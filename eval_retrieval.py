import json
import os
import re
import sys
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

# =============================================================================
# Domain Delivery RAG - Retrieval-only Eval
# =============================================================================
#
# Purpose:
#   Automated regression checks for retrieval quality.
#
# Where it fits:
#   python3 ingest.py
#   python3 eval_retrieval.py
#   python3 ask_qwen.py "..."
#
# This script does NOT call the chat model.
# It only verifies that retrieval finds expected source files/chunks.
#
# It mirrors ask_qwen.py retrieval:
#   query -> embedding -> Qdrant vector search
#   -> bounded heuristic metadata reranking
#   -> file diversity
#   -> neighbor expansion
#
# Eval checks:
#   - expected files selected/expanded
#   - expected chunks selected/expanded
#   - forbidden files/chunks absent
#   - drop chunks absent
#
# Exit code:
#   0 = all eval cases pass
#   1 = at least one eval case fails
# =============================================================================


EMBED_URL = os.environ.get("RAG_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")
QDRANT_URL = os.environ.get("RAG_QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("RAG_COLLECTION", "rag_v1_chunks")

EVAL_FILE = os.environ.get(
    "RAG_EVAL_FILE",
    os.path.expanduser("~/rag_v1/eval_queries.json"),
)

DEFAULT_TOP_K = int(os.environ.get("RAG_ASK_TOP_K", "4"))
DEFAULT_PRE_K = int(os.environ.get("RAG_ASK_PRE_K", "24"))
MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", "2"))
NEIGHBOR_RADIUS = int(os.environ.get("RAG_NEIGHBOR_RADIUS", "1"))

VERBOSE = os.environ.get("RAG_VERBOSE", "1") != "0"
DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"


# =============================================================================
# Basic logging
# =============================================================================

def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)


def debug_print(title, value):
    if DEBUG:
        print("=" * 100)
        print(f"DEBUG: {title}")
        print("=" * 100)
        if isinstance(value, (dict, list)):
            print(json.dumps(value, indent=2, ensure_ascii=False)[:30000])
        else:
            print(str(value)[:30000])
        print()


# =============================================================================
# Embedding
# =============================================================================

def embed(text: str):
    r = requests.post(EMBED_URL, json={"input": text}, timeout=120)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


# =============================================================================
# Query profiling and bounded heuristic metadata reranking
# Keep this in sync with ask_qwen.py/search.py.
# =============================================================================

def tokenize_query(query: str):
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9_#+.-]{3,}", query.lower())
    stop = {
        "the", "and", "for", "with", "from", "into", "what", "does", "this",
        "that", "are", "how", "why", "when", "where", "which", "about",
        "likely", "implications", "context", "corpus",
        "что", "как", "для", "или", "это", "про", "при", "над", "под",
    }
    return [t for t in tokens if t not in stop]


def query_profile(query: str):
    q = query.lower()
    tokens = set(tokenize_query(query))

    profile = {
        "roles": set(),
        "facets": set(),
        "layers": set(),
        "stages": set(),
        "flags": set(),
    }

    def has_any(words):
        return any(w in q for w in words) or any(w in tokens for w in words)

    if has_any(["warning", "alert", "alarm", "hmi", "driver", "stage", "stages", "yellow", "red"]):
        profile["facets"].update({"system_behavior", "interface", "constraints"})
        profile["layers"].update({"hmi", "decision_logic", "vehicle_integration"})
        profile["flags"].add("has_behavioral_requirements")

    if has_any(["blind spot", "pedestrian", "cyclist", "vru", "detection", "perception", "person"]):
        profile["facets"].update({"system_behavior", "performance", "constraints"})
        profile["layers"].update({"perception", "sensor", "decision_logic"})

    if has_any([
        "false positive", "false positives", "false negative", "false negatives",
        "fp", "fn", "validation", "verification", "test", "tests", "scenario", "scenarios"
    ]):
        profile["roles"].update({"validation", "test"})
        profile["facets"].update({"validation", "test_scenarios", "performance", "failure_modes"})
        profile["stages"].add("verification")
        profile["flags"].add("has_validation_or_test_evidence")

    if has_any([
        "degraded", "fallback", "shutdown", "failure", "contamination",
        "blocked", "lighting", "unavailable", "fault"
    ]):
        profile["facets"].update({"failure_modes", "system_behavior", "constraints"})
        profile["layers"].update({"sensor", "decision_logic", "hmi"})
        profile["flags"].add("has_failure_or_degraded_mode")

    if has_any(["interface", "can", "gpio", "api", "contract", "integration", "signal"]):
        profile["roles"].add("interface_contract")
        profile["facets"].update({"interface", "configuration", "implementation"})
        profile["layers"].update({"vehicle_integration", "embedded_runtime", "hmi"})
        profile["flags"].add("has_interface_or_contract")

    if has_any([
        "deploy", "deployment", "runtime", "embedded", "edge",
        "latency", "performance", "quantized", "int8"
    ]):
        profile["roles"].add("deployment")
        profile["facets"].update({"deployment", "performance", "implementation", "constraints"})
        profile["layers"].update({"embedded_runtime", "perception"})
        profile["stages"].update({"implementation", "release"})

    if has_any([
        "regulation", "regulatory", "compliance", "standard", "standards",
        "certification", "approval", "ece", "gsr", "iso"
    ]):
        profile["roles"].add("regulation")
        profile["facets"].update({"regulatory", "constraints"})
        profile["layers"].add("compliance")
        profile["stages"].update({"verification", "release"})
        profile["flags"].add("has_regulatory_or_compliance")

    if has_any(["data", "annotation", "training", "dataset", "augmentation", "label", "labels"]):
        profile["roles"].add("data_management")
        profile["facets"].update({"data", "implementation", "examples"})
        profile["layers"].add("data_pipeline")
        profile["stages"].update({"implementation", "verification"})

    return profile


def pretty_profile(profile: dict):
    return {
        "roles": sorted(profile["roles"]),
        "facets": sorted(profile["facets"]),
        "layers": sorted(profile["layers"]),
        "stages": sorted(profile["stages"]),
        "flags": sorted(profile["flags"]),
    }


def intersect_count(payload_value, wanted: set):
    if not wanted:
        return 0

    if isinstance(payload_value, list):
        return len(set(payload_value) & wanted)

    if isinstance(payload_value, str):
        return 1 if payload_value in wanted else 0

    return 0


def clamp(x: float, lo: float, hi: float):
    return max(lo, min(hi, x))


def metadata_prior(payload: dict, query: str = "") -> float:
    """
    Bounded heuristic metadata reranking.

    Same scoring as ask_qwen.py/search.py.
    Clamp range: [-0.045, +0.050].
    """
    bonus = 0.0

    corpus_decision = payload.get("corpus_decision")
    delivery_value = payload.get("delivery_value")
    safety_relevance = payload.get("safety_relevance")
    chunk_role = payload.get("chunk_role")
    confidence = payload.get("confidence") or 0.0

    if corpus_decision == "primary":
        bonus += 0.005
    elif corpus_decision == "drop":
        bonus -= 0.040

    if delivery_value == "high":
        bonus += 0.005
    elif delivery_value == "medium":
        bonus += 0.002
    elif delivery_value == "low":
        bonus -= 0.002

    if safety_relevance == "high":
        bonus += 0.004
    elif safety_relevance == "medium":
        bonus += 0.001

    if chunk_role == "noise":
        bonus -= 0.040
    elif chunk_role == "unknown":
        bonus -= 0.010

    try:
        bonus += min(float(confidence), 1.0) * 0.0025
    except Exception:
        pass

    profile = query_profile(query)

    role_hits = intersect_count(payload.get("chunk_role"), profile["roles"])
    facet_hits = intersect_count(payload.get("content_facets"), profile["facets"])
    layer_hits = intersect_count(payload.get("system_layers"), profile["layers"])
    stage_hits = intersect_count(payload.get("workflow_stages"), profile["stages"])

    bonus += min(role_hits * 0.005, 0.008)
    bonus += min(facet_hits * 0.004, 0.016)
    bonus += min(layer_hits * 0.003, 0.009)
    bonus += min(stage_hits * 0.002, 0.006)

    for flag in profile["flags"]:
        if payload.get(flag) is True:
            bonus += 0.005

    return clamp(bonus, -0.045, 0.050)


# =============================================================================
# Retrieval and neighbor expansion
# =============================================================================

def retrieve(query: str, top_k: int, pre_k: int, max_per_file: int):
    client = QdrantClient(url=QDRANT_URL)
    qvec = embed(query)

    raw = client.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=pre_k,
        with_payload=True,
    ).points

    candidates = []
    for point in raw:
        payload = point.payload or {}
        vector_score = float(point.score)
        meta_bonus = metadata_prior(payload, query=query)
        final_score = vector_score + meta_bonus

        candidates.append(
            {
                "final_score": final_score,
                "vector_score": vector_score,
                "meta_bonus": meta_bonus,
                "payload": payload,
                "is_selected_hit": True,
            }
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    selected = []
    per_file = {}

    for item in candidates:
        payload = item["payload"]
        file_path = payload.get("file_path", "unknown")

        if per_file.get(file_path, 0) >= max_per_file:
            continue

        selected.append(item)
        per_file[file_path] = per_file.get(file_path, 0) + 1

        if len(selected) >= top_k:
            break

    return selected, candidates


def fetch_chunk_payloads_for_file(client: QdrantClient, file_path: str, indices: set):
    if not indices:
        return {}

    min_idx = min(indices)
    max_idx = max(indices)
    out = {}

    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_path",
                        match=models.MatchValue(value=file_path),
                    ),
                    models.FieldCondition(
                        key="chunk_index",
                        range=models.Range(gte=min_idx, lte=max_idx),
                    ),
                ]
            ),
            limit=max(64, max_idx - min_idx + 1),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for rec in records:
            payload = rec.payload or {}
            idx = payload.get("chunk_index")
            if idx in indices:
                out[idx] = payload

        if offset is None:
            break

    return out


def expand_results_with_neighbors(results, radius: int):
    if radius <= 0:
        groups = []
        for rank, item in enumerate(results, start=1):
            p = item["payload"]
            groups.append(
                {
                    "group_rank": rank,
                    "file_path": p.get("file_path", "unknown"),
                    "file_name": p.get("file_name", "unknown"),
                    "selected_indices": [p.get("chunk_index")],
                    "expanded_indices": [p.get("chunk_index")],
                    "best_final_score": item.get("final_score"),
                    "chunks": [item],
                }
            )
        return groups

    client = QdrantClient(url=QDRANT_URL)

    requested_by_file = {}
    selected_by_file = {}
    best_rank_by_file = {}
    best_score_by_file = {}
    selected_lookup = {}

    for rank, item in enumerate(results, start=1):
        p = item["payload"]
        file_path = p.get("file_path", "unknown")
        chunk_index = p.get("chunk_index")

        if not isinstance(chunk_index, int):
            continue

        requested_by_file.setdefault(file_path, set())
        selected_by_file.setdefault(file_path, set())

        selected_by_file[file_path].add(chunk_index)
        selected_lookup[(file_path, chunk_index)] = item

        if file_path not in best_rank_by_file:
            best_rank_by_file[file_path] = rank
            best_score_by_file[file_path] = item.get("final_score")
        else:
            if item.get("final_score", 0.0) > (best_score_by_file.get(file_path) or 0.0):
                best_score_by_file[file_path] = item.get("final_score")

        for idx in range(chunk_index - radius, chunk_index + radius + 1):
            if idx >= 0:
                requested_by_file[file_path].add(idx)

    groups = []
    file_order = sorted(requested_by_file.keys(), key=lambda fp: best_rank_by_file.get(fp, 999999))

    for group_rank, file_path in enumerate(file_order, start=1):
        requested_indices = requested_by_file[file_path]
        neighbor_payloads = fetch_chunk_payloads_for_file(
            client,
            file_path=file_path,
            indices=requested_indices,
        )

        chunks = []
        for idx in sorted(requested_indices):
            key = (file_path, idx)

            if key in selected_lookup:
                item = dict(selected_lookup[key])
                item["is_selected_hit"] = True
            else:
                payload = neighbor_payloads.get(idx)
                if payload is None:
                    continue

                item = {
                    "final_score": None,
                    "vector_score": None,
                    "meta_bonus": None,
                    "payload": payload,
                    "is_selected_hit": False,
                }

            chunks.append(item)

        if not chunks:
            continue

        first_payload = chunks[0]["payload"]
        expanded_indices = [
            c["payload"].get("chunk_index")
            for c in chunks
            if isinstance(c["payload"].get("chunk_index"), int)
        ]

        groups.append(
            {
                "group_rank": group_rank,
                "file_path": file_path,
                "file_name": first_payload.get("file_name", "unknown"),
                "selected_indices": sorted(selected_by_file.get(file_path, set())),
                "expanded_indices": sorted(expanded_indices),
                "best_final_score": best_score_by_file.get(file_path),
                "chunks": chunks,
            }
        )

    return groups


# =============================================================================
# Eval helpers
# =============================================================================

def load_eval_cases(path: str):
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"Eval file not found: {path}\n"
            f"Create ~/rag_v1/eval_queries.json first."
        )

    with p.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, list):
        raise SystemExit("eval_queries.json must contain a JSON list")

    return cases


def file_name_from_payload(payload: dict):
    return payload.get("file_name") or os.path.basename(payload.get("file_path", ""))


def selected_file_set(results):
    return {file_name_from_payload(item["payload"]) for item in results}


def expanded_file_set(groups):
    out = set()
    for group in groups:
        for item in group["chunks"]:
            out.add(file_name_from_payload(item["payload"]))
    return out


def selected_chunk_set(results):
    out = set()
    for item in results:
        p = item["payload"]
        out.add((file_name_from_payload(p), p.get("chunk_index")))
    return out


def expanded_chunk_set(groups):
    out = set()
    for group in groups:
        for item in group["chunks"]:
            p = item["payload"]
            out.add((file_name_from_payload(p), p.get("chunk_index")))
    return out


def has_any_chunk(chunk_set: set, spec: dict):
    file_name = spec["file"]
    chunks = spec.get("chunks", [])
    return any((file_name, c) in chunk_set for c in chunks)


def has_all_chunks(chunk_set: set, spec: dict):
    file_name = spec["file"]
    chunks = spec.get("chunks", [])
    return all((file_name, c) in chunk_set for c in chunks)


def chunk_hits(chunk_set: set, spec: dict):
    file_name = spec["file"]
    chunks = spec.get("chunks", [])
    return [c for c in chunks if (file_name, c) in chunk_set]


def collect_drop_hits(results, groups):
    hits = []

    for item in results:
        p = item["payload"]
        if p.get("corpus_decision") == "drop":
            hits.append(("selected", file_name_from_payload(p), p.get("chunk_index")))

    for group in groups:
        for item in group["chunks"]:
            p = item["payload"]
            if p.get("corpus_decision") == "drop":
                hit = ("expanded", file_name_from_payload(p), p.get("chunk_index"))
                if hit not in hits:
                    hits.append(hit)

    return hits


def fmt_file_list(values):
    if not values:
        return "[]"
    return "[" + ", ".join(sorted(values)) + "]"


def fmt_chunk_set(values, limit=20):
    ordered = sorted(values, key=lambda x: (x[0], -1 if x[1] is None else x[1]))
    shown = ordered[:limit]
    text = ", ".join(f"{f}:#{c}" for f, c in shown)
    if len(ordered) > limit:
        text += f", ... +{len(ordered) - limit} more"
    return "[" + text + "]"


def check_case(case: dict):
    case_id = case.get("id", "unnamed")
    query = case["query"]

    top_k = int(case.get("top_k", DEFAULT_TOP_K))
    pre_k = int(case.get("pre_k", max(DEFAULT_PRE_K, top_k * 4)))
    max_per_file = int(case.get("max_per_file", MAX_PER_FILE))
    neighbor_radius = int(case.get("neighbor_radius", NEIGHBOR_RADIUS))

    results, candidates = retrieve(
        query=query,
        top_k=top_k,
        pre_k=pre_k,
        max_per_file=max_per_file,
    )

    groups = expand_results_with_neighbors(results, radius=neighbor_radius)

    selected_files = selected_file_set(results)
    expanded_files = expanded_file_set(groups)

    selected_chunks = selected_chunk_set(results)
    expanded_chunks = expanded_chunk_set(groups)

    failures = []
    warnings = []

    # File checks use expanded files because neighbor expansion is part of the
    # actual ask context. The selected-only set is still printed for debugging.
    for file_name in case.get("expected_files_all", []):
        if file_name not in expanded_files:
            failures.append(f"missing expected file: {file_name}")

    expected_files_any = case.get("expected_files_any", [])
    if expected_files_any and not any(f in expanded_files for f in expected_files_any):
        failures.append(f"none of expected_files_any found: {expected_files_any}")

    for file_name in case.get("forbidden_files", []):
        if file_name in expanded_files:
            failures.append(f"forbidden file retrieved: {file_name}")

    # Chunk checks.
    for spec in case.get("expected_selected_chunks_any", []):
        if not has_any_chunk(selected_chunks, spec):
            failures.append(
                f"missing any selected chunk for {spec['file']} "
                f"expected_any={spec.get('chunks', [])}"
            )

    for spec in case.get("expected_selected_chunks_all", []):
        if not has_all_chunks(selected_chunks, spec):
            hits = chunk_hits(selected_chunks, spec)
            failures.append(
                f"missing selected chunks for {spec['file']} "
                f"expected_all={spec.get('chunks', [])} hits={hits}"
            )

    for spec in case.get("expected_expanded_chunks_any", []):
        if not has_any_chunk(expanded_chunks, spec):
            failures.append(
                f"missing any expanded chunk for {spec['file']} "
                f"expected_any={spec.get('chunks', [])}"
            )

    for spec in case.get("expected_expanded_chunks_all", []):
        if not has_all_chunks(expanded_chunks, spec):
            hits = chunk_hits(expanded_chunks, spec)
            failures.append(
                f"missing expanded chunks for {spec['file']} "
                f"expected_all={spec.get('chunks', [])} hits={hits}"
            )

    for spec in case.get("forbidden_chunks", []):
        file_name = spec["file"]
        for chunk_index in spec.get("chunks", []):
            if (file_name, chunk_index) in expanded_chunks:
                failures.append(f"forbidden chunk retrieved: {file_name}#{chunk_index}")

    drop_hits = collect_drop_hits(results, groups)
    if drop_hits:
        failures.append(f"drop chunks retrieved despite default policy: {drop_hits}")

    min_selected = int(case.get("min_selected_results", 1))
    if len(results) < min_selected:
        failures.append(f"selected results below minimum: {len(results)} < {min_selected}")

    min_groups = int(case.get("min_source_groups", 1))
    if len(groups) < min_groups:
        failures.append(f"source groups below minimum: {len(groups)} < {min_groups}")

    passed = not failures

    return {
        "id": case_id,
        "query": query,
        "passed": passed,
        "failures": failures,
        "warnings": warnings,
        "results": results,
        "candidates": candidates,
        "groups": groups,
        "selected_files": selected_files,
        "expanded_files": expanded_files,
        "selected_chunks": selected_chunks,
        "expanded_chunks": expanded_chunks,
        "top_k": top_k,
        "pre_k": pre_k,
        "max_per_file": max_per_file,
        "neighbor_radius": neighbor_radius,
        "notes": case.get("notes", ""),
    }


def print_case_report(report: dict):
    status = "PASS" if report["passed"] else "FAIL"

    print("=" * 100)
    print(f"{status}: {report['id']}")
    print("=" * 100)
    print(f"Query: {report['query']}")
    print(
        f"top_k={report['top_k']}; pre_k={report['pre_k']}; "
        f"max_per_file={report['max_per_file']}; "
        f"neighbor_radius={report['neighbor_radius']}"
    )
    if report["notes"]:
        print(f"Notes: {report['notes']}")
    print()

    print(f"Selected files: {fmt_file_list(report['selected_files'])}")
    print(f"Expanded files: {fmt_file_list(report['expanded_files'])}")
    print(f"Selected chunks: {fmt_chunk_set(report['selected_chunks'])}")
    print(f"Expanded chunks: {fmt_chunk_set(report['expanded_chunks'])}")
    print()

    print("Selected results:")
    for rank, item in enumerate(report["results"], start=1):
        p = item["payload"]
        print(
            f"  {rank}. final={item['final_score']:.4f} "
            f"vector={item['vector_score']:.4f} "
            f"meta={item['meta_bonus']:+.4f} "
            f"file={file_name_from_payload(p)} "
            f"chunk={p.get('chunk_index')} "
            f"role={p.get('chunk_role')} "
            f"facets={p.get('content_facets')} "
            f"safety={p.get('safety_relevance')} "
            f"delivery={p.get('delivery_value')} "
            f"decision={p.get('corpus_decision')}"
        )

    print()

    if report["groups"]:
        print("Source groups after neighbor expansion:")
        for idx, group in enumerate(report["groups"], start=1):
            print(
                f"  S{idx}. file={group.get('file_name')} "
                f"selected={group.get('selected_indices')} "
                f"expanded={group.get('expanded_indices')} "
                f"best_final={group.get('best_final_score'):.4f}"
                if group.get("best_final_score") is not None
                else f"  S{idx}. file={group.get('file_name')} "
                     f"selected={group.get('selected_indices')} "
                     f"expanded={group.get('expanded_indices')}"
            )

    print()

    if report["failures"]:
        print("Failures:")
        for failure in report["failures"]:
            print(f"  - {failure}")
        print()

    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
        print()


def main():
    only_id = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 100)
    print("RAG RETRIEVAL EVAL START")
    print("=" * 100)
    print(f"Eval file: {EVAL_FILE}")
    print(f"Collection: {COLLECTION}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Embedding URL: {EMBED_URL}")
    print(f"Defaults: top_k={DEFAULT_TOP_K}; pre_k={DEFAULT_PRE_K}; max_per_file={MAX_PER_FILE}; neighbor_radius={NEIGHBOR_RADIUS}")
    print(f"metadata_rerank_clamp=[-0.045,+0.050]")
    print(f"Verbose: {VERBOSE}; Debug raw payloads: {DEBUG}")
    if only_id:
        print(f"Filter case id: {only_id}")
    print()

    cases = load_eval_cases(EVAL_FILE)

    if only_id:
        cases = [c for c in cases if c.get("id") == only_id]
        if not cases:
            raise SystemExit(f"No eval case found with id: {only_id}")

    reports = []
    for case in cases:
        report = check_case(case)
        reports.append(report)
        print_case_report(report)

    passed = sum(1 for r in reports if r["passed"])
    failed = len(reports) - passed

    print("=" * 100)
    print("RAG RETRIEVAL EVAL SUMMARY")
    print("=" * 100)
    print(f"Cases total: {len(reports)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed:
        print()
        print("Failed case ids:")
        for r in reports:
            if not r["passed"]:
                print(f"  - {r['id']}")

    print("=" * 100)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
