import json
import os
import re
import sys

import requests
from qdrant_client import QdrantClient

# =============================================================================
# Domain Delivery RAG - Diagnostic Search
# =============================================================================
#
# This mirrors ask_qwen.py retrieval logic, but does not call the chat model.
# Use it to inspect ranking, metadata rerank, diversity, and chunk content.
# =============================================================================


EMBED_URL = os.environ.get("RAG_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")
QDRANT_URL = os.environ.get("RAG_QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("RAG_COLLECTION", "rag_v1_chunks")

# Defaults intentionally match ask_qwen.py.
DEFAULT_TOP_K = int(os.environ.get("RAG_SEARCH_TOP_K", os.environ.get("RAG_ASK_TOP_K", "4")))
DEFAULT_PRE_K = int(os.environ.get("RAG_SEARCH_PRE_K", os.environ.get("RAG_ASK_PRE_K", "24")))
MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", "2"))

VERBOSE = os.environ.get("RAG_VERBOSE", "1") != "0"
DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"


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


def embed(text: str):
    r = requests.post(EMBED_URL, json={"input": text}, timeout=120)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


# =============================================================================
# Query profiling and bounded heuristic metadata reranking
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

    Same scoring as ask_qwen.py.
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
# Retrieval
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


# =============================================================================
# Output formatting
# =============================================================================

def fmt_list(value):
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)


def preview(text: str, n: int = 1400):
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "\n...[truncated]"


def print_candidate_summary(candidates, limit=10):
    print("=" * 100)
    print("TOP RAW CANDIDATES BEFORE DIVERSITY")
    print("=" * 100)

    for i, item in enumerate(candidates[:limit], start=1):
        p = item["payload"]
        print(
            f"{i:02d}. final={item['final_score']:.4f} "
            f"vector={item['vector_score']:.4f} "
            f"meta={item['meta_bonus']:+.4f} "
            f"file={p.get('file_name')} "
            f"chunk={p.get('chunk_index')} "
            f"role={p.get('chunk_role')} "
            f"facets={fmt_list(p.get('content_facets'))} "
            f"layers={fmt_list(p.get('system_layers'))} "
            f"safety={p.get('safety_relevance')} "
            f"delivery={p.get('delivery_value')} "
            f"decision={p.get('corpus_decision')}"
        )

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print('Usage: python3 ~/rag_v1/search.py "your query" [top_k] [pre_k]')
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOP_K
    pre_k = int(sys.argv[3]) if len(sys.argv) > 3 else max(DEFAULT_PRE_K, top_k * 4)

    print("=" * 100)
    print("RAG SEARCH START")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Collection: {COLLECTION}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Embedding URL: {EMBED_URL}")
    print(f"top_k={top_k}; pre_k={pre_k}; max_per_file={MAX_PER_FILE}")
    print(f"metadata_rerank_clamp=[-0.045,+0.050]")
    print(f"Verbose: {VERBOSE}; Debug raw payloads: {DEBUG}")
    print()

    profile = query_profile(query)
    print("Query profile:")
    print(json.dumps(pretty_profile(profile), ensure_ascii=False, indent=2))
    print()

    results, candidates = retrieve(
        query=query,
        top_k=top_k,
        pre_k=pre_k,
        max_per_file=MAX_PER_FILE,
    )

    print_candidate_summary(candidates, limit=min(10, len(candidates)))

    print("=" * 100)
    print("SELECTED RESULTS AFTER DIVERSITY")
    print("=" * 100)

    for rank, item in enumerate(results, start=1):
        p = item["payload"]

        print("=" * 100)
        print(f"RANK: {rank}")
        print(f"FINAL_SCORE: {item['final_score']:.4f}")
        print(f"VECTOR_SCORE: {item['vector_score']:.4f}")
        print(f"META_BONUS: {item['meta_bonus']:+.4f}")
        print(f"FILE: {p.get('file_path')}")
        print(f"CHUNK: {p.get('chunk_index')}")
        print("-" * 100)

        print("CHUNK METADATA")
        print(f"chunk_role: {p.get('chunk_role')}")
        print(f"content_facets: {fmt_list(p.get('content_facets'))}")
        print(f"system_layers: {fmt_list(p.get('system_layers'))}")
        print(f"workflow_stages: {fmt_list(p.get('workflow_stages'))}")
        print(f"safety_relevance: {p.get('safety_relevance')}")
        print(f"delivery_value: {p.get('delivery_value')}")
        print(f"corpus_decision: {p.get('corpus_decision')}")
        print(f"has_behavioral_requirements: {p.get('has_behavioral_requirements')}")
        print(f"has_interface_or_contract: {p.get('has_interface_or_contract')}")
        print(f"has_validation_or_test_evidence: {p.get('has_validation_or_test_evidence')}")
        print(f"has_failure_or_degraded_mode: {p.get('has_failure_or_degraded_mode')}")
        print(f"has_regulatory_or_compliance: {p.get('has_regulatory_or_compliance')}")
        print(f"confidence: {p.get('confidence')}")
        print(f"reason_short: {p.get('reason_short')}")

        print("-" * 100)
        print("DOCUMENT METADATA")
        print(f"document_primary_role: {p.get('document_primary_role')}")
        print(f"document_roles: {fmt_list(p.get('document_roles'))}")
        print(f"document_content_facets: {fmt_list(p.get('document_content_facets'))}")
        print(f"document_system_layers: {fmt_list(p.get('document_system_layers'))}")
        print(f"document_workflow_stages: {fmt_list(p.get('document_workflow_stages'))}")
        print(f"document_safety_relevance: {p.get('document_safety_relevance')}")
        print(f"document_delivery_value: {p.get('document_delivery_value')}")
        print(f"document_corpus_decision: {p.get('document_corpus_decision')}")
        print(f"document_confidence: {p.get('document_confidence')}")
        print(f"document_signal_chunks: {p.get('document_signal_chunks')}")

        print("-" * 100)
        print("CONTENT PREVIEW")
        print(preview(p.get("content") or ""))
        print()

        if DEBUG:
            debug_print("FULL PAYLOAD", p)

    print("=" * 100)
    print("RAG SEARCH DONE")
    print("=" * 100)
    print(f"Raw candidates: {len(candidates)}")
    print(f"Selected results: {len(results)}")
    print("=" * 100)


if __name__ == "__main__":
    main()