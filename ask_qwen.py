import json
import os
import re
import sys

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

# =============================================================================
# Domain Delivery RAG - Ask Pipeline
# =============================================================================
#
# Flow:
#   user question -> embed query -> Qdrant vector search
#   -> bounded heuristic metadata reranking -> diversity by file
#   -> neighbor expansion -> evidence-only context -> Qwen answer
#
# Design principles:
#   - Vector similarity is the primary first-stage retrieval signal.
#   - Metadata is used for bounded heuristic reranking and diagnostics.
#   - Metadata is NOT included as answer evidence in the LLM prompt.
#   - Selected hit chunks are included fully.
#   - Neighbor chunks are included as snippets to restore local context cheaply.
#   - Qwen reasoning may be enabled, but stdout ANSWER must contain only final answer.
# =============================================================================


EMBED_URL = os.environ.get("RAG_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")
CHAT_URL = os.environ.get("RAG_CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions")
QDRANT_URL = os.environ.get("RAG_QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("RAG_COLLECTION", "rag_v1_chunks")

DEFAULT_TOP_K = int(os.environ.get("RAG_ASK_TOP_K", "4"))
DEFAULT_PRE_K = int(os.environ.get("RAG_ASK_PRE_K", "24"))
MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", "2"))

NEIGHBOR_RADIUS = int(os.environ.get("RAG_NEIGHBOR_RADIUS", "1"))

# Selected chunks go into the prompt fully.
# Neighbor chunks use query-dense snippets.
NEIGHBOR_SNIPPET_CHARS = int(os.environ.get("RAG_NEIGHBOR_SNIPPET_CHARS", "700"))

MAX_TOKENS = int(os.environ.get("RAG_ANSWER_MAX_TOKENS", "10000"))

VERBOSE = os.environ.get("RAG_VERBOSE", "1") != "0"
DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"


SYSTEM_PROMPT = """You are a senior domain delivery assistant for safety-relevant technical systems, embedded vision, ADAS, and edge AI.

Strict rules:
- Retrieved content is the only source of truth.
- Source ids, file names, and chunk ids are provenance, not evidence by themselves.
- Do not reveal reasoning, scratchpad, self-correction, or hidden analysis.
- Return only the final answer in the requested answer format.
- Do not invent thresholds, interfaces, warning logic, state machines, safety claims, or compliance details not explicitly supported by retrieved content.
- Separate supported facts from inferences.
- If evidence is insufficient, say: Not enough evidence in retrieved context.
- Be conservative on safety-relevant topics.
- Cite sources inline as [S1], [S2], etc.
- A source [Sx] may contain an expanded local context window around a retrieved chunk.

Answer format:
1. Conclusion
2. Supported facts
3. Inferences
4. Implementation implications
5. Unknowns / verification needed
6. Source mapping
"""


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

    Clamp range: [-0.045, +0.050].
    Metadata can reorder very close vector matches, but should not dominate
    retrieval.
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

def retrieve(query: str, top_k: int = DEFAULT_TOP_K, pre_k: int = DEFAULT_PRE_K, max_per_file: int = MAX_PER_FILE):
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


def expand_results_with_neighbors(results, radius: int = NEIGHBOR_RADIUS):
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
        neighbor_payloads = fetch_chunk_payloads_for_file(client, file_path=file_path, indices=requested_indices)

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
# Evidence-only context packing
# =============================================================================

def best_snippet(text: str, query: str, limit: int = NEIGHBOR_SNIPPET_CHARS):
    text = (text or "").strip()
    if len(text) <= limit:
        return text

    q_tokens = tokenize_query(query)
    if not q_tokens:
        return text[:limit] + "\n...[truncated]"

    lower = text.lower()
    positions = []

    for token in q_tokens:
        start = 0
        while True:
            idx = lower.find(token, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + len(token)

    if not positions:
        return text[:limit] + "\n...[truncated]"

    positions.sort()

    best_start = 0
    best_hits = -1
    half = limit // 2

    for pos in positions:
        start = max(0, min(pos - half, len(text) - limit))
        end = min(len(text), start + limit)
        hits = sum(1 for p in positions if start <= p <= end)

        if hits > best_hits:
            best_hits = hits
            best_start = start

    snippet = text[best_start:best_start + limit].strip()
    prefix = "...[truncated before]\n" if best_start > 0 else ""
    suffix = "\n...[truncated after]" if best_start + limit < len(text) else ""

    return prefix + snippet + suffix


def fmt_list(value):
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)


def fmt_score(value):
    if value is None:
        return "neighbor"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def evidence_text_for_item(item, query: str):
    p = item["payload"]
    content = (p.get("content") or "").strip()

    if item.get("is_selected_hit"):
        return content

    return best_snippet(content, query=query, limit=NEIGHBOR_SNIPPET_CHARS)


def build_context(source_groups, query: str):
    """
    Build evidence-only LLM context.

    Metadata is intentionally not included here.
    It remains visible in stdout diagnostics, but Qwen answers only from content.
    """
    blocks = []

    for rank, group in enumerate(source_groups, start=1):
        chunks = group["chunks"]
        chunk_blocks = []

        for item in chunks:
            p = item["payload"]
            hit_label = "selected_hit_full_content" if item.get("is_selected_hit") else "neighbor_snippet"
            evidence = evidence_text_for_item(item, query=query)

            chunk_blocks.append(
                f"--- chunk {p.get('chunk_index')} ({hit_label}) ---\n"
                f"{evidence}\n"
            )

        blocks.append(
            f"[S{rank}]\n"
            f"file={group.get('file_name')}\n"
            f"selected_chunks={group.get('selected_indices')}\n"
            f"expanded_chunks={group.get('expanded_indices')}\n"
            f"\n"
            f"content:\n"
            f"{chr(10).join(chunk_blocks)}"
        )

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(blocks)


# =============================================================================
# Answer generation and cleanup
# =============================================================================

def clean_model_answer(text: str) -> str:
    """
    Strip reasoning leaks from local Qwen responses.

    Qwen reasoning mode may occasionally put draft/self-correction into
    message.content. The public stdout answer should contain only final answer.
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    cleaned = re.sub(r"^```(?:markdown|text)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # If model leaked draft before the requested final structure, keep final part.
    final_start = re.search(
        r"(?mi)^\s*(?:#{1,6}\s*)?1[\.\)]\s*(?:\*\*)?(Conclusion|Вывод|Краткий вывод)\b",
        cleaned,
    )
    if final_start:
        cleaned = cleaned[final_start.start():].strip()

    return cleaned


def extract_answer(data):
    try:
        msg = data["choices"][0]["message"]
        content = msg.get("content", "")
        cleaned = clean_model_answer(content)
        if cleaned:
            return cleaned
    except Exception:
        pass

    try:
        text = data["choices"][0].get("text", "")
        cleaned = clean_model_answer(text)
        if cleaned:
            return cleaned
    except Exception:
        pass

    if DEBUG:
        try:
            msg = data["choices"][0]["message"]
            reasoning = msg.get("reasoning_content", "")
            cleaned = clean_model_answer(reasoning)
            if cleaned:
                return "[DEBUG FALLBACK: reasoning_content]\n\n" + cleaned
        except Exception:
            pass

    return None


def ask_model(question: str, context: str):
    user_prompt = f"""Question:
{question}

Retrieved evidence:
{context}

Instructions:
- Use retrieved evidence first.
- Neighbor chunks are provided only to restore local context around selected hits.
- Source ids, file names, and chunk ids are provenance, not evidence by themselves.
- Return only the final answer. Do not include reasoning, draft, self-correction, or analysis.
- Cite source groups inline as [S1], [S2], etc.
- Separate supported facts from inferences.
- If evidence is insufficient, say so clearly.
"""

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": MAX_TOKENS,
    }

    debug_print("CHAT PAYLOAD", payload)

    r = requests.post(CHAT_URL, json=payload, timeout=900)
    debug_print("HTTP STATUS", r.status_code)
    debug_print("RAW RESPONSE TEXT", r.text)

    r.raise_for_status()
    data = r.json()
    debug_print("PARSED JSON", data)

    answer = extract_answer(data)
    if answer is None:
        raise RuntimeError("Model response did not contain a non-empty final answer")

    return answer


# =============================================================================
# Observability
# =============================================================================

def print_candidate_summary(candidates, limit=10):
    log("Top raw candidates before diversity:")
    for i, item in enumerate(candidates[:limit], start=1):
        p = item["payload"]
        log(
            f"  {i:02d}. final={item['final_score']:.4f} "
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


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print('Usage: python3 ask_qwen.py "your question" [top_k] [pre_k]')
        sys.exit(1)

    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOP_K
    pre_k = int(sys.argv[3]) if len(sys.argv) > 3 else max(DEFAULT_PRE_K, top_k * 4)

    print("=" * 100)
    print("RAG ASK START")
    print("=" * 100)
    print(f"Question: {question}")
    print(f"Collection: {COLLECTION}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Embedding URL: {EMBED_URL}")
    print(f"Chat URL: {CHAT_URL}")
    print(f"top_k={top_k}; pre_k={pre_k}; max_per_file={MAX_PER_FILE}; neighbor_radius={NEIGHBOR_RADIUS}")
    print(f"neighbor_snippet_chars={NEIGHBOR_SNIPPET_CHARS}; answer_max_tokens={MAX_TOKENS}")
    print(f"metadata_rerank_clamp=[-0.045,+0.050]")
    print(f"Verbose: {VERBOSE}; Debug raw payloads: {DEBUG}")
    print()

    profile = query_profile(question)
    print("Query profile:")
    print(json.dumps(pretty_profile(profile), ensure_ascii=False, indent=2))
    print()

    results, candidates = retrieve(
        query=question,
        top_k=top_k,
        pre_k=pre_k,
        max_per_file=MAX_PER_FILE,
    )

    print_candidate_summary(candidates, limit=min(10, len(candidates)))

    source_groups = expand_results_with_neighbors(results, radius=NEIGHBOR_RADIUS)
    context = build_context(source_groups, query=question)

    print()
    print("=" * 100)
    print("RETRIEVED SOURCE GROUPS")
    print("=" * 100)

    selected_chunks_total = 0
    expanded_chunks_total = 0

    for rank, group in enumerate(source_groups, start=1):
        selected_chunks_total += len(group.get("selected_indices") or [])
        expanded_chunks_total += len(group.get("expanded_indices") or [])

        print(
            f"{rank}. file={group.get('file_path')} "
            f"selected_chunks={group.get('selected_indices')} "
            f"expanded_chunks={group.get('expanded_indices')} "
            f"best_final={fmt_score(group.get('best_final_score'))}"
        )

        for item in group["chunks"]:
            p = item["payload"]
            marker = "*" if item.get("is_selected_hit") else "-"
            content_mode = "full" if item.get("is_selected_hit") else "snippet"
            print(
                f"   {marker} chunk={p.get('chunk_index')} "
                f"mode={content_mode} "
                f"score={fmt_score(item.get('final_score'))} "
                f"role={p.get('chunk_role')} "
                f"facets={fmt_list(p.get('content_facets'))} "
                f"layers={fmt_list(p.get('system_layers'))} "
                f"safety={p.get('safety_relevance')} "
                f"delivery={p.get('delivery_value')} "
                f"decision={p.get('corpus_decision')}"
            )

    print()
    print("Context summary:")
    print(f"  source_groups={len(source_groups)}")
    print(f"  selected_chunks={selected_chunks_total}")
    print(f"  expanded_chunks={expanded_chunks_total}")
    print(f"  context_chars={len(context)}")
    print(f"  approx_context_tokens={len(context) // 4}")
    print("  llm_context_contains_metadata=False")
    print("  answer_cleanup=strip_reasoning_after_think_and_keep_final_structure")
    print()

    if DEBUG:
        debug_print("CONTEXT SENT TO MODEL", context)

    answer = ask_model(question, context)

    print("=" * 100)
    print("ANSWER")
    print("=" * 100)
    print(answer)
    print()
    print("=" * 100)
    print("RAG ASK DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()