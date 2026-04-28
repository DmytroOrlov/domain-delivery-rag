#!/usr/bin/env python3
"""
Shared dense-retrieval core for the local ADAS / Embedded Vision Delivery RAG v1.

This module is the single source of truth for the v1 runtime retrieval contract.

Scope note:
  Query-profile rules and metadata-prior weights are loaded from the active
  domain config. Ingest ontology is still domain-specific in code and is
  expected to move into the domain pack in a later refactor.

Runtime contract:

    question
    -> embedding server
    -> Qdrant dense retrieval
    -> bounded metadata rerank
    -> per-file diversity
    -> neighbor expansion
    -> evidence-only context packing

Important semantics:
- Hybrid lexical/RRF is intentionally not used here. A naive hybrid experiment
  improved exact-term recall but hurt broad semantic questions, so v1 keeps the
  proven dense baseline as default.
- Classification metadata is used only for retrieval/reranking/diagnostics.
  It is not used as answer evidence.
- The LLM context contains only source provenance (source id, sanitized file
  name, chunk indices) plus chunk text. Full local file paths are deliberately
  excluded from the answer prompt.
- corpus_decision="drop" means the ingest pipeline should exclude the chunk from
  the Qdrant collection by default. Any retrieved drop chunk is treated as a
  configuration/indexing problem and receives a negative metadata prior.

Used by:
- rag_proxy.py        browser /rag UX through stock llama.cpp UI
- search.py           retrieval microscope and context debugger
- eval_retrieval.py   retrieval-only regression eval
- ask_qwen.py         optional legacy/debug CLI if kept
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import requests
from domain_config import load_domain_config
from qdrant_client import QdrantClient
from qdrant_client.http import models

# =============================================================================
# Configuration
# =============================================================================

DOMAIN = load_domain_config()

EMBED_URL = os.environ.get("RAG_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")
QDRANT_URL = os.environ.get("RAG_QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("RAG_COLLECTION", DOMAIN.collection)

DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", str(DOMAIN.retrieval_defaults.get("top_k", 5))))
DEFAULT_PRE_K = int(os.environ.get("RAG_PRE_K", str(DOMAIN.retrieval_defaults.get("pre_k", 24))))
DEFAULT_MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", str(DOMAIN.retrieval_defaults.get("max_per_file", 2))))
DEFAULT_NEIGHBOR_RADIUS = int(os.environ.get("RAG_NEIGHBOR_RADIUS", str(DOMAIN.retrieval_defaults.get("neighbor_radius", 1))))

SELECTED_MAX_CHARS = int(os.environ.get("RAG_SELECTED_MAX_CHARS", str(DOMAIN.context_defaults.get("selected_max_chars", 2200))))
NEIGHBOR_SNIPPET_CHARS = int(os.environ.get("RAG_NEIGHBOR_SNIPPET_CHARS", str(DOMAIN.context_defaults.get("neighbor_snippet_chars", 700))))
CONTEXT_MAX_CHARS = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", str(DOMAIN.context_defaults.get("context_max_chars", 18000))))

# Bounded metadata rerank. This is a heuristic, not a learned ranker.
# It can reorder close dense candidates; this is intentional but must remain
# measurable. `eval_metadata_ablation.py` compares raw dense retrieval against
# dense+metadata reranking so weight changes are reviewable instead of guessed.
#
# RAG_METADATA_PRIOR=0 disables the heuristic globally for normal callers.
# Ablation code can also override it per call through retrieve_dense(...,
# use_metadata_prior=False).
RERANK_CONFIG = getattr(DOMAIN, "rerank", {}) or {}
RERANK_CLAMP = RERANK_CONFIG.get("clamp", [-0.045, 0.050])
META_MIN = float(os.environ.get("RAG_META_MIN", str(RERANK_CLAMP[0])))
META_MAX = float(os.environ.get("RAG_META_MAX", str(RERANK_CLAMP[1])))
METADATA_PRIOR_ENABLED = os.environ.get("RAG_METADATA_PRIOR", "1") != "0"

VERBOSE = os.environ.get("RAG_VERBOSE", "1") != "0"
DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"

DEFAULT_ANSWER_GROUNDING_RULES = [
    "Use the retrieved context first.",
    "The retrieved chunk text is the source of truth.",
    "Use source ids, file names, and chunk ids only for citation/source mapping.",
    "Separate supported facts from inference.",
    "If the retrieved context does not support a claim, say so explicitly.",
    "Be conservative with safety-relevant claims.",
]

DEFAULT_CITATION_RULE = (
    "Citation rule: use only exact citations like [S1], [S2]. Do not put chunk ids, "
    "file names, commas, or extra text inside citation brackets. No [S#] citation "
    "means no claim. Cite every factual claim, inference, limitation, and "
    '"not specified" statement.'
)

DEFAULT_ANSWER_SECTIONS = [
    "Conclusion",
    "Supported facts",
    "Inferences",
    "Implementation implications",
    "Unknowns / verification needed",
    "Source mapping",
]

DEFAULT_REPAIR_RULES = [
    "Use the same retrieved context from the original prompt.",
    "Use exact source citations like [S1], [S2]. Do not put chunk ids, file names, commas, or extra text inside citation brackets.",
    "No [S#] citation means no claim.",
    "For missing evidence, cite the retrieved sources reviewed and state what is not specified.",
    "Do not reproduce malformed tables, orphaned table captions, or repeated list items from context.",
    "Do not repeat the same phrase or bullet.",
    "Keep the answer concise.",
]


def answer_config() -> dict[str, Any]:
    return dict(getattr(DOMAIN, "answer", {}) or {})


def answer_persona() -> str:
    cfg = answer_config()
    return str(cfg.get("persona") or getattr(DOMAIN, "answer_persona", "You are a senior domain delivery assistant."))


def answer_grounding_rules() -> list[str]:
    cfg = answer_config()
    rules = cfg.get("grounding_rules") or DEFAULT_ANSWER_GROUNDING_RULES
    return [str(rule) for rule in rules if str(rule).strip()]


def answer_citation_rule() -> str:
    cfg = answer_config()
    return str(cfg.get("citation_rule") or DEFAULT_CITATION_RULE)


def answer_sections() -> list[str]:
    cfg = answer_config()
    sections = cfg.get("sections") or DEFAULT_ANSWER_SECTIONS
    return [str(section) for section in sections if str(section).strip()]


def answer_repair_rules() -> list[str]:
    cfg = answer_config()
    rules = cfg.get("repair_rules") or DEFAULT_REPAIR_RULES
    return [str(rule) for rule in rules if str(rule).strip()]


def render_answer_format(sections: list[str] | None = None) -> str:
    sections = sections or answer_sections()
    return "\n".join(f"{idx}. {section}" for idx, section in enumerate(sections, start=1))


def render_answer_prompt(question: str, context: str) -> str:
    """Render the domain answer prompt from the active domain config.

    Retrieval/context assembly is engine code; answer persona, grounding rules,
    citation policy, and section contract are domain policy.
    """
    grounding = "\n".join(answer_grounding_rules())
    return f"""{answer_persona()}

{grounding}

{answer_citation_rule()}

Answer format:
{render_answer_format()}

Question:
{question}

Retrieved context:
{context}
"""


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class RetrievalItem:
    payload: dict[str, Any]
    dense_rank: int | None
    vector_score: float | None
    meta_bonus: float | None
    final_score: float | None
    is_selected_hit: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "payload": self.payload,
            "dense_rank": self.dense_rank,
            "vector_score": self.vector_score,
            "meta_bonus": self.meta_bonus,
            "final_score": self.final_score,
            "is_selected_hit": self.is_selected_hit,
        }


# =============================================================================
# Logging and formatting helpers
# =============================================================================

def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)


def debug_print(title: str, value: Any):
    if not DEBUG:
        return
    print("=" * 100)
    print(f"DEBUG: {title}")
    print("=" * 100)
    if isinstance(value, (dict, list)):
        print(json.dumps(value, ensure_ascii=False, indent=2)[:30000])
    else:
        print(str(value)[:30000])
    print()


def fmt_score(value: Any) -> str:
    if value is None:
        return "none"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def fmt_list(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)


def file_name(payload: dict[str, Any]) -> str:
    return payload.get("file_name") or os.path.basename(payload.get("file_path", "unknown"))


# =============================================================================
# Embedding
# =============================================================================

def embed(text: str) -> list[float]:
    r = requests.post(EMBED_URL, json={"input": text}, timeout=120)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


# =============================================================================
# Query profiling and bounded metadata reranking
# =============================================================================

def tokenize_query(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9_#+.-]{2,}", query.lower())
    stop = {
        "the", "and", "for", "with", "from", "into", "what", "does", "this",
        "that", "are", "how", "why", "when", "where", "which", "about",
        "likely", "implications", "context", "corpus",
        "что", "как", "для", "или", "это", "про", "при", "над", "под",
    }
    return [t for t in tokens if t not in stop]


def _empty_query_profile() -> dict[str, set[str]]:
    return {
        "roles": set(),
        "facets": set(),
        "layers": set(),
        "stages": set(),
        "flags": set(),
    }


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def _query_profile_rule_matches(query_lower: str, tokens: set[str], rule: dict[str, Any]) -> bool:
    """
    Match a domain-configured query-profile rule.

    The ADAS domain pack intentionally preserves the old behavior: a rule hits
    if any configured trigger appears as a substring in the lower-cased query or
    as an exact token. That keeps multi-word triggers such as "blind spot" and
    short technical triggers such as "CAN" working as before.
    """
    triggers = _as_list(rule.get("triggers"))
    if not triggers:
        return False

    return any(
        trigger.lower() in query_lower or trigger.lower() in tokens
        for trigger in triggers
    )


def query_profile(query: str) -> dict[str, set[str]]:
    q = query.lower()
    tokens = set(tokenize_query(query))
    profile = _empty_query_profile()

    for rule in getattr(DOMAIN, "query_profiles", []):
        if not _query_profile_rule_matches(q, tokens, rule):
            continue

        profile["roles"].update(_as_list(rule.get("add_roles")))
        profile["facets"].update(_as_list(rule.get("add_facets")))
        profile["layers"].update(_as_list(rule.get("add_layers")))
        profile["stages"].update(_as_list(rule.get("add_stages")))
        profile["flags"].update(_as_list(rule.get("add_flags")))

    return profile


def pretty_profile(profile: dict[str, set[str]]) -> dict[str, list[str]]:
    return {k: sorted(v) for k, v in profile.items()}


def _intersect_count(payload_value: Any, wanted: set[str]) -> int:
    if not wanted:
        return 0
    if isinstance(payload_value, list):
        return len(set(payload_value) & wanted)
    if isinstance(payload_value, str):
        return 1 if payload_value in wanted else 0
    return 0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _value_weight(weights: dict[str, Any], field: str, value: Any) -> float:
    field_weights = weights.get(field, {})
    try:
        return float(field_weights.get(str(value), 0.0))
    except Exception:
        return 0.0


def _hit_weight(hit_cfg: dict[str, Any], hits: int) -> float:
    if not hits:
        return 0.0
    per_hit = float(hit_cfg.get("per_hit", 0.0))
    cap = float(hit_cfg.get("cap", per_hit * hits))
    return min(hits * per_hit, cap)


def metadata_prior(payload: dict[str, Any], query: str = "") -> float:
    """
    Bounded heuristic rerank.

    Classification metadata is a routing/ranking signal only. It is not placed in
    answer evidence and should not be cited by the model.

    The mechanism is generic; the weights are domain policy and are loaded from
    the active domain config.
    """
    bonus = 0.0
    rerank = RERANK_CONFIG
    base_weights = rerank.get("base_weights", {})
    hit_weights = rerank.get("hit_weights", {})

    bonus += _value_weight(base_weights, "corpus_decision", payload.get("corpus_decision"))
    bonus += _value_weight(base_weights, "delivery_value", payload.get("delivery_value"))
    bonus += _value_weight(base_weights, "safety_relevance", payload.get("safety_relevance"))
    bonus += _value_weight(base_weights, "chunk_role", payload.get("chunk_role"))

    try:
        confidence_weight = float(rerank.get("confidence_weight", 0.0))
        bonus += min(float(payload.get("confidence") or 0.0), 1.0) * confidence_weight
    except Exception:
        pass

    profile = query_profile(query)

    role_hits = _intersect_count(payload.get("chunk_role"), profile["roles"])
    facet_hits = _intersect_count(payload.get("content_facets"), profile["facets"])
    layer_hits = _intersect_count(payload.get("system_layers"), profile["layers"])
    stage_hits = _intersect_count(payload.get("workflow_stages"), profile["stages"])

    bonus += _hit_weight(hit_weights.get("role", {}), role_hits)
    bonus += _hit_weight(hit_weights.get("facet", {}), facet_hits)
    bonus += _hit_weight(hit_weights.get("layer", {}), layer_hits)
    bonus += _hit_weight(hit_weights.get("stage", {}), stage_hits)

    flag_weight = float(hit_weights.get("flag", 0.0))
    for flag in profile["flags"]:
        if payload.get(flag) is True:
            bonus += flag_weight

    return _clamp(bonus, META_MIN, META_MAX)


# =============================================================================
# Dense retrieval and neighbor expansion
# =============================================================================

def retrieve_dense(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    pre_k: int = DEFAULT_PRE_K,
    max_per_file: int = DEFAULT_MAX_PER_FILE,
    use_metadata_prior: bool | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return selected candidates and all pre-diversity candidates.

    Ranking contract:
      dense_raw:       final_score = dense cosine score
      dense+metadata:  final_score = dense cosine score + bounded metadata_prior

    `use_metadata_prior=None` follows RAG_METADATA_PRIOR. Ablation callers pass
    True/False explicitly so both modes can be compared in one process.
    """
    if use_metadata_prior is None:
        use_metadata_prior = METADATA_PRIOR_ENABLED

    qvec = embed(question)
    client = QdrantClient(url=QDRANT_URL)
    actual_pre_k = max(pre_k, top_k * 4)

    raw = client.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=actual_pre_k,
        with_payload=True,
    ).points

    candidates = []
    for rank, point in enumerate(raw, start=1):
        payload = point.payload or {}
        vector_score = float(point.score)
        meta_bonus = metadata_prior(payload, query=question) if use_metadata_prior else 0.0
        final_score = vector_score + meta_bonus
        candidates.append(
            RetrievalItem(
                payload=payload,
                dense_rank=rank,
                vector_score=vector_score,
                meta_bonus=meta_bonus,
                final_score=final_score,
                is_selected_hit=True,
            ).as_dict()
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    selected = []
    per_file: dict[str, int] = {}
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


def fetch_chunk_payloads_for_file(client: QdrantClient, file_path: str, indices: set[int]) -> dict[int, dict[str, Any]]:
    if not indices:
        return {}

    min_idx = min(indices)
    max_idx = max(indices)
    out: dict[int, dict[str, Any]] = {}
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="file_path", match=models.MatchValue(value=file_path)),
                    models.FieldCondition(key="chunk_index", range=models.Range(gte=min_idx, lte=max_idx)),
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


def expand_results_with_neighbors(
    results: list[dict[str, Any]],
    radius: int = DEFAULT_NEIGHBOR_RADIUS,
) -> list[dict[str, Any]]:
    """
    Group selected chunks by file and include radius-neighbors from the same file.

    Selected chunks keep scores. Neighbor-only chunks have score fields as None.
    """
    client = QdrantClient(url=QDRANT_URL)

    requested_by_file: dict[str, set[int]] = {}
    selected_by_file: dict[str, set[int]] = {}
    best_rank_by_file: dict[str, int] = {}
    best_score_by_file: dict[str, float | None] = {}
    selected_lookup: dict[tuple[str, int], dict[str, Any]] = {}

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
        elif (item.get("final_score") or 0.0) > (best_score_by_file.get(file_path) or 0.0):
            best_score_by_file[file_path] = item.get("final_score")

        for idx in range(chunk_index - radius, chunk_index + radius + 1):
            if idx >= 0:
                requested_by_file[file_path].add(idx)

    groups = []
    file_order = sorted(requested_by_file, key=lambda fp: best_rank_by_file.get(fp, 999999))

    for group_rank, file_path in enumerate(file_order, start=1):
        requested_indices = requested_by_file[file_path]
        neighbor_payloads = fetch_chunk_payloads_for_file(client, file_path, requested_indices)

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
                item = RetrievalItem(
                    payload=payload,
                    dense_rank=None,
                    vector_score=None,
                    meta_bonus=None,
                    final_score=None,
                    is_selected_hit=False,
                ).as_dict()
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
                "file_name": file_name(first_payload),
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

def clip_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def best_snippet(text: str, question: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text

    terms = tokenize_query(question)
    if not terms:
        return clip_text(text, max_chars)

    lower = text.lower()
    best_pos = None
    for term in terms:
        pos = lower.find(term.lower())
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best_pos = pos

    if best_pos is None:
        return clip_text(text, max_chars)

    start = max(0, best_pos - max_chars // 3)
    end = min(len(text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)

    snippet = text[start:end].strip()
    if start > 0:
        snippet = "...[before]\n" + snippet
    if end < len(text):
        snippet = snippet + "\n...[after]"
    return snippet


def build_context(
    question: str,
    source_groups: list[dict[str, Any]],
    selected_max_chars: int = SELECTED_MAX_CHARS,
    neighbor_snippet_chars: int = NEIGHBOR_SNIPPET_CHARS,
    context_max_chars: int = CONTEXT_MAX_CHARS,
    include_full_path: bool = False,
) -> str:
    """
    Build the exact answer context contract used by proxy/search/eval.

    By default, full local paths are not included in the LLM context. Source
    provenance is limited to source id, file name, and chunk ids. Classification
    metadata such as roles/facets/reason_short is deliberately excluded.
    """
    blocks = []
    total_chars = 0

    for source_idx, group in enumerate(source_groups, start=1):
        source_id = f"S{source_idx}"
        header = (
            f"[{source_id}]\n"
            f"file={group.get('file_name', 'unknown')}\n"
            f"selected_chunks={group.get('selected_indices', [])}\n"
            f"expanded_chunks={group.get('expanded_indices', [])}\n"
        )
        if include_full_path:
            header += f"path={group.get('file_path', 'unknown')}\n"

        chunk_blocks = []
        for item in group["chunks"]:
            payload = item["payload"]
            chunk_index = payload.get("chunk_index")
            content = payload.get("content") or ""

            if item.get("is_selected_hit"):
                mode = "selected"
                content_out = clip_text(content, selected_max_chars)
            else:
                mode = "neighbor"
                content_out = best_snippet(content, question, neighbor_snippet_chars)

            chunk_blocks.append(f"--- chunk={chunk_index} mode={mode} ---\n{content_out}")

        block = header + "content:\n" + "\n\n".join(chunk_blocks)
        if total_chars + len(block) > context_max_chars:
            remaining = context_max_chars - total_chars
            if remaining > 1000:
                blocks.append(block[:remaining].rstrip() + "\n...[context truncated]")
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(blocks)


def build_augmented_prompt(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    pre_k: int = DEFAULT_PRE_K,
    max_per_file: int = DEFAULT_MAX_PER_FILE,
    neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
    selected_max_chars: int = SELECTED_MAX_CHARS,
    neighbor_snippet_chars: int = NEIGHBOR_SNIPPET_CHARS,
    context_max_chars: int = CONTEXT_MAX_CHARS,
) -> tuple[str, dict[str, Any]]:
    """
    Retrieve, expand, pack context, and build the final prompt for the model.

    Returns (prompt, debug_info). debug_info is for stdout/logs only, not for the
    LLM prompt.
    """
    selected, candidates = retrieve_dense(
        question=question,
        top_k=top_k,
        pre_k=pre_k,
        max_per_file=max_per_file,
    )
    source_groups = expand_results_with_neighbors(selected, radius=neighbor_radius)
    context = build_context(
        question=question,
        source_groups=source_groups,
        selected_max_chars=selected_max_chars,
        neighbor_snippet_chars=neighbor_snippet_chars,
        context_max_chars=context_max_chars,
        include_full_path=False,
    )

    prompt = render_answer_prompt(question=question, context=context)

    debug_info = {
        "top_k": top_k,
        "pre_k": max(pre_k, top_k * 4),
        "max_per_file": max_per_file,
        "neighbor_radius": neighbor_radius,
        "candidates_count": len(candidates),
        "selected_count": len(selected),
        "source_groups_count": len(source_groups),
        "context_chars": len(context),
        "prompt_chars": len(prompt),
        "answer_sections": answer_sections(),
        "source_groups": source_groups,
        "selected": selected,
        "candidates": candidates,
    }
    return prompt, debug_info


def retrieval_config_summary() -> dict[str, Any]:
    return {
        "domain_id": DOMAIN.id,
        "domain_display_name": DOMAIN.display_name,
        "mode": "dense",
        "embed_url": EMBED_URL,
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION,
        "top_k": DEFAULT_TOP_K,
        "pre_k": DEFAULT_PRE_K,
        "max_per_file": DEFAULT_MAX_PER_FILE,
        "neighbor_radius": DEFAULT_NEIGHBOR_RADIUS,
        "selected_max_chars": SELECTED_MAX_CHARS,
        "neighbor_snippet_chars": NEIGHBOR_SNIPPET_CHARS,
        "context_max_chars": CONTEXT_MAX_CHARS,
        "metadata_prior": "enabled" if METADATA_PRIOR_ENABLED else "disabled",
        "metadata_rerank_clamp": f"[{META_MIN:+.3f},{META_MAX:+.3f}]",
        "hybrid": "disabled/deferred",
    }


# =============================================================================
# Qwen / llama-server final-answer parsing for local debug/eval clients
# =============================================================================

def parse_qwen_final_answer(raw_content: str) -> tuple[str, dict[str, Any]]:
    """
    Parse the final answer from the default Qwen / llama-server message.content.

    This helper does not change server behavior and does not modify facts,
    citations, or section structure. It only removes visible Qwen thinking
    wrappers from the client-visible answer used by non-streaming debug/eval
    tools:
      - if </think> exists, keep only text after the last </think>;
      - otherwise use message.content as-is;
      - if there is preamble before "1. Conclusion", trim to that section;
      - if an unclosed <think> remains, report it in cleanup metadata.

    rag_proxy.py intentionally does not call this: browser streaming stays a
    transparent llama-server response.
    """
    raw = raw_content or ""
    text = raw.strip()
    cleanup = {
        "raw_answer_chars": len(raw),
        "parsed_answer_chars_before_section_trim": None,
        "removed_think_prefix": False,
        "removed_dangling_end_think": False,
        "trimmed_to_section_start": False,
        "unclosed_think_after_parse": False,
    }

    if "</think>" in text:
        before, after = text.rsplit("</think>", 1)
        text = after.strip()
        cleanup["removed_think_prefix"] = bool(before.strip())
        cleanup["removed_dangling_end_think"] = not bool(before.strip())

    text = re.sub(r"^```(?:markdown|text)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip()).strip()

    cleanup["parsed_answer_chars_before_section_trim"] = len(text)

    section_start = re.search(
        r"(?mi)^\s*(?:#{1,6}\s*)?1[.)]\s*(?:\*\*)?Conclusion\b",
        text,
    )
    if section_start and section_start.start() > 0:
        text = text[section_start.start():].strip()
        cleanup["trimmed_to_section_start"] = True

    if "<think>" in text and "</think>" not in text:
        cleanup["unclosed_think_after_parse"] = True

    cleanup["parsed_answer_chars"] = len(text)
    return text, cleanup
