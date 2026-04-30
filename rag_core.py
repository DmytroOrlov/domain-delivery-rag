#!/usr/bin/env python3
"""
Shared dense-retrieval core for the local Domain Delivery RAG.

This module is the single source of truth for the v1 runtime retrieval contract.

Scope note:
  Query-profile rules, metadata-prior weights, answer contract, and metadata
  field mapping are loaded from the active domain config.

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
- The active domain's configured drop decision means the ingest pipeline should
  exclude the chunk from the Qdrant collection by default. Any retrieved drop
  chunk is treated as a configuration/indexing problem and receives the domain's
  configured metadata prior.

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

def _required_config_value(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context}.{key} is required in the active domain config")
    return mapping[key]


DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", str(_required_config_value(DOMAIN.retrieval_defaults, "top_k", "retrieval_defaults"))))
DEFAULT_PRE_K = int(os.environ.get("RAG_PRE_K", str(_required_config_value(DOMAIN.retrieval_defaults, "pre_k", "retrieval_defaults"))))
DEFAULT_MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", str(_required_config_value(DOMAIN.retrieval_defaults, "max_per_file", "retrieval_defaults"))))
DEFAULT_NEIGHBOR_RADIUS = int(os.environ.get("RAG_NEIGHBOR_RADIUS", str(_required_config_value(DOMAIN.retrieval_defaults, "neighbor_radius", "retrieval_defaults"))))

SELECTED_MAX_CHARS = int(os.environ.get("RAG_SELECTED_MAX_CHARS", str(_required_config_value(DOMAIN.context_defaults, "selected_max_chars", "context_defaults"))))
NEIGHBOR_SNIPPET_CHARS = int(os.environ.get("RAG_NEIGHBOR_SNIPPET_CHARS", str(_required_config_value(DOMAIN.context_defaults, "neighbor_snippet_chars", "context_defaults"))))
CONTEXT_MAX_CHARS = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", str(_required_config_value(DOMAIN.context_defaults, "context_max_chars", "context_defaults"))))

# Bounded metadata rerank. This is a heuristic, not a learned ranker.
# It can reorder close dense candidates; this is intentional but must remain
# measurable. `eval_metadata_ablation.py` compares raw dense retrieval against
# dense+metadata reranking so weight changes are reviewable instead of guessed.
#
# RAG_METADATA_PRIOR=0 disables the heuristic globally for normal callers.
# Ablation code can also override it per call through retrieve_dense(...,
# use_metadata_prior=False).
RERANK_CONFIG = dict(getattr(DOMAIN, "rerank", {}) or {})
RERANK_CLAMP = _required_config_value(RERANK_CONFIG, "clamp", "rerank")
META_MIN = float(os.environ.get("RAG_META_MIN", str(RERANK_CLAMP[0])))
META_MAX = float(os.environ.get("RAG_META_MAX", str(RERANK_CLAMP[1])))
def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


ALLOWED_RERANK_MODES = {"full", "value_weights_only", "disabled"}

def _configured_rerank_mode() -> str:
    """Return the domain-configured rerank mode, optionally overridden for eval.

    `rerank.mode` is production domain policy. `RAG_RERANK_MODE` exists only as
    an explicit run/eval override so ablation variants can compare current
    production behavior against full or disabled rerank without editing JSON.
    """
    raw = os.environ.get("RAG_RERANK_MODE", str(_required_config_value(RERANK_CONFIG, "mode", "rerank")))
    mode = raw.strip().lower()
    if mode not in ALLOWED_RERANK_MODES:
        raise ValueError(f"Unsupported rerank.mode/RAG_RERANK_MODE {raw!r}; allowed: {sorted(ALLOWED_RERANK_MODES)}")
    return mode


RERANK_MODE = _configured_rerank_mode()


def _parse_retrieval_ensemble_modes() -> list[str]:
    """Return optional deterministic retrieval-ensemble modes for eval/study runs.

    This is intentionally an explicit runtime override, not domain policy. It
    lets eval compare "run one ranker" with "union evidence from two rankers"
    without changing ingestion or the domain config. Example:

        RAG_RETRIEVAL_ENSEMBLE=disabled,value_weights_only

    Normal production/runtime calls leave this unset and use the single active
    rerank mode from the domain config / RAG_RERANK_MODE override.
    """
    raw = os.environ.get("RAG_RETRIEVAL_ENSEMBLE", "").strip()
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode not in ALLOWED_RERANK_MODES:
            raise ValueError(
                f"Unsupported RAG_RETRIEVAL_ENSEMBLE mode {mode!r}; "
                f"allowed: {sorted(ALLOWED_RERANK_MODES)}"
            )
        if mode not in seen:
            out.append(mode)
            seen.add(mode)
    return out


RETRIEVAL_ENSEMBLE_MODES = _parse_retrieval_ensemble_modes()

# RAG_METADATA_PRIOR remains a coarse production kill switch.
# RAG_ABLATION_NO_METADATA_RERANK is a one-way eval switch. Neither can
# accidentally enable rerank if the domain config says mode=disabled.
METADATA_PRIOR_ENABLED = (
    RERANK_MODE != "disabled"
    and os.environ.get("RAG_METADATA_PRIOR", "1") != "0"
    and not env_flag("RAG_ABLATION_NO_METADATA_RERANK")
)
EFFECTIVE_RERANK_MODE = RERANK_MODE if METADATA_PRIOR_ENABLED else "disabled"

ABLATION_NO_QUERY_PROFILES = env_flag("RAG_ABLATION_NO_QUERY_PROFILES")
ABLATION_NO_ROLE_HITS = env_flag("RAG_ABLATION_NO_ROLE_HITS_IN_RERANK")
ABLATION_NO_FACET_HITS = env_flag("RAG_ABLATION_NO_FACETS_IN_RERANK")
ABLATION_NO_LAYER_HITS = env_flag("RAG_ABLATION_NO_LAYERS_IN_RERANK")
ABLATION_NO_STAGE_HITS = env_flag("RAG_ABLATION_NO_STAGES_IN_RERANK")
ABLATION_NO_FLAG_HITS = env_flag("RAG_ABLATION_NO_FLAGS_IN_RERANK")
ABLATION_NO_CONFIDENCE = env_flag("RAG_ABLATION_NO_CONFIDENCE_IN_RERANK")
ABLATION_NO_VALUE_WEIGHTS = env_flag("RAG_ABLATION_NO_VALUE_WEIGHTS_IN_RERANK")

# Optional comma-separated allow-list for query profile names. This does not
# delete profiles from the domain config; it only lets eval runs measure whether
# a smaller hot-path subset is enough for a domain.
_PROFILE_ALLOWLIST_RAW = os.environ.get("RAG_ABLATION_QUERY_PROFILE_NAMES", "").strip()
ABLATION_QUERY_PROFILE_NAMES = {
    x.strip() for x in _PROFILE_ALLOWLIST_RAW.split(",") if x.strip()
}

VERBOSE = os.environ.get("RAG_VERBOSE", "1") != "0"
DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"

def metadata_fields_config() -> dict[str, Any]:
    return dict(getattr(DOMAIN, "metadata_fields", {}) or {})


def metadata_field(logical_name: str) -> str:
    """Return the payload field name for a logical metadata concept.

    Chunk-level logical fields are defined by DOMAIN.metadata_fields. Document-level
    aliases are defined by DOMAIN.metadata_field_map. Missing entries fail fast.
    """
    fields = metadata_fields_config()
    field_cfg = fields.get(logical_name) if isinstance(fields.get(logical_name), dict) else {}
    payload = field_cfg.get("payload") if isinstance(field_cfg, dict) else None
    if isinstance(payload, str) and payload.strip():
        return payload.strip()

    field_map = getattr(DOMAIN, "metadata_field_map", {}) or {}
    value = field_map.get(logical_name)
    if isinstance(value, str) and value.strip():
        return value.strip()

    raise ValueError(f"No payload mapping for logical metadata field {logical_name!r} in active domain config")


def boolean_flag_fields() -> list[str]:
    schema = getattr(DOMAIN, "metadata_schema", {}) or {}
    flags = _required_config_value(schema, "boolean_flags", "metadata_schema")
    return [str(flag) for flag in flags]


def payload_metadata(payload: dict[str, Any], logical_name: str, default: Any = None) -> Any:
    return payload.get(metadata_field(logical_name), default)


def payload_decision(payload: dict[str, Any]) -> Any:
    return payload_metadata(payload, "decision")


def payload_delivery_value(payload: dict[str, Any]) -> Any:
    return payload_metadata(payload, "delivery_value")


def payload_criticality(payload: dict[str, Any]) -> Any:
    return payload_metadata(payload, "criticality")


def payload_role(payload: dict[str, Any]) -> Any:
    return payload_metadata(payload, "role")


def payload_facets(payload: dict[str, Any]) -> Any:
    return payload_metadata(payload, "facets")


def payload_layers(payload: dict[str, Any]) -> Any:
    return payload_metadata(payload, "layers")


def payload_stages(payload: dict[str, Any]) -> Any:
    return payload_metadata(payload, "stages")


def answer_config() -> dict[str, Any]:
    return dict(getattr(DOMAIN, "answer", {}) or {})


def _required_answer_value(key: str) -> Any:
    cfg = answer_config()
    if key not in cfg:
        raise ValueError(f"answer.{key} is required in the active domain config")
    return cfg[key]


def answer_persona() -> str:
    return str(_required_answer_value("persona"))


def answer_grounding_rules() -> list[str]:
    rules = _required_answer_value("grounding_rules")
    return [str(rule) for rule in rules if str(rule).strip()]


def answer_citation_rule() -> str:
    return str(_required_answer_value("citation_rule"))


def answer_sections() -> list[str]:
    sections = _required_answer_value("sections")
    return [str(section) for section in sections if str(section).strip()]


def answer_repair_rules() -> list[str]:
    rules = _required_answer_value("repair_rules")
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

    A rule hits if any configured trigger appears as a substring in the
    lower-cased query or as an exact token. That keeps multi-word triggers and
    short technical triggers working without embedding domain terms in code.
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

    if ABLATION_NO_QUERY_PROFILES:
        return profile

    for rule in getattr(DOMAIN, "query_profiles", []):
        if ABLATION_QUERY_PROFILE_NAMES:
            rule_name = str(rule.get("name") or "").strip()
            if rule_name not in ABLATION_QUERY_PROFILE_NAMES:
                continue
        if not _query_profile_rule_matches(q, tokens, rule):
            continue

        if not ABLATION_NO_ROLE_HITS:
            profile["roles"].update(_as_list(rule.get("add_roles")))
        if not ABLATION_NO_FACET_HITS:
            profile["facets"].update(_as_list(rule.get("add_facets")))
        if not ABLATION_NO_LAYER_HITS:
            profile["layers"].update(_as_list(rule.get("add_layers")))
        if not ABLATION_NO_STAGE_HITS:
            profile["stages"].update(_as_list(rule.get("add_stages")))
        if not ABLATION_NO_FLAG_HITS:
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
    per_hit = float(_required_config_value(hit_cfg, "per_hit", "rerank.hit_weights"))
    cap = float(_required_config_value(hit_cfg, "cap", "rerank.hit_weights"))
    return min(hits * per_hit, cap)


def _value_weight_for(
    weights: dict[str, Any],
    logical_name: str,
    payload: dict[str, Any],
) -> float:
    field = metadata_field(logical_name)
    value = payload.get(field)

    # Prefer the concrete payload field name. Domain packs may also key weights by
    # logical concept name for portability across payload schemas.
    key = field if field in weights else logical_name
    return _value_weight(weights, key, value)


def metadata_prior(payload: dict[str, Any], query: str = "", mode: str | None = None) -> float:
    """
    Bounded heuristic rerank.

    Classification metadata is a routing/ranking signal only. It is not placed in
    answer evidence and should not be cited by the model.

    The mechanism is generic; the weights and metadata field mapping are domain
    policy loaded from the active domain config. `mode` is only for eval/study
    code that needs to score the same dense candidate set under multiple rankers
    in one process.
    """
    effective_mode = (mode or EFFECTIVE_RERANK_MODE).strip().lower()
    if effective_mode not in ALLOWED_RERANK_MODES:
        raise ValueError(f"Unsupported rerank mode {effective_mode!r}; allowed: {sorted(ALLOWED_RERANK_MODES)}")
    if effective_mode == "disabled":
        return 0.0

    bonus = 0.0
    rerank = RERANK_CONFIG
    base_weights = _required_config_value(rerank, "base_weights", "rerank")
    hit_weights = _required_config_value(rerank, "hit_weights", "rerank")

    use_value_weights = effective_mode in {"full", "value_weights_only"} and not ABLATION_NO_VALUE_WEIGHTS
    use_confidence_weight = effective_mode in {"full", "value_weights_only"} and not ABLATION_NO_CONFIDENCE
    use_hit_weights = effective_mode == "full"

    if use_value_weights:
        bonus += _value_weight_for(base_weights, "decision", payload)
        bonus += _value_weight_for(base_weights, "delivery_value", payload)
        bonus += _value_weight_for(base_weights, "criticality", payload)
        bonus += _value_weight_for(base_weights, "role", payload)

    if use_confidence_weight:
        try:
            confidence_weight = float(_required_config_value(rerank, "confidence_weight", "rerank"))
            bonus += min(float(payload.get("confidence") or 0.0), 1.0) * confidence_weight
        except Exception:
            pass

    profile = query_profile(query)

    role_hits = _intersect_count(payload_role(payload), profile["roles"])
    facet_hits = _intersect_count(payload_facets(payload), profile["facets"])
    layer_hits = _intersect_count(payload_layers(payload), profile["layers"])
    stage_hits = _intersect_count(payload_stages(payload), profile["stages"])

    if use_hit_weights and not ABLATION_NO_ROLE_HITS:
        bonus += _hit_weight(_required_config_value(hit_weights, "role", "rerank.hit_weights"), role_hits)
    if use_hit_weights and not ABLATION_NO_FACET_HITS:
        bonus += _hit_weight(_required_config_value(hit_weights, "facet", "rerank.hit_weights"), facet_hits)
    if use_hit_weights and not ABLATION_NO_LAYER_HITS:
        bonus += _hit_weight(_required_config_value(hit_weights, "layer", "rerank.hit_weights"), layer_hits)
    if use_hit_weights and not ABLATION_NO_STAGE_HITS:
        bonus += _hit_weight(_required_config_value(hit_weights, "stage", "rerank.hit_weights"), stage_hits)

    if use_hit_weights and not ABLATION_NO_FLAG_HITS:
        flag_weight = float(_required_config_value(hit_weights, "flag", "rerank.hit_weights"))
        for flag in profile["flags"]:
            if payload.get(flag) is True:
                bonus += flag_weight

    return _clamp(bonus, META_MIN, META_MAX)

# =============================================================================
# Dense retrieval and neighbor expansion
# =============================================================================

def _candidate_key(item: dict[str, Any]) -> tuple[str, int | str, str]:
    payload = item.get("payload") or {}
    file_path = str(payload.get("file_path") or "")
    chunk_index = payload.get("chunk_index")
    try:
        chunk_key: int | str = int(chunk_index)
    except Exception:
        chunk_key = str(chunk_index or "")
    fallback = str(payload.get("chunk_id") or payload.get("source_id") or payload.get("text", "")[:80])
    return (file_path, chunk_key, fallback)


def _rank_raw_points(
    raw: list[Any],
    question: str,
    mode: str,
    use_metadata_prior: bool,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for rank, point in enumerate(raw, start=1):
        payload = point.payload or {}
        vector_score = float(point.score)
        meta_bonus = metadata_prior(payload, query=question, mode=mode) if use_metadata_prior else 0.0
        final_score = vector_score + meta_bonus
        item = RetrievalItem(
            payload=payload,
            dense_rank=rank,
            vector_score=vector_score,
            meta_bonus=meta_bonus,
            final_score=final_score,
            is_selected_hit=True,
        ).as_dict()
        item["rerank_mode"] = mode
        candidates.append(item)

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates


def _select_diverse(candidates: list[dict[str, Any]], top_k: int, max_per_file: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
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
    return selected


def _select_ensemble_union(
    ranked_by_mode: dict[str, list[dict[str, Any]]],
    top_k: int,
    max_per_file: int,
) -> list[dict[str, Any]]:
    """Stable union of each ensemble ranker's selected top-k.

    This intentionally may return more than `top_k` items: it is an experimental
    evidence-union candidate for eval, not a replacement for the normal single
    ranker. Neighbor/context caps still bound the final prompt.
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int | str, str]] = set()
    selected_rank_by_mode: dict[tuple[str, int | str, str], dict[str, int]] = {}

    for mode in RETRIEVAL_ENSEMBLE_MODES:
        mode_selected = _select_diverse(ranked_by_mode[mode], top_k=top_k, max_per_file=max_per_file)
        for rank, item in enumerate(mode_selected, start=1):
            key = _candidate_key(item)
            selected_rank_by_mode.setdefault(key, {})[mode] = rank
            if key in seen:
                continue
            seen.add(key)
            copied = dict(item)
            copied["retrieval_ensemble_modes"] = list(RETRIEVAL_ENSEMBLE_MODES)
            copied["retrieval_ensemble_selected_by"] = [mode]
            out.append(copied)

    for item in out:
        key = _candidate_key(item)
        ranks = selected_rank_by_mode.get(key, {})
        item["retrieval_ensemble_selected_rank_by_mode"] = ranks
        item["retrieval_ensemble_selected_by"] = [m for m in RETRIEVAL_ENSEMBLE_MODES if m in ranks]

    return out


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

    # Normal runtime uses one ranker. Study mode can request a deterministic
    # evidence union from multiple rankers by setting RAG_RETRIEVAL_ENSEMBLE.
    # Explicit retrieve_dense(..., use_metadata_prior=True/False) calls keep the
    # old single-ranker behavior, which matters for legacy ablation tests.
    if use_metadata_prior is None and RETRIEVAL_ENSEMBLE_MODES:
        ranked_by_mode = {
            mode: _rank_raw_points(
                raw,
                question=question,
                mode=mode,
                use_metadata_prior=(mode != "disabled"),
            )
            for mode in RETRIEVAL_ENSEMBLE_MODES
        }
        selected = _select_ensemble_union(ranked_by_mode, top_k=top_k, max_per_file=max_per_file)
        # Return the first ensemble member's full candidate list as the dense/rank
        # diagnostic backbone, and annotate it so run artifacts make the ensemble
        # behavior explicit.
        candidates = ranked_by_mode[RETRIEVAL_ENSEMBLE_MODES[0]]
        for item in candidates:
            item["retrieval_ensemble_modes"] = list(RETRIEVAL_ENSEMBLE_MODES)
        return selected, candidates

    candidates = _rank_raw_points(
        raw,
        question=question,
        mode=EFFECTIVE_RERANK_MODE,
        use_metadata_prior=bool(use_metadata_prior),
    )
    selected = _select_diverse(candidates, top_k=top_k, max_per_file=max_per_file)

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



def ablation_config_summary() -> dict[str, Any]:
    return {
        "rerank_mode_configured": RERANK_MODE,
        "rerank_mode_effective": EFFECTIVE_RERANK_MODE,
        "no_metadata_rerank": env_flag("RAG_ABLATION_NO_METADATA_RERANK"),
        "no_query_profiles": ABLATION_NO_QUERY_PROFILES,
        "query_profile_names": sorted(ABLATION_QUERY_PROFILE_NAMES),
        "no_role_hits_in_rerank": ABLATION_NO_ROLE_HITS,
        "no_facets_in_rerank": ABLATION_NO_FACET_HITS,
        "no_layers_in_rerank": ABLATION_NO_LAYER_HITS,
        "no_stages_in_rerank": ABLATION_NO_STAGE_HITS,
        "no_flags_in_rerank": ABLATION_NO_FLAG_HITS,
        "no_confidence_in_rerank": ABLATION_NO_CONFIDENCE,
        "no_value_weights_in_rerank": ABLATION_NO_VALUE_WEIGHTS,
        "retrieval_ensemble_modes": list(RETRIEVAL_ENSEMBLE_MODES),
    }


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
        "rerank_mode_configured": RERANK_MODE,
        "rerank_mode_effective": EFFECTIVE_RERANK_MODE,
        "retrieval_ensemble_modes": list(RETRIEVAL_ENSEMBLE_MODES),
        "metadata_rerank_clamp": f"[{META_MIN:+.3f},{META_MAX:+.3f}]",
        "ablation": ablation_config_summary(),
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
