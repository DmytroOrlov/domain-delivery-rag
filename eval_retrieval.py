#!/usr/bin/env python3
"""
Retrieval-only eval for the local Domain Delivery RAG.

This is intentionally NOT an answer-quality eval. It checks whether the shared
retrieval/context path can find expected source files/chunks and avoids forbidden
or dropped chunks.

This script uses rag_core.py, the same retrieval and context-packing contract as
rag_proxy.py and search.py.

Public-safe eval files
----------------------
`eval_queries.json` should not need to contain real/private file names. Prefer
logical source aliases:

  {
    "id": "case_id",
    "query": "...",
    "expected_sources_all": ["edge_ai_product"],
    "expected_sources_any": ["ai_model_verification_report"],
    "expected_expanded_chunks_any": [
      {"source": "edge_ai_product", "chunks": [0, 1]}
    ]
  }

Aliases are resolved through a local, untracked source map:

  RAG_EVAL_SOURCE_MAP=eval_source_map.local.json

The source map contains the real file names and should not be committed. Direct
file-name expectations are still supported for local/private use, but aliases are
the recommended review/public format. Cases without expected source/chunk
constraints are allowed for answer-only checks such as insufficient-evidence
queries; retrieval eval then only enforces generic index hygiene such as no
retrieved drop chunks.

Usage:
  python3 eval_retrieval.py
  python3 eval_retrieval.py blind_spot_warning_implications
  python3 eval_retrieval.py --case blind_spot_warning_implications

Fast iteration pattern:
  # Cheap sweep: retrieval-only, seconds instead of hours.
  python3 eval_retrieval.py --compare --retrieval-only

  # Expensive sweep: answer + optional LLM judge only for shortlisted variants/cases.
  python3 eval_retrieval.py --compare --answer --judge \
    --variant production_default --variant baseline_full --variant no_metadata_rerank \
    --case-file eval_runs/problem_cases.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import rag_core as rc

EVAL_FILE = Path(os.environ.get("RAG_EVAL_FILE", rc.DOMAIN.eval_file)).expanduser()
SOURCE_MAP_FILE = Path(os.environ.get("RAG_EVAL_SOURCE_MAP", rc.DOMAIN.eval_source_map)).expanduser()
DEFAULT_TOP_K = int(os.environ.get("RAG_EVAL_TOP_K", str(rc.DEFAULT_TOP_K)))
DEFAULT_PRE_K = int(os.environ.get("RAG_EVAL_PRE_K", str(rc.DEFAULT_PRE_K)))
DEFAULT_MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", str(rc.DEFAULT_MAX_PER_FILE)))
DEFAULT_NEIGHBOR_RADIUS = int(os.environ.get("RAG_NEIGHBOR_RADIUS", str(rc.DEFAULT_NEIGHBOR_RADIUS)))
RUNS_DIR = Path(
    os.environ.get(
        "RAG_EVAL_RUNS_DIR",
        os.environ.get("RAG_EVAL_RUN_DIR", rc.DOMAIN.eval_run_dir),
    )
).expanduser()


ABLATION_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "production_default",
        "description": "Active domain config as production runs it. Current ADAS champion is conservative rerank.mode=disabled; value_weights_only remains a measured challenger.",
        "env": {},
    },
    {
        "name": "baseline_full",
        "description": "Force full metadata rerank: value weights + query-profile hit weights.",
        "env": {"RAG_RERANK_MODE": "full"},
    },
    {
        "name": "no_metadata_rerank",
        "description": "Force dense retrieval only; metadata bonus is zero.",
        "env": {"RAG_RERANK_MODE": "disabled"},
    },
    {
        "name": "no_query_profiles",
        "description": "Force full mode but disable query-profile expansion entirely.",
        "env": {"RAG_RERANK_MODE": "full", "RAG_ABLATION_NO_QUERY_PROFILES": "1"},
    },
    {
        "name": "value_weights_only",
        "description": "Force value-only rerank: decision/delivery/criticality/role/confidence; no query-profile hit boosts.",
        "env": {"RAG_RERANK_MODE": "value_weights_only"},
    },
    {
        "name": "dual_union_disabled_value",
        "description": "Experimental two-ranker union: dense/disabled top-k plus value_weights_only top-k. Measures whether two rankers are worth the extra context/noise.",
        "env": {"RAG_RETRIEVAL_ENSEMBLE": "disabled,value_weights_only", "RAG_RERANK_MODE": "value_weights_only"},
    },
    {
        "name": "no_flags_in_rerank",
        "description": "Force full mode and disable boolean flag hit bonus only.",
        "env": {"RAG_RERANK_MODE": "full", "RAG_ABLATION_NO_FLAGS_IN_RERANK": "1"},
    },
    {
        "name": "no_layers_in_rerank",
        "description": "Force full mode and disable layer hit bonus only.",
        "env": {"RAG_RERANK_MODE": "full", "RAG_ABLATION_NO_LAYERS_IN_RERANK": "1"},
    },
    {
        "name": "no_stages_in_rerank",
        "description": "Force full mode and disable stage hit bonus only.",
        "env": {"RAG_RERANK_MODE": "full", "RAG_ABLATION_NO_STAGES_IN_RERANK": "1"},
    },
    {
        "name": "core_query_profiles",
        "description": "Force full mode but allow only high-risk/core ADAS profiles: validation/failure/interface/regulation.",
        "env": {
            "RAG_RERANK_MODE": "full",
            "RAG_ABLATION_QUERY_PROFILE_NAMES": "validation_and_test_evidence,failure_or_degraded_mode,interface_or_contract,regulation_compliance"
        },
    },
]

RETRIEVAL_COMPARE_KEYS = [
    "rerank_mode_effective",
    "retrieval_ensemble_modes",
    "metadata_prior_enabled",
    "pass_rate",
    "positive_retrieval_cases",
    "positive_expected_source_hit_at_1",
    "positive_expected_source_hit_at_3",
    "positive_expected_source_hit_at_5",
    "positive_expected_source_mrr_selected",
    "positive_selected_source_ndcg_at_5",
    "positive_metadata_rank_delta_dense_minus_final_avg",
    "query_profile_hit_count_avg",
    "selected_drop_count_total",
    "expanded_drop_count_total",
    "total_elapsed_sec_avg",
]

ANSWER_COMPARE_KEYS = [
    "pass_rate",
    "avg_answer_chars",
    "avg_citation_count",
    "avg_generation_elapsed_sec",
    "total_tokens",
    "repair_attempted_count",
    "repaired_count",
    "regex_overclaim_risk_cases",
    "judge_parse_failed_cases",
    "judge_semantic_failed_cases",
    "llm_judge_call_success_rate",
    "llm_judge_semantic_pass_rate",
    "llm_judge_end_to_end_pass_rate",
    "llm_judge_avg_overall_score",
    "llm_judge_avg_groundedness_score",
    "llm_judge_avg_citation_quality_score",
    "llm_judge_judge_pass_rate",
]

RANKING_KEYS = [
    "rank",
    "rerank_mode_effective",
    "retrieval_ensemble_modes",
    "recommendation",
    "score",
    "positive_hit_at_1",
    "positive_mrr",
    "retrieval_failed",
    "answer_pass_rate",
    "judge_pass_rate",
    "avg_overall",
    "avg_answer_sec",
    "total_tokens",
]


STUDY_VARIANTS = ["no_metadata_rerank", "value_weights_only", "baseline_full"]
STUDY_PRIMARY_A = "no_metadata_rerank"
STUDY_PRIMARY_B = "value_weights_only"

SLICE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("insufficient_evidence", ("unknown", "no_answer", "not specified", "insufficient evidence")),
    ("regulatory_scope", ("regulation", "regulatory", "compliance", "bsis", "mois", "r159", "r151")),
    ("validation_test", ("validation", "verification", "test", "scenario", "false positive", "false negative", "latency")),
    ("deployment_runtime", ("deploy", "deployment", "runtime", "embedded", "edge", "on_device", "cloud", "ecu")),
    ("interface_contract", ("interface", "can", "gpio", "signal", "contract")),
    ("failure_degraded_mode", ("failure", "degraded", "fallback", "rollback", "update", "recovery")),
    ("marketing_overclaim", ("marketing", "limitation", "low_light", "privacy", "pixelation")),
]

HARD_GATE_POLICY = {
    "positive_expected_source_hit_at_5_min": 1.0,
    "answer_pass_rate_min": 0.95,
    "judge_call_success_rate_min": 1.0,
    "judge_semantic_pass_rate_min": 0.90,
    "regex_overclaim_risk_max": 0,
    "insufficient_evidence_semantic_pass_min": 1.0,
}

# Keep the research trail in executable code rather than in a README that drifts.
# This ledger summarizes why the study harness exists and what prior runs taught us
# since structured metrics were added. It is also copied into study_report.json so
# future runs carry the assumptions/results that shaped the current decision rule.
RERANK_SELECTION_EVIDENCE_LEDGER: list[dict[str, Any]] = [
    {
        "hypothesis": "structured_metrics_required",
        "result": "Persisted retrieval/answer JSONL and summary metrics made rerank changes measurable instead of anecdotal.",
        "decision": "Keep structured eval artifacts as the baseline observability layer.",
    },
    {
        "hypothesis": "llm_judge_adds_useful_signal",
        "result": "LLM judge exposed unsupported inference and abstention-quality regressions that deterministic checks alone missed.",
        "decision": "Use judge as an advisory/gated semantic layer, but separate judge infrastructure failures from semantic failures.",
    },
    {
        "hypothesis": "eval_should_control_generation_policy",
        "result": "Eval-side max_tokens/temp/top_p defaults conflicted with the llama-server launch policy and caused misleading budget debates.",
        "decision": "Do not send generation overrides from eval; the server launch is the source of truth.",
    },
    {
        "hypothesis": "full_metadata_rerank_improves_adas",
        "result": "Full rerank repeatedly worsened rank deltas and produced more safety-tail semantic failures than simpler candidates.",
        "decision": "Keep full rerank as a lab/reference mode, not as the ADAS production default.",
    },
    {
        "hypothesis": "value_weights_only_keeps_the_useful_metadata_signal",
        "result": "Value-only rerank sometimes won focused runs, but was unstable on insufficient-evidence, regulatory, deployment, and latency slices.",
        "decision": "Keep value_weights_only as a challenger/study variant; require decisive paired uplift before promoting it.",
    },
    {
        "hypothesis": "disabled_rerank_is_too_primitive",
        "result": "Disabled/no_metadata_rerank most consistently cleared hard gates and avoided metadata-induced over-inference on ADAS safety-tail cases.",
        "decision": "Use disabled as the conservative ADAS champion unless a challenger beats it on paired safety-tail metrics.",
    },
    {
        "hypothesis": "dual_union_can_combine_disabled_and_value_only",
        "result": "Retrieval-only study showed dual_union matched value_weights_only without clear uplift, adding context complexity without proven benefit.",
        "decision": "Keep dual_union experimental and exclude it from expensive default sweeps unless retrieval disagreement suggests value.",
    },
    {
        "hypothesis": "metadata_system_should_be_deleted_if_rerank_is_disabled",
        "result": "ADAS evidence only argues against metadata scoring in the hot path, not against metadata extraction/schema for diagnostics or generic domains.",
        "decision": "Do not delete metadata capabilities; make domain configs choose which scoring layers pay rent.",
    },
]


def base_name(path_or_name: str) -> str:
    return os.path.basename(path_or_name or "")



def classify_case_slices(case_or_result: dict[str, Any]) -> list[str]:
    """Assign stable eval slices from case id/query/text fields.

    Slices are deliberately heuristic and local to eval reporting. They do not
    affect retrieval. The goal is to avoid choosing a winner by average score
    while hiding safety-tail regressions in no-answer, regulatory, rollback, or
    latency cases.
    """
    text = " ".join(
        str(case_or_result.get(key) or "")
        for key in ("id", "query", "expected_answer_mode")
    ).lower().replace("-", "_")
    out = []
    for name, needles in SLICE_RULES:
        if any(needle in text for needle in needles):
            out.append(name)
    if not out:
        out.append("general")
    return sorted(set(out))


def load_source_map() -> dict[str, str]:
    """Load optional alias -> real file name map.

    This keeps eval_queries.json public-safe while allowing local validation
    against the actual Qdrant payload file_name values.
    """
    path = Path(SOURCE_MAP_FILE).expanduser()
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Source map must be a JSON object: {path}")

    out = {}
    for alias, file_name in data.items():
        if not isinstance(alias, str) or not isinstance(file_name, str):
            raise SystemExit(f"Invalid source map entry in {path}: {alias!r} -> {file_name!r}")
        out[alias] = file_name
    return out


def resolve_source_name(value: str, source_map: dict[str, str]) -> str:
    """Resolve an alias to a real file name; fall back to direct file name.

    If eval_queries.json uses aliases, missing aliases should fail loudly instead
    of silently treating the alias as a real file name. Direct filenames remain
    supported for private/local eval files.
    """
    if value in source_map:
        return source_map[value]

    # Heuristic: values with common file suffixes are direct file names.
    if value.endswith((".txt", ".pdf", ".md", ".docx", ".pptx")):
        return value

    raise SystemExit(
        f"Unknown eval source alias {value!r}. Add it to {SOURCE_MAP_FILE}, "
        "or use a direct local filename in the eval case."
    )


def resolve_case_sources(case: dict[str, Any], source_map: dict[str, str]) -> dict[str, Any]:
    """Return a copy of an eval case with aliases resolved to file names."""
    resolved = dict(case)

    def resolve_list(alias_key: str, file_key: str) -> None:
        values = []
        values.extend(case.get(file_key, []) or [])
        values.extend(case.get(alias_key, []) or [])
        if values:
            resolved[file_key] = [resolve_source_name(v, source_map) for v in values]

    resolve_list("expected_sources_all", "expected_files_all")
    resolve_list("expected_sources_any", "expected_files_any")
    resolve_list("forbidden_sources", "forbidden_files")

    def resolve_chunk_specs(key: str) -> None:
        out = []
        for spec in case.get(key, []) or []:
            spec = dict(spec)
            source = spec.pop("source", None)
            file_name = spec.get("file")
            if source is not None:
                spec["file"] = resolve_source_name(source, source_map)
            elif file_name is not None:
                spec["file"] = resolve_source_name(file_name, source_map)
            out.append(spec)
        if out:
            resolved[key] = out

    resolve_chunk_specs("expected_selected_chunks_any")
    resolve_chunk_specs("expected_expanded_chunks_any")

    return resolved


def file_set_from_items(items: list[dict[str, Any]]) -> set[str]:
    return {rc.file_name(i["payload"]) for i in items}


def chunk_refs_from_items(items: list[dict[str, Any]]) -> set[str]:
    out = set()
    for item in items:
        p = item["payload"]
        out.add(f"{rc.file_name(p)}:#{p.get('chunk_index')}")
    return out


def chunk_refs_from_groups(groups: list[dict[str, Any]]) -> set[str]:
    out = set()
    for group in groups:
        fname = group.get("file_name", "unknown")
        for item in group.get("chunks", []):
            p = item["payload"]
            out.add(f"{fname}:#{p.get('chunk_index')}")
    return out


def selected_items_from_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group in groups:
        for item in group.get("chunks", []):
            if item.get("is_selected_hit"):
                out.append(item)
    return out


def expanded_items_from_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group in groups:
        out.extend(group.get("chunks", []))
    return out


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def item_ref(item: dict[str, Any]) -> str:
    p = item["payload"]
    return f"{rc.file_name(p)}:#{p.get('chunk_index')}"


def item_summary(item: dict[str, Any], rank: int | None = None) -> dict[str, Any]:
    p = item["payload"]
    out = {
        "ref": item_ref(item),
        "file": rc.file_name(p),
        "chunk_index": p.get("chunk_index"),
        "dense_rank": item.get("dense_rank"),
        "vector_score": item.get("vector_score"),
        "meta_bonus": item.get("meta_bonus"),
        "final_score": item.get("final_score"),
        "decision": rc.payload_decision(p),
        "delivery_value": rc.payload_delivery_value(p),
        "criticality": rc.payload_criticality(p),
        "role": rc.payload_role(p),
    }
    if rank is not None:
        out["rank"] = rank
    return out


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _expected_file_pool(case: dict[str, Any]) -> list[str]:
    values: list[str] = []
    values.extend(case.get("expected_files_all", []) or [])
    values.extend(case.get("expected_files_any", []) or [])
    seen = set()
    out = []
    for value in values:
        base = base_name(value)
        if base and base not in seen:
            seen.add(base)
            out.append(base)
    return out


def _has_positive_retrieval_expectation(case: dict[str, Any]) -> bool:
    return bool(
        case.get("expected_files_all")
        or case.get("expected_files_any")
        or case.get("expected_selected_chunks_any")
        or case.get("expected_expanded_chunks_any")
    )


def _retrieval_expectation_type(case: dict[str, Any]) -> str:
    if _has_positive_retrieval_expectation(case):
        return "positive"
    if case.get("forbidden_files"):
        return "negative_only"
    return "abstention_or_answer_only"


def _hit_at(rank: int | None, k: int) -> int:
    return int(rank is not None and rank <= k)


def _source_dcg_at(selected: list[dict[str, Any]], expected_files: set[str], k: int) -> float | None:
    """Source-level DCG@k with one gain per expected file.

    A single source can contribute several selected chunks. Counting every chunk
    made the old "source nDCG" exceed 1.0. De-duplicating by file keeps this a
    source-recall ranking metric; chunk-level evidence can be measured separately.
    """
    if not expected_files:
        return None
    dcg = 0.0
    credited: set[str] = set()
    for rank, item in enumerate(selected[:k], start=1):
        fname = base_name(rc.file_name(item["payload"]))
        if fname in expected_files and fname not in credited:
            credited.add(fname)
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def _source_ndcg_at(selected: list[dict[str, Any]], expected_files: set[str], k: int) -> float | None:
    """Normalized source-level DCG@k, bounded to [0, 1]."""
    if not expected_files:
        return None
    dcg = _source_dcg_at(selected, expected_files, k) or 0.0
    ideal_hits = min(len(expected_files), k)
    if ideal_hits <= 0:
        return None
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return min(1.0, dcg / idcg) if idcg else None


def compute_retrieval_metrics(
    case: dict[str, Any],
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    failures: list[str],
    timings: dict[str, float],
) -> dict[str, Any]:
    expanded_items = expanded_items_from_groups(groups)
    selected_files = [base_name(rc.file_name(item["payload"])) for item in selected]
    expanded_files = [base_name(rc.file_name(item["payload"])) for item in expanded_items]
    selected_file_set = set(selected_files)
    expanded_file_set = set(expanded_files)
    expected_files = set(_expected_file_pool(case))

    selected_rank_by_file: dict[str, int] = {}
    for rank, item in enumerate(selected, start=1):
        selected_rank_by_file.setdefault(base_name(rc.file_name(item["payload"])), rank)

    expanded_group_rank_by_file: dict[str, int] = {}
    for rank, group in enumerate(groups, start=1):
        expanded_group_rank_by_file.setdefault(base_name(group.get("file_name", "")), rank)

    final_rank_by_file: dict[str, int] = {}
    dense_rank_by_file: dict[str, int] = {}
    for final_rank, item in enumerate(candidates, start=1):
        fname = base_name(rc.file_name(item["payload"]))
        final_rank_by_file.setdefault(fname, final_rank)
        dense_rank = item.get("dense_rank")
        if isinstance(dense_rank, int):
            dense_rank_by_file[fname] = min(dense_rank_by_file.get(fname, dense_rank), dense_rank)

    expected_selected_ranks = [selected_rank_by_file[f] for f in expected_files if f in selected_rank_by_file]
    expected_expanded_ranks = [expanded_group_rank_by_file[f] for f in expected_files if f in expanded_group_rank_by_file]
    expected_dense_ranks = [dense_rank_by_file[f] for f in expected_files if f in dense_rank_by_file]
    expected_final_ranks = [final_rank_by_file[f] for f in expected_files if f in final_rank_by_file]

    best_selected_rank = min(expected_selected_ranks) if expected_selected_ranks else None
    best_expanded_group_rank = min(expected_expanded_ranks) if expected_expanded_ranks else None
    best_dense_rank = min(expected_dense_ranks) if expected_dense_ranks else None
    best_final_rank = min(expected_final_ranks) if expected_final_ranks else None

    expected_rank_delta = None
    expected_rank_delta_direction = None
    if best_dense_rank is not None and best_final_rank is not None:
        # Positive means metadata rerank moved the best expected source upward.
        expected_rank_delta = best_dense_rank - best_final_rank
        if expected_rank_delta > 0:
            expected_rank_delta_direction = "improved"
        elif expected_rank_delta < 0:
            expected_rank_delta_direction = "worsened"
        else:
            expected_rank_delta_direction = "unchanged"

    meta_bonuses = [float(x.get("meta_bonus") or 0.0) for x in candidates]
    selected_meta_bonuses = [float(x.get("meta_bonus") or 0.0) for x in selected]

    selected_decisions = Counter(str(rc.payload_decision(item["payload"])) for item in selected)
    selected_roles = Counter(str(rc.payload_role(item["payload"])) for item in selected)
    selected_criticality = Counter(str(rc.payload_criticality(item["payload"])) for item in selected)
    selected_delivery = Counter(str(rc.payload_delivery_value(item["payload"])) for item in selected)

    expected_all = [base_name(x) for x in case.get("expected_files_all", []) or []]
    expected_any = [base_name(x) for x in case.get("expected_files_any", []) or []]
    query_profile = rc.pretty_profile(rc.query_profile(case["query"]))
    query_profile_hit_count = sum(len(v) for v in query_profile.values())
    positive_retrieval_case = _has_positive_retrieval_expectation(case)
    slices = classify_case_slices(case)

    return {
        "slices": slices,
        "passed": not failures,
        "failure_count": len(failures),
        "retrieval_expectation_type": _retrieval_expectation_type(case),
        "positive_retrieval_case": positive_retrieval_case,
        "top_k": int(case.get("top_k", DEFAULT_TOP_K)),
        "pre_k": int(case.get("pre_k", DEFAULT_PRE_K)),
        "max_per_file": int(case.get("max_per_file", DEFAULT_MAX_PER_FILE)),
        "neighbor_radius": int(case.get("neighbor_radius", DEFAULT_NEIGHBOR_RADIUS)),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_unique_files": len(selected_file_set),
        "expanded_chunk_count": len(expanded_items),
        "expanded_unique_files": len(expanded_file_set),
        "source_group_count": len(groups),
        "selected_drop_count": selected_decisions.get("drop", 0),
        "expanded_drop_count": sum(1 for item in expanded_items if rc.payload_decision(item["payload"]) == "drop"),
        "expected_files": sorted(expected_files),
        "expected_files_count": len(expected_files),
        "selected_expected_files": sorted(f for f in expected_files if f in selected_file_set),
        "expanded_expected_files": sorted(f for f in expected_files if f in expanded_file_set),
        "expected_files_selected_count": sum(1 for f in expected_files if f in selected_file_set),
        "expected_files_expanded_count": sum(1 for f in expected_files if f in expanded_file_set),
        "expected_files_all_count": len(expected_all),
        "expected_files_all_selected_count": sum(1 for f in expected_all if f in selected_file_set),
        "expected_files_all_expanded_count": sum(1 for f in expected_all if f in expanded_file_set),
        "expected_files_any_count": len(expected_any),
        "expected_files_any_selected_hit": bool(expected_any and any(f in selected_file_set for f in expected_any)),
        "expected_files_any_expanded_hit": bool(expected_any and any(f in expanded_file_set for f in expected_any)),
        "best_expected_selected_rank": best_selected_rank,
        "best_expected_expanded_group_rank": best_expanded_group_rank,
        "best_expected_dense_candidate_rank": best_dense_rank,
        "best_expected_final_candidate_rank": best_final_rank,
        "best_expected_rank_delta_dense_minus_final": expected_rank_delta,
        "best_expected_rank_delta_direction": expected_rank_delta_direction,
        "expected_source_hit_at_1": _hit_at(best_selected_rank, 1),
        "expected_source_hit_at_3": _hit_at(best_selected_rank, 3),
        "expected_source_hit_at_5": _hit_at(best_selected_rank, 5),
        "expected_source_mrr_selected": (1.0 / best_selected_rank) if best_selected_rank else 0.0,
        "selected_source_dcg_at_5": _source_dcg_at(selected, expected_files, 5),
        "selected_source_ndcg_at_5": _source_ndcg_at(selected, expected_files, 5),
        "metadata_prior_enabled": rc.METADATA_PRIOR_ENABLED,
        "rerank_mode_effective": rc.EFFECTIVE_RERANK_MODE,
        "query_profile": query_profile,
        "query_profile_hit_count": query_profile_hit_count,
        "candidate_meta_bonus": _stats(meta_bonuses),
        "selected_meta_bonus": _stats(selected_meta_bonuses),
        "selected_decision_counts": dict(selected_decisions),
        "selected_role_counts": dict(selected_roles),
        "selected_criticality_counts": dict(selected_criticality),
        "selected_delivery_value_counts": dict(selected_delivery),
        "retrieve_elapsed_sec": timings.get("retrieve_elapsed_sec"),
        "neighbor_expand_elapsed_sec": timings.get("neighbor_expand_elapsed_sec"),
        "total_elapsed_sec": timings.get("total_elapsed_sec"),
    }


def compact_case_record(
    case: dict[str, Any],
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    failures: list[str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "query": case.get("query"),
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
        "selected": [item_summary(item, rank=i) for i, item in enumerate(selected, start=1)],
        "top_candidates": [item_summary(item, rank=i) for i, item in enumerate(candidates[: max(10, DEFAULT_TOP_K)], start=1)],
        "source_groups": [
            {
                "group_rank": group.get("group_rank"),
                "file_name": group.get("file_name"),
                "selected_indices": group.get("selected_indices"),
                "expanded_indices": group.get("expanded_indices"),
                "best_final_score": group.get("best_final_score"),
            }
            for group in groups
        ],
    }


def aggregate_retrieval_metrics(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not case_records:
        return {}

    metrics = [r.get("metrics", {}) for r in case_records]
    total = len(case_records)
    positive_metrics = [m for m in metrics if m.get("positive_retrieval_case")]
    non_positive_metrics = [m for m in metrics if not m.get("positive_retrieval_case")]

    def avg_for(items: list[dict[str, Any]], key: str) -> float | None:
        vals = [m.get(key) for m in items if isinstance(m.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    def avg(key: str) -> float | None:
        return avg_for(metrics, key)

    failure_counter: Counter[str] = Counter()
    expectation_counter: Counter[str] = Counter()
    rank_delta_counter: Counter[str] = Counter()
    for record in case_records:
        for failure in record.get("failures", []) or []:
            failure_counter[failure.split(":", 1)[0]] += 1
        m = record.get("metrics", {}) or {}
        expectation_counter[str(m.get("retrieval_expectation_type") or "unknown")] += 1
        direction = m.get("best_expected_rank_delta_direction")
        if direction:
            rank_delta_counter[str(direction)] += 1

    selected_drop_total = sum(int(m.get("selected_drop_count") or 0) for m in metrics)
    expanded_drop_total = sum(int(m.get("expanded_drop_count") or 0) for m in metrics)

    slice_metrics: dict[str, dict[str, Any]] = {}
    slice_names = sorted({s for m in metrics for s in (m.get("slices") or ["general"])})
    for slice_name in slice_names:
        items = [m for m in metrics if slice_name in (m.get("slices") or [])]
        pos_items = [m for m in items if m.get("positive_retrieval_case")]
        slice_metrics[slice_name] = {
            "cases": len(items),
            "positive_cases": len(pos_items),
            "pass_rate": avg_for(items, "expected_source_hit_at_5") if pos_items else None,
            "positive_hit_at_1": avg_for(pos_items, "expected_source_hit_at_1"),
            "positive_hit_at_5": avg_for(pos_items, "expected_source_hit_at_5"),
            "positive_mrr": avg_for(pos_items, "expected_source_mrr_selected"),
            "positive_ndcg_at_5": avg_for(pos_items, "selected_source_ndcg_at_5"),
        }

    return {
        "cases_total": total,
        "passed": sum(1 for r in case_records if r.get("passed")),
        "failed": sum(1 for r in case_records if not r.get("passed")),
        "pass_rate": sum(1 for r in case_records if r.get("passed")) / total,
        "expectation_type_counts": dict(expectation_counter),
        "positive_retrieval_cases": len(positive_metrics),
        "non_positive_retrieval_cases": len(non_positive_metrics),

        # Overall metrics preserve the previous aggregate shape for compatibility.
        # For retrieval-quality decisions, use the positive_* metrics below.
        "expected_source_hit_at_1": avg("expected_source_hit_at_1"),
        "expected_source_hit_at_3": avg("expected_source_hit_at_3"),
        "expected_source_hit_at_5": avg("expected_source_hit_at_5"),
        "expected_source_mrr_selected": avg("expected_source_mrr_selected"),
        "selected_source_ndcg_at_5": avg("selected_source_ndcg_at_5"),

        # Positive-only retrieval metrics exclude answer-only/no-answer cases that
        # have no expected source by design. These are the main metrics for A/B/C
        # retrieval ablations.
        "positive_expected_source_hit_at_1": avg_for(positive_metrics, "expected_source_hit_at_1"),
        "positive_expected_source_hit_at_3": avg_for(positive_metrics, "expected_source_hit_at_3"),
        "positive_expected_source_hit_at_5": avg_for(positive_metrics, "expected_source_hit_at_5"),
        "positive_expected_source_mrr_selected": avg_for(positive_metrics, "expected_source_mrr_selected"),
        "positive_selected_source_dcg_at_5": avg_for(positive_metrics, "selected_source_dcg_at_5"),
        "positive_selected_source_ndcg_at_5": avg_for(positive_metrics, "selected_source_ndcg_at_5"),
        "positive_expected_files_selected_avg": avg_for(positive_metrics, "expected_files_selected_count"),
        "positive_expected_files_expanded_avg": avg_for(positive_metrics, "expected_files_expanded_count"),
        "positive_metadata_rank_delta_dense_minus_final_avg": avg_for(positive_metrics, "best_expected_rank_delta_dense_minus_final"),
        "positive_rank_delta_direction_counts": dict(rank_delta_counter),

        "expected_files_selected_avg": avg("expected_files_selected_count"),
        "expected_files_expanded_avg": avg("expected_files_expanded_count"),
        "candidate_count_avg": avg("candidate_count"),
        "selected_unique_files_avg": avg("selected_unique_files"),
        "expanded_chunk_count_avg": avg("expanded_chunk_count"),
        "query_profile_hit_count_avg": avg("query_profile_hit_count"),
        "retrieve_elapsed_sec_avg": avg("retrieve_elapsed_sec"),
        "neighbor_expand_elapsed_sec_avg": avg("neighbor_expand_elapsed_sec"),
        "total_elapsed_sec_avg": avg("total_elapsed_sec"),
        "selected_drop_count_total": selected_drop_total,
        "expanded_drop_count_total": expanded_drop_total,
        "metadata_rank_delta_dense_minus_final_avg": avg("best_expected_rank_delta_dense_minus_final"),
        "failure_kinds": dict(failure_counter),
        "slice_metrics": slice_metrics,
    }


def has_any_chunk(refs: set[str], file_name: str, chunks: list[int]) -> bool:
    fname = base_name(file_name)
    return any(f"{fname}:#{idx}" in refs for idx in chunks)


def validate_case(case: dict[str, Any], selected: list[dict[str, Any]], groups: list[dict[str, Any]]) -> list[str]:
    failures = []

    selected_files = file_set_from_items(selected)
    expanded_items = expanded_items_from_groups(groups)
    expanded_files = file_set_from_items(expanded_items)
    selected_refs = chunk_refs_from_items(selected)
    expanded_refs = chunk_refs_from_groups(groups)

    for fname in case.get("expected_files_all", []):
        if base_name(fname) not in expanded_files:
            failures.append(f"missing expected file: {fname}")

    any_files = [base_name(x) for x in case.get("expected_files_any", [])]
    if any_files and not any(fname in expanded_files for fname in any_files):
        failures.append(f"none of expected_files_any found: {case.get('expected_files_any')}")

    for fname in case.get("forbidden_files", []):
        if base_name(fname) in expanded_files:
            failures.append(f"forbidden file retrieved: {fname}")

    for spec in case.get("expected_selected_chunks_any", []):
        fname = spec.get("file")
        chunks = spec.get("chunks", [])
        if fname and chunks and not has_any_chunk(selected_refs, fname, chunks):
            failures.append(f"missing any selected chunk for {fname} expected_any={chunks}")

    for spec in case.get("expected_expanded_chunks_any", []):
        fname = spec.get("file")
        chunks = spec.get("chunks", [])
        if fname and chunks and not has_any_chunk(expanded_refs, fname, chunks):
            failures.append(f"missing any expanded chunk for {fname} expected_any={chunks}")

    # This is an ingest policy check more than a reranker check: drop chunks should
    # normally not be indexed at all. If they appear, either RAG_INDEX_DROPPED_CHUNKS
    # was enabled or the collection is stale/misconfigured.
    for item in expanded_items:
        p = item["payload"]
        if rc.payload_decision(p) == "drop":
            failures.append(
                f"drop chunk retrieved: {rc.file_name(p)}:#{p.get('chunk_index')} "
                "(drop means exclude from index by default)"
            )

    return failures


def print_case_result(case: dict[str, Any], selected: list[dict[str, Any]], groups: list[dict[str, Any]], failures: list[str]):
    status = "PASS" if not failures else "FAIL"
    print("=" * 100)
    print(f"{status}: {case['id']}")
    print("=" * 100)
    print(f"Query: {case['query']}")
    print(
        f"top_k={case.get('top_k', DEFAULT_TOP_K)}; "
        f"pre_k={case.get('pre_k', DEFAULT_PRE_K)}; "
        f"max_per_file={case.get('max_per_file', DEFAULT_MAX_PER_FILE)}; "
        f"neighbor_radius={case.get('neighbor_radius', DEFAULT_NEIGHBOR_RADIUS)}"
    )
    if case.get("notes"):
        print(f"Notes: {case['notes']}")
    print()

    selected_refs = sorted(chunk_refs_from_items(selected))
    expanded_refs = sorted(chunk_refs_from_groups(groups))
    selected_files = sorted(file_set_from_items(selected))
    expanded_files = sorted(file_set_from_items(expanded_items_from_groups(groups)))

    print(f"Selected files: [{', '.join(selected_files)}]")
    print(f"Expanded files: [{', '.join(expanded_files)}]")
    print(f"Selected chunks: [{', '.join(selected_refs)}]")
    print(f"Expanded chunks: [{', '.join(expanded_refs)}]")
    print()

    print("Selected results:")
    for rank, item in enumerate(selected, start=1):
        p = item["payload"]
        print(
            f"  {rank}. "
            f"final={rc.fmt_score(item.get('final_score'))} "
            f"vector={rc.fmt_score(item.get('vector_score'))} "
            f"meta={rc.fmt_score(item.get('meta_bonus'))} "
            f"file={rc.file_name(p)} "
            f"chunk={p.get('chunk_index')} "
            f"role={rc.payload_role(p)} "
            f"facets={rc.payload_facets(p)} "
            f"criticality={rc.payload_criticality(p)} "
            f"delivery={rc.payload_delivery_value(p)} "
            f"decision={rc.payload_decision(p)}"
        )
    print()

    print("Source groups after neighbor expansion:")
    for i, group in enumerate(groups, start=1):
        print(
            f"  S{i}. file={group.get('file_name')} "
            f"selected={group.get('selected_indices')} "
            f"expanded={group.get('expanded_indices')} "
            f"best_final={rc.fmt_score(group.get('best_final_score'))}"
        )
    print()

    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
        print()



def latest_summary(root: Path, prefix: str) -> Path:
    matches = sorted(root.glob(f"{prefix}_*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No {prefix}_*/summary.json under {root}")
    return matches[0]


def try_latest_summary(root: Path, prefix: str) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    """Best-effort summary loader for compare mode.

    Eval subprocesses intentionally return non-zero when a case fails. In an
    ablation sweep that failure is data, not a reason to discard the whole run.
    This helper lets compare-mode preserve partial metrics whenever the child
    wrote summary.json before exiting.
    """
    try:
        path = latest_summary(root, prefix)
    except FileNotFoundError as exc:
        return None, None, str(exc)
    try:
        return path, json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return path, None, f"Could not read {path}: {exc}"


def run_subprocess(cmd: list[str], env: dict[str, str], cwd: Path, label: str = "subprocess") -> dict[str, Any]:
    """Run a child eval while streaming its stdout as a progress pulse."""
    started = datetime.now().isoformat(timespec="seconds")
    tail: deque[str] = deque(maxlen=200)
    print(f"--- {label}: start {' '.join(cmd)} ---", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        print(line, flush=True)
    returncode = proc.wait()
    print(f"--- {label}: exit {returncode} ---", flush=True)
    return {
        "cmd": cmd,
        "started_at": started,
        "returncode": returncode,
        "stdout_tail": "\n".join(tail),
    }


def compact_retrieval_summary(summary: dict[str, Any]) -> dict[str, Any]:
    agg = summary.get("aggregate_metrics", {}) or {}
    out = {key: agg.get(key) for key in RETRIEVAL_COMPARE_KEYS}
    config = summary.get("retrieval_config", {}) or summary.get("config", {}) or {}
    out["rerank_mode_configured"] = config.get("rerank_mode_configured")
    out["rerank_mode_effective"] = config.get("rerank_mode_effective")
    out["retrieval_ensemble_modes"] = config.get("retrieval_ensemble_modes") or []
    out["metadata_prior_enabled"] = config.get("metadata_prior") == "enabled"
    out["cases_total"] = summary.get("cases_total") or agg.get("cases_total")
    out["failed"] = summary.get("failed") or agg.get("failed")
    out["failed_ids"] = summary.get("failed_ids", [])
    out["expectation_type_counts"] = agg.get("expectation_type_counts", {})
    out["positive_rank_delta_direction_counts"] = agg.get("positive_rank_delta_direction_counts", {})
    out["slice_metrics"] = agg.get("slice_metrics", {})
    return out


def compact_answer_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Normalize eval_answer.py summary names for compare tables.

    eval_answer.py keeps stable aggregate names such as answer_chars_avg. The
    compare table uses human-oriented names such as avg_answer_chars. Keeping the
    mapping here avoids spreading one-off metric aliases into run artifacts.
    """
    agg = summary.get("aggregate_metrics", {}) or {}
    judge = summary.get("llm_judge_aggregate", {}) or {}
    usage = agg.get("usage_totals") or {}
    out = {
        "pass_rate": agg.get("pass_rate"),
        "avg_answer_chars": agg.get("answer_chars_avg"),
        "avg_citation_count": agg.get("citation_count_avg"),
        "avg_generation_elapsed_sec": agg.get("answer_elapsed_sec_avg"),
        "total_tokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
        "repair_attempted_count": agg.get("repair_attempted_cases"),
        "repaired_count": agg.get("repaired_cases"),
        "regex_overclaim_risk_cases": agg.get("regex_overclaim_risk_cases"),
        "failure_category_counts": agg.get("failure_category_counts", {}),
        "judge_parse_failed_cases": agg.get("judge_parse_failed_cases"),
        "judge_semantic_failed_cases": agg.get("judge_semantic_failed_cases"),
    }
    if judge.get("enabled"):
        judged = judge.get("judged_cases") or 0
        passed = judge.get("judge_passed_cases") or 0
        out.update({
            "llm_judge_avg_overall_score": judge.get("avg_overall_score"),
            "llm_judge_avg_groundedness_score": judge.get("avg_groundedness_score"),
            "llm_judge_avg_citation_quality_score": judge.get("avg_citation_quality_score"),
            "llm_judge_judge_pass_rate": (passed / judged) if judged else None,
            "llm_judge_call_success_rate": judge.get("judge_call_success_rate"),
            "llm_judge_semantic_pass_rate": judge.get("judge_semantic_pass_rate"),
            "llm_judge_end_to_end_pass_rate": judge.get("judge_end_to_end_pass_rate"),
            "llm_judge_failure_type_counts": judge.get("judge_failure_type_counts", {}),
        })
    out["cases_total"] = summary.get("cases_total")
    out["failed"] = summary.get("failed")
    out["failed_ids"] = summary.get("failed_ids", [])
    return out


def markdown_table(rows: list[dict[str, Any]], metric_keys: list[str], title: str) -> str:
    cols = ["variant"] + metric_keys
    lines = [f"## {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        values = [str(row.get("variant"))]
        metrics = row.get("metrics", {}) or {}
        for key in metric_keys:
            value = metrics.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)




def _num(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def rank_variants(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Build an opinionated but transparent shortlist ranking.

    The score is not a new scientific metric. It is a triage helper for deciding
    which variants deserve expensive answer+judge reruns. Retrieval quality is
    weighted most because it is cheap and deterministic; answer/judge scores are
    included only when present. Runtime/tokens are small penalties.
    """
    rows: list[dict[str, Any]] = []
    for variant in comparison.get("variants", []) or []:
        name = variant.get("name")
        r = ((variant.get("retrieval") or {}).get("metrics") or {})
        a = ((variant.get("answer") or {}).get("metrics") or {})
        retrieval_failed = bool(r.get("subprocess_failed"))
        answer_failed = bool(a.get("subprocess_failed")) if a else False

        hit1 = _num(r.get("positive_expected_source_hit_at_1"))
        mrr = _num(r.get("positive_expected_source_mrr_selected"))
        hit5 = _num(r.get("positive_expected_source_hit_at_5"))
        retrieval_pass = _num(r.get("pass_rate"))
        answer_pass = _num(a.get("pass_rate"), 0.0) if a else None
        judge_pass = _num(a.get("llm_judge_judge_pass_rate"), 0.0) if a else None
        overall = _num(a.get("llm_judge_avg_overall_score"), 0.0) if a else None
        answer_sec = _num(a.get("avg_generation_elapsed_sec"), 0.0) if a else None
        tokens = _num(a.get("total_tokens"), 0.0) if a else None

        score = 0.0
        score += 4.0 * hit1
        score += 2.0 * mrr
        score += 1.5 * hit5
        score += 1.0 * retrieval_pass
        if answer_pass is not None:
            score += 3.0 * answer_pass
        if judge_pass is not None:
            score += 1.5 * judge_pass
        if overall is not None:
            score += 0.4 * overall
        if retrieval_failed:
            score -= 2.0
        if answer_failed:
            score -= 1.0
        if answer_sec:
            score -= min(answer_sec / 120.0, 1.0)
        if tokens:
            score -= min(tokens / 200000.0, 1.0)

        rows.append({
            "variant": name,
            "rerank_mode_effective": r.get("rerank_mode_effective"),
            "retrieval_ensemble_modes": ",".join(r.get("retrieval_ensemble_modes") or []),
            "recommendation": "shortlist" if score >= 9.0 and not retrieval_failed else "watch",
            "score": score,
            "positive_hit_at_1": hit1,
            "positive_mrr": mrr,
            "retrieval_failed": retrieval_failed,
            "answer_pass_rate": answer_pass,
            "judge_pass_rate": judge_pass,
            "avg_overall": overall,
            "avg_answer_sec": answer_sec,
            "total_tokens": tokens,
        })

    rows.sort(key=lambda row: row["score"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        if idx == 1 and row.get("recommendation") != "watch":
            row["recommendation"] = "best_current_candidate"
    return rows


def _infer_answer_failure_categories(result: dict[str, Any]) -> set[str]:
    """Best-effort per-case failure categories for rerun selection.

    eval_answer.py already writes aggregate category counts. For case files we
    need case ids, so this helper reads the per-case result and tolerates older
    runs that may not yet contain metrics.failure_category_counts.
    """
    categories: set[str] = set()
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    counts = metrics.get("failure_category_counts") if isinstance(metrics.get("failure_category_counts"), dict) else {}
    categories.update(str(k) for k, v in counts.items() if v)
    for failure in result.get("failures") or []:
        text = str(failure).lower()
        if "forbidden_answer_pattern" in text or "overclaim" in text:
            categories.add("regex_overclaim_risk")
        elif "citation" in text:
            categories.add("citation_contract")
        elif "missing_required_section" in text or "required section" in text:
            categories.add("answer_contract")
        elif text:
            categories.add("deterministic_failure")
    return categories


def _collect_answer_problem_sets(comparison: dict[str, Any]) -> dict[str, set[str]]:
    """Collect problem case ids by reason from answer/judge artifacts."""
    out: dict[str, set[str]] = {
        "deterministic_failed": set(),
        "regex_overclaim_risk": set(),
        "judge_infra_failed": set(),
        "judge_semantic_failed": set(),
        "safety_tail": set(),
    }
    for variant in comparison.get("variants", []) or []:
        for result in _load_answer_results_for_variant(variant):
            case_id = str(result.get("id") or "").strip()
            if not case_id:
                continue
            slices = set(classify_case_slices(result))
            if slices & {"insufficient_evidence", "regulatory_scope", "validation_test", "deployment_runtime", "failure_degraded_mode"}:
                out["safety_tail"].add(case_id)
            summary = _case_success_from_answer_result(result)
            if not summary["deterministic_pass"]:
                out["deterministic_failed"].add(case_id)
            categories = _infer_answer_failure_categories(result)
            if "regex_overclaim_risk" in categories:
                out["regex_overclaim_risk"].add(case_id)
            if summary["judge_enabled"] and not summary["judge_ok"]:
                out["judge_infra_failed"].add(case_id)
            if summary["judge_enabled"] and summary["judge_ok"] and not summary["judge_passed"]:
                out["judge_semantic_failed"].add(case_id)
    return out


def _collect_paired_disagreement_cases(study_report: dict[str, Any] | None) -> set[str]:
    """Cases where primary A/B candidates disagree or one fails end-to-end."""
    if not study_report:
        return set()
    out: set[str] = set()
    for row in study_report.get("paired_rows") or []:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            continue
        a_pass = bool(row.get(f"{STUDY_PRIMARY_A}_end_to_end_pass"))
        b_pass = bool(row.get(f"{STUDY_PRIMARY_B}_end_to_end_pass"))
        if row.get("winner") != "tie" or a_pass != b_pass:
            out.add(case_id)
    return out


def _write_case_id_file(path: Path, case_ids: set[str]) -> list[str]:
    out = sorted(case_ids)
    if out:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return out


def write_problem_case_files(run_root: Path, comparison: dict[str, Any], study_report: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Write targeted rerun case files.

    `problem_cases.txt` is intentionally broad: it is the expensive rerun set,
    not just deterministic failures. Prior runs showed that semantic judge
    failures and paired disagreements were the cases that actually changed the
    rerank decision, so they must be included.
    """
    reason_sets = _collect_answer_problem_sets(comparison)

    retrieval_failed: set[str] = set()
    for variant in comparison.get("variants", []) or []:
        metrics = ((variant.get("retrieval") or {}).get("metrics") or {})
        for case_id in metrics.get("failed_ids") or []:
            if case_id:
                retrieval_failed.add(str(case_id))
    reason_sets["retrieval_failed"] = retrieval_failed
    reason_sets["paired_disagreement"] = _collect_paired_disagreement_cases(study_report)

    semantic_problem_cases = set(reason_sets["judge_semantic_failed"]) | set(reason_sets["judge_infra_failed"])
    paired_disagreement_cases = set(reason_sets["paired_disagreement"])
    safety_tail_cases = set(reason_sets["safety_tail"])
    all_problem_cases = set().union(*reason_sets.values()) if reason_sets else set()

    files = {
        "problem_cases": _write_case_id_file(run_root / "problem_cases.txt", all_problem_cases),
        "semantic_problem_cases": _write_case_id_file(run_root / "semantic_problem_cases.txt", semantic_problem_cases),
        "paired_disagreement_cases": _write_case_id_file(run_root / "paired_disagreement_cases.txt", paired_disagreement_cases),
        "safety_tail_cases": _write_case_id_file(run_root / "safety_tail_cases.txt", safety_tail_cases),
    }
    files["by_reason"] = {key: sorted(value) for key, value in sorted(reason_sets.items())}  # type: ignore[assignment]
    return files



def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _case_success_from_answer_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {}) or {}
    judge = result.get("llm_judge") or {}
    judge_enabled = bool(metrics.get("llm_judge_enabled") or judge.get("enabled"))
    judge_ok = bool(metrics.get("llm_judge_ok") or judge.get("ok")) if judge_enabled else True
    judge_passed = bool(metrics.get("llm_judge_passed") or judge.get("passed")) if judge_enabled else True
    deterministic_pass = bool(result.get("passed"))
    end_to_end = deterministic_pass and judge_ok and judge_passed
    judge_result = judge.get("result", {}) if isinstance(judge, dict) else {}
    return {
        "deterministic_pass": deterministic_pass,
        "judge_enabled": judge_enabled,
        "judge_ok": judge_ok,
        "judge_passed": judge_passed,
        "end_to_end_pass": end_to_end,
        "overall_score": judge_result.get("overall_score"),
        "failure_categories": metrics.get("failure_category_counts", {}),
        "slices": classify_case_slices(result),
    }


def _load_answer_results_for_variant(record: dict[str, Any]) -> list[dict[str, Any]]:
    answer = record.get("answer") or {}
    summary_path = answer.get("summary_path")
    if not summary_path:
        return []
    path = Path(summary_path)
    case_path = path.parent / "cases.jsonl"
    if case_path.exists():
        return _read_jsonl(case_path)
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        return summary.get("results") or []
    except Exception:
        return []


def _study_base_name(record: dict[str, Any]) -> str:
    return str(record.get("base_variant") or record.get("name") or "")


def build_study_report(comparison: dict[str, Any]) -> dict[str, Any]:
    """Build paired/slice evidence for rerank-mode selection.

    This report is meant to answer a narrower question than the normal ranking:
    if disabled and value-only rerank keep trading wins, are they actually
    distinguishable on paired cases and safety-tail slices? If not, the decision
    rule favors the simpler conservative candidate.
    """
    records = comparison.get("variants", []) or []
    by_base: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_base.setdefault(_study_base_name(record), []).append(record)

    candidate_summaries: dict[str, Any] = {}
    for base, recs in sorted(by_base.items()):
        retrieval_metrics = [((r.get("retrieval") or {}).get("metrics") or {}) for r in recs]
        answer_metrics = [((r.get("answer") or {}).get("metrics") or {}) for r in recs if r.get("answer")]
        def mean_numeric(rows: list[dict[str, Any]], key: str):
            vals = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
            return (sum(vals) / len(vals)) if vals else None
        candidate_summaries[base] = {
            "runs": len(recs),
            "retrieval_positive_hit_at_1_mean": mean_numeric(retrieval_metrics, "positive_expected_source_hit_at_1"),
            "retrieval_positive_hit_at_5_mean": mean_numeric(retrieval_metrics, "positive_expected_source_hit_at_5"),
            "retrieval_positive_mrr_mean": mean_numeric(retrieval_metrics, "positive_expected_source_mrr_selected"),
            "retrieval_positive_ndcg_at_5_mean": mean_numeric(retrieval_metrics, "positive_selected_source_ndcg_at_5"),
            "answer_pass_rate_mean": mean_numeric(answer_metrics, "pass_rate"),
            "judge_call_success_rate_mean": mean_numeric(answer_metrics, "llm_judge_call_success_rate"),
            "judge_semantic_pass_rate_mean": mean_numeric(answer_metrics, "llm_judge_semantic_pass_rate"),
            "judge_end_to_end_pass_rate_mean": mean_numeric(answer_metrics, "llm_judge_end_to_end_pass_rate"),
            "avg_overall_score_mean": mean_numeric(answer_metrics, "llm_judge_avg_overall_score"),
            "regex_overclaim_risk_total": sum(int(m.get("regex_overclaim_risk_cases") or 0) for m in answer_metrics),
        }

    # Slice metrics from answer results, grouped by base candidate.
    slice_metrics: dict[str, dict[str, Any]] = {}
    for base, recs in by_base.items():
        per_slice: dict[str, list[dict[str, Any]]] = {}
        for rec in recs:
            for result in _load_answer_results_for_variant(rec):
                summary = _case_success_from_answer_result(result)
                for slice_name in summary["slices"]:
                    per_slice.setdefault(slice_name, []).append(summary)
        slice_metrics[base] = {}
        for slice_name, rows in sorted(per_slice.items()):
            n = len(rows)
            if not n:
                continue
            scores = [r.get("overall_score") for r in rows if isinstance(r.get("overall_score"), (int, float))]
            slice_metrics[base][slice_name] = {
                "cases": n,
                "deterministic_pass_rate": sum(1 for r in rows if r["deterministic_pass"]) / n,
                "judge_semantic_pass_rate": sum(1 for r in rows if r["judge_passed"]) / n,
                "end_to_end_pass_rate": sum(1 for r in rows if r["end_to_end_pass"]) / n,
                "avg_overall_score": (sum(scores) / len(scores)) if scores else None,
            }

    # Paired disabled-vs-value comparison over matching repeat index + case id.
    paired_rows: list[dict[str, Any]] = []
    a_recs = by_base.get(STUDY_PRIMARY_A, [])
    b_recs = by_base.get(STUDY_PRIMARY_B, [])
    for idx, (a_rec, b_rec) in enumerate(zip(a_recs, b_recs), start=1):
        a_cases = {r.get("id"): r for r in _load_answer_results_for_variant(a_rec)}
        b_cases = {r.get("id"): r for r in _load_answer_results_for_variant(b_rec)}
        for case_id in sorted(set(a_cases) & set(b_cases)):
            a_sum = _case_success_from_answer_result(a_cases[case_id])
            b_sum = _case_success_from_answer_result(b_cases[case_id])
            if a_sum["end_to_end_pass"] and not b_sum["end_to_end_pass"]:
                winner = STUDY_PRIMARY_A
            elif b_sum["end_to_end_pass"] and not a_sum["end_to_end_pass"]:
                winner = STUDY_PRIMARY_B
            else:
                ao = a_sum.get("overall_score")
                bo = b_sum.get("overall_score")
                if isinstance(ao, (int, float)) and isinstance(bo, (int, float)) and ao != bo:
                    winner = STUDY_PRIMARY_A if ao > bo else STUDY_PRIMARY_B
                else:
                    winner = "tie"
            paired_rows.append({
                "repeat": idx,
                "case_id": case_id,
                "slices": sorted(set(a_sum["slices"]) | set(b_sum["slices"])),
                f"{STUDY_PRIMARY_A}_end_to_end_pass": a_sum["end_to_end_pass"],
                f"{STUDY_PRIMARY_B}_end_to_end_pass": b_sum["end_to_end_pass"],
                f"{STUDY_PRIMARY_A}_overall": a_sum.get("overall_score"),
                f"{STUDY_PRIMARY_B}_overall": b_sum.get("overall_score"),
                "winner": winner,
            })

    pair_counts = Counter(row["winner"] for row in paired_rows)

    def hard_gate(candidate: str) -> dict[str, Any]:
        c = candidate_summaries.get(candidate, {})
        insufficient = (slice_metrics.get(candidate, {}) or {}).get("insufficient_evidence", {})
        checks = {
            "positive_hit_at_5": (c.get("retrieval_positive_hit_at_5_mean") or 0.0) >= HARD_GATE_POLICY["positive_expected_source_hit_at_5_min"],
            "answer_pass_rate": (c.get("answer_pass_rate_mean") or 0.0) >= HARD_GATE_POLICY["answer_pass_rate_min"],
            "judge_call_success_rate": (c.get("judge_call_success_rate_mean") or 0.0) >= HARD_GATE_POLICY["judge_call_success_rate_min"],
            "judge_semantic_pass_rate": (c.get("judge_semantic_pass_rate_mean") or 0.0) >= HARD_GATE_POLICY["judge_semantic_pass_rate_min"],
            "regex_overclaim_risk": int(c.get("regex_overclaim_risk_total") or 0) <= HARD_GATE_POLICY["regex_overclaim_risk_max"],
            "insufficient_evidence_semantic_pass": (insufficient.get("judge_semantic_pass_rate") or 0.0) >= HARD_GATE_POLICY["insufficient_evidence_semantic_pass_min"] if insufficient else True,
        }
        return {"passed": all(checks.values()), "checks": checks}

    hard_gates = {candidate: hard_gate(candidate) for candidate in sorted(candidate_summaries)}
    disabled_gate = hard_gates.get(STUDY_PRIMARY_A, {}).get("passed")
    value_gate = hard_gates.get(STUDY_PRIMARY_B, {}).get("passed")
    value_wins = pair_counts.get(STUDY_PRIMARY_B, 0)
    disabled_wins = pair_counts.get(STUDY_PRIMARY_A, 0)

    if value_gate and value_wins >= disabled_wins + 2:
        recommendation = STUDY_PRIMARY_B
        reason = "value_weights_only cleared hard gates and won paired cases by the configured practical margin."
    elif disabled_gate:
        recommendation = STUDY_PRIMARY_A
        reason = "disabled/no_metadata_rerank cleared hard gates; tie-breaker favors simpler conservative retrieval when uplift is not decisive."
    elif value_gate:
        recommendation = STUDY_PRIMARY_B
        reason = "disabled failed hard gates while value_weights_only cleared them."
    else:
        recommendation = "no_candidate_cleared_hard_gates"
        reason = "Neither primary candidate cleared the hard gates; inspect slice failures before changing production policy."

    return {
        "study_name": comparison.get("study"),
        "policy": {
            "primary_a": STUDY_PRIMARY_A,
            "primary_b": STUDY_PRIMARY_B,
            "reference": "baseline_full",
            "tie_breaker": "favor disabled/no_metadata_rerank for safety-relevant ADAS unless value_weights_only shows decisive paired uplift without safety-tail regressions",
            "hard_gate_policy": HARD_GATE_POLICY,
        },
        "evidence_ledger": RERANK_SELECTION_EVIDENCE_LEDGER,
        "candidate_summaries": candidate_summaries,
        "hard_gates": hard_gates,
        "slice_metrics": slice_metrics,
        "paired_counts": dict(pair_counts),
        "paired_rows": paired_rows,
        "recommendation": recommendation,
        "recommendation_reason": reason,
    }


def write_study_report_files(run_root: Path, study_report: dict[str, Any]) -> None:
    write_json(run_root / "study_report.json", study_report)
    lines = [
        f"# RAG rerank selection study: {run_root.name}",
        "",
        f"Recommendation: `{study_report.get('recommendation')}`",
        "",
        study_report.get("recommendation_reason") or "",
        "",
        "## Paired counts",
        "",
    ]
    for key, value in sorted((study_report.get("paired_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Hard gates", ""]
    for candidate, gate in sorted((study_report.get("hard_gates") or {}).items()):
        lines.append(f"- {candidate}: {'PASS' if gate.get('passed') else 'FAIL'} {gate.get('checks')}")
    lines += ["", "## Candidate summaries", ""]
    for candidate, summary in sorted((study_report.get("candidate_summaries") or {}).items()):
        lines.append(f"- {candidate}: {summary}")
    (run_root / "study_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommended_next_steps(run_root: Path, ranking_rows: list[dict[str, Any]], problem_cases: list[str], answer_enabled: bool) -> None:
    top_variants = [row["variant"] for row in ranking_rows[:3] if row.get("variant")]
    lines = [
        "Recommended next commands",
        "=========================",
        "",
        "Cheap sweep, safe to run often:",
        "  python3 eval_retrieval.py --compare --retrieval-only",
        "",
    ]
    if top_variants:
        variant_args = " ".join(f"--variant {name}" for name in top_variants)
        lines += [
            "Expensive shortlist only:",
            f"  python3 eval_retrieval.py --compare --answer --judge {variant_args}",
            "",
        ]
    if problem_cases:
        variant_args = " ".join(f"--variant {name}" for name in top_variants[:2]) if top_variants else ""
        lines += [
            "Problem-case loop only:",
            f"  python3 eval_retrieval.py --compare --answer --judge {variant_args} --case-file {run_root / 'problem_cases.txt'}",
            "",
        ]
    if not answer_enabled:
        lines += [
            "This run was retrieval-only. Use --answer --judge only after retrieval narrows the shortlist.",
            "",
        ]
    (run_root / "recommended_next_commands.txt").write_text("\n".join(lines), encoding="utf-8")

def run_compare(args: argparse.Namespace) -> None:
    """Run retrieval ablations and optional answer/judge study.

    Compare mode is deliberately tolerant: a child eval returning non-zero is
    treated as a measured failure for that variant, not as a reason to abort the
    whole sweep. `--study adas_rerank_selection` narrows the default variants to
    the academic A/B protocol: disabled vs value-only rerank plus baseline_full
    as reference; `--repeat N` repeats expensive answer/judge calls so stochastic
    LLM output is measured instead of mistaken for deterministic truth.
    """
    case_ids = selected_case_ids(args)
    if args.retrieval_only and (args.answer or args.judge):
        raise SystemExit("Use either --retrieval-only or --answer/--judge, not both")

    repo_dir = Path(__file__).resolve().parent
    run_root = RUNS_DIR / f"ablation_compare_{now_run_id()}"
    run_root.mkdir(parents=True, exist_ok=True)

    if args.study == "adas_rerank_selection" and not args.variant:
        wanted_names = list(STUDY_VARIANTS)
        if args.include_dual_union:
            wanted_names.append("dual_union_disabled_value")
        wanted = set(wanted_names)
        variants = [v for v in ABLATION_VARIANTS if v["name"] in wanted]
        # Study mode is about answer/judge selection. Keep retrieval-only allowed
        # for sanity, but otherwise imply answer+judge so users do not forget the
        # expensive but necessary semantic part of the protocol.
        if not args.retrieval_only:
            args.answer = True
            args.judge = True
    elif args.variant:
        wanted = set(args.variant)
        variants = [v for v in ABLATION_VARIANTS if v["name"] in wanted]
        missing = wanted - {v["name"] for v in variants}
        if missing:
            raise SystemExit(f"Unknown variant(s): {sorted(missing)}")
    else:
        variants = ABLATION_VARIANTS

    comparison: dict[str, Any] = {
        "run_root": str(run_root),
        "entrypoint": "eval_retrieval.py --compare",
        "study": args.study,
        "repeat": int(args.repeat),
        "answer_eval_enabled": bool(args.answer or args.judge),
        "llm_judge_enabled": bool(args.judge),
        "case_filter": case_ids,
        "compare_policy": {
            "continue_after_child_exit_nonzero": True,
            "fail_fast": bool(args.fail_fast),
            "child_nonzero_is_metric": True,
            "repeat_applies_to_answer_judge_study": True,
        },
        "variants": [],
    }

    retrieval_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    child_failures: list[dict[str, Any]] = []

    repeat_count = max(1, int(args.repeat or 1)) if (args.answer or args.judge) else 1

    for variant in variants:
        base_name_variant = variant["name"]
        for repeat_index in range(1, repeat_count + 1):
            name = base_name_variant if repeat_count == 1 else f"{base_name_variant}__r{repeat_index}"
            variant_dir = run_root / name
            variant_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env.update(variant["env"])
            env["RAG_EVAL_RUNS_DIR"] = str(variant_dir)

            retrieval_cmd = [sys.executable, "eval_retrieval.py"]
            for case_id in case_ids:
                retrieval_cmd += ["--case", case_id]

            print(f"=== {name}: retrieval ===", flush=True)
            retrieval_run = run_subprocess(retrieval_cmd, env, repo_dir, label=f"{name}: retrieval")
            (variant_dir / "retrieval_stdout_tail.txt").write_text(retrieval_run["stdout_tail"], encoding="utf-8")

            retrieval_summary_path, retrieval_summary, retrieval_summary_error = try_latest_summary(variant_dir, "retrieval_eval")
            if retrieval_summary is not None:
                retrieval_metrics = compact_retrieval_summary(retrieval_summary)
            else:
                retrieval_metrics = {
                    "summary_missing": True,
                    "summary_error": retrieval_summary_error,
                    "subprocess_returncode": retrieval_run["returncode"],
                }
            retrieval_metrics["subprocess_returncode"] = retrieval_run["returncode"]
            retrieval_metrics["subprocess_failed"] = retrieval_run["returncode"] != 0
            if retrieval_run["returncode"] != 0:
                child_failures.append({"variant": name, "base_variant": base_name_variant, "stage": "retrieval", "returncode": retrieval_run["returncode"]})
                print(
                    f"WARNING: {name} retrieval exited {retrieval_run['returncode']}; "
                    "keeping partial metrics and continuing.",
                    flush=True,
                )

            retrieval_rows.append({"variant": name, "metrics": retrieval_metrics})

            variant_record: dict[str, Any] = {
                "name": name,
                "base_variant": base_name_variant,
                "repeat_index": repeat_index,
                "description": variant["description"],
                "env": variant["env"],
                "retrieval": {
                    "summary_path": str(retrieval_summary_path) if retrieval_summary_path else None,
                    "summary_error": retrieval_summary_error,
                    "returncode": retrieval_run["returncode"],
                    "failed": retrieval_run["returncode"] != 0,
                    "metrics": retrieval_metrics,
                },
            }

            if args.fail_fast and retrieval_run["returncode"] != 0:
                comparison["variants"].append(variant_record)
                comparison["child_failures"] = child_failures
                write_json(run_root / "comparison.partial.json", comparison)
                raise SystemExit(f"Variant {name} retrieval failed. See {variant_dir}/retrieval_stdout_tail.txt")

            if args.answer or args.judge:
                answer_cmd = [sys.executable, "eval_answer.py"]
                for case_id in case_ids:
                    answer_cmd += ["--case", case_id]
                if args.judge:
                    env["RAG_ANSWER_EVAL_LLM_JUDGE"] = "1"
                print(f"=== {name}: answer {'+ LLM judge' if args.judge else ''} ===", flush=True)
                answer_run = run_subprocess(answer_cmd, env, repo_dir, label=f"{name}: answer")
                (variant_dir / "answer_stdout_tail.txt").write_text(answer_run["stdout_tail"], encoding="utf-8")

                answer_summary_path, answer_summary, answer_summary_error = try_latest_summary(variant_dir, "answer_eval")
                if answer_summary is not None:
                    answer_metrics = compact_answer_summary(answer_summary)
                else:
                    answer_metrics = {
                        "summary_missing": True,
                        "summary_error": answer_summary_error,
                        "subprocess_returncode": answer_run["returncode"],
                    }
                answer_metrics["subprocess_returncode"] = answer_run["returncode"]
                answer_metrics["subprocess_failed"] = answer_run["returncode"] != 0
                if answer_run["returncode"] != 0:
                    child_failures.append({"variant": name, "base_variant": base_name_variant, "stage": "answer", "returncode": answer_run["returncode"]})
                    print(
                        f"WARNING: {name} answer exited {answer_run['returncode']}; "
                        "keeping partial metrics and continuing.",
                        flush=True,
                    )

                answer_rows.append({"variant": name, "metrics": answer_metrics})
                variant_record["answer"] = {
                    "summary_path": str(answer_summary_path) if answer_summary_path else None,
                    "summary_error": answer_summary_error,
                    "returncode": answer_run["returncode"],
                    "failed": answer_run["returncode"] != 0,
                    "metrics": answer_metrics,
                }

                if args.fail_fast and answer_run["returncode"] != 0:
                    comparison["variants"].append(variant_record)
                    comparison["child_failures"] = child_failures
                    write_json(run_root / "comparison.partial.json", comparison)
                    raise SystemExit(f"Variant {name} answer failed. See {variant_dir}/answer_stdout_tail.txt")

            comparison["variants"].append(variant_record)
            comparison["child_failures"] = child_failures
            write_json(run_root / "comparison.partial.json", comparison)

    comparison["child_failures"] = child_failures
    comparison["completed_variants"] = len(comparison["variants"])
    comparison["variant_child_failure_count"] = len(child_failures)
    ranking_rows = rank_variants(comparison)
    comparison["variant_ranking"] = ranking_rows
    study_report = None
    if args.study:
        study_report = build_study_report(comparison)
        comparison["study_report_path"] = str(run_root / "study_report.json")
        comparison["study_recommendation"] = study_report.get("recommendation")
        comparison["study_recommendation_reason"] = study_report.get("recommendation_reason")
        write_study_report_files(run_root, study_report)
    problem_case_files = write_problem_case_files(run_root, comparison, study_report)
    problem_cases = problem_case_files.get("problem_cases", [])
    comparison["problem_cases"] = problem_cases
    comparison["problem_case_files"] = problem_case_files
    write_recommended_next_steps(run_root, ranking_rows, problem_cases, answer_enabled=bool(answer_rows))
    write_json(run_root / "comparison.json", comparison)

    md_parts = [
        f"# RAG ablation comparison: {run_root.name}",
        "",
        f"Child eval non-zero exits: {len(child_failures)}",
        "",
    ]
    if args.study:
        md_parts += [
            f"Study recommendation: `{comparison.get('study_recommendation')}`",
            "",
            comparison.get("study_recommendation_reason") or "",
            "",
        ]
    md_parts += [
        markdown_table([{"variant": row["variant"], "metrics": row} for row in ranking_rows], RANKING_KEYS, "Variant ranking / shortlist"),
        "",
        markdown_table(retrieval_rows, RETRIEVAL_COMPARE_KEYS + ["subprocess_failed"], "Retrieval"),
    ]
    if answer_rows:
        md_parts += ["", markdown_table(answer_rows, ANSWER_COMPARE_KEYS + ["subprocess_failed"], "Answer / LLM judge")]
    if problem_cases:
        md_parts += ["", "## Problem cases for the next expensive loop", ""]
        md_parts.append("These include retrieval/deterministic failures, judge semantic or infrastructure failures, regex overclaim risks, and paired A/B disagreements; they were written to `problem_cases.txt`.")
        md_parts.append("")
        for case_id in problem_cases:
            md_parts.append(f"- {case_id}")
    if child_failures:
        md_parts += ["", "## Child failures", ""]
        for item in child_failures:
            md_parts.append(f"- {item['variant']} / {item['stage']} exited {item['returncode']}")
    (run_root / "comparison.md").write_text("\n".join(md_parts) + "\n", encoding="utf-8")

    print("=" * 100)
    print(f"Comparison written to: {run_root}")
    print(f"- {run_root / 'comparison.json'}")
    print(f"- {run_root / 'comparison.md'}")
    if args.study:
        print(f"- {run_root / 'study_report.json'}")
        print(f"- {run_root / 'study_report.md'}")
    print(f"- {run_root / 'recommended_next_commands.txt'}")
    if problem_cases:
        print(f"- {run_root / 'problem_cases.txt'}")
        for extra_name in ("semantic_problem_cases.txt", "paired_disagreement_cases.txt", "safety_tail_cases.txt"):
            extra_path = run_root / extra_name
            if extra_path.exists():
                print(f"- {extra_path}")
    if child_failures:
        print(f"Child eval non-zero exits recorded: {len(child_failures)}")
        print("These are kept as metrics; compare mode did not discard the sweep.")
    print("=" * 100)

def load_cases() -> list[dict[str, Any]]:
    path = Path(EVAL_FILE).expanduser()
    if not path.exists():
        raise SystemExit(f"Eval file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("eval_queries.json must contain a JSON list")
    return data


def read_case_file(path: str | os.PathLike[str]) -> list[str]:
    """Read a newline-delimited case-id list used for fast expensive sweeps.

    Empty lines and # comments are ignored. Keeping this in the eval script rather
    than a separate note file makes the optimization discoverable via --help and
    keeps the run artifact self-describing.
    """
    case_path = Path(path).expanduser()
    if not case_path.exists():
        raise SystemExit(f"Case file not found: {case_path}")
    out: list[str] = []
    for raw in case_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def selected_case_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    ids.extend(args.case or [])
    for path in args.case_file or []:
        ids.extend(read_case_file(path))
    seen: set[str] = set()
    out: list[str] = []
    for case_id in ids:
        if case_id not in seen:
            seen.add(case_id)
            out.append(case_id)
    return out


def run_retrieval_eval(case_ids: list[str] | None = None) -> None:
    source_map = load_source_map()
    cases = load_cases()
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.get("id") in wanted]
        found = {str(c.get("id")) for c in cases}
        missing = [case_id for case_id in case_ids if case_id not in found]
        if missing:
            raise SystemExit(f"No eval case(s) found: {missing}")

    cases = [resolve_case_sources(case, source_map) for case in cases]

    cfg = rc.retrieval_config_summary()
    print("=" * 100)
    print("RAG RETRIEVAL-ONLY EVAL START")
    print("=" * 100)
    print(f"Domain: {cfg.get('domain_id')} ({cfg.get('domain_display_name')})")
    print(f"Eval file: {EVAL_FILE}")
    print(f"Source map: {SOURCE_MAP_FILE} ({'loaded' if source_map else 'not loaded'})")
    print(f"Retrieval mode: {cfg['mode']} (hybrid {cfg['hybrid']})")
    print(f"Collection: {cfg['collection']}")
    print(f"Qdrant URL: {cfg['qdrant_url']}")
    print(f"Embedding URL: {cfg['embed_url']}")
    print(
        f"Defaults: top_k={DEFAULT_TOP_K}; pre_k={DEFAULT_PRE_K}; "
        f"max_per_file={DEFAULT_MAX_PER_FILE}; neighbor_radius={DEFAULT_NEIGHBOR_RADIUS}"
    )
    print(f"metadata_rerank_clamp={cfg['metadata_rerank_clamp']}")
    print("Eval scope: retrieval only; answer faithfulness and proxy/e2e eval are deferred.")
    print()

    run_id = now_run_id()
    run_dir = RUNS_DIR / f"retrieval_eval_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    case_records: list[dict[str, Any]] = []

    print(f"Run dir: {run_dir}")
    print()

    passed = 0
    failed = 0
    failed_ids = []

    for case in cases:
        top_k = int(case.get("top_k", DEFAULT_TOP_K))
        pre_k = int(case.get("pre_k", DEFAULT_PRE_K))
        max_per_file = int(case.get("max_per_file", DEFAULT_MAX_PER_FILE))
        neighbor_radius = int(case.get("neighbor_radius", DEFAULT_NEIGHBOR_RADIUS))

        case_started = time.time()
        retrieve_started = time.time()
        selected, candidates = rc.retrieve_dense(
            question=case["query"],
            top_k=top_k,
            pre_k=pre_k,
            max_per_file=max_per_file,
        )
        retrieve_elapsed = time.time() - retrieve_started

        expand_started = time.time()
        groups = rc.expand_results_with_neighbors(selected, radius=neighbor_radius)
        expand_elapsed = time.time() - expand_started

        failures = validate_case(case, selected, groups)
        timings = {
            "retrieve_elapsed_sec": retrieve_elapsed,
            "neighbor_expand_elapsed_sec": expand_elapsed,
            "total_elapsed_sec": time.time() - case_started,
        }
        metrics = compute_retrieval_metrics(case, selected, candidates, groups, failures, timings)
        record = compact_case_record(case, selected, candidates, groups, failures, metrics)
        case_records.append(record)
        append_jsonl(run_dir / "cases.jsonl", record)
        write_json(run_dir / f"{case['id']}.json", record)

        print_case_result(case, selected, groups, failures)
        print(
            "Metrics: "
            f"hit@1={metrics['expected_source_hit_at_1']} "
            f"hit@3={metrics['expected_source_hit_at_3']} "
            f"hit@5={metrics['expected_source_hit_at_5']} "
            f"mrr={metrics['expected_source_mrr_selected']:.3f} "
            f"ndcg@5={metrics['selected_source_ndcg_at_5']} "
            f"rank_delta={metrics['best_expected_rank_delta_dense_minus_final']} "
            f"elapsed={metrics['total_elapsed_sec']:.2f}s"
        )
        print()

        if failures:
            failed += 1
            failed_ids.append(case["id"])
        else:
            passed += 1

    aggregate = aggregate_retrieval_metrics(case_records)
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "domain_id": cfg.get("domain_id"),
        "domain_display_name": cfg.get("domain_display_name"),
        "eval_file": str(EVAL_FILE),
        "source_map": str(SOURCE_MAP_FILE),
        "retrieval_config": cfg,
        "aggregate_metrics": aggregate,
        "failed_ids": failed_ids,
        "case_result_count": len(case_records),
    }
    write_json(run_dir / "summary.json", summary)

    print("=" * 100)
    print("RAG RETRIEVAL-ONLY EVAL SUMMARY")
    print("=" * 100)
    print(f"Run dir: {run_dir}")
    print(f"Cases total: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("Aggregate metrics:")
    for key, value in aggregate.items():
        if key == "failure_kinds":
            continue
        print(f"  {key}: {value}")
    if failed_ids:
        print("\nFailed case ids:")
        for cid in failed_ids:
            print(f"  - {cid}")
    print("=" * 100)

    if failed:
        sys.exit(1)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval eval or built-in ablation comparison for local RAG.",
        epilog=(
            "Fast loop: run --compare --retrieval-only across many variants first; "
            "then rerun --answer --judge only for the top variants and --case-file problem_cases.txt."
        ),
    )
    parser.add_argument("case_pos", nargs="?", help="Optional single case id, kept for backward compatibility.")
    parser.add_argument("--case", action="append", help="Run one case id. Repeat to run a small problem-case set.")
    parser.add_argument("--case-file", action="append", help="Newline-delimited case ids; # comments and blanks are ignored.")
    parser.add_argument("--compare", action="store_true", help="Run built-in ablation comparison instead of a single retrieval eval.")
    parser.add_argument("--study", choices=["adas_rerank_selection"], help="Run a predefined paired/slice study. Implies --compare; unless --retrieval-only, also implies --answer --judge.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat expensive answer/judge runs per variant for stochastic LLM stability. Retrieval-only compare ignores repeats.")
    parser.add_argument("--include-dual-union", action="store_true", help="With --study, also test experimental dual_union_disabled_value.")
    parser.add_argument("--retrieval-only", action="store_true", help="With --compare/--study, make the cheap retrieval-only mode explicit.")
    parser.add_argument("--answer", action="store_true", help="With --compare, also run eval_answer.py for each variant.")
    parser.add_argument("--judge", action="store_true", help="With --compare, enable eval_answer.py LLM judge. Implies --answer.")
    parser.add_argument("--variant", action="append", help="With --compare, run only named variant(s). Can be repeated.")
    parser.add_argument("--keep-going", action="store_true", help=argparse.SUPPRESS)  # Backward compatible no-op; compare now keeps going by default.
    parser.add_argument("--fail-fast", action="store_true", help="With --compare, stop after the first child eval exits non-zero.")
    parser.add_argument("--list-variants", action="store_true", help="List built-in ablation variants and exit.")
    args = parser.parse_args()
    if args.study:
        args.compare = True
        if not args.retrieval_only:
            args.answer = True
            args.judge = True
    if args.judge:
        args.answer = True
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.case_pos:
        if args.case:
            raise SystemExit("Use either positional case id or --case, not both")
        args.case = [args.case_pos]
    return args


def main() -> None:
    args = parse_args()
    if args.list_variants:
        for variant in ABLATION_VARIANTS:
            print(f"{variant['name']}: {variant['description']}")
        return
    if args.compare:
        run_compare(args)
        return
    run_retrieval_eval(case_ids=selected_case_ids(args))


if __name__ == "__main__":
    main()
