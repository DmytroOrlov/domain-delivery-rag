#!/usr/bin/env python3
"""
Metadata-prior ablation for the local Domain Delivery RAG.

Purpose
-------
This script answers two narrow production-readiness questions without adding a
separate fourth eval layer:

    1. Does the LLM-generated classification metadata prior break retrieval?
    2. When both dense_raw and dense+metadata pass, does metadata improve the
       expected-evidence composition of the selected context?

It compares two modes through the same rag_core retrieval path:

    dense_raw       final_score = dense cosine score
    dense_metadata  final_score = dense cosine score + bounded metadata_prior

This is not an answer eval and not a semantic faithfulness eval. It is a
retrieval-ranking audit for a hand-written heuristic. Classification metadata is
used only for ranking; it is not answer evidence.

Scoring contract
----------------
Binary PASS/FAIL remains delegated to eval_retrieval.validate_case().

This script additionally computes a small graded retrieval score from the same
retrieval expectations already present in eval_queries.json:

    - expected source/file coverage
    - expected selected chunks
    - expected expanded chunks from neighbor expansion
    - rank placement of expected selected evidence
    - small penalty for low-value selected distractors

The score is intentionally a review aid, not a production truth metric. It makes
"both modes pass, but one selected better evidence" visible without creating a
new eval script.

Tiny rank-only score drops are reported separately as MINOR_RANK_SHIFT when the
selected and expanded evidence sets are unchanged. They still contribute to the
total graded score, but they are not counted as meaningful score regressions.

Cases with no retrieval expectations, such as no-answer/insufficient-evidence
answer-only cases, still participate in PASS/FAIL ablation but are excluded from
the graded retrieval-score total.

Usage
-----
  python3 eval_metadata_ablation.py
  python3 eval_metadata_ablation.py blind_spot_warning_implications

Helpful env vars are the same as eval_retrieval.py:
  RAG_EVAL_FILE=eval_queries.json
  RAG_EVAL_SOURCE_MAP=eval_source_map.local.json
  RAG_EVAL_TOP_K=5
  RAG_EVAL_PRE_K=24
  RAG_MAX_PER_FILE=2
  RAG_NEIGHBOR_RADIUS=1

The script writes a machine-readable summary to:
  eval_runs/metadata_ablation_<timestamp>/summary.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import eval_retrieval as ev
import rag_core as rc


RUN_ROOT = Path(
    os.environ.get(
        "RAG_EVAL_RUN_DIR",
        getattr(rc.DOMAIN, "eval_run_dir", "eval_runs"),
    )
).expanduser()

# A small negative score delta caused only by reordering the same selected
# evidence is diagnostic noise, not a meaningful retrieval regression. Keep the
# score delta visible, but classify it separately from SCORE_REGRESSED.
MINOR_RANK_SHIFT_MAX_DELTA = 0.25


def item_ref(item: dict[str, Any]) -> str:
    p = item["payload"]
    return f"{rc.file_name(p)}:#{p.get('chunk_index')}"


def refs(items: list[dict[str, Any]]) -> list[str]:
    return [item_ref(item) for item in items]


def selected_rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    return {item_ref(item): i for i, item in enumerate(items, start=1)}


def compact_failures(failures: list[str]) -> list[str]:
    return list(failures)


def has_retrieval_rubric(case: dict[str, Any]) -> bool:
    keys = (
        "expected_files_all",
        "expected_files_any",
        "expected_selected_chunks_any",
        "expected_expanded_chunks_any",
        "forbidden_files",
    )
    return any(case.get(k) for k in keys)


def chunk_ref(file_name: str, chunk_index: int) -> str:
    return f"{ev.base_name(file_name)}:#{chunk_index}"


def best_rank_for_chunks(rank: dict[str, int], file_name: str, chunks: list[int]) -> int | None:
    ranks = [rank[chunk_ref(file_name, idx)] for idx in chunks if chunk_ref(file_name, idx) in rank]
    return min(ranks) if ranks else None


def count_matching_chunks(ref_set: set[str], file_name: str, chunks: list[int]) -> int:
    return sum(1 for idx in chunks if chunk_ref(file_name, idx) in ref_set)


def selected_distractor_penalty(selected: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """Small, transparent penalty for selected chunks that are likely support/noise.

    This is intentionally mild. It should not overpower explicit expected source
    and chunk hits; it only helps make context-composition changes visible when a
    low-value data/example chunk is swapped for a domain-relevant source.
    """
    penalty = 0.0
    reasons: list[str] = []

    for item in selected:
        p = item["payload"]
        ref = item_ref(item)
        decision = rc.payload_decision(p)
        delivery = rc.payload_delivery_value(p)
        criticality = rc.payload_criticality(p)

        if decision == "drop":
            penalty += 4.0
            reasons.append(f"{ref}: drop chunk selected (-4.0)")
        elif decision == "secondary":
            penalty += 0.4
            reasons.append(f"{ref}: secondary chunk selected (-0.4)")

        if delivery == "low":
            penalty += 0.3
            reasons.append(f"{ref}: low delivery value (-0.3)")
        if criticality == "low":
            penalty += 0.2
            reasons.append(f"{ref}: low criticality (-0.2)")

    return penalty, reasons


def retrieval_quality_score(case: dict[str, Any], mode: dict[str, Any]) -> dict[str, Any] | None:
    """Compute a graded score from the same expectations used by retrieval eval.

    The score deliberately rewards *coverage and placement of expected evidence*,
    not model metadata labels. It uses metadata only indirectly through whatever
    chunks the retrieval mode selected.
    """
    if not has_retrieval_rubric(case):
        return None

    selected = mode["selected_items"]
    groups = mode["groups"]
    selected_files = mode["selected_files_set"]
    expanded_files = mode["expanded_files_set"]
    selected_refs = mode["selected_refs_set"]
    expanded_refs = mode["expanded_refs_set"]
    rank = mode["selected_rank"]
    top_k = max(1, int(case.get("top_k", ev.DEFAULT_TOP_K)))

    score = 0.0
    max_score = 0.0
    details: list[str] = []

    for fname in case.get("expected_files_all", []) or []:
        base = ev.base_name(fname)
        max_score += 2.5
        if base in expanded_files:
            score += 2.0
            details.append(f"expected source expanded: {base} (+2.0)")
        else:
            details.append(f"expected source missing: {base} (+0.0/{2.0})")
        if base in selected_files:
            score += 0.5
            details.append(f"expected source selected: {base} (+0.5)")

    any_files = [ev.base_name(x) for x in case.get("expected_files_any", []) or []]
    if any_files:
        max_score += 2.0
        expanded_hit = any(fname in expanded_files for fname in any_files)
        selected_hit = any(fname in selected_files for fname in any_files)
        if expanded_hit:
            score += 1.5
            details.append(f"expected any-source expanded: {any_files} (+1.5)")
        else:
            details.append(f"expected any-source missing: {any_files} (+0.0/{1.5})")
        if selected_hit:
            score += 0.5
            details.append(f"expected any-source selected: {any_files} (+0.5)")

    for spec in case.get("expected_selected_chunks_any", []) or []:
        fname = spec.get("file")
        chunks = spec.get("chunks", []) or []
        if not fname or not chunks:
            continue

        max_score += 4.0
        match_count = count_matching_chunks(selected_refs, fname, chunks)
        best_rank = best_rank_for_chunks(rank, fname, chunks)
        label = f"{ev.base_name(fname)} chunks {chunks}"

        if match_count:
            score += 3.0
            details.append(f"expected selected chunk hit: {label} (+3.0)")
            extra_hits = max(0, match_count - 1)
            if extra_hits:
                extra = min(0.5, 0.25 * extra_hits)
                score += extra
                details.append(f"additional selected expected chunks: {label} (+{extra:.2f})")
            if best_rank is not None:
                rank_bonus = max(0.0, (top_k - best_rank + 1) / top_k)
                score += rank_bonus
                details.append(f"selected expected chunk rank bonus: rank #{best_rank} (+{rank_bonus:.2f})")
        else:
            details.append(f"expected selected chunk missing: {label} (+0.0/{3.0})")

    for spec in case.get("expected_expanded_chunks_any", []) or []:
        fname = spec.get("file")
        chunks = spec.get("chunks", []) or []
        if not fname or not chunks:
            continue

        max_score += 2.0
        match_count = count_matching_chunks(expanded_refs, fname, chunks)
        label = f"{ev.base_name(fname)} chunks {chunks}"

        if match_count:
            score += 1.5
            details.append(f"expected expanded chunk hit: {label} (+1.5)")
            extra_hits = max(0, match_count - 1)
            if extra_hits:
                extra = min(0.5, 0.15 * extra_hits)
                score += extra
                details.append(f"additional expanded expected chunks: {label} (+{extra:.2f})")
        else:
            details.append(f"expected expanded chunk missing: {label} (+0.0/{1.5})")

    forbidden_files = [ev.base_name(x) for x in case.get("forbidden_files", []) or []]
    for fname in forbidden_files:
        if fname in expanded_files:
            score -= 3.0
            details.append(f"forbidden source expanded: {fname} (-3.0)")

    penalty, penalty_reasons = selected_distractor_penalty(selected)
    if penalty:
        score -= penalty
        details.extend(penalty_reasons)

    return {
        "score": round(score, 4),
        "max_score": round(max_score, 4),
        "normalized": round(score / max_score, 4) if max_score > 0 else None,
        "details": details,
    }


def run_mode(case: dict[str, Any], *, use_metadata_prior: bool) -> dict[str, Any]:
    top_k = int(case.get("top_k", ev.DEFAULT_TOP_K))
    pre_k = int(case.get("pre_k", ev.DEFAULT_PRE_K))
    max_per_file = int(case.get("max_per_file", ev.DEFAULT_MAX_PER_FILE))
    neighbor_radius = int(case.get("neighbor_radius", ev.DEFAULT_NEIGHBOR_RADIUS))

    selected, candidates = rc.retrieve_dense(
        question=case["query"],
        top_k=top_k,
        pre_k=pre_k,
        max_per_file=max_per_file,
        use_metadata_prior=use_metadata_prior,
    )
    groups = rc.expand_results_with_neighbors(selected, radius=neighbor_radius)
    failures = ev.validate_case(case, selected, groups)

    expanded_items = ev.expanded_items_from_groups(groups)

    mode = {
        "passed": not failures,
        "failures": compact_failures(failures),
        "selected": refs(selected),
        "expanded": sorted(ev.chunk_refs_from_groups(groups)),
        "selected_files": sorted(ev.file_set_from_items(selected)),
        "expanded_files": sorted(ev.file_set_from_items(expanded_items)),
        "selected_rank": selected_rank_map(selected),
        "top_candidates": [
            {
                "rank": i,
                "ref": item_ref(item),
                "file": rc.file_name(item["payload"]),
                "chunk": item["payload"].get("chunk_index"),
                "vector_score": item.get("vector_score"),
                "meta_bonus": item.get("meta_bonus"),
                "final_score": item.get("final_score"),
                "dense_rank": item.get("dense_rank"),
                "role": rc.payload_role(item["payload"]),
                "facets": rc.payload_facets(item["payload"]),
            }
            for i, item in enumerate(candidates[:12], start=1)
        ],
        # Internal fields used by scoring. Removed before writing summary JSON.
        "selected_items": selected,
        "groups": groups,
        "selected_files_set": ev.file_set_from_items(selected),
        "expanded_files_set": ev.file_set_from_items(expanded_items),
        "selected_refs_set": ev.chunk_refs_from_items(selected),
        "expanded_refs_set": ev.chunk_refs_from_groups(groups),
    }
    mode["quality_score"] = retrieval_quality_score(case, mode)
    return mode


def public_mode(mode: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mode.items() if not k.endswith("_set") and k not in {"selected_items", "groups"}}


def compare_case(case: dict[str, Any]) -> dict[str, Any]:
    raw = run_mode(case, use_metadata_prior=False)
    meta = run_mode(case, use_metadata_prior=True)

    raw_set = set(raw["selected"])
    meta_set = set(meta["selected"])
    added = sorted(meta_set - raw_set)
    removed = sorted(raw_set - meta_set)

    rank_changes = []
    for ref in sorted(raw_set & meta_set):
        raw_rank = raw["selected_rank"][ref]
        meta_rank = meta["selected_rank"][ref]
        if raw_rank != meta_rank:
            rank_changes.append({"ref": ref, "dense_raw_rank": raw_rank, "dense_metadata_rank": meta_rank})

    raw_score = raw["quality_score"]
    meta_score = meta["quality_score"]
    score_delta = None
    if raw_score is not None and meta_score is not None:
        score_delta = round(meta_score["score"] - raw_score["score"], 4)

    selected_set_same = raw_set == meta_set
    expanded_set_same = set(raw["expanded"]) == set(meta["expanded"])
    minor_rank_shift = (
        raw["passed"]
        and meta["passed"]
        and score_delta is not None
        and score_delta < 0
        and abs(score_delta) <= MINOR_RANK_SHIFT_MAX_DELTA
        and selected_set_same
        and expanded_set_same
        and bool(rank_changes)
    )

    status = "UNCHANGED"
    if raw["passed"] and not meta["passed"]:
        status = "REGRESSION"
    elif not raw["passed"] and meta["passed"]:
        status = "IMPROVEMENT"
    elif raw["passed"] and meta["passed"] and score_delta is not None and score_delta > 0:
        status = "SCORE_IMPROVED"
    elif minor_rank_shift:
        status = "MINOR_RANK_SHIFT"
    elif raw["passed"] and meta["passed"] and score_delta is not None and score_delta < 0:
        status = "SCORE_REGRESSED"
    elif raw["passed"] and meta["passed"] and (added or removed or rank_changes):
        status = "PASS_CHANGED"
    elif not raw["passed"] and not meta["passed"]:
        status = "BOTH_FAIL"

    return {
        "id": case["id"],
        "query": case["query"],
        "notes": case.get("notes", ""),
        "status": status,
        "score_delta": score_delta,
        "dense_raw": public_mode(raw),
        "dense_metadata": public_mode(meta),
        "selection_added_by_metadata": added,
        "selection_removed_by_metadata": removed,
        "rank_changes": rank_changes,
    }


def format_score(score: dict[str, Any] | None) -> str:
    if score is None:
        return "score=n/a"
    norm = score.get("normalized")
    norm_txt = "n/a" if norm is None else f"{norm:.3f}"
    return f"score={score['score']:.2f}/{score['max_score']:.2f} norm={norm_txt}"


def print_score_details(label: str, score: dict[str, Any] | None) -> None:
    if score is None:
        return
    print(f"{label} score details:")
    for detail in score["details"]:
        print(f"  - {detail}")


def print_case(result: dict[str, Any]) -> None:
    print("=" * 100)
    print(f"{result['status']}: {result['id']}")
    print("=" * 100)
    print(f"Query: {result['query']}")
    if result.get("notes"):
        print(f"Notes: {result['notes']}")
    print()

    raw = result["dense_raw"]
    meta = result["dense_metadata"]
    delta = result["score_delta"]
    delta_text = "n/a" if delta is None else f"{delta:+.2f}"

    print(f"dense_raw:      {'PASS' if raw['passed'] else 'FAIL'} {format_score(raw['quality_score'])}")
    print(f"dense_metadata: {'PASS' if meta['passed'] else 'FAIL'} {format_score(meta['quality_score'])}")
    print(f"score_delta:    {delta_text}")
    print()
    print(f"dense_raw selected:      [{', '.join(raw['selected'])}]")
    print(f"dense_metadata selected: [{', '.join(meta['selected'])}]")

    if result["selection_added_by_metadata"] or result["selection_removed_by_metadata"]:
        print()
        print("Selection diff after metadata prior:")
        for ref in result["selection_added_by_metadata"]:
            print(f"  + {ref}")
        for ref in result["selection_removed_by_metadata"]:
            print(f"  - {ref}")

    if result["rank_changes"]:
        print()
        print("Selected rank changes:")
        for change in result["rank_changes"]:
            print(f"  {change['ref']}: raw #{change['dense_raw_rank']} -> metadata #{change['dense_metadata_rank']}")

    if delta:
        print()
        print_score_details("dense_raw", raw["quality_score"])
        print_score_details("dense_metadata", meta["quality_score"])

    if raw["failures"] or meta["failures"]:
        print()
        if raw["failures"]:
            print("dense_raw failures:")
            for failure in raw["failures"]:
                print(f"  - {failure}")
        if meta["failures"]:
            print("dense_metadata failures:")
            for failure in meta["failures"]:
                print(f"  - {failure}")

    print()


def load_resolved_cases(only_id: str | None) -> list[dict[str, Any]]:
    source_map = ev.load_source_map()
    cases = ev.load_cases()
    if only_id:
        cases = [c for c in cases if c.get("id") == only_id]
        if not cases:
            raise SystemExit(f"No eval case with id={only_id!r}")
    return [ev.resolve_case_sources(case, source_map) for case in cases]


def main() -> None:
    only_id = sys.argv[1] if len(sys.argv) > 1 else None
    cases = load_resolved_cases(only_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"metadata_ablation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = rc.retrieval_config_summary()

    print("=" * 100)
    print("RAG METADATA PRIOR ABLATION START")
    print("=" * 100)
    print(f"Domain: {cfg.get('domain_id')} ({cfg.get('domain_display_name')})")
    print(f"Eval file: {ev.EVAL_FILE}")
    print(f"Source map: {ev.SOURCE_MAP_FILE} ({'loaded' if ev.load_source_map() else 'not loaded'})")
    print(f"Run dir: {run_dir}")
    print(f"Collection: {cfg['collection']}")
    print(f"Qdrant URL: {cfg['qdrant_url']}")
    print(f"Embedding URL: {cfg['embed_url']}")
    print(
        f"Defaults: top_k={ev.DEFAULT_TOP_K}; pre_k={ev.DEFAULT_PRE_K}; "
        f"max_per_file={ev.DEFAULT_MAX_PER_FILE}; neighbor_radius={ev.DEFAULT_NEIGHBOR_RADIUS}"
    )
    print(f"metadata_rerank_clamp={cfg['metadata_rerank_clamp']}")
    print("Compares dense_raw against dense+metadata; classification metadata is not answer evidence.")
    print("Quality score is graded from existing retrieval expectations; no fourth eval layer is introduced.")
    print()

    results = []
    for case in cases:
        result = compare_case(case)
        results.append(result)
        print_case(result)

    total = len(results)
    raw_pass = sum(1 for r in results if r["dense_raw"]["passed"])
    meta_pass = sum(1 for r in results if r["dense_metadata"]["passed"])
    regressions = [r["id"] for r in results if r["status"] == "REGRESSION"]
    improvements = [r["id"] for r in results if r["status"] == "IMPROVEMENT"]
    score_regressions = [r["id"] for r in results if r["status"] == "SCORE_REGRESSED"]
    score_improvements = [r["id"] for r in results if r["status"] == "SCORE_IMPROVED"]
    minor_rank_shifts = [r["id"] for r in results if r["status"] == "MINOR_RANK_SHIFT"]
    changed = [r["id"] for r in results if r["selection_added_by_metadata"] or r["selection_removed_by_metadata"]]

    scored_results = [r for r in results if r["score_delta"] is not None]
    raw_score_total = round(sum(r["dense_raw"]["quality_score"]["score"] for r in scored_results), 4)
    meta_score_total = round(sum(r["dense_metadata"]["quality_score"]["score"] for r in scored_results), 4)
    score_delta_total = round(meta_score_total - raw_score_total, 4)

    summary = {
        "total": total,
        "scored_cases": len(scored_results),
        "dense_raw_passed": raw_pass,
        "dense_metadata_passed": meta_pass,
        "dense_raw_quality_score_total": raw_score_total,
        "dense_metadata_quality_score_total": meta_score_total,
        "quality_score_delta_total": score_delta_total,
        "regressions": regressions,
        "improvements": improvements,
        "score_regressions": score_regressions,
        "score_improvements": score_improvements,
        "minor_rank_shifts": minor_rank_shifts,
        "selection_changed": changed,
        "results": results,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 100)
    print("RAG METADATA PRIOR ABLATION SUMMARY")
    print("=" * 100)
    print(f"Run dir: {run_dir}")
    print(f"Cases total: {total}")
    print(f"Scored retrieval-rubric cases: {len(scored_results)}")
    print(f"dense_raw passed: {raw_pass}")
    print(f"dense_metadata passed: {meta_pass}")
    print(f"dense_raw quality score total: {raw_score_total:.2f}")
    print(f"dense_metadata quality score total: {meta_score_total:.2f}")
    print(f"quality score delta total: {score_delta_total:+.2f}")
    print(f"Binary improvements: {len(improvements)}")
    for cid in improvements:
        print(f"  + {cid}")
    print(f"Binary regressions: {len(regressions)}")
    for cid in regressions:
        print(f"  - {cid}")
    print(f"Score improvements: {len(score_improvements)}")
    for cid in score_improvements:
        print(f"  + {cid}")
    print(f"Score regressions: {len(score_regressions)}")
    for cid in score_regressions:
        print(f"  - {cid}")
    print(f"Minor rank-only shifts: {len(minor_rank_shifts)}")
    for cid in minor_rank_shifts:
        print(f"  ~ {cid}")
    print(f"Selection changed: {len(changed)}")
    for cid in changed:
        print(f"  * {cid}")
    print("=" * 100)

    # Current production baseline is dense+metadata. Keep shell/CI behavior
    # useful: fail if the baseline fails any eval case, if metadata causes a
    # binary regression, or if the total graded retrieval score gets worse.
    if meta_pass != total or regressions or score_delta_total < 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
