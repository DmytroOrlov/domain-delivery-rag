#!/usr/bin/env python3
"""
Retrieval-only eval for the local ADAS / Embedded Vision Delivery RAG v1.

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
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import rag_core as rc

EVAL_FILE = os.environ.get("RAG_EVAL_FILE", rc.DOMAIN.eval_file)
SOURCE_MAP_FILE = os.environ.get("RAG_EVAL_SOURCE_MAP", rc.DOMAIN.eval_source_map)
DEFAULT_TOP_K = int(os.environ.get("RAG_EVAL_TOP_K", str(rc.DEFAULT_TOP_K)))
DEFAULT_PRE_K = int(os.environ.get("RAG_EVAL_PRE_K", str(rc.DEFAULT_PRE_K)))
DEFAULT_MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", str(rc.DEFAULT_MAX_PER_FILE)))
DEFAULT_NEIGHBOR_RADIUS = int(os.environ.get("RAG_NEIGHBOR_RADIUS", str(rc.DEFAULT_NEIGHBOR_RADIUS)))


def base_name(path_or_name: str) -> str:
    return os.path.basename(path_or_name or "")


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
        if p.get("corpus_decision") == "drop":
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
            f"role={p.get('chunk_role')} "
            f"facets={p.get('content_facets')} "
            f"safety={p.get('safety_relevance')} "
            f"delivery={p.get('delivery_value')} "
            f"decision={p.get('corpus_decision')}"
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


def load_cases() -> list[dict[str, Any]]:
    path = Path(EVAL_FILE)
    if not path.exists():
        raise SystemExit(f"Eval file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("eval_queries.json must contain a JSON list")
    return data


def main():
    only_id = sys.argv[1] if len(sys.argv) > 1 else None
    source_map = load_source_map()
    cases = load_cases()
    if only_id:
        cases = [c for c in cases if c.get("id") == only_id]
        if not cases:
            raise SystemExit(f"No eval case with id={only_id!r}")

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

    passed = 0
    failed = 0
    failed_ids = []

    for case in cases:
        top_k = int(case.get("top_k", DEFAULT_TOP_K))
        pre_k = int(case.get("pre_k", DEFAULT_PRE_K))
        max_per_file = int(case.get("max_per_file", DEFAULT_MAX_PER_FILE))
        neighbor_radius = int(case.get("neighbor_radius", DEFAULT_NEIGHBOR_RADIUS))

        selected, _candidates = rc.retrieve_dense(
            question=case["query"],
            top_k=top_k,
            pre_k=pre_k,
            max_per_file=max_per_file,
        )
        groups = rc.expand_results_with_neighbors(selected, radius=neighbor_radius)
        failures = validate_case(case, selected, groups)
        print_case_result(case, selected, groups, failures)

        if failures:
            failed += 1
            failed_ids.append(case["id"])
        else:
            passed += 1

    print("=" * 100)
    print("RAG RETRIEVAL-ONLY EVAL SUMMARY")
    print("=" * 100)
    print(f"Cases total: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if failed_ids:
        print("\nFailed case ids:")
        for cid in failed_ids:
            print(f"  - {cid}")
    print("=" * 100)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
