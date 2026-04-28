#!/usr/bin/env python3
"""
Diagnostic search for the local Domain Delivery RAG.

This script is intentionally a thin shell around rag_core.py. It must use the
same retrieval, neighbor expansion, and context-packing contract as rag_proxy.py
and eval_retrieval.py. That prevents the old drift where search/ask/proxy looked
similar but were actually different products.

Usage:
  python3 search.py "query" [top_k] [pre_k]

Useful env vars:
  RAG_SHOW_CONTEXT=1                  print exact packed LLM context
  RAG_SEARCH_CONTENT_PREVIEW_CHARS=1600
  RAG_PRE_K=24
  RAG_MAX_PER_FILE=2
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import rag_core as rc

DEFAULT_TOP_K = int(os.environ.get("RAG_SEARCH_TOP_K", str(rc.DEFAULT_TOP_K)))
DEFAULT_PRE_K = int(os.environ.get("RAG_SEARCH_PRE_K", str(rc.DEFAULT_PRE_K)))
MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", str(rc.DEFAULT_MAX_PER_FILE)))
NEIGHBOR_RADIUS = int(os.environ.get("RAG_NEIGHBOR_RADIUS", str(rc.DEFAULT_NEIGHBOR_RADIUS)))
SHOW_CONTEXT = os.environ.get("RAG_SHOW_CONTEXT", "0") == "1"
CONTENT_PREVIEW_CHARS = int(os.environ.get("RAG_SEARCH_CONTENT_PREVIEW_CHARS", "1600"))


def preview(text: str, n: int = CONTENT_PREVIEW_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "\n...[truncated]"


def fmt_rank(value: Any) -> str:
    return "-" if value is None else str(value)


def print_config(query: str, top_k: int, pre_k: int):
    cfg = rc.retrieval_config_summary()
    print("=" * 100)
    print("RAG SEARCH START")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Domain: {cfg.get('domain_id', 'unknown')} ({cfg.get('domain_display_name', 'unknown')})")
    print(f"Retrieval mode: {cfg['mode']} (hybrid {cfg['hybrid']})")
    print(f"Collection: {cfg['collection']}")
    print(f"Qdrant URL: {cfg['qdrant_url']}")
    print(f"Embedding URL: {cfg['embed_url']}")
    print(
        f"top_k={top_k}; pre_k={pre_k}; max_per_file={MAX_PER_FILE}; "
        f"neighbor_radius={NEIGHBOR_RADIUS}"
    )
    print(
        f"context packing: selected_max_chars={cfg['selected_max_chars']}; "
        f"neighbor_snippet_chars={cfg['neighbor_snippet_chars']}; "
        f"context_max_chars={cfg['context_max_chars']}"
    )
    print(f"metadata_rerank_clamp={cfg['metadata_rerank_clamp']}")
    print("classification metadata used for ranking only; not answer evidence")
    print()
    print("Query profile:")
    print(json.dumps(rc.pretty_profile(rc.query_profile(query)), ensure_ascii=False, indent=2))
    print()


def print_candidates(candidates: list[dict[str, Any]], limit: int = 12):
    print("=" * 100)
    print("TOP RAW CANDIDATES BEFORE DIVERSITY")
    print("=" * 100)
    for i, item in enumerate(candidates[:limit], start=1):
        p = item["payload"]
        print(
            f"{i:02d}. "
            f"final={rc.fmt_score(item.get('final_score'))} "
            f"vector={rc.fmt_score(item.get('vector_score'))} "
            f"dense_rank={fmt_rank(item.get('dense_rank'))} "
            f"meta={rc.fmt_score(item.get('meta_bonus'))} "
            f"file={rc.file_name(p)} "
            f"chunk={p.get('chunk_index')} "
            f"role={rc.payload_role(p)} "
            f"facets={rc.fmt_list(rc.payload_facets(p))} "
            f"layers={rc.fmt_list(rc.payload_layers(p))} "
            f"criticality={rc.payload_criticality(p)} "
            f"delivery={rc.payload_delivery_value(p)} "
            f"decision={rc.payload_decision(p)}"
        )
    print()


def print_selected_result(rank: int, item: dict[str, Any]):
    p = item["payload"]
    print("=" * 100)
    print(f"RANK: {rank}")
    print(f"FINAL_SCORE: {rc.fmt_score(item.get('final_score'))}")
    print(f"VECTOR_SCORE: {rc.fmt_score(item.get('vector_score'))}")
    print(f"DENSE_RANK: {fmt_rank(item.get('dense_rank'))}")
    print(f"META_BONUS: {rc.fmt_score(item.get('meta_bonus'))}")
    print(f"FILE: {p.get('file_path')}")
    print(f"CHUNK: {p.get('chunk_index')}")
    print("-" * 100)
    print("CHUNK METADATA (diagnostic only, not answer evidence)")
    print(f"role: {rc.payload_role(p)}")
    print(f"facets: {rc.fmt_list(rc.payload_facets(p))}")
    print(f"layers: {rc.fmt_list(rc.payload_layers(p))}")
    print(f"stages: {rc.fmt_list(rc.payload_stages(p))}")
    print(f"criticality: {rc.payload_criticality(p)}")
    print(f"delivery_value: {rc.payload_delivery_value(p)}")
    print(f"decision: {rc.payload_decision(p)}")
    flag_fields = rc.boolean_flag_fields()
    if flag_fields:
        print("boolean flags:")
        for flag in flag_fields:
            print(f"  {flag}: {p.get(flag)}")
    print(f"confidence: {p.get('confidence')}")
    print(f"reason_short: {p.get('reason_short')}")
    print("-" * 100)
    print("CONTENT PREVIEW")
    print(preview(p.get("content") or ""))
    print()


def print_source_groups(source_groups: list[dict[str, Any]]):
    print("=" * 100)
    print("SOURCE GROUPS AFTER NEIGHBOR EXPANSION")
    print("=" * 100)
    selected_total = 0
    expanded_total = 0
    for i, group in enumerate(source_groups, start=1):
        selected = group.get("selected_indices") or []
        expanded = group.get("expanded_indices") or []
        selected_total += len(selected)
        expanded_total += len(expanded)
        print(
            f"{i}. file={group.get('file_name')} "
            f"selected_chunks={selected} "
            f"expanded_chunks={expanded} "
            f"best_final={rc.fmt_score(group.get('best_final_score'))}"
        )
        for item in group["chunks"]:
            p = item["payload"]
            marker = "*" if item.get("is_selected_hit") else "-"
            mode = "selected" if item.get("is_selected_hit") else "neighbor"
            print(
                f"   {marker} chunk={p.get('chunk_index')} "
                f"mode={mode} "
                f"final={rc.fmt_score(item.get('final_score'))} "
                f"dense_rank={fmt_rank(item.get('dense_rank'))} "
                f"role={rc.payload_role(p)} "
                f"facets={rc.fmt_list(rc.payload_facets(p))} "
                f"criticality={rc.payload_criticality(p)} "
                f"delivery={rc.payload_delivery_value(p)} "
                f"decision={rc.payload_decision(p)}"
            )
    print()
    print("Source group summary:")
    print(f"  source_groups={len(source_groups)}")
    print(f"  selected_chunks={selected_total}")
    print(f"  expanded_chunks={expanded_total}")
    print()


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 search.py "your query" [top_k] [pre_k]')
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOP_K
    pre_k = int(sys.argv[3]) if len(sys.argv) > 3 else max(DEFAULT_PRE_K, top_k * 4)

    print_config(query, top_k, pre_k)

    selected, candidates = rc.retrieve_dense(
        question=query,
        top_k=top_k,
        pre_k=pre_k,
        max_per_file=MAX_PER_FILE,
    )
    source_groups = rc.expand_results_with_neighbors(selected, radius=NEIGHBOR_RADIUS)
    context = rc.build_context(query, source_groups)

    print_candidates(candidates)

    print("=" * 100)
    print("SELECTED RESULTS AFTER DIVERSITY")
    print("=" * 100)
    for rank, item in enumerate(selected, start=1):
        print_selected_result(rank, item)

    print_source_groups(source_groups)

    print("=" * 100)
    print("CONTEXT CONTRACT SUMMARY")
    print("=" * 100)
    print("This is the same context-packing contract used by rag_proxy.py.")
    print("Full local paths are excluded from LLM context by default.")
    print("Classification metadata is excluded from LLM context.")
    print(f"context_chars={len(context)}")
    print(f"approx_context_tokens={len(context) // 4}")
    print(f"show_context={SHOW_CONTEXT}")
    print()
    if SHOW_CONTEXT:
        print(context)
        print()

    print("=" * 100)
    print("RAG SEARCH DONE")
    print("=" * 100)
    print(f"Raw candidates: {len(candidates)}")
    print(f"Selected results: {len(selected)}")
    print(f"Source groups: {len(source_groups)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
