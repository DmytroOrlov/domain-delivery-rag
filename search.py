import os
import sys

import rag_retrieval as rr

# =============================================================================
# Domain Delivery RAG - Diagnostic Search
# =============================================================================
#
# Purpose:
#   Manual microscope for retrieval debugging.
#
# This script now uses rag_retrieval.py as the single source of truth.
# It supports both:
#   - dense retrieval
#   - hybrid retrieval: Qdrant dense + SQLite FTS5 lexical + RRF fusion
#
# Typical usage:
#   python3 search.py "warning logic for pedestrian detection in blind spot scenarios" 5
#
# Useful env vars:
#   RAG_RETRIEVAL_MODE=hybrid   # default if set in rag_retrieval.py
#   RAG_RETRIEVAL_MODE=dense
#   RAG_REQUIRE_LEXICAL=1       # fail if lexical.sqlite is unavailable
#   RAG_SHOW_NEIGHBORS=1        # default; show neighbor expansion groups
#   RAG_DEBUG=1                 # print extra raw diagnostics from shared module
#
# How to read scores in hybrid mode:
#   dense_rank:
#     rank from Qdrant vector search
#
#   lexical_rank:
#     rank from SQLite FTS5 search
#
#   rrf:
#     raw reciprocal-rank-fusion score
#
#   retrieval:
#     scaled RRF score, easier to read in logs
#
#   meta:
#     bounded metadata rerank bonus
#
#   final:
#     retrieval + metadata bonus
#
# Metadata is used for retrieval diagnostics/reranking only.
# It is not answer evidence.
# =============================================================================


DEFAULT_TOP_K = int(os.environ.get("RAG_SEARCH_TOP_K", str(rr.DEFAULT_TOP_K)))
DEFAULT_PRE_K = int(os.environ.get("RAG_SEARCH_PRE_K", str(rr.DEFAULT_PRE_K)))
MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", str(rr.MAX_PER_FILE)))

SHOW_NEIGHBORS = os.environ.get("RAG_SHOW_NEIGHBORS", "1") != "0"
CONTENT_PREVIEW_CHARS = int(os.environ.get("RAG_SEARCH_CONTENT_PREVIEW_CHARS", "1600"))


def preview(text: str, n: int = CONTENT_PREVIEW_CHARS):
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "\n...[truncated]"


def fmt(value):
    return rr.fmt_score(value)


def fmt_rank(value):
    return "-" if value is None else str(value)


def print_query_profile(query: str):
    profile = rr.query_profile(query)
    print("Query profile:")
    print(rr.json.dumps(rr.pretty_profile(profile), ensure_ascii=False, indent=2))
    print()


def print_top_candidates(candidates, limit=10):
    print("=" * 100)
    print("TOP RAW CANDIDATES BEFORE DIVERSITY")
    print("=" * 100)

    for i, item in enumerate(candidates[:limit], start=1):
        p = item["payload"]

        print(
            f"{i:02d}. "
            f"mode={item.get('retrieval_mode')} "
            f"final={fmt(item.get('final_score'))} "
            f"retrieval={fmt(item.get('retrieval_score'))} "
            f"vector={fmt(item.get('vector_score'))} "
            f"rrf={fmt(item.get('rrf_score'))} "
            f"dense_rank={fmt_rank(item.get('dense_rank'))} "
            f"lexical_rank={fmt_rank(item.get('lexical_rank'))} "
            f"meta={fmt(item.get('meta_bonus'))} "
            f"file={p.get('file_name')} "
            f"chunk={p.get('chunk_index')} "
            f"role={p.get('chunk_role')} "
            f"facets={rr.fmt_list(p.get('content_facets'))} "
            f"layers={rr.fmt_list(p.get('system_layers'))} "
            f"safety={p.get('safety_relevance')} "
            f"delivery={p.get('delivery_value')} "
            f"decision={p.get('corpus_decision')}"
        )

    print()


def print_selected_result(rank: int, item: dict):
    p = item["payload"]

    print("=" * 100)
    print(f"RANK: {rank}")
    print(f"MODE: {item.get('retrieval_mode')}")
    print(f"FINAL_SCORE: {fmt(item.get('final_score'))}")
    print(f"RETRIEVAL_SCORE: {fmt(item.get('retrieval_score'))}")
    print(f"VECTOR_SCORE: {fmt(item.get('vector_score'))}")
    print(f"RRF_SCORE: {fmt(item.get('rrf_score'))}")
    print(f"DENSE_RANK: {fmt_rank(item.get('dense_rank'))}")
    print(f"DENSE_SCORE: {fmt(item.get('dense_score'))}")
    print(f"LEXICAL_RANK: {fmt_rank(item.get('lexical_rank'))}")
    print(f"LEXICAL_SCORE: {fmt(item.get('lexical_score'))}")
    print(f"META_BONUS: {fmt(item.get('meta_bonus'))}")
    print(f"FILE: {p.get('file_path')}")
    print(f"CHUNK: {p.get('chunk_index')}")
    print("-" * 100)

    print("CHUNK METADATA")
    print(f"chunk_role: {p.get('chunk_role')}")
    print(f"content_facets: {rr.fmt_list(p.get('content_facets'))}")
    print(f"system_layers: {rr.fmt_list(p.get('system_layers'))}")
    print(f"workflow_stages: {rr.fmt_list(p.get('workflow_stages'))}")
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
    print(f"document_roles: {rr.fmt_list(p.get('document_roles'))}")
    print(f"document_content_facets: {rr.fmt_list(p.get('document_content_facets'))}")
    print(f"document_system_layers: {rr.fmt_list(p.get('document_system_layers'))}")
    print(f"document_workflow_stages: {rr.fmt_list(p.get('document_workflow_stages'))}")
    print(f"document_safety_relevance: {p.get('document_safety_relevance')}")
    print(f"document_delivery_value: {p.get('document_delivery_value')}")
    print(f"document_corpus_decision: {p.get('document_corpus_decision')}")
    print(f"document_confidence: {p.get('document_confidence')}")
    print(f"document_signal_chunks: {p.get('document_signal_chunks')}")

    print("-" * 100)
    print("CONTENT PREVIEW")
    print(preview(p.get("content") or ""))
    print()


def print_source_groups(source_groups):
    print("=" * 100)
    print("SOURCE GROUPS AFTER NEIGHBOR EXPANSION")
    print("=" * 100)

    selected_total = 0
    expanded_total = 0

    for rank, group in enumerate(source_groups, start=1):
        selected = group.get("selected_indices") or []
        expanded = group.get("expanded_indices") or []

        selected_total += len(selected)
        expanded_total += len(expanded)

        print(
            f"{rank}. file={group.get('file_path')} "
            f"selected_chunks={selected} "
            f"expanded_chunks={expanded} "
            f"best_final={fmt(group.get('best_final_score'))}"
        )

        for item in group["chunks"]:
            p = item["payload"]
            marker = "*" if item.get("is_selected_hit") else "-"
            mode = "selected" if item.get("is_selected_hit") else "neighbor"

            print(
                f"   {marker} chunk={p.get('chunk_index')} "
                f"mode={mode} "
                f"retrieval_mode={item.get('retrieval_mode')} "
                f"final={fmt(item.get('final_score'))} "
                f"dense_rank={fmt_rank(item.get('dense_rank'))} "
                f"lexical_rank={fmt_rank(item.get('lexical_rank'))} "
                f"role={p.get('chunk_role')} "
                f"facets={rr.fmt_list(p.get('content_facets'))} "
                f"safety={p.get('safety_relevance')} "
                f"delivery={p.get('delivery_value')} "
                f"decision={p.get('corpus_decision')}"
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

    print("=" * 100)
    print("RAG SEARCH START")
    print("=" * 100)
    print(f"Query: {query}")
    rr.print_retrieval_config()
    print(f"search_top_k={top_k}; search_pre_k={pre_k}; max_per_file={MAX_PER_FILE}")
    print(f"show_neighbors={SHOW_NEIGHBORS}; content_preview_chars={CONTENT_PREVIEW_CHARS}")
    print(f"Verbose: {rr.VERBOSE}; Debug raw payloads: {rr.DEBUG}")
    print()

    print_query_profile(query)

    results, candidates = rr.retrieve(
        query=query,
        top_k=top_k,
        pre_k=pre_k,
        max_per_file=MAX_PER_FILE,
    )

    print_top_candidates(candidates, limit=min(12, len(candidates)))

    print("=" * 100)
    print("SELECTED RESULTS AFTER DIVERSITY")
    print("=" * 100)

    for rank, item in enumerate(results, start=1):
        print_selected_result(rank, item)

    source_groups = []
    if SHOW_NEIGHBORS:
        source_groups = rr.expand_results_with_neighbors(
            results,
            radius=rr.NEIGHBOR_RADIUS,
        )
        print_source_groups(source_groups)

    print("=" * 100)
    print("RAG SEARCH DONE")
    print("=" * 100)
    print(f"Retrieval mode: {rr.RETRIEVAL_MODE}")
    print(f"Lexical available: {rr.lexical_schema_available()}")
    print(f"Raw candidates: {len(candidates)}")
    print(f"Selected results: {len(results)}")
    if SHOW_NEIGHBORS:
        print(f"Source groups: {len(source_groups)}")
    print("=" * 100)


if __name__ == "__main__":
    main()