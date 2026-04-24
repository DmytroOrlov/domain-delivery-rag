import json
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

# =============================================================================
# Domain Delivery RAG - Shared Retrieval Module
# =============================================================================
#
# Purpose:
#   Single source of truth for retrieval logic used by:
#     - search.py
#     - ask_qwen.py
#     - eval_retrieval.py
#
# Retrieval modes:
#   dense:
#     query -> Qdrant dense vector search -> metadata rerank -> diversity
#
#   hybrid:
#     query -> Qdrant dense vector search
#           -> SQLite FTS5 lexical search
#           -> RRF fusion
#           -> bounded metadata rerank
#           -> diversity
#
# Why hybrid:
#   Dense embeddings are good for semantic/domain similarity.
#   Lexical search is better for exact engineering terms:
#     CAN, GPIO, RKNN, R151, IP69k, 120 ms, GS-Bau-70, TC-01, etc.
#
# Expected lexical SQLite schema:
#
#   CREATE TABLE chunks (
#       rowid INTEGER PRIMARY KEY,
#       file_path TEXT NOT NULL,
#       file_name TEXT NOT NULL,
#       chunk_index INTEGER NOT NULL,
#       content TEXT NOT NULL
#   );
#
#   CREATE VIRTUAL TABLE chunks_fts USING fts5(
#       content,
#       file_name,
#       tokenize='unicode61',
#       content='chunks',
#       content_rowid='rowid'
#   );
#
# The ingest pipeline should keep Qdrant and lexical.sqlite in sync.
#
# Current design:
#   - metadata is used for retrieval/rerank/debug only
#   - metadata is NOT answer evidence
#   - chunks with corpus_decision="drop" should normally not be indexed
# =============================================================================


# =============================================================================
# Configuration
# =============================================================================

EMBED_URL = os.environ.get("RAG_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")
QDRANT_URL = os.environ.get("RAG_QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("RAG_COLLECTION", "rag_v1_chunks")

LEXICAL_DB_PATH = os.environ.get(
    "RAG_LEXICAL_DB",
    os.path.expanduser("~/rag_v1/lexical.sqlite"),
)

RETRIEVAL_MODE = os.environ.get("RAG_RETRIEVAL_MODE", "hybrid").lower()
REQUIRE_LEXICAL = os.environ.get("RAG_REQUIRE_LEXICAL", "0") == "1"

DEFAULT_TOP_K = int(os.environ.get("RAG_ASK_TOP_K", "4"))
DEFAULT_PRE_K = int(os.environ.get("RAG_ASK_PRE_K", "24"))
MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", "2"))

# Candidate pools before final diversity.
DENSE_LIMIT = int(os.environ.get("RAG_DENSE_LIMIT", "80"))
LEXICAL_LIMIT = int(os.environ.get("RAG_LEXICAL_LIMIT", "80"))

# Reciprocal Rank Fusion.
# RRF score is rank-based and avoids score calibration between dense and lexical.
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
RRF_DENSE_WEIGHT = float(os.environ.get("RAG_RRF_DENSE_WEIGHT", "1.0"))
RRF_LEXICAL_WEIGHT = float(os.environ.get("RAG_RRF_LEXICAL_WEIGHT", "1.0"))

# RRF raw values are small, roughly ~0.01-0.03.
# Scaling makes final scores easier to read in logs.
RRF_SCALE = float(os.environ.get("RAG_RRF_SCALE", "10.0"))

NEIGHBOR_RADIUS = int(os.environ.get("RAG_NEIGHBOR_RADIUS", "1"))

VERBOSE = os.environ.get("RAG_VERBOSE", "1") != "0"
DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"


# =============================================================================
# Logging
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
# Query profiling
# =============================================================================

def tokenize_query(query: str):
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9_#+.-]{2,}", query.lower())

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


# =============================================================================
# Bounded heuristic metadata reranking
# =============================================================================

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
    Metadata can reorder very close retrieval candidates, but should not dominate
    dense/lexical retrieval.
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
# Payload/key helpers
# =============================================================================

def file_name_from_payload(payload: dict):
    return payload.get("file_name") or os.path.basename(payload.get("file_path", ""))


def payload_key(payload: dict):
    return (payload.get("file_path"), payload.get("chunk_index"))


def make_key(file_path: str, chunk_index: int):
    return (file_path, chunk_index)


def fmt_list(value):
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)


def fmt_score(value):
    if value is None:
        return "none"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


# =============================================================================
# Dense retrieval from Qdrant
# =============================================================================

def dense_search(query: str, limit: int):
    client = QdrantClient(url=QDRANT_URL)
    qvec = embed(query)

    raw = client.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=limit,
        with_payload=True,
    ).points

    results = []
    for rank, point in enumerate(raw, start=1):
        payload = point.payload or {}

        results.append(
            {
                "key": payload_key(payload),
                "payload": payload,
                "dense_rank": rank,
                "dense_score": float(point.score),
            }
        )

    return results


# =============================================================================
# Lexical retrieval from SQLite FTS5
# =============================================================================

def lexical_db_exists():
    return Path(LEXICAL_DB_PATH).exists()


def lexical_schema_available():
    if not lexical_db_exists():
        return False

    try:
        conn = sqlite3.connect(LEXICAL_DB_PATH)
        cur = conn.cursor()

        rows = cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name IN ('chunks', 'chunks_fts')
            """
        ).fetchall()

        conn.close()

        names = {r[0] for r in rows}
        return "chunks" in names and "chunks_fts" in names
    except Exception:
        return False


def make_fts_query(query: str, max_terms: int = 32):
    """
    Convert a natural language query into a safe FTS5 OR query.

    We intentionally use OR rather than AND:
    - OR improves recall for engineering queries.
    - Final ranking/fusion will decide what survives.
    """
    terms = tokenize_query(query)

    unique_terms = []
    seen = set()
    for term in terms:
        cleaned = term.strip().replace('"', "")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique_terms.append(cleaned)

    unique_terms = unique_terms[:max_terms]

    if not unique_terms:
        return ""

    return " OR ".join(f'"{t}"' for t in unique_terms)


def lexical_search(query: str, limit: int):
    """
    Return lexical hits from SQLite FTS5.

    Output items contain:
      key=(file_path, chunk_index)
      file_path/file_name/chunk_index
      lexical_rank
      lexical_score

    Lower bm25() is better in SQLite FTS5, but we use rank for RRF.
    """
    if not lexical_schema_available():
        msg = (
            f"Lexical DB unavailable or schema missing: {LEXICAL_DB_PATH}. "
            f"Hybrid retrieval will fall back to dense-only unless RAG_REQUIRE_LEXICAL=1."
        )

        if REQUIRE_LEXICAL:
            raise RuntimeError(msg)

        debug_print("LEXICAL UNAVAILABLE", msg)
        return []

    fts_query = make_fts_query(query)
    if not fts_query:
        return []

    debug_print("FTS QUERY", fts_query)

    try:
        conn = sqlite3.connect(LEXICAL_DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        rows = cur.execute(
            """
            SELECT
                c.file_path AS file_path,
                c.file_name AS file_name,
                c.chunk_index AS chunk_index,
                bm25(chunks_fts) AS lexical_score
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY lexical_score ASC
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()

        conn.close()

    except sqlite3.OperationalError as e:
        msg = f"SQLite FTS query failed: {e}; query={fts_query!r}"

        if REQUIRE_LEXICAL:
            raise RuntimeError(msg) from e

        debug_print("LEXICAL QUERY FAILED", msg)
        return []

    results = []
    for rank, row in enumerate(rows, start=1):
        file_path = row["file_path"]
        chunk_index = int(row["chunk_index"])

        results.append(
            {
                "key": make_key(file_path, chunk_index),
                "file_path": file_path,
                "file_name": row["file_name"],
                "chunk_index": chunk_index,
                "lexical_rank": rank,
                "lexical_score": float(row["lexical_score"]),
            }
        )

    return results


# =============================================================================
# Qdrant payload fetch for lexical-only hits and neighbor expansion
# =============================================================================

def fetch_chunk_payloads_for_file(client: QdrantClient, file_path: str, indices: set[int]):
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


def fetch_payloads_by_keys(keys: set[tuple]):
    """
    Fetch Qdrant payloads for (file_path, chunk_index) keys.

    Used for lexical-only results because FTS stores only minimal fields.
    """
    client = QdrantClient(url=QDRANT_URL)

    by_file = defaultdict(set)
    for file_path, chunk_index in keys:
        if file_path is None or chunk_index is None:
            continue
        by_file[file_path].add(chunk_index)

    out = {}
    for file_path, indices in by_file.items():
        payloads = fetch_chunk_payloads_for_file(client, file_path=file_path, indices=indices)
        for idx, payload in payloads.items():
            out[(file_path, idx)] = payload

    return out


# =============================================================================
# Fusion and final selection
# =============================================================================

def rrf_score(
    dense_rank: int | None,
    lexical_rank: int | None,
    rrf_k: int = RRF_K,
):
    score = 0.0

    if dense_rank is not None:
        score += RRF_DENSE_WEIGHT / (rrf_k + dense_rank)

    if lexical_rank is not None:
        score += RRF_LEXICAL_WEIGHT / (rrf_k + lexical_rank)

    return score


def apply_diversity(candidates: list[dict], top_k: int, max_per_file: int):
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

    return selected


def dense_retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    pre_k: int = DEFAULT_PRE_K,
    max_per_file: int = MAX_PER_FILE,
):
    dense_limit = max(pre_k, top_k * 4)
    dense_hits = dense_search(query, limit=dense_limit)

    candidates = []

    for hit in dense_hits:
        payload = hit["payload"]
        vector_score = hit["dense_score"]
        meta_bonus = metadata_prior(payload, query=query)
        final_score = vector_score + meta_bonus

        candidates.append(
            {
                "retrieval_mode": "dense",
                "key": hit["key"],
                "payload": payload,
                "is_selected_hit": True,

                "dense_rank": hit["dense_rank"],
                "dense_score": hit["dense_score"],
                "vector_score": hit["dense_score"],

                "lexical_rank": None,
                "lexical_score": None,

                "rrf_score": None,
                "retrieval_score": vector_score,

                "meta_bonus": meta_bonus,
                "final_score": final_score,
            }
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    selected = apply_diversity(candidates, top_k=top_k, max_per_file=max_per_file)

    return selected, candidates


def hybrid_retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    pre_k: int = DEFAULT_PRE_K,
    max_per_file: int = MAX_PER_FILE,
):
    dense_limit = max(DENSE_LIMIT, pre_k, top_k * 4)
    lexical_limit = max(LEXICAL_LIMIT, pre_k, top_k * 4)

    dense_hits = dense_search(query, limit=dense_limit)
    lexical_hits = lexical_search(query, limit=lexical_limit)

    # If lexical is unavailable, fall back to dense-only.
    if not lexical_hits:
        selected, candidates = dense_retrieve(
            query=query,
            top_k=top_k,
            pre_k=pre_k,
            max_per_file=max_per_file,
        )

        for item in candidates:
            item["retrieval_mode"] = "dense_fallback"

        for item in selected:
            item["retrieval_mode"] = "dense_fallback"

        return selected, candidates

    dense_by_key = {hit["key"]: hit for hit in dense_hits}
    lexical_by_key = {hit["key"]: hit for hit in lexical_hits}

    all_keys = set(dense_by_key) | set(lexical_by_key)

    # Dense hits already have full Qdrant payload.
    payload_by_key = {
        key: hit["payload"]
        for key, hit in dense_by_key.items()
        if hit.get("payload")
    }

    # Fetch payloads for lexical-only hits.
    lexical_only_keys = all_keys - set(payload_by_key)
    payload_by_key.update(fetch_payloads_by_keys(lexical_only_keys))

    candidates = []

    for key in all_keys:
        payload = payload_by_key.get(key)
        if not payload:
            # This can happen if lexical.sqlite is out of sync with Qdrant.
            continue

        dense_hit = dense_by_key.get(key)
        lexical_hit = lexical_by_key.get(key)

        dense_rank = dense_hit["dense_rank"] if dense_hit else None
        dense_score = dense_hit["dense_score"] if dense_hit else None

        lexical_rank = lexical_hit["lexical_rank"] if lexical_hit else None
        lexical_score = lexical_hit["lexical_score"] if lexical_hit else None

        raw_rrf = rrf_score(dense_rank=dense_rank, lexical_rank=lexical_rank)
        retrieval_score = raw_rrf * RRF_SCALE
        meta_bonus = metadata_prior(payload, query=query)
        final_score = retrieval_score + meta_bonus

        candidates.append(
            {
                "retrieval_mode": "hybrid",
                "key": key,
                "payload": payload,
                "is_selected_hit": True,

                "dense_rank": dense_rank,
                "dense_score": dense_score,
                "vector_score": dense_score if dense_score is not None else 0.0,

                "lexical_rank": lexical_rank,
                "lexical_score": lexical_score,

                "rrf_score": raw_rrf,
                "retrieval_score": retrieval_score,

                "meta_bonus": meta_bonus,
                "final_score": final_score,
            }
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    selected = apply_diversity(candidates, top_k=top_k, max_per_file=max_per_file)

    return selected, candidates


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    pre_k: int = DEFAULT_PRE_K,
    max_per_file: int = MAX_PER_FILE,
    mode: str | None = None,
):
    """
    Main retrieval entry point.

    Returns:
      selected, candidates

    selected:
      final top_k after diversity

    candidates:
      full sorted candidate list before diversity

    Each item contains at least:
      payload
      final_score
      vector_score
      meta_bonus
      is_selected_hit

    Hybrid items additionally contain:
      dense_rank
      lexical_rank
      rrf_score
      retrieval_score
    """
    mode = (mode or RETRIEVAL_MODE).lower()

    if mode == "dense":
        return dense_retrieve(
            query=query,
            top_k=top_k,
            pre_k=pre_k,
            max_per_file=max_per_file,
        )

    if mode == "hybrid":
        return hybrid_retrieve(
            query=query,
            top_k=top_k,
            pre_k=pre_k,
            max_per_file=max_per_file,
        )

    raise ValueError(f"Unknown retrieval mode: {mode!r}. Use 'dense' or 'hybrid'.")


# =============================================================================
# Neighbor expansion
# =============================================================================

def expand_results_with_neighbors(results: list[dict], radius: int = NEIGHBOR_RADIUS):
    """
    Expand selected hits with neighboring chunks from the same file.

    Selected hits remain marked as is_selected_hit=True.
    Neighbor-only chunks are marked as is_selected_hit=False.
    """
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
                    "retrieval_mode": "neighbor",
                    "key": key,
                    "payload": payload,
                    "is_selected_hit": False,

                    "dense_rank": None,
                    "dense_score": None,
                    "vector_score": None,

                    "lexical_rank": None,
                    "lexical_score": None,

                    "rrf_score": None,
                    "retrieval_score": None,

                    "meta_bonus": None,
                    "final_score": None,
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
# Diagnostics
# =============================================================================

def print_candidate_summary(candidates: list[dict], limit: int = 10):
    print("Top raw candidates before diversity:")

    for i, item in enumerate(candidates[:limit], start=1):
        p = item["payload"]

        print(
            f"  {i:02d}. "
            f"mode={item.get('retrieval_mode')} "
            f"final={fmt_score(item.get('final_score'))} "
            f"retrieval={fmt_score(item.get('retrieval_score'))} "
            f"vector={fmt_score(item.get('vector_score'))} "
            f"rrf={fmt_score(item.get('rrf_score'))} "
            f"dense_rank={item.get('dense_rank')} "
            f"lexical_rank={item.get('lexical_rank')} "
            f"meta={fmt_score(item.get('meta_bonus'))} "
            f"file={p.get('file_name')} "
            f"chunk={p.get('chunk_index')} "
            f"role={p.get('chunk_role')} "
            f"facets={fmt_list(p.get('content_facets'))} "
            f"layers={fmt_list(p.get('system_layers'))} "
            f"safety={p.get('safety_relevance')} "
            f"delivery={p.get('delivery_value')} "
            f"decision={p.get('corpus_decision')}"
        )


def retrieval_config_summary():
    return {
        "retrieval_mode": RETRIEVAL_MODE,
        "collection": COLLECTION,
        "qdrant_url": QDRANT_URL,
        "embed_url": EMBED_URL,
        "lexical_db_path": LEXICAL_DB_PATH,
        "lexical_available": lexical_schema_available(),
        "require_lexical": REQUIRE_LEXICAL,
        "default_top_k": DEFAULT_TOP_K,
        "default_pre_k": DEFAULT_PRE_K,
        "max_per_file": MAX_PER_FILE,
        "dense_limit": DENSE_LIMIT,
        "lexical_limit": LEXICAL_LIMIT,
        "rrf_k": RRF_K,
        "rrf_dense_weight": RRF_DENSE_WEIGHT,
        "rrf_lexical_weight": RRF_LEXICAL_WEIGHT,
        "rrf_scale": RRF_SCALE,
        "neighbor_radius": NEIGHBOR_RADIUS,
        "metadata_rerank_clamp": "[-0.045,+0.050]",
    }


def print_retrieval_config():
    cfg = retrieval_config_summary()

    print(f"retrieval_mode={cfg['retrieval_mode']}")
    print(f"Collection: {cfg['collection']}")
    print(f"Qdrant URL: {cfg['qdrant_url']}")
    print(f"Embedding URL: {cfg['embed_url']}")
    print(f"Lexical DB: {cfg['lexical_db_path']}")
    print(f"Lexical available: {cfg['lexical_available']}")
    print(f"Require lexical: {cfg['require_lexical']}")
    print(
        f"defaults: top_k={cfg['default_top_k']}; "
        f"pre_k={cfg['default_pre_k']}; "
        f"max_per_file={cfg['max_per_file']}; "
        f"neighbor_radius={cfg['neighbor_radius']}"
    )
    print(
        f"hybrid: dense_limit={cfg['dense_limit']}; "
        f"lexical_limit={cfg['lexical_limit']}; "
        f"rrf_k={cfg['rrf_k']}; "
        f"rrf_scale={cfg['rrf_scale']}"
    )
    print(f"metadata_rerank_clamp={cfg['metadata_rerank_clamp']}")
