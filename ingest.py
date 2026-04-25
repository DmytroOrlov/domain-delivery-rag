import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

# =============================================================================
# ADAS / Embedded Vision Delivery RAG - Ingestion Pipeline
# =============================================================================
#
# This is a local v1 pipeline for safety-relevant embedded vision / ADAS-style
# corpora. It is not a generic production RAG framework yet. Domain-specific
# ontology and metadata are intentionally kept explicit so they can be reviewed,
# measured, and later moved into configuration if the project becomes multi-domain.
# =============================================================================
#
# Flow:
#   text files -> semantic-ish chunks -> Qwen metadata extraction -> embeddings
#   -> Qdrant collection with rich payload metadata
#
# Design goals:
#   - local-first: llama-server for embeddings and instruct model
#   - trustable ingestion: strict metadata schema, fail-fast on protocol errors
#   - senior-ish chunking: paragraph/heading-aware instead of raw char slicing
#   - scalable enough for v1: Qdrant payload indexes, batch metadata extraction
#   - observable by default: stdout shows chunk stats, batch plan, retries, summary
#
# Important:
#   - This script still does full re-ingest by design.
#   - Incremental ingest / cache / manifest are intentionally not implemented yet.
#   - Chunks marked corpus_decision="drop" are not indexed by default.
# =============================================================================


INPUT_DIR = os.path.expanduser("~/rag_v1/files")
FAILURE_DIR = os.path.expanduser("~/rag_v1/metadata_failures")

EMBED_URL = os.environ.get("RAG_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")
CHAT_URL = os.environ.get("RAG_CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions")
QDRANT_URL = os.environ.get("RAG_QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("RAG_COLLECTION", "rag_v1_chunks")

# CHUNK_SIZE is a target, not a hard maximum.
# Paragraph/heading-aware chunking preserves readable units and overlap as much
# as possible, so some chunks can exceed this target moderately.
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "1400"))
OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "250"))

METADATA_TARGET_BATCH_SIZE = int(os.environ.get("RAG_METADATA_TARGET_BATCH_SIZE", "10"))
# Target number of initial LLM metadata batches per file.
# This is NOT a hard request cap: large files may require more batches to respect
# METADATA_MAX_BATCH_SIZE. Retry splits can also increase the actual request count.
METADATA_TARGET_MAX_INITIAL_BATCHES = int(os.environ.get("RAG_METADATA_TARGET_MAX_INITIAL_BATCHES", "3"))
METADATA_MAX_BATCH_SIZE = int(os.environ.get("RAG_METADATA_MAX_BATCH_SIZE", "30"))

# Default: do NOT index chunks classified as drop.
# For debugging, run:
#   RAG_INDEX_DROPPED_CHUNKS=1 python3 ~/rag_v1/ingest.py
INDEX_DROPPED_CHUNKS = os.environ.get("RAG_INDEX_DROPPED_CHUNKS", "0") == "1"

VERBOSE = os.environ.get("RAG_VERBOSE", "1") != "0"
DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"

TEXT_EXTS = {".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".xml", ".csv"}


# =============================================================================
# Metadata schema
# =============================================================================

CHUNK_ROLE_VALUES = {
    "overview",
    "requirements",
    "architecture",
    "interface_contract",
    "configuration",
    "deployment",
    "validation",
    "test",
    "regulation",
    "operations",
    "data_management",
    "monitoring",
    "incident",
    "simulation",
    "example",
    "noise",
    "unknown",
}

CONTENT_FACET_VALUES = {
    "system_behavior",
    "constraints",
    "interface",
    "configuration",
    "validation",
    "test_scenarios",
    "failure_modes",
    "monitoring",
    "deployment",
    "data",
    "regulatory",
    "examples",
    "implementation",
    "performance",
    "security",
    "privacy",
    "unknown",
}

SYSTEM_LAYER_VALUES = {
    "sensor",
    "perception",
    "decision_logic",
    "hmi",
    "embedded_runtime",
    "data_pipeline",
    "vehicle_integration",
    "ops_monitoring",
    "compliance",
    "simulation",
    "unknown",
}

WORKFLOW_STAGE_VALUES = {
    "discovery",
    "implementation",
    "verification",
    "release",
    "operation",
    "unknown",
}

SAFETY_RELEVANCE_VALUES = {"high", "medium", "low", "unknown"}
DELIVERY_VALUE_VALUES = {"high", "medium", "low"}
CORPUS_DECISION_VALUES = {"primary", "secondary", "drop"}

SAFETY_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
DELIVERY_RANK = {"low": 0, "medium": 1, "high": 2}


STATS = {
    "metadata_retries": 0,
    "metadata_initial_batches": 0,
    "metadata_actual_requests": 0,
    "chunks_total": 0,
    "points_total": 0,
    "dropped_chunks_skipped": 0,
}


class MetadataExtractionError(RuntimeError):
    pass


class MetadataValidationError(RuntimeError):
    pass


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


def preview(text: str, n: int = 160):
    text = (text or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n:
        return text
    return text[:n] + "..."


# =============================================================================
# Text normalization and chunking
# =============================================================================

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\f", "\n\n")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def looks_like_heading(line: str) -> bool:
    """
    Lightweight heading detector for exported PDF/wiki/plain text.

    It intentionally uses generic structure, not domain keywords.
    """
    s = line.strip()
    if not s or len(s) > 120:
        return False

    if re.match(r"^\d+(\.\d+)*\s+\S+", s):
        return True

    if re.match(r"^[A-Z][A-Z0-9 /&()\-:]{6,}$", s) and len(s.split()) <= 12:
        return True

    if re.match(r"^(Appendix|Chapter|Section|Figure|Table)\s+\w+", s, flags=re.I):
        return True

    return False


def hard_split_text(text: str, max_len: int):
    """
    Last-resort splitter for very long paragraphs.

    It tries to cut on newline/sentence/word boundaries before raw length.
    """
    text = text.strip()
    if len(text) <= max_len:
        return [text] if text else []

    parts = []
    start = 0

    while start < len(text):
        end = min(start + max_len, len(text))
        if end < len(text):
            cut_candidates = [
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("; ", start, end),
                text.rfind(", ", start, end),
                text.rfind(" ", start, end),
            ]
            cut = max(cut_candidates)
            if cut > start + max_len * 0.55:
                end = cut + 1

        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end

    return parts


def split_into_units(text: str, max_unit_chars: int):
    """
    Converts raw text into units that are closer to paragraphs/headings.

    This avoids chunks starting mid-word or mid-sentence.
    """
    text = normalize_text(text)
    if not text:
        return []

    units = []
    current_lines = []

    def flush_current():
        nonlocal current_lines
        if not current_lines:
            return
        block = "\n".join(current_lines).strip()
        current_lines = []
        if not block:
            return
        units.extend(hard_split_text(block, max_unit_chars))

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            flush_current()
            continue

        if looks_like_heading(line):
            flush_current()
            units.append(line)
            continue

        current_lines.append(line)

    flush_current()
    return [u.strip() for u in units if u.strip()]


def merge_units_to_chunks(units, chunk_size: int, overlap_chars: int):
    """
    Merge units into chunks.

    chunk_size is a target. Overlap is done by reusing previous units, not by
    slicing raw characters. This preserves local context while keeping chunks
    readable. Some chunks may moderately exceed the target when units/overlap
    would otherwise be broken awkwardly.
    """
    if not units:
        return []

    chunks = []
    current = []
    current_len = 0

    def emit_current():
        if not current:
            return
        chunk = "\n\n".join(current).strip()
        if chunk:
            chunks.append(chunk)

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        unit_len = len(unit)

        if not current:
            current = [unit]
            current_len = unit_len
            continue

        projected = current_len + 2 + unit_len

        if projected <= chunk_size:
            current.append(unit)
            current_len = projected
            continue

        emit_current()

        overlap_units = []
        overlap_len = 0
        for prev in reversed(current):
            prev_len = len(prev)
            if overlap_units and overlap_len + prev_len > overlap_chars:
                break
            overlap_units.insert(0, prev)
            overlap_len += prev_len + 2
            if overlap_len >= overlap_chars:
                break

        current = overlap_units + [unit]
        current_len = sum(len(x) for x in current) + max(0, len(current) - 1) * 2

    emit_current()
    return chunks


def chunk_text(text: str):
    units = split_into_units(text, max_unit_chars=CHUNK_SIZE)
    return merge_units_to_chunks(units, chunk_size=CHUNK_SIZE, overlap_chars=OVERLAP)


def print_chunk_stats(chunks):
    lens = [len(c) for c in chunks]
    if not lens:
        log("  chunk stats: count=0")
        return

    sorted_lens = sorted(lens)
    p50 = sorted_lens[len(sorted_lens) // 2]
    avg = sum(lens) // len(lens)

    log(
        f"  chunk stats: count={len(chunks)} "
        f"min={min(lens)} p50={p50} avg={avg} max={max(lens)} "
        f"(target={CHUNK_SIZE}, not hard cap)"
    )
    log(f"  first chunk preview: {preview(chunks[0])}")
    if len(chunks) > 1:
        log(f"  last chunk preview:  {preview(chunks[-1])}")


# =============================================================================
# Local model API helpers
# =============================================================================

def embed(text: str):
    r = requests.post(EMBED_URL, json={"input": text}, timeout=120)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def extract_json_payload(text: str):
    """
    Robust JSON extractor for Qwen reasoning-mode outputs.
    """
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    if "</think>" in text:
        after_think = text.split("</think>")[-1].strip()
        after_think = re.sub(r"^```json\s*", "", after_think)
        after_think = re.sub(r"^```\s*", "", after_think)
        after_think = re.sub(r"\s*```$", "", after_think)
        try:
            parsed = json.loads(after_think)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    candidates = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue

        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
                    start = None

    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    return None


def _safe_name(file_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", file_name)[:120]


def save_batch_metadata_failure(
        file_name: str,
        start_chunk_index: int,
        reason: str,
        raw_response: str,
        raw_content: str,
        batch_chunks: list,
):
    os.makedirs(FAILURE_DIR, exist_ok=True)

    digest_source = raw_response + json.dumps(batch_chunks, ensure_ascii=False)
    digest = hashlib.sha1(digest_source.encode("utf-8", errors="ignore")).hexdigest()[:12]
    fail_path = os.path.join(
        FAILURE_DIR,
        f"{_safe_name(file_name)}.batch_from_{start_chunk_index}.{digest}.txt",
    )

    with open(fail_path, "w", encoding="utf-8") as f:
        f.write("=== REASON ===\n")
        f.write(reason)
        f.write("\n\n=== FILE ===\n")
        f.write(file_name)
        f.write("\n\n=== START CHUNK INDEX ===\n")
        f.write(str(start_chunk_index))
        f.write("\n\n=== RAW CONTENT ===\n")
        f.write(raw_content or "")
        f.write("\n\n=== FULL RAW RESPONSE ===\n")
        f.write(raw_response or "")
        f.write("\n\n=== BATCH CHUNKS ===\n")
        f.write(json.dumps(batch_chunks, ensure_ascii=False, indent=2))

    return fail_path


# =============================================================================
# Metadata validation
# =============================================================================

def normalize_chunk_index(value):
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise MetadataValidationError(f"chunk_index must be int-like, got {value!r}")


def require_enum(raw: dict, field: str, allowed: set):
    value = raw.get(field)
    if not isinstance(value, str) or value not in allowed:
        raise MetadataValidationError(
            f"{field} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def require_list_enum(raw: dict, field: str, allowed: set):
    values = raw.get(field)
    if not isinstance(values, list):
        raise MetadataValidationError(f"{field} must be a list, got {values!r}")

    cleaned = []
    for value in values:
        if not isinstance(value, str) or value not in allowed:
            raise MetadataValidationError(
                f"{field} items must be one of {sorted(allowed)}, got {value!r}"
            )
        if value not in cleaned:
            cleaned.append(value)

    if not cleaned:
        raise MetadataValidationError(f"{field} must not be empty")

    if "unknown" in cleaned and len(cleaned) > 1:
        raise MetadataValidationError(
            f'{field} cannot mix "unknown" with specific values: {cleaned!r}'
        )

    return cleaned


def require_bool(raw: dict, field: str):
    value = raw.get(field)
    if isinstance(value, bool):
        return value
    raise MetadataValidationError(f"{field} must be boolean, got {value!r}")


def require_confidence(raw: dict):
    value = raw.get("confidence")
    try:
        x = float(value)
    except Exception:
        raise MetadataValidationError(f"confidence must be float 0..1, got {value!r}")

    if not 0.0 <= x <= 1.0:
        raise MetadataValidationError(f"confidence must be 0..1, got {x}")

    return x


def validate_chunk_metadata(raw):
    if not isinstance(raw, dict):
        raise MetadataValidationError("metadata must be a JSON object")

    return {
        "chunk_role": require_enum(raw, "chunk_role", CHUNK_ROLE_VALUES),
        "content_facets": require_list_enum(raw, "content_facets", CONTENT_FACET_VALUES),
        "system_layers": require_list_enum(raw, "system_layers", SYSTEM_LAYER_VALUES),
        "workflow_stages": require_list_enum(raw, "workflow_stages", WORKFLOW_STAGE_VALUES),
        "safety_relevance": require_enum(raw, "safety_relevance", SAFETY_RELEVANCE_VALUES),
        "delivery_value": require_enum(raw, "delivery_value", DELIVERY_VALUE_VALUES),
        "corpus_decision": require_enum(raw, "corpus_decision", CORPUS_DECISION_VALUES),
        "has_behavioral_requirements": require_bool(raw, "has_behavioral_requirements"),
        "has_interface_or_contract": require_bool(raw, "has_interface_or_contract"),
        "has_validation_or_test_evidence": require_bool(raw, "has_validation_or_test_evidence"),
        "has_failure_or_degraded_mode": require_bool(raw, "has_failure_or_degraded_mode"),
        "has_regulatory_or_compliance": require_bool(raw, "has_regulatory_or_compliance"),
        "confidence": require_confidence(raw),
        "reason_short": str(raw.get("reason_short", ""))[:160],
    }


def compact_item_to_metadata(item):
    if not isinstance(item, list):
        raise MetadataValidationError(f"compact item must be list, got {type(item).__name__}")

    if len(item) != 15:
        raise MetadataValidationError(f"compact item must have 15 fields, got {len(item)}: {item!r}")

    chunk_index = normalize_chunk_index(item[0])

    raw_metadata = {
        "chunk_role": item[1],
        "content_facets": item[2],
        "system_layers": item[3],
        "workflow_stages": item[4],
        "safety_relevance": item[5],
        "delivery_value": item[6],
        "corpus_decision": item[7],
        "has_behavioral_requirements": item[8],
        "has_interface_or_contract": item[9],
        "has_validation_or_test_evidence": item[10],
        "has_failure_or_degraded_mode": item[11],
        "has_regulatory_or_compliance": item[12],
        "confidence": item[13],
        "reason_short": item[14],
    }

    return chunk_index, validate_chunk_metadata(raw_metadata)


def item_to_metadata(item):
    if isinstance(item, list):
        return compact_item_to_metadata(item)

    if isinstance(item, dict):
        chunk_index = normalize_chunk_index(item.get("chunk_index"))
        metadata_raw = item.get("metadata")

        if isinstance(metadata_raw, dict) and "metadata" in metadata_raw:
            chunk_index = normalize_chunk_index(metadata_raw.get("chunk_index", chunk_index))
            metadata_raw = metadata_raw.get("metadata")

        return chunk_index, validate_chunk_metadata(metadata_raw)

    raise MetadataValidationError(f"item must be list or dict, got {type(item).__name__}")


# =============================================================================
# Metadata extraction batching and strict retry policy
# =============================================================================

def make_balanced_batches(
        chunks,
        target_batch_size: int,
        target_max_initial_batches: int,
        max_batch_size: int,
):
    n = len(chunks)
    if n == 0:
        return []

    target_batch_size = max(1, int(target_batch_size))
    target_max_initial_batches = max(1, int(target_max_initial_batches))
    max_batch_size = max(1, int(max_batch_size))

    desired_by_target = (n + target_batch_size - 1) // target_batch_size
    desired_by_cap = (n + max_batch_size - 1) // max_batch_size

    num_batches = max(
        desired_by_cap,
        min(target_max_initial_batches, desired_by_target),
    )

    base = n // num_batches
    remainder = n % num_batches

    batches = []
    start = 0

    for batch_idx in range(num_batches):
        size = base + (1 if batch_idx < remainder else 0)
        end = start + size
        batch = [{"chunk_index": i, "text": chunks[i]} for i in range(start, end)]
        batches.append(batch)
        start = end

    return batches


def extract_chunk_metadata_batch(batch_chunks: list, file_name: str):
    STATS["metadata_actual_requests"] += 1

    allowed_chunk_role = ", ".join(sorted(CHUNK_ROLE_VALUES))
    allowed_content_facets = ", ".join(sorted(CONTENT_FACET_VALUES))
    allowed_system_layers = ", ".join(sorted(SYSTEM_LAYER_VALUES))
    allowed_workflow_stages = ", ".join(sorted(WORKFLOW_STAGE_VALUES))
    allowed_safety_relevance = ", ".join(sorted(SAFETY_RELEVANCE_VALUES))
    allowed_delivery_value = ", ".join(sorted(DELIVERY_VALUE_VALUES))
    allowed_corpus_decision = ", ".join(sorted(CORPUS_DECISION_VALUES))

    start_chunk_index = batch_chunks[0]["chunk_index"] if batch_chunks else -1
    expected_indices = [x["chunk_index"] for x in batch_chunks]
    expected_set = set(expected_indices)

    system_prompt = """You are a strict metadata extractor for a technical RAG corpus.

Return JSON only.
Do not explain.
Do not include markdown.
Use only the provided chunk texts.
Use chunk_index only as an identifier, never as evidence.
Do not infer missing context from surrounding chunks.
Use exact enum values only.
Classify each chunk independently.
Do not create metadata for chunks that are not explicitly listed.
Return exactly one compact row for every expected chunk_index.
If unsure, use "unknown" where allowed.
"""

    user_prompt = f"""Classify each chunk independently for a Domain Delivery RAG focused on engineering delivery for safety-relevant technical systems.

Expected chunk_index values:
{json.dumps(expected_indices)}

Allowed values:
chunk_role: {allowed_chunk_role}
content_facets: {allowed_content_facets}
system_layers: {allowed_system_layers}
workflow_stages: {allowed_workflow_stages}
safety_relevance: {allowed_safety_relevance}
delivery_value: {allowed_delivery_value}
corpus_decision: {allowed_corpus_decision}

Definitions:
- chunk_role: the primary role of this chunk.
- content_facets: what kind of useful content appears here. Multiple values are allowed.
- system_layers: which system layer this chunk concerns. Multiple values are allowed.
- workflow_stages: where this chunk helps in delivery lifecycle. Multiple values are allowed.
- safety_relevance: high only if this chunk directly affects safety behavior, safety assurance, compliance, validation, failure/degraded modes, or safety-critical deployment.
- delivery_value: high only if this chunk helps implementation, verification, release, debugging, integration, or decision-making.
- corpus_decision: primary if this chunk should be indexed for v1 delivery RAG; secondary if useful but not core; drop if mostly noise.

Return compact JSON only.

Output schema:
{{
  "items": [
    [
      chunk_index,
      chunk_role,
      content_facets,
      system_layers,
      workflow_stages,
      safety_relevance,
      delivery_value,
      corpus_decision,
      has_behavioral_requirements,
      has_interface_or_contract,
      has_validation_or_test_evidence,
      has_failure_or_degraded_mode,
      has_regulatory_or_compliance,
      confidence,
      reason_short
    ]
  ]
}}

Example item:
[0, "test", ["validation"], ["perception"], ["verification"], "high", "high", "primary", false, false, true, false, false, 0.85, "Short reason."]

Chunks:
{json.dumps(batch_chunks, ensure_ascii=False, indent=2)}
"""

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 16384,
        "response_format": {"type": "json_object"},
    }

    debug_print("METADATA PAYLOAD", payload)

    r = requests.post(CHAT_URL, json=payload, timeout=1800)
    r.raise_for_status()

    data = r.json()
    raw_response = json.dumps(data, ensure_ascii=False, indent=2)
    msg = data["choices"][0]["message"]
    raw_content = msg.get("content", "")

    debug_print("METADATA RAW CONTENT", raw_content)

    if not raw_content.strip():
        fail_path = save_batch_metadata_failure(
            file_name=file_name,
            start_chunk_index=start_chunk_index,
            reason="empty message.content",
            raw_response=raw_response,
            raw_content=raw_content,
            batch_chunks=batch_chunks,
        )
        raise MetadataExtractionError(
            f"Batch metadata extraction returned empty content → {fail_path}"
        )

    parsed = extract_json_payload(raw_content)

    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        fail_path = save_batch_metadata_failure(
            file_name=file_name,
            start_chunk_index=start_chunk_index,
            reason='could not parse JSON object with "items" list',
            raw_response=raw_response,
            raw_content=raw_content,
            batch_chunks=batch_chunks,
        )
        raise MetadataExtractionError(f"Batch metadata JSON parse failed → {fail_path}")

    seen = set()
    result = {}
    extra_indices = []

    for item in parsed["items"]:
        try:
            chunk_index, metadata = item_to_metadata(item)
        except MetadataValidationError as e:
            fail_path = save_batch_metadata_failure(
                file_name=file_name,
                start_chunk_index=start_chunk_index,
                reason=f"metadata item validation failed: {e}",
                raw_response=raw_response,
                raw_content=raw_content,
                batch_chunks=batch_chunks,
            )
            raise MetadataExtractionError(f"Batch metadata validation failed → {fail_path}") from e

        if chunk_index not in expected_set:
            extra_indices.append(chunk_index)
            continue

        if chunk_index in seen:
            fail_path = save_batch_metadata_failure(
                file_name=file_name,
                start_chunk_index=start_chunk_index,
                reason=f"duplicate chunk_index: {chunk_index!r}",
                raw_response=raw_response,
                raw_content=raw_content,
                batch_chunks=batch_chunks,
            )
            raise MetadataExtractionError(f"Batch metadata duplicate index → {fail_path}")

        result[chunk_index] = metadata
        seen.add(chunk_index)

    missing = sorted(expected_set - seen)
    if missing:
        fail_path = save_batch_metadata_failure(
            file_name=file_name,
            start_chunk_index=start_chunk_index,
            reason=f"missing chunk indices: expected {expected_indices!r}, got {sorted(seen)!r}, missing {missing!r}",
            raw_response=raw_response,
            raw_content=raw_content,
            batch_chunks=batch_chunks,
        )
        raise MetadataExtractionError(f"Batch metadata missing items → {fail_path}")

    if extra_indices:
        fail_path = save_batch_metadata_failure(
            file_name=file_name,
            start_chunk_index=start_chunk_index,
            reason=f"extra chunk indices returned: {extra_indices!r}; expected only {expected_indices!r}",
            raw_response=raw_response,
            raw_content=raw_content,
            batch_chunks=batch_chunks,
        )
        raise MetadataExtractionError(f"Batch metadata returned extra chunk indices → {fail_path}")

    return result


def extract_chunk_metadata_batch_with_retry(batch_chunks: list, file_name: str):
    try:
        return extract_chunk_metadata_batch(batch_chunks, file_name=file_name)
    except MetadataExtractionError:
        if len(batch_chunks) <= 1:
            raise

        STATS["metadata_retries"] += 1

        mid = len(batch_chunks) // 2
        left = batch_chunks[:mid]
        right = batch_chunks[mid:]

        left_start = left[0]["chunk_index"]
        left_end = left[-1]["chunk_index"]
        right_start = right[0]["chunk_index"]
        right_end = right[-1]["chunk_index"]

        log(
            f"  ↳ metadata protocol violation on batch size={len(batch_chunks)}; "
            f"retry split {left_start + 1}-{left_end + 1} and {right_start + 1}-{right_end + 1}"
        )

        result = {}
        result.update(extract_chunk_metadata_batch_with_retry(left, file_name=file_name))
        result.update(extract_chunk_metadata_batch_with_retry(right, file_name=file_name))
        return result


# =============================================================================
# Document-level metadata aggregation
# =============================================================================

def score_for_metadata(meta: dict) -> float:
    score = 1.0
    score += meta.get("confidence", 0.0)

    if meta["delivery_value"] == "high":
        score += 1.0
    elif meta["delivery_value"] == "medium":
        score += 0.5

    if meta["safety_relevance"] == "high":
        score += 1.0
    elif meta["safety_relevance"] == "medium":
        score += 0.5

    if meta["corpus_decision"] == "primary":
        score += 1.0
    elif meta["corpus_decision"] == "drop":
        score -= 1.0

    if meta["chunk_role"] in {"noise", "unknown"}:
        score -= 0.5

    return max(0.0, score)


def weighted_top_values(chunk_metas, field: str, limit: int = 6, ignore_values=None):
    ignore_values = ignore_values or set()
    weights = defaultdict(float)

    for meta in chunk_metas:
        value = meta[field]
        score = score_for_metadata(meta)

        if isinstance(value, list):
            for item in value:
                if item not in ignore_values:
                    weights[item] += score
        else:
            if value not in ignore_values:
                weights[value] += score

    if not weights:
        return ["unknown"]

    ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    return [v for v, _ in ranked[:limit]]


def max_enum(chunk_metas, field: str, rank_map: dict, default: str):
    best = default
    best_rank = rank_map.get(default, 0)

    for meta in chunk_metas:
        value = meta[field]
        r = rank_map.get(value, 0)
        if r > best_rank:
            best = value
            best_rank = r

    return best


def aggregate_document_metadata(chunk_metas):
    if not chunk_metas:
        return {
            "document_primary_role": "unknown",
            "document_roles": ["unknown"],
            "document_content_facets": ["unknown"],
            "document_system_layers": ["unknown"],
            "document_workflow_stages": ["unknown"],
            "document_safety_relevance": "unknown",
            "document_delivery_value": "low",
            "document_corpus_decision": "secondary",
            "document_confidence": 0.0,
            "document_signal_chunks": [],
        }

    roles = weighted_top_values(chunk_metas, "chunk_role", limit=6, ignore_values={"unknown", "noise"})
    facets = weighted_top_values(chunk_metas, "content_facets", limit=10, ignore_values={"unknown"})
    layers = weighted_top_values(chunk_metas, "system_layers", limit=10, ignore_values={"unknown"})
    stages = weighted_top_values(chunk_metas, "workflow_stages", limit=8, ignore_values={"unknown"})

    document_primary_role = roles[0] if roles else "unknown"
    document_safety_relevance = max_enum(chunk_metas, "safety_relevance", SAFETY_RANK, "unknown")
    document_delivery_value = max_enum(chunk_metas, "delivery_value", DELIVERY_RANK, "low")

    decisions = [m["corpus_decision"] for m in chunk_metas]
    if "primary" in decisions:
        document_corpus_decision = "primary"
    elif all(d == "drop" for d in decisions):
        document_corpus_decision = "drop"
    else:
        document_corpus_decision = "secondary"

    avg_confidence = sum(m.get("confidence", 0.0) for m in chunk_metas) / len(chunk_metas)

    signal_chunks = []
    for i, meta in enumerate(chunk_metas):
        if (
                meta["corpus_decision"] == "primary"
                or meta["delivery_value"] == "high"
                or meta["safety_relevance"] == "high"
                or meta["has_interface_or_contract"]
                or meta["has_validation_or_test_evidence"]
                or meta["has_failure_or_degraded_mode"]
        ):
            signal_chunks.append(i)

    return {
        "document_primary_role": document_primary_role,
        "document_roles": roles or ["unknown"],
        "document_content_facets": facets or ["unknown"],
        "document_system_layers": layers or ["unknown"],
        "document_workflow_stages": stages or ["unknown"],
        "document_safety_relevance": document_safety_relevance,
        "document_delivery_value": document_delivery_value,
        "document_corpus_decision": document_corpus_decision,
        "document_confidence": round(avg_confidence, 3),
        "document_signal_chunks": signal_chunks[:20],
    }


# =============================================================================
# Qdrant setup
# =============================================================================

def create_payload_indexes(client: QdrantClient):
    index_specs = [
        ("file_path", models.PayloadSchemaType.KEYWORD),
        ("file_name", models.PayloadSchemaType.KEYWORD),
        ("chunk_index", models.PayloadSchemaType.INTEGER),
        ("chunk_role", models.PayloadSchemaType.KEYWORD),
        ("content_facets", models.PayloadSchemaType.KEYWORD),
        ("system_layers", models.PayloadSchemaType.KEYWORD),
        ("workflow_stages", models.PayloadSchemaType.KEYWORD),
        ("safety_relevance", models.PayloadSchemaType.KEYWORD),
        ("delivery_value", models.PayloadSchemaType.KEYWORD),
        ("corpus_decision", models.PayloadSchemaType.KEYWORD),
        ("document_primary_role", models.PayloadSchemaType.KEYWORD),
        ("document_roles", models.PayloadSchemaType.KEYWORD),
        ("document_content_facets", models.PayloadSchemaType.KEYWORD),
        ("document_system_layers", models.PayloadSchemaType.KEYWORD),
        ("document_workflow_stages", models.PayloadSchemaType.KEYWORD),
        ("document_safety_relevance", models.PayloadSchemaType.KEYWORD),
        ("document_delivery_value", models.PayloadSchemaType.KEYWORD),
        ("document_corpus_decision", models.PayloadSchemaType.KEYWORD),
    ]

    log(f"Creating payload indexes: {', '.join(field for field, _ in index_specs)}")
    for field, schema in index_specs:
        client.create_payload_index(COLLECTION, field, schema)


def print_document_metadata_summary(document_meta, chunk_metas):
    role_counts = defaultdict(int)
    facet_counts = defaultdict(int)
    layer_counts = defaultdict(int)
    decision_counts = defaultdict(int)

    for meta in chunk_metas:
        role_counts[meta["chunk_role"]] += 1
        decision_counts[meta["corpus_decision"]] += 1
        for f in meta["content_facets"]:
            facet_counts[f] += 1
        for layer in meta["system_layers"]:
            layer_counts[layer] += 1

    top_roles = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_facets = sorted(facet_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    top_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:8]

    log(
        f"  document: primary_role={document_meta['document_primary_role']} "
        f"safety={document_meta['document_safety_relevance']} "
        f"delivery={document_meta['document_delivery_value']} "
        f"decision={document_meta['document_corpus_decision']} "
        f"confidence={document_meta['document_confidence']}"
    )
    log(f"  corpus decisions: {dict(decision_counts)}")
    log(f"  top chunk roles: {top_roles}")
    log(f"  top facets: {top_facets}")
    log(f"  top layers: {top_layers}")


# =============================================================================
# Main
# =============================================================================

def main():
    files = []
    for root, _, names in os.walk(INPUT_DIR):
        for name in names:
            path = Path(root) / name
            if path.suffix.lower() in TEXT_EXTS:
                files.append(path)

    files.sort()

    if not files:
        raise SystemExit(f"No supported text files found in {INPUT_DIR}")

    print("=" * 100)
    print("RAG INGEST START")
    print("=" * 100)
    print(f"Input dir: {INPUT_DIR}")
    print(f"Collection: {COLLECTION}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Embedding URL: {EMBED_URL}")
    print(f"Chat URL: {CHAT_URL}")
    print(f"Files found: {len(files)}")
    print(f"Chunking: paragraph/heading-aware; chunk_size_target={CHUNK_SIZE}; overlap={OVERLAP}")
    print(
        f"Metadata batching: target_batch_size={METADATA_TARGET_BATCH_SIZE}; "
        f"target_max_initial_batches={METADATA_TARGET_MAX_INITIAL_BATCHES}; "
        f"hard_max_batch_size={METADATA_MAX_BATCH_SIZE}; "
        "actual requests may exceed target for large files or retry splits"
    )
    print(f"Index dropped chunks: {INDEX_DROPPED_CHUNKS}")
    print(f"Verbose: {VERBOSE}; Debug raw payloads: {DEBUG}")
    print()

    client = QdrantClient(url=QDRANT_URL)
    dim = len(embed("probe"))
    print(f"Embedding dimension probe: {dim}")

    if client.collection_exists(COLLECTION):
        print(f"Deleting existing collection: {COLLECTION}")
        client.delete_collection(COLLECTION)

    print(f"Creating collection: {COLLECTION}")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    create_payload_indexes(client)

    point_id = 1

    for file_num, path in enumerate(files, start=1):
        print()
        print("=" * 100)
        print(f"FILE {file_num}/{len(files)}: {path.name}")
        print("=" * 100)

        text = path.read_text(encoding="utf-8", errors="ignore")
        print(f"  raw chars: {len(text)}")

        chunks = chunk_text(text)
        print_chunk_stats(chunks)

        STATS["chunks_total"] += len(chunks)

        chunk_metas = [None] * len(chunks)

        batches = make_balanced_batches(
            chunks,
            target_batch_size=METADATA_TARGET_BATCH_SIZE,
            target_max_initial_batches=METADATA_TARGET_MAX_INITIAL_BATCHES,
            max_batch_size=METADATA_MAX_BATCH_SIZE,
        )

        STATS["metadata_initial_batches"] += len(batches)
        print(f"  metadata batches planned: {[len(b) for b in batches]}")

        for batch_num, batch in enumerate(batches, start=1):
            start_index = batch[0]["chunk_index"]
            end_index = batch[-1]["chunk_index"]

            print(
                f"  metadata batch {batch_num}/{len(batches)} "
                f"chunks {start_index + 1}-{end_index + 1}/{len(chunks)} "
                f"size={len(batch)}",
                flush=True,
            )

            try:
                batch_result = extract_chunk_metadata_batch_with_retry(batch, file_name=path.name)
            except Exception:
                print(f"❌ FAILED: {path.name} batch={batch_num} start_chunk={start_index}")
                raise

            for chunk_index, meta in batch_result.items():
                chunk_metas[chunk_index] = meta

        if any(m is None for m in chunk_metas):
            missing = [i for i, m in enumerate(chunk_metas) if m is None]
            raise RuntimeError(f"Missing metadata for chunks: {missing}")

        document_meta = aggregate_document_metadata(chunk_metas)
        print_document_metadata_summary(document_meta, chunk_metas)

        points = []
        skipped_drop = 0

        for i, chunk in enumerate(chunks):
            chunk_meta = chunk_metas[i]

            if chunk_meta["corpus_decision"] == "drop" and not INDEX_DROPPED_CHUNKS:
                skipped_drop += 1
                continue

            vec = embed(chunk)

            payload = {
                "file_path": str(path),
                "file_name": path.name,
                "chunk_index": i,
                "content": chunk,
                **chunk_meta,
                **document_meta,
            }

            points.append(models.PointStruct(id=point_id, vector=vec, payload=payload))
            point_id += 1

        if points:
            client.upsert(collection_name=COLLECTION, points=points)

        STATS["points_total"] += len(points)
        STATS["dropped_chunks_skipped"] += skipped_drop

        print(f"  skipped drop chunks: {skipped_drop}")
        print(f"  upserted points: {len(points)}")

    print()
    print("=" * 100)
    print("RAG INGEST DONE")
    print("=" * 100)
    print(f"Files processed: {len(files)}")
    print(f"Chunks total: {STATS['chunks_total']}")
    print(f"Initial metadata batches: {STATS['metadata_initial_batches']}")
    print(f"Actual metadata requests including retries: {STATS['metadata_actual_requests']}")
    print(f"Metadata retry splits: {STATS['metadata_retries']}")
    print(f"Drop chunks skipped: {STATS['dropped_chunks_skipped']}")
    print(f"Points inserted: {STATS['points_total']}")
    print(f"Collection: {COLLECTION}")
    print("=" * 100)


if __name__ == "__main__":
    main()