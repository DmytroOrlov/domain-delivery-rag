import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

from domain_config import load_domain_config

# =============================================================================
# Domain Delivery RAG - Ingestion Pipeline
# =============================================================================
#
# Local v1 ingestion pipeline for a domain-configured RAG collection.
# Domain ontology, metadata schema values, logical metadata fields, document
# aggregation policy, and extraction prompt policy are loaded from the active
# domain config.
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


DOMAIN = load_domain_config()

INPUT_DIR = os.path.expanduser(os.environ.get("RAG_INPUT_DIR", DOMAIN.input_dir))
FAILURE_DIR = os.path.expanduser(os.environ.get("RAG_FAILURE_DIR", DOMAIN.failure_dir))

EMBED_URL = os.environ.get("RAG_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")
CHAT_URL = os.environ.get("RAG_CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions")
QDRANT_URL = os.environ.get("RAG_QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("RAG_COLLECTION", DOMAIN.collection)

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
METADATA_MAX_BATCH_SIZE = int(os.environ.get("RAG_METADATA_MAX_BATCH_SIZE", "25"))

# Default: do NOT index chunks classified as drop.
# For debugging, run:
#   RAG_INDEX_DROPPED_CHUNKS=1 python3 ingest.py
INDEX_DROPPED_CHUNKS = os.environ.get("RAG_INDEX_DROPPED_CHUNKS", "0") == "1"

VERBOSE = os.environ.get("RAG_VERBOSE", "1") != "0"
DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"

TEXT_EXTS = {".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".xml", ".csv"}


# =============================================================================
# Metadata schema
# =============================================================================

DEFAULT_CHUNK_ROLE_VALUES = {
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

DEFAULT_CONTENT_FACET_VALUES = {
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

DEFAULT_SYSTEM_LAYER_VALUES = {
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

DEFAULT_WORKFLOW_STAGE_VALUES = {
    "discovery",
    "implementation",
    "verification",
    "release",
    "operation",
    "unknown",
}

DEFAULT_SAFETY_RELEVANCE_VALUES = {"high", "medium", "low", "unknown"}
DEFAULT_DELIVERY_VALUE_VALUES = {"high", "medium", "low"}
DEFAULT_CORPUS_DECISION_VALUES = {"primary", "secondary", "drop"}
DEFAULT_BOOLEAN_FLAG_FIELDS = (
    "has_behavioral_requirements",
    "has_interface_or_contract",
    "has_validation_or_test_evidence",
    "has_failure_or_degraded_mode",
    "has_regulatory_or_compliance",
)


def schema_values(field: str, default_values) -> set[str]:
    values = DOMAIN.metadata_schema.get(field, default_values)
    if not isinstance(values, (list, tuple, set)) or not values:
        raise ValueError(f"metadata_schema.{field} must be a non-empty list")
    result = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"metadata_schema.{field} values must be non-empty strings")
        result.add(value)
    return result


def schema_list(field: str, default_values) -> list[str]:
    values = DOMAIN.metadata_schema.get(field, default_values)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"metadata_schema.{field} must be a non-empty list")
    result = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"metadata_schema.{field} values must be non-empty strings")
        if value not in result:
            result.append(value)
    return result


DEFAULT_METADATA_FIELD_MAP = {
    "role": "chunk_role",
    "facets": "content_facets",
    "layers": "system_layers",
    "stages": "workflow_stages",
    "criticality": "safety_relevance",
    "delivery_value": "delivery_value",
    "decision": "corpus_decision",
    "document_primary_role": "document_primary_role",
    "document_roles": "document_roles",
    "document_facets": "document_content_facets",
    "document_layers": "document_system_layers",
    "document_stages": "document_workflow_stages",
    "document_criticality": "document_safety_relevance",
    "document_delivery_value": "document_delivery_value",
    "document_decision": "document_corpus_decision",
    "document_confidence": "document_confidence",
    "document_signal_chunks": "document_signal_chunks",
}


def metadata_field(logical_name: str, default: str | None = None) -> str:
    field_map = dict(DEFAULT_METADATA_FIELD_MAP)
    field_map.update(getattr(DOMAIN, "metadata_field_map", {}) or {})
    fallback = default if default is not None else logical_name
    value = field_map.get(logical_name, fallback)
    return str(value or fallback)

LOGICAL_METADATA_ORDER = (
    "role",
    "facets",
    "layers",
    "stages",
    "criticality",
    "delivery_value",
    "decision",
)

DEFAULT_METADATA_FIELDS = {
    "role": {
        "payload": "chunk_role",
        "prompt_label": "chunk_role",
        "schema_key": "chunk_role",
        "kind": "enum",
        "default": "unknown",
    },
    "facets": {
        "payload": "content_facets",
        "prompt_label": "content_facets",
        "schema_key": "content_facets",
        "kind": "list_enum",
        "default": ["unknown"],
    },
    "layers": {
        "payload": "system_layers",
        "prompt_label": "system_layers",
        "schema_key": "system_layers",
        "kind": "list_enum",
        "default": ["unknown"],
    },
    "stages": {
        "payload": "workflow_stages",
        "prompt_label": "workflow_stages",
        "schema_key": "workflow_stages",
        "kind": "list_enum",
        "default": ["unknown"],
    },
    "criticality": {
        "payload": "safety_relevance",
        "prompt_label": "safety_relevance",
        "schema_key": "safety_relevance",
        "kind": "enum",
        "default": "unknown",
    },
    "delivery_value": {
        "payload": "delivery_value",
        "prompt_label": "delivery_value",
        "schema_key": "delivery_value",
        "kind": "enum",
        "default": "low",
    },
    "decision": {
        "payload": "corpus_decision",
        "prompt_label": "corpus_decision",
        "schema_key": "corpus_decision",
        "kind": "enum",
        "default": "secondary",
    },
}


def metadata_fields_config() -> dict:
    configured = getattr(DOMAIN, "metadata_fields", {}) or {}
    merged = {}
    for logical_name, default_cfg in DEFAULT_METADATA_FIELDS.items():
        configured_cfg = configured.get(logical_name) if isinstance(configured.get(logical_name), dict) else {}
        cfg = dict(default_cfg)
        cfg.update(configured_cfg)

        # Backward compatible fallback: metadata_field_map can still override payload names.
        cfg["payload"] = str(cfg.get("payload") or metadata_field(logical_name, default_cfg["payload"]))
        cfg["prompt_label"] = str(cfg.get("prompt_label") or cfg["payload"] or logical_name)

        # Important: if a domain pack supplies values_ref but omits schema_key,
        # values_ref must win over the ADAS/default schema_key. Otherwise a new
        # domain can silently validate against the wrong enum.
        if configured_cfg.get("schema_key"):
            schema_key = configured_cfg["schema_key"]
        elif configured_cfg.get("values_ref"):
            schema_key = str(configured_cfg["values_ref"]).split(".")[-1]
        else:
            schema_key = cfg.get("schema_key") or cfg.get("values_ref", "").split(".")[-1] or cfg["prompt_label"]
        cfg["schema_key"] = str(schema_key)

        cfg["kind"] = str(cfg.get("kind") or default_cfg["kind"])
        merged[logical_name] = cfg
    return merged


def metadata_field_config(logical_name: str) -> dict:
    return metadata_fields_config()[logical_name]


def metadata_prompt_label(logical_name: str) -> str:
    return str(metadata_field_config(logical_name).get("prompt_label") or metadata_field(logical_name))


def metadata_payload_field(logical_name: str) -> str:
    return str(metadata_field_config(logical_name).get("payload") or metadata_field(logical_name))


def metadata_schema_key(logical_name: str) -> str:
    return str(metadata_field_config(logical_name).get("schema_key") or metadata_prompt_label(logical_name))


def metadata_kind(logical_name: str) -> str:
    return str(metadata_field_config(logical_name).get("kind") or "enum")


def metadata_allowed_values(logical_name: str) -> set[str]:
    schema_key = metadata_schema_key(logical_name)
    default_by_logical = {
        "role": DEFAULT_CHUNK_ROLE_VALUES,
        "facets": DEFAULT_CONTENT_FACET_VALUES,
        "layers": DEFAULT_SYSTEM_LAYER_VALUES,
        "stages": DEFAULT_WORKFLOW_STAGE_VALUES,
        "criticality": DEFAULT_SAFETY_RELEVANCE_VALUES,
        "delivery_value": DEFAULT_DELIVERY_VALUE_VALUES,
        "decision": DEFAULT_CORPUS_DECISION_VALUES,
    }
    return schema_values(schema_key, default_by_logical[logical_name])


def metadata_raw_value(raw: dict, logical_name: str):
    cfg = metadata_field_config(logical_name)
    candidate_keys = [
        cfg.get("prompt_label"),
        cfg.get("payload"),
        metadata_field(logical_name),
        logical_name,
    ]
    # Backward compatible aliases for old ADAS payloads.
    candidate_keys.extend([
        DEFAULT_METADATA_FIELDS[logical_name]["prompt_label"],
        DEFAULT_METADATA_FIELDS[logical_name]["payload"],
    ])
    for key in candidate_keys:
        if key and key in raw:
            return raw.get(key)
    return None


ROLE_FIELD = metadata_field("role", "chunk_role")
FACETS_FIELD = metadata_field("facets", "content_facets")
LAYERS_FIELD = metadata_field("layers", "system_layers")
STAGES_FIELD = metadata_field("stages", "workflow_stages")
CRITICALITY_FIELD = metadata_field("criticality", "safety_relevance")
DELIVERY_FIELD = metadata_field("delivery_value", "delivery_value")
DECISION_FIELD = metadata_field("decision", "corpus_decision")

DOC_PRIMARY_ROLE_FIELD = metadata_field("document_primary_role", "document_primary_role")
DOC_ROLES_FIELD = metadata_field("document_roles", "document_roles")
DOC_FACETS_FIELD = metadata_field("document_facets", "document_content_facets")
DOC_LAYERS_FIELD = metadata_field("document_layers", "document_system_layers")
DOC_STAGES_FIELD = metadata_field("document_stages", "document_workflow_stages")
DOC_CRITICALITY_FIELD = metadata_field("document_criticality", "document_safety_relevance")
DOC_DELIVERY_FIELD = metadata_field("document_delivery_value", "document_delivery_value")
DOC_DECISION_FIELD = metadata_field("document_decision", "document_corpus_decision")
DOC_CONFIDENCE_FIELD = metadata_field("document_confidence", "document_confidence")
DOC_SIGNAL_CHUNKS_FIELD = metadata_field("document_signal_chunks", "document_signal_chunks")


def document_aggregation_config() -> dict:
    return dict(getattr(DOMAIN, "document_aggregation", {}) or {})


def aggregation_rank_map(name: str, default: dict[str, int]) -> dict[str, int]:
    raw = document_aggregation_config().get(name, default)
    if not isinstance(raw, dict):
        return dict(default)
    out = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value)
        except Exception:
            continue
    return out or dict(default)


def aggregation_list(path: tuple[str, ...], default: list[str]) -> list[str]:
    cfg = document_aggregation_config()
    value = cfg
    for part in path:
        if not isinstance(value, dict):
            return list(default)
        value = value.get(part)
    if not isinstance(value, list):
        return list(default)
    return [str(x) for x in value]


def unique_index_fields(fields: list[tuple[str, object]]) -> list[tuple[str, object]]:
    seen = set()
    out = []
    for field, schema in fields:
        if field and field not in seen:
            seen.add(field)
            out.append((field, schema))
    return out


CHUNK_ROLE_VALUES = metadata_allowed_values("role")
CONTENT_FACET_VALUES = metadata_allowed_values("facets")
SYSTEM_LAYER_VALUES = metadata_allowed_values("layers")
WORKFLOW_STAGE_VALUES = metadata_allowed_values("stages")
SAFETY_RELEVANCE_VALUES = metadata_allowed_values("criticality")
DELIVERY_VALUE_VALUES = metadata_allowed_values("delivery_value")
CORPUS_DECISION_VALUES = metadata_allowed_values("decision")
BOOLEAN_FLAG_FIELDS = tuple(schema_list("boolean_flags", DEFAULT_BOOLEAN_FLAG_FIELDS))

METADATA_ALLOWED_VALUES = {
    "role": CHUNK_ROLE_VALUES,
    "facets": CONTENT_FACET_VALUES,
    "layers": SYSTEM_LAYER_VALUES,
    "stages": WORKFLOW_STAGE_VALUES,
    "criticality": SAFETY_RELEVANCE_VALUES,
    "delivery_value": DELIVERY_VALUE_VALUES,
    "decision": CORPUS_DECISION_VALUES,
}

SAFETY_RANK = aggregation_rank_map("criticality_rank", {"unknown": 0, "low": 1, "medium": 2, "high": 3})
DELIVERY_RANK = aggregation_rank_map("delivery_rank", {"low": 0, "medium": 1, "high": 2})


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

    metadata = {}
    for logical_name in LOGICAL_METADATA_ORDER:
        payload_field = metadata_payload_field(logical_name)
        prompt_label = metadata_prompt_label(logical_name)
        allowed = METADATA_ALLOWED_VALUES[logical_name]
        value = metadata_raw_value(raw, logical_name)
        temp_raw = {prompt_label: value}
        if metadata_kind(logical_name) == "list_enum":
            metadata[payload_field] = require_list_enum(temp_raw, prompt_label, allowed)
        else:
            metadata[payload_field] = require_enum(temp_raw, prompt_label, allowed)

    for flag in BOOLEAN_FLAG_FIELDS:
        metadata[flag] = require_bool(raw, flag)

    metadata["confidence"] = require_confidence(raw)
    metadata["reason_short"] = str(raw.get("reason_short", ""))[:160]
    return metadata


def compact_item_to_metadata(item):
    if not isinstance(item, list):
        raise MetadataValidationError(f"compact item must be list, got {type(item).__name__}")

    logical_count = len(LOGICAL_METADATA_ORDER)
    expected_len = 1 + logical_count + len(BOOLEAN_FLAG_FIELDS) + 2
    if len(item) != expected_len:
        raise MetadataValidationError(
            f"compact item must have {expected_len} fields, got {len(item)}: {item!r}"
        )

    chunk_index = normalize_chunk_index(item[0])
    raw_metadata = {}
    for offset, logical_name in enumerate(LOGICAL_METADATA_ORDER, start=1):
        raw_metadata[metadata_prompt_label(logical_name)] = item[offset]

    flag_start = 1 + logical_count
    flag_end = flag_start + len(BOOLEAN_FLAG_FIELDS)
    for offset, flag in enumerate(BOOLEAN_FLAG_FIELDS):
        raw_metadata[flag] = item[flag_start + offset]

    raw_metadata["confidence"] = item[flag_end]
    raw_metadata["reason_short"] = item[flag_end + 1]

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


DEFAULT_METADATA_EXTRACTION = {
    "system_rules": [
        "You are a strict metadata extractor for a technical RAG corpus.",
        "Return JSON only.",
        "Do not explain.",
        "Do not include markdown.",
        "Use only the provided chunk texts.",
        "Use chunk_index only as an identifier, never as evidence.",
        "Do not infer missing context from surrounding chunks.",
        "Use exact enum values only.",
        "Classify each chunk independently.",
        "Do not create metadata for chunks that are not explicitly listed.",
        "Return exactly one compact row for every expected chunk_index.",
        "If unsure, use \"unknown\" where allowed.",
    ],
    "objective": (
        "Classify each chunk independently for a Domain Delivery RAG focused on "
        "engineering delivery for safety-relevant technical systems."
    ),
    "field_definitions": {
        "chunk_role": "the primary role of this chunk.",
        "content_facets": "what kind of useful content appears here. Multiple values are allowed.",
        "system_layers": "which system layer this chunk concerns. Multiple values are allowed.",
        "workflow_stages": "where this chunk helps in delivery lifecycle. Multiple values are allowed.",
        "safety_relevance": (
            "high only if this chunk directly affects safety behavior, safety assurance, "
            "compliance, validation, failure/degraded modes, or safety-critical deployment."
        ),
        "delivery_value": (
            "high only if this chunk helps implementation, verification, release, debugging, "
            "integration, or decision-making."
        ),
        "corpus_decision": (
            "primary if this chunk should be indexed for v1 delivery RAG; secondary if useful "
            "but not core; drop if mostly noise."
        ),
        "boolean_flags": "true only when the chunk explicitly contains that type of evidence.",
    },
    "return_instruction": "Return compact JSON only.",
}


def metadata_extraction_config() -> dict:
    config = dict(DEFAULT_METADATA_EXTRACTION)
    domain_config = getattr(DOMAIN, "metadata_extraction", {}) or {}

    if isinstance(domain_config.get("system_rules"), list) and domain_config["system_rules"]:
        config["system_rules"] = [str(x) for x in domain_config["system_rules"] if str(x).strip()]

    if isinstance(domain_config.get("objective"), str) and domain_config["objective"].strip():
        config["objective"] = domain_config["objective"].strip()

    field_definitions = dict(DEFAULT_METADATA_EXTRACTION["field_definitions"])
    if isinstance(domain_config.get("field_definitions"), dict):
        for key, value in domain_config["field_definitions"].items():
            if isinstance(value, str) and value.strip():
                field_definitions[str(key)] = value.strip()
    config["field_definitions"] = field_definitions

    if isinstance(domain_config.get("return_instruction"), str) and domain_config["return_instruction"].strip():
        config["return_instruction"] = domain_config["return_instruction"].strip()

    return config


def render_metadata_system_prompt() -> str:
    rules = metadata_extraction_config()["system_rules"]
    return "\n".join(rules).strip() + "\n"


def metadata_definition_for(logical_name: str) -> str:
    config = metadata_extraction_config()
    definitions = config["field_definitions"]
    prompt_label = metadata_prompt_label(logical_name)
    payload_field = metadata_payload_field(logical_name)
    return str(
        definitions.get(prompt_label)
        or definitions.get(payload_field)
        or definitions.get(logical_name)
        or DEFAULT_METADATA_EXTRACTION["field_definitions"].get(DEFAULT_METADATA_FIELDS[logical_name]["prompt_label"])
        or f"metadata field {prompt_label}."
    )


def metadata_allowed_lines() -> str:
    lines = []
    for logical_name in LOGICAL_METADATA_ORDER:
        label = metadata_prompt_label(logical_name)
        values = ", ".join(sorted(METADATA_ALLOWED_VALUES[logical_name]))
        lines.append(f"{label}: {values}")
    lines.append(f"boolean flags: {', '.join(BOOLEAN_FLAG_FIELDS)}")
    return "\n".join(lines)


def metadata_definition_lines() -> str:
    lines = []
    for logical_name in LOGICAL_METADATA_ORDER:
        lines.append(f"- {metadata_prompt_label(logical_name)}: {metadata_definition_for(logical_name)}")
    definitions = metadata_extraction_config()["field_definitions"]
    flag_definition = definitions.get("boolean_flags") or DEFAULT_METADATA_EXTRACTION["field_definitions"]["boolean_flags"]
    lines.append(f"- boolean flags: {flag_definition}")
    return "\n".join(lines)


def compact_row_field_names() -> list[str]:
    return [
        "chunk_index",
        *[metadata_prompt_label(name) for name in LOGICAL_METADATA_ORDER],
        *BOOLEAN_FLAG_FIELDS,
        "confidence",
        "reason_short",
    ]


def compact_example_value(logical_name: str):
    allowed = sorted(METADATA_ALLOWED_VALUES[logical_name])
    cfg_default = metadata_field_config(logical_name).get("default")
    if metadata_kind(logical_name) == "list_enum":
        if isinstance(cfg_default, list) and cfg_default:
            return cfg_default
        for preferred in ("validation", "perception", "verification", "implementation", "unknown"):
            if preferred in allowed:
                return [preferred]
        return [allowed[0]]
    if isinstance(cfg_default, str) and cfg_default in allowed:
        return cfg_default
    for preferred in ("test", "high", "primary", "unknown", "secondary", "low"):
        if preferred in allowed:
            return preferred
    return allowed[0]


def compact_example_item() -> list:
    return [
        0,
        *[compact_example_value(name) for name in LOGICAL_METADATA_ORDER],
        *([False] * len(BOOLEAN_FLAG_FIELDS)),
        0.85,
        "Short reason.",
    ]


def render_metadata_user_prompt(
        expected_indices,
        batch_chunks: list,
) -> str:
    config = metadata_extraction_config()
    compact_row_fields = ",\n      ".join(compact_row_field_names())

    return f"""{config['objective']}

Expected chunk_index values:
{json.dumps(expected_indices)}

Allowed values:
{metadata_allowed_lines()}

Definitions:
{metadata_definition_lines()}

{config['return_instruction']}

Output schema:
{{
  "items": [
    [
      {compact_row_fields}
    ]
  ]
}}

Example item:
{json.dumps(compact_example_item(), ensure_ascii=False)}

Chunks:
{json.dumps(batch_chunks, ensure_ascii=False, indent=2)}
"""

def make_balanced_batches(
        chunks,
        target_batch_size: int,
        target_max_initial_batches: int,
        max_batch_size: int,
):
    n = len(chunks)
    if n == 0:
        return []

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

    start_chunk_index = batch_chunks[0]["chunk_index"] if batch_chunks else -1
    expected_indices = [x["chunk_index"] for x in batch_chunks]
    expected_set = set(expected_indices)

    system_prompt = render_metadata_system_prompt()
    user_prompt = render_metadata_user_prompt(
        expected_indices=expected_indices,
        batch_chunks=batch_chunks,
    )

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

def _score_weight(mapping: dict, value, default: float = 0.0) -> float:
    try:
        return float(mapping.get(str(value), default))
    except Exception:
        return default


def score_for_metadata(meta: dict) -> float:
    weights = document_aggregation_config().get("score_weights") or {}
    score = float(weights.get("base", 1.0))

    try:
        score += float(meta.get("confidence", 0.0)) * float(weights.get("confidence", 1.0))
    except Exception:
        pass

    score += _score_weight(weights.get("delivery_value", {}), meta.get(DELIVERY_FIELD))
    score += _score_weight(weights.get("criticality", {}), meta.get(CRITICALITY_FIELD))
    score += _score_weight(weights.get("decision", {}), meta.get(DECISION_FIELD))
    score += _score_weight(weights.get("role", {}), meta.get(ROLE_FIELD))

    return max(0.0, score)


def weighted_top_values(chunk_metas, field: str, limit: int = 6, ignore_values=None):
    ignore_values = ignore_values or set()
    weights = defaultdict(float)

    for meta in chunk_metas:
        value = meta.get(field)
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
        value = meta.get(field)
        r = rank_map.get(value, 0)
        if r > best_rank:
            best = value
            best_rank = r

    return best


def aggregate_document_metadata(chunk_metas):
    signal_cfg = (document_aggregation_config().get("signal") or {})
    decision_cfg = (document_aggregation_config().get("decision") or {})
    limits = document_aggregation_config().get("weighted_top_limits") or {}
    ignore_values = set(document_aggregation_config().get("ignore_values") or ["unknown", "noise"])

    primary_decisions = set(str(x) for x in decision_cfg.get("primary_values", ["primary"]))
    drop_decisions = set(str(x) for x in decision_cfg.get("drop_values", ["drop"]))
    default_decision = str(decision_cfg.get("default", "secondary"))

    if not chunk_metas:
        return {
            DOC_PRIMARY_ROLE_FIELD: "unknown",
            DOC_ROLES_FIELD: ["unknown"],
            DOC_FACETS_FIELD: ["unknown"],
            DOC_LAYERS_FIELD: ["unknown"],
            DOC_STAGES_FIELD: ["unknown"],
            DOC_CRITICALITY_FIELD: "unknown",
            DOC_DELIVERY_FIELD: "low",
            DOC_DECISION_FIELD: default_decision,
            DOC_CONFIDENCE_FIELD: 0.0,
            DOC_SIGNAL_CHUNKS_FIELD: [],
        }

    roles = weighted_top_values(
        chunk_metas,
        ROLE_FIELD,
        limit=int(limits.get("roles", 6)),
        ignore_values=ignore_values,
    )
    facets = weighted_top_values(
        chunk_metas,
        FACETS_FIELD,
        limit=int(limits.get("facets", 10)),
        ignore_values={"unknown"},
    )
    layers = weighted_top_values(
        chunk_metas,
        LAYERS_FIELD,
        limit=int(limits.get("layers", 10)),
        ignore_values={"unknown"},
    )
    stages = weighted_top_values(
        chunk_metas,
        STAGES_FIELD,
        limit=int(limits.get("stages", 8)),
        ignore_values={"unknown"},
    )

    document_primary_role = roles[0] if roles else "unknown"
    document_criticality = max_enum(chunk_metas, CRITICALITY_FIELD, SAFETY_RANK, "unknown")
    document_delivery_value = max_enum(chunk_metas, DELIVERY_FIELD, DELIVERY_RANK, "low")

    decisions = [m.get(DECISION_FIELD) for m in chunk_metas]
    if any(d in primary_decisions for d in decisions):
        document_decision = "primary" if "primary" in primary_decisions else sorted(primary_decisions)[0]
    elif decisions and all(d in drop_decisions for d in decisions):
        document_decision = "drop" if "drop" in drop_decisions else sorted(drop_decisions)[0]
    else:
        document_decision = default_decision

    avg_confidence = sum(m.get("confidence", 0.0) for m in chunk_metas) / len(chunk_metas)

    signal_decisions = set(str(x) for x in signal_cfg.get("decision_values", ["primary"]))
    signal_delivery_values = set(str(x) for x in signal_cfg.get("delivery_values", ["high"]))
    signal_criticality_values = set(str(x) for x in signal_cfg.get("criticality_values", ["high"]))
    signal_roles = set(str(x) for x in signal_cfg.get("roles", []))
    signal_flags = [str(x) for x in signal_cfg.get("flags", [])]
    max_signal_chunks = int(signal_cfg.get("max_signal_chunks", 20))

    signal_chunks = []
    for i, meta in enumerate(chunk_metas):
        role = meta.get(ROLE_FIELD)
        if (
                meta.get(DECISION_FIELD) in signal_decisions
                or meta.get(DELIVERY_FIELD) in signal_delivery_values
                or meta.get(CRITICALITY_FIELD) in signal_criticality_values
                or role in signal_roles
                or any(meta.get(flag) is True for flag in signal_flags)
        ):
            signal_chunks.append(i)

    return {
        DOC_PRIMARY_ROLE_FIELD: document_primary_role,
        DOC_ROLES_FIELD: roles or ["unknown"],
        DOC_FACETS_FIELD: facets or ["unknown"],
        DOC_LAYERS_FIELD: layers or ["unknown"],
        DOC_STAGES_FIELD: stages or ["unknown"],
        DOC_CRITICALITY_FIELD: document_criticality,
        DOC_DELIVERY_FIELD: document_delivery_value,
        DOC_DECISION_FIELD: document_decision,
        DOC_CONFIDENCE_FIELD: round(avg_confidence, 3),
        DOC_SIGNAL_CHUNKS_FIELD: signal_chunks[:max_signal_chunks],
    }


# =============================================================================
# Qdrant setup
# =============================================================================

def create_payload_indexes(client: QdrantClient):
    index_specs = unique_index_fields([
        ("file_path", models.PayloadSchemaType.KEYWORD),
        ("file_name", models.PayloadSchemaType.KEYWORD),
        ("chunk_index", models.PayloadSchemaType.INTEGER),
        (ROLE_FIELD, models.PayloadSchemaType.KEYWORD),
        (FACETS_FIELD, models.PayloadSchemaType.KEYWORD),
        (LAYERS_FIELD, models.PayloadSchemaType.KEYWORD),
        (STAGES_FIELD, models.PayloadSchemaType.KEYWORD),
        (CRITICALITY_FIELD, models.PayloadSchemaType.KEYWORD),
        (DELIVERY_FIELD, models.PayloadSchemaType.KEYWORD),
        (DECISION_FIELD, models.PayloadSchemaType.KEYWORD),
        (DOC_PRIMARY_ROLE_FIELD, models.PayloadSchemaType.KEYWORD),
        (DOC_ROLES_FIELD, models.PayloadSchemaType.KEYWORD),
        (DOC_FACETS_FIELD, models.PayloadSchemaType.KEYWORD),
        (DOC_LAYERS_FIELD, models.PayloadSchemaType.KEYWORD),
        (DOC_STAGES_FIELD, models.PayloadSchemaType.KEYWORD),
        (DOC_CRITICALITY_FIELD, models.PayloadSchemaType.KEYWORD),
        (DOC_DELIVERY_FIELD, models.PayloadSchemaType.KEYWORD),
        (DOC_DECISION_FIELD, models.PayloadSchemaType.KEYWORD),
    ])

    log(f"Creating payload indexes: {', '.join(field for field, _ in index_specs)}")
    for field, schema in index_specs:
        client.create_payload_index(COLLECTION, field, schema)


def print_document_metadata_summary(document_meta, chunk_metas):
    role_counts = defaultdict(int)
    facet_counts = defaultdict(int)
    layer_counts = defaultdict(int)
    decision_counts = defaultdict(int)

    for meta in chunk_metas:
        role_counts[meta.get(ROLE_FIELD)] += 1
        decision_counts[meta.get(DECISION_FIELD)] += 1
        for f in meta.get(FACETS_FIELD) or []:
            facet_counts[f] += 1
        for layer in meta.get(LAYERS_FIELD) or []:
            layer_counts[layer] += 1

    top_roles = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_facets = sorted(facet_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    top_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:8]

    log(
        f"  document: primary_role={document_meta.get(DOC_PRIMARY_ROLE_FIELD)} "
        f"criticality={document_meta.get(DOC_CRITICALITY_FIELD)} "
        f"delivery={document_meta.get(DOC_DELIVERY_FIELD)} "
        f"decision={document_meta.get(DOC_DECISION_FIELD)} "
        f"confidence={document_meta.get(DOC_CONFIDENCE_FIELD)}"
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
    print(f"Domain: {DOMAIN.id} ({DOMAIN.display_name})")
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

            if chunk_meta.get(DECISION_FIELD) == "drop" and not INDEX_DROPPED_CHUNKS:
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