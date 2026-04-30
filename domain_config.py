#!/usr/bin/env python3
"""
Domain configuration loader for the local Domain Delivery RAG.

Default domain:
  RAG_DOMAIN=adas_embedded_vision

The loader intentionally starts small. It creates a domain boundary for runtime
paths/defaults/persona, query profiling rules, and metadata-prior
weights, answer contract, ingestion metadata schema, metadata extraction prompt, logical metadata field contract, metadata field mapping, and document aggregation policy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from collections.abc import Iterable


DEFAULT_DOMAIN_ID = "adas_embedded_vision"


@dataclass(frozen=True)
class DomainConfig:
    id: str
    display_name: str
    collection: str
    input_dir: str
    failure_dir: str
    eval_file: str
    eval_source_map: str
    eval_run_dir: str
    answer: dict[str, Any]
    retrieval_defaults: dict[str, Any]
    context_defaults: dict[str, Any]
    query_profiles: list[dict[str, Any]]
    rerank: dict[str, Any]
    metadata_schema: dict[str, Any]
    metadata_extraction: dict[str, Any]
    metadata_field_map: dict[str, str]
    metadata_fields: dict[str, Any]
    document_aggregation: dict[str, Any]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DomainConfig":
        """Build a DomainConfig without synthesizing domain-policy defaults.

        The only implicit selector is DEFAULT_DOMAIN_ID in load_domain_config().
        Every runtime policy used by ingest/retrieval/answer/eval must be present
        in the selected JSON domain pack.
        """
        required_keys = (
            "id",
            "display_name",
            "collection",
            "input_dir",
            "failure_dir",
            "eval_file",
            "eval_source_map",
            "eval_run_dir",
            "answer",
            "retrieval_defaults",
            "context_defaults",
            "query_profiles",
            "rerank",
            "metadata_schema",
            "metadata_extraction",
            "metadata_field_map",
            "metadata_fields",
            "document_aggregation",
        )
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise ValueError("domain config is missing required keys: " + ", ".join(missing))

        return DomainConfig(
            id=str(data["id"]),
            display_name=str(data["display_name"]),
            collection=str(data["collection"]),
            input_dir=str(data["input_dir"]),
            failure_dir=str(data["failure_dir"]),
            eval_file=str(data["eval_file"]),
            eval_source_map=str(data["eval_source_map"]),
            eval_run_dir=str(data["eval_run_dir"]),
            answer=dict(data["answer"]),
            retrieval_defaults=dict(data["retrieval_defaults"]),
            context_defaults=dict(data["context_defaults"]),
            query_profiles=list(data["query_profiles"]),
            rerank=dict(data["rerank"]),
            metadata_schema=dict(data["metadata_schema"]),
            metadata_extraction=dict(data["metadata_extraction"]),
            metadata_field_map={str(k): str(v) for k, v in dict(data["metadata_field_map"]).items()},
            metadata_fields=dict(data["metadata_fields"]),
            document_aggregation=dict(data["document_aggregation"]),
        )


REQUIRED_METADATA_LOGICAL_FIELDS = (
    "role",
    "facets",
    "layers",
    "stages",
    "criticality",
    "delivery_value",
    "decision",
)


def _schema_key_from_field_config(field_cfg: dict[str, Any]) -> str:
    values_ref = str(field_cfg.get("values_ref") or "")
    if field_cfg.get("schema_key"):
        return str(field_cfg["schema_key"])
    if values_ref.startswith("metadata_schema."):
        return values_ref.split(".", 1)[1]
    if "." in values_ref:
        return values_ref.rsplit(".", 1)[-1]
    return str(field_cfg.get("prompt_label") or field_cfg.get("payload") or "")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return list(value)
    return [value]


def _validate_allowed_values(config: DomainConfig, logical_name: str, field_cfg: dict[str, Any]) -> None:
    schema_key = _schema_key_from_field_config(field_cfg)
    if not schema_key:
        raise ValueError(f"metadata_fields.{logical_name}: missing schema_key or values_ref")
    if schema_key not in config.metadata_schema:
        raise ValueError(
            f"metadata_fields.{logical_name}: schema_key {schema_key!r} is not present in metadata_schema"
        )

    allowed = {str(x) for x in _as_list(config.metadata_schema.get(schema_key))}
    kind = str(field_cfg.get("kind") or "enum")
    default = field_cfg.get("default")
    default_values = _as_list(default)

    if kind not in {"enum", "list_enum"}:
        raise ValueError(f"metadata_fields.{logical_name}: unsupported kind {kind!r}")

    for item in default_values:
        if str(item) not in allowed:
            raise ValueError(
                f"metadata_fields.{logical_name}: default {item!r} is not in metadata_schema.{schema_key}"
            )


def _schema_values(config: DomainConfig, logical_name: str) -> set[str]:
    field_cfg = config.metadata_fields[logical_name]
    schema_key = _schema_key_from_field_config(field_cfg)
    return {str(x) for x in _as_list(config.metadata_schema.get(schema_key))}


def _payload_field_for_logical(config: DomainConfig, logical_name: str) -> str:
    field_cfg = config.metadata_fields.get(logical_name) or {}
    return str(field_cfg.get("payload") or logical_name)


def _logical_for_weight_key(config: DomainConfig, key: str) -> str | None:
    key = str(key)
    if key in REQUIRED_METADATA_LOGICAL_FIELDS:
        return key
    for logical_name in REQUIRED_METADATA_LOGICAL_FIELDS:
        if key == _payload_field_for_logical(config, logical_name):
            return logical_name
    return None


def _validate_weight_value_keys(config: DomainConfig, *, context: str, logical_name: str, mapping: Any) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{context} must be an object")
    allowed = _schema_values(config, logical_name)
    _validate_value_members(context=context, values=mapping.keys(), allowed=allowed)
    _validate_numeric_mapping(context, mapping)


def _validate_value_members(
    *,
    context: str,
    values: Any,
    allowed: set[str],
    allow_empty: bool = True,
) -> None:
    items = [str(x) for x in _as_list(values)]
    if not items and allow_empty:
        return
    for value in items:
        if value not in allowed:
            raise ValueError(f"{context} contains {value!r}, allowed values: {sorted(allowed)}")


def _validate_numeric_mapping(context: str, mapping: Any) -> None:
    if mapping is None:
        return
    if not isinstance(mapping, dict):
        raise ValueError(f"{context} must be an object")
    for key, value in mapping.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"{context}.{key} must be numeric, got {type(value).__name__}")



def _require_object(mapping: dict[str, Any], context: str, key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return value


def _require_list(mapping: dict[str, Any], context: str, key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty list")
    return value


def _require_numeric(mapping: dict[str, Any], context: str, key: str) -> None:
    if not isinstance(mapping.get(key), (int, float)):
        raise ValueError(f"{context}.{key} must be numeric")



def _validate_metadata_field_map(config: DomainConfig) -> None:
    """Validate aliases in metadata_field_map.

    The map contains document-level aliases used by aggregation/debug payloads.
    Chunk-level fields are defined by metadata_fields.<logical>.payload. Unknown
    keys are usually typos and should fail early instead of silently producing an
    inconsistent domain pack.
    """
    if not isinstance(config.metadata_field_map, dict):
        raise ValueError("metadata_field_map must be an object")

    allowed = {
        "document_primary_role",
        "document_roles",
        "document_facets",
        "document_layers",
        "document_stages",
        "document_criticality",
        "document_delivery_value",
        "document_decision",
        "document_confidence",
        "document_signal_chunks",
        "source_file",
        "chunk_index",
        "text",
    }
    for key, value in config.metadata_field_map.items():
        if key not in allowed:
            raise ValueError(
                f"metadata_field_map has unknown logical key {key!r}; "
                f"allowed keys: {sorted(allowed)}"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata_field_map.{key} must be a non-empty string")


def _validate_metadata_extraction(config: DomainConfig) -> None:
    extraction = config.metadata_extraction or {}
    if not isinstance(extraction, dict):
        raise ValueError("metadata_extraction must be an object")

    required_keys = ("system_rules", "objective", "field_definitions", "return_instruction")
    missing_keys = [key for key in required_keys if key not in extraction]
    if missing_keys:
        raise ValueError("metadata_extraction is missing required keys: " + ", ".join(missing_keys))

    if not isinstance(extraction["system_rules"], list) or not extraction["system_rules"]:
        raise ValueError("metadata_extraction.system_rules must be a non-empty list")
    for idx, rule in enumerate(extraction["system_rules"]):
        if not isinstance(rule, str) or not rule.strip():
            raise ValueError(f"metadata_extraction.system_rules[{idx}] must be a non-empty string")

    if not isinstance(extraction["objective"], str) or not extraction["objective"].strip():
        raise ValueError("metadata_extraction.objective must be a non-empty string")

    definitions = extraction["field_definitions"]
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("metadata_extraction.field_definitions must be a non-empty object")

    prompt_labels = {
        str((config.metadata_fields.get(logical_name) or {}).get("prompt_label") or "")
        for logical_name in REQUIRED_METADATA_LOGICAL_FIELDS
    }
    payload_fields = {
        _payload_field_for_logical(config, logical_name)
        for logical_name in REQUIRED_METADATA_LOGICAL_FIELDS
    }
    logical_names = set(REQUIRED_METADATA_LOGICAL_FIELDS)
    allowed_definition_keys = logical_names | prompt_labels | payload_fields | {"boolean_flags", "confidence", "reason_short"}
    allowed_definition_keys = {x for x in allowed_definition_keys if x}

    missing = []
    for logical_name in REQUIRED_METADATA_LOGICAL_FIELDS:
        field_cfg = config.metadata_fields[logical_name]
        candidates = {
            logical_name,
            str(field_cfg.get("prompt_label") or ""),
            str(field_cfg.get("payload") or ""),
        }
        if not any(c in definitions for c in candidates if c):
            missing.append(logical_name)
    if missing:
        raise ValueError(
            "metadata_extraction.field_definitions is missing definitions for logical fields: "
            + ", ".join(missing)
        )

    for key, value in definitions.items():
        if key not in allowed_definition_keys:
            raise ValueError(
                f"metadata_extraction.field_definitions has unknown key {key!r}; "
                f"allowed keys: {sorted(allowed_definition_keys)}"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata_extraction.field_definitions.{key} must be a non-empty string")

    if "boolean_flags" in (config.metadata_schema or {}) and "boolean_flags" not in definitions:
        raise ValueError("metadata_extraction.field_definitions.boolean_flags is required when metadata_schema.boolean_flags is defined")

    return_instruction = extraction["return_instruction"]
    if not isinstance(return_instruction, str) or not return_instruction.strip():
        raise ValueError("metadata_extraction.return_instruction must be a non-empty string")

def _validate_answer_contract(config: DomainConfig) -> None:
    answer = config.answer or {}
    if not isinstance(answer, dict):
        raise ValueError("answer must be an object")

    persona = str(answer.get("persona") or "").strip()
    if not persona:
        raise ValueError("answer.persona is required")

    sections = answer.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("answer.sections must be a non-empty list")
    normalized: set[str] = set()
    for idx, section in enumerate(sections):
        if not isinstance(section, str) or not section.strip():
            raise ValueError(f"answer.sections[{idx}] must be a non-empty string")
        key = section.strip().casefold()
        if key in normalized:
            raise ValueError(f"answer.sections contains duplicate section {section!r}")
        normalized.add(key)

    citation_rule = str(answer.get("citation_rule") or "").strip()
    if not citation_rule:
        raise ValueError("answer.citation_rule is required")

    for key in ("grounding_rules", "repair_rules"):
        if key not in answer:
            raise ValueError(f"answer.{key} is required")
        if not isinstance(answer[key], list) or not answer[key]:
            raise ValueError(f"answer.{key} must be a non-empty list of strings")
        for idx, value in enumerate(answer[key]):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"answer.{key}[{idx}] must be a non-empty string")


def _validate_document_aggregation(config: DomainConfig) -> None:
    aggregation = config.document_aggregation or {}
    if not isinstance(aggregation, dict):
        raise ValueError("document_aggregation must be an object")

    required_keys = (
        "criticality_rank",
        "delivery_rank",
        "decision",
        "weighted_top_limits",
        "ignore_values",
        "signal",
        "score_weights",
    )
    missing = [key for key in required_keys if key not in aggregation]
    if missing:
        raise ValueError("document_aggregation is missing required keys: " + ", ".join(missing))

    role_values = _schema_values(config, "role")
    criticality_values = _schema_values(config, "criticality")
    delivery_values = _schema_values(config, "delivery_value")
    decision_values = _schema_values(config, "decision")

    # Rank maps are policy, but their keys must be valid domain values.
    for context, mapping, allowed in (
        ("document_aggregation.criticality_rank", aggregation.get("criticality_rank"), criticality_values),
        ("document_aggregation.delivery_rank", aggregation.get("delivery_rank"), delivery_values),
    ):
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"{context} must be a non-empty object")
        _validate_value_members(context=context, values=mapping.keys(), allowed=allowed)
        _validate_numeric_mapping(context, mapping)

    decision = _require_object(aggregation, "document_aggregation", "decision")
    for key in ("primary_values", "drop_values"):
        _require_list(decision, "document_aggregation.decision", key)
    _validate_value_members(
        context="document_aggregation.decision.primary_values",
        values=decision.get("primary_values"),
        allowed=decision_values,
        allow_empty=False,
    )
    _validate_value_members(
        context="document_aggregation.decision.drop_values",
        values=decision.get("drop_values"),
        allowed=decision_values,
        allow_empty=False,
    )
    if "default" not in decision:
        raise ValueError("document_aggregation.decision.default is required")
    _validate_value_members(
        context="document_aggregation.decision.default",
        values=[decision.get("default")],
        allowed=decision_values,
        allow_empty=False,
    )

    limits = _require_object(aggregation, "document_aggregation", "weighted_top_limits")
    for key in ("roles", "facets", "layers", "stages"):
        _require_numeric(limits, "document_aggregation.weighted_top_limits", key)
        if int(limits[key]) < 0:
            raise ValueError(f"document_aggregation.weighted_top_limits.{key} must be >= 0")

    ignore_values_raw = _require_list(aggregation, "document_aggregation", "ignore_values")
    ignore_values = set()
    for logical_name in REQUIRED_METADATA_LOGICAL_FIELDS:
        ignore_values.update(_schema_values(config, logical_name))
    _validate_value_members(
        context="document_aggregation.ignore_values",
        values=ignore_values_raw,
        allowed=ignore_values,
        allow_empty=False,
    )

    signal = _require_object(aggregation, "document_aggregation", "signal")
    boolean_flags = {str(x) for x in _as_list(config.metadata_schema.get("boolean_flags"))}
    for key in ("decision_values", "delivery_values", "criticality_values", "roles", "flags"):
        _require_list(signal, "document_aggregation.signal", key)
    _validate_value_members(
        context="document_aggregation.signal.decision_values",
        values=signal.get("decision_values"),
        allowed=decision_values,
        allow_empty=False,
    )
    _validate_value_members(
        context="document_aggregation.signal.delivery_values",
        values=signal.get("delivery_values"),
        allowed=delivery_values,
        allow_empty=False,
    )
    _validate_value_members(
        context="document_aggregation.signal.criticality_values",
        values=signal.get("criticality_values"),
        allowed=criticality_values,
        allow_empty=False,
    )
    _validate_value_members(
        context="document_aggregation.signal.roles",
        values=signal.get("roles"),
        allowed=role_values,
        allow_empty=False,
    )
    _validate_value_members(
        context="document_aggregation.signal.flags",
        values=signal.get("flags"),
        allowed=boolean_flags,
        allow_empty=False,
    )
    if not isinstance(signal.get("max_signal_chunks"), int) or int(signal["max_signal_chunks"]) < 0:
        raise ValueError("document_aggregation.signal.max_signal_chunks must be a non-negative integer")

    score_weights = _require_object(aggregation, "document_aggregation", "score_weights")
    for scalar_key in ("base", "confidence"):
        _require_numeric(score_weights, "document_aggregation.score_weights", scalar_key)

    weighted_fields = {
        "role": role_values,
        "criticality": criticality_values,
        "delivery_value": delivery_values,
        "decision": decision_values,
    }
    for field_key, allowed in weighted_fields.items():
        if field_key not in score_weights:
            continue
        mapping = score_weights[field_key]
        if not isinstance(mapping, dict):
            raise ValueError(f"document_aggregation.score_weights.{field_key} must be an object")
        _validate_value_members(
            context=f"document_aggregation.score_weights.{field_key}",
            values=mapping.keys(),
            allowed=allowed,
        )
        _validate_numeric_mapping(f"document_aggregation.score_weights.{field_key}", mapping)

    for key in score_weights.keys():
        if key not in {"base", "confidence", *weighted_fields.keys()}:
            raise ValueError(f"document_aggregation.score_weights has unknown key {key!r}")

def _validate_rerank(config: DomainConfig) -> None:
    """Validate metadata prior configuration.

    Field keys may be logical names (criticality, decision, role, ...) or the
    configured payload names from the active domain pack. Mapping keys inside
    each field must be valid enum values for that logical field; this catches
    typos like "primarry" before a rerank silently stops working.
    """
    rerank = config.rerank or {}
    if not isinstance(rerank, dict):
        raise ValueError("rerank must be an object")

    required_keys = ("mode", "clamp", "base_weights", "confidence_weight", "hit_weights")
    missing = [key for key in required_keys if key not in rerank]
    if missing:
        raise ValueError("rerank is missing required keys: " + ", ".join(missing))

    mode = rerank.get("mode")
    allowed_modes = {"full", "value_weights_only", "disabled"}
    if not isinstance(mode, str) or mode not in allowed_modes:
        raise ValueError(f"rerank.mode must be one of {sorted(allowed_modes)}")

    base_weights = rerank.get("base_weights")
    if not isinstance(base_weights, dict):
        raise ValueError("rerank.base_weights must be an object")
    for key, mapping in base_weights.items():
        logical_name = _logical_for_weight_key(config, str(key))
        if logical_name is None:
            raise ValueError(
                f"rerank.base_weights has unknown field key {key!r}; "
                "use a logical field name or a configured payload field"
            )
        _validate_weight_value_keys(
            config,
            context=f"rerank.base_weights.{key}",
            logical_name=logical_name,
            mapping=mapping,
        )

    if not isinstance(rerank["confidence_weight"], (int, float)):
        raise ValueError("rerank.confidence_weight must be numeric")

    clamp = rerank["clamp"]
    if not isinstance(clamp, list) or len(clamp) != 2 or not all(isinstance(x, (int, float)) for x in clamp):
        raise ValueError("rerank.clamp must be a two-number list")
    if float(clamp[0]) > float(clamp[1]):
        raise ValueError("rerank.clamp lower bound must be <= upper bound")

    hit_weights = rerank.get("hit_weights")
    if not isinstance(hit_weights, dict):
        raise ValueError("rerank.hit_weights must be an object")
    for required_hit in ("role", "facet", "layer", "stage", "flag"):
        if required_hit not in hit_weights:
            raise ValueError(f"rerank.hit_weights.{required_hit} is required")

    allowed_hit_keys = {"role", "facet", "facets", "layer", "layers", "stage", "stages", "flag", "flags"}
    for key, value in hit_weights.items():
        if key not in allowed_hit_keys:
            raise ValueError(
                f"rerank.hit_weights has unknown key {key!r}; allowed keys: {sorted(allowed_hit_keys)}"
            )
        if isinstance(value, dict):
            allowed_inner = {"per_hit", "cap"}
            for required_inner in allowed_inner:
                if required_inner not in value:
                    raise ValueError(f"rerank.hit_weights.{key}.{required_inner} is required")
            for inner_key in value.keys():
                if inner_key not in allowed_inner:
                    raise ValueError(
                        f"rerank.hit_weights.{key} has unknown key {inner_key!r}; "
                        f"allowed keys: {sorted(allowed_inner)}"
                    )
            _validate_numeric_mapping(f"rerank.hit_weights.{key}", value)
        elif not isinstance(value, (int, float)):
            raise ValueError(f"rerank.hit_weights.{key} must be numeric or an object")

def _validate_runtime_defaults(config: DomainConfig) -> None:
    if not isinstance(config.retrieval_defaults, dict):
        raise ValueError("retrieval_defaults must be an object")
    for key in ("top_k", "pre_k", "max_per_file", "neighbor_radius"):
        _require_numeric(config.retrieval_defaults, "retrieval_defaults", key)

    if not isinstance(config.context_defaults, dict):
        raise ValueError("context_defaults must be an object")
    for key in ("selected_max_chars", "neighbor_snippet_chars", "context_max_chars"):
        _require_numeric(config.context_defaults, "context_defaults", key)

    if not isinstance(config.query_profiles, list):
        raise ValueError("query_profiles must be a list")


def validate_domain_config(config: DomainConfig) -> None:
    """Fail fast when a domain config is internally inconsistent.

    This validates the domain boundary rather than relying on code-level policy
    fallbacks. A domain pack should fail loudly if extraction, validation,
    reranking, answer contract, or document aggregation policies are internally
    inconsistent.
    """
    if not config.id:
        raise ValueError("domain config id is required")
    if not config.collection:
        raise ValueError("domain config collection is required")

    _validate_runtime_defaults(config)

    metadata_schema = config.metadata_schema or {}
    metadata_fields = config.metadata_fields or {}

    if not metadata_schema:
        raise ValueError("metadata_schema is required in domain config")
    if not metadata_fields:
        raise ValueError("metadata_fields logical contract is required in domain config")

    for logical_name in REQUIRED_METADATA_LOGICAL_FIELDS:
        field_cfg = metadata_fields.get(logical_name)
        if not isinstance(field_cfg, dict):
            raise ValueError(f"metadata_fields.{logical_name} must be defined as an object")
        payload = str(field_cfg.get("payload") or "")
        prompt_label = str(field_cfg.get("prompt_label") or payload or "")
        if not payload:
            raise ValueError(f"metadata_fields.{logical_name}: payload is required")
        if not prompt_label:
            raise ValueError(f"metadata_fields.{logical_name}: prompt_label is required")
        _validate_allowed_values(config, logical_name, field_cfg)

    _validate_metadata_field_map(config)
    _validate_metadata_extraction(config)

    boolean_flags = {str(x) for x in _as_list(metadata_schema.get("boolean_flags"))}

    # Query profile additions must refer to values in the active domain schema.
    profile_schema = {
        "add_roles": _schema_key_from_field_config(metadata_fields["role"]),
        "add_facets": _schema_key_from_field_config(metadata_fields["facets"]),
        "add_layers": _schema_key_from_field_config(metadata_fields["layers"]),
        "add_stages": _schema_key_from_field_config(metadata_fields["stages"]),
    }
    for idx, rule in enumerate(config.query_profiles or []):
        if not isinstance(rule, dict):
            raise ValueError(f"query_profiles[{idx}] must be an object")
        for key, schema_key in profile_schema.items():
            allowed = {str(x) for x in _as_list(metadata_schema.get(schema_key))}
            for value in _as_list(rule.get(key)):
                if str(value) not in allowed:
                    raise ValueError(
                        f"query_profiles[{idx}].{key} contains {value!r}, "
                        f"not present in metadata_schema.{schema_key}"
                    )
        for flag in _as_list(rule.get("add_flags")):
            if str(flag) not in boolean_flags:
                raise ValueError(f"query_profiles[{idx}].add_flags contains unknown flag {flag!r}")

    _validate_answer_contract(config)
    _validate_document_aggregation(config)
    _validate_rerank(config)

def _default_config_path(domain_id: str) -> Path:
    return Path(__file__).resolve().parent / "domains" / f"{domain_id}.json"


def _path_base_for_config(config_path: Path) -> Path:
    """Return the base directory for relative paths inside a domain config.

    Built-in configs live in ``<repo>/domains/*.json`` and conventionally use
    repo-relative paths such as ``evals/...`` or ``examples/...``. For configs
    outside a ``domains`` directory, relative paths are resolved next to the
    config file so ad-hoc external domain packs remain portable.
    """
    resolved = config_path.expanduser().resolve()
    if resolved.parent.name == "domains":
        return resolved.parent.parent
    return resolved.parent


def _resolve_config_path(value: str, *, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _normalize_domain_paths(config: DomainConfig, *, config_path: Path) -> DomainConfig:
    base_dir = _path_base_for_config(config_path)
    return replace(
        config,
        input_dir=_resolve_config_path(config.input_dir, base_dir=base_dir),
        failure_dir=_resolve_config_path(config.failure_dir, base_dir=base_dir),
        eval_file=_resolve_config_path(config.eval_file, base_dir=base_dir),
        eval_source_map=_resolve_config_path(config.eval_source_map, base_dir=base_dir),
        eval_run_dir=_resolve_config_path(config.eval_run_dir, base_dir=base_dir),
    )


def load_domain_config() -> DomainConfig:
    explicit_domain = os.environ.get("RAG_DOMAIN")
    explicit_config = os.environ.get("RAG_DOMAIN_CONFIG")
    requested_domain = explicit_domain or DEFAULT_DOMAIN_ID
    config_path = Path(explicit_config or str(_default_config_path(requested_domain))).expanduser()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Domain config not found: {config_path}. "
            f"Set RAG_DOMAIN_CONFIG or create domains/{requested_domain}.json"
        )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    config = DomainConfig.from_dict(data)
    config = _normalize_domain_paths(config, config_path=config_path)
    validate_domain_config(config)

    # If a config path is explicitly supplied but RAG_DOMAIN is omitted, trust
    # the config's own id. This makes one-off smoke commands ergonomic while
    # still failing fast when the caller explicitly sets a conflicting domain id.
    if explicit_domain and config.id != explicit_domain:
        raise ValueError(
            f"RAG_DOMAIN={explicit_domain!r} but config id is {config.id!r}: {config_path}"
        )

    return config
