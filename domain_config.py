#!/usr/bin/env python3
"""
Domain configuration loader for the local Domain Delivery RAG.

Default domain:
  RAG_DOMAIN=adas_embedded_vision

The loader intentionally starts small. It creates a domain boundary for runtime
paths/defaults/persona, query profiling rules, and metadata-prior
weights, answer contract, ingestion metadata schema, and metadata extraction prompt.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    answer_persona: str
    answer: dict[str, Any]
    retrieval_defaults: dict[str, Any]
    context_defaults: dict[str, Any]
    query_profiles: list[dict[str, Any]]
    rerank: dict[str, Any]
    metadata_schema: dict[str, Any]
    metadata_extraction: dict[str, Any]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DomainConfig":
        answer = dict(data.get("answer", {}))
        legacy_persona = data.get("answer_persona")
        persona = str(answer.get("persona") or legacy_persona or "You are a senior domain delivery assistant.")

        # Keep the legacy field for old callers, but make the structured answer
        # contract the preferred source for prompt rendering and repair prompts.
        answer.setdefault("persona", persona)

        return DomainConfig(
            id=str(data["id"]),
            display_name=str(data["display_name"]),
            collection=str(data["collection"]),
            input_dir=str(data.get("input_dir", "files")),
            failure_dir=str(data.get("failure_dir", "metadata_failures")),
            eval_file=str(data.get("eval_file", "eval_queries.json")),
            eval_source_map=str(data.get("eval_source_map", "eval_source_map.local.json")),
            eval_run_dir=str(data.get("eval_run_dir", "eval_runs")),
            answer_persona=persona,
            answer=answer,
            retrieval_defaults=dict(data.get("retrieval_defaults", {})),
            context_defaults=dict(data.get("context_defaults", {})),
            query_profiles=list(data.get("query_profiles", [])),
            rerank=dict(data.get("rerank", {})),
            metadata_schema=dict(data.get("metadata_schema", {})),
            metadata_extraction=dict(data.get("metadata_extraction", {})),
        )


def _default_config_path(domain_id: str) -> Path:
    return Path(__file__).resolve().parent / "domains" / f"{domain_id}.json"


def load_domain_config() -> DomainConfig:
    domain_id = os.environ.get("RAG_DOMAIN", DEFAULT_DOMAIN_ID)
    config_path = Path(os.environ.get("RAG_DOMAIN_CONFIG", str(_default_config_path(domain_id)))).expanduser()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Domain config not found: {config_path}. "
            f"Set RAG_DOMAIN_CONFIG or create domains/{domain_id}.json"
        )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    config = DomainConfig.from_dict(data)

    if config.id != domain_id:
        raise ValueError(
            f"RAG_DOMAIN={domain_id!r} but config id is {config.id!r}: {config_path}"
        )

    return config
