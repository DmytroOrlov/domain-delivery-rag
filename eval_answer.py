#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

import rag_core as rc

# =============================================================================
# ADAS / Embedded Vision Delivery RAG - Answer Contract Eval
# =============================================================================
#
# Purpose:
#   Lightweight answer-level regression gate for the local RAG system.
#
# This is NOT a full semantic faithfulness evaluator. It is a practical v1 gate
# that catches answer-contract regressions and obvious grounding hygiene issues.
#
# What this eval checks:
#   - the answer has the required six-section structure;
#   - no-answer/insufficient-evidence cases explicitly abstain instead of
#     inventing thresholds, ASIL/SIL assignments, or formal safety goals;
#   - the answer cites retrieved sources as [S1], [S2], ...;
#   - visible reasoning does not remain in the final parsed answer;
#   - raw /rag command does not leak;
#   - full local paths do not leak;
#   - classification metadata (chunk_role, content_facets, reason_short, etc.)
#     does not appear as answer evidence.
#
# Qwen / llama-server contract:
#   This script does not send server-side response-routing overrides.
#
#   The goal is to evaluate the default response shape produced by the same
#   Qwen-instruct / llama-server setup used by the rest of the local system.
#
#   Qwen-style responses may appear in one of these shapes:
#     1. message.content is already the final answer;
#     2. message.reasoning_content contains reasoning and message.content is final;
#     3. message.content contains <think>...</think> followed by the final answer;
#     4. message.content starts with a dangling </think> followed by final answer.
#
#   eval_answer.py therefore stores the raw message.content and evaluates a
#   narrowly parsed final answer:
#     - if </think> appears, keep only text after the last </think>;
#     - otherwise keep message.content as-is;
#     - then, if needed, trim to the requested "1. Conclusion" section start.
#
#   This is parsing the default model response, not changing llama-server
#   behavior. A malformed unclosed <think> remains a hard failure.
#
# Recommended local sequence:
#   python3 eval_retrieval.py
#   python3 eval_answer.py
#
# One-case smoke test:
#   python3 eval_answer.py --case blind_spot_warning_implications
# =============================================================================


BASE_DIR = Path(os.environ.get("RAG_BASE_DIR", os.path.expanduser("~/rag_v1")))
DEFAULT_EVAL_FILE = BASE_DIR / "eval_queries.json"
RUNS_DIR = Path(os.environ.get("RAG_EVAL_RUNS_DIR", str(BASE_DIR / "eval_runs")))

CHAT_URL = os.environ.get("RAG_CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions")

ANSWER_MAX_TOKENS = int(os.environ.get("RAG_ANSWER_EVAL_MAX_TOKENS", "16384"))
ANSWER_TIMEOUT = int(os.environ.get("RAG_ANSWER_EVAL_TIMEOUT", "1800"))
DEFAULT_TOP_K = int(os.environ.get("RAG_ANSWER_EVAL_TOP_K", "5"))

MIN_ANSWER_CHARS = int(os.environ.get("RAG_ANSWER_EVAL_MIN_CHARS", "600"))
MIN_CITATION_COUNT = int(os.environ.get("RAG_ANSWER_EVAL_MIN_CITATIONS", "2"))

VERBOSE = os.environ.get("RAG_ANSWER_EVAL_VERBOSE", "1") != "0"
SAVE_PROMPTS = os.environ.get("RAG_ANSWER_EVAL_SAVE_PROMPTS", "1") != "0"


# =============================================================================
# Small utilities
# =============================================================================

def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_file_name(value: str) -> str:
    value = value or "case"
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value[:120].strip("_") or "case"


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def preview(text: str, n: int = 280) -> str:
    text = (text or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= n:
        return text
    return text[:n] + f"...[truncated {len(text) - n} chars]"


def load_eval_cases(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Eval file must contain a list of cases: {path}")

    cases = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Eval case #{i + 1} must be object")

        case_id = item.get("id") or f"case_{i + 1}"
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Eval case {case_id!r} has no query")

        cases.append(item)

    return cases


# =============================================================================
# Prompt building through shared rag_core
# =============================================================================

def build_augmented_prompt(case: dict) -> str:
    """
    Build prompt through the current shared rag_core contract.

    No fallback signatures, no compatibility layer. If rag_core changes, this
    eval should fail loudly so the contract is updated deliberately.
    """
    question = case["query"]
    top_k = int(case.get("top_k") or DEFAULT_TOP_K)

    prompt, _info = rc.build_augmented_prompt(
        question=question,
        top_k=top_k,
    )
    return prompt


# =============================================================================
# llama-server answer generation
# =============================================================================

def call_chat(prompt: str):
    """
    Call llama-server with the default Qwen-instruct response behavior.

    Important: do not send server-side response-routing overrides here.
    The eval parses the default model/server response.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": ANSWER_MAX_TOKENS,
    }

    started = time.time()
    r = requests.post(CHAT_URL, json=payload, timeout=ANSWER_TIMEOUT)
    elapsed = time.time() - started

    try:
        data = r.json()
    except Exception:
        data = {"raw_text": r.text}

    if r.status_code >= 400:
        raise RuntimeError(
            f"Chat request failed HTTP {r.status_code}: "
            f"{json.dumps(data, ensure_ascii=False)[:2000]}"
        )

    try:
        msg = data["choices"][0]["message"]
    except Exception as e:
        raise RuntimeError(
            f"Unexpected chat response shape: {json.dumps(data, ensure_ascii=False)[:2000]}"
        ) from e

    raw_content = msg.get("content") or ""
    reasoning_content = msg.get("reasoning_content") or ""
    parsed_answer, cleanup = rc.parse_qwen_final_answer(raw_content)

    return {
        "raw_answer": raw_content,
        "answer": parsed_answer,
        "answer_cleanup": cleanup,
        "reasoning_content_chars": len(reasoning_content),
        "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
        "elapsed_sec": elapsed,
        "usage": data.get("usage"),
        "raw_response": data,
    }



# =============================================================================
# Answer checks
# =============================================================================

# Accept plain numbered headings and common Markdown-bold variants, e.g.
#   1. Conclusion
#   1. **Conclusion**
# The eval is checking the answer contract, not enforcing one exact Markdown style.
REQUIRED_SECTION_PATTERNS = [
    ("conclusion", r"(?im)^\s*1[.)]\s*(?:\*\*)?\s*conclusion\b\s*(?:\*\*)?"),
    ("supported_facts", r"(?im)^\s*2[.)]\s*(?:\*\*)?\s*supported facts\b\s*(?:\*\*)?"),
    ("inferences", r"(?im)^\s*3[.)]\s*(?:\*\*)?\s*inferences\b\s*(?:\*\*)?"),
    ("implementation_implications", r"(?im)^\s*4[.)]\s*(?:\*\*)?\s*implementation implications\b\s*(?:\*\*)?"),
    ("unknowns", r"(?im)^\s*5[.)]\s*(?:\*\*)?\s*unknowns\s*/\s*verification needed\b\s*(?:\*\*)?"),
    ("source_mapping", r"(?im)^\s*6[.)]\s*(?:\*\*)?\s*source mapping\b\s*(?:\*\*)?"),
]

REASONING_LEAK_PATTERNS = [
    r"<think>",
    r"</think>",
    r"(?i)\bhere'?s a thinking process\b",
    r"(?i)\bself-correction\b",
    r"(?i)\boutput generation\b",
    r"(?i)\bproceeds\b",
]

LOCAL_PATH_PATTERNS = [
    r"/Users/[^ \n\t]+",
    r"/home/[^ \n\t]+",
    r"/mnt/[^ \n\t]+",
    r"/tmp/[^ \n\t]+",
    r"[A-Za-z]:\\[^ \n\t]+",
]

INSUFFICIENT_EVIDENCE_PATTERNS = [
    r"(?i)\bnot specified\b",
    r"(?i)\bnot provided\b",
    r"(?i)\bnot enough evidence\b",
    r"(?i)\binsufficient evidence\b",
    r"(?i)\bnot explicitly\b",
    r"(?i)\bcannot determine\b",
    r"(?i)\bunknown\b",
    r"(?i)\bno retrieved evidence\b",
]

CLASSIFICATION_METADATA_LEAK_PATTERNS = [
    r"\bchunk_role\b",
    r"\bcontent_facets\b",
    r"\bsystem_layers\b",
    r"\bworkflow_stages\b",
    r"\breason_short\b",
    r"\bcorpus_decision\b",
    r"\bdocument_primary_role\b",
    r"\bdocument_content_facets\b",
    r"\bdocument_system_layers\b",
    r"\bdocument_signal_chunks\b",
    r"\bhas_behavioral_requirements\b",
    r"\bhas_interface_or_contract\b",
    r"\bhas_validation_or_test_evidence\b",
    r"\bhas_failure_or_degraded_mode\b",
    r"\bhas_regulatory_or_compliance\b",
]


def collect_regex_hits(text: str, patterns: list[str]):
    hits = []
    for pattern in patterns:
        m = re.search(pattern, text or "")
        if m:
            hits.append({"pattern": pattern, "match": m.group(0)[:180], "pos": m.start()})
    return hits


def citation_ids(answer: str):
    return [int(x) for x in re.findall(r"\[S(\d+)\]", answer or "")]


def check_required_sections(answer: str):
    failures = []
    present = {}
    for name, pattern in REQUIRED_SECTION_PATTERNS:
        ok = bool(re.search(pattern, answer or ""))
        present[name] = ok
        if not ok:
            failures.append(f"missing required section: {name}")
    return present, failures


def run_answer_checks(answer: str, case: dict, cleanup: dict[str, Any]):
    failures = []
    warnings = []

    if cleanup.get("unclosed_think_after_parse"):
        failures.append("unclosed <think> remained after Qwen final-answer parsing")

    if not answer.strip():
        failures.append("empty assistant content")
        return {"passed": False, "failures": failures, "warnings": warnings, "metrics": {}, "details": {}}

    min_chars = int(case.get("min_answer_chars") or MIN_ANSWER_CHARS)
    if len(answer) < min_chars:
        failures.append(f"answer too short: chars={len(answer)} min={min_chars}")

    sections, section_failures = check_required_sections(answer)
    failures.extend(section_failures)

    citations = citation_ids(answer)
    unique_citations = sorted(set(citations))
    min_citations = int(case.get("min_citation_count") or MIN_CITATION_COUNT)

    if len(citations) < min_citations:
        failures.append(f"too few source citations: citations={len(citations)} min={min_citations}")
    if not unique_citations:
        failures.append("no [S#] citations found")

    suspicious_source_ids = [x for x in unique_citations if x > 12]
    if suspicious_source_ids:
        warnings.append(f"suspiciously high source ids: {suspicious_source_ids}")

    reasoning_hits = collect_regex_hits(answer, REASONING_LEAK_PATTERNS)
    if reasoning_hits:
        failures.append(f"visible reasoning leak detected after parsing: {reasoning_hits[:3]}")

    if re.search(r"(?im)^\s*/rag\b", answer):
        failures.append("raw /rag command leaked into answer")

    path_hits = collect_regex_hits(answer, LOCAL_PATH_PATTERNS)
    if path_hits:
        failures.append(f"full local path leaked into answer: {path_hits[:3]}")

    metadata_hits = collect_regex_hits(answer, CLASSIFICATION_METADATA_LEAK_PATTERNS)
    if metadata_hits:
        failures.append(f"classification metadata leaked into answer: {metadata_hits[:5]}")

    if case.get("expected_answer_mode") == "insufficient_evidence":
        insufficiency_hits = collect_regex_hits(answer, INSUFFICIENT_EVIDENCE_PATTERNS)
        if not insufficiency_hits:
            failures.append(
                "expected insufficient-evidence behavior, but answer did not contain "
                "an explicit uncertainty/abstention phrase"
            )

    for pattern in case.get("required_answer_patterns", []):
        if not re.search(pattern, answer, flags=re.I | re.M):
            failures.append(f"missing required answer pattern: {pattern}")

    for pattern in case.get("forbidden_answer_patterns", []):
        m = re.search(pattern, answer, flags=re.I | re.M)
        if m:
            failures.append(f"forbidden answer pattern matched: {pattern}; match={m.group(0)[:160]!r}")

    metrics = {
        "answer_chars": len(answer),
        "citation_count": len(citations),
        "unique_citation_count": len(unique_citations),
        "unique_citations": [f"S{x}" for x in unique_citations],
        "section_count": sum(1 for ok in sections.values() if ok),
    }

    details = {
        "sections": sections,
        "reasoning_hits": reasoning_hits,
        "path_hits": path_hits,
        "metadata_hits": metadata_hits,
        "answer_cleanup": cleanup,
    }

    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "metrics": metrics,
        "details": details,
    }


# =============================================================================
# Eval runner
# =============================================================================

def eval_case(case: dict, run_dir: Path, ordinal: int):
    case_id = case.get("id") or f"case_{ordinal}"

    print("=" * 100)
    print(f"CASE {ordinal}: {case_id}")
    print("=" * 100)
    print(f"Query: {case['query']}")
    print(f"top_k={case.get('top_k') or DEFAULT_TOP_K}")
    if case.get("notes"):
        print(f"Notes: {case['notes']}")

    started = time.time()
    prompt = build_augmented_prompt(case)
    prompt_elapsed = time.time() - started

    prompt_warning = None
    if re.search(r"/Users/|/home/|/mnt/|/tmp/", prompt):
        prompt_warning = "prompt contains full local path"

    print(f"Prompt built: chars={len(prompt)} elapsed={prompt_elapsed:.2f}s")
    if prompt_warning:
        print(f"⚠️  {prompt_warning}")

    response = call_chat(prompt)
    raw_answer = response["raw_answer"]
    answer = response["answer"]
    cleanup = response["answer_cleanup"]

    print(
        f"Answer generated: raw_chars={len(raw_answer)} parsed_chars={len(answer)} "
        f"elapsed={response['elapsed_sec']:.2f}s "
        f"finish_reason={response['finish_reason']} "
        f"reasoning_content_chars={response['reasoning_content_chars']}"
    )
    print(f"Answer cleanup: {json.dumps(cleanup, ensure_ascii=False)}")
    print(f"Answer preview: {preview(answer)}")

    checks = run_answer_checks(answer, case, cleanup)
    if prompt_warning:
        checks["warnings"].append(prompt_warning)

    case_dir = run_dir / f"{ordinal:02d}_{safe_file_name(case_id)}"
    case_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_PROMPTS:
        write_text(case_dir / "prompt.txt", prompt)
    write_text(case_dir / "raw_answer.txt", raw_answer)
    write_text(case_dir / "answer.txt", answer)

    result = {
        "id": case_id,
        "query": case["query"],
        "top_k": int(case.get("top_k") or DEFAULT_TOP_K),
        "passed": checks["passed"],
        "failures": checks["failures"],
        "warnings": checks["warnings"],
        "metrics": checks["metrics"],
        "details": checks["details"],
        "prompt_chars": len(prompt),
        "prompt_build_elapsed_sec": prompt_elapsed,
        "answer_elapsed_sec": response["elapsed_sec"],
        "finish_reason": response["finish_reason"],
        "reasoning_content_chars": response["reasoning_content_chars"],
        "answer_cleanup": cleanup,
        "usage": response["usage"],
    }
    write_json(case_dir / "result.json", result)

    if checks["passed"]:
        print(f"PASS: {case_id}")
    else:
        print(f"FAIL: {case_id}")
        for failure in checks["failures"]:
            print(f"  - {failure}")

    if checks["warnings"]:
        print("Warnings:")
        for warning in checks["warnings"]:
            print(f"  - {warning}")

    print()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run answer contract / grounding hygiene eval for local RAG.")
    parser.add_argument(
        "eval_file",
        nargs="?",
        default=str(DEFAULT_EVAL_FILE),
        help=f"Path to eval queries JSON. Default: {DEFAULT_EVAL_FILE}",
    )
    parser.add_argument("--case", dest="case_id", default=None, help="Run only one eval case id.")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases after filtering. 0 means all.")
    return parser.parse_args()


def main():
    args = parse_args()
    eval_file = Path(args.eval_file).expanduser()

    cases = load_eval_cases(eval_file)
    if args.case_id:
        cases = [c for c in cases if c.get("id") == args.case_id]
        if not cases:
            raise SystemExit(f"No eval case found with id={args.case_id!r}")
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    run_id = now_run_id()
    run_dir = RUNS_DIR / f"answer_eval_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("RAG ANSWER CONTRACT EVAL START")
    print("=" * 100)
    print(f"Eval file: {eval_file}")
    print(f"Run dir: {run_dir}")
    print(f"Chat URL: {CHAT_URL}")
    print(f"Answer max tokens: {ANSWER_MAX_TOKENS}")
    print(f"Answer timeout: {ANSWER_TIMEOUT}")
    print("Qwen response handling: default server response parsed locally; no response-routing override is sent")
    print(f"Cases: {len(cases)}")
    print("Eval scope: answer contract / grounding hygiene; not full semantic faithfulness.")
    print("Retrieval/context packing is delegated to rag_core.py.")
    print("=" * 100)
    print()

    results = []
    for i, case in enumerate(cases, start=1):
        try:
            result = eval_case(case, run_dir=run_dir, ordinal=i)
        except Exception as e:
            case_id = case.get("id") or f"case_{i}"
            print("=" * 100)
            print(f"ERROR: {case_id}")
            print("=" * 100)
            print(f"{type(e).__name__}: {e}")
            print()
            result = {
                "id": case_id,
                "query": case.get("query"),
                "passed": False,
                "failures": [f"eval exception: {type(e).__name__}: {e}"],
                "warnings": [],
                "metrics": {},
                "details": {},
            }
        results.append(result)

    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed
    failed_ids = [r["id"] for r in results if not r.get("passed")]

    summary = {
        "run_id": run_id,
        "eval_file": str(eval_file),
        "run_dir": str(run_dir),
        "chat_url": CHAT_URL,
        "qwen_answer_parsing": "default server response parsed locally; no response-routing override sent",
        "cases_total": len(results),
        "passed": passed,
        "failed": failed,
        "failed_ids": failed_ids,
        "scope": "answer contract / grounding hygiene; not full semantic faithfulness",
        "results": results,
    }
    write_json(run_dir / "summary.json", summary)

    print("=" * 100)
    print("RAG ANSWER CONTRACT EVAL SUMMARY")
    print("=" * 100)
    print(f"Run dir: {run_dir}")
    print(f"Cases total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if failed_ids:
        print()
        print("Failed case ids:")
        for case_id in failed_ids:
            print(f"  - {case_id}")
    print("=" * 100)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
