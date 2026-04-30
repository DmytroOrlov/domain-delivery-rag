#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

import rag_core as rc

# =============================================================================
# Domain Delivery RAG - Answer Contract Eval
# =============================================================================
#
# Purpose:
#   Lightweight answer-level regression gate for the local RAG system.
#
# This is NOT a full semantic faithfulness evaluator. It is a practical v1 gate
# that catches answer-contract regressions and obvious grounding hygiene issues.
#
# What this eval checks:
#   - the answer has the sections required by the active domain answer contract;
#   - no-answer/insufficient-evidence cases explicitly abstain instead of
#     inventing unsupported values or domain-specific assignments;
#   - the answer cites retrieved sources as [S1], [S2], ...;
#   - visible reasoning does not remain in the final parsed answer;
#   - raw /rag command does not leak;
#   - full local paths do not leak;
#   - classification metadata from the active domain config does not appear as answer evidence.
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
#     - then, if needed, trim to the first configured answer section.
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
#
# Fast expensive loop:
#   python3 eval_answer.py --case-file eval_runs/problem_cases.txt
# =============================================================================


BASE_DIR = Path(os.environ.get("RAG_BASE_DIR", os.path.expanduser("~/rag_v1")))
DEFAULT_EVAL_FILE = Path(os.environ.get("RAG_EVAL_FILE", rc.DOMAIN.eval_file)).expanduser()
RUNS_DIR = Path(
    os.environ.get(
        "RAG_EVAL_RUNS_DIR",
        os.environ.get("RAG_EVAL_RUN_DIR", rc.DOMAIN.eval_run_dir),
    )
).expanduser()

CHAT_URL = os.environ.get("RAG_CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions")

# This eval intentionally does not set generation parameters on chat-completion
# requests. The llama-server launch is the single source of truth for generation
# policy, e.g. the user's launch currently sets:
#   --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --reasoning on
#   --reasoning-budget -1 --n-predict 16384
# Keep eval-side controls to transport/evaluation behavior only.
ANSWER_TIMEOUT = int(os.environ.get("RAG_ANSWER_EVAL_TIMEOUT", "3600"))
DEFAULT_TOP_K = int(os.environ.get("RAG_ANSWER_EVAL_TOP_K", str(rc.DEFAULT_TOP_K)))

MIN_ANSWER_CHARS = int(os.environ.get("RAG_ANSWER_EVAL_MIN_CHARS", "600"))
MIN_CITATION_COUNT = int(os.environ.get("RAG_ANSWER_EVAL_MIN_CITATIONS", "2"))
STRICT_CITATION_FORMAT = os.environ.get("RAG_ANSWER_EVAL_STRICT_CITATIONS", "0") == "1"

VERBOSE = os.environ.get("RAG_ANSWER_EVAL_VERBOSE", "1") != "0"
SAVE_PROMPTS = os.environ.get("RAG_ANSWER_EVAL_SAVE_PROMPTS", "1") != "0"

# Eval-only guardrail. This does not change retrieval; it prevents one stochastic
# malformed generation (loop/truncation/missing citation format) from making the
# whole regression gate noisy. Set to 0 to evaluate first-shot compliance only.
REPAIR_ON_CONTRACT_FAILURE = os.environ.get("RAG_ANSWER_EVAL_REPAIR_ON_CONTRACT_FAILURE", "0") != "0"
MAX_REPAIR_ATTEMPTS = int(os.environ.get("RAG_ANSWER_EVAL_REPAIR_ATTEMPTS", "1"))

# Optional LLM-as-judge pass. Disabled by default so deterministic answer-contract
# eval remains cheap and stable. Enable it for richer semantic/grounding metrics:
#   RAG_ANSWER_EVAL_LLM_JUDGE=1 python3 eval_answer.py
# The judge is advisory by default. To make it fail the eval gate, set:
#   RAG_ANSWER_EVAL_LLM_JUDGE_GATE=1
LLM_JUDGE_ENABLED = os.environ.get("RAG_ANSWER_EVAL_LLM_JUDGE", "0") == "1"
LLM_JUDGE_GATE = os.environ.get("RAG_ANSWER_EVAL_LLM_JUDGE_GATE", "0") == "1"
JUDGE_CHAT_URL = os.environ.get("RAG_ANSWER_EVAL_JUDGE_URL", CHAT_URL)

# Judge uses the same llama-server generation policy as normal answers.
# Do not send judge-side max_tokens/sampling overrides from eval_answer.py.
JUDGE_TIMEOUT = int(os.environ.get("RAG_ANSWER_EVAL_JUDGE_TIMEOUT", str(ANSWER_TIMEOUT)))

JUDGE_PROMPT_MAX_CHARS = int(os.environ.get("RAG_ANSWER_EVAL_JUDGE_PROMPT_MAX_CHARS", "50000"))
JUDGE_ANSWER_MAX_CHARS = int(os.environ.get("RAG_ANSWER_EVAL_JUDGE_ANSWER_MAX_CHARS", "24000"))
JUDGE_MIN_OVERALL_SCORE = float(os.environ.get("RAG_ANSWER_EVAL_JUDGE_MIN_OVERALL_SCORE", "4.0"))

# Progress heartbeat for long local LLM calls. This does not change generation policy;
# it only prints liveness while requests.post is waiting for llama-server.
PROGRESS_INTERVAL_SEC = int(os.environ.get("RAG_ANSWER_EVAL_PROGRESS_INTERVAL", "30"))


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def post_with_progress(url: str, payload: dict[str, Any], timeout: int, label: str):
    """POST with periodic heartbeat so long llama.cpp generations are visible."""
    started = time.time()

    def _post():
        return requests.post(url, json=payload, timeout=timeout)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_post)
        last_print = started
        while True:
            try:
                return future.result(timeout=max(PROGRESS_INTERVAL_SEC, 1))
            except FutureTimeoutError:
                if PROGRESS_INTERVAL_SEC <= 0:
                    continue
                now = time.time()
                if now - last_print >= PROGRESS_INTERVAL_SEC:
                    print(f"… {label}: still waiting for llama-server ({now - started:.0f}s)", flush=True)
                    last_print = now


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

def _retrieval_item_ref(item: dict[str, Any]) -> str:
    p = item["payload"]
    return f"{rc.file_name(p)}:#{p.get('chunk_index')}"


def compact_retrieval_info(info: dict[str, Any], question: str) -> dict[str, Any]:
    """Keep answer-eval retrieval diagnostics useful without storing chunk text."""
    selected = info.get("selected", []) or []
    candidates = info.get("candidates", []) or []
    source_groups = info.get("source_groups", []) or []

    selected_summary = []
    for rank, item in enumerate(selected, start=1):
        p = item["payload"]
        selected_summary.append({
            "rank": rank,
            "ref": _retrieval_item_ref(item),
            "file": rc.file_name(p),
            "chunk_index": p.get("chunk_index"),
            "dense_rank": item.get("dense_rank"),
            "vector_score": item.get("vector_score"),
            "meta_bonus": item.get("meta_bonus"),
            "final_score": item.get("final_score"),
            "decision": rc.payload_decision(p),
            "delivery_value": rc.payload_delivery_value(p),
            "criticality": rc.payload_criticality(p),
            "role": rc.payload_role(p),
        })

    return {
        "top_k": info.get("top_k"),
        "pre_k": info.get("pre_k"),
        "max_per_file": info.get("max_per_file"),
        "neighbor_radius": info.get("neighbor_radius"),
        "candidates_count": info.get("candidates_count", len(candidates)),
        "selected_count": info.get("selected_count", len(selected)),
        "source_groups_count": info.get("source_groups_count", len(source_groups)),
        "context_chars": info.get("context_chars"),
        "prompt_chars": info.get("prompt_chars"),
        "answer_sections": info.get("answer_sections"),
        "query_profile": rc.pretty_profile(rc.query_profile(question)),
        "selected": selected_summary,
        "source_groups": [
            {
                "group_rank": group.get("group_rank"),
                "file_name": group.get("file_name"),
                "selected_indices": group.get("selected_indices"),
                "expanded_indices": group.get("expanded_indices"),
                "best_final_score": group.get("best_final_score"),
            }
            for group in source_groups
        ],
    }


def build_augmented_prompt(case: dict) -> tuple[str, dict[str, Any]]:
    """
    Build prompt through the current shared rag_core contract.

    No fallback signatures, no compatibility layer. If rag_core changes, this
    eval should fail loudly so the contract is updated deliberately.
    """
    question = case["query"]
    top_k = int(case.get("top_k") or DEFAULT_TOP_K)

    prompt, info = rc.build_augmented_prompt(
        question=question,
        top_k=top_k,
    )
    return prompt, compact_retrieval_info(info, question=question)


# =============================================================================
# llama-server answer generation
# =============================================================================

def call_chat(prompt: str, label: str = "answer generation"):
    """
    Call llama-server with the default Qwen-instruct response behavior.

    Important: do not send server-side response-routing overrides here.
    The eval parses the default model/server response.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    started = time.time()
    r = post_with_progress(CHAT_URL, payload, ANSWER_TIMEOUT, label)
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
# Optional LLM judge
# =============================================================================

def clip_text_for_judge(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return (
        text[:head]
        + f"\n\n...[judge input truncated: {len(text) - max_chars} chars omitted]...\n\n"
        + text[-tail:]
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON object extraction for local instruct models."""
    raw = text or ""
    parsed, _cleanup = rc.parse_qwen_final_answer(raw)
    candidates = [parsed.strip(), raw.strip()]

    for candidate in list(candidates):
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.S | re.I)
        candidates.extend(fenced)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(candidate[start : end + 1])
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    raise ValueError(f"Could not parse judge JSON from: {preview(raw, 1200)}")


def clamp_score(value: Any) -> float | None:
    try:
        score = float(value)
    except Exception:
        return None
    if score < 0:
        return 0.0
    if score > 5:
        return 5.0
    return score


def build_llm_judge_prompt(prompt: str, answer: str, case: dict, checks: dict[str, Any]) -> str:
    sections = rc.answer_sections() if hasattr(rc, "answer_sections") else []
    case_requirements = {
        "expected_answer_mode": case.get("expected_answer_mode"),
        "required_answer_patterns": case.get("required_answer_patterns", []),
        "forbidden_answer_patterns": case.get("forbidden_answer_patterns", []),
        "min_citation_count": case.get("min_citation_count", MIN_CITATION_COUNT),
        "required_sections": sections,
    }
    deterministic_summary = {
        "passed": checks.get("passed"),
        "failures": checks.get("failures", []),
        "warnings": checks.get("warnings", []),
        "metrics": checks.get("metrics", {}),
    }

    schema = {
        "pass": True,
        "overall_score": 0,
        "groundedness_score": 0,
        "citation_quality_score": 0,
        "completeness_score": 0,
        "abstention_quality_score": 0,
        "contradiction_risk_score": 0,
        "unsupported_claims": [],
        "citation_errors": [],
        "missing_important_points": [],
        "overclaims": [],
        "notes": "brief explanation",
    }

    return f"""You are a strict evaluator for a Retrieval-Augmented Generation answer.
Return JSON only. Do not include markdown. Do not include chain-of-thought.

Evaluate whether ANSWER_TO_EVALUATE is supported by RETRIEVED_CONTEXT_AND_TASK.
Use only the retrieved context in the prompt, not outside knowledge.
Do not reward plausible domain knowledge if it is not supported by the retrieved context.

Scoring scale for all *_score fields: 0=bad, 5=excellent.
- groundedness_score: factual claims are supported by retrieved context and cited.
- citation_quality_score: citations point to relevant [S#] sources and are not decorative.
- completeness_score: answer addresses the user question with the available evidence.
- abstention_quality_score: answer clearly states unknowns when evidence is missing.
- contradiction_risk_score: 0=no contradiction risk, 5=high contradiction/overclaim risk.
- overall_score: holistic score considering the criteria above.

Pass guideline:
- pass=true only if overall_score >= {JUDGE_MIN_OVERALL_SCORE}, groundedness_score >= 4,
  citation_quality_score >= 4, and contradiction_risk_score <= 2.
- pass=false if there are serious unsupported claims, bad citations, or unsafe overclaims.

Return exactly one JSON object matching this schema shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

CASE_ID:
{case.get('id')}

USER_QUERY:
{case.get('query')}

CASE_REQUIREMENTS_JSON:
{json.dumps(case_requirements, ensure_ascii=False, indent=2)}

DETERMINISTIC_CHECKS_JSON:
{json.dumps(deterministic_summary, ensure_ascii=False, indent=2)}

RETRIEVED_CONTEXT_AND_TASK:
{clip_text_for_judge(prompt, JUDGE_PROMPT_MAX_CHARS)}

ANSWER_TO_EVALUATE:
{clip_text_for_judge(answer, JUDGE_ANSWER_MAX_CHARS)}
"""


def call_llm_judge(prompt: str, answer: str, case: dict, checks: dict[str, Any], label: str = "LLM judge") -> dict[str, Any]:
    judge_prompt = build_llm_judge_prompt(prompt, answer, case, checks)
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict RAG evaluation judge. Return one JSON object only. "
                    "Do not include markdown or chain-of-thought."
                ),
            },
            {"role": "user", "content": judge_prompt},
        ],
        "stream": False,
    }
    started = time.time()
    try:
        r = post_with_progress(JUDGE_CHAT_URL, payload, JUDGE_TIMEOUT, label)
        elapsed = time.time() - started
        try:
            data = r.json()
        except Exception:
            data = {"raw_text": r.text}
        if r.status_code >= 400:
            return {
                "enabled": True,
                "ok": False,
                "passed": False,
                "failure_type": "http_error",
                "elapsed_sec": elapsed,
                "url": JUDGE_CHAT_URL,
                "http_status": r.status_code,
                "error": (
                    f"Judge request failed HTTP {r.status_code}: "
                    f"{json.dumps(data, ensure_ascii=False)[:2000]}"
                ),
            }

        msg = data.get("choices", [{}])[0].get("message", {})
        raw_content = msg.get("content") or ""
        try:
            parsed = extract_json_object(raw_content)
        except Exception as parse_error:
            return {
                "enabled": True,
                "ok": False,
                "passed": False,
                "failure_type": "parse_error",
                "elapsed_sec": elapsed,
                "url": JUDGE_CHAT_URL,
                "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                "usage": data.get("usage"),
                "raw_content": raw_content,
                "raw_preview": preview(raw_content, 2000),
                "error": f"{type(parse_error).__name__}: {parse_error}",
            }

        scores = {
            key: clamp_score(parsed.get(key))
            for key in (
                "overall_score",
                "groundedness_score",
                "citation_quality_score",
                "completeness_score",
                "abstention_quality_score",
                "contradiction_risk_score",
            )
        }
        for key, value in scores.items():
            if value is not None:
                parsed[key] = value

        judge_pass = bool(parsed.get("pass"))
        # If the model omitted pass, derive it from the stable numeric scores.
        if "pass" not in parsed and scores.get("overall_score") is not None:
            judge_pass = (
                (scores.get("overall_score") or 0) >= JUDGE_MIN_OVERALL_SCORE
                and (scores.get("groundedness_score") or 0) >= 4
                and (scores.get("citation_quality_score") or 0) >= 4
                and (scores.get("contradiction_risk_score") or 5) <= 2
            )
            parsed["pass"] = judge_pass

        return {
            "enabled": True,
            "ok": True,
            "passed": judge_pass,
            "failure_type": None if judge_pass else "semantic_failure",
            "elapsed_sec": elapsed,
            "url": JUDGE_CHAT_URL,
            "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
            "usage": data.get("usage"),
            "raw_content": raw_content,
            "result": parsed,
        }
    except Exception as e:
        return {
            "enabled": True,
            "ok": False,
            "passed": False,
            "failure_type": "request_error",
            "elapsed_sec": time.time() - started,
            "url": JUDGE_CHAT_URL,
            "error": f"{type(e).__name__}: {e}",
        }

def add_llm_judge_metrics(checks: dict[str, Any], judge: dict[str, Any]):
    metrics = checks.setdefault("metrics", {})
    details = checks.setdefault("details", {})
    details["llm_judge"] = judge

    metrics["llm_judge_enabled"] = bool(judge.get("enabled"))
    metrics["llm_judge_ok"] = bool(judge.get("ok"))
    metrics["llm_judge_passed"] = bool(judge.get("passed")) if judge.get("ok") else False
    metrics["llm_judge_elapsed_sec"] = judge.get("elapsed_sec")
    metrics["llm_judge_failure_type"] = judge.get("failure_type")
    metrics["llm_judge_parse_failed"] = judge.get("failure_type") == "parse_error"
    metrics["llm_judge_semantic_failed"] = bool(judge.get("ok")) and not bool(judge.get("passed"))

    result = judge.get("result") if isinstance(judge.get("result"), dict) else {}
    for key in (
        "overall_score",
        "groundedness_score",
        "citation_quality_score",
        "completeness_score",
        "abstention_quality_score",
        "contradiction_risk_score",
    ):
        if key in result:
            metrics[f"llm_judge_{key}"] = result.get(key)

    if not judge.get("ok"):
        failure_type = judge.get("failure_type") or "unknown"
        checks.setdefault("warnings", []).append(
            f"LLM judge call failed ({failure_type}): {judge.get('error')}"
        )
        if LLM_JUDGE_GATE:
            checks.setdefault("failures", []).append(
                f"LLM judge call failed ({failure_type}) while gate enabled: {judge.get('error')}"
            )
    elif not judge.get("passed"):
        result = judge.get("result", {})
        note = result.get("notes") if isinstance(result, dict) else None
        msg = "LLM judge did not pass answer"
        if note:
            msg += f": {str(note)[:300]}"
        if LLM_JUDGE_GATE:
            checks.setdefault("failures", []).append(msg)
        else:
            checks.setdefault("warnings", []).append(msg)

    checks["passed"] = not checks.get("failures")


def aggregate_llm_judge(results: list[dict[str, Any]]) -> dict[str, Any]:
    judges = [r.get("llm_judge") for r in results if isinstance(r.get("llm_judge"), dict)]
    ok_judges = [j for j in judges if j.get("ok")]
    failed_judges = [j for j in judges if not j.get("ok")]
    semantic_failed_judges = [j for j in ok_judges if not j.get("passed")]
    failure_types = Counter(str(j.get("failure_type") or "unknown") for j in failed_judges)

    aggregate = {
        "enabled": LLM_JUDGE_ENABLED,
        "gate": LLM_JUDGE_GATE,
        "cases_total": len(results),
        "judged_cases": len(judges),
        "ok_cases": len(ok_judges),
        "failed_judge_calls": len(failed_judges),
        "judge_call_success_rate": (len(ok_judges) / len(judges)) if judges else None,
        "judge_passed_cases": sum(1 for j in ok_judges if j.get("passed")),
        "judge_semantic_failed_cases": len(semantic_failed_judges),
        "judge_semantic_pass_rate": (
            sum(1 for j in ok_judges if j.get("passed")) / len(ok_judges)
        ) if ok_judges else None,
        "judge_end_to_end_pass_rate": (
            sum(1 for j in ok_judges if j.get("passed")) / len(judges)
        ) if judges else None,
        "judge_failure_type_counts": dict(failure_types),
        "judge_parse_error_cases": failure_types.get("parse_error", 0),
        "judge_http_error_cases": failure_types.get("http_error", 0),
        "judge_request_error_cases": failure_types.get("request_error", 0),
    }

    score_keys = (
        "overall_score",
        "groundedness_score",
        "citation_quality_score",
        "completeness_score",
        "abstention_quality_score",
        "contradiction_risk_score",
    )
    for key in score_keys:
        values = []
        for j in ok_judges:
            result = j.get("result") if isinstance(j.get("result"), dict) else {}
            value = result.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            aggregate[f"avg_{key}"] = sum(values) / len(values)
            aggregate[f"min_{key}"] = min(values)
            aggregate[f"max_{key}"] = max(values)
    return aggregate


# =============================================================================
# Answer checks
# =============================================================================

# Accept plain numbered headings and common Markdown variants, e.g.
#   1. Conclusion
#   1. **Conclusion**
#   ### 1. Conclusion
# Section titles are derived from the active domain answer contract.
def _section_key(title: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return key or "section"


def _section_title_regex(title: str) -> str:
    # Escape the literal title but allow flexible whitespace and slash spacing.
    parts = []
    for ch in title.strip():
        if ch.isspace():
            parts.append(r"\s+")
        elif ch == "/":
            parts.append(r"\s*/\s*")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def required_section_patterns() -> list[tuple[str, str]]:
    try:
        sections = rc.answer_sections()
    except Exception:
        sections = [
            "Conclusion",
            "Supported facts",
            "Inferences",
            "Implementation implications",
            "Unknowns / verification needed",
            "Source mapping",
        ]

    patterns = []
    for idx, title in enumerate(sections, start=1):
        title_re = _section_title_regex(title)
        patterns.append((
            _section_key(title),
            rf"(?im)^\s*(?:#{{1,6}}\s*)?(?:\*\*\s*)?{idx}[.)]\s*(?:\*\*\s*)?{title_re}\b(?:\s*\*\*)?",
        ))
    return patterns

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
    # Short forms.
    r"(?i)\bnot specified\b",
    r"(?i)\bnot provided\b",
    r"(?i)\bnot stated\b",
    r"(?i)\bnot defined\b",
    r"(?i)\bnot enough evidence\b",
    r"(?i)\binsufficient evidence\b",
    r"(?i)\bnot explicitly\b",
    r"(?i)\bcannot determine\b",
    r"(?i)\bunknown\b",
    r"(?i)\bno retrieved evidence\b",
    r"(?i)\bno direct evidence\b",
    r"(?i)\bno evidence\b",
    r"(?i)\b(?:does|do|did)\s+not\s+(?:specify|provide|state|define|contain|document)\b",
    r"(?i)\b(?:is|are|was|were)\s+not\s+(?:specified|provided|stated|defined|documented)\b",

    # Natural Qwen phrasings observed in no-answer cases.
    r"(?i)\bdoes not provide\b",
    r"(?i)\bdoes not specify\b",
    r"(?i)\bdoes not state\b",
    r"(?i)\bdoes not define\b",
    r"(?i)\bdoes not contain\b",
    r"(?i)\bno exact\b",
    r"(?i)\bno explicit\b",
    r"(?i)\bnor does it provide\b",
]

CLASSIFICATION_METADATA_LEAK_PATTERNS = [
    r"\breason_short\b",
]


def classification_metadata_leak_patterns() -> list[str]:
    patterns = list(CLASSIFICATION_METADATA_LEAK_PATTERNS)
    dynamic_names = set()
    try:
        for logical_name, cfg in (rc.metadata_fields_config() or {}).items():
            for key in ("payload", "prompt_label"):
                value = cfg.get(key) if isinstance(cfg, dict) else None
                if value:
                    dynamic_names.add(str(value))

        # Include document-level mapped payload names as classification metadata
        # too. These are useful inside Qdrant payload/debug output, but final
        # answers should cite source text rather than expose fields such as
        # document_business_impact or document_evidence_decision.
        for value in getattr(rc.DOMAIN, "metadata_field_map", {}).values():
            if value:
                dynamic_names.add(str(value))

        dynamic_names.update(rc.boolean_flag_fields())
    except Exception:
        pass

    for name in sorted(dynamic_names):
        if name:
            patterns.append(rf"\b{re.escape(name)}\b")
    return patterns


def normalize_answer_for_contract_checks(text: str) -> str:
    """Normalize lightweight Markdown that should not affect answer-contract checks.

    Qwen sometimes emits emphasis inside phrases that the eval contract checks,
    for example ``does **not** specify``. The contract should judge the phrase,
    not the Markdown decoration.
    """
    normalized = text or ""
    normalized = re.sub(r"[`*_~]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def collect_regex_hits(text: str, patterns: list[str]):
    hits = []
    for pattern in patterns:
        m = re.search(pattern, text or "")
        if m:
            hits.append({"pattern": pattern, "match": m.group(0)[:180], "pos": m.start()})
    return hits


def failure_category(failure: str) -> str:
    """Stable, coarse failure taxonomy for comparing noisy local LLM runs.

    The raw failure string stays human-readable and case-specific. This category
    is for aggregation only, so regex/contract risk is not mixed with transport,
    formatting, citation, or judge-infrastructure failures.
    """
    text = (failure or "").lower()
    if "regex overclaim risk" in text or "forbidden answer pattern matched" in text:
        return "regex_overclaim_risk"
    if text.startswith("missing required section"):
        return "format_missing_section"
    if "too few source citations" in text or "no source citations" in text:
        return "citation_contract"
    if "generation truncated" in text or "generation loop detected" in text or "answer too short" in text:
        return "generation_shape"
    if "insufficient-evidence" in text or "uncertainty/abstention" in text:
        return "abstention_contract"
    if "visible reasoning leak" in text or "<think>" in text:
        return "reasoning_leak"
    if "classification metadata leaked" in text:
        return "metadata_leak"
    if "full local path leaked" in text:
        return "path_leak"
    if "raw /rag command" in text:
        return "command_leak"
    if "llm judge" in text and "parse" in text:
        return "judge_parse_failure"
    if "llm judge" in text:
        return "judge_failure"
    if text.startswith("missing required answer pattern"):
        return "required_pattern_missing"
    return "other"


def failure_category_counts(failures: list[str]) -> dict[str, int]:
    return dict(Counter(failure_category(f) for f in failures or []))


def citation_ids(answer: str):
    """Extract source citation ids from bracketed RAG citations.

    Canonical output is [S1], [S2], ... . By default the eval accepts common
    model drift such as [S1, chunk 11] because it still explicitly cites a
    retrieved source id. Set RAG_ANSWER_EVAL_STRICT_CITATIONS=1 for CI or
    strict contract review to accept only exact [S#] citations.
    """
    if STRICT_CITATION_FORMAT:
        return [int(x) for x in re.findall(r"\[S(\d+)\]", answer or "")]

    ids: list[int] = []
    for bracket_body in re.findall(r"\[([^\]]+)\]", answer or ""):
        for value in re.findall(r"\bS(\d+)\b", bracket_body):
            ids.append(int(value))
    return ids


def check_required_sections(answer: str):
    failures = []
    present = {}
    for name, pattern in required_section_patterns():
        ok = bool(re.search(pattern, answer or ""))
        present[name] = ok
        if not ok:
            failures.append(f"missing required section: {name}")
    return present, failures



def repeated_line_hits(text: str, min_repeats: int = 4):
    """Detect obvious generation loops without judging answer semantics."""
    hits = []
    previous = None
    count = 0
    for line_no, raw_line in enumerate((text or "").splitlines(), start=1):
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue
        if len(line) < 12:
            previous = line
            count = 1
            continue
        if line == previous:
            count += 1
            if count == min_repeats:
                hits.append({"line": line[:180], "line_no": line_no, "repeats": count})
        else:
            previous = line
            count = 1
    return hits


def is_format_or_generation_failure(failures: list[str], finish_reason: str | None) -> bool:
    """Return True for failures that are safe to retry as formatting/generation repair.

    Do not retry semantic failures such as forbidden answer patterns, metadata/path
    leaks, or visible reasoning leaks. Those should stay hard failures.
    """
    if not failures:
        return False
    safe_prefixes = (
        "missing required section:",
        "too few source citations:",
        "no source citations found",
        "generation loop detected",
        "generation truncated:",
        "answer too short:",
    )
    unsafe_markers = (
        "forbidden answer pattern matched",
        "regex overclaim risk",
        "classification metadata leaked",
        "full local path leaked",
        "visible reasoning leak",
        "raw /rag command leaked",
        "unclosed <think>",
    )
    if any(any(marker in failure for marker in unsafe_markers) for failure in failures):
        return False
    if finish_reason == "length":
        return True
    return all(any(failure.startswith(prefix) for prefix in safe_prefixes) for failure in failures)


def build_repair_prompt(original_prompt: str, previous_answer: str, failures: list[str]) -> str:
    failure_text = "\n".join(f"- {failure}" for failure in failures[:12])
    previous_preview = (previous_answer or "")[:5000]

    sections = rc.render_answer_format() if hasattr(rc, "render_answer_format") else (
        "1. Conclusion\n"
        "2. Supported facts\n"
        "3. Inferences\n"
        "4. Implementation implications\n"
        "5. Unknowns / verification needed\n"
        "6. Source mapping"
    )
    citation_rule = rc.answer_citation_rule() if hasattr(rc, "answer_citation_rule") else (
        "Citation rule: use only exact citations like [S1], [S2]. "
        "Do not put chunk ids, file names, commas, or extra text inside citation brackets. "
        "No [S#] citation means no claim."
    )
    repair_rules = rc.answer_repair_rules() if hasattr(rc, "answer_repair_rules") else [
        "Use the same retrieved context from the original prompt.",
        "Use exact source citations like [S1], [S2]. Do not put chunk ids or extra text inside citation brackets.",
        "No [S#] citation means no claim.",
        "For missing evidence, cite the retrieved sources reviewed and state what is not specified.",
        "Do not reproduce malformed tables, orphaned table captions, or repeated list items from context.",
        "Do not repeat the same phrase or bullet.",
        "Keep the answer concise.",
    ]
    repair_rules_text = "\n".join(f"- {rule}" for rule in repair_rules)

    return f"""Your previous answer failed the RAG answer-contract check.

Failures:
{failure_text}

Rewrite the answer using the same retrieved context from the original prompt.

Required answer format:
{sections}

{citation_rule}

Repair rules:
{repair_rules_text}

Original prompt:
{original_prompt}

Previous malformed answer preview:
{previous_preview}
"""

def run_answer_checks(answer: str, case: dict, cleanup: dict[str, Any], finish_reason: str | None = None):
    failures = []
    warnings = []

    if cleanup.get("unclosed_think_after_parse"):
        failures.append("unclosed <think> remained after Qwen final-answer parsing")

    if not answer.strip():
        failures.append("empty assistant content")
        return {"passed": False, "failures": failures, "warnings": warnings, "metrics": {}, "details": {}}

    contract_answer = normalize_answer_for_contract_checks(answer)

    min_chars = int(case.get("min_answer_chars") or MIN_ANSWER_CHARS)
    if len(answer) < min_chars:
        failures.append(f"answer too short: chars={len(answer)} min={min_chars}")

    if finish_reason == "length":
        failures.append("generation truncated: finish_reason=length")

    loop_hits = repeated_line_hits(answer)
    if loop_hits:
        failures.append(f"generation loop detected: {loop_hits[:3]}")

    sections, section_failures = check_required_sections(answer)
    failures.extend(section_failures)

    citations = citation_ids(answer)
    unique_citations = sorted(set(citations))
    min_citations = int(case["min_citation_count"]) if "min_citation_count" in case else MIN_CITATION_COUNT

    if len(citations) < min_citations:
        failures.append(f"too few source citations: citations={len(citations)} min={min_citations}")
    if min_citations > 0 and not unique_citations:
        failures.append("no source citations found in accepted [S#] citation format")

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

    metadata_hits = collect_regex_hits(answer, classification_metadata_leak_patterns())
    if metadata_hits:
        failures.append(f"classification metadata leaked into answer: {metadata_hits[:5]}")

    answer_mode = case.get("expected_answer_mode")
    if answer_mode == "insufficient_evidence":
        insufficiency_hits = collect_regex_hits(contract_answer, INSUFFICIENT_EVIDENCE_PATTERNS)
        if not insufficiency_hits:
            failures.append(
                "expected insufficient-evidence behavior, but answer did not contain "
                "an explicit uncertainty/abstention phrase"
            )
    elif answer_mode == "limitation":
        # Limitation/boundary cases are not pure no-answer cases. They can be
        # supported by partial evidence (for example documented low-light noise)
        # while still requiring the answer not to overclaim. Case-specific
        # required_answer_patterns and forbidden_answer_patterns enforce that
        # contract without a brittle global abstention-phrase check.
        pass

    for pattern in case.get("required_answer_patterns", []):
        if not (
            re.search(pattern, answer, flags=re.I | re.M)
            or re.search(pattern, contract_answer, flags=re.I | re.M)
        ):
            failures.append(f"missing required answer pattern: {pattern}")

    forbidden_pattern_hits = []
    for pattern in case.get("forbidden_answer_patterns", []):
        m = re.search(pattern, answer, flags=re.I | re.M) or re.search(
            pattern, contract_answer, flags=re.I | re.M
        )
        if m:
            hit = {"pattern": pattern, "match": m.group(0)[:160], "pos": m.start()}
            forbidden_pattern_hits.append(hit)
            failures.append(
                f"regex overclaim risk: forbidden answer pattern matched: {pattern}; "
                f"match={m.group(0)[:160]!r}"
            )

    category_counts = failure_category_counts(failures)

    metrics = {
        "answer_chars": len(answer),
        "citation_count": len(citations),
        "unique_citation_count": len(unique_citations),
        "unique_citations": [f"S{x}" for x in unique_citations],
        "section_count": sum(1 for ok in sections.values() if ok),
        "forbidden_pattern_hit_count": len(forbidden_pattern_hits),
        "regex_overclaim_risk_count": category_counts.get("regex_overclaim_risk", 0),
        "failure_category_counts": category_counts,
    }

    details = {
        "sections": sections,
        "reasoning_hits": reasoning_hits,
        "path_hits": path_hits,
        "metadata_hits": metadata_hits,
        "loop_hits": loop_hits,
        "forbidden_pattern_hits": forbidden_pattern_hits,
        "failure_category_counts": category_counts,
        "finish_reason": finish_reason,
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
    prompt, retrieval_info = build_augmented_prompt(case)
    prompt_elapsed = time.time() - started

    prompt_warning = None
    if re.search(r"/Users/|/home/|/mnt/|/tmp/", prompt):
        prompt_warning = "prompt contains full local path"

    print(
        f"Prompt built: chars={len(prompt)} elapsed={prompt_elapsed:.2f}s "
        f"context_chars={retrieval_info.get('context_chars')} "
        f"selected={retrieval_info.get('selected_count')} "
        f"sources={retrieval_info.get('source_groups_count')}"
    )
    if prompt_warning:
        print(f"⚠️  {prompt_warning}")

    print(f"Answer request: sending to llama-server (timeout={ANSWER_TIMEOUT}s)", flush=True)
    response = call_chat(prompt, label=f"CASE {ordinal} answer")
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

    checks = run_answer_checks(answer, case, cleanup, finish_reason=response["finish_reason"])

    repair_attempts = []
    if (
        REPAIR_ON_CONTRACT_FAILURE
        and MAX_REPAIR_ATTEMPTS > 0
        and not checks["passed"]
        and is_format_or_generation_failure(checks["failures"], response["finish_reason"])
    ):
        for repair_i in range(1, MAX_REPAIR_ATTEMPTS + 1):
            print(f"Repair attempt {repair_i}: format/generation-only failure detected")
            repair_prompt = build_repair_prompt(prompt, answer, checks["failures"])
            repair_response = call_chat(repair_prompt, label=f"CASE {ordinal} repair {repair_i}")
            repair_answer = repair_response["answer"]
            repair_cleanup = repair_response["answer_cleanup"]
            repair_checks = run_answer_checks(
                repair_answer,
                case,
                repair_cleanup,
                finish_reason=repair_response["finish_reason"],
            )
            repair_attempts.append({
                "attempt": repair_i,
                "finish_reason": repair_response["finish_reason"],
                "raw_answer_chars": len(repair_response["raw_answer"]),
                "answer_chars": len(repair_answer),
                "passed": repair_checks["passed"],
                "failures": repair_checks["failures"],
            })
            if repair_checks["passed"]:
                raw_answer = repair_response["raw_answer"]
                answer = repair_answer
                cleanup = repair_cleanup
                response = repair_response
                checks = repair_checks
                checks["warnings"].append("answer repaired after first-shot format/generation failure")
                print(f"Repair attempt {repair_i}: PASS")
                break
            print(f"Repair attempt {repair_i}: still failing")

    llm_judge = None
    if LLM_JUDGE_ENABLED:
        print(f"LLM judge: enabled; sending to llama-server (timeout={JUDGE_TIMEOUT}s)", flush=True)
        llm_judge = call_llm_judge(prompt, answer, case, checks, label=f"CASE {ordinal} LLM judge")
        add_llm_judge_metrics(checks, llm_judge)
        if llm_judge.get("ok"):
            judge_result = llm_judge.get("result", {})
            print(
                "LLM judge: "
                f"pass={llm_judge.get('passed')} "
                f"overall={judge_result.get('overall_score')} "
                f"groundedness={judge_result.get('groundedness_score')} "
                f"citations={judge_result.get('citation_quality_score')} "
                f"contradiction_risk={judge_result.get('contradiction_risk_score')} "
                f"elapsed={llm_judge.get('elapsed_sec'):.2f}s"
            )
        else:
            print(
                f"LLM judge call failed: type={llm_judge.get('failure_type')} "
                f"error={llm_judge.get('error')}"
            )

    if prompt_warning:
        checks["warnings"].append(prompt_warning)

    case_dir = run_dir / f"{ordinal:02d}_{safe_file_name(case_id)}"
    case_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_PROMPTS:
        write_text(case_dir / "prompt.txt", prompt)
    write_json(case_dir / "retrieval_metrics.json", retrieval_info)
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
        "retrieval_metrics": retrieval_info,
        "prompt_chars": len(prompt),
        "prompt_build_elapsed_sec": prompt_elapsed,
        "answer_elapsed_sec": response["elapsed_sec"],
        "finish_reason": response["finish_reason"],
        "reasoning_content_chars": response["reasoning_content_chars"],
        "answer_cleanup": cleanup,
        "repair_attempts": repair_attempts,
        "usage": response["usage"],
        "llm_judge": llm_judge,
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


def aggregate_answer_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}

    total = len(results)

    def avg(path: tuple[str, ...]) -> float | None:
        vals = []
        for result in results:
            cur: Any = result
            for key in path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(key)
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return (sum(vals) / len(vals)) if vals else None

    finish_reasons = Counter(str(r.get("finish_reason")) for r in results)
    failure_kinds: Counter[str] = Counter()
    failure_categories: Counter[str] = Counter()
    warning_kinds: Counter[str] = Counter()
    regex_overclaim_risk_cases = 0
    judge_parse_failed_cases = 0
    judge_semantic_failed_cases = 0
    for result in results:
        case_categories = Counter()
        for failure in result.get("failures", []) or []:
            failure_kinds[failure.split(":", 1)[0]] += 1
            category = failure_category(failure)
            failure_categories[category] += 1
            case_categories[category] += 1
        if case_categories.get("regex_overclaim_risk", 0) > 0:
            regex_overclaim_risk_cases += 1
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        if metrics.get("llm_judge_parse_failed"):
            judge_parse_failed_cases += 1
        if metrics.get("llm_judge_semantic_failed"):
            judge_semantic_failed_cases += 1
        for warning in result.get("warnings", []) or []:
            warning_kinds[warning.split(":", 1)[0]] += 1

    usage_totals: Counter[str] = Counter()
    for result in results:
        usage = result.get("usage") or {}
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    usage_totals[key] += value

    repaired = 0
    repair_attempted = 0
    for result in results:
        attempts = result.get("repair_attempts") or []
        if attempts:
            repair_attempted += 1
            if result.get("passed") and any(a.get("passed") for a in attempts):
                repaired += 1

    return {
        "cases_total": total,
        "passed": sum(1 for r in results if r.get("passed")),
        "failed": sum(1 for r in results if not r.get("passed")),
        "pass_rate": sum(1 for r in results if r.get("passed")) / total,
        "answer_chars_avg": avg(("metrics", "answer_chars")),
        "citation_count_avg": avg(("metrics", "citation_count")),
        "unique_citation_count_avg": avg(("metrics", "unique_citation_count")),
        "section_count_avg": avg(("metrics", "section_count")),
        "prompt_chars_avg": avg(("prompt_chars",)),
        "prompt_build_elapsed_sec_avg": avg(("prompt_build_elapsed_sec",)),
        "answer_elapsed_sec_avg": avg(("answer_elapsed_sec",)),
        "reasoning_content_chars_avg": avg(("reasoning_content_chars",)),
        "retrieval_context_chars_avg": avg(("retrieval_metrics", "context_chars")),
        "retrieval_selected_count_avg": avg(("retrieval_metrics", "selected_count")),
        "retrieval_source_groups_count_avg": avg(("retrieval_metrics", "source_groups_count")),
        "finish_reasons": dict(finish_reasons),
        "failure_kinds": dict(failure_kinds),
        "failure_category_counts": dict(failure_categories),
        "regex_overclaim_risk_cases": regex_overclaim_risk_cases,
        "judge_parse_failed_cases": judge_parse_failed_cases,
        "judge_semantic_failed_cases": judge_semantic_failed_cases,
        "warning_kinds": dict(warning_kinds),
        "repair_attempted_cases": repair_attempted,
        "repaired_cases": repaired,
        "usage_totals": dict(usage_totals),
    }




def read_case_file(path: str | os.PathLike[str]) -> list[str]:
    """Read newline-delimited case ids for expensive answer/judge reruns."""
    case_path = Path(path).expanduser()
    if not case_path.exists():
        raise SystemExit(f"Case file not found: {case_path}")
    out: list[str] = []
    for raw in case_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def selected_case_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    ids.extend(args.case_id or [])
    for path in args.case_file or []:
        ids.extend(read_case_file(path))
    seen: set[str] = set()
    out: list[str] = []
    for case_id in ids:
        if case_id not in seen:
            seen.add(case_id)
            out.append(case_id)
    return out

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run answer contract / grounding hygiene eval for local RAG.",
        epilog="Use repeated --case or --case-file to make LLM judge loops cheap after retrieval narrows the shortlist.",
    )
    parser.add_argument(
        "eval_file",
        nargs="?",
        default=str(DEFAULT_EVAL_FILE),
        help=f"Path to eval queries JSON. Default: {DEFAULT_EVAL_FILE}",
    )
    parser.add_argument("--case", dest="case_id", action="append", help="Run one eval case id. Repeat for multiple cases.")
    parser.add_argument("--case-file", action="append", help="Newline-delimited case ids; # comments and blanks are ignored.")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases after filtering. 0 means all.")
    return parser.parse_args()


def main():
    args = parse_args()
    eval_file = Path(args.eval_file).expanduser()

    cases = load_eval_cases(eval_file)
    case_ids = selected_case_ids(args)
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.get("id") in wanted]
        found = {str(c.get("id")) for c in cases}
        missing = [case_id for case_id in case_ids if case_id not in found]
        if missing:
            raise SystemExit(f"No eval case(s) found: {missing}")
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
    print("Answer generation params: omitted; llama-server launch controls max_tokens/temp/top_p/top_k/min_p/reasoning")
    print(f"Answer timeout: {ANSWER_TIMEOUT}")
    print(f"Repair on contract failure: {REPAIR_ON_CONTRACT_FAILURE}; attempts={MAX_REPAIR_ATTEMPTS}")
    print(f"LLM judge enabled: {LLM_JUDGE_ENABLED}; gate={LLM_JUDGE_GATE}; url={JUDGE_CHAT_URL}")
    print(f"Progress heartbeat interval: {PROGRESS_INTERVAL_SEC}s")
    if LLM_JUDGE_ENABLED:
        print("Judge generation params: omitted; same llama-server launch policy as answer generation")
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
        append_jsonl(run_dir / "cases.jsonl", result)

    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed
    failed_ids = [r["id"] for r in results if not r.get("passed")]

    aggregate = aggregate_answer_metrics(results)

    summary = {
        "run_id": run_id,
        "eval_file": str(eval_file),
        "run_dir": str(run_dir),
        "case_filter": case_ids,
        "chat_url": CHAT_URL,
        "qwen_answer_parsing": "default server response parsed locally; no response-routing override sent",
        "llm_judge_enabled": LLM_JUDGE_ENABLED,
        "llm_judge_gate": LLM_JUDGE_GATE,
        "generation_settings": {
            "eval_sent_generation_params": [],
            "source_of_truth": "llama-server launch",
            "server_launch_assumption": "eval_answer.py omits max_tokens/temp/top_p/top_k/min_p/reasoning; user's launch uses --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --reasoning on --reasoning-budget -1 --n-predict 16384",
        },
        "llm_judge_aggregate": aggregate_llm_judge(results),
        "cases_total": len(results),
        "passed": passed,
        "failed": failed,
        "failed_ids": failed_ids,
        "aggregate_metrics": aggregate,
        "scope": "answer contract / grounding hygiene plus optional LLM judge; not a substitute for human review",
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
    if LLM_JUDGE_ENABLED:
        judge_agg = summary["llm_judge_aggregate"]
        print(
            "LLM judge: "
            f"ok={judge_agg.get('ok_cases')}/{judge_agg.get('judged_cases')} "
            f"passed={judge_agg.get('judge_passed_cases')} "
            f"avg_overall={judge_agg.get('avg_overall_score')}"
        )
    print("Aggregate metrics:")
    for key, value in aggregate.items():
        if key in {"failure_kinds", "warning_kinds"}:
            continue
        print(f"  {key}: {value}")
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
