#!/usr/bin/env python3
"""
Debug CLI for asking the local Qwen/llama-server with the same RAG prompt used by
rag_proxy.py.

Primary UX:
    Open the llama.cpp browser UI through rag_proxy.py and type:

        /rag 5
        question...

Why this file still exists:
    This is a thin debug wrapper for terminal use when the browser UI is not
    convenient. It delegates retrieval, neighbor expansion, and context packing
    to rag_core.py. It must not maintain a separate retrieval implementation.

What this script does:
    1. Call rag_core.build_augmented_prompt(question, top_k, ...)
    2. Print the same source-group summary used by the proxy/search path
    3. Send a plain non-streaming chat request to llama-server
    4. Parse the default Qwen final answer locally and print it

What this script does NOT do:
    - no SQLite fallback
    - no custom cosine search
    - no duplicate metadata rerank implementation
    - no different context packing contract
    - no server-side response-routing overrides

Qwen / llama-server note:
    This CLI does not try to steer llama-server answer routing. If the default
    Qwen response contains <think>...</think> or a dangling </think>, the final
    answer is parsed with rag_core.parse_qwen_final_answer().
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests

import rag_core as rc

CHAT_URL = os.environ.get("RAG_CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_TOP_K = int(os.environ.get("RAG_ASK_TOP_K", str(rc.DEFAULT_TOP_K)))
DEFAULT_PRE_K = int(os.environ.get("RAG_ASK_PRE_K", str(rc.DEFAULT_PRE_K)))
MAX_PER_FILE = int(os.environ.get("RAG_MAX_PER_FILE", str(rc.DEFAULT_MAX_PER_FILE)))
NEIGHBOR_RADIUS = int(os.environ.get("RAG_NEIGHBOR_RADIUS", str(rc.DEFAULT_NEIGHBOR_RADIUS)))
MAX_TOKENS = int(os.environ.get("RAG_ASK_MAX_TOKENS", "10000"))
TIMEOUT = int(os.environ.get("RAG_ASK_TIMEOUT", "1800"))
TEMPERATURE = float(os.environ.get("RAG_ASK_TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("RAG_ASK_TOP_P", "0.9"))
DEBUG = os.environ.get("RAG_DEBUG", "0") == "1"


def debug_print(title: str, value: Any) -> None:
    if not DEBUG:
        return
    print("=" * 100)
    print(f"DEBUG: {title}")
    print("=" * 100)
    if isinstance(value, (dict, list)):
        print(json.dumps(value, ensure_ascii=False, indent=2)[:20000])
    else:
        print(str(value)[:20000])
    print()


def extract_answer(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        choice = data["choices"][0]
        message = choice.get("message") or {}
        raw_content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason")
    except Exception as exc:
        raise RuntimeError(
            "Unexpected chat response shape: "
            + json.dumps(data, ensure_ascii=False)[:2000]
        ) from exc

    if not isinstance(raw_content, str) or not raw_content.strip():
        raise RuntimeError("Model response did not contain non-empty message.content")

    answer, cleanup = rc.parse_qwen_final_answer(raw_content)
    if not answer.strip():
        raise RuntimeError("Parsed final answer is empty after Qwen response parsing")

    info = {
        "finish_reason": finish_reason,
        "reasoning_content_chars": len(reasoning_content),
        "raw_answer_chars": len(raw_content),
        "parsed_answer_chars": len(answer),
        "answer_cleanup": cleanup,
    }
    return answer, info


def call_chat(prompt: str) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }

    debug_print("CHAT PAYLOAD", payload)

    started = time.time()
    response = requests.post(CHAT_URL, json=payload, timeout=TIMEOUT)
    elapsed = time.time() - started

    debug_print("HTTP STATUS", response.status_code)
    debug_print("RAW RESPONSE TEXT", response.text)

    response.raise_for_status()
    data = response.json()
    debug_print("PARSED JSON", data)

    answer, info = extract_answer(data)
    info["elapsed_sec"] = elapsed
    info["usage"] = data.get("usage")
    return answer, info


def print_source_groups(debug_info: dict[str, Any]) -> None:
    groups = debug_info.get("source_groups") or []

    print("=" * 100)
    print("RETRIEVED SOURCE GROUPS")
    print("=" * 100)
    for idx, group in enumerate(groups, start=1):
        print(
            f"{idx}. file={group.get('file_name')} "
            f"selected_chunks={group.get('selected_indices')} "
            f"expanded_chunks={group.get('expanded_indices')} "
            f"best_final={rc.fmt_score(group.get('best_final_score'))}"
        )
        for item in group.get("chunks", []):
            payload = item["payload"]
            marker = "*" if item.get("is_selected_hit") else "-"
            mode = "selected" if item.get("is_selected_hit") else "neighbor"
            print(
                f"   {marker} chunk={payload.get('chunk_index')} "
                f"mode={mode} "
                f"final={rc.fmt_score(item.get('final_score'))} "
                f"dense_rank={item.get('dense_rank') if item.get('dense_rank') is not None else '-'} "
                f"role={payload.get('chunk_role')} "
                f"facets={rc.fmt_list(payload.get('content_facets'))} "
                f"safety={payload.get('safety_relevance')} "
                f"delivery={payload.get('delivery_value')} "
                f"decision={payload.get('corpus_decision')}"
            )
    print()
    print("Context summary:")
    print(f"  candidates={debug_info.get('candidates_count')}")
    print(f"  selected_chunks={debug_info.get('selected_count')}")
    print(f"  source_groups={debug_info.get('source_groups_count')}")
    print(f"  context_chars={debug_info.get('context_chars')}")
    print(f"  prompt_chars={debug_info.get('prompt_chars')}")
    print("  full_paths_in_prompt=False")
    print("  classification_metadata_in_prompt=False")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python3 ask_qwen.py "your question" [top_k] [pre_k]')
        sys.exit(1)

    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOP_K
    pre_k = int(sys.argv[3]) if len(sys.argv) > 3 else max(DEFAULT_PRE_K, top_k * 4)

    cfg = rc.retrieval_config_summary()

    print("=" * 100)
    print("RAG ASK DEBUG START")
    print("=" * 100)
    print(f"Question: {question}")
    print(f"Chat URL: {CHAT_URL}")
    print(f"Retrieval mode: {cfg['mode']} (hybrid {cfg['hybrid']})")
    print(f"Collection: {cfg['collection']}")
    print(f"Qdrant URL: {cfg['qdrant_url']}")
    print(f"Embedding URL: {cfg['embed_url']}")
    print(
        f"top_k={top_k}; pre_k={pre_k}; max_per_file={MAX_PER_FILE}; "
        f"neighbor_radius={NEIGHBOR_RADIUS}"
    )
    print("Qwen response handling: default server response parsed locally; no response-routing override is sent")
    print("This CLI is a thin debug wrapper; browser UX is rag_proxy.py.")
    print()

    prompt, debug_info = rc.build_augmented_prompt(
        question=question,
        top_k=top_k,
        pre_k=pre_k,
        max_per_file=MAX_PER_FILE,
        neighbor_radius=NEIGHBOR_RADIUS,
    )

    print_source_groups(debug_info)
    debug_print("AUGMENTED PROMPT", prompt)

    answer, chat_info = call_chat(prompt)

    print("=" * 100)
    print("ANSWER")
    print("=" * 100)
    print(answer)
    print()
    print("=" * 100)
    print("RAG ASK DEBUG DONE")
    print("=" * 100)
    print(f"finish_reason={chat_info['finish_reason']}")
    print(f"elapsed_sec={chat_info['elapsed_sec']:.2f}")
    print(f"reasoning_content_chars={chat_info['reasoning_content_chars']}")
    print(f"raw_answer_chars={chat_info['raw_answer_chars']}")
    print(f"parsed_answer_chars={chat_info['parsed_answer_chars']}")
    print(f"answer_cleanup={json.dumps(chat_info['answer_cleanup'], ensure_ascii=False)}")
    if chat_info.get("usage") is not None:
        print(f"usage={chat_info['usage']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
