#!/usr/bin/env python3
"""
Reverse proxy RAG UX for the stock llama.cpp browser chat.

Why this exists:
- We want to use the nice llama-server browser UI at /#/ without patching
  llama.cpp.
- The proxy forwards everything to llama-server unchanged except explicit
  `/rag` chat messages.
- `/rag` messages are rewritten into an evidence-only RAG prompt using the
  shared retrieval/context contract from rag_core.py.

Architecture:
  Browser UI -> rag_proxy.py (:8088) -> llama-server chat (:8080)
                                -> embedding server (:8081)
                                -> Qdrant (:6333)

User command:
  /rag 5
  What are the likely warning-related and safety-relevant implications ...

Behavior:
  - proxy all HTTP paths/methods to llama-server
  - preserve streaming responses
  - inspect JSON chat requests
  - if last user message starts with /rag:
      parse top_k and question
      call rag_core.build_augmented_prompt()
      replace only that user message content
      forward patched JSON to llama-server

Security / privacy:
  - Bind to 127.0.0.1 by default.
  - Do not expose this proxy publicly.
  - Logs may contain user questions and request previews.
  - Full local source paths are not placed in the LLM prompt; source provenance
    inside the prompt is limited to source id, file name, and chunk ids.

Important semantics:
  - Current runtime uses the proven dense retrieval baseline.
  - Hybrid lexical/RRF is intentionally deferred because the naive version hurt
    broad semantic queries.
  - Classification metadata is used only by retrieval/reranking diagnostics, not
    as answer evidence.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from starlette.background import BackgroundTask
from starlette.responses import Response, StreamingResponse

import rag_core as rc

# =============================================================================
# Proxy config
# =============================================================================

LLAMA_URL = os.environ.get("RAG_LLAMA_URL", "http://127.0.0.1:8080").rstrip("/")
PROXY_HOST = os.environ.get("RAG_PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("RAG_PROXY_PORT", "8088"))

# Default: active rewrite. Use RAG_PROXY_DRY_RUN=1 to only detect/log /rag.
DRY_RUN = os.environ.get("RAG_PROXY_DRY_RUN", "0") == "1"

RAG_COMMAND = os.environ.get("RAG_COMMAND", "/rag")
DEFAULT_RAG_K = int(os.environ.get("RAG_DEFAULT_K", str(rc.DEFAULT_TOP_K)))
LOG_BODY_CHARS = int(os.environ.get("RAG_PROXY_LOG_BODY_CHARS", "500"))

CONNECT_TIMEOUT = float(os.environ.get("RAG_PROXY_CONNECT_TIMEOUT", "10"))
WRITE_TIMEOUT = float(os.environ.get("RAG_PROXY_WRITE_TIMEOUT", "60"))
READ_TIMEOUT = os.environ.get("RAG_PROXY_READ_TIMEOUT", "none")
POOL_TIMEOUT = float(os.environ.get("RAG_PROXY_POOL_TIMEOUT", "10"))

if READ_TIMEOUT.lower() in {"none", "null", "0", "-1"}:
    READ_TIMEOUT_VALUE = None
else:
    READ_TIMEOUT_VALUE = float(READ_TIMEOUT)


# =============================================================================
# HTTP proxy header rules
# =============================================================================

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

STRIP_REQUEST_HEADERS = HOP_BY_HOP_HEADERS | {"accept-encoding"}
STRIP_RESPONSE_HEADERS = HOP_BY_HOP_HEADERS | {"content-length"}


# =============================================================================
# Logging helpers
# =============================================================================

def now_ms() -> int:
    return int(time.time() * 1000)


def short(text: str, n: int = LOG_BODY_CHARS) -> str:
    text = text or ""
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= n:
        return text
    return text[:n] + f"...[truncated {len(text) - n} chars]"


def log(request_id: str, msg: str) -> None:
    print(f"[rag-proxy {request_id}] {msg}", flush=True)


def safe_decode_body(body: bytes) -> str:
    try:
        return body.decode("utf-8", errors="replace")
    except Exception:
        return repr(body)


# =============================================================================
# /rag parser and chat payload helpers
# =============================================================================

def parse_rag_command(text: str):
    """
    Supports both preferred newline format and one-line fallback:
      /rag 5\nquestion
      /rag\nquestion
      /rag 5 question
    """
    if not isinstance(text, str):
        return None

    command = re.escape(RAG_COMMAND)
    pattern = rf"^\s*{command}(?:\s+(\d+))?(?:\s*\n|\s+)([\s\S]*?)\s*$"
    m = re.match(pattern, text, flags=re.IGNORECASE)

    if not m:
        bare = re.match(rf"^\s*{command}(?:\s+(\d+))?\s*$", text, flags=re.IGNORECASE)
        if not bare:
            return None
        return {
            "top_k": int(bare.group(1) or DEFAULT_RAG_K),
            "question": "",
            "raw": text,
        }

    return {
        "top_k": int(m.group(1) or DEFAULT_RAG_K),
        "question": m.group(2).strip(),
        "raw": text,
    }


def find_last_user_message(messages: list[dict[str, Any]]):
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return idx, msg
    return None, None


def extract_text_from_message_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "\n".join(parts)
    return None


def replace_message_content(message: dict[str, Any], new_text: str):
    """Preserve OpenAI-compatible content shape where possible."""
    content = message.get("content")

    if isinstance(content, str):
        message["content"] = new_text
        return

    if isinstance(content, list):
        replaced = False
        new_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and not replaced:
                new_part = dict(part)
                new_part["text"] = new_text
                new_parts.append(new_part)
                replaced = True
            else:
                new_parts.append(part)
        if not replaced:
            new_parts.insert(0, {"type": "text", "text": new_text})
        message["content"] = new_parts
        return

    message["content"] = new_text


def detect_rag_in_json(data: Any):
    """Return detection dict for common OpenAI/llama.cpp request shapes."""
    if not isinstance(data, dict):
        return None

    messages = data.get("messages")
    if isinstance(messages, list):
        idx, msg = find_last_user_message(messages)
        if msg is not None:
            content_text = extract_text_from_message_content(msg.get("content"))
            parsed = parse_rag_command(content_text or "")
            if parsed:
                return {
                    "shape": "messages",
                    "message_index": idx,
                    "top_k": parsed["top_k"],
                    "question": parsed["question"],
                    "roles": [m.get("role", "?") if isinstance(m, dict) else "?" for m in messages],
                }

    prompt = data.get("prompt")
    if isinstance(prompt, str):
        parsed = parse_rag_command(prompt)
        if parsed:
            return {"shape": "prompt", "message_index": None, "top_k": parsed["top_k"], "question": parsed["question"], "roles": []}

    content = data.get("content")
    if isinstance(content, str):
        parsed = parse_rag_command(content)
        if parsed:
            return {"shape": "content", "message_index": None, "top_k": parsed["top_k"], "question": parsed["question"], "roles": []}

    return None


# =============================================================================
# JSON request patching
# =============================================================================

async def maybe_patch_json_body(request_id: str, path: str, data: Any):
    detection = detect_rag_in_json(data)
    if not detection:
        return data, False

    top_k = detection["top_k"]
    question = detection["question"]

    log(
        request_id,
        "RAG_DETECTED "
        f"path=/{path} "
        f"shape={detection['shape']} "
        f"message_index={detection['message_index']} "
        f"top_k={top_k} "
        f"question_preview={short(question, 300)!r} "
        f"roles={detection.get('roles', [])}",
    )

    if not question.strip():
        raise ValueError("RAG command detected, but question is empty")

    if DRY_RUN:
        log(request_id, "RAG_DRY_RUN active: request body is NOT modified")
        return data, True

    # Copy request JSON without mutating the object retained by FastAPI/debug code.
    patched = json.loads(json.dumps(data, ensure_ascii=False))

    augmented, info = await asyncio.to_thread(
        rc.build_augmented_prompt,
        question,
        top_k,
        rc.DEFAULT_PRE_K,
        rc.DEFAULT_MAX_PER_FILE,
        rc.DEFAULT_NEIGHBOR_RADIUS,
        rc.SELECTED_MAX_CHARS,
        rc.NEIGHBOR_SNIPPET_CHARS,
        rc.CONTEXT_MAX_CHARS,
    )

    log(
        request_id,
        "RAG_RETRIEVE "
        f"top_k={info['top_k']} "
        f"pre_k={info['pre_k']} "
        f"selected={info['selected_count']} "
        f"source_groups={info['source_groups_count']} "
        f"context_chars={info['context_chars']} "
        f"prompt_chars={info['prompt_chars']} "
        f"candidates={info['candidates_count']}",
    )

    for idx, group in enumerate(info["source_groups"], start=1):
        log(
            request_id,
            "RAG_SOURCE "
            f"S{idx} file={group.get('file_name')} "
            f"selected={group.get('selected_indices')} "
            f"expanded={group.get('expanded_indices')} "
            f"best_final={rc.fmt_score(group.get('best_final_score'))}",
        )

    shape = detection["shape"]
    if shape == "messages":
        idx = detection["message_index"]
        replace_message_content(patched["messages"][idx], augmented)
    elif shape == "prompt":
        patched["prompt"] = augmented
    elif shape == "content":
        patched["content"] = augmented
    else:
        raise ValueError(f"Unsupported RAG detection shape: {shape}")

    log(
        request_id,
        "RAG_REWRITE "
        f"old_question_chars={len(question)} "
        f"new_prompt_chars={len(augmented)} "
        "full_paths_in_prompt=False "
        "classification_metadata_in_prompt=False",
    )

    return patched, True


# =============================================================================
# HTTP proxy helpers
# =============================================================================

def filter_request_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in STRIP_REQUEST_HEADERS}


def filter_response_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in STRIP_RESPONSE_HEADERS}


def should_try_json(content_type: str, body: bytes) -> bool:
    if not body:
        return False
    ct = (content_type or "").lower()
    if "application/json" in ct:
        return True
    stripped = body.lstrip()
    return stripped.startswith(b"{") or stripped.startswith(b"[")


def target_url(path: str, query_string: bytes) -> str:
    url = f"{LLAMA_URL}/{path}"
    if query_string:
        url += "?" + query_string.decode("utf-8", errors="replace")
    return url


# =============================================================================
# FastAPI app
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT_VALUE,
        write=WRITE_TIMEOUT,
        pool=POOL_TIMEOUT,
    )
    app.state.client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)

    cfg = rc.retrieval_config_summary()
    print("=" * 100)
    print("RAG PROXY START")
    print("=" * 100)
    print(f"Proxy URL:        http://{PROXY_HOST}:{PROXY_PORT}")
    print(f"Upstream llama:   {LLAMA_URL}")
    print(f"RAG command:      {RAG_COMMAND}")
    print(f"Default RAG k:    {DEFAULT_RAG_K}")
    print(f"Dry run:          {DRY_RUN}")
    print(f"Retrieval mode:   {cfg['mode']} (hybrid {cfg['hybrid']})")
    print(f"Embedding URL:    {cfg['embed_url']}")
    print(f"Qdrant URL:       {cfg['qdrant_url']}")
    print(f"Collection:       {cfg['collection']}")
    print(f"RAG pre_k:        {cfg['pre_k']}")
    print(f"RAG max_per_file: {cfg['max_per_file']}")
    print(f"Neighbor radius:  {cfg['neighbor_radius']}")
    print(f"Context max chars:{cfg['context_max_chars']}")
    print(f"Metadata clamp:   {cfg['metadata_rerank_clamp']}")
    print("Prompt policy:    full_paths=False; classification_metadata=False")
    print(f"Read timeout:     {READ_TIMEOUT_VALUE}")
    print("=" * 100, flush=True)

    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_all(path: str, request: Request):
    request_id = uuid.uuid4().hex[:8]
    t0 = now_ms()

    method = request.method
    url = target_url(path, request.scope.get("query_string", b""))
    body = await request.body()
    content_type = request.headers.get("content-type", "")

    log(request_id, f"IN {method} /{path} query={request.url.query!r} ct={content_type!r} body_bytes={len(body)}")

    outbound_body = body
    body_was_patched = False
    rag_detected = False

    if should_try_json(content_type, body):
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
            patched_data, rag_detected = await maybe_patch_json_body(request_id, path, data)
            if rag_detected and not DRY_RUN:
                outbound_body = json.dumps(patched_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                body_was_patched = True
        except Exception as e:
            log(request_id, f"RAG_OR_JSON_PATCH_FAILED error={type(e).__name__}: {e}")
            return Response(
                content=(
                    "rag_proxy RAG preprocessing error\n"
                    f"path=/{path}\n"
                    f"error={type(e).__name__}: {e}\n"
                ),
                status_code=500,
                media_type="text/plain",
            )
    elif body and len(body) < 64_000:
        parsed = parse_rag_command(safe_decode_body(body))
        if parsed:
            rag_detected = True
            log(
                request_id,
                "RAG_DETECTED_NON_JSON "
                f"path=/{path} top_k={parsed['top_k']} "
                f"question_preview={short(parsed['question'], 300)!r}",
            )

    if rag_detected:
        log(request_id, f"BODY_PREVIEW {short(safe_decode_body(body))!r}")

    upstream_headers = filter_request_headers(request.headers)
    if body_was_patched:
        upstream_headers["content-type"] = "application/json"

    client: httpx.AsyncClient = request.app.state.client

    try:
        upstream_request = client.build_request(method=method, url=url, headers=upstream_headers, content=outbound_body)
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as e:
        elapsed = now_ms() - t0
        log(request_id, f"UPSTREAM_ERROR after_ms={elapsed} error={type(e).__name__}: {e}")
        return Response(
            content=("rag_proxy upstream error\n" f"target={url}\n" f"error={type(e).__name__}: {e}\n"),
            status_code=502,
            media_type="text/plain",
        )

    response_headers = filter_response_headers(upstream_response.headers)
    elapsed = now_ms() - t0
    log(
        request_id,
        f"OUT status={upstream_response.status_code} after_ms={elapsed} "
        f"upstream_ct={upstream_response.headers.get('content-type')!r} streaming=True",
    )

    async def response_stream():
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        except asyncio.CancelledError:
            log(request_id, "STREAM_CANCELLED client disconnected")
            raise
        except Exception as e:
            log(request_id, f"STREAM_ERROR {type(e).__name__}: {e}")
            raise

    return StreamingResponse(
        response_stream(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream_response.aclose),
    )


def main():
    try:
        uvicorn.run(
            app,
            host=PROXY_HOST,
            port=PROXY_PORT,
            log_level=os.environ.get("RAG_PROXY_UVICORN_LOG_LEVEL", "warning"),
            access_log=False,
        )
    except KeyboardInterrupt:
        print("\nRAG proxy stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
