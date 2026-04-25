#!/usr/bin/env python3
"""
Minimal reverse-proxy RAG UX for the stock llama.cpp browser chat.

Scope: local ADAS / embedded vision delivery RAG v1. This proxy is deliberately
narrow and does not attempt to be a general OpenAI-compatible middleware.

Why this exists
---------------
We want to keep llama.cpp / llama-server unpatched and still use its browser UI.
Open the browser through this proxy:

    http://127.0.0.1:8088/#/

The proxy forwards every request to the upstream llama-server, except for one
explicit, observed UI request shape:

    POST /v1/chat/completions
    Content-Type: application/json
    {
      "messages": [
        {"role": "user", "content": "/rag 5\nquestion..."}
      ],
      ... other llama.cpp UI fields ...
    }

Only in that exact chat/messages shape, if the last user message content is a
plain string matching the newline-only /rag command, the proxy replaces that
single user message content with a RAG-augmented prompt built by rag_core.py.
Everything else in the JSON request is preserved unchanged.

Supported /rag command contract
-------------------------------
Supported:

    /rag 5
    What are the likely warning-related implications ...

    /rag
    What are the likely warning-related implications ...

Not supported intentionally:

    /rag 5 question on one line
    prompt=... request shapes
    content=... request shapes
    non-JSON /rag bodies
    multimodal/content-parts message shapes

This is deliberate. The goal is not to invent a generic OpenAI-compatible RAG
middleware. The goal is to minimally patch the exact llama.cpp browser UI traffic
observed in this project.

What the proxy changes
----------------------
For a detected /rag request:

    data["messages"][last_user_index]["content"] = augmented_prompt

What the proxy does NOT change
------------------------------
It does not modify:

    stream
    return_progress
    backend_sampling
    timings_per_token
    temperature / sampling options
    max_tokens / n_predict
    model settings
    any non-/rag request

This matters for Qwen reasoning models: the browser UI already sends a working
llama-server payload. The proxy must not second-guess or override it.

Architecture
------------
    Browser UI -> rag_proxy.py (:8088) -> llama-server chat (:8080)
                                  -> rag_core.py retrieval/context packing
                                  -> embedding server (:8081)
                                  -> Qdrant (:6333)

Runtime semantics
-----------------
    - Retrieval mode is the dense baseline from rag_core.py.
    - Hybrid lexical/RRF is deferred because the naive version hurt broad
      semantic queries in this corpus.
    - Classification metadata may be used by rag_core for ranking/diagnostics,
      but classification metadata is not included as answer evidence.
    - Full local source paths are not placed in the LLM prompt; prompt
      provenance uses source id, file name, and chunk ids.

Security / privacy
------------------
    - Bind to 127.0.0.1 by default.
    - Do not expose this proxy publicly.
    - Logs may contain user questions and request previews.

Run
---
    RAG_LLAMA_URL=http://127.0.0.1:8080 python3 rag_proxy.py
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

LLAMA_URL = os.environ.get("RAG_LLAMA_URL", "http://127.0.0.1:8080").rstrip("/")
PROXY_HOST = os.environ.get("RAG_PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("RAG_PROXY_PORT", "8088"))

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


def parse_rag_command(text: str):
    """
    Parse only the newline command format used in the llama.cpp browser UI.

    Accepted:
        /rag 5\nquestion...
        /rag\nquestion...

    Rejected intentionally:
        /rag 5 question...
        bare /rag without a question
    """
    if not isinstance(text, str):
        return None

    command = re.escape(RAG_COMMAND)
    pattern = rf"^\s*{command}(?:\s+(\d+))?\s*\n([\s\S]+?)\s*$"
    match = re.match(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None

    question = match.group(2).strip()
    if not question:
        return None

    return {
        "top_k": int(match.group(1) or DEFAULT_RAG_K),
        "question": question,
        "raw": text,
    }


def find_last_user_message(messages: list[dict[str, Any]]):
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return idx, msg
    return None, None


def detect_llama_ui_rag_request(path: str, method: str, data: Any):
    """
    Detect only the observed llama.cpp browser UI RAG request shape.

    This intentionally ignores other OpenAI-compatible shapes so the proxy stays
    minimal and predictable.
    """
    if method.upper() != "POST":
        return None

    if path.strip("/") != "v1/chat/completions":
        return None

    if not isinstance(data, dict):
        return None

    messages = data.get("messages")
    if not isinstance(messages, list):
        return None

    idx, msg = find_last_user_message(messages)
    if msg is None:
        return None

    content = msg.get("content")
    if not isinstance(content, str):
        return None

    parsed = parse_rag_command(content)
    if not parsed:
        return None

    return {
        "message_index": idx,
        "top_k": parsed["top_k"],
        "question": parsed["question"],
        "roles": [m.get("role", "?") if isinstance(m, dict) else "?" for m in messages],
    }


def should_try_json(content_type: str, body: bytes) -> bool:
    if not body:
        return False
    ct = (content_type or "").lower()
    if "application/json" in ct:
        return True
    stripped = body.lstrip()
    return stripped.startswith(b"{") or stripped.startswith(b"[")


def filter_request_headers(headers) -> dict[str, str]:
    out = {}
    for key, value in headers.items():
        if key.lower() in STRIP_REQUEST_HEADERS:
            continue
        out[key] = value
    return out


def filter_response_headers(headers) -> dict[str, str]:
    out = {}
    for key, value in headers.items():
        if key.lower() in STRIP_RESPONSE_HEADERS:
            continue
        out[key] = value
    return out


def target_url(path: str, query_string: bytes) -> str:
    url = f"{LLAMA_URL}/{path}"
    if query_string:
        url += "?" + query_string.decode("utf-8", errors="replace")
    return url


async def patch_json_body_if_rag(request_id: str, path: str, method: str, data: Any):
    detection = detect_llama_ui_rag_request(path=path, method=method, data=data)
    if not detection:
        return data, False

    top_k = detection["top_k"]
    question = detection["question"]

    log(
        request_id,
        "RAG_DETECTED "
        f"path=/{path} "
        f"message_index={detection['message_index']} "
        f"top_k={top_k} "
        f"question_preview={short(question, 300)!r} "
        f"roles={detection.get('roles', [])}",
    )

    patched = json.loads(json.dumps(data, ensure_ascii=False))

    prompt, debug_info = await asyncio.to_thread(
        rc.build_augmented_prompt,
        question=question,
        top_k=top_k,
    )

    for idx, group in enumerate(debug_info.get("source_groups", []), start=1):
        log(
            request_id,
            "RAG_SOURCE "
            f"S{idx} file={group.get('file_name')} "
            f"selected={group.get('selected_indices')} "
            f"expanded={group.get('expanded_indices')} "
            f"best_final={rc.fmt_score(group.get('best_final_score'))}",
        )

    patched["messages"][detection["message_index"]]["content"] = prompt

    log(
        request_id,
        "RAG_REWRITE "
        f"old_question_chars={len(question)} "
        f"new_prompt_chars={len(prompt)} "
        f"context_chars={debug_info.get('context_chars')} "
        "full_paths_in_prompt=False "
        "classification_metadata_in_prompt=False",
    )

    return patched, True


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
    print(f"Patch contract:   only POST /v1/chat/completions messages[last user].content")
    print(f"Command format:   newline-only '/rag [k]\\nquestion' (one-line commands ignored)")
    print(f"Payload policy:   preserve all non-content JSON fields; do not touch sampling/stream/server options")
    print(f"Retrieval mode:   {cfg['mode']} (hybrid {cfg['hybrid']})")
    print(f"Collection:       {cfg['collection']}")
    print(f"Qdrant URL:       {cfg['qdrant_url']}")
    print(f"Embedding URL:    {cfg['embed_url']}")
    print(f"Read timeout:     {READ_TIMEOUT_VALUE}")
    print("=" * 100, flush=True)

    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_all(path: str, request: Request):
    request_id = uuid.uuid4().hex[:8]
    t0 = now_ms()

    method = request.method
    url = target_url(path, request.scope.get("query_string", b""))
    body = await request.body()
    content_type = request.headers.get("content-type", "")

    log(
        request_id,
        f"IN {method} /{path} "
        f"query={request.url.query!r} "
        f"ct={content_type!r} "
        f"body_bytes={len(body)}",
    )

    outbound_body = body
    body_was_patched = False

    if should_try_json(content_type, body):
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
            patched_data, body_was_patched = await patch_json_body_if_rag(
                request_id=request_id,
                path=path,
                method=method,
                data=data,
            )
            if body_was_patched:
                outbound_body = json.dumps(
                    patched_data,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                log(request_id, f"BODY_PREVIEW {short(body.decode('utf-8', errors='replace'))!r}")
        except Exception as exc:
            log(request_id, f"RAG_OR_JSON_PATCH_FAILED error={type(exc).__name__}: {exc}")
            return Response(
                content=(
                    "rag_proxy RAG preprocessing error\n"
                    f"path=/{path}\n"
                    f"error={type(exc).__name__}: {exc}\n"
                ),
                status_code=500,
                media_type="text/plain",
            )

    upstream_headers = filter_request_headers(request.headers)
    if body_was_patched:
        upstream_headers["content-type"] = "application/json"

    client: httpx.AsyncClient = request.app.state.client

    try:
        upstream_request = client.build_request(
            method=method,
            url=url,
            headers=upstream_headers,
            content=outbound_body,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        elapsed = now_ms() - t0
        log(request_id, f"UPSTREAM_ERROR after_ms={elapsed} error={type(exc).__name__}: {exc}")
        return Response(
            content=(
                "rag_proxy upstream error\n"
                f"target={url}\n"
                f"error={type(exc).__name__}: {exc}\n"
            ),
            status_code=502,
            media_type="text/plain",
        )

    response_headers = filter_response_headers(upstream_response.headers)
    elapsed = now_ms() - t0
    log(
        request_id,
        f"OUT status={upstream_response.status_code} "
        f"after_ms={elapsed} "
        f"upstream_ct={upstream_response.headers.get('content-type')!r} "
        f"streaming=True",
    )

    async def response_stream():
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        except asyncio.CancelledError:
            log(request_id, "STREAM_CANCELLED client disconnected")
            raise
        except Exception as exc:
            log(request_id, f"STREAM_ERROR {type(exc).__name__}: {exc}")
            raise

    return StreamingResponse(
        response_stream(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream_response.aclose),
    )


def main() -> None:
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
