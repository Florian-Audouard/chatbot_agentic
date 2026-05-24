from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from openai_compat.schemas import (
    ChatCompletionRequest,
    validate_messages_are_strings,
)
from openai_compat.service import invoke_non_streaming, stream_openai_sse
from openai_compat.utils import choose_thread_id
from services.agent import build_agent_bundle


router = APIRouter()


@router.get("/v1/models")
async def list_models(
    x_target_url: str | None = Header(default=None, alias="X-Target-URL"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> Response:
    if not x_target_url:
        raise HTTPException(status_code=400, detail="Missing X-Target-URL")
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            upstream = await client.get(
                f"{x_target_url.rstrip('/')}/models",
                headers={"Authorization": authorization},
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

    # Return exactly what provider returns (status, content-type, body).
    content_type = upstream.headers.get("content-type")
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=content_type,
    )


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    x_target_url: str | None = Header(default=None, alias="X-Target-URL"),
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_thread_id: str | None = Header(default=None, alias="x-thread-id"),
) -> Any:
    if not x_target_url:
        raise HTTPException(status_code=400, detail="Missing X-Target-URL")
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization must be Bearer")

    api_key = authorization.split(" ", 1)[1].strip()
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    payload = body.model_dump()

    # Per-request model selection.
    model_id = payload.get("model")
    if not isinstance(model_id, str) or not model_id:
        raise HTTPException(status_code=400, detail="Missing model")

    try:
        validate_messages_are_strings(payload["messages"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    thread_id = choose_thread_id(x_thread_id, payload)
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"messages": payload["messages"]}

    temperature = payload.get("temperature")
    if temperature is None:
        temperature = 0.2
    try:
        temperature_f = float(temperature)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid temperature")

    agent = build_agent_bundle(
        api_key=api_key,
        model=model_id,
        base_url=x_target_url,
        temperature=temperature_f,
    )

    if not payload.get("stream"):
        resp = await invoke_non_streaming(
            agent=agent,
            model_id=model_id,
            input_state=input_state,
            config=config,
        )
        return JSONResponse(resp)

    return StreamingResponse(
        stream_openai_sse(
            agent=agent,
            model_id=model_id,
            input_state=input_state,
            config=config,
        ),
        media_type="text/event-stream",
    )
