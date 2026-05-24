from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from openai_compat.responses import build_models_list
from openai_compat.schemas import (
    ChatCompletionRequest,
    validate_messages_are_strings,
)
from openai_compat.service import invoke_non_streaming, stream_openai_sse
from openai_compat.utils import choose_thread_id


router = APIRouter()


@router.get("/v1/models")
def list_models(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    return build_models_list(model_id=settings.openrouter_model)


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    x_thread_id: str | None = Header(default=None, alias="x-thread-id"),
) -> Any:
    settings = request.app.state.settings
    payload = body.model_dump()

    # If caller specifies a different model than our configured OpenRouter model, reject to avoid surprises.
    if payload.get("model") and payload["model"] != settings.openrouter_model:
        raise HTTPException(
            status_code=400,
            detail=f"Only model '{settings.openrouter_model}' is supported by this server",
        )

    try:
        validate_messages_are_strings(payload["messages"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    thread_id = choose_thread_id(x_thread_id, payload)
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"messages": payload["messages"]}

    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=500, detail="OPENROUTER_API_KEY is missing. Add it to .env"
        )

    if not payload.get("stream"):
        resp = await invoke_non_streaming(
            agent=agent,
            model_id=settings.openrouter_model,
            input_state=input_state,
            config=config,
        )
        return JSONResponse(resp)

    return StreamingResponse(
        stream_openai_sse(
            agent=agent,
            model_id=settings.openrouter_model,
            input_state=input_state,
            config=config,
        ),
        media_type="text/event-stream",
    )
