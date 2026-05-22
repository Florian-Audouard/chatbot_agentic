from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException
from langchain.messages import AIMessage, AIMessageChunk, ToolMessage

from server.openai_compat.tool_blocks import (
    extract_assistant_text,
    tool_details_block,
    tool_blocks_from_state,
)
from server.openai_compat.utils import now


async def invoke_non_streaming(
    *, agent: Any, model_id: str, input_state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    try:
        state = await agent.ainvoke(input_state, config=config)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream model error: {exc}")

    tool_blocks = tool_blocks_from_state(state)
    text = extract_assistant_text(state)
    content = (tool_blocks + "\n" + text).strip()
    created = now()
    return {
        "id": f"chatcmpl_{created}",
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


async def stream_openai_sse(
    *,
    agent: Any,
    model_id: str,
    input_state: dict[str, Any],
    config: dict[str, Any],
) -> AsyncIterator[bytes]:
    # OpenAI-style SSE stream. For a self-contained agent, we MUST NOT emit delta.tool_calls.
    # Instead we render tool execution as assistant-visible <details ...> blocks.
    created = now()
    stream_id = f"chatcmpl_{created}"

    # First chunk: establish role.
    first = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(first)}\n\n".encode("utf-8")

    tool_call_args: dict[str, dict[str, Any]] = {}

    try:
        async for chunk in agent.astream(
            input_state,
            config=config,
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if chunk.get("type") == "messages":
                token, _metadata = chunk["data"]
                if isinstance(token, AIMessageChunk) and token.content:
                    data = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token.content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n".encode("utf-8")
            elif chunk.get("type") == "updates":
                for _node, update in (chunk.get("data") or {}).items():
                    msgs = update.get("messages") if isinstance(update, dict) else None
                    if not msgs:
                        continue
                    last = msgs[-1]

                    if isinstance(last, AIMessage) and getattr(
                        last, "tool_calls", None
                    ):
                        for call in last.tool_calls:
                            if not isinstance(call, dict):
                                continue
                            call_id = call.get("id")
                            if not call_id:
                                continue
                            tool_call_args[call_id] = {
                                "name": call.get("name"),
                                "args": call.get("args") or {},
                            }

                    if isinstance(last, ToolMessage):
                        call_id = getattr(last, "tool_call_id", None)
                        info = tool_call_args.get(call_id or "", {})
                        details = tool_details_block(
                            tool_call_id=call_id,
                            name=info.get("name") or getattr(last, "name", None),
                            arguments=info.get("args") or {},
                            result=last.content,
                        )
                        data = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": details},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n".encode("utf-8")
    except Exception as exc:
        # Best-effort: emit a final assistant-visible error so the UI doesn't just hang.
        msg = f"\n[error] Upstream model error: {exc}\n"
        data = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"content": msg}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(data)}\n\n".encode("utf-8")

    final = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"
