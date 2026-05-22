import html
import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not OPENROUTER_API_KEY:
    # Keep this loud: server won't work without model access.
    raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to .env")


app = FastAPI(title="LangChain OpenAI-Compatible Adapter", version="0.1.0")


@tool
def simple_math(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '2+2*3'."""
    try:
        allowed = set("0123456789+-*/(). ")
        if any(ch not in allowed for ch in expression):
            return "Unsupported characters in expression."
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as exc:
        return f"Error: {exc}"


@tool
def lookup_definition(term: str) -> str:
    """Return a short definition for a small fixed glossary."""
    glossary = {
        "lcel": "LangChain Expression Language for composing runnables.",
        "retriever": "A runnable that returns Documents for a query.",
        "tool calling": "A model capability to request function/tool execution with structured args.",
    }
    return glossary.get(term.lower(), "No definition found.")


# Ephemeral memory: kept in-process only.
_checkpointer = InMemorySaver()
_model = ChatOpenRouter(model=OPENROUTER_MODEL, api_key=OPENROUTER_API_KEY, temperature=0.2)
_agent = create_agent(
    model=_model,
    tools=[simple_math, lookup_definition],
    system_prompt=(
        "You are a concise assistant. Use tools when helpful. "
        "If you use a tool, incorporate the result into the final answer."
    ),
    checkpointer=_checkpointer,
)


def _now() -> int:
    return int(time.time())


def _tool_details_block(*, tool_call_id: str | None, name: str | None, arguments: Any, result: Any) -> str:
    # OpenWebUI renders tool activity when embedded as a <details type="tool_calls" ...> block.
    call_id = tool_call_id or "call_" + str(_now())
    tool_name = name or "tool"

    args_json = json.dumps(arguments, ensure_ascii=False, default=str)
    res_json = json.dumps(result, ensure_ascii=False, default=str)

    return (
        f'<details type="tool_calls" done="true" '
        f'id="{html.escape(call_id)}" '
        f'name="{html.escape(tool_name)}" '
        f'arguments="{html.escape(args_json)}">\n'
        f"<summary>Tool Executed</summary>\n"
        f"{html.escape(res_json)}\n"
        f"</details>\n"
    )


def _extract_assistant_text(state: dict[str, Any]) -> str:
    # Agent returns {'messages': [...]} where the final message is the latest AIMessage.
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content or ""
    return ""


def _choose_thread_id(x_thread_id: str | None, body: dict[str, Any]) -> str:
    # Preferred: explicit header set via OpenWebUI connection custom headers: x-thread-id: {{chat_id}}
    if x_thread_id:
        return x_thread_id
    # Fallbacks for direct callers.
    for key in ("chat_id", "session_id", "conversation_id"):
        val = body.get(key)
        if isinstance(val, str) and val:
            return val
    # Stateless fallback: one request = one thread.
    return f"req_{_now()}"


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    stream: bool = False
    temperature: float | None = None


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": OPENROUTER_MODEL,
                "object": "model",
                "created": _now(),
                "owned_by": "openrouter",
                "meta": {
                    "description": "LangChain create_agent (server-side tools)",
                    "capabilities": {"chat": True, "tools": True, "stream": True},
                },
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    x_thread_id: str | None = Header(default=None, alias="x-thread-id"),
) -> Any:
    payload = body.model_dump()
    # If caller specifies a different model than our configured OpenRouter model, reject to avoid surprises.
    if payload.get("model") and payload["model"] != OPENROUTER_MODEL:
        raise HTTPException(status_code=400, detail=f"Only model '{OPENROUTER_MODEL}' is supported by this server")

    # Basic message validation: only string content supported for now.
    for m in payload["messages"]:
        if not isinstance(m, dict) or "role" not in m:
            raise HTTPException(status_code=400, detail="Invalid messages format")
        content = m.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="Only string message content is supported")

    thread_id = _choose_thread_id(x_thread_id, payload)
    config = {"configurable": {"thread_id": thread_id}}

    # We pass messages through; agent owns tool loop and execution.
    input_state = {"messages": payload["messages"]}

    if not payload.get("stream"):
        state = await _agent.ainvoke(input_state, config=config)
        text = _extract_assistant_text(state)
        resp = {
            "id": f"chatcmpl_{_now()}",
            "object": "chat.completion",
            "created": _now(),
            "model": OPENROUTER_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
        return JSONResponse(resp)

    async def event_stream() -> AsyncIterator[bytes]:
        # OpenAI-style SSE stream. For a self-contained agent, we MUST NOT emit delta.tool_calls.
        # Instead we render tool execution as assistant-visible <details ...> blocks.
        created = _now()
        stream_id = f"chatcmpl_{created}"

        # First chunk: establish role.
        first = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": OPENROUTER_MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n".encode("utf-8")

        async for chunk in _agent.astream(
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
                        "model": OPENROUTER_MODEL,
                        "choices": [{"index": 0, "delta": {"content": token.content}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data)}\n\n".encode("utf-8")
            elif chunk.get("type") == "updates":
                for _node, update in (chunk.get("data") or {}).items():
                    msgs = update.get("messages") if isinstance(update, dict) else None
                    if not msgs:
                        continue
                    last = msgs[-1]
                    if isinstance(last, ToolMessage):
                        details = _tool_details_block(
                            tool_call_id=getattr(last, "tool_call_id", None),
                            name=getattr(last, "name", None),
                            arguments=getattr(last, "args", {}) or {},
                            result=last.content,
                        )
                        data = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": OPENROUTER_MODEL,
                            "choices": [{"index": 0, "delta": {"content": details}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n".encode("utf-8")

        final = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": OPENROUTER_MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "ok": "true",
        "models": "/v1/models",
        "chat_completions": "/v1/chat/completions",
    }
