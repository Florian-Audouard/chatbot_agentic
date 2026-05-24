from __future__ import annotations

from typing import Any

from openai_compat.utils import now


def build_models_list(*, model_id: str) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": now(),
                "owned_by": "openrouter",
                "meta": {
                    "description": "LangChain create_agent (server-side tools)",
                    "capabilities": {"chat": True, "tools": True, "stream": True},
                },
            }
        ],
    }


def build_chat_completion(*, model_id: str, content: str) -> dict[str, Any]:
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
