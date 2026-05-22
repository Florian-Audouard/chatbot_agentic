from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatCompletionRequest(BaseModel):
    # OpenAI clients may send extra fields; we ignore them.
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]]
    stream: bool = False
    temperature: float | None = None


def validate_messages_are_strings(messages: list[dict[str, Any]]) -> None:
    # Basic message validation: only string content supported for now.
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise ValueError("Invalid messages format")
        content = m.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            raise ValueError("Only string message content is supported")
