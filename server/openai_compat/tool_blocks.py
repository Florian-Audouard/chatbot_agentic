from __future__ import annotations

import html
import json
from typing import Any

from langchain.messages import AIMessage, ToolMessage

from openai_compat.utils import now


def tool_details_block(
    *, tool_call_id: str | None, name: str | None, arguments: Any, result: Any
) -> str:
    # OpenWebUI renders tool activity when embedded as a <details type="tool_calls" ...> block.
    call_id = tool_call_id or "call_" + str(now())
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


def extract_assistant_text(state: dict[str, Any]) -> str:
    # Agent returns {'messages': [...]} where the final message is the latest AIMessage.
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content or ""
    return ""


def tool_blocks_from_state(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    calls: dict[str, dict[str, Any]] = {}
    blocks: list[str] = []

    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for call in m.tool_calls:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                if not call_id:
                    continue
                calls[call_id] = {
                    "name": call.get("name"),
                    "args": call.get("args"),
                }

    for m in messages:
        if isinstance(m, ToolMessage):
            call_id = getattr(m, "tool_call_id", None)
            info = calls.get(call_id or "", {})
            blocks.append(
                tool_details_block(
                    tool_call_id=call_id,
                    name=info.get("name") or getattr(m, "name", None),
                    arguments=info.get("args") or {},
                    result=m.content,
                )
            )

    return "".join(blocks)
