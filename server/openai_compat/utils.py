from __future__ import annotations

import time
from typing import Any


def now() -> int:
    return int(time.time())


def choose_thread_id(x_thread_id: str | None, body: dict[str, Any]) -> str:
    # Preferred: explicit header set via OpenWebUI connection custom headers: x-thread-id: {{chat_id}}
    if x_thread_id:
        return x_thread_id

    # Fallbacks for direct callers.
    for key in ("chat_id", "session_id", "conversation_id"):
        val = body.get(key)
        if isinstance(val, str) and val:
            return val

    # Stateless fallback: one request = one thread.
    return f"req_{now()}"
