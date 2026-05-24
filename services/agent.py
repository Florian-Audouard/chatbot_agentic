from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from tools.math_tool import simple_math
from tools.weather_tool import get_weather


def build_agent_bundle(
    *, api_key: str, model: str, base_url: str, temperature: float
) -> tuple[Any, InMemorySaver]:
    """Construct the agent + checkpointer.

    Stateless server: built per-request.
    """

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if not model:
        raise HTTPException(status_code=400, detail="Missing model")
    if not base_url:
        raise HTTPException(status_code=400, detail="Missing X-Target-URL")

    chat_model = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

    agent = create_agent(
        model=chat_model,
        tools=[simple_math, get_weather],
        system_prompt=(
            "You are a concise assistant. Use tools when helpful. "
            "If you use a tool, incorporate the result into the final answer."
        ),
    )

    return agent
