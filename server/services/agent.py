from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver

from server.core.settings import Settings
from server.tools.math_tool import simple_math
from server.tools.weather_tool import get_weather


def build_agent_bundle(settings: Settings) -> tuple[Any, InMemorySaver]:
    """Construct the agent + checkpointer.

    Stored on `app.state` during lifespan startup.
    """

    if not settings.openrouter_api_key:
        # Defensive: create_app() also checks this.
        raise HTTPException(
            status_code=500, detail="OPENROUTER_API_KEY is missing. Add it to .env"
        )

    checkpointer = InMemorySaver()
    model = ChatOpenRouter(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        temperature=settings.openrouter_temperature,
    )

    agent = create_agent(
        model=model,
        tools=[simple_math, get_weather],
        system_prompt=(
            "You are a concise assistant. Use tools when helpful. "
            "If you use a tool, incorporate the result into the final answer."
        ),
        checkpointer=checkpointer,
    )

    return agent, checkpointer
