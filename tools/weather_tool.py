from __future__ import annotations

from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Fake weather tool for demonstration."""

    return f"The weather in {location} is sunny with a high of 25°C."
