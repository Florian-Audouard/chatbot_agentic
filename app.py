from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.settings import get_settings
from openai_compat.router import router as openai_router
from services.agent import build_agent_bundle


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Make settings available to handlers even when key is missing.
    app.state.settings = settings

    if settings.openrouter_api_key:
        # Build once per process. The checkpointer is in-process only.
        agent, checkpointer = build_agent_bundle(settings)
        app.state.agent = agent
        app.state.checkpointer = checkpointer

    yield


settings = get_settings()

app = FastAPI(
    title="LangChain OpenAI-Compatible Adapter",
    version="0.1.0",
    lifespan=lifespan,
)

# Ensure request handlers can read settings even if lifespan isn't executed
# (e.g., some test setups).
app.state.settings = settings

# API routers
app.include_router(openai_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "ok": "true",
        "models": "/v1/models",
        "chat_completions": "/v1/chat/completions",
    }
