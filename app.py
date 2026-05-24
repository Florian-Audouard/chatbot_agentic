from __future__ import annotations

from fastapi import FastAPI

from openai_compat.router import router as openai_router


app = FastAPI(title="LangChain OpenAI-Compatible Adapter", version="0.2.0")

app.include_router(openai_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "ok": "true",
        "models": "/v1/models",
        "chat_completions": "/v1/chat/completions",
    }
