# LangChain Core Concepts (Python, OpenRouter)

This repo contains a focused, modern LangChain notebook series using OpenRouter.
It stays fully offline except for model calls.

## Quickstart (uv)

1) Install and start Jupyter:

```bash
uv sync
uv run jupyter notebook
```

## Notebooks

- `notebooks/00_setup_openrouter.ipynb`
- `notebooks/01_lcel_runnables.ipynb`
- `notebooks/02_retrieval_rag_offline.ipynb`
- `notebooks/03_tools_agents_memory.ipynb`

## OpenWebUI (OpenAI-compatible server)

This repo includes a small FastAPI server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint backed by `create_agent`.

### Run

```bash
uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

This server is stateless. Provider base URL and API key are passed per-request:

- `X-Target-URL: <provider base url, including /v1>` (example: `https://openrouter.ai/api/v1`)
- `Authorization: Bearer <api key>`

Then verify:

- `GET http://localhost:8000/v1/models` (returns exactly what the provider returns)
- `POST http://localhost:8000/v1/chat/completions` (LangChain agent with server-side tools)

### Tool calls in OpenWebUI

This server executes tools server-side. To display tool activity (name, args, result) in OpenWebUI without triggering OpenWebUI's own tool-executor loop, the server embeds tool executions as assistant-visible `<details type="tool_calls" ...>` blocks in the streamed content.

### Conversation memory (ephemeral)

Memory is in-process only (lost on restart). To keep per-chat continuity, configure OpenWebUI to send a stable thread header:

- Connection settings: add custom header `x-thread-id: {{chat_id}}`

## Notes

- Model is selected via the request body `model`.
- The notebooks use only modern LangChain APIs (LCEL + `create_agent`).
- Persistence uses LangGraph checkpointers (`thread_id`), not deprecated history wrappers.
