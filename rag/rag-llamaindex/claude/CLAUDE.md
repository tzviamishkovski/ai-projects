# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A NotebookLM-style grounded research assistant, built as a staged learning project for **LangChain v1**. The code is organized as a finished product, by feature, not by development stage. Stack: Anthropic Claude for chat, Cohere for embeddings, FastAPI backend, a static (no-build-step) HTML/JS web client.

## Commands

```bash
uv sync                      # install dependencies (uv-managed; Python >=3.13)
uv run notebooklm-serve      # start the FastAPI backend + web client (default http://127.0.0.1:4040, override with NOTEBOOKLM_PORT/NOTEBOOKLM_HOST)
uv run notebooklm "prompt"   # one-shot CLI chat (src/app.py), no server, no persistence
```

Required env vars live in `.env` (loaded via `python-dotenv`): `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, optional `FIRECRAWL_API_KEY` (enables the web search/scrape/crawl tools; without it those tools return a friendly "unavailable" message instead of failing). `.env` is gitignored — never commit it.

There is no test suite and no linter/type-checker config in this repo yet.

## Architecture

**Import roots.** `pyproject.toml` builds with `sources = ["src"]`, so everything under `src/` is a top-level import — `agents`, `api`, `core`, `tools` — not `src.agents` or a package-name prefix. Match this when adding new modules.

- **`api/`** — FastAPI HTTP layer. `app.py` defines routes; `schemas.py` is the stable Pydantic contract the web client is written against (kept additive across stages); `services.py` is the thin translation layer between routes and the `agents`/`core` implementations; `serve.py` is the process entrypoint — it calls `load_dotenv()` before any other import, which matters because some modules (e.g. `core/store.py`) read env vars at import time.
- **`agents/`** — one agent per feature. Currently only `chat.py`: builds a LangChain `create_agent` per request, with an `InMemorySaver` checkpointer keyed by `thread_id` (memory lives only for the process lifetime — nothing is persisted across restarts).
- **`core/`** — the notebook's own domain state. `store.py` holds a process-lifetime, in-memory `SourceStore` singleton (a dict of sources + an `InMemoryVectorStore` for retrieval via Cohere embeddings). `sources.py` chunks source text with `RecursiveCharacterTextSplitter` before embedding. `SourceStore.search()` is filtered to only the currently *active* sources, so toggling a source in the UI actually changes retrieval, not just display.
- **`tools/`** — reusable LangChain tool factories that aren't tied to one agent or to the notebook's own data (external APIs). Currently `firecrawl.py`: `make_firecrawl_tools()` builds `web_search` / `scrape_url` / `crawl_site` for the chat agent, gated on `FIRECRAWL_API_KEY`; it also exposes a plain `scrape_page(url)` helper reused directly by `api/services.py` to add a source from a URL (`POST /api/sources/from-url`) without going through the agent tool-calling layer.
- **`client/`** — static HTML/JS/CSS, no build step, served by FastAPI's `StaticFiles` mount at `/` (mounted last in `app.py` so it never shadows `/api/*`). Talks to the backend only via `fetch` calls in `app.js` against `/api/*`. A `no_store` middleware in `app.py` disables all HTTP caching so client edits show up on a plain refresh.

**Everything is in-memory, no database** — `SourceStore`, saved notes, and chat checkpointer state all reset on server restart.

**Source ingestion has three paths**, all converging on `services.add_source`: paste text (`POST /api/sources`), file upload (`POST /api/sources/upload`, dispatched by extension in `api/extract.py` — `.pdf` via `pypdf`, `.docx` via `python-docx`, known text extensions via an encoding fallback chain, anything else rejected with a clear 400), and scrape-from-URL (`POST /api/sources/from-url`, via `tools/firecrawl.scrape_page`).

Studio artifact generation (`POST /api/studio/generate`) is stubbed for every kind — it raises `ComingSoon` → HTTP 501. See the feature roadmap table in `README.md` for what's built vs. planned.
