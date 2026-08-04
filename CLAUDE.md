# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multimodal RAG chat application. Two parts in one repo:

- **Backend** (`src/`, root `pyproject.toml`) — FastAPI + LangGraph agent, Postgres, Qdrant, Redis, Celery.
- **Frontend** (`frontend/`) — Next.js 16 / React 19 chat UI (ChatGPT-style: projects, threads, artifacts, model picker).

The `README.md` documents an older, CLI-only slice of this project (`main.py ingest/ask/chat` against a fixed paper via Docling+Qdrant+Ollama). That CLI still works and shares the same core (`src/agent/rag_core.py`), but the FastAPI app + Next.js frontend are the primary product now — most new work happens there, not in the README's documented flow.

## Commands

### Backend (run from repo root, via `uv`)

```bash
uv sync                                   # install deps
uv run main.py api                        # start FastAPI (dev: reload on, host 0.0.0.0:8000)
uv run main.py worker                     # start Celery worker (background artifact ingestion)
uv run main.py ingest --rebuild           # CLI: ingest default paper(s) into Qdrant
uv run main.py ask "question"             # CLI: one-shot RAG question
uv run main.py chat                       # CLI: interactive RAG chat
uv run pytest                             # run tests
uv run pytest tests/test_note_helpers.py::test_clean_markdown_strips_wrapping_fence  # single test
uv run alembic revision --autogenerate -m "..."   # after editing src/db/models/*
uv run alembic upgrade head               # apply migrations (also auto-run on `main.py api` via _check_database_connection)
pre-commit run --all-files                # ruff check --fix + ruff format + biome (frontend) + detect-secrets
```

Infra dependencies (Postgres, Qdrant, Redis, Localstack for S3) are started via:

```bash
docker compose up -d
```

Ollama and/or the OMLX server (see `src/lib/omlx.py`, `OMLX_BASE_URL`) must be running separately for local model inference — they are not in `docker-compose.yaml`.

### Frontend (run from `frontend/`)

```bash
bun install        # or: npm install (bun.lock is the lockfile in use)
bun run dev         # next dev --webpack
bun run build
bun run lint        # biome check
bun run format      # biome format --write
```

## Architecture

### Backend request flow

`main.py` is the single CLI entrypoint for every backend process (`api`, `worker`, `ingest`, `ask`, `chat`). `src/config/bootstrap.py` is imported for side effects first thing (`import src.config.bootstrap  # noqa: F401`) — it wires loguru + rich tracebacks and monkey-patches `builtins.print` to route through the logger with level markup. Anything printed anywhere in the backend goes through this.

`src.api.create_api()` (`src/api/__init__.py`) builds the FastAPI app: CORS from `CORS_ALLOW_ORIGINS`, lifespan hook that inits/closes the LangGraph Postgres checkpointer, routes mounted under `/api`. Route modules live under `src/api/routes/{auth,config,projects,models}`, each with its own `router`; `projects` further nests `routes/{artifacts,threads}`. Most routes depend on `try_get_current_user` (JWT-based).

The RAG agent has two layers:
- `src/agent/rag_core.py` — framework-agnostic core: builds the LangGraph agent (`build_rag_agent`), streams events (`stream_rag_events`), Qdrant filter helpers. Shared by both CLI and API.
- `src/agent/rag_agent.py` — CLI-specific adapter (colored terminal tool-call tracing, `answer_question`/`interactive_chat`).

Tools the agent can call live in `src/agent/tools/` (`document_retriever`, `web_fetch`, `duckduckgo_search`, `image_search`, `next_chunk`, plus `builtin_tools.py` for third-party LangChain tools like YouTube search).

`src/side_agents/` holds small, single-purpose LLM agents invoked as steps in larger flows, not exposed as chat tools: `note_generator_agent` (turns sliding-window chunk triples into per-chunk notes, see item 12 in `todo.md`), `update_project_name_agent`, `update_thread_name_agent`. Each is a package with `__init__.py` (logic) + `prompt.md` (the prompt template kept out of Python source).

### Model selection

Models are referenced everywhere by a `"provider:model"` string constant (e.g. `DEFAULT_LLM_MODEL = "omlx:gemma-4-e4b-it-4bit"` in `src/config/constants.py`). `src/config/model_config.py`'s `MODEL_CONFIGS` maps these strings to a `Model` (provider, pricing, context window, reasoning effort). `model_selector()` resolves a model string to a `BaseChatModel`: the `omlx` provider routes to a custom `ChatOmlx` client (`src/lib/omlx.py`, talks to a local OMLX server via `OMLX_BASE_URL`); every other provider goes through LangChain's `init_chat_model`. Adding a new model = add an entry to `MODEL_CONFIGS`, not a new code path.

### Data layer

Postgres via SQLAlchemy (async engine for the API, sync engine for Alembic/LangGraph's `PostgresSaver`) — both built from the same connection parts in `src/db/__init__.py`. Models under `src/db/models/`: `User`, `Project`, `ProjectDocument`, `Document`, `Thread`, `Chat`, `Chunk`, `Preferences`. Schema changes always go through Alembic (`alembic/versions/`) — never hand-edit tables; `create_all_tables()` just runs `alembic upgrade head`.

Qdrant (`src/vector_store/qdrant_store.py`) is the vector store for document chunks (hybrid dense+sparse via `langchain-qdrant`); `src/vector_store/faiss_store.py` is a legacy/alternate local store, no longer the default path (see `todo.md` item 2 — migrated to Qdrant). Ingestion (`src/module/upload_docs.py`, `src/lib/docling_lib.py`) uses Docling to parse PDFs into markdown + extracted image/formula artifacts, hashes each source paper (SHA256) and stores the hash in point metadata so re-ingesting an already-indexed paper is a no-op unless `--rebuild` is passed.

Long-running ingestion work (artifact processing, note generation) runs on Celery (`src/lib/celery_lib.py`, `src/tasks/`) backed by Redis, not inline in the request path — `main.py worker` runs the consumer.

### Frontend

Next.js App Router under `frontend/src/app/` (`auth/{login,register}`, `chat`, `settings`). `frontend/src/components/ai-elements/` is a large library of chat/agent UI primitives (message, tool call, reasoning, artifact panel, code block, etc.) composed by `frontend/src/components/chat/*` into the actual chat screen. API calls are centralized in `frontend/src/lib/api/{auth,chat,models,preferences,projects}.ts`. `frontend/CLAUDE.md` just imports `AGENTS.md` (frontend-specific conventions, if any, live there — check it alongside this file when touching `frontend/`).

## Conventions

- Root `AGENTS.md` sets the working style expected in this repo (also pulled in by `frontend/CLAUDE.md`): favor plan mode for non-trivial (3+ step) changes, delegate research/parallel work to subagents, find root causes rather than patch symptoms, keep diffs minimal, verify behavior (tests/logs) before calling something done.
- `todo.md` at repo root is the running, checkbox-style feature log — check it for in-flight/planned work and the reasoning behind past features before assuming something is unimplemented.
- Pre-commit enforces `ruff check --fix` + `ruff format` on Python (ignoring E402/E731) and `biome check --write` on `frontend/`, plus `detect-secrets` against `.secrets.baseline`. Run `pre-commit run --all-files` before considering backend/frontend changes done.
- **After changing code, make a commit in Conventional Commits format** (`feat:`, `fix:`, `chore:`, `refactor:`, etc., matching the existing git log style).
