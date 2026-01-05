# Repository Guidelines

## Project Structure & Module Organization
- `app/` holds core runtime logic: LangGraph workflow, router, state, and framework nodes.
- `app/nodes/` contains framework implementations grouped by category (`strategy/`, `search/`, `iterative/`, `code/`, `context/`, `fast/`).
- `server/` contains the MCP server entry point (`server/main.py`).
- `scripts/` provides dev utilities (ingestion, search/debug helpers, startup).
- `data/` is runtime persistence (ChromaDB, checkpoints); keep out of commits.
- `mcp-config-examples/`, `MCP_SETUP.md`, `ARCHITECTURE.md`, `CLAUDE.md` document setup and design.

## Build, Test, and Development Commands
- `./setup.sh` creates `.env`, builds the Docker image, configures MCP, and starts the service.
- `docker-compose build` builds the image.
- `docker-compose up -d` runs the MCP server; `docker-compose logs -f` tails logs; `docker-compose down` stops.
- `pip install -r requirements.txt` then `python -m server.main` runs locally without Docker.

## Coding Style & Naming Conventions
- Python with 4-space indentation, type hints where practical, async-first for framework nodes.
- Use `snake_case` for modules/functions, `CamelCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Framework nodes follow `async def <name>_node(state: GraphState) -> GraphState` in `app/nodes/<category>/`.

## Testing Guidelines
- No formal test suite yet; use scripts for checks:
- `python scripts/verify_learning_offline.py` validates learning flow with mock embeddings.
- `python scripts/test_mcp_search.py` exercises Chroma search (requires embeddings + data).
- `python -m scripts.debug_search` diagnoses Chroma/OpenAI configuration.
- If you add new scripts or checks, document them here or in `README.md`.

## Commit & Pull Request Guidelines
- Use Conventional Commits seen in history: `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`, `chore: ...`.
- PRs should include a short summary, testing evidence (commands run), and any config changes.
- New frameworks must update `app/graph.py`, `app/core/router.py`, and `app/core/vibe_dictionary.py`.

## Configuration & Runtime Notes
- Copy `.env.example` to `.env`; never commit API keys.
- Stdout is reserved for MCP JSON-RPC; log to stderr in server/runtime code (see `scripts/startup.sh`).
