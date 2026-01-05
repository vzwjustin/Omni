# Omni-Cortex Architecture

Technical documentation for contributors and curious developers.

## Pass-Through Design

Omni-Cortex doesn't call any LLMs itself. When your IDE agent calls a framework tool:

1. Server returns a token-optimized prompt
2. Your IDE's LLM executes the prompt
3. Framework guides the reasoning, IDE does the work

This means:
- No API keys needed for frameworks (only for optional RAG features)
- Works with any LLM your IDE uses
- ~60-80 tokens per framework call

## Components

| Component | Purpose |
|-----------|---------|
| **LangGraph** | Workflow orchestration, 62 framework nodes |
| **LangChain Memory** | Conversation history per thread |
| **ChromaDB** | Vector store for RAG search (6 collections) |
| **HyperRouter** | Vibe-based framework selection |

## Tool Counts

| Configuration | Tools |
|---------------|-------|
| With API key (RAG enabled) | 70 tools |
| Without API key | 63 tools |

### Breakdown

- 62 `think_*` framework tools (always available)
- 1 `reason` smart router (always available)
- 7 RAG/search tools (require API key)

## Framework Categories

```
app/nodes/
├── strategy/      # 7 frameworks - Architecture, planning
├── search/        # 4 frameworks - Exploration, optimization
├── iterative/     # 8 frameworks - Debugging, refinement
├── code/          # 17 frameworks - Code-specific patterns
├── context/       # 6 frameworks - Research, analogies
├── fast/          # 2 frameworks - Quick responses
├── verification/  # 8 frameworks - Testing, validation
├── agent/         # 5 frameworks - Multi-agent patterns
└── rag/           # 5 frameworks - Retrieval-augmented generation
```

## Project Structure

```
omni_cortex/
├── app/
│   ├── state.py              # GraphState TypedDict
│   ├── graph.py              # LangGraph workflow
│   ├── langchain_integration.py  # Memory + RAG
│   ├── collection_manager.py # ChromaDB collections
│   ├── core/
│   │   ├── config.py         # Settings (Pydantic)
│   │   └── router.py         # HyperRouter
│   └── nodes/                # 62 framework implementations
├── server/
│   └── main.py               # MCP server entry point
├── setup.sh                  # Automated setup
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Configuration

```bash
# .env file
LLM_PROVIDER=pass-through          # Server doesn't call LLMs
EMBEDDING_PROVIDER=openai          # For RAG/ChromaDB (or 'none')
OPENAI_API_KEY=sk-...              # Only if using RAG features
ENABLE_AUTO_INGEST=true            # Index repo on startup
```

## Docker

```bash
# Build
docker-compose build

# Run (foreground)
docker-compose up

# Run (background)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Adding a New Framework

1. Create node in `app/nodes/{category}/`
2. Implement: `async def my_framework_node(state: GraphState) -> GraphState`
3. Export from `app/nodes/{category}/__init__.py`
4. Add to `FRAMEWORK_NODES` in `app/graph.py`
5. Add metadata to `HyperRouter.FRAMEWORKS` in `app/core/router.py`
6. Add vibe patterns to `HyperRouter.VIBE_DICTIONARY`

## Token Efficiency

Prompts are compressed to minimize costs:

| Component | Tokens |
|-----------|--------|
| Framework header | ~12 |
| Prompt body | ~50-70 |
| **Total per call** | **~60-80** |

Compare to verbose prompts (~150+ tokens) - ~50% savings per call.
