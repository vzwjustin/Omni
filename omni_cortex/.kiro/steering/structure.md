# Project Structure & Organization

## Top-Level Structure

```
omni_cortex/
├── app/                      # Core runtime logic
├── server/                   # MCP server entry point
├── scripts/                  # Development utilities
├── data/                     # Runtime persistence (excluded from commits)
├── tests/                    # Test suite
├── docs/                     # Documentation
├── mcp-config-examples/      # MCP configuration templates
└── knowledge/                # Knowledge base files
```

## Core Application (`app/`)

```
app/
├── __init__.py
├── state.py                  # GraphState TypedDict definition
├── graph.py                  # LangGraph workflow orchestration
├── langchain_integration.py  # Memory + RAG integration
├── collection_manager.py     # ChromaDB collections management
├── core/                     # Core utilities and configuration
├── nodes/                    # Framework implementations (62 frameworks)
├── orchestrators/            # Framework factory and orchestration
├── prompts/                  # Prompt templates and parsers
├── retrieval/                # Vector search and embeddings
├── memory/                   # Memory management
├── models/                   # Model abstractions
├── callbacks/                # Monitoring and callbacks
└── frameworks/               # Framework registry and validation
```

## Framework Categories (`app/nodes/`)

Frameworks are organized by reasoning type:

- `strategy/` - Architecture, planning (7 frameworks)
- `search/` - Exploration, optimization (4 frameworks) 
- `iterative/` - Debugging, refinement (8 frameworks)
- `code/` - Code-specific patterns (17 frameworks)
- `context/` - Research, analogies (6 frameworks)
- `fast/` - Quick responses (2 frameworks)
- `verification/` - Testing, validation (8 frameworks)
- `agent/` - Multi-agent patterns (5 frameworks)
- `rag/` - Retrieval-augmented generation (5 frameworks)

## Server & Configuration (`server/`)

```
server/
├── main.py                   # MCP server entry point
├── framework_prompts.py      # Framework prompt definitions
└── handlers/                 # MCP tool handlers
    ├── framework_handlers.py # Framework tool implementations
    ├── rag_handlers.py       # RAG/search tool handlers
    ├── reason_handler.py     # Smart routing handler
    └── utility_handlers.py   # Utility tool handlers
```

## Development Scripts (`scripts/`)

- `debug_search.py` - Diagnose ChromaDB/OpenAI issues
- `test_mcp_search.py` - Test search functionality
- `verify_learning_offline.py` - Validate learning flow
- `validate_frameworks.py` - Check framework definitions
- `ingest_*.py` - Data ingestion utilities
- `startup.sh` - Container startup script

## Naming Conventions

- **Modules/Functions**: `snake_case`
- **Classes**: `CamelCase` 
- **Constants**: `UPPER_SNAKE_CASE`
- **Framework nodes**: `async def <name>_node(state: GraphState) -> GraphState`
- **MCP tools**: `think_<framework_name>` pattern
- **Files**: Use descriptive names, group related functionality

## Key Files to Know

- `app/graph.py` - Main workflow definition
- `app/core/router.py` - HyperRouter for framework selection
- `app/core/vibe_dictionary.py` - Vibe patterns for routing
- `server/main.py` - MCP server startup
- `setup.sh` - Automated project setup
- `.env.example` - Configuration template