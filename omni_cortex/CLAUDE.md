# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Omni-Cortex is a headless MCP (Model Context Protocol) server that routes coding tasks to 62 specialized AI reasoning frameworks. It acts as a "Brain" for IDE agents, using LangGraph for workflow orchestration and LangChain for memory/tools.

## Quick Start

```bash
# Docker (recommended)
cp .env.example .env  # Edit with your API keys
docker-compose up -d
docker-compose logs -f omni-cortex

# Local development
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"  # For embeddings
python -m server.main
```

## MCP Client Configuration

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": ["exec", "-i", "omni-cortex", "python", "-m", "server.main"]
    }
  }
}
```

Or for local development:
```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "python",
      "args": ["-m", "server.main"],
      "cwd": "/path/to/omni_cortex"
    }
  }
}
```

---

## Critical Rules

### Synchronization Requirements
- **NEVER** modify `FRAMEWORK_NODES` in `app/graph.py` without also updating:
  - `FRAMEWORKS` dict in `app/core/router.py`
  - `VIBE_DICTIONARY` in `app/core/vibe_dictionary.py`
  - `get_framework_info()` in `app/core/router.py`
- All three locations must have matching framework names

### Async Patterns
- All LangGraph nodes are async: `async def node(state: GraphState) -> GraphState`
- Use `await` for all async calls - missing awaits cause silent failures
- Use `asyncio.Lock()` for shared mutable state (see `app/langchain_integration.py`)

### State Management
- All framework nodes must accept `GraphState` and return `GraphState`
- Use `state.get("key", default)` pattern - never direct dict access
- Track tokens: `state["tokens_used"] += tokens`
- Set `state["confidence_score"]` (0-1) before returning

### Error Handling
- Use `OmniCortexError` hierarchy from `app/core/errors.py`:
  - `RoutingError`, `FrameworkNotFoundError` for routing issues
  - `SandboxSecurityError`, `SandboxTimeoutError` for code execution
  - `MemoryError`, `ThreadNotFoundError` for memory issues
  - `RAGError`, `EmbeddingError` for vector store issues
  - `LLMError`, `ProviderNotConfiguredError` for LLM calls

### Security
- Code sandbox uses AST-based validation (`_SafetyValidator` class)
- Never bypass sandbox checks - use `ALLOWED_IMPORTS` whitelist
- Never include API keys in code - use environment variables
- Validate all user input in MCP tool handlers

---

## Project Structure

```
omni_cortex/
├── server/main.py          # MCP server entry point (LEAN_MODE controls tool exposure)
├── app/
│   ├── graph.py            # LangGraph workflow (FRAMEWORK_NODES dict)
│   ├── state.py            # GraphState TypedDict
│   ├── langchain_integration.py  # Memory, RAG, callbacks
│   ├── collection_manager.py     # ChromaDB collections
│   ├── core/
│   │   ├── router.py       # HyperRouter (FRAMEWORKS dict)
│   │   ├── vibe_dictionary.py   # VIBE_DICTIONARY + match_vibes()
│   │   ├── settings.py     # OmniCortexSettings (Pydantic)
│   │   ├── errors.py       # Exception hierarchy
│   │   └── config.py       # Legacy settings
│   ├── frameworks/         # Framework registry (single source of truth)
│   │   ├── registry.py     # FrameworkSpec, register(), get_framework()
│   │   └── definitions.py  # Framework metadata definitions
│   └── nodes/              # Framework implementations
│       ├── common.py       # Shared utilities (@quiet_star, LLM wrappers)
│       ├── strategy/       # Architecture, patterns (reason_flux, self_discover)
│       ├── search/         # Tree/graph search (mcts, tot, got, eot)
│       ├── iterative/      # Refinement loops (active_inference, debate)
│       ├── context/        # Context enhancement (chain_of_note, step_back)
│       ├── fast/           # Quick responses (system1, skeleton_of_thought)
│       └── verification/   # Validation (chain_of_verification, critic)
├── tests/
│   ├── conftest.py         # Pytest fixtures
│   ├── unit/               # Unit tests (state, memory, sandbox)
│   └── integration/        # MCP tool tests
└── docs/
    ├── API_REFERENCE.md    # Full MCP API documentation
    └── FRAMEWORK_GUIDE.md  # Framework selection guide
```

---

## Configuration

### LEAN_MODE (Token Optimization)

```bash
# .env
LEAN_MODE=true   # Default: Only expose 14 tools (reason + utilities)
LEAN_MODE=false  # Expose all 76 tools (62 think_* + 14 utilities)
```

**LEAN_MODE=true (default):**
- Reduces MCP tool definitions from ~60k to ~5k tokens
- HyperRouter handles framework selection internally
- Use `reason` tool - it auto-selects the best framework

**LEAN_MODE=false:**
- Exposes all `think_*` tools for direct framework access
- Useful for testing specific frameworks

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LEAN_MODE` | `true` | Token-optimized tool exposure |
| `LLM_PROVIDER` | `pass-through` | `pass-through`, `openrouter`, `anthropic`, `openai`, `google` |
| `EMBEDDING_PROVIDER` | `openrouter` | `openrouter`, `openai`, `huggingface` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `ENABLE_AUTO_INGEST` | `true` | Auto-index codebase on startup |
| `ENABLE_DSPY_OPTIMIZATION` | `true` | DSPy prompt optimization |
| `ENABLE_PRM_SCORING` | `true` | Process reward model for search |
| `MAX_MEMORY_THREADS` | `100` | LRU memory thread limit |
| `SANDBOX_TIMEOUT` | `5.0` | Code execution timeout (seconds) |

---

## Architecture

### Request Flow (MCP → Response)

```
MCP call_tool("reason", {query, context})
    ↓
server/main.py: call_tool handler
    ↓
create_initial_state() → GraphState
    ↓
get_memory(thread_id) → OmniCortexMemory
    ↓
graph.ainvoke(state) → LangGraph workflow
    ├── route_node: HyperRouter.route()
    │   ├── match_vibes(query) → quick pattern match
    │   └── LLM selection → complex/ambiguous tasks
    ↓
    └── execute_node: framework_node(state)
        ├── enhance_state_with_langchain()
        ├── call_deep_reasoner() / call_fast_synthesizer()
        └── save_to_langchain_memory()
    ↓
format_reasoning_response() → MCP response
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| MCP Server | `server/main.py` | Tool registration, LEAN_MODE |
| HyperRouter | `app/core/router.py` | Framework selection (vibe + LLM) |
| VIBE_DICTIONARY | `app/core/vibe_dictionary.py` | Fast pattern matching |
| LangGraph | `app/graph.py` | Workflow orchestration |
| Memory | `app/langchain_integration.py` | Thread-based conversation memory |
| RAG | `app/collection_manager.py` | ChromaDB vector store |
| Settings | `app/core/settings.py` | Pydantic configuration |
| Errors | `app/core/errors.py` | Exception hierarchy |

### Memory System

| Scope | Storage | Lifetime |
|-------|---------|----------|
| Short-term | `GraphState` | Single request |
| Medium-term | `ConversationBufferMemory` | Same `thread_id` |
| Long-term | ChromaDB | Persists across threads |

---

## Adding a New Framework

1. **Create node** in `app/nodes/{category}/my_framework.py`:
   ```python
   from app.state import GraphState
   from app.nodes.common import quiet_star, add_reasoning_step

   @quiet_star
   async def my_framework_node(state: GraphState) -> GraphState:
       # Implementation
       state["final_answer"] = "..."
       state["confidence_score"] = 0.85
       return state
   ```

2. **Export** from `app/nodes/{category}/__init__.py`:
   ```python
   from .my_framework import my_framework_node
   __all__ = [..., "my_framework_node"]
   ```

3. **Register in graph** (`app/graph.py`):
   ```python
   FRAMEWORK_NODES = {
       ...,
       "my_framework": my_framework_node,
   }
   ```

4. **Add router metadata** (`app/core/router.py`):
   ```python
   FRAMEWORKS = {
       ...,
       "my_framework": {
           "category": "strategy",
           "description": "...",
           "best_for": ["use case 1", "use case 2"],
       },
   }
   ```

5. **Add vibe patterns** (`app/core/vibe_dictionary.py`):
   ```python
   VIBE_DICTIONARY = {
       ...,
       "my_framework": [
           "casual phrase 1",
           "another vibe pattern",
       ],
   }
   ```

---

## Testing

```bash
# Run all tests
docker-compose exec omni-cortex pytest tests/ -v

# Unit tests only
docker-compose exec omni-cortex pytest tests/unit/ -v

# Integration tests
docker-compose exec omni-cortex pytest tests/integration/ -v

# With coverage
docker-compose exec omni-cortex pytest tests/ --cov=app --cov-report=term-missing
```

### Test with both LEAN modes:
```bash
# Test LEAN_MODE=true (default)
LEAN_MODE=true docker-compose up -d && docker-compose logs -f

# Test LEAN_MODE=false (all tools)
LEAN_MODE=false docker-compose up -d && docker-compose logs -f
```

---

## Code Quality Rules

### Python Style
- Follow PEP 8
- Use type hints for all function signatures
- Docstrings for public functions (Google style)
- Keep functions under 50 lines
- Avoid deep nesting (max 3 levels)

### Async Best Practices
- Use `asyncio.gather()` for parallel operations
- Never block the event loop with sync I/O
- Use `async with` for context managers

### Performance
- Use LRU eviction for memory (`MAX_MEMORY_THREADS`)
- Lazy load heavy resources (embeddings, models)
- Profile before optimizing

### Context Optimization (for Claude Code)
- Use specific file paths instead of broad searches
- Read only necessary portions of large files
- Use `/compact` when context grows large
- Prefer `grep` with `files_with_matches` mode

---

## GraphState Key Fields

```python
# Input
query: str                  # Natural language request
code_snippet: Optional[str] # Code context
file_list: List[str]        # Relevant files
thread_id: str              # Memory persistence key

# Routing
task_type: str              # 'debug', 'refactor', 'architect'
complexity_estimate: float  # 0.0-1.0
selected_framework: str     # Chosen framework name

# Output
reasoning_steps: List[Dict] # Thought/Action/Observation
final_answer: str           # Response text
final_code: str             # Generated code
confidence_score: float     # 0.0-1.0
tokens_used: int            # Token count
```

---

## Troubleshooting

### Common Issues

**"No embedding provider available"**
- Set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in `.env`
- Or use `EMBEDDING_PROVIDER=huggingface` for local embeddings

**Framework not found**
- Check all three locations: `graph.py`, `router.py`, `vibe_dictionary.py`
- Names must match exactly (case-sensitive)

**Memory not persisting**
- Ensure same `thread_id` is passed across calls
- Check `MAX_MEMORY_THREADS` limit (default: 100)

**Sandbox timeout**
- Increase `SANDBOX_TIMEOUT` in `.env`
- Check for infinite loops in user code

---

## Docker Commands

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f omni-cortex

# Restart after changes
docker-compose restart omni-cortex

# Run tests in container
docker-compose exec omni-cortex pytest tests/ -v

# Check health
docker-compose exec omni-cortex python -c "from server.main import FRAMEWORKS; print(f'{len(FRAMEWORKS)} frameworks')"
```
