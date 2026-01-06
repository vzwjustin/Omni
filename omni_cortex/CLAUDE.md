# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## ENFORCED RULES

**These rules are MANDATORY. Violations will break the system.**

### Critical Synchronization (ENFORCED)
```
VIOLATION = SYSTEM FAILURE
```
- **NEVER** modify `FRAMEWORK_NODES` in `app/graph.py` without ALSO updating:
  1. `FRAMEWORKS` dict in `app/core/router.py`
  2. `VIBE_DICTIONARY` in `app/core/vibe_dictionary.py`
  3. `get_framework_info()` in `app/core/router.py`
- All four locations MUST have matching framework names - no exceptions

### Async/Await (ENFORCED)
```
MISSING AWAIT = SILENT FAILURE
```
- ALL LangGraph nodes are async: `async def node(state: GraphState) -> GraphState`
- ALL MCP tool handlers are async
- ALWAYS use `await` for async calls - missing awaits cause silent failures
- Use `asyncio.Lock()` for shared mutable state

### State Access (ENFORCED)
```
DIRECT ACCESS = KEYERROR CRASH
```
- ALWAYS use `state.get("key", default)` - NEVER `state["key"]` directly
- ALL framework nodes must accept `GraphState` and return `GraphState`
- ALWAYS track tokens: `state["tokens_used"] += tokens`
- ALWAYS set `state["confidence_score"]` (0-1) before returning

### Error Handling (ENFORCED)
```
GENERIC EXCEPTIONS = HIDDEN BUGS
```
- ALWAYS use `OmniCortexError` hierarchy from `app/core/errors.py`
- `RoutingError`, `FrameworkNotFoundError` for routing
- `SandboxSecurityError`, `SandboxTimeoutError` for code execution
- `MemoryError`, `ThreadNotFoundError` for memory
- `RAGError`, `EmbeddingError` for vector store
- `LLMError`, `ProviderNotConfiguredError` for LLM calls

### Security (ENFORCED)
```
BYPASS = VULNERABILITY
```
- NEVER bypass sandbox checks - use `ALLOWED_IMPORTS` whitelist only
- NEVER include API keys, passwords, or secrets in code
- ALWAYS validate user input in MCP tool handlers
- ALWAYS sanitize output to prevent injection

### Testing (ENFORCED)
```
UNTESTED = UNSHIPPED
```
- ALWAYS test with both `LEAN_MODE=true` AND `LEAN_MODE=false`
- ALWAYS verify changes in Docker container before committing
- ALWAYS run `pytest tests/` before pushing
- Test edge cases and error conditions

### Code Quality (ENFORCED)
```
SLOPPY CODE = REJECTED
```
- Keep functions under 50 lines
- Avoid deep nesting (max 3 levels)
- Remove dead code and unused imports
- Use type hints for all function signatures
- Use meaningful names for variables and functions

### Context Optimization (ENFORCED)
```
WASTED TOKENS = WASTED MONEY
```
- Use specific file paths instead of broad searches
- Read only necessary portions of large files
- Use `/compact` when context grows large
- Prefer `grep` with `files_with_matches` mode
- Start new sessions for unrelated tasks

---

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

---

## Project Structure

```
omni_cortex/
├── server/main.py              # MCP entry (LEAN_MODE controls tools)
├── app/
│   ├── graph.py                # LangGraph (FRAMEWORK_NODES) ⚠️ SYNC REQUIRED
│   ├── state.py                # GraphState TypedDict
│   ├── langchain_integration.py
│   ├── collection_manager.py   # ChromaDB
│   ├── core/
│   │   ├── router.py           # HyperRouter (FRAMEWORKS) ⚠️ SYNC REQUIRED
│   │   ├── vibe_dictionary.py  # VIBE_DICTIONARY ⚠️ SYNC REQUIRED
│   │   ├── settings.py         # OmniCortexSettings (Pydantic)
│   │   ├── errors.py           # Exception hierarchy
│   │   └── config.py           # Legacy settings
│   ├── frameworks/             # Registry (single source of truth)
│   └── nodes/                  # Framework implementations
│       ├── common.py           # @quiet_star, LLM wrappers
│       ├── strategy/           # reason_flux, self_discover
│       ├── search/             # mcts, tot, got, eot
│       ├── iterative/          # active_inference, debate
│       ├── context/            # chain_of_note, step_back
│       ├── fast/               # system1, skeleton_of_thought
│       └── verification/       # chain_of_verification
├── tests/
│   ├── unit/                   # State, memory, sandbox tests
│   └── integration/            # MCP tool tests
└── docs/
    ├── API_REFERENCE.md
    └── FRAMEWORK_GUIDE.md
```

---

## Configuration

### LEAN_MODE (Token Optimization)

```bash
LEAN_MODE=true   # Default: 14 tools (reason + utilities) - saves ~55k tokens
LEAN_MODE=false  # All 76 tools (62 think_* + 14 utilities)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LEAN_MODE` | `true` | Token-optimized tool exposure |
| `LLM_PROVIDER` | `pass-through` | `pass-through`, `openrouter`, `anthropic`, `openai`, `google` |
| `EMBEDDING_PROVIDER` | `openrouter` | `openrouter`, `openai`, `huggingface` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `ENABLE_AUTO_INGEST` | `true` | Auto-index codebase |
| `MAX_MEMORY_THREADS` | `100` | LRU memory limit |
| `SANDBOX_TIMEOUT` | `5.0` | Code execution timeout |

---

## Architecture

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
    │   └── LLM selection → complex tasks
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
| HyperRouter | `app/core/router.py` | Framework selection |
| VIBE_DICTIONARY | `app/core/vibe_dictionary.py` | Pattern matching |
| LangGraph | `app/graph.py` | Workflow orchestration |
| Memory | `app/langchain_integration.py` | Conversation memory |
| RAG | `app/collection_manager.py` | ChromaDB |
| Settings | `app/core/settings.py` | Pydantic config |
| Errors | `app/core/errors.py` | Exception hierarchy |

### Memory System

| Scope | Storage | Lifetime |
|-------|---------|----------|
| Short-term | `GraphState` | Single request |
| Medium-term | `ConversationBufferMemory` | Same `thread_id` |
| Long-term | ChromaDB | Persists globally |

---

## Adding a New Framework

**⚠️ ALL FOUR LOCATIONS MUST BE UPDATED - NO EXCEPTIONS**

1. **Create node** in `app/nodes/{category}/my_framework.py`:
   ```python
   from app.state import GraphState
   from app.nodes.common import quiet_star

   @quiet_star
   async def my_framework_node(state: GraphState) -> GraphState:
       state["final_answer"] = "..."
       state["confidence_score"] = 0.85
       return state
   ```

2. **Export** from `app/nodes/{category}/__init__.py`

3. **Register in graph** (`app/graph.py`):
   ```python
   FRAMEWORK_NODES["my_framework"] = my_framework_node
   ```

4. **Add router metadata** (`app/core/router.py`):
   ```python
   FRAMEWORKS["my_framework"] = {"category": "...", "description": "...", "best_for": [...]}
   ```

5. **Add vibe patterns** (`app/core/vibe_dictionary.py`):
   ```python
   VIBE_DICTIONARY["my_framework"] = ["casual phrase", "another vibe"]
   ```

---

## Testing

```bash
# REQUIRED before any commit
docker-compose exec omni-cortex pytest tests/ -v

# Test both LEAN modes
LEAN_MODE=true docker-compose up -d
LEAN_MODE=false docker-compose up -d
```

---

## GraphState Key Fields

```python
# Input
query: str                  # Natural language request
code_snippet: Optional[str] # Code context
file_list: List[str]        # Relevant files
thread_id: str              # Memory key

# Routing
task_type: str              # 'debug', 'refactor', 'architect'
complexity_estimate: float  # 0.0-1.0
selected_framework: str     # Chosen framework

# Output
reasoning_steps: List[Dict] # Thought/Action/Observation
final_answer: str           # Response
final_code: str             # Generated code
confidence_score: float     # 0.0-1.0 (REQUIRED)
tokens_used: int            # Token count (REQUIRED)
```

---

## Docker Commands

```bash
docker-compose up -d --build      # Build and start
docker-compose logs -f omni-cortex # View logs
docker-compose restart omni-cortex # Restart
docker-compose exec omni-cortex pytest tests/ -v  # Run tests
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| "No embedding provider" | Set `OPENAI_API_KEY` or use `EMBEDDING_PROVIDER=huggingface` |
| Framework not found | Check all 4 sync locations match exactly |
| Memory not persisting | Use same `thread_id` across calls |
| Sandbox timeout | Increase `SANDBOX_TIMEOUT` or check for infinite loops |
