# CLAUDE.md

Omni-Cortex: Headless MCP server routing tasks to 62 AI reasoning frameworks via LangGraph/LangChain.

## Enforced Rules

### Framework Registry
**Source of truth: `app/frameworks/registry.py`** - all 62 frameworks defined here only.
- Add framework: 1) `FrameworkDefinition` in registry.py → 2) node in `app/nodes/{category}/` → 3) import in `app/graph.py` FRAMEWORK_NODES

### Async/Await
- ALL nodes/handlers are async. Missing `await` = silent failure
- Use `asyncio.Lock()` for shared state

### State Access
- ALWAYS `state.get("key", default)` - never `state["key"]`
- ALWAYS set `confidence_score` (0-1) and track `tokens_used`

### Errors
Use `app/core/errors.py` hierarchy: `RoutingError`, `FrameworkNotFoundError`, `SandboxSecurityError`, `SandboxTimeoutError`, `MemoryError`, `ThreadNotFoundError`, `RAGError`, `EmbeddingError`, `LLMError`, `ProviderNotConfiguredError`

### Security
- Never bypass sandbox - use `ALLOWED_IMPORTS` whitelist only
- Never include secrets in code
- Validate MCP inputs, sanitize outputs

### Testing
- Test both `LEAN_MODE=true` and `LEAN_MODE=false`
- Run `pytest tests/` before pushing
- Verify in Docker before commit

### Code Quality
Functions <50 lines, nesting <3 levels, type hints required, no dead code

## Structure

```
server/main.py          # MCP entry, LEAN_MODE
app/graph.py            # LangGraph + FRAMEWORK_NODES
app/state.py            # GraphState TypedDict
app/frameworks/registry.py  # ALL 62 frameworks (SINGLE SOURCE)
app/core/router.py      # HyperRouter
app/core/settings.py    # Pydantic config
app/core/errors.py      # Exception hierarchy
app/nodes/{category}/   # Framework implementations
app/langchain_integration.py  # Memory
app/collection_manager.py     # ChromaDB RAG
```

## Config

| Var | Default | Notes |
|-----|---------|-------|
| `LEAN_MODE` | `true` | 14 tools vs 76 |
| `LLM_PROVIDER` | `pass-through` | openrouter/anthropic/openai/google |
| `EMBEDDING_PROVIDER` | `openrouter` | openrouter/openai/huggingface |
| `SANDBOX_TIMEOUT` | `5.0` | seconds |

## Flow

```
MCP call_tool("reason") → server/main.py → GraphState
  → HyperRouter.route() → framework_node(state) → MCP response
```

Memory: Short=GraphState, Medium=ConversationBufferMemory(thread_id), Long=ChromaDB

## Adding Framework

```python
# 1. app/frameworks/registry.py
register(FrameworkDefinition(
    name="my_framework", category=FrameworkCategory.ITERATIVE,
    description="...", best_for=["..."], vibes=["..."],
    node_function="app.nodes.iterative.my_framework_node",
    complexity="medium", task_type="debug"
))

# 2. app/nodes/{category}/my_framework.py
@quiet_star
async def my_framework_node(state: GraphState) -> GraphState:
    state["final_answer"] = "..."
    state["confidence_score"] = 0.85
    return state

# 3. app/graph.py
FRAMEWORK_NODES["my_framework"] = my_framework_node
```

## GraphState

Input: `query`, `code_snippet`, `file_list`, `thread_id`
Routing: `task_type`, `complexity_estimate`, `selected_framework`
Output: `reasoning_steps`, `final_answer`, `final_code`, `confidence_score`(req), `tokens_used`(req)

## Commands

```bash
docker-compose up -d --build && docker-compose logs -f omni-cortex
docker-compose exec omni-cortex pytest tests/ -v
```

## Troubleshooting

- "No embedding provider" → Set `OPENAI_API_KEY` or `EMBEDDING_PROVIDER=huggingface`
- Framework not found → Check `app/frameworks/registry.py`
- Memory not persisting → Use same `thread_id`
