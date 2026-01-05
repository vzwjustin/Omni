# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Omni-Cortex is a headless MCP (Model Context Protocol) server that routes coding tasks to 62 specialized AI reasoning frameworks. It acts as a "Brain" for IDE agents, using LangGraph for workflow orchestration and LangChain for memory/tools.

## Commands

### Running the Server

```bash
# Local development
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"  # or OPENROUTER_API_KEY
python -m server.main

# Docker
cp .env.example .env  # Edit with your API keys
docker-compose up -d
docker-compose logs -f omni-cortex
```

### MCP Client Configuration

For local development:
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

## Architecture

### Request Flow (MCP → Response)

1. MCP tool handler receives `reason` call (`server/main.py:call_tool`)
2. Create initial state with query and context (`create_initial_state`)
3. Retrieve memory for thread (`get_memory`)
4. Invoke LangGraph workflow (`graph.ainvoke`)
5. Enhance state with memory context (`enhance_state_with_langchain`)
6. Execute framework node dynamically (`execute_framework_node`)
7. Format response for IDE (`format_reasoning_response`)

### Framework Selection Flow

1. Route node invokes `HyperRouter.route()`
2. `auto_select_framework()` attempts selection:
   - First: Check `VIBE_DICTIONARY` for quick pattern match
   - Fallback: Call LLM with `FRAMEWORK_SELECTION_TEMPLATE` for complex/ambiguous tasks
3. `estimate_complexity()` scores task (0-1) based on query length, code size, file count
4. Update state with `selected_framework`, `complexity_estimate`, `task_type`

### Framework Execution Flow

1. Create `OmniCortexCallback` handler for LLM tracking
2. Surface recommended tools via `list_tools_for_framework()`
3. Framework calls `call_deep_reasoner()` or `call_fast_synthesizer()`
4. Callback tracks LLM start/end and token usage
5. Route to provider-specific client (OpenRouter/Anthropic/OpenAI)
6. Accumulate tokens in `state["tokens_used"]`
7. Save query-answer exchange to memory (`save_to_langchain_memory`)

### Memory System

- `get_memory(thread_id)` → get or create `OmniCortexMemory`
- LRU eviction when exceeding `MAX_MEMORY_THREADS` (100)
- `buffer_memory`: Recent conversation history (LangChain `ConversationBufferMemory`)
- `summary_memory`: Summarized long conversations
- `framework_history`: List of frameworks used in session

### Tool Execution Flow

1. Framework calls `run_tool(name, input, state)` wrapper
2. Lookup tool by name in `AVAILABLE_TOOLS`
3. Async invoke LangChain tool
4. For `execute_code`: runs in sandbox via `_safe_execute()` with pattern filtering
5. Return result to framework

### Vector Store (RAG)

- Startup: `ingest_repo_main()` reads files → `add_documents_with_metadata()` → embed with OpenAI → store in Chroma
- Runtime: `search_documentation` tool → `search_vectorstore()` → similarity search returns top-k
- Persist directory: `/app/data/chroma`

### LLM Client Management (`app/core/config.py`)

1. `ModelConfig` wrapper receives call
2. `_get_model_name()` resolves format (OpenRouter: `provider/model`, direct: `model`)
3. Select provider from `LLM_PROVIDER` env var
4. Route to appropriate client:
   - `openrouter`/`openai`: `_call_openai_compatible()`
   - `anthropic`: Direct `anthropic.messages.create()`
5. Extract response text and token count

### Docker Startup Flow

1. Entrypoint runs `python -m server.main`
2. `main()` validates API keys for configured provider
3. If `ENABLE_AUTO_INGEST=true`: run `ingest_repo_main()` to populate vector store
4. Create MCP server instance via `create_server()`
5. Create stdio transport via `stdio_server()`
6. Start server event loop with `server.run()`

### Key Components

**`server/main.py`** - MCP server entry point. Registers all tools: `reason` (smart router), `fw_{framework}` (direct access), `list_frameworks`, `health`, plus LangChain tools.

**`app/core/router.py`** - `HyperRouter` class. Two-stage selection:
1. `VIBE_DICTIONARY`: Fast pattern matching for casual queries ("wtf is wrong with this" → `active_inference`)
2. LLM-based selection via `FRAMEWORK_SELECTION_TEMPLATE` for complex/ambiguous tasks

**`app/graph.py`** - LangGraph workflow with SQLite checkpointing. Nodes: `route` → `execute` → `END`

**`app/state.py`** - `GraphState` TypedDict with fields for each framework type (search state, debate state, verification checks, PRM scores, etc.)

**`app/langchain_integration.py`** - Memory management (`OmniCortexMemory` with buffer + summary), vector store (Chroma), chat model factory, and prompt templates

### Framework Categories

Located in `app/nodes/`:

| Category | Frameworks | Key Use Cases |
|----------|-----------|---------------|
| `strategy/` | reason_flux, self_discover, buffer_of_thoughts, coala | Architecture, novel problems, patterns |
| `search/` | mcts_rstar, tree_of_thoughts, graph_of_thoughts, everything_of_thought | Algorithms, refactoring, complex bugs |
| `iterative/` | active_inference, multi_agent_debate, adaptive_injection, re2 | Debugging, decisions, specs |
| `code/` | program_of_thoughts, chain_of_verification, critic | Math/data, security, API validation |
| `context/` | chain_of_note, step_back, analogical | Research, performance, creative solutions |
| `fast/` | skeleton_of_thought, system1 | Boilerplate, quick fixes |

### Shared Framework Utilities (`app/nodes/common.py`)

- `@quiet_star` decorator: Forces `<quiet_thought>` blocks for explicit reasoning
- `process_reward_model()`: Scores intermediate steps (0-1) for MCTS/ToT
- `optimize_prompt()`: DSPy-style prompt rewriting
- `call_deep_reasoner()` / `call_fast_synthesizer()`: LLM wrappers with token tracking

## LLM Provider Configuration

Set `LLM_PROVIDER` in `.env`:
- `openrouter` (default): Single API key for all models. Model names use `provider/model` format.
- `anthropic`: Direct Claude API. Model names without prefix.
- `openai`: Direct GPT API. Model names without prefix.

Key env vars:
- `DEEP_REASONING_MODEL`: For complex reasoning (default: `anthropic/claude-4.5-sonnet`)
- `FAST_SYNTHESIS_MODEL`: For quick generation (default: `openai/gpt-5.2`)
- `ENABLE_DSPY_OPTIMIZATION`: Prompt optimization (default: true)
- `ENABLE_PRM_SCORING`: Process reward model for search (default: true)

## Adding a New Framework

1. Create node in appropriate `app/nodes/{category}/` directory
2. Implement async function: `async def my_framework_node(state: GraphState) -> GraphState`
3. Use `@quiet_star` decorator for reasoning transparency
4. Export from `app/nodes/{category}/__init__.py`
5. Add to `FRAMEWORK_NODES` dict in `app/graph.py`
6. Add metadata to `HyperRouter.FRAMEWORKS` and `get_framework_info()` in `app/core/router.py`
7. Add vibe patterns to `HyperRouter.VIBE_DICTIONARY`

## GraphState Key Fields

```python
# Input Context
query: str                  # User's natural language request
code_snippet: Optional[str] # Selected code from IDE
file_list: List[str]        # Relevant workspace files
ide_context: Optional[str]  # Cursor position, linter errors, etc.

# Routing Metadata
task_type: str              # 'debug', 'refactor', 'architect', etc.
complexity_estimate: float  # 0.0-1.0 driving resource allocation
selected_framework: str     # Chosen by HyperRouter

# Execution Artifacts
reasoning_steps: List[Dict] # Thought/Action/Observation trace
final_answer: str           # Synthesized response
final_code: str             # Generated code blocks
quiet_thoughts: List[str]   # Hidden scratchpad (Quiet-STaR)

# Working Memory (transient, per-request)
working_memory: Dict        # Contains: thread_id, chat_history,
                            # framework_history, available_tools,
                            # langchain_callback, recommended_tools

# Metrics
tokens_used: int            # Total token consumption
confidence_score: float     # 0.0-1.0 result confidence
```

## Memory Persistence Model

| Scope | Storage | Lifetime |
|-------|---------|----------|
| Short-term | `GraphState` | Single `graph.ainvoke()` |
| Medium-term | `ConversationBufferMemory` | Same `thread_id` across turns |
| Long-term | Chroma VectorStore | Global, persists across all threads |

## Error Handling Patterns

- **Routing fallback**: If LLM selection fails → defaults to `self_discover`
- **Tool failures**: Frameworks return partial state rather than crashing graph
- **Sandbox safety**: `_safe_execute` uses AST parsing to block dangerous imports, 5s timeout
- **Context limits**: `ConversationSummaryMemory` auto-compresses when history exceeds thresholds
- **Missing thread_id**: New UUID generated, treated as stateless one-off request
- **Vector store errors**: `search_vectorstore` returns empty list on failure, frameworks proceed with "general knowledge"

## State Management Conventions

- Use `add_reasoning_step()` from `common.py` to log steps
- Track tokens: `state["tokens_used"] += tokens`
- Store framework-specific state in `state["working_memory"]`
- Final outputs go to `state["final_answer"]` and `state["final_code"]`
- Set `state["confidence_score"]` (0-1) before returning

## Prompting Best Practices (for IDE agents calling Omni-Cortex)

- **Vibe over rigidity**: Describe the problem nature, not the framework
  - Good: "This feels sluggish, worried about O(n^2) complexity"
  - Bad: "Use Tree of Thoughts to optimize this"
- **Provide context**: Always populate `code_snippet` and `file_list` - router uses them for framework selection
- **Use thread_id**: For iterative debugging, reuse the same `thread_id` so Omni-Cortex sees prior attempts and can switch strategies
