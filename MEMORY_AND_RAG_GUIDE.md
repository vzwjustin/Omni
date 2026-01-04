# Memory & RAG Guide for Omni-Cortex

## Overview

Omni-Cortex has **two types of tools**:

1. **Framework Tools** (`think_*`, `reason`) - Stateless, return prompt templates
2. **Utility Tools** (14 tools) - Stateful, provide memory/RAG/execution

## Memory System (Thread-Based Context)

### How It Works

```python
# In-memory storage (survives for server uptime)
_memory_store = {
    "thread-id-1": OmniCortexMemory(
        messages=[HumanMessage(...), AIMessage(...)],  # Last 20 messages
        framework_history=["active_inference", "alphacodium"]
    ),
    "thread-id-2": OmniCortexMemory(...),
    # ... up to 100 threads (LRU eviction)
}
```

### Available Memory Tools

#### 1. `save_context`
**Purpose**: Save a Q&A exchange to a thread

**Parameters**:
- `thread_id` (string): Unique identifier for conversation thread
- `query` (string): User's question
- `answer` (string): AI's response
- `framework` (string): Which framework was used

**Example**:
```json
{
  "tool": "save_context",
  "arguments": {
    "thread_id": "user-alice-project-auth",
    "query": "How should I implement JWT tokens?",
    "answer": "Use HS256 with 15min expiry...",
    "framework": "alphacodium"
  }
}
```

#### 2. `get_context`
**Purpose**: Retrieve conversation history for a thread

**Parameters**:
- `thread_id` (string): Thread to retrieve

**Returns**:
```json
{
  "chat_history": [
    {"type": "human", "content": "How should I implement JWT tokens?"},
    {"type": "ai", "content": "Use HS256 with 15min expiry..."}
  ],
  "framework_history": ["alphacodium"]
}
```

### Thread ID Strategy

**Recommended Patterns**:

1. **Per-User Session**: `user-{user_id}-session-{date}`
   - Maintains context within a work session

2. **Per-Project**: `user-{user_id}-project-{project_name}`
   - Maintains context across all work on a specific project

3. **Per-Feature**: `user-{user_id}-feature-{feature_name}`
   - Maintains context for implementing a specific feature

**Example Flow**:
```
Day 1:
- thread_id: "alice-auth-feature"
- Query: "Design JWT auth system"
- Framework: reason_flux
- Save context

Day 2:
- thread_id: "alice-auth-feature" (same!)
- Get context → retrieves Day 1 conversation
- Query: "How do I test the JWT implementation?"
- Framework: tdd_prompting
- Save context
```

### Memory Persistence

**Current Implementation**:
- ✅ In-memory (fast)
- ✅ LRU cache (100 threads max)
- ✅ Per-thread buffer (20 messages max)
- ❌ Not persisted to disk (lost on server restart)

**To Add Disk Persistence** (future enhancement):
- Store in SQLite or Redis
- Load on server start
- Auto-save on updates

---

## RAG System (ChromaDB Vector Search)

### Architecture

```
┌─────────────────────────────────────────┐
│  Codebase Files                         │
│  • Python files                         │
│  • Markdown docs                        │
│  • Config files                         │
└────────────┬────────────────────────────┘
             │ Ingestion
             ▼
┌─────────────────────────────────────────┐
│  ChromaDB Vector Store                  │
│  ┌─────────────────────────────────┐   │
│  │ Collection: frameworks          │   │
│  │ Collection: documentation       │   │
│  │ Collection: configs             │   │
│  │ Collection: utilities           │   │
│  │ Collection: tests               │   │
│  │ Collection: integrations        │   │
│  └─────────────────────────────────┘   │
└────────────┬────────────────────────────┘
             │ Semantic Search
             ▼
┌─────────────────────────────────────────┐
│  RAG Tools (search_*)                   │
└─────────────────────────────────────────┘
```

### Setup RAG

**1. Set API Key** (for embeddings):
```bash
export OPENAI_API_KEY=sk-...
# OR
export OPENROUTER_API_KEY=sk-or-...
```

**2. Ingest Repository**:
```bash
cd omni_cortex
python -m app.ingest_repo
```

This will:
- Scan all Python/Markdown/config files
- Generate embeddings using OpenAI
- Store in ChromaDB at `/app/data/chroma`

### Available RAG Tools

#### 1. `search_documentation`
**Purpose**: General semantic search across all indexed content

**Parameters**:
- `query` (string): What to search for
- `k` (int, optional): Number of results (default: 5)

**Example**:
```json
{
  "tool": "search_documentation",
  "arguments": {
    "query": "how to implement Tree of Thoughts",
    "k": 5
  }
}
```

**Returns**: Top-k matching documents with file paths

#### 2. `search_by_category`
**Purpose**: Search within a specific category

**Parameters**:
- `category` (string): One of `framework`, `documentation`, `config`, `utility`, `test`, `integration`
- `query` (string): Search query
- `k` (int, optional): Number of results

**Example**:
```json
{
  "tool": "search_by_category",
  "arguments": {
    "category": "framework",
    "query": "MCTS implementation",
    "k": 3
  }
}
```

#### 3. `search_function`
**Purpose**: Find a specific function definition

**Parameters**:
- `function_name` (string): Name of function
- `k` (int, optional): Number of results

**Example**:
```json
{
  "tool": "search_function",
  "arguments": {
    "function_name": "create_reasoning_graph",
    "k": 1
  }
}
```

#### 4. `search_class`
**Purpose**: Find a specific class definition

**Parameters**:
- `class_name` (string): Name of class
- `k` (int, optional): Number of results

**Example**:
```json
{
  "tool": "search_class",
  "arguments": {
    "class_name": "OmniCortexMemory"
  }
}
```

#### 5. `search_frameworks_by_name`
**Purpose**: Search within a specific framework's code

**Parameters**:
- `framework_name` (string): Framework name (e.g., "active_inference")
- `query` (string): What to find
- `k` (int, optional): Number of results

#### 6-10. Enhanced Search Tools
- `search_by_file_type`: Search by file extension
- `search_recent`: Search recently modified files
- `search_multi_collection`: Search across multiple collections
- `get_file_context`: Get full file with surrounding context
- `search_related`: Find related code/docs

---

## Code Execution Tool

### `execute_code`
**Purpose**: Run Python code in a sandboxed environment

**Parameters**:
- `code` (string): Python code to execute
- `language` (string, optional): Always "python"

**Safety Features**:
- AST filtering (blocks dangerous imports)
- 5-second timeout
- Restricted environment

**Example**:
```json
{
  "tool": "execute_code",
  "arguments": {
    "code": "result = sum([1, 2, 3, 4, 5])\nprint(f'Sum: {result}')"
  }
}
```

**Returns**:
```json
{
  "success": true,
  "output": "Sum: 15\n",
  "error": null
}
```

---

## Complete Workflow Example

### Scenario: Building an Auth System Across Multiple Sessions

#### Session 1 - Initial Design
```
User: "I need to add JWT authentication to my FastAPI app"

LLM:
1. Calls: reason(query="design JWT auth for FastAPI")
   → Returns: reason_flux template (hierarchical planning)

2. Follows template, creates design

3. Calls: search_documentation(query="FastAPI JWT best practices")
   → Gets relevant code examples

4. Calls: save_context(
     thread_id="user-bob-auth-system",
     query="design JWT auth for FastAPI",
     answer="[Full design plan]",
     framework="reason_flux"
   )
```

#### Session 2 - Implementation (Next Day)
```
User: "Can you help me implement the auth system we designed?"

LLM:
1. Calls: get_context(thread_id="user-bob-auth-system")
   → Retrieves previous design discussion

2. Calls: think_alphacodium(query="implement JWT auth", context="[previous design]")
   → Returns AlphaCodium template (test-based iterative)

3. Follows template, writes code

4. Calls: execute_code(code="[test JWT encoding]")
   → Validates implementation

5. Calls: save_context(
     thread_id="user-bob-auth-system",
     query="implement JWT auth",
     answer="[Implementation code]",
     framework="alphacodium"
   )
```

#### Session 3 - Security Review (Later)
```
User: "Let's make sure this auth is secure"

LLM:
1. Calls: get_context(thread_id="user-bob-auth-system")
   → Has full context: design + implementation

2. Calls: think_chain_of_verification(query="security review JWT auth")
   → Returns verification framework template

3. Calls: think_red_team(query="find vulnerabilities in JWT auth")
   → Returns red-teaming template

4. Follows both frameworks, finds issues

5. Calls: search_documentation(query="JWT security OWASP")
   → Gets security best practices

6. Calls: save_context(...)
```

---

## API Key Requirements

### What Needs API Keys?

| Tool Category | Needs API Key? | Which Key? | Why? |
|--------------|----------------|------------|------|
| Framework tools (`think_*`) | ❌ No | None | Just returns templates |
| Memory tools | ❌ No | None | In-memory storage |
| Code execution | ❌ No | None | Local sandbox |
| RAG/Search tools | ✅ Yes | `OPENAI_API_KEY` or `OPENROUTER_API_KEY` | Embeddings for vector search |

### Minimal Setup (No API Keys)

You can use Omni-Cortex **without any API keys** if you skip RAG:

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

**Available without API keys**:
- ✅ All 40 framework tools
- ✅ Smart routing (reason tool)
- ✅ Memory/context tools
- ✅ Code execution
- ❌ RAG/search tools (need embeddings)

### With RAG Setup

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "python",
      "args": ["-m", "server.main"],
      "cwd": "/path/to/omni_cortex",
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

**Everything available**:
- ✅ All framework tools
- ✅ Memory/context
- ✅ Code execution
- ✅ RAG/search tools

---

## Best Practices

### Thread Management

1. **Use descriptive thread IDs**: `{user}-{project}-{feature}` not just random UUIDs
2. **Reuse threads for related work**: Same feature = same thread
3. **Start new threads for new features**: Don't mix unrelated contexts
4. **Clean up**: Memory evicts automatically (LRU), but be mindful of 100-thread limit

### Memory Usage

1. **Save after significant answers**: Don't save every tiny exchange
2. **Include context in saves**: The more context, the better retrieval
3. **Track framework usage**: Helps understand what approaches worked
4. **Check context before answering**: `get_context` first in multi-session work

### RAG Usage

1. **Ingest regularly**: Re-run ingestion when codebase changes
2. **Use specific searches**: `search_function` > `search_documentation` for finding specific code
3. **Combine with memory**: RAG for code knowledge, memory for conversation history
4. **Filter by category**: Narrow search scope for better results

---

## Troubleshooting

### Memory not persisting across sessions

**Problem**: Context is lost when server restarts
**Cause**: In-memory storage only
**Solution**: Implement disk persistence (SQLite/Redis) or keep server running

### RAG search returns nothing

**Problem**: Search tools return "No results found"
**Causes**:
1. ChromaDB not ingested → Run `python -m app.ingest_repo`
2. Wrong API key → Check `OPENAI_API_KEY` or `OPENROUTER_API_KEY`
3. Query too specific → Try broader terms

### Code execution fails

**Problem**: `execute_code` returns error
**Causes**:
1. Blocked import → AST filter prevents dangerous imports
2. Timeout → Code takes >5 seconds
3. Syntax error → Check code syntax

---

## Architecture Summary

```
┌─────────────────────────────────────────────┐
│         MCP Client (Claude Code)            │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    ▼                             ▼
┌───────────────┐          ┌──────────────────┐
│ Framework     │          │ Utility Tools    │
│ Tools         │          │                  │
│ • think_*     │          │ Memory:          │
│ • reason      │          │ • get_context    │
│               │          │ • save_context   │
│ Returns:      │          │                  │
│ Templates     │          │ RAG:             │
│ (stateless)   │          │ • search_docs    │
│               │          │ • search_*       │
│               │          │                  │
│               │          │ Execution:       │
│               │          │ • execute_code   │
│               │          │                  │
│               │          │ Uses:            │
│               │          │ • LangChain mem  │
│               │          │ • ChromaDB RAG   │
│               │          │ (stateful)       │
└───────────────┘          └──────────────────┘
```

The key insight: **Framework tools are stateless** (just prompt templates), while **utility tools are stateful** (maintain memory/RAG across sessions).
