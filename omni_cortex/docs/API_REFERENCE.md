# Omni-Cortex API Reference

Complete API documentation for all 76 MCP tools exposed by Omni-Cortex.

---

## Overview

Omni-Cortex is a Model Context Protocol (MCP) server that provides 62 specialized reasoning frameworks plus 14 utility tools. It runs as a headless service that IDE agents connect to for advanced reasoning capabilities.

### Connecting to Omni-Cortex

**Docker (recommended):**
```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "vzwjustin/omni-cortex:latest"]
    }
  }
}
```

**Local development:**
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

### Tool Categories

| Category | Count | Description |
|:---------|:-----:|:------------|
| Smart Routing | 1 | Auto-selects best framework |
| Framework Tools | 62 | Direct framework access (`think_*`) |
| Discovery | 2 | List and recommend frameworks |
| Memory | 2 | Conversation context management |
| Search/RAG | 7 | Vector store and code search |
| Execution | 1 | Python code sandbox |
| Health | 1 | Server status |

---

## Smart Routing

### `reason`

Auto-selects the best framework based on task analysis using the HyperRouter. Returns a structured prompt with memory context.

**Parameters:**

| Name | Type | Required | Description |
|:-----|:-----|:--------:|:------------|
| `query` | string | Yes | Your task or question |
| `context` | string | No | Code or additional context |
| `thread_id` | string | No | Thread ID for memory persistence across turns |

**Returns:** Structured brief with selected framework pipeline, confidence score, and execution prompt.

**Example:**
```json
{
  "name": "reason",
  "arguments": {
    "query": "This async handler crashes randomly under load",
    "context": "async def handle_request(req): ...",
    "thread_id": "debug-session-123"
  }
}
```

**Response format:**
```
[self_ask->active_inference->verify_and_edit] conf=85% risk=medium signals=debug,complexity

# Task Analysis
...

# Execution Plan
...
```

---

## Framework Tools

All framework tools follow the same schema. The framework name is prefixed with `think_`.

### Common Parameters

| Name | Type | Required | Description |
|:-----|:-----|:--------:|:------------|
| `query` | string | Yes | Your task or problem |
| `context` | string | No | Code snippet or additional context |
| `thread_id` | string | No | Thread ID for memory persistence across turns |

### Common Response Format

```
# Framework: {name}
Category: {category}
Best for: {use_cases}

---

{structured_prompt_or_result}
```

---

### Strategy Frameworks

#### `think_reason_flux`
Hierarchical planning: Template -> Expand -> Refine. Best for architecture, system design, complex planning.

#### `think_self_discover`
Discover and apply reasoning patterns. Best for novel problems, unknown domains.

#### `think_buffer_of_thoughts`
Build context in a thought buffer. Best for multi-part problems, complex context.

#### `think_coala`
Cognitive architecture for agents. Best for autonomous tasks, agent behavior.

#### `think_least_to_most`
Bottom-up atomic function decomposition. Best for complex systems, monolith refactoring, dependency management.

#### `think_comparative_arch`
Multi-approach comparison (readability/memory/speed). Best for optimization, architecture decisions, trade-off analysis.

#### `think_plan_and_solve`
Explicit planning before execution. Best for complex features, methodical development, avoiding rushed code.

---

### Search Frameworks

#### `think_mcts_rstar`
Monte Carlo Tree Search exploration for code. Best for complex bugs, multi-step optimization, thorough search.

#### `think_tree_of_thoughts`
Explore multiple paths, pick best. Best for design decisions, multiple valid approaches.

#### `think_graph_of_thoughts`
Non-linear reasoning with idea graphs. Best for complex dependencies, interconnected problems.

#### `think_everything_of_thought`
Combine multiple reasoning approaches. Best for complex novel problems, when one approach isn't enough.

---

### Iterative Frameworks

#### `think_active_inference`
Hypothesis testing loop. Best for debugging, investigation, root cause analysis.

#### `think_multi_agent_debate`
Multiple perspectives debate. Best for design decisions, trade-off analysis.

#### `think_adaptive_injection`
Inject strategies as needed. Best for evolving understanding, adaptive problem solving.

#### `think_re2`
Read-Execute-Evaluate loop. Best for specifications, requirements implementation.

#### `think_rubber_duck`
Socratic questioning for self-discovery. Best for architectural issues, blind spots, stuck problems.

#### `think_react`
Reasoning + Acting with tools. Best for multi-step tasks, tool use, investigation.

#### `think_reflexion`
Self-evaluation with memory-based learning. Best for learning from failures, iterative improvement, retry scenarios.

#### `think_self_refine`
Iterative self-critique and improvement. Best for code quality, documentation, polish.

---

### Code Frameworks

#### `think_program_of_thoughts`
Step-by-step code reasoning. Best for algorithms, data processing, math.

#### `think_chain_of_verification`
Draft-Verify-Patch cycle. Best for security review, code quality, bug prevention.

#### `think_critic`
Generate then critique. Best for API design, interface validation.

#### `think_chain_of_code`
Code-based problem decomposition. Best for logic puzzles, algorithmic debugging, structured thinking.

#### `think_self_debugging`
Mental execution trace before presenting. Best for preventing bugs, edge case handling, off-by-one errors.

#### `think_tdd_prompting`
Test-first development approach. Best for new features, edge case coverage, robust implementations.

#### `think_reverse_cot`
Backward reasoning from output delta. Best for silent bugs, wrong outputs, calculation errors.

#### `think_alphacodium`
Test-based multi-stage iterative code generation. Best for competitive programming, complex algorithms, interview problems.

#### `think_codechain`
Chain of self-revisions guided by sub-modules. Best for modular code generation, incremental refinement, component development.

#### `think_evol_instruct`
Evolutionary instruction complexity for code. Best for challenging code problems, constraint-based coding, extending solutions.

#### `think_llmloop`
Automated iterative feedback loops for code+tests. Best for code quality assurance, production-ready code, CI/CD preparation.

#### `think_procoder`
Compiler-feedback-guided iterative refinement. Best for project-level code generation, API usage, type-safe code.

#### `think_recode`
Multi-candidate validation with CFG-based debugging. Best for reliable code generation, high-stakes code, mission-critical systems.

#### `think_pal`
Program-Aided Language - code as reasoning substrate. Best for algorithms, parsing, numeric logic, validation.

#### `think_scratchpads`
Structured intermediate reasoning workspace. Best for multi-step fixes, multi-constraint reasoning, state tracking.

#### `think_parsel`
Compositional code generation from natural language specs. Best for complex functions, dependency graphs, spec-to-code, modular systems.

#### `think_docprompting`
Documentation-driven code generation. Best for API usage, library integration, following docs, correct usage.

---

### Context Frameworks

#### `think_chain_of_note`
Research and note-taking approach. Best for understanding code, documentation, exploration.

#### `think_step_back`
Abstract principles first, then apply. Best for optimization, performance, architectural decisions.

#### `think_analogical`
Find and adapt similar solutions. Best for creative solutions, pattern matching.

#### `think_red_team`
Adversarial security analysis (STRIDE, OWASP). Best for security audits, vulnerability scanning, threat modeling.

#### `think_state_machine`
Formal FSM design before coding. Best for UI logic, workflow systems, game states.

#### `think_chain_of_thought`
Step-by-step reasoning chain. Best for complex reasoning, logical deduction, problem solving.

---

### Fast Frameworks

#### `think_skeleton_of_thought`
Outline first, fill in details. Best for boilerplate, quick scaffolding.

#### `think_system1`
Fast intuitive response. Best for simple questions, quick fixes.

---

### Verification Frameworks

#### `think_self_consistency`
Multi-sample voting for reliable answers. Best for ambiguous bugs, tricky logic, multiple plausible fixes.

#### `think_self_ask`
Sub-question decomposition before solving. Best for unclear tickets, missing requirements, multi-part debugging.

#### `think_rar`
Rephrase-and-Respond for clarity. Best for vague prompts, poorly written bug reports, ambiguous requirements.

#### `think_verify_and_edit`
Verify claims, edit only failures. Best for code review, security guidance, implementation plans, surgical edits.

#### `think_rarr`
Research, Augment, Revise - evidence-driven revision. Best for external docs, repo knowledge, prove-it requirements.

#### `think_selfcheckgpt`
Hallucination detection via sampling consistency. Best for high-stakes guidance, unfamiliar libraries, final pre-flight gate.

#### `think_metaqa`
Metamorphic testing for reasoning reliability. Best for brittle reasoning, edge cases, policy consistency.

#### `think_ragas`
RAG Assessment - evaluate retrieval quality. Best for RAG pipelines, retrieval quality, source grounding.

---

### Agent Frameworks

#### `think_rewoo`
Reasoning Without Observation - plan then execute. Best for multi-step tasks, cost control, plan-once-execute-clean.

#### `think_lats`
Language Agent Tree Search over action sequences. Best for complex repo changes, multiple fix paths, uncertain root cause.

#### `think_mrkl`
Modular Reasoning with specialized modules. Best for big systems, mixed domains, tool-rich setups.

#### `think_swe_agent`
Repo-first execution loop - inspect/edit/run/iterate. Best for multi-file bugfixes, CI failures, make tests pass.

#### `think_toolformer`
Smart tool selection policy. Best for router logic, preventing pointless calls, standardized prompts.

---

### RAG Frameworks

#### `think_self_rag`
Self-triggered selective retrieval. Best for mixed knowledge tasks, large corpora, minimizing irrelevant retrieval.

#### `think_hyde`
Hypothetical Document Embeddings for better retrieval. Best for fuzzy search, unclear intent, broad problems.

#### `think_rag_fusion`
Multi-query retrieval with rank fusion. Best for improving recall, complex queries, noisy corpora.

#### `think_raptor`
Hierarchical abstraction retrieval for large docs. Best for huge repos, long design docs, monorepos.

#### `think_graphrag`
Entity-relation grounding for dependencies. Best for architecture questions, module relationships, impact analysis.

---

## Discovery Tools

### `list_frameworks`

List all 62 thinking frameworks organized by category.

**Parameters:** None

**Returns:** Markdown-formatted list of all frameworks grouped by category with descriptions.

**Example:**
```json
{
  "name": "list_frameworks",
  "arguments": {}
}
```

---

### `recommend`

Get a framework recommendation based on task description.

**Parameters:**

| Name | Type | Required | Description |
|:-----|:-----|:--------:|:------------|
| `task` | string | Yes | Description of your task |

**Returns:** Recommended framework with description and best-for use cases.

**Example:**
```json
{
  "name": "recommend",
  "arguments": {
    "task": "debug a race condition in async code"
  }
}
```

**Response:**
```
Recommended: `think_active_inference`

Hypothesis testing loop. Best for debugging, investigation, root cause analysis.
```

---

## Memory Tools

### `get_context`

Retrieve conversation history and framework usage for a thread.

**Parameters:**

| Name | Type | Required | Description |
|:-----|:-----|:--------:|:------------|
| `thread_id` | string | Yes | Thread ID to get context for |

**Returns:** JSON object containing chat history and framework usage history.

**Example:**
```json
{
  "name": "get_context",
  "arguments": {
    "thread_id": "session-abc-123"
  }
}
```

**Response:**
```json
{
  "chat_history": [...],
  "framework_history": ["active_inference", "verify_and_edit"],
  "summary": "..."
}
```

---

### `save_context`

Save a query-answer exchange to memory for a thread.

**Parameters:**

| Name | Type | Required | Description |
|:-----|:-----|:--------:|:------------|
| `thread_id` | string | Yes | Thread ID to save to |
| `query` | string | Yes | The original query |
| `answer` | string | Yes | The generated answer |
| `framework` | string | Yes | The framework used |

**Returns:** Confirmation message.

**Example:**
```json
{
  "name": "save_context",
  "arguments": {
    "thread_id": "session-abc-123",
    "query": "Fix the null pointer exception",
    "answer": "The issue is in line 42...",
    "framework": "active_inference"
  }
}
```

---

## Search Tools

### `search_documentation`

Search indexed documentation and code via vector store (RAG).

**Parameters:**

| Name | Type | Required | Default | Description |
|:-----|:-----|:--------:|:-------:|:------------|
| `query` | string | Yes | - | Search query |
| `k` | integer | No | 5 | Number of results to return |

**Returns:** Formatted search results with file paths and content excerpts.

**Example:**
```json
{
  "name": "search_documentation",
  "arguments": {
    "query": "how to configure memory persistence",
    "k": 3
  }
}
```

---

### `search_frameworks_by_name`

Search within a specific framework's implementation.

**Parameters:**

| Name | Type | Required | Default | Description |
|:-----|:-----|:--------:|:-------:|:------------|
| `framework_name` | string | Yes | - | Framework to search (e.g., 'active_inference') |
| `query` | string | Yes | - | Search query |
| `k` | integer | No | 3 | Number of results |

**Returns:** Code snippets from the specified framework.

**Example:**
```json
{
  "name": "search_frameworks_by_name",
  "arguments": {
    "framework_name": "active_inference",
    "query": "hypothesis generation"
  }
}
```

---

### `search_by_category`

Search within a code category.

**Parameters:**

| Name | Type | Required | Default | Description |
|:-----|:-----|:--------:|:-------:|:------------|
| `query` | string | Yes | - | Search query |
| `category` | string | Yes | - | One of: `framework`, `documentation`, `config`, `utility`, `test`, `integration` |
| `k` | integer | No | 5 | Number of results |

**Returns:** Matching code from the specified category.

**Example:**
```json
{
  "name": "search_by_category",
  "arguments": {
    "query": "router",
    "category": "utility"
  }
}
```

---

### `search_function`

Find specific function implementations by name.

**Parameters:**

| Name | Type | Required | Default | Description |
|:-----|:-----|:--------:|:-------:|:------------|
| `function_name` | string | Yes | - | Function name to search |
| `k` | integer | No | 3 | Number of results |

**Returns:** Function implementations matching the name.

**Example:**
```json
{
  "name": "search_function",
  "arguments": {
    "function_name": "get_memory"
  }
}
```

---

### `search_class`

Find specific class implementations by name.

**Parameters:**

| Name | Type | Required | Default | Description |
|:-----|:-----|:--------:|:-------:|:------------|
| `class_name` | string | Yes | - | Class name to search |
| `k` | integer | No | 3 | Number of results |

**Returns:** Class implementations matching the name.

**Example:**
```json
{
  "name": "search_class",
  "arguments": {
    "class_name": "HyperRouter"
  }
}
```

---

### `search_docs_only`

Search only markdown documentation files.

**Parameters:**

| Name | Type | Required | Default | Description |
|:-----|:-----|:--------:|:-------:|:------------|
| `query` | string | Yes | - | Search query |
| `k` | integer | No | 5 | Number of results |

**Returns:** Matching documentation excerpts.

**Example:**
```json
{
  "name": "search_docs_only",
  "arguments": {
    "query": "framework chaining"
  }
}
```

---

### `search_framework_category`

Search within a framework category.

**Parameters:**

| Name | Type | Required | Default | Description |
|:-----|:-----|:--------:|:-------:|:------------|
| `query` | string | Yes | - | Search query |
| `framework_category` | string | Yes | - | One of: `strategy`, `search`, `iterative`, `code`, `context`, `fast`, `verification`, `agent`, `rag` |
| `k` | integer | No | 5 | Number of results |

**Returns:** Framework implementations in the specified category.

**Example:**
```json
{
  "name": "search_framework_category",
  "arguments": {
    "query": "debugging loop",
    "framework_category": "iterative"
  }
}
```

---

## Execution Tools

### `execute_code`

Execute Python code in a sandboxed environment.

**Parameters:**

| Name | Type | Required | Default | Description |
|:-----|:-----|:--------:|:-------:|:------------|
| `code` | string | Yes | - | Python code to execute |
| `language` | string | No | python | Language (only 'python' supported) |

**Returns:** JSON with execution result or error.

**Example:**
```json
{
  "name": "execute_code",
  "arguments": {
    "code": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n\nprint(factorial(5))"
  }
}
```

**Response:**
```json
{
  "success": true,
  "output": "120\n",
  "error": null
}
```

**Security notes:**
- 5 second timeout
- Dangerous imports blocked (os, subprocess, etc.)
- AST-based pattern filtering

---

## Health Check

### `health`

Check server health and available capabilities.

**Parameters:** None

**Returns:** JSON status object.

**Example:**
```json
{
  "name": "health",
  "arguments": {}
}
```

**Response:**
```json
{
  "status": "healthy",
  "frameworks": 62,
  "tools": 76,
  "collections": ["frameworks", "documentation", "configs", "utilities", "tests", "integrations"],
  "memory_enabled": true,
  "rag_enabled": true
}
```

---

## Best Practices

### Vibe-Based Routing

Instead of manually selecting frameworks, use natural language with the `reason` tool:

| You Say | Selected Framework |
|:--------|:-------------------|
| "WTF is wrong with this?" | `active_inference` |
| "This code is spaghetti" | `graph_of_thoughts` |
| "Is this secure?" | `chain_of_verification` |
| "I have no idea how to start" | `self_discover` |
| "Make it faster" | `tree_of_thoughts` |

### Memory Persistence

For multi-turn debugging sessions, always provide a consistent `thread_id`:

```json
{
  "name": "think_active_inference",
  "arguments": {
    "query": "The bug is still happening after my fix",
    "thread_id": "debug-session-123"
  }
}
```

### Context is Key

Always provide relevant code in the `context` parameter:

```json
{
  "name": "reason",
  "arguments": {
    "query": "Why does this timeout randomly?",
    "context": "async def fetch_data():\n    async with aiohttp.ClientSession() as session:\n        return await session.get(url, timeout=30)"
  }
}
```

---

## Error Handling

All tools return `TextContent` responses. Errors are returned as text with descriptive messages:

- **Unknown tool:** `"Unknown tool: {name}"`
- **Missing arguments:** `"Missing required arguments: {list}"`
- **Search failures:** `"Search failed: {error}"`
- **No results:** `"No results found. Try refining your query."`
- **Invalid category:** `"Invalid category. Use: {valid_options}"`
