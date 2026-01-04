# Omni-Cortex: 40 Thinking Frameworks MCP Server

A headless MCP server exposing 40 thinking frameworks for AI-assisted coding. Pass-through architecture - your IDE's LLM does the reasoning, Omni-Cortex provides the structured prompts.

## Overview

Omni-Cortex provides cognitive scaffolding via **Model Context Protocol (MCP)**. When your IDE agent calls a framework tool, it receives a token-optimized prompt that guides its reasoning process.

**Built for vibe coders** - describe what you want naturally, the router picks the right framework.

## Quick Start

### Automated Setup (Recommended)

```bash
cd omni_cortex
./setup.sh
```

This will:
- Prompt for your OpenAI API key (for embeddings)
- Create `.env` file
- Build Docker image
- Configure MCP for Claude Code

### Manual Setup

```bash
# Copy environment file
cp .env.example .env
# Edit .env with your OpenAI API key

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

## MCP Client Configuration

After running `setup.sh`, MCP is auto-configured for Claude Code.

For other IDEs, see **[MCP_SETUP.md](MCP_SETUP.md)** for copy-paste configs:
- Claude Code
- Cursor
- VS Code (Cline)
- Windsurf
- Generic MCP clients

**Quick example (Claude Code):**
```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--env-file", "/path/to/omni_cortex/.env",
        "-v", "/path/to/omni_cortex/data:/app/data",
        "omni_cortex-omni-cortex:latest"
      ]
    }
  }
}
```

## Vibe Coding - Just Say What You Want

| What You Say | Framework Selected |
|--------------|-------------------|
| "wtf is wrong with this" | `active_inference` |
| "clean this up" | `graph_of_thoughts` |
| "make it faster" | `tree_of_thoughts` |
| "pros and cons" | `multi_agent_debate` |
| "quick fix" | `system1` |
| "major rewrite" | `everything_of_thought` |
| "is this secure" | `chain_of_verification` |
| "write tests first" | `tdd_prompting` |

## 55 Tools Available

### 40 Framework Tools (`think_*`)

| Category | Frameworks |
|----------|------------|
| **Strategy** | `reason_flux`, `self_discover`, `buffer_of_thoughts`, `coala`, `least_to_most`, `comparative_arch`, `plan_and_solve` |
| **Search** | `mcts_rstar`, `tree_of_thoughts`, `graph_of_thoughts`, `everything_of_thought` |
| **Iterative** | `active_inference`, `multi_agent_debate`, `adaptive_injection`, `re2`, `rubber_duck`, `react`, `reflexion`, `self_refine` |
| **Code** | `program_of_thoughts`, `chain_of_verification`, `critic`, `chain_of_code`, `self_debugging`, `tdd_prompting`, `reverse_cot`, `alphacodium`, `codechain`, `evol_instruct`, `llmloop`, `procoder`, `recode` |
| **Context** | `chain_of_note`, `step_back`, `analogical`, `red_team`, `state_machine`, `chain_of_thought` |
| **Fast** | `skeleton_of_thought`, `system1` |

### 15 Utility Tools

| Tool | Purpose |
|------|---------|
| `reason` | Smart auto-routing to best framework |
| `list_frameworks` | List all 40 frameworks by category |
| `recommend` | Get framework recommendation |
| `get_context` | Retrieve conversation memory |
| `save_context` | Save to memory |
| `search_documentation` | RAG search via ChromaDB |
| `search_frameworks_by_name` | Search specific framework |
| `search_by_category` | Search by code category |
| `search_function` | Find function by name |
| `search_class` | Find class by name |
| `search_docs_only` | Search markdown docs |
| `search_framework_category` | Search framework category |
| `execute_code` | Sandboxed Python execution |
| `health` | Health check |

## Architecture

```
Pass-Through Mode (No API keys needed for frameworks)
=====================================================

IDE Agent (Claude Code, Cursor, etc.)
         |
         | calls think_active_inference(query="debug this")
         v
+------------------+
|  Omni-Cortex MCP |
|------------------|
| Returns prompt:  |
| [active_inference] Hypothesis testing loop
| TASK:{query}|CTX:{context}
| 1.OBSERVE: Current state, form hypotheses
| 2.PREDICT: Expected behavior if hypothesis true
| 3.TEST: Gather evidence, update beliefs
| 4.ACT: Implement fix
| 5.VERIFY: Confirm fix worked
+------------------+
         |
         v
IDE Agent executes the prompt with its own LLM
```

## Token Efficiency

Prompts are optimized to minimize token usage:

| Component | Tokens |
|-----------|--------|
| Framework header | ~12 |
| Prompt body | ~50-70 |
| **Total per call** | **~60-80** |

Compared to verbose alternatives (~150+ tokens), this saves ~50% on every framework call.

## Components

| Component | Purpose |
|-----------|---------|
| **LangGraph** | Workflow orchestration, 40 framework nodes |
| **LangChain Memory** | Conversation history per thread |
| **ChromaDB** | Vector store for RAG, 6 collections |
| **HyperRouter** | Vibe-based framework selection |

## Configuration

```bash
# .env file
LLM_PROVIDER=pass-through          # Server doesn't call LLMs
EMBEDDING_PROVIDER=openai          # For RAG/ChromaDB
OPENAI_API_KEY=sk-...              # Required for embeddings
ENABLE_AUTO_INGEST=true            # Index repo on startup
```

## Project Structure

```
omni_cortex/
├── app/
│   ├── state.py              # GraphState
│   ├── graph.py              # LangGraph workflow
│   ├── langchain_integration.py  # Memory + RAG
│   ├── collection_manager.py # ChromaDB collections
│   ├── core/
│   │   ├── config.py         # Settings
│   │   └── router.py         # HyperRouter
│   └── nodes/
│       ├── common.py         # Shared utilities
│       ├── strategy/         # 7 frameworks
│       ├── search/           # 4 frameworks
│       ├── iterative/        # 8 frameworks
│       ├── code/             # 13 frameworks
│       ├── context/          # 6 frameworks
│       └── fast/             # 2 frameworks
├── server/
│   └── main.py               # MCP server (55 tools)
├── setup.sh                  # Automated setup
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## License

MIT
