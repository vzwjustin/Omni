# Omni-Cortex: Headless AI Reasoning MCP Server

A headless AI microservice that acts as a "Brain" for IDE agents, routing complex coding tasks to 20 specialized reasoning frameworks.

## 🧠 Overview

Omni-Cortex exposes cognitive reasoning capabilities via the **Model Context Protocol (MCP)**, enabling seamless integration with IDE agents like Cursor, Claude Code, and any MCP-compatible client.

**Built for vibe coders** - just describe what you want naturally, and the AI picks the right framework.

## ✨ Vibe Coding - Just Say What You Want

Don't know which framework to use? Just describe the vibe:

| What You Say | Framework Selected |
|--------------|-------------------|
| "wtf is wrong with this", "why is this broken" | Active Inference (debugging) |
| "clean this up", "this code is ugly" | Graph of Thoughts (refactoring) |
| "make it faster", "too slow" | Tree of Thoughts (optimization) |
| "pros and cons", "should I use A or B" | Multi-Agent Debate (decisions) |
| "quick question", "easy fix" | System1 (fast mode) |
| "major rewrite", "big migration" | Everything of Thought (deep) |
| "I have no idea", "weird problem" | Self-Discover (exploration) |
| "is this secure", "audit this" | Chain of Verification (security) |
| "just generate", "boilerplate" | Skeleton of Thought (fast gen) |

## 🚀 Quick Start

### Docker (Recommended)

```bash
# Clone and navigate
cd omni_cortex

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f omni-cortex
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Run the server
python -m server.main
```

## 🔧 MCP Client Configuration

Add to your MCP client configuration:

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

## 📚 Available Frameworks (20)

### Strategy & Hierarchical
| Framework | Best For |
|-----------|----------|
| **ReasonFlux** | Architecture, system design, planning |
| **Self-Discover** | Novel problems, unclear requirements |
| **Buffer-of-Thoughts** | Repetitive tasks, known patterns |
| **CoALA** | Long-context, multi-file, stateful tasks |

### Search & Exploration
| Framework | Best For |
|-----------|----------|
| **rStar-Code MCTS** | Complex bugs, optimization |
| **Tree of Thoughts** | Algorithms, problem solving |
| **Graph of Thoughts** | Refactoring, restructuring |
| **Everything-of-Thought** | Complex migrations |

### Iterative & Adversarial
| Framework | Best For |
|-----------|----------|
| **Active Inference** | Debugging, root cause analysis |
| **Multi-Agent Debate** | Design decisions, trade-offs |
| **Adaptive Injection** | Variable complexity tasks |
| **RE2 (Re-Reading)** | Complex specifications |

### Code & Verification
| Framework | Best For |
|-----------|----------|
| **Program of Thoughts** | Math, data processing, testing |
| **Chain of Verification** | Security, code review |
| **CRITIC** | API usage, library integration |

### Context & Research
| Framework | Best For |
|-----------|----------|
| **Chain-of-Note** | Research, documentation |
| **Step-Back** | Performance, complexity |
| **Analogical** | Creative solutions, patterns |

### Fast Execution
| Framework | Best For |
|-----------|----------|
| **Skeleton-of-Thought** | Docs, boilerplate, scaffolding |
| **System1** | Simple queries, quick fixes |

## 🛠 MCP Tools

### Two Access Patterns

**Pattern 1: Smart Router (Recommended)**
- Use `reason` tool - AI automatically selects the best framework
- Best for: "vibe coding" where you describe what you want naturally

**Pattern 2: Direct Framework Access**
- Use `fw_{framework_name}` tools - call specific frameworks directly
- Best for: sequential multi-agent workflows, explicit framework control
- Example: `fw_active_inference`, `fw_tree_of_thoughts`, `fw_mcts_rstar`

### Main Tools

#### `reason` (Smart Router)
Automatically routes to optimal framework based on task analysis.

```json
{
    "query": "Debug this null pointer exception",
    "code_snippet": "def foo(): return x.bar()",
    "file_list": ["main.py"],
    "ide_context": "Python 3.11 project",
    "preferred_framework": "active_inference",
    "max_iterations": 5,
    "thread_id": "optional-for-continuity"
}
```

#### `fw_{framework_name}` (Direct Access)
Call any of the 20 frameworks directly. Same input schema as `reason`.

Available:
- `fw_reason_flux`, `fw_self_discover`, `fw_buffer_of_thoughts`, `fw_coala`
- `fw_mcts_rstar`, `fw_tree_of_thoughts`, `fw_graph_of_thoughts`, `fw_everything_of_thought`
- `fw_active_inference`, `fw_multi_agent_debate`, `fw_adaptive_injection`, `fw_re2`
- `fw_program_of_thoughts`, `fw_chain_of_verification`, `fw_critic`
- `fw_chain_of_note`, `fw_step_back`, `fw_analogical`
- `fw_skeleton_of_thought`, `fw_system1`

#### `list_frameworks`
List all 20 frameworks with descriptions and best-use cases.

#### `health`
Server health check and statistics.

### LangChain Tools (RAG & Code Execution)

#### `search_documentation`
Search indexed documentation/code via vector store (legacy).

#### `execute_code`
Execute Python code in sandboxed environment.

#### `retrieve_context`
Retrieve recent conversation/framework history.

### Enhanced Vector Search Tools (Production Schema)

**New in Enhanced Schema**: 6 specialized search tools with rich metadata filtering.

#### `search_frameworks_by_name`
Search within a specific framework's implementation.

#### `search_by_category`
Search by code category (framework, documentation, config, utility, test, integration).

#### `search_function_implementation`
Find function implementations by name with AST metadata.

#### `search_class_implementation`
Find class implementations by name with structure info.

#### `search_documentation_only`
Search only markdown documentation with section-level chunks.

#### `search_with_framework_context`
Search within framework categories (strategy, search, iterative, code, context, fast).

**See `ENHANCED_VECTOR_SCHEMA.md` for complete details on the production-grade vector database.**

## ⚙️ Configuration

### Provider Selection

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `openrouter`, `anthropic`, or `openai` | openrouter |

### API Keys

| Variable | Description | When Required |
|----------|-------------|---------------|
| `OPENROUTER_API_KEY` | OpenRouter API key (access all models) | When `LLM_PROVIDER=openrouter` |
| `ANTHROPIC_API_KEY` | Anthropic API key (Claude models) | When `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | OpenAI API key (GPT models) | When `LLM_PROVIDER=openai` |

### Model Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEP_REASONING_MODEL` | Model for deep reasoning | anthropic/claude-4.5-sonnet |
| `FAST_SYNTHESIS_MODEL` | Model for fast generation | openai/gpt-5.2 |
| `ENABLE_DSPY_OPTIMIZATION` | DSPy prompt optimization | true |
| `ENABLE_PRM_SCORING` | Process Reward Model | true |

> **Note**: When using OpenRouter, model names use the format `provider/model-name`. When using direct APIs, just use `model-name`.

## 🏗 Architecture

```
┌─────────────────────────────────────────────────┐
│                 MCP Client (IDE)                │
└─────────────────────┬───────────────────────────┘
                      │ stdio/SSE
┌─────────────────────▼───────────────────────────┐
│              Omni-Cortex MCP Server             │
├─────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────────────────┐  │
│  │   Router    │──│   Framework Dispatcher   │  │
│  └─────────────┘  └──────────────────────────┘  │
├─────────────────────────────────────────────────┤
│  ┌─────────┐ ┌────────┐ ┌─────────┐ ┌────────┐  │
│  │Strategy │ │ Search │ │Iterative│ │  Code  │  │
│  └─────────┘ └────────┘ └─────────┘ └────────┘  │
│  ┌─────────┐ ┌────────┐                         │
│  │ Context │ │  Fast  │                         │
│  └─────────┘ └────────┘                         │
├─────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────┐  │
│  │            OpenRouter (Unified)              │  │
│  │  ┌─────────────────┐  ┌──────────────────┐  │  │
│  │  │  Claude Sonnet  │  │     GPT-5.2      │  │  │
│  │  │ (Deep Thinking) │  │ (Fast Synthesis) │  │  │
│  │  └─────────────────┘  └──────────────────┘  │  │
│  └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
omni_cortex/
├── app/
│   ├── __init__.py
│   ├── schemas.py          # Pydantic models
│   ├── state.py            # GraphState management
│   ├── core/
│   │   ├── config.py       # Settings & model clients
│   │   └── router.py       # Hyper-Dispatcher
│   └── nodes/
│       ├── common.py       # Quiet-STaR, PRM, DSPy
│       ├── strategy/       # 4 frameworks
│       ├── search/         # 4 frameworks
│       ├── iterative/      # 4 frameworks
│       ├── code/           # 3 frameworks
│       ├── context/        # 3 frameworks
│       └── fast/           # 2 frameworks
├── server/
│   └── main.py             # MCP server entry
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 📜 License

MIT
