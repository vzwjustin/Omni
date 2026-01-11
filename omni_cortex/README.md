# Omni-Cortex

Headless MCP server routing tasks to 62 AI reasoning frameworks via LangGraph/LangChain.

## Overview

Omni-Cortex is an intelligent reasoning orchestrator that helps AI coding assistants think more effectively. It provides:

- **62 Thinking Frameworks** - From debugging (Active Inference) to architecture (ReasonFlux) to optimization (Tree of Thoughts)
- **Smart Routing** - HyperRouter automatically selects the best framework based on your task
- **Memory Persistence** - LangChain memory for conversation continuity across sessions
- **RAG Search** - ChromaDB-powered search across documentation and code

## Quick Start

### Docker (Recommended)

```bash
# Clone and setup
git clone https://github.com/vzwjustin/Omni.git
cd Omni

# Configure environment
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY

# Build and run
docker-compose up -d

# View logs
docker-compose logs -f omni-cortex
```

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run server
python -m server.main
```

### Single Command Setup

```bash
./setup.sh
```

## Configuration

Key environment variables (see `.env.example` for full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `LEAN_MODE` | `true` | 4 essential tools vs 76 full tools |
| `LLM_PROVIDER` | `pass-through` | `google`, `openrouter`, `anthropic`, `openai` |
| `GOOGLE_API_KEY` | - | Required for Gemini routing |
| `EMBEDDING_PROVIDER` | `gemini` | `gemini`, `openai`, `huggingface` |

## Architecture

```
MCP Client (Claude Code, etc.)
    ↓
server/main.py (MCP Server)
    ↓
HyperRouter (Framework Selection)
    ↓
LangGraph (Workflow Orchestration)
    ↓
Framework Nodes (62 frameworks in app/nodes/)
    ↓
Response (via MCP)
```

### Components

| Component | Purpose |
|-----------|---------|
| **LangGraph** | Workflow orchestration, state management |
| **LangChain Memory** | Conversation history per thread |
| **ChromaDB** | Vector store for RAG (6 collections) |
| **HyperRouter** | AI-powered framework selection |

## Framework Categories

| Category | Count | Examples |
|----------|-------|----------|
| Strategy | 7 | ReasonFlux, Self-Discover, CoALA |
| Search | 4 | MCTS, Tree of Thoughts, Graph of Thoughts |
| Iterative | 8 | Active Inference, ReAct, Reflexion |
| Code | 17 | TDD Prompting, AlphaCodium, Chain of Code |
| Context | 6 | Chain of Note, Step-Back, Red-Teaming |
| Fast | 2 | Skeleton of Thought, System1 |
| Verification | 8 | Self-Consistency, Verify-and-Edit |

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app --cov=server

# Run only unit tests
pytest tests/unit/ -m unit
```

## Documentation

- [AGENTS.md](AGENTS.md) - Development guidelines for AI agents
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture details
- [FRAMEWORKS.md](FRAMEWORKS.md) - Complete framework reference
- [MCP_SETUP.md](MCP_SETUP.md) - MCP client configuration

## License

MIT License - see [LICENSE](LICENSE)
