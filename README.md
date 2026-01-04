# Omni-Cortex: AI Thinking Frameworks MCP Server

An MCP (Model Context Protocol) server that provides 20 advanced reasoning frameworks for AI assistants. Built with LangGraph for orchestration and LangChain for memory/RAG capabilities.

## Overview

Omni-Cortex exposes specialized thinking frameworks as MCP tools, allowing AI assistants to apply structured reasoning strategies for different types of tasks. The server itself doesn't call LLMs - it provides prompts and orchestration while the calling AI (Claude, GPT, etc.) does the actual reasoning.

## 🧠 Available Frameworks (20 Total)

### Strategy (4 frameworks)
- **ReasonFlux** - Hierarchical planning: Template → Expand → Refine
- **Self-Discover** - Discover and compose custom reasoning patterns
- **Buffer of Thoughts** - Build context in a thought buffer
- **CoALA** - Cognitive architecture with memory systems

### Search (4 frameworks)
- **rStar-Code MCTS** - Monte Carlo Tree Search for code exploration
- **Tree of Thoughts** - Explore multiple solution paths, pick best
- **Graph of Thoughts** - Non-linear reasoning with idea graphs
- **Everything of Thought** - Combine multiple reasoning approaches

### Iterative (4 frameworks)
- **Active Inference** - Hypothesis testing loop for debugging
- **Multi-Agent Debate** - Multiple perspectives argue trade-offs
- **Adaptive Injection** - Inject strategies as needed
- **RE2** - Read-Execute-Evaluate loop for requirements

### Code (3 frameworks)
- **Program of Thoughts** - Generate executable code to solve problems
- **Chain of Verification** - Draft → Verify → Patch cycle
- **CRITIC** - Generate then critique with external validation

### Context (3 frameworks)
- **Chain of Note** - Research and note-taking approach
- **Step-Back** - Abstract principles first, then apply
- **Analogical** - Find and adapt similar solutions

### Fast (2 frameworks)
- **Skeleton of Thought** - Outline first, fill in details
- **System1** - Quick intuitive responses

## 🎯 Key Features

- **Smart Routing**: Auto-selects optimal framework based on task analysis
- **Vibe Dictionary**: Natural language activation ("wtf is broken" → Active Inference)
- **Memory Systems**: LangChain-powered conversation history and framework tracking
- **RAG Integration**: ChromaDB vector store with 6 specialized collections
- **Tool Integration**: Execute code, search docs, retrieve context
- **Quiet-STaR**: Internal thought processes for enhanced reasoning

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              MCP Client (Claude/Cursor)              │
└───────────────────┬─────────────────────────────────┘
                    │ MCP Protocol
┌───────────────────▼─────────────────────────────────┐
│           Omni-Cortex MCP Server                     │
│  ┌────────────────────────────────────────────┐     │
│  │  35 MCP Tools                               │     │
│  │  • 20 think_* framework tools               │     │
│  │  • 1 reason (smart routing)                 │     │
│  │  • 14 utility tools (search, memory, etc)   │     │
│  └────────────────┬───────────────────────────┘     │
│                   │                                   │
│  ┌────────────────▼───────────────────────────┐     │
│  │  HyperRouter (AI-powered selection)         │     │
│  │  • Vibe Dictionary (casual phrase matching) │     │
│  │  • LLM analysis (complex selection)         │     │
│  │  • Heuristic fallback                       │     │
│  └────────────────┬───────────────────────────┘     │
│                   │                                   │
│  ┌────────────────▼───────────────────────────┐     │
│  │  LangGraph Workflow                         │     │
│  │  • Route Node (framework selection)         │     │
│  │  • Execute Node (run framework)             │     │
│  │  • Checkpointing (SQLite)                   │     │
│  └────────────────┬───────────────────────────┘     │
│                   │                                   │
│  ┌────────────────▼───────────────────────────┐     │
│  │  20 Framework Nodes                         │     │
│  │  • Each implements specific strategy         │     │
│  │  • PRM scoring for search algorithms        │     │
│  │  • Tool integration where needed            │     │
│  └────────────────┬───────────────────────────┘     │
│                   │                                   │
│  ┌────────────────▼───────────────────────────┐     │
│  │  LangChain Integration                      │     │
│  │  • Memory (conversation history)            │     │
│  │  • Tools (code exec, search)                │     │
│  │  • RAG (ChromaDB vector store)              │     │
│  └─────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.11+
- OpenAI API key or OpenRouter API key (for embeddings)
- Optional: Anthropic API key (if using Anthropic models)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd thinking-frameworks/omni_cortex

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...              # For embeddings
# OR
OPENROUTER_API_KEY=sk-or-...       # Alternative for embeddings

# Optional
ANTHROPIC_API_KEY=sk-ant-...       # If using Anthropic models
CHROMA_PERSIST_DIR=/app/data/chroma  # Vector store location
LOG_LEVEL=INFO                     # Logging level
```

## 🚀 Usage

### Running the MCP Server

```bash
# From omni_cortex directory
python -m server.main
```

The server runs via stdio and communicates using the MCP protocol.

### MCP Configuration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "python",
      "args": ["-m", "server.main"],
      "cwd": "/path/to/thinking-frameworks/omni_cortex",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "CHROMA_PERSIST_DIR": "/path/to/data/chroma"
      }
    }
  }
}
```

### Using in Claude/Cursor

Once configured, the tools are available:

```
# Let the system auto-select the best framework
Use the "reason" tool with your query

# Or explicitly select a framework
Use "think_active_inference" for debugging
Use "think_mcts_rstar" for complex optimization
Use "think_chain_of_verification" for security review
```

### Ingesting Documentation for RAG

```bash
# Ingest repository files into vector store
python -m app.ingest_repo

# For enhanced ingestion with metadata
python -m app.enhanced_ingestion
```

## 🛠️ Development

### Project Structure

```
omni_cortex/
├── app/
│   ├── core/
│   │   ├── config.py          # Settings and configuration
│   │   └── router.py          # HyperRouter for framework selection
│   ├── nodes/
│   │   ├── strategy/          # Strategic planning frameworks
│   │   ├── search/            # Tree/graph search frameworks
│   │   ├── iterative/         # Iterative refinement frameworks
│   │   ├── code/              # Code-focused frameworks
│   │   ├── context/           # Context-building frameworks
│   │   ├── fast/              # Quick response frameworks
│   │   ├── common.py          # Shared utilities
│   │   └── langchain_tools.py # Tool integration
│   ├── graph.py               # LangGraph workflow definition
│   ├── state.py               # State management
│   ├── langchain_integration.py  # Memory, RAG, callbacks
│   ├── collection_manager.py  # Multi-collection vector store
│   └── schemas.py             # Pydantic models
├── server/
│   └── main.py                # MCP server entry point
└── mcp-config-examples/       # Example configurations
```

### Adding a New Framework

1. Create node file in appropriate category: `app/nodes/category/my_framework.py`
2. Implement the node function with `@quiet_star` decorator
3. Register in `app/graph.py` FRAMEWORK_NODES dict
4. Add to FRAMEWORKS dict in `server/main.py`
5. Update HyperRouter VIBE_DICTIONARY in `app/core/router.py`

## 📊 Collections (RAG)

The system maintains 6 specialized ChromaDB collections:

- **frameworks** - Framework implementations and reasoning nodes
- **documentation** - Markdown docs, READMEs, guides
- **configs** - Configuration files
- **utilities** - Helper functions
- **tests** - Test files
- **integrations** - LangChain/LangGraph integration code

## 🔧 Configuration

Key settings in `app/core/config.py`:

```python
max_reasoning_depth: int = 10        # Max recursion depth
mcts_max_rollouts: int = 50          # MCTS exploration limit
debate_max_rounds: int = 5           # Multi-agent debate rounds
enable_prm_scoring: bool = True      # Process Reward Model
enable_dspy_optimization: bool = True # Prompt optimization
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run specific test
pytest tests/test_router.py

# With coverage
pytest --cov=app tests/
```

## 📝 Example Use Cases

- **Debugging**: "Why is this throwing a null pointer?" → Active Inference
- **Architecture**: "Design a REST API for user management" → ReasonFlux
- **Optimization**: "Make this algorithm faster" → Tree of Thoughts
- **Security**: "Audit this code for vulnerabilities" → Chain of Verification
- **Math**: "Calculate the optimal portfolio allocation" → Program of Thoughts
- **Research**: "Understand how this codebase works" → Chain of Note

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

Built with:
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol
- [LangGraph](https://github.com/langchain-ai/langgraph) - Graph-based LLM orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [ChromaDB](https://www.trychroma.com/) - Vector database

Inspired by research in:
- Tree of Thoughts, Graph of Thoughts, Buffer of Thoughts
- rStar-Code (MCTS for code)
- Active Inference for debugging
- Self-Discover, CoALA, and other reasoning frameworks
