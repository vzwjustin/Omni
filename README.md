# Omni-Cortex: AI Thinking Frameworks MCP Server

An MCP (Model Context Protocol) server that provides 40 advanced reasoning frameworks for AI assistants. Built with LangGraph for orchestration and LangChain for memory/RAG capabilities.

## Overview

Omni-Cortex is a fully orchestrated MCP server that routes AI reasoning through 40 specialized frameworks using LangGraph workflows. Each framework is exposed as an MCP tool, with automatic routing via HyperRouter or explicit selection.

**Key Architecture:**
- **LangGraph Orchestration**: All execution flows through graph.ainvoke() with state management
- **Smart Routing**: HyperRouter with vibe dictionary and LLM-based analysis
- **Memory Persistence**: LangChain integration with conversation history
- **RAG Integration**: ChromaDB vector store with 6 specialized collections
- **Checkpointing**: SQLite-based workflow state persistence

The server orchestrates reasoning workflows while the calling AI (Claude, GPT, etc.) performs the actual reasoning within each framework's structured approach.

## 🧠 Available Frameworks (40 Total)

### Strategy (7 frameworks)
- **ReasonFlux** - Hierarchical planning: Template → Expand → Refine
- **Self-Discover** - Discover and compose custom reasoning patterns
- **Buffer of Thoughts** - Build context in a thought buffer
- **CoALA** - Cognitive architecture with memory systems
- **Least-to-Most** - Bottom-up atomic function decomposition
- **Comparative Architecture** - Multiple solution approaches (readability/memory/speed)
- **Plan-and-Solve** - Explicit planning before execution

### Search (4 frameworks)
- **rStar-Code MCTS** - Monte Carlo Tree Search for code exploration
- **Tree of Thoughts** - Explore multiple solution paths, pick best
- **Graph of Thoughts** - Non-linear reasoning with idea graphs
- **Everything of Thought** - Combine multiple reasoning approaches

### Iterative (8 frameworks)
- **Active Inference** - Hypothesis testing loop for debugging
- **Multi-Agent Debate** - Multiple perspectives argue trade-offs
- **Adaptive Injection** - Inject strategies as needed
- **RE2** - Read-Execute-Evaluate loop for requirements
- **Rubber Duck Debugging** - Socratic questioning for self-discovery
- **ReAct** - Interleaved reasoning and acting with tools
- **Reflexion** - Self-evaluation with memory-based learning
- **Self-Refine** - Iterative self-critique and improvement

### Code (13 frameworks)
- **Program of Thoughts** - Generate executable code to solve problems
- **Chain of Verification** - Draft → Verify → Patch cycle
- **CRITIC** - Generate then critique with external validation
- **Chain-of-Code** - Break problems into code blocks for structured thinking
- **Self-Debugging** - Mental execution trace before presenting code
- **TDD Prompting** - Write tests first, then implementation
- **Reverse Chain-of-Thought** - Work backward from buggy output to source
- **AlphaCodium** - Test-based multi-stage iterative code generation (competitive programming)
- **CodeChain** - Chain of self-revisions guided by sub-modules
- **Evol-Instruct** - Evolutionary instruction complexity with constraints
- **LLMLOOP** - Automated iterative feedback loops (compilation, tests, mutation)
- **ProCoder** - Compiler-feedback-guided iterative refinement
- **RECODE** - Multi-candidate validation with CFG-based debugging

### Context (6 frameworks)
- **Chain of Note** - Research and note-taking approach
- **Step-Back** - Abstract principles first, then apply
- **Analogical** - Find and adapt similar solutions
- **Red-Teaming** - Adversarial security analysis (STRIDE, OWASP)
- **State-Machine Reasoning** - Formal FSM design before coding
- **Chain-of-Thought** - Basic step-by-step reasoning

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
│  │  55 MCP Tools                               │     │
│  │  • 40 think_* framework tools               │     │
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
│  │  40 Framework Nodes                         │     │
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

Once configured, the tools are available through the MCP protocol:

```
# Auto-select framework via LangGraph routing
Use the "reason" tool with your query
→ Creates GraphState
→ Invokes graph.ainvoke(state)
→ HyperRouter selects optimal framework
→ Executes through LangGraph workflow
→ Returns structured result with confidence

# Or explicitly select a framework
Use "think_active_inference" for debugging
Use "think_alphacodium" for competitive programming
Use "think_llmloop" for production-ready code
Use "think_chain_of_verification" for security review

# All tools execute through LangGraph orchestration
→ Full state management
→ Memory persistence (LangChain)
→ Checkpointing (SQLite)
→ Tool integration (code execution, RAG)
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
│   │   ├── strategy/          # Strategic planning frameworks (7)
│   │   ├── search/            # Tree/graph search frameworks (4)
│   │   ├── iterative/         # Iterative refinement frameworks (8)
│   │   ├── code/              # Code-focused frameworks (13)
│   │   │   ├── pot.py         # Program of Thoughts
│   │   │   ├── alphacodium.py # Test-based multi-stage (NEW)
│   │   │   ├── codechain.py   # Sub-module self-revision (NEW)
│   │   │   ├── evol_instruct.py # Evolutionary complexity (NEW)
│   │   │   ├── llmloop.py     # 5-loop refinement (NEW)
│   │   │   ├── procoder.py    # Compiler-guided (NEW)
│   │   │   └── recode.py      # Multi-candidate CFG (NEW)
│   │   ├── context/           # Context-building frameworks (6)
│   │   ├── fast/              # Quick response frameworks (2)
│   │   ├── common.py          # Shared utilities (@quiet_star decorator)
│   │   └── langchain_tools.py # Tool integration
│   ├── graph.py               # LangGraph workflow (route→execute nodes)
│   ├── state.py               # GraphState management
│   ├── langchain_integration.py  # Memory, RAG, callbacks
│   ├── collection_manager.py  # Multi-collection vector store
│   └── schemas.py             # Pydantic models
├── server/
│   └── main.py                # MCP server (wired to graph.ainvoke)
└── mcp-config-examples/       # Example configurations
```

### Adding a New Framework

1. **Create node implementation**: `app/nodes/category/my_framework.py`
   ```python
   from ...state import GraphState
   from ..common import quiet_star, add_reasoning_step, format_code_context

   @quiet_star
   async def my_framework_node(state: GraphState) -> GraphState:
       # Your framework logic here
       state["final_answer"] = "..."
       state["confidence_score"] = 0.85
       return state
   ```

2. **Export from category**: Add to `app/nodes/category/__init__.py`
   ```python
   from .my_framework import my_framework_node
   __all__ = [..., "my_framework_node"]
   ```

3. **Register in graph**: Add to `app/graph.py` FRAMEWORK_NODES dict
   ```python
   from .nodes.category import my_framework_node
   FRAMEWORK_NODES = {
       "my_framework": my_framework_node,
   }
   ```

4. **Add MCP tool definition**: Update `server/main.py` FRAMEWORKS dict
   ```python
   FRAMEWORKS = {
       "my_framework": {
           "category": "code",
           "description": "Brief description",
           "best_for": ["use case 1", "use case 2"],
           "prompt": """Framework prompt template..."""
       }
   }
   ```

5. **Update router vibes** (optional): Add to `app/core/router.py` VIBE_DICTIONARY
   ```python
   VIBE_DICTIONARY = {
       "my_framework": ["keyword1", "keyword2", "phrase"],
   }
   ```

All execution automatically flows through LangGraph - no additional wiring needed!

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

All examples execute through full LangGraph orchestration:

- **Debugging**: "Why is this throwing a null pointer?" → Active Inference (hypothesis testing loop)
- **Architecture**: "Design a REST API for user management" → ReasonFlux (hierarchical planning)
- **Competitive Programming**: "Solve this LeetCode hard problem" → AlphaCodium (test-based iterative)
- **Production Code**: "Generate production-ready user auth" → LLMLOOP (5-loop refinement)
- **Large Codebase Integration**: "Add this feature to existing system" → ProCoder (compiler-guided)
- **High-Stakes Code**: "Generate critical payment processing logic" → RECODE (multi-candidate validation)
- **Security**: "Audit this code for vulnerabilities" → Chain of Verification + Red-Teaming
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
