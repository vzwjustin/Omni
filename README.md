# 🧠 Omni Cortex
### The Gemini-Powered Context Gateway & Reasoning Engine for Claude

[![Docker](https://img.shields.io/badge/docker-latest-blue?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/vzwjustin/omni-cortex)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge&logo=github)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green?style=for-the-badge)](https://modelcontextprotocol.io)
[![Frameworks](https://img.shields.io/badge/Frameworks-62-purple?style=for-the-badge)](omni_cortex/FRAMEWORKS.md)

**Omni Cortex** is an MCP server that supercharges Claude with:
- 🎯 **Gemini-Powered Context Gateway**: Gemini 3 Flash does the "egg hunting" - analyzing queries, discovering relevant files, searching code/docs, and structuring rich context so Claude can focus on deep reasoning
- 🧠 **62 Real Reasoning Frameworks**: Multi-turn orchestrations (not templates) with actual algorithmic flows - branching, voting, iteration, evaluation
- 🗜️ **Context Optimization Tools**: Token counting, 30-70% content compression, truncation detection, CLAUDE.md rule management
- 📚 **16K+ Training Examples**: ChromaDB knowledge base with bug-fix patterns, reasoning chains, and code examples
- ⚡ **Ultra-Lean Mode**: Just 8 tools exposed - full power, zero bloat

> **"Gemini finds the needles in the haystack. Claude does the deep thinking."**

---

## 🚀 Quick Start

### 1. Get Gemini API Key (Free)
Visit https://aistudio.google.com/apikey and grab a free API key (1500 embedding requests/day).

### 2. Add to Claude Desktop / Cursor

Edit your MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "/path/to/your/code:/code",
        "-e", "GOOGLE_API_KEY=your_gemini_key_here",
        "vzwjustin/omni-cortex:latest"
      ]
    }
  }
}
```

### 3. Restart Your IDE

Claude now has access to:
- **prepare_context**: Gemini analyzes your codebase, finds relevant files, searches docs
- **reason**: Auto-selects best framework from 62 options
- **compress_content**: Reduce token usage by 30-70%
- **count_tokens**: Know exactly how many tokens you're using
- **execute_code**: Run Python in sandbox for validation

### 4. Usage Example

In Claude Desktop:

```
You: "This async handler crashes under load. Find the bug."

Claude uses:
1. prepare_context → Gemini discovers handlers, searches git history, finds similar bugs in knowledge base
2. reason → Auto-selects Debug Detective chain (self_ask → active_inference → verify_and_edit)
3. Returns fix with detailed reasoning trace
```

**That's it!** No need to manually select frameworks or specify which files to read. Gemini does the prep, Claude does the reasoning.

---

## 🎯 Context Gateway: Gemini → Claude Intelligence Pipeline

**The Problem**: Claude wastes tokens searching files, grepping code, fetching docs - burning context on "egg hunting" instead of deep reasoning.

**The Solution**: Gemini 3 Flash (cheap, fast) does the prep work and hands Claude a perfectly organized brief.

### How It Works

```
User Query → Gemini Context Gateway → Structured Context → Claude Deep Reasoning
                    ↓
        1. Analyze intent & extract keywords
        2. Discover relevant files (with scoring)
        3. Search code (grep/ripgrep/git)
        4. Fetch documentation from web
        5. Query ChromaDB knowledge base (16K+ examples)
        6. Structure everything into organized brief
```

### What You Get

- **File Discovery**: Gemini scores files by relevance (0-1), extracts key elements (functions, classes), summarizes content
- **Code Search**: Automated grep/ripgrep/git searches with match counts and context
- **Documentation**: Fetches relevant docs from web with snippets and relevance scores
- **Knowledge Base**: Pulls from 16K+ bug-fixes, reasoning patterns, and code examples in ChromaDB
- **Structured Output**: Organized brief with execution plan, relevant files, and context - ready for Claude

### Example

```json
{
  "query": "Fix the async handler that crashes under load",
  "files": [
    {
      "path": "src/handlers/async.py",
      "relevance": 0.95,
      "summary": "Main async request handler with connection pooling",
      "key_elements": ["handle_request", "ConnectionPool", "async_timeout"]
    }
  ],
  "code_search": {
    "type": "grep",
    "query": "async.*handler.*crash",
    "matches": 12,
    "context": "..."
  },
  "knowledge_base_insights": [
    "Similar async race condition in PyResBugs #4231",
    "Common pattern: missing await in error handler"
  ],
  "plan": "1. Check error handler for missing awaits\n2. Verify connection pool cleanup\n3. Add timeout guards"
}
```

**Cost Savings**: Gemini 3 Flash is **15x cheaper** than Claude Opus 4.5 for context prep. Let the cheap model do the grunt work.

---

## 🗜️ Context Optimization Tools

Four specialized tools to manage token budgets and optimize context:

### 1. **count_tokens**
Uses Claude's tokenizer (tiktoken cl100k_base) to accurately count tokens before sending to Claude.

```python
count_tokens(text="your code here")
# Returns: {"tokens": 1234, "characters": 5678}
```

### 2. **compress_content**
Intelligently removes comments, whitespace, and redundant formatting while preserving code structure.

- **30-70% token reduction** on typical codebases
- Preserves function signatures, class definitions, and logic
- Configurable target reduction (default: 30%)

```python
compress_content(content="...", target_reduction=0.5)
# Returns compressed code with ~50% fewer tokens
```

### 3. **detect_truncation**
Identifies incomplete code blocks, unclosed syntax, and truncated content with confidence scoring.

```python
detect_truncation(text="def foo():\n    return...")
# Returns: {
#   "is_truncated": true,
#   "confidence": 0.85,
#   "issues": ["Incomplete function body", "Missing closing brace"],
#   "last_complete_line": 1
# }
```

### 4. **manage_claude_md**
Analyze, generate, or inject rules into CLAUDE.md files for consistent agent behavior.

**Actions**:
- `analyze`: Scan project and suggest rules
- `generate`: Create CLAUDE.md template for project type (Python, TypeScript, React, Rust)
- `inject`: Add rules to existing CLAUDE.md
- `list_presets`: Show available rule categories

**Rule Presets**: security, performance, testing, documentation, code_quality, git, context_optimization

```python
manage_claude_md(
  action="generate",
  project_type="python",
  presets=["security", "testing", "performance"]
)
```

**Why This Matters**: Stay within Claude's context limits, reduce API costs, and maintain code integrity when working with large codebases.

---

## 🔥 How It Works: The Two-Tier Workflow

### Step 1: Context Preparation (Gemini)
When you call **prepare_context**, Gemini 3 Flash:
- Analyzes your query to understand intent
- Discovers relevant files with relevance scoring
- Searches your codebase (grep/ripgrep/git)
- Fetches documentation from the web
- Queries ChromaDB knowledge base (16K+ examples)
- Structures everything into an organized brief

**Cost**: ~$0.0001 per query (virtually free with Gemini's free tier)

### Step 2: Deep Reasoning (Claude)
When you call **reason** with the prepared context:
- HyperRouter analyzes the task (2-stage routing: category → specialist)
- Specialist agent selects best framework(s) from 62 options
- Framework(s) execute as **real multi-turn orchestrations** (not templates):
  - Tree of Thoughts: Branches, evaluates, selects winner
  - Active Inference: Hypothesis loop with evidence gathering
  - Chain of Verification: Generate → Verify → Revise cycle
- Returns structured results with reasoning traces

**Cost**: Standard Claude API pricing, but you're using it for deep reasoning (its strength) not file searching (waste)

### The Intelligence Division

| Task | Who Does It | Why |
|:---|:---|:---|
| Find relevant files | Gemini (cheap) | Pattern matching, scoring |
| Search git history | Gemini (cheap) | Grep/regex operations |
| Fetch documentation | Gemini (cheap) | Web scraping, summarization |
| Query knowledge base | Gemini (cheap) | Embedding search |
| **Deep reasoning** | **Claude (expensive)** | **Complex logic, code generation** |
| **Framework orchestration** | **Claude (expensive)** | **Multi-turn algorithms** |

**Result**: 15x cost savings by using each model for what it does best.

---

## 🔗 Smart Routing & Framework Chaining

The **reason** tool uses sophisticated 2-stage routing to select the optimal approach:

### Hierarchical Routing

```
Query → Stage 1: Category Detection → Stage 2: Specialist Selection → Framework Chain
         (9 categories)              (9 domain experts)              (1-4 frameworks)
```

**Stage 1 - Category Detection**: Fast pattern matching to one of 9 domains
- `debug`, `code_gen`, `refactor`, `architecture`, `verification`, `agent`, `retrieval`, `explore`, `fast`

**Stage 2 - Specialist Selection**: Domain expert picks best framework(s)
- Each specialist has deep knowledge of their domain's framework toolbox
- Can select single framework or chain 2-4 frameworks for complex tasks

### The 9 Specialist Agents

| Specialist | Domain | Example Chain |
|:---|:---|:---|
| **Debug Detective** | Bug hunting, root cause analysis | `self_ask → active_inference → verify_and_edit` |
| **Code Architect** | New feature implementation | `plan_and_solve → parsel → tdd_prompting → self_refine` |
| **Refactor Surgeon** | Code cleanup, restructuring | `plan_and_solve → graph_of_thoughts → verify_and_edit` |
| **System Architect** | High-level design decisions | `reason_flux → multi_agent_debate → plan_and_solve` |
| **Verification Expert** | Security, correctness audits | `red_team → chain_of_verification → verify_and_edit` |
| **Agent Orchestrator** | Multi-step task execution | `swe_agent → tdd_prompting → verify_and_edit` |
| **Retrieval Specialist** | Documentation research | `hyde → rag_fusion → rarr` |
| **Explorer** | Novel problems, unknowns | `self_discover → analogical → self_refine` |
| **Speed Demon** | Quick fixes, simple tasks | `system1` (single framework) |

### Example: Debugging Workflow

**Your query**: *"This async handler crashes randomly under load"*

**Omni's process**:
1. **prepare_context** (Gemini):
   - Finds `src/handlers/async.py` with 0.95 relevance
   - Greps for async error patterns
   - Searches knowledge base: "PyResBugs #4231 - similar race condition"
   - Returns structured context

2. **reason** (Claude + HyperRouter):
   - Category: `debug` (detected from "crashes", "randomly")
   - Specialist: **Debug Detective**
   - Chain selected: `complex_bug` → `self_ask → active_inference → verify_and_edit`

3. **Framework execution**:
   ```
   self_ask:          "What conditions trigger the crash? What's the async flow?"
   active_inference:  Test hypotheses (race condition, timeout, pool exhaustion)
   verify_and_edit:   Validate fix against test cases, apply minimal patch
   ```

**Result**: Root cause identified with high-confidence fix, backed by knowledge base patterns.

---

## ⚡ Installation

### Option 1: Docker (Recommended)

```bash
docker pull vzwjustin/omni-cortex:latest
```

### Option 2: Add to MCP Config

Add to your IDE's MCP settings file (`claude_desktop_config.json`, Cursor settings, etc.):

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-v", "/path/to/your/code:/code",
        "-e", "GOOGLE_API_KEY=your_gemini_key_here",
        "vzwjustin/omni-cortex:latest"
      ]
    }
  }
}
```

**Important**:
- Replace `/path/to/your/code` with your project directory
- Get a **free Gemini API key** at https://aistudio.google.com/apikey
- Gemini 3 Flash is **free** with generous limits (1500 requests/day for embeddings)
- Code volume mounting enables Context Gateway to discover and analyze your files

### Configuration

Set these environment variables (optional, sensible defaults provided):

```bash
# API Keys (Gemini recommended for cost-effectiveness)
GOOGLE_API_KEY=your_key          # For Context Gateway + embeddings (FREE)
ANTHROPIC_API_KEY=your_key       # Optional: if using Anthropic models
OPENAI_API_KEY=your_key          # Optional: for OpenAI embeddings
OPENROUTER_API_KEY=your_key      # Optional: for OpenRouter models

# Models (defaults to Gemini 3 Flash)
LLM_PROVIDER=google              # google, anthropic, openrouter
DEEP_REASONING_MODEL=gemini-3-flash-preview
ROUTING_MODEL=gemini-3-flash-preview

# Embedding Provider (defaults to Google/Gemini - FREE)
EMBEDDING_PROVIDER=google        # google, openai, openrouter
EMBEDDING_MODEL=text-embedding-004

# Mode
LEAN_MODE=true                   # true = 8 tools, false = 77 tools
```

See [omni_cortex/.env.example](omni_cortex/.env.example) for complete configuration options.

---

## 🎛️ Operating Modes

### Ultra-Lean Mode (Default) - 8 Tools, Full Power

**The Problem**: Exposing 62+ framework tools bloats Claude's context window with tool descriptions, leaving less room for your code.

**The Solution**: Expose only 8 essential tools. All 62 frameworks available internally via smart routing.

**Tools Exposed**:
1. **prepare_context** - Gemini-powered context gateway
2. **reason** - Auto-selects best framework(s) from all 62
3. **execute_code** - Sandboxed Python execution
4. **health** - Server status check
5. **count_tokens** - Claude tokenizer
6. **compress_content** - 30-70% token reduction
7. **detect_truncation** - Find incomplete code blocks
8. **manage_claude_md** - CLAUDE.md rule management

**Benefits**:
- ✅ **Minimal context overhead** - only 8 tool descriptions vs 77
- ✅ **Full framework access** - all 62 frameworks available via routing
- ✅ **Smart selection** - HyperRouter picks optimal framework(s)
- ✅ **Automatic chaining** - complex tasks get 2-4 frameworks in sequence

**When to Use**: Default for most users. Let the router do the thinking.

### Full Mode - 77 Tools Exposed

Set `LEAN_MODE=false` to expose all tools individually:
- 62 `think_*` framework tools (direct access to each framework)
- 15 utility tools (memory, RAG, search, execution, etc.)

**When to Use**: When you want explicit control over which framework to use, or for testing/debugging specific frameworks.

**Trade-off**: More context overhead from tool descriptions, but direct framework selection.

---

## 🧩 The 62 Frameworks
Omni contains the world's largest collection of formalized cognitive architectures for coding.

<details>
<summary><h3>🔍 Debugging & Verification (7)</h3></summary>

| Framework | Best For |
|:---|:---|
| **Active Inference** | Root cause analysis of "impossible" bugs. |
| **Chain of Verification** | Security audits and logic checking. |
| **Self Debugging** | Pre-computation mental traces before coding. |
| **Reverse CoT** | Working backward from a wrong output to the error. |
| **Red Team** | Adversarial attack simulation. |
| **Reflexion** | Learning from past failures in a loop. |
| **TDD Prompting** | Writing tests before implementation. |
</details>

<details>
<summary><h3>🏗️ Architecture & Planning (8)</h3></summary>

| Framework | Best For |
|:---|:---|
| **Reason Flux** | Hierarchical system design (Template -> Expand -> Refine). |
| **Plan and Solve** | Explicit roadmap creation before execution. |
| **State Machine** | Designing robust FSMs and workflows. |
| **CoALA** | Agentic loop with episodic memory. |
| **Buffer of Thoughts** | Managing massive context requirements. |
| **Least-to-Most** | Bottom-up decomposition of complex systems. |
| **Comparative Arch** | Weighing trade-offs between multiple approaches. |
| **Step Back** | Abstraction and first-principles thinking. |
</details>

<details>
<summary><h3>🚀 Optimization & Code Gen (15)</h3></summary>

| Framework | Best For |
|:---|:---|
| **Tree of Thoughts** | Exploring multiple optimization paths. |
| **Graph of Thoughts** | Non-linear refactoring of spaghetti code. |
| **AlphaCodium** | Competitive programming-style iterative solutions. |
| **Chain of Code** | Execution-based logic reasoning. |
| **ProCoder** | Compiler-feedback guided iteration. |
| **LLM Loop** | Continuous integration/test loops. |
| **Evol-Instruct** | Increasing constraint complexity. |
| **CodeChain** | Modular code generation with self-revisions. |
| **RECODE** | Multi-candidate validation with CFG debugging. |
| **PAL** | Program-Aided Language - code as reasoning. |
| **Scratchpads** | Structured intermediate reasoning workspace. |
| **Critic** | Generate then critique pattern. |
| **Program of Thoughts** | Step-by-step code reasoning. |
| *(and more...)* | *See [FRAMEWORKS.md](omni_cortex/FRAMEWORKS.md) for full list.* |
</details>

<details>
<summary><h3>💡 Creativity & Learning (6)</h3></summary>

| Framework | Best For |
|:---|:---|
| **Self Discover** | Solving novel problems with no known pattern. |
| **Analogical** | Finding similar solutions in history. |
| **Multi-Agent Debate** | Arguing pros/cons of a decision. |
| **Rubber Duck** | Socratic questioning to help you think. |
| **System 1** | Fast, intuitive "gut check" answers. |
| **Chain of Note** | Researching and summarizing massive docs. |
</details>

<details>
<summary><h3>✅ Verification & Claim Integrity (8) — NEW</h3></summary>

| Framework | Best For |
|:---|:---|
| **Self-Consistency** | Multi-sample voting for reliable answers. |
| **Self-Ask** | Sub-question decomposition before solving. |
| **RaR** | Rephrase-and-Respond for clarity. |
| **Verify-and-Edit** | Verify claims, edit only failures. |
| **RARR** | Research, Augment, Revise - evidence-driven. |
| **SelfCheckGPT** | Hallucination detection via sampling. |
| **MetaQA** | Metamorphic testing for reasoning reliability. |
| **RAGAS** | RAG Assessment for retrieval quality. |
</details>

<details>
<summary><h3>🤖 Agent Orchestration (5) — NEW</h3></summary>

| Framework | Best For |
|:---|:---|
| **ReWOO** | Plan then execute - minimize tool calls. |
| **LATS** | Tree search over action sequences. |
| **MRKL** | Modular reasoning with specialized modules. |
| **SWE-Agent** | Repo-first execution loop (inspect/edit/run). |
| **Toolformer** | Smart tool selection policy. |
</details>

<details>
<summary><h3>📚 RAG & Retrieval Grounding (5) — NEW</h3></summary>

| Framework | Best For |
|:---|:---|
| **Self-RAG** | Self-triggered selective retrieval. |
| **HyDE** | Hypothetical Document Embeddings. |
| **RAG-Fusion** | Multi-query retrieval with rank fusion. |
| **RAPTOR** | Hierarchical abstraction retrieval. |
| **GraphRAG** | Entity-relation grounding for dependencies. |
</details>

---

## 🧠 Architecture: Gemini + Multi-Turn Orchestration

Omni-Cortex uses a **two-tier intelligence architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TIER 1: GEMINI CONTEXT GATEWAY                       │
│  (Cheap, Fast - Egg Hunting & Prep Work)                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Query Analysis → File Discovery → Code Search → Doc Fetch → ChromaDB  │
│       ↓               ↓                ↓             ↓           ↓       │
│   Intent +      Relevance        grep/git       Web Docs    16K+ KB     │
│   Keywords      Scoring 0-1      matches        snippets    patterns    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                         Structured Context
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    TIER 2: CLAUDE REASONING ENGINE                      │
│  (Expensive, Powerful - Deep Reasoning)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  HyperRouter → Specialist Agent → Framework Chain → Multi-Turn Exec    │
│       ↓              ↓                   ↓                    ↓          │
│  Category    Domain Expert        [fw1→fw2→fw3]      Branching,        │
│  Detection   Picks Chain          Sequence Exec      Voting, Iteration │
└─────────────────────────────────────────────────────────────────────────┘
```

### System Components

1. **Context Gateway (Gemini)**: Preprocessing layer that discovers files, searches code, fetches docs, queries knowledge base
2. **HyperRouter**: Two-stage routing - category detection → specialist selection
3. **Specialist Agents**: 9 domain experts (Debug Detective, Code Architect, etc.)
4. **Framework Chaining**: Complex tasks get 2-4 frameworks in sequence
5. **Multi-Turn Orchestration**: Each framework executes its algorithm via MCP sampling
6. **ChromaDB Knowledge Base**: 16K+ training examples across debugging, reasoning, and instruction datasets
7. **Memory System**: Episodic memory for cross-session learning
8. **Context Optimization**: Token counting, compression, truncation detection

### Example: Tree of Thoughts

When you use Tree of Thoughts, the server:
1. Requests 3 solution branches from client (temp=0.8)
2. Requests evaluation of each branch (temp=0.2)
3. Selects best based on extracted scores
4. Requests expansion of winner (temp=0.3)
5. Returns final solution + metadata

**No API keys needed** - all inference happens in your local Claude Desktop client.

---

## 📊 Framework Categories

| Category | Count | Focus |
|:---|:---:|:---|
| Strategy | 7 | Architecture, planning, system design |
| Search | 4 | Optimization, exploration, complex bugs |
| Iterative | 8 | Debugging, refinement, learning loops |
| Code | 17 | Code generation, testing, algorithms |
| Context | 6 | Research, abstraction, security |
| Fast | 2 | Quick fixes, scaffolding |
| Verification | 8 | Claim integrity, hallucination detection |
| Agent | 5 | Tool orchestration, execution loops |
| RAG | 5 | Retrieval grounding, evidence-based |
| **Total** | **62** | |

See [FRAMEWORKS.md](omni_cortex/FRAMEWORKS.md) for complete documentation.

---

## 📊 Recent Updates (2026-01-06)

### 🎯 NEW: Gemini-Powered Context Gateway
The biggest update yet - a complete intelligence pipeline redesign:

**Context Gateway (prepare_context tool)**:
- ✅ Gemini 3 Flash analyzes queries and discovers relevant files with scoring
- ✅ Automated code search via grep/ripgrep/git with match counts
- ✅ Web documentation fetching with relevance scoring
- ✅ ChromaDB knowledge base integration (16K+ training examples)
- ✅ Structured output with execution plan ready for Claude
- ✅ **15x cheaper** than using Claude for context prep
- ✅ Gemini does the "egg hunting", Claude does deep reasoning

**Gemini Integration**:
- ✅ Default LLM provider now Google/Gemini (free tier available)
- ✅ Gemini 3 Flash Preview as default model
- ✅ Gemini embeddings support (FREE, 1500 requests/day)
- ✅ Get free API key at https://aistudio.google.com/apikey

### 🗜️ NEW: Context Optimization Tools (4 Tools)
Merged from context-optimizer project:

- **count_tokens**: Claude tokenizer (tiktoken cl100k_base) for accurate token counting
- **compress_content**: 30-70% token reduction while preserving code structure
- **detect_truncation**: Identify incomplete code blocks with confidence scoring
- **manage_claude_md**: Analyze, generate, inject rules into CLAUDE.md files
  - 7 rule presets: security, performance, testing, documentation, code_quality, git, context_optimization

### ⚡ Ultra-Lean Mode (Default)
**The Problem**: 62+ tool descriptions bloat Claude's context window

**The Solution**: Expose only 8 essential tools, all 62 frameworks available via auto-routing
- ✅ 8 tools exposed (vs 77 in full mode)
- ✅ Minimal context overhead
- ✅ Full framework access via HyperRouter
- ✅ Automatic framework chaining for complex tasks

**Tool Counts**:
- LEAN_MODE=true (default): **8 tools** exposed, 62 frameworks internal
- LEAN_MODE=false: **77 tools** exposed (62 think_* + 15 utilities)

### 📚 Knowledge Base (Existing, Now Enhanced)
16K+ training examples across 3 categories:

**Debugging** (10K examples):
- PyResBugs, HaPy-Bug, Learning-Fixes, python-bugs, bugnet, Code-Feedback

**Reasoning** (6K examples):
- General_Inquiry_Thinking-Chain-Of-Thought, chain-of-thoughts-chatml

**Instructions** (12K examples):
- tiny-codes (sampled), helpful_instructions

**Setup**: `python3 scripts/ingest_training_data.py --all`
**Requirements**: API key for embeddings (Gemini FREE, OpenAI, or OpenRouter)

### 🏗️ Architecture Updates
- ✅ Two-tier intelligence: Gemini (context prep) → Claude (deep reasoning)
- ✅ Sophisticated 2-stage hierarchical routing (category → specialist)
- ✅ 9 specialist agents for domain expertise
- ✅ 24 pre-defined framework chains
- ✅ Multi-collection ChromaDB RAG (10 collections)
- ✅ All 62 frameworks as real multi-turn orchestrations (not templates)

---

## 📄 License
MIT License. Open source and free to use.

---
*Built with ❤️ by [Justin Adams](https://github.com/vzwjustin)*
