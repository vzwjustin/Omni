# 🧠 Omni Cortex
### The Headless Strategy Engine for AI Coding Agents

[![Docker Version](https://img.shields.io/docker/v/vzwjustin/omni-cortex?sort=semver&style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/vzwjustin/omni-cortex)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge&logo=github)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green?style=for-the-badge)](https://modelcontextprotocol.io)
[![Frameworks](https://img.shields.io/badge/Frameworks-62-purple?style=for-the-badge)](FRAMEWORKS.md)

**Omni Cortex** is an MCP server that gives Claude access to **62 advanced reasoning frameworks** through Gemini-powered orchestration. Gemini thinks deeply about your problem and generates ultra-efficient execution briefs for Claude.

> **"Gemini orchestrates. Claude executes. You ship faster."**

---

## 🏗️ Architecture: Two Gemini Paths

Omni-Cortex uses Gemini for orchestration through **two parallel tools**:

### Path 1: `prepare_context` — Context Preparation
```
Claude calls prepare_context → ContextGateway
    ├── QueryAnalyzer (Gemini)     → Understands task intent
    ├── FileDiscoverer (Gemini)    → Finds relevant files  
    ├── DocumentationSearcher      → Fetches web docs
    └── CodeSearcher               → grep/git searches
                     ↓
        StructuredContext → Claude uses for file discovery
```
**Use when:** Claude needs to understand the codebase, find files, get documentation.

### Path 2: `reason` — Framework Selection + Execution Brief
```
Claude calls reason → HyperRouter
    ├── _route_to_category()              → Fast local pattern match
    ├── _select_with_specialist(Gemini)   → Picks framework chain
    └── StructuredBriefGenerator
        ├── gemini_analyze_task()         → Rich execution plan
        └── enrich_evidence_from_chroma() → RAG knowledge
                     ↓
        ClaudeCodeBrief (~200 tokens) → Claude executes
```
**Use when:** Claude needs a thinking strategy and step-by-step execution plan.

### Combined Flow
```
┌─────────────────────────────────────────────────────────────────────┐
│  USER → Claude Code                                                 │
│            ↓                                                        │
│      prepare_context → Gemini discovers files, docs, context        │
│            ↓                                                        │
│      reason → Gemini selects frameworks, generates brief            │
│            ↓                                                        │
│      Claude executes with full context + optimal strategy           │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Design:**
- **Gemini burns tokens freely** (1M context) - does ALL the heavy thinking
- **Claude gets surgical briefs** (~200 tokens) - focuses on execution
- **20% token savings** vs verbose formats, with zero information loss

---

## 🔮 Vibe-Based Routing

You don't need to ask for "Active Inference" or "Chain of Verification." Just speak naturally:

| You Say | Selected Strategy |
|:---|:---|
| *"WTF is wrong with this? It's failing randomly!"* | `active_inference` → Hypothesis Testing Loop |
| *"This code is spaghetti. Kill it with fire."* | `graph_of_thoughts` → Dependency disentanglement |
| *"Is this actually secure? Check for hacks."* | `chain_of_verification` → Red Teaming & Auditing |
| *"I have no idea how to start this weird problem."* | `self_discover` → First Principles exploration |
| *"Make it faster. It's too slow."* | `tree_of_thoughts` → Optimization Search |
| *"Make the tests pass, fix the CI."* | `tdd_prompting` → Test-Driven Development |

---

## 🔗 Framework Chaining

For complex tasks, Omni chains multiple frameworks together in a pipeline:

```
Category Router → Specialist Agent → Framework Chain → Pipeline Executor
      ↓                   ↓                  ↓               ↓
  "debug"         Debug Detective      [fw1 → fw2 → fw3]   Execute each
  "code_gen"      Code Architect               ↓           in sequence
  "refactor"      Refactor Surgeon     Pass state between
      ...                              frameworks
```

### Example: Complex Bug Fix

When you say *"This async handler crashes randomly under load"*:

1. **Category Match** → `debug` (vibe: "crashes", "randomly")
2. **Specialist Decision** → Debug Detective selects chain
3. **Pipeline Execution**:
   ```
   self_ask         →  "What exactly are we debugging?"
   active_inference →  Hypothesis testing loop
   verify_and_edit  →  Validate fix, patch only what's wrong
   ```

---

## 📋 ClaudeCodeBrief Format (NEW)

The structured handoff protocol optimizes for Claude Max subscriptions:

```
[DEBUG] Fix user auth failing after password reset
→ auth/password.py:L45-67 auth/session.py
  • authentication
  ⊘ auth/oauth.py

1. Check reset_password() return value
2. Verify session invalidation after reset
3. Add token refresh call after password change

✓ pytest tests/auth/ -v
• All tests pass
• Auth flow works

⚠ Preserve existing functionality
⚠ Do not break API

• [FILE] auth/password.py:L45
  → returns None instead of new token
• [FILE] auth/session.py:L78
  → does not invalidate old tokens

⛔ If required inputs missing, request them
```

**~200 tokens** with full actionability.

---

## ⚡ Installation

### Option 1: Docker Pull
```bash
docker pull vzwjustin/omni-cortex:latest
```

### Option 2: Add to MCP Config
Add to your IDE's MCP settings (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "/path/to/your/code:/code",
        "vzwjustin/omni-cortex:latest"
      ]
    }
  }
}
```

---

## 🧩 Framework Architecture (Modular)

The 62 frameworks are organized into a clean, modular structure:

### Single Source of Truth
```
app/frameworks/
├── __init__.py         # Exposes FRAMEWORKS dict
└── registry.py         # ALL 62 framework definitions (76KB)
```

Each framework is defined as a `FrameworkDefinition` dataclass:
```python
FrameworkDefinition(
    name="active_inference",
    display_name="Active Inference",
    category=FrameworkCategory.ITERATIVE,
    description="Debugging loop: hypothesis → predict → compare → update",
    best_for=["debugging", "error analysis", "root cause investigation"],
    vibes=["why is this broken", "wtf is wrong", "find the bug", ...],
    steps=["HYPOTHESIS: Form hypothesis", "PREDICT: Expected behavior", ...],
    complexity="medium",
    task_type="debug",
)
```

### Node Implementations (By Category)
```
app/nodes/
├── common.py           # Shared logic for all nodes
├── generator.py        # Dynamic prompt generator (uses registry)
│
├── strategy/           # ReasonFlux, Self-Discover, Plan-and-Solve...
├── search/             # Tree of Thoughts, Graph of Thoughts, MCTS...
├── iterative/          # Active Inference, Reflexion, Self-Refine...
├── code/               # Program of Thoughts, Chain of Code, TDD...
├── context/            # Chain of Note, Step-Back, Buffer of Thoughts...
├── fast/               # System1, Scaffolding (quick responses)
├── verification/       # Chain of Verification, Self-Consistency...
├── agent/              # SWE-Agent, ReWOO, LATS...
└── rag/                # HyDE, RAG-Fusion, RAPTOR, GraphRAG...
```

### How It Works
```
1. User query → HyperRouter matches vibes in registry.py
2. Category identified → Specialist selects framework(s)
3. Framework chain selected → generator.py builds prompts from steps
4. Node executes → Category-specific logic in nodes/{category}/
5. Result returned → Formatted as ClaudeCodeBrief
```

### Why This Structure?
- **Single Source of Truth**: Add/modify frameworks in ONE file
- **Vibe Matching**: Natural language → framework selection
- **Modular Nodes**: Category-specific execution logic
- **Prompt Generation**: Steps are templates, generator fills in context

### Categories

| Category | Count | Focus |
|:---|:---:|:---|
| Strategy | 7 | Architecture, planning, system design |
| Search | 4 | Optimization, exploration, complex bugs |
| Iterative | 8 | Debugging, refinement, learning loops |
| Code | 15 | Code generation, testing, algorithms |
| Context | 6 | Research, abstraction, security |
| Fast | 2 | Quick fixes, scaffolding |
| Verification | 8 | Claim integrity, hallucination detection |
| Agent | 5 | Tool orchestration, execution loops |
| RAG | 5 | Retrieval grounding, evidence-based |
| **Total** | **62** | |

<details>
<summary><h3>🔍 Debugging & Verification (7)</h3></summary>

| Framework | Best For |
|:---|:---|
| **Active Inference** | Root cause analysis of "impossible" bugs |
| **Chain of Verification** | Security audits and logic checking |
| **Self Debugging** | Pre-computation mental traces before coding |
| **Reverse CoT** | Working backward from a wrong output to the error |
| **Red Team** | Adversarial attack simulation |
| **Reflexion** | Learning from past failures in a loop |
| **TDD Prompting** | Writing tests before implementation |
</details>

<details>
<summary><h3>🏗️ Architecture & Planning (7)</h3></summary>

| Framework | Best For |
|:---|:---|
| **ReasonFlux** | Hierarchical system design |
| **Plan-and-Solve** | Explicit roadmap creation before execution |
| **Self-Discover** | Solving novel problems with no known pattern |
| **CoALA** | Agentic loop with episodic memory |
| **Buffer of Thoughts** | Managing massive context requirements |
| **Least-to-Most** | Bottom-up decomposition of complex systems |
| **Comparative Arch** | Weighing trade-offs between multiple approaches |
</details>

<details>
<summary><h3>🚀 Optimization & Code Gen (15)</h3></summary>

| Framework | Best For |
|:---|:---|
| **Tree of Thoughts** | Exploring multiple optimization paths |
| **Graph of Thoughts** | Non-linear refactoring of spaghetti code |
| **Program of Thoughts** | Math, data processing, computational problems |
| **Chain-of-Code** | Execution-based logic reasoning |
| **CRITIC** | API usage validation with external tools |
| **Self-Debugging** | Mental execution trace before presenting code |
| **Reverse Chain-of-Thought** | Backward debugging from wrong outputs |
| *(and more...)* | *See [FRAMEWORKS.md](FRAMEWORKS.md)* |
</details>

<details>
<summary><h3>✅ Verification & Integrity (8)</h3></summary>

| Framework | Best For |
|:---|:---|
| **Self-Consistency** | Multi-sample voting for reliable answers |
| **Self-Ask** | Sub-question decomposition before solving |
| **RaR** | Rephrase-and-Respond for clarity |
| **Verify-and-Edit** | Verify claims, edit only failures |
| **RARR** | Research, Augment, Revise - evidence-driven |
| **SelfCheckGPT** | Hallucination detection via sampling |
| **MetaQA** | Metamorphic testing for reasoning reliability |
| **RAGAS** | RAG Assessment for retrieval quality |
</details>

<details>
<summary><h3>🤖 Agent Orchestration (5)</h3></summary>

| Framework | Best For |
|:---|:---|
| **ReWOO** | Plan then execute - minimize tool calls |
| **LATS** | Tree search over action sequences |
| **MRKL** | Modular reasoning with specialized modules |
| **SWE-Agent** | Repo-first execution loop (inspect/edit/run) |
| **Toolformer** | Smart tool selection policy |
</details>

<details>
<summary><h3>📚 RAG & Retrieval (5)</h3></summary>

| Framework | Best For |
|:---|:---|
| **Self-RAG** | Self-triggered selective retrieval |
| **HyDE** | Hypothetical Document Embeddings |
| **RAG-Fusion** | Multi-query retrieval with rank fusion |
| **RAPTOR** | Hierarchical abstraction retrieval |
| **GraphRAG** | Entity-relation grounding for dependencies |
</details>

---

## 🔧 Recent Changes

### Code Consolidation
- **Single Registry**: All frameworks now in `app/frameworks/registry.py`
- **Config Unified**: Deprecated `core/config.py`, using `core/settings.py`
- **Token-Efficient Briefs**: `ClaudeCodeBrief.to_surgical_prompt()` for 20% savings

### Architecture
- **Gemini Orchestration**: Task analysis, context prep, framework selection
- **ChromaDB Integration**: Knowledge buffer for cross-session learning
- **Structured Handoff Protocol**: GeminiRouterOutput → ClaudeCodeBrief

---

## 📄 License
MIT License. Open source and free to use.

---
*Built with ❤️ by [Justin Adams](https://github.com/vzwjustin)*
