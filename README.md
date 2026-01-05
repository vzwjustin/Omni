# 🧠 Omni Cortex
### The Headless Strategy Engine for AI Coding Agents

[![Docker](https://img.shields.io/badge/docker-latest-blue?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/vzwjustin/omni-cortex)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge&logo=github)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green?style=for-the-badge)](https://modelcontextprotocol.io)
[![Frameworks](https://img.shields.io/badge/Frameworks-60-purple?style=for-the-badge)](omni_cortex/FRAMEWORKS.md)

**Omni Cortex** is an MCP server that gives your IDE's AI access to **60 advanced reasoning frameworks**. It doesn't write the code for you; it gives your AI the *strategy* to write better code.

> **"Don't memorize complex prompt engineering. Just tell Omni how you feel about the code, and it picks the perfect cognitive strategy."**

---

## 🔮 Vibe-Based Routing
You don't need to ask for "Active Inference" or "Chain of Verification." Just speak naturally. The **Smart Router** analyzes your intent using a comprehensive "Vibe Dictionary."

| You Say | Omni Hears | Selected Strategy |
|:---|:---|:---|
| *"WTF is wrong with this? It's failing randomly!"* | **Debugging Panic** | `active_inference` (Hypothesis Testing Loop) |
| *"This code is spaghetti. Kill it with fire."* | **Refactoring Rage** | `graph_of_thoughts` (Dependency disentanglement) |
| *"Is this actually secure? Check for hacks."* | **Security Anxiety** | `chain_of_verification` (Red Teaming & Auditing) |
| *"I have no idea how to start this weird problem."* | **Novelty Confusion** | `self_discover` (First Principles exploration) |
| *"Make it faster. It's too slow."* | **Performance Need** | `tree_of_thoughts` (Optimization Search) |
| *"Prove it with evidence from the docs."* | **Verification Need** | `rarr` (Research, Augment, Revise) |
| *"Make the tests pass, fix the CI."* | **Execution Mode** | `swe_agent` (Repo-first execution loop) |
| *"How do these modules relate?"* | **Architecture Query** | `graphrag` (Entity-relation grounding) |

---

## ⚡ Installation

### Option 1: The One-Liner (Docker)
The easiest way to perform a "Headless Transformation" on your IDE.
```bash
docker pull vzwjustin/omni-cortex:latest
```

### Option 2: Add to MCP Config
Add this to your IDE's MCP settings file (e.g., `claude_desktop_config.json` or Cursor settings):

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
        "vzwjustin/omni-cortex:latest"
      ]
    }
  }
}
```
*(Replace `/path/to/your/code` with your actual project directory)*
*(Note: Mounting your code volume allows Omni to read your codebase for context-aware strategies.)*

---

## 🧩 The 60 Frameworks
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

## 🧠 Architecture: "Headless" Protocols
Omni-Cortex acts as a **Protocol Provider**.
1.  **Router**: Determines the best strategy based on your "vibe".
2.  **Context**: Retrieves relevant memories or docs (RAG).
3.  **Protocol Generation**: Constructs a highly structured, scientifically-backed system prompt.
4.  **Handoff**: Returns this protocol to your **Client LLM** (Claude, GPT-4, etc.) to execute.

This means **Omni doesn't need your API keys**. It just tells your AI *how* to think.

---

## 📊 Framework Categories

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
| **Total** | **60** | |

See [FRAMEWORKS.md](omni_cortex/FRAMEWORKS.md) for complete documentation.

---

## 📄 License
MIT License. Open source and free to use.

---
*Built with ❤️ by [Justin Adams](https://github.com/vzwjustin)*
