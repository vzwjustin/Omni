# Omni-Cortex: 60 Thinking Frameworks

A comprehensive library of AI reasoning frameworks for code-focused tasks. Each framework is accessible via MCP tools (`think_*`) or auto-selected by the `reason` tool.

---

## Strategy Frameworks (7)

| Framework | Tool | Description | Best For |
|-----------|------|-------------|----------|
| **ReasonFlux** | `think_reason_flux` | Hierarchical planning: Template → Expand → Refine | Architecture, system design, complex planning |
| **Self-Discover** | `think_self_discover` | Discover and apply custom reasoning patterns | Novel problems, unknown domains |
| **Buffer of Thoughts** | `think_buffer_of_thoughts` | Build context progressively in a thought buffer | Multi-part problems, complex context |
| **CoALA** | `think_coala` | Cognitive architecture with working + episodic memory | Autonomous tasks, agent behavior, multi-file |
| **Least-to-Most** | `think_least_to_most` | Bottom-up atomic function decomposition | Complex systems, monolith refactoring |
| **Comparative Architecture** | `think_comparative_arch` | Multi-approach comparison (readability/memory/speed) | Optimization, architecture decisions |
| **Plan-and-Solve** | `think_plan_and_solve` | Explicit planning before execution | Complex features, methodical development |

---

## Search Frameworks (4)

| Framework | Tool | Description | Best For |
|-----------|------|-------------|----------|
| **MCTS rStar** | `think_mcts_rstar` | Monte Carlo Tree Search exploration for code | Complex bugs, multi-step optimization |
| **Tree of Thoughts** | `think_tree_of_thoughts` | Explore multiple solution paths, pick best | Design decisions, multiple valid approaches |
| **Graph of Thoughts** | `think_graph_of_thoughts` | Non-linear reasoning with idea graphs | Complex dependencies, interconnected problems |
| **Everything of Thought** | `think_everything_of_thought` | Combine multiple reasoning approaches | Complex novel problems, major rewrites |

---

## Iterative Frameworks (8)

| Framework | Tool | Description | Best For |
|-----------|------|-------------|----------|
| **Active Inference** | `think_active_inference` | Hypothesis testing loop: Observe → Predict → Test → Act | Debugging, investigation, root cause analysis |
| **Multi-Agent Debate** | `think_multi_agent_debate` | Multiple perspectives debate (Pragmatist, Architect, Security, Performance) | Design decisions, trade-off analysis |
| **Adaptive Injection** | `think_adaptive_injection` | Inject strategies dynamically as needed | Evolving understanding, adaptive problem solving |
| **RE2** | `think_re2` | Read-Execute-Evaluate loop | Specifications, requirements implementation |
| **Rubber Duck** | `think_rubber_duck` | Socratic questioning for self-discovery | Architectural issues, blind spots, stuck problems |
| **ReAct** | `think_react` | Reasoning + Acting with tools | Multi-step tasks, tool use, investigation |
| **Reflexion** | `think_reflexion` | Self-evaluation with memory-based learning | Learning from failures, iterative improvement |
| **Self-Refine** | `think_self_refine` | Iterative self-critique and improvement | Code quality, documentation, polish |

---

## Code Frameworks (15)

| Framework | Tool | Description | Best For |
|-----------|------|-------------|----------|
| **Program of Thoughts** | `think_program_of_thoughts` | Step-by-step code reasoning | Algorithms, data processing, math |
| **Chain of Verification** | `think_chain_of_verification` | Draft-Verify-Patch cycle | Security review, code quality, bug prevention |
| **CRITIC** | `think_critic` | Generate then critique | API design, interface validation |
| **Chain-of-Code** | `think_chain_of_code` | Code-based problem decomposition | Logic puzzles, algorithmic debugging |
| **Self-Debugging** | `think_self_debugging` | Mental execution trace before presenting | Preventing bugs, edge case handling |
| **TDD Prompting** | `think_tdd_prompting` | Test-first development approach | New features, edge case coverage |
| **Reverse CoT** | `think_reverse_cot` | Backward reasoning from output delta | Silent bugs, wrong outputs, calculation errors |
| **AlphaCodium** | `think_alphacodium` | Test-based multi-stage iterative code generation | Competitive programming, complex algorithms |
| **CodeChain** | `think_codechain` | Chain of self-revisions guided by sub-modules | Modular code generation, incremental refinement |
| **Evol-Instruct** | `think_evol_instruct` | Evolutionary instruction complexity for code | Challenging code problems, constraint-based coding |
| **LLMLoop** | `think_llmloop` | Automated iterative feedback loops for code+tests | Code quality assurance, production-ready code |
| **ProCoder** | `think_procoder` | Compiler-feedback-guided iterative refinement | Project-level code generation, API usage |
| **RECODE** | `think_recode` | Multi-candidate validation with CFG-based debugging | Reliable code generation, high-stakes code |
| **PAL** | `think_pal` | Program-Aided Language - code as reasoning substrate | Algorithms, parsing, numeric logic, validation |
| **Scratchpads** | `think_scratchpads` | Structured intermediate reasoning workspace | Multi-step fixes, multi-constraint reasoning |

---

## Context Frameworks (6)

| Framework | Tool | Description | Best For |
|-----------|------|-------------|----------|
| **Chain of Note** | `think_chain_of_note` | Research and note-taking approach | Understanding code, documentation, exploration |
| **Step-Back** | `think_step_back` | Abstract principles first, then apply | Optimization, performance, architectural decisions |
| **Analogical** | `think_analogical` | Find and adapt similar solutions | Creative solutions, pattern matching |
| **Red-Team** | `think_red_team` | Adversarial security analysis (STRIDE, OWASP) | Security audits, vulnerability scanning |
| **State-Machine** | `think_state_machine` | Formal FSM design before coding | UI logic, workflow systems, game states |
| **Chain-of-Thought** | `think_chain_of_thought` | Step-by-step reasoning chain | Complex reasoning, logical deduction |

---

## Fast Frameworks (2)

| Framework | Tool | Description | Best For |
|-----------|------|-------------|----------|
| **Skeleton of Thought** | `think_skeleton_of_thought` | Outline first, fill in details | Boilerplate, quick scaffolding |
| **System1** | `think_system1` | Fast intuitive response | Simple questions, quick fixes |

---

## Verification Frameworks (8) ✨ NEW

| Framework | Tool | Description | Best For |
|-----------|------|-------------|----------|
| **Self-Consistency** | `think_self_consistency` | Multi-sample voting for reliable answers | Ambiguous bugs, tricky logic, multiple plausible fixes |
| **Self-Ask** | `think_self_ask` | Sub-question decomposition before solving | Unclear tickets, missing requirements, multi-part debugging |
| **RaR** | `think_rar` | Rephrase-and-Respond for clarity | Vague prompts, poorly written bug reports, ambiguous requirements |
| **Verify-and-Edit** | `think_verify_and_edit` | Verify claims, edit only failures | Code review, security guidance, surgical edits |
| **RARR** | `think_rarr` | Research, Augment, Revise loop | External docs, repo knowledge, prove-it requirements |
| **SelfCheckGPT** | `think_selfcheckgpt` | Hallucination detection via sampling | High-stakes guidance, unfamiliar libraries, final gate |
| **MetaQA** | `think_metaqa` | Metamorphic testing for reasoning reliability | Brittle reasoning, edge cases, policy consistency |
| **RAGAS** | `think_ragas` | RAG Assessment for retrieval quality | RAG pipelines, retrieval quality, source grounding |

---

## Agent Frameworks (5) ✨ NEW

| Framework | Tool | Description | Best For |
|-----------|------|-------------|----------|
| **ReWOO** | `think_rewoo` | Reasoning Without Observation - plan then execute | Multi-step tasks, cost control, plan-once-execute-clean |
| **LATS** | `think_lats` | Language Agent Tree Search over action sequences | Complex repo changes, multiple fix paths, uncertain root cause |
| **MRKL** | `think_mrkl` | Modular Reasoning with specialized modules | Big systems, mixed domains, tool-rich setups |
| **SWE-Agent** | `think_swe_agent` | Repo-first execution loop - inspect/edit/run/iterate | Multi-file bugfixes, CI failures, make tests pass |
| **Toolformer** | `think_toolformer` | Smart tool selection policy | Router logic, preventing pointless calls, tool efficiency |

---

## RAG Frameworks (5) ✨ NEW

| Framework | Tool | Description | Best For |
|-----------|------|-------------|----------|
| **Self-RAG** | `think_self_rag` | Self-triggered selective retrieval | Mixed knowledge tasks, large corpora, minimizing irrelevant retrieval |
| **HyDE** | `think_hyde` | Hypothetical Document Embeddings for better retrieval | Fuzzy search, unclear intent, broad problems |
| **RAG-Fusion** | `think_rag_fusion` | Multi-query retrieval with rank fusion | Improving recall, complex queries, noisy corpora |
| **RAPTOR** | `think_raptor` | Hierarchical abstraction retrieval for large docs | Huge repos, long design docs, monorepos |
| **GraphRAG** | `think_graphrag` | Entity-relation grounding for dependencies | Architecture questions, module relationships, impact analysis |

---

## Usage

### Auto-Selection (Recommended)
```
Use the `reason` tool - it automatically selects the best framework based on your task.
```

### Direct Selection
```
Use `think_{framework_name}` to invoke a specific framework.
Example: think_active_inference for debugging
```

### Vibe-Based Selection
Just describe your problem naturally:
- "wtf is wrong with this code" → Active Inference
- "make it faster" → Tree of Thoughts
- "design a system" → ReasonFlux
- "is this secure?" → Chain of Verification
- "prove it with evidence" → RARR
- "make tests pass" → SWE-Agent
- "how do modules relate" → GraphRAG

---

## Framework Categories by Use Case

| Use Case | Recommended Frameworks |
|----------|----------------------|
| **Debugging** | active_inference, mcts_rstar, reverse_cot, self_debugging |
| **Refactoring** | graph_of_thoughts, least_to_most, everything_of_thought |
| **Architecture** | reason_flux, plan_and_solve, comparative_arch, graphrag |
| **Security** | chain_of_verification, red_team |
| **Performance** | tree_of_thoughts, step_back |
| **Testing** | tdd_prompting, program_of_thoughts |
| **Documentation** | chain_of_note, skeleton_of_thought |
| **Quick Fixes** | system1, skeleton_of_thought |
| **Complex Problems** | everything_of_thought, mcts_rstar, coala |
| **Code Generation** | alphacodium, codechain, procoder, recode, pal |
| **Verification** | self_consistency, verify_and_edit, selfcheckgpt, ragas |
| **Agent Tasks** | rewoo, lats, mrkl, swe_agent, toolformer |
| **Retrieval/RAG** | self_rag, hyde, rag_fusion, raptor, graphrag |

---

## High-Risk Output Recommendations

For high-stakes outputs (security, auth, payments, data loss):

1. **Primary**: Use your best-fit framework for the task
2. **Verification Gate**: `think_verify_and_edit` or `think_rarr`
3. **Final Sanity**: `think_selfcheckgpt` or `think_metaqa`

## Large Repo/Docs Grounding

For large codebase or documentation grounding:

1. **Start**: `think_hyde` or `think_rag_fusion`
2. **If very large**: `think_raptor`
3. **If relationships matter**: `think_graphrag`
4. **Evaluate**: `think_ragas`

## Multi-Step Tool-Heavy Work

For complex tool-based tasks:

1. **Plan/Schedule**: `think_rewoo` for efficiency
2. **Multiple paths**: `think_lats` when backtracking likely
3. **Make CI green**: `think_swe_agent` for execution discipline
