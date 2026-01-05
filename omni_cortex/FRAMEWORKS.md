# Omni-Cortex: 40 Thinking Frameworks

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

## Code Frameworks (13)

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

---

## Framework Categories by Use Case

| Use Case | Recommended Frameworks |
|----------|----------------------|
| **Debugging** | active_inference, mcts_rstar, reverse_cot, self_debugging |
| **Refactoring** | graph_of_thoughts, least_to_most, everything_of_thought |
| **Architecture** | reason_flux, plan_and_solve, comparative_arch |
| **Security** | chain_of_verification, red_team |
| **Performance** | tree_of_thoughts, step_back |
| **Testing** | tdd_prompting, program_of_thoughts |
| **Documentation** | chain_of_note, skeleton_of_thought |
| **Quick Fixes** | system1, skeleton_of_thought |
| **Complex Problems** | everything_of_thought, mcts_rstar, coala |
| **Code Generation** | alphacodium, codechain, procoder, recode |
