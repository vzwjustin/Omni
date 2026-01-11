---
name: reasoning-framework-selection
description: Select and apply the optimal AI reasoning framework for any task
triggers:
  - "which framework"
  - "best approach for"
  - "how to think about"
  - "reasoning strategy"
---

# Reasoning Framework Selection

Use this skill when deciding which of the 62 thinking frameworks to apply.

## Framework Categories

### Debugging & Root Cause
- **active_inference** - Hypothesis → Predict → Compare → Update loop
- **mcts_rstar** - Monte Carlo Tree Search for complex multi-step bugs
- **reverse_cot** - Work backward from wrong output to source

### Architecture & Design
- **reason_flux** - Hierarchical: Template → Expand → Refine
- **comparative_arch** - Multiple approaches (readability/memory/speed)
- **state_machine** - FSM design before implementation

### Optimization
- **tree_of_thoughts** - BFS/DFS exploration of solutions
- **graph_of_thoughts** - Non-linear with merge/aggregate

### Quick Tasks
- **system1** - Fast heuristic for trivial fixes
- **skeleton_of_thought** - Outline-first with parallel expansion

## Selection Heuristics

1. **Bug/Error?** → active_inference or mcts_rstar
2. **Architecture?** → reason_flux or comparative_arch
3. **Optimization?** → tree_of_thoughts
4. **Simple fix?** → system1
5. **Unknown domain?** → self_discover
