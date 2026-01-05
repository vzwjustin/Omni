# Omni-Cortex Framework Selection Guide

A comprehensive guide to choosing the right thinking framework for your coding task. Omni-Cortex offers 62 specialized reasoning frameworks organized into 9 categories.

---

## Quick Reference Table

| Task Type | Primary Frameworks | When to Use |
|-----------|-------------------|-------------|
| **Debugging** | `active_inference`, `self_debugging`, `reverse_cot` | Bug hunting, root cause analysis, error investigation |
| **Architecture** | `reason_flux`, `plan_and_solve`, `multi_agent_debate` | System design, high-level decisions, trade-offs |
| **Code Generation** | `alphacodium`, `parsel`, `tdd_prompting` | New features, algorithms, implementations |
| **Refactoring** | `graph_of_thoughts`, `least_to_most`, `self_refine` | Code cleanup, restructuring, modernization |
| **Security** | `red_team`, `chain_of_verification` | Vulnerability scanning, security audits |
| **Quick Fixes** | `system1`, `skeleton_of_thought` | Trivial tasks, boilerplate, scaffolding |
| **Verification** | `verify_and_edit`, `self_consistency`, `selfcheckgpt` | Validation, proving correctness |
| **Agent Tasks** | `swe_agent`, `react`, `rewoo` | Multi-step automation, CI/CD fixes |
| **Research/RAG** | `chain_of_note`, `graphrag`, `raptor` | Understanding codebases, documentation |

---

## Frameworks by Task Type

### Debugging

When you have bugs, errors, or unexpected behavior.

| Framework | Best For | Vibe Phrases |
|-----------|----------|--------------|
| `active_inference` | Root cause analysis, hypothesis testing | "why is this broken", "wtf is wrong", "track down the bug", "investigate" |
| `self_debugging` | Preventing bugs, edge cases, off-by-one errors | "check my work", "trace through", "before committing", "dry run" |
| `reverse_cot` | Silent bugs, wrong outputs, calculation errors | "wrong output", "expected vs actual", "backwards debugging" |
| `mcts_rstar` | Complex multi-step bugs, thorough exploration | "really hard bug", "been stuck for hours", "deep issue" |
| `chain_of_code` | Logic puzzles, algorithmic debugging | "trace execution", "code flow", "step through" |
| `rubber_duck` | Architectural issues, blind spots | "explain to me", "help me think", "talk it through" |

### Architecture & Design

When you need to plan, design systems, or make high-level decisions.

| Framework | Best For | Vibe Phrases |
|-----------|----------|--------------|
| `reason_flux` | System design, hierarchical planning | "architect this", "design a system", "big picture", "greenfield" |
| `plan_and_solve` | Complex features, methodical development | "plan first", "strategy", "roadmap", "step by step plan" |
| `multi_agent_debate` | Trade-off analysis, design decisions | "pros and cons", "which approach", "React or Vue", "should I use" |
| `comparative_arch` | Multiple solution comparison | "compare approaches", "trade-offs", "show me options" |
| `state_machine` | UI logic, workflows, game states | "FSM", "state diagram", "transitions", "Redux" |
| `coala` | Multi-file awareness, large codebases | "across multiple files", "whole codebase", "monorepo" |

### Code Generation

When you need to write new code, algorithms, or features.

| Framework | Best For | Vibe Phrases |
|-----------|----------|--------------|
| `alphacodium` | Competitive programming, complex algorithms | "leetcode", "interview question", "algorithm challenge" |
| `parsel` | Complex functions from specs, modular systems | "dependency graph", "decompose into functions", "spec to code" |
| `tdd_prompting` | Test-first development, edge case coverage | "test first", "TDD", "write tests", "red green refactor" |
| `codechain` | Modular code, incremental refinement | "sub-modules", "component based", "modular code" |
| `docprompting` | API usage, library integration | "follow the docs", "api reference", "according to documentation" |
| `procoder` | Project-level code, type-safe code | "compiler errors", "fix imports", "codebase integration" |
| `program_of_thoughts` | Algorithms, data processing, math | "calculate", "compute", "data processing" |
| `pal` | Code as reasoning, numeric logic | "code to reason", "compute answer", "algorithm as code" |

### Refactoring

When you need to clean up, restructure, or modernize code.

| Framework | Best For | Vibe Phrases |
|-----------|----------|--------------|
| `graph_of_thoughts` | Spaghetti code, complex dependencies | "clean this up", "untangle this mess", "restructure" |
| `least_to_most` | Monolith refactoring, dependency management | "atomic functions", "bottom up", "building blocks" |
| `self_refine` | Code quality, iterative improvement | "polish", "make it better", "quality pass" |
| `everything_of_thought` | Major rewrites, complete redesigns | "major rewrite", "overhaul", "v2" |
| `codechain` | Module-by-module refinement | "incremental", "chain revisions", "refine modules" |

### Security & Verification

When you need to audit, validate, or ensure correctness.

| Framework | Best For | Vibe Phrases |
|-----------|----------|--------------|
| `red_team` | Security audits, vulnerability scanning | "security audit", "pen test", "OWASP", "find exploits" |
| `chain_of_verification` | Code review, bug prevention | "sanity check", "review my code", "verify", "PR review" |
| `verify_and_edit` | Surgical edits, fact-checking | "verify claims", "edit only wrong parts", "targeted fix" |
| `self_consistency` | Ambiguous bugs, tricky logic | "multiple answers", "consensus", "are you sure" |
| `selfcheckgpt` | Hallucination detection, high-stakes guidance | "sanity check", "might be wrong", "is this real" |
| `metaqa` | Edge case testing, policy consistency | "test with variations", "invariants", "works for this but not that" |
| `rarr` | Evidence-based answers, documentation | "prove it", "cite sources", "evidence based" |

### Quick Fixes

When you need fast, simple solutions.

| Framework | Best For | Vibe Phrases |
|-----------|----------|--------------|
| `system1` | Trivial tasks, one-liners | "quick question", "simple fix", "trivial", "no brainer" |
| `skeleton_of_thought` | Boilerplate, scaffolding | "scaffold", "quick template", "get me started" |
| `rar` | Clarifying vague requests | "rephrase", "what exactly", "be more specific" |
| `scratchpads` | Multi-step notes, state tracking | "working notes", "track state", "organized thinking" |

### Agent & Automation

When you need multi-step execution, tool use, or automation.

| Framework | Best For | Vibe Phrases |
|-----------|----------|--------------|
| `swe_agent` | Multi-file bugfixes, CI failures | "make tests pass", "CI green", "fix build" |
| `react` | Multi-step tasks with tools | "use tools", "step and observe", "API calls" |
| `rewoo` | Planned tool execution, cost control | "plan first then execute", "batch tools", "tool budget" |
| `lats` | Complex repo changes, backtracking | "multiple paths", "backtrack if fails", "action tree" |
| `mrkl` | Big systems, mixed domains | "specialized modules", "route to expert", "mixed domains" |
| `toolformer` | Smart tool selection | "when to use tools", "tool decision", "tool policy" |
| `reflexion` | Learning from failures | "learn from mistakes", "retry", "what went wrong" |

### Research & Understanding

When you need to understand, explore, or learn a codebase.

| Framework | Best For | Vibe Phrases |
|-----------|----------|--------------|
| `chain_of_note` | Understanding code, documentation | "explain", "what does this do", "learn the codebase" |
| `graphrag` | Architecture questions, dependencies | "entity relations", "how things connect", "impact analysis" |
| `raptor` | Large codebases, long documents | "hierarchical search", "drill down", "huge document" |
| `self_rag` | Selective retrieval, mixed knowledge | "retrieve when needed", "confidence based" |
| `hyde` | Fuzzy search, unclear queries | "hypothetical document", "fuzzy query", "vague search" |
| `rag_fusion` | Comprehensive search, better recall | "multiple queries", "combine searches", "thorough retrieval" |
| `step_back` | First principles, performance analysis | "big O", "fundamentals", "under the hood" |
| `analogical` | Creative solutions, pattern matching | "similar to", "like when", "pattern from" |

---

## Frameworks by Complexity

### Simple Tasks (Fast)

For quick fixes that need minimal thinking.

| Framework | Use When | Time |
|-----------|----------|------|
| `system1` | Trivial fixes, renaming, typos | ~seconds |
| `skeleton_of_thought` | Basic scaffolding, templates | ~seconds |
| `rar` | Need to clarify a vague request | ~seconds |

### Medium Complexity (Iterative)

For tasks requiring multiple passes or refinement.

| Framework | Use When | Iterations |
|-----------|----------|------------|
| `active_inference` | Hypothesis-driven debugging | 3-5 cycles |
| `self_refine` | Iterative quality improvement | 2-4 passes |
| `reflexion` | Learning from failed attempts | Until success |
| `tdd_prompting` | Test-first development | Red-green-refactor |
| `self_debugging` | Pre-validation of code | Single trace |

### Complex Tasks (Search)

For challenging problems requiring deep exploration.

| Framework | Use When | Approach |
|-----------|----------|----------|
| `mcts_rstar` | Multi-step bugs, optimization | Monte Carlo tree search |
| `tree_of_thoughts` | Multiple valid solutions exist | BFS/DFS exploration |
| `everything_of_thought` | Major changes, multiple approaches needed | Combined strategies |
| `lats` | Complex repo changes, uncertain root cause | Action tree search |
| `alphacodium` | Complex algorithms, competitive problems | Test-based iteration |

---

## Framework Chains

For complex tasks, Omni-Cortex can chain multiple frameworks. Here are proven patterns:

### Debugging Chains

| Pattern | Chain | Use Case |
|---------|-------|----------|
| Complex Bug | `self_ask` -> `active_inference` -> `verify_and_edit` | Multi-faceted bugs requiring decomposition |
| Silent Bug | `reverse_cot` -> `self_debugging` -> `selfcheckgpt` | Wrong outputs with no obvious error |
| Flaky Test | `active_inference` -> `tdd_prompting` -> `self_consistency` | Intermittent test failures |

### Code Generation Chains

| Pattern | Chain | Use Case |
|---------|-------|----------|
| Complex Feature | `plan_and_solve` -> `parsel` -> `tdd_prompting` -> `self_refine` | Large feature development |
| API Integration | `docprompting` -> `critic` -> `verify_and_edit` | Third-party API usage |
| Algorithm | `step_back` -> `alphacodium` -> `self_debugging` | Complex algorithm implementation |

### Refactoring Chains

| Pattern | Chain | Use Case |
|---------|-------|----------|
| Major Rewrite | `plan_and_solve` -> `graph_of_thoughts` -> `verify_and_edit` | Large-scale restructuring |
| Modular Extract | `least_to_most` -> `parsel` -> `self_refine` | Breaking down monoliths |
| Legacy Cleanup | `chain_of_note` -> `graph_of_thoughts` -> `tdd_prompting` | Modernizing old code |

### Architecture Chains

| Pattern | Chain | Use Case |
|---------|-------|----------|
| New System | `reason_flux` -> `multi_agent_debate` -> `plan_and_solve` | Greenfield architecture |
| Scale Decision | `step_back` -> `comparative_arch` -> `verify_and_edit` | Scaling/performance decisions |
| Workflow Design | `state_machine` -> `plan_and_solve` -> `critic` | Complex workflow implementation |

### Verification Chains

| Pattern | Chain | Use Case |
|---------|-------|----------|
| Security Audit | `red_team` -> `chain_of_verification` -> `verify_and_edit` | Security review |
| Claim Check | `self_ask` -> `rarr` -> `selfcheckgpt` | Verifying documentation/claims |
| Code Review | `chain_of_verification` -> `self_consistency` -> `verify_and_edit` | Thorough PR review |

### Agent Chains

| Pattern | Chain | Use Case |
|---------|-------|----------|
| CI Fix | `swe_agent` -> `tdd_prompting` -> `verify_and_edit` | Fixing broken builds |
| Multi-File | `coala` -> `swe_agent` -> `self_refine` | Changes spanning many files |
| Tool Heavy | `rewoo` -> `react` -> `reflexion` | Complex automation tasks |

### Research Chains

| Pattern | Chain | Use Case |
|---------|-------|----------|
| Large Codebase | `raptor` -> `graphrag` -> `chain_of_note` | Understanding huge repos |
| Fuzzy Search | `hyde` -> `rag_fusion` -> `rarr` | Finding info with vague queries |
| Dependency Map | `graphrag` -> `least_to_most` -> `chain_of_note` | Understanding module relationships |

---

## Vibe Examples

Omni-Cortex is designed for "vibe coding" - just describe what you want naturally.

### Debugging Vibes

```
"wtf is wrong with this code" -> active_inference
"why does this keep crashing" -> active_inference
"been stuck on this bug for hours" -> mcts_rstar
"wrong output, should be X but getting Y" -> reverse_cot
"works on my machine but not in prod" -> active_inference
"help me think through this problem" -> rubber_duck
```

### Architecture Vibes

```
"design a system for handling user uploads" -> reason_flux
"should I use microservices or monolith" -> multi_agent_debate
"plan out this feature before I code" -> plan_and_solve
"show me different approaches" -> comparative_arch
"what are the trade-offs" -> multi_agent_debate
```

### Code Generation Vibes

```
"write a leetcode solution" -> alphacodium
"implement this based on the API docs" -> docprompting
"write tests first then the code" -> tdd_prompting
"break this into smaller functions" -> parsel
"just scaffold the basic structure" -> skeleton_of_thought
```

### Refactoring Vibes

```
"clean up this spaghetti code" -> graph_of_thoughts
"this code is ugly, make it not suck" -> self_refine
"break this monolith into modules" -> least_to_most
"major rewrite needed" -> everything_of_thought
"polish and improve" -> self_refine
```

### Security Vibes

```
"is this code secure" -> chain_of_verification
"find vulnerabilities in this" -> red_team
"security audit please" -> red_team
"before I merge this, check for issues" -> chain_of_verification
```

### Quick Fix Vibes

```
"quick question" -> system1
"simple fix" -> system1
"just generate basic boilerplate" -> skeleton_of_thought
"trivial change" -> system1
```

### Understanding Vibes

```
"explain what this code does" -> chain_of_note
"how does this module connect to others" -> graphrag
"understand this codebase" -> chain_of_note
"what calls this function" -> graphrag
```

### Adaptive Vibes

```
"just figure it out" -> adaptive_injection
"do your thing" -> adaptive_injection
"whatever works" -> adaptive_injection
"your call" -> adaptive_injection
```

---

## Category Reference

All 62 frameworks organized by their primary category:

### Strategy (Hierarchical Planning)
- `reason_flux` - Template -> Expand -> Refine
- `self_discover` - Discover and apply reasoning patterns
- `buffer_of_thoughts` - Build context in thought buffer
- `coala` - Cognitive architecture for agents
- `least_to_most` - Bottom-up atomic decomposition
- `comparative_arch` - Multi-approach comparison
- `plan_and_solve` - Explicit planning before execution

### Search (Exploration)
- `mcts_rstar` - Monte Carlo Tree Search for code
- `tree_of_thoughts` - BFS/DFS exploration
- `graph_of_thoughts` - Non-linear reasoning with idea graphs
- `everything_of_thought` - Combined multi-approach

### Iterative (Refinement)
- `active_inference` - Hypothesis testing loop
- `multi_agent_debate` - Multiple perspectives debate
- `adaptive_injection` - Inject strategies as needed
- `re2` - Read-Execute-Evaluate loop
- `rubber_duck` - Socratic questioning
- `react` - Reasoning + Acting with tools
- `reflexion` - Self-evaluation with memory
- `self_refine` - Iterative self-critique

### Code (Generation & Analysis)
- `program_of_thoughts` - Step-by-step code reasoning
- `chain_of_verification` - Draft-Verify-Patch
- `critic` - Generate then critique
- `chain_of_code` - Code-based decomposition
- `self_debugging` - Mental execution trace
- `tdd_prompting` - Test-first development
- `reverse_cot` - Backward reasoning from output
- `alphacodium` - Test-based iterative generation
- `codechain` - Self-revisions with sub-modules
- `evol_instruct` - Evolutionary instruction complexity
- `llmloop` - Automated feedback loops
- `procoder` - Compiler-feedback refinement
- `recode` - Multi-candidate validation
- `pal` - Program-Aided Language
- `scratchpads` - Structured reasoning workspace
- `parsel` - Compositional code from specs
- `docprompting` - Documentation-driven generation

### Context (Understanding)
- `chain_of_note` - Research and note-taking
- `step_back` - Abstract principles first
- `analogical` - Find and adapt similar solutions
- `red_team` - Adversarial security analysis
- `state_machine` - Formal FSM design
- `chain_of_thought` - Step-by-step reasoning

### Fast (Quick)
- `skeleton_of_thought` - Outline first, fill in
- `system1` - Fast intuitive response

### Verification (Checking)
- `self_consistency` - Multi-sample voting
- `self_ask` - Sub-question decomposition
- `rar` - Rephrase-and-Respond
- `verify_and_edit` - Verify claims, edit failures
- `rarr` - Research, Augment, Revise
- `selfcheckgpt` - Hallucination detection
- `metaqa` - Metamorphic testing
- `ragas` - RAG Assessment

### Agent (Execution)
- `rewoo` - Reasoning Without Observation
- `lats` - Language Agent Tree Search
- `mrkl` - Modular Reasoning
- `swe_agent` - Repo-first execution
- `toolformer` - Smart tool selection

### RAG (Retrieval)
- `self_rag` - Self-triggered retrieval
- `hyde` - Hypothetical Document Embeddings
- `rag_fusion` - Multi-query rank fusion
- `raptor` - Hierarchical abstraction
- `graphrag` - Entity-relation grounding

---

## Best Practices

### 1. Let the Router Decide

Use the `reason` tool and describe your problem naturally. The router will pick the best framework:

```python
# Good - describe the vibe
reason("this function is returning wrong values and I can't figure out why")

# Less optimal - specifying framework
think_active_inference("debug this function")
```

### 2. Provide Context

Always include relevant code and file context for better framework selection:

```python
reason(
    query="why is this slow",
    context="""
    def process_items(items):
        result = []
        for item in items:
            for other in items:  # O(n^2) smell
                if item.matches(other):
                    result.append(item)
        return result
    """
)
```

### 3. Use Thread IDs for Multi-Turn

When debugging iteratively, use the same thread_id so Omni-Cortex remembers previous attempts:

```python
reason("this is broken", thread_id="debug-session-1")
# ... later ...
reason("still not working, tried your suggestion", thread_id="debug-session-1")
```

### 4. Trust Chain Recommendations

For complex tasks, the router may recommend chains. Let them run - they're designed to work together:

```
[plan_and_solve -> parsel -> tdd_prompting]
```

### 5. Match Complexity to Framework

- Simple typo? Use `system1`
- Debugging a nasty bug? Use `active_inference` or `mcts_rstar`
- Major architecture decision? Use `multi_agent_debate` or `reason_flux`

---

## Quick Decision Tree

```
START
  |
  v
Is it a quick fix or trivial question?
  |-- YES --> system1 or skeleton_of_thought
  |-- NO
      |
      v
    Is it a bug or error?
      |-- YES --> active_inference, self_debugging, or reverse_cot
      |-- NO
          |
          v
        Is it architecture or design?
          |-- YES --> reason_flux, plan_and_solve, or multi_agent_debate
          |-- NO
              |
              v
            Is it new code to write?
              |-- YES --> alphacodium, parsel, or tdd_prompting
              |-- NO
                  |
                  v
                Is it refactoring?
                  |-- YES --> graph_of_thoughts, least_to_most, or self_refine
                  |-- NO
                      |
                      v
                    Is it security-related?
                      |-- YES --> red_team or chain_of_verification
                      |-- NO
                          |
                          v
                        Need to understand the codebase?
                          |-- YES --> chain_of_note, graphrag, or raptor
                          |-- NO --> self_discover (novel/unknown problem)
```

---

## Summary

Omni-Cortex provides 62 frameworks across 9 categories. For most tasks:

1. **Just describe your problem** - The router will select the best framework
2. **Use vibe phrases** - Natural language triggers smart routing
3. **For complex tasks** - Trust the chain recommendations
4. **For iterative work** - Use thread_id for memory persistence

The goal is to let you focus on coding while Omni-Cortex handles the reasoning strategy.
