"""
Framework Prompt Templates for MCP Server

Contains the full prompt templates for each reasoning framework.
These are the actual prompts sent to the LLM when a framework tool is called.
"""

from typing import Any

# Framework definitions - what the LLM gets when it calls the tool
FRAMEWORKS: dict[str, dict[str, Any]] = {
    "reason_flux": {
        "category": "strategy",
        "description": "Hierarchical planning: Template -> Expand -> Refine",
        "best_for": ["architecture", "system design", "complex planning"],
        "prompt": """Apply ReasonFlux hierarchical planning:

TASK: {query}
CONTEXT: {context}

PHASE 1 - TEMPLATE: Create high-level structure with 3-5 major components
PHASE 2 - EXPAND: Detail each component (classes, functions, interfaces)
PHASE 3 - REFINE: Integrate into final plan with code skeleton""",
    },
    "self_discover": {
        "category": "strategy",
        "description": "Discover and apply reasoning patterns",
        "best_for": ["novel problems", "unknown domains"],
        "prompt": """Apply Self-Discover reasoning:

TASK: {query}
CONTEXT: {context}

1. SELECT: Which patterns apply? (decomposition, analogy, abstraction, constraints)
2. ADAPT: Customize patterns for this specific task
3. IMPLEMENT: Apply your customized approach
4. VERIFY: Check completeness""",
    },
    "buffer_of_thoughts": {
        "category": "strategy",
        "description": "Build context in a thought buffer",
        "best_for": ["multi-part problems", "complex context"],
        "prompt": """Apply Buffer of Thoughts:

TASK: {query}
CONTEXT: {context}

Build your thought buffer:
- INIT: Key facts and constraints
- ADD: Problem analysis
- ADD: Possible approaches
- ADD: Decision and reasoning
- OUTPUT: Synthesize final solution""",
    },
    "coala": {
        "category": "strategy",
        "description": "Cognitive architecture for agents",
        "best_for": ["autonomous tasks", "agent behavior"],
        "prompt": """Apply COALA cognitive architecture:

TASK: {query}
CONTEXT: {context}

1. PERCEPTION: Current state and resources
2. MEMORY: Relevant knowledge and patterns
3. REASONING: Analyze and plan
4. ACTION: Execute plan
5. LEARNING: What worked? What to improve?""",
    },
    "mcts_rstar": {
        "category": "search",
        "description": "Monte Carlo Tree Search exploration for code",
        "best_for": ["complex bugs", "multi-step optimization", "thorough search"],
        "prompt": """Apply rStar-Code MCTS reasoning:

TASK: {query}
CONTEXT: {context}

1. SELECT: Most promising focus area using UCT scoring
2. EXPAND: Generate 2-3 possible modifications
3. SIMULATE: Trace consequences of each path
4. EVALUATE: Score each path (0-1)
5. BACKPROPAGATE: Update parent scores
6. ITERATE: Repeat until confidence threshold or max depth""",
    },
    "tree_of_thoughts": {
        "category": "search",
        "description": "Explore multiple paths, pick best",
        "best_for": ["design decisions", "multiple valid approaches"],
        "prompt": """Apply Tree of Thoughts:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create 3 distinct approaches with pros/cons
2. EVALUATE: Score each (feasibility, effectiveness, simplicity)
3. EXPAND: Develop the best approach fully
4. SYNTHESIZE: Final solution with reasoning""",
    },
    "graph_of_thoughts": {
        "category": "search",
        "description": "Non-linear reasoning with idea graphs",
        "best_for": ["complex dependencies", "interconnected problems"],
        "prompt": """Apply Graph of Thoughts:

TASK: {query}
CONTEXT: {context}

1. NODES: Identify key concepts/components
2. EDGES: Map relationships between them
3. TRAVERSE: Find the solution path through the graph
4. SYNTHESIZE: Combine insights into solution""",
    },
    "everything_of_thought": {
        "category": "search",
        "description": "Combine multiple reasoning approaches",
        "best_for": ["complex novel problems", "when one approach isn't enough"],
        "prompt": """Apply Everything of Thought:

TASK: {query}
CONTEXT: {context}

1. MULTI-APPROACH: Apply analytical, creative, critical, practical thinking
2. CROSS-POLLINATE: Find synergies between approaches
3. SYNTHESIZE: Create unified solution
4. VALIDATE: Check against all perspectives""",
    },
    "active_inference": {
        "category": "iterative",
        "description": "Hypothesis testing loop",
        "best_for": ["debugging", "investigation", "root cause analysis"],
        "prompt": """Apply Active Inference:

TASK: {query}
CONTEXT: {context}

1. OBSERVE: Current state, form hypotheses
2. PREDICT: What should we expect if hypothesis is true?
3. TEST: Gather evidence, update beliefs
4. ACT: Implement fix based on best hypothesis
5. VERIFY: Confirm the fix worked""",
    },
    "multi_agent_debate": {
        "category": "iterative",
        "description": "Multiple perspectives debate",
        "best_for": ["design decisions", "trade-off analysis"],
        "prompt": """Apply Multi-Agent Debate:

TASK: {query}
CONTEXT: {context}

Argue from these perspectives:
- PRAGMATIST: What's the simplest working solution?
- ARCHITECT: What's most maintainable/scalable?
- SECURITY: What are the risks?
- PERFORMANCE: What's most efficient?

DEBATE the trade-offs, then SYNTHESIZE a balanced solution.""",
    },
    "adaptive_injection": {
        "category": "iterative",
        "description": "Inject strategies as needed",
        "best_for": ["evolving understanding", "adaptive problem solving"],
        "prompt": """Apply Adaptive Injection:

TASK: {query}
CONTEXT: {context}

As you work, inject strategies when needed:
- If stuck -> step back and abstract
- If complex -> decompose into parts
- If uncertain -> explore alternatives
- If risky -> add verification steps

Continue until complete.""",
    },
    "re2": {
        "category": "iterative",
        "description": "Read-Execute-Evaluate loop",
        "best_for": ["specifications", "requirements implementation"],
        "prompt": """Apply RE2 (Read-Execute-Evaluate):

TASK: {query}
CONTEXT: {context}

1. READ: Parse requirements, list acceptance criteria
2. EXECUTE: Implement, referencing each requirement
3. EVALUATE: Check against requirements, fix gaps

Repeat until all requirements are satisfied.""",
    },
    "program_of_thoughts": {
        "category": "code",
        "description": "Step-by-step code reasoning",
        "best_for": ["algorithms", "data processing", "math"],
        "prompt": """Apply Program of Thoughts:

TASK: {query}
CONTEXT: {context}

1. UNDERSTAND: What's input? What's output? What transformations?
2. DECOMPOSE: Break into computational steps
3. CODE: Write each step with clear comments
4. TRACE: Walk through with sample input
5. OUTPUT: Complete solution""",
    },
    "chain_of_verification": {
        "category": "code",
        "description": "Draft-Verify-Patch cycle",
        "best_for": ["security review", "code quality", "bug prevention"],
        "prompt": """Apply Chain of Verification:

TASK: {query}
CONTEXT: {context}

1. DRAFT: Create initial solution
2. VERIFY: Check for security issues, bugs, best practice violations
3. PATCH: Fix all identified issues
4. VALIDATE: Confirm fixes, no regressions""",
    },
    "critic": {
        "category": "code",
        "description": "Generate then critique",
        "best_for": ["API design", "interface validation"],
        "prompt": """Apply Critic framework:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create initial solution
2. CRITIQUE: What works? What's missing? What could break?
3. REVISE: Address each criticism
4. FINAL CHECK: Verify improvements""",
    },
    "chain_of_note": {
        "category": "context",
        "description": "Research and note-taking approach",
        "best_for": ["understanding code", "documentation", "exploration"],
        "prompt": """Apply Chain of Note:

TASK: {query}
CONTEXT: {context}

NOTE 1 - Observations: What do you see?
NOTE 2 - Connections: How do pieces relate?
NOTE 3 - Inferences: What can you conclude?
NOTE 4 - Synthesis: Complete answer""",
    },
    "step_back": {
        "category": "context",
        "description": "Abstract principles first, then apply",
        "best_for": ["optimization", "performance", "architectural decisions"],
        "prompt": """Apply Step-Back reasoning:

TASK: {query}
CONTEXT: {context}

1. STEP BACK: What category is this? What principles apply?
2. ABSTRACT: Key constraints, trade-offs, proven approaches
3. APPLY: Map abstract principles to concrete solution
4. VERIFY: Solution follows identified principles""",
    },
    "analogical": {
        "category": "context",
        "description": "Find and adapt similar solutions",
        "best_for": ["creative solutions", "pattern matching"],
        "prompt": """Apply Analogical reasoning:

TASK: {query}
CONTEXT: {context}

1. FIND ANALOGIES: What similar problems have been solved? (2-3)
2. MAP: How does the best analogy map to this problem?
3. ADAPT: What transfers? What's different?
4. IMPLEMENT: Build solution using adapted approach""",
    },
    "skeleton_of_thought": {
        "category": "fast",
        "description": "Outline first, fill in details",
        "best_for": ["boilerplate", "quick scaffolding"],
        "prompt": """Apply Skeleton of Thought:

TASK: {query}
CONTEXT: {context}

1. SKELETON: High-level structure (components, interfaces)
2. FLESH OUT: Add implementation details
3. CONNECT: Handle integration and edge cases""",
    },
    "system1": {
        "category": "fast",
        "description": "Fast intuitive response",
        "best_for": ["simple questions", "quick fixes"],
        "prompt": """Quick response for: {query}

Context: {context}

Provide a direct, efficient answer. Focus on the most likely correct solution.""",
    },
    # 2026 Edition frameworks
    "chain_of_code": {
        "category": "code",
        "description": "Code-based problem decomposition",
        "best_for": ["logic puzzles", "algorithmic debugging", "structured thinking"],
        "prompt": """Apply Chain-of-Code reasoning:

TASK: {query}
CONTEXT: {context}

1. DECOMPOSE: Break the problem into code blocks
2. TRACE: Walk through execution step by step
3. IDENTIFY: Pinpoint where logic diverges from intent
4. FIX: Apply targeted corrections with explanation
5. VERIFY: Trace corrected code to confirm fix""",
    },
    "self_debugging": {
        "category": "code",
        "description": "Mental execution trace before presenting",
        "best_for": ["preventing bugs", "edge case handling", "off-by-one errors"],
        "prompt": """Apply Self-Debugging:

TASK: {query}
CONTEXT: {context}

1. DRAFT: Write initial solution
2. TRACE: Mentally execute with sample inputs
3. EDGES: Test boundary conditions (0, 1, empty, null, max)
4. CATCH: Identify potential bugs before presenting
5. FIX: Present corrected code with trace verification""",
    },
    "tdd_prompting": {
        "category": "code",
        "description": "Test-first development approach",
        "best_for": ["new features", "edge case coverage", "robust implementations"],
        "prompt": """Apply TDD Prompting:

TASK: {query}
CONTEXT: {context}

1. TESTS FIRST: Write test cases covering:
   - Happy path
   - Edge cases
   - Error conditions
2. IMPLEMENT: Write minimal code to pass tests
3. REFACTOR: Clean up while keeping tests green
4. VERIFY: All tests pass with clean implementation""",
    },
    "reverse_cot": {
        "category": "code",
        "description": "Backward reasoning from output delta",
        "best_for": ["silent bugs", "wrong outputs", "calculation errors"],
        "prompt": """Apply Reverse Chain-of-Thought:

TASK: {query}
CONTEXT: {context}

1. EXPECTED: What output should the code produce?
2. ACTUAL: What is it actually producing?
3. DELTA: What's the difference?
4. BACKTRACK: Work backward from the delta to find the cause
5. FIX: Apply correction and verify expected output""",
    },
    "rubber_duck": {
        "category": "iterative",
        "description": "Socratic questioning for self-discovery",
        "best_for": ["architectural issues", "blind spots", "stuck problems"],
        "prompt": """Apply Rubber Duck Debugging:

TASK: {query}
CONTEXT: {context}

Guide through questions:
1. What are you trying to accomplish?
2. What have you tried so far?
3. What happened vs what you expected?
4. What assumptions are you making?
5. What haven't you checked yet?

Lead to insight through questioning.""",
    },
    "react": {
        "category": "iterative",
        "description": "Reasoning + Acting with tools",
        "best_for": ["multi-step tasks", "tool use", "investigation"],
        "prompt": """Apply ReAct (Reasoning + Acting):

TASK: {query}
CONTEXT: {context}

Iterate:
THOUGHT: What do I need to figure out next?
ACTION: What tool/search/command would help?
OBSERVATION: What did I learn?

Continue until solution is found. Show your reasoning chain.""",
    },
    "reflexion": {
        "category": "iterative",
        "description": "Self-evaluation with memory-based learning",
        "best_for": ["learning from failures", "iterative improvement", "retry scenarios"],
        "prompt": """Apply Reflexion:

TASK: {query}
CONTEXT: {context}

1. ATTEMPT: Make first solution attempt
2. EVALUATE: Did it work? What went wrong?
3. REFLECT: What lessons to remember?
4. RETRY: Apply lessons to improved attempt
5. CONVERGE: Iterate until successful""",
    },
    "self_refine": {
        "category": "iterative",
        "description": "Iterative self-critique and improvement",
        "best_for": ["code quality", "documentation", "polish"],
        "prompt": """Apply Self-Refine:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create initial solution
2. CRITIQUE: What could be better? (readability, efficiency, edge cases)
3. REFINE: Address each critique point
4. REPEAT: Continue until no more improvements
5. FINALIZE: Present polished solution""",
    },
    "least_to_most": {
        "category": "strategy",
        "description": "Bottom-up atomic function decomposition",
        "best_for": ["complex systems", "monolith refactoring", "dependency management"],
        "prompt": """Apply Least-to-Most Decomposition:

TASK: {query}
CONTEXT: {context}

1. IDENTIFY: What are the atomic sub-problems?
2. ORDER: Sort by dependency (least dependent first)
3. SOLVE: Build solutions bottom-up
4. COMPOSE: Combine into full solution
5. VERIFY: Check all pieces integrate correctly""",
    },
    "comparative_arch": {
        "category": "strategy",
        "description": "Multi-approach comparison (readability/memory/speed)",
        "best_for": ["optimization", "architecture decisions", "trade-off analysis"],
        "prompt": """Apply Comparative Architecture:

TASK: {query}
CONTEXT: {context}

Generate THREE approaches:
1. READABLE: Prioritize clarity and maintainability
2. EFFICIENT: Optimize for memory/space
3. FAST: Optimize for speed/time

For each: show code, pros, cons, big-O analysis.
RECOMMEND: Which approach and why.""",
    },
    "plan_and_solve": {
        "category": "strategy",
        "description": "Explicit planning before execution",
        "best_for": ["complex features", "methodical development", "avoiding rushed code"],
        "prompt": """Apply Plan-and-Solve:

TASK: {query}
CONTEXT: {context}

PHASE 1 - PLAN:
- What components are needed?
- What's the order of implementation?
- What are the risks/blockers?

PHASE 2 - SOLVE:
- Implement according to plan
- Note any deviations and why
- Verify against plan""",
    },
    "red_team": {
        "category": "context",
        "description": "Adversarial security analysis (STRIDE, OWASP)",
        "best_for": ["security audits", "vulnerability scanning", "threat modeling"],
        "prompt": """Apply Red-Team Security Analysis:

TASK: {query}
CONTEXT: {context}

Analyze using STRIDE:
- SPOOFING: Identity issues?
- TAMPERING: Data integrity issues?
- REPUDIATION: Audit issues?
- INFO DISCLOSURE: Leakage issues?
- DENIAL OF SERVICE: Availability issues?
- ELEVATION OF PRIVILEGE: Access control issues?

OWASP Top 10 check. Provide fixes for each finding.""",
    },
    "state_machine": {
        "category": "context",
        "description": "Formal FSM design before coding",
        "best_for": ["UI logic", "workflow systems", "game states"],
        "prompt": """Apply State-Machine Reasoning:

TASK: {query}
CONTEXT: {context}

1. STATES: Enumerate all possible states
2. TRANSITIONS: What triggers state changes?
3. GUARDS: What conditions must be met?
4. ACTIONS: What happens on transition?
5. IMPLEMENT: Code the state machine with clear structure""",
    },
    "chain_of_thought": {
        "category": "context",
        "description": "Step-by-step reasoning chain",
        "best_for": ["complex reasoning", "logical deduction", "problem solving"],
        "prompt": """Apply Chain-of-Thought:

TASK: {query}
CONTEXT: {context}

Think step by step:
STEP 1: [First logical step]
STEP 2: [Building on step 1]
STEP 3: [Continue reasoning]
...
CONCLUSION: [Final answer based on chain]

Show your complete reasoning process.""",
    },
    "alphacodium": {
        "category": "code",
        "description": "Test-based multi-stage iterative code generation",
        "best_for": ["competitive programming", "complex algorithms", "interview problems"],
        "prompt": """Apply AlphaCodium:

TASK: {query}
CONTEXT: {context}

STAGE 1 - UNDERSTAND:
- Parse problem constraints
- Identify input/output format
- Note edge cases

STAGE 2 - GENERATE TESTS:
- Public test cases
- Edge cases
- Large input cases

STAGE 3 - ITERATIVE CODE:
- Generate solution
- Test against cases
- Fix failures
- Repeat until all pass""",
    },
    "codechain": {
        "category": "code",
        "description": "Chain of self-revisions guided by sub-modules",
        "best_for": ["modular code generation", "incremental refinement", "component development"],
        "prompt": """Apply CodeChain:

TASK: {query}
CONTEXT: {context}

1. MODULES: Identify independent sub-modules needed
2. GENERATE: Create each module with clear interface
3. INTEGRATE: Connect modules together
4. REVISE: Self-critique each module, improve
5. VALIDATE: Test integrated solution""",
    },
    "evol_instruct": {
        "category": "code",
        "description": "Evolutionary instruction complexity for code",
        "best_for": ["challenging code problems", "constraint-based coding", "extending solutions"],
        "prompt": """Apply Evol-Instruct:

TASK: {query}
CONTEXT: {context}

1. BASE: Solve the base problem
2. EVOLVE: Add constraint/complexity:
   - Additional requirements
   - Performance constraints
   - Edge case handling
3. SOLVE EVOLVED: Handle new complexity
4. REPEAT: Continue evolving until robust""",
    },
    "llmloop": {
        "category": "code",
        "description": "Automated iterative feedback loops for code+tests",
        "best_for": ["code quality assurance", "production-ready code", "CI/CD preparation"],
        "prompt": """Apply LLMLoop:

TASK: {query}
CONTEXT: {context}

LOOP:
1. GENERATE: Create code + tests
2. RUN: Execute tests (mentally trace if needed)
3. ANALYZE: What failed? Why?
4. FIX: Apply corrections
5. ITERATE: Until all quality checks pass

Quality checks: correctness, edge cases, readability, efficiency.""",
    },
    "procoder": {
        "category": "code",
        "description": "Compiler-feedback-guided iterative refinement",
        "best_for": ["project-level code generation", "API usage", "type-safe code"],
        "prompt": """Apply ProCoder:

TASK: {query}
CONTEXT: {context}

1. ANALYZE: Understand project context, types, APIs
2. GENERATE: Create type-safe code using project patterns
3. COMPILE: Check for type errors, import issues
4. FIX: Address any compiler/linter feedback
5. INTEGRATE: Ensure clean integration with existing code""",
    },
    "recode": {
        "category": "code",
        "description": "Multi-candidate validation with CFG-based debugging",
        "best_for": ["reliable code generation", "high-stakes code", "mission-critical systems"],
        "prompt": """Apply RECODE:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create 3 candidate solutions
2. VALIDATE: Test each independently
3. ANALYZE CFG: Check control flow for issues
4. VOTE: Select best candidate based on:
   - Correctness
   - Robustness
   - Clarity
5. FINALIZE: Present winning candidate with confidence assessment""",
    },
    # Verification frameworks
    "self_consistency": {
        "category": "verification",
        "description": "Multi-sample voting for reliable answers",
        "best_for": ["ambiguous bugs", "tricky logic", "multiple plausible fixes"],
        "prompt": """Apply Self-Consistency:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create 3-7 independent solution paths (don't reuse reasoning)
2. NORMALIZE: Structure as (hypothesis -> fix -> expected evidence)
3. SCORE: Rate consistency, constraint fit, simplicity, testability
4. SELECT: Choose winner; keep runner-up if high risk
5. OUTPUT: Final solution + why it won + validation checks""",
    },
    "self_ask": {
        "category": "verification",
        "description": "Sub-question decomposition before solving",
        "best_for": ["unclear tickets", "missing requirements", "multi-part debugging"],
        "prompt": """Apply Self-Ask:

TASK: {query}
CONTEXT: {context}

1. GENERATE: List 5-12 sub-questions that must be answered
2. CLASSIFY: Mark each as must-know vs nice-to-know
3. ANSWER: Address must-know using context, tools, or docs
4. RECOMPOSE: Build final solution with stated assumptions
5. VALIDATE: Check against acceptance criteria""",
    },
    "rar": {
        "category": "verification",
        "description": "Rephrase-and-Respond for clarity",
        "best_for": ["vague prompts", "poorly written bug reports", "ambiguous requirements"],
        "prompt": """Apply Rephrase-and-Respond (RaR):

TASK: {query}
CONTEXT: {context}

1. REPHRASE: Write precise task spec (objective, constraints, acceptance criteria)
2. CONFIRM: Check for internal consistency
3. SOLVE: Implement strictly against rephrased spec
4. VERIFY: Map results to acceptance criteria""",
    },
    "verify_and_edit": {
        "category": "verification",
        "description": "Verify claims, edit only failures",
        "best_for": ["code review", "security guidance", "implementation plans", "surgical edits"],
        "prompt": """Apply Verify-and-Edit:

TASK: {query}
CONTEXT: {context}

1. DRAFT: Create initial solution
2. EXTRACT: Identify verifiable claims and risky assertions
3. VERIFY: Check each via context, tests, docs (mark assumptions)
4. EDIT: Fix ONLY failing sections; preserve good sections
5. FINALIZE: Produce verification ledger""",
    },
    "rarr": {
        "category": "verification",
        "description": "Research, Augment, Revise - evidence-driven revision",
        "best_for": ["external docs", "repo knowledge", "prove-it requirements"],
        "prompt": """Apply RARR (Research, Augment, Revise):

TASK: {query}
CONTEXT: {context}

1. DRAFT: Create initial output
2. GENERATE: Create 3-8 targeted evidence queries
3. RETRIEVE: Gather relevant evidence
4. REVISE: Update to align with evidence; remove unsupported claims
5. CITE: Provide anchors (file:line when possible)""",
    },
    "selfcheckgpt": {
        "category": "verification",
        "description": "Hallucination detection via sampling consistency",
        "best_for": ["high-stakes guidance", "unfamiliar libraries", "final pre-flight gate"],
        "prompt": """Apply SelfCheckGPT:

TASK: {query}
CONTEXT: {context}

1. IDENTIFY: Flag high-risk claims in draft
2. SAMPLE: Generate multiple re-answers focusing on risky claims
3. CHECK: Compare answers; flag disagreement hotspots
4. REPLACE: Disputed content with verified evidence or explicit uncertainty
5. OUTPUT: Final result + risk register""",
    },
    "metaqa": {
        "category": "verification",
        "description": "Metamorphic testing for reasoning reliability",
        "best_for": ["brittle reasoning", "edge cases", "policy consistency"],
        "prompt": """Apply MetaQA (Metamorphic QA):

TASK: {query}
CONTEXT: {context}

1. DEFINE: Invariants that must stay true
2. GENERATE: 3-10 transformed variants (rewording, constraint tweaks)
3. SOLVE: Answer each variant
4. CHECK: Identify contradictions between answers
5. PATCH: Fix solution to satisfy invariants across variants""",
    },
    "ragas": {
        "category": "verification",
        "description": "RAG Assessment - evaluate retrieval quality",
        "best_for": ["RAG pipelines", "retrieval quality", "source grounding"],
        "prompt": """Apply RAGAS (RAG Assessment):

TASK: {query}
CONTEXT: {context}

Evaluate across dimensions:
1. RELEVANCE: Are retrieved chunks relevant?
2. FAITHFULNESS: Does answer stick to sources?
3. COMPLETENESS: All aspects covered?
4. NOISE: Any irrelevant/misleading content?
5. DIAGNOSE: Failure modes + corrective actions""",
    },
    # Agent frameworks
    "rewoo": {
        "category": "agent",
        "description": "Reasoning Without Observation - plan then execute",
        "best_for": ["multi-step tasks", "cost control", "plan-once-execute-clean"],
        "prompt": """Apply ReWOO (Reasoning Without Observation):

TASK: {query}
CONTEXT: {context}

1. PLAN: Create tool-free plan with expected observations
2. SCHEDULE: Convert to tool call schedule (what, when, why)
3. EXECUTE: Run tools in batches; collect observations
4. REVISE: Update plan only if observations contradict
5. FINALIZE: Result + checks + next actions""",
    },
    "lats": {
        "category": "agent",
        "description": "Language Agent Tree Search over action sequences",
        "best_for": ["complex repo changes", "multiple fix paths", "uncertain root cause"],
        "prompt": """Apply LATS (Language Agent Tree Search):

TASK: {query}
CONTEXT: {context}

1. PRIMITIVES: Define actions (edit, test, inspect, search)
2. EXPAND: Generate multiple action sequence branches
3. SCORE: Rate by likelihood, risk, effort, rollback ease
4. EXECUTE: Run best branch; backtrack if fails
5. FINALIZE: Chosen path + alternatives considered""",
    },
    "mrkl": {
        "category": "agent",
        "description": "Modular Reasoning with specialized modules",
        "best_for": ["big systems", "mixed domains", "tool-rich setups"],
        "prompt": """Apply MRKL (Modular Reasoning, Knowledge, Language):

TASK: {query}
CONTEXT: {context}

1. DECOMPOSE: Break into module tasks (Security, Perf, Test, Product)
2. ROUTE: Specify input/output/validation for each module
3. EXECUTE: Run modules with clear interfaces
4. RECONCILE: Resolve conflicting outputs
5. SYNTHESIZE: Combine into final decision + verification plan""",
    },
    "swe_agent": {
        "category": "agent",
        "description": "Repo-first execution loop - inspect/edit/run/iterate",
        "best_for": ["multi-file bugfixes", "CI failures", "make tests pass"],
        "prompt": """Apply SWE-Agent:

TASK: {query}
CONTEXT: {context}

1. INSPECT: Entry points, failing tests, logs, config
2. IDENTIFY: Minimal change set
3. PATCH: Apply changes in small increments
4. VERIFY: Run tests/lint/typecheck
5. ITERATE: Until green, then summarize + remaining risks""",
    },
    "toolformer": {
        "category": "agent",
        "description": "Smart tool selection policy",
        "best_for": ["router logic", "preventing pointless calls", "standardized prompts"],
        "prompt": """Apply Toolformer (Tool Selection Policy):

TASK: {query}
CONTEXT: {context}

1. IDENTIFY: What claims require external confirmation?
2. JUSTIFY: Does tool call materially reduce uncertainty?
3. OPTIMIZE: Specify tight inputs + expected outputs
4. INTEGRATE: Parse results, update confidence
5. DOCUMENT: Tool decision rationale""",
    },
    # RAG frameworks
    "self_rag": {
        "category": "rag",
        "description": "Self-triggered selective retrieval",
        "best_for": ["mixed knowledge tasks", "large corpora", "minimizing irrelevant retrieval"],
        "prompt": """Apply Self-RAG:

TASK: {query}
CONTEXT: {context}

1. DRAFT: Write with confidence tags (HIGH/MEDIUM/LOW)
2. IDENTIFY: Gaps in LOW-confidence segments
3. RETRIEVE: Fetch evidence only for uncertain parts
4. UPDATE: Revise only uncertain segments
5. CRITIQUE: Confirm groundedness; remove unsupported claims""",
    },
    "hyde": {
        "category": "rag",
        "description": "Hypothetical Document Embeddings for better retrieval",
        "best_for": ["fuzzy search", "unclear intent", "broad problems"],
        "prompt": """Apply HyDE (Hypothetical Document Embeddings):

TASK: {query}
CONTEXT: {context}

1. HYPOTHESIZE: Write ideal answer document
2. EXTRACT: Convert to strong retrieval queries
3. RETRIEVE: Find real documents/snippets
4. GROUND: Answer based on retrieved evidence
5. CITE: Evidence anchors for claims""",
    },
    "rag_fusion": {
        "category": "rag",
        "description": "Multi-query retrieval with rank fusion",
        "best_for": ["improving recall", "complex queries", "noisy corpora"],
        "prompt": """Apply RAG-Fusion:

TASK: {query}
CONTEXT: {context}

1. GENERATE: 3-8 diverse queries (synonyms, facets, constraints)
2. RETRIEVE: Top-K per query
3. FUSE: Dedupe + reciprocal rank merge
4. SYNTHESIZE: Answer using fused evidence
5. COVERAGE: Ensure all query facets addressed""",
    },
    "raptor": {
        "category": "rag",
        "description": "Hierarchical abstraction retrieval for large docs",
        "best_for": ["huge repos", "long design docs", "monorepos"],
        "prompt": """Apply RAPTOR (Recursive Abstraction Retrieval):

TASK: {query}
CONTEXT: {context}

1. HIERARCHY: Summaries at chunk/section/doc levels
2. RETRIEVE TOP-DOWN: High-level first, then drill down
3. GATHER: Supporting details from lower levels
4. SYNTHESIZE: Combine abstraction with specifics
5. ANCHOR: Both overview context and specific citations""",
    },
    "graphrag": {
        "category": "rag",
        "description": "Entity-relation grounding for dependencies",
        "best_for": ["architecture questions", "module relationships", "impact analysis"],
        "prompt": """Apply GraphRAG:

TASK: {query}
CONTEXT: {context}

1. EXTRACT: Entities (modules, APIs, tables, services)
2. MAP: Relations (calls, reads/writes, owns, triggers)
3. GRAPH: Build conceptual relation map
4. QUERY: Trace paths, find blast radius, identify dependencies
5. CITE: Show relationship chains supporting claims""",
    },
    # Additional code frameworks
    "pal": {
        "category": "code",
        "description": "Program-Aided Language - code as reasoning substrate",
        "best_for": ["algorithms", "parsing", "numeric logic", "validation"],
        "prompt": """Apply PAL (Program-Aided Language):

TASK: {query}
CONTEXT: {context}

1. TRANSLATE: Convert reasoning to a small program
2. PSEUDOCODE: Start with pseudocode if unsure
3. IMPLEMENT: Write executable code
4. VALIDATE: Test with examples (normal + edge cases)
5. CONVERT: Translate verified code to final solution""",
    },
    "scratchpads": {
        "category": "code",
        "description": "Structured intermediate reasoning workspace",
        "best_for": ["multi-step fixes", "multi-constraint reasoning", "state tracking"],
        "prompt": """Apply Scratchpads:

TASK: {query}
CONTEXT: {context}

Maintain structured scratchpad:
- FACTS: Key known information
- CONSTRAINTS: Must-do and cannot-do
- PLAN: Ordered approach
- RISKS: Potential issues + mitigations
- CHECKS: Verification steps

Update as you work; present final with scratchpad summary.""",
    },
    "parsel": {
        "category": "code",
        "description": "Compositional code generation from natural language specs",
        "best_for": ["complex functions", "dependency graphs", "spec-to-code", "modular systems"],
        "prompt": """Apply Parsel (Compositional Code Generation):

TASK: {query}
CONTEXT: {context}

1. DECOMPOSE: Break task into individual function specs
   - Name, description, inputs, outputs, dependencies
2. GRAPH: Build dependency order (no cycles)
3. BASE: Generate leaf functions first (no dependencies)
4. COMPOSE: Build dependent functions using generated ones
5. INTEGRATE: Combine into cohesive module with entry point""",
    },
    "docprompting": {
        "category": "code",
        "description": "Documentation-driven code generation",
        "best_for": ["API usage", "library integration", "following docs", "correct usage"],
        "prompt": """Apply DocPrompting (Documentation-Driven):

TASK: {query}
CONTEXT: {context}

1. IDENTIFY: List required APIs/libraries
2. RETRIEVE: Find relevant documentation + examples
3. EXTRACT: Note function signatures, usage patterns, idioms
4. GENERATE: Write code following doc patterns
5. VERIFY: Cross-check against doc for correct params, types, error handling""",
    },
}


def get_framework_prompt(name: str, query: str, context: str = "None provided") -> str:
    """Get formatted prompt for a framework."""
    if name not in FRAMEWORKS:
        return f"Unknown framework: {name}"
    fw = FRAMEWORKS[name]
    return fw["prompt"].format(query=query, context=context)


def get_framework_categories() -> list[str]:
    """Get list of unique categories."""
    return list(set(fw["category"] for fw in FRAMEWORKS.values()))
