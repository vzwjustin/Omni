"""
Framework Definitions - First 10 framework specifications.

Registers framework specs using data from server/main.py FRAMEWORKS dict.
"""

from .registry import register, FrameworkSpec, FrameworkCategory


# =============================================================================
# STRATEGY FRAMEWORKS
# =============================================================================

register(FrameworkSpec(
    name="reason_flux",
    category=FrameworkCategory.STRATEGY,
    description="Hierarchical planning: Template -> Expand -> Refine",
    best_for=["architecture", "system design", "complex planning"],
    vibes=[
        "design a system", "architect this", "how should I structure",
        "plan this out", "big picture", "system design", "make it scalable",
        "build from scratch", "greenfield", "new project", "overall approach",
        "microservices", "monolith", "architecture", "high level design",
        "design patterns", "SOLID", "clean architecture", "hexagonal",
        "domain driven", "DDD", "event driven", "CQRS", "event sourcing",
        "API design", "schema design", "data model", "ERD",
        "infrastructure", "deployment", "CI/CD pipeline",
        "aws architecture", "cloud native", "serverless", "kubernetes",
        "folder structure", "project layout", "directory structure"
    ],
    prompt_template="""Apply ReasonFlux hierarchical planning:

TASK: {query}
CONTEXT: {context}

PHASE 1 - TEMPLATE: Create high-level structure with 3-5 major components
PHASE 2 - EXPAND: Detail each component (classes, functions, interfaces)
PHASE 3 - REFINE: Integrate into final plan with code skeleton"""
))


register(FrameworkSpec(
    name="self_discover",
    category=FrameworkCategory.STRATEGY,
    description="Discover and apply reasoning patterns",
    best_for=["novel problems", "unknown domains"],
    vibes=[
        "I have no idea", "weird problem", "never seen this",
        "creative solution", "think outside box", "novel approach",
        "unconventional", "unique situation", "edge case",
        "bizarre", "strange bug", "unusual", "one-off",
        "special case", "corner case", "rare scenario",
        "no documentation", "undocumented", "black box",
        "reverse engineering", "figure it out", "puzzling",
        "magic", "voodoo", "haunted code", "ghost in the machine"
    ],
    prompt_template="""Apply Self-Discover reasoning:

TASK: {query}
CONTEXT: {context}

1. SELECT: Which patterns apply? (decomposition, analogy, abstraction, constraints)
2. ADAPT: Customize patterns for this specific task
3. IMPLEMENT: Apply your customized approach
4. VERIFY: Check completeness"""
))


register(FrameworkSpec(
    name="buffer_of_thoughts",
    category=FrameworkCategory.STRATEGY,
    description="Build context in a thought buffer",
    best_for=["multi-part problems", "complex context"],
    vibes=[
        "I've done this before", "standard pattern", "common task",
        "typical", "usual approach", "boilerplate", "routine",
        "CRUD", "login flow", "auth", "pagination",
        "search filter", "form validation", "file upload",
        "email sending", "notifications", "logging",
        "error handling", "retry logic", "rate limiting",
        "design pattern", "singleton", "factory", "observer"
    ],
    prompt_template="""Apply Buffer of Thoughts:

TASK: {query}
CONTEXT: {context}

Build your thought buffer:
- INIT: Key facts and constraints
- ADD: Problem analysis
- ADD: Possible approaches
- ADD: Decision and reasoning
- OUTPUT: Synthesize final solution"""
))


register(FrameworkSpec(
    name="coala",
    category=FrameworkCategory.STRATEGY,
    description="Cognitive architecture for agents",
    best_for=["autonomous tasks", "agent behavior"],
    vibes=[
        "lots of files", "whole codebase", "across multiple files",
        "context from", "remember earlier", "stateful", "keep track",
        "monorepo", "large codebase", "enterprise", "legacy system",
        "multiple services", "cross-cutting", "shared code",
        "dependencies between", "imports from", "circular dependency",
        "file structure", "project organization", "modules",
        "global search", "find usages", "where is this defined"
    ],
    prompt_template="""Apply COALA cognitive architecture:

TASK: {query}
CONTEXT: {context}

1. PERCEPTION: Current state and resources
2. MEMORY: Relevant knowledge and patterns
3. REASONING: Analyze and plan
4. ACTION: Execute plan
5. LEARNING: What worked? What to improve?"""
))


# =============================================================================
# SEARCH FRAMEWORKS
# =============================================================================

register(FrameworkSpec(
    name="mcts_rstar",
    category=FrameworkCategory.SEARCH,
    description="Monte Carlo Tree Search exploration for code",
    best_for=["complex bugs", "multi-step optimization", "thorough search"],
    vibes=[
        "really hard bug", "been stuck for hours", "complex issue",
        "multi-step problem", "deep issue", "intricate bug",
        "need to explore options", "thorough search",
        "exhaustive", "all possibilities", "brute force",
        "search space", "combinatorial", "permutations",
        "game tree", "decision tree", "branch and bound",
        "stuck for days", "impossible bug", "nightmare",
        "burning out", "hitting a wall", "desperate"
    ],
    prompt_template="""Apply rStar-Code MCTS reasoning:

TASK: {query}
CONTEXT: {context}

1. SELECT: Most promising focus area using UCT scoring
2. EXPAND: Generate 2-3 possible modifications
3. SIMULATE: Trace consequences of each path
4. EVALUATE: Score each path (0-1)
5. BACKPROPAGATE: Update parent scores
6. ITERATE: Repeat until confidence threshold or max depth"""
))


register(FrameworkSpec(
    name="tree_of_thoughts",
    category=FrameworkCategory.SEARCH,
    description="Explore multiple paths, pick best",
    best_for=["design decisions", "multiple valid approaches"],
    vibes=[
        "make it faster", "optimize this", "too slow", "performance sucks",
        "better algorithm", "more efficient", "reduce complexity",
        "speed this up", "it's laggy", "bottleneck", "O(n^2)",
        "time complexity", "space complexity", "memory leak",
        "CPU intensive", "optimize query", "slow query", "N+1",
        "caching", "memoization", "lazy loading", "pagination",
        "batch processing", "async", "parallel", "concurrent",
        "profiling", "benchmark", "perf issues", "latency",
        "indexing", "database tuning", "load time", "render performance"
    ],
    prompt_template="""Apply Tree of Thoughts:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create 3 distinct approaches with pros/cons
2. EVALUATE: Score each (feasibility, effectiveness, simplicity)
3. EXPAND: Develop the best approach fully
4. SYNTHESIZE: Final solution with reasoning"""
))


register(FrameworkSpec(
    name="graph_of_thoughts",
    category=FrameworkCategory.SEARCH,
    description="Non-linear reasoning with idea graphs",
    best_for=["complex dependencies", "interconnected problems"],
    vibes=[
        "clean this up", "this code is ugly", "make it not suck",
        "untangle this mess", "spaghetti code", "needs refactoring",
        "reorganize", "restructure", "make it cleaner", "simplify this",
        "code smell", "tech debt", "legacy code", "maintainable",
        "readable", "DRY this up", "extract method", "decompose",
        "too complex", "cyclomatic complexity", "nested hell",
        "callback hell", "pyramid of doom", "deeply nested",
        "hard to follow", "confusing code", "wtf per minute",
        "kill it with fire", "rewrite this junk", "modernize",
        "from class to functional", "hooks refactor", "migration"
    ],
    prompt_template="""Apply Graph of Thoughts:

TASK: {query}
CONTEXT: {context}

1. NODES: Identify key concepts/components
2. EDGES: Map relationships between them
3. TRAVERSE: Find the solution path through the graph
4. SYNTHESIZE: Combine insights into solution"""
))


# =============================================================================
# ITERATIVE FRAMEWORKS
# =============================================================================

register(FrameworkSpec(
    name="active_inference",
    category=FrameworkCategory.ITERATIVE,
    description="Hypothesis testing loop",
    best_for=["debugging", "investigation", "root cause analysis"],
    vibes=[
        "why is this broken", "wtf is wrong", "this doesn't work",
        "find the bug", "debug this", "what's causing this",
        "figure out why", "track down", "root cause", "investigate",
        "something's off", "it's acting weird", "unexpected behavior",
        "why does this", "what the hell", "broken af", "not working",
        "keeps crashing", "throwing errors", "fails randomly",
        "intermittent bug", "heisenbug", "works on my machine",
        "production bug", "regression", "stopped working",
        "used to work", "suddenly broke", "after update",
        "null pointer", "undefined", "NaN", "empty result",
        "segfault", "memory corruption", "race condition",
        "deadlock", "timeout", "connection refused", "500 error",
        "stack overflow", "infinite loop", "recursion error"
    ],
    prompt_template="""Apply Active Inference:

TASK: {query}
CONTEXT: {context}

1. OBSERVE: Current state, form hypotheses
2. PREDICT: What should we expect if hypothesis is true?
3. TEST: Gather evidence, update beliefs
4. ACT: Implement fix based on best hypothesis
5. VERIFY: Confirm the fix worked"""
))


register(FrameworkSpec(
    name="multi_agent_debate",
    category=FrameworkCategory.ITERATIVE,
    description="Multiple perspectives debate",
    best_for=["design decisions", "trade-off analysis"],
    vibes=[
        "should I use A or B", "trade-offs", "pros and cons",
        "which approach", "compare options", "decision", "evaluate",
        "weigh options", "what would you recommend", "best choice",
        "React or Vue", "SQL or NoSQL", "REST or GraphQL",
        "which framework", "which library", "which database",
        "monorepo or multirepo", "serverless or containers",
        "build vs buy", "roll own or use existing", "opinions",
        "debate", "argument", "settle this", "tie breaker"
    ],
    prompt_template="""Apply Multi-Agent Debate:

TASK: {query}
CONTEXT: {context}

Argue from these perspectives:
- PRAGMATIST: What's the simplest working solution?
- ARCHITECT: What's most maintainable/scalable?
- SECURITY: What are the risks?
- PERFORMANCE: What's most efficient?

DEBATE the trade-offs, then SYNTHESIZE a balanced solution."""
))


register(FrameworkSpec(
    name="adaptive_injection",
    category=FrameworkCategory.ITERATIVE,
    description="Inject strategies as needed",
    best_for=["evolving understanding", "adaptive problem solving"],
    vibes=[
        "just figure it out", "do your thing", "whatever works",
        "adapt", "flex", "go with the flow", "surprise me",
        "your call", "you decide", "best judgment",
        "improvise", "wing it", "play it by ear",
        "dynamic approach", "flexible solution",
        "smart select", "auto mode", "magic mode"
    ],
    prompt_template="""Apply Adaptive Injection:

TASK: {query}
CONTEXT: {context}

As you work, inject strategies when needed:
- If stuck -> step back and abstract
- If complex -> decompose into parts
- If uncertain -> explore alternatives
- If risky -> add verification steps

Continue until complete."""
))
