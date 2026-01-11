"""
Single Source of Truth for Framework Definitions.

All framework metadata in ONE place. Other files import from here.
This eliminates the 4-location sync requirement.

Previous sync locations (now deprecated - import from here instead):
1. FRAMEWORK_NODES in app/graph.py
2. FRAMEWORKS dict in app/core/routing/framework_registry.py
3. VIBE_DICTIONARY in app/core/vibe_dictionary.py
4. get_framework_info() in app/core/routing/framework_registry.py
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any


class FrameworkCategory(Enum):
    """Categories of thinking frameworks."""

    STRATEGY = "strategy"
    SEARCH = "search"
    ITERATIVE = "iterative"
    CODE = "code"
    CONTEXT = "context"
    FAST = "fast"
    VERIFICATION = "verification"
    AGENT = "agent"
    RAG = "rag"


@dataclass
class FrameworkDefinition:
    """Complete definition of a reasoning framework.

    This is the single source of truth for all framework metadata.
    All other locations should import from this registry.
    """

    name: str  # e.g., "active_inference"
    display_name: str  # e.g., "Active Inference"
    category: FrameworkCategory
    description: str  # Short description for router
    best_for: list[str]  # Use cases this framework excels at
    vibes: list[str]  # Pattern matching phrases for vibe-based routing
    steps: list[str] = field(default_factory=list)  # Framework reasoning steps
    use_case: str = ""  # One-line use case description
    node_function: str | None = None  # Import path to node function (None = use generator)
    complexity: str = "medium"  # low, medium, high
    task_type: str = "unknown"  # Task type for routing heuristics
    example_type: str = ""  # Type of examples to search: "debugging", "reasoning", "code", ""
    prompt_template: str = ""  # Optional custom prompt template (overrides generator)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.display_name,
            "category": self.category.value,
            "description": self.description,
            "best_for": self.best_for,
            "vibes": self.vibes,
            "complexity": self.complexity,
            "task_type": self.task_type,
        }


# Import centralized error class instead of defining duplicate
from app.core.errors import FrameworkNotFoundError

# =============================================================================
# FRAMEWORK DEFINITIONS - Single Source of Truth for all 62 frameworks
# =============================================================================

FRAMEWORKS: dict[str, FrameworkDefinition] = {}


def register(framework: FrameworkDefinition) -> FrameworkDefinition:
    """Register a framework in the global registry."""
    FRAMEWORKS[framework.name] = framework
    return framework


# =============================================================================
# STRATEGY FRAMEWORKS (7 frameworks)
# =============================================================================

register(
    FrameworkDefinition(
        name="reason_flux",
        display_name="ReasonFlux",
        category=FrameworkCategory.STRATEGY,
        description="Hierarchical planning: template -> expand -> refine. Best for architecture and system design.",
        best_for=["architecture", "system design", "complex planning", "greenfield projects"],
        vibes=[
            "design a system",
            "architect this",
            "how should I structure",
            "plan this out",
            "big picture",
            "system design",
            "make it scalable",
            "build from scratch",
            "greenfield",
            "new project",
            "overall approach",
            "microservices",
            "monolith",
            "architecture",
            "high level design",
            "design patterns",
            "SOLID",
            "clean architecture",
            "hexagonal",
            "domain driven",
            "DDD",
            "event driven",
            "CQRS",
            "event sourcing",
            "API design",
            "schema design",
            "data model",
            "ERD",
            "infrastructure",
            "deployment",
            "CI/CD pipeline",
            "aws architecture",
            "cloud native",
            "serverless",
            "kubernetes",
            "folder structure",
            "project layout",
            "directory structure",
        ],
        steps=[
            "TEMPLATE: Create high-level structure/skeleton",
            "EXPAND: Fill in each section with details",
            "REFINE: Review, identify gaps, iterate",
            "VALIDATE: Check against requirements and constraints",
        ],
        use_case="Architecture design, system planning, large changes",
        complexity="high",
        task_type="architecture",
    )
)

register(
    FrameworkDefinition(
        name="self_discover",
        display_name="Self-Discover",
        category=FrameworkCategory.STRATEGY,
        description="Compose custom reasoning from atomic modules. Best for novel/unclear problems.",
        best_for=["novel problems", "exploration", "unknown domains", "creative solutions"],
        vibes=[
            "I have no idea",
            "weird problem",
            "never seen this",
            "creative solution",
            "think outside box",
            "novel approach",
            "unconventional",
            "unique situation",
            "edge case",
            "bizarre",
            "strange bug",
            "unusual",
            "one-off",
            "special case",
            "corner case",
            "rare scenario",
            "no documentation",
            "undocumented",
            "black box",
            "reverse engineering",
            "figure it out",
            "puzzling",
            "magic",
            "voodoo",
            "haunted code",
            "ghost in the machine",
        ],
        steps=[
            "SELECT: Choose relevant reasoning modules (critical thinking, creative brainstorming, etc.)",
            "ADAPT: Customize modules to this specific task",
            "COMPOSE: Build a reasoning structure from selected modules",
            "EXECUTE: Apply the composed structure to solve the problem",
        ],
        use_case="Novel problems, unclear requirements, exploration",
        example_type="reasoning",
        complexity="high",
        task_type="exploration",
    )
)

register(
    FrameworkDefinition(
        name="buffer_of_thoughts",
        display_name="Buffer of Thoughts",
        category=FrameworkCategory.STRATEGY,
        description="Retrieve proven thought templates. Best for repetitive known patterns.",
        best_for=["repetitive tasks", "known patterns", "standard implementations"],
        vibes=[
            "I've done this before",
            "standard pattern",
            "common task",
            "typical",
            "usual approach",
            "boilerplate",
            "routine",
            "CRUD",
            "login flow",
            "auth",
            "pagination",
            "search filter",
            "form validation",
            "file upload",
            "email sending",
            "notifications",
            "logging",
            "error handling",
            "retry logic",
            "rate limiting",
            "design pattern",
            "singleton",
            "factory",
            "observer",
        ],
        steps=[
            "RETRIEVE: Search memory for similar solved problems",
            "MATCH: Find the best-matching thought template",
            "ADAPT: Modify template for current context",
            "APPLY: Execute the adapted template",
        ],
        use_case="Repetitive tasks, standard patterns, boilerplate",
        complexity="low",
        task_type="docs",
    )
)

register(
    FrameworkDefinition(
        name="coala",
        display_name="CoALA",
        category=FrameworkCategory.STRATEGY,
        description="Cognitive architecture with working + episodic memory. Best for long-context multi-file tasks.",
        best_for=["long context", "multi-file tasks", "stateful operations", "large codebases"],
        vibes=[
            "lots of files",
            "whole codebase",
            "across multiple files",
            "context from",
            "remember earlier",
            "stateful",
            "keep track",
            "monorepo",
            "large codebase",
            "enterprise",
            "legacy system",
            "multiple services",
            "cross-cutting",
            "shared code",
            "dependencies between",
            "imports from",
            "circular dependency",
            "file structure",
            "project organization",
            "modules",
            "global search",
            "find usages",
            "where is this defined",
        ],
        steps=[
            "GATHER: Collect relevant context from multiple files",
            "ORGANIZE: Build working memory with key relationships",
            "REASON: Use episodic memory for similar past solutions",
            "SYNTHESIZE: Combine insights across files into solution",
        ],
        use_case="Long context, multi-file tasks, large codebases",
        complexity="high",
        task_type="context",
    )
)

register(
    FrameworkDefinition(
        name="least_to_most",
        display_name="Least-to-Most Decomposition",
        category=FrameworkCategory.STRATEGY,
        description="Bottom-up atomic function decomposition. Best for complex systems and refactoring monoliths.",
        best_for=["complex systems", "monolith refactoring", "hierarchical problems"],
        vibes=[
            "atomic functions",
            "dependency graph",
            "bottom up",
            "layered",
            "base functions first",
            "decompose completely",
            "building blocks",
            "hierarchical build",
            "least dependent first",
            "utils first",
            "helpers",
            "primitives",
            "foundation",
            "core functions",
            "base layer",
            "component library",
            "atoms",
            "molecules",
            "organisms",
        ],
        steps=[
            "IDENTIFY: Find the leaf-level atomic subproblems",
            "ORDER: Sort by dependencies (least dependent first)",
            "SOLVE: Implement each atomic piece bottom-up",
            "COMPOSE: Build higher-level components from solved pieces",
        ],
        use_case="Complex systems, monolith refactoring, hierarchical problems",
        complexity="high",
        task_type="architecture",
    )
)

register(
    FrameworkDefinition(
        name="comparative_arch",
        display_name="Comparative Architecture",
        category=FrameworkCategory.STRATEGY,
        description="Multiple solution approaches (readability/memory/speed). Best for optimization and architecture decisions.",
        best_for=["optimization", "architecture decisions", "trade-off analysis"],
        vibes=[
            "compare approaches",
            "readability vs performance",
            "trade-offs",
            "multiple solutions",
            "which is faster",
            "optimize for",
            "different versions",
            "performance vs memory",
            "three approaches",
            "show me options",
            "alternatives",
            "variations",
            "different ways",
            "multiple implementations",
            "compare",
            "benchmark results",
            "A/B test",
            "matrix",
        ],
        steps=[
            "APPROACH A: Design for readability/maintainability",
            "APPROACH B: Design for performance/speed",
            "APPROACH C: Design for memory efficiency",
            "COMPARE: Analyze trade-offs and recommend best fit",
        ],
        use_case="Trade-off analysis, architecture decisions, optimization",
        complexity="high",
        task_type="architecture",
    )
)

register(
    FrameworkDefinition(
        name="plan_and_solve",
        display_name="Plan-and-Solve",
        category=FrameworkCategory.STRATEGY,
        description="Explicit planning before execution. Best for complex features and avoiding rushed implementations.",
        best_for=["complex features", "methodical development", "planning"],
        vibes=[
            "plan first",
            "think before coding",
            "explicit plan",
            "strategy",
            "outline approach",
            "plan then execute",
            "methodical",
            "step by step plan",
            "planning phase",
            "roadmap",
            "game plan",
            "action plan",
            "checklist",
            "todo list",
            "phases",
            "milestones",
            "technical spec",
            "design document",
            "RFC",
        ],
        steps=[
            "UNDERSTAND: Deeply analyze the problem requirements",
            "PLAN: Create explicit step-by-step implementation plan",
            "EXECUTE: Implement according to the plan",
            "VERIFY: Check that implementation matches plan",
        ],
        use_case="Complex features, methodical development, planning",
        complexity="medium",
        task_type="planning",
    )
)

# =============================================================================
# SEARCH FRAMEWORKS (4 frameworks)
# =============================================================================

register(
    FrameworkDefinition(
        name="mcts_rstar",
        display_name="rStar-Code MCTS",
        category=FrameworkCategory.SEARCH,
        description="Monte Carlo Tree Search for code. Best for complex multi-step bugs and optimization.",
        best_for=["complex bugs", "optimization", "multi-step problems"],
        vibes=[
            "really hard bug",
            "been stuck for hours",
            "complex issue",
            "multi-step problem",
            "deep issue",
            "intricate bug",
            "need to explore options",
            "thorough search",
            "exhaustive",
            "all possibilities",
            "brute force",
            "search space",
            "combinatorial",
            "permutations",
            "game tree",
            "decision tree",
            "branch and bound",
            "stuck for days",
            "impossible bug",
            "nightmare",
            "burning out",
            "hitting a wall",
            "desperate",
        ],
        steps=[
            "SELECT: Pick promising node to explore (UCB1)",
            "EXPAND: Generate candidate solutions/hypotheses",
            "SIMULATE: Evaluate each candidate's potential",
            "BACKPROPAGATE: Update scores based on results",
            "REPEAT: Continue until solution found or budget exhausted",
        ],
        use_case="Complex bugs, exhaustive search, hard problems",
        example_type="debugging",
        complexity="high",
        task_type="debug",
    )
)

register(
    FrameworkDefinition(
        name="tree_of_thoughts",
        display_name="Tree of Thoughts",
        category=FrameworkCategory.SEARCH,
        description="BFS/DFS exploration of solutions. Best for algorithms and optimization problems.",
        best_for=["algorithms", "optimization", "multiple valid approaches"],
        vibes=[
            "make it faster",
            "optimize this",
            "too slow",
            "performance sucks",
            "better algorithm",
            "more efficient",
            "reduce complexity",
            "speed this up",
            "it's laggy",
            "bottleneck",
            "O(n^2)",
            "time complexity",
            "space complexity",
            "memory leak",
            "CPU intensive",
            "optimize query",
            "slow query",
            "N+1",
            "caching",
            "memoization",
            "lazy loading",
            "pagination",
            "batch processing",
            "async",
            "parallel",
            "concurrent",
            "profiling",
            "benchmark",
            "perf issues",
            "latency",
            "indexing",
            "database tuning",
            "load time",
            "render performance",
        ],
        steps=[
            "GENERATE: Create initial thought branches (2-3 approaches)",
            "EVALUATE: Score each branch's promise",
            "EXPAND: Develop the most promising branches further",
            "PRUNE: Abandon low-scoring branches",
            "CONVERGE: Select best solution path",
        ],
        use_case="Algorithms, optimization, problem solving",
        example_type="reasoning",
        complexity="medium",
        task_type="algorithm",
    )
)

register(
    FrameworkDefinition(
        name="graph_of_thoughts",
        display_name="Graph of Thoughts",
        category=FrameworkCategory.SEARCH,
        description="Non-linear thinking with merge/aggregate. Best for refactoring spaghetti code and restructuring.",
        best_for=["refactoring", "code restructuring", "complex dependencies"],
        vibes=[
            "clean this up",
            "this code is ugly",
            "make it not suck",
            "untangle this mess",
            "spaghetti code",
            "needs refactoring",
            "reorganize",
            "restructure",
            "make it cleaner",
            "simplify this",
            "code smell",
            "tech debt",
            "legacy code",
            "maintainable",
            "readable",
            "DRY this up",
            "extract method",
            "decompose",
            "too complex",
            "cyclomatic complexity",
            "nested hell",
            "callback hell",
            "pyramid of doom",
            "deeply nested",
            "hard to follow",
            "confusing code",
            "wtf per minute",
            "kill it with fire",
            "rewrite this junk",
            "modernize",
            "from class to functional",
            "hooks refactor",
            "migration",
        ],
        steps=[
            "MAP: Identify all interconnected pieces",
            "BRANCH: Explore multiple refactoring approaches",
            "MERGE: Combine insights from different branches",
            "AGGREGATE: Synthesize best elements into solution",
        ],
        use_case="Refactoring, untangling complexity, restructuring",
        complexity="high",
        task_type="refactor",
    )
)

register(
    FrameworkDefinition(
        name="everything_of_thought",
        display_name="Everything of Thought",
        category=FrameworkCategory.SEARCH,
        description="MCTS + fast generation + verification. Best for large complex changes.",
        best_for=["complex refactoring", "migration", "major overhaul"],
        vibes=[
            "major rewrite",
            "big migration",
            "overhaul",
            "massive change",
            "complete redesign",
            "from scratch",
            "total refactor",
            "modernize",
            "upgrade everything",
            "v2",
            "next version",
            "breaking changes",
            "deprecate",
            "sunset",
            "EOL",
            "migration path",
            "upgrade path",
            "replatform",
            "technical transformation",
            "rebuild",
            "redo everything",
            "strangler fig",
            "rip and replace",
            "lift and shift",
        ],
        steps=[
            "SEARCH: Use MCTS to explore solution space",
            "GENERATE: Fast-generate candidate implementations",
            "VERIFY: Test and validate each candidate",
            "ITERATE: Refine based on verification results",
            "DELIVER: Present best verified solution",
        ],
        use_case="Major rewrites, migrations, large overhauls",
        complexity="high",
        task_type="refactor",
    )
)

# =============================================================================
# ITERATIVE FRAMEWORKS (8 frameworks)
# =============================================================================

register(
    FrameworkDefinition(
        name="active_inference",
        display_name="Active Inference",
        category=FrameworkCategory.ITERATIVE,
        description="Debugging loop: hypothesis -> predict -> compare -> update. Best for nasty bugs and root cause analysis.",
        best_for=["debugging", "error analysis", "root cause investigation", "hypothesis testing"],
        vibes=[
            "why is this broken",
            "wtf is wrong",
            "this doesn't work",
            "find the bug",
            "debug this",
            "what's causing this",
            "figure out why",
            "track down",
            "root cause",
            "investigate",
            "something's off",
            "it's acting weird",
            "unexpected behavior",
            "why does this",
            "what the hell",
            "broken af",
            "not working",
            "keeps crashing",
            "throwing errors",
            "fails randomly",
            "intermittent bug",
            "heisenbug",
            "works on my machine",
            "production bug",
            "regression",
            "stopped working",
            "used to work",
            "suddenly broke",
            "after update",
            "null pointer",
            "undefined",
            "NaN",
            "empty result",
            "segfault",
            "memory corruption",
            "race condition",
            "deadlock",
            "timeout",
            "connection refused",
            "500 error",
            "stack overflow",
            "infinite loop",
            "recursion error",
        ],
        steps=[
            "HYPOTHESIS: Form a hypothesis about the bug's cause",
            "PREDICT: What would we expect to see if hypothesis is true?",
            "COMPARE: Check actual behavior against prediction",
            "UPDATE: Revise hypothesis based on evidence",
            "REPEAT: Continue until root cause identified",
        ],
        use_case="Debugging, error analysis, root cause investigation",
        example_type="debugging",
        complexity="medium",
        task_type="debug",
    )
)

register(
    FrameworkDefinition(
        name="multi_agent_debate",
        display_name="Multi-Agent Debate",
        category=FrameworkCategory.ITERATIVE,
        description="Proponent vs Critic argumentation. Best for design decisions and trade-offs.",
        best_for=["design decisions", "trade-offs", "weighing options"],
        vibes=[
            "should I use A or B",
            "trade-offs",
            "pros and cons",
            "which approach",
            "compare options",
            "decision",
            "evaluate",
            "weigh options",
            "what would you recommend",
            "best choice",
            "React or Vue",
            "SQL or NoSQL",
            "REST or GraphQL",
            "which framework",
            "which library",
            "which database",
            "monorepo or multirepo",
            "serverless or containers",
            "build vs buy",
            "roll own or use existing",
            "opinions",
            "debate",
            "argument",
            "settle this",
            "tie breaker",
        ],
        steps=[
            "PROPONENT: Make the case for Option A",
            "CRITIC: Challenge with counterarguments",
            "PROPONENT: Defend and address concerns",
            "CRITIC: Make the case for Option B",
            "SYNTHESIZE: Weigh arguments and recommend",
        ],
        use_case="Design decisions, trade-offs, weighing options",
        complexity="high",
        task_type="architecture",
    )
)

register(
    FrameworkDefinition(
        name="adaptive_injection",
        display_name="Adaptive Injection",
        category=FrameworkCategory.ITERATIVE,
        description="Dynamic thinking depth based on complexity. Best for mixed-complexity tasks.",
        best_for=["variable complexity", "adaptive problem solving"],
        vibes=[
            "just figure it out",
            "do your thing",
            "whatever works",
            "adapt",
            "flex",
            "go with the flow",
            "surprise me",
            "your call",
            "you decide",
            "best judgment",
            "improvise",
            "wing it",
            "play it by ear",
            "dynamic approach",
            "flexible solution",
            "smart select",
            "auto mode",
            "magic mode",
        ],
        steps=[
            "ASSESS: Evaluate problem complexity (low/medium/high)",
            "CALIBRATE: Select appropriate thinking depth",
            "EXECUTE: Apply calibrated reasoning",
            "ADJUST: Increase depth if initial attempt insufficient",
        ],
        use_case="Variable complexity, adaptive problem solving",
        complexity="medium",
        task_type="adaptive",
    )
)

register(
    FrameworkDefinition(
        name="re2",
        display_name="Re-Reading (RE2)",
        category=FrameworkCategory.ITERATIVE,
        description="Two-pass: goals -> constraints. Best for complex specs and requirements.",
        best_for=["complex specs", "requirements", "detailed analysis"],
        vibes=[
            "requirements",
            "spec",
            "constraints",
            "must have",
            "needs to",
            "requirements doc",
            "acceptance criteria",
            "user story",
            "ticket",
            "JIRA",
            "specification",
            "functional requirements",
            "non-functional",
            "NFR",
            "SLA",
            "compliance",
            "regulatory",
            "legal requirements",
            "business rules",
            "validation rules",
            "invariants",
            "as a user",
            "given when then",
            "checklist",
        ],
        steps=[
            "PASS 1 - GOALS: Extract what needs to be achieved",
            "PASS 2 - CONSTRAINTS: Identify limitations and requirements",
            "SYNTHESIZE: Combine goals with constraints",
            "VALIDATE: Ensure solution meets all requirements",
        ],
        use_case="Complex specs, requirements, detailed analysis",
        complexity="medium",
        task_type="requirements",
    )
)

register(
    FrameworkDefinition(
        name="rubber_duck",
        display_name="Rubber Duck Debugging",
        category=FrameworkCategory.ITERATIVE,
        description="Socratic questioning for self-discovery. Best for architectural bottlenecks and blind spots.",
        best_for=["architectural issues", "blind spots", "thinking through problems"],
        vibes=[
            "explain to me",
            "walk me through",
            "ask me questions",
            "guide me",
            "help me think",
            "rubber duck",
            "Socratic method",
            "lead me to answer",
            "questioning approach",
            "talk it through",
            "think aloud",
            "verbalize",
            "explain my thinking",
            "work through it",
            "reasoning",
            "pair program",
            "code buddy",
            "sounding board",
        ],
        steps=[
            "EXPLAIN: Describe what the code is supposed to do",
            "QUESTION: What assumptions am I making?",
            "TRACE: Walk through execution step by step",
            "DISCOVER: Where does expectation diverge from reality?",
        ],
        use_case="Thinking through problems, finding blind spots",
        complexity="medium",
        task_type="debug",
    )
)

register(
    FrameworkDefinition(
        name="react",
        display_name="ReAct",
        category=FrameworkCategory.ITERATIVE,
        description="Interleaved reasoning and acting with tools. Best for multi-step tasks requiring tool use.",
        best_for=["multi-step tasks", "tool use", "action sequences"],
        vibes=[
            "use tools",
            "multi-step",
            "action reasoning",
            "tool use",
            "step and observe",
            "interact with",
            "reasoning acting",
            "ReAct pattern",
            "observe results",
            "API calls",
            "external tools",
            "shell commands",
            "file operations",
            "database queries",
            "web requests",
            "scrape",
            "curl",
            "wget",
            "grep",
            "find",
        ],
        steps=[
            "THOUGHT: Reason about what to do next",
            "ACTION: Execute a tool or command",
            "OBSERVATION: Observe and interpret the result",
            "(Repeat THOUGHT-ACTION-OBSERVATION until goal achieved)",
            "ANSWER: Synthesize final solution",
        ],
        use_case="Tool use, API exploration, multi-step tasks",
        complexity="high",
        task_type="agent",
    )
)

register(
    FrameworkDefinition(
        name="reflexion",
        display_name="Reflexion",
        category=FrameworkCategory.ITERATIVE,
        description="Self-evaluation with memory-based learning. Best for learning from failed attempts.",
        best_for=["learning from failures", "iterative improvement", "retrying"],
        vibes=[
            "learn from mistakes",
            "retry",
            "failed attempt",
            "try again",
            "reflect on",
            "what went wrong",
            "iterative learning",
            "self-evaluation",
            "memory-based",
            "previous attempt",
            "last time",
            "improve on",
            "lessons learned",
            "retrospective",
            "post-mortem",
            "root cause analysis",
            "RCA",
            "incident report",
        ],
        steps=[
            "ATTEMPT: Try to solve the problem",
            "EVALUATE: Assess if solution is correct",
            "REFLECT: If wrong, analyze what went wrong",
            "REMEMBER: Store reflection in memory",
            "RETRY: Use reflection to inform next attempt",
        ],
        use_case="Learning from failures, iterative improvement",
        complexity="high",
        task_type="iterative",
    )
)

register(
    FrameworkDefinition(
        name="self_refine",
        display_name="Self-Refine",
        category=FrameworkCategory.ITERATIVE,
        description="Iterative self-critique and improvement. Best for code quality and documentation.",
        best_for=["code quality", "documentation", "iterative improvement"],
        vibes=[
            "improve quality",
            "polish",
            "refine",
            "make it better",
            "iterative improvement",
            "critique and improve",
            "self-critique",
            "refinement loop",
            "quality pass",
            "clean up",
            "tighten up",
            "optimize",
            "beautify",
            "format",
            "lint",
            "prettier",
            "black",
            "isort",
            "ruff",
            "flake8",
            "pylint",
        ],
        steps=[
            "GENERATE: Create initial solution",
            "CRITIQUE: Identify weaknesses and issues",
            "REFINE: Address each critique",
            "REPEAT: Continue until quality threshold met",
        ],
        use_case="Code quality, documentation, polishing",
        complexity="medium",
        task_type="quality",
    )
)

# =============================================================================
# CODE FRAMEWORKS (15 frameworks)
# =============================================================================

register(
    FrameworkDefinition(
        name="program_of_thoughts",
        display_name="Program of Thoughts",
        category=FrameworkCategory.CODE,
        description="Generate executable Python to compute. Best for math, data processing, testing.",
        best_for=["math", "data processing", "computational problems"],
        vibes=[
            "calculate",
            "compute",
            "do the math",
            "run the numbers",
            "data processing",
            "transform data",
            "crunch",
            "analyze data",
            "formula",
            "equation",
            "statistics",
            "aggregate",
            "sum",
            "average",
            "median",
            "percentile",
            "distribution",
            "pandas",
            "numpy",
            "dataframe",
            "CSV",
            "JSON transform",
            "ETL",
            "data pipeline",
            "data munging",
            "wrangling",
            "fibonacci",
            "factorial",
            "prime numbers",
            "matrix multiplication",
        ],
        steps=[
            "UNDERSTAND: Parse the computational problem",
            "PLAN: Design the algorithm/approach",
            "CODE: Generate executable Python code",
            "EXECUTE: Run in sandbox and verify result",
        ],
        use_case="Math, data processing, computational problems",
        node_function="app.nodes.code.pot",  # Special node with sandbox
        complexity="medium",
        task_type="compute",
    )
)

register(
    FrameworkDefinition(
        name="chain_of_verification",
        display_name="Chain of Verification",
        category=FrameworkCategory.CODE,
        description="Draft -> verify -> patch cycle. Best for security review and code validation.",
        best_for=["security", "code review", "validation"],
        vibes=[
            "is this secure",
            "security check",
            "audit this",
            "vulnerabilities",
            "could this be hacked",
            "pen test",
            "code review",
            "sanity check",
            "validate this",
            "double check",
            "verify",
            "confirm",
            "make sure",
            "correct?",
            "looks right?",
            "review my code",
            "spot check",
            "find issues",
            "catch bugs",
            "QA",
            "quality check",
            "before merging",
            "PR review",
            "pull request",
            "lgtm",
            "nitpick",
            "blocker",
            "critical issue",
        ],
        steps=[
            "DRAFT: Generate initial code/solution",
            "VERIFY: Check each claim/assumption against facts",
            "IDENTIFY: List all verification failures",
            "PATCH: Fix only the verified failures",
        ],
        use_case="Security review, code validation, QA",
        complexity="medium",
        task_type="security",
    )
)

register(
    FrameworkDefinition(
        name="critic",
        display_name="CRITIC",
        category=FrameworkCategory.CODE,
        description="External tool verification. Best for API usage validation and library integration.",
        best_for=["API usage", "library integration", "external validation"],
        vibes=[
            "using this library",
            "api integration",
            "third party",
            "how do I use",
            "sdk",
            "package",
            "library docs",
            "correct usage",
            "best practice for",
            "npm package",
            "pip install",
            "import",
            "dependency",
            "module",
            "REST API",
            "GraphQL",
            "webhook",
            "OAuth",
            "JWT",
            "external service",
            "integration",
            "connect to",
            "API call",
            "fetch",
            "axios",
            "requests",
            "read the docs",
            "documentation says",
            "api reference",
        ],
        steps=[
            "GENERATE: Produce initial code using the library/API",
            "CRITIQUE: Verify against official documentation",
            "TOOL-CHECK: Use external tools to validate (linter, type checker)",
            "REVISE: Fix issues based on external feedback",
        ],
        use_case="API usage, library integration, external validation",
        complexity="medium",
        task_type="api",
    )
)

register(
    FrameworkDefinition(
        name="chain_of_code",
        display_name="Chain-of-Code",
        category=FrameworkCategory.CODE,
        description="Break problems into code blocks for structured thinking. Best for logic puzzles and algorithmic debugging.",
        best_for=["logic puzzles", "algorithmic debugging", "structured thinking"],
        vibes=[
            "code blocks",
            "pseudocode",
            "execution trace",
            "logic puzzle",
            "recursive logic",
            "algorithmic complexity",
            "structured thinking",
            "break into code",
            "code decomposition",
            "step through",
            "trace execution",
            "follow the code",
            "code flow",
            "control flow",
            "data flow",
            "call stack",
            "breakpoints",
            "debugger",
            "line by line",
            "inspect",
            "variable state",
            "heap",
            "stack",
        ],
        steps=[
            "DECOMPOSE: Break problem into discrete code blocks",
            "TRACE: Execute each block mentally, track state",
            "VERIFY: Check intermediate results at each step",
            "INTEGRATE: Combine blocks into final solution",
        ],
        use_case="Logic puzzles, algorithmic debugging, structured thinking",
        complexity="medium",
        task_type="debug",
    )
)

register(
    FrameworkDefinition(
        name="self_debugging",
        display_name="Self-Debugging",
        category=FrameworkCategory.CODE,
        description="Mental execution trace before presenting code. Best for preventing off-by-one and null pointer bugs.",
        best_for=["preventing bugs", "edge case handling", "pre-validation"],
        vibes=[
            "test before showing",
            "mental execution",
            "trace through",
            "prevent bugs",
            "check my work",
            "simulate execution",
            "off by one",
            "edge case check",
            "dry run",
            "desk check",
            "code review myself",
            "self review",
            "before committing",
            "sanity test",
            "smoke test",
            "manual testing",
            "walkthrough",
            "trace",
            "pre-flight",
            "double check logic",
        ],
        steps=[
            "GENERATE: Write the initial code",
            "TRACE: Mentally execute with sample inputs",
            "CHECK EDGES: Test boundary conditions (0, 1, empty, max)",
            "FIX: Correct any issues found during trace",
        ],
        use_case="Preventing bugs, edge case handling, pre-validation",
        complexity="medium",
        task_type="debug",
    )
)

register(
    FrameworkDefinition(
        name="tdd_prompting",
        display_name="TDD Prompting",
        category=FrameworkCategory.CODE,
        description="Write tests first, then implementation. Best for edge case coverage and new features.",
        best_for=["new features", "edge case coverage", "test-driven development"],
        vibes=[
            "test first",
            "write tests",
            "TDD",
            "test driven",
            "unit tests",
            "edge cases",
            "test coverage",
            "red green refactor",
            "tests then code",
            "jest",
            "pytest",
            "mocha",
            "junit",
            "testing framework",
            "mock",
            "stub",
            "spy",
            "fixture",
            "assertion",
            "expect",
            "should",
            "test case",
            "integration test",
            "e2e test",
            "selenium",
            "cypress",
            "playwright",
        ],
        steps=[
            "RED: Write failing tests for the expected behavior",
            "GREEN: Write minimal code to make tests pass",
            "REFACTOR: Clean up code while keeping tests green",
            "REPEAT: Add more tests for edge cases",
        ],
        use_case="Test-driven development, edge case coverage",
        complexity="medium",
        task_type="test",
    )
)

register(
    FrameworkDefinition(
        name="reverse_cot",
        display_name="Reverse Chain-of-Thought",
        category=FrameworkCategory.CODE,
        description="Work backward from buggy output to source. Best for silent bugs with wrong outputs.",
        best_for=["silent bugs", "wrong outputs", "backward debugging"],
        vibes=[
            "wrong output",
            "expected vs actual",
            "why different output",
            "output delta",
            "silent bug",
            "calculation error",
            "backwards debugging",
            "work backward",
            "reverse engineer bug",
            "expected X got Y",
            "off by",
            "incorrect result",
            "wrong answer",
            "bad output",
            "unexpected result",
            "diff",
            "mismatch",
            "discrepancy",
            "log analysis",
            "traceback analysis",
        ],
        steps=[
            "OBSERVE: Document the wrong output precisely",
            "TRACE BACK: What code produced this output?",
            "COMPARE: What should have produced correct output?",
            "IDENTIFY: Find the divergence point",
            "FIX: Correct the root cause",
        ],
        use_case="Silent bugs, wrong outputs, backward debugging",
        example_type="debugging",
        complexity="medium",
        task_type="debug",
    )
)

register(
    FrameworkDefinition(
        name="alphacodium",
        display_name="AlphaCodium",
        category=FrameworkCategory.CODE,
        description="Test-based multi-stage iterative code generation. Best for competitive programming and complex algorithms.",
        best_for=["competitive programming", "complex algorithms", "test-based development"],
        vibes=[
            "competitive programming",
            "code contest",
            "algorithm challenge",
            "iterative code",
            "test-based",
            "multi-stage",
            "code generation",
            "contest problem",
            "leetcode",
            "hackerrank",
            "codeforces",
            "topcoder",
            "advent of code",
            "interview question",
            "coding interview",
            "whiteboard",
            "DSA",
            "data structures and algorithms",
            "dynamic programming",
            "graph theory",
            "greedy algorithm",
        ],
        steps=[
            "ANALYZE: Understand problem, identify edge cases",
            "GENERATE: Create initial solution",
            "TEST: Run against public test cases",
            "ITERATE: Fix failures, handle edge cases",
            "OPTIMIZE: Improve time/space complexity",
        ],
        use_case="Competitive programming, complex algorithms",
        example_type="code",
        complexity="high",
        task_type="competitive",
    )
)

register(
    FrameworkDefinition(
        name="codechain",
        display_name="CodeChain",
        category=FrameworkCategory.CODE,
        description="Chain of self-revisions guided by sub-modules. Best for modular code generation and incremental refinement.",
        best_for=["modular code generation", "incremental refinement", "sub-module development"],
        vibes=[
            "modular code",
            "sub-modules",
            "self-revision",
            "incremental",
            "chain revisions",
            "module by module",
            "component based",
            "build incrementally",
            "refine modules",
            "microservices",
            "packages",
            "libraries",
            "separation of concerns",
            "single responsibility",
            "interface segregation",
            "dependency injection",
        ],
        steps=[
            "DECOMPOSE: Break into independent sub-modules",
            "GENERATE: Create each sub-module",
            "REVISE: Self-critique and improve each module",
            "INTEGRATE: Combine sub-modules into complete solution",
        ],
        use_case="Modular code generation, incremental refinement",
        complexity="high",
        task_type="code_gen",
    )
)

register(
    FrameworkDefinition(
        name="evol_instruct",
        display_name="Evol-Instruct",
        category=FrameworkCategory.CODE,
        description="Evolutionary instruction complexity for code. Best for challenging code problems and constraint-based coding.",
        best_for=["challenging code problems", "constraint-based coding", "evolutionary solutions"],
        vibes=[
            "evolve solution",
            "add constraints",
            "increase complexity",
            "challenging problem",
            "constraint-based",
            "evolutionary",
            "harder version",
            "more constraints",
            "complex requirements",
            "additional requirements",
            "scope creep",
            "feature creep",
            "extend",
            "enhance",
            "augment",
            "expand",
            "version 2",
            "next iteration",
            "advanced features",
        ],
        steps=[
            "BASE: Solve the simple version first",
            "EVOLVE: Add constraints incrementally",
            "ADAPT: Modify solution to handle new constraints",
            "VALIDATE: Ensure all constraints are satisfied",
        ],
        use_case="Challenging problems, constraint-based coding",
        complexity="high",
        task_type="code_gen",
    )
)

register(
    FrameworkDefinition(
        name="llmloop",
        display_name="LLMLoop",
        category=FrameworkCategory.CODE,
        description="Automated iterative feedback loops for code+tests. Best for code quality assurance and production-ready code.",
        best_for=["code quality assurance", "production-ready code", "automated testing"],
        vibes=[
            "feedback loop",
            "iterate until",
            "compile and fix",
            "test loop",
            "automated testing",
            "quality assurance",
            "production ready",
            "lint and fix",
            "keep iterating",
            "CI/CD",
            "build pipeline",
            "automated checks",
            "pre-commit hooks",
            "continuous improvement",
            "nightly build",
            "regression testing",
        ],
        steps=[
            "GENERATE: Create initial code",
            "RUN: Execute tests/linter/compiler",
            "ANALYZE: Parse error messages",
            "FIX: Address each error",
            "REPEAT: Until all checks pass",
        ],
        use_case="Code quality assurance, production-ready code",
        complexity="high",
        task_type="ci_cd",
    )
)

register(
    FrameworkDefinition(
        name="procoder",
        display_name="ProCoder",
        category=FrameworkCategory.CODE,
        description="Compiler-feedback-guided iterative refinement. Best for project-level code generation and API usage.",
        best_for=["project-level code generation", "API usage", "compiler feedback"],
        vibes=[
            "compiler feedback",
            "project level",
            "codebase integration",
            "API usage",
            "large project",
            "integrate with",
            "project context",
            "compiler errors",
            "fix imports",
            "type errors",
            "typescript",
            "mypy",
            "pylint",
            "ESLint errors",
            "build errors",
            "dependency issues",
            "linker error",
            "module not found",
            "circular import",
        ],
        steps=[
            "CONTEXT: Gather project context and dependencies",
            "GENERATE: Create code that fits project structure",
            "COMPILE: Run compiler/type checker",
            "ITERATE: Fix issues based on compiler feedback",
        ],
        use_case="Project-level code generation, compiler feedback",
        complexity="high",
        task_type="code_gen",
    )
)

register(
    FrameworkDefinition(
        name="recode",
        display_name="RECODE",
        category=FrameworkCategory.CODE,
        description="Multi-candidate validation with CFG-based debugging. Best for reliable code generation and high-stakes code.",
        best_for=["reliable code generation", "high-stakes code", "multi-candidate validation"],
        vibes=[
            "multiple candidates",
            "cross validate",
            "CFG debugging",
            "control flow",
            "reliable code",
            "high stakes",
            "validate candidates",
            "majority voting",
            "robust solution",
            "mission critical",
            "production critical",
            "can't fail",
            "financial",
            "healthcare",
            "safety critical",
            "consensus",
            "redundancy",
            "fault tolerant",
            "zero downtime",
            "utility class",
            "nuclear",
        ],
        steps=[
            "GENERATE: Create multiple candidate solutions",
            "ANALYZE: Build control flow graph for each",
            "VALIDATE: Cross-validate candidates against each other",
            "SELECT: Choose most reliable (majority voting)",
        ],
        use_case="High-stakes code, reliable generation",
        complexity="high",
        task_type="code_gen",
    )
)

register(
    FrameworkDefinition(
        name="pal",
        display_name="PAL",
        category=FrameworkCategory.CODE,
        description="Program-Aided Language - code as reasoning substrate. Best for algorithms and numeric logic.",
        best_for=["algorithms", "parsing", "numeric logic", "validation"],
        vibes=[
            "code to reason",
            "compute answer",
            "executable logic",
            "algorithm as code",
            "run to verify",
            "code substrate",
            "program for answer",
            "computation",
            "calculate with code",
            "numeric reasoning",
            "code-based math",
        ],
        steps=[
            "TRANSLATE: Convert problem to code representation",
            "IMPLEMENT: Write executable program",
            "EXECUTE: Run program to get answer",
            "VERIFY: Check result makes sense",
        ],
        use_case="Algorithms, numeric logic, computation",
        complexity="medium",
        task_type="compute",
    )
)

register(
    FrameworkDefinition(
        name="scratchpads",
        display_name="Scratchpads",
        category=FrameworkCategory.CODE,
        description="Structured intermediate reasoning workspace. Best for multi-step fixes.",
        best_for=["multi-step fixes", "multi-constraint reasoning", "state tracking"],
        vibes=[
            "working notes",
            "scratch space",
            "intermediate work",
            "track state",
            "multi-step notes",
            "organized thinking",
            "structured notes",
            "work area",
            "keep track",
            "progressive notes",
            "step tracker",
        ],
        steps=[
            "SETUP: Create structured workspace",
            "TRACK: Record intermediate results at each step",
            "VERIFY: Check each intermediate result",
            "COMPLETE: Synthesize final answer from workspace",
        ],
        use_case="Multi-step fixes, state tracking",
        complexity="low",
        task_type="reasoning",
    )
)

register(
    FrameworkDefinition(
        name="parsel",
        display_name="Parsel",
        category=FrameworkCategory.CODE,
        description="Compositional code generation from natural language specs. Builds dependency graph of functions.",
        best_for=["complex functions", "dependency graphs", "spec-to-code", "modular systems"],
        vibes=[
            "compositional",
            "dependency graph",
            "function specs",
            "decompose into functions",
            "build from specs",
            "spec to code",
            "natural language specs",
            "function dependencies",
            "bottom up build",
            "break into functions",
            "modular from spec",
            "compose functions",
            "spec driven",
            "from requirements",
            "hierarchical functions",
            "dependency order",
            "build order",
            "layered implementation",
        ],
        steps=[
            "PARSE: Extract function specs from natural language",
            "GRAPH: Build dependency graph of functions",
            "ORDER: Topologically sort by dependencies",
            "IMPLEMENT: Generate each function in order",
            "COMPOSE: Combine into complete solution",
        ],
        use_case="Complex functions, spec-to-code, dependency graphs",
        complexity="high",
        task_type="code_gen",
    )
)

register(
    FrameworkDefinition(
        name="docprompting",
        display_name="DocPrompting",
        category=FrameworkCategory.CODE,
        description="Documentation-driven code generation. Retrieves docs and examples to guide code generation.",
        best_for=["API usage", "library integration", "following docs", "correct usage"],
        vibes=[
            "from documentation",
            "follow the docs",
            "docs say",
            "according to documentation",
            "api docs",
            "official docs",
            "documentation example",
            "doc-driven",
            "read the docs",
            "rtfm",
            "manual says",
            "reference docs",
            "usage example",
            "as documented",
            "per documentation",
            "library docs",
            "sdk documentation",
            "api reference",
            "doc examples",
        ],
        steps=[
            "RETRIEVE: Find relevant documentation",
            "EXTRACT: Pull out key patterns and examples",
            "GENERATE: Create code following doc patterns",
            "VERIFY: Check code matches documentation",
        ],
        use_case="API usage, library integration, following docs",
        complexity="medium",
        task_type="code_gen",
    )
)

# =============================================================================
# CONTEXT FRAMEWORKS (6 frameworks)
# =============================================================================

register(
    FrameworkDefinition(
        name="chain_of_note",
        display_name="Chain of Note",
        category=FrameworkCategory.CONTEXT,
        description="Research mode with gap analysis. Best for documentation and learning codebases.",
        best_for=["research", "learning", "documentation", "gap analysis"],
        vibes=[
            "understand this code",
            "explain",
            "what does this do",
            "learn the codebase",
            "document",
            "figure out how",
            "reverse engineer",
            "how does it work",
            "explain like",
            "ELI5",
            "walk through",
            "code walkthrough",
            "documentation",
            "comments",
            "README",
            "wiki",
            "onboarding",
            "new to codebase",
            "ramping up",
            "summary",
            "tldr",
            "synopsis",
            "notes",
        ],
        steps=[
            "READ: Examine the code/documentation thoroughly",
            "NOTE: Record key observations and patterns",
            "GAP: Identify what's unclear or missing",
            "SYNTHESIZE: Build understanding from notes",
        ],
        use_case="Research, learning, documentation, gap analysis",
        complexity="medium",
        task_type="research",
    )
)

register(
    FrameworkDefinition(
        name="step_back",
        display_name="Step-Back Prompting",
        category=FrameworkCategory.CONTEXT,
        description="Abstract first, then implement. Best for performance and complexity analysis.",
        best_for=["performance", "complexity analysis", "first principles"],
        vibes=[
            "big O",
            "complexity analysis",
            "fundamentals",
            "first principles",
            "underlying concept",
            "theory behind",
            "abstract thinking",
            "CS fundamentals",
            "data structures",
            "algorithms",
            "design principles",
            "patterns",
            "anti-patterns",
            "why does this work",
            "how does this work",
            "under the hood",
            "internals",
            "implementation details",
            "deep dive",
            "zoom out",
            "high level view",
            "conceptual",
        ],
        steps=[
            "ABSTRACT: What's the underlying principle/concept?",
            "ANALYZE: How does theory apply to this case?",
            "GROUND: Connect abstract understanding to concrete problem",
            "IMPLEMENT: Apply insights to solution",
        ],
        use_case="First principles, complexity analysis, deep understanding",
        complexity="medium",
        task_type="analysis",
    )
)

register(
    FrameworkDefinition(
        name="analogical",
        display_name="Analogical Prompting",
        category=FrameworkCategory.CONTEXT,
        description="Find analogies to solve problems. Best for creative solutions and pattern recognition.",
        best_for=["creative solutions", "pattern recognition", "analogies"],
        vibes=[
            "like when",
            "similar to",
            "pattern from",
            "reminds me of",
            "same as",
            "analogous",
            "comparable to",
            "like in",
            "inspired by",
            "borrowed from",
            "adapted from",
            "seen this before",
            "familiar pattern",
            "like that other",
            "copy from",
            "based on",
            "reference implementation",
            "metaphor",
            "example",
            "instance",
        ],
        steps=[
            "RECALL: What similar problems have I solved before?",
            "MAP: How does the analogy apply here?",
            "ADAPT: Modify the analogous solution for this context",
            "VERIFY: Ensure adaptation is valid",
        ],
        use_case="Creative solutions, pattern recognition, analogies",
        complexity="medium",
        task_type="creative",
    )
)

register(
    FrameworkDefinition(
        name="red_team",
        display_name="Red-Teaming",
        category=FrameworkCategory.CONTEXT,
        description="Adversarial security analysis (STRIDE, OWASP). Best for security audits and vulnerability scanning.",
        best_for=["security audits", "vulnerability scanning", "adversarial analysis"],
        vibes=[
            "security audit",
            "vulnerabilities",
            "pen test",
            "security review",
            "find exploits",
            "attack vectors",
            "OWASP",
            "SQLi",
            "XSS",
            "security threats",
            "hack this",
            "injection",
            "CSRF",
            "SSRF",
            "RCE",
            "privilege escalation",
            "authentication bypass",
            "broken access control",
            "sensitive data exposure",
            "data breach",
            "leak",
            "encryption",
            "hashing",
            "salt",
        ],
        steps=[
            "THREAT MODEL: Identify assets and attack surface (STRIDE)",
            "ENUMERATE: List potential vulnerabilities (OWASP Top 10)",
            "EXPLOIT: Attempt to exploit each vulnerability",
            "REPORT: Document findings with severity and remediation",
        ],
        use_case="Security audits, vulnerability scanning, adversarial analysis",
        complexity="high",
        task_type="security",
    )
)

register(
    FrameworkDefinition(
        name="state_machine",
        display_name="State-Machine Reasoning",
        category=FrameworkCategory.CONTEXT,
        description="Formal FSM design before coding. Best for UI/UX logic, game dev, and workflows.",
        best_for=["UI logic", "workflow systems", "state management", "game dev"],
        vibes=[
            "state machine",
            "FSM",
            "states and transitions",
            "workflow",
            "state diagram",
            "UI states",
            "game states",
            "state management",
            "transitions",
            "Redux",
            "Zustand",
            "MobX",
            "Vuex",
            "finite automata",
            "statechart",
            "xstate",
            "loading states",
            "error states",
            "success states",
            "hydration",
            "client state",
            "server state",
        ],
        steps=[
            "ENUMERATE STATES: List all possible states",
            "DEFINE TRANSITIONS: What events cause state changes?",
            "VALIDATE: Check for unreachable states, missing transitions",
            "IMPLEMENT: Translate FSM to code",
        ],
        use_case="UI logic, workflow systems, state management",
        complexity="medium",
        task_type="architecture",
    )
)

register(
    FrameworkDefinition(
        name="chain_of_thought",
        display_name="Chain-of-Thought",
        category=FrameworkCategory.CONTEXT,
        description="Basic step-by-step reasoning. Best for complex reasoning and logical deduction.",
        best_for=["complex reasoning", "logical deduction", "step-by-step analysis"],
        vibes=[
            "think step by step",
            "reason through",
            "work through",
            "logical steps",
            "step by step",
            "reasoning chain",
            "think carefully",
            "show your work",
            "explicit reasoning",
            "break it down",
            "one step at a time",
            "systematically",
            "logically",
            "methodically",
            "carefully",
            "first",
            "then",
            "finally",
            "consequently",
        ],
        steps=[
            "UNDERSTAND: Parse and comprehend the problem",
            "REASON: Work through each logical step explicitly",
            "VERIFY: Check each step's validity",
            "CONCLUDE: State the final answer with reasoning chain",
        ],
        use_case="Complex reasoning, logical deduction, step-by-step analysis",
        example_type="reasoning",
        complexity="low",
        task_type="reasoning",
    )
)

# =============================================================================
# FAST FRAMEWORKS (2 frameworks)
# =============================================================================

register(
    FrameworkDefinition(
        name="skeleton_of_thought",
        display_name="Skeleton of Thought",
        category=FrameworkCategory.FAST,
        description="Outline-first with parallel expansion. Best for docs, boilerplate, fast generation.",
        best_for=["documentation", "boilerplate", "fast generation", "scaffolding"],
        vibes=[
            "just generate",
            "scaffold",
            "boilerplate",
            "quick template",
            "stub this out",
            "starter code",
            "basic structure",
            "skeleton",
            "rough draft",
            "get me started",
            "wireframe",
            "prototype",
            "MVP",
            "proof of concept",
            "POC",
            "quick prototype",
            "bare bones",
            "minimal version",
            "starting point",
            "init project",
            "create new",
            "bootstrap",
            "setup",
            "hello world",
            "base plate",
            "cookie cutter",
        ],
        steps=[
            "OUTLINE: Create high-level skeleton/structure",
            "EXPAND: Fill in each section in parallel",
            "CONNECT: Link sections together",
            "POLISH: Quick cleanup pass",
        ],
        use_case="Scaffolding, boilerplate, fast generation",
        complexity="low",
        task_type="docs",
    )
)

register(
    FrameworkDefinition(
        name="system1",
        display_name="System1 Fast",
        category=FrameworkCategory.FAST,
        description="Fast heuristic, minimal thinking. Best for trivial quick fixes.",
        best_for=["simple queries", "quick fixes", "trivial tasks"],
        vibes=[
            "quick question",
            "simple fix",
            "easy",
            "obvious",
            "just do it",
            "no brainer",
            "trivial",
            "one liner",
            "fast answer",
            "real quick",
            "simple",
            "straightforward",
            "basic",
            "elementary",
            "beginner",
            "101",
            "quick fix",
            "hotfix",
            "patch",
            "typo",
            "rename",
            "move",
            "delete",
            "add comment",
            "format code",
            "indentation",
            "whitespace",
        ],
        steps=[
            "RECOGNIZE: Pattern-match to known solution",
            "EXECUTE: Apply quick fix immediately",
        ],
        use_case="Simple queries, quick fixes, trivial tasks",
        complexity="low",
        task_type="quick",
    )
)

# =============================================================================
# VERIFICATION FRAMEWORKS (8 frameworks)
# =============================================================================

register(
    FrameworkDefinition(
        name="self_consistency",
        display_name="Self-Consistency",
        category=FrameworkCategory.VERIFICATION,
        description="Multi-sample voting for reliable answers. Best for ambiguous bugs and tricky logic.",
        best_for=["ambiguous bugs", "tricky logic", "multiple plausible fixes"],
        vibes=[
            "multiple answers",
            "vote",
            "consensus",
            "majority",
            "check multiple times",
            "sample answers",
            "consistency check",
            "are you sure",
            "double check",
            "verify answer",
            "ambiguous",
            "uncertain",
            "could be either",
            "tricky logic",
            "multiple plausible",
            "which is right",
        ],
        steps=[
            "SAMPLE: Generate multiple independent solutions",
            "COMPARE: Identify agreements and disagreements",
            "VOTE: Select answer with highest consistency",
            "VERIFY: Validate majority answer",
        ],
        use_case="Ambiguous bugs, tricky logic, verification",
        complexity="medium",
        task_type="verification",
    )
)

register(
    FrameworkDefinition(
        name="self_ask",
        display_name="Self-Ask",
        category=FrameworkCategory.VERIFICATION,
        description="Sub-question decomposition before solving. Best for unclear tickets and missing requirements.",
        best_for=["unclear tickets", "missing requirements", "multi-part debugging"],
        vibes=[
            "break down",
            "sub-questions",
            "what do I need to know",
            "before I can answer",
            "first need to understand",
            "unclear requirements",
            "missing info",
            "need clarification",
            "what questions should I ask",
            "decompose the problem",
            "prerequisite knowledge",
            "dependencies between parts",
        ],
        steps=[
            "ASK: What sub-questions must I answer first?",
            "ANSWER: Resolve each sub-question",
            "INTEGRATE: Combine sub-answers into final answer",
            "VERIFY: Check completeness",
        ],
        use_case="Unclear tickets, missing requirements, decomposition",
        complexity="medium",
        task_type="verification",
    )
)

register(
    FrameworkDefinition(
        name="rar",
        display_name="Rephrase-and-Respond",
        category=FrameworkCategory.VERIFICATION,
        description="Clarify request before solving. Best for vague prompts and ambiguous requirements.",
        best_for=["vague prompts", "ambiguous requirements", "clarification"],
        vibes=[
            "rephrase",
            "clarify the question",
            "what exactly",
            "be more specific",
            "unclear request",
            "vague",
            "what do you mean",
            "interpret as",
            "restate",
            "poorly written",
            "confusing request",
            "ambiguous ask",
        ],
        steps=[
            "REPHRASE: Restate the question clearly",
            "CONFIRM: Verify interpretation is correct",
            "RESPOND: Answer the clarified question",
        ],
        use_case="Vague prompts, ambiguous requirements, clarification",
        complexity="low",
        task_type="verification",
    )
)

register(
    FrameworkDefinition(
        name="verify_and_edit",
        display_name="Verify-and-Edit",
        category=FrameworkCategory.VERIFICATION,
        description="Verify claims then edit only failures. Best for surgical edits and code review.",
        best_for=["code review", "surgical edits", "implementation plans"],
        vibes=[
            "verify claims",
            "fact check",
            "edit only wrong parts",
            "surgical fix",
            "minimal changes",
            "preserve good parts",
            "check each claim",
            "verify then edit",
            "patch specific",
            "don't rewrite everything",
            "targeted fix",
            "precision edit",
        ],
        steps=[
            "VERIFY: Check each claim/assumption for correctness",
            "IDENTIFY: Mark only the incorrect parts",
            "EDIT: Fix only the verified failures",
            "PRESERVE: Keep all correct parts unchanged",
        ],
        use_case="Code review, surgical edits, minimal changes",
        complexity="medium",
        task_type="verification",
    )
)

register(
    FrameworkDefinition(
        name="rarr",
        display_name="RARR",
        category=FrameworkCategory.VERIFICATION,
        description="Research, Augment, Revise loop. Best for evidence-based answers and documentation.",
        best_for=["evidence-based answers", "documentation", "prove-it requirements"],
        vibes=[
            "evidence based",
            "prove it",
            "cite sources",
            "grounded",
            "research first",
            "find evidence",
            "back it up",
            "documentation says",
            "according to docs",
            "verified by",
            "need proof",
            "show me where",
            "source please",
        ],
        steps=[
            "RESEARCH: Find relevant evidence and sources",
            "AUGMENT: Enhance answer with found evidence",
            "REVISE: Update answer based on evidence",
            "CITE: Provide sources for claims",
        ],
        use_case="Evidence-based answers, documentation",
        complexity="medium",
        task_type="verification",
    )
)

register(
    FrameworkDefinition(
        name="selfcheckgpt",
        display_name="SelfCheckGPT",
        category=FrameworkCategory.VERIFICATION,
        description="Hallucination detection via sampling consistency. Best for high-stakes guidance.",
        best_for=["high-stakes guidance", "unfamiliar libraries", "final gate"],
        vibes=[
            "am I hallucinating",
            "is this real",
            "sanity check",
            "might be wrong",
            "not sure if true",
            "verify accuracy",
            "could be making this up",
            "confidence check",
            "before I trust this",
            "credibility check",
            "final check",
            "pre-flight",
            "gate check",
        ],
        steps=[
            "GENERATE: Create multiple independent answers",
            "COMPARE: Check consistency across samples",
            "FLAG: Identify inconsistent/potentially hallucinated claims",
            "VERIFY: Confirm or reject flagged claims",
        ],
        use_case="Hallucination detection, high-stakes verification",
        complexity="medium",
        task_type="verification",
    )
)

register(
    FrameworkDefinition(
        name="metaqa",
        display_name="MetaQA",
        category=FrameworkCategory.VERIFICATION,
        description="Metamorphic testing for reasoning reliability. Best for brittle reasoning and edge cases.",
        best_for=["brittle reasoning", "edge cases", "policy consistency"],
        vibes=[
            "test with variations",
            "edge cases",
            "what if slightly different",
            "invariants",
            "should work for all",
            "consistency across",
            "metamorphic",
            "transform and check",
            "robustness test",
            "works for this but not that",
            "brittle",
            "fragile logic",
        ],
        steps=[
            "TRANSFORM: Create variations of the input",
            "APPLY: Run reasoning on each variation",
            "CHECK: Verify outputs follow expected metamorphic relations",
            "FLAG: Identify inconsistencies",
        ],
        use_case="Metamorphic testing, edge cases, robustness",
        complexity="medium",
        task_type="verification",
    )
)

register(
    FrameworkDefinition(
        name="ragas",
        display_name="RAGAS",
        category=FrameworkCategory.VERIFICATION,
        description="RAG Assessment for retrieval quality. Best for evaluating RAG pipelines.",
        best_for=["RAG pipelines", "retrieval quality", "source grounding"],
        vibes=[
            "evaluate retrieval",
            "RAG quality",
            "were sources relevant",
            "faithfulness",
            "did it use sources",
            "grounded in docs",
            "retrieval quality",
            "chunk relevance",
            "answer quality",
            "RAG pipeline",
            "evaluate RAG",
            "retrieval assessment",
        ],
        steps=[
            "RETRIEVE: Get relevant documents",
            "SCORE: Evaluate relevance, faithfulness, context recall",
            "ANALYZE: Identify retrieval gaps",
            "IMPROVE: Suggest retrieval improvements",
        ],
        use_case="RAG pipeline evaluation, retrieval quality",
        complexity="medium",
        task_type="rag",
    )
)

# =============================================================================
# AGENT FRAMEWORKS (5 frameworks)
# =============================================================================

register(
    FrameworkDefinition(
        name="rewoo",
        display_name="ReWOO",
        category=FrameworkCategory.AGENT,
        description="Reasoning Without Observation - plan then execute. Best for multi-step tasks with tools.",
        best_for=["multi-step tasks", "cost control", "tool efficiency"],
        vibes=[
            "plan first then execute",
            "tool schedule",
            "batch tools",
            "reasoning without observation",
            "plan once",
            "efficient tools",
            "tool-free plan",
            "expected observations",
            "tool budget",
            "minimize tool calls",
            "strategic tools",
            "planned execution",
        ],
        steps=[
            "PLAN: Create complete action plan without tool calls",
            "PREDICT: Anticipate expected observations",
            "EXECUTE: Run all planned tools",
            "SYNTHESIZE: Combine results into answer",
        ],
        use_case="Multi-step tasks, tool efficiency, cost control",
        complexity="medium",
        task_type="agent",
    )
)

register(
    FrameworkDefinition(
        name="lats",
        display_name="LATS",
        category=FrameworkCategory.AGENT,
        description="Language Agent Tree Search over action sequences. Best for complex repo changes.",
        best_for=["complex repo changes", "multiple fix paths", "uncertain root cause"],
        vibes=[
            "action tree",
            "multiple paths",
            "backtrack if fails",
            "tree search",
            "action sequences",
            "branch and bound",
            "rollback possible",
            "try different paths",
            "agent search",
            "strategic agent",
            "action planning",
            "branch exploration",
        ],
        steps=[
            "EXPAND: Generate possible action sequences",
            "EVALUATE: Score each sequence's promise",
            "BACKTRACK: If stuck, return to earlier decision point",
            "COMMIT: Execute best path when confident",
        ],
        use_case="Complex repo changes, multiple fix paths",
        complexity="high",
        task_type="agent",
    )
)

register(
    FrameworkDefinition(
        name="mrkl",
        display_name="MRKL",
        category=FrameworkCategory.AGENT,
        description="Modular Reasoning with specialized modules. Best for big systems and mixed domains.",
        best_for=["big systems", "mixed domains", "tool-rich setups"],
        vibes=[
            "specialized modules",
            "security module",
            "perf module",
            "modular reasoning",
            "route to expert",
            "different perspectives",
            "orchestrate modules",
            "domain experts",
            "specialized handling",
            "big system",
            "mixed domains",
            "module routing",
        ],
        steps=[
            "CLASSIFY: Determine which expert module(s) needed",
            "ROUTE: Send to specialized module",
            "COLLECT: Gather module outputs",
            "SYNTHESIZE: Combine expert opinions",
        ],
        use_case="Big systems, mixed domains, expert routing",
        complexity="high",
        task_type="agent",
    )
)

register(
    FrameworkDefinition(
        name="swe_agent",
        display_name="SWE-Agent",
        category=FrameworkCategory.AGENT,
        description="Repo-first execution loop. Best for multi-file bugfixes and CI failures.",
        best_for=["multi-file bugfixes", "CI failures", "make tests pass"],
        vibes=[
            "make tests pass",
            "CI green",
            "fix build",
            "inspect repo",
            "multi-file fix",
            "iterate until passing",
            "run tests",
            "check lint",
            "verify build",
            "software engineering",
            "repo workflow",
            "CI/CD fix",
        ],
        steps=[
            "EXPLORE: Understand repo structure and codebase",
            "LOCALIZE: Find files causing the issue",
            "EDIT: Make necessary changes",
            "TEST: Run tests until all pass",
            "ITERATE: Repeat if tests fail",
        ],
        use_case="Multi-file bugfixes, CI failures, make tests pass",
        complexity="high",
        task_type="agent",
    )
)

register(
    FrameworkDefinition(
        name="toolformer",
        display_name="Toolformer",
        category=FrameworkCategory.AGENT,
        description="Smart tool selection policy. Best for preventing pointless tool calls.",
        best_for=["router logic", "preventing pointless calls", "tool efficiency"],
        vibes=[
            "when to use tools",
            "tool decision",
            "justify tool use",
            "tool policy",
            "smart tools",
            "efficient tool use",
            "tool selection",
            "optimize tool calls",
            "tool strategy",
            "when is tool needed",
            "tool cost benefit",
        ],
        steps=[
            "ASSESS: Can I answer without tools?",
            "JUSTIFY: What value would tool call add?",
            "SELECT: Choose minimal necessary tools",
            "EXECUTE: Call only justified tools",
        ],
        use_case="Smart tool selection, cost efficiency",
        complexity="medium",
        task_type="agent",
    )
)

# =============================================================================
# RAG FRAMEWORKS (5 frameworks)
# =============================================================================

register(
    FrameworkDefinition(
        name="self_rag",
        display_name="Self-RAG",
        category=FrameworkCategory.RAG,
        description="Self-triggered selective retrieval. Best for mixed knowledge tasks.",
        best_for=["mixed knowledge tasks", "large corpora", "targeted retrieval"],
        vibes=[
            "retrieve when needed",
            "selective retrieval",
            "confidence based",
            "not sure need to check",
            "maybe retrieve",
            "targeted search",
            "retrieve for uncertain",
            "gap-driven retrieval",
            "don't retrieve everything",
            "smart retrieval",
        ],
        steps=[
            "ASSESS: Do I need external knowledge?",
            "RETRIEVE: If uncertain, fetch relevant docs",
            "CRITIQUE: Evaluate retrieval relevance",
            "GENERATE: Answer using retrieved context",
        ],
        use_case="Selective retrieval, mixed knowledge tasks",
        complexity="medium",
        task_type="rag",
    )
)

register(
    FrameworkDefinition(
        name="hyde",
        display_name="HyDE",
        category=FrameworkCategory.RAG,
        description="Hypothetical Document Embeddings for better retrieval. Best for fuzzy search.",
        best_for=["fuzzy search", "unclear intent", "broad problems"],
        vibes=[
            "hypothetical document",
            "ideal answer",
            "what would doc say",
            "fuzzy query",
            "vague search",
            "broad search",
            "unclear what to search",
            "improve retrieval",
            "query expansion",
            "semantic search improvement",
        ],
        steps=[
            "HYPOTHESIZE: Generate hypothetical ideal answer",
            "EMBED: Create embedding from hypothetical doc",
            "RETRIEVE: Search using hypothetical embedding",
            "ANSWER: Generate from real retrieved docs",
        ],
        use_case="Fuzzy search, query expansion, better retrieval",
        complexity="medium",
        task_type="rag",
    )
)

register(
    FrameworkDefinition(
        name="rag_fusion",
        display_name="RAG-Fusion",
        category=FrameworkCategory.RAG,
        description="Multi-query retrieval with rank fusion. Best for improving recall.",
        best_for=["improving recall", "complex queries", "noisy corpora"],
        vibes=[
            "multiple queries",
            "query variations",
            "fuse results",
            "rank fusion",
            "better recall",
            "diverse queries",
            "query diversity",
            "combine searches",
            "aggregate results",
            "comprehensive search",
            "thorough retrieval",
        ],
        steps=[
            "EXPAND: Generate multiple query variations",
            "RETRIEVE: Search with each query variant",
            "FUSE: Combine results using rank fusion",
            "SELECT: Take top-k from fused results",
        ],
        use_case="Multi-query retrieval, improving recall",
        complexity="medium",
        task_type="rag",
    )
)

register(
    FrameworkDefinition(
        name="raptor",
        display_name="RAPTOR",
        category=FrameworkCategory.RAG,
        description="Hierarchical abstraction retrieval. Best for huge repos and long docs.",
        best_for=["huge repos", "long docs", "monorepos"],
        vibes=[
            "hierarchical search",
            "summary first",
            "drill down",
            "abstraction levels",
            "overview then details",
            "huge document",
            "long document",
            "large codebase",
            "lost in chunks",
            "tree retrieval",
            "multi-level",
        ],
        steps=[
            "CLUSTER: Group chunks into hierarchical summaries",
            "SEARCH TOP: Query at highest abstraction level",
            "DRILL DOWN: Descend to relevant lower levels",
            "RETRIEVE: Get specific chunks from target level",
        ],
        use_case="Huge repos, long docs, hierarchical retrieval",
        complexity="high",
        task_type="rag",
    )
)

register(
    FrameworkDefinition(
        name="graphrag",
        display_name="GraphRAG",
        category=FrameworkCategory.RAG,
        description="Entity-relation grounding for dependencies. Best for architecture questions.",
        best_for=["architecture questions", "module relationships", "impact analysis"],
        vibes=[
            "entity relations",
            "dependency graph",
            "how things connect",
            "module relationships",
            "call graph",
            "data flow",
            "impact analysis",
            "blast radius",
            "what depends on",
            "architecture map",
            "relation extraction",
            "knowledge graph",
        ],
        steps=[
            "EXTRACT: Identify entities and relationships",
            "BUILD: Construct knowledge graph",
            "QUERY: Traverse graph for relevant connections",
            "GROUND: Use graph context for answer",
        ],
        use_case="Architecture questions, dependency analysis",
        complexity="high",
        task_type="rag",
    )
)


# =============================================================================
# CONVENIENCE ACCESSORS
# =============================================================================


@lru_cache(maxsize=128)
def get_framework(name: str) -> FrameworkDefinition:
    """Get a framework by name.

    Args:
        name: Framework name (e.g., "active_inference")

    Returns:
        FrameworkDefinition for the requested framework

    Raises:
        FrameworkNotFoundError: If framework doesn't exist
        
    Note:
        Cached for performance (maxsize=128). Same name returns cached result.
    """
    if name not in FRAMEWORKS:
        raise FrameworkNotFoundError(
            f"Unknown framework: {name}",
            details={"requested": name, "available": list(FRAMEWORKS.keys())},
        )
    return FRAMEWORKS[name]


@lru_cache(maxsize=64)
def get_framework_safe(name: str) -> FrameworkDefinition | None:
    """Get a framework by name, returning None if not found.
    
    Note:
        Cached for performance (maxsize=64). Same name returns cached result.
    """
    return FRAMEWORKS.get(name)


@lru_cache(maxsize=16)
def get_frameworks_by_category(category: FrameworkCategory) -> tuple[FrameworkDefinition, ...]:
    """Get all frameworks in a specific category.
    
    Note:
        Returns tuple instead of list to enable caching (lists aren't hashable).
        Cached for performance (maxsize=16, one per category).
    """
    return tuple(f for f in FRAMEWORKS.values() if f.category == category)


@lru_cache(maxsize=1)
def get_all_vibes() -> dict[str, list[str]]:
    """Get all vibes for all frameworks (for vibe dictionary compatibility).
    
    Note:
        Cached (maxsize=1). Dictionary is immutable after framework registration.
    """
    return {name: fw.vibes for name, fw in FRAMEWORKS.items()}


def get_framework_names() -> list[str]:
    """Get list of all registered framework names."""
    return list(FRAMEWORKS.keys())


def infer_task_type(framework: str) -> str:
    """Infer task type from chosen framework."""
    if framework in FRAMEWORKS:
        return FRAMEWORKS[framework].task_type
    return "unknown"


def get_frameworks_dict() -> dict[str, str]:
    """Get frameworks as name -> description dict (for router compatibility)."""
    return {name: fw.description for name, fw in FRAMEWORKS.items()}


@lru_cache(maxsize=1)
def list_by_category() -> dict[str, list[str]]:
    """List framework names organized by category.
    
    Note:
        Cached (maxsize=1). Dictionary is immutable after framework registration.
    """
    result: dict[str, list[str]] = {}
    for cat in FrameworkCategory:
        frameworks = get_frameworks_by_category(cat)
        if frameworks:
            result[cat.value] = [f.name for f in frameworks]
    return result


def count() -> int:
    """Return total number of registered frameworks."""
    return len(FRAMEWORKS)


@lru_cache(maxsize=256)
def find_by_vibe(vibe: str) -> FrameworkDefinition | None:
    """Find a framework that matches a vibe pattern.
    
    Note:
        Cached for performance (maxsize=256). Same vibe returns cached result.
    """
    vibe_lower = vibe.lower()
    for spec in FRAMEWORKS.values():
        if any(v.lower() in vibe_lower for v in spec.vibes):
            return spec
    return None


# =============================================================================
# VIBE DICTIONARY COMPATIBILITY
# =============================================================================

# This provides backward compatibility with the VIBE_DICTIONARY import
VIBE_DICTIONARY = get_all_vibes()


def match_vibes(query: str) -> str | None:
    """
    Check query against vibe dictionary with weighted scoring.

    Scoring rules:
    - Longer phrase matches score higher (phrase length * 2)
    - Multiple matches accumulate
    - Multi-word phrases get bonus weight

    Args:
        query: The user's natural language query

    Returns:
        Framework name if strong match found, None otherwise.
    """
    query_lower = query.lower()

    # Score each framework by vibe matches with weighted scoring
    scores = {}
    for framework, vibes in VIBE_DICTIONARY.items():
        total_score = 0.0

        for vibe in vibes:
            if vibe in query_lower:
                # Weight by phrase length - longer phrases are more specific
                word_count = len(vibe.split())
                # Multi-word phrases get bonus (2x for 2 words, 3x for 3+ words)
                weight = word_count if word_count >= 2 else 0.5
                total_score += weight

        if total_score > 0:
            scores[framework] = total_score

    # Return if we have a clear winner
    if scores:
        best = max(scores, key=scores.get)
        second_best_score = 0
        if len(scores) > 1:
            second_best_score = sorted(scores.values(), reverse=True)[1]

        # Require:
        # - Score >= 2 (at least one 2-word match or two 1-word matches), OR
        # - Clear leader (50% more than second place), OR
        # - Single match with score >= 1
        score_margin = scores[best] / (second_best_score + 0.001)
        if scores[best] >= 2 or score_margin >= 1.5 or (len(scores) == 1 and scores[best] >= 0.5):
            return best

    return None
