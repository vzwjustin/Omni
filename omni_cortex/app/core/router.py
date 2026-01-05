"""
Hyper-Router: Intelligent LLM-Powered Framework Selection

Uses hierarchical category routing with specialist agents.
Supports framework chaining for complex multi-step tasks.
Designed for vibe coders and senior engineers who just want it to work.
"""

import re
from typing import Optional, List, Tuple
from ..state import GraphState


class HyperRouter:
    """
    The Hyper-Dispatcher: Hierarchical AI-powered framework selection.

    Three-stage routing:
    1. CATEGORY: Fast pattern match to one of 9 categories
    2. SPECIALIST: Category agent picks 1+ frameworks (supports chaining)
    3. EXECUTE: Run framework(s) in sequence, passing state

    Modes:
    - AUTO (default): Hierarchical routing with specialist agents
    - HEURISTIC: Fast pattern matching only (fallback)
    """

    # ==========================================================================
    # CATEGORY DEFINITIONS - First stage routing
    # ==========================================================================

    CATEGORIES = {
        "debug": {
            "description": "Bug hunting, error analysis, root cause investigation",
            "specialist": "Debug Detective",
            "frameworks": ["active_inference", "self_debugging", "reverse_cot",
                          "mcts_rstar", "chain_of_code", "rubber_duck"],
            "chain_patterns": {
                "complex_bug": ["self_ask", "active_inference", "verify_and_edit"],
                "silent_bug": ["reverse_cot", "self_debugging", "selfcheckgpt"],
                "flaky_test": ["active_inference", "tdd_prompting", "self_consistency"],
            }
        },
        "code_gen": {
            "description": "Writing new code, algorithms, implementations",
            "specialist": "Code Architect",
            "frameworks": ["alphacodium", "codechain", "parsel", "docprompting",
                          "procoder", "recode", "pal", "program_of_thoughts",
                          "evol_instruct", "llmloop", "tdd_prompting"],
            "chain_patterns": {
                "complex_feature": ["plan_and_solve", "parsel", "tdd_prompting", "self_refine"],
                "api_integration": ["docprompting", "critic", "verify_and_edit"],
                "algorithm": ["step_back", "alphacodium", "self_debugging"],
            }
        },
        "refactor": {
            "description": "Code cleanup, restructuring, modernization",
            "specialist": "Refactor Surgeon",
            "frameworks": ["graph_of_thoughts", "everything_of_thought",
                          "least_to_most", "codechain", "self_refine"],
            "chain_patterns": {
                "major_rewrite": ["plan_and_solve", "graph_of_thoughts", "verify_and_edit"],
                "modular_extract": ["least_to_most", "parsel", "self_refine"],
                "legacy_cleanup": ["chain_of_note", "graph_of_thoughts", "tdd_prompting"],
            }
        },
        "architecture": {
            "description": "System design, planning, high-level decisions",
            "specialist": "System Architect",
            "frameworks": ["reason_flux", "plan_and_solve", "comparative_arch",
                          "multi_agent_debate", "state_machine", "coala"],
            "chain_patterns": {
                "new_system": ["reason_flux", "multi_agent_debate", "plan_and_solve"],
                "scale_decision": ["step_back", "comparative_arch", "verify_and_edit"],
                "workflow_design": ["state_machine", "plan_and_solve", "critic"],
            }
        },
        "verification": {
            "description": "Checking, validating, proving correctness",
            "specialist": "Verification Expert",
            "frameworks": ["chain_of_verification", "verify_and_edit", "self_consistency",
                          "selfcheckgpt", "metaqa", "rarr", "red_team"],
            "chain_patterns": {
                "security_audit": ["red_team", "chain_of_verification", "verify_and_edit"],
                "claim_check": ["self_ask", "rarr", "selfcheckgpt"],
                "code_review": ["chain_of_verification", "self_consistency", "verify_and_edit"],
            }
        },
        "agent": {
            "description": "Multi-step tasks, tool use, autonomous execution",
            "specialist": "Agent Orchestrator",
            "frameworks": ["swe_agent", "react", "rewoo", "lats",
                          "mrkl", "toolformer", "reflexion"],
            "chain_patterns": {
                "ci_fix": ["swe_agent", "tdd_prompting", "verify_and_edit"],
                "multi_file": ["coala", "swe_agent", "self_refine"],
                "tool_heavy": ["rewoo", "react", "reflexion"],
            }
        },
        "rag": {
            "description": "Retrieval, documentation, knowledge grounding",
            "specialist": "Knowledge Navigator",
            "frameworks": ["self_rag", "hyde", "rag_fusion", "raptor",
                          "graphrag", "chain_of_note", "ragas"],
            "chain_patterns": {
                "large_codebase": ["raptor", "graphrag", "chain_of_note"],
                "fuzzy_search": ["hyde", "rag_fusion", "rarr"],
                "dependency_map": ["graphrag", "least_to_most", "chain_of_note"],
            }
        },
        "exploration": {
            "description": "Novel problems, learning, creative solutions",
            "specialist": "Explorer",
            "frameworks": ["self_discover", "analogical", "buffer_of_thoughts",
                          "adaptive_injection", "chain_of_thought", "step_back"],
            "chain_patterns": {
                "novel_problem": ["self_ask", "self_discover", "verify_and_edit"],
                "learn_codebase": ["chain_of_note", "graphrag", "analogical"],
                "creative_solution": ["analogical", "multi_agent_debate", "self_refine"],
            }
        },
        "fast": {
            "description": "Quick fixes, simple tasks, scaffolding",
            "specialist": "Speed Demon",
            "frameworks": ["system1", "skeleton_of_thought", "rar", "scratchpads"],
            "chain_patterns": {}  # Fast category doesn't chain
        },
    }

    # Quick category vibes for first-stage routing
    CATEGORY_VIBES = {
        "debug": [
            "bug", "error", "broken", "crash", "fix", "debug", "failing", "wrong",
            "doesn't work", "not working", "exception", "null", "undefined",
            "wtf", "what the", "why is this", "investigate", "root cause",
            "stack trace", "segfault", "memory", "race condition", "deadlock"
        ],
        "code_gen": [
            "write", "create", "implement", "generate", "build", "code",
            "function", "class", "algorithm", "feature", "add", "new",
            "from scratch", "scaffold", "boilerplate", "template"
        ],
        "refactor": [
            "refactor", "clean", "restructure", "reorganize", "simplify",
            "modernize", "legacy", "spaghetti", "tech debt", "extract",
            "modular", "split", "combine", "rename", "move"
        ],
        "architecture": [
            "design", "architect", "system", "structure", "plan", "scale",
            "microservice", "monolith", "pattern", "decision", "trade-off",
            "how should", "best approach", "high level"
        ],
        "verification": [
            "verify", "check", "validate", "secure", "review", "audit",
            "correct", "prove", "test", "confirm", "safe", "vulnerability",
            "owasp", "injection", "xss"
        ],
        "agent": [
            "multi-step", "automate", "ci", "cd", "pipeline", "workflow",
            "tool", "execute", "run", "iterate", "loop", "until"
        ],
        "rag": [
            "find", "search", "retrieve", "documentation", "docs", "where",
            "which file", "codebase", "understand", "learn", "explore"
        ],
        "exploration": [
            "novel", "creative", "new approach", "different", "alternative",
            "no idea", "stuck", "help me think", "weird", "unusual"
        ],
        "fast": [
            "quick", "simple", "easy", "trivial", "just", "only",
            "one line", "minor", "small", "typo", "rename"
        ],
    }
    
    # All available frameworks with descriptions for the AI
    FRAMEWORKS = {
        "active_inference": "Debugging loop: hypothesis → predict → compare → update. Best for nasty bugs and root cause analysis.",
        "graph_of_thoughts": "Non-linear thinking with merge/aggregate. Best for refactoring spaghetti code and restructuring.",
        "reason_flux": "Hierarchical planning: template → expand → refine. Best for architecture and system design.",
        "tree_of_thoughts": "BFS/DFS exploration of solutions. Best for algorithms and optimization problems.",
        "skeleton_of_thought": "Outline-first with parallel expansion. Best for docs, boilerplate, fast generation.",
        "critic": "External tool verification. Best for API usage validation and library integration.",
        "chain_of_verification": "Draft → verify → patch cycle. Best for security review and code validation.",
        "program_of_thoughts": "Generate executable Python to compute. Best for math, data processing, testing.",
        "self_discover": "Compose custom reasoning from atomic modules. Best for novel/unclear problems.",
        "mcts_rstar": "Monte Carlo Tree Search for code. Best for complex multi-step bugs and optimization.",
        "multi_agent_debate": "Proponent vs Critic argumentation. Best for design decisions and trade-offs.",
        "everything_of_thought": "MCTS + fast generation + verification. Best for large complex changes.",
        "buffer_of_thoughts": "Retrieve proven thought templates. Best for repetitive known patterns.",
        "coala": "Cognitive architecture with working + episodic memory. Best for long-context multi-file tasks.",
        "chain_of_note": "Research mode with gap analysis. Best for documentation and learning codebases.",
        "step_back": "Abstract first, then implement. Best for performance and complexity analysis.",
        "analogical": "Find analogies to solve problems. Best for creative solutions and pattern recognition.",
        "adaptive_injection": "Dynamic thinking depth based on complexity. Best for mixed-complexity tasks.",
        "re2": "Two-pass: goals → constraints. Best for complex specs and requirements.",
        "system1": "Fast heuristic, minimal thinking. Best for trivial quick fixes.",
        # New frameworks (2026 Edition + Modern LLM)
        "chain_of_code": "Break problems into code blocks for structured thinking. Best for logic puzzles and algorithmic debugging.",
        "self_debugging": "Mental execution trace before presenting code. Best for preventing off-by-one and null pointer bugs.",
        "tdd_prompting": "Write tests first, then implementation. Best for edge case coverage and new features.",
        "reverse_cot": "Work backward from buggy output to source. Best for silent bugs with wrong outputs.",
        "rubber_duck": "Socratic questioning for self-discovery. Best for architectural bottlenecks and blind spots.",
        "react": "Interleaved reasoning and acting with tools. Best for multi-step tasks requiring tool use.",
        "reflexion": "Self-evaluation with memory-based learning. Best for learning from failed attempts.",
        "self_refine": "Iterative self-critique and improvement. Best for code quality and documentation.",
        "least_to_most": "Bottom-up atomic function decomposition. Best for complex systems and refactoring monoliths.",
        "comparative_arch": "Multiple solution approaches (readability/memory/speed). Best for optimization and architecture decisions.",
        "plan_and_solve": "Explicit planning before execution. Best for complex features and avoiding rushed implementations.",
        "red_team": "Adversarial security analysis (STRIDE, OWASP). Best for security audits and vulnerability scanning.",
        "state_machine": "Formal FSM design before coding. Best for UI/UX logic, game dev, and workflows.",
        "chain_of_thought": "Basic step-by-step reasoning. Best for complex reasoning and logical deduction.",
        # Additional coding frameworks (2026 expansion)
        "alphacodium": "Test-based multi-stage iterative code generation. Best for competitive programming and complex algorithms.",
        "codechain": "Chain of self-revisions guided by sub-modules. Best for modular code generation and incremental refinement.",
        "evol_instruct": "Evolutionary instruction complexity for code. Best for challenging code problems and constraint-based coding.",
        "llmloop": "Automated iterative feedback loops for code+tests. Best for code quality assurance and production-ready code.",
        "procoder": "Compiler-feedback-guided iterative refinement. Best for project-level code generation and API usage.",
        "recode": "Multi-candidate validation with CFG-based debugging. Best for reliable code generation and high-stakes code.",
        # Verification frameworks (2026 Expansion)
        "self_consistency": "Multi-sample voting for reliable answers. Best for ambiguous bugs and tricky logic.",
        "self_ask": "Sub-question decomposition before solving. Best for unclear tickets and missing requirements.",
        "rar": "Rephrase-and-Respond for clarity. Best for vague prompts and ambiguous requirements.",
        "verify_and_edit": "Verify claims then edit only failures. Best for surgical edits and code review.",
        "rarr": "Research, Augment, Revise loop. Best for evidence-based answers and documentation.",
        "selfcheckgpt": "Hallucination detection via sampling consistency. Best for high-stakes guidance.",
        "metaqa": "Metamorphic testing for reasoning reliability. Best for brittle reasoning and edge cases.",
        "ragas": "RAG Assessment for retrieval quality. Best for evaluating RAG pipelines.",
        # Agent frameworks (2026 Expansion)
        "rewoo": "Reasoning Without Observation - plan then execute. Best for multi-step tasks with tools.",
        "lats": "Language Agent Tree Search over action sequences. Best for complex repo changes.",
        "mrkl": "Modular Reasoning with specialized modules. Best for big systems and mixed domains.",
        "swe_agent": "Repo-first execution loop. Best for multi-file bugfixes and CI failures.",
        "toolformer": "Smart tool selection policy. Best for preventing pointless tool calls.",
        # RAG frameworks (2026 Expansion)
        "self_rag": "Self-triggered selective retrieval. Best for mixed knowledge tasks.",
        "hyde": "Hypothetical Document Embeddings for better retrieval. Best for fuzzy search.",
        "rag_fusion": "Multi-query retrieval with rank fusion. Best for improving recall.",
        "raptor": "Hierarchical abstraction retrieval. Best for huge repos and long docs.",
        "graphrag": "Entity-relation grounding for dependencies. Best for architecture questions.",
        # Additional code frameworks (2026 Expansion)
        "pal": "Program-Aided Language - code as reasoning substrate. Best for algorithms and numeric logic.",
        "scratchpads": "Structured intermediate reasoning workspace. Best for multi-step fixes.",
        # Additional code frameworks (2026 Expansion - Part 2)
        "parsel": "Compositional code generation from natural language specs. Builds dependency graph of functions.",
        "docprompting": "Documentation-driven code generation. Retrieves docs and examples to guide code generation."
    }
    
    # Heuristic patterns (fallback)
    PATTERNS = {
        "debug": [r"bug", r"error", r"exception", r"crash", r"fix", r"broken", r"fails", r"null", r"undefined"],
        "refactor": [r"refactor", r"clean", r"reorganize", r"legacy", r"spaghetti", r"debt"],
        "architecture": [r"architect", r"design", r"structure", r"system", r"microservice", r"scale"],
        "algorithm": [r"algorithm", r"optimize", r"complexity", r"performance", r"efficient", r"sort", r"search"],
        "docs": [r"document", r"boilerplate", r"template", r"scaffold", r"generate"],
        "api": [r"api", r"endpoint", r"rest", r"integration", r"library"],
        "security": [r"security", r"vulnerab", r"injection", r"xss", r"auth", r"owasp", r"pen.?test"],
        "test": [r"test", r"unit", r"coverage", r"mock", r"tdd", r"pytest", r"jest"],
        "math": [r"calculate", r"compute", r"math", r"formula"],
        # New categories for 60 framework support
        "verification": [r"verify", r"validate", r"check", r"confirm", r"prove", r"evidence", r"hallucin"],
        "agent": [r"agent", r"multi.?step", r"tool.?use", r"automat", r"workflow", r"ci.?cd", r"pipeline"],
        "rag": [r"retriev", r"rag", r"search.*doc", r"vector", r"embed", r"knowledge.?base", r"corpus"],
        "competitive": [r"leetcode", r"hackerrank", r"codeforce", r"contest", r"competitive", r"interview"],
        "research": [r"understand", r"explain", r"learn", r"research", r"onboard", r"document"],
        "planning": [r"plan", r"roadmap", r"strategy", r"methodic", r"step.?by.?step"]
    }
    
    HEURISTIC_MAP = {
        "debug": "active_inference",
        "refactor": "graph_of_thoughts",
        "architecture": "reason_flux",
        "algorithm": "tree_of_thoughts",
        "docs": "skeleton_of_thought",
        "api": "critic",
        "security": "chain_of_verification",
        "test": "tdd_prompting",
        "math": "program_of_thoughts",
        # New categories for 60 framework support
        "verification": "verify_and_edit",
        "agent": "swe_agent",
        "rag": "self_rag",
        "competitive": "alphacodium",
        "research": "chain_of_note",
        "planning": "plan_and_solve",
        "unknown": "self_discover"
    }
    
    # =========================================================================
    # VIBE DICTIONARY: Casual activation phrases for vibe coders
    # =========================================================================
    # Just say what you want naturally - we'll figure out the right framework
    
    VIBE_DICTIONARY = {
        # Debugging vibes - hypothesis testing loop
        "active_inference": [
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

        # Refactoring vibes - structured cleanup
        "graph_of_thoughts": [
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

        # Architecture vibes - system design
        "reason_flux": [
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

        # Algorithm/optimization vibes
        "tree_of_thoughts": [
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

        # Fast generation vibes - scaffolding
        "skeleton_of_thought": [
            "just generate", "scaffold", "boilerplate", "quick template",
            "stub this out", "starter code", "basic structure", "skeleton",
            "rough draft", "get me started", "wireframe", "prototype",
            "MVP", "proof of concept", "POC", "quick prototype",
            "bare bones", "minimal version", "starting point",
            "init project", "create new", "bootstrap", "setup",
            "hello world", "base plate", "cookie cutter"
        ],

        # API/library vibes - external integration
        "critic": [
            "using this library", "api integration", "third party",
            "how do I use", "sdk", "package", "library docs",
            "correct usage", "best practice for", "npm package",
            "pip install", "import", "dependency", "module",
            "REST API", "GraphQL", "webhook", "OAuth", "JWT",
            "external service", "integration", "connect to",
            "API call", "fetch", "axios", "requests",
            "read the docs", "documentation says", "api reference"
        ],

        # Security/verification vibes
        "chain_of_verification": [
            "is this secure", "security check", "audit this", "vulnerabilities",
            "could this be hacked", "pen test", "code review",
            "sanity check", "validate this", "double check",
            "verify", "confirm", "make sure", "correct?",
            "looks right?", "review my code", "spot check",
            "find issues", "catch bugs", "QA", "quality check",
            "before merging", "PR review", "pull request",
            "lgtm", "nitpick", "blocker", "critical issue"
        ],

        # Math/compute vibes - calculations
        "program_of_thoughts": [
            "calculate", "compute", "do the math", "run the numbers",
            "data processing", "transform data", "crunch", "analyze data",
            "formula", "equation", "statistics", "aggregate",
            "sum", "average", "median", "percentile", "distribution",
            "pandas", "numpy", "dataframe", "CSV", "JSON transform",
            "ETL", "data pipeline", "data munging", "wrangling",
            "fibonacci", "factorial", "prime numbers", "matrix multiplication"
        ],

        # Creative/novel vibes - unique problems
        "self_discover": [
            "I have no idea", "weird problem", "never seen this",
            "creative solution", "think outside box", "novel approach",
            "unconventional", "unique situation", "edge case",
            "bizarre", "strange bug", "unusual", "one-off",
            "special case", "corner case", "rare scenario",
            "no documentation", "undocumented", "black box",
            "reverse engineering", "figure it out", "puzzling",
            "magic", "voodoo", "haunted code", "ghost in the machine"
        ],

        # Deep search vibes - complex exploration
        "mcts_rstar": [
            "really hard bug", "been stuck for hours", "complex issue",
            "multi-step problem", "deep issue", "intricate bug",
            "need to explore options", "thorough search",
            "exhaustive", "all possibilities", "brute force",
            "search space", "combinatorial", "permutations",
            "game tree", "decision tree", "branch and bound",
            "stuck for days", "impossible bug", "nightmare",
            "burning out", "hitting a wall", "desperate"
        ],

        # Decision vibes - weighing options
        "multi_agent_debate": [
            "should I use A or B", "trade-offs", "pros and cons",
            "which approach", "compare options", "decision", "evaluate",
            "weigh options", "what would you recommend", "best choice",
            "React or Vue", "SQL or NoSQL", "REST or GraphQL",
            "which framework", "which library", "which database",
            "monorepo or multirepo", "serverless or containers",
            "build vs buy", "roll own or use existing", "opinions",
            "debate", "argument", "settle this", "tie breaker"
        ],

        # Big change vibes - major overhaul
        "everything_of_thought": [
            "major rewrite", "big migration", "overhaul", "massive change",
            "complete redesign", "from scratch", "total refactor",
            "modernize", "upgrade everything", "v2", "next version",
            "breaking changes", "deprecate", "sunset", "EOL",
            "migration path", "upgrade path", "replatform",
            "technical transformation", "rebuild", "redo everything",
            "strangler fig", "rip and replace", "lift and shift"
        ],

        # Pattern vibes - known solutions
        "buffer_of_thoughts": [
            "I've done this before", "standard pattern", "common task",
            "typical", "usual approach", "boilerplate", "routine",
            "CRUD", "login flow", "auth", "pagination",
            "search filter", "form validation", "file upload",
            "email sending", "notifications", "logging",
            "error handling", "retry logic", "rate limiting",
            "design pattern", "singleton", "factory", "observer"
        ],

        # Long context vibes - multi-file awareness
        "coala": [
            "lots of files", "whole codebase", "across multiple files",
            "context from", "remember earlier", "stateful", "keep track",
            "monorepo", "large codebase", "enterprise", "legacy system",
            "multiple services", "cross-cutting", "shared code",
            "dependencies between", "imports from", "circular dependency",
            "file structure", "project organization", "modules",
            "global search", "find usages", "where is this defined"
        ],

        # Research vibes - understanding code
        "chain_of_note": [
            "understand this code", "explain", "what does this do",
            "learn the codebase", "document", "figure out how",
            "reverse engineer", "how does it work", "explain like",
            "ELI5", "walk through", "code walkthrough",
            "documentation", "comments", "README", "wiki",
            "onboarding", "new to codebase", "ramping up",
            "summary", "tldr", "synopsis", "notes"
        ],

        # Abstraction vibes - first principles
        "step_back": [
            "big O", "complexity analysis", "fundamentals", "first principles",
            "underlying concept", "theory behind", "abstract thinking",
            "CS fundamentals", "data structures", "algorithms",
            "design principles", "patterns", "anti-patterns",
            "why does this work", "how does this work", "under the hood",
            "internals", "implementation details", "deep dive",
            "zoom out", "high level view", "conceptual"
        ],

        # Analogy vibes - similar solutions
        "analogical": [
            "like when", "similar to", "pattern from", "reminds me of",
            "same as", "analogous", "comparable to", "like in",
            "inspired by", "borrowed from", "adapted from",
            "seen this before", "familiar pattern", "like that other",
            "copy from", "based on", "reference implementation",
            "metaphor", "example", "instance"
        ],

        # Adaptive vibes - flexible approach
        "adaptive_injection": [
            "just figure it out", "do your thing", "whatever works",
            "adapt", "flex", "go with the flow", "surprise me",
            "your call", "you decide", "best judgment",
            "improvise", "wing it", "play it by ear",
            "dynamic approach", "flexible solution",
            "smart select", "auto mode", "magic mode"
        ],

        # Requirements vibes - spec driven
        "re2": [
            "requirements", "spec", "constraints", "must have",
            "needs to", "requirements doc", "acceptance criteria",
            "user story", "ticket", "JIRA", "specification",
            "functional requirements", "non-functional", "NFR",
            "SLA", "compliance", "regulatory", "legal requirements",
            "business rules", "validation rules", "invariants",
            "as a user", "given when then", "checklist"
        ],

        # Quick vibes - simple tasks
        "system1": [
            "quick question", "simple fix", "easy", "obvious",
            "just do it", "no brainer", "trivial", "one liner",
            "fast answer", "real quick", "simple", "straightforward",
            "basic", "elementary", "beginner", "101",
            "quick fix", "hotfix", "patch", "typo",
            "rename", "move", "delete", "add comment",
            "format code", "indentation", "whitespace"
        ],

        # Code decomposition vibes
        "chain_of_code": [
            "code blocks", "pseudocode", "execution trace", "logic puzzle",
            "recursive logic", "algorithmic complexity", "structured thinking",
            "break into code", "code decomposition", "step through",
            "trace execution", "follow the code", "code flow",
            "control flow", "data flow", "call stack",
            "breakpoints", "debugger", "line by line",
            "inspect", "variable state", "heap", "stack"
        ],

        # Pre-validation vibes
        "self_debugging": [
            "test before showing", "mental execution", "trace through",
            "prevent bugs", "check my work", "simulate execution",
            "off by one", "edge case check", "dry run",
            "desk check", "code review myself", "self review",
            "before committing", "sanity test", "smoke test",
            "manual testing", "walkthrough", "trace",
            "pre-flight", "double check logic"
        ],

        # TDD vibes
        "tdd_prompting": [
            "test first", "write tests", "TDD", "test driven",
            "unit tests", "edge cases", "test coverage",
            "red green refactor", "tests then code", "jest",
            "pytest", "mocha", "junit", "testing framework",
            "mock", "stub", "spy", "fixture",
            "assertion", "expect", "should", "test case",
            "integration test", "e2e test", "selenium", "cypress", "playwright"
        ],

        # Reverse debugging vibes
        "reverse_cot": [
            "wrong output", "expected vs actual", "why different output",
            "output delta", "silent bug", "calculation error",
            "backwards debugging", "work backward", "reverse engineer bug",
            "expected X got Y", "off by", "incorrect result",
            "wrong answer", "bad output", "unexpected result",
            "diff", "mismatch", "discrepancy",
            "log analysis", "traceback analysis"
        ],

        # Socratic vibes
        "rubber_duck": [
            "explain to me", "walk me through", "ask me questions",
            "guide me", "help me think", "rubber duck",
            "Socratic method", "lead me to answer", "questioning approach",
            "talk it through", "think aloud", "verbalize",
            "explain my thinking", "work through it", "reasoning",
            "pair program", "code buddy", "sounding board"
        ],

        # Tool use vibes
        "react": [
            "use tools", "multi-step", "action reasoning",
            "tool use", "step and observe", "interact with",
            "reasoning acting", "ReAct pattern", "observe results",
            "API calls", "external tools", "shell commands",
            "file operations", "database queries", "web requests",
            "scrape", "curl", "wget", "grep", "find"
        ],

        # Learning vibes
        "reflexion": [
            "learn from mistakes", "retry", "failed attempt",
            "try again", "reflect on", "what went wrong",
            "iterative learning", "self-evaluation", "memory-based",
            "previous attempt", "last time", "improve on",
            "lessons learned", "retrospective", "post-mortem",
            "root cause analysis", "RCA", "incident report"
        ],

        # Quality vibes
        "self_refine": [
            "improve quality", "polish", "refine",
            "make it better", "iterative improvement", "critique and improve",
            "self-critique", "refinement loop", "quality pass",
            "clean up", "tighten up", "optimize",
            "beautify", "format", "lint", "prettier",
            "black", "isort", "ruff", "flake8", "pylint"
        ],

        # Bottom-up vibes
        "least_to_most": [
            "atomic functions", "dependency graph", "bottom up",
            "layered", "base functions first", "decompose completely",
            "building blocks", "hierarchical build", "least dependent first",
            "utils first", "helpers", "primitives",
            "foundation", "core functions", "base layer",
            "component library", "atoms", "molecules", "organisms"
        ],

        # Comparison vibes
        "comparative_arch": [
            "compare approaches", "readability vs performance", "trade-offs",
            "multiple solutions", "which is faster", "optimize for",
            "different versions", "performance vs memory", "three approaches",
            "show me options", "alternatives", "variations",
            "different ways", "multiple implementations", "compare",
            "benchmark results", "A/B test", "matrix"
        ],

        # Planning vibes
        "plan_and_solve": [
            "plan first", "think before coding", "explicit plan",
            "strategy", "outline approach", "plan then execute",
            "methodical", "step by step plan", "planning phase",
            "roadmap", "game plan", "action plan",
            "checklist", "todo list", "phases", "milestones",
            "technical spec", "design document", "RFC"
        ],

        # Security audit vibes
        "red_team": [
            "security audit", "vulnerabilities", "pen test",
            "security review", "find exploits", "attack vectors",
            "OWASP", "SQLi", "XSS", "security threats", "hack this",
            "injection", "CSRF", "SSRF", "RCE",
            "privilege escalation", "authentication bypass",
            "broken access control", "sensitive data exposure",
            "data breach", "leak", "encryption", "hashing", "salt"
        ],

        # State management vibes
        "state_machine": [
            "state machine", "FSM", "states and transitions",
            "workflow", "state diagram", "UI states",
            "game states", "state management", "transitions",
            "Redux", "Zustand", "MobX", "Vuex",
            "finite automata", "statechart", "xstate",
            "loading states", "error states", "success states",
            "hydration", "client state", "server state"
        ],

        # Basic reasoning vibes
        "chain_of_thought": [
            "think step by step", "reason through", "work through",
            "logical steps", "step by step", "reasoning chain",
            "think carefully", "show your work", "explicit reasoning",
            "break it down", "one step at a time", "systematically",
            "logically", "methodically", "carefully",
            "first", "then", "finally", "consequently"
        ],

        # Competitive programming vibes
        "alphacodium": [
            "competitive programming", "code contest", "algorithm challenge",
            "iterative code", "test-based", "multi-stage",
            "code generation", "contest problem", "leetcode",
            "hackerrank", "codeforces", "topcoder", "advent of code",
            "interview question", "coding interview", "whiteboard",
            "DSA", "data structures and algorithms",
            "dynamic programming", "graph theory", "greedy algorithm"
        ],

        # Modular vibes
        "codechain": [
            "modular code", "sub-modules", "self-revision",
            "incremental", "chain revisions", "module by module",
            "component based", "build incrementally", "refine modules",
            "microservices", "packages", "libraries",
            "separation of concerns", "single responsibility",
            "interface segregation", "dependency injection"
        ],

        # Evolutionary vibes
        "evol_instruct": [
            "evolve solution", "add constraints", "increase complexity",
            "challenging problem", "constraint-based", "evolutionary",
            "harder version", "more constraints", "complex requirements",
            "additional requirements", "scope creep", "feature creep",
            "extend", "enhance", "augment", "expand",
            "version 2", "next iteration", "advanced features"
        ],

        # Iteration vibes
        "llmloop": [
            "feedback loop", "iterate until", "compile and fix",
            "test loop", "automated testing", "quality assurance",
            "production ready", "lint and fix", "keep iterating",
            "CI/CD", "build pipeline", "automated checks",
            "pre-commit hooks", "continuous improvement",
            "nightly build", "regression testing"
        ],

        # Project integration vibes
        "procoder": [
            "compiler feedback", "project level", "codebase integration",
            "API usage", "large project", "integrate with",
            "project context", "compiler errors", "fix imports",
            "type errors", "typescript", "mypy", "pylint",
            "ESLint errors", "build errors", "dependency issues",
            "linker error", "module not found", "circular import"
        ],

        # High-stakes vibes
        "recode": [
            "multiple candidates", "cross validate", "CFG debugging",
            "control flow", "reliable code", "high stakes",
            "validate candidates", "majority voting", "robust solution",
            "mission critical", "production critical", "can't fail",
            "financial", "healthcare", "safety critical",
            "consensus", "redundancy", "fault tolerant",
            "zero downtime", "utility class", "nuclear"
        ],

        # =========================================================================
        # VERIFICATION FRAMEWORKS (2026 Expansion)
        # =========================================================================

        # Multi-sample voting vibes
        "self_consistency": [
            "multiple answers", "vote", "consensus", "majority",
            "check multiple times", "sample answers", "consistency check",
            "are you sure", "double check", "verify answer",
            "ambiguous", "uncertain", "could be either",
            "tricky logic", "multiple plausible", "which is right"
        ],

        # Sub-question vibes
        "self_ask": [
            "break down", "sub-questions", "what do I need to know",
            "before I can answer", "first need to understand",
            "unclear requirements", "missing info", "need clarification",
            "what questions should I ask", "decompose the problem",
            "prerequisite knowledge", "dependencies between parts"
        ],

        # Rephrase vibes
        "rar": [
            "rephrase", "clarify the question", "what exactly",
            "be more specific", "unclear request", "vague",
            "what do you mean", "interpret as", "restate",
            "poorly written", "confusing request", "ambiguous ask"
        ],

        # Verify-edit vibes
        "verify_and_edit": [
            "verify claims", "fact check", "edit only wrong parts",
            "surgical fix", "minimal changes", "preserve good parts",
            "check each claim", "verify then edit", "patch specific",
            "don't rewrite everything", "targeted fix", "precision edit"
        ],

        # Research-revise vibes
        "rarr": [
            "evidence based", "prove it", "cite sources", "grounded",
            "research first", "find evidence", "back it up",
            "documentation says", "according to docs", "verified by",
            "need proof", "show me where", "source please"
        ],

        # Hallucination check vibes
        "selfcheckgpt": [
            "am I hallucinating", "is this real", "sanity check",
            "might be wrong", "not sure if true", "verify accuracy",
            "could be making this up", "confidence check",
            "before I trust this", "credibility check",
            "final check", "pre-flight", "gate check"
        ],

        # Metamorphic test vibes
        "metaqa": [
            "test with variations", "edge cases", "what if slightly different",
            "invariants", "should work for all", "consistency across",
            "metamorphic", "transform and check", "robustness test",
            "works for this but not that", "brittle", "fragile logic"
        ],

        # RAG assessment vibes
        "ragas": [
            "evaluate retrieval", "RAG quality", "were sources relevant",
            "faithfulness", "did it use sources", "grounded in docs",
            "retrieval quality", "chunk relevance", "answer quality",
            "RAG pipeline", "evaluate RAG", "retrieval assessment"
        ],

        # =========================================================================
        # AGENT FRAMEWORKS (2026 Expansion)
        # =========================================================================

        # Plan-execute vibes
        "rewoo": [
            "plan first then execute", "tool schedule", "batch tools",
            "reasoning without observation", "plan once", "efficient tools",
            "tool-free plan", "expected observations", "tool budget",
            "minimize tool calls", "strategic tools", "planned execution"
        ],

        # Tree search agent vibes
        "lats": [
            "action tree", "multiple paths", "backtrack if fails",
            "tree search", "action sequences", "branch and bound",
            "rollback possible", "try different paths", "agent search",
            "strategic agent", "action planning", "branch exploration"
        ],

        # Modular agent vibes
        "mrkl": [
            "specialized modules", "security module", "perf module",
            "modular reasoning", "route to expert", "different perspectives",
            "orchestrate modules", "domain experts", "specialized handling",
            "big system", "mixed domains", "module routing"
        ],

        # SWE execution vibes
        "swe_agent": [
            "make tests pass", "CI green", "fix build",
            "inspect repo", "multi-file fix", "iterate until passing",
            "run tests", "check lint", "verify build",
            "software engineering", "repo workflow", "CI/CD fix"
        ],

        # Tool policy vibes
        "toolformer": [
            "when to use tools", "tool decision", "justify tool use",
            "tool policy", "smart tools", "efficient tool use",
            "tool selection", "optimize tool calls", "tool strategy",
            "when is tool needed", "tool cost benefit"
        ],

        # =========================================================================
        # RAG FRAMEWORKS (2026 Expansion)
        # =========================================================================

        # Self-triggered retrieval vibes
        "self_rag": [
            "retrieve when needed", "selective retrieval", "confidence based",
            "not sure need to check", "maybe retrieve", "targeted search",
            "retrieve for uncertain", "gap-driven retrieval",
            "don't retrieve everything", "smart retrieval"
        ],

        # Hypothetical doc vibes
        "hyde": [
            "hypothetical document", "ideal answer", "what would doc say",
            "fuzzy query", "vague search", "broad search",
            "unclear what to search", "improve retrieval",
            "query expansion", "semantic search improvement"
        ],

        # Multi-query fusion vibes
        "rag_fusion": [
            "multiple queries", "query variations", "fuse results",
            "rank fusion", "better recall", "diverse queries",
            "query diversity", "combine searches", "aggregate results",
            "comprehensive search", "thorough retrieval"
        ],

        # Hierarchical retrieval vibes
        "raptor": [
            "hierarchical search", "summary first", "drill down",
            "abstraction levels", "overview then details",
            "huge document", "long document", "large codebase",
            "lost in chunks", "tree retrieval", "multi-level"
        ],

        # Entity-relation vibes
        "graphrag": [
            "entity relations", "dependency graph", "how things connect",
            "module relationships", "call graph", "data flow",
            "impact analysis", "blast radius", "what depends on",
            "architecture map", "relation extraction", "knowledge graph"
        ],

        # =========================================================================
        # CODE FRAMEWORKS (2026 Expansion - Additional)
        # =========================================================================

        # Program-aided vibes
        "pal": [
            "code to reason", "compute answer", "executable logic",
            "algorithm as code", "run to verify", "code substrate",
            "program for answer", "computation", "calculate with code",
            "numeric reasoning", "code-based math"
        ],

        # Scratchpad vibes
        "scratchpads": [
            "working notes", "scratch space", "intermediate work",
            "track state", "multi-step notes", "organized thinking",
            "structured notes", "work area", "keep track",
            "progressive notes", "step tracker"
        ],

        # =========================================================================
        # CODE FRAMEWORKS (2026 Expansion - Part 2)
        # =========================================================================

        # Compositional code vibes
        "parsel": [
            "compositional", "dependency graph", "function specs",
            "decompose into functions", "build from specs", "spec to code",
            "natural language specs", "function dependencies", "bottom up build",
            "break into functions", "modular from spec", "compose functions",
            "spec driven", "from requirements", "hierarchical functions",
            "dependency order", "build order", "layered implementation"
        ],

        # Documentation-driven vibes
        "docprompting": [
            "from documentation", "follow the docs", "docs say",
            "according to documentation", "api docs", "official docs",
            "documentation example", "doc-driven", "read the docs",
            "rtfm", "manual says", "reference docs", "usage example",
            "as documented", "per documentation", "library docs",
            "sdk documentation", "api reference", "doc examples"
        ]
    }
    
    def __init__(self):
        self._compiled_patterns = {
            task_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for task_type, patterns in self.PATTERNS.items()
        }

    # ==========================================================================
    # HIERARCHICAL ROUTING - Stage 1: Category Selection
    # ==========================================================================

    def _route_to_category(self, query: str) -> Tuple[str, float]:
        """
        Fast first-stage routing to a category.
        Returns (category_name, confidence_score).
        """
        query_lower = query.lower()
        scores = {}

        for category, vibes in self.CATEGORY_VIBES.items():
            score = 0.0
            for vibe in vibes:
                if vibe in query_lower:
                    # Weight by phrase length
                    word_count = len(vibe.split())
                    score += word_count if word_count >= 2 else 0.5
            if score > 0:
                scores[category] = score

        if not scores:
            return "exploration", 0.3  # Default fallback

        best = max(scores, key=scores.get)
        confidence = min(scores[best] / 5.0, 1.0)  # Normalize to 0-1
        return best, confidence

    # ==========================================================================
    # HIERARCHICAL ROUTING - Stage 2: Specialist Agent Selection
    # ==========================================================================

    def _get_specialist_prompt(self, category: str, query: str, context: str = "") -> str:
        """
        Generate a specialist agent prompt for framework selection within a category.
        The specialist can recommend single frameworks or chains.
        """
        cat_info = self.CATEGORIES.get(category, self.CATEGORIES["exploration"])
        frameworks = cat_info["frameworks"]
        chain_patterns = cat_info.get("chain_patterns", {})
        specialist = cat_info["specialist"]

        # Build framework descriptions for this category
        fw_descriptions = []
        for fw in frameworks:
            desc = self.FRAMEWORKS.get(fw, "Unknown framework")
            vibes = self.VIBE_DICTIONARY.get(fw, [])[:3]
            fw_descriptions.append(f"  - {fw}: {desc} (vibes: {', '.join(vibes)})")

        # Build chain pattern descriptions
        chain_descriptions = []
        for pattern_name, chain in chain_patterns.items():
            chain_descriptions.append(f"  - {pattern_name}: {' → '.join(chain)}")

        return f"""You are the **{specialist}** - a specialist agent for {cat_info['description']}.

TASK: {query}
{f'CONTEXT: {context}' if context else ''}

## Available Frameworks (pick 1 or chain multiple):
{chr(10).join(fw_descriptions)}

## Pre-defined Chains (for complex tasks):
{chr(10).join(chain_descriptions) if chain_descriptions else '  (none - single framework recommended)'}

## Instructions:
1. Analyze the task complexity
2. For SIMPLE tasks: recommend a single framework
3. For COMPLEX tasks: recommend a chain of 2-4 frameworks

Respond in this format:
COMPLEXITY: simple|complex
FRAMEWORKS: framework1 [→ framework2 → framework3]
REASONING: Brief explanation of your choice
"""

    async def _select_with_specialist(
        self,
        category: str,
        query: str,
        code_snippet: Optional[str] = None,
        ide_context: Optional[str] = None
    ) -> Tuple[List[str], str]:
        """
        Use category specialist agent to select framework(s).
        Returns (list_of_frameworks, reasoning).
        """
        cat_info = self.CATEGORIES.get(category, self.CATEGORIES["exploration"])
        frameworks = cat_info["frameworks"]
        chain_patterns = cat_info.get("chain_patterns", {})

        # Build context
        context = ""
        if code_snippet:
            context += f"Code:\n{code_snippet[:500]}..."
        if ide_context:
            context += f"\nIDE Context: {ide_context}"

        # Check for chain pattern match first (fast path)
        query_lower = query.lower()
        for pattern_name, chain in chain_patterns.items():
            pattern_words = pattern_name.replace("_", " ").split()
            if all(word in query_lower for word in pattern_words):
                return chain, f"Matched chain pattern: {pattern_name}"

        # Try specialist agent for nuanced selection
        try:
            from ..langchain_integration import get_chat_model

            prompt = self._get_specialist_prompt(category, query, context)
            llm = get_chat_model("fast")
            response = await llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            # Parse response
            frameworks_line = ""
            reasoning = ""
            for line in response_text.split("\n"):
                if line.startswith("FRAMEWORKS:"):
                    frameworks_line = line.replace("FRAMEWORKS:", "").strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()

            if frameworks_line:
                # Parse chain: "fw1 → fw2 → fw3" or "fw1"
                selected = [fw.strip() for fw in frameworks_line.replace("→", ",").replace("->", ",").split(",")]
                selected = [fw for fw in selected if fw in self.FRAMEWORKS]
                if selected:
                    return selected, reasoning or f"Specialist selected: {frameworks_line}"

        except Exception as e:
            pass  # Fall through to default

        # Default: pick first framework in category
        return [frameworks[0]], f"Default selection for {category}"

    async def select_framework_chain(
        self,
        query: str,
        code_snippet: Optional[str] = None,
        ide_context: Optional[str] = None
    ) -> Tuple[List[str], str, str]:
        """
        Hierarchical routing with chain support.

        Returns: (framework_chain, reasoning, category)
        - framework_chain: List of 1+ frameworks to execute in sequence
        - reasoning: Explanation of selection
        - category: The matched category
        """
        # Stage 1: Route to category
        category, confidence = self._route_to_category(query)

        # Stage 2: Specialist agent selection
        if confidence < 0.5:
            # Low confidence - let specialist decide more carefully
            chain, reasoning = await self._select_with_specialist(
                category, query, code_snippet, ide_context
            )
        else:
            # High confidence - check for quick vibe match within category
            cat_frameworks = self.CATEGORIES[category]["frameworks"]
            vibe_match = self._check_vibe_dictionary(query)

            if vibe_match and vibe_match in cat_frameworks:
                chain = [vibe_match]
                reasoning = f"Vibe match in {category}: {query[:30]}..."
            else:
                # Use specialist for nuanced selection
                chain, reasoning = await self._select_with_specialist(
                    category, query, code_snippet, ide_context
                )

        return chain, reasoning, category

    async def auto_select_framework(
        self,
        query: str,
        code_snippet: Optional[str] = None,
        ide_context: Optional[str] = None
    ) -> tuple[str, str]:
        """
        AI-powered framework selection using hierarchical routing.

        Uses two-stage approach:
        1. Route to category (fast pattern matching)
        2. Specialist agent picks framework(s) within category

        Returns: (first_framework, reasoning)
        For chain access, use select_framework_chain() instead.
        """
        try:
            chain, reasoning, category = await self.select_framework_chain(
                query, code_snippet, ide_context
            )
            # Return first framework for backwards compatibility
            # Full chain is available via select_framework_chain()
            if chain:
                if len(chain) > 1:
                    reasoning = f"[Chain: {' → '.join(chain)}] {reasoning}"
                return chain[0], reasoning
        except Exception as e:
            pass  # Fall through to legacy method

        # Legacy fallback: direct vibe matching
        vibe_match = self._check_vibe_dictionary(query)
        if vibe_match:
            return vibe_match, f"Matched vibe: {query[:50]}..."

        return "self_discover", "Fallback: routing failed, using self_discover."

    def _extract_framework(self, response: str) -> Optional[str]:
        """
        Extract framework name from LLM response using regex/string matching.
        Fallback when structured parser fails.
        """
        response_lower = response.lower()

        # Look for framework names in the response
        for framework in self.FRAMEWORKS:
            # Check for exact matches or common patterns like "use X" or "select X"
            if framework in response_lower:
                return framework

        return None

    def _check_vibe_dictionary(self, query: str) -> Optional[str]:
        """
        Quick check against vibe dictionary with weighted scoring.

        Scoring rules:
        - Longer phrase matches score higher (phrase length * 2)
        - Multiple matches accumulate
        - Multi-word phrases get bonus weight

        Returns framework name if strong match found.
        """
        query_lower = query.lower()

        # Score each framework by vibe matches with weighted scoring
        scores = {}
        for framework, vibes in self.VIBE_DICTIONARY.items():
            total_score = 0.0
            matched_vibes = []

            for vibe in vibes:
                if vibe in query_lower:
                    # Weight by phrase length - longer phrases are more specific
                    word_count = len(vibe.split())
                    # Multi-word phrases get bonus (2x for 2 words, 3x for 3+ words)
                    weight = word_count if word_count >= 2 else 0.5
                    total_score += weight
                    matched_vibes.append(vibe)

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
    
    def _heuristic_select(self, query: str, code_snippet: Optional[str] = None) -> str:
        """Fast heuristic selection (fallback)."""
        combined = query + (" " + code_snippet if code_snippet else "")
        
        scores = {}
        for task_type, patterns in self._compiled_patterns.items():
            scores[task_type] = sum(1 for p in patterns if p.search(combined))
        
        if max(scores.values()) > 0:
            task_type = max(scores, key=scores.get)
            return self.HEURISTIC_MAP.get(task_type, "self_discover")
        
        return "self_discover"
    
    def estimate_complexity(
        self,
        query: str,
        code_snippet: Optional[str] = None,
        file_list: Optional[list[str]] = None
    ) -> float:
        """Estimate task complexity on 0-1 scale."""
        complexity = 0.3
        
        if len(query.split()) > 50:
            complexity += 0.15
        if len(query.split()) > 100:
            complexity += 0.1
        
        if code_snippet:
            lines = code_snippet.count('\n') + 1
            if lines > 50:
                complexity += 0.1
            if lines > 200:
                complexity += 0.15
        
        if file_list and len(file_list) > 5:
            complexity += 0.1
        
        indicators = [r"complex", r"difficult", r"tricky", r"interdependent", r"legacy", r"distributed"]
        for ind in indicators:
            if re.search(ind, query, re.IGNORECASE):
                complexity += 0.05
        
        return min(complexity, 1.0)
    
    async def route(self, state: GraphState, use_ai: bool = True) -> GraphState:
        """
        Route the task to the best framework(s).

        Supports framework chaining for complex tasks.

        Args:
            state: Current graph state
            use_ai: If True, use hierarchical AI routing (default). If False, use heuristics.
        """
        framework_chain = []
        category = "unknown"
        reason = ""

        # Check for explicit preference first
        if state.get("preferred_framework") and state["preferred_framework"] in self.FRAMEWORKS:
            framework_chain = [state["preferred_framework"]]
            reason = "User specified"
        elif use_ai:
            # Hierarchical AI-powered selection with chain support
            try:
                chain, reason, category = await self.select_framework_chain(
                    state["query"],
                    state.get("code_snippet"),
                    state.get("ide_context")
                )
                framework_chain = chain
            except Exception as e:
                # Fallback to single framework
                framework, reason = await self.auto_select_framework(
                    state["query"],
                    state.get("code_snippet"),
                    state.get("ide_context")
                )
                framework_chain = [framework]
        else:
            # Heuristic fallback (single framework only)
            framework = self._heuristic_select(state["query"], state.get("code_snippet"))
            framework_chain = [framework]
            reason = "Heuristic selection"

        complexity = self.estimate_complexity(
            state["query"],
            state.get("code_snippet"),
            state.get("file_list")
        )

        # Update state with chain support
        state["selected_framework"] = framework_chain[0] if framework_chain else "self_discover"
        state["framework_chain"] = framework_chain  # Full chain for pipeline execution
        state["complexity_estimate"] = complexity
        state["task_type"] = self._infer_task_type(framework_chain[0] if framework_chain else "self_discover")
        state["routing_category"] = category

        # Inject Past Learnings (Episodic Memory)
        try:
            from ..collection_manager import get_collection_manager
            cm = get_collection_manager()
            learnings = cm.search_learnings(state["query"], k=3)
            if learnings:
                state["episodic_memory"] = learnings
        except Exception as e:
            import structlog
            logger = structlog.get_logger("router")
            logger.warning("episodic_memory_search_failed", error=str(e), query=state["query"][:100])

        state["reasoning_steps"].append({
            "step": "routing",
            "framework": framework_chain[0] if framework_chain else "self_discover",
            "framework_chain": framework_chain,
            "category": category,
            "reason": reason,
            "complexity": complexity,
            "method": "hierarchical_ai" if use_ai else "heuristic"
        })

        return state
    
    def _infer_task_type(self, framework: str) -> str:
        """Infer task type from chosen framework."""
        type_map = {
            # Original 20 frameworks
            "active_inference": "debug",
            "mcts_rstar": "debug",
            "graph_of_thoughts": "refactor",
            "everything_of_thought": "refactor",
            "reason_flux": "architecture",
            "multi_agent_debate": "architecture",
            "tree_of_thoughts": "algorithm",
            "skeleton_of_thought": "docs",
            "buffer_of_thoughts": "docs",
            "critic": "api",
            "chain_of_verification": "security",
            "program_of_thoughts": "compute",
            "self_discover": "exploration",
            "coala": "context",
            "chain_of_note": "research",
            "step_back": "analysis",
            "analogical": "creative",
            "adaptive_injection": "adaptive",
            "re2": "requirements",
            "system1": "quick",
            # 2026 Edition frameworks
            "chain_of_code": "debug",
            "self_debugging": "debug",
            "reverse_cot": "debug",
            "tdd_prompting": "test",
            "rubber_duck": "debug",
            "react": "agent",
            "reflexion": "iterative",
            "self_refine": "quality",
            "least_to_most": "architecture",
            "comparative_arch": "architecture",
            "plan_and_solve": "planning",
            "red_team": "security",
            "state_machine": "architecture",
            "chain_of_thought": "reasoning",
            # Additional coding frameworks
            "alphacodium": "competitive",
            "codechain": "code_gen",
            "evol_instruct": "code_gen",
            "llmloop": "ci_cd",
            "procoder": "code_gen",
            "recode": "code_gen",
            "pal": "compute",
            "scratchpads": "reasoning",
            "parsel": "code_gen",
            "docprompting": "code_gen",
            # Verification frameworks
            "self_consistency": "verification",
            "self_ask": "verification",
            "rar": "verification",
            "verify_and_edit": "verification",
            "rarr": "verification",
            "selfcheckgpt": "verification",
            "metaqa": "verification",
            "ragas": "rag",
            # Agent frameworks
            "rewoo": "agent",
            "lats": "agent",
            "mrkl": "agent",
            "swe_agent": "agent",
            "toolformer": "agent",
            # RAG frameworks
            "self_rag": "rag",
            "hyde": "rag",
            "rag_fusion": "rag",
            "raptor": "rag",
            "graphrag": "rag",
        }
        return type_map.get(framework, "unknown")
    
    def get_framework_info(self, framework: str) -> dict:
        """Get metadata about a framework."""
        INFO = {
            "active_inference": {
                "name": "Active Inference",
                "category": "iterative",
                "description": "Debugging loop with hypothesis testing",
                "best_for": ["debugging", "error analysis", "root cause"],
                "complexity": "medium"
            },
            "graph_of_thoughts": {
                "name": "Graph of Thoughts",
                "category": "search",
                "description": "Non-linear thinking with merge operations",
                "best_for": ["refactoring", "code restructuring"],
                "complexity": "high"
            },
            "reason_flux": {
                "name": "ReasonFlux",
                "category": "strategy",
                "description": "Hierarchical planning with template expansion",
                "best_for": ["architecture", "system design"],
                "complexity": "high"
            },
            "tree_of_thoughts": {
                "name": "Tree of Thoughts",
                "category": "search",
                "description": "BFS/DFS exploration of solution space",
                "best_for": ["algorithms", "optimization"],
                "complexity": "medium"
            },
            "skeleton_of_thought": {
                "name": "Skeleton of Thought",
                "category": "fast",
                "description": "Outline-first parallel expansion",
                "best_for": ["documentation", "boilerplate"],
                "complexity": "low"
            },
            "critic": {
                "name": "CRITIC",
                "category": "code",
                "description": "External tool verification",
                "best_for": ["API usage", "library integration"],
                "complexity": "medium"
            },
            "chain_of_verification": {
                "name": "Chain of Verification",
                "category": "code",
                "description": "Draft-verify-patch cycle",
                "best_for": ["security", "code review"],
                "complexity": "medium"
            },
            "program_of_thoughts": {
                "name": "Program of Thoughts",
                "category": "code",
                "description": "Generate executable code",
                "best_for": ["math", "data processing"],
                "complexity": "medium"
            },
            "self_discover": {
                "name": "Self-Discover",
                "category": "strategy",
                "description": "Compose custom reasoning",
                "best_for": ["novel problems", "exploration"],
                "complexity": "high"
            },
            "mcts_rstar": {
                "name": "rStar-Code MCTS",
                "category": "search",
                "description": "Monte Carlo Tree Search",
                "best_for": ["complex bugs", "optimization"],
                "complexity": "high"
            },
            "multi_agent_debate": {
                "name": "Multi-Agent Debate",
                "category": "iterative",
                "description": "Proponent vs Critic",
                "best_for": ["design decisions", "trade-offs"],
                "complexity": "high"
            },
            "everything_of_thought": {
                "name": "Everything of Thought",
                "category": "search",
                "description": "MCTS + fast generation",
                "best_for": ["complex refactoring", "migration"],
                "complexity": "high"
            },
            "buffer_of_thoughts": {
                "name": "Buffer of Thoughts",
                "category": "strategy",
                "description": "Template retrieval",
                "best_for": ["repetitive tasks", "patterns"],
                "complexity": "low"
            },
            "coala": {
                "name": "CoALA",
                "category": "strategy",
                "description": "Cognitive architecture with memory",
                "best_for": ["long context", "multi-file"],
                "complexity": "high"
            },
            "chain_of_note": {
                "name": "Chain of Note",
                "category": "context",
                "description": "Research with note-taking",
                "best_for": ["research", "learning"],
                "complexity": "medium"
            },
            "step_back": {
                "name": "Step-Back Prompting",
                "category": "context",
                "description": "Abstraction first",
                "best_for": ["performance", "complexity analysis"],
                "complexity": "medium"
            },
            "analogical": {
                "name": "Analogical Prompting",
                "category": "context",
                "description": "Analogy-based solving",
                "best_for": ["creative solutions", "patterns"],
                "complexity": "medium"
            },
            "adaptive_injection": {
                "name": "Adaptive Injection",
                "category": "iterative",
                "description": "Dynamic thinking depth",
                "best_for": ["variable complexity"],
                "complexity": "medium"
            },
            "re2": {
                "name": "Re-Reading (RE2)",
                "category": "iterative",
                "description": "Two-pass processing",
                "best_for": ["complex specs", "requirements"],
                "complexity": "medium"
            },
            "system1": {
                "name": "System1 Fast",
                "category": "fast",
                "description": "Quick heuristic responses",
                "best_for": ["simple queries", "quick fixes"],
                "complexity": "low"
            },
            # New frameworks (2026 Edition)
            "chain_of_code": {
                "name": "Chain-of-Code",
                "category": "code",
                "description": "Code-based problem decomposition",
                "best_for": ["logic puzzles", "algorithmic debugging"],
                "complexity": "medium"
            },
            "self_debugging": {
                "name": "Self-Debugging",
                "category": "code",
                "description": "Mental execution before presenting",
                "best_for": ["preventing bugs", "edge case handling"],
                "complexity": "medium"
            },
            "tdd_prompting": {
                "name": "TDD Prompting",
                "category": "code",
                "description": "Test-first development",
                "best_for": ["new features", "edge case coverage"],
                "complexity": "medium"
            },
            "reverse_cot": {
                "name": "Reverse Chain-of-Thought",
                "category": "code",
                "description": "Backward reasoning from output delta",
                "best_for": ["silent bugs", "wrong outputs"],
                "complexity": "medium"
            },
            "rubber_duck": {
                "name": "Rubber Duck Debugging",
                "category": "iterative",
                "description": "Socratic questioning",
                "best_for": ["architectural issues", "blind spots"],
                "complexity": "medium"
            },
            "react": {
                "name": "ReAct",
                "category": "iterative",
                "description": "Reasoning + Acting with tools",
                "best_for": ["multi-step tasks", "tool use"],
                "complexity": "high"
            },
            "reflexion": {
                "name": "Reflexion",
                "category": "iterative",
                "description": "Self-evaluation with memory",
                "best_for": ["learning from failures", "iterative improvement"],
                "complexity": "high"
            },
            "self_refine": {
                "name": "Self-Refine",
                "category": "iterative",
                "description": "Iterative self-critique",
                "best_for": ["code quality", "documentation"],
                "complexity": "medium"
            },
            "least_to_most": {
                "name": "Least-to-Most Decomposition",
                "category": "strategy",
                "description": "Bottom-up atomic decomposition",
                "best_for": ["complex systems", "monolith refactoring"],
                "complexity": "high"
            },
            "comparative_arch": {
                "name": "Comparative Architecture",
                "category": "strategy",
                "description": "Multi-approach comparison",
                "best_for": ["optimization", "architecture decisions"],
                "complexity": "high"
            },
            "plan_and_solve": {
                "name": "Plan-and-Solve",
                "category": "strategy",
                "description": "Explicit planning before execution",
                "best_for": ["complex features", "methodical development"],
                "complexity": "medium"
            },
            "red_team": {
                "name": "Red-Teaming",
                "category": "context",
                "description": "Adversarial security analysis",
                "best_for": ["security audits", "vulnerability scanning"],
                "complexity": "high"
            },
            "state_machine": {
                "name": "State-Machine Reasoning",
                "category": "context",
                "description": "Formal FSM design",
                "best_for": ["UI logic", "workflow systems"],
                "complexity": "medium"
            },
            "chain_of_thought": {
                "name": "Chain-of-Thought",
                "category": "context",
                "description": "Step-by-step reasoning",
                "best_for": ["complex reasoning", "logical deduction"],
                "complexity": "low"
            },
            # Additional coding frameworks (2026 expansion)
            "alphacodium": {
                "name": "AlphaCodium",
                "category": "code",
                "description": "Test-based multi-stage iterative code generation",
                "best_for": ["competitive programming", "complex algorithms"],
                "complexity": "high"
            },
            "codechain": {
                "name": "CodeChain",
                "category": "code",
                "description": "Chain of self-revisions guided by sub-modules",
                "best_for": ["modular code generation", "incremental refinement"],
                "complexity": "high"
            },
            "evol_instruct": {
                "name": "Evol-Instruct",
                "category": "code",
                "description": "Evolutionary instruction complexity for code",
                "best_for": ["challenging code problems", "constraint-based coding"],
                "complexity": "high"
            },
            "llmloop": {
                "name": "LLMLoop",
                "category": "code",
                "description": "Automated iterative feedback loops for code+tests",
                "best_for": ["code quality assurance", "production-ready code"],
                "complexity": "high"
            },
            "procoder": {
                "name": "ProCoder",
                "category": "code",
                "description": "Compiler-feedback-guided iterative refinement",
                "best_for": ["project-level code generation", "API usage"],
                "complexity": "high"
            },
            "recode": {
                "name": "RECODE",
                "category": "code",
                "description": "Multi-candidate validation with CFG-based debugging",
                "best_for": ["reliable code generation", "high-stakes code"],
                "complexity": "high"
            },
            # Verification frameworks (2026 Expansion)
            "self_consistency": {
                "name": "Self-Consistency",
                "category": "verification",
                "description": "Multi-sample voting for reliable answers",
                "best_for": ["ambiguous bugs", "tricky logic", "multiple plausible fixes"],
                "complexity": "medium"
            },
            "self_ask": {
                "name": "Self-Ask",
                "category": "verification",
                "description": "Sub-question decomposition before solving",
                "best_for": ["unclear tickets", "missing requirements", "multi-part debugging"],
                "complexity": "medium"
            },
            "rar": {
                "name": "Rephrase-and-Respond",
                "category": "verification",
                "description": "Clarify request before solving",
                "best_for": ["vague prompts", "ambiguous requirements"],
                "complexity": "low"
            },
            "verify_and_edit": {
                "name": "Verify-and-Edit",
                "category": "verification",
                "description": "Verify claims, edit only failures",
                "best_for": ["code review", "surgical edits", "implementation plans"],
                "complexity": "medium"
            },
            "rarr": {
                "name": "RARR",
                "category": "verification",
                "description": "Research, Augment, Revise loop",
                "best_for": ["evidence-based answers", "documentation", "prove-it requirements"],
                "complexity": "medium"
            },
            "selfcheckgpt": {
                "name": "SelfCheckGPT",
                "category": "verification",
                "description": "Hallucination detection via sampling",
                "best_for": ["high-stakes guidance", "unfamiliar libraries", "final gate"],
                "complexity": "medium"
            },
            "metaqa": {
                "name": "MetaQA",
                "category": "verification",
                "description": "Metamorphic testing for reasoning",
                "best_for": ["brittle reasoning", "edge cases", "policy consistency"],
                "complexity": "medium"
            },
            "ragas": {
                "name": "RAGAS",
                "category": "verification",
                "description": "RAG Assessment for retrieval quality",
                "best_for": ["RAG pipelines", "retrieval quality", "source grounding"],
                "complexity": "medium"
            },
            # Agent frameworks (2026 Expansion)
            "rewoo": {
                "name": "ReWOO",
                "category": "agent",
                "description": "Plan then execute with tools",
                "best_for": ["multi-step tasks", "cost control", "tool efficiency"],
                "complexity": "medium"
            },
            "lats": {
                "name": "LATS",
                "category": "agent",
                "description": "Tree search over action sequences",
                "best_for": ["complex repo changes", "multiple fix paths", "uncertain root cause"],
                "complexity": "high"
            },
            "mrkl": {
                "name": "MRKL",
                "category": "agent",
                "description": "Modular reasoning with specialized modules",
                "best_for": ["big systems", "mixed domains", "tool-rich setups"],
                "complexity": "high"
            },
            "swe_agent": {
                "name": "SWE-Agent",
                "category": "agent",
                "description": "Repo-first execution loop",
                "best_for": ["multi-file bugfixes", "CI failures", "make tests pass"],
                "complexity": "high"
            },
            "toolformer": {
                "name": "Toolformer",
                "category": "agent",
                "description": "Smart tool selection policy",
                "best_for": ["router logic", "preventing pointless calls", "tool efficiency"],
                "complexity": "medium"
            },
            # RAG frameworks (2026 Expansion)
            "self_rag": {
                "name": "Self-RAG",
                "category": "rag",
                "description": "Self-triggered selective retrieval",
                "best_for": ["mixed knowledge tasks", "large corpora", "targeted retrieval"],
                "complexity": "medium"
            },
            "hyde": {
                "name": "HyDE",
                "category": "rag",
                "description": "Hypothetical Document Embeddings",
                "best_for": ["fuzzy search", "unclear intent", "broad problems"],
                "complexity": "medium"
            },
            "rag_fusion": {
                "name": "RAG-Fusion",
                "category": "rag",
                "description": "Multi-query retrieval with rank fusion",
                "best_for": ["improving recall", "complex queries", "noisy corpora"],
                "complexity": "medium"
            },
            "raptor": {
                "name": "RAPTOR",
                "category": "rag",
                "description": "Hierarchical abstraction retrieval",
                "best_for": ["huge repos", "long docs", "monorepos"],
                "complexity": "high"
            },
            "graphrag": {
                "name": "GraphRAG",
                "category": "rag",
                "description": "Entity-relation grounding",
                "best_for": ["architecture questions", "module relationships", "impact analysis"],
                "complexity": "high"
            },
            # Additional code frameworks (2026 Expansion)
            "pal": {
                "name": "PAL",
                "category": "code",
                "description": "Program-Aided Language reasoning",
                "best_for": ["algorithms", "parsing", "numeric logic", "validation"],
                "complexity": "medium"
            },
            "scratchpads": {
                "name": "Scratchpads",
                "category": "code",
                "description": "Structured intermediate reasoning",
                "best_for": ["multi-step fixes", "multi-constraint reasoning", "state tracking"],
                "complexity": "low"
            },
            # Additional code frameworks (2026 Expansion - Part 2)
            "parsel": {
                "name": "Parsel",
                "category": "code",
                "description": "Compositional code generation from natural language specs",
                "best_for": ["complex functions", "dependency graphs", "spec-to-code", "modular systems"],
                "complexity": "high"
            },
            "docprompting": {
                "name": "DocPrompting",
                "category": "code",
                "description": "Documentation-driven code generation",
                "best_for": ["API usage", "library integration", "following docs", "correct usage"],
                "complexity": "medium"
            }
        }
        return INFO.get(framework, {
            "name": framework,
            "category": "unknown",
            "description": "Unknown framework",
            "best_for": [],
            "complexity": "unknown"
        })


# Global router instance
router = HyperRouter()
