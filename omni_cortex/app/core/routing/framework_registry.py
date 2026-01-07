"""
Framework Registry for Omni-Cortex

Contains all framework metadata, categories, and routing patterns.
This is the single source of truth for framework information.
"""

from typing import Dict, List, Any

from ..errors import FrameworkNotFoundError


# =============================================================================
# CATEGORY DEFINITIONS - First stage routing
# =============================================================================

CATEGORIES: Dict[str, Dict[str, Any]] = {
    "debug": {
        "description": "Bug hunting, error analysis, root cause investigation",
        "specialist": "Debug Detective",
        "frameworks": [
            "active_inference", "self_debugging", "reverse_cot",
            "mcts_rstar", "chain_of_code", "rubber_duck"
        ],
        "chain_patterns": {
            "complex_bug": ["self_ask", "active_inference", "verify_and_edit"],
            "silent_bug": ["reverse_cot", "self_debugging", "selfcheckgpt"],
            "flaky_test": ["active_inference", "tdd_prompting", "self_consistency"],
        }
    },
    "code_gen": {
        "description": "Writing new code, algorithms, implementations",
        "specialist": "Code Architect",
        "frameworks": [
            "alphacodium", "codechain", "parsel", "docprompting",
            "procoder", "recode", "pal", "program_of_thoughts",
            "evol_instruct", "llmloop", "tdd_prompting"
        ],
        "chain_patterns": {
            "complex_feature": ["plan_and_solve", "parsel", "tdd_prompting", "self_refine"],
            "api_integration": ["docprompting", "critic", "verify_and_edit"],
            "algorithm": ["step_back", "alphacodium", "self_debugging"],
        }
    },
    "refactor": {
        "description": "Code cleanup, restructuring, modernization",
        "specialist": "Refactor Surgeon",
        "frameworks": [
            "graph_of_thoughts", "everything_of_thought",
            "least_to_most", "codechain", "self_refine"
        ],
        "chain_patterns": {
            "major_rewrite": ["plan_and_solve", "graph_of_thoughts", "verify_and_edit"],
            "modular_extract": ["least_to_most", "parsel", "self_refine"],
            "legacy_cleanup": ["chain_of_note", "graph_of_thoughts", "tdd_prompting"],
        }
    },
    "architecture": {
        "description": "System design, planning, high-level decisions",
        "specialist": "System Architect",
        "frameworks": [
            "reason_flux", "plan_and_solve", "comparative_arch",
            "multi_agent_debate", "state_machine", "coala"
        ],
        "chain_patterns": {
            "new_system": ["reason_flux", "multi_agent_debate", "plan_and_solve"],
            "scale_decision": ["step_back", "comparative_arch", "verify_and_edit"],
            "workflow_design": ["state_machine", "plan_and_solve", "critic"],
        }
    },
    "verification": {
        "description": "Checking, validating, proving correctness",
        "specialist": "Verification Expert",
        "frameworks": [
            "chain_of_verification", "verify_and_edit", "self_consistency",
            "selfcheckgpt", "metaqa", "rarr", "red_team"
        ],
        "chain_patterns": {
            "security_audit": ["red_team", "chain_of_verification", "verify_and_edit"],
            "claim_check": ["self_ask", "rarr", "selfcheckgpt"],
            "code_review": ["chain_of_verification", "self_consistency", "verify_and_edit"],
        }
    },
    "agent": {
        "description": "Multi-step tasks, tool use, autonomous execution",
        "specialist": "Agent Orchestrator",
        "frameworks": [
            "swe_agent", "react", "rewoo", "lats",
            "mrkl", "toolformer", "reflexion"
        ],
        "chain_patterns": {
            "ci_fix": ["swe_agent", "tdd_prompting", "verify_and_edit"],
            "multi_file": ["coala", "swe_agent", "self_refine"],
            "tool_heavy": ["rewoo", "react", "reflexion"],
        }
    },
    "rag": {
        "description": "Retrieval, documentation, knowledge grounding",
        "specialist": "Knowledge Navigator",
        "frameworks": [
            "self_rag", "hyde", "rag_fusion", "raptor",
            "graphrag", "chain_of_note", "ragas"
        ],
        "chain_patterns": {
            "large_codebase": ["raptor", "graphrag", "chain_of_note"],
            "fuzzy_search": ["hyde", "rag_fusion", "rarr"],
            "dependency_map": ["graphrag", "least_to_most", "chain_of_note"],
        }
    },
    "exploration": {
        "description": "Novel problems, learning, creative solutions",
        "specialist": "Explorer",
        "frameworks": [
            "self_discover", "analogical", "buffer_of_thoughts",
            "adaptive_injection", "chain_of_thought", "step_back"
        ],
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
CATEGORY_VIBES: Dict[str, List[str]] = {
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

# All available frameworks with descriptions
FRAMEWORKS: Dict[str, str] = {
    "active_inference": "Debugging loop: hypothesis -> predict -> compare -> update. Best for nasty bugs and root cause analysis.",
    "graph_of_thoughts": "Non-linear thinking with merge/aggregate. Best for refactoring spaghetti code and restructuring.",
    "reason_flux": "Hierarchical planning: template -> expand -> refine. Best for architecture and system design.",
    "tree_of_thoughts": "BFS/DFS exploration of solutions. Best for algorithms and optimization problems.",
    "skeleton_of_thought": "Outline-first with parallel expansion. Best for docs, boilerplate, fast generation.",
    "critic": "External tool verification. Best for API usage validation and library integration.",
    "chain_of_verification": "Draft -> verify -> patch cycle. Best for security review and code validation.",
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
    "re2": "Two-pass: goals -> constraints. Best for complex specs and requirements.",
    "system1": "Fast heuristic, minimal thinking. Best for trivial quick fixes.",
    # 2026 Edition frameworks
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
    # Additional coding frameworks
    "alphacodium": "Test-based multi-stage iterative code generation. Best for competitive programming and complex algorithms.",
    "codechain": "Chain of self-revisions guided by sub-modules. Best for modular code generation and incremental refinement.",
    "evol_instruct": "Evolutionary instruction complexity for code. Best for challenging code problems and constraint-based coding.",
    "llmloop": "Automated iterative feedback loops for code+tests. Best for code quality assurance and production-ready code.",
    "procoder": "Compiler-feedback-guided iterative refinement. Best for project-level code generation and API usage.",
    "recode": "Multi-candidate validation with CFG-based debugging. Best for reliable code generation and high-stakes code.",
    # Verification frameworks
    "self_consistency": "Multi-sample voting for reliable answers. Best for ambiguous bugs and tricky logic.",
    "self_ask": "Sub-question decomposition before solving. Best for unclear tickets and missing requirements.",
    "rar": "Rephrase-and-Respond for clarity. Best for vague prompts and ambiguous requirements.",
    "verify_and_edit": "Verify claims then edit only failures. Best for surgical edits and code review.",
    "rarr": "Research, Augment, Revise loop. Best for evidence-based answers and documentation.",
    "selfcheckgpt": "Hallucination detection via sampling consistency. Best for high-stakes guidance.",
    "metaqa": "Metamorphic testing for reasoning reliability. Best for brittle reasoning and edge cases.",
    "ragas": "RAG Assessment for retrieval quality. Best for evaluating RAG pipelines.",
    # Agent frameworks
    "rewoo": "Reasoning Without Observation - plan then execute. Best for multi-step tasks with tools.",
    "lats": "Language Agent Tree Search over action sequences. Best for complex repo changes.",
    "mrkl": "Modular Reasoning with specialized modules. Best for big systems and mixed domains.",
    "swe_agent": "Repo-first execution loop. Best for multi-file bugfixes and CI failures.",
    "toolformer": "Smart tool selection policy. Best for preventing pointless tool calls.",
    # RAG frameworks
    "self_rag": "Self-triggered selective retrieval. Best for mixed knowledge tasks.",
    "hyde": "Hypothetical Document Embeddings for better retrieval. Best for fuzzy search.",
    "rag_fusion": "Multi-query retrieval with rank fusion. Best for improving recall.",
    "raptor": "Hierarchical abstraction retrieval. Best for huge repos and long docs.",
    "graphrag": "Entity-relation grounding for dependencies. Best for architecture questions.",
    # Additional code frameworks
    "pal": "Program-Aided Language - code as reasoning substrate. Best for algorithms and numeric logic.",
    "scratchpads": "Structured intermediate reasoning workspace. Best for multi-step fixes.",
    "parsel": "Compositional code generation from natural language specs. Builds dependency graph of functions.",
    "docprompting": "Documentation-driven code generation. Retrieves docs and examples to guide code generation."
}

# Heuristic patterns (fallback)
PATTERNS: Dict[str, List[str]] = {
    "debug": [r"bug", r"error", r"exception", r"crash", r"fix", r"broken", r"fails", r"null", r"undefined"],
    "refactor": [r"refactor", r"clean", r"reorganize", r"legacy", r"spaghetti", r"debt"],
    "architecture": [r"architect", r"design", r"structure", r"system", r"microservice", r"scale"],
    "algorithm": [r"algorithm", r"optimize", r"complexity", r"performance", r"efficient", r"sort", r"search"],
    "docs": [r"document", r"boilerplate", r"template", r"scaffold", r"generate"],
    "api": [r"api", r"endpoint", r"rest", r"integration", r"library"],
    "security": [r"security", r"vulnerab", r"injection", r"xss", r"auth", r"owasp", r"pen.?test"],
    "test": [r"test", r"unit", r"coverage", r"mock", r"tdd", r"pytest", r"jest"],
    "math": [r"calculate", r"compute", r"math", r"formula"],
    "verification": [r"verify", r"validate", r"check", r"confirm", r"prove", r"evidence", r"hallucin"],
    "agent": [r"agent", r"multi.?step", r"tool.?use", r"automat", r"workflow", r"ci.?cd", r"pipeline"],
    "rag": [r"retriev", r"rag", r"search.*doc", r"vector", r"embed", r"knowledge.?base", r"corpus"],
    "competitive": [r"leetcode", r"hackerrank", r"codeforce", r"contest", r"competitive", r"interview"],
    "research": [r"understand", r"explain", r"learn", r"research", r"onboard", r"document"],
    "planning": [r"plan", r"roadmap", r"strategy", r"methodic", r"step.?by.?step"]
}

HEURISTIC_MAP: Dict[str, str] = {
    "debug": "active_inference",
    "refactor": "graph_of_thoughts",
    "architecture": "reason_flux",
    "algorithm": "tree_of_thoughts",
    "docs": "skeleton_of_thought",
    "api": "critic",
    "security": "chain_of_verification",
    "test": "tdd_prompting",
    "math": "program_of_thoughts",
    "verification": "verify_and_edit",
    "agent": "swe_agent",
    "rag": "self_rag",
    "competitive": "alphacodium",
    "research": "chain_of_note",
    "planning": "plan_and_solve",
    "unknown": "self_discover"
}


# =============================================================================
# Framework Metadata
# =============================================================================

_FRAMEWORK_INFO: Dict[str, Dict[str, Any]] = {
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
    # 2026 Edition frameworks
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
    # Additional coding frameworks
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
    # Verification frameworks
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
    # Agent frameworks
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
    # RAG frameworks
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
    # Additional code frameworks
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

# Task type mapping
_TASK_TYPE_MAP: Dict[str, str] = {
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


def get_framework_info(framework: str, raise_on_unknown: bool = False) -> Dict[str, Any]:
    """Get metadata about a framework.

    Args:
        framework: Name of the framework
        raise_on_unknown: If True, raise FrameworkNotFoundError for unknown frameworks

    Returns:
        Framework metadata dict
    """
    if framework in _FRAMEWORK_INFO:
        return _FRAMEWORK_INFO[framework]

    if raise_on_unknown:
        raise FrameworkNotFoundError(
            f"Unknown framework: {framework}",
            details={"requested": framework, "available": list(FRAMEWORKS.keys())}
        )

    return {
        "name": framework,
        "category": "unknown",
        "description": "Unknown framework",
        "best_for": [],
        "complexity": "unknown"
    }


def infer_task_type(framework: str) -> str:
    """Infer task type from chosen framework."""
    return _TASK_TYPE_MAP.get(framework, "unknown")
