"""
Hyper-Router: Intelligent LLM-Powered Framework Selection

Uses AI to analyze tasks and select the optimal reasoning framework.
Designed for vibe coders and senior engineers who just want it to work.
"""

import re
from typing import Optional
from ..state import GraphState


class HyperRouter:
    """
    The Hyper-Dispatcher: AI-powered framework selection.
    
    Two modes:
    - AUTO (default): Uses LLM to intelligently pick the best framework
    - HEURISTIC: Uses fast pattern matching (fallback)
    """
    
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
        "recode": "Multi-candidate validation with CFG-based debugging. Best for reliable code generation and high-stakes code."
    }
    
    # Heuristic patterns (fallback)
    PATTERNS = {
        "debug": [r"bug", r"error", r"exception", r"crash", r"fix", r"broken", r"fails", r"null", r"undefined"],
        "refactor": [r"refactor", r"clean", r"reorganize", r"legacy", r"spaghetti", r"debt"],
        "architecture": [r"architect", r"design", r"structure", r"system", r"microservice", r"scale"],
        "algorithm": [r"algorithm", r"optimize", r"complexity", r"performance", r"efficient", r"sort", r"search"],
        "docs": [r"document", r"boilerplate", r"template", r"scaffold", r"generate"],
        "api": [r"api", r"endpoint", r"rest", r"integration", r"library"],
        "security": [r"security", r"vulnerab", r"injection", r"xss", r"auth"],
        "test": [r"test", r"unit", r"coverage", r"mock"],
        "math": [r"calculate", r"compute", r"math", r"formula"]
    }
    
    HEURISTIC_MAP = {
        "debug": "active_inference",
        "refactor": "graph_of_thoughts",
        "architecture": "reason_flux",
        "algorithm": "tree_of_thoughts",
        "docs": "skeleton_of_thought",
        "api": "critic",
        "security": "chain_of_verification",
        "test": "program_of_thoughts",
        "math": "program_of_thoughts",
        "unknown": "self_discover"
    }
    
    # =========================================================================
    # VIBE DICTIONARY: Casual activation phrases for vibe coders
    # =========================================================================
    # Just say what you want naturally - we'll figure out the right framework
    
    VIBE_DICTIONARY = {
        # Debugging vibes
        "active_inference": [
            "why is this broken", "wtf is wrong", "this doesn't work", 
            "find the bug", "debug this", "what's causing this",
            "figure out why", "track down", "root cause", "investigate",
            "something's off", "it's acting weird", "unexpected behavior"
        ],
        
        # Refactoring vibes  
        "graph_of_thoughts": [
            "clean this up", "this code is ugly", "make it not suck",
            "untangle this mess", "spaghetti code", "needs refactoring",
            "reorganize", "restructure", "make it cleaner", "simplify this"
        ],
        
        # Architecture vibes
        "reason_flux": [
            "design a system", "architect this", "how should I structure",
            "plan this out", "big picture", "system design", "make it scalable",
            "build from scratch", "greenfield", "new project", "overall approach"
        ],
        
        # Algorithm vibes
        "tree_of_thoughts": [
            "make it faster", "optimize this", "too slow", "performance sucks",
            "better algorithm", "more efficient", "reduce complexity",
            "speed this up", "it's laggy", "bottleneck"
        ],
        
        # Fast generation vibes
        "skeleton_of_thought": [
            "just generate", "scaffold", "boilerplate", "quick template",
            "stub this out", "starter code", "basic structure", "skeleton",
            "rough draft", "get me started", "wireframe"
        ],
        
        # API vibes
        "critic": [
            "using this library", "api integration", "third party", 
            "how do I use", "sdk", "package", "library docs",
            "correct usage", "best practice for"
        ],
        
        # Security vibes
        "chain_of_verification": [
            "is this secure", "security check", "audit this", "vulnerabilities",
            "could this be hacked", "pen test", "code review",
            "sanity check", "validate this", "double check"
        ],
        
        # Math/compute vibes
        "program_of_thoughts": [
            "calculate", "compute", "do the math", "run the numbers",
            "data processing", "transform data", "crunch", "analyze data"
        ],
        
        # Creative/novel vibes
        "self_discover": [
            "I have no idea", "weird problem", "never seen this",
            "creative solution", "think outside box", "novel approach",
            "unconventional", "unique situation", "edge case"
        ],
        
        # Deep debugging vibes
        "mcts_rstar": [
            "really hard bug", "been stuck for hours", "complex issue",
            "multi-step problem", "deep issue", "intricate bug",
            "need to explore options", "thorough search"
        ],
        
        # Decision vibes
        "multi_agent_debate": [
            "should I use A or B", "trade-offs", "pros and cons",
            "which approach", "compare options", "decision", "evaluate",
            "weigh options", "what would you recommend"
        ],
        
        # Big change vibes
        "everything_of_thought": [
            "major rewrite", "big migration", "overhaul", "massive change",
            "complete redesign", "from scratch", "total refactor",
            "modernize", "upgrade everything"
        ],
        
        # Pattern vibes
        "buffer_of_thoughts": [
            "I've done this before", "standard pattern", "common task",
            "typical", "usual approach", "boilerplate", "routine"
        ],
        
        # Long context vibes
        "coala": [
            "lots of files", "whole codebase", "across multiple files",
            "context from", "remember earlier", "stateful", "keep track"
        ],
        
        # Research vibes
        "chain_of_note": [
            "understand this code", "explain", "what does this do",
            "learn the codebase", "document", "figure out how",
            "reverse engineer", "how does it work"
        ],
        
        # Abstraction vibes
        "step_back": [
            "big O", "complexity analysis", "fundamentals", "first principles",
            "underlying concept", "theory behind", "abstract thinking"
        ],
        
        # Analogy vibes
        "analogical": [
            "like when", "similar to", "pattern from", "reminds me of",
            "same as", "analogous", "comparable to"
        ],
        
        # Mixed vibes
        "adaptive_injection": [
            "just figure it out", "do your thing", "whatever works",
            "adapt", "flex", "go with the flow"
        ],
        
        # Requirements vibes
        "re2": [
            "requirements", "spec", "constraints", "must have",
            "needs to", "requirements doc", "acceptance criteria"
        ],
        
        # Quick vibes
        "system1": [
            "quick question", "simple fix", "easy", "obvious",
            "just do it", "no brainer", "trivial", "one liner",
            "fast answer", "real quick"
        ],

        # New framework vibes (2026 Edition)
        "chain_of_code": [
            "code blocks", "pseudocode", "execution trace", "logic puzzle",
            "recursive logic", "algorithmic complexity", "structured thinking",
            "break into code", "code decomposition"
        ],

        "self_debugging": [
            "test before showing", "mental execution", "trace through",
            "prevent bugs", "check my work", "simulate execution",
            "off by one", "edge case check", "dry run"
        ],

        "tdd_prompting": [
            "test first", "write tests", "TDD", "test driven",
            "unit tests", "edge cases", "test coverage",
            "red green refactor", "tests then code"
        ],

        "reverse_cot": [
            "wrong output", "expected vs actual", "why different output",
            "output delta", "silent bug", "calculation error",
            "backwards debugging", "work backward", "reverse engineer bug"
        ],

        "rubber_duck": [
            "explain to me", "walk me through", "ask me questions",
            "guide me", "help me think", "rubber duck",
            "Socratic method", "lead me to answer", "questioning approach"
        ],

        "react": [
            "use tools", "multi-step", "action reasoning",
            "tool use", "step and observe", "interact with",
            "reasoning acting", "ReAct pattern", "observe results"
        ],

        "reflexion": [
            "learn from mistakes", "retry", "failed attempt",
            "try again", "reflect on", "what went wrong",
            "iterative learning", "self-evaluation", "memory-based"
        ],

        "self_refine": [
            "improve quality", "polish", "refine",
            "make it better", "iterative improvement", "critique and improve",
            "self-critique", "refinement loop", "quality pass"
        ],

        "least_to_most": [
            "atomic functions", "dependency graph", "bottom up",
            "layered", "base functions first", "decompose completely",
            "building blocks", "hierarchical build", "least dependent first"
        ],

        "comparative_arch": [
            "compare approaches", "readability vs performance", "trade-offs",
            "multiple solutions", "which is faster", "optimize for",
            "different versions", "performance vs memory", "three approaches"
        ],

        "plan_and_solve": [
            "plan first", "think before coding", "explicit plan",
            "strategy", "outline approach", "plan then execute",
            "methodical", "step by step plan", "planning phase"
        ],

        "red_team": [
            "security audit", "vulnerabilities", "pen test",
            "security review", "find exploits", "attack vectors",
            "OWASP", "SQLi", "XSS", "security threats", "hack this"
        ],

        "state_machine": [
            "state machine", "FSM", "states and transitions",
            "workflow", "state diagram", "UI states",
            "game states", "state management", "transitions"
        ],

        "chain_of_thought": [
            "think step by step", "reason through", "work through",
            "logical steps", "step by step", "reasoning chain",
            "think carefully", "show your work", "explicit reasoning"
        ],

        # Additional coding frameworks (2026 expansion)
        "alphacodium": [
            "competitive programming", "code contest", "algorithm challenge",
            "iterative code", "test-based", "multi-stage",
            "code generation", "contest problem", "leetcode"
        ],

        "codechain": [
            "modular code", "sub-modules", "self-revision",
            "incremental", "chain revisions", "module by module",
            "component based", "build incrementally", "refine modules"
        ],

        "evol_instruct": [
            "evolve solution", "add constraints", "increase complexity",
            "challenging problem", "constraint-based", "evolutionary",
            "harder version", "more constraints", "complex requirements"
        ],

        "llmloop": [
            "feedback loop", "iterate until", "compile and fix",
            "test loop", "automated testing", "quality assurance",
            "production ready", "lint and fix", "keep iterating"
        ],

        "procoder": [
            "compiler feedback", "project level", "codebase integration",
            "API usage", "large project", "integrate with",
            "project context", "compiler errors", "fix imports"
        ],

        "recode": [
            "multiple candidates", "cross validate", "CFG debugging",
            "control flow", "reliable code", "high stakes",
            "validate candidates", "majority voting", "robust solution"
        ]
    }
    
    def __init__(self):
        self._compiled_patterns = {
            task_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for task_type, patterns in self.PATTERNS.items()
        }
    
    async def auto_select_framework(
        self,
        query: str,
        code_snippet: Optional[str] = None,
        ide_context: Optional[str] = None
    ) -> tuple[str, str]:
        """
        AI-powered framework selection.
        
        First checks the vibe dictionary for quick matches,
        then falls back to LLM for complex/ambiguous cases.
        Returns: (framework_name, reasoning)
        """
        # First, check vibe dictionary for quick match
        vibe_match = self._check_vibe_dictionary(query)
        if vibe_match:
            return vibe_match, f"Matched vibe: {query[:50]}..."
        
        # Build framework list with vibes for the AI using prompt template
        framework_list = []
        for name, desc in self.FRAMEWORKS.items():
            vibes = ", ".join(self.VIBE_DICTIONARY.get(name, [])[:3])
            framework_list.append(f"{name}: {desc}. Vibes: {vibes}")
        
        try:
            from ..langchain_integration import (
                FRAMEWORK_SELECTION_TEMPLATE,
                framework_parser,
                get_chat_model,
            )
            messages = FRAMEWORK_SELECTION_TEMPLATE.format_messages(
                frameworks="\n".join(framework_list),
                task_type="auto",
                complexity="auto",
                has_code=bool(code_snippet),
                framework_history="",
                query=query,
            )
            # Append code/IDE context in the human message
            extra_context = f"\n\nCode:\n{code_snippet or 'N/A'}\n\nIDE:\n{ide_context or 'N/A'}"
            messages[-1].content = messages[-1].content + extra_context
            
            llm = get_chat_model("fast")
            lc_response = await llm.ainvoke(messages)
            response = lc_response.content if hasattr(lc_response, "content") else str(lc_response)
            try:
                parsed = framework_parser.parse(response)
                selected = parsed.selected_framework
                reasoning = parsed.reasoning
                if selected:
                    return selected, reasoning
            except Exception:
                pass
            
            # Fallback to regex extract
            selected = self._extract_framework(response)
            if selected:
                return selected, response
            return "self_discover", "Fallback: could not parse framework, using self_discover."
        except Exception:
            return "self_discover", "Fallback: router LLM failed, using self_discover."

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
        Quick check against vibe dictionary.
        Returns framework name if strong match found.
        """
        query_lower = query.lower()
        
        # Score each framework by vibe matches
        scores = {}
        for framework, vibes in self.VIBE_DICTIONARY.items():
            score = sum(1 for vibe in vibes if vibe in query_lower)
            if score > 0:
                scores[framework] = score
        
        # Return if we have a clear winner (score >= 2 or single match)
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] >= 2 or (len(scores) == 1 and scores[best] >= 1):
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
        Route the task to the best framework.
        
        Args:
            state: Current graph state
            use_ai: If True, use LLM to select (default). If False, use heuristics.
        """
        # Check for explicit preference first
        if state.get("preferred_framework") and state["preferred_framework"] in self.FRAMEWORKS:
            framework = state["preferred_framework"]
            reason = "User specified"
        elif use_ai:
            # AI-powered selection
            framework, reason = await self.auto_select_framework(
                state["query"],
                state.get("code_snippet"),
                state.get("ide_context")
            )
        else:
            # Heuristic fallback
            framework = self._heuristic_select(state["query"], state.get("code_snippet"))
            reason = "Heuristic selection"
        
        complexity = self.estimate_complexity(
            state["query"],
            state.get("code_snippet"),
            state.get("file_list")
        )
        
        # Update state
        state["selected_framework"] = framework
        state["complexity_estimate"] = complexity
        state["task_type"] = self._infer_task_type(framework)
        
        state["reasoning_steps"].append({
            "step": "routing",
            "framework": framework,
            "reason": reason,
            "complexity": complexity,
            "method": "ai" if use_ai else "heuristic"
        })
        
        return state
    
    def _infer_task_type(self, framework: str) -> str:
        """Infer task type from chosen framework."""
        type_map = {
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
            "system1": "quick"
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
