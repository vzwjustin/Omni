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
        "system1": "Fast heuristic, minimal thinking. Best for trivial quick fixes."
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
