"""
Hyper-Router: Intelligent LLM-Powered Framework Selection

Uses hierarchical category routing with specialist agents.
Supports framework chaining for complex multi-step tasks.
Designed for vibe coders and senior engineers who just want it to work.
"""

import re
import structlog
from typing import Optional, List, Tuple

from ..state import GraphState
from .constants import CONTENT
from .errors import RoutingError, FrameworkNotFoundError
from .vibe_dictionary import VIBE_DICTIONARY, match_vibes
from .routing import (
    CATEGORIES,
    CATEGORY_VIBES,
    FRAMEWORKS,
    PATTERNS,
    HEURISTIC_MAP,
    get_framework_info,
    infer_task_type,
)

logger = structlog.get_logger("router")


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

    # Re-export class-level constants for backward compatibility
    CATEGORIES = CATEGORIES
    CATEGORY_VIBES = CATEGORY_VIBES
    FRAMEWORKS = FRAMEWORKS
    PATTERNS = PATTERNS
    HEURISTIC_MAP = HEURISTIC_MAP

    def __init__(self):
        self._compiled_patterns = {
            task_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for task_type, patterns in PATTERNS.items()
        }
        self._brief_generator = None

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

        for category, vibes in CATEGORY_VIBES.items():
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
        cat_info = CATEGORIES.get(category, CATEGORIES["exploration"])
        frameworks = cat_info["frameworks"]
        chain_patterns = cat_info.get("chain_patterns", {})
        specialist = cat_info["specialist"]

        # Build framework descriptions for this category
        fw_descriptions = []
        for fw in frameworks:
            desc = FRAMEWORKS.get(fw, "Unknown framework")
            vibes = VIBE_DICTIONARY.get(fw, [])[:3]
            fw_descriptions.append(f"  - {fw}: {desc} (vibes: {', '.join(vibes)})")

        # Build chain pattern descriptions
        chain_descriptions = []
        for pattern_name, chain in chain_patterns.items():
            chain_descriptions.append(f"  - {pattern_name}: {' -> '.join(chain)}")

        return f"""You are the **{specialist}** - a specialist agent for {cat_info['description']}.

TASK: {query}
{f'CONTEXT: {context}' if context else ''}

## Available Frameworks (pick 1 or chain multiple):
{chr(10).join(fw_descriptions)}

## Pre-defined Chains (for complex tasks):
{chr(10).join(chain_descriptions) if chain_descriptions else '  (none - single framework recommended)'}

## When to Chain (IMPORTANT):
Use a chain of 2-4 frameworks when the task has ANY of these signals:
- Multiple distinct phases (e.g., "first... then... finally...")
- Requires both analysis AND implementation
- Mentions verification, testing, or review as a final step
- Involves understanding before changing
- Security-sensitive changes that need audit
- Complex debugging that needs hypothesis -> test -> verify

Single framework only for: quick fixes, simple questions, one-step tasks.

## Instructions:
1. Analyze the task - look for multi-phase signals
2. For SIMPLE tasks: recommend a single framework
3. For COMPLEX tasks: recommend a chain of 2-4 frameworks in logical order

Respond EXACTLY in this format:
COMPLEXITY: simple|complex
FRAMEWORKS: framework1 -> framework2 -> framework3
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
        cat_info = CATEGORIES.get(category, CATEGORIES["exploration"])
        frameworks = cat_info["frameworks"]
        chain_patterns = cat_info.get("chain_patterns", {})

        # Build context
        context = ""
        if code_snippet:
            context += f"Code:\n{code_snippet[:CONTENT.SNIPPET_SHORT]}..."
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

            # Handle different response formats
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                response_text = content[0].get('text', str(content)) if content else ""
            else:
                response_text = content

            # Parse response
            frameworks_line = ""
            reasoning = ""
            for line in response_text.split("\n"):
                if line.startswith("FRAMEWORKS:"):
                    frameworks_line = line.replace("FRAMEWORKS:", "").strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()

            if frameworks_line:
                # Parse chain: "fw1 -> fw2 -> fw3" or "fw1"
                selected = [fw.strip() for fw in frameworks_line.replace("->", ",").replace("→", ",").split(",")]
                selected = [fw for fw in selected if fw in FRAMEWORKS]
                if selected:
                    return selected, reasoning or f"Specialist selected: {frameworks_line}"

        except Exception as e:
            error_msg = str(e).lower()
            
            # Detect specific Gemini API errors
            if "insufficient" in error_msg or "quota" in error_msg or "billing" in error_msg:
                logger.warning(
                    "gemini_billing_issue",
                    error="Insufficient funds or quota exceeded",
                    category=category,
                    hint="Using local pattern matching. Add GOOGLE_API_KEY with credits for AI routing."
                )
            elif "api_key" in error_msg or "unauthorized" in error_msg:
                logger.warning(
                    "gemini_auth_issue", 
                    error="API key missing or invalid",
                    hint="Set GOOGLE_API_KEY for Gemini-powered routing."
                )
            else:
                logger.warning(
                    "specialist_selection_failed",
                    error=str(e)[:CONTENT.QUERY_LOG],
                    category=category
                )

        # Default: pick first framework in category (graceful fallback)
        fallback_reason = f"[Fallback] Local pattern match for {category}"
        return [frameworks[0]], fallback_reason

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
        from ..core.settings import get_settings

        # Stage 1: Route to category
        category, confidence = self._route_to_category(query)

        # Stage 2: Specialist agent selection
        llm_enabled = get_settings().llm_provider not in ("pass-through", "none", "")

        if llm_enabled:
            # LLM-powered routing - always use specialist agent
            chain, reasoning = await self._select_with_specialist(
                category, query, code_snippet, ide_context
            )
            reasoning = f"[Gemini] {reasoning}"
        elif confidence < 0.5:
            # Low confidence without LLM - try specialist anyway
            chain, reasoning = await self._select_with_specialist(
                category, query, code_snippet, ide_context
            )
        else:
            # High confidence, no LLM - use vibe matching
            cat_frameworks = CATEGORIES[category]["frameworks"]
            vibe_match = self._check_vibe_dictionary(query)

            if vibe_match and vibe_match in cat_frameworks:
                chain = [vibe_match]
                reasoning = f"Vibe match in {category}: {query[:30]}..."
            else:
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

        Returns: (first_framework, reasoning)
        For chain access, use select_framework_chain() instead.
        """
        try:
            chain, reasoning, category = await self.select_framework_chain(
                query, code_snippet, ide_context
            )
            if chain:
                if len(chain) > 1:
                    reasoning = f"[Chain: {' -> '.join(chain)}] {reasoning}"
                return chain[0], reasoning
        except Exception as e:
            logger.warning("framework_chain_selection_failed", error=str(e)[:CONTENT.QUERY_LOG])

        # Legacy fallback: direct vibe matching
        vibe_match = self._check_vibe_dictionary(query)
        if vibe_match:
            return vibe_match, f"Matched vibe: {query[:CONTENT.QUERY_PREVIEW]}..."

        return "self_discover", "Fallback: routing failed, using self_discover."

    def _extract_framework(self, response: str) -> Optional[str]:
        """Extract framework name from LLM response."""
        response_lower = response.lower()
        for framework in FRAMEWORKS:
            if framework in response_lower:
                return framework
        return None

    def _check_vibe_dictionary(self, query: str) -> Optional[str]:
        """Quick check against vibe dictionary with weighted scoring."""
        return match_vibes(query)

    def _heuristic_select(self, query: str, code_snippet: Optional[str] = None) -> str:
        """Fast heuristic selection (fallback)."""
        combined = query + (" " + code_snippet if code_snippet else "")

        scores = {}
        for task_type, patterns in self._compiled_patterns.items():
            scores[task_type] = sum(1 for p in patterns if p.search(combined))

        if max(scores.values()) > 0:
            task_type = max(scores, key=scores.get)
            return HEURISTIC_MAP.get(task_type, "self_discover")

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
        """
        framework_chain = []
        category = "unknown"
        reason = ""

        query = state.get("query", "")
        code_snippet = state.get("code_snippet")
        ide_context = state.get("ide_context")
        file_list = state.get("file_list")
        preferred = state.get("preferred_framework")

        # Check for explicit preference first
        if preferred and preferred in FRAMEWORKS:
            framework_chain = [preferred]
            reason = "User specified"
        elif use_ai:
            # Hierarchical AI-powered selection with chain support
            try:
                chain, reason, category = await self.select_framework_chain(
                    query, code_snippet, ide_context
                )
                framework_chain = chain
            except Exception as e:
                # Fallback to single framework
                framework, reason = await self.auto_select_framework(
                    query, code_snippet, ide_context
                )
                framework_chain = [framework]
        else:
            # Heuristic fallback (single framework only)
            framework = self._heuristic_select(query, code_snippet)
            framework_chain = [framework]
            reason = "Heuristic selection"

        complexity = self.estimate_complexity(query, code_snippet, file_list)

        # Update state with chain support
        state["selected_framework"] = framework_chain[0] if framework_chain else "self_discover"
        state["framework_chain"] = framework_chain
        state["complexity_estimate"] = complexity
        state["task_type"] = infer_task_type(framework_chain[0] if framework_chain else "self_discover")
        state["routing_category"] = category

        # Inject Past Learnings (Episodic Memory)
        try:
            from ..collection_manager import get_collection_manager
            cm = get_collection_manager()
            learnings = cm.search_learnings(query, k=3)
            if learnings:
                state["episodic_memory"] = learnings
        except Exception as e:
            logger.warning(
                "episodic_memory_search_failed",
                error=str(e)[:CONTENT.QUERY_LOG],
                query=query[:CONTENT.QUERY_LOG] if query else ""
            )

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
        return infer_task_type(framework)

    def get_framework_info(self, framework: str) -> dict:
        """Get metadata about a framework."""
        return get_framework_info(framework)

    # ==========================================================================
    # STRUCTURED BRIEF GENERATION
    # ==========================================================================

    async def generate_structured_brief(
        self,
        query: str,
        context: Optional[str] = None,
        code_snippet: Optional[str] = None,
        ide_context: Optional[str] = None,
        file_list: Optional[List[str]] = None
    ) -> "GeminiRouterOutput":
        """
        Generate a structured GeminiRouterOutput with ClaudeCodeBrief.

        This is the main entry point for the new structured handoff protocol.
        Gemini orchestrates, Claude executes.
        """
        from .routing import StructuredBriefGenerator

        if self._brief_generator is None:
            self._brief_generator = StructuredBriefGenerator(self)

        return await self._brief_generator.generate(
            query=query,
            context=context,
            code_snippet=code_snippet,
            ide_context=ide_context,
            file_list=file_list
        )


# Global router instance
router = HyperRouter()
