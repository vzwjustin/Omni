"""
Hyper-Router: Intelligent LLM-Powered Framework Selection

Uses hierarchical category routing with specialist agents.
Supports framework chaining for complex multi-step tasks.
Designed for vibe coders and senior engineers who just want it to work.
"""

import asyncio
import hashlib
import heapq
import json
import time
import structlog
from functools import lru_cache
from typing import Optional, List, Tuple

from ..state import GraphState
from .constants import CONTENT
from .errors import (
    RoutingError,
    FrameworkNotFoundError,
    LLMError,
    ProviderNotConfiguredError,
    RateLimitError,
)
from .vibe_dictionary import VIBE_DICTIONARY
from .routing import (
    CATEGORIES,
    FRAMEWORKS,
    get_framework_info,
    infer_task_type,
)
from .routing.complexity import ComplexityEstimator
from .routing.vibes import VibeMatcher
from .settings import get_settings

# Precomputed frozenset for O(1) framework membership testing
# Used in _parse_specialist_response and _extract_framework for fast validation
FRAMEWORK_SET: frozenset[str] = frozenset(FRAMEWORKS.keys())

logger = structlog.get_logger("router")


# ==========================================================================
# PROMPT TEMPLATES
# ==========================================================================

SPECIALIST_PROMPT_TEMPLATE = """You are the **{specialist}** - a specialist agent for {category_description}.

TASK: {query}
{context_section}

## Available Frameworks (pick 1 or chain multiple):
{framework_descriptions}

## Pre-defined Chains (for complex tasks):
{chain_descriptions}

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
    FRAMEWORKS = FRAMEWORKS

    def __init__(self):
        self._complexity_estimator = ComplexityEstimator()
        self._vibe_matcher = VibeMatcher()
        self._brief_generator = None
        # Routing decision cache: query_hash -> (chain, reasoning, category, timestamp)
        self._routing_cache: dict[str, tuple] = {}
        # Min-heap for O(log n) eviction: (timestamp, cache_key) pairs
        self._cache_heap: list[tuple[float, str]] = []
        # Use settings for cache configuration
        settings = get_settings()
        self._cache_max_size = settings.routing_cache_max_size
        self._cache_ttl_seconds = settings.routing_cache_ttl_seconds
        # Lazy initialization: asyncio.Lock() requires an event loop
        # which may not exist at module load time when global router is created
        self.__cache_lock: Optional[asyncio.Lock] = None

    @property
    def _cache_lock(self) -> asyncio.Lock:
        """Lazily initialize the asyncio lock when first needed."""
        if self.__cache_lock is None:
            self.__cache_lock = asyncio.Lock()
        return self.__cache_lock

    # ==========================================================================
    # ROUTING CACHE
    # ==========================================================================


    def _get_cache_key(self, query: str, code_snippet: Optional[str] = None) -> str:
        """Generate cache key from query (normalized)."""
        # Normalize: lowercase, strip, first 500 chars
        normalized = query.lower().strip()[:500]
        if code_snippet:
            normalized += "|" + code_snippet[:200]
        return hashlib.sha256(normalized.encode()).hexdigest()

    async def _get_cached_routing(self, cache_key: str) -> Optional[Tuple[List[str], str, str]]:
        """Get cached routing decision if valid (thread-safe)."""
        async with self._cache_lock:
            if cache_key not in self._routing_cache:
                return None
            chain, reasoning, category, timestamp = self._routing_cache[cache_key]
            # Check TTL
            if time.time() - timestamp > self._cache_ttl_seconds:
                del self._routing_cache[cache_key]
                return None
            logger.debug("routing_cache_hit", cache_key=cache_key[:8])
            return chain, reasoning, category

    async def _set_cached_routing(
        self,
        cache_key: str,
        chain: List[str],
        reasoning: str,
        category: str
    ) -> None:
        """Cache a routing decision (thread-safe)."""
        async with self._cache_lock:
            timestamp = time.time()

            # Evict oldest entries if at capacity using O(log n) heap operations
            while len(self._routing_cache) >= self._cache_max_size and self._cache_heap:
                # Pop the oldest entry from the heap
                oldest_ts, oldest_key = heapq.heappop(self._cache_heap)

                # Handle stale heap entries: key was already deleted or updated
                if oldest_key not in self._routing_cache:
                    # Key was deleted (e.g., TTL expiration in _get_cached_routing)
                    continue

                # Check if heap entry is stale (key was updated with a newer timestamp)
                cached_entry = self._routing_cache[oldest_key]
                cached_timestamp = cached_entry[3]
                if cached_timestamp != oldest_ts:
                    # Entry was updated; skip this stale heap entry
                    continue

                # Valid entry found - evict it
                del self._routing_cache[oldest_key]
                logger.debug("routing_cache_evicted", key=oldest_key[:8])
                break

            # Add new entry to cache and heap
            self._routing_cache[cache_key] = (chain, reasoning, category, timestamp)
            heapq.heappush(self._cache_heap, (timestamp, cache_key))

    # ==========================================================================
    # HIERARCHICAL ROUTING - Stage 2: Specialist Agent Selection
    # ==========================================================================

    @lru_cache(maxsize=16)
    def _get_specialist_prompt_template(self, category: str) -> tuple:
        """
        Generate the static parts of a specialist agent prompt for a category.
        Returns (template, specialist_name) to be formatted with query/context.

        Cached by category only to prevent cache thrashing from unique queries.
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

        chain_desc_text = "\n".join(chain_descriptions) if chain_descriptions else "  (none - single framework recommended)"
        fw_desc_text = "\n".join(fw_descriptions)
        
        return (specialist, fw_desc_text, chain_desc_text)

    def _get_specialist_prompt(self, category: str, query: str, context: str = "") -> str:
        """
        Generate a specialist agent prompt for framework selection within a category.
        The specialist can recommend single frameworks or chains.

        Uses cached template parts for efficiency.
        """
        specialist, fw_desc_text, chain_desc_text = self._get_specialist_prompt_template(category)
        context_section = f"CONTEXT: {context}" if context else ""
        cat_info = CATEGORIES.get(category, CATEGORIES["exploration"])

        return SPECIALIST_PROMPT_TEMPLATE.format(
            specialist=specialist,
            category_description=cat_info["description"],
            query=query,
            context_section=context_section,
            framework_descriptions=fw_desc_text,
            chain_descriptions=chain_desc_text,
        )

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
        chain_match = self._check_chain_patterns(query, chain_patterns)
        if chain_match:
            return chain_match

        # Try specialist agent for nuanced selection
        result = await self._invoke_specialist_agent(category, query, context)
        if result:
            return result

        # Default: pick first framework in category (graceful fallback)
        fallback_reason = f"[Fallback] Local pattern match for {category}"
        return [frameworks[0]], fallback_reason

    def _check_chain_patterns(
        self,
        query: str,
        chain_patterns: dict
    ) -> Optional[Tuple[List[str], str]]:
        """Check if query matches any predefined chain pattern."""
        query_lower = query.lower()
        for pattern_name, chain in chain_patterns.items():
            pattern_words = pattern_name.replace("_", " ").split()
            if all(word in query_lower for word in pattern_words):
                return chain, f"Matched chain pattern: {pattern_name}"
        return None

    async def _invoke_specialist_agent(
        self,
        category: str,
        query: str,
        context: str
    ) -> Optional[Tuple[List[str], str]]:
        """
        Invoke the specialist LLM agent for framework selection.
        Returns None on failure to allow graceful fallback.
        """
        # Timeout for LLM calls to prevent indefinite hangs
        LLM_TIMEOUT_SECONDS = 30.0

        try:
            from ..langchain_integration import get_chat_model

            prompt = self._get_specialist_prompt(category, query, context)
            llm = get_chat_model("fast")

            # Wrap LLM call with timeout to prevent indefinite blocking
            response = await asyncio.wait_for(
                llm.ainvoke(prompt),
                timeout=LLM_TIMEOUT_SECONDS
            )

            response_text = self._extract_response_text(response)
            return self._parse_specialist_response(response_text)

        except asyncio.TimeoutError:
            logger.warning(
                "specialist_agent_timeout",
                category=category,
                timeout_seconds=LLM_TIMEOUT_SECONDS,
                hint="LLM call timed out, falling back to local pattern matching"
            )
        except RateLimitError as e:
            logger.warning(
                "gemini_billing_issue",
                error=repr(e),
                hint="Using local pattern matching. Add GOOGLE_API_KEY with credits for AI routing."
            )
        except ProviderNotConfiguredError as e:
            logger.warning(
                "gemini_auth_issue",
                error=repr(e),
                hint="Set GOOGLE_API_KEY for Gemini-powered routing."
            )
        except LLMError as e:
            logger.warning("specialist_selection_failed", error=repr(e))
        except json.JSONDecodeError as e:
            logger.warning(
                "specialist_response_parse_error",
                error=str(e)[:100],
                hint="LLM returned malformed response"
            )
        except (AttributeError, TypeError, KeyError) as e:
            # Response format errors from unexpected LLM output structure
            wrapped_error = LLMError(
                f"Unexpected response format: {str(e)[:100]}",
                details={"category": category, "original_error": type(e).__name__}
            )
            logger.warning("specialist_response_format_error", error=repr(wrapped_error))

        return None

    def _extract_response_text(self, response) -> str:
        """Extract text content from LLM response, handling various formats."""
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            return content[0].get("text", str(content)) if content else ""
        return content

    def _parse_specialist_response(
        self,
        response_text: str
    ) -> Optional[Tuple[List[str], str]]:
        """Parse the specialist agent response to extract frameworks and reasoning."""
        frameworks_line = ""
        reasoning = ""

        for line in response_text.split("\n"):
            if line.startswith("FRAMEWORKS:"):
                frameworks_line = line.replace("FRAMEWORKS:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        if not frameworks_line:
            return None

        # Parse chain: "fw1 -> fw2 -> fw3" or "fw1"
        selected = [
            fw.strip()
            for fw in frameworks_line.replace("->", ",").replace("→", ",").split(",")
        ]
        # O(1) set membership check for each framework in the chain
        selected = [fw for fw in selected if fw in FRAMEWORK_SET]

        if not selected:
            return None

        return selected, reasoning or f"Specialist selected: {frameworks_line}"

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

        # Check cache first
        cache_key = self._get_cache_key(query, code_snippet)
        cached = await self._get_cached_routing(cache_key)
        if cached:
            chain, reasoning, category = cached
            return chain, f"[Cached] {reasoning}", category

        # Stage 1: Route to category
        category, confidence = self._vibe_matcher.route_to_category(query)

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
            vibe_match = self._vibe_matcher.check_vibe_dictionary(query)

            if vibe_match and vibe_match in cat_frameworks:
                chain = [vibe_match]
                reasoning = f"Vibe match in {category}: {query[:30]}..."
            else:
                chain, reasoning = await self._select_with_specialist(
                    category, query, code_snippet, ide_context
                )

        # Cache the result
        await self._set_cached_routing(cache_key, chain, reasoning, category)

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
            # INTENTIONAL BROAD CATCH: Framework selection must not fail the entire request.
            # Graceful degradation to vibe matching ensures the system remains operational
            # even when hierarchical routing encounters unexpected errors.
            logger.warning(
                "framework_chain_selection_failed",
                error=str(e)[:CONTENT.QUERY_LOG],
                error_type=type(e).__name__
            )

        # Legacy fallback: direct vibe matching
        vibe_match = self._vibe_matcher.check_vibe_dictionary(query)
        if vibe_match:
            return vibe_match, f"Matched vibe: {query[:CONTENT.QUERY_PREVIEW]}..."

        return "self_discover", "Fallback: routing failed, using self_discover."

    def _extract_framework(self, response: str) -> Optional[str]:
        """Extract framework name from LLM response.

        Note: This method iterates over all 62 frameworks to find substring matches.
        This is intentional as framework names may appear anywhere in the response
        and may contain underscores (e.g., active_inference).

        The O(62 * M) complexity is acceptable because:
        - 62 frameworks is a fixed constant
        - Response text M is typically < 500 chars
        - This method is called infrequently (only as fallback)
        """
        response_lower = response.lower()
        for framework in FRAMEWORK_SET:
            if framework in response_lower:
                return framework
        return None

    def _heuristic_select(self, query: str, code_snippet: Optional[str] = None) -> str:
        """Fast heuristic selection (fallback)."""
        return self._vibe_matcher.heuristic_select(query, code_snippet)

    def estimate_complexity(
        self,
        query: str,
        code_snippet: Optional[str] = None,
        file_list: Optional[list[str]] = None
    ) -> float:
        """Estimate task complexity on 0-1 scale."""
        return self._complexity_estimator.estimate(query, code_snippet, file_list)

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

        # Check for explicit preference first (O(1) set lookup)
        if preferred and preferred in FRAMEWORK_SET:
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
                # INTENTIONAL BROAD CATCH: Chain selection failure should not block routing.
                # Fallback to auto_select_framework provides a simpler but functional path.
                # This ensures observability while maintaining system availability.
                logger.warning(
                    "framework_chain_in_route_failed",
                    error=str(e)[:CONTENT.QUERY_LOG],
                    error_type=type(e).__name__,
                    query=query[:CONTENT.QUERY_LOG] if query else ""
                )
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
        # Always initialize to empty list for consistent downstream access
        state["episodic_memory"] = []
        try:
            from ..collection_manager import get_collection_manager
            cm = get_collection_manager()
            learnings = cm.search_learnings(query, k=3)
            if learnings:
                state["episodic_memory"] = learnings
        except Exception as e:
            # INTENTIONAL BROAD CATCH: Episodic memory is an enhancement, not a requirement.
            # Routing should succeed even if memory retrieval fails (e.g., ChromaDB unavailable,
            # embedding service down). The core routing logic must remain operational.
            # state["episodic_memory"] already set to [] above for graceful degradation
            logger.warning(
                "episodic_memory_search_failed",
                error=str(e)[:CONTENT.QUERY_LOG],
                error_type=type(e).__name__,
                query=query[:CONTENT.QUERY_LOG] if query else ""
            )

        # Add reasoning step with size limit to prevent unbounded growth
        MAX_REASONING_STEPS = 100
        if "reasoning_steps" not in state:
            state["reasoning_steps"] = []
        reasoning_steps = state["reasoning_steps"]
        # Ensure it's a list (defensive check for malformed state)
        if not isinstance(reasoning_steps, list):
            reasoning_steps = []
        reasoning_steps.append({
            "step": "routing",
            "framework": framework_chain[0] if framework_chain else "self_discover",
            "framework_chain": framework_chain,
            "category": category,
            "reason": reason,
            "complexity": complexity,
            "method": "hierarchical_ai" if use_ai else "heuristic"
        })
        # Trim to max size, keeping most recent entries
        if len(reasoning_steps) > MAX_REASONING_STEPS:
            reasoning_steps = reasoning_steps[-MAX_REASONING_STEPS:]
        state["reasoning_steps"] = reasoning_steps

        return state

    def _infer_task_type(self, framework: str) -> str:
        """Infer task type from chosen framework."""
        return infer_task_type(framework)

    def get_framework_info(self, framework: str, raise_on_unknown: bool = False) -> dict:
        """Get metadata about a framework.

        Args:
            framework: Name of the framework
            raise_on_unknown: If True, raise FrameworkNotFoundError for unknown frameworks

        Returns:
            Framework metadata dict
        """
        return get_framework_info(framework, raise_on_unknown=raise_on_unknown)

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
