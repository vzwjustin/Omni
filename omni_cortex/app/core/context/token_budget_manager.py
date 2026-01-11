"""
Token Budget Manager for Context Gateway

Implements intelligent token budget management with:
- Dynamic budget allocation based on task complexity
- Content prioritization algorithms
- Budget utilization tracking
- Gemini-based content ranking and optimization
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

# Try to import Google AI (with graceful fallback)
try:
    from google import genai
    from google.genai import types

    GOOGLE_AI_AVAILABLE = True
except ImportError:
    # Fallback to deprecated package
    try:
        import google.generativeai as genai

        types = None
        GOOGLE_AI_AVAILABLE = True
    except ImportError:
        GOOGLE_AI_AVAILABLE = False
        genai = None
        types = None

from ..settings import get_settings
from .enhanced_models import (
    EnhancedDocumentationContext,
    EnhancedFileContext,
    TokenBudgetAllocation,
    TokenBudgetUsage,
)

logger = structlog.get_logger("token_budget_manager")


class TokenBudgetManager:
    """
    Manages token budget allocation and optimization for context preparation.

    Responsibilities:
    - Calculate appropriate budget based on task complexity
    - Allocate budget across components (query analysis, file discovery, docs, code search)
    - Prioritize content when budget is limited
    - Track budget utilization
    - Provide optimization recommendations
    """

    def __init__(self):
        """Initialize TokenBudgetManager with settings."""
        self._settings = get_settings()

        # Complexity multipliers for budget calculation
        self._complexity_multipliers = {
            "low": 0.6,
            "medium": 1.0,
            "high": 1.5,
            "very_high": 2.0,
        }

        # Base budgets from settings
        self._base_budgets = {
            "low": self._settings.token_budget_low_complexity,
            "medium": self._settings.token_budget_medium_complexity,
            "high": self._settings.token_budget_high_complexity,
            "very_high": self._settings.token_budget_very_high_complexity,
        }

        # Component allocation percentages (default distribution)
        self._default_allocation = {
            "query_analysis": 0.15,  # 15% for query understanding
            "file_discovery": 0.35,  # 35% for file summaries
            "documentation_search": 0.30,  # 30% for documentation
            "code_search": 0.10,  # 10% for code patterns
            "context_assembly": 0.05,  # 5% for final assembly
            "reserve": 0.05,  # 5% emergency reserve
        }

        # Task type specific adjustments
        self._task_type_adjustments = {
            "debug": {
                "code_search": 1.5,  # More code search for debugging
                "file_discovery": 1.2,
                "documentation_search": 0.8,
            },
            "implement": {
                "documentation_search": 1.3,  # More docs for implementation
                "file_discovery": 1.1,
                "code_search": 0.9,
            },
            "refactor": {
                "file_discovery": 1.4,  # More files for refactoring
                "code_search": 1.2,
                "documentation_search": 0.7,
            },
            "architect": {
                "documentation_search": 1.4,  # More docs for architecture
                "file_discovery": 1.2,
                "code_search": 0.7,
            },
        }

    def calculate_budget(self, complexity: str, task_type: str | None = None) -> int:
        """
        Calculate appropriate token budget based on complexity and task type.

        Args:
            complexity: Task complexity ("low", "medium", "high", "very_high")
            task_type: Optional task type for fine-tuning ("debug", "implement", etc.)

        Returns:
            Total token budget for context preparation
        """
        if not self._settings.enable_dynamic_token_budget:
            # Use medium complexity as default when dynamic budgeting is disabled
            return self._base_budgets["medium"]

        # Get base budget for complexity
        base_budget = self._base_budgets.get(complexity, self._base_budgets["medium"])

        # Note: task_type adjustments are applied at the component allocation level
        # (see allocate_budget), not by changing the total budget here.

        logger.info(
            "budget_calculated", complexity=complexity, task_type=task_type, budget=base_budget
        )

        return base_budget

    def allocate_budget(
        self, total_budget: int, task_type: str | None = None, _complexity: str | None = None
    ) -> TokenBudgetAllocation:
        """
        Allocate total budget across components.

        Args:
            total_budget: Total token budget available
            task_type: Optional task type for allocation adjustments
            complexity: Optional complexity for allocation adjustments

        Returns:
            TokenBudgetAllocation with per-component budgets
        """
        # Start with default allocation percentages
        allocation_percentages = self._default_allocation.copy()

        # Apply task type adjustments if provided
        if task_type and task_type in self._task_type_adjustments:
            adjustments = self._task_type_adjustments[task_type]
            for component, multiplier in adjustments.items():
                if component in allocation_percentages:
                    allocation_percentages[component] *= multiplier

        # Normalize percentages to sum to 1.0
        total_percentage = sum(allocation_percentages.values())
        allocation_percentages = {
            k: v / total_percentage for k, v in allocation_percentages.items()
        }

        # Calculate actual token allocations
        allocation = TokenBudgetAllocation(
            total_budget=total_budget,
            query_analysis=int(total_budget * allocation_percentages["query_analysis"]),
            file_discovery=int(total_budget * allocation_percentages["file_discovery"]),
            documentation_search=int(total_budget * allocation_percentages["documentation_search"]),
            code_search=int(total_budget * allocation_percentages["code_search"]),
            context_assembly=int(total_budget * allocation_percentages["context_assembly"]),
            reserve=int(total_budget * allocation_percentages["reserve"]),
        )

        logger.info(
            "budget_allocated",
            total=total_budget,
            task_type=task_type,
            query_analysis=allocation.query_analysis,
            file_discovery=allocation.file_discovery,
            documentation_search=allocation.documentation_search,
            code_search=allocation.code_search,
        )

        return allocation

    def prioritize_files(
        self, files: list[EnhancedFileContext], budget: int, avg_tokens_per_file: int = 200
    ) -> tuple[list[EnhancedFileContext], list[str]]:
        """
        Prioritize files to fit within budget.

        Args:
            files: List of file contexts to prioritize
            budget: Token budget for files
            avg_tokens_per_file: Average tokens per file summary

        Returns:
            Tuple of (prioritized files, optimization details)
        """
        if not files:
            return [], []

        optimizations = []

        # Calculate how many files we can include
        max_files = budget // avg_tokens_per_file

        if len(files) <= max_files:
            # All files fit within budget
            return files, ["All files included within budget"]

        # Need to prioritize - sort by relevance score
        sorted_files = sorted(files, key=lambda f: f.relevance_score, reverse=True)

        # Take top files that fit in budget
        prioritized = sorted_files[:max_files]
        excluded_count = len(files) - max_files

        optimizations.append(
            f"Prioritized top {max_files} files by relevance (excluded {excluded_count} lower-scoring files)"
        )

        # Log excluded files for transparency
        if excluded_count > 0:
            min_included_score = prioritized[-1].relevance_score if prioritized else 0
            logger.info(
                "files_prioritized",
                included=max_files,
                excluded=excluded_count,
                min_score=min_included_score,
            )

        return prioritized, optimizations

    def prioritize_documentation(
        self, docs: list[EnhancedDocumentationContext], budget: int, avg_tokens_per_doc: int = 300
    ) -> tuple[list[EnhancedDocumentationContext], list[str]]:
        """
        Prioritize documentation to fit within budget.

        Args:
            docs: List of documentation contexts to prioritize
            budget: Token budget for documentation
            avg_tokens_per_doc: Average tokens per doc snippet

        Returns:
            Tuple of (prioritized docs, optimization details)
        """
        if not docs:
            return [], []

        optimizations = []

        # Calculate how many docs we can include
        max_docs = budget // avg_tokens_per_doc

        if len(docs) <= max_docs:
            # All docs fit within budget
            return docs, ["All documentation included within budget"]

        # Need to prioritize - sort by multiple factors:
        # 1. Official documentation first (if attribution available)
        # 2. Authority score
        # 3. Relevance score
        def doc_priority_score(doc: EnhancedDocumentationContext) -> float:
            score = doc.relevance_score

            # Boost official documentation
            if doc.attribution and doc.attribution.is_official:
                score += 0.3

            # Boost high authority sources
            if doc.attribution and doc.attribution.authority_score > 0.8:
                score += 0.2
            elif doc.attribution and doc.attribution.authority_score > 0.6:
                score += 0.1

            return score

        sorted_docs = sorted(docs, key=doc_priority_score, reverse=True)

        # Take top docs that fit in budget
        prioritized = sorted_docs[:max_docs]
        excluded_count = len(docs) - max_docs

        optimizations.append(
            f"Prioritized top {max_docs} documentation sources by authority and relevance (excluded {excluded_count})"
        )

        # Count official docs included
        official_count = sum(1 for d in prioritized if d.attribution and d.attribution.is_official)
        if official_count > 0:
            optimizations.append(f"Included {official_count} official documentation sources")

        logger.info(
            "docs_prioritized",
            included=max_docs,
            excluded=excluded_count,
            official_included=official_count,
        )

        return prioritized, optimizations

    def estimate_token_usage(
        self,
        files: list[EnhancedFileContext],
        docs: list[EnhancedDocumentationContext],
        code_search_results: list[Any],
        query_analysis: dict[str, Any],
    ) -> int:
        """
        Estimate total token usage for context.

        Args:
            files: File contexts
            docs: Documentation contexts
            code_search_results: Code search results
            query_analysis: Query analysis results

        Returns:
            Estimated token count
        """
        # Rough token estimation (1 token ≈ 4 characters)
        tokens = 0

        # Query analysis
        if query_analysis:
            analysis_text = str(query_analysis)
            tokens += len(analysis_text) // 4

        # Files (summary + key elements)
        for file in files:
            file_text = f"{file.path} {file.summary} {' '.join(file.key_elements)}"
            tokens += len(file_text) // 4

        # Documentation (title + snippet)
        for doc in docs:
            doc_text = f"{doc.title} {doc.snippet}"
            tokens += len(doc_text) // 4

        # Code search results
        for result in code_search_results:
            if hasattr(result, "results"):
                tokens += len(result.results) // 4

        # Add overhead for formatting (20%)
        tokens = int(tokens * 1.2)

        return tokens

    def create_usage_report(
        self, allocation: TokenBudgetAllocation, actual_usage: int, optimizations_applied: list[str]
    ) -> TokenBudgetUsage:
        """
        Create token budget usage report.

        Args:
            allocation: Original budget allocation
            actual_usage: Actual tokens used
            optimizations_applied: List of optimizations that were applied

        Returns:
            TokenBudgetUsage report
        """
        utilization = (
            (actual_usage / allocation.total_budget * 100) if allocation.total_budget > 0 else 0
        )

        usage = TokenBudgetUsage(
            allocated_budget=allocation.total_budget,
            actual_usage=actual_usage,
            utilization_percentage=utilization,
            component_allocation={
                "query_analysis": allocation.query_analysis,
                "file_discovery": allocation.file_discovery,
                "documentation_search": allocation.documentation_search,
                "code_search": allocation.code_search,
                "context_assembly": allocation.context_assembly,
                "reserve": allocation.reserve,
            },
            component_usage={},  # Would be populated by actual component tracking
            optimization_applied=len(optimizations_applied) > 0,
            optimization_details=optimizations_applied,
        )

        logger.info(
            "budget_usage_report",
            allocated=allocation.total_budget,
            used=actual_usage,
            utilization=f"{utilization:.1f}%",
            optimizations=len(optimizations_applied),
        )

        return usage


# Global singleton with thread-safe initialization
_token_budget_manager: TokenBudgetManager | None = None
_manager_lock = __import__("threading").Lock()


def get_token_budget_manager() -> TokenBudgetManager:
    """Get the global TokenBudgetManager singleton (thread-safe)."""
    global _token_budget_manager

    # Fast path: already initialized
    if _token_budget_manager is not None:
        return _token_budget_manager

    # Thread-safe initialization
    with _manager_lock:
        if _token_budget_manager is None:
            _token_budget_manager = TokenBudgetManager()

    return _token_budget_manager


class GeminiContentRanker:
    """
    Uses Gemini to intelligently rank and filter content for optimal relevance.

    Responsibilities:
    - Rank documentation snippets by relevance to query
    - Summarize code search patterns
    - Filter low-value content
    - Optimize content for token efficiency
    """

    def __init__(self):
        """Initialize GeminiContentRanker with settings."""
        self._settings = get_settings()
        self._use_new_api = types is not None  # Track which API version we're using

        # Configure Gemini
        if not GOOGLE_AI_AVAILABLE:
            self._model = None
            logger.warning(
                "gemini_ranker_no_package",
                msg="google-genai not installed, ranking will use fallback",
            )
        elif self._settings.google_api_key:
            if self._use_new_api:
                # New google-genai package (preferred)
                self._model = genai.Client(api_key=self._settings.google_api_key)
                logger.info("gemini_ranker_initialized", api_version="new (google-genai)")
            else:
                # Fallback to deprecated google.generativeai package
                genai.configure(api_key=self._settings.google_api_key)
                self._model = genai.GenerativeModel("gemini-2.0-flash-exp")
                logger.info("gemini_ranker_initialized", api_version="legacy (google.generativeai)")
        else:
            self._model = None
            logger.warning(
                "gemini_ranker_no_api_key",
                msg="Gemini API key not configured, ranking will use fallback",
            )

    async def rank_documentation(
        self, query: str, docs: list[EnhancedDocumentationContext], max_docs: int = 5
    ) -> tuple[list[EnhancedDocumentationContext], list[str]]:
        """
        Use Gemini to rank documentation snippets by relevance.

        Args:
            query: User's query
            docs: List of documentation contexts
            max_docs: Maximum number of docs to return

        Returns:
            Tuple of (ranked docs, optimization details)
        """
        if not docs:
            return [], []

        if not self._model:
            # Fallback to simple relevance score sorting
            sorted_docs = sorted(docs, key=lambda d: d.relevance_score, reverse=True)
            return sorted_docs[:max_docs], [
                "Fallback: sorted by relevance score (Gemini unavailable)"
            ]

        optimizations = []

        try:
            # Prepare documentation summaries for Gemini
            doc_summaries = []
            for i, doc in enumerate(docs):
                summary = f"{i}. {doc.title} ({doc.source}): {doc.snippet[:200]}..."
                doc_summaries.append(summary)

            # Ask Gemini to rank by relevance
            prompt = f"""Given this user query: "{query}"

Rank these documentation sources by relevance (most relevant first).
Return ONLY the indices (0-{len(docs) - 1}) in order, comma-separated.

Documentation sources:
{chr(10).join(doc_summaries)}

Ranking (indices only, comma-separated):"""

            # Use appropriate API based on which package is available
            if self._use_new_api:
                # New google-genai API
                model_name = self._settings.routing_model or "gemini-2.0-flash-exp"
                response = await asyncio.to_thread(
                    self._model.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.3),
                )
            else:
                # Legacy google.generativeai API
                response = await self._model.generate_content_async(prompt)

            ranking_text = response.text.strip()

            # Parse ranking indices
            try:
                indices = [int(idx.strip()) for idx in ranking_text.split(",")]
                # Validate indices
                indices = [idx for idx in indices if 0 <= idx < len(docs)]

                # Reorder docs based on Gemini ranking
                ranked_docs = [docs[idx] for idx in indices[:max_docs]]

                optimizations.append(
                    f"Gemini ranked {len(docs)} docs, selected top {len(ranked_docs)}"
                )

                logger.info(
                    "gemini_docs_ranked", total=len(docs), ranked=len(ranked_docs), query=query[:50]
                )

                return ranked_docs, optimizations

            except (ValueError, IndexError) as e:
                logger.warning("gemini_ranking_parse_failed", error=str(e), response=ranking_text)
                # Fallback to relevance score
                sorted_docs = sorted(docs, key=lambda d: d.relevance_score, reverse=True)
                return sorted_docs[:max_docs], [
                    "Fallback: Gemini ranking parse failed, using relevance scores"
                ]

        except Exception as e:
            logger.warning("gemini_ranking_failed", error=str(e))
            # Fallback to relevance score sorting
            sorted_docs = sorted(docs, key=lambda d: d.relevance_score, reverse=True)
            return sorted_docs[:max_docs], [
                f"Fallback: Gemini ranking failed ({str(e)}), using relevance scores"
            ]

    async def summarize_code_patterns(
        self, query: str, code_search_results: list[Any], max_length: int = 500
    ) -> tuple[str, list[str]]:
        """
        Use Gemini to summarize code search patterns.

        Args:
            query: User's query
            code_search_results: Raw code search results
            max_length: Maximum length of summary

        Returns:
            Tuple of (summary text, optimization details)
        """
        if not code_search_results:
            return "", []

        if not self._model:
            # Fallback to truncation
            combined = "\n".join([str(r.results)[:200] for r in code_search_results[:3]])
            return combined[:max_length], ["Fallback: truncated results (Gemini unavailable)"]

        optimizations = []

        try:
            # Combine code search results
            results_text = []
            for result in code_search_results[:5]:  # Limit to first 5 results
                if hasattr(result, "results"):
                    results_text.append(f"{result.search_type}: {result.results[:500]}")

            combined_results = "\n\n".join(results_text)

            # Ask Gemini to summarize patterns
            prompt = f"""Given this user query: "{query}"

Summarize the key patterns found in these code search results.
Focus on: common patterns, important findings, relevant code locations.
Keep summary under {max_length} characters.

Code search results:
{combined_results}

Summary:"""

            # Use appropriate API based on which package is available
            if self._use_new_api:
                # New google-genai API
                model_name = self._settings.routing_model or "gemini-2.0-flash-exp"
                response = await asyncio.to_thread(
                    self._model.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.3),
                )
            else:
                # Legacy google.generativeai API
                response = await self._model.generate_content_async(prompt)

            summary = response.text.strip()

            # Ensure summary fits within max_length
            if len(summary) > max_length:
                summary = summary[: max_length - 3] + "..."

            optimizations.append(
                f"Gemini summarized {len(code_search_results)} code search results into {len(summary)} chars"
            )

            logger.info(
                "gemini_code_summarized",
                results=len(code_search_results),
                summary_length=len(summary),
            )

            return summary, optimizations

        except Exception as e:
            logger.warning("gemini_summarization_failed", error=str(e))
            # Fallback to truncation
            combined = "\n".join([str(r.results)[:200] for r in code_search_results[:3]])
            return combined[:max_length], [
                f"Fallback: Gemini summarization failed ({str(e)}), using truncation"
            ]

    async def filter_low_value_content(
        self, _query: str, files: list[EnhancedFileContext], relevance_threshold: float = 0.3
    ) -> tuple[list[EnhancedFileContext], list[str]]:
        """
        Filter out low-value files based on relevance.

        Args:
            query: User's query
            files: List of file contexts
            relevance_threshold: Minimum relevance score to keep

        Returns:
            Tuple of (filtered files, optimization details)
        """
        if not files:
            return [], []

        optimizations = []

        # Filter by relevance threshold
        filtered = [f for f in files if f.relevance_score >= relevance_threshold]
        removed_count = len(files) - len(filtered)

        if removed_count > 0:
            optimizations.append(
                f"Filtered {removed_count} low-relevance files (threshold: {relevance_threshold})"
            )

            logger.info(
                "low_value_filtered",
                original=len(files),
                filtered=len(filtered),
                removed=removed_count,
                threshold=relevance_threshold,
            )

        return filtered, optimizations

    async def optimize_content_for_budget(
        self,
        query: str,
        files: list[EnhancedFileContext],
        docs: list[EnhancedDocumentationContext],
        code_search_results: list[Any],
        budget: int,
    ) -> tuple[list[EnhancedFileContext], list[EnhancedDocumentationContext], str, list[str]]:
        """
        Optimize all content to fit within token budget using Gemini intelligence.

        Args:
            query: User's query
            files: File contexts
            docs: Documentation contexts
            code_search_results: Code search results
            budget: Total token budget

        Returns:
            Tuple of (optimized files, optimized docs, code summary, optimization details)
        """
        all_optimizations = []

        # Allocate budget across content types
        file_budget = int(budget * 0.4)
        doc_budget = int(budget * 0.4)
        code_budget = int(budget * 0.2)

        # Rank and filter documentation
        ranked_docs, doc_opts = await self.rank_documentation(query, docs, max_docs=5)
        all_optimizations.extend(doc_opts)

        # Filter low-value files
        filtered_files, file_opts = await self.filter_low_value_content(
            query, files, relevance_threshold=0.3
        )
        all_optimizations.extend(file_opts)

        # Summarize code patterns
        code_summary, code_opts = await self.summarize_code_patterns(
            query, code_search_results, max_length=code_budget * 4
        )
        all_optimizations.extend(code_opts)

        # Further prioritize files if needed
        if filtered_files:
            prioritized_files, prio_opts = TokenBudgetManager().prioritize_files(
                filtered_files, file_budget, avg_tokens_per_file=200
            )
            all_optimizations.extend(prio_opts)
        else:
            prioritized_files = []

        # Further prioritize docs if needed
        if ranked_docs:
            prioritized_docs, doc_prio_opts = TokenBudgetManager().prioritize_documentation(
                ranked_docs, doc_budget, avg_tokens_per_doc=300
            )
            all_optimizations.extend(doc_prio_opts)
        else:
            prioritized_docs = []

        logger.info(
            "content_optimized",
            original_files=len(files),
            optimized_files=len(prioritized_files),
            original_docs=len(docs),
            optimized_docs=len(prioritized_docs),
            code_summary_length=len(code_summary),
            budget=budget,
        )

        return prioritized_files, prioritized_docs, code_summary, all_optimizations


# Global singleton for Gemini content ranker
_gemini_ranker: GeminiContentRanker | None = None
_ranker_lock = __import__("threading").Lock()


def get_gemini_content_ranker() -> GeminiContentRanker:
    """Get the global GeminiContentRanker singleton (thread-safe)."""
    global _gemini_ranker

    # Fast path: already initialized
    if _gemini_ranker is not None:
        return _gemini_ranker

    # Thread-safe initialization
    with _ranker_lock:
        if _gemini_ranker is None:
            _gemini_ranker = GeminiContentRanker()

    return _gemini_ranker
