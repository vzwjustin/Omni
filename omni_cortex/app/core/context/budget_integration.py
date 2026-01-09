"""
Token Budget Integration for Context Gateway

Integrates TokenBudgetManager and GeminiContentRanker into the context preparation flow.
"""

import structlog
from typing import List, Dict, Any, Optional

from .enhanced_models import (
    EnhancedFileContext,
    EnhancedDocumentationContext,
    TokenBudgetUsage,
)
from .token_budget_manager import (
    TokenBudgetManager,
    get_token_budget_manager,
    GeminiContentRanker,
    get_gemini_content_ranker,
)
from ..settings import get_settings

logger = structlog.get_logger("budget_integration")


class BudgetIntegration:
    """
    Integrates token budget management into context preparation.
    
    Responsibilities:
    - Calculate appropriate budget for task
    - Optimize content to fit budget
    - Track budget utilization
    - Provide transparency metrics
    """
    
    def __init__(
        self,
        budget_manager: Optional[TokenBudgetManager] = None,
        content_ranker: Optional[GeminiContentRanker] = None
    ):
        """
        Initialize BudgetIntegration.
        
        Args:
            budget_manager: Optional TokenBudgetManager instance (for testing)
            content_ranker: Optional GeminiContentRanker instance (for testing)
        """
        self._budget_manager = budget_manager or get_token_budget_manager()
        self._content_ranker = content_ranker or get_gemini_content_ranker()
        self._settings = get_settings()
    
    async def optimize_context_for_budget(
        self,
        query: str,
        task_type: str,
        complexity: str,
        files: List[EnhancedFileContext],
        docs: List[EnhancedDocumentationContext],
        code_search_results: List[Any],
    ) -> tuple[
        List[EnhancedFileContext],
        List[EnhancedDocumentationContext],
        List[Any],
        TokenBudgetUsage
    ]:
        """
        Optimize all context content to fit within appropriate token budget.
        
        Args:
            query: User's query
            task_type: Type of task (debug, implement, etc.)
            complexity: Task complexity (low, medium, high, very_high)
            files: File contexts
            docs: Documentation contexts
            code_search_results: Code search results
        
        Returns:
            Tuple of (optimized files, optimized docs, optimized code results, budget usage)
        """
        if not self._settings.enable_content_optimization:
            # No optimization - return original content
            logger.info("content_optimization_disabled")
            
            # Still calculate budget for transparency
            budget = self._budget_manager.calculate_budget(complexity, task_type)
            allocation = self._budget_manager.allocate_budget(budget, task_type, complexity)
            actual_usage = self._budget_manager.estimate_token_usage(
                files, docs, code_search_results, {}
            )
            
            usage = self._budget_manager.create_usage_report(
                allocation,
                actual_usage,
                ["Content optimization disabled"]
            )
            
            return files, docs, code_search_results, usage
        
        # Calculate appropriate budget
        budget = self._budget_manager.calculate_budget(complexity, task_type)
        allocation = self._budget_manager.allocate_budget(budget, task_type, complexity)
        
        logger.info(
            "budget_optimization_start",
            budget=budget,
            complexity=complexity,
            task_type=task_type,
            files=len(files),
            docs=len(docs),
            code_results=len(code_search_results)
        )
        
        # Use Gemini to optimize content
        optimized_files, optimized_docs, code_summary, optimizations = \
            await self._content_ranker.optimize_content_for_budget(
                query,
                files,
                docs,
                code_search_results,
                budget
            )
        
        # Create optimized code search results with summary
        optimized_code_results = []
        if code_summary:
            # Create a single summarized result
            from ..context_gateway import CodeSearchContext
            optimized_code_results = [
                CodeSearchContext(
                    search_type="gemini_summary",
                    query=query,
                    results=code_summary,
                    file_count=len(code_search_results),
                    match_count=0
                )
            ]
        
        # Estimate actual token usage
        actual_usage = self._budget_manager.estimate_token_usage(
            optimized_files,
            optimized_docs,
            optimized_code_results,
            {}
        )
        
        # Create usage report
        usage = self._budget_manager.create_usage_report(
            allocation,
            actual_usage,
            optimizations
        )
        
        logger.info(
            "budget_optimization_complete",
            original_files=len(files),
            optimized_files=len(optimized_files),
            original_docs=len(docs),
            optimized_docs=len(optimized_docs),
            budget=budget,
            actual_usage=actual_usage,
            utilization=f"{usage.utilization_percentage:.1f}%"
        )
        
        return optimized_files, optimized_docs, optimized_code_results, usage
    
    async def optimize_files_only(
        self,
        query: str,
        files: List[EnhancedFileContext],
        budget: int
    ) -> tuple[List[EnhancedFileContext], List[str]]:
        """
        Optimize just files to fit within budget.
        
        Args:
            query: User's query
            files: File contexts
            budget: Token budget for files
        
        Returns:
            Tuple of (optimized files, optimization details)
        """
        # Filter low-value files first
        filtered_files, filter_opts = await self._content_ranker.filter_low_value_content(
            query, files, relevance_threshold=0.3
        )
        
        # Then prioritize by budget
        prioritized_files, prio_opts = self._budget_manager.prioritize_files(
            filtered_files, budget, avg_tokens_per_file=200
        )
        
        all_opts = filter_opts + prio_opts
        
        return prioritized_files, all_opts
    
    async def optimize_docs_only(
        self,
        query: str,
        docs: List[EnhancedDocumentationContext],
        budget: int
    ) -> tuple[List[EnhancedDocumentationContext], List[str]]:
        """
        Optimize just documentation to fit within budget.
        
        Args:
            query: User's query
            docs: Documentation contexts
            budget: Token budget for docs
        
        Returns:
            Tuple of (optimized docs, optimization details)
        """
        # Use Gemini to rank docs
        max_docs = budget // 300  # Avg 300 tokens per doc
        ranked_docs, rank_opts = await self._content_ranker.rank_documentation(
            query, docs, max_docs=max_docs
        )
        
        return ranked_docs, rank_opts


# Global singleton
_budget_integration: Optional[BudgetIntegration] = None
_integration_lock = __import__('threading').Lock()


def get_budget_integration() -> BudgetIntegration:
    """Get the global BudgetIntegration singleton (thread-safe)."""
    global _budget_integration
    
    # Fast path: already initialized
    if _budget_integration is not None:
        return _budget_integration
    
    # Thread-safe initialization
    with _integration_lock:
        if _budget_integration is None:
            _budget_integration = BudgetIntegration()
    
    return _budget_integration
