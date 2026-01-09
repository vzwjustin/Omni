"""
Context Module: Specialized Context Gathering Components

This module provides focused, single-responsibility classes for context preparation:

- QueryAnalyzer: Analyzes queries using Gemini to understand intent
- FileDiscoverer: Discovers relevant files in workspace using Gemini scoring
- DocumentationSearcher: Searches web and knowledge base for documentation
- CodeSearcher: Searches codebase using grep/ripgrep and git

Enhanced functionality includes:
- Intelligent caching with TTL management
- Streaming progress events
- Multi-repository discovery
- Enhanced documentation grounding with source attribution
- Comprehensive metrics and monitoring
- Token budget management
- Advanced resilience patterns

Usage:
    from app.core.context import (
        QueryAnalyzer,
        FileDiscoverer,
        DocumentationSearcher,
        CodeSearcher,
    )

    # Enhanced models and utilities
    from app.core.context.enhanced_models import (
        EnhancedStructuredContext,
        ProgressEvent,
        RepoInfo,
        SourceAttribution,
    )

    analyzer = QueryAnalyzer()
    analysis = await analyzer.analyze("Fix the authentication bug")
"""

from .query_analyzer import QueryAnalyzer
from .file_discoverer import FileDiscoverer, FileContext
from .doc_searcher import DocumentationSearcher, DocumentationContext
from .code_searcher import CodeSearcher, CodeSearchContext
from .multi_repo_discoverer import MultiRepoFileDiscoverer

# Enhanced models and utilities
from .enhanced_models import (
    # Cache models
    CacheEntry,
    CacheMetadata,
    
    # Streaming models
    ProgressEvent,
    ProgressStatus,
    
    # Multi-repository models
    RepoInfo,
    CrossRepoDependency,
    
    # Enhanced documentation models
    SourceAttribution,
    EnhancedDocumentationContext,
    
    # Metrics models
    ComponentMetrics,
    QualityMetrics,
    ContextGatewayMetrics,
    
    # Token budget models
    TokenBudgetAllocation,
    TokenBudgetUsage,
    
    # Circuit breaker models
    CircuitBreakerState,
    CircuitBreakerStatus,
    ComponentStatus,
    ComponentStatusInfo,
    
    # Enhanced context models
    EnhancedFileContext,
    EnhancedStructuredContext,
)

from .context_cache import (
    ContextCache,
    get_context_cache,
    reset_context_cache,
)

from .relevance_tracker import (
    RelevanceTracker,
    get_relevance_tracker,
    ElementUsage,
    ContextUsageSession,
)

from .token_budget_manager import (
    TokenBudgetManager,
    get_token_budget_manager,
    GeminiContentRanker,
    get_gemini_content_ranker,
)

from .budget_integration import (
    BudgetIntegration,
    get_budget_integration,
)

from .gateway_metrics import (
    ContextGatewayMetrics as GatewayMetricsClass,
    get_gateway_metrics,
    reset_gateway_metrics,
    APICallMetrics,
    ComponentPerformanceMetrics,
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker,
    get_all_circuit_breakers,
    reset_all_circuit_breakers,
)

from .fallback_analysis import (
    EnhancedFallbackAnalyzer,
    ComponentFallbackMethods,
    FallbackQualityIndicator,
    get_fallback_analyzer,
)

from .status_tracking import (
    ComponentStatusTracker,
    DetailedErrorReport,
    StatusFormatter,
    get_status_tracker,
    reset_status_tracker,
)

from .thinking_mode_optimizer import (
    ThinkingModeOptimizer,
    ThinkingLevel,
    ThinkingModeMetrics,
    ThinkingModeDecision,
    get_thinking_mode_optimizer,
)

# Note: enhanced_utils will be implemented in future tasks
# from .enhanced_utils import (
#     CacheKeyGenerator,
#     TokenBudgetCalculator,
#     QualityScorer,
#     CircuitBreakerUtils,
#     MultiRepoUtils,
# )

__all__ = [
    # Original classes
    "QueryAnalyzer",
    "FileDiscoverer",
    "DocumentationSearcher",
    "CodeSearcher",
    "MultiRepoFileDiscoverer",
    
    # Original dataclasses
    "FileContext",
    "DocumentationContext",
    "CodeSearchContext",
    
    # Enhanced models - Cache
    "CacheEntry",
    "CacheMetadata",
    
    # Enhanced models - Streaming
    "ProgressEvent",
    "ProgressStatus",
    
    # Enhanced models - Multi-repository
    "RepoInfo",
    "CrossRepoDependency",
    
    # Enhanced models - Documentation
    "SourceAttribution",
    "EnhancedDocumentationContext",
    
    # Enhanced models - Metrics
    "ComponentMetrics",
    "QualityMetrics",
    "ContextGatewayMetrics",
    
    # Enhanced models - Token budget
    "TokenBudgetAllocation",
    "TokenBudgetUsage",
    
    # Enhanced models - Circuit breaker
    "CircuitBreakerState",
    "CircuitBreakerStatus",
    "ComponentStatus",
    "ComponentStatusInfo",
    
    # Enhanced models - Context
    "EnhancedFileContext",
    "EnhancedStructuredContext",
    
    # Cache system
    "ContextCache",
    "get_context_cache",
    "reset_context_cache",

    # Streaming gateway - NOT exported to avoid circular import
    # Import from app.core.context.streaming_gateway directly

    # Relevance tracking
    "RelevanceTracker",
    "get_relevance_tracker",
    "ElementUsage",
    "ContextUsageSession",
    
    # Token budget management
    "TokenBudgetManager",
    "get_token_budget_manager",
    "GeminiContentRanker",
    "get_gemini_content_ranker",
    "BudgetIntegration",
    "get_budget_integration",
    
    # Gateway metrics
    "GatewayMetricsClass",
    "get_gateway_metrics",
    "reset_gateway_metrics",
    "APICallMetrics",
    "ComponentPerformanceMetrics",
    
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "get_circuit_breaker",
    "get_all_circuit_breakers",
    "reset_all_circuit_breakers",
    
    # Fallback analysis
    "EnhancedFallbackAnalyzer",
    "ComponentFallbackMethods",
    "FallbackQualityIndicator",
    "get_fallback_analyzer",
    
    # Status tracking
    "ComponentStatusTracker",
    "DetailedErrorReport",
    "StatusFormatter",
    "get_status_tracker",
    "reset_status_tracker",
    
    # Thinking mode optimization
    "ThinkingModeOptimizer",
    "ThinkingLevel",
    "ThinkingModeMetrics",
    "ThinkingModeDecision",
    "get_thinking_mode_optimizer",
    
    # Enhanced utilities (to be implemented)
    # "CacheKeyGenerator",
    # "TokenBudgetCalculator",
    # "QualityScorer",
    # "CircuitBreakerUtils",
    # "MultiRepoUtils",
]

# Note: StreamingContextGateway is NOT imported here to avoid circular dependency
# (it inherits from ContextGateway which imports from this module).
# Import directly from app.core.context.streaming_gateway where needed.
