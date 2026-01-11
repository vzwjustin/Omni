"""
Prometheus Metrics for Omni-Cortex

Provides observability for framework execution, routing decisions,
and system performance.

Requires prometheus_client to be installed (included in requirements.txt).
"""

import structlog
from prometheus_client import Counter, Gauge, Histogram, Info

logger = structlog.get_logger("metrics")

PROMETHEUS_AVAILABLE = True


# =============================================================================
# Framework Execution Metrics
# =============================================================================

FRAMEWORK_EXECUTIONS = Counter(
    "omni_cortex_framework_executions_total",
    "Total framework executions",
    ["framework", "category", "success"],
)

FRAMEWORK_DURATION = Histogram(
    "omni_cortex_framework_duration_seconds",
    "Framework execution duration in seconds",
    ["framework", "category"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

FRAMEWORK_TOKENS = Histogram(
    "omni_cortex_framework_tokens_used",
    "Tokens used per framework execution",
    ["framework"],
    buckets=(100, 500, 1000, 2000, 4000, 8000, 16000, 32000),
)

FRAMEWORK_CONFIDENCE = Histogram(
    "omni_cortex_framework_confidence",
    "Framework confidence scores",
    ["framework"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# =============================================================================
# Router Metrics
# =============================================================================

ROUTER_DECISIONS = Counter(
    "omni_cortex_router_decisions_total",
    "Total routing decisions",
    ["category", "method"],  # method: hierarchical_ai, heuristic, user_specified
)

ROUTER_DURATION = Histogram(
    "omni_cortex_router_duration_seconds",
    "Router decision duration in seconds",
    ["method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

ROUTER_CHAIN_LENGTH = Histogram(
    "omni_cortex_router_chain_length",
    "Number of frameworks in selected chain",
    buckets=(1, 2, 3, 4, 5),
)

# =============================================================================
# MCP Tool Metrics
# =============================================================================

MCP_REQUESTS = Counter(
    "omni_cortex_mcp_requests_total", "Total MCP tool requests", ["tool", "success"]
)

MCP_REQUEST_DURATION = Histogram(
    "omni_cortex_mcp_request_duration_seconds",
    "MCP request duration in seconds",
    ["tool"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# =============================================================================
# Rate Limiter Metrics
# =============================================================================

RATE_LIMIT_REJECTIONS = Counter(
    "omni_cortex_rate_limit_rejections_total", "Total rate limit rejections", ["tool", "category"]
)

RATE_LIMIT_TOKENS = Gauge(
    "omni_cortex_rate_limit_tokens", "Current tokens in rate limit bucket", ["category"]
)

# =============================================================================
# Memory & Cache Metrics
# =============================================================================

MEMORY_THREADS = Gauge("omni_cortex_memory_threads_active", "Number of active memory threads")

CACHE_HITS = Counter(
    "omni_cortex_cache_hits_total",
    "Cache hits",
    ["cache_type"],  # embedding, router, rag, context_cache
)

CACHE_MISSES = Counter("omni_cortex_cache_misses_total", "Cache misses", ["cache_type"])

# Context Cache Effectiveness Metrics
CONTEXT_CACHE_HITS = Counter(
    "omni_cortex_context_cache_hits_total",
    "Context cache hits by cache type",
    [
        "cache_type",
        "stale",
    ],  # cache_type: query_analysis, file_discovery, documentation; stale: true/false
)

CONTEXT_CACHE_MISSES = Counter(
    "omni_cortex_context_cache_misses_total", "Context cache misses by cache type", ["cache_type"]
)

CONTEXT_CACHE_TOKENS_SAVED = Counter(
    "omni_cortex_context_cache_tokens_saved_total",
    "Total tokens saved by context cache",
    ["cache_type"],
)

CONTEXT_CACHE_SIZE = Gauge(
    "omni_cortex_context_cache_size_bytes", "Current context cache size in bytes"
)

CONTEXT_CACHE_ENTRIES = Gauge(
    "omni_cortex_context_cache_entries",
    "Number of entries in context cache by type",
    ["cache_type"],
)

CONTEXT_CACHE_AGE = Histogram(
    "omni_cortex_context_cache_age_seconds",
    "Age of cache entries when accessed",
    ["cache_type"],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400),
)

CONTEXT_CACHE_INVALIDATIONS = Counter(
    "omni_cortex_context_cache_invalidations_total",
    "Number of cache invalidations",
    ["reason"],  # workspace_change, ttl_expired, size_limit, manual
)

CONTEXT_CACHE_HIT_RATE = Gauge(
    "omni_cortex_context_cache_hit_rate",
    "Cache hit rate by cache type (0.0 to 1.0)",
    ["cache_type"],
)

# Context Relevance Tracking Metrics
CONTEXT_RELEVANCE_USAGE_RATE = Histogram(
    "omni_cortex_context_relevance_usage_rate",
    "Usage rate of context elements (0.0 to 1.0)",
    ["element_type"],  # file, documentation, code_search
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

CONTEXT_RELEVANCE_ELEMENTS_TRACKED = Gauge(
    "omni_cortex_context_relevance_elements_tracked",
    "Number of context elements being tracked",
    ["element_type"],
)

CONTEXT_RELEVANCE_HIGH_VALUE = Gauge(
    "omni_cortex_context_relevance_high_value_count",
    "Number of high-value context elements (usage_rate > 0.7)",
    ["element_type"],
)

CONTEXT_RELEVANCE_LOW_VALUE = Gauge(
    "omni_cortex_context_relevance_low_value_count",
    "Number of low-value context elements (usage_rate < 0.2)",
    ["element_type"],
)

CONTEXT_RELEVANCE_OPTIMIZATIONS = Counter(
    "omni_cortex_context_relevance_optimizations_total",
    "Number of relevance score optimizations applied",
    ["task_type", "adjustment_type"],  # adjustment_type: boosted, reduced, unchanged
)

CONTEXT_RELEVANCE_SESSIONS = Counter(
    "omni_cortex_context_relevance_sessions_total",
    "Total context usage sessions tracked",
    ["task_type"],
)

CONTEXT_RELEVANCE_ELEMENT_USAGE = Counter(
    "omni_cortex_context_relevance_element_usage_total",
    "Total times context elements were actually used in solutions",
    ["element_type"],
)

# =============================================================================
# RAG Metrics
# =============================================================================

RAG_SEARCHES = Counter("omni_cortex_rag_searches_total", "Total RAG searches", ["collection"])

RAG_SEARCH_DURATION = Histogram(
    "omni_cortex_rag_search_duration_seconds",
    "RAG search duration in seconds",
    ["collection"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

RAG_RESULTS_COUNT = Histogram(
    "omni_cortex_rag_results_count",
    "Number of results returned from RAG search",
    ["collection"],
    buckets=(0, 1, 2, 3, 5, 10, 20),
)

# =============================================================================
# System Info
# =============================================================================

SYSTEM_INFO = Info("omni_cortex", "Omni-Cortex system information")


# =============================================================================
# Helper Functions
# =============================================================================


def record_framework_execution(
    framework: str,
    category: str,
    duration_seconds: float,
    tokens_used: int,
    confidence: float,
    success: bool = True,
) -> None:
    """Record metrics for a framework execution."""
    FRAMEWORK_EXECUTIONS.labels(
        framework=framework, category=category, success=str(success).lower()
    ).inc()

    FRAMEWORK_DURATION.labels(framework=framework, category=category).observe(duration_seconds)

    FRAMEWORK_TOKENS.labels(framework=framework).observe(tokens_used)
    FRAMEWORK_CONFIDENCE.labels(framework=framework).observe(confidence)


def record_router_decision(
    category: str, method: str, duration_seconds: float, chain_length: int
) -> None:
    """Record metrics for a routing decision."""
    ROUTER_DECISIONS.labels(category=category, method=method).inc()
    ROUTER_DURATION.labels(method=method).observe(duration_seconds)
    ROUTER_CHAIN_LENGTH.observe(chain_length)


def record_mcp_request(tool: str, duration_seconds: float, success: bool = True) -> None:
    """Record metrics for an MCP tool request."""
    MCP_REQUESTS.labels(tool=tool, success=str(success).lower()).inc()
    MCP_REQUEST_DURATION.labels(tool=tool).observe(duration_seconds)


def record_rate_limit_rejection(tool: str, category: str) -> None:
    """Record a rate limit rejection."""
    RATE_LIMIT_REJECTIONS.labels(tool=tool, category=category).inc()


def record_cache_access(cache_type: str, hit: bool) -> None:
    """Record a cache hit or miss."""
    if hit:
        CACHE_HITS.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISSES.labels(cache_type=cache_type).inc()


def record_context_cache_access(
    cache_type: str,
    hit: bool,
    tokens_saved: int = 0,
    cache_age_seconds: float = 0.0,
    is_stale: bool = False,
) -> None:
    """
    Record context cache access with detailed metrics.

    Args:
        cache_type: Type of cache (query_analysis, file_discovery, documentation)
        hit: Whether it was a cache hit
        tokens_saved: Number of tokens saved by cache hit
        cache_age_seconds: Age of cache entry in seconds
        is_stale: Whether this was a stale cache fallback
    """
    if hit:
        CONTEXT_CACHE_HITS.labels(cache_type=cache_type, stale=str(is_stale).lower()).inc()

        if tokens_saved > 0:
            CONTEXT_CACHE_TOKENS_SAVED.labels(cache_type=cache_type).inc(tokens_saved)

        if cache_age_seconds > 0:
            CONTEXT_CACHE_AGE.labels(cache_type=cache_type).observe(cache_age_seconds)
    else:
        CONTEXT_CACHE_MISSES.labels(cache_type=cache_type).inc()


def record_context_cache_invalidation(reason: str, count: int = 1) -> None:
    """
    Record cache invalidation events.

    Args:
        reason: Reason for invalidation (workspace_change, ttl_expired, size_limit, manual)
        count: Number of entries invalidated
    """
    CONTEXT_CACHE_INVALIDATIONS.labels(reason=reason).inc(count)


def update_context_cache_stats(
    total_entries: int, entries_by_type: dict, size_bytes: int, hit_rates: dict
) -> None:
    """
    Update context cache statistics gauges.

    Args:
        total_entries: Total number of cache entries
        entries_by_type: Dictionary of entry counts by cache type
        size_bytes: Total cache size in bytes
        hit_rates: Dictionary of hit rates by cache type (0.0 to 1.0)
    """
    CONTEXT_CACHE_SIZE.set(size_bytes)

    for cache_type, count in entries_by_type.items():
        CONTEXT_CACHE_ENTRIES.labels(cache_type=cache_type).set(count)

    for cache_type, hit_rate in hit_rates.items():
        CONTEXT_CACHE_HIT_RATE.labels(cache_type=cache_type).set(hit_rate)


def record_relevance_tracking(
    element_type: str,
    usage_rate: float,
    elements_tracked: int,
    high_value_count: int,
    low_value_count: int,
) -> None:
    """
    Record context relevance tracking metrics.

    Args:
        element_type: Type of element (file, documentation, code_search)
        usage_rate: Average usage rate for this element type
        elements_tracked: Number of elements being tracked
        high_value_count: Number of high-value elements
        low_value_count: Number of low-value elements
    """
    CONTEXT_RELEVANCE_USAGE_RATE.labels(element_type=element_type).observe(usage_rate)
    CONTEXT_RELEVANCE_ELEMENTS_TRACKED.labels(element_type=element_type).set(elements_tracked)
    CONTEXT_RELEVANCE_HIGH_VALUE.labels(element_type=element_type).set(high_value_count)
    CONTEXT_RELEVANCE_LOW_VALUE.labels(element_type=element_type).set(low_value_count)


def record_relevance_optimization(task_type: str, adjustment_type: str, count: int = 1) -> None:
    """
    Record relevance score optimization.

    Args:
        task_type: Type of task (debug, implement, etc.)
        adjustment_type: Type of adjustment (boosted, reduced, unchanged)
        count: Number of optimizations applied
    """
    CONTEXT_RELEVANCE_OPTIMIZATIONS.labels(
        task_type=task_type, adjustment_type=adjustment_type
    ).inc(count)


def record_relevance_session(task_type: str, elements_included: int, elements_used: int) -> None:
    """
    Record a context usage session.

    Args:
        task_type: Type of task
        elements_included: Number of elements included in context
        elements_used: Number of elements actually used in solution
    """
    CONTEXT_RELEVANCE_SESSIONS.labels(task_type=task_type).inc()

    # Record element usage
    if elements_used > 0:
        CONTEXT_RELEVANCE_ELEMENT_USAGE.labels(element_type="all").inc(elements_used)


def record_rag_search(collection: str, duration_seconds: float, results_count: int) -> None:
    """Record metrics for a RAG search."""
    RAG_SEARCHES.labels(collection=collection).inc()
    RAG_SEARCH_DURATION.labels(collection=collection).observe(duration_seconds)
    RAG_RESULTS_COUNT.labels(collection=collection).observe(results_count)


def set_system_info(version: str, framework_count: int, lean_mode: bool, llm_provider: str) -> None:
    """Set system information metrics."""
    SYSTEM_INFO.info(
        {
            "version": version,
            "framework_count": str(framework_count),
            "lean_mode": str(lean_mode).lower(),
            "llm_provider": llm_provider,
        }
    )


def update_memory_threads(count: int) -> None:
    """Update the active memory threads gauge."""
    MEMORY_THREADS.set(count)


def is_prometheus_available() -> bool:
    """Check if Prometheus metrics are available."""
    return PROMETHEUS_AVAILABLE
