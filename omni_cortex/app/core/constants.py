"""
Constants for Omni-Cortex

Single source of truth for all magic numbers previously scattered across 20+ files.
Import from here instead of hardcoding values.

Usage:
    from app.core.constants import CONTENT, SEARCH, LIMITS

    # Instead of: doc.page_content[:500]
    doc.page_content[:CONTENT.SNIPPET_SHORT]

    # Instead of: search(k=3)
    search(k=SEARCH.K_STANDARD)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContentLimits:
    """Text/code content truncation limits (characters).

    These replace the scattered [:500], [:1000], [:1500], [:2000] slices.
    """

    # Display/logging - safe, non-critical truncation
    API_KEY_PREVIEW: int = 20  # chars shown for API key redaction
    QUERY_PREVIEW: int = 50  # very short query preview
    QUERY_LOG: int = 100  # query logging preview
    ERROR_PREVIEW: int = 200  # error message preview
    SUMMARY_PREVIEW: int = 400  # summary/outcome preview
    ERROR_FULL: int = 500  # full error message

    # RAG/Search result content
    SNIPPET_TINY: int = 100  # minimal content preview
    SNIPPET_SHORT: int = 500  # short snippet (most common)
    SNIPPET_MEDIUM: int = 800  # medium snippet
    SNIPPET_STANDARD: int = 1000  # standard content
    SNIPPET_LONG: int = 1200  # longer content
    SNIPPET_EXTENDED: int = 1500  # extended context
    SNIPPET_FULL: int = 1800  # full code context
    SNIPPET_MAX: int = 2000  # maximum single snippet

    # Command/output caps
    COMMAND_OUTPUT: int = 3000  # shell command output cap

    # Observation/answer limits for graph state
    OBSERVATION_LIMIT: int = 500  # state observation truncation


@dataclass(frozen=True)
class SearchLimits:
    """RAG and vector search limits."""

    # Default k values for retrieval
    K_SINGLE: int = 1  # single best match
    K_FEW: int = 2  # 2 results
    K_STANDARD: int = 3  # standard 3-result (most common)
    K_DEFAULT: int = 5  # default RAG retrieval
    K_EXTENDED: int = 10  # extended search

    # Search result filtering
    GREP_MAX_COUNT: int = 10  # max grep results per query
    SEARCH_QUERIES_MAX: int = 3  # max search queries to run
    GIT_LOG_MAX: int = 10  # max git log results
    VIBE_ITEMS_MAX: int = 4  # max vibe dict items to display
    ROUTER_SIGNALS_MAX: int = 3  # max detection signals


@dataclass(frozen=True)
class WorkspaceLimits:
    """File discovery and workspace analysis limits."""

    MAX_FILES_CONTEXT: int = 15  # max files in final context
    MAX_FILES_LIST: int = 50  # max from pre-specified list
    MAX_FILES_ANALYZE: int = 100  # max for LLM analysis/ranking
    MAX_FILES_SCAN: int = 200  # absolute cap for workspace scan

    # Exclude patterns (directories to skip)
    EXCLUDE_DIRS: tuple[str, ...] = (
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
        ".egg-info",
        ".tox",
        "htmlcov",
        ".coverage",
        ".idea",
        ".vscode",
    )

    # Include extensions (files to consider)
    CODE_EXTENSIONS: tuple[str, ...] = (
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".sql",
        ".sh",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".md",
        ".rst",
    )


@dataclass(frozen=True)
class ResourceLimits:
    """Memory, timeout, and batch processing limits."""

    # Memory (threads/conversations)
    MAX_MEMORY_THREADS: int = 100  # concurrent conversation threads
    MAX_MESSAGES_PER_THREAD: int = 20  # message history per thread

    # Execution timeouts (seconds)
    SANDBOX_TIMEOUT: float = 5.0  # code execution timeout
    SUBPROCESS_TIMEOUT: float = 5.0  # subprocess timeout (grep/find)
    GIT_TIMEOUT: float = 3.0  # git command timeout
    LLM_TIMEOUT: float = 30.0  # LLM sampling timeout
    TEST_TIMEOUT: float = 0.5  # test execution timeout

    # Batch processing
    INGEST_BATCH_SIZE: int = 100  # document ingestion batch size

    # Episode limits (CoALA framework)
    MAX_EPISODES: int = 100  # max stored episodes


@dataclass(frozen=True)
class FrameworkLimits:
    """Reasoning framework iteration/search limits."""

    MAX_REASONING_DEPTH: int = 10  # max chain-of-thought steps
    MCTS_MAX_ROLLOUTS: int = 50  # Monte Carlo tree search rollouts
    DEBATE_MAX_ROUNDS: int = 5  # max debate rounds
    TOT_MAX_BREADTH: int = 3  # Tree-of-Thought branching
    TOT_MAX_DEPTH: int = 5  # Tree-of-Thought depth


@dataclass(frozen=True)
class LLMParams:
    """LLM sampling parameters."""

    # Temperature settings
    ROUTING_TEMP: float = 0.7  # routing decision temp
    ANALYSIS_TEMP: float = 0.2  # deterministic analysis
    SEARCH_TEMP: float = 0.3  # documentation search
    CREATIVE_TEMP: float = 0.9  # creative generation

    # Sampling
    TOP_P: float = 1.0  # nucleus sampling
    TOP_K: int = 40  # top-k sampling

    # Token budgets
    CONTEXT_TOKEN_BUDGET: int = 50000  # max tokens for prepared context


@dataclass(frozen=True)
class CacheConfig:
    """Context cache configuration constants."""

    # TTL settings (seconds)
    QUERY_ANALYSIS_TTL: int = 3600  # 1 hour - query analysis results
    FILE_DISCOVERY_TTL: int = 1800  # 30 minutes - file relevance scores
    DOCUMENTATION_TTL: int = 86400  # 24 hours - documentation search results
    CODE_SEARCH_TTL: int = 1800  # 30 minutes - code search results

    # Cache size limits
    MAX_CACHE_ENTRIES: int = 1000  # maximum cache entries
    MAX_CACHE_SIZE_MB: int = 100  # maximum cache size in MB

    # Cache key settings
    QUERY_SIMILARITY_THRESHOLD: float = 0.85  # similarity threshold for cache hits
    WORKSPACE_FINGERPRINT_DEPTH: int = 3  # directory depth for fingerprinting

    # Stale cache settings
    STALE_CACHE_MAX_AGE: int = 604800  # 7 days - maximum age for stale cache fallback
    ENABLE_STALE_FALLBACK: bool = True  # enable serving stale cache on API failures


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Circuit breaker configuration constants."""

    # Failure thresholds
    FAILURE_THRESHOLD: int = 5  # failures before opening circuit
    SUCCESS_THRESHOLD: int = 3  # successes needed to close circuit in half-open

    # Timeout settings (seconds)
    RECOVERY_TIMEOUT: int = 60  # time before attempting recovery
    HALF_OPEN_TIMEOUT: int = 30  # timeout for half-open state

    # Backoff settings
    INITIAL_BACKOFF: float = 1.0  # initial backoff delay (seconds)
    MAX_BACKOFF: float = 300.0  # maximum backoff delay (5 minutes)
    BACKOFF_MULTIPLIER: float = 2.0  # exponential backoff multiplier
    JITTER_FACTOR: float = 0.1  # jitter factor for backoff randomization

    # Component-specific settings
    GEMINI_API_THRESHOLD: int = 3  # lower threshold for Gemini API
    FILE_SYSTEM_THRESHOLD: int = 10  # higher threshold for file system operations
    NETWORK_THRESHOLD: int = 5  # standard threshold for network operations


@dataclass(frozen=True)
class StreamingConfig:
    """Streaming context preparation configuration."""

    # Progress reporting
    PROGRESS_UPDATE_INTERVAL: float = 0.5  # seconds between progress updates
    MIN_PROGRESS_DELTA: float = 0.05  # minimum progress change to report

    # Estimation settings
    ENABLE_TIME_ESTIMATION: bool = True  # enable completion time estimation
    ESTIMATION_WINDOW_SIZE: int = 10  # number of recent operations for estimation

    # Cancellation settings
    CANCELLATION_CHECK_INTERVAL: float = 0.1  # seconds between cancellation checks
    CLEANUP_TIMEOUT: float = 5.0  # timeout for cleanup operations

    # Buffer settings
    EVENT_BUFFER_SIZE: int = 100  # maximum buffered progress events


@dataclass(frozen=True)
class MultiRepoConfig:
    """Multi-repository discovery configuration."""

    # Repository detection
    MAX_REPO_DEPTH: int = 3  # maximum directory depth to search for repos
    MAX_REPOSITORIES: int = 10  # maximum repositories to analyze

    # Parallel processing
    MAX_CONCURRENT_REPOS: int = 5  # maximum concurrent repository analysis
    REPO_ANALYSIS_TIMEOUT: float = 30.0  # timeout per repository analysis

    # Dependency analysis
    ENABLE_CROSS_REPO_DEPS: bool = True  # enable cross-repository dependency analysis
    MAX_DEPENDENCY_DEPTH: int = 2  # maximum dependency traversal depth

    # Access control
    RESPECT_GITIGNORE: bool = True  # respect .gitignore patterns
    SKIP_INACCESSIBLE: bool = True  # skip inaccessible repositories


@dataclass(frozen=True)
class TokenBudgetConfig:
    """Token budget management configuration."""

    # Base budgets by complexity
    LOW_COMPLEXITY_BUDGET: int = 30000  # tokens for low complexity tasks
    MEDIUM_COMPLEXITY_BUDGET: int = 50000  # tokens for medium complexity tasks
    HIGH_COMPLEXITY_BUDGET: int = 80000  # tokens for high complexity tasks
    VERY_HIGH_COMPLEXITY_BUDGET: int = 120000  # tokens for very high complexity tasks

    # Component allocation percentages
    QUERY_ANALYSIS_PERCENT: float = 0.15  # 15% for query analysis
    FILE_DISCOVERY_PERCENT: float = 0.30  # 30% for file discovery
    DOCUMENTATION_PERCENT: float = 0.25  # 25% for documentation search
    CODE_SEARCH_PERCENT: float = 0.20  # 20% for code search
    ASSEMBLY_PERCENT: float = 0.05  # 5% for context assembly
    RESERVE_PERCENT: float = 0.05  # 5% reserve

    # Optimization settings
    ENABLE_CONTENT_RANKING: bool = True  # enable Gemini-based content ranking
    ENABLE_PATTERN_SUMMARIZATION: bool = True  # enable pattern summarization
    SNIPPET_PRIORITY_THRESHOLD: float = 0.7  # threshold for snippet prioritization


@dataclass(frozen=True)
class MetricsConfig:
    """Metrics collection configuration."""

    # Collection settings
    ENABLE_DETAILED_METRICS: bool = True  # enable detailed component metrics
    ENABLE_QUALITY_SCORING: bool = True  # enable context quality scoring
    ENABLE_PERFORMANCE_TRACKING: bool = True  # enable performance tracking

    # Retention settings
    METRICS_RETENTION_DAYS: int = 30  # days to retain detailed metrics
    AGGREGATED_RETENTION_DAYS: int = 365  # days to retain aggregated metrics

    # Export settings
    PROMETHEUS_ENABLED: bool = True  # enable Prometheus metrics export
    METRICS_EXPORT_INTERVAL: int = 60  # seconds between metrics exports


@dataclass(frozen=True)
class ServerConfig:
    """Server configuration constants."""

    DEFAULT_HOST: str = "0.0.0.0"
    DEFAULT_PORT: int = 8000


# ============================================================================
# Global Singleton Instances
# ============================================================================

CONTENT = ContentLimits()
SEARCH = SearchLimits()
WORKSPACE = WorkspaceLimits()
LIMITS = ResourceLimits()
FRAMEWORK = FrameworkLimits()
LLM = LLMParams()
SERVER = ServerConfig()

# Enhanced configuration constants
CACHE = CacheConfig()
CIRCUIT_BREAKER = CircuitBreakerConfig()
STREAMING = StreamingConfig()
MULTI_REPO = MultiRepoConfig()
TOKEN_BUDGET = TokenBudgetConfig()
METRICS = MetricsConfig()


# ============================================================================
# Backward Compatibility / Direct Access
# ============================================================================

# For places that just need the number without the namespace
# Usage: from app.core.constants import SNIPPET_SHORT
SNIPPET_TINY = CONTENT.SNIPPET_TINY
SNIPPET_SHORT = CONTENT.SNIPPET_SHORT
SNIPPET_MEDIUM = CONTENT.SNIPPET_MEDIUM
SNIPPET_STANDARD = CONTENT.SNIPPET_STANDARD
SNIPPET_LONG = CONTENT.SNIPPET_LONG
SNIPPET_EXTENDED = CONTENT.SNIPPET_EXTENDED
SNIPPET_FULL = CONTENT.SNIPPET_FULL
SNIPPET_MAX = CONTENT.SNIPPET_MAX

K_SINGLE = SEARCH.K_SINGLE
K_FEW = SEARCH.K_FEW
K_STANDARD = SEARCH.K_STANDARD
K_DEFAULT = SEARCH.K_DEFAULT

MAX_MEMORY_THREADS = LIMITS.MAX_MEMORY_THREADS
SANDBOX_TIMEOUT = LIMITS.SANDBOX_TIMEOUT
INGEST_BATCH_SIZE = LIMITS.INGEST_BATCH_SIZE
