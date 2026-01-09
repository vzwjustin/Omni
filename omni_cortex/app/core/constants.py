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

from dataclasses import dataclass, field
from typing import Set, Tuple


@dataclass(frozen=True)
class ContentLimits:
    """Text/code content truncation limits (characters).

    These replace the scattered [:500], [:1000], [:1500], [:2000] slices.
    """

    # Display/logging - safe, non-critical truncation
    API_KEY_PREVIEW: int = 20       # chars shown for API key redaction
    QUERY_PREVIEW: int = 50         # very short query preview
    QUERY_LOG: int = 100            # query logging preview
    ERROR_PREVIEW: int = 200        # error message preview
    SUMMARY_PREVIEW: int = 400      # summary/outcome preview
    ERROR_FULL: int = 500           # full error message

    # RAG/Search result content
    SNIPPET_TINY: int = 100         # minimal content preview
    SNIPPET_SHORT: int = 500        # short snippet (most common)
    SNIPPET_MEDIUM: int = 800       # medium snippet
    SNIPPET_STANDARD: int = 1000    # standard content
    SNIPPET_LONG: int = 1200        # longer content
    SNIPPET_EXTENDED: int = 1500    # extended context
    SNIPPET_FULL: int = 1800        # full code context
    SNIPPET_MAX: int = 2000         # maximum single snippet

    # Command/output caps
    COMMAND_OUTPUT: int = 3000      # shell command output cap

    # Observation/answer limits for graph state
    OBSERVATION_LIMIT: int = 500    # state observation truncation


@dataclass(frozen=True)
class SearchLimits:
    """RAG and vector search limits."""

    # Default k values for retrieval
    K_SINGLE: int = 1               # single best match
    K_FEW: int = 2                  # 2 results
    K_STANDARD: int = 3             # standard 3-result (most common)
    K_DEFAULT: int = 5              # default RAG retrieval
    K_EXTENDED: int = 10            # extended search

    # Search result filtering
    GREP_MAX_COUNT: int = 10        # max grep results per query
    SEARCH_QUERIES_MAX: int = 3     # max search queries to run
    GIT_LOG_MAX: int = 10           # max git log results
    VIBE_ITEMS_MAX: int = 4         # max vibe dict items to display
    ROUTER_SIGNALS_MAX: int = 3     # max detection signals


@dataclass(frozen=True)
class WorkspaceLimits:
    """File discovery and workspace analysis limits."""

    MAX_FILES_CONTEXT: int = 15     # max files in final context
    MAX_FILES_LIST: int = 50        # max from pre-specified list
    MAX_FILES_ANALYZE: int = 100    # max for LLM analysis/ranking
    MAX_FILES_SCAN: int = 200       # absolute cap for workspace scan

    # Exclude patterns (directories to skip)
    EXCLUDE_DIRS: Tuple[str, ...] = (
        "__pycache__", ".git", "node_modules", ".venv", "venv",
        ".pytest_cache", ".mypy_cache", "dist", "build", ".egg-info",
        ".tox", "htmlcov", ".coverage", ".idea", ".vscode"
    )

    # Include extensions (files to consider)
    CODE_EXTENSIONS: Tuple[str, ...] = (
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".sql", ".sh", ".yaml", ".yml", ".json",
        ".toml", ".md", ".rst"
    )


@dataclass(frozen=True)
class ResourceLimits:
    """Memory, timeout, and batch processing limits."""

    # Memory (threads/conversations)
    MAX_MEMORY_THREADS: int = 100       # concurrent conversation threads
    MAX_MESSAGES_PER_THREAD: int = 20   # message history per thread

    # Execution timeouts (seconds)
    SANDBOX_TIMEOUT: float = 5.0        # code execution timeout
    SUBPROCESS_TIMEOUT: float = 5.0     # subprocess timeout (grep/find)
    GIT_TIMEOUT: float = 3.0            # git command timeout
    LLM_TIMEOUT: float = 30.0           # LLM sampling timeout
    TEST_TIMEOUT: float = 0.5           # test execution timeout

    # Batch processing
    INGEST_BATCH_SIZE: int = 100        # document ingestion batch size

    # Episode limits (CoALA framework)
    MAX_EPISODES: int = 100             # max stored episodes


@dataclass(frozen=True)
class FrameworkLimits:
    """Reasoning framework iteration/search limits."""

    MAX_REASONING_DEPTH: int = 10       # max chain-of-thought steps
    MCTS_MAX_ROLLOUTS: int = 50         # Monte Carlo tree search rollouts
    DEBATE_MAX_ROUNDS: int = 5          # max debate rounds
    TOT_MAX_BREADTH: int = 3            # Tree-of-Thought branching
    TOT_MAX_DEPTH: int = 5              # Tree-of-Thought depth


@dataclass(frozen=True)
class LLMParams:
    """LLM sampling parameters."""

    # Temperature settings
    ROUTING_TEMP: float = 0.7           # routing decision temp
    ANALYSIS_TEMP: float = 0.2          # deterministic analysis
    SEARCH_TEMP: float = 0.3            # documentation search
    CREATIVE_TEMP: float = 0.9          # creative generation

    # Sampling
    TOP_P: float = 1.0                  # nucleus sampling
    TOP_K: int = 40                     # top-k sampling

    # Token budgets
    CONTEXT_TOKEN_BUDGET: int = 50000   # max tokens for prepared context


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
