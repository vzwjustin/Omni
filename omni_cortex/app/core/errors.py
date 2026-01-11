"""
Omni-Cortex Error Taxonomy

Provides structured exceptions for all error conditions.
"""


class OmniCortexError(Exception):
    """Base exception for all Omni-Cortex errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

    def __repr__(self) -> str:
        name = self.__class__.__name__
        if self.details:
            return f"{name}({self.args[0]!r}, details={self.details!r})"
        return f"{name}({self.args[0]!r})"


# Validation errors
class ValidationError(OmniCortexError):
    """Input validation failed."""


# Routing errors
class RoutingError(OmniCortexError):
    """Framework selection or routing failed."""


class FrameworkNotFoundError(RoutingError):
    """Requested framework does not exist."""


class CategoryNotFoundError(RoutingError):
    """Requested category does not exist."""


# Execution errors
class ExecutionError(OmniCortexError):
    """Framework execution failed."""


class SandboxSecurityError(ExecutionError):
    """Code blocked by sandbox security checks."""


# Alias for backward compatibility
SecurityError = SandboxSecurityError


class SandboxTimeoutError(ExecutionError):
    """Code execution exceeded timeout."""


# Memory errors
class OmniMemoryError(OmniCortexError):
    """Memory operations failed."""


# Alias for backward compatibility (shadows built-in, use OmniMemoryError for clarity)
MemoryError = OmniMemoryError


class ThreadNotFoundError(OmniMemoryError):
    """Requested thread_id not in memory store."""


# RAG errors
class RAGError(OmniCortexError):
    """Vector store or retrieval failed."""


class CollectionNotFoundError(RAGError):
    """Requested collection does not exist."""


class EmbeddingError(RAGError):
    """Embedding generation failed."""


class ContextRetrievalError(RAGError):
    """Context retrieval failed (file discovery, doc search, etc)."""


# LLM errors
class LLMError(OmniCortexError):
    """LLM call failed."""


class ProviderNotConfiguredError(LLMError):
    """No API key for requested provider."""


class RateLimitError(LLMError):
    """API rate limit exceeded."""


class SamplerTimeoutError(LLMError):
    """Sampler request timed out."""


# Backwards compatibility alias
SamplerTimeout = SamplerTimeoutError  # noqa: N818


class SamplerCircuitOpenError(LLMError):
    """Circuit breaker is open, sampler unavailable."""


# Backwards compatibility alias
SamplerCircuitOpen = SamplerCircuitOpenError  # noqa: N818


# Context Gateway Enhancement errors
class ContextCacheError(OmniCortexError):
    """Cache-related errors."""


class CacheInvalidationError(ContextCacheError):
    """Cache invalidation failed."""


class CacheCorruptionError(ContextCacheError):
    """Cache data is corrupted or invalid."""


class StreamingError(OmniCortexError):
    """Streaming operation errors."""


class StreamingCancellationError(StreamingError):
    """Streaming operation was cancelled."""


class ProgressEventError(StreamingError):
    """Progress event emission failed."""


class MultiRepoError(OmniCortexError):
    """Multi-repository operation errors."""


class RepositoryAccessError(MultiRepoError):
    """Repository access denied or failed."""


class CrossRepoDependencyError(MultiRepoError):
    """Cross-repository dependency analysis failed."""


class CircuitBreakerError(OmniCortexError):
    """Circuit breaker activation errors."""


class CircuitBreakerOpenError(CircuitBreakerError):
    """Circuit breaker is open, operation blocked."""


class TokenBudgetError(OmniCortexError):
    """Token budget management errors."""


class TokenBudgetExceededError(TokenBudgetError):
    """Token budget exceeded during operation."""


class ContentOptimizationError(TokenBudgetError):
    """Content optimization failed."""


class DocumentationGroundingError(ContextRetrievalError):
    """Enhanced documentation grounding failed."""


class SourceAttributionError(DocumentationGroundingError):
    """Source attribution extraction failed."""


class MetricsCollectionError(OmniCortexError):
    """Metrics collection or recording failed."""
