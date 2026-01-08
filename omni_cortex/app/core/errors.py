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
class MemoryError(OmniCortexError):
    """Memory operations failed."""


class ThreadNotFoundError(MemoryError):
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


class SamplerTimeout(LLMError):
    """Sampler request timed out."""


class SamplerCircuitOpen(LLMError):
    """Circuit breaker is open, sampler unavailable."""
