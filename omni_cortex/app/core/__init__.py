"""Core module initialization."""

from .correlation import clear_correlation_id, get_correlation_id, set_correlation_id
from .errors import (
    CategoryNotFoundError,
    CollectionNotFoundError,
    EmbeddingError,
    ExecutionError,
    FrameworkNotFoundError,
    LLMError,
    MemoryError,
    OmniCortexError,
    ProviderNotConfiguredError,
    RAGError,
    RateLimitError,
    RoutingError,
    SandboxSecurityError,
    SandboxTimeoutError,
    ThreadNotFoundError,
)
from .logging import add_correlation_id
from .router import HyperRouter
from .settings import OmniCortexSettings, get_settings, reset_settings

__all__ = [
    "get_settings",
    "OmniCortexSettings",
    "reset_settings",
    "HyperRouter",
    # Correlation ID utilities
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "add_correlation_id",
    # Error hierarchy
    "OmniCortexError",
    "RoutingError",
    "FrameworkNotFoundError",
    "CategoryNotFoundError",
    "ExecutionError",
    "SandboxSecurityError",
    "SandboxTimeoutError",
    "MemoryError",
    "ThreadNotFoundError",
    "RAGError",
    "CollectionNotFoundError",
    "EmbeddingError",
    "LLMError",
    "ProviderNotConfiguredError",
    "RateLimitError",
]
