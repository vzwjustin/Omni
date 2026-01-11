"""
Comprehensive tests for Omni-Cortex Error Taxonomy.

Tests all error classes defined in app/core/errors.py including:
- Base class behavior
- Inheritance hierarchy
- Exception chaining
- Aliases
- Details dict handling
- __repr__ formatting
"""

import pytest

from app.core.errors import (
    CacheCorruptionError,
    CacheInvalidationError,
    CategoryNotFoundError,
    CircuitBreakerError,
    CircuitBreakerOpenError,
    CollectionNotFoundError,
    ContentOptimizationError,
    # Context Gateway
    ContextCacheError,
    ContextRetrievalError,
    CrossRepoDependencyError,
    DocumentationGroundingError,
    EmbeddingError,
    # Execution
    ExecutionError,
    FrameworkNotFoundError,
    # LLM
    LLMError,
    MemoryError,  # alias (shadows built-in)
    MetricsCollectionError,
    MultiRepoError,
    # Base
    OmniCortexError,
    # Memory
    OmniMemoryError,
    ProgressEventError,
    ProviderNotConfiguredError,
    # RAG
    RAGError,
    RateLimitError,
    RepositoryAccessError,
    # Routing
    RoutingError,
    SamplerCircuitOpen,
    SamplerTimeout,
    SandboxSecurityError,
    SandboxTimeoutError,
    SecurityError,  # alias
    SourceAttributionError,
    StreamingCancellationError,
    StreamingError,
    ThreadNotFoundError,
    TokenBudgetError,
    TokenBudgetExceededError,
)


class TestOmniCortexErrorBase:
    """Tests for the base OmniCortexError class."""

    def test_basic_instantiation(self):
        """Test creating error with just a message."""
        err = OmniCortexError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.details == {}

    def test_with_details(self):
        """Test creating error with message and details."""
        details = {"code": 500, "context": "test"}
        err = OmniCortexError("Error occurred", details)
        assert str(err) == "Error occurred"
        assert err.details == details
        assert err.details["code"] == 500
        assert err.details["context"] == "test"

    def test_details_none_becomes_empty_dict(self):
        """Test that None details becomes empty dict."""
        err = OmniCortexError("Error", None)
        assert err.details == {}
        assert isinstance(err.details, dict)

    def test_details_is_mutable(self):
        """Test that details dict can be modified after creation."""
        err = OmniCortexError("Error")
        err.details["added"] = "later"
        assert err.details["added"] == "later"

    def test_repr_without_details(self):
        """Test __repr__ output without details."""
        err = OmniCortexError("Test message")
        assert repr(err) == "OmniCortexError('Test message')"

    def test_repr_with_details(self):
        """Test __repr__ output with details."""
        err = OmniCortexError("Test message", {"key": "value"})
        assert repr(err) == "OmniCortexError('Test message', details={'key': 'value'})"

    def test_repr_with_empty_details(self):
        """Test __repr__ with explicitly empty details still shows nothing."""
        err = OmniCortexError("Test message", {})
        # Empty dict is falsy, so should not show details
        assert repr(err) == "OmniCortexError('Test message')"

    def test_inherits_from_exception(self):
        """Test that OmniCortexError is an Exception."""
        assert issubclass(OmniCortexError, Exception)
        err = OmniCortexError("Error")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self):
        """Test raising and catching OmniCortexError."""
        with pytest.raises(OmniCortexError) as exc_info:
            raise OmniCortexError("Test error")
        assert str(exc_info.value) == "Test error"

    def test_exception_args(self):
        """Test that message is accessible via args."""
        err = OmniCortexError("Test message")
        assert err.args == ("Test message",)


class TestRoutingErrors:
    """Tests for routing-related errors."""

    def test_routing_error_inherits_from_base(self):
        """Test RoutingError inheritance."""
        assert issubclass(RoutingError, OmniCortexError)

    def test_framework_not_found_error_inheritance(self):
        """Test FrameworkNotFoundError inheritance chain."""
        assert issubclass(FrameworkNotFoundError, RoutingError)
        assert issubclass(FrameworkNotFoundError, OmniCortexError)
        assert issubclass(FrameworkNotFoundError, Exception)

    def test_category_not_found_error_inheritance(self):
        """Test CategoryNotFoundError inheritance chain."""
        assert issubclass(CategoryNotFoundError, RoutingError)
        assert issubclass(CategoryNotFoundError, OmniCortexError)

    def test_framework_not_found_can_be_caught_by_routing_error(self):
        """Test that FrameworkNotFoundError can be caught as RoutingError."""
        with pytest.raises(RoutingError):
            raise FrameworkNotFoundError("Unknown framework: foo")

    def test_framework_not_found_can_be_caught_by_base(self):
        """Test that FrameworkNotFoundError can be caught as OmniCortexError."""
        with pytest.raises(OmniCortexError):
            raise FrameworkNotFoundError("Unknown framework")

    def test_routing_error_with_details(self):
        """Test routing errors with details."""
        details = {"framework": "nonexistent", "available": ["a", "b"]}
        err = FrameworkNotFoundError("Framework not found", details)
        assert err.details["framework"] == "nonexistent"
        assert "nonexistent" in repr(err)

    def test_category_not_found_error_usage(self):
        """Test CategoryNotFoundError practical usage."""
        with pytest.raises(CategoryNotFoundError) as exc_info:
            raise CategoryNotFoundError(
                "Category 'unknown' not found",
                {"category": "unknown", "valid": ["STRATEGY", "SEARCH"]}
            )
        assert exc_info.value.details["category"] == "unknown"


class TestExecutionErrors:
    """Tests for execution-related errors."""

    def test_execution_error_inherits_from_base(self):
        """Test ExecutionError inheritance."""
        assert issubclass(ExecutionError, OmniCortexError)

    def test_sandbox_security_error_inheritance(self):
        """Test SandboxSecurityError inheritance chain."""
        assert issubclass(SandboxSecurityError, ExecutionError)
        assert issubclass(SandboxSecurityError, OmniCortexError)

    def test_sandbox_timeout_error_inheritance(self):
        """Test SandboxTimeoutError inheritance chain."""
        assert issubclass(SandboxTimeoutError, ExecutionError)
        assert issubclass(SandboxTimeoutError, OmniCortexError)

    def test_security_error_alias(self):
        """Test that SecurityError is an alias for SandboxSecurityError."""
        assert SecurityError is SandboxSecurityError

    def test_security_error_alias_catch(self):
        """Test catching SandboxSecurityError via SecurityError alias."""
        with pytest.raises(SecurityError):
            raise SandboxSecurityError("Blocked import: os")

    def test_sandbox_security_error_with_details(self):
        """Test SandboxSecurityError with security context."""
        err = SandboxSecurityError(
            "Import not allowed",
            {"blocked_import": "subprocess", "allowed": ["math", "json"]}
        )
        assert err.details["blocked_import"] == "subprocess"

    def test_sandbox_timeout_with_details(self):
        """Test SandboxTimeoutError with timeout context."""
        err = SandboxTimeoutError(
            "Execution timed out",
            {"timeout_seconds": 5.0, "code_length": 100}
        )
        assert err.details["timeout_seconds"] == 5.0


class TestMemoryErrors:
    """Tests for memory-related errors."""

    def test_omni_memory_error_inherits_from_base(self):
        """Test OmniMemoryError inheritance."""
        assert issubclass(OmniMemoryError, OmniCortexError)

    def test_memory_error_alias(self):
        """Test that MemoryError is an alias for OmniMemoryError."""
        assert MemoryError is OmniMemoryError

    def test_memory_error_alias_catch(self):
        """Test catching OmniMemoryError via MemoryError alias."""
        with pytest.raises(MemoryError):
            raise OmniMemoryError("Memory operation failed")

    def test_thread_not_found_error_inheritance(self):
        """Test ThreadNotFoundError inheritance chain."""
        assert issubclass(ThreadNotFoundError, OmniMemoryError)
        assert issubclass(ThreadNotFoundError, OmniCortexError)

    def test_thread_not_found_can_be_caught_by_memory_error(self):
        """Test that ThreadNotFoundError can be caught as OmniMemoryError."""
        with pytest.raises(OmniMemoryError):
            raise ThreadNotFoundError("Thread abc123 not found")

    def test_thread_not_found_can_be_caught_by_alias(self):
        """Test that ThreadNotFoundError can be caught via MemoryError alias."""
        with pytest.raises(MemoryError):
            raise ThreadNotFoundError("Thread not found")

    def test_thread_not_found_with_details(self):
        """Test ThreadNotFoundError with thread context."""
        err = ThreadNotFoundError(
            "Thread not found",
            {"thread_id": "abc123", "store_size": 50}
        )
        assert err.details["thread_id"] == "abc123"


class TestRAGErrors:
    """Tests for RAG-related errors."""

    def test_rag_error_inherits_from_base(self):
        """Test RAGError inheritance."""
        assert issubclass(RAGError, OmniCortexError)

    def test_collection_not_found_error_inheritance(self):
        """Test CollectionNotFoundError inheritance chain."""
        assert issubclass(CollectionNotFoundError, RAGError)
        assert issubclass(CollectionNotFoundError, OmniCortexError)

    def test_embedding_error_inheritance(self):
        """Test EmbeddingError inheritance chain."""
        assert issubclass(EmbeddingError, RAGError)
        assert issubclass(EmbeddingError, OmniCortexError)

    def test_context_retrieval_error_inheritance(self):
        """Test ContextRetrievalError inheritance chain."""
        assert issubclass(ContextRetrievalError, RAGError)
        assert issubclass(ContextRetrievalError, OmniCortexError)

    def test_all_rag_errors_caught_by_parent(self):
        """Test all RAG errors can be caught by RAGError."""
        rag_errors = [
            CollectionNotFoundError("Collection missing"),
            EmbeddingError("Embedding failed"),
            ContextRetrievalError("Retrieval failed"),
        ]
        for error in rag_errors:
            with pytest.raises(RAGError):
                raise error

    def test_embedding_error_with_provider_details(self):
        """Test EmbeddingError with provider context."""
        err = EmbeddingError(
            "Embedding generation failed",
            {"provider": "openai", "model": "text-embedding-3-small", "status": 429}
        )
        assert err.details["provider"] == "openai"
        assert err.details["status"] == 429


class TestLLMErrors:
    """Tests for LLM-related errors."""

    def test_llm_error_inherits_from_base(self):
        """Test LLMError inheritance."""
        assert issubclass(LLMError, OmniCortexError)

    def test_provider_not_configured_error_inheritance(self):
        """Test ProviderNotConfiguredError inheritance chain."""
        assert issubclass(ProviderNotConfiguredError, LLMError)
        assert issubclass(ProviderNotConfiguredError, OmniCortexError)

    def test_rate_limit_error_inheritance(self):
        """Test RateLimitError inheritance chain."""
        assert issubclass(RateLimitError, LLMError)
        assert issubclass(RateLimitError, OmniCortexError)

    def test_sampler_timeout_inheritance(self):
        """Test SamplerTimeout inheritance chain."""
        assert issubclass(SamplerTimeout, LLMError)
        assert issubclass(SamplerTimeout, OmniCortexError)

    def test_sampler_circuit_open_inheritance(self):
        """Test SamplerCircuitOpen inheritance chain."""
        assert issubclass(SamplerCircuitOpen, LLMError)
        assert issubclass(SamplerCircuitOpen, OmniCortexError)

    def test_all_llm_errors_caught_by_parent(self):
        """Test all LLM errors can be caught by LLMError."""
        llm_errors = [
            ProviderNotConfiguredError("No API key"),
            RateLimitError("Rate limited"),
            SamplerTimeout("Timeout"),
            SamplerCircuitOpen("Circuit open"),
        ]
        for error in llm_errors:
            with pytest.raises(LLMError):
                raise error

    def test_rate_limit_error_with_retry_info(self):
        """Test RateLimitError with retry context."""
        err = RateLimitError(
            "Rate limit exceeded",
            {"retry_after": 60, "limit": 100, "remaining": 0}
        )
        assert err.details["retry_after"] == 60

    def test_provider_not_configured_with_provider_info(self):
        """Test ProviderNotConfiguredError with provider context."""
        err = ProviderNotConfiguredError(
            "Provider not configured",
            {"provider": "anthropic", "env_var": "ANTHROPIC_API_KEY"}
        )
        assert err.details["provider"] == "anthropic"


class TestContextCacheErrors:
    """Tests for context cache errors."""

    def test_context_cache_error_inherits_from_base(self):
        """Test ContextCacheError inheritance."""
        assert issubclass(ContextCacheError, OmniCortexError)

    def test_cache_invalidation_error_inheritance(self):
        """Test CacheInvalidationError inheritance chain."""
        assert issubclass(CacheInvalidationError, ContextCacheError)
        assert issubclass(CacheInvalidationError, OmniCortexError)

    def test_cache_corruption_error_inheritance(self):
        """Test CacheCorruptionError inheritance chain."""
        assert issubclass(CacheCorruptionError, ContextCacheError)
        assert issubclass(CacheCorruptionError, OmniCortexError)

    def test_cache_errors_caught_by_parent(self):
        """Test all cache errors can be caught by ContextCacheError."""
        cache_errors = [
            CacheInvalidationError("Invalidation failed"),
            CacheCorruptionError("Data corrupted"),
        ]
        for error in cache_errors:
            with pytest.raises(ContextCacheError):
                raise error


class TestStreamingErrors:
    """Tests for streaming errors."""

    def test_streaming_error_inherits_from_base(self):
        """Test StreamingError inheritance."""
        assert issubclass(StreamingError, OmniCortexError)

    def test_streaming_cancellation_error_inheritance(self):
        """Test StreamingCancellationError inheritance chain."""
        assert issubclass(StreamingCancellationError, StreamingError)
        assert issubclass(StreamingCancellationError, OmniCortexError)

    def test_progress_event_error_inheritance(self):
        """Test ProgressEventError inheritance chain."""
        assert issubclass(ProgressEventError, StreamingError)
        assert issubclass(ProgressEventError, OmniCortexError)

    def test_streaming_errors_caught_by_parent(self):
        """Test all streaming errors can be caught by StreamingError."""
        streaming_errors = [
            StreamingCancellationError("Cancelled by user"),
            ProgressEventError("Event emission failed"),
        ]
        for error in streaming_errors:
            with pytest.raises(StreamingError):
                raise error


class TestMultiRepoErrors:
    """Tests for multi-repository errors."""

    def test_multi_repo_error_inherits_from_base(self):
        """Test MultiRepoError inheritance."""
        assert issubclass(MultiRepoError, OmniCortexError)

    def test_repository_access_error_inheritance(self):
        """Test RepositoryAccessError inheritance chain."""
        assert issubclass(RepositoryAccessError, MultiRepoError)
        assert issubclass(RepositoryAccessError, OmniCortexError)

    def test_cross_repo_dependency_error_inheritance(self):
        """Test CrossRepoDependencyError inheritance chain."""
        assert issubclass(CrossRepoDependencyError, MultiRepoError)
        assert issubclass(CrossRepoDependencyError, OmniCortexError)

    def test_multi_repo_errors_caught_by_parent(self):
        """Test all multi-repo errors can be caught by MultiRepoError."""
        repo_errors = [
            RepositoryAccessError("Access denied"),
            CrossRepoDependencyError("Dependency analysis failed"),
        ]
        for error in repo_errors:
            with pytest.raises(MultiRepoError):
                raise error


class TestCircuitBreakerErrors:
    """Tests for circuit breaker errors."""

    def test_circuit_breaker_error_inherits_from_base(self):
        """Test CircuitBreakerError inheritance."""
        assert issubclass(CircuitBreakerError, OmniCortexError)

    def test_circuit_breaker_open_error_inheritance(self):
        """Test CircuitBreakerOpenError inheritance chain."""
        assert issubclass(CircuitBreakerOpenError, CircuitBreakerError)
        assert issubclass(CircuitBreakerOpenError, OmniCortexError)

    def test_circuit_breaker_open_caught_by_parent(self):
        """Test CircuitBreakerOpenError can be caught by CircuitBreakerError."""
        with pytest.raises(CircuitBreakerError):
            raise CircuitBreakerOpenError("Circuit breaker open")


class TestTokenBudgetErrors:
    """Tests for token budget errors."""

    def test_token_budget_error_inherits_from_base(self):
        """Test TokenBudgetError inheritance."""
        assert issubclass(TokenBudgetError, OmniCortexError)

    def test_token_budget_exceeded_error_inheritance(self):
        """Test TokenBudgetExceededError inheritance chain."""
        assert issubclass(TokenBudgetExceededError, TokenBudgetError)
        assert issubclass(TokenBudgetExceededError, OmniCortexError)

    def test_content_optimization_error_inheritance(self):
        """Test ContentOptimizationError inheritance chain."""
        assert issubclass(ContentOptimizationError, TokenBudgetError)
        assert issubclass(ContentOptimizationError, OmniCortexError)

    def test_token_budget_errors_caught_by_parent(self):
        """Test all token budget errors can be caught by TokenBudgetError."""
        budget_errors = [
            TokenBudgetExceededError("Budget exceeded"),
            ContentOptimizationError("Optimization failed"),
        ]
        for error in budget_errors:
            with pytest.raises(TokenBudgetError):
                raise error

    def test_token_budget_exceeded_with_details(self):
        """Test TokenBudgetExceededError with token context."""
        err = TokenBudgetExceededError(
            "Token budget exceeded",
            {"budget": 10000, "used": 15000, "overage": 5000}
        )
        assert err.details["budget"] == 10000
        assert err.details["overage"] == 5000


class TestDocumentationGroundingErrors:
    """Tests for documentation grounding errors."""

    def test_documentation_grounding_error_inheritance(self):
        """Test DocumentationGroundingError inheritance chain."""
        # DocumentationGroundingError inherits from ContextRetrievalError
        assert issubclass(DocumentationGroundingError, ContextRetrievalError)
        assert issubclass(DocumentationGroundingError, RAGError)
        assert issubclass(DocumentationGroundingError, OmniCortexError)

    def test_source_attribution_error_inheritance(self):
        """Test SourceAttributionError inheritance chain."""
        # SourceAttributionError inherits from DocumentationGroundingError
        assert issubclass(SourceAttributionError, DocumentationGroundingError)
        assert issubclass(SourceAttributionError, ContextRetrievalError)
        assert issubclass(SourceAttributionError, RAGError)
        assert issubclass(SourceAttributionError, OmniCortexError)

    def test_deep_hierarchy_catch_by_rag_error(self):
        """Test deeply nested error can be caught by RAGError."""
        with pytest.raises(RAGError):
            raise SourceAttributionError("Attribution failed")

    def test_deep_hierarchy_catch_by_context_retrieval(self):
        """Test deeply nested error can be caught by ContextRetrievalError."""
        with pytest.raises(ContextRetrievalError):
            raise SourceAttributionError("Attribution failed")

    def test_deep_hierarchy_catch_by_documentation_grounding(self):
        """Test error can be caught by immediate parent."""
        with pytest.raises(DocumentationGroundingError):
            raise SourceAttributionError("Attribution failed")


class TestMetricsCollectionError:
    """Tests for metrics collection error."""

    def test_metrics_collection_error_inherits_from_base(self):
        """Test MetricsCollectionError inheritance."""
        assert issubclass(MetricsCollectionError, OmniCortexError)

    def test_metrics_error_with_details(self):
        """Test MetricsCollectionError with metrics context."""
        err = MetricsCollectionError(
            "Failed to record metric",
            {"metric_name": "token_usage", "timestamp": "2024-01-01T00:00:00Z"}
        )
        assert err.details["metric_name"] == "token_usage"


class TestExceptionChaining:
    """Tests for exception chaining behavior."""

    def test_exception_chaining_with_from(self):
        """Test exception chaining using 'from'."""
        original = ValueError("Original error")
        try:
            try:
                raise original
            except ValueError as e:
                raise OmniCortexError("Wrapped error") from e
        except OmniCortexError as e:
            assert e.__cause__ is original
            assert str(e.__cause__) == "Original error"

    def test_exception_chaining_implicit(self):
        """Test implicit exception chaining."""
        try:
            try:
                raise ValueError("Original")
            except ValueError as err:
                raise OmniCortexError("Wrapped") from err
        except OmniCortexError as e:
            assert e.__context__ is not None
            assert isinstance(e.__context__, ValueError)

    def test_nested_omni_cortex_errors(self):
        """Test chaining OmniCortexError subclasses."""
        try:
            try:
                raise EmbeddingError("Embedding failed")
            except EmbeddingError as e:
                raise ContextRetrievalError("Retrieval failed") from e
        except ContextRetrievalError as e:
            assert isinstance(e.__cause__, EmbeddingError)

    def test_chained_error_preserves_details(self):
        """Test that chained errors preserve their details."""
        original = EmbeddingError("Failed", {"provider": "openai"})
        try:
            try:
                raise original
            except EmbeddingError as e:
                raise RAGError("RAG failed", {"step": "embedding"}) from e
        except RAGError as e:
            assert e.details["step"] == "embedding"
            assert e.__cause__.details["provider"] == "openai"


class TestCompleteHierarchy:
    """Tests verifying the complete error hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """Verify all custom errors inherit from OmniCortexError."""
        all_error_classes = [
            RoutingError,
            FrameworkNotFoundError,
            CategoryNotFoundError,
            ExecutionError,
            SandboxSecurityError,
            SandboxTimeoutError,
            OmniMemoryError,
            ThreadNotFoundError,
            RAGError,
            CollectionNotFoundError,
            EmbeddingError,
            ContextRetrievalError,
            LLMError,
            ProviderNotConfiguredError,
            RateLimitError,
            SamplerTimeout,
            SamplerCircuitOpen,
            ContextCacheError,
            CacheInvalidationError,
            CacheCorruptionError,
            StreamingError,
            StreamingCancellationError,
            ProgressEventError,
            MultiRepoError,
            RepositoryAccessError,
            CrossRepoDependencyError,
            CircuitBreakerError,
            CircuitBreakerOpenError,
            TokenBudgetError,
            TokenBudgetExceededError,
            ContentOptimizationError,
            DocumentationGroundingError,
            SourceAttributionError,
            MetricsCollectionError,
        ]
        for error_class in all_error_classes:
            assert issubclass(error_class, OmniCortexError), (
                f"{error_class.__name__} should inherit from OmniCortexError"
            )

    def test_all_errors_are_exceptions(self):
        """Verify all custom errors are Exception subclasses."""
        all_error_classes = [
            OmniCortexError,
            RoutingError,
            FrameworkNotFoundError,
            CategoryNotFoundError,
            ExecutionError,
            SandboxSecurityError,
            SandboxTimeoutError,
            OmniMemoryError,
            ThreadNotFoundError,
            RAGError,
            CollectionNotFoundError,
            EmbeddingError,
            ContextRetrievalError,
            LLMError,
            ProviderNotConfiguredError,
            RateLimitError,
            SamplerTimeout,
            SamplerCircuitOpen,
            ContextCacheError,
            CacheInvalidationError,
            CacheCorruptionError,
            StreamingError,
            StreamingCancellationError,
            ProgressEventError,
            MultiRepoError,
            RepositoryAccessError,
            CrossRepoDependencyError,
            CircuitBreakerError,
            CircuitBreakerOpenError,
            TokenBudgetError,
            TokenBudgetExceededError,
            ContentOptimizationError,
            DocumentationGroundingError,
            SourceAttributionError,
            MetricsCollectionError,
        ]
        for error_class in all_error_classes:
            assert issubclass(error_class, Exception), (
                f"{error_class.__name__} should be an Exception"
            )

    def test_all_errors_support_details(self):
        """Verify all errors support the details parameter."""
        all_error_classes = [
            OmniCortexError,
            RoutingError,
            FrameworkNotFoundError,
            CategoryNotFoundError,
            ExecutionError,
            SandboxSecurityError,
            SandboxTimeoutError,
            OmniMemoryError,
            ThreadNotFoundError,
            RAGError,
            CollectionNotFoundError,
            EmbeddingError,
            ContextRetrievalError,
            LLMError,
            ProviderNotConfiguredError,
            RateLimitError,
            SamplerTimeout,
            SamplerCircuitOpen,
            ContextCacheError,
            CacheInvalidationError,
            CacheCorruptionError,
            StreamingError,
            StreamingCancellationError,
            ProgressEventError,
            MultiRepoError,
            RepositoryAccessError,
            CrossRepoDependencyError,
            CircuitBreakerError,
            CircuitBreakerOpenError,
            TokenBudgetError,
            TokenBudgetExceededError,
            ContentOptimizationError,
            DocumentationGroundingError,
            SourceAttributionError,
            MetricsCollectionError,
        ]
        details = {"test_key": "test_value"}
        for error_class in all_error_classes:
            err = error_class("Test message", details)
            assert err.details == details, (
                f"{error_class.__name__} should support details parameter"
            )

    def test_all_errors_have_repr(self):
        """Verify all errors have working __repr__."""
        all_error_classes = [
            OmniCortexError,
            RoutingError,
            FrameworkNotFoundError,
            CategoryNotFoundError,
            ExecutionError,
            SandboxSecurityError,
            SandboxTimeoutError,
            OmniMemoryError,
            ThreadNotFoundError,
            RAGError,
            CollectionNotFoundError,
            EmbeddingError,
            ContextRetrievalError,
            LLMError,
            ProviderNotConfiguredError,
            RateLimitError,
            SamplerTimeout,
            SamplerCircuitOpen,
            ContextCacheError,
            CacheInvalidationError,
            CacheCorruptionError,
            StreamingError,
            StreamingCancellationError,
            ProgressEventError,
            MultiRepoError,
            RepositoryAccessError,
            CrossRepoDependencyError,
            CircuitBreakerError,
            CircuitBreakerOpenError,
            TokenBudgetError,
            TokenBudgetExceededError,
            ContentOptimizationError,
            DocumentationGroundingError,
            SourceAttributionError,
            MetricsCollectionError,
        ]
        for error_class in all_error_classes:
            err = error_class("Test message")
            repr_str = repr(err)
            assert error_class.__name__ in repr_str, (
                f"{error_class.__name__}.__repr__ should include class name"
            )
            assert "Test message" in repr_str, (
                f"{error_class.__name__}.__repr__ should include message"
            )


class TestErrorUsagePatterns:
    """Tests for common error usage patterns."""

    def test_catch_specific_then_general(self):
        """Test catching specific error before general."""
        def may_raise(error_type):
            if error_type == "framework":
                raise FrameworkNotFoundError("Not found")
            elif error_type == "routing":
                raise RoutingError("General routing")
            else:
                raise OmniCortexError("Base error")

        # Specific error caught
        with pytest.raises(FrameworkNotFoundError):
            may_raise("framework")

        # Parent catches child
        with pytest.raises(RoutingError):
            may_raise("framework")

    def test_multiple_except_blocks(self):
        """Test error handling with multiple except blocks."""
        def handle_error(error):
            try:
                raise error
            except FrameworkNotFoundError:
                return "framework_not_found"
            except RoutingError:
                return "routing_error"
            except OmniCortexError:
                return "omni_cortex_error"

        assert handle_error(FrameworkNotFoundError("test")) == "framework_not_found"
        assert handle_error(CategoryNotFoundError("test")) == "routing_error"
        assert handle_error(LLMError("test")) == "omni_cortex_error"

    def test_error_in_context_manager(self):
        """Test error handling in context manager pattern."""
        class ErrorContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return exc_type is not None and issubclass(exc_type, RoutingError)

        # Routing errors suppressed
        with ErrorContext():
            raise FrameworkNotFoundError("Suppressed")

        # Other errors propagate
        with pytest.raises(LLMError), ErrorContext():
            raise LLMError("Not suppressed")

    def test_reraise_with_context(self):
        """Test reraising errors with additional context."""
        def inner():
            raise EmbeddingError("Embedding failed", {"model": "text-3-small"})

        def outer():
            try:
                inner()
            except EmbeddingError as e:
                raise RAGError(
                    f"RAG operation failed: {e}",
                    {"original_error": type(e).__name__, **e.details}
                ) from e

        with pytest.raises(RAGError) as exc_info:
            outer()

        err = exc_info.value
        assert "RAG operation failed" in str(err)
        assert err.details["original_error"] == "EmbeddingError"
        assert err.details["model"] == "text-3-small"
        assert isinstance(err.__cause__, EmbeddingError)
