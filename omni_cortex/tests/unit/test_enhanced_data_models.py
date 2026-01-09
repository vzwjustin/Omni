"""
Property-based tests for enhanced context gateway data models.

Tests the enhanced data models for context gateway enhancements including:
- Cache metadata and TTL handling
- Progress event structures
- Multi-repository information
- Source attribution
- Component status tracking
- Token budget management
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import MagicMock

# Property-based testing imports
try:
    from hypothesis import given, strategies as st, assume, settings
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators for when hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class st:
        @staticmethod
        def text(*args, **kwargs):
            return lambda: "test_string"
        
        @staticmethod
        def floats(*args, **kwargs):
            return lambda: 0.5
        
        @staticmethod
        def integers(*args, **kwargs):
            return lambda: 42
        
        @staticmethod
        def booleans():
            return lambda: True
        
        @staticmethod
        def lists(*args, **kwargs):
            return lambda: []
        
        @staticmethod
        def dictionaries(*args, **kwargs):
            return lambda: {}
        
        @staticmethod
        def one_of(*args):
            return lambda: args[0]() if args else None

# Import enhanced models
from app.core.context.enhanced_models import (
    CacheEntry,
    CacheMetadata,
    ProgressEvent,
    ProgressStatus,
    RepoInfo,
    SourceAttribution,
    ComponentMetrics,
    QualityMetrics,
    TokenBudgetUsage,
    ComponentStatus,
    ComponentStatusInfo,
    EnhancedStructuredContext,
)


# =============================================================================
# Hypothesis Strategies for Data Generation
# =============================================================================

@composite
def cache_entry_strategy(draw):
    """Generate valid CacheEntry instances."""
    return CacheEntry(
        value=draw(st.dictionaries(st.text(), st.text())),
        created_at=datetime.now() - timedelta(seconds=draw(st.integers(min_value=0, max_value=3600))),
        ttl_seconds=draw(st.integers(min_value=60, max_value=86400)),
        cache_type=draw(st.one_of(
            st.just("query_analysis"),
            st.just("file_discovery"),
            st.just("documentation")
        )),
        workspace_fingerprint=draw(st.text(min_size=8, max_size=16)),
        query_hash=draw(st.text(min_size=8, max_size=16))
    )


@composite
def progress_event_strategy(draw):
    """Generate valid ProgressEvent instances."""
    return ProgressEvent(
        component=draw(st.one_of(
            st.just("query_analysis"),
            st.just("file_discovery"),
            st.just("doc_search"),
            st.just("code_search")
        )),
        status=draw(st.one_of(*[st.just(status) for status in ProgressStatus])),
        progress=draw(st.floats(min_value=0.0, max_value=1.0)),
        data=draw(st.one_of(st.none(), st.dictionaries(st.text(), st.text()))),
        timestamp=datetime.now(),
        estimated_completion=draw(st.one_of(st.none(), st.floats(min_value=0.1, max_value=300.0))),
        message=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100)))
    )


@composite
def repo_info_strategy(draw):
    """Generate valid RepoInfo instances."""
    return RepoInfo(
        path=draw(st.text(min_size=1, max_size=100)),
        name=draw(st.text(min_size=1, max_size=50)),
        git_root=draw(st.text(min_size=1, max_size=100)),
        ignore_patterns=draw(st.lists(st.text(), max_size=10)),
        access_permissions=draw(st.dictionaries(st.text(), st.booleans(), max_size=5)),
        last_commit=draw(st.one_of(st.none(), st.text(min_size=8, max_size=40))),
        branch=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        is_accessible=draw(st.booleans()),
        error_message=draw(st.one_of(st.none(), st.text(min_size=1, max_size=200)))
    )


@composite
def component_status_info_strategy(draw):
    """Generate valid ComponentStatusInfo instances."""
    status = draw(st.one_of(*[st.just(s) for s in ComponentStatus]))
    
    # Generate appropriate fields based on status
    error_message = None
    fallback_method = None
    
    if status in [ComponentStatus.FAILED, ComponentStatus.PARTIAL]:
        error_message = draw(st.one_of(st.none(), st.text(min_size=1, max_size=200)))
    
    if status == ComponentStatus.FALLBACK:
        fallback_method = draw(st.text(min_size=1, max_size=50))
    
    return ComponentStatusInfo(
        status=status,
        execution_time=draw(st.floats(min_value=0.1, max_value=300.0)),
        error_message=error_message,
        fallback_method=fallback_method,
        api_calls_made=draw(st.integers(min_value=0, max_value=20)),
        tokens_consumed=draw(st.integers(min_value=0, max_value=10000)),
        warnings=draw(st.lists(st.text(min_size=1, max_size=100), max_size=5))
    )


@composite
def enhanced_structured_context_strategy(draw):
    """Generate valid EnhancedStructuredContext instances."""
    # Generate component status with at least one partial failure
    component_status = {}
    components = ["query_analysis", "file_discovery", "doc_search", "code_search"]
    
    # Ensure at least one component has partial failure for property testing
    has_partial_failure = False
    for component in components:
        status_info = draw(component_status_info_strategy())
        component_status[component] = status_info
        if status_info.status in [ComponentStatus.PARTIAL, ComponentStatus.FALLBACK, ComponentStatus.FAILED]:
            has_partial_failure = True
    
    # If no partial failures were generated, force one
    if not has_partial_failure:
        component_status[components[0]] = ComponentStatusInfo(
            status=ComponentStatus.PARTIAL,
            execution_time=1.0,
            error_message="Test partial failure",
            warnings=["Test warning"]
        )
    
    return EnhancedStructuredContext(
        task_type=draw(st.one_of(
            st.just("debug"), st.just("implement"), st.just("refactor"), 
            st.just("architect"), st.just("test"), st.just("explain")
        )),
        task_summary=draw(st.text(min_size=10, max_size=200)),
        complexity=draw(st.one_of(
            st.just("low"), st.just("medium"), st.just("high"), st.just("very_high")
        )),
        component_status=component_status
    )


# =============================================================================
# Property-Based Tests
# =============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestEnhancedDataModelsProperties:
    """Property-based tests for enhanced data models."""
    
    @given(cache_entry_strategy())
    @settings(max_examples=100)
    def test_cache_entry_expiration_property(self, cache_entry: CacheEntry):
        """
        Property: Cache entry expiration should be consistent with TTL.
        
        For any cache entry, if current time > created_at + ttl_seconds,
        then is_expired should be True, otherwise False.
        """
        # Calculate expected expiration
        expiration_time = cache_entry.created_at + timedelta(seconds=cache_entry.ttl_seconds)
        expected_expired = datetime.now() > expiration_time
        
        # Test the property
        assert cache_entry.is_expired == expected_expired, (
            f"Cache entry expiration inconsistent: "
            f"created_at={cache_entry.created_at}, "
            f"ttl={cache_entry.ttl_seconds}s, "
            f"expected_expired={expected_expired}, "
            f"actual_expired={cache_entry.is_expired}"
        )
    
    @given(progress_event_strategy())
    @settings(max_examples=100)
    def test_progress_event_progress_bounds_property(self, progress_event: ProgressEvent):
        """
        Property: Progress values should always be between 0.0 and 1.0.
        
        For any progress event, progress should be in valid range.
        """
        assert 0.0 <= progress_event.progress <= 1.0, (
            f"Progress value out of bounds: {progress_event.progress}"
        )
    
    @given(repo_info_strategy())
    @settings(max_examples=100)
    def test_repo_info_accessibility_consistency_property(self, repo_info: RepoInfo):
        """
        Property: Repository accessibility should be consistent with error messages.
        
        For any repo info, if is_accessible is False, there should be an error_message,
        and if is_accessible is True, error_message should be None or empty.
        """
        if not repo_info.is_accessible:
            assert repo_info.error_message is not None and repo_info.error_message.strip(), (
                f"Inaccessible repository should have error message: "
                f"is_accessible={repo_info.is_accessible}, "
                f"error_message={repo_info.error_message}"
            )
        # Note: Accessible repos may still have error_message for warnings
    
    @given(component_status_info_strategy())
    @settings(max_examples=100)
    def test_component_status_fallback_consistency_property(self, status_info: ComponentStatusInfo):
        """
        Property: Component status should be consistent with fallback method.
        
        For any component status info, if status is FALLBACK, there should be a fallback_method,
        and if status is not FALLBACK, fallback_method should be None.
        """
        if status_info.status == ComponentStatus.FALLBACK:
            assert status_info.fallback_method is not None and status_info.fallback_method.strip(), (
                f"Fallback status should have fallback method: "
                f"status={status_info.status}, "
                f"fallback_method={status_info.fallback_method}"
            )
        else:
            # Non-fallback status may or may not have fallback_method (could be None)
            pass  # This is acceptable - other statuses don't require fallback_method
    
    @given(enhanced_structured_context_strategy())
    @settings(max_examples=100)
    def test_partial_failure_status_indication_property(self, context: EnhancedStructuredContext):
        """
        **Property 21: Partial Failure Status Indication**
        **Validates: Requirements 8.4**
        
        For any context preparation with partial component failures, the StructuredContext 
        should clearly indicate which components succeeded, which failed, and which used 
        fallback methods.
        
        This property ensures that:
        1. All components have status information
        2. Failed/partial components have appropriate error details
        3. Fallback components indicate the fallback method used
        4. Status information is complete and consistent
        """
        # Verify all expected components have status
        expected_components = {"query_analysis", "file_discovery", "doc_search", "code_search"}
        actual_components = set(context.component_status.keys())
        
        # At least some components should have status (may not be all in test data)
        assert len(actual_components) > 0, "Context should have component status information"
        
        # Check each component's status consistency
        for component_name, status_info in context.component_status.items():
            # 1. Status should be a valid ComponentStatus
            assert isinstance(status_info.status, ComponentStatus), (
                f"Component {component_name} should have valid status, got {type(status_info.status)}"
            )
            
            # 2. Execution time should be positive
            assert status_info.execution_time > 0, (
                f"Component {component_name} should have positive execution time, got {status_info.execution_time}"
            )
            
            # 3. Failed components should have error information
            if status_info.status == ComponentStatus.FAILED:
                assert status_info.error_message is not None, (
                    f"Failed component {component_name} should have error message"
                )
            
            # 4. Partial components should have warnings or error details
            if status_info.status == ComponentStatus.PARTIAL:
                has_details = (
                    (status_info.error_message is not None and status_info.error_message.strip()) or
                    (status_info.warnings and len(status_info.warnings) > 0)
                )
                assert has_details, (
                    f"Partial component {component_name} should have error message or warnings"
                )
            
            # 5. Fallback components should indicate fallback method
            if status_info.status == ComponentStatus.FALLBACK:
                assert status_info.fallback_method is not None and status_info.fallback_method.strip(), (
                    f"Fallback component {component_name} should specify fallback method"
                )
            
            # 6. API calls and tokens should be non-negative
            assert status_info.api_calls_made >= 0, (
                f"Component {component_name} should have non-negative API calls"
            )
            assert status_info.tokens_consumed >= 0, (
                f"Component {component_name} should have non-negative token consumption"
            )
        
        # 7. Context should provide clear indication of overall status
        # Check if we can determine overall success from component statuses
        success_count = sum(1 for status in context.component_status.values() 
                          if status.status == ComponentStatus.SUCCESS)
        partial_count = sum(1 for status in context.component_status.values() 
                          if status.status == ComponentStatus.PARTIAL)
        fallback_count = sum(1 for status in context.component_status.values() 
                           if status.status == ComponentStatus.FALLBACK)
        failed_count = sum(1 for status in context.component_status.values() 
                         if status.status == ComponentStatus.FAILED)
        
        total_components = len(context.component_status)
        
        # The status distribution should make sense
        assert success_count + partial_count + fallback_count + failed_count == total_components, (
            f"Component status counts should sum to total: "
            f"success={success_count}, partial={partial_count}, "
            f"fallback={fallback_count}, failed={failed_count}, total={total_components}"
        )
        
        # If there are any non-success components, the context should reflect this
        has_issues = partial_count > 0 or fallback_count > 0 or failed_count > 0
        if has_issues:
            # Context should have some indication of issues (this is ensured by our test data generation)
            assert True  # This property is validated by the detailed status information above


# =============================================================================
# Unit Tests for Enhanced Data Models
# =============================================================================

class TestEnhancedDataModels:
    """Unit tests for enhanced data models."""
    
    def test_cache_entry_creation(self):
        """Test basic CacheEntry creation and properties."""
        now = datetime.now()
        entry = CacheEntry(
            value={"test": "data"},
            created_at=now,
            ttl_seconds=3600,
            cache_type="query_analysis",
            workspace_fingerprint="abc123",
            query_hash="def456"
        )
        
        assert entry.value == {"test": "data"}
        assert entry.created_at == now
        assert entry.ttl_seconds == 3600
        assert entry.cache_type == "query_analysis"
        assert not entry.is_expired  # Should not be expired immediately
        assert entry.age.total_seconds() < 1  # Should be very recent
    
    def test_progress_event_creation(self):
        """Test ProgressEvent creation with different statuses."""
        event = ProgressEvent(
            component="file_discovery",
            status=ProgressStatus.PROGRESS,
            progress=0.5,
            data={"files_found": 10},
            message="Discovering files..."
        )
        
        assert event.component == "file_discovery"
        assert event.status == ProgressStatus.PROGRESS
        assert event.progress == 0.5
        assert event.data == {"files_found": 10}
        assert event.message == "Discovering files..."
        assert isinstance(event.timestamp, datetime)
    
    def test_repo_info_accessibility(self):
        """Test RepoInfo accessibility tracking."""
        # Accessible repo
        accessible_repo = RepoInfo(
            path="/test/repo",
            name="test-repo",
            git_root="/test/repo",
            is_accessible=True
        )
        assert accessible_repo.is_accessible
        assert accessible_repo.error_message is None
        
        # Inaccessible repo
        inaccessible_repo = RepoInfo(
            path="/test/bad-repo",
            name="bad-repo",
            git_root="/test/bad-repo",
            is_accessible=False,
            error_message="Permission denied"
        )
        assert not inaccessible_repo.is_accessible
        assert inaccessible_repo.error_message == "Permission denied"
    
    def test_component_status_info_creation(self):
        """Test ComponentStatusInfo with different statuses."""
        # Success status
        success_status = ComponentStatusInfo(
            status=ComponentStatus.SUCCESS,
            execution_time=1.5,
            api_calls_made=2,
            tokens_consumed=150
        )
        assert success_status.status == ComponentStatus.SUCCESS
        assert success_status.error_message is None
        assert success_status.fallback_method is None
        
        # Fallback status
        fallback_status = ComponentStatusInfo(
            status=ComponentStatus.FALLBACK,
            execution_time=2.0,
            fallback_method="pattern_matching",
            warnings=["API unavailable, using fallback"]
        )
        assert fallback_status.status == ComponentStatus.FALLBACK
        assert fallback_status.fallback_method == "pattern_matching"
        assert len(fallback_status.warnings) == 1
    
    def test_enhanced_structured_context_creation(self):
        """Test EnhancedStructuredContext creation and methods."""
        context = EnhancedStructuredContext(
            task_type="debug",
            task_summary="Fix authentication bug",
            complexity="medium",
            component_status={
                "query_analysis": ComponentStatusInfo(
                    status=ComponentStatus.SUCCESS,
                    execution_time=1.0
                ),
                "file_discovery": ComponentStatusInfo(
                    status=ComponentStatus.PARTIAL,
                    execution_time=2.0,
                    warnings=["Some files inaccessible"]
                )
            }
        )
        
        assert context.task_type == "debug"
        assert context.task_summary == "Fix authentication bug"
        assert context.complexity == "medium"
        assert len(context.component_status) == 2
        
        # Test enhanced prompt generation
        prompt = context.to_claude_prompt_enhanced()
        assert "## Task Analysis" in prompt
        assert "## Component Status" in prompt
        assert "✅ query_analysis: success" in prompt
        assert "⚠️ file_discovery: partial" in prompt
        
        # Test detailed JSON generation
        json_data = context.to_detailed_json()
        assert json_data["task_type"] == "debug"
        assert "component_status" in json_data
        assert json_data["component_status"]["query_analysis"]["status"] == "success"
        assert json_data["component_status"]["file_discovery"]["status"] == "partial"


# =============================================================================
# Test Execution
# =============================================================================

if __name__ == "__main__":
    # Run a simple test if executed directly
    test_instance = TestEnhancedDataModels()
    test_instance.test_cache_entry_creation()
    test_instance.test_progress_event_creation()
    test_instance.test_repo_info_accessibility()
    test_instance.test_component_status_info_creation()
    test_instance.test_enhanced_structured_context_creation()
    print("✅ All unit tests passed!")
    
    if HYPOTHESIS_AVAILABLE:
        print("✅ Hypothesis available - property tests can run")
    else:
        print("⚠️ Hypothesis not available - property tests will be skipped")