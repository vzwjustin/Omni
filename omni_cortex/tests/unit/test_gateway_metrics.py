"""
Unit tests for ContextGatewayMetrics

Tests the comprehensive metrics collection system for context gateway operations.
"""

import pytest
from datetime import datetime, timedelta
from app.core.context.gateway_metrics import (
    ContextGatewayMetrics,
    get_gateway_metrics,
    reset_gateway_metrics,
    APICallMetrics,
    ComponentPerformanceMetrics
)
from app.core.context.enhanced_models import (
    ComponentMetrics,
    QualityMetrics
)


class TestContextGatewayMetrics:
    """Test suite for ContextGatewayMetrics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = ContextGatewayMetrics(retention_days=7)
    
    def test_initialization(self):
        """Test metrics initialization."""
        assert self.metrics._retention_days == 7
        assert len(self.metrics._api_calls) == 0
        assert len(self.metrics._component_performance) == 0
        assert self.metrics._total_tokens_used == 0
        assert self.metrics._total_sessions == 0
    
    def test_record_api_call(self):
        """Test API call recording."""
        self.metrics.record_api_call(
            component="query_analyzer",
            model="gemini-flash-2.0",
            provider="google",
            tokens=500,
            duration=1.5,
            success=True,
            thinking_mode=True
        )
        
        assert len(self.metrics._api_calls) == 1
        assert self.metrics._total_tokens_used == 500
        assert self.metrics._token_usage_by_component["query_analyzer"] == 500
        assert self.metrics._token_usage_by_model["gemini-flash-2.0"] == 500
        
        call = self.metrics._api_calls[0]
        assert call.component == "query_analyzer"
        assert call.model == "gemini-flash-2.0"
        assert call.tokens_used == 500
        assert call.thinking_mode_used is True
    
    def test_record_multiple_api_calls(self):
        """Test recording multiple API calls."""
        self.metrics.record_api_call("query_analyzer", "gemini-flash-2.0", "google", 500, 1.5)
        self.metrics.record_api_call("file_discoverer", "gemini-flash-2.0", "google", 1000, 2.0)
        self.metrics.record_api_call("doc_searcher", "gemini-pro", "google", 750, 1.8)
        
        assert len(self.metrics._api_calls) == 3
        assert self.metrics._total_tokens_used == 2250
        assert self.metrics._token_usage_by_component["query_analyzer"] == 500
        assert self.metrics._token_usage_by_component["file_discoverer"] == 1000
        assert self.metrics._token_usage_by_model["gemini-flash-2.0"] == 1500
        assert self.metrics._token_usage_by_model["gemini-pro"] == 750
    
    def test_record_component_performance(self):
        """Test component performance recording."""
        self.metrics.record_component_performance(
            component="file_discoverer",
            duration=2.3,
            success=True,
            api_calls=2,
            tokens=1000,
            cache_hit=False,
            fallback_used=False
        )
        
        assert "file_discoverer" in self.metrics._component_performance
        perf = self.metrics._component_performance["file_discoverer"]
        assert perf.total_executions == 1
        assert perf.successful_executions == 1
        assert perf.total_duration == 2.3
        assert perf.total_tokens == 1000
        assert perf.cache_misses == 1
    
    def test_component_performance_aggregation(self):
        """Test component performance aggregation over multiple executions."""
        # Record multiple executions
        self.metrics.record_component_performance("file_discoverer", 2.0, True, 2, 1000, False, False)
        self.metrics.record_component_performance("file_discoverer", 3.0, True, 2, 1500, True, False)
        self.metrics.record_component_performance("file_discoverer", 1.5, False, 1, 500, False, True)
        
        perf = self.metrics._component_performance["file_discoverer"]
        assert perf.total_executions == 3
        assert perf.successful_executions == 2
        assert perf.failed_executions == 1
        assert perf.total_duration == 6.5
        assert perf.avg_duration == pytest.approx(2.167, rel=0.01)
        assert perf.total_tokens == 3000
        assert perf.avg_tokens == 1000
        assert perf.cache_hits == 1
        assert perf.cache_misses == 2
        assert perf.fallback_uses == 1
        assert perf.min_duration == 1.5
        assert perf.max_duration == 3.0
    
    def test_record_context_quality(self):
        """Test context quality recording."""
        quality_metrics = self.metrics.record_context_quality(
            quality_score=0.85,
            task_type="debug",
            component_scores={
                "query_analyzer": 0.9,
                "file_discoverer": 0.8,
                "doc_searcher": 0.85
            },
            relevance_scores=[0.9, 0.8, 0.7, 0.85],
            completeness_score=0.9
        )
        
        assert quality_metrics.overall_quality_score == 0.85
        assert quality_metrics.completeness_score == 0.9
        assert len(quality_metrics.component_quality_scores) == 3
        assert len(quality_metrics.relevance_distribution) == 4
        assert len(self.metrics._quality_scores) == 1
        assert len(self.metrics._quality_by_task_type["debug"]) == 1
    
    def test_record_session(self):
        """Test session recording."""
        self.metrics.record_session(duration=5.5, success=True, complexity="medium")
        
        assert self.metrics._total_sessions == 1
        assert self.metrics._successful_sessions == 1
        assert self.metrics._failed_sessions == 0
        assert len(self.metrics._execution_times) == 1
        assert len(self.metrics._execution_times_by_complexity["medium"]) == 1
    
    def test_get_api_call_summary(self):
        """Test API call summary generation."""
        # Record some calls
        self.metrics.record_api_call("query_analyzer", "gemini-flash-2.0", "google", 500, 1.5, True)
        self.metrics.record_api_call("file_discoverer", "gemini-flash-2.0", "google", 1000, 2.0, True)
        self.metrics.record_api_call("doc_searcher", "gemini-pro", "google", 750, 1.8, False)
        
        summary = self.metrics.get_api_call_summary()
        
        assert summary["total_calls"] == 3
        assert summary["successful_calls"] == 2
        assert summary["failed_calls"] == 1
        assert summary["success_rate"] == pytest.approx(0.667, rel=0.01)
        assert summary["total_tokens"] == 2250
        assert summary["avg_tokens_per_call"] == 750
        assert summary["thinking_mode_calls"] == 0
    
    def test_get_api_call_summary_with_filters(self):
        """Test API call summary with filters."""
        # Record calls
        self.metrics.record_api_call("query_analyzer", "gemini-flash-2.0", "google", 500, 1.5)
        self.metrics.record_api_call("file_discoverer", "gemini-flash-2.0", "google", 1000, 2.0)
        self.metrics.record_api_call("doc_searcher", "gemini-pro", "google", 750, 1.8)
        
        # Filter by component
        summary = self.metrics.get_api_call_summary(component="query_analyzer")
        assert summary["total_calls"] == 1
        assert summary["total_tokens"] == 500
        
        # Filter by model
        summary = self.metrics.get_api_call_summary(model="gemini-flash-2.0")
        assert summary["total_calls"] == 2
        assert summary["total_tokens"] == 1500
    
    def test_get_component_performance_summary(self):
        """Test component performance summary."""
        # Record performance for multiple components
        self.metrics.record_component_performance("file_discoverer", 2.0, True, 2, 1000, False, False)
        self.metrics.record_component_performance("file_discoverer", 3.0, True, 2, 1500, True, False)
        self.metrics.record_component_performance("doc_searcher", 1.5, True, 1, 500, False, False)
        
        # Get summary for specific component
        summary = self.metrics.get_component_performance_summary(component="file_discoverer")
        assert summary["component"] == "file_discoverer"
        assert summary["total_executions"] == 2
        assert summary["success_rate"] == 1.0
        assert summary["avg_duration"] == 2.5
        assert summary["cache_hit_rate"] == 0.5
        
        # Get summary for all components
        all_summary = self.metrics.get_component_performance_summary()
        assert len(all_summary) == 2
        assert "file_discoverer" in all_summary
        assert "doc_searcher" in all_summary
    
    def test_get_quality_summary(self):
        """Test quality summary generation."""
        # Record quality scores
        self.metrics.record_context_quality(0.85, task_type="debug")
        self.metrics.record_context_quality(0.90, task_type="debug")
        self.metrics.record_context_quality(0.75, task_type="implement")
        
        summary = self.metrics.get_quality_summary()
        assert summary["total_measurements"] == 3
        assert summary["avg_quality"] == pytest.approx(0.833, rel=0.01)
        assert summary["min_quality"] == 0.75
        assert summary["max_quality"] == 0.90
        
        # Filter by task type
        debug_summary = self.metrics.get_quality_summary(task_type="debug")
        assert debug_summary["task_type_avg"] == pytest.approx(0.875, rel=0.01)
        assert debug_summary["task_type_count"] == 2
    
    def test_get_token_usage_summary(self):
        """Test token usage summary."""
        self.metrics.record_api_call("query_analyzer", "gemini-flash-2.0", "google", 500, 1.5)
        self.metrics.record_api_call("file_discoverer", "gemini-flash-2.0", "google", 1000, 2.0)
        self.metrics.record_api_call("doc_searcher", "gemini-pro", "google", 750, 1.8)
        self.metrics.record_session(5.5, True)
        
        summary = self.metrics.get_token_usage_summary()
        assert summary["total_tokens"] == 2250
        assert summary["by_component"]["query_analyzer"] == 500
        assert summary["by_component"]["file_discoverer"] == 1000
        assert summary["by_model"]["gemini-flash-2.0"] == 1500
        assert summary["avg_tokens_per_session"] == 2250
    
    def test_get_session_summary(self):
        """Test session summary."""
        self.metrics.record_session(5.5, True, "medium")
        self.metrics.record_session(3.2, True, "low")
        self.metrics.record_session(8.1, False, "high")
        
        summary = self.metrics.get_session_summary()
        assert summary["total_sessions"] == 3
        assert summary["successful_sessions"] == 2
        assert summary["failed_sessions"] == 1
        assert summary["success_rate"] == pytest.approx(0.667, rel=0.01)
        assert summary["avg_duration"] == pytest.approx(5.6, rel=0.01)
        assert "by_complexity" in summary
        assert len(summary["by_complexity"]) == 3
    
    def test_get_comprehensive_dashboard(self):
        """Test comprehensive dashboard generation."""
        # Record various metrics
        self.metrics.record_api_call("query_analyzer", "gemini-flash-2.0", "google", 500, 1.5)
        self.metrics.record_component_performance("file_discoverer", 2.0, True, 2, 1000)
        self.metrics.record_context_quality(0.85, task_type="debug")
        self.metrics.record_session(5.5, True, "medium")
        
        dashboard = self.metrics.get_comprehensive_dashboard()
        
        assert "overview" in dashboard
        assert "api_calls" in dashboard
        assert "component_performance" in dashboard
        assert "quality_metrics" in dashboard
        assert "token_usage" in dashboard
        assert "session_stats" in dashboard
        
        assert dashboard["overview"]["total_sessions"] == 1
        assert "100.0%" in dashboard["overview"]["success_rate"]
    
    def test_cleanup_old_data(self):
        """Test cleanup of old data."""
        # Create metrics with short retention
        metrics = ContextGatewayMetrics(retention_days=1)
        
        # Record some data
        metrics.record_api_call("query_analyzer", "gemini-flash-2.0", "google", 500, 1.5)
        metrics.record_context_quality(0.85)
        metrics.record_session(5.5, True)
        
        # Manually set old timestamps
        old_time = datetime.now() - timedelta(days=2)
        metrics._api_calls[0].timestamp = old_time
        metrics._quality_scores[0] = (old_time, 0.85)
        metrics._execution_times[0] = (old_time, 5.5)
        
        # Cleanup
        removed = metrics.cleanup_old_data()
        
        assert removed == 3
        assert len(metrics._api_calls) == 0
        assert len(metrics._quality_scores) == 0
        assert len(metrics._execution_times) == 0
    
    def test_reset(self):
        """Test metrics reset."""
        # Record some data
        self.metrics.record_api_call("query_analyzer", "gemini-flash-2.0", "google", 500, 1.5)
        self.metrics.record_component_performance("file_discoverer", 2.0, True, 2, 1000)
        self.metrics.record_session(5.5, True)
        
        # Reset
        self.metrics.reset()
        
        assert len(self.metrics._api_calls) == 0
        assert len(self.metrics._component_performance) == 0
        assert self.metrics._total_tokens_used == 0
        assert self.metrics._total_sessions == 0
    
    def test_singleton(self):
        """Test singleton pattern."""
        metrics1 = get_gateway_metrics()
        metrics2 = get_gateway_metrics()
        
        assert metrics1 is metrics2
        
        # Reset singleton
        reset_gateway_metrics()
        metrics3 = get_gateway_metrics()
        
        # Should be a new instance
        assert metrics3 is not metrics1


class TestAPICallMetrics:
    """Test suite for APICallMetrics dataclass."""
    
    def test_creation(self):
        """Test APICallMetrics creation."""
        call = APICallMetrics(
            model="gemini-flash-2.0",
            provider="google",
            component="query_analyzer",
            tokens_used=500,
            duration_seconds=1.5,
            success=True,
            thinking_mode_used=True
        )
        
        assert call.model == "gemini-flash-2.0"
        assert call.provider == "google"
        assert call.component == "query_analyzer"
        assert call.tokens_used == 500
        assert call.duration_seconds == 1.5
        assert call.success is True
        assert call.thinking_mode_used is True
        assert isinstance(call.timestamp, datetime)


class TestComponentPerformanceMetrics:
    """Test suite for ComponentPerformanceMetrics dataclass."""
    
    def test_creation(self):
        """Test ComponentPerformanceMetrics creation."""
        perf = ComponentPerformanceMetrics(component_name="file_discoverer")
        
        assert perf.component_name == "file_discoverer"
        assert perf.total_executions == 0
        assert perf.avg_duration == 0.0
        assert perf.cache_hits == 0
    
    def test_update(self):
        """Test performance metrics update."""
        perf = ComponentPerformanceMetrics(component_name="file_discoverer")
        
        metrics = ComponentMetrics(
            component_name="file_discoverer",
            execution_time=2.5,
            api_calls_made=2,
            tokens_consumed=1000,
            success=True,
            cache_hit=True
        )
        
        perf.update(metrics)
        
        assert perf.total_executions == 1
        assert perf.successful_executions == 1
        assert perf.total_duration == 2.5
        assert perf.avg_duration == 2.5
        assert perf.total_tokens == 1000
        assert perf.cache_hits == 1
