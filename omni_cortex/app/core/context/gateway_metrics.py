"""
Context Gateway Metrics System

Comprehensive metrics collection for context gateway operations including:
- API call tracking (counts, tokens, timing)
- Component performance monitoring
- Context quality scoring
- Integration with Prometheus metrics
- Detailed performance analytics

This module provides the ContextGatewayMetrics class that serves as the
central metrics collection point for all context gateway operations.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import structlog

from ..metrics import PROMETHEUS_AVAILABLE
from .enhanced_models import (
    ComponentMetrics,
    QualityMetrics,
)

logger = structlog.get_logger("gateway_metrics")


@dataclass
class APICallMetrics:
    """Detailed metrics for a single API call."""

    model: str
    provider: str  # "google", "openai", etc.
    component: str  # Which component made the call
    tokens_used: int
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str | None = None
    thinking_mode_used: bool = False


@dataclass
class ComponentPerformanceMetrics:
    """Performance metrics for a component over time."""

    component_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration: float = 0.0
    total_api_calls: int = 0
    total_tokens: int = 0
    avg_duration: float = 0.0
    avg_tokens: int = 0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    fallback_uses: int = 0

    def update(self, metrics: ComponentMetrics) -> None:
        """Update performance metrics with new execution data."""
        self.total_executions += 1

        if metrics.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        self.total_duration += metrics.execution_time
        self.total_api_calls += metrics.api_calls_made
        self.total_tokens += metrics.tokens_consumed

        # Update averages
        self.avg_duration = self.total_duration / self.total_executions
        self.avg_tokens = (
            self.total_tokens / self.total_executions if self.total_executions > 0 else 0
        )

        # Update min/max
        self.min_duration = min(self.min_duration, metrics.execution_time)
        self.max_duration = max(self.max_duration, metrics.execution_time)

        # Track cache and fallback usage
        if metrics.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if metrics.fallback_used:
            self.fallback_uses += 1


class ContextGatewayMetrics:
    """
    Comprehensive metrics collection for context gateway operations.

    Features:
    - API call tracking with detailed breakdowns
    - Component performance monitoring
    - Context quality scoring
    - Token usage analytics
    - Prometheus integration
    - Historical performance tracking
    """

    def __init__(self, retention_days: int = 30):
        """
        Initialize context gateway metrics.

        Args:
            retention_days: How many days of metrics to retain
        """
        self._retention_days = retention_days

        # API call tracking
        self._api_calls: list[APICallMetrics] = []
        self._api_calls_by_model: dict[str, list[APICallMetrics]] = defaultdict(list)
        self._api_calls_by_component: dict[str, list[APICallMetrics]] = defaultdict(list)

        # Component performance tracking
        self._component_performance: dict[str, ComponentPerformanceMetrics] = {}

        # Context quality tracking
        self._quality_scores: list[tuple[datetime, float]] = []
        self._quality_by_task_type: dict[str, list[float]] = defaultdict(list)

        # Token usage tracking
        self._token_usage_by_component: dict[str, int] = defaultdict(int)
        self._token_usage_by_model: dict[str, int] = defaultdict(int)
        self._total_tokens_used: int = 0

        # Timing tracking
        self._execution_times: list[tuple[datetime, float]] = []
        self._execution_times_by_complexity: dict[str, list[float]] = defaultdict(list)

        # Session tracking
        self._total_sessions: int = 0
        self._successful_sessions: int = 0
        self._failed_sessions: int = 0

        logger.info(
            "gateway_metrics_initialized",
            retention_days=retention_days,
            prometheus_available=PROMETHEUS_AVAILABLE,
        )

    def record_api_call(
        self,
        component: str,
        model: str,
        provider: str,
        tokens: int,
        duration: float,
        success: bool = True,
        error_message: str | None = None,
        thinking_mode: bool = False,
    ) -> None:
        """
        Record an API call with detailed metrics.

        Args:
            component: Component that made the call
            model: Model name (e.g., "gemini-flash-2.0")
            provider: Provider name (e.g., "google")
            tokens: Number of tokens used
            duration: Call duration in seconds
            success: Whether the call succeeded
            error_message: Error message if failed
            thinking_mode: Whether thinking mode was used
        """
        call_metrics = APICallMetrics(
            model=model,
            provider=provider,
            component=component,
            tokens_used=tokens,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            thinking_mode_used=thinking_mode,
        )

        # Store in various indexes
        self._api_calls.append(call_metrics)
        self._api_calls_by_model[model].append(call_metrics)
        self._api_calls_by_component[component].append(call_metrics)

        # Update token usage
        self._token_usage_by_component[component] += tokens
        self._token_usage_by_model[model] += tokens
        self._total_tokens_used += tokens

        logger.debug(
            "api_call_recorded",
            component=component,
            model=model,
            tokens=tokens,
            duration=duration,
            success=success,
            thinking_mode=thinking_mode,
        )

    def record_component_performance(
        self,
        component: str,
        duration: float,
        success: bool,
        api_calls: int = 0,
        tokens: int = 0,
        cache_hit: bool = False,
        fallback_used: bool = False,
        error_message: str | None = None,
    ) -> None:
        """
        Record component performance metrics.

        Args:
            component: Component name
            duration: Execution duration in seconds
            success: Whether execution succeeded
            api_calls: Number of API calls made
            tokens: Total tokens consumed
            cache_hit: Whether cache was hit
            fallback_used: Whether fallback was used
            error_message: Error message if failed
        """
        metrics = ComponentMetrics(
            component_name=component,
            execution_time=duration,
            api_calls_made=api_calls,
            tokens_consumed=tokens,
            success=success,
            error_message=error_message,
            fallback_used=fallback_used,
            cache_hit=cache_hit,
        )

        # Initialize component performance if needed
        if component not in self._component_performance:
            self._component_performance[component] = ComponentPerformanceMetrics(
                component_name=component
            )

        # Update performance metrics
        self._component_performance[component].update(metrics)

        logger.debug(
            "component_performance_recorded",
            component=component,
            duration=duration,
            success=success,
            cache_hit=cache_hit,
            fallback_used=fallback_used,
        )

    def record_context_quality(
        self,
        quality_score: float,
        task_type: str | None = None,
        component_scores: dict[str, float] | None = None,
        relevance_scores: list[float] | None = None,
        completeness_score: float | None = None,
    ) -> QualityMetrics:
        """
        Record context quality metrics.

        Args:
            quality_score: Overall quality score (0.0 to 1.0)
            task_type: Type of task
            component_scores: Quality scores by component
            relevance_scores: List of relevance scores for context elements
            completeness_score: How complete the context is

        Returns:
            QualityMetrics object with computed metrics
        """
        # Store quality score with timestamp
        self._quality_scores.append((datetime.now(), quality_score))

        # Track by task type
        if task_type:
            self._quality_by_task_type[task_type].append(quality_score)

        # Compute confidence intervals for component scores
        confidence_intervals = {}
        if component_scores:
            for component, score in component_scores.items():
                # Simple confidence interval based on score
                # In production, this would use historical data
                margin = 0.1 * (1.0 - score)  # Higher uncertainty for lower scores
                confidence_intervals[component] = (
                    max(0.0, score - margin),
                    min(1.0, score + margin),
                )

        # Create quality metrics object
        quality_metrics = QualityMetrics(
            overall_quality_score=quality_score,
            component_quality_scores=component_scores or {},
            confidence_intervals=confidence_intervals,
            completeness_score=completeness_score or 0.0,
            relevance_distribution=relevance_scores or [],
        )

        logger.info(
            "context_quality_recorded",
            quality_score=quality_score,
            task_type=task_type,
            completeness=completeness_score,
        )

        return quality_metrics

    def record_session(self, duration: float, success: bool, complexity: str | None = None) -> None:
        """
        Record a complete context preparation session.

        Args:
            duration: Total session duration in seconds
            success: Whether session succeeded
            complexity: Task complexity level
        """
        self._total_sessions += 1

        if success:
            self._successful_sessions += 1
        else:
            self._failed_sessions += 1

        # Track execution time
        self._execution_times.append((datetime.now(), duration))

        if complexity:
            self._execution_times_by_complexity[complexity].append(duration)

        logger.info(
            "session_recorded",
            duration=duration,
            success=success,
            complexity=complexity,
            total_sessions=self._total_sessions,
        )

    def get_api_call_summary(
        self,
        component: str | None = None,
        model: str | None = None,
        time_window_hours: int | None = None,
    ) -> dict[str, Any]:
        """
        Get summary of API calls.

        Args:
            component: Filter by component
            model: Filter by model
            time_window_hours: Only include calls from last N hours

        Returns:
            Dictionary with API call summary
        """
        # Filter calls
        calls = self._api_calls

        if time_window_hours:
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            calls = [c for c in calls if c.timestamp > cutoff]

        if component:
            calls = [c for c in calls if c.component == component]

        if model:
            calls = [c for c in calls if c.model == model]

        if not calls:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_tokens": 0,
                "avg_duration": 0.0,
                "thinking_mode_calls": 0,
            }

        # Compute summary
        successful = [c for c in calls if c.success]
        failed = [c for c in calls if not c.success]
        thinking_mode = [c for c in calls if c.thinking_mode_used]

        total_tokens = sum(c.tokens_used for c in calls)
        durations = [c.duration_seconds for c in calls]

        return {
            "total_calls": len(calls),
            "successful_calls": len(successful),
            "failed_calls": len(failed),
            "success_rate": len(successful) / len(calls) if calls else 0.0,
            "total_tokens": total_tokens,
            "avg_tokens_per_call": total_tokens / len(calls) if calls else 0,
            "avg_duration": statistics.mean(durations) if durations else 0.0,
            "min_duration": min(durations) if durations else 0.0,
            "max_duration": max(durations) if durations else 0.0,
            "thinking_mode_calls": len(thinking_mode),
            "thinking_mode_percentage": len(thinking_mode) / len(calls) * 100 if calls else 0.0,
            "calls_by_model": self._group_by_field(calls, "model"),
            "calls_by_component": self._group_by_field(calls, "component"),
        }

    def get_component_performance_summary(self, component: str | None = None) -> dict[str, Any]:
        """
        Get component performance summary.

        Args:
            component: Specific component to get metrics for

        Returns:
            Dictionary with component performance metrics
        """
        if component:
            if component not in self._component_performance:
                return {"error": f"No metrics for component: {component}"}

            perf = self._component_performance[component]
            return {
                "component": component,
                "total_executions": perf.total_executions,
                "successful_executions": perf.successful_executions,
                "failed_executions": perf.failed_executions,
                "success_rate": perf.successful_executions / perf.total_executions
                if perf.total_executions > 0
                else 0.0,
                "avg_duration": perf.avg_duration,
                "min_duration": perf.min_duration if perf.min_duration != float("inf") else 0.0,
                "max_duration": perf.max_duration,
                "avg_tokens": perf.avg_tokens,
                "total_tokens": perf.total_tokens,
                "cache_hit_rate": perf.cache_hits / (perf.cache_hits + perf.cache_misses)
                if (perf.cache_hits + perf.cache_misses) > 0
                else 0.0,
                "fallback_rate": perf.fallback_uses / perf.total_executions
                if perf.total_executions > 0
                else 0.0,
            }

        # Return summary for all components
        return {
            component_name: {
                "total_executions": perf.total_executions,
                "success_rate": perf.successful_executions / perf.total_executions
                if perf.total_executions > 0
                else 0.0,
                "avg_duration": perf.avg_duration,
                "avg_tokens": perf.avg_tokens,
                "cache_hit_rate": perf.cache_hits / (perf.cache_hits + perf.cache_misses)
                if (perf.cache_hits + perf.cache_misses) > 0
                else 0.0,
                "fallback_rate": perf.fallback_uses / perf.total_executions
                if perf.total_executions > 0
                else 0.0,
            }
            for component_name, perf in self._component_performance.items()
        }

    def get_quality_summary(
        self, task_type: str | None = None, time_window_hours: int | None = None
    ) -> dict[str, Any]:
        """
        Get context quality summary.

        Args:
            task_type: Filter by task type
            time_window_hours: Only include recent scores

        Returns:
            Dictionary with quality metrics summary
        """
        # Filter scores
        scores = self._quality_scores

        if time_window_hours:
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            scores = [(ts, score) for ts, score in scores if ts > cutoff]

        if not scores:
            return {
                "total_measurements": 0,
                "avg_quality": 0.0,
                "min_quality": 0.0,
                "max_quality": 0.0,
            }

        score_values = [score for _, score in scores]

        summary = {
            "total_measurements": len(scores),
            "avg_quality": statistics.mean(score_values),
            "median_quality": statistics.median(score_values),
            "min_quality": min(score_values),
            "max_quality": max(score_values),
            "std_dev": statistics.stdev(score_values) if len(score_values) > 1 else 0.0,
        }

        # Add task type breakdown
        if task_type:
            task_scores = self._quality_by_task_type.get(task_type, [])
            if task_scores:
                summary["task_type_avg"] = statistics.mean(task_scores)
                summary["task_type_count"] = len(task_scores)
        else:
            summary["by_task_type"] = {
                tt: {"avg_quality": statistics.mean(scores), "count": len(scores)}
                for tt, scores in self._quality_by_task_type.items()
                if scores
            }

        return summary

    def get_token_usage_summary(self) -> dict[str, Any]:
        """
        Get token usage summary.

        Returns:
            Dictionary with token usage breakdown
        """
        return {
            "total_tokens": self._total_tokens_used,
            "by_component": dict(self._token_usage_by_component),
            "by_model": dict(self._token_usage_by_model),
            "avg_tokens_per_session": self._total_tokens_used / self._total_sessions
            if self._total_sessions > 0
            else 0,
        }

    def get_session_summary(self) -> dict[str, Any]:
        """
        Get session summary statistics.

        Returns:
            Dictionary with session statistics
        """
        durations = [duration for _, duration in self._execution_times]

        summary = {
            "total_sessions": self._total_sessions,
            "successful_sessions": self._successful_sessions,
            "failed_sessions": self._failed_sessions,
            "success_rate": self._successful_sessions / self._total_sessions
            if self._total_sessions > 0
            else 0.0,
        }

        if durations:
            summary.update(
                {
                    "avg_duration": statistics.mean(durations),
                    "median_duration": statistics.median(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                }
            )

        # Add complexity breakdown
        if self._execution_times_by_complexity:
            summary["by_complexity"] = {
                complexity: {"avg_duration": statistics.mean(times), "count": len(times)}
                for complexity, times in self._execution_times_by_complexity.items()
                if times
            }

        return summary

    def get_comprehensive_dashboard(self) -> dict[str, Any]:
        """
        Get comprehensive metrics dashboard.

        Returns:
            Dictionary with all metrics formatted for dashboard display
        """
        return {
            "overview": {
                "total_sessions": self._total_sessions,
                "success_rate": f"{(self._successful_sessions / self._total_sessions * 100) if self._total_sessions > 0 else 0:.1f}%",
                "total_api_calls": len(self._api_calls),
                "total_tokens": f"{self._total_tokens_used:,}",
                "avg_tokens_per_session": f"{(self._total_tokens_used / self._total_sessions) if self._total_sessions > 0 else 0:,.0f}",
            },
            "api_calls": self.get_api_call_summary(),
            "component_performance": self.get_component_performance_summary(),
            "quality_metrics": self.get_quality_summary(),
            "token_usage": self.get_token_usage_summary(),
            "session_stats": self.get_session_summary(),
        }

    def cleanup_old_data(self) -> int:
        """
        Remove metrics older than retention period.

        Returns:
            Number of items removed
        """
        cutoff = datetime.now() - timedelta(days=self._retention_days)

        # Clean API calls
        old_count = len(self._api_calls)
        self._api_calls = [c for c in self._api_calls if c.timestamp > cutoff]
        removed = old_count - len(self._api_calls)

        # Rebuild indexes
        self._api_calls_by_model.clear()
        self._api_calls_by_component.clear()
        for call in self._api_calls:
            self._api_calls_by_model[call.model].append(call)
            self._api_calls_by_component[call.component].append(call)

        # Clean quality scores
        old_quality_count = len(self._quality_scores)
        self._quality_scores = [(ts, score) for ts, score in self._quality_scores if ts > cutoff]
        removed += old_quality_count - len(self._quality_scores)

        # Clean execution times
        old_exec_count = len(self._execution_times)
        self._execution_times = [
            (ts, duration) for ts, duration in self._execution_times if ts > cutoff
        ]
        removed += old_exec_count - len(self._execution_times)

        logger.info(
            "metrics_cleanup_completed", items_removed=removed, retention_days=self._retention_days
        )

        return removed

    def _group_by_field(self, calls: list[APICallMetrics], field: str) -> dict[str, int]:
        """Group API calls by a field and count."""
        groups = defaultdict(int)
        for call in calls:
            value = getattr(call, field)
            groups[value] += 1
        return dict(groups)

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        self._api_calls.clear()
        self._api_calls_by_model.clear()
        self._api_calls_by_component.clear()
        self._component_performance.clear()
        self._quality_scores.clear()
        self._quality_by_task_type.clear()
        self._token_usage_by_component.clear()
        self._token_usage_by_model.clear()
        self._total_tokens_used = 0
        self._execution_times.clear()
        self._execution_times_by_complexity.clear()
        self._total_sessions = 0
        self._successful_sessions = 0
        self._failed_sessions = 0

        logger.info("gateway_metrics_reset")


# =============================================================================
# Global singleton
# =============================================================================

_gateway_metrics: ContextGatewayMetrics | None = None


def get_gateway_metrics() -> ContextGatewayMetrics:
    """Get the global ContextGatewayMetrics singleton."""
    global _gateway_metrics

    if _gateway_metrics is None:
        from ..settings import get_settings

        settings = get_settings()
        _gateway_metrics = ContextGatewayMetrics(retention_days=settings.metrics_retention_days)

    return _gateway_metrics


def reset_gateway_metrics() -> None:
    """Reset gateway metrics singleton (useful for testing)."""
    global _gateway_metrics
    if _gateway_metrics is not None:
        _gateway_metrics.reset()
    _gateway_metrics = None
