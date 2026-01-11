"""
Enhanced Status Tracking for Context Gateway Components

Provides comprehensive status tracking, error reporting, and component
health monitoring for the context gateway system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from .enhanced_models import (
    ComponentMetrics,
    ComponentStatus,
    ComponentStatusInfo,
)

logger = structlog.get_logger("status_tracking")


@dataclass
class DetailedErrorReport:
    """Detailed error report for component failures."""

    component: str
    error_type: str
    error_message: str
    timestamp: datetime
    stack_trace: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False


class ComponentStatusTracker:
    """
    Tracks status of all context gateway components.

    Provides:
    - Real-time component status
    - Error tracking and reporting
    - Success/failure/fallback indicators
    - Component health metrics
    """

    def __init__(self):
        self._component_status: dict[str, ComponentStatusInfo] = {}
        self._error_reports: list[DetailedErrorReport] = []
        self._component_metrics: dict[str, ComponentMetrics] = {}

    def start_component(self, component: str) -> None:
        """Mark a component as started."""
        logger.debug("component_started", component=component)

        self._component_status[component] = ComponentStatusInfo(
            status=ComponentStatus.SUCCESS,  # Optimistic start
            execution_time=0.0,
        )

    def record_success(
        self,
        component: str,
        execution_time: float,
        api_calls: int = 0,
        tokens: int = 0,
    ) -> None:
        """Record successful component execution."""
        logger.info(
            "component_success",
            component=component,
            execution_time=execution_time,
            api_calls=api_calls,
            tokens=tokens,
        )

        self._component_status[component] = ComponentStatusInfo(
            status=ComponentStatus.SUCCESS,
            execution_time=execution_time,
            api_calls_made=api_calls,
            tokens_consumed=tokens,
        )

        self._component_metrics[component] = ComponentMetrics(
            component_name=component,
            execution_time=execution_time,
            api_calls_made=api_calls,
            tokens_consumed=tokens,
            success=True,
        )

    def record_partial(
        self,
        component: str,
        execution_time: float,
        warnings: list[str],
        api_calls: int = 0,
        tokens: int = 0,
    ) -> None:
        """Record partial success with warnings."""
        logger.warning(
            "component_partial",
            component=component,
            execution_time=execution_time,
            warnings=warnings,
        )

        self._component_status[component] = ComponentStatusInfo(
            status=ComponentStatus.PARTIAL,
            execution_time=execution_time,
            api_calls_made=api_calls,
            tokens_consumed=tokens,
            warnings=warnings,
        )

        self._component_metrics[component] = ComponentMetrics(
            component_name=component,
            execution_time=execution_time,
            api_calls_made=api_calls,
            tokens_consumed=tokens,
            success=True,  # Partial is still considered success
        )

    def record_fallback(
        self,
        component: str,
        execution_time: float,
        fallback_method: str,
        error_message: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """Record fallback method usage."""
        logger.warning(
            "component_fallback",
            component=component,
            fallback_method=fallback_method,
            error=error_message,
        )

        self._component_status[component] = ComponentStatusInfo(
            status=ComponentStatus.FALLBACK,
            execution_time=execution_time,
            fallback_method=fallback_method,
            error_message=error_message,
            warnings=warnings or [],
        )

        self._component_metrics[component] = ComponentMetrics(
            component_name=component,
            execution_time=execution_time,
            api_calls_made=0,
            tokens_consumed=0,
            success=True,  # Fallback is still functional
            fallback_used=True,
        )

    def record_failure(
        self,
        component: str,
        execution_time: float,
        error_message: str,
        error_type: str | None = None,
        stack_trace: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record component failure."""
        logger.error(
            "component_failure",
            component=component,
            error=error_message,
            error_type=error_type,
        )

        self._component_status[component] = ComponentStatusInfo(
            status=ComponentStatus.FAILED,
            execution_time=execution_time,
            error_message=error_message,
        )

        self._component_metrics[component] = ComponentMetrics(
            component_name=component,
            execution_time=execution_time,
            api_calls_made=0,
            tokens_consumed=0,
            success=False,
            error_message=error_message,
        )

        # Create detailed error report
        error_report = DetailedErrorReport(
            component=component,
            error_type=error_type or "UnknownError",
            error_message=error_message,
            timestamp=datetime.now(),
            stack_trace=stack_trace,
            context=context or {},
        )
        self._error_reports.append(error_report)

    def get_component_status(self, component: str) -> ComponentStatusInfo | None:
        """Get status for a specific component."""
        return self._component_status.get(component)

    def get_all_status(self) -> dict[str, ComponentStatusInfo]:
        """Get status for all components."""
        return dict(self._component_status)

    def get_component_metrics(self, component: str) -> ComponentMetrics | None:
        """Get metrics for a specific component."""
        return self._component_metrics.get(component)

    def get_all_metrics(self) -> list[ComponentMetrics]:
        """Get metrics for all components."""
        return list(self._component_metrics.values())

    def get_error_reports(self) -> list[DetailedErrorReport]:
        """Get all error reports."""
        return list(self._error_reports)

    def get_health_summary(self) -> dict[str, Any]:
        """Get overall health summary."""
        total = len(self._component_status)
        if total == 0:
            return {
                "overall_health": "unknown",
                "total_components": 0,
                "successful": 0,
                "partial": 0,
                "fallback": 0,
                "failed": 0,
            }

        status_counts = {
            "successful": 0,
            "partial": 0,
            "fallback": 0,
            "failed": 0,
        }

        for status_info in self._component_status.values():
            if status_info.status == ComponentStatus.SUCCESS:
                status_counts["successful"] += 1
            elif status_info.status == ComponentStatus.PARTIAL:
                status_counts["partial"] += 1
            elif status_info.status == ComponentStatus.FALLBACK:
                status_counts["fallback"] += 1
            elif status_info.status == ComponentStatus.FAILED:
                status_counts["failed"] += 1

        # Determine overall health
        if status_counts["failed"] > 0:
            overall_health = "degraded"
        elif status_counts["fallback"] > 0:
            overall_health = "fallback"
        elif status_counts["partial"] > 0:
            overall_health = "partial"
        else:
            overall_health = "healthy"

        return {
            "overall_health": overall_health,
            "total_components": total,
            **status_counts,
        }

    def clear(self) -> None:
        """Clear all tracking data."""
        self._component_status.clear()
        self._error_reports.clear()
        self._component_metrics.clear()


class StatusFormatter:
    """Formats status information for display."""

    @staticmethod
    def format_component_status(status_info: ComponentStatusInfo) -> str:
        """Format component status as human-readable string."""
        status_emoji = {
            ComponentStatus.SUCCESS: "✅",
            ComponentStatus.PARTIAL: "⚠️",
            ComponentStatus.FALLBACK: "🔄",
            ComponentStatus.FAILED: "❌",
        }

        emoji = status_emoji.get(status_info.status, "❓")
        status_text = status_info.status.value.upper()

        parts = [f"{emoji} {status_text}"]

        if status_info.execution_time > 0:
            parts.append(f"({status_info.execution_time:.2f}s)")

        if status_info.fallback_method:
            parts.append(f"[Fallback: {status_info.fallback_method}]")

        if status_info.error_message:
            parts.append(f"Error: {status_info.error_message}")

        if status_info.warnings:
            parts.append(f"Warnings: {len(status_info.warnings)}")

        return " ".join(parts)

    @staticmethod
    def format_health_summary(health: dict[str, Any]) -> str:
        """Format health summary as human-readable string."""
        health_emoji = {
            "healthy": "✅",
            "partial": "⚠️",
            "fallback": "🔄",
            "degraded": "❌",
            "unknown": "❓",
        }

        emoji = health_emoji.get(health["overall_health"], "❓")

        return (
            f"{emoji} Overall Health: {health['overall_health'].upper()}\n"
            f"  Total Components: {health['total_components']}\n"
            f"  ✅ Successful: {health['successful']}\n"
            f"  ⚠️  Partial: {health['partial']}\n"
            f"  🔄 Fallback: {health['fallback']}\n"
            f"  ❌ Failed: {health['failed']}"
        )

    @staticmethod
    def format_error_report(error: DetailedErrorReport) -> str:
        """Format error report as human-readable string."""
        lines = [
            f"❌ Error in {error.component}",
            f"  Type: {error.error_type}",
            f"  Message: {error.error_message}",
            f"  Time: {error.timestamp.isoformat()}",
        ]

        if error.recovery_attempted:
            recovery_status = "✅ Success" if error.recovery_successful else "❌ Failed"
            lines.append(f"  Recovery: {recovery_status}")

        if error.context:
            lines.append(f"  Context: {error.context}")

        return "\n".join(lines)


# =============================================================================
# Global singleton
# =============================================================================

_status_tracker: ComponentStatusTracker | None = None


def get_status_tracker() -> ComponentStatusTracker:
    """Get the global status tracker singleton."""
    global _status_tracker
    if _status_tracker is None:
        _status_tracker = ComponentStatusTracker()
    return _status_tracker


def reset_status_tracker() -> None:
    """Reset the global status tracker."""
    global _status_tracker
    if _status_tracker is not None:
        _status_tracker.clear()
    _status_tracker = None
