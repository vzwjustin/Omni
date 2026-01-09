"""
Metrics Manager for Dashboard Integration

Thread-safe metrics collection and aggregation for the experimental dashboard.
Tracks reasoning sessions, framework usage, token consumption, and system health.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from .settings import get_settings


@dataclass
class ReasoningSession:
    """Represents a single reasoning session."""

    session_id: str
    framework: str
    query: str
    started_at: float
    completed_at: Optional[float] = None
    status: str = "running"  # running, completed, failed
    tokens_used: int = 0
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class FrameworkMetrics:
    """Metrics for a specific framework."""

    name: str
    total_invocations: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    avg_time_ms: float = 0.0
    last_used: Optional[float] = None


@dataclass
class SystemMetrics:
    """Overall system metrics."""

    uptime_seconds: float = 0.0
    total_sessions: int = 0
    active_sessions: int = 0
    total_tokens: int = 0
    total_frameworks_used: int = 0
    avg_session_time_ms: float = 0.0
    success_rate: float = 0.0


class MetricsManager:
    """
    Thread-safe metrics manager for tracking Omni Cortex operations.

    Designed to have zero performance impact when dashboard is disabled.
    """

    def __init__(self, max_sessions: int = 1000):
        """
        Initialize metrics manager.

        Args:
            max_sessions: Maximum number of sessions to retain in memory
        """
        self.settings = get_settings()
        self._lock = threading.RLock()
        self._enabled = self.settings.enable_dashboard

        # Session tracking
        self._active_sessions: Dict[str, ReasoningSession] = {}
        self._completed_sessions: Deque[ReasoningSession] = deque(
            maxlen=max_sessions
        )

        # Framework metrics
        self._framework_metrics: Dict[str, FrameworkMetrics] = defaultdict(
            lambda: FrameworkMetrics(name="")
        )

        # System metrics
        self._start_time = time.time()
        self._total_sessions = 0
        self._total_tokens = 0

    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled

    def start_session(
        self,
        session_id: str,
        framework: str,
        query: str
    ) -> None:
        """
        Record the start of a reasoning session.

        Args:
            session_id: Unique session identifier
            framework: Framework being used
            query: User query/task
        """
        if not self._enabled:
            return

        with self._lock:
            session = ReasoningSession(
                session_id=session_id,
                framework=framework,
                query=query,
                started_at=time.time()
            )
            self._active_sessions[session_id] = session
            self._total_sessions += 1

    def complete_session(
        self,
        session_id: str,
        status: str = "completed",
        tokens_used: int = 0,
        result: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Record the completion of a reasoning session.

        Args:
            session_id: Session identifier
            status: Final status (completed, failed)
            tokens_used: Number of tokens consumed
            result: Result summary
            error: Error message if failed
        """
        if not self._enabled:
            return

        with self._lock:
            session = self._active_sessions.pop(session_id, None)
            if not session:
                return

            session.completed_at = time.time()
            session.status = status
            session.tokens_used = tokens_used
            session.result = result
            session.error = error

            # Update framework metrics
            framework = session.framework
            metrics = self._framework_metrics[framework]
            if not metrics.name:
                metrics.name = framework

            metrics.total_invocations += 1
            metrics.total_tokens += tokens_used

            if session.completed_at and session.started_at:
                duration_ms = (session.completed_at - session.started_at) * 1000
                metrics.total_time_ms += duration_ms
                metrics.avg_time_ms = (
                    metrics.total_time_ms / metrics.total_invocations
                )

            if status == "completed":
                metrics.success_count += 1
            else:
                metrics.failure_count += 1

            metrics.last_used = session.completed_at

            # Update system totals
            self._total_tokens += tokens_used

            # Archive session
            self._completed_sessions.append(session)

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all currently active sessions."""
        if not self._enabled:
            return []

        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "framework": s.framework,
                    "query": s.query[:100] + "..." if len(s.query) > 100 else s.query,
                    "started_at": datetime.fromtimestamp(s.started_at).isoformat(),
                    "duration_seconds": time.time() - s.started_at,
                    "status": s.status
                }
                for s in self._active_sessions.values()
            ]

    def get_recent_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently completed sessions."""
        if not self._enabled:
            return []

        with self._lock:
            sessions = list(self._completed_sessions)[-limit:]
            return [
                {
                    "session_id": s.session_id,
                    "framework": s.framework,
                    "query": s.query[:100] + "..." if len(s.query) > 100 else s.query,
                    "started_at": datetime.fromtimestamp(s.started_at).isoformat(),
                    "completed_at": datetime.fromtimestamp(s.completed_at).isoformat() if s.completed_at else None,
                    "duration_ms": (s.completed_at - s.started_at) * 1000 if s.completed_at else None,
                    "status": s.status,
                    "tokens_used": s.tokens_used,
                    "error": s.error
                }
                for s in reversed(sessions)
            ]

    def get_framework_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics for all frameworks."""
        if not self._enabled:
            return []

        with self._lock:
            return [
                {
                    "name": metrics.name,
                    "total_invocations": metrics.total_invocations,
                    "total_tokens": metrics.total_tokens,
                    "avg_time_ms": round(metrics.avg_time_ms, 2),
                    "success_count": metrics.success_count,
                    "failure_count": metrics.failure_count,
                    "success_rate": round(
                        metrics.success_count / max(metrics.total_invocations, 1) * 100,
                        2
                    ),
                    "last_used": datetime.fromtimestamp(metrics.last_used).isoformat() if metrics.last_used else None
                }
                for metrics in self._framework_metrics.values()
                if metrics.name
            ]

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics."""
        if not self._enabled:
            return {}

        with self._lock:
            uptime = time.time() - self._start_time
            total_completed = len(self._completed_sessions)

            # Calculate success rate
            success_count = sum(
                1 for s in self._completed_sessions
                if s.status == "completed"
            )
            success_rate = (
                success_count / max(total_completed, 1) * 100
            )

            # Calculate average session time
            total_time = sum(
                (s.completed_at - s.started_at) if s.completed_at else 0
                for s in self._completed_sessions
            )
            avg_time_ms = (
                total_time / max(total_completed, 1) * 1000
            )

            return {
                "uptime_seconds": round(uptime, 2),
                "uptime_formatted": self._format_duration(uptime),
                "total_sessions": self._total_sessions,
                "active_sessions": len(self._active_sessions),
                "completed_sessions": total_completed,
                "total_tokens": self._total_tokens,
                "total_frameworks_used": len(self._framework_metrics),
                "avg_session_time_ms": round(avg_time_ms, 2),
                "success_rate": round(success_rate, 2),
                "tokens_per_session": round(
                    self._total_tokens / max(total_completed, 1),
                    2
                )
            }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data snapshot."""
        if not self._enabled:
            return {
                "enabled": False,
                "message": "Dashboard is disabled. Set ENABLE_DASHBOARD=true to enable."
            }

        return {
            "enabled": True,
            "timestamp": datetime.now().isoformat(),
            "system": self.get_system_metrics(),
            "active_sessions": self.get_active_sessions(),
            "recent_sessions": self.get_recent_sessions(limit=10),
            "frameworks": self.get_framework_metrics(),
            "settings": {
                "lean_mode": self.settings.lean_mode,
                "enable_auto_ingest": self.settings.enable_auto_ingest,
                "llm_provider": self.settings.llm_provider,
                "routing_model": self.settings.routing_model
            }
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        elif seconds < 86400:
            return f"{seconds / 3600:.1f}h"
        else:
            return f"{seconds / 86400:.1f}d"

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._active_sessions.clear()
            self._completed_sessions.clear()
            self._framework_metrics.clear()
            self._start_time = time.time()
            self._total_sessions = 0
            self._total_tokens = 0


# Global singleton
_metrics_manager: Optional[MetricsManager] = None
_metrics_lock = threading.Lock()


def get_metrics_manager() -> MetricsManager:
    """Get the global metrics manager singleton (thread-safe)."""
    global _metrics_manager

    if _metrics_manager is not None:
        return _metrics_manager

    with _metrics_lock:
        if _metrics_manager is None:
            _metrics_manager = MetricsManager(
                max_sessions=get_settings().dashboard_metrics_retention
            )
    return _metrics_manager


def reset_metrics_manager() -> None:
    """Reset metrics manager singleton (useful for testing)."""
    global _metrics_manager
    with _metrics_lock:
        _metrics_manager = None
