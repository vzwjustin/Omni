"""
Audit Logging for Omni-Cortex

Provides security audit trail for all framework executions and tool calls.
"""

from datetime import datetime
from typing import Any

import structlog

from .correlation import get_correlation_id

audit_logger = structlog.get_logger("audit")


def log_framework_execution(
    framework: str,
    query: str,
    thread_id: str | None = None,
    tokens_used: int = 0,
    confidence: float = 0.0,
    duration_ms: float = 0.0,
    success: bool = True,
    error: str | None = None,
) -> None:
    """Log a framework execution for audit trail."""
    audit_logger.info(
        "framework_execution",
        correlation_id=get_correlation_id(),
        timestamp=datetime.utcnow().isoformat(),
        framework=framework,
        query_preview=query[:100] if query else "",
        thread_id=thread_id,
        tokens_used=tokens_used,
        confidence=confidence,
        duration_ms=duration_ms,
        success=success,
        error=error,
    )


def log_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    thread_id: str | None = None,
    success: bool = True,
    error: str | None = None,
) -> None:
    """Log a tool call for audit trail."""
    # Redact sensitive arguments
    safe_args = {
        k: "***" if "key" in k.lower() or "secret" in k.lower() or "password" in k.lower() else v
        for k, v in arguments.items()
    }

    audit_logger.info(
        "tool_call",
        correlation_id=get_correlation_id(),
        timestamp=datetime.utcnow().isoformat(),
        tool=tool_name,
        arguments=safe_args,
        thread_id=thread_id,
        success=success,
        error=error,
    )
