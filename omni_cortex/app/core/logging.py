"""Logging utilities and structlog processors for omni-cortex."""
from typing import Any, Dict


def add_correlation_id(
    logger: Any,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Structlog processor to add correlation ID to all log events."""
    from app.core.correlation import get_correlation_id
    event_dict['correlation_id'] = get_correlation_id()
    return event_dict
