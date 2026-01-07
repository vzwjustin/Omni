"""Request correlation ID management for distributed tracing."""
import uuid
import contextvars
from typing import Optional

# Context variable for request-scoped correlation ID
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)


def get_correlation_id() -> str:
    """Get current correlation ID, generating one if needed."""
    cid = _correlation_id.get()
    if cid is None:
        cid = str(uuid.uuid4())[:8]  # Short ID for readability
        _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    _correlation_id.set(cid)


def clear_correlation_id() -> None:
    """Clear correlation ID."""
    _correlation_id.set(None)
