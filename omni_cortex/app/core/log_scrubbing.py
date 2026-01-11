"""
Log scrubbing utilities for PII and sensitive data redaction.

Provides structlog processors to sanitize logs before output.
"""

import re
from typing import Any

# Patterns for sensitive data detection
SENSITIVE_PATTERNS = {
    # API Keys
    "api_key": re.compile(r'(?i)(api[_-]?key|apikey)["\s:=]+["\']?([a-zA-Z0-9_-]{20,})["\']?'),
    "bearer_token": re.compile(r"(?i)bearer\s+([a-zA-Z0-9_-]{20,})"),
    "auth_header": re.compile(r'(?i)authorization["\s:=]+["\']?([a-zA-Z0-9_-]{20,})["\']?'),
    # Common API key prefixes
    "openai_key": re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    "anthropic_key": re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}"),
    "google_key": re.compile(r"AIza[a-zA-Z0-9_-]{35}"),
    # Credentials
    "password": re.compile(r'(?i)(password|passwd|pwd)["\s:=]+["\']?([^\s"\']{4,})["\']?'),
    "secret": re.compile(r'(?i)(secret|token)["\s:=]+["\']?([a-zA-Z0-9_-]{8,})["\']?'),
    # PII
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    # IP addresses (internal)
    "ip_address": re.compile(
        r"\b(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|192\.168\.)\d{1,3}\.\d{1,3}\b"
    ),
}

# Fields to always redact
REDACT_FIELDS = {
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "credentials",
    "private_key",
    "access_token",
    "refresh_token",
    "session_id",
    "cookie",
    "x-api-key",
}

REDACTED = "[REDACTED]"


def scrub_value(value: Any) -> Any:
    """Scrub sensitive data from a single value."""
    if not isinstance(value, str):
        return value

    result = value
    for pattern_name, pattern in SENSITIVE_PATTERNS.items():
        result = pattern.sub(f"[{pattern_name.upper()}_REDACTED]", result)

    return result


def scrub_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively scrub sensitive data from a dictionary."""
    result = {}
    for key, value in data.items():
        key_lower = key.lower()

        # Check if key itself indicates sensitive data
        if any(sensitive in key_lower for sensitive in REDACT_FIELDS):
            result[key] = REDACTED
        elif isinstance(value, dict):
            result[key] = scrub_dict(value)
        elif isinstance(value, list):
            result[key] = [scrub_dict(v) if isinstance(v, dict) else scrub_value(v) for v in value]
        else:
            result[key] = scrub_value(value)

    return result


def log_scrubber(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Structlog processor that scrubs sensitive data from log events.

    Usage:
        structlog.configure(
            processors=[
                ...,
                log_scrubber,
                ...,
            ]
        )
    """
    return scrub_dict(event_dict)


# Export for easy import
__all__ = ["log_scrubber", "scrub_value", "scrub_dict", "REDACTED"]
