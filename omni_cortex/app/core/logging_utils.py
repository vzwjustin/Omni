"""
Logging Utilities with Secret Sanitization

Prevents accidental exposure of API keys, tokens, and other sensitive
data in logs and error messages.
"""

import re
from typing import Any, Dict, Union


# Patterns for detecting sensitive data
API_KEY_PATTERNS = [
    r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})',
    r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})',
    r'secret["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})',
    r'password["\']?\s*[:=]\s*["\']?([^\s"\']+)',
    r'authorization["\']?\s*:\s*["\']?(Bearer\s+[a-zA-Z0-9_-]+)',
]

# Environment variable patterns
ENV_VAR_PATTERNS = [
    r'(OPENAI_API_KEY)',
    r'(ANTHROPIC_API_KEY)',
    r'(GOOGLE_API_KEY)',
    r'(OPENROUTER_API_KEY)',
]


def sanitize_api_keys(text: str) -> str:
    """
    Remove or redact API keys from text.

    Args:
        text: Text that may contain API keys

    Returns:
        Text with API keys redacted
    """
    sanitized = text

    # Replace API keys with redacted versions
    for pattern in API_KEY_PATTERNS:
        sanitized = re.sub(
            pattern,
            lambda m: m.group(0).replace(m.group(1), '[REDACTED]'),
            sanitized,
            flags=re.IGNORECASE
        )

    return sanitized


def sanitize_env_vars(text: str) -> str:
    """
    Redact environment variable values from text.

    Args:
        text: Text that may contain env var values

    Returns:
        Text with env var values redacted
    """
    sanitized = text

    for pattern in ENV_VAR_PATTERNS:
        # Match the env var name followed by its value
        sanitized = re.sub(
            rf'{pattern}\s*=\s*["\']?([^"\'\s]+)',
            lambda m: f'{m.group(1)}="[REDACTED]"',
            sanitized
        )

    return sanitized


def sanitize_error(error: Exception) -> str:
    """
    Sanitize exception message to remove sensitive data.

    Args:
        error: Exception to sanitize

    Returns:
        Sanitized error message safe for logging
    """
    error_msg = str(error)

    # Sanitize API keys
    error_msg = sanitize_api_keys(error_msg)

    # Sanitize env vars
    error_msg = sanitize_env_vars(error_msg)

    # Truncate very long error messages
    if len(error_msg) > 1000:
        error_msg = error_msg[:1000] + "... [truncated]"

    return error_msg


def sanitize_dict(data: Dict[str, Any], redact_keys: list = None) -> Dict[str, Any]:
    """
    Sanitize dictionary by redacting sensitive keys.

    Args:
        data: Dictionary to sanitize
        redact_keys: Additional keys to redact (beyond defaults)

    Returns:
        Sanitized dictionary
    """
    # Default sensitive keys
    sensitive_keys = {
        'api_key', 'apikey', 'api-key',
        'token', 'auth_token', 'access_token',
        'secret', 'password', 'pwd',
        'authorization', 'auth',
    }

    # Add custom keys
    if redact_keys:
        sensitive_keys.update(k.lower() for k in redact_keys)

    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()

        # Check if key is sensitive
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            # Redact but show first few chars
            if isinstance(value, str) and len(value) > 4:
                sanitized[key] = value[:4] + '...[REDACTED]'
            else:
                sanitized[key] = '[REDACTED]'
        # Recursively sanitize nested dicts
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, redact_keys)
        # Sanitize lists of dicts
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_dict(item, redact_keys) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized


def safe_repr(obj: Any, max_length: int = 500) -> str:
    """
    Create a safe repr of an object for logging.

    Args:
        obj: Object to represent
        max_length: Maximum length of repr

    Returns:
        Safe string representation
    """
    try:
        # Get repr
        repr_str = repr(obj)

        # Sanitize API keys
        repr_str = sanitize_api_keys(repr_str)

        # Truncate if too long
        if len(repr_str) > max_length:
            repr_str = repr_str[:max_length] + "... [truncated]"

        return repr_str

    except Exception as e:
        # Fallback if repr fails
        return f"<{type(obj).__name__} [repr failed: {str(e)[:100]}]>"


def sanitize_log_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize a structured log record.

    Args:
        record: Log record dictionary

    Returns:
        Sanitized log record
    """
    sanitized = record.copy()

    # Sanitize message
    if 'message' in sanitized:
        sanitized['message'] = sanitize_api_keys(str(sanitized['message']))

    # Sanitize any dict fields
    for key, value in list(sanitized.items()):
        if isinstance(value, dict):
            sanitized[key] = sanitize_dict(value)
        elif isinstance(value, str):
            sanitized[key] = sanitize_api_keys(value)

    return sanitized


def create_safe_error_details(error: Exception) -> Dict[str, Any]:
    """
    Create safe error details dict for logging.

    Args:
        error: Exception

    Returns:
        Dictionary with safe error details
    """
    details = {
        'type': type(error).__name__,
        'message': sanitize_error(error),
    }

    # Add details from OmniCortexError if available
    if hasattr(error, 'details'):
        details['details'] = sanitize_dict(error.details)

    return details
