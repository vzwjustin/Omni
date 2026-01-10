"""
Input Validation and Sanitization

Protects against prompt injection, XSS, and other injection attacks.
All user input should be sanitized through these functions.
"""

import re
from typing import Optional
from app.core.errors import OmniCortexError, ValidationError


# Maximum input lengths (from constants)
MAX_QUERY_LENGTH = 10000
MAX_CODE_SNIPPET_LENGTH = 50000
MAX_CONTEXT_LENGTH = 100000


# Suspicious patterns that could indicate injection attacks
SUSPICIOUS_PATTERNS = [
    r'<\s*script',  # Script tags
    r'javascript:',  # JavaScript protocol
    r'on\w+\s*=',   # Event handlers (onclick=, onerror=, etc.)
    r'\\x[0-9a-f]{2}',  # Hex escapes
    r'\\u[0-9a-f]{4}',  # Unicode escapes
    r'eval\s*\(',  # eval() calls
    r'exec\s*\(',  # exec() calls
    r'__import__',  # Dynamic imports
    r'<iframe',  # iframe injection
    r'<embed',   # embed tags
    r'<object',  # object tags
]


def sanitize_user_input(
    text: str,
    max_length: int = MAX_QUERY_LENGTH,
    allow_code: bool = False
) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        text: Raw user input
        max_length: Maximum allowed length
        allow_code: If True, allows code-like patterns (for code_snippet field)

    Returns:
        Sanitized text

    Raises:
        ValidationError: If input is invalid or suspicious
    """
    if not isinstance(text, str):
        raise ValidationError(
            "Input must be a string",
            details={"type": type(text).__name__}
        )

    # Check length
    if len(text) > max_length:
        raise ValidationError(
            f"Input exceeds maximum length of {max_length} characters",
            details={"length": len(text), "max": max_length}
        )

    # Check for suspicious patterns (unless code is explicitly allowed)
    if not allow_code:
        for pattern in SUSPICIOUS_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raise ValidationError(
                    f"Input contains suspicious pattern: {match.group()}",
                    details={"pattern": pattern, "match": match.group()}
                )

    # Strip leading/trailing whitespace
    sanitized = text.strip()

    # Ensure not empty after sanitization
    if not sanitized:
        raise ValidationError("Input cannot be empty")

    return sanitized


def sanitize_query(query: str) -> str:
    """
    Sanitize user query input.

    Applies strict validation suitable for natural language queries.
    """
    return sanitize_user_input(
        query,
        max_length=MAX_QUERY_LENGTH,
        allow_code=False
    )


def sanitize_code_snippet(code: Optional[str]) -> Optional[str]:
    """
    Sanitize code snippet input.

    Allows code-like patterns but still enforces length limits.
    """
    if code is None:
        return None

    return sanitize_user_input(
        code,
        max_length=MAX_CODE_SNIPPET_LENGTH,
        allow_code=True  # Allow code patterns
    )


def sanitize_context(context: Optional[str]) -> Optional[str]:
    """
    Sanitize IDE context input.

    Allows larger input for full context but still validates.
    """
    if context is None:
        return None

    return sanitize_user_input(
        context,
        max_length=MAX_CONTEXT_LENGTH,
        allow_code=True  # Context may contain code
    )


def validate_thread_id(thread_id: Optional[str]) -> Optional[str]:
    """
    Validate thread ID format.

    Thread IDs should be alphanumeric with hyphens/underscores only.
    """
    if thread_id is None:
        return None

    if not isinstance(thread_id, str):
        raise ValidationError("thread_id must be a string")

    # Allow alphanumeric, hyphens, underscores only
    if not re.match(r'^[a-zA-Z0-9_-]+$', thread_id):
        raise ValidationError(
            "thread_id contains invalid characters",
            details={"thread_id": thread_id[:50]}
        )

    if len(thread_id) > 256:
        raise ValidationError(
            "thread_id too long (max 256 characters)",
            details={"length": len(thread_id)}
        )

    return thread_id


def validate_framework_name(framework: Optional[str]) -> Optional[str]:
    """
    Validate framework name.

    Framework names must:
    - Start with a lowercase letter
    - Contain only lowercase letters, numbers, and underscores
    - Be at most 100 characters long

    This is the authoritative validation used by both core and handlers.
    """
    if framework is None:
        return None

    if not isinstance(framework, str):
        raise ValidationError("framework must be a string")

    if len(framework) > 100:
        raise ValidationError("framework name too long (max 100 chars)")

    # Framework names must start with letter, then alphanumeric with underscores
    if not re.match(r'^[a-z][a-z0-9_]*$', framework):
        raise ValidationError(
            "framework name contains invalid characters",
            details={"framework": framework}
        )

    return framework


def sanitize_file_path(path: str) -> str:
    """
    Sanitize file path to prevent directory traversal.

    Args:
        path: File path from user

    Returns:
        Sanitized path

    Raises:
        ValidationError: If path contains traversal attempts
    """
    # Check for directory traversal
    if '..' in path:
        raise ValidationError(
            "Path contains directory traversal",
            details={"path": path[:100]}
        )

    # Check for absolute paths (should be relative)
    if path.startswith('/'):
        raise ValidationError(
            "Absolute paths not allowed",
            details={"path": path[:100]}
        )

    # Check for null bytes
    if '\x00' in path:
        raise ValidationError("Path contains null bytes")

    return path


def validate_boolean(value: any, field_name: str = "value") -> bool:
    """
    Validate and convert boolean input.

    Accepts: True/False, "true"/"false", 1/0
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lower = value.lower()
        if lower in ('true', '1', 'yes', 'on'):
            return True
        if lower in ('false', '0', 'no', 'off'):
            return False

    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)

    raise ValidationError(
        f"{field_name} must be a boolean",
        details={"value": str(value)[:100]}
    )


def validate_integer(
    value: any,
    field_name: str = "value",
    min_value: Optional[int] = None,
    max_value: Optional[int] = None
) -> int:
    """
    Validate and convert integer input with range checking.
    """
    try:
        int_value = int(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"{field_name} must be an integer",
            details={"value": str(value)[:100]}
        )

    if min_value is not None and int_value < min_value:
        raise ValidationError(
            f"{field_name} must be >= {min_value}",
            details={"value": int_value, "min": min_value}
        )

    if max_value is not None and int_value > max_value:
        raise ValidationError(
            f"{field_name} must be <= {max_value}",
            details={"value": int_value, "max": max_value}
        )

    return int_value


def validate_float(
    value: any,
    field_name: str = "value",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> float:
    """
    Validate and convert float input with range checking.
    """
    try:
        float_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"{field_name} must be a number",
            details={"value": str(value)[:100]}
        )

    if min_value is not None and float_value < min_value:
        raise ValidationError(
            f"{field_name} must be >= {min_value}",
            details={"value": float_value, "min": min_value}
        )

    if max_value is not None and float_value > max_value:
        raise ValidationError(
            f"{field_name} must be <= {max_value}",
            details={"value": float_value, "max": max_value}
        )

    return float_value
