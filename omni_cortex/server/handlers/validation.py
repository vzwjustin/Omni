"""
Input Validation Utilities for MCP Tool Handlers

Provides validation helpers to sanitize and validate input parameters
before they are used in handlers.
"""

import re
from typing import Any, Optional, List

from app.core.errors import OmniCortexError


class ValidationError(OmniCortexError):
    """Input validation failed."""
    pass


def validate_thread_id(thread_id: Any, required: bool = False) -> Optional[str]:
    """
    Validate thread_id parameter.

    Args:
        thread_id: The thread_id value to validate
        required: If True, raises ValidationError when thread_id is missing

    Returns:
        Validated thread_id string, or None if not provided and not required

    Raises:
        ValidationError: If validation fails
    """
    if not thread_id:
        if required:
            raise ValidationError("thread_id is required")
        return None

    if not isinstance(thread_id, str):
        raise ValidationError("thread_id must be a string")

    if len(thread_id) > 256:
        raise ValidationError("thread_id too long (max 256 chars)")

    # Basic sanitization - alphanumeric, dash, underscore only
    if not re.match(r'^[a-zA-Z0-9_-]+$', thread_id):
        raise ValidationError("thread_id contains invalid characters (alphanumeric, dash, underscore only)")

    return thread_id


def validate_query(query: Any, max_length: int = 50000, required: bool = True) -> str:
    """
    Validate query parameter.

    Args:
        query: The query value to validate
        max_length: Maximum allowed length (default 50000)
        required: If True, raises ValidationError when query is missing

    Returns:
        Validated query string

    Raises:
        ValidationError: If validation fails
    """
    if not query:
        if required:
            raise ValidationError("query is required")
        return ""

    if not isinstance(query, str):
        raise ValidationError("query must be a string")

    if len(query) > max_length:
        raise ValidationError(f"query too long (max {max_length} chars)")

    return query


def validate_context(context: Any, max_length: int = 100000) -> str:
    """
    Validate context parameter.

    Args:
        context: The context value to validate
        max_length: Maximum allowed length (default 100000)

    Returns:
        Validated context string, or empty string if not provided

    Raises:
        ValidationError: If validation fails
    """
    if not context:
        return ""

    if not isinstance(context, str):
        raise ValidationError("context must be a string")

    if len(context) > max_length:
        raise ValidationError(f"context too long (max {max_length} chars)")

    return context


def validate_code(code: Any, max_length: int = 100000) -> str:
    """
    Validate code parameter for execution.

    Args:
        code: The code value to validate
        max_length: Maximum allowed length (default 100000)

    Returns:
        Validated code string

    Raises:
        ValidationError: If validation fails
    """
    if not code:
        raise ValidationError("code is required")

    if not isinstance(code, str):
        raise ValidationError("code must be a string")

    if len(code) > max_length:
        raise ValidationError(f"code too long (max {max_length} chars)")

    return code


def validate_text(text: Any, param_name: str = "text", max_length: int = 500000, required: bool = True) -> str:
    """
    Validate generic text parameter.

    Args:
        text: The text value to validate
        param_name: Name of parameter for error messages
        max_length: Maximum allowed length
        required: If True, raises ValidationError when text is missing

    Returns:
        Validated text string

    Raises:
        ValidationError: If validation fails
    """
    if not text:
        if required:
            raise ValidationError(f"{param_name} is required")
        return ""

    if not isinstance(text, str):
        raise ValidationError(f"{param_name} must be a string")

    if len(text) > max_length:
        raise ValidationError(f"{param_name} too long (max {max_length} chars)")

    return text


def validate_positive_int(value: Any, param_name: str, default: int, max_value: int = 1000) -> int:
    """
    Validate positive integer parameter.

    Args:
        value: The value to validate
        param_name: Name of parameter for error messages
        default: Default value if not provided
        max_value: Maximum allowed value

    Returns:
        Validated integer

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return default

    if isinstance(value, bool):
        raise ValidationError(f"{param_name} must be an integer")

    if not isinstance(value, int):
        raise ValidationError(f"{param_name} must be an integer")

    if value < 1:
        raise ValidationError(f"{param_name} must be positive")

    if value > max_value:
        raise ValidationError(f"{param_name} too large (max {max_value})")

    return value


def validate_framework_name(name: Any) -> str:
    """
    Validate framework name parameter.

    Args:
        name: The framework name to validate

    Returns:
        Validated framework name string

    Raises:
        ValidationError: If validation fails
    """
    if not name:
        raise ValidationError("framework_name is required")

    if not isinstance(name, str):
        raise ValidationError("framework_name must be a string")

    if len(name) > 100:
        raise ValidationError("framework_name too long (max 100 chars)")

    # Framework names should be alphanumeric with underscores
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
        raise ValidationError("framework_name contains invalid characters")

    return name


def validate_category(category: Any, valid_categories: List[str], param_name: str = "category") -> str:
    """
    Validate category parameter against allowed values.

    Args:
        category: The category value to validate
        valid_categories: List of valid category values
        param_name: Name of parameter for error messages

    Returns:
        Validated category string

    Raises:
        ValidationError: If validation fails
    """
    if not category:
        raise ValidationError(f"{param_name} is required")

    if not isinstance(category, str):
        raise ValidationError(f"{param_name} must be a string")

    if category not in valid_categories:
        raise ValidationError(f"Invalid {param_name}. Use: {', '.join(valid_categories)}")

    return category


def validate_path(path: Any, param_name: str = "path", required: bool = False) -> Optional[str]:
    """
    Validate file/directory path parameter.

    Args:
        path: The path value to validate
        param_name: Name of parameter for error messages
        required: If True, raises ValidationError when path is missing

    Returns:
        Validated path string, or None if not provided and not required

    Raises:
        ValidationError: If validation fails
    """
    if not path:
        if required:
            raise ValidationError(f"{param_name} is required")
        return None

    if not isinstance(path, str):
        raise ValidationError(f"{param_name} must be a string")

    if len(path) > 4096:
        raise ValidationError(f"{param_name} too long (max 4096 chars)")

    # Basic path traversal protection
    if ".." in path:
        raise ValidationError(f"{param_name} cannot contain path traversal sequences")

    # Prevent null bytes
    if "\x00" in path:
        raise ValidationError(f"{param_name} contains invalid characters")

    return path


def validate_action(action: Any, valid_actions: List[str]) -> str:
    """
    Validate action parameter against allowed values.

    Args:
        action: The action value to validate
        valid_actions: List of valid action values

    Returns:
        Validated action string

    Raises:
        ValidationError: If validation fails
    """
    if not action:
        raise ValidationError("action is required")

    if not isinstance(action, str):
        raise ValidationError("action must be a string")

    if action not in valid_actions:
        raise ValidationError(f"Invalid action. Use: {', '.join(valid_actions)}")

    return action


def validate_file_list(file_list: Any, max_files: int = 100) -> Optional[List[str]]:
    """
    Validate file list parameter.

    Args:
        file_list: The file list to validate
        max_files: Maximum number of files allowed

    Returns:
        Validated list of file paths, or None if not provided

    Raises:
        ValidationError: If validation fails
    """
    if not file_list:
        return None

    if not isinstance(file_list, list):
        raise ValidationError("file_list must be a list")

    if len(file_list) > max_files:
        raise ValidationError(f"file_list too long (max {max_files} files)")

    validated = []
    for i, path in enumerate(file_list):
        if not isinstance(path, str):
            raise ValidationError(f"file_list[{i}] must be a string")
        validated_path = validate_path(path, f"file_list[{i}]", required=True)
        validated.append(validated_path)

    return validated


def validate_string_list(items: Any, param_name: str, max_items: int = 100, max_item_length: int = 1000) -> Optional[List[str]]:
    """
    Validate list of strings parameter.

    Args:
        items: The list to validate
        param_name: Name of parameter for error messages
        max_items: Maximum number of items allowed
        max_item_length: Maximum length per item

    Returns:
        Validated list of strings, or None if not provided

    Raises:
        ValidationError: If validation fails
    """
    if not items:
        return None

    if not isinstance(items, list):
        raise ValidationError(f"{param_name} must be a list")

    if len(items) > max_items:
        raise ValidationError(f"{param_name} too long (max {max_items} items)")

    validated = []
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise ValidationError(f"{param_name}[{i}] must be a string")
        if len(item) > max_item_length:
            raise ValidationError(f"{param_name}[{i}] too long (max {max_item_length} chars)")
        validated.append(item)

    return validated


def validate_boolean(value: Any, param_name: str, default: bool) -> bool:
    """
    Validate boolean parameter.

    Args:
        value: The value to validate
        param_name: Name of parameter for error messages
        default: Default value if not provided

    Returns:
        Validated boolean

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return default

    if not isinstance(value, bool):
        raise ValidationError(f"{param_name} must be a boolean")

    return value


def validate_float(value: Any, param_name: str, default: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    Validate float parameter within range.

    Args:
        value: The value to validate
        param_name: Name of parameter for error messages
        default: Default value if not provided
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Validated float

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return default

    if not isinstance(value, (int, float)):
        raise ValidationError(f"{param_name} must be a number")

    value = float(value)

    if value < min_value or value > max_value:
        raise ValidationError(f"{param_name} must be between {min_value} and {max_value}")

    return value
