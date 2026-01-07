"""
MCP Tool Handlers for Omni-Cortex

Modular handlers for different tool categories.
"""

from .reason_handler import handle_reason
from .framework_handlers import handle_think_framework
from .rag_handlers import (
    handle_search_documentation,
    handle_search_frameworks_by_name,
    handle_search_by_category,
    handle_search_function,
    handle_search_class,
    handle_search_docs_only,
    handle_search_framework_category,
)
from .utility_handlers import (
    handle_list_frameworks,
    handle_recommend,
    handle_get_context,
    handle_save_context,
    handle_execute_code,
    handle_health,
    handle_prepare_context,
    handle_count_tokens,
    handle_compress_content,
    handle_detect_truncation,
    handle_manage_claude_md,
)
from .validation import (
    ValidationError,
    validate_thread_id,
    validate_query,
    validate_context,
    validate_code,
    validate_text,
    validate_positive_int,
    validate_framework_name,
    validate_category,
    validate_path,
    validate_action,
    validate_file_list,
    validate_string_list,
    validate_boolean,
    validate_float,
)

__all__ = [
    # Core
    "handle_reason",
    "handle_think_framework",
    # RAG
    "handle_search_documentation",
    "handle_search_frameworks_by_name",
    "handle_search_by_category",
    "handle_search_function",
    "handle_search_class",
    "handle_search_docs_only",
    "handle_search_framework_category",
    # Utilities
    "handle_list_frameworks",
    "handle_recommend",
    "handle_get_context",
    "handle_save_context",
    "handle_execute_code",
    "handle_health",
    "handle_prepare_context",
    "handle_count_tokens",
    "handle_compress_content",
    "handle_detect_truncation",
    "handle_manage_claude_md",
    # Validation
    "ValidationError",
    "validate_thread_id",
    "validate_query",
    "validate_context",
    "validate_code",
    "validate_text",
    "validate_positive_int",
    "validate_framework_name",
    "validate_category",
    "validate_path",
    "validate_action",
    "validate_file_list",
    "validate_string_list",
    "validate_boolean",
    "validate_float",
]
