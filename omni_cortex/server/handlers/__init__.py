"""
MCP Tool Handlers for Omni-Cortex

Modular handlers for different tool categories.
"""

"""
This package intentionally uses **lazy imports**.

Why:
- Importing the full handler suite pulls in optional/large dependencies (e.g. MCP types).
- Many callers only need lightweight utilities (like validation helpers).

We expose a stable import surface (e.g. `from server.handlers import handle_reason`)
without importing every submodule at package import time.
"""

from importlib import import_module
from typing import Any

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
    "handle_prepare_context_streaming",
    "handle_context_cache_status",
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

_EXPORTS: dict[str, tuple[str, str]] = {
    # Core
    "handle_reason": (".reason_handler", "handle_reason"),
    "handle_think_framework": (".framework_handlers", "handle_think_framework"),
    # RAG
    "handle_search_documentation": (".rag_handlers", "handle_search_documentation"),
    "handle_search_frameworks_by_name": (".rag_handlers", "handle_search_frameworks_by_name"),
    "handle_search_by_category": (".rag_handlers", "handle_search_by_category"),
    "handle_search_function": (".rag_handlers", "handle_search_function"),
    "handle_search_class": (".rag_handlers", "handle_search_class"),
    "handle_search_docs_only": (".rag_handlers", "handle_search_docs_only"),
    "handle_search_framework_category": (".rag_handlers", "handle_search_framework_category"),
    # Utilities
    "handle_list_frameworks": (".utility_handlers", "handle_list_frameworks"),
    "handle_recommend": (".utility_handlers", "handle_recommend"),
    "handle_get_context": (".utility_handlers", "handle_get_context"),
    "handle_save_context": (".utility_handlers", "handle_save_context"),
    "handle_execute_code": (".utility_handlers", "handle_execute_code"),
    "handle_health": (".utility_handlers", "handle_health"),
    "handle_prepare_context": (".utility_handlers", "handle_prepare_context"),
    "handle_count_tokens": (".utility_handlers", "handle_count_tokens"),
    "handle_compress_content": (".utility_handlers", "handle_compress_content"),
    "handle_detect_truncation": (".utility_handlers", "handle_detect_truncation"),
    "handle_manage_claude_md": (".utility_handlers", "handle_manage_claude_md"),
    "handle_prepare_context_streaming": (".utility_handlers", "handle_prepare_context_streaming"),
    "handle_context_cache_status": (".utility_handlers", "handle_context_cache_status"),
    # Validation
    "ValidationError": (".validation", "ValidationError"),
    "validate_thread_id": (".validation", "validate_thread_id"),
    "validate_query": (".validation", "validate_query"),
    "validate_context": (".validation", "validate_context"),
    "validate_code": (".validation", "validate_code"),
    "validate_text": (".validation", "validate_text"),
    "validate_positive_int": (".validation", "validate_positive_int"),
    "validate_framework_name": (".validation", "validate_framework_name"),
    "validate_category": (".validation", "validate_category"),
    "validate_path": (".validation", "validate_path"),
    "validate_action": (".validation", "validate_action"),
    "validate_file_list": (".validation", "validate_file_list"),
    "validate_string_list": (".validation", "validate_string_list"),
    "validate_boolean": (".validation", "validate_boolean"),
    "validate_float": (".validation", "validate_float"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache for future lookups
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
