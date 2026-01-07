"""
Frameworks Package - Single Source of Truth for all 62 thinking frameworks.

This package eliminates the 4-location sync requirement by providing a
centralized registry for all framework metadata.

Previous sync locations (now deprecated - import from here instead):
1. FRAMEWORK_NODES in app/graph.py
2. FRAMEWORKS dict in app/core/routing/framework_registry.py
3. VIBE_DICTIONARY in app/core/vibe_dictionary.py
4. get_framework_info() in app/core/routing/framework_registry.py

Usage:
    from app.frameworks import (
        # Core types
        FrameworkDefinition,
        FrameworkCategory,
        FrameworkNotFoundError,

        # Registry access
        FRAMEWORKS,
        get_framework,
        get_framework_safe,
        get_frameworks_by_category,
        get_framework_names,
        get_framework_info,
        get_frameworks_dict,

        # Vibe-based routing
        VIBE_DICTIONARY,
        match_vibes,
        get_all_vibes,
        find_by_vibe,

        # Utilities
        infer_task_type,
        list_by_category,
        count,
    )

    # Get a specific framework
    spec = get_framework("active_inference")

    # List all frameworks in a category
    search_frameworks = get_frameworks_by_category(FrameworkCategory.SEARCH)

    # Match vibes from natural language
    framework = match_vibes("wtf is wrong with this code")  # "active_inference"

    # Get total count (usage in code should use logger)
"""

from .registry import (
    # Core types
    FrameworkDefinition,
    FrameworkCategory,
    FrameworkNotFoundError,

    # Registry
    FRAMEWORKS,
    register,

    # Framework access
    get_framework,
    get_framework_safe,
    get_frameworks_by_category,
    get_framework_names,
    get_framework_info,
    get_frameworks_dict,

    # Vibe-based routing (backward compatibility)
    VIBE_DICTIONARY,
    match_vibes,
    get_all_vibes,
    find_by_vibe,

    # Utilities
    infer_task_type,
    list_by_category,
    count,
)

__all__ = [
    # Core types
    "FrameworkDefinition",
    "FrameworkCategory",
    "FrameworkNotFoundError",

    # Registry
    "FRAMEWORKS",
    "register",

    # Framework access
    "get_framework",
    "get_framework_safe",
    "get_frameworks_by_category",
    "get_framework_names",
    "get_framework_info",
    "get_frameworks_dict",

    # Vibe-based routing
    "VIBE_DICTIONARY",
    "match_vibes",
    "get_all_vibes",
    "find_by_vibe",

    # Utilities
    "infer_task_type",
    "list_by_category",
    "count",
]
