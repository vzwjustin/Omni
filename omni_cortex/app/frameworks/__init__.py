"""
Frameworks Package - Centralized framework registry.

This package provides a single source of truth for all 62 thinking frameworks.
Import framework definitions to register them, then use registry functions
to access framework metadata.

Usage:
    from app.frameworks import (
        FRAMEWORK_REGISTRY,
        get_framework,
        get_by_category,
        list_all,
        list_by_category,
        FrameworkSpec,
        FrameworkCategory,
    )

    # Get a specific framework
    spec = get_framework("active_inference")

    # List all frameworks in a category
    search_frameworks = get_by_category(FrameworkCategory.SEARCH)

    # List all registered framework names
    all_names = list_all()
"""

# Import registry components
from .registry import (
    FRAMEWORK_REGISTRY,
    FrameworkSpec,
    FrameworkCategory,
    register,
    get_framework,
    get_by_category,
    list_all,
    list_by_category,
    get_vibes_for_framework,
    find_by_vibe,
    count,
)

# Import definitions to populate the registry
from . import definitions  # noqa: F401 - imported for side effects

__all__ = [
    # Registry
    "FRAMEWORK_REGISTRY",
    # Types
    "FrameworkSpec",
    "FrameworkCategory",
    # Functions
    "register",
    "get_framework",
    "get_by_category",
    "list_all",
    "list_by_category",
    "get_vibes_for_framework",
    "find_by_vibe",
    "count",
]
