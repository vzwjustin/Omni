"""
Framework Registry - Single source of truth for all 62 thinking frameworks.

This module provides a centralized registry for framework metadata including:
- Framework specifications (name, category, description, best_for, vibes)
- Prompt templates
- Registry functions for lookup and filtering
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class FrameworkCategory(Enum):
    """Categories of thinking frameworks."""
    STRATEGY = "strategy"
    SEARCH = "search"
    ITERATIVE = "iterative"
    CODE = "code"
    CONTEXT = "context"
    FAST = "fast"
    VERIFICATION = "verification"
    AGENT = "agent"
    RAG = "rag"


@dataclass
class FrameworkSpec:
    """Specification for a thinking framework."""
    name: str
    category: FrameworkCategory
    description: str
    best_for: list[str]
    vibes: list[str] = field(default_factory=list)
    prompt_template: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "best_for": self.best_for,
            "vibes": self.vibes,
            "prompt": self.prompt_template,
        }


# Registry of all frameworks - populated by definitions.py
FRAMEWORK_REGISTRY: dict[str, FrameworkSpec] = {}


def register(spec: FrameworkSpec) -> None:
    """Register a framework in the global registry."""
    FRAMEWORK_REGISTRY[spec.name] = spec


def get_framework(name: str) -> Optional[FrameworkSpec]:
    """Get a framework specification by name."""
    return FRAMEWORK_REGISTRY.get(name)


def get_by_category(category: FrameworkCategory) -> list[FrameworkSpec]:
    """Get all frameworks in a specific category."""
    return [f for f in FRAMEWORK_REGISTRY.values() if f.category == category]


def list_all() -> list[str]:
    """List all registered framework names."""
    return list(FRAMEWORK_REGISTRY.keys())


def list_by_category() -> dict[str, list[str]]:
    """List framework names organized by category."""
    result: dict[str, list[str]] = {}
    for cat in FrameworkCategory:
        frameworks = get_by_category(cat)
        if frameworks:
            result[cat.value] = [f.name for f in frameworks]
    return result


def get_vibes_for_framework(name: str) -> list[str]:
    """Get the vibe patterns for a framework."""
    spec = get_framework(name)
    return spec.vibes if spec else []


def find_by_vibe(vibe: str) -> Optional[FrameworkSpec]:
    """Find a framework that matches a vibe pattern."""
    vibe_lower = vibe.lower()
    for spec in FRAMEWORK_REGISTRY.values():
        if any(v.lower() in vibe_lower for v in spec.vibes):
            return spec
    return None


def count() -> int:
    """Return total number of registered frameworks."""
    return len(FRAMEWORK_REGISTRY)
