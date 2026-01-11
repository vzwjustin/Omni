"""
Routing Module for Omni-Cortex

Hierarchical framework routing with Gemini-powered analysis.
"""

from .brief_generator import StructuredBriefGenerator
from .framework_registry import (
    CATEGORIES,
    CATEGORY_VIBES,
    FRAMEWORKS,
    HEURISTIC_MAP,
    PATTERNS,
    get_framework_info,
    infer_task_type,
)
from .task_analysis import (
    enrich_evidence_from_chroma,
    gemini_analyze_task,
    get_relevant_learnings,
    save_task_analysis,
)

__all__ = [
    # Registry
    "CATEGORIES",
    "CATEGORY_VIBES",
    "FRAMEWORKS",
    "PATTERNS",
    "HEURISTIC_MAP",
    "get_framework_info",
    "infer_task_type",
    # Task Analysis
    "gemini_analyze_task",
    "get_relevant_learnings",
    "enrich_evidence_from_chroma",
    "save_task_analysis",
    # Brief Generator
    "StructuredBriefGenerator",
]
