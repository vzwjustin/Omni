"""
Vibe Matching Service

Handles heuristic and keyword-based routing ("vibes") to categorize tasks.
"""

import re
from re import Pattern

from ..vibe_dictionary import match_vibes
from . import (
    CATEGORY_VIBES,
    HEURISTIC_MAP,
    PATTERNS,
)


class VibeMatcher:
    """Service for vibe-based routing and categorization."""

    def __init__(self):
        self._compiled_patterns: dict[str, list[Pattern]] = {
            task_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for task_type, patterns in PATTERNS.items()
        }

    def route_to_category(self, query: str) -> tuple[str, float]:
        """
        Fast first-stage routing to a category based on vibes.

        Args:
            query: User query

        Returns:
            Tuple of (category_name, confidence_score)
        """
        query_lower = query.lower()
        scores = {}

        for category, vibes in CATEGORY_VIBES.items():
            score = 0.0
            for vibe in vibes:
                if vibe in query_lower:
                    # Weight by phrase length
                    word_count = len(vibe.split())
                    score += word_count if word_count >= 2 else 0.5
            if score > 0:
                scores[category] = score

        if not scores:
            return "exploration", 0.3  # Default fallback

        best = max(scores, key=scores.get)
        confidence = min(scores[best] / 5.0, 1.0)  # Normalize to 0-1
        return best, confidence

    def check_vibe_dictionary(self, query: str) -> str | None:
        """
        Quick check against vibe dictionary with weighted scoring.

        Args:
            query: User query

        Returns:
            Matched framework name or None
        """
        return match_vibes(query)

    def heuristic_select(self, query: str, code_snippet: str | None = None) -> str:
        """
        Fast heuristic selection (fallback) using regex patterns.

        Args:
            query: User query
            code_snippet: Optional code context

        Returns:
            Selected framework name
        """
        combined = query + (" " + code_snippet if code_snippet else "")

        scores = {}
        for task_type, patterns in self._compiled_patterns.items():
            scores[task_type] = sum(1 for p in patterns if p.search(combined))

        if max(scores.values()) > 0:
            task_type = max(scores, key=scores.get)
            return HEURISTIC_MAP.get(task_type, "self_discover")

        return "self_discover"
