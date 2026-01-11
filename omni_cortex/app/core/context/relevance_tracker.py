"""
Context Relevance Tracking

Tracks which context elements are most relevant to Claude's final solutions,
enabling feedback loops for context optimization.

This module provides:
- Relevance scoring for context elements (files, docs, code search results)
- Usage tracking to identify which elements Claude actually uses
- Feedback loop for improving context preparation over time
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger("relevance_tracker")


@dataclass
class ElementUsage:
    """Tracks usage of a single context element."""

    element_id: str  # Unique identifier (file path, doc URL, etc.)
    element_type: str  # "file", "documentation", "code_search"
    times_included: int = 0  # How many times included in context
    times_used: int = 0  # How many times actually referenced in solution
    usage_rate: float = 0.0  # times_used / times_included
    last_included: datetime | None = None
    last_used: datetime | None = None
    relevance_scores: list[float] = field(default_factory=list)  # Historical scores
    avg_relevance_score: float = 0.0


@dataclass
class ContextUsageSession:
    """Tracks context usage for a single query/solution session."""

    session_id: str
    query: str
    timestamp: datetime
    included_elements: set[str] = field(default_factory=set)  # Element IDs included
    used_elements: set[str] = field(default_factory=set)  # Element IDs actually used
    solution_text: str | None = None  # Claude's solution for analysis
    task_type: str | None = None
    complexity: str | None = None


class RelevanceTracker:
    """
    Tracks context element relevance and usage patterns.

    Provides feedback loop for optimizing context preparation by learning
    which elements are most valuable for different task types.
    """

    def __init__(self, max_history_days: int = 30):
        """
        Initialize relevance tracker.

        Args:
            max_history_days: How many days of history to keep
        """
        self._element_usage: dict[str, ElementUsage] = {}
        self._sessions: list[ContextUsageSession] = []
        self._max_history_days = max_history_days
        self._usage_by_task_type: dict[str, dict[str, ElementUsage]] = defaultdict(dict)

    def start_session(
        self, query: str, task_type: str | None = None, complexity: str | None = None
    ) -> str:
        """
        Start a relevance tracking session.

        Returns:
            Generated session identifier.
        """
        session_id = f"session-{uuid4().hex}"
        session = ContextUsageSession(
            session_id=session_id,
            query=query,
            timestamp=datetime.now(),
            task_type=task_type,
            complexity=complexity,
        )
        self._sessions.append(session)
        logger.info("relevance_session_started", session_id=session_id)
        return session_id

    def track_element_provided(
        self,
        session_id: str,
        element_type: str,
        element_identifier: str,
        relevance_score: float | None = None,
    ) -> None:
        """
        Track an element included in the prepared context.

        Args:
            session_id: Active session identifier
            element_type: "file", "doc", or "code_search"
            element_identifier: File path, doc URL, or search signature
            relevance_score: Optional relevance score (0-1)
        """
        session = None
        for s in self._sessions:
            if s.session_id == session_id:
                session = s
                break

        if not session:
            logger.warning("session_not_found", session_id=session_id)
            return

        normalized_type = element_type.lower()
        if normalized_type in ("doc", "documentation"):
            element_id = f"doc:{element_identifier}"
            usage_type = "documentation"
        elif normalized_type in ("code_search", "code"):
            element_id = f"code_search:{element_identifier}"
            usage_type = "code_search"
        else:
            element_id = f"file:{element_identifier}"
            usage_type = "file"

        session.included_elements.add(element_id)

        if element_id not in self._element_usage:
            self._element_usage[element_id] = ElementUsage(
                element_id=element_id, element_type=usage_type
            )

        usage = self._element_usage[element_id]
        usage.times_included += 1
        usage.last_included = datetime.now()

        if relevance_score is not None:
            usage.relevance_scores.append(relevance_score)
            usage.avg_relevance_score = sum(usage.relevance_scores) / len(usage.relevance_scores)

    def record_context_preparation(
        self,
        session_id: str,
        query: str,
        files: list[dict[str, Any]],
        documentation: list[dict[str, Any]],
        code_search: list[dict[str, Any]],
        task_type: str | None = None,
        complexity: str | None = None,
    ) -> None:
        """
        Record which elements were included in prepared context.

        Args:
            session_id: Unique session identifier
            query: User's query
            files: List of file contexts with path and relevance_score
            documentation: List of documentation contexts with source
            code_search: List of code search results
            task_type: Type of task (debug, implement, etc.)
            complexity: Task complexity level
        """
        session = ContextUsageSession(
            session_id=session_id,
            query=query,
            timestamp=datetime.now(),
            task_type=task_type,
            complexity=complexity,
        )

        # Track file inclusions
        for file_ctx in files:
            element_id = f"file:{file_ctx.get('path', '')}"
            session.included_elements.add(element_id)

            # Update or create element usage
            if element_id not in self._element_usage:
                self._element_usage[element_id] = ElementUsage(
                    element_id=element_id, element_type="file"
                )

            usage = self._element_usage[element_id]
            usage.times_included += 1
            usage.last_included = datetime.now()

            # Record relevance score
            if "relevance_score" in file_ctx:
                usage.relevance_scores.append(file_ctx["relevance_score"])
                usage.avg_relevance_score = sum(usage.relevance_scores) / len(
                    usage.relevance_scores
                )

        # Track documentation inclusions
        for doc_ctx in documentation:
            element_id = f"doc:{doc_ctx.get('source', '')}"
            session.included_elements.add(element_id)

            if element_id not in self._element_usage:
                self._element_usage[element_id] = ElementUsage(
                    element_id=element_id, element_type="documentation"
                )

            usage = self._element_usage[element_id]
            usage.times_included += 1
            usage.last_included = datetime.now()

            if "relevance_score" in doc_ctx:
                usage.relevance_scores.append(doc_ctx["relevance_score"])
                usage.avg_relevance_score = sum(usage.relevance_scores) / len(
                    usage.relevance_scores
                )

        # Track code search inclusions
        for code_ctx in code_search:
            element_id = (
                f"code_search:{code_ctx.get('search_type', '')}:{code_ctx.get('query', '')}"
            )
            session.included_elements.add(element_id)

            if element_id not in self._element_usage:
                self._element_usage[element_id] = ElementUsage(
                    element_id=element_id, element_type="code_search"
                )

            usage = self._element_usage[element_id]
            usage.times_included += 1
            usage.last_included = datetime.now()

        self._sessions.append(session)

        logger.info(
            "context_preparation_recorded",
            session_id=session_id,
            files=len(files),
            docs=len(documentation),
            code_searches=len(code_search),
            task_type=task_type,
        )

    def record_solution_usage(self, session_id: str, solution_text: str) -> dict[str, int]:
        """
        Analyze Claude's solution to identify which context elements were used.

        Uses pattern matching to detect references to files, documentation, and
        code patterns in the solution text.

        Args:
            session_id: Session identifier
            solution_text: Claude's solution text

        Returns:
            Dictionary mapping element IDs to usage counts
        """
        # Find the session
        session = None
        for s in self._sessions:
            if s.session_id == session_id:
                session = s
                break

        if not session:
            logger.warning("session_not_found", session_id=session_id)
            return {}

        session.solution_text = solution_text
        usage_counts = {}

        # Analyze solution text for element references
        for element_id in session.included_elements:
            usage_count = self._detect_element_usage(element_id, solution_text)

            if usage_count > 0:
                session.used_elements.add(element_id)
                usage_counts[element_id] = usage_count

                # Update element usage statistics
                if element_id in self._element_usage:
                    usage = self._element_usage[element_id]
                    usage.times_used += 1
                    usage.last_used = datetime.now()

                    # Update usage rate
                    if usage.times_included > 0:
                        usage.usage_rate = usage.times_used / usage.times_included

                    # Track by task type
                    if session.task_type:
                        if element_id not in self._usage_by_task_type[session.task_type]:
                            self._usage_by_task_type[session.task_type][element_id] = ElementUsage(
                                element_id=element_id, element_type=usage.element_type
                            )

                        task_usage = self._usage_by_task_type[session.task_type][element_id]
                        task_usage.times_used += 1
                        task_usage.times_included += 1
                        task_usage.usage_rate = task_usage.times_used / task_usage.times_included

        logger.info(
            "solution_usage_recorded",
            session_id=session_id,
            included_count=len(session.included_elements),
            used_count=len(session.used_elements),
            usage_rate=len(session.used_elements) / len(session.included_elements)
            if session.included_elements
            else 0,
        )

        return usage_counts

    def _detect_element_usage(self, element_id: str, solution_text: str) -> int:
        """
        Detect if and how many times an element is referenced in solution text.

        Args:
            element_id: Element identifier (e.g., "file:path/to/file.py")
            solution_text: Solution text to analyze

        Returns:
            Number of times element is referenced
        """
        element_type, element_value = element_id.split(":", 1)
        count = 0

        if element_type == "file":
            # Look for file path references
            # Match full path or just filename
            filename = element_value.split("/")[-1]

            # Count occurrences of full path
            count += solution_text.count(element_value)

            # Count occurrences of filename (if not already counted)
            if element_value not in solution_text:
                count += solution_text.count(filename)

            # Look for code block references with file path
            code_block_pattern = rf"```[^\n]*{re.escape(filename)}"
            count += len(re.findall(code_block_pattern, solution_text, re.IGNORECASE))

        elif element_type == "doc":
            # Look for documentation URL or title references
            count += solution_text.count(element_value)

            # Look for domain references
            if "://" in element_value:
                domain = element_value.split("://")[1].split("/")[0]
                count += solution_text.count(domain)

        elif element_type == "code_search":
            # Look for code pattern references
            _, search_query = element_value.split(":", 1)
            count += solution_text.count(search_query)

        return count

    def get_element_statistics(
        self,
        element_type: str | None = None,
        min_usage_rate: float = 0.0,
        task_type: str | None = None,
    ) -> list[ElementUsage]:
        """
        Get usage statistics for context elements.

        Args:
            element_type: Filter by element type (file, documentation, code_search)
            min_usage_rate: Minimum usage rate to include
            task_type: Filter by task type

        Returns:
            List of ElementUsage objects sorted by usage rate
        """
        if task_type and task_type in self._usage_by_task_type:
            elements = list(self._usage_by_task_type[task_type].values())
        else:
            elements = list(self._element_usage.values())

        # Filter by element type
        if element_type:
            elements = [e for e in elements if e.element_type == element_type]

        # Filter by minimum usage rate
        elements = [e for e in elements if e.usage_rate >= min_usage_rate]

        # Sort by usage rate (descending)
        elements.sort(key=lambda e: e.usage_rate, reverse=True)

        return elements

    def get_relevance_feedback(
        self, task_type: str | None = None, top_n: int = 10
    ) -> dict[str, Any]:
        """
        Get feedback for optimizing context preparation.

        Identifies high-value and low-value elements to improve future
        context preparation.

        Args:
            task_type: Filter by task type
            top_n: Number of top/bottom elements to return

        Returns:
            Dictionary with high_value and low_value elements
        """
        elements = self.get_element_statistics(task_type=task_type)

        # Separate by element type
        files = [e for e in elements if e.element_type == "file"]
        docs = [e for e in elements if e.element_type == "documentation"]
        [e for e in elements if e.element_type == "code_search"]

        feedback = {
            "high_value_files": [
                {
                    "path": e.element_id.replace("file:", ""),
                    "usage_rate": e.usage_rate,
                    "avg_relevance_score": e.avg_relevance_score,
                    "times_used": e.times_used,
                    "times_included": e.times_included,
                }
                for e in files[:top_n]
                if e.usage_rate > 0.5
            ],
            "low_value_files": [
                {
                    "path": e.element_id.replace("file:", ""),
                    "usage_rate": e.usage_rate,
                    "avg_relevance_score": e.avg_relevance_score,
                    "times_used": e.times_used,
                    "times_included": e.times_included,
                }
                for e in files[-top_n:]
                if e.usage_rate < 0.2 and e.times_included >= 3
            ],
            "high_value_docs": [
                {
                    "source": e.element_id.replace("doc:", ""),
                    "usage_rate": e.usage_rate,
                    "times_used": e.times_used,
                    "times_included": e.times_included,
                }
                for e in docs[:top_n]
                if e.usage_rate > 0.5
            ],
            "low_value_docs": [
                {
                    "source": e.element_id.replace("doc:", ""),
                    "usage_rate": e.usage_rate,
                    "times_used": e.times_used,
                    "times_included": e.times_included,
                }
                for e in docs[-top_n:]
                if e.usage_rate < 0.2 and e.times_included >= 3
            ],
            "overall_stats": {
                "total_elements": len(elements),
                "avg_usage_rate": sum(e.usage_rate for e in elements) / len(elements)
                if elements
                else 0,
                "high_value_count": len([e for e in elements if e.usage_rate > 0.5]),
                "low_value_count": len(
                    [e for e in elements if e.usage_rate < 0.2 and e.times_included >= 3]
                ),
            },
        }

        return feedback

    def optimize_relevance_scores(
        self, files: list[dict[str, Any]], task_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Adjust relevance scores based on historical usage patterns.

        Boosts scores for elements with high usage rates, reduces scores
        for elements with low usage rates.

        Args:
            files: List of file contexts with path and relevance_score
            task_type: Task type for context-specific optimization

        Returns:
            List of file contexts with optimized relevance scores
        """
        optimized_files = []

        for file_ctx in files:
            element_id = f"file:{file_ctx.get('path', '')}"

            # Get historical usage
            usage = None
            if task_type and task_type in self._usage_by_task_type:
                usage = self._usage_by_task_type[task_type].get(element_id)

            if not usage:
                usage = self._element_usage.get(element_id)

            # Apply optimization
            optimized_ctx = file_ctx.copy()
            original_score = file_ctx.get("relevance_score", 0.5)

            if usage and usage.times_included >= 3:
                # Adjust based on usage rate
                if usage.usage_rate > 0.7:
                    # High value - boost score
                    boost = 0.2 * (usage.usage_rate - 0.7) / 0.3
                    optimized_ctx["relevance_score"] = min(1.0, original_score + boost)
                    optimized_ctx["score_adjustment"] = "boosted"
                elif usage.usage_rate < 0.2:
                    # Low value - reduce score
                    penalty = 0.2 * (0.2 - usage.usage_rate) / 0.2
                    optimized_ctx["relevance_score"] = max(0.0, original_score - penalty)
                    optimized_ctx["score_adjustment"] = "reduced"
                else:
                    optimized_ctx["score_adjustment"] = "unchanged"
            else:
                optimized_ctx["score_adjustment"] = "no_history"

            optimized_files.append(optimized_ctx)

        # Re-sort by optimized scores
        optimized_files.sort(key=lambda f: f.get("relevance_score", 0), reverse=True)

        return optimized_files

    def cleanup_old_data(self) -> int:
        """
        Remove data older than max_history_days.

        Returns:
            Number of sessions removed
        """
        cutoff_date = datetime.now() - timedelta(days=self._max_history_days)

        # Remove old sessions
        old_count = len(self._sessions)
        self._sessions = [s for s in self._sessions if s.timestamp > cutoff_date]
        removed_count = old_count - len(self._sessions)

        # Remove element usage data for elements not seen recently
        elements_to_remove = []
        for element_id, usage in self._element_usage.items():
            if usage.last_included and usage.last_included < cutoff_date:
                elements_to_remove.append(element_id)

        for element_id in elements_to_remove:
            del self._element_usage[element_id]

        logger.info(
            "relevance_tracker_cleanup",
            sessions_removed=removed_count,
            elements_removed=len(elements_to_remove),
        )

        return removed_count

    def get_summary_statistics(self) -> dict[str, Any]:
        """
        Get summary statistics for monitoring.

        Returns:
            Dictionary with summary statistics
        """
        all_elements = list(self._element_usage.values())

        files = [e for e in all_elements if e.element_type == "file"]
        docs = [e for e in all_elements if e.element_type == "documentation"]
        code_searches = [e for e in all_elements if e.element_type == "code_search"]

        return {
            "total_sessions": len(self._sessions),
            "total_elements_tracked": len(all_elements),
            "files_tracked": len(files),
            "docs_tracked": len(docs),
            "code_searches_tracked": len(code_searches),
            "avg_file_usage_rate": sum(f.usage_rate for f in files) / len(files) if files else 0,
            "avg_doc_usage_rate": sum(d.usage_rate for d in docs) / len(docs) if docs else 0,
            "high_value_files": len([f for f in files if f.usage_rate > 0.7]),
            "low_value_files": len(
                [f for f in files if f.usage_rate < 0.2 and f.times_included >= 3]
            ),
            "task_types_tracked": list(self._usage_by_task_type.keys()),
        }


# =============================================================================
# Global singleton
# =============================================================================

_relevance_tracker: RelevanceTracker | None = None


def get_relevance_tracker() -> RelevanceTracker:
    """Get the global RelevanceTracker singleton."""
    global _relevance_tracker

    if _relevance_tracker is None:
        _relevance_tracker = RelevanceTracker()

    return _relevance_tracker
