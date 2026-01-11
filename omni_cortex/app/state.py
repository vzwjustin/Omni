"""
Graph State Management for LangGraph Workflows

Defines the state structure that flows through reasoning frameworks.
Refactored to use composition and strictly typed sub-states.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, TypedDict


class InputState(TypedDict, total=False):
    """Immutable input parameters."""

    query: str
    code_snippet: str | None
    file_list: list[str]
    ide_context: str | None
    preferred_framework: str | None
    max_iterations: int


class ReasoningState(TypedDict, total=False):
    """Internal working state for the reasoning engine."""

    selected_framework: str
    framework_chain: list[str]
    routing_category: str
    task_type: str
    complexity_estimate: float
    working_memory: dict[str, Any]
    reasoning_steps: list[dict[str, Any]]
    step_counter: int
    tokens_used: int
    quiet_thoughts: list[str]
    episodic_memory: list[dict[str, Any]]


class OutputState(TypedDict, total=False):
    """Final results."""

    final_code: str | None
    final_answer: str | None
    confidence_score: float
    error: str | None


class GraphState(TypedDict, total=False):
    """
    Central state object passed through LangGraph nodes.

    Refactored to flatten the structure for backward compatibility with existing nodes,
    while internally supporting the composed types if needed in future refactors.

    Currently flattened to maintain compatibility with existing
    node implementations like `state['query']` or `state['working_memory']`.
    """

    # Input fields
    query: str
    code_snippet: str | None
    file_list: list[str]
    ide_context: str | None
    preferred_framework: str | None
    max_iterations: int

    # Routing & Framework Selection
    selected_framework: str
    framework_chain: list[str]
    routing_category: str
    task_type: str
    complexity_estimate: float

    # Working Memory
    working_memory: dict[str, Any]
    reasoning_steps: list[dict[str, Any]]
    step_counter: int

    # Episodic Memory
    episodic_memory: list[dict[str, Any]]

    # Output fields
    final_code: str | None
    final_answer: str | None
    confidence_score: float
    tokens_used: int

    # Quiet-STaR internal thoughts
    quiet_thoughts: list[str]

    # Error handling
    error: str | None

    # Retry state (used by graph retry logic)
    retry_count: int
    last_error: str | None


@dataclass
class MemoryStore:
    """
    Persistent memory store for cross-session learning.

    Implements both working memory (current session) and
    episodic memory (historical patterns).

    Uses deque for thread-safe bounded collections that automatically
    discard oldest items when maxlen is reached.
    """

    # Working memory: cleared each session
    working: dict[str, Any] = field(default_factory=dict)

    # Episodic memory: persists across sessions (thread-safe bounded deque)
    episodic: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=1000))

    # Thought buffer: successful reasoning templates (thread-safe bounded deque)
    thought_templates: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))

    # Statistics
    total_queries: int = 0
    successful_queries: int = 0
    framework_usage: dict[str, int] = field(default_factory=dict)

    def add_episode(self, episode: dict[str, Any]) -> None:
        """Add a completed reasoning episode to memory (thread-safe)."""
        self.episodic.append(episode)  # deque auto-discards oldest when full

    def add_thought_template(self, template: dict[str, Any]) -> None:
        """Add a successful thought template to the buffer (thread-safe)."""
        self.thought_templates.append(template)  # deque auto-discards oldest when full

    def find_similar_templates(
        self, task_type: str, keywords: list[str], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Find similar thought templates from the buffer."""
        matches = []
        for template in self.thought_templates:
            # Simple keyword matching (could be enhanced with embeddings)
            template_keywords = template.get("keywords", [])
            overlap = len(set(keywords) & set(template_keywords))
            if template.get("task_type") == task_type or overlap > 0:
                matches.append((overlap, template))

        # Sort by overlap and return top matches
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]

    def record_framework_usage(self, framework: str) -> None:
        """Track which frameworks are being used."""
        self.framework_usage[framework] = self.framework_usage.get(framework, 0) + 1

    def clear_working_memory(self) -> None:
        """Clear working memory for a new session."""
        self.working = {}

    def __repr__(self) -> str:
        """Concise representation showing stats, not full data."""
        return (
            f"MemoryStore("
            f"working_keys={len(self.working)}, "
            f"episodes={len(self.episodic)}, "
            f"templates={len(self.thought_templates)}, "
            f"queries={self.total_queries}, "
            f"success={self.successful_queries})"
        )


def create_initial_state(
    query: str,
    code_snippet: str | None = None,
    file_list: list[str] | None = None,
    ide_context: str | None = None,
    preferred_framework: str | None = None,
    max_iterations: int = 5,
) -> GraphState:
    """Create a fresh state for a new reasoning request."""
    return GraphState(
        # Input
        query=query,
        code_snippet=code_snippet,
        file_list=file_list or [],
        ide_context=ide_context,
        preferred_framework=preferred_framework,
        max_iterations=max_iterations,
        # Routing
        selected_framework="",
        framework_chain=[],
        routing_category="unknown",
        task_type="unknown",
        complexity_estimate=0.5,
        # Working memory
        working_memory={},
        reasoning_steps=[],
        step_counter=0,
        # Episodic memory
        episodic_memory=[],
        # Output
        final_code=None,
        final_answer=None,
        confidence_score=0.5,
        tokens_used=0,
        # Quiet-STaR
        quiet_thoughts=[],
        # Error handling
        error=None,
        # Retry state
        retry_count=0,
        last_error=None,
    )
