"""
Graph State Management for LangGraph Workflows

Defines the state structure that flows through reasoning frameworks.
Refactored to use composition and strictly typed sub-states.
"""

from typing import TypedDict, Optional, Any, List, Dict
from dataclasses import dataclass, field


class InputState(TypedDict, total=False):
    """Immutable input parameters."""
    query: str
    code_snippet: Optional[str]
    file_list: List[str]
    ide_context: Optional[str]
    preferred_framework: Optional[str]
    max_iterations: int


class ReasoningState(TypedDict, total=False):
    """Internal working state for the reasoning engine."""
    selected_framework: str
    framework_chain: List[str]
    routing_category: str
    task_type: str
    complexity_estimate: float
    working_memory: Dict[str, Any]
    reasoning_steps: List[Dict[str, Any]]
    step_counter: int
    tokens_used: int
    quiet_thoughts: List[str]
    episodic_memory: List[Dict[str, Any]]


class OutputState(TypedDict, total=False):
    """Final results."""
    final_code: Optional[str]
    final_answer: Optional[str]
    confidence_score: float
    error: Optional[str]


class GraphState(TypedDict, total=False):
    """
    Central state object passed through LangGraph nodes.
    
    Refactored to flatten the structure for backward compatibility with existing nodes,
    while internally supporting the composed types if needed in future refactors.
    
    NOTE: Currently flattened to maintain compatibility with existing
    node implementations like `state['query']` or `state['working_memory']`.
    """
    # Input fields
    query: str
    code_snippet: Optional[str]
    file_list: List[str]
    ide_context: Optional[str]
    preferred_framework: Optional[str]
    max_iterations: int

    # Routing & Framework Selection
    selected_framework: str
    framework_chain: List[str]
    routing_category: str
    task_type: str
    complexity_estimate: float

    # Working Memory
    working_memory: Dict[str, Any]
    reasoning_steps: List[Dict[str, Any]]
    step_counter: int

    # Episodic Memory
    episodic_memory: List[Dict[str, Any]]

    # Output fields
    final_code: Optional[str]
    final_answer: Optional[str]
    confidence_score: float
    tokens_used: int

    # Quiet-STaR internal thoughts
    quiet_thoughts: List[str]

    # Error handling
    error: Optional[str]


@dataclass
class MemoryStore:
    """
    Persistent memory store for cross-session learning.

    Implements both working memory (current session) and
    episodic memory (historical patterns).
    """

    # Working memory: cleared each session
    working: dict[str, Any] = field(default_factory=dict)

    # Episodic memory: persists across sessions
    episodic: list[dict[str, Any]] = field(default_factory=list)

    # Thought buffer: successful reasoning templates
    thought_templates: list[dict[str, Any]] = field(default_factory=list)

    # Statistics
    total_queries: int = 0
    successful_queries: int = 0
    framework_usage: dict[str, int] = field(default_factory=dict)

    def add_episode(self, episode: dict[str, Any]) -> None:
        """Add a completed reasoning episode to memory."""
        self.episodic.append(episode)
        # Keep only the last 1000 episodes
        if len(self.episodic) > 1000:
            self.episodic = self.episodic[-1000:]

    def add_thought_template(self, template: dict[str, Any]) -> None:
        """Add a successful thought template to the buffer."""
        self.thought_templates.append(template)
        # Keep only the last 500 templates
        if len(self.thought_templates) > 500:
            self.thought_templates = self.thought_templates[-500:]

    def find_similar_templates(
        self,
        task_type: str,
        keywords: list[str],
        limit: int = 5
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
    code_snippet: Optional[str] = None,
    file_list: Optional[list[str]] = None,
    ide_context: Optional[str] = None,
    preferred_framework: Optional[str] = None,
    max_iterations: int = 5
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
    )
