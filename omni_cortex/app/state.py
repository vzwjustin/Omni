"""
Graph State Management for LangGraph Workflows

Defines the state structure that flows through reasoning frameworks.
"""

from typing import TypedDict, Optional, Any
from dataclasses import dataclass, field


class GraphState(TypedDict, total=False):
    """
    Central state object passed through LangGraph nodes.
    
    Uses TypedDict for LangGraph compatibility while maintaining
    full type hints for IDE support.
    """
    
    # Input fields
    query: str
    code_snippet: Optional[str]
    file_list: list[str]
    ide_context: Optional[str]
    preferred_framework: Optional[str]
    max_iterations: int
    
    # Routing & Framework Selection
    selected_framework: str
    task_type: str  # "debug", "architecture", "algorithm", "refactor", "docs", "unknown"
    complexity_estimate: float
    
    # Working Memory (short-term, current task)
    working_memory: dict[str, Any]
    current_hypothesis: Optional[str]
    current_code: Optional[str]
    reasoning_steps: list[dict[str, Any]]
    
    # Episodic Memory (long-term, cross-task patterns)
    episodic_memory: list[dict[str, Any]]
    
    # Buffer of Thoughts (successful templates)
    thought_buffer: list[dict[str, Any]]
    matched_template: Optional[dict[str, Any]]
    
    # Search State (for MCTS, ToT, GoT)
    search_tree: dict[str, Any]
    current_node_id: str
    best_path: list[str]
    explored_nodes: list[str]
    
    # Debate State (for Multi-Agent Debate)
    proponent_arguments: list[str]
    critic_arguments: list[str]
    debate_round: int
    consensus_reached: bool
    
    # Verification State
    verification_checks: list[dict[str, Any]]
    passed_checks: int
    failed_checks: int
    
    # Process Reward Model Scores
    prm_scores: list[float]
    cumulative_prm_score: float
    
    # Output fields
    final_code: Optional[str]
    final_answer: Optional[str]
    confidence_score: float
    tokens_used: int
    
    # Quiet-STaR internal thoughts
    quiet_thoughts: list[str]
    
    # Error handling
    error: Optional[str]
    should_retry: bool
    retry_count: int


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
        task_type="unknown",
        complexity_estimate=0.5,
        
        # Working memory
        working_memory={},
        current_hypothesis=None,
        current_code=code_snippet,
        reasoning_steps=[],
        
        # Episodic memory
        episodic_memory=[],
        
        # Buffer
        thought_buffer=[],
        matched_template=None,
        
        # Search
        search_tree={},
        current_node_id="root",
        best_path=[],
        explored_nodes=[],
        
        # Debate
        proponent_arguments=[],
        critic_arguments=[],
        debate_round=0,
        consensus_reached=False,
        
        # Verification
        verification_checks=[],
        passed_checks=0,
        failed_checks=0,
        
        # PRM
        prm_scores=[],
        cumulative_prm_score=0.0,
        
        # Output
        final_code=None,
        final_answer=None,
        confidence_score=0.5,
        tokens_used=0,
        
        # Quiet-STaR
        quiet_thoughts=[],
        
        # Error handling
        error=None,
        should_retry=False,
        retry_count=0
    )
