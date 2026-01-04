"""
CoALA: Cognitive Architecture with Layered Memory

Implements working memory vs episodic memory management
for long-context, multi-file, stateful tasks.
"""

import logging
from typing import Optional, Any
from datetime import datetime
from ...state import GraphState, MemoryStore
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    run_tool,
)

logger = logging.getLogger(__name__)


# Memory layers
class WorkingMemory:
    """Short-term, task-specific memory with limited capacity."""
    
    MAX_ITEMS = 7  # Miller's Law: 7 ± 2
    
    def __init__(self):
        self.items: list[dict[str, Any]] = []
        self.focus: Optional[str] = None
        self.goals: list[str] = []
    
    def add(self, key: str, value: Any, importance: float = 0.5) -> None:
        """Add item to working memory, evicting if necessary."""
        # Evict least important BEFORE adding if at capacity
        if len(self.items) >= self.MAX_ITEMS:
            self.items.sort(key=lambda x: x["importance"])
            self.items.pop(0)
        
        self.items.append({
            "key": key,
            "value": value,
            "importance": importance,
            "timestamp": datetime.now().isoformat()
        })
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from working memory."""
        for item in self.items:
            if item["key"] == key:
                return item["value"]
        return None
    
    def set_focus(self, focus: str) -> None:
        """Set current focus/attention."""
        self.focus = focus
    
    def to_context(self) -> str:
        """Serialize working memory for LLM context."""
        lines = [f"[FOCUS]: {self.focus or 'None'}"]
        lines.append(f"[GOALS]: {', '.join(self.goals) if self.goals else 'None'}")
        lines.append("[ACTIVE ITEMS]:")
        for item in sorted(self.items, key=lambda x: x["importance"], reverse=True):
            lines.append(f"  - {item['key']}: {str(item['value'])[:100]}...")
        return "\n".join(lines)


class EpisodicMemory:
    """Long-term memory of past experiences and patterns."""
    
    MAX_EPISODES = 100
    
    def __init__(self):
        self.episodes: list[dict[str, Any]] = []
    
    def store_episode(
        self,
        task: str,
        context: str,
        actions: list[str],
        outcome: str,
        success: bool
    ) -> None:
        """Store a completed task episode."""
        self.episodes.append({
            "timestamp": datetime.now().isoformat(),
            "task_summary": task[:200],
            "context_hash": hash(context),
            "actions": actions,
            "outcome": outcome,
            "success": success
        })
        
        # Trim old episodes
        if len(self.episodes) > self.MAX_EPISODES:
            self.episodes = self.episodes[-self.MAX_EPISODES:]
    
    def recall_similar(self, task: str, limit: int = 3) -> list[dict]:
        """Recall episodes similar to current task."""
        # Simple keyword matching (could use embeddings)
        task_words = set(task.lower().split())
        
        scored = []
        for episode in self.episodes:
            ep_words = set(episode["task_summary"].lower().split())
            overlap = len(task_words & ep_words)
            if overlap > 0:
                scored.append((overlap, episode))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e[1] for e in scored[:limit]]
    
    def get_success_patterns(self) -> list[str]:
        """Extract patterns from successful episodes."""
        successful = [e for e in self.episodes if e["success"]]
        
        if not successful:
            return []
        
        # Find common actions in successful episodes
        action_counts: dict[str, int] = {}
        for ep in successful:
            for action in ep["actions"]:
                action_counts[action] = action_counts.get(action, 0) + 1
        
        # Return most common successful actions
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        return [a[0] for a in sorted_actions[:5]]


# Memory instances - created per request to avoid pollution
# Global storage removed to prevent cross-request data leakage


@quiet_star
async def coala_node(state: GraphState) -> GraphState:
    """
    CoALA: Cognitive Architecture with Layered Memory.
    
    Five-phase cognitive cycle:
    1. PERCEIVE: Process input into working memory
    2. RECALL: Retrieve relevant episodes from long-term memory
    3. REASON: Deliberate with both memory layers
    4. ACT: Execute the decided action
    5. LEARN: Store experience in episodic memory
    
    Best for: Long-context tasks, multi-file, stateful reasoning
    """
    # Create per-request memory instances to avoid cross-request pollution
    _working_memory = WorkingMemory()
    _episodic_memory = EpisodicMemory()
    
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )
    
    # Retrieve prior conversation/context to enrich reasoning
    try:
        retrieved_context = await run_tool("retrieve_context", state.get("query", ""), state)
    except Exception as e:
        # Log context retrieval failure but continue with empty context
        logger.debug("Context retrieval failed", error=str(e))
        retrieved_context = ""
    
    # Learn from similar framework implementations for complex tasks
    framework_examples = ""
    try:
        # If task mentions specific frameworks or patterns, search for them
        if any(keyword in query.lower() for keyword in ["pattern", "implement", "framework", "architecture"]):
            framework_examples = await run_tool("search_with_framework_context",
                                               {"query": query[:100], "framework_category": "strategy", "k": 2},
                                               state)
    except Exception as e:
        # Log framework search failure but continue - this is optional context enrichment
        logger.debug("Framework context search failed", error=str(e))
    
    # =========================================================================
    # Phase 1: PERCEIVE - Process input into working memory
    # =========================================================================
    
    # Set focus
    _working_memory.set_focus(query[:100])
    
    # Extract key elements for working memory
    perception_prompt = f"""Analyze this task and extract KEY ELEMENTS for working memory.

TASK: {query}

CONTEXT:
{code_context}

Extract:
1. MAIN GOAL: What is the primary objective?
2. SUB-GOALS: What are the component tasks? (max 3)
3. KEY ENTITIES: Important code elements (classes, functions, files)
4. CONSTRAINTS: Any limitations or requirements
5. PRIORITY ITEMS: What needs immediate attention?

Format as structured key-value pairs."""

    perception_response, tokens1 = await call_fast_synthesizer(
        prompt=perception_prompt,
        state=state,
        max_tokens=500
    )
    
    # Populate working memory
    _working_memory.add("main_goal", perception_response.split("MAIN GOAL:")[-1].split("\n")[0] if "MAIN GOAL:" in perception_response else query[:100], importance=1.0)
    _working_memory.add("full_query", query, importance=0.9)
    _working_memory.add("code_context", code_context[:500], importance=0.8)
    
    add_reasoning_step(
        state=state,
        framework="coala",
        thought="Perceived and processed input into working memory",
        action="perception",
        observation=f"Focus set: {_working_memory.focus}"
    )
    
    # =========================================================================
    # Phase 2: RECALL - Retrieve from episodic memory
    # =========================================================================
    
    similar_episodes = _episodic_memory.recall_similar(query)
    success_patterns = _episodic_memory.get_success_patterns()
    
    episodic_context = ""
    if similar_episodes:
        episodic_context = "\n\nRELEVANT PAST EXPERIENCES:\n"
        for ep in similar_episodes:
            episodic_context += f"- Task: {ep['task_summary']}\n"
            episodic_context += f"  Outcome: {ep['outcome'][:100]}\n"
            episodic_context += f"  Success: {'Yes' if ep['success'] else 'No'}\n"
    
    if success_patterns:
        episodic_context += f"\nSUCCESSFUL PATTERNS: {', '.join(success_patterns)}\n"
    
    add_reasoning_step(
        state=state,
        framework="coala",
        thought=f"Retrieved {len(similar_episodes)} similar episodes from memory",
        action="memory_recall",
        observation=f"Success patterns: {success_patterns[:3] if success_patterns else 'None'}"
    )
    
    # =========================================================================
    # Phase 3: REASON - Deliberate with full cognitive context
    # =========================================================================
    
    working_context = _working_memory.to_context()
    
    reasoning_prompt = f"""You are CoALA, a cognitive architecture with layered memory.

CURRENT WORKING MEMORY:
{working_context}

{episodic_context}

TASK: {query}

CONTEXT:
{code_context}

Using BOTH your working memory and episodic memory, reason through this task:

1. **Analysis**: What do these memories tell you about the task?
2. **Strategy**: What approach should you take? (Informed by past success patterns)
3. **Potential Issues**: What problems might arise? (Learned from past failures)
4. **Action Plan**: Ordered list of specific actions to take

Be thorough - you have access to rich cognitive context."""

    reasoning_response, tokens2 = await call_deep_reasoner(
        prompt=reasoning_prompt,
        state=state,
        system="You are CoALA in REASONING mode. Use your full cognitive capacity.",
        temperature=0.7
    )
    
    add_reasoning_step(
        state=state,
        framework="coala",
        thought="Deliberated using working and episodic memory",
        action="reasoning",
        observation="Action plan generated from cognitive synthesis"
    )
    
    # =========================================================================
    # Phase 4: ACT - Execute the action plan
    # =========================================================================
    
    action_prompt = f"""Execute the action plan you developed.

YOUR REASONING AND PLAN:
{reasoning_response}

TASK: {query}

CONTEXT:
{code_context}

Now EXECUTE each action in your plan:

For each action:
1. State what you're doing
2. Show the work/output
3. Verify it's correct

Provide:
- **FINAL SOLUTION**: Complete answer/implementation
- **CODE** (if applicable): Ready-to-use code
- **SUMMARY**: What was accomplished"""

    action_response, tokens3 = await call_deep_reasoner(
        prompt=action_prompt,
        state=state,
        system="You are CoALA in ACTION mode. Execute precisely.",
        temperature=0.6,
        max_tokens=6000
    )
    
    add_reasoning_step(
        state=state,
        framework="coala",
        thought="Executed action plan",
        action="execution",
        observation="Solution implemented"
    )
    
    # =========================================================================
    # Phase 5: LEARN - Store in episodic memory
    # =========================================================================
    
    # Extract actions taken
    actions = [step["action"] for step in state["reasoning_steps"]]
    
    # Store in episodic memory for future reference
    _episodic_memory.store_episode(
        task=query,
        context=code_context,
        actions=actions,
        outcome=action_response[:500],
        success=True  # Assume success; could add verification
    )
    
    add_reasoning_step(
        state=state,
        framework="coala",
        thought="Stored experience in episodic memory for future learning",
        action="learning",
        observation="Episode stored successfully"
    )
    
    # Extract code if present
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, action_response, re.DOTALL)
    
    # Update final state
    state["final_answer"] = action_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.8 + (0.1 if similar_episodes else 0)  # Higher if had relevant episodes
    
    # Clear working memory for next task
    _working_memory.items = []
    
    return state


def reset_cognitive_state() -> None:
    """Reset all cognitive state for testing (deprecated - memory is now per-request)."""
    # No-op: Memory is now created per-request, no global state to reset
    pass
