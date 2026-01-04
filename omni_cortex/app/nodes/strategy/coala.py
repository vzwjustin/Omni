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
        """Store a completed task episode in the vector db."""
        try:
            from ...collection_manager import get_collection_manager
            manager = get_collection_manager()
            
            # Format episode for storage
            content = f"TASK: {task}\nACTIONS: {actions}\nOUTCOME: {outcome}\nSUCCESS: {success}"
            
            # Use 'frameworks' collection or a dedicated 'memories' one if available
            # For now, tagging it as 'coala_episode' in frameworks
            metadata = {
                "path": f"memory/coala/{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                "file_name": "episode.md",
                "file_type": ".md",
                "category": "framework",  # Storing in frameworks for now
                "chunk_type": "memory",
                "tags": f"coala,episode,success={success}",
            }
            
            manager.add_documents([content], [metadata], collection_name="frameworks")
            logger.info("episode_stored", task=task[:50])
            
        except Exception as e:
            logger.error("store_episode_failed", error=str(e))
    
    def recall_similar(self, task: str, limit: int = 3) -> list[dict]:
        """Recall episodes similar to current task from vector db."""
        try:
            from ...collection_manager import get_collection_manager
            manager = get_collection_manager()
            
            # Search specifically for coala episodes
            docs = manager.search(
                task, 
                collection_names=["frameworks"], 
                k=limit,
                filter_dict={"chunk_type": "memory"}
            )
            
            episodes = []
            for d in docs:
                episodes.append({
                    "task_summary": d.page_content[:200], # Approximate
                    "outcome": d.page_content[200:400],
                    "success": "SUCCESS: True" in d.page_content,
                    "content": d.page_content
                })
            return episodes
            
        except Exception as e:
            logger.error("recall_failed", error=str(e))
            return []
    
    def get_success_patterns(self) -> list[str]:
        """Extract patterns (placeholder for sophisticated analytics)."""
        return ["Break down complex tasks", "Verify code before running", "Check constraints"]


# Memory instances
@quiet_star
async def coala_node(state: GraphState) -> GraphState:
    """
    CoALA: Cognitive Architecture with Layered Memory.
    (Headless Mode: Returns Context & Prompts for Client Execution)
    """
    _episodic_memory = EpisodicMemory()
    
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )
    
    # 1. PERCEIVE & RECALL (Server-side RAG)
    similar_episodes = _episodic_memory.recall_similar(query)
    success_patterns = _episodic_memory.get_success_patterns()
    
    episodic_context = ""
    if similar_episodes:
        episodic_context = "\n\n### Relevant Past Episodes\n"
        for ep in similar_episodes:
            episodic_context += f"```\n{ep['content']}\n```\n"
            
    if success_patterns:
        episodic_context += f"\n### Suggested Patterns\n{', '.join(success_patterns)}\n"
        
    # 2. CONSTRUCT PROMPT (For Client)
    prompt = f"""# CoALA Framework Execution

I have prepared the cognitive context for this task using the CoALA architecture.
Please execute the following reasoning steps using your internal LLM.

## 🧠 Cognitive Context
**Task**: {query}
**Patterns**: {', '.join(success_patterns)}

{episodic_context}

## 📋 Execution Plan (Client-Side)

1. **Working Memory Analysis**: 
   - Identify Main Goal, Sub-goals, and Constraints.
   - List key entities from the code context.

2. **Reasoning**: 
   - Based on the Past Episodes above, what strategy works best?
   - What pitfalls should be avoided?

3. **Action**: 
   - Generate the solution/code.
   - Verify it against the constraints.

## 📝 Code Context
{code_context}
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0
    return state


def reset_cognitive_state() -> None:
    """Reset all cognitive state for testing (deprecated - memory is now per-request)."""
    # No-op: Memory is now created per-request, no global state to reset
    pass
