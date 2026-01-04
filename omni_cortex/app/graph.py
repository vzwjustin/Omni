"""
LangGraph Workflow for Omni-Cortex

Defines the graph structure for orchestrating reasoning frameworks
with proper state management and memory persistence.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
import structlog

from .state import GraphState
from .core.router import HyperRouter
from .langchain_integration import (
    enhance_state_with_langchain,
    save_to_langchain_memory,
    OmniCortexCallback,
    AVAILABLE_TOOLS
)

CHECKPOINT_PATH = "/app/data/checkpoints.sqlite"
logger = structlog.get_logger("graph")

# Import all framework nodes
from .nodes.strategy import (
    reason_flux_node,
    self_discover_node,
    buffer_of_thoughts_node,
    coala_node,
    least_to_most_node,
    comparative_architecture_node,
    plan_and_solve_node
)
from .nodes.search import (
    mcts_rstar_node,
    tree_of_thoughts_node,
    graph_of_thoughts_node,
    everything_of_thought_node
)
from .nodes.iterative import (
    active_inference_node,
    multi_agent_debate_node,
    adaptive_injection_node,
    re2_node,
    rubber_duck_debugging_node,
    react_node,
    reflexion_node,
    self_refine_node
)
from .nodes.code import (
    program_of_thoughts_node,
    chain_of_verification_node,
    critic_node,
    chain_of_code_node,
    self_debugging_node,
    tdd_prompting_node,
    reverse_chain_of_thought_node
)
from .nodes.context import (
    chain_of_note_node,
    step_back_node,
    analogical_node,
    red_team_node,
    state_machine_node,
    chain_of_thought_node
)
from .nodes.fast import (
    skeleton_of_thought_node,
    system1_node
)


# Framework registry
FRAMEWORK_NODES = {
    # Strategy
    "reason_flux": reason_flux_node,
    "self_discover": self_discover_node,
    "buffer_of_thoughts": buffer_of_thoughts_node,
    "coala": coala_node,
    "least_to_most": least_to_most_node,
    "comparative_arch": comparative_architecture_node,
    "plan_and_solve": plan_and_solve_node,
    # Search
    "mcts_rstar": mcts_rstar_node,
    "tree_of_thoughts": tree_of_thoughts_node,
    "graph_of_thoughts": graph_of_thoughts_node,
    "everything_of_thought": everything_of_thought_node,
    # Iterative
    "active_inference": active_inference_node,
    "multi_agent_debate": multi_agent_debate_node,
    "adaptive_injection": adaptive_injection_node,
    "re2": re2_node,
    "rubber_duck": rubber_duck_debugging_node,
    "react": react_node,
    "reflexion": reflexion_node,
    "self_refine": self_refine_node,
    # Code
    "program_of_thoughts": program_of_thoughts_node,
    "chain_of_verification": chain_of_verification_node,
    "critic": critic_node,
    "chain_of_code": chain_of_code_node,
    "self_debugging": self_debugging_node,
    "tdd_prompting": tdd_prompting_node,
    "reverse_cot": reverse_chain_of_thought_node,
    # Context
    "chain_of_note": chain_of_note_node,
    "step_back": step_back_node,
    "analogical": analogical_node,
    "red_team": red_team_node,
    "state_machine": state_machine_node,
    "chain_of_thought": chain_of_thought_node,
    # Fast
    "skeleton_of_thought": skeleton_of_thought_node,
    "system1": system1_node,
}


# Initialize router
router = HyperRouter()


async def route_node(state: GraphState) -> GraphState:
    """
    Routing node: AI-powered framework selection.
    
    Uses HyperRouter with LangChain memory context to analyze the task
    and select the optimal framework.
    """
    # Enhance state with LangChain memory if thread_id available
    thread_id = state.get("working_memory", {}).get("thread_id")
    if thread_id:
        state = enhance_state_with_langchain(state, thread_id)
        logger.info("state_enhanced_with_memory", thread_id=thread_id)
    
    # Make LangChain tools available to router if needed
    state["working_memory"]["available_tools"] = [
        tool.name for tool in AVAILABLE_TOOLS
    ]
    
    return await router.route(state, use_ai=True)


async def execute_framework_node(state: GraphState) -> GraphState:
    """
    Execution node: Run the selected framework with LangChain monitoring.
    
    Dynamically calls the appropriate framework based on routing decision,
    with callback tracking and memory persistence.
    """
    selected_framework = state.get("selected_framework")
    thread_id = state.get("working_memory", {}).get("thread_id")
    
    # Create callback handler for this execution
    if thread_id:
        callback = OmniCortexCallback(thread_id)
        state["working_memory"]["langchain_callback"] = callback
    
    # Surface recommended tools for this framework (for prompt/context usage)
    from .nodes.common import list_tools_for_framework  # local import to avoid cycle at module load
    state["working_memory"]["recommended_tools"] = list_tools_for_framework(
        selected_framework or "unknown", state
    )
    
    # Execute the framework
    if selected_framework and selected_framework in FRAMEWORK_NODES:
        framework_fn = FRAMEWORK_NODES[selected_framework]
        state = await framework_fn(state)
    else:
        # Fallback to self_discover
        state = await self_discover_node(state)
        state["selected_framework"] = "self_discover (fallback)"
    
    # Save to LangChain memory after execution
    if thread_id and state.get("final_answer"):
        save_to_langchain_memory(
            thread_id=thread_id,
            query=state["query"],
            answer=state["final_answer"],
            framework=selected_framework or "unknown"
        )
        logger.info("saved_to_langchain_memory", thread_id=thread_id, framework=selected_framework)
    
    return state


def should_continue(state: GraphState) -> Literal["execute", "end"]:
    """
    Conditional edge: Determine if we should execute or end.
    
    Always execute after routing (could add retry logic here in the future).
    """
    # Check if we have a selected framework
    if state.get("selected_framework"):
        return "execute"
    else:
        return "end"


def create_reasoning_graph(checkpointer=None) -> StateGraph:
    """
    Create the LangGraph workflow for Omni-Cortex reasoning.

    Graph structure:
    1. START -> route_node (AI selects framework)
    2. route_node -> should_continue (conditional)
    3. should_continue -> execute_framework_node (run selected framework)
    4. execute_framework_node -> END

    Returns compiled graph with optional memory checkpointing.
    """
    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("route", route_node)
    workflow.add_node("execute", execute_framework_node)

    # Add edges
    workflow.set_entry_point("route")
    workflow.add_conditional_edges(
        "route",
        should_continue,
        {
            "execute": "execute",
            "end": END
        }
    )
    workflow.add_edge("execute", END)

    # Compile with optional checkpointer
    compiled_graph = workflow.compile(checkpointer=checkpointer)

    return compiled_graph


async def get_checkpointer():
    """Get async SQLite checkpointer for LangGraph."""
    import os
    os.makedirs("/app/data", exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(CHECKPOINT_PATH)


# Create the global graph instance (without checkpointer for import-time initialization)
# Checkpointer should be added at runtime when needed
graph = create_reasoning_graph()
