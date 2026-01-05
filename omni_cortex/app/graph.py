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
import os

from .state import GraphState
from .core.router import HyperRouter
from .langchain_integration import (
    enhance_state_with_langchain,
    save_to_langchain_memory,
    OmniCortexCallback,
    AVAILABLE_TOOLS
)

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/app/data/checkpoints.sqlite")
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
    reverse_chain_of_thought_node,
    alphacodium_node,
    codechain_node,
    evol_instruct_node,
    llmloop_node,
    procoder_node,
    recode_node,
    parsel_node,
    docprompting_node
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
from .nodes.verification import (
    self_consistency_node,
    self_ask_node,
    rar_node,
    verify_and_edit_node,
    rarr_node,
    selfcheckgpt_node,
    metaqa_node,
    ragas_node
)
from .nodes.agent import (
    rewoo_node,
    lats_node,
    mrkl_node,
    swe_agent_node,
    toolformer_node
)
from .nodes.rag import (
    self_rag_node,
    hyde_node,
    rag_fusion_node,
    raptor_node,
    graphrag_node
)
from .nodes.code import (
    pal_node,
    scratchpads_node
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
    "alphacodium": alphacodium_node,
    "codechain": codechain_node,
    "evol_instruct": evol_instruct_node,
    "llmloop": llmloop_node,
    "procoder": procoder_node,
    "recode": recode_node,
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
    # Verification
    "self_consistency": self_consistency_node,
    "self_ask": self_ask_node,
    "rar": rar_node,
    "verify_and_edit": verify_and_edit_node,
    "rarr": rarr_node,
    "selfcheckgpt": selfcheckgpt_node,
    "metaqa": metaqa_node,
    "ragas": ragas_node,
    # Agent
    "rewoo": rewoo_node,
    "lats": lats_node,
    "mrkl": mrkl_node,
    "swe_agent": swe_agent_node,
    "toolformer": toolformer_node,
    # RAG
    "self_rag": self_rag_node,
    "hyde": hyde_node,
    "rag_fusion": rag_fusion_node,
    "raptor": raptor_node,
    "graphrag": graphrag_node,
    # Code (additional)
    "pal": pal_node,
    "scratchpads": scratchpads_node,
    "parsel": parsel_node,
    "docprompting": docprompting_node,
}


# Initialize router
router = HyperRouter()


async def route_node(state: GraphState) -> GraphState:
    """
    Routing node: AI-powered framework selection.

    Uses HyperRouter with LangChain memory context to analyze the task
    and select the optimal framework.
    """
    # Ensure working_memory exists
    if "working_memory" not in state or state["working_memory"] is None:
        state["working_memory"] = {}

    # Enhance state with LangChain memory if thread_id available
    thread_id = state.get("working_memory", {}).get("thread_id")
    if thread_id:
        state = await enhance_state_with_langchain(state, thread_id)
        logger.info("state_enhanced_with_memory", thread_id=thread_id)

    # Make LangChain tools available to router if needed
    state["working_memory"]["available_tools"] = [
        tool.name for tool in AVAILABLE_TOOLS
    ]

    return await router.route(state, use_ai=True)


async def execute_framework_node(state: GraphState) -> GraphState:
    """
    Execution node: Run frameworks with LangChain monitoring.

    Supports both single framework execution and pipeline execution
    when framework_chain contains multiple frameworks.

    Pipeline Execution Flow:
    1. Each framework in chain receives output of previous
    2. Intermediate results stored in reasoning_steps
    3. Final framework's output becomes final_answer
    4. Token usage aggregated across all frameworks
    """
    # Ensure working_memory exists
    if "working_memory" not in state or state["working_memory"] is None:
        state["working_memory"] = {}

    thread_id = state.get("working_memory", {}).get("thread_id")
    framework_chain = state.get("framework_chain", [])
    selected_framework = state.get("selected_framework")

    # Create callback handler for this execution
    if thread_id:
        callback = OmniCortexCallback(thread_id)
        state["working_memory"]["langchain_callback"] = callback

    # Local import to avoid cycle at module load
    from .nodes.common import list_tools_for_framework

    # Determine execution mode: pipeline (chain) vs single framework
    if framework_chain and len(framework_chain) > 1:
        # Pipeline execution: run multiple frameworks in sequence
        logger.info(
            "pipeline_execution_start",
            chain=framework_chain,
            chain_length=len(framework_chain)
        )

        total_tokens = state.get("tokens_used", 0)
        executed_frameworks = []

        for i, framework_name in enumerate(framework_chain):
            if framework_name not in FRAMEWORK_NODES:
                logger.warning("unknown_framework_in_chain", framework=framework_name)
                continue

            # Update current framework context
            state["selected_framework"] = framework_name
            state["working_memory"]["recommended_tools"] = list_tools_for_framework(
                framework_name, state
            )
            state["working_memory"]["pipeline_position"] = {
                "index": i,
                "total": len(framework_chain),
                "is_first": i == 0,
                "is_last": i == len(framework_chain) - 1,
                "previous_frameworks": executed_frameworks.copy()
            }

            logger.info(
                "pipeline_step_start",
                step=i + 1,
                framework=framework_name,
                total_steps=len(framework_chain)
            )

            # Execute framework
            framework_fn = FRAMEWORK_NODES[framework_name]
            pre_tokens = state.get("tokens_used", 0)
            state = await framework_fn(state)
            post_tokens = state.get("tokens_used", 0)

            # Track execution
            executed_frameworks.append(framework_name)
            step_tokens = post_tokens - pre_tokens

            # Store intermediate result if not last in chain
            if i < len(framework_chain) - 1:
                intermediate_result = {
                    "framework": framework_name,
                    "step": i + 1,
                    "answer": state.get("final_answer", ""),
                    "code": state.get("final_code"),
                    "confidence": state.get("confidence_score", 0.5),
                    "tokens": step_tokens
                }

                # Add to reasoning steps for context
                if "reasoning_steps" not in state:
                    state["reasoning_steps"] = []
                state["reasoning_steps"].append({
                    "thought": f"Pipeline step {i + 1}: {framework_name}",
                    "action": "framework_execution",
                    "observation": state.get("final_answer", "")[:500]
                })

                # For next framework, the previous answer becomes context
                state["working_memory"]["pipeline_context"] = intermediate_result

                logger.info(
                    "pipeline_step_complete",
                    step=i + 1,
                    framework=framework_name,
                    tokens=step_tokens
                )

        # Record all executed frameworks
        state["working_memory"]["executed_chain"] = executed_frameworks
        logger.info(
            "pipeline_execution_complete",
            executed=executed_frameworks,
            total_tokens=state.get("tokens_used", 0)
        )

    else:
        # Single framework execution (original behavior)
        state["working_memory"]["recommended_tools"] = list_tools_for_framework(
            selected_framework or "unknown", state
        )

        if selected_framework and selected_framework in FRAMEWORK_NODES:
            framework_fn = FRAMEWORK_NODES[selected_framework]
            state = await framework_fn(state)
        else:
            # Fallback to self_discover
            state = await self_discover_node(state)
            state["selected_framework"] = "self_discover (fallback)"

    # Save to LangChain memory after execution
    if thread_id and state.get("final_answer"):
        # For pipelines, record all frameworks used
        frameworks_used = state.get("working_memory", {}).get(
            "executed_chain",
            [state.get("selected_framework", "unknown")]
        )
        framework_str = " -> ".join(frameworks_used) if isinstance(frameworks_used, list) else frameworks_used

        await save_to_langchain_memory(
            thread_id=thread_id,
            query=state.get("query", ""),
            answer=state.get("final_answer", ""),
            framework=framework_str
        )
        logger.info(
            "saved_to_langchain_memory",
            thread_id=thread_id,
            framework=framework_str,
            is_pipeline=len(frameworks_used) > 1 if isinstance(frameworks_used, list) else False
        )

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
    """
    Get async SQLite checkpointer for LangGraph.

    IMPORTANT: Lifecycle Management
    -------------------------------
    The returned AsyncSqliteSaver maintains an internal connection pool.
    Callers are responsible for cleanup when done:

        checkpointer = await get_checkpointer()
        try:
            graph = create_reasoning_graph(checkpointer=checkpointer)
            # ... use graph ...
        finally:
            await checkpointer.conn.close()  # Close the underlying connection

    For long-running servers, consider using the checkpointer as a singleton
    that persists for the application lifetime, closing only on shutdown.
    """
    # Ensure directory exists
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(CHECKPOINT_PATH)


# Create the global graph instance (without checkpointer for import-time initialization)
# Checkpointer should be added at runtime when needed
graph = create_reasoning_graph()
