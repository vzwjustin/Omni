"""
LangGraph Workflow for Omni-Cortex

Defines the graph structure for orchestrating reasoning frameworks
with proper state management and memory persistence.
"""

import asyncio
import os
import time
from typing import Awaitable, Callable, Dict, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import structlog

from .state import GraphState
from .core.router import HyperRouter
from .core.settings import get_settings
from .core.audit import log_framework_execution
from .langchain_integration import (
    enhance_state_with_langchain,
    save_to_langchain_memory,
    OmniCortexCallback,
    AVAILABLE_TOOLS
)

# Import generated nodes - SINGLE SOURCE OF TRUTH
# This replaces 62 manual imports with one line
from .nodes.generator import GENERATED_NODES

CHECKPOINT_PATH = str(get_settings().checkpoint_path)
logger = structlog.get_logger("graph")

# Retry configuration
MAX_RETRIES = 3
BASE_BACKOFF_MS = 100  # Base delay for exponential backoff

# Framework registry - now auto-generated from FrameworkDefinition
# Type annotation for the framework nodes dictionary
FRAMEWORK_NODES: Dict[str, Callable[[GraphState], Awaitable[GraphState]]] = GENERATED_NODES


# Initialize router
router = HyperRouter()


def _log_framework_metrics(
    framework_name: str,
    tokens_used: int,
    duration_ms: float,
    confidence_score: float
) -> None:
    """Log execution metrics for a framework."""
    logger.info(
        "framework_execution_metrics",
        framework=framework_name,
        tokens_used=tokens_used,
        duration_ms=round(duration_ms, 2),
        confidence_score=round(confidence_score, 3)
    )


async def _execute_pipeline(
    state: GraphState,
    framework_chain: list[str],
    list_tools_fn: Callable
) -> GraphState:
    """
    Execute multiple frameworks in sequence (pipeline mode).

    Each framework receives the output of the previous one.
    Intermediate results are stored in reasoning_steps.

    Args:
        state: Current graph state
        framework_chain: List of framework names to execute in order
        list_tools_fn: Function to get recommended tools for a framework

    Returns:
        Updated graph state after all frameworks have executed
    """
    logger.info(
        "pipeline_execution_start",
        chain=framework_chain,
        chain_length=len(framework_chain)
    )

    executed_frameworks = []

    for i, framework_name in enumerate(framework_chain):
        if framework_name not in FRAMEWORK_NODES:
            logger.warning("unknown_framework_in_chain", framework=framework_name)
            continue

        # Update current framework context
        state["selected_framework"] = framework_name
        state["working_memory"]["recommended_tools"] = list_tools_fn(
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

        # Execute framework with metrics
        framework_fn = FRAMEWORK_NODES[framework_name]
        pre_tokens = state.get("tokens_used", 0)
        start_time = time.perf_counter()

        state = await framework_fn(state)

        end_time = time.perf_counter()
        post_tokens = state.get("tokens_used", 0)
        step_tokens = post_tokens - pre_tokens
        duration_ms = (end_time - start_time) * 1000

        # Log metrics for this framework
        _log_framework_metrics(
            framework_name=framework_name,
            tokens_used=step_tokens,
            duration_ms=duration_ms,
            confidence_score=state.get("confidence_score", 0.0)
        )

        # Track execution
        executed_frameworks.append(framework_name)

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

    return state


async def _execute_single(
    state: GraphState,
    selected_framework: str | None,
    list_tools_fn: Callable
) -> GraphState:
    """
    Execute a single framework.

    Args:
        state: Current graph state
        selected_framework: Name of the framework to execute
        list_tools_fn: Function to get recommended tools for a framework

    Returns:
        Updated graph state after framework execution
    """
    state["working_memory"]["recommended_tools"] = list_tools_fn(
        selected_framework or "unknown", state
    )

    framework_name = selected_framework
    pre_tokens = state.get("tokens_used", 0)
    start_time = time.perf_counter()

    if selected_framework and selected_framework in FRAMEWORK_NODES:
        framework_fn = FRAMEWORK_NODES[selected_framework]
        state = await framework_fn(state)
    else:
        # Fallback to self_discover (from generated nodes)
        fallback_fn = FRAMEWORK_NODES.get("self_discover")
        if fallback_fn:
            state = await fallback_fn(state)
        state["selected_framework"] = "self_discover (fallback)"
        framework_name = "self_discover"

    end_time = time.perf_counter()
    post_tokens = state.get("tokens_used", 0)
    duration_ms = (end_time - start_time) * 1000

    # Log metrics for this framework
    _log_framework_metrics(
        framework_name=framework_name or "unknown",
        tokens_used=post_tokens - pre_tokens,
        duration_ms=duration_ms,
        confidence_score=state.get("confidence_score", 0.0)
    )

    return state


async def route_node(state: GraphState) -> GraphState:
    """
    Routing node: AI-powered framework selection.

    Uses HyperRouter with LangChain memory context to analyze the task
    and select the optimal framework.
    """
    # Ensure working_memory exists (defensive - use .get() for safety)
    state["working_memory"] = state.get("working_memory") or {}

    # Enhance state with LangChain memory if thread_id available
    thread_id = state.get("working_memory", {}).get("thread_id")
    if thread_id:
        state = await enhance_state_with_langchain(state, thread_id)
        # Re-ensure working_memory after enhancement (state may be replaced)
        state["working_memory"] = state.get("working_memory") or {}
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
    # Ensure working_memory exists (defensive - use .get() for safety)
    state["working_memory"] = state.get("working_memory") or {}

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
        state = await _execute_pipeline(state, framework_chain, list_tools_for_framework)
    else:
        state = await _execute_single(state, selected_framework, list_tools_for_framework)

    # Audit log the framework execution
    frameworks_used = state.get("working_memory", {}).get(
        "executed_chain",
        [state.get("selected_framework", "unknown")]
    )
    framework_str = " -> ".join(frameworks_used) if isinstance(frameworks_used, list) else frameworks_used

    log_framework_execution(
        framework=framework_str,
        query=state.get("query", ""),
        thread_id=thread_id,
        tokens_used=state.get("tokens_used", 0),
        confidence=state.get("confidence_score", 0.0),
        duration_ms=0.0,  # Duration tracked per-step in pipeline/single execution
        success=True,
        error=None
    )

    # Save to LangChain memory after execution
    if thread_id and state.get("final_answer"):
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


def should_continue(state: GraphState) -> Literal["execute", "end", "retry"]:
    """
    Conditional edge: Determine if we should execute, retry, or end.

    Supports retry logic with exponential backoff for transient failures.
    Checks retry_count against MAX_RETRIES before allowing retry.

    Returns:
        "execute" - Proceed with framework execution
        "retry" - Retry routing (after backoff delay)
        "end" - Stop the workflow
    """
    # Check if we have a selected framework
    if state.get("selected_framework"):
        return "execute"

    # Check retry eligibility
    retry_count = state.get("retry_count", 0)
    last_error = state.get("last_error")

    if last_error and retry_count < MAX_RETRIES:
        # Calculate exponential backoff delay for logging
        backoff_ms = BASE_BACKOFF_MS * (2 ** retry_count)
        logger.warning(
            "retry_scheduled",
            retry_count=retry_count + 1,
            max_retries=MAX_RETRIES,
            backoff_ms=backoff_ms,
            error=last_error
        )
        return "retry"

    if retry_count >= MAX_RETRIES:
        logger.error(
            "max_retries_exceeded",
            retry_count=retry_count,
            max_retries=MAX_RETRIES,
            last_error=last_error
        )

    return "end"


async def retry_with_backoff(state: GraphState) -> GraphState:
    """
    Retry node: Apply exponential backoff before re-routing.

    Increments retry_count and waits before allowing another attempt.
    """
    retry_count = state.get("retry_count", 0)
    backoff_ms = BASE_BACKOFF_MS * (2 ** retry_count)

    # Apply backoff delay
    await asyncio.sleep(backoff_ms / 1000.0)

    # Increment retry count
    state["retry_count"] = retry_count + 1

    # Clear error for fresh attempt
    state["last_error"] = None

    logger.info(
        "retry_attempt",
        retry_count=state["retry_count"],
        backoff_applied_ms=backoff_ms
    )

    return state


def create_reasoning_graph(checkpointer=None) -> StateGraph:
    """
    Create the LangGraph workflow for Omni-Cortex reasoning.

    Graph structure:
    1. START -> route_node (AI selects framework)
    2. route_node -> should_continue (conditional)
    3. should_continue -> execute_framework_node (run selected framework)
       OR should_continue -> retry_with_backoff (if transient error)
    4. retry_with_backoff -> route_node (re-attempt routing)
    5. execute_framework_node -> END

    Retry behavior:
    - Up to MAX_RETRIES attempts with exponential backoff
    - Backoff starts at BASE_BACKOFF_MS and doubles each retry

    Returns compiled graph with optional memory checkpointing.
    """
    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("route", route_node)
    workflow.add_node("execute", execute_framework_node)
    workflow.add_node("retry", retry_with_backoff)

    # Add edges
    workflow.set_entry_point("route")
    workflow.add_conditional_edges(
        "route",
        should_continue,
        {
            "execute": "execute",
            "retry": "retry",
            "end": END
        }
    )
    workflow.add_edge("retry", "route")  # Retry loops back to routing
    workflow.add_edge("execute", END)

    # Compile with optional checkpointer
    compiled_graph = workflow.compile(checkpointer=checkpointer)

    return compiled_graph


# =============================================================================
# Checkpointer Lifecycle Management
# =============================================================================

_checkpointer: AsyncSqliteSaver | None = None
_checkpointer_lock = None  # Lazy init to avoid event loop issues


async def get_checkpointer() -> AsyncSqliteSaver:
    """
    Get async SQLite checkpointer singleton for LangGraph.

    This maintains a single checkpointer instance for the application lifetime.
    Call cleanup_checkpointer() on shutdown to properly close connections.
    """
    global _checkpointer, _checkpointer_lock
    import asyncio
    
    # Lazy init lock in async context
    if _checkpointer_lock is None:
        _checkpointer_lock = asyncio.Lock()
    
    if _checkpointer is not None:
        return _checkpointer
    
    async with _checkpointer_lock:
        if _checkpointer is None:
            # Ensure directory exists
            checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
            os.makedirs(checkpoint_dir, exist_ok=True)
            _checkpointer = await AsyncSqliteSaver.from_conn_string(CHECKPOINT_PATH)
            logger.info("checkpointer_initialized", path=CHECKPOINT_PATH)
    
    return _checkpointer


async def cleanup_checkpointer() -> None:
    """
    Clean up checkpointer resources on shutdown.
    
    Call this from your shutdown handler to properly close database connections.
    """
    global _checkpointer
    if _checkpointer is not None:
        try:
            if hasattr(_checkpointer, 'conn') and _checkpointer.conn:
                await _checkpointer.conn.close()
            logger.info("checkpointer_cleaned_up")
        except Exception as e:
            logger.warning("checkpointer_cleanup_error", error=str(e))
        finally:
            _checkpointer = None


async def get_graph_with_memory() -> StateGraph:
    """
    Get the reasoning graph with checkpointing enabled.
    
    Use this for operations that need memory persistence.
    """
    checkpointer = await get_checkpointer()
    return create_reasoning_graph(checkpointer=checkpointer)


# Create the global graph instance (without checkpointer for import-time initialization)
# For memory-enabled operations, use get_graph_with_memory() instead
graph = create_reasoning_graph()

