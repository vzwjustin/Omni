"""
LangChain Tools Integration for Framework Nodes

Makes LangChain tools available within reasoning framework nodes
for enhanced capabilities like code execution, documentation search, etc.

Refactored to include security checks for dangerous tools.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from ..core.settings import get_settings
from ..langchain_integration import AVAILABLE_TOOLS
from ..state import GraphState

logger = structlog.get_logger("langchain-tools")


async def call_langchain_tool(
    tool_name: str, tool_input: Any, state: GraphState | None = None
) -> Any:
    """
    Call a LangChain tool by name.

    Args:
        tool_name: Name of the tool to call
        tool_input: Input for the tool (str or dict depending on tool)
        state: Optional GraphState for context

    Returns:
        Tool output (type depends on tool - typically str or dict)
    """
    # Find the tool
    tool = next((t for t in AVAILABLE_TOOLS if t.name == tool_name), None)

    if not tool:
        logger.warning("tool_not_found", tool_name=tool_name)
        return f"Tool '{tool_name}' not found"

    # SECURITY CHECK: Sandbox enforcement for dangerous tools
    if tool_name == "execute_code":
        # Check environment setting
        get_settings()

        # STRICT ENFORCEMENT: Unless explicitly overridden by env var, BLOCK unsafe execution.
        # This prevents accidental RCE in production environments.
        allow_unsafe = os.environ.get("ALLOW_UNSAFE_CODE_EXECUTION", "false").lower() == "true"

        # In a real production system, we would check for a sandbox provider here.
        # For now, we block by default.
        if not allow_unsafe:
            logger.error(
                "security_violation_blocked",
                tool="execute_code",
                reason="Unsafe code execution blocked. Set ALLOW_UNSAFE_CODE_EXECUTION=true to override (DEV ONLY).",
            )
            return "SecurityError: Unsafe code execution is blocked in this environment. Use a sandboxed provider."

        # If allowed, log a critical warning
        logger.warning(
            "executing_code_locally_unsafe",
            tool="execute_code",
            input_preview=str(tool_input)[:100],
            hint="Ensure this is running in a containerized environment or use a sandbox provider.",
        )

    try:
        # Execute the tool
        result = await tool.ainvoke(tool_input)
        logger.info("tool_executed", tool_name=tool_name, success=True)
        return result
    except Exception as e:
        # Graceful degradation: Tool failures should not crash the reasoning pipeline.
        # Return error message instead so the framework can continue with partial results.
        logger.error(
            "tool_execution_failed", tool_name=tool_name, error=str(e), error_type=type(e).__name__
        )
        return f"Tool execution failed: {str(e)}"


def get_available_tools_for_framework(framework_name: str, state: GraphState) -> list[str]:
    """
    Get list of tool names available for a specific framework.

    Different frameworks might benefit from different tools:
    - Code frameworks: execute_code, search_documentation
    - Context frameworks: retrieve_context, search_documentation
    - Search frameworks: All tools

    Note: This is a sync function (no async operations needed).
    """
    # Map frameworks to recommended tools
    framework_tool_map = {
        "program_of_thoughts": ["execute_code"],
        "chain_of_verification": ["execute_code", "search_documentation"],
        "critic": ["search_documentation"],
        "chain_of_note": ["retrieve_context", "search_documentation"],
        "coala": ["retrieve_context"],
        "buffer_of_thoughts": ["retrieve_context"],
    }

    # Get recommended tools for this framework, or all if not specified
    recommended = framework_tool_map.get(framework_name, [t.name for t in AVAILABLE_TOOLS])

    return recommended


def format_tool_descriptions() -> str:
    """Format tool descriptions for LLM prompting."""
    descriptions = []
    for tool in AVAILABLE_TOOLS:
        descriptions.append(f"- **{tool.name}**: {tool.description}")

    # Add explicit instruction for learning
    descriptions.append("\n**IMPORTANT SYSTEM INSTRUCTION**:")
    descriptions.append(
        "When you have successfully solved a problem or generated a useful solution, you MUST use the `save_learning` tool to persist this knowledge for future reference. This is critical for the system's long-term improvement."
    )

    return "\n".join(descriptions)


# Export for easy import
__all__ = ["call_langchain_tool", "get_available_tools_for_framework", "format_tool_descriptions"]
