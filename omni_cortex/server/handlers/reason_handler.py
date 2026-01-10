"""
Reason Tool Handler

The smart routing handler that uses HyperRouter for framework selection.
Integrates with ContextGateway for automatic context preparation.

Now supports EXECUTION MODE: When execute=true, Gemini actually analyzes
the code and returns specific findings instead of just returning a template.
"""

import asyncio
import sys
import traceback

import structlog
from mcp.types import TextContent

from app.core.audit import log_tool_call
from app.core.context_gateway import get_context_gateway
from app.core.analysis_executor import execute_code_analysis, AnalysisResult
from ..framework_prompts import FRAMEWORKS
from .validation import (
    ValidationError,
    validate_query,
    validate_context,
    validate_thread_id,
)

logger = structlog.get_logger("omni-cortex.reason_handler")


async def handle_reason(
    arguments: dict,
    router,
) -> list[TextContent]:
    """
    Handle the 'reason' tool - smart routing with structured brief OR execution.

    Args:
        arguments: Tool arguments (query, context, thread_id, execute, workspace_path)
        router: HyperRouter instance for framework selection

    Modes:
        - Default (execute=false): Returns structured brief/template for Claude
        - Execution (execute=true): Gemini actually analyzes code and returns findings

    Returns:
        List with single TextContent containing either brief or analysis results
    """
    # Validate inputs
    try:
        query = validate_query(arguments.get("query"), required=True)
        context = validate_context(arguments.get("context"))
        thread_id = validate_thread_id(arguments.get("thread_id"), required=False)
        workspace_path = arguments.get("workspace_path")

        # Auto-detect execution mode based on query patterns
        # Code analysis queries should use Gemini to actually analyze code
        analysis_keywords = [
            "analyze", "audit", "review", "stability", "issues", "problems",
            "bugs", "errors", "security", "performance", "refactor", "improve"
        ]
        query_lower = query.lower() if query else ""
        is_analysis_query = any(kw in query_lower for kw in analysis_keywords)

        # Auto-enable execute mode for analysis queries or when workspace_path provided
        execute_mode = arguments.get("execute", False) or bool(workspace_path) or is_analysis_query

        logger.info(
            "reason_mode_selected",
            execute_mode=execute_mode,
            query_preview=query[:50] if query else "",
        )
    except ValidationError as e:
        log_tool_call(
            tool_name="reason",
            arguments=arguments,
            success=False,
            error=str(e)
        )
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    # EXECUTION MODE: Actually analyze code with Gemini
    if execute_mode:
        return await _handle_execute_mode(
            query=query,
            context=context,
            workspace_path=workspace_path,
            arguments=arguments,
            thread_id=thread_id,
            router=router,
        )

    # AUTO-CONTEXT: If no context provided, use ContextGateway to prepare rich context
    # This bridges the gap between prepare_context and reason tools
    context_was_prepared = False
    if not context or context == "None provided":
        try:
            gateway = get_context_gateway()
            structured_context = await gateway.prepare_context(
                query=query,
                workspace_path=arguments.get("workspace_path"),
                code_context=arguments.get("code_context"),
                file_list=arguments.get("file_list"),
                search_docs=True,
            )
            context = structured_context.to_claude_prompt()
            context_was_prepared = True
            logger.info("auto_context_prepared", query_preview=query[:50])
        except Exception as e:
            # Graceful degradation: proceed without auto-context if gateway fails
            logger.warning(
                "auto_context_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            context = None

    # Generate structured brief using the new protocol
    # Timeout for LLM calls to prevent indefinite hangs
    BRIEF_TIMEOUT_SECONDS = 30.0

    try:
        router_output = await asyncio.wait_for(
            router.generate_structured_brief(
                query=query,
                context=context,
                code_snippet=None,
                ide_context=None,
                file_list=None
            ),
            timeout=BRIEF_TIMEOUT_SECONDS
        )

        # Build output with pipeline metadata + Claude brief
        brief = router_output.claude_code_brief
        pipeline = router_output.pipeline
        gate = router_output.integrity_gate
        telemetry = router_output.telemetry

        # Compact header - single line metadata
        stages = "→".join([s.framework_id for s in pipeline.stages])
        signals = ",".join([s.type.value for s in router_output.detected_signals[:3]]) if router_output.detected_signals else ""

        output = f"[{stages}] conf={gate.confidence.score:.0%} risk={router_output.task_profile.risk_level.value}"
        if signals:
            output += f" signals={signals}"
        output += "\n\n"

        # The actual brief for Claude - compact but rich
        output += brief.to_compact_prompt()

        # Gate warning only if not proceeding
        if gate.recommendation.action.value != "PROCEED":
            output += f"\n\n⚠️ {gate.recommendation.action.value}: {gate.recommendation.notes}"

        # Audit log successful call
        log_tool_call(
            tool_name="reason",
            arguments=arguments,
            thread_id=thread_id,
            success=True
        )

    except asyncio.TimeoutError:
        # Graceful degradation: structured brief generation timed out (likely LLM hang).
        # Fall back to simple template mode to ensure user always gets a response.
        logger.warning(
            "structured_brief_timeout",
            timeout_seconds=BRIEF_TIMEOUT_SECONDS,
            query_preview=query[:50] if query else "",
            hint="LLM call timed out, falling back to simple template mode"
        )

        # Use the same fallback logic as generic exception handler below
        selected = router._vibe_matcher.check_vibe_dictionary(query)
        if not selected:
            selected = router._heuristic_select(query, context)
        if not selected or selected not in FRAMEWORKS:
            selected = "self_discover"

        fw_info = router.get_framework_info(selected)
        complexity = router.estimate_complexity(query, context if context != "None provided" else None)

        fw = FRAMEWORKS.get(selected, {
            "category": "unknown",
            "best_for": [],
            "prompt": "Apply your best reasoning to: {query}\n\nContext: {context}"
        })
        prompt = fw["prompt"].format(query=query, context=context or "None provided")

        output = f"# Framework: {selected}\n"
        output += f"Category: {fw_info.get('category', fw.get('category', 'unknown'))} | Complexity: {complexity:.2f}\n"
        output += f"Best for: {', '.join(fw_info.get('best_for', fw.get('best_for', [])))}\n"
        output += "\n---\n\n"
        output += prompt

        # Audit log timeout fallback
        log_tool_call(
            tool_name="reason",
            arguments=arguments,
            thread_id=thread_id,
            success=True,
            error=f"timeout_fallback: {BRIEF_TIMEOUT_SECONDS}s"
        )

    except Exception as e:
        # Graceful degradation: intentionally catching all exceptions to ensure
        # the user always receives a usable response. If structured brief generation
        # fails for any reason, we fall back to simple template mode rather than
        # returning an error to the MCP client.
        err_detail = traceback.format_exc()
        logger.warning(
            "Structured brief generation failed",
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=err_detail,
        )
        print(f"BRIEF FAILED: {e}\n{err_detail}", file=sys.stderr)

        selected = router._vibe_matcher.check_vibe_dictionary(query)
        if not selected:
            selected = router._heuristic_select(query, context)
        if not selected or selected not in FRAMEWORKS:
            selected = "self_discover"

        fw_info = router.get_framework_info(selected)
        complexity = router.estimate_complexity(query, context if context != "None provided" else None)

        fw = FRAMEWORKS.get(selected, {
            "category": "unknown",
            "best_for": [],
            "prompt": "Apply your best reasoning to: {query}\n\nContext: {context}"
        })
        prompt = fw["prompt"].format(query=query, context=context or "None provided")

        output = f"# Framework: {selected}\n"
        output += f"Category: {fw_info.get('category', fw.get('category', 'unknown'))} | Complexity: {complexity:.2f}\n"
        output += f"Best for: {', '.join(fw_info.get('best_for', fw.get('best_for', [])))}\n"
        output += "\n---\n\n"
        output += prompt

        # Audit log fallback (still successful, but with degraded mode)
        log_tool_call(
            tool_name="reason",
            arguments=arguments,
            thread_id=thread_id,
            success=True,
            error=f"fallback_mode: {type(e).__name__}"
        )

    return [TextContent(type="text", text=output)]


async def _handle_execute_mode(
    query: str,
    context: str | None,
    workspace_path: str | None,
    arguments: dict,
    thread_id: str | None,
    router,
) -> list[TextContent]:
    """
    Execute actual code analysis using Gemini.

    This is the key difference from the default mode:
    - Default: Returns templates/plans for Claude to follow
    - Execute: Gemini actually analyzes code and returns specific findings

    Args:
        query: Analysis query
        context: Pre-prepared context
        workspace_path: Path to workspace for file discovery
        arguments: Original tool arguments
        thread_id: Thread ID for memory
        router: HyperRouter instance

    Returns:
        TextContent with analysis results in markdown format
    """
    import os

    # Default to current working directory if no workspace provided
    if not workspace_path:
        workspace_path = os.getcwd()
        # If running in Docker, use /app as the default workspace
        if os.path.exists("/app/app"):
            workspace_path = "/app"

    try:
        # Determine framework to use for analysis
        framework = "chain_of_verification"  # Default for stability analysis
        try:
            chain, reasoning, category = await router.select_framework_chain(
                query, None, None
            )
            if chain:
                framework = chain[0]
        except Exception as e:
            logger.warning("framework_selection_failed_in_execute", error=str(e))

        # Execute the analysis
        logger.info(
            "executing_analysis",
            query=query[:100],
            workspace=workspace_path,
            framework=framework,
        )

        result = await execute_code_analysis(
            query=query,
            workspace_path=workspace_path,
            context=context,
            file_list=arguments.get("file_list"),
            framework=framework,
            max_files=15,  # Analyze up to 15 files
        )

        # Log success
        log_tool_call(
            tool_name="reason",
            arguments={**arguments, "execute": True},
            thread_id=thread_id,
            success=True,
        )

        # Return markdown-formatted results
        output = result.to_markdown()

        # Add metadata header
        header = f"**Gemini Analysis** | Framework: `{framework}` | Time: {result.execution_time_ms}ms\n\n"
        output = header + output

        return [TextContent(type="text", text=output)]

    except Exception as e:
        err_detail = traceback.format_exc()
        logger.error(
            "execute_mode_failed",
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=err_detail,
        )

        log_tool_call(
            tool_name="reason",
            arguments={**arguments, "execute": True},
            thread_id=thread_id,
            success=False,
            error=str(e),
        )

        return [TextContent(
            type="text",
            text=f"Analysis execution failed: {e}\n\nFallback to template mode - set execute=false to get framework guidance."
        )]
