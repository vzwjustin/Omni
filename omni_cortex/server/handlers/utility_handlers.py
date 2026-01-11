"""
Utility Tool Handlers

Handles memory, context, code execution, and utility tools.
"""

import asyncio
import json
import os

import structlog
from mcp.types import TextContent

from app.core.audit import log_tool_call
from app.core.context_gateway import get_context_gateway
from app.core.context_utils import (
    RULE_PRESETS,
    analyze_claude_md,
    compress_content,
    count_tokens,
    detect_truncation,
    generate_claude_md_template,
    inject_rules,
)
from app.langchain_integration import get_memory, save_to_langchain_memory

from ..framework_prompts import FRAMEWORKS
from .validation import (
    ValidationError,
    validate_action,
    validate_boolean,
    validate_category,
    validate_code,
    validate_file_list,
    validate_float,
    validate_int,
    validate_path,
    validate_query,
    validate_string_list,
    validate_text,
    validate_thread_id,
)

logger = structlog.get_logger("omni-cortex")


async def handle_list_frameworks(arguments: dict) -> list[TextContent]:
    """List all thinking frameworks by category."""
    output = f"# Omni-Cortex: {len(FRAMEWORKS)} Thinking Frameworks\n\n"
    for cat in [
        "strategy",
        "search",
        "iterative",
        "code",
        "context",
        "fast",
        "verification",
        "agent",
        "rag",
    ]:
        output += f"## {cat.upper()}\n"
        for n, fw in FRAMEWORKS.items():
            if fw["category"] == cat:
                output += f"- `think_{n}`: {fw['description']}\n"
        output += "\n"
    return [TextContent(type="text", text=output)]


async def handle_recommend(arguments: dict) -> list[TextContent]:
    """Get framework recommendation for a task."""
    # Validate inputs
    try:
        task = validate_text(arguments.get("task"), "task", max_length=10000, required=True)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    task = task.lower()
    if any(w in task for w in ["debug", "bug", "fix", "error", "issue"]):
        rec = "active_inference"
    elif any(w in task for w in ["security", "verify", "review", "audit"]):
        rec = "chain_of_verification"
    elif any(w in task for w in ["design", "architect", "plan", "system"]):
        rec = "reason_flux"
    elif any(w in task for w in ["refactor", "improve", "optimize"]):
        rec = "tree_of_thoughts"
    elif any(w in task for w in ["algorithm", "math", "data", "compute"]):
        rec = "program_of_thoughts"
    elif any(w in task for w in ["understand", "explore", "learn"]):
        rec = "chain_of_note"
    elif any(w in task for w in ["quick", "simple", "fast"]):
        rec = "system1"
    else:
        rec = "self_discover"

    fw = FRAMEWORKS[rec]
    return [
        TextContent(
            type="text",
            text=f"Recommended: `think_{rec}`\n\n{fw['description']}\nBest for: {', '.join(fw['best_for'])}",
        )
    ]


async def handle_get_context(arguments: dict) -> list[TextContent]:
    """Retrieve conversation history for a thread."""
    # Validate inputs
    try:
        thread_id = validate_thread_id(arguments.get("thread_id"), required=True)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    memory = await get_memory(thread_id)
    context = memory.get_context()
    return [TextContent(type="text", text=json.dumps(context, default=str, indent=2))]


async def handle_save_context(arguments: dict) -> list[TextContent]:
    """Save a query-answer exchange to memory."""
    # Validate inputs
    try:
        thread_id = validate_thread_id(arguments.get("thread_id"), required=True)
        query = validate_query(arguments.get("query"), required=True)
        answer = validate_text(arguments.get("answer"), "answer", max_length=100000, required=True)
        framework = validate_text(
            arguments.get("framework"), "framework", max_length=100, required=True
        )
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    try:
        await save_to_langchain_memory(thread_id, query, answer, framework)
        return [TextContent(type="text", text="Context saved successfully")]
    except Exception as e:
        # Catch-all for unexpected memory/storage failures (e.g., database connection issues,
        # serialization errors). We log the error type for debugging and return a user-friendly
        # message rather than crashing the handler.
        logger.error("save_context_failed", error=str(e), error_type=type(e).__name__)
        return [TextContent(type="text", text=f"Failed to save context: {str(e)}")]


async def handle_execute_code(arguments: dict) -> list[TextContent]:
    """Execute Python code in sandboxed environment."""
    import time

    from app.core.settings import get_settings
    from app.nodes.code.pot import _safe_execute

    # Code execution rate limiter state (sliding window)
    if not hasattr(handle_execute_code, "_executions"):
        handle_execute_code._executions = []

    settings = get_settings()
    max_rpm = settings.rate_limit_execute_rpm
    now = time.time()
    window_start = now - 60  # 1-minute sliding window

    # Clean old entries
    handle_execute_code._executions = [
        t for t in handle_execute_code._executions if t > window_start
    ]

    # Check rate limit
    if len(handle_execute_code._executions) >= max_rpm:
        wait_time = int(handle_execute_code._executions[0] - window_start) + 1
        logger.warning(
            "execute_code_rate_limited",
            executions_in_window=len(handle_execute_code._executions),
            limit=max_rpm,
            wait_seconds=wait_time,
        )
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": False,
                        "error": f"Rate limit exceeded: {max_rpm} executions/minute. Try again in {wait_time}s.",
                    }
                ),
            )
        ]

    # Record this execution
    handle_execute_code._executions.append(now)

    # Validate inputs
    try:
        code = validate_code(arguments.get("code"))
        language = (
            validate_text(arguments.get("language"), "language", max_length=50, required=False)
            or "python"
        )
    except ValidationError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps({"success": False, "error": f"Validation error: {str(e)}"}),
            )
        ]

    if language.lower() != "python":
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"success": False, "error": f"Only python supported, got: {language}"}
                ),
            )
        ]

    result = await _safe_execute(code)
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_health(arguments: dict, manager, lean_mode: bool) -> list[TextContent]:
    """Check server health and capabilities."""
    from app.core.context.context_cache import get_context_cache
    from app.core.settings import get_settings

    settings = get_settings()
    collections = list(manager.COLLECTIONS.keys())

    if lean_mode:
        exposed_tools = 10
        note = (
            "ULTRA-LEAN MODE: 10 tools exposed (prepare_context, prepare_context_streaming, "
            "context_cache_status, reason, execute_code, health + count_tokens, compress_content, "
            "detect_truncation, manage_claude_md). Gemini handles context prep, 62 frameworks available internally."
        )
    else:
        exposed_tools = len(FRAMEWORKS) + 21
        note = "FULL MODE: All 83 tools exposed (62 think_* + 21 utilities)"

    # Get cache status
    cache_status = "disabled"
    cache_entries = 0
    cache_hit_rate = 0.0
    try:
        if settings.enable_context_cache:
            cache = get_context_cache()
            stats = cache.get_statistics()
            cache_status = "enabled"
            cache_entries = stats["active_entries"]
            cache_hit_rate = stats["hit_rate"]
    except Exception as e:
        logger.warning("health_check_cache_status_failed", error=str(e))

    # Build enhanced health response
    health_data = {
        "status": "healthy",
        "mode": "ultra-lean" if lean_mode else "full",
        "tools_exposed": exposed_tools,
        "frameworks_available": len(FRAMEWORKS),
        "gemini_context_gateway": lean_mode,
        "collections": collections,
        "memory_enabled": True,
        "rag_enabled": True,
        "note": note,
        "enhancements": {
            "context_cache": {
                "enabled": settings.enable_context_cache,
                "status": cache_status,
                "active_entries": cache_entries,
                "hit_rate": cache_hit_rate,
            },
            "streaming_context": {
                "enabled": settings.enable_streaming_context,
            },
            "multi_repo_discovery": {
                "enabled": settings.enable_multi_repo_discovery,
                "max_repositories": settings.multi_repo_max_repositories,
            },
            "circuit_breaker": {
                "enabled": settings.enable_circuit_breaker,
                "failure_threshold": settings.circuit_breaker_failure_threshold,
            },
            "dynamic_token_budget": {
                "enabled": settings.enable_dynamic_token_budget,
            },
            "enhanced_metrics": {
                "enabled": settings.enable_enhanced_metrics,
                "prometheus": settings.enable_prometheus_metrics,
            },
        },
    }

    return [TextContent(type="text", text=json.dumps(health_data, indent=2))]


async def handle_prepare_context(arguments: dict) -> list[TextContent]:
    """Gemini-powered context preparation with enhanced features."""
    from app.core.settings import get_settings

    valid_formats = ["prompt", "json"]
    settings = get_settings()

    # Validate inputs
    try:
        query = validate_query(arguments.get("query"), required=True)
        workspace_path = validate_path(
            arguments.get("workspace_path"), "workspace_path", required=False
        )
        code_context = (
            validate_text(
                arguments.get("code_context"), "code_context", max_length=100000, required=False
            )
            or None
        )
        file_list = validate_file_list(arguments.get("file_list"))
        search_docs = validate_boolean(arguments.get("search_docs"), "search_docs", default=True)
        output_format = validate_category(
            arguments.get("output_format") or "prompt", valid_formats, "output_format"
        )
        # New enhanced options
        enable_cache = validate_boolean(
            arguments.get("enable_cache"), "enable_cache", default=settings.enable_context_cache
        )
        enable_multi_repo = validate_boolean(
            arguments.get("enable_multi_repo"),
            "enable_multi_repo",
            default=settings.enable_multi_repo_discovery,
        )
        enable_source_attribution = validate_boolean(
            arguments.get("enable_source_attribution"),
            "enable_source_attribution",
            default=settings.enable_source_attribution,
        )
    except ValidationError as e:
        log_tool_call(tool_name="prepare_context", arguments=arguments, success=False, error=str(e))
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
            )
        ]

    try:
        gateway = get_context_gateway()

        # Temporarily override settings if requested
        original_cache_setting = gateway._enable_cache
        try:
            gateway._enable_cache = enable_cache

            context = await gateway.prepare_context(
                query=query,
                workspace_path=workspace_path,
                code_context=code_context,
                file_list=file_list,
                search_docs=search_docs,
                enable_multi_repo=enable_multi_repo,
                enable_source_attribution=enable_source_attribution,
            )
        finally:
            # Restore original setting
            gateway._enable_cache = original_cache_setting

        # Audit log successful call
        log_tool_call(tool_name="prepare_context", arguments=arguments, success=True)

        if output_format == "json":
            result = context.to_dict()
            # Add cache metadata if available
            if hasattr(context, "cache_metadata") and context.cache_metadata:
                result["cache_metadata"] = {
                    "cache_hit": context.cache_metadata.cache_hit,
                    "cache_age_seconds": context.cache_metadata.cache_age.total_seconds(),
                    "is_stale_fallback": context.cache_metadata.is_stale_fallback,
                }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        else:
            output = "# Context Prepared by Gemini\n\n"

            # Add enhanced metadata section
            metadata_parts = []

            # Cache status
            if hasattr(context, "cache_metadata") and context.cache_metadata:
                if context.cache_metadata.cache_hit:
                    cache_age = int(context.cache_metadata.cache_age.total_seconds())
                    cache_status = f"Cache Hit ({cache_age}s old"
                    if context.cache_metadata.is_stale_fallback:
                        cache_status += ", stale fallback"
                    cache_status += ")"
                    metadata_parts.append(cache_status)

            # Quality metrics
            if hasattr(context, "quality_metrics") and context.quality_metrics:
                qm = context.quality_metrics
                quality_info = f"Quality: {qm.context_coverage_score:.0%} coverage, {qm.avg_file_relevance:.0%} file relevance"
                metadata_parts.append(quality_info)

            # Token budget usage
            if hasattr(context, "token_budget_usage") and context.token_budget_usage:
                tbu = context.token_budget_usage
                budget_info = f"Budget: {tbu.used_tokens}/{tbu.allocated_tokens} tokens ({tbu.utilization_percentage:.0f}%)"
                if not tbu.within_budget:
                    budget_info += " ⚠️"
                metadata_parts.append(budget_info)

            # Add metadata if we have any
            if metadata_parts:
                output += "*" + " | ".join(metadata_parts) + "*\n\n"

            output += context.to_claude_prompt()
            return [TextContent(type="text", text=output)]

    except Exception as e:
        # Catch-all for context gateway failures (e.g., API errors, network issues,
        # missing credentials). We log with error_type for diagnostics and provide
        # a hint about common configuration issues rather than exposing raw errors.
        logger.error("context_gateway_failed", error=str(e), error_type=type(e).__name__)
        log_tool_call(tool_name="prepare_context", arguments=arguments, success=False, error=str(e))
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": str(e),
                        "hint": "Ensure GOOGLE_API_KEY is set for Gemini-powered context preparation",
                    },
                    indent=2,
                ),
            )
        ]


async def handle_count_tokens(arguments: dict) -> list[TextContent]:
    """Count tokens in text."""
    # Validate inputs
    try:
        text = validate_text(arguments.get("text"), "text", max_length=500000, required=True)
    except ValidationError as e:
        return [TextContent(type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}))]

    tokens = count_tokens(text)
    return [TextContent(type="text", text=json.dumps({"tokens": tokens, "characters": len(text)}))]


async def handle_compress_content(arguments: dict) -> list[TextContent]:
    """Compress file content by removing comments/whitespace."""
    # Validate inputs
    try:
        content = validate_text(
            arguments.get("content"), "content", max_length=500000, required=True
        )
        target = validate_float(
            arguments.get("target_reduction"),
            "target_reduction",
            default=0.3,
            min_value=0.0,
            max_value=1.0,
        )
    except ValidationError as e:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
            )
        ]

    result = compress_content(content, target)
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_detect_truncation(arguments: dict) -> list[TextContent]:
    """Detect if text is truncated."""
    # Validate inputs
    try:
        text = validate_text(arguments.get("text"), "text", max_length=500000, required=True)
    except ValidationError as e:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
            )
        ]

    result = detect_truncation(text)
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_manage_claude_md(arguments: dict) -> list[TextContent]:
    """Analyze, generate, or inject rules into CLAUDE.md files."""
    valid_actions = ["analyze", "generate", "inject", "list_presets"]
    valid_project_types = ["general", "python", "typescript", "react", "rust"]

    # Validate action first
    try:
        action = validate_action(arguments.get("action") or "analyze", valid_actions)
    except ValidationError as e:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
            )
        ]

    if action == "analyze":
        try:
            directory = (
                validate_path(arguments.get("directory"), "directory", required=False)
                or os.getcwd()
            )
        except ValidationError as e:
            return [
                TextContent(
                    type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
                )
            ]
        result = analyze_claude_md(directory)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif action == "generate":
        try:
            project_type = validate_category(
                arguments.get("project_type") or "general", valid_project_types, "project_type"
            )
            rules = validate_string_list(arguments.get("rules"), "rules", max_items=100) or []
            presets = validate_string_list(arguments.get("presets"), "presets", max_items=20) or []
        except ValidationError as e:
            return [
                TextContent(
                    type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
                )
            ]
        result = generate_claude_md_template(project_type, rules, presets)
        return [TextContent(type="text", text=result)]

    elif action == "inject":
        try:
            existing = (
                validate_text(
                    arguments.get("existing_content"),
                    "existing_content",
                    max_length=100000,
                    required=False,
                )
                or ""
            )
            rules = validate_string_list(arguments.get("rules"), "rules", max_items=100) or []
            presets = validate_string_list(arguments.get("presets"), "presets", max_items=20) or []
            section = (
                validate_text(arguments.get("section"), "section", max_length=100, required=False)
                or "Rules"
            )
        except ValidationError as e:
            return [
                TextContent(
                    type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
                )
            ]
        result = inject_rules(existing, rules, section, presets)
        return [TextContent(type="text", text=result)]

    elif action == "list_presets":
        result = {
            name: {"rules": rules, "count": len(rules)} for name, rules in RULE_PRESETS.items()
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return [TextContent(type="text", text=f"Unknown action: {action}")]


async def handle_prepare_context_streaming(arguments: dict) -> list[TextContent]:
    """Gemini-powered context preparation with streaming progress."""
    from app.core.context.streaming_gateway import get_streaming_context_gateway

    valid_formats = ["prompt", "json"]

    # Validate inputs
    try:
        query = validate_query(arguments.get("query"), required=True)
        workspace_path = validate_path(
            arguments.get("workspace_path"), "workspace_path", required=False
        )
        code_context = (
            validate_text(
                arguments.get("code_context"), "code_context", max_length=100000, required=False
            )
            or None
        )
        file_list = validate_file_list(arguments.get("file_list"))
        search_docs = validate_boolean(arguments.get("search_docs"), "search_docs", default=True)
        output_format = validate_category(
            arguments.get("output_format") or "prompt", valid_formats, "output_format"
        )
    except ValidationError as e:
        log_tool_call(
            tool_name="prepare_context_streaming", arguments=arguments, success=False, error=str(e)
        )
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
            )
        ]

    try:
        gateway = get_streaming_context_gateway()

        # Collect progress events
        progress_events = []

        def progress_callback(event):
            """Collect progress events."""
            progress_events.append(
                {
                    "component": event.component,
                    "status": event.status.value,
                    "progress": event.progress,
                    "message": event.message,
                    "timestamp": event.timestamp.isoformat(),
                    "estimated_completion": event.estimated_completion,
                }
            )

        # Create cancellation token (not used in MCP, but required by API)
        cancellation_token = asyncio.Event()

        # Prepare context with streaming
        context = await gateway.prepare_context_streaming(
            query=query,
            workspace_path=workspace_path,
            code_context=code_context,
            file_list=file_list,
            search_docs=search_docs,
            progress_callback=progress_callback,
            cancellation_token=cancellation_token,
        )

        # Audit log successful call
        log_tool_call(tool_name="prepare_context_streaming", arguments=arguments, success=True)

        # Format output
        if output_format == "json":
            result = context.to_detailed_json()
            result["progress_events"] = progress_events
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        else:
            output = "# Context Prepared by Gemini (Streaming)\n\n"

            # Add progress summary
            output += "## Progress Summary\n"
            for event in progress_events:
                if event["status"] == "completed":
                    output += f"- ✅ {event['component']}: {event['message']}\n"
                elif event["status"] == "failed":
                    output += f"- ❌ {event['component']}: {event['message']}\n"
            output += "\n"

            # Add main context
            output += context.to_claude_prompt_enhanced()
            return [TextContent(type="text", text=output)]

    except Exception as e:
        logger.error("streaming_context_gateway_failed", error=str(e), error_type=type(e).__name__)
        log_tool_call(
            tool_name="prepare_context_streaming", arguments=arguments, success=False, error=str(e)
        )
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": str(e),
                        "hint": "Ensure GOOGLE_API_KEY is set and streaming is enabled",
                    },
                    indent=2,
                ),
            )
        ]


async def handle_context_cache_status(arguments: dict) -> list[TextContent]:
    """Get context cache status and statistics."""
    from app.core.context.context_cache import get_context_cache

    try:
        cache = get_context_cache()

        # Get cache statistics
        stats = cache.get_statistics()

        # Format output
        result = {
            "cache_enabled": cache._enabled,
            "total_entries": stats["total_entries"],
            "expired_entries": stats["expired_entries"],
            "active_entries": stats["active_entries"],
            "cache_size_mb": stats["cache_size_mb"],
            "max_size_mb": cache._max_size_mb,
            "hit_rate": stats["hit_rate"],
            "entries_by_type": stats["entries_by_type"],
            "oldest_entry_age_seconds": stats["oldest_entry_age_seconds"],
            "newest_entry_age_seconds": stats["newest_entry_age_seconds"],
            "ttl_settings": {
                "query_analysis": cache._ttl_settings.get("query_analysis", 3600),
                "file_discovery": cache._ttl_settings.get("file_discovery", 1800),
                "documentation": cache._ttl_settings.get("documentation", 86400),
            },
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error("cache_status_failed", error=str(e), error_type=type(e).__name__)
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": str(e), "hint": "Cache may not be initialized or enabled"}, indent=2
                ),
            )
        ]


# =============================================================================
# Token Reduction Tools (TOON + LLMLingua-2)
# =============================================================================


async def handle_serialize_to_toon(arguments: dict) -> list[TextContent]:
    """
    Serialize structured data to TOON format for token reduction.

    TOON (Token-Oriented Object Notation) reduces tokens by 20-60% compared to JSON,
    especially for arrays of uniform objects.

    Args:
        data: JSON string or structured data to serialize

    Returns:
        TOON-formatted string with comparison statistics
    """
    try:
        data_str = validate_text(arguments.get("data"), "data", max_length=500000, required=True)
    except ValidationError as e:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
            )
        ]

    try:
        from app.core.token_reduction import get_manager

        # Parse JSON input
        data = json.loads(data_str)

        # Get token reduction manager
        manager = get_manager()

        # Serialize to TOON
        toon_str = manager.serialize_to_toon(data)

        # Get comparison stats
        comparison = manager.get_format_comparison(data)

        result = {"toon_output": toon_str, "statistics": comparison, "success": True}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except json.JSONDecodeError as e:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Invalid JSON input: {str(e)}"}, indent=2)
            )
        ]
    except Exception as e:
        logger.error("toon_serialization_failed", error=str(e))
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


async def handle_deserialize_from_toon(arguments: dict) -> list[TextContent]:
    """
    Deserialize TOON format back to JSON.

    Args:
        toon_data: TOON-formatted string

    Returns:
        JSON representation of the data
    """
    try:
        toon_str = validate_text(
            arguments.get("toon_data"), "toon_data", max_length=500000, required=True
        )
    except ValidationError as e:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
            )
        ]

    try:
        from app.core.token_reduction import deserialize_from_toon

        # Deserialize from TOON
        data = deserialize_from_toon(toon_str)

        result = {"data": data, "success": True}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error("toon_deserialization_failed", error=str(e))
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


async def handle_compress_prompt(arguments: dict) -> list[TextContent]:
    """
    Compress prompt using LLMLingua-2 for 50-80% token reduction.

    Microsoft's LLMLingua-2 uses BERT-based token classification to intelligently
    compress prompts while preserving semantic meaning.

    Args:
        prompt: Prompt text to compress
        rate: Compression rate (0.1-0.9, default 0.5)
        min_tokens: Only compress if prompt exceeds this token count (default from settings)

    Returns:
        Compressed prompt with statistics
    """
    try:
        prompt = validate_text(arguments.get("prompt"), "prompt", max_length=500000, required=True)
        rate = validate_float(
            arguments.get("rate"), "rate", default=0.5, min_value=0.1, max_value=0.9
        )
        min_tokens = validate_int(
            arguments.get("min_tokens"), "min_tokens", default=None, min_value=1000
        )
    except ValidationError as e:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
            )
        ]

    try:
        from app.core.token_reduction import get_manager, get_reduction_stats

        manager = get_manager()
        result = manager.compress_prompt(prompt, rate=rate, min_tokens=min_tokens)

        # Add stats if compression succeeded
        if result.get("compressed", False):
            stats = get_reduction_stats(
                original=prompt, reduced=result.get("compressed_prompt", prompt), method="llmlingua"
            )
            result["statistics"] = stats

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error("prompt_compression_failed", error=str(e))
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": str(e),
                        "hint": "LLMLingua may not be installed or initialized. Run: pip install llmlingua",
                    },
                    indent=2,
                ),
            )
        ]


async def handle_compress_context(arguments: dict) -> list[TextContent]:
    """
    Compress context while preserving instruction (useful for RAG).

    This is ideal for scenarios where you have a user instruction and retrieved context.
    The instruction is preserved while the context is compressed.

    Args:
        instruction: User instruction to preserve
        context: Context to compress
        rate: Compression rate (0.1-0.9, default 0.5)

    Returns:
        Compressed result with preserved instruction
    """
    try:
        instruction = validate_text(
            arguments.get("instruction"), "instruction", max_length=50000, required=True
        )
        context = validate_text(
            arguments.get("context"), "context", max_length=500000, required=True
        )
        rate = validate_float(
            arguments.get("rate"), "rate", default=0.5, min_value=0.1, max_value=0.9
        )
    except ValidationError as e:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
            )
        ]

    try:
        from app.core.token_reduction import get_manager

        manager = get_manager()
        result = manager.compress_context(instruction, context, rate=rate)

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error("context_compression_failed", error=str(e))
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": str(e),
                        "hint": "LLMLingua may not be installed. Run: pip install llmlingua",
                    },
                    indent=2,
                ),
            )
        ]


async def handle_token_reduction_compare(arguments: dict) -> list[TextContent]:
    """
    Compare different token reduction strategies.

    Args:
        content: Content to analyze (text or JSON string)

    Returns:
        Comparison of JSON, TOON, and LLMLingua compression methods
    """
    try:
        content = validate_text(
            arguments.get("content"), "content", max_length=500000, required=True
        )
    except ValidationError as e:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Validation error: {str(e)}"}, indent=2)
            )
        ]

    try:
        from app.core.context_utils import count_tokens
        from app.core.token_reduction import get_manager

        manager = get_manager()
        result = {
            "original": {
                "content": content[:200] + "..." if len(content) > 200 else content,
                "tokens": count_tokens(content),
                "chars": len(content),
            },
            "methods": {},
        }

        # Try TOON serialization
        try:
            data = json.loads(content)
            if isinstance(data, (dict, list)):
                comparison = manager.get_format_comparison(data)
                result["methods"]["toon"] = {
                    "applicable": True,
                    "tokens": comparison["toon"]["tokens"],
                    "chars": comparison["toon"]["chars"],
                    "reduction_percent": comparison["savings"]["reduction_percent"],
                    "recommendation": comparison["recommendation"],
                }
        except json.JSONDecodeError:
            result["methods"]["toon"] = {"applicable": False, "reason": "Content is not valid JSON"}

        # Try LLMLingua compression
        try:
            compress_result = manager.compress_prompt(content, rate=0.5)
            if compress_result.get("compressed", False):
                result["methods"]["llmlingua"] = {
                    "applicable": True,
                    "compressed_tokens": compress_result.get("compressed_tokens", -1),
                    "original_tokens": compress_result.get("origin_tokens", -1),
                    "ratio": compress_result.get("ratio", 1.0),
                    "reduction_percent": (1 - compress_result.get("ratio", 1.0)) * 100,
                }
            else:
                result["methods"]["llmlingua"] = {
                    "applicable": False,
                    "reason": compress_result.get("reason", "Compression not available"),
                }
        except Exception as e:
            result["methods"]["llmlingua"] = {"applicable": False, "reason": str(e)}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error("token_reduction_compare_failed", error=str(e))
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
