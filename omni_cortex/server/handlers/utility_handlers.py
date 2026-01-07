"""
Utility Tool Handlers

Handles memory, context, code execution, and utility tools.
"""

import json
import os
import structlog

from mcp.types import TextContent

from app.langchain_integration import get_memory, save_to_langchain_memory
from app.core.context_gateway import get_context_gateway
from app.core.context_utils import (
    count_tokens,
    compress_content,
    detect_truncation,
    analyze_claude_md,
    generate_claude_md_template,
    inject_rules,
    RULE_PRESETS,
)
from ..framework_prompts import FRAMEWORKS
from .validation import (
    ValidationError,
    validate_thread_id,
    validate_query,
    validate_text,
    validate_code,
    validate_path,
    validate_action,
    validate_file_list,
    validate_string_list,
    validate_boolean,
    validate_float,
    validate_category,
)

logger = structlog.get_logger("omni-cortex")


async def handle_list_frameworks(arguments: dict) -> list[TextContent]:
    """List all thinking frameworks by category."""
    output = f"# Omni-Cortex: {len(FRAMEWORKS)} Thinking Frameworks\n\n"
    for cat in ["strategy", "search", "iterative", "code", "context", "fast", "verification", "agent", "rag"]:
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
    return [TextContent(
        type="text",
        text=f"Recommended: `think_{rec}`\n\n{fw['description']}\nBest for: {', '.join(fw['best_for'])}"
    )]


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
        framework = validate_text(arguments.get("framework"), "framework", max_length=100, required=True)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    try:
        await save_to_langchain_memory(thread_id, query, answer, framework)
        return [TextContent(type="text", text="Context saved successfully")]
    except Exception as e:
        logger.error("save_context_failed", error=str(e))
        return [TextContent(type="text", text=f"Failed to save context: {str(e)}")]


async def handle_execute_code(arguments: dict) -> list[TextContent]:
    """Execute Python code in sandboxed environment."""
    from app.nodes.code.pot import _safe_execute

    # Validate inputs
    try:
        code = validate_code(arguments.get("code"))
        language = validate_text(arguments.get("language"), "language", max_length=50, required=False) or "python"
    except ValidationError as e:
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": f"Validation error: {str(e)}"
        }))]

    if language.lower() != "python":
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": f"Only python supported, got: {language}"
        }))]

    result = await _safe_execute(code)
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_health(arguments: dict, manager, lean_mode: bool) -> list[TextContent]:
    """Check server health and capabilities."""
    collections = list(manager.COLLECTIONS.keys())
    if lean_mode:
        exposed_tools = 8
        note = (
            "ULTRA-LEAN MODE: 8 tools exposed (prepare_context, reason, execute_code, health "
            "+ count_tokens, compress_content, detect_truncation, manage_claude_md). "
            "Gemini handles context prep, 62 frameworks available internally."
        )
    else:
        exposed_tools = len(FRAMEWORKS) + 19
        note = "FULL MODE: All 81 tools exposed (62 think_* + 19 utilities)"

    return [TextContent(type="text", text=json.dumps({
        "status": "healthy",
        "mode": "ultra-lean" if lean_mode else "full",
        "tools_exposed": exposed_tools,
        "frameworks_available": len(FRAMEWORKS),
        "gemini_context_gateway": lean_mode,
        "collections": collections,
        "memory_enabled": True,
        "rag_enabled": True,
        "note": note
    }, indent=2))]


async def handle_prepare_context(arguments: dict) -> list[TextContent]:
    """Gemini-powered context preparation."""
    valid_formats = ["prompt", "json"]

    # Validate inputs
    try:
        query = validate_query(arguments.get("query"), required=True)
        workspace_path = validate_path(arguments.get("workspace_path"), "workspace_path", required=False)
        code_context = validate_text(arguments.get("code_context"), "code_context", max_length=100000, required=False) or None
        file_list = validate_file_list(arguments.get("file_list"))
        search_docs = validate_boolean(arguments.get("search_docs"), "search_docs", default=True)
        output_format = validate_category(
            arguments.get("output_format") or "prompt",
            valid_formats,
            "output_format"
        )
    except ValidationError as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Validation error: {str(e)}"
        }, indent=2))]

    try:
        gateway = get_context_gateway()
        context = await gateway.prepare_context(
            query=query,
            workspace_path=workspace_path,
            code_context=code_context,
            file_list=file_list,
            search_docs=search_docs,
        )

        if output_format == "json":
            return [TextContent(type="text", text=json.dumps(context.to_dict(), indent=2))]
        else:
            output = "# Context Prepared by Gemini\n\n"
            output += context.to_claude_prompt()
            return [TextContent(type="text", text=output)]

    except Exception as e:
        logger.error(f"Context gateway failed: {e}")
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "hint": "Ensure GOOGLE_API_KEY is set for Gemini-powered context preparation"
        }, indent=2))]


async def handle_count_tokens(arguments: dict) -> list[TextContent]:
    """Count tokens in text."""
    # Validate inputs
    try:
        text = validate_text(arguments.get("text"), "text", max_length=500000, required=True)
    except ValidationError as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Validation error: {str(e)}"
        }))]

    tokens = count_tokens(text)
    return [TextContent(type="text", text=json.dumps({
        "tokens": tokens,
        "characters": len(text)
    }))]


async def handle_compress_content(arguments: dict) -> list[TextContent]:
    """Compress file content by removing comments/whitespace."""
    # Validate inputs
    try:
        content = validate_text(arguments.get("content"), "content", max_length=500000, required=True)
        target = validate_float(arguments.get("target_reduction"), "target_reduction", default=0.3, min_value=0.0, max_value=1.0)
    except ValidationError as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Validation error: {str(e)}"
        }, indent=2))]

    result = compress_content(content, target)
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_detect_truncation(arguments: dict) -> list[TextContent]:
    """Detect if text is truncated."""
    # Validate inputs
    try:
        text = validate_text(arguments.get("text"), "text", max_length=500000, required=True)
    except ValidationError as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Validation error: {str(e)}"
        }, indent=2))]

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
        return [TextContent(type="text", text=json.dumps({
            "error": f"Validation error: {str(e)}"
        }, indent=2))]

    if action == "analyze":
        try:
            directory = validate_path(arguments.get("directory"), "directory", required=False) or os.getcwd()
        except ValidationError as e:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Validation error: {str(e)}"
            }, indent=2))]
        result = analyze_claude_md(directory)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif action == "generate":
        try:
            project_type = validate_category(
                arguments.get("project_type") or "general",
                valid_project_types,
                "project_type"
            )
            rules = validate_string_list(arguments.get("rules"), "rules", max_items=100) or []
            presets = validate_string_list(arguments.get("presets"), "presets", max_items=20) or []
        except ValidationError as e:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Validation error: {str(e)}"
            }, indent=2))]
        result = generate_claude_md_template(project_type, rules, presets)
        return [TextContent(type="text", text=result)]

    elif action == "inject":
        try:
            existing = validate_text(arguments.get("existing_content"), "existing_content", max_length=100000, required=False) or ""
            rules = validate_string_list(arguments.get("rules"), "rules", max_items=100) or []
            presets = validate_string_list(arguments.get("presets"), "presets", max_items=20) or []
            section = validate_text(arguments.get("section"), "section", max_length=100, required=False) or "Rules"
        except ValidationError as e:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Validation error: {str(e)}"
            }, indent=2))]
        result = inject_rules(existing, rules, section, presets)
        return [TextContent(type="text", text=result)]

    elif action == "list_presets":
        result = {
            name: {"rules": rules, "count": len(rules)}
            for name, rules in RULE_PRESETS.items()
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return [TextContent(type="text", text=f"Unknown action: {action}")]
