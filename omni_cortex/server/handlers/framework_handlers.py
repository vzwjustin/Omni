"""
Framework Tool Handlers

Handles think_* tools for specific framework execution.
"""

import logging

from mcp.types import TextContent

from app.langchain_integration import get_memory, save_to_langchain_memory
from app.core.sampling import (
    SamplingNotSupportedError,
    call_llm_with_fallback,
    LANGCHAIN_LLM_ENABLED,
)
from app.orchestrators import FRAMEWORK_ORCHESTRATORS
from ..framework_prompts import FRAMEWORKS
from .validation import (
    ValidationError,
    validate_query,
    validate_context,
    validate_thread_id,
    validate_framework_name,
)

logger = logging.getLogger("omni-cortex")


async def handle_think_framework(
    fw_name: str,
    arguments: dict,
    sampler,
) -> list[TextContent]:
    """
    Handle think_* framework tools.

    Args:
        fw_name: Framework name (without 'think_' prefix)
        arguments: Tool arguments (query, context, thread_id)
        sampler: ClientSampler for MCP sampling

    Returns:
        List with single TextContent containing framework output
    """
    # Validate inputs
    try:
        fw_name = validate_framework_name(fw_name)
        query = validate_query(arguments.get("query"), required=True)
        context = validate_context(arguments.get("context"))
        thread_id = validate_thread_id(arguments.get("thread_id"), required=False)
    except ValidationError as e:
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]

    # Enhance context with memory if thread_id provided
    if thread_id:
        memory = await get_memory(thread_id)
        mem_context = memory.get_context()
        if mem_context.get("chat_history"):
            history_str = "\n".join(str(m) for m in mem_context["chat_history"][-5:])
            context = f"CONVERSATION HISTORY:\n{history_str}\n\n{context}"
        if mem_context.get("framework_history"):
            context = f"PREVIOUSLY USED FRAMEWORKS: {mem_context['framework_history'][-5:]}\n\n{context}"

    # Try orchestrator with MCP sampling first (if client supports it)
    if fw_name in FRAMEWORK_ORCHESTRATORS:
        try:
            orchestrator = FRAMEWORK_ORCHESTRATORS[fw_name]
            result = await orchestrator(sampler, query, context or "None provided")

            # Save to memory if thread_id provided
            if thread_id:
                await save_to_langchain_memory(
                    thread_id,
                    query,
                    result["final_answer"],
                    fw_name
                )

            # Return result with metadata
            output = f"# Framework: {fw_name}\n"
            if "metadata" in result:
                metadata = result["metadata"]
                output += f"**Metadata:** {metadata}\n\n"
            output += "---\n\n"
            output += result["final_answer"]

            return [TextContent(type="text", text=output)]

        except SamplingNotSupportedError as e:
            # Client doesn't support sampling, try LangChain fallback
            logger.info(f"Sampling not supported: {e}")

    # LangChain direct API fallback (if USE_LANGCHAIN_LLM=true)
    if LANGCHAIN_LLM_ENABLED and fw_name in FRAMEWORKS:
        try:
            fw = FRAMEWORKS[fw_name]
            prompt = fw["prompt"].format(query=query, context=context or "None provided")

            logger.info(f"Using LangChain LLM for {fw_name}")
            response = await call_llm_with_fallback(
                prompt=prompt,
                sampler=None,  # Skip sampling, go straight to LangChain
                max_tokens=4000,
                temperature=0.7
            )

            # Save to memory if thread_id provided
            if thread_id:
                await save_to_langchain_memory(thread_id, query, response, fw_name)

            output = f"# Framework: {fw_name} (via LangChain)\n"
            output += f"Category: {fw.get('category', 'unknown')}\n"
            output += f"Best for: {', '.join(fw.get('best_for', []))}\n\n"
            output += "---\n\n"
            output += response

            return [TextContent(type="text", text=output)]

        except Exception as e:
            logger.warning(f"LangChain fallback failed: {e}, using template mode")

    # Template mode: return structured prompt for client to execute
    if fw_name in FRAMEWORKS:
        fw = FRAMEWORKS[fw_name]
        prompt = fw["prompt"].format(query=query, context=context or "None provided")

        output = f"# Framework: {fw_name}\n"
        output += f"Category: {fw.get('category', 'unknown')}\n"
        output += f"Best for: {', '.join(fw.get('best_for', []))}\n\n"
        output += "---\n\n"
        output += prompt

        return [TextContent(type="text", text=output)]

    return [TextContent(type="text", text=f"Unknown framework: {fw_name}")]
