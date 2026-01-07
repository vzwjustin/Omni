"""
Omni-Cortex MCP Server

Exposes 60 thinking framework tools + utility tools.
The calling LLM uses these tools and does the reasoning.
LangGraph orchestrates, LangChain handles memory/RAG.
"""

import asyncio
import logging
import sys
from typing import Any

import structlog

# CRITICAL: Configure ALL logging to stderr BEFORE any other imports
# MCP uses stdio - stdout is for JSON-RPC, stderr for logs
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(levelname)s:%(name)s:%(message)s"
)

# Import correlation ID utilities
from app.core.correlation import set_correlation_id, clear_correlation_id
from app.core.logging import add_correlation_id

# Configure structlog to use stdlib logging (required for ChromaDB/LangChain compatibility)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_correlation_id,  # Add correlation ID to all log events
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import from graph for orchestration
from app.graph import FRAMEWORK_NODES, router
from app.collection_manager import get_collection_manager
from app.core.vibe_dictionary import VIBE_DICTIONARY

# Import framework prompts
from .framework_prompts import FRAMEWORKS

# Import handlers
from .handlers import (
    handle_reason,
    handle_think_framework,
    handle_search_documentation,
    handle_search_frameworks_by_name,
    handle_search_by_category,
    handle_search_function,
    handle_search_class,
    handle_search_docs_only,
    handle_search_framework_category,
    handle_list_frameworks,
    handle_recommend,
    handle_get_context,
    handle_save_context,
    handle_execute_code,
    handle_health,
    handle_prepare_context,
    handle_count_tokens,
    handle_compress_content,
    handle_detect_truncation,
    handle_manage_claude_md,
)

# Import MCP sampling and settings
from app.core.sampling import ClientSampler
from app.core.settings import get_settings

# LEAN_MODE: Only expose essential tools to reduce MCP token overhead
LEAN_MODE = get_settings().lean_mode

logger = structlog.get_logger("omni-cortex")


def create_server() -> Server:
    """Create the MCP server with all tools."""
    server = Server("omni-cortex")

    # Initialize client sampler for multi-turn orchestration
    sampler = ClientSampler(server)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        tools = []

        if LEAN_MODE:
            # ULTRA-LEAN: 8 tools - Gemini does the heavy lifting
            tools.append(Tool(
                name="prepare_context",
                description="Gemini 3 Flash prepares rich, structured context for Claude. Analyzes query, discovers relevant files (with relevance scoring), fetches documentation, generates execution plan. Returns organized brief so Claude can focus on deep reasoning instead of searching.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The task or problem to prepare context for"},
                        "workspace_path": {"type": "string", "description": "Path to workspace/project directory"},
                        "code_context": {"type": "string", "description": "Any code snippets to consider"},
                        "file_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Pre-specified files to analyze"
                        },
                        "search_docs": {"type": "boolean", "description": "Search web for documentation (default: true)"},
                        "output_format": {"type": "string", "enum": ["prompt", "json"], "description": "Output format (default: prompt)"}
                    },
                    "required": ["query"]
                }
            ))

            tools.append(Tool(
                name="reason",
                description="Execute reasoning with auto-selected framework. Uses 62 thinking frameworks internally. Pass context from prepare_context for best results.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Your task or question"},
                        "context": {"type": "string", "description": "Context from prepare_context or code snippets"},
                        "thread_id": {"type": "string", "description": "Thread ID for memory persistence"}
                    },
                    "required": ["query"]
                }
            ))

            tools.append(Tool(
                name="execute_code",
                description="Execute Python code in sandboxed environment. Use for testing, validation, and verification.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "language": {"type": "string", "description": "Language (only 'python' supported)"}
                    },
                    "required": ["code"]
                }
            ))

            tools.append(Tool(
                name="health",
                description="Check server health, available frameworks, and capabilities",
                inputSchema={"type": "object", "properties": {}}
            ))

            # Context optimization tools
            tools.append(Tool(
                name="count_tokens",
                description="Count tokens in text using Claude's tokenizer",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "Text to count tokens for"}},
                    "required": ["text"]
                }
            ))

            tools.append(Tool(
                name="compress_content",
                description="Compress file content by removing comments/whitespace. Achieves 30-70% token reduction.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to compress"},
                        "target_reduction": {"type": "number", "description": "Target reduction 0.0-1.0 (default: 0.3)"}
                    },
                    "required": ["content"]
                }
            ))

            tools.append(Tool(
                name="detect_truncation",
                description="Detect if text is truncated (unclosed blocks, incomplete sentences)",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "Text to check"}},
                    "required": ["text"]
                }
            ))

            tools.append(Tool(
                name="manage_claude_md",
                description="Analyze, generate, or inject rules into CLAUDE.md files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["analyze", "generate", "inject", "list_presets"], "description": "Action to perform"},
                        "directory": {"type": "string", "description": "Project directory (for analyze)"},
                        "project_type": {"type": "string", "enum": ["general", "python", "typescript", "react", "rust"], "description": "Project type (for generate)"},
                        "rules": {"type": "array", "items": {"type": "string"}, "description": "Custom rules to add"},
                        "presets": {"type": "array", "items": {"type": "string"}, "description": "Presets: security, performance, testing, documentation, code_quality, git, context_optimization"},
                        "existing_content": {"type": "string", "description": "Existing CLAUDE.md content (for inject)"},
                        "section": {"type": "string", "description": "Section name for injection (default: Rules)"}
                    },
                    "required": ["action"]
                }
            ))

            return tools

        # FULL MODE: All tools exposed
        # 62 Framework tools (think_*)
        for name, fw in FRAMEWORKS.items():
            vibes = VIBE_DICTIONARY.get(name, [])[:4]
            vibe_str = f" Vibes: {', '.join(vibes)}" if vibes else ""

            tools.append(Tool(
                name=f"think_{name}",
                description=f"[{fw['category'].upper()}] {fw['description']}. Best for: {', '.join(fw['best_for'])}.{vibe_str}",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Your task or problem"},
                        "context": {"type": "string", "description": "Code snippet or additional context"},
                        "thread_id": {"type": "string", "description": "Thread ID for memory persistence across turns"}
                    },
                    "required": ["query"]
                }
            ))

        # Smart routing tool
        tools.append(Tool(
            name="reason",
            description="Smart reasoning: auto-selects best framework based on task analysis, returns structured prompt with memory context",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Your task or question"},
                    "context": {"type": "string", "description": "Code or additional context"},
                    "thread_id": {"type": "string", "description": "Thread ID for memory persistence"}
                },
                "required": ["query"]
            }
        ))

        # Framework discovery tools
        tools.append(Tool(
            name="list_frameworks",
            description="List all 62 thinking frameworks by category",
            inputSchema={"type": "object", "properties": {}}
        ))

        tools.append(Tool(
            name="recommend",
            description="Get framework recommendation for your task",
            inputSchema={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"]
            }
        ))

        # Memory tools
        tools.append(Tool(
            name="get_context",
            description="Retrieve conversation history and framework usage for a thread",
            inputSchema={
                "type": "object",
                "properties": {"thread_id": {"type": "string", "description": "Thread ID to get context for"}},
                "required": ["thread_id"]
            }
        ))

        tools.append(Tool(
            name="save_context",
            description="Save a query-answer exchange to memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "thread_id": {"type": "string"},
                    "query": {"type": "string"},
                    "answer": {"type": "string"},
                    "framework": {"type": "string"}
                },
                "required": ["thread_id", "query", "answer", "framework"]
            }
        ))

        # RAG/Search tools
        tools.append(Tool(
            name="search_documentation",
            description="Search indexed documentation and code via vector store (RAG)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {"type": "integer", "description": "Number of results (default: 5)"}
                },
                "required": ["query"]
            }
        ))

        tools.append(Tool(
            name="search_frameworks_by_name",
            description="Search within a specific framework's implementation",
            inputSchema={
                "type": "object",
                "properties": {
                    "framework_name": {"type": "string", "description": "Framework to search (e.g., 'active_inference')"},
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "Number of results"}
                },
                "required": ["framework_name", "query"]
            }
        ))

        tools.append(Tool(
            name="search_by_category",
            description="Search within a code category: framework, documentation, config, utility, test, integration",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string", "description": "One of: framework, documentation, config, utility, test, integration"},
                    "k": {"type": "integer"}
                },
                "required": ["query", "category"]
            }
        ))

        tools.append(Tool(
            name="search_function",
            description="Find specific function implementations by name",
            inputSchema={
                "type": "object",
                "properties": {"function_name": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["function_name"]
            }
        ))

        tools.append(Tool(
            name="search_class",
            description="Find specific class implementations by name",
            inputSchema={
                "type": "object",
                "properties": {"class_name": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["class_name"]
            }
        ))

        tools.append(Tool(
            name="search_docs_only",
            description="Search only markdown documentation files",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["query"]
            }
        ))

        tools.append(Tool(
            name="search_framework_category",
            description="Search within a framework category: strategy, search, iterative, code, context, fast",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "framework_category": {"type": "string", "description": "One of: strategy, search, iterative, code, context, fast"},
                    "k": {"type": "integer"}
                },
                "required": ["query", "framework_category"]
            }
        ))

        # Code execution tool
        tools.append(Tool(
            name="execute_code",
            description="Execute Python code in a sandboxed environment",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "language": {"type": "string", "description": "Language (only 'python' supported)"}
                },
                "required": ["code"]
            }
        ))

        # Health check
        tools.append(Tool(
            name="health",
            description="Check server health and available capabilities",
            inputSchema={"type": "object", "properties": {}}
        ))

        # Context Gateway
        tools.append(Tool(
            name="prepare_context",
            description="Gemini prepares rich, structured context for Claude. Does the heavy lifting: analyzes query, discovers relevant files, fetches documentation, ranks by relevance. Returns organized context packet so Claude can focus on deep reasoning instead of egg-hunting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The task or problem to prepare context for"},
                    "workspace_path": {"type": "string", "description": "Path to workspace/project directory"},
                    "code_context": {"type": "string", "description": "Any code snippets to consider"},
                    "file_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Pre-specified files to analyze"
                    },
                    "search_docs": {"type": "boolean", "description": "Whether to search web for documentation (default: true)"},
                    "output_format": {"type": "string", "enum": ["prompt", "json"], "description": "Output as Claude prompt or raw JSON (default: prompt)"}
                },
                "required": ["query"]
            }
        ))

        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        # Set a new correlation ID for this request
        import uuid
        correlation_id = str(uuid.uuid4())[:8]
        set_correlation_id(correlation_id)

        try:
            logger.info("call_tool_start", tool=name, args=list(arguments.keys()))

            manager = get_collection_manager()

            # Route to handlers
            if name == "reason":
                return await handle_reason(arguments, router)

            if name.startswith("think_"):
                fw_name = name[6:]
                return await handle_think_framework(fw_name, arguments, sampler)

            if name == "list_frameworks":
                return await handle_list_frameworks(arguments)

            if name == "recommend":
                return await handle_recommend(arguments)

            if name == "get_context":
                return await handle_get_context(arguments)

            if name == "save_context":
                return await handle_save_context(arguments)

            if name == "search_documentation":
                return await handle_search_documentation(arguments, manager)

            if name == "search_frameworks_by_name":
                return await handle_search_frameworks_by_name(arguments, manager)

            if name == "search_by_category":
                return await handle_search_by_category(arguments, manager)

            if name == "search_function":
                return await handle_search_function(arguments, manager)

            if name == "search_class":
                return await handle_search_class(arguments, manager)

            if name == "search_docs_only":
                return await handle_search_docs_only(arguments, manager)

            if name == "search_framework_category":
                return await handle_search_framework_category(arguments, manager)

            if name == "execute_code":
                return await handle_execute_code(arguments)

            if name == "health":
                return await handle_health(arguments, manager, LEAN_MODE)

            if name == "prepare_context":
                return await handle_prepare_context(arguments)

            if name == "count_tokens":
                return await handle_count_tokens(arguments)

            if name == "compress_content":
                return await handle_compress_content(arguments)

            if name == "detect_truncation":
                return await handle_detect_truncation(arguments)

            if name == "manage_claude_md":
                return await handle_manage_claude_md(arguments)

            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        finally:
            # Clear correlation ID after request completes
            clear_correlation_id()
            logger.info("call_tool_end", tool=name)

    return server


async def main():
    logger.info("=" * 60)
    logger.info("Omni-Cortex MCP - Operating System for Vibe Coders")
    logger.info("=" * 60)
    logger.info(f"Frameworks: {len(FRAMEWORKS)} thinking frameworks (internal)")
    logger.info(f"Graph nodes: {len(FRAMEWORK_NODES)} LangGraph nodes")
    logger.info("Memory: LangChain ConversationBufferMemory")
    logger.info("RAG: ChromaDB with 6 collections")
    if LEAN_MODE:
        logger.info("Mode: ULTRA-LEAN (8 tools)")
        logger.info("  Core: prepare_context, reason, execute_code, health")
        logger.info("  Context: count_tokens, compress_content, detect_truncation, manage_claude_md")
        logger.info("  Gemini 3 Flash: Context prep, file discovery, doc search")
        logger.info("  62 frameworks available internally via HyperRouter")
        logger.info("  Set LEAN_MODE=false for full 81-tool access")
    else:
        logger.info(f"Mode: FULL ({len(FRAMEWORKS) + 19} tools)")
        logger.info(f"  {len(FRAMEWORKS)} think_* tools + 19 utilities")
    logger.info("=" * 60)

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
