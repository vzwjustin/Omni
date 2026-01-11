"""
Omni-Cortex MCP Server

Exposes 60 thinking framework tools + utility tools.
The calling LLM uses these tools and does the reasoning.
LangGraph orchestrates, LangChain handles memory/RAG.
"""

import asyncio
import logging
import signal
import sys
from typing import Any

import structlog

# CRITICAL: Configure ALL logging to stderr BEFORE any other imports
# MCP uses stdio - stdout is for JSON-RPC, stderr for logs
logging.basicConfig(
    level=logging.INFO, stream=sys.stderr, format="%(levelname)s:%(name)s:%(message)s"
)

# Import correlation ID utilities
from app.core.correlation import clear_correlation_id, set_correlation_id
from app.core.logging import add_correlation_id

# Import settings early to determine logging mode
from app.core.settings import get_settings

_settings = get_settings()

# Choose renderer based on production mode
if _settings.production_logging:
    _log_renderer = structlog.processors.JSONRenderer()
else:
    _log_renderer = structlog.dev.ConsoleRenderer()

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
        _log_renderer,
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

import contextlib

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from app.collection_manager import get_collection_manager

# Import MCP sampling and settings
from app.core.sampling import ClientSampler
from app.core.settings import get_settings
from app.core.vibe_dictionary import VIBE_DICTIONARY

# Import from graph for orchestration
from app.graph import FRAMEWORK_NODES, router

# Import framework prompts
from .framework_prompts import FRAMEWORKS

# Import handlers
from .handlers import (
    handle_compress_content,
    handle_compress_context,
    handle_compress_prompt,
    handle_context_cache_status,
    handle_count_tokens,
    handle_deserialize_from_toon,
    handle_detect_truncation,
    handle_execute_code,
    handle_get_context,
    handle_health,
    handle_list_frameworks,
    handle_manage_claude_md,
    handle_prepare_context,
    handle_prepare_context_streaming,
    handle_reason,
    handle_recommend,
    handle_save_context,
    handle_search_by_category,
    handle_search_class,
    handle_search_docs_only,
    handle_search_documentation,
    handle_search_framework_category,
    handle_search_frameworks_by_name,
    handle_search_function,
    # Token reduction tools
    handle_serialize_to_toon,
    handle_think_framework,
    handle_token_reduction_compare,
)

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
            # ULTRA-LEAN: 4 tools only - Claude talks to Gemini, Gemini does all preprocessing
            # All compression, token counting, TOON, etc. happens internally in prepare_context
            tools.append(
                Tool(
                    name="prepare_context",
                    description="Gemini 3 Flash prepares rich, structured context for Claude. Analyzes query, discovers relevant files (with relevance scoring), fetches documentation, generates execution plan. All compression and token optimization happens internally. Returns organized brief so Claude can focus on deep reasoning.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The task or problem to prepare context for",
                            },
                            "workspace_path": {
                                "type": "string",
                                "description": "Path to workspace/project directory",
                            },
                            "code_context": {
                                "type": "string",
                                "description": "Any code snippets to consider",
                            },
                            "file_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Pre-specified files to analyze",
                            },
                            "search_docs": {
                                "type": "boolean",
                                "description": "Search web for documentation (default: true)",
                            },
                            "output_format": {
                                "type": "string",
                                "enum": ["prompt", "json"],
                                "description": "Output format (default: prompt)",
                            },
                            "streaming": {
                                "type": "boolean",
                                "description": "Enable streaming progress (default: false)",
                            },
                        },
                        "required": ["query"],
                    },
                )
            )

            tools.append(
                Tool(
                    name="reason",
                    description="Execute reasoning with auto-selected framework. Set execute=true for Gemini to actually analyze code and return specific findings. Uses 62 thinking frameworks internally.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Your task or question"},
                            "context": {
                                "type": "string",
                                "description": "Context from prepare_context or code snippets",
                            },
                            "thread_id": {
                                "type": "string",
                                "description": "Thread ID for memory persistence",
                            },
                            "execute": {
                                "type": "boolean",
                                "description": "If true, Gemini executes actual analysis and returns specific findings. If false (default), returns framework template.",
                            },
                            "workspace_path": {
                                "type": "string",
                                "description": "Path to workspace for file discovery (used with execute=true)",
                            },
                            "file_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific files to analyze (optional, used with execute=true)",
                            },
                        },
                        "required": ["query"],
                    },
                )
            )

            tools.append(
                Tool(
                    name="execute_code",
                    description="Execute Python code in sandboxed environment. Use for testing and validation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute"},
                            "language": {
                                "type": "string",
                                "description": "Language (only 'python' supported)",
                            },
                        },
                        "required": ["code"],
                    },
                )
            )

            tools.append(
                Tool(
                    name="health",
                    description="Check server health, available frameworks, and capabilities",
                    inputSchema={"type": "object", "properties": {}},
                )
            )

            return tools

        # FULL MODE: All tools exposed
        # 62 Framework tools (think_*)
        for name, fw in FRAMEWORKS.items():
            vibes = VIBE_DICTIONARY.get(name, [])[:4]
            vibe_str = f" Vibes: {', '.join(vibes)}" if vibes else ""

            tools.append(
                Tool(
                    name=f"think_{name}",
                    description=f"[{fw['category'].upper()}] {fw['description']}. Best for: {', '.join(fw['best_for'])}.{vibe_str}",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Your task or problem"},
                            "context": {
                                "type": "string",
                                "description": "Code snippet or additional context",
                            },
                            "thread_id": {
                                "type": "string",
                                "description": "Thread ID for memory persistence across turns",
                            },
                        },
                        "required": ["query"],
                    },
                )
            )

        # Smart routing tool with execution mode
        tools.append(
            Tool(
                name="reason",
                description="Smart reasoning: auto-selects best framework. Set execute=true for Gemini to actually analyze code and return specific findings with file:line locations.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Your task or question"},
                        "context": {"type": "string", "description": "Code or additional context"},
                        "thread_id": {
                            "type": "string",
                            "description": "Thread ID for memory persistence",
                        },
                        "execute": {
                            "type": "boolean",
                            "description": "If true, Gemini executes actual code analysis. If false (default), returns framework guidance.",
                        },
                        "workspace_path": {
                            "type": "string",
                            "description": "Path to workspace for auto file discovery (used with execute=true)",
                        },
                        "file_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific files to analyze (optional, used with execute=true)",
                        },
                    },
                    "required": ["query"],
                },
            )
        )

        # Framework discovery tools
        tools.append(
            Tool(
                name="list_frameworks",
                description="List all 62 thinking frameworks by category",
                inputSchema={"type": "object", "properties": {}},
            )
        )

        tools.append(
            Tool(
                name="recommend",
                description="Get framework recommendation for your task",
                inputSchema={
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                    "required": ["task"],
                },
            )
        )

        # Memory tools
        tools.append(
            Tool(
                name="get_context",
                description="Retrieve conversation history and framework usage for a thread",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "thread_id": {
                            "type": "string",
                            "description": "Thread ID to get context for",
                        }
                    },
                    "required": ["thread_id"],
                },
            )
        )

        tools.append(
            Tool(
                name="save_context",
                description="Save a query-answer exchange to memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "thread_id": {"type": "string"},
                        "query": {"type": "string"},
                        "answer": {"type": "string"},
                        "framework": {"type": "string"},
                    },
                    "required": ["thread_id", "query", "answer", "framework"],
                },
            )
        )

        # RAG/Search tools
        tools.append(
            Tool(
                name="search_documentation",
                description="Search indexed documentation and code via vector store (RAG)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "k": {"type": "integer", "description": "Number of results (default: 5)"},
                    },
                    "required": ["query"],
                },
            )
        )

        tools.append(
            Tool(
                name="search_frameworks_by_name",
                description="Search within a specific framework's implementation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "framework_name": {
                            "type": "string",
                            "description": "Framework to search (e.g., 'active_inference')",
                        },
                        "query": {"type": "string"},
                        "k": {"type": "integer", "description": "Number of results"},
                    },
                    "required": ["framework_name", "query"],
                },
            )
        )

        tools.append(
            Tool(
                name="search_by_category",
                description="Search within a code category: framework, documentation, config, utility, test, integration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "category": {
                            "type": "string",
                            "description": "One of: framework, documentation, config, utility, test, integration",
                        },
                        "k": {"type": "integer"},
                    },
                    "required": ["query", "category"],
                },
            )
        )

        tools.append(
            Tool(
                name="search_function",
                description="Find specific function implementations by name",
                inputSchema={
                    "type": "object",
                    "properties": {"function_name": {"type": "string"}, "k": {"type": "integer"}},
                    "required": ["function_name"],
                },
            )
        )

        tools.append(
            Tool(
                name="search_class",
                description="Find specific class implementations by name",
                inputSchema={
                    "type": "object",
                    "properties": {"class_name": {"type": "string"}, "k": {"type": "integer"}},
                    "required": ["class_name"],
                },
            )
        )

        tools.append(
            Tool(
                name="search_docs_only",
                description="Search only markdown documentation files",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                    "required": ["query"],
                },
            )
        )

        tools.append(
            Tool(
                name="search_framework_category",
                description="Search within a framework category: strategy, search, iterative, code, context, fast",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "framework_category": {
                            "type": "string",
                            "description": "One of: strategy, search, iterative, code, context, fast",
                        },
                        "k": {"type": "integer"},
                    },
                    "required": ["query", "framework_category"],
                },
            )
        )

        # Code execution tool
        tools.append(
            Tool(
                name="execute_code",
                description="Execute Python code in a sandboxed environment",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "language": {
                            "type": "string",
                            "description": "Language (only 'python' supported)",
                        },
                    },
                    "required": ["code"],
                },
            )
        )

        # Health check
        tools.append(
            Tool(
                name="health",
                description="Check server health and available capabilities",
                inputSchema={"type": "object", "properties": {}},
            )
        )

        # Context Gateway
        tools.append(
            Tool(
                name="prepare_context",
                description="Gemini prepares rich, structured context for Claude. Does the heavy lifting: analyzes query, discovers relevant files, fetches documentation, ranks by relevance. Returns organized context packet so Claude can focus on deep reasoning instead of egg-hunting.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The task or problem to prepare context for",
                        },
                        "workspace_path": {
                            "type": "string",
                            "description": "Path to workspace/project directory",
                        },
                        "code_context": {
                            "type": "string",
                            "description": "Any code snippets to consider",
                        },
                        "file_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Pre-specified files to analyze",
                        },
                        "search_docs": {
                            "type": "boolean",
                            "description": "Whether to search web for documentation (default: true)",
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["prompt", "json"],
                            "description": "Output as Claude prompt or raw JSON (default: prompt)",
                        },
                        "enable_cache": {
                            "type": "boolean",
                            "description": "Enable intelligent caching (default: true)",
                        },
                        "enable_multi_repo": {
                            "type": "boolean",
                            "description": "Enable multi-repository discovery (default: true)",
                        },
                        "enable_source_attribution": {
                            "type": "boolean",
                            "description": "Include source attribution for docs (default: true)",
                        },
                    },
                    "required": ["query"],
                },
            )
        )

        tools.append(
            Tool(
                name="prepare_context_streaming",
                description="Gemini-powered context preparation with real-time streaming progress. Shows what's happening during file discovery, doc search, and analysis.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The task or problem to prepare context for",
                        },
                        "workspace_path": {
                            "type": "string",
                            "description": "Path to workspace/project directory",
                        },
                        "code_context": {
                            "type": "string",
                            "description": "Any code snippets to consider",
                        },
                        "file_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Pre-specified files to analyze",
                        },
                        "search_docs": {
                            "type": "boolean",
                            "description": "Whether to search web for documentation (default: true)",
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["prompt", "json"],
                            "description": "Output as Claude prompt or raw JSON (default: prompt)",
                        },
                    },
                    "required": ["query"],
                },
            )
        )

        tools.append(
            Tool(
                name="context_cache_status",
                description="Get context cache status and statistics. Shows cache hit rates, token savings, and cache health.",
                inputSchema={"type": "object", "properties": {}},
            )
        )

        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        # Set a new correlation ID for this request
        import uuid

        correlation_id = str(uuid.uuid4())[:8]
        set_correlation_id(correlation_id)

        try:
            logger.info("call_tool_start", tool=name, args=list(arguments.keys()))

            # Rate limiting check
            from app.core.rate_limiter import get_rate_limiter

            rate_limiter = await get_rate_limiter()

            allowed, error_msg = await rate_limiter.check_rate_limit(name)
            if not allowed:
                logger.warning("rate_limit_rejected", tool=name, error=error_msg)
                return [TextContent(type="text", text=f"Rate limit exceeded: {error_msg}")]

            # Input size validation
            valid, size_error = rate_limiter.validate_input_size(arguments, name)
            if not valid:
                logger.warning("input_size_rejected", tool=name, error=size_error)
                return [TextContent(type="text", text=f"Input validation failed: {size_error}")]

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
                # Route to streaming handler if streaming=true
                if arguments.get("streaming", False):
                    return await handle_prepare_context_streaming(arguments)
                return await handle_prepare_context(arguments)

            if name == "context_cache_status":
                return await handle_context_cache_status(arguments)

            if name == "count_tokens":
                return await handle_count_tokens(arguments)

            if name == "compress_content":
                return await handle_compress_content(arguments)

            if name == "detect_truncation":
                return await handle_detect_truncation(arguments)

            if name == "manage_claude_md":
                return await handle_manage_claude_md(arguments)

            # Token Reduction Tools
            if name == "serialize_to_toon":
                return await handle_serialize_to_toon(arguments)

            if name == "deserialize_from_toon":
                return await handle_deserialize_from_toon(arguments)

            if name == "compress_prompt":
                return await handle_compress_prompt(arguments)

            if name == "compress_context":
                return await handle_compress_context(arguments)

            if name == "token_reduction_compare":
                return await handle_token_reduction_compare(arguments)

            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        finally:
            # Clear correlation ID after request completes
            clear_correlation_id()
            logger.info("call_tool_end", tool=name)

    # Attach handlers to server for testing access
    server.list_tools_handler = list_tools
    server.call_tool_handler = call_tool

    return server


async def graceful_shutdown(shutdown_event: asyncio.Event, timeout: float = 10.0) -> None:
    """Perform graceful shutdown with timeout.

    Args:
        shutdown_event: Event to signal shutdown completion
        timeout: Maximum time to wait for cleanup (default: 10s)
    """
    from app.graph import cleanup_checkpointer

    logger.info("graceful_shutdown_started", timeout_seconds=timeout)

    try:
        # Run cleanup with timeout
        await asyncio.wait_for(cleanup_checkpointer(), timeout=timeout)
        logger.info("graceful_shutdown_complete", status="success")
    except asyncio.TimeoutError:
        logger.warning("graceful_shutdown_timeout", timeout_seconds=timeout)
    except Exception as e:
        logger.error("graceful_shutdown_error", error=str(e), error_type=type(e).__name__)
    finally:
        shutdown_event.set()


def setup_signal_handlers(loop: asyncio.AbstractEventLoop, shutdown_event: asyncio.Event) -> None:
    """Set up signal handlers for graceful shutdown.

    Uses asyncio.add_signal_handler on Unix systems for proper async signal handling.
    Falls back to signal.signal on Windows (with limitations).

    Args:
        loop: The asyncio event loop
        shutdown_event: Event to signal when shutdown should begin
    """

    def handle_signal(sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        sig_name = sig.name if hasattr(sig, "name") else str(sig)
        logger.info("signal_received", signal=sig_name, action="initiating_shutdown")

        # Create shutdown task if not already shutting down
        if not shutdown_event.is_set():
            loop.create_task(graceful_shutdown(shutdown_event))

    # Unix-specific: use asyncio.add_signal_handler for proper async handling
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
                logger.debug("signal_handler_registered", signal=sig.name)
            except (ValueError, OSError) as e:
                # May fail if not in main thread or if signal is not valid
                logger.warning("signal_handler_registration_failed", signal=sig.name, error=str(e))
    else:
        # Windows: limited signal support, use signal.signal
        # Note: On Windows, only SIGINT (Ctrl+C) is reliably supported
        def sync_handler(signum: int, frame: Any) -> None:
            sig = signal.Signals(signum)
            handle_signal(sig)

        try:
            signal.signal(signal.SIGINT, sync_handler)
            signal.signal(signal.SIGTERM, sync_handler)
            logger.debug("signal_handlers_registered", platform="windows")
        except (ValueError, OSError) as e:
            logger.warning("signal_handler_registration_failed", error=str(e))


async def main():
    logger.info("=" * 60)
    logger.info("Omni-Cortex MCP - Operating System for Vibe Coders")
    logger.info("=" * 60)
    logger.info(f"Frameworks: {len(FRAMEWORKS)} thinking frameworks (internal)")
    logger.info(f"Graph nodes: {len(FRAMEWORK_NODES)} LangGraph nodes")
    logger.info("Memory: LangChain ConversationBufferMemory")
    logger.info("RAG: ChromaDB with 6 collections")
    if LEAN_MODE:
        logger.info("Mode: ULTRA-LEAN (4 tools)")
        logger.info("  Tools: prepare_context, reason, execute_code, health")
        logger.info("  Gemini handles: file discovery, doc search, compression, token optimization")
        logger.info("  62 frameworks available internally via HyperRouter")
        logger.info("  Set LEAN_MODE=false for full tool access")
    else:
        logger.info(f"Mode: FULL ({len(FRAMEWORKS) + 21} tools)")
        logger.info(f"  {len(FRAMEWORKS)} think_* tools + 21 utilities")
    logger.info("=" * 60)

    # Set up shutdown event and signal handlers
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    setup_signal_handlers(loop, shutdown_event)
    logger.info("signal_handlers_configured", signals=["SIGTERM", "SIGINT"])

    try:
        server = create_server()
        async with stdio_server() as (read_stream, write_stream):
            # Run server until shutdown is signaled or server exits
            server_task = asyncio.create_task(
                server.run(read_stream, write_stream, server.create_initialization_options())
            )
            shutdown_task = asyncio.create_task(shutdown_event.wait())

            # Wait for either server to exit or shutdown signal
            done, pending = await asyncio.wait(
                [server_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # Check if server raised an exception
            for task in done:
                if task is server_task and task.exception():
                    raise task.exception()

    except asyncio.CancelledError:
        logger.info("server_cancelled", reason="shutdown_signal")
    finally:
        # CRITICAL: Always cleanup resources on shutdown (especially important for Docker)
        # Ensures SQLite connections are closed and prevents "database locked" errors on restart
        if not shutdown_event.is_set():
            # If we haven't done graceful shutdown yet, do it now
            from app.graph import cleanup_checkpointer

            logger.info("shutting_down", action="cleanup_resources")
            try:
                await asyncio.wait_for(cleanup_checkpointer(), timeout=10.0)
                logger.info("server_shutdown_complete")
            except asyncio.TimeoutError:
                logger.warning("shutdown_cleanup_timeout", timeout_seconds=10.0)
            except Exception as e:
                logger.error("shutdown_cleanup_failed", error=str(e), error_type=type(e).__name__)
        else:
            logger.info("server_shutdown_complete", reason="graceful_shutdown_already_completed")


if __name__ == "__main__":
    asyncio.run(main())
