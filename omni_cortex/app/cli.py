#!/usr/bin/env python3
"""
Omni-Cortex CLI - Local Testing and Development Tool

Provides command-line interface for testing frameworks, routing,
and RAG search without needing MCP client configuration.

Usage:
    python -m app.cli reason "debug this code" --context "def foo(): pass"
    python -m app.cli framework active_inference --query "optimize this"
    python -m app.cli route "fix performance bug"
    python -m app.cli search "connection pool"
    python -m app.cli health
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Optional

import structlog

# Configure logging to stderr
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

logger = structlog.get_logger("cli")


async def cmd_reason(args: argparse.Namespace) -> int:
    """Run the reason tool with auto-framework selection."""
    from app.graph import router, FRAMEWORK_NODES
    from app.state import GraphState

    print(f"\n{'='*60}")
    print("Omni-Cortex Reason")
    print(f"{'='*60}\n")

    # Create initial state
    state: GraphState = {
        "query": args.query,
        "code_snippet": args.context,
        "file_list": args.files.split(",") if args.files else [],
        "working_memory": {"thread_id": args.thread_id or "cli-session"},
        "reasoning_steps": [],
        "tokens_used": 0,
        "confidence_score": 0.0,
    }

    # Route to framework
    print(f"Query: {args.query[:100]}{'...' if len(args.query) > 100 else ''}")
    print(f"\nRouting...")

    start = time.perf_counter()
    state = await router.route(state, use_ai=not args.heuristic)
    route_time = (time.perf_counter() - start) * 1000

    framework = state.get("selected_framework", "unknown")
    chain = state.get("framework_chain", [framework])
    category = state.get("routing_category", "unknown")
    complexity = state.get("complexity_estimate", 0.0)

    print(f"\n  Framework: {framework}")
    print(f"  Chain: {' -> '.join(chain)}")
    print(f"  Category: {category}")
    print(f"  Complexity: {complexity:.2f}")
    print(f"  Route time: {route_time:.1f}ms")

    if args.dry_run:
        print("\n[Dry run - skipping execution]")
        return 0

    # Execute framework
    print(f"\nExecuting {framework}...")

    if framework not in FRAMEWORK_NODES:
        print(f"Error: Framework '{framework}' not found")
        return 1

    start = time.perf_counter()
    try:
        state = await FRAMEWORK_NODES[framework](state)
        exec_time = (time.perf_counter() - start) * 1000

        print(f"\n{'='*60}")
        print("Result")
        print(f"{'='*60}\n")
        print(state.get("final_answer", "No answer generated"))

        print(f"\n{'='*60}")
        print("Metrics")
        print(f"{'='*60}")
        print(f"  Tokens used: {state.get('tokens_used', 0)}")
        print(f"  Confidence: {state.get('confidence_score', 0.0):.2f}")
        print(f"  Execution time: {exec_time:.1f}ms")

        if state.get("final_code"):
            print(f"\nGenerated Code:")
            print("```")
            print(state["final_code"][:1000])
            print("```")

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


async def cmd_framework(args: argparse.Namespace) -> int:
    """Run a specific framework directly."""
    from app.graph import FRAMEWORK_NODES
    from app.state import GraphState

    if args.framework not in FRAMEWORK_NODES:
        print(f"Error: Framework '{args.framework}' not found")
        print(f"\nAvailable frameworks ({len(FRAMEWORK_NODES)}):")
        for fw in sorted(FRAMEWORK_NODES.keys())[:20]:
            print(f"  - {fw}")
        print(f"  ... and {len(FRAMEWORK_NODES) - 20} more")
        return 1

    print(f"\n{'='*60}")
    print(f"Running: {args.framework}")
    print(f"{'='*60}\n")

    state: GraphState = {
        "query": args.query,
        "code_snippet": args.context,
        "file_list": [],
        "selected_framework": args.framework,
        "working_memory": {"thread_id": args.thread_id or "cli-session"},
        "reasoning_steps": [],
        "tokens_used": 0,
        "confidence_score": 0.0,
    }

    start = time.perf_counter()
    try:
        state = await FRAMEWORK_NODES[args.framework](state)
        exec_time = (time.perf_counter() - start) * 1000

        print(state.get("final_answer", "No answer generated"))

        print(f"\n{'='*60}")
        print(f"Tokens: {state.get('tokens_used', 0)} | "
              f"Confidence: {state.get('confidence_score', 0.0):.2f} | "
              f"Time: {exec_time:.1f}ms")

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


async def cmd_route(args: argparse.Namespace) -> int:
    """Test routing without executing the framework."""
    from app.graph import router
    from app.state import GraphState

    print(f"\n{'='*60}")
    print("Router Test")
    print(f"{'='*60}\n")

    state: GraphState = {
        "query": args.query,
        "code_snippet": args.context,
        "file_list": [],
        "working_memory": {},
        "reasoning_steps": [],
        "tokens_used": 0,
        "confidence_score": 0.0,
    }

    print(f"Query: {args.query}\n")

    # Test both heuristic and AI routing
    if args.compare:
        print("Heuristic routing:")
        start = time.perf_counter()
        heuristic_state = await router.route(dict(state), use_ai=False)
        h_time = (time.perf_counter() - start) * 1000
        print(f"  Framework: {heuristic_state.get('selected_framework')}")
        print(f"  Time: {h_time:.1f}ms\n")

        print("AI routing:")
        start = time.perf_counter()
        ai_state = await router.route(dict(state), use_ai=True)
        ai_time = (time.perf_counter() - start) * 1000
        print(f"  Framework: {ai_state.get('selected_framework')}")
        print(f"  Chain: {ai_state.get('framework_chain', [])}")
        print(f"  Category: {ai_state.get('routing_category')}")
        print(f"  Time: {ai_time:.1f}ms")
    else:
        start = time.perf_counter()
        state = await router.route(state, use_ai=not args.heuristic)
        route_time = (time.perf_counter() - start) * 1000

        print(f"Framework: {state.get('selected_framework')}")
        print(f"Chain: {state.get('framework_chain', [])}")
        print(f"Category: {state.get('routing_category')}")
        print(f"Complexity: {state.get('complexity_estimate', 0.0):.2f}")
        print(f"Time: {route_time:.1f}ms")

        # Show reasoning step
        if state.get("reasoning_steps"):
            step = state["reasoning_steps"][0]
            print(f"\nReason: {step.get('reason', 'N/A')}")

    return 0


async def cmd_search(args: argparse.Namespace) -> int:
    """Search the RAG vector store."""
    from app.collection_manager import get_collection_manager

    print(f"\n{'='*60}")
    print("RAG Search")
    print(f"{'='*60}\n")

    cm = get_collection_manager()

    print(f"Query: {args.query}")
    print(f"Collection: {args.collection or 'all'}")
    print(f"Results: {args.k}\n")

    start = time.perf_counter()

    if args.collection:
        results = cm.search(args.query, collection_name=args.collection, k=args.k)
    else:
        results = cm.search(args.query, k=args.k)

    search_time = (time.perf_counter() - start) * 1000

    if not results:
        print("No results found.")
    else:
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Score: {doc.metadata.get('score', 'N/A')}")
            content = doc.page_content[:500]
            print(f"Content: {content}{'...' if len(doc.page_content) > 500 else ''}")

    print(f"\n{'='*60}")
    print(f"Found {len(results)} results in {search_time:.1f}ms")

    return 0


async def cmd_list(args: argparse.Namespace) -> int:
    """List available frameworks."""
    from app.graph import FRAMEWORK_NODES
    from app.frameworks.registry import get_all_frameworks, FrameworkCategory

    print(f"\n{'='*60}")
    print(f"Available Frameworks ({len(FRAMEWORK_NODES)})")
    print(f"{'='*60}\n")

    frameworks = get_all_frameworks()

    # Group by category
    by_category: dict[str, list] = {}
    for fw in frameworks:
        cat = fw.category.value if hasattr(fw.category, 'value') else str(fw.category)
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(fw)

    for category in sorted(by_category.keys()):
        fws = by_category[category]
        print(f"\n{category.upper()} ({len(fws)} frameworks)")
        print("-" * 40)
        for fw in sorted(fws, key=lambda x: x.name):
            if args.verbose:
                print(f"  {fw.name}")
                print(f"    {fw.description[:80]}...")
                print(f"    Best for: {', '.join(fw.best_for[:3])}")
            else:
                print(f"  - {fw.name}: {fw.description[:60]}...")

    return 0


async def cmd_health(args: argparse.Namespace) -> int:
    """Check system health."""
    from app.graph import FRAMEWORK_NODES
    from app.core.settings import get_settings

    print(f"\n{'='*60}")
    print("Omni-Cortex Health Check")
    print(f"{'='*60}\n")

    settings = get_settings()

    checks = []

    # Framework check
    fw_count = len(FRAMEWORK_NODES)
    checks.append(("Frameworks", f"{fw_count} loaded", fw_count >= 60))

    # Settings check
    checks.append(("LLM Provider", settings.llm_provider, settings.llm_provider != ""))
    checks.append(("LEAN Mode", str(settings.lean_mode), True))

    # ChromaDB check
    try:
        from app.collection_manager import get_collection_manager
        cm = get_collection_manager()
        collections = cm.list_collections()
        checks.append(("ChromaDB", f"{len(collections)} collections", len(collections) > 0))
    except Exception as e:
        checks.append(("ChromaDB", f"Error: {e}", False))

    # Memory check
    try:
        from app.memory.manager import get_memory
        mem = await get_memory("health-check")
        checks.append(("Memory", "OK", True))
    except Exception as e:
        checks.append(("Memory", f"Error: {e}", False))

    # Print results
    all_ok = True
    for name, status, ok in checks:
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name}: {status}")
        if not ok:
            all_ok = False

    print(f"\n{'='*60}")
    print(f"Status: {'HEALTHY' if all_ok else 'DEGRADED'}")

    return 0 if all_ok else 1


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="omni-cortex",
        description="Omni-Cortex CLI - Local testing and development tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # reason command
    reason_parser = subparsers.add_parser("reason", help="Run reasoning with auto-framework selection")
    reason_parser.add_argument("query", help="The query or task")
    reason_parser.add_argument("--context", "-c", help="Code context")
    reason_parser.add_argument("--files", "-f", help="Comma-separated file list")
    reason_parser.add_argument("--thread-id", "-t", help="Thread ID for memory")
    reason_parser.add_argument("--heuristic", action="store_true", help="Use heuristic routing only")
    reason_parser.add_argument("--dry-run", action="store_true", help="Only route, don't execute")

    # framework command
    fw_parser = subparsers.add_parser("framework", help="Run a specific framework")
    fw_parser.add_argument("framework", help="Framework name (e.g., active_inference)")
    fw_parser.add_argument("--query", "-q", required=True, help="The query or task")
    fw_parser.add_argument("--context", "-c", help="Code context")
    fw_parser.add_argument("--thread-id", "-t", help="Thread ID for memory")

    # route command
    route_parser = subparsers.add_parser("route", help="Test routing without execution")
    route_parser.add_argument("query", help="The query to route")
    route_parser.add_argument("--context", "-c", help="Code context")
    route_parser.add_argument("--heuristic", action="store_true", help="Use heuristic routing only")
    route_parser.add_argument("--compare", action="store_true", help="Compare heuristic vs AI routing")

    # search command
    search_parser = subparsers.add_parser("search", help="Search RAG vector store")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--collection", "-c", help="Collection to search")
    search_parser.add_argument("--k", "-k", type=int, default=5, help="Number of results")

    # list command
    list_parser = subparsers.add_parser("list", help="List available frameworks")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Show details")

    # health command
    subparsers.add_parser("health", help="Check system health")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run the appropriate command
    commands = {
        "reason": cmd_reason,
        "framework": cmd_framework,
        "route": cmd_route,
        "search": cmd_search,
        "list": cmd_list,
        "health": cmd_health,
    }

    return asyncio.run(commands[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
