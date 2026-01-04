"""
Omni-Cortex MCP Server

Exposes 20 thinking framework tools + utility tools.
The calling LLM uses these tools and does the reasoning.
LangGraph orchestrates, LangChain handles memory/RAG.
"""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import LangGraph for orchestration
from app.graph import FRAMEWORK_NODES, router
from app.state import GraphState

# Import LangChain integration for memory and RAG
from app.langchain_integration import (
    get_memory,
    save_to_langchain_memory,
    search_vectorstore,
    enhance_state_with_langchain,
    AVAILABLE_TOOLS,
    OmniCortexCallback,
)
from app.collection_manager import get_collection_manager
from app.core.router import HyperRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-cortex")

# Framework definitions - what the LLM gets when it calls the tool
FRAMEWORKS = {
    "reason_flux": {
        "category": "strategy",
        "description": "Hierarchical planning: Template -> Expand -> Refine",
        "best_for": ["architecture", "system design", "complex planning"],
        "prompt": """Apply ReasonFlux hierarchical planning:

TASK: {query}
CONTEXT: {context}

PHASE 1 - TEMPLATE: Create high-level structure with 3-5 major components
PHASE 2 - EXPAND: Detail each component (classes, functions, interfaces)
PHASE 3 - REFINE: Integrate into final plan with code skeleton"""
    },
    "self_discover": {
        "category": "strategy",
        "description": "Discover and apply reasoning patterns",
        "best_for": ["novel problems", "unknown domains"],
        "prompt": """Apply Self-Discover reasoning:

TASK: {query}
CONTEXT: {context}

1. SELECT: Which patterns apply? (decomposition, analogy, abstraction, constraints)
2. ADAPT: Customize patterns for this specific task
3. IMPLEMENT: Apply your customized approach
4. VERIFY: Check completeness"""
    },
    "buffer_of_thoughts": {
        "category": "strategy",
        "description": "Build context in a thought buffer",
        "best_for": ["multi-part problems", "complex context"],
        "prompt": """Apply Buffer of Thoughts:

TASK: {query}
CONTEXT: {context}

Build your thought buffer:
- INIT: Key facts and constraints
- ADD: Problem analysis
- ADD: Possible approaches
- ADD: Decision and reasoning
- OUTPUT: Synthesize final solution"""
    },
    "coala": {
        "category": "strategy",
        "description": "Cognitive architecture for agents",
        "best_for": ["autonomous tasks", "agent behavior"],
        "prompt": """Apply COALA cognitive architecture:

TASK: {query}
CONTEXT: {context}

1. PERCEPTION: Current state and resources
2. MEMORY: Relevant knowledge and patterns
3. REASONING: Analyze and plan
4. ACTION: Execute plan
5. LEARNING: What worked? What to improve?"""
    },
    "mcts_rstar": {
        "category": "search",
        "description": "Monte Carlo Tree Search exploration for code",
        "best_for": ["complex bugs", "multi-step optimization", "thorough search"],
        "prompt": """Apply rStar-Code MCTS reasoning:

TASK: {query}
CONTEXT: {context}

1. SELECT: Most promising focus area using UCT scoring
2. EXPAND: Generate 2-3 possible modifications
3. SIMULATE: Trace consequences of each path
4. EVALUATE: Score each path (0-1)
5. BACKPROPAGATE: Update parent scores
6. ITERATE: Repeat until confidence threshold or max depth"""
    },
    "tree_of_thoughts": {
        "category": "search",
        "description": "Explore multiple paths, pick best",
        "best_for": ["design decisions", "multiple valid approaches"],
        "prompt": """Apply Tree of Thoughts:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create 3 distinct approaches with pros/cons
2. EVALUATE: Score each (feasibility, effectiveness, simplicity)
3. EXPAND: Develop the best approach fully
4. SYNTHESIZE: Final solution with reasoning"""
    },
    "graph_of_thoughts": {
        "category": "search",
        "description": "Non-linear reasoning with idea graphs",
        "best_for": ["complex dependencies", "interconnected problems"],
        "prompt": """Apply Graph of Thoughts:

TASK: {query}
CONTEXT: {context}

1. NODES: Identify key concepts/components
2. EDGES: Map relationships between them
3. TRAVERSE: Find the solution path through the graph
4. SYNTHESIZE: Combine insights into solution"""
    },
    "everything_of_thought": {
        "category": "search",
        "description": "Combine multiple reasoning approaches",
        "best_for": ["complex novel problems", "when one approach isn't enough"],
        "prompt": """Apply Everything of Thought:

TASK: {query}
CONTEXT: {context}

1. MULTI-APPROACH: Apply analytical, creative, critical, practical thinking
2. CROSS-POLLINATE: Find synergies between approaches
3. SYNTHESIZE: Create unified solution
4. VALIDATE: Check against all perspectives"""
    },
    "active_inference": {
        "category": "iterative",
        "description": "Hypothesis testing loop",
        "best_for": ["debugging", "investigation", "root cause analysis"],
        "prompt": """Apply Active Inference:

TASK: {query}
CONTEXT: {context}

1. OBSERVE: Current state, form hypotheses
2. PREDICT: What should we expect if hypothesis is true?
3. TEST: Gather evidence, update beliefs
4. ACT: Implement fix based on best hypothesis
5. VERIFY: Confirm the fix worked"""
    },
    "multi_agent_debate": {
        "category": "iterative",
        "description": "Multiple perspectives debate",
        "best_for": ["design decisions", "trade-off analysis"],
        "prompt": """Apply Multi-Agent Debate:

TASK: {query}
CONTEXT: {context}

Argue from these perspectives:
- PRAGMATIST: What's the simplest working solution?
- ARCHITECT: What's most maintainable/scalable?
- SECURITY: What are the risks?
- PERFORMANCE: What's most efficient?

DEBATE the trade-offs, then SYNTHESIZE a balanced solution."""
    },
    "adaptive_injection": {
        "category": "iterative",
        "description": "Inject strategies as needed",
        "best_for": ["evolving understanding", "adaptive problem solving"],
        "prompt": """Apply Adaptive Injection:

TASK: {query}
CONTEXT: {context}

As you work, inject strategies when needed:
- If stuck → step back and abstract
- If complex → decompose into parts
- If uncertain → explore alternatives
- If risky → add verification steps

Continue until complete."""
    },
    "re2": {
        "category": "iterative",
        "description": "Read-Execute-Evaluate loop",
        "best_for": ["specifications", "requirements implementation"],
        "prompt": """Apply RE2 (Read-Execute-Evaluate):

TASK: {query}
CONTEXT: {context}

1. READ: Parse requirements, list acceptance criteria
2. EXECUTE: Implement, referencing each requirement
3. EVALUATE: Check against requirements, fix gaps

Repeat until all requirements are satisfied."""
    },
    "program_of_thoughts": {
        "category": "code",
        "description": "Step-by-step code reasoning",
        "best_for": ["algorithms", "data processing", "math"],
        "prompt": """Apply Program of Thoughts:

TASK: {query}
CONTEXT: {context}

1. UNDERSTAND: What's input? What's output? What transformations?
2. DECOMPOSE: Break into computational steps
3. CODE: Write each step with clear comments
4. TRACE: Walk through with sample input
5. OUTPUT: Complete solution"""
    },
    "chain_of_verification": {
        "category": "code",
        "description": "Draft-Verify-Patch cycle",
        "best_for": ["security review", "code quality", "bug prevention"],
        "prompt": """Apply Chain of Verification:

TASK: {query}
CONTEXT: {context}

1. DRAFT: Create initial solution
2. VERIFY: Check for security issues, bugs, best practice violations
3. PATCH: Fix all identified issues
4. VALIDATE: Confirm fixes, no regressions"""
    },
    "critic": {
        "category": "code",
        "description": "Generate then critique",
        "best_for": ["API design", "interface validation"],
        "prompt": """Apply Critic framework:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create initial solution
2. CRITIQUE: What works? What's missing? What could break?
3. REVISE: Address each criticism
4. FINAL CHECK: Verify improvements"""
    },
    "chain_of_note": {
        "category": "context",
        "description": "Research and note-taking approach",
        "best_for": ["understanding code", "documentation", "exploration"],
        "prompt": """Apply Chain of Note:

TASK: {query}
CONTEXT: {context}

NOTE 1 - Observations: What do you see?
NOTE 2 - Connections: How do pieces relate?
NOTE 3 - Inferences: What can you conclude?
NOTE 4 - Synthesis: Complete answer"""
    },
    "step_back": {
        "category": "context",
        "description": "Abstract principles first, then apply",
        "best_for": ["optimization", "performance", "architectural decisions"],
        "prompt": """Apply Step-Back reasoning:

TASK: {query}
CONTEXT: {context}

1. STEP BACK: What category is this? What principles apply?
2. ABSTRACT: Key constraints, trade-offs, proven approaches
3. APPLY: Map abstract principles to concrete solution
4. VERIFY: Solution follows identified principles"""
    },
    "analogical": {
        "category": "context",
        "description": "Find and adapt similar solutions",
        "best_for": ["creative solutions", "pattern matching"],
        "prompt": """Apply Analogical reasoning:

TASK: {query}
CONTEXT: {context}

1. FIND ANALOGIES: What similar problems have been solved? (2-3)
2. MAP: How does the best analogy map to this problem?
3. ADAPT: What transfers? What's different?
4. IMPLEMENT: Build solution using adapted approach"""
    },
    "skeleton_of_thought": {
        "category": "fast",
        "description": "Outline first, fill in details",
        "best_for": ["boilerplate", "quick scaffolding"],
        "prompt": """Apply Skeleton of Thought:

TASK: {query}
CONTEXT: {context}

1. SKELETON: High-level structure (components, interfaces)
2. FLESH OUT: Add implementation details
3. CONNECT: Handle integration and edge cases"""
    },
    "system1": {
        "category": "fast",
        "description": "Fast intuitive response",
        "best_for": ["simple questions", "quick fixes"],
        "prompt": """Quick response for: {query}

Context: {context}

Provide a direct, efficient answer. Focus on the most likely correct solution."""
    },
}


def create_server() -> Server:
    """Create the MCP server with all tools."""
    server = Server("omni-cortex")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        tools = []

        # 20 Framework tools (think_*) - LLM selects based on task
        for name, fw in FRAMEWORKS.items():
            # Build vibes from router for better LLM selection
            vibes = router.VIBE_DICTIONARY.get(name, [])[:4]
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

        # Smart routing tool (uses HyperRouter)
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
            description="List all 20 thinking frameworks by category",
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
                "properties": {
                    "thread_id": {"type": "string", "description": "Thread ID to get context for"}
                },
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
                "properties": {
                    "function_name": {"type": "string"},
                    "k": {"type": "integer"}
                },
                "required": ["function_name"]
            }
        ))

        tools.append(Tool(
            name="search_class",
            description="Find specific class implementations by name",
            inputSchema={
                "type": "object",
                "properties": {
                    "class_name": {"type": "string"},
                    "k": {"type": "integer"}
                },
                "required": ["class_name"]
            }
        ))

        tools.append(Tool(
            name="search_docs_only",
            description="Search only markdown documentation files",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer"}
                },
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

        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        manager = get_collection_manager()

        # Smart routing with HyperRouter (reason tool)
        if name == "reason":
            query = arguments.get("query", "")
            context = arguments.get("context", "None provided")
            thread_id = arguments.get("thread_id")

            # Use HyperRouter for vibe-based selection
            hyper_router = HyperRouter()

            # First check vibe dictionary, then heuristics
            selected = hyper_router._check_vibe_dictionary(query)
            if not selected:
                selected = hyper_router._heuristic_select(query, context if context != "None provided" else None)

            # Get framework info from router
            fw_info = hyper_router.get_framework_info(selected)
            complexity = hyper_router.estimate_complexity(query, context if context != "None provided" else None)

            # Get the framework prompt (fallback to self_discover if not found)
            fw = FRAMEWORKS.get(selected, FRAMEWORKS.get("self_discover"))
            if not fw:
                selected = "self_discover"
                fw = FRAMEWORKS["self_discover"]

            prompt = fw["prompt"].format(query=query, context=context)

            # Prepend memory context if thread_id provided
            if thread_id:
                memory = get_memory(thread_id)
                mem_context = memory.get_context()
                if mem_context.get("chat_history"):
                    history_str = "\n".join(str(m) for m in mem_context["chat_history"][-5:])
                    prompt = f"CONVERSATION HISTORY:\n{history_str}\n\n{prompt}"
                if mem_context.get("framework_history"):
                    prompt = f"PREVIOUSLY USED FRAMEWORKS: {mem_context['framework_history'][-5:]}\n\n{prompt}"

            # Return with routing metadata
            output = f"# Auto-selected Framework: {selected}\n"
            output += f"Category: {fw_info.get('category', fw['category'])} | Complexity: {complexity:.2f}\n"
            output += f"Best for: {', '.join(fw_info.get('best_for', fw['best_for']))}\n\n"
            output += prompt

            return [TextContent(type="text", text=output)]

        # Framework tools (think_*)
        if name.startswith("think_"):
            fw_name = name[6:]
            if fw_name in FRAMEWORKS:
                query = arguments.get("query", "")
                context = arguments.get("context", "None provided")
                thread_id = arguments.get("thread_id")

                prompt = FRAMEWORKS[fw_name]["prompt"].format(query=query, context=context)

                # Include memory context if thread_id provided
                if thread_id:
                    memory = get_memory(thread_id)
                    mem_context = memory.get_context()
                    if mem_context.get("chat_history"):
                        history_str = "\n".join(str(m) for m in mem_context["chat_history"][-5:])
                        prompt = f"CONVERSATION HISTORY:\n{history_str}\n\n{prompt}"
                    if mem_context.get("framework_history"):
                        prompt = f"PREVIOUSLY USED FRAMEWORKS: {mem_context['framework_history'][-5:]}\n\n{prompt}"

                return [TextContent(type="text", text=prompt)]

        # List frameworks
        if name == "list_frameworks":
            output = "# Omni-Cortex: 20 Thinking Frameworks\n\n"
            for cat in ["strategy", "search", "iterative", "code", "context", "fast"]:
                output += f"## {cat.upper()}\n"
                for n, fw in FRAMEWORKS.items():
                    if fw["category"] == cat:
                        output += f"- `think_{n}`: {fw['description']}\n"
                output += "\n"
            return [TextContent(type="text", text=output)]

        # Recommend framework
        if name == "recommend":
            task = arguments.get("task", "").lower()
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
            return [TextContent(type="text", text=f"Recommended: `think_{rec}`\n\n{fw['description']}\nBest for: {', '.join(fw['best_for'])}")]

        # Memory: get context
        if name == "get_context":
            thread_id = arguments.get("thread_id", "")
            memory = get_memory(thread_id)
            context = memory.get_context()
            return [TextContent(type="text", text=json.dumps(context, default=str, indent=2))]

        # Memory: save context
        if name == "save_context":
            save_to_langchain_memory(
                arguments["thread_id"],
                arguments["query"],
                arguments["answer"],
                arguments["framework"]
            )
            return [TextContent(type="text", text="Context saved successfully")]

        # RAG: search documentation
        if name == "search_documentation":
            query = arguments.get("query", "")
            k = arguments.get("k", 5)
            docs = search_vectorstore(query, k=k)
            if not docs:
                return [TextContent(type="text", text="No results found. Try refining your query.")]
            formatted = []
            for d in docs:
                meta = d.metadata or {}
                path = meta.get("path", "unknown")
                formatted.append(f"### {path}\n{d.page_content[:1500]}")
            return [TextContent(type="text", text="\n\n".join(formatted))]

        # RAG: search frameworks by name
        if name == "search_frameworks_by_name":
            docs = manager.search_frameworks(
                arguments.get("query", ""),
                framework_name=arguments.get("framework_name"),
                k=arguments.get("k", 3)
            )
            if not docs:
                return [TextContent(type="text", text=f"No results in framework '{arguments.get('framework_name')}'")]
            formatted = [f"### {d.metadata.get('path', 'unknown')}\n{d.page_content[:1000]}" for d in docs]
            return [TextContent(type="text", text="\n\n".join(formatted))]

        # RAG: search by category
        if name == "search_by_category":
            category = arguments.get("category", "")
            collection_map = {
                "framework": ["frameworks"],
                "documentation": ["documentation"],
                "config": ["configs"],
                "utility": ["utilities"],
                "test": ["tests"],
                "integration": ["integrations"]
            }
            collections = collection_map.get(category)
            if not collections:
                return [TextContent(type="text", text=f"Invalid category. Use: framework, documentation, config, utility, test, integration")]
            docs = manager.search(arguments.get("query", ""), collection_names=collections, k=arguments.get("k", 5))
            if not docs:
                return [TextContent(type="text", text=f"No results in category '{category}'")]
            formatted = [f"### {d.metadata.get('path', 'unknown')}\n{d.page_content[:800]}" for d in docs]
            return [TextContent(type="text", text="\n\n".join(formatted))]

        # RAG: search function
        if name == "search_function":
            docs = manager.search_by_function(arguments.get("function_name", ""), k=arguments.get("k", 3))
            if not docs:
                return [TextContent(type="text", text=f"No function '{arguments.get('function_name')}' found")]
            formatted = [f"### {d.metadata.get('path', 'unknown')}\n{d.page_content[:1200]}" for d in docs]
            return [TextContent(type="text", text="\n\n".join(formatted))]

        # RAG: search class
        if name == "search_class":
            docs = manager.search_by_class(arguments.get("class_name", ""), k=arguments.get("k", 3))
            if not docs:
                return [TextContent(type="text", text=f"No class '{arguments.get('class_name')}' found")]
            formatted = [f"### {d.metadata.get('path', 'unknown')}\n{d.page_content[:1200]}" for d in docs]
            return [TextContent(type="text", text="\n\n".join(formatted))]

        # RAG: search docs only
        if name == "search_docs_only":
            docs = manager.search_documentation(arguments.get("query", ""), k=arguments.get("k", 5))
            if not docs:
                return [TextContent(type="text", text="No documentation found")]
            formatted = [f"### {d.metadata.get('file_name', 'unknown')}\n{d.page_content[:1000]}" for d in docs]
            return [TextContent(type="text", text="\n\n".join(formatted))]

        # RAG: search framework category
        if name == "search_framework_category":
            category = arguments.get("framework_category", "")
            valid = ["strategy", "search", "iterative", "code", "context", "fast"]
            if category not in valid:
                return [TextContent(type="text", text=f"Invalid category. Use: {', '.join(valid)}")]
            docs = manager.search_frameworks(
                arguments.get("query", ""),
                framework_category=category,
                k=arguments.get("k", 5)
            )
            if not docs:
                return [TextContent(type="text", text=f"No results in category '{category}'")]
            formatted = [f"### Framework: {d.metadata.get('framework_name', 'unknown')}\nPath: {d.metadata.get('path', 'unknown')}\n\n{d.page_content[:900]}" for d in docs]
            return [TextContent(type="text", text="\n\n".join(formatted))]

        # Code execution
        if name == "execute_code":
            # Import at runtime to avoid circular import
            from app.nodes.code.pot import _safe_execute

            code = arguments.get("code", "")
            language = arguments.get("language", "python")

            if language.lower() != "python":
                return [TextContent(type="text", text=json.dumps({"success": False, "error": f"Only python supported, got: {language}"}))]

            result = await _safe_execute(code)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # Health check
        if name == "health":
            collections = list(manager.COLLECTIONS.keys())
            return [TextContent(type="text", text=json.dumps({
                "status": "healthy",
                "frameworks": len(FRAMEWORKS),
                "tools": 20 + 14,  # 20 think_* + 14 utility tools
                "collections": collections,
                "memory_enabled": True,
                "rag_enabled": True
            }, indent=2))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def main():
    logger.info("=" * 60)
    logger.info("Omni-Cortex MCP - Operating System for Vibe Coders")
    logger.info("=" * 60)
    logger.info(f"Frameworks: {len(FRAMEWORKS)} thinking frameworks")
    logger.info(f"Graph nodes: {len(FRAMEWORK_NODES)} LangGraph nodes")
    logger.info("Memory: LangChain ConversationBufferMemory")
    logger.info("RAG: ChromaDB with 6 collections")
    logger.info("Tools: 20 think_* + 1 reason + 14 utility = 35 total")
    logger.info("=" * 60)

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
