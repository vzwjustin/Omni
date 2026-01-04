"""
Omni-Cortex MCP Server

Exposes 40 thinking framework tools + utility tools.
The calling LLM uses these tools and does the reasoning.
LangGraph orchestrates, LangChain handles memory/RAG.
"""

import asyncio
import json
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

# Configure structlog to use stderr
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=True,
)

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
    # =========================================================================
    # NEW FRAMEWORKS (2026 Edition - Aligning with router.py)
    # =========================================================================
    "chain_of_code": {
        "category": "code",
        "description": "Code-based problem decomposition",
        "best_for": ["logic puzzles", "algorithmic debugging", "structured thinking"],
        "prompt": """Apply Chain-of-Code reasoning:

TASK: {query}
CONTEXT: {context}

1. DECOMPOSE: Break the problem into code blocks
2. TRACE: Walk through execution step by step
3. IDENTIFY: Pinpoint where logic diverges from intent
4. FIX: Apply targeted corrections with explanation
5. VERIFY: Trace corrected code to confirm fix"""
    },
    "self_debugging": {
        "category": "code",
        "description": "Mental execution trace before presenting",
        "best_for": ["preventing bugs", "edge case handling", "off-by-one errors"],
        "prompt": """Apply Self-Debugging:

TASK: {query}
CONTEXT: {context}

1. DRAFT: Write initial solution
2. TRACE: Mentally execute with sample inputs
3. EDGES: Test boundary conditions (0, 1, empty, null, max)
4. CATCH: Identify potential bugs before presenting
5. FIX: Present corrected code with trace verification"""
    },
    "tdd_prompting": {
        "category": "code",
        "description": "Test-first development approach",
        "best_for": ["new features", "edge case coverage", "robust implementations"],
        "prompt": """Apply TDD Prompting:

TASK: {query}
CONTEXT: {context}

1. TESTS FIRST: Write test cases covering:
   - Happy path
   - Edge cases
   - Error conditions
2. IMPLEMENT: Write minimal code to pass tests
3. REFACTOR: Clean up while keeping tests green
4. VERIFY: All tests pass with clean implementation"""
    },
    "reverse_cot": {
        "category": "code",
        "description": "Backward reasoning from output delta",
        "best_for": ["silent bugs", "wrong outputs", "calculation errors"],
        "prompt": """Apply Reverse Chain-of-Thought:

TASK: {query}
CONTEXT: {context}

1. EXPECTED: What output should the code produce?
2. ACTUAL: What is it actually producing?
3. DELTA: What's the difference?
4. BACKTRACK: Work backward from the delta to find the cause
5. FIX: Apply correction and verify expected output"""
    },
    "rubber_duck": {
        "category": "iterative",
        "description": "Socratic questioning for self-discovery",
        "best_for": ["architectural issues", "blind spots", "stuck problems"],
        "prompt": """Apply Rubber Duck Debugging:

TASK: {query}
CONTEXT: {context}

Guide through questions:
1. What are you trying to accomplish?
2. What have you tried so far?
3. What happened vs what you expected?
4. What assumptions are you making?
5. What haven't you checked yet?

Lead to insight through questioning."""
    },
    "react": {
        "category": "iterative",
        "description": "Reasoning + Acting with tools",
        "best_for": ["multi-step tasks", "tool use", "investigation"],
        "prompt": """Apply ReAct (Reasoning + Acting):

TASK: {query}
CONTEXT: {context}

Iterate:
THOUGHT: What do I need to figure out next?
ACTION: What tool/search/command would help?
OBSERVATION: What did I learn?

Continue until solution is found. Show your reasoning chain."""
    },
    "reflexion": {
        "category": "iterative",
        "description": "Self-evaluation with memory-based learning",
        "best_for": ["learning from failures", "iterative improvement", "retry scenarios"],
        "prompt": """Apply Reflexion:

TASK: {query}
CONTEXT: {context}

1. ATTEMPT: Make first solution attempt
2. EVALUATE: Did it work? What went wrong?
3. REFLECT: What lessons to remember?
4. RETRY: Apply lessons to improved attempt
5. CONVERGE: Iterate until successful"""
    },
    "self_refine": {
        "category": "iterative",
        "description": "Iterative self-critique and improvement",
        "best_for": ["code quality", "documentation", "polish"],
        "prompt": """Apply Self-Refine:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create initial solution
2. CRITIQUE: What could be better? (readability, efficiency, edge cases)
3. REFINE: Address each critique point
4. REPEAT: Continue until no more improvements
5. FINALIZE: Present polished solution"""
    },
    "least_to_most": {
        "category": "strategy",
        "description": "Bottom-up atomic function decomposition",
        "best_for": ["complex systems", "monolith refactoring", "dependency management"],
        "prompt": """Apply Least-to-Most Decomposition:

TASK: {query}
CONTEXT: {context}

1. IDENTIFY: What are the atomic sub-problems?
2. ORDER: Sort by dependency (least dependent first)
3. SOLVE: Build solutions bottom-up
4. COMPOSE: Combine into full solution
5. VERIFY: Check all pieces integrate correctly"""
    },
    "comparative_arch": {
        "category": "strategy",
        "description": "Multi-approach comparison (readability/memory/speed)",
        "best_for": ["optimization", "architecture decisions", "trade-off analysis"],
        "prompt": """Apply Comparative Architecture:

TASK: {query}
CONTEXT: {context}

Generate THREE approaches:
1. READABLE: Prioritize clarity and maintainability
2. EFFICIENT: Optimize for memory/space
3. FAST: Optimize for speed/time

For each: show code, pros, cons, big-O analysis.
RECOMMEND: Which approach and why."""
    },
    "plan_and_solve": {
        "category": "strategy",
        "description": "Explicit planning before execution",
        "best_for": ["complex features", "methodical development", "avoiding rushed code"],
        "prompt": """Apply Plan-and-Solve:

TASK: {query}
CONTEXT: {context}

PHASE 1 - PLAN:
- What components are needed?
- What's the order of implementation?
- What are the risks/blockers?

PHASE 2 - SOLVE:
- Implement according to plan
- Note any deviations and why
- Verify against plan"""
    },
    "red_team": {
        "category": "context",
        "description": "Adversarial security analysis (STRIDE, OWASP)",
        "best_for": ["security audits", "vulnerability scanning", "threat modeling"],
        "prompt": """Apply Red-Team Security Analysis:

TASK: {query}
CONTEXT: {context}

Analyze using STRIDE:
- SPOOFING: Identity issues?
- TAMPERING: Data integrity issues?
- REPUDIATION: Audit issues?
- INFO DISCLOSURE: Leakage issues?
- DENIAL OF SERVICE: Availability issues?
- ELEVATION OF PRIVILEGE: Access control issues?

OWASP Top 10 check. Provide fixes for each finding."""
    },
    "state_machine": {
        "category": "context",
        "description": "Formal FSM design before coding",
        "best_for": ["UI logic", "workflow systems", "game states"],
        "prompt": """Apply State-Machine Reasoning:

TASK: {query}
CONTEXT: {context}

1. STATES: Enumerate all possible states
2. TRANSITIONS: What triggers state changes?
3. GUARDS: What conditions must be met?
4. ACTIONS: What happens on transition?
5. IMPLEMENT: Code the state machine with clear structure"""
    },
    "chain_of_thought": {
        "category": "context",
        "description": "Step-by-step reasoning chain",
        "best_for": ["complex reasoning", "logical deduction", "problem solving"],
        "prompt": """Apply Chain-of-Thought:

TASK: {query}
CONTEXT: {context}

Think step by step:
STEP 1: [First logical step]
STEP 2: [Building on step 1]
STEP 3: [Continue reasoning]
...
CONCLUSION: [Final answer based on chain]

Show your complete reasoning process."""
    },
    "alphacodium": {
        "category": "code",
        "description": "Test-based multi-stage iterative code generation",
        "best_for": ["competitive programming", "complex algorithms", "interview problems"],
        "prompt": """Apply AlphaCodium:

TASK: {query}
CONTEXT: {context}

STAGE 1 - UNDERSTAND:
- Parse problem constraints
- Identify input/output format
- Note edge cases

STAGE 2 - GENERATE TESTS:
- Public test cases
- Edge cases
- Large input cases

STAGE 3 - ITERATIVE CODE:
- Generate solution
- Test against cases
- Fix failures
- Repeat until all pass"""
    },
    "codechain": {
        "category": "code",
        "description": "Chain of self-revisions guided by sub-modules",
        "best_for": ["modular code generation", "incremental refinement", "component development"],
        "prompt": """Apply CodeChain:

TASK: {query}
CONTEXT: {context}

1. MODULES: Identify independent sub-modules needed
2. GENERATE: Create each module with clear interface
3. INTEGRATE: Connect modules together
4. REVISE: Self-critique each module, improve
5. VALIDATE: Test integrated solution"""
    },
    "evol_instruct": {
        "category": "code",
        "description": "Evolutionary instruction complexity for code",
        "best_for": ["challenging code problems", "constraint-based coding", "extending solutions"],
        "prompt": """Apply Evol-Instruct:

TASK: {query}
CONTEXT: {context}

1. BASE: Solve the base problem
2. EVOLVE: Add constraint/complexity:
   - Additional requirements
   - Performance constraints
   - Edge case handling
3. SOLVE EVOLVED: Handle new complexity
4. REPEAT: Continue evolving until robust"""
    },
    "llmloop": {
        "category": "code",
        "description": "Automated iterative feedback loops for code+tests",
        "best_for": ["code quality assurance", "production-ready code", "CI/CD preparation"],
        "prompt": """Apply LLMLoop:

TASK: {query}
CONTEXT: {context}

LOOP:
1. GENERATE: Create code + tests
2. RUN: Execute tests (mentally trace if needed)
3. ANALYZE: What failed? Why?
4. FIX: Apply corrections
5. ITERATE: Until all quality checks pass

Quality checks: correctness, edge cases, readability, efficiency."""
    },
    "procoder": {
        "category": "code",
        "description": "Compiler-feedback-guided iterative refinement",
        "best_for": ["project-level code generation", "API usage", "type-safe code"],
        "prompt": """Apply ProCoder:

TASK: {query}
CONTEXT: {context}

1. ANALYZE: Understand project context, types, APIs
2. GENERATE: Create type-safe code using project patterns
3. COMPILE: Check for type errors, import issues
4. FIX: Address any compiler/linter feedback
5. INTEGRATE: Ensure clean integration with existing code"""
    },
    "recode": {
        "category": "code",
        "description": "Multi-candidate validation with CFG-based debugging",
        "best_for": ["reliable code generation", "high-stakes code", "mission-critical systems"],
        "prompt": """Apply RECODE:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create 3 candidate solutions
2. VALIDATE: Test each independently
3. ANALYZE CFG: Check control flow for issues
4. VOTE: Select best candidate based on:
   - Correctness
   - Robustness
   - Clarity
5. FINALIZE: Present winning candidate with confidence assessment"""
    },
}


def create_server() -> Server:
    """Create the MCP server with all tools."""
    server = Server("omni-cortex")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        tools = []

        # 40 Framework tools (think_*) - LLM selects based on task
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
            description="List all 40 thinking frameworks by category",
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
            context = arguments.get("context")  # None if not provided
            thread_id = arguments.get("thread_id")

            # Use singleton router imported from app.graph (not new instance each time)
            hyper_router = router

            # First check vibe dictionary, then heuristics, ensure fallback
            selected = hyper_router._check_vibe_dictionary(query)
            if not selected:
                selected = hyper_router._heuristic_select(query, context)
            
            # Guaranteed fallback to self_discover if both methods fail
            if not selected or selected not in FRAMEWORKS:
                selected = "self_discover"

            # Get framework info from router
            fw_info = hyper_router.get_framework_info(selected)
            complexity = hyper_router.estimate_complexity(query, context if context != "None provided" else None)

            # Get the framework prompt (guaranteed to exist after fallback)
            fw = FRAMEWORKS[selected]

            prompt = fw["prompt"].format(query=query, context=context or "None provided")

            # Prepend memory context if thread_id provided
            if thread_id:
                memory = await get_memory(thread_id)
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
                context = arguments.get("context")  # None if not provided
                thread_id = arguments.get("thread_id")

                prompt = FRAMEWORKS[fw_name]["prompt"].format(query=query, context=context or "None provided")

                # Include memory context if thread_id provided
                if thread_id:
                    memory = await get_memory(thread_id)
                    mem_context = memory.get_context()
                    if mem_context.get("chat_history"):
                        history_str = "\n".join(str(m) for m in mem_context["chat_history"][-5:])
                        prompt = f"CONVERSATION HISTORY:\n{history_str}\n\n{prompt}"
                    if mem_context.get("framework_history"):
                        prompt = f"PREVIOUSLY USED FRAMEWORKS: {mem_context['framework_history'][-5:]}\n\n{prompt}"

                return [TextContent(type="text", text=prompt)]

        # List frameworks
        if name == "list_frameworks":
            output = "# Omni-Cortex: 40 Thinking Frameworks\n\n"
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
            memory = await get_memory(thread_id)
            context = memory.get_context()
            return [TextContent(type="text", text=json.dumps(context, default=str, indent=2))]

        # Memory: save context
        if name == "save_context":
            await save_to_langchain_memory(
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
                "tools": len(FRAMEWORKS) + 14,  # 40 think_* + 14 utility tools
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
    logger.info(f"Tools: {len(FRAMEWORKS)} think_* + 14 utility = {len(FRAMEWORKS) + 14} total")
    logger.info("=" * 60)

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
