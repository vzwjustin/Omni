"""
Omni-Cortex MCP Server

Exposes 40 thinking framework tools + utility tools.
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
    # New frameworks (2026 Edition + Modern LLM)
    "chain_of_code": {
        "category": "code",
        "description": "Break problems into code blocks for structured thinking",
        "best_for": ["logic puzzles", "algorithmic debugging", "recursive logic"],
        "prompt": """Apply Chain-of-Code reasoning:

TASK: {query}
CONTEXT: {context}

1. TRANSLATE: Express problem as computational representation
2. DECOMPOSE: Break into code blocks/functions
3. EXECUTE: Mental execution trace of each block
4. SYNTHESIZE: Extract answer from execution traces"""
    },
    "self_debugging": {
        "category": "code",
        "description": "Mental execution trace before presenting code",
        "best_for": ["preventing bugs", "edge case handling", "off-by-one errors"],
        "prompt": """Apply Self-Debugging framework:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Write initial code
2. IDENTIFY: Generate test cases (normal + edge cases)
3. TRACE: Line-by-line mental execution
4. DEBUG: Fix identified errors
5. VERIFY: Confirm fixes work"""
    },
    "tdd_prompting": {
        "category": "code",
        "description": "Write tests first, then implementation",
        "best_for": ["new features", "edge case coverage", "TDD methodology"],
        "prompt": """Apply TDD Prompting:

TASK: {query}
CONTEXT: {context}

1. SPECIFY: Clarify requirements
2. TEST: Write 5-8 unit tests FIRST (happy path, edge cases, errors)
3. IMPLEMENT: Write minimal code to pass tests
4. VERIFY: Mental test execution
5. REFACTOR: Improve code quality"""
    },
    "reverse_cot": {
        "category": "code",
        "description": "Work backward from buggy output to source",
        "best_for": ["silent bugs", "wrong outputs", "calculation errors"],
        "prompt": """Apply Reverse Chain-of-Thought:

TASK: {query}
CONTEXT: {context}

1. COMPARE: Analyze actual vs expected output delta
2. HYPOTHESIZE: What could cause this specific difference?
3. TRACE_BACK: Work backward through code
4. LOCATE: Identify specific buggy lines
5. FIX: Correct root cause"""
    },
    "rubber_duck": {
        "category": "iterative",
        "description": "Socratic questioning for self-discovery",
        "best_for": ["architectural issues", "blind spots", "unclear problems"],
        "prompt": """Apply Rubber Duck Debugging:

TASK: {query}
CONTEXT: {context}

1. LISTEN: Understand the problem
2. QUESTION: Ask clarifying questions about assumptions
3. PROBE: Challenge logic gaps
4. GUIDE: Lead toward self-discovery (don't give answers)
5. REFLECT: Summarize insights revealed"""
    },
    "react": {
        "category": "iterative",
        "description": "Interleaved reasoning and acting with tools",
        "best_for": ["multi-step tasks", "tool use", "information gathering"],
        "prompt": """Apply ReAct (Reasoning + Acting):

TASK: {query}
CONTEXT: {context}

Iterate: THOUGHT -> ACTION -> OBSERVATION
1. THOUGHT: What should I do next?
2. ACTION: Execute tool/command
3. OBSERVATION: Record result
(Repeat until ready for final answer)
4. FINAL ANSWER: Synthesize from trace"""
    },
    "reflexion": {
        "category": "iterative",
        "description": "Self-evaluation with memory-based learning",
        "best_for": ["learning from failures", "iterative improvement", "complex debugging"],
        "prompt": """Apply Reflexion framework:

TASK: {query}
CONTEXT: {context}

Loop (max 3 attempts):
1. ATTEMPT: Try to solve
2. EVALUATE: Assess success/failure
3. REFLECT: Analyze what went wrong and why
4. MEMORIZE: Store reflection insights
5. RETRY: Use reflection to improve"""
    },
    "self_refine": {
        "category": "iterative",
        "description": "Iterative self-critique and improvement",
        "best_for": ["code quality", "documentation", "polishing solutions"],
        "prompt": """Apply Self-Refine:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create initial solution
2-4. Loop (3 refinements):
   - CRITIQUE: Find flaws as editor
   - REFINE: Improve based on critique
5. FINALIZE: Present polished solution"""
    },
    "least_to_most": {
        "category": "strategy",
        "description": "Bottom-up atomic function decomposition",
        "best_for": ["complex systems", "monolith refactoring", "large features"],
        "prompt": """Apply Least-to-Most Decomposition:

TASK: {query}
CONTEXT: {context}

1. DECOMPOSE: Break into atomic functions + dependencies
2. ORDER: Topological sort (least dependent first)
3. IMPLEMENT_BASE: Build Level 0 (no dependencies)
4. BUILD_UP: Implement higher levels using base
5. INTEGRATE: Combine into complete solution"""
    },
    "comparative_arch": {
        "category": "strategy",
        "description": "Multiple solution approaches comparison",
        "best_for": ["optimization", "architecture decisions", "trade-off analysis"],
        "prompt": """Apply Comparative Architecture:

TASK: {query}
CONTEXT: {context}

1. ANALYZE: Requirements and constraints
2. GENERATE_3: Create versions optimized for:
   - Readability/Maintainability
   - Memory Efficiency
   - Execution Speed
3. COMPARE: Trade-off analysis (table format)
4. RECOMMEND: Best choice for context"""
    },
    "plan_and_solve": {
        "category": "strategy",
        "description": "Explicit planning before execution",
        "best_for": ["complex features", "methodical development", "avoiding rushed code"],
        "prompt": """Apply Plan-and-Solve:

TASK: {query}
CONTEXT: {context}

1. UNDERSTAND: Clarify problem thoroughly
2. PLAN: Detailed step-by-step plan (don't code yet!)
3. VERIFY_PLAN: Check completeness
4. EXECUTE: Implement following plan
5. VALIDATE: Ensure execution matches plan"""
    },
    "red_team": {
        "category": "context",
        "description": "Adversarial security analysis (STRIDE, OWASP)",
        "best_for": ["security audits", "vulnerability scanning", "penetration testing"],
        "prompt": """Apply Red-Teaming (Security Analysis):

TASK: {query}
CONTEXT: {context}

1. RECONNAISSANCE: Map attack surface
2. THREAT_MODEL: Identify attack vectors (STRIDE, OWASP Top 10)
3. EXPLOIT: Find specific vulnerabilities
4. ASSESS: Rate severity (CVSS-style)
5. PATCH: Provide secure fixes"""
    },
    "state_machine": {
        "category": "context",
        "description": "Formal FSM design before coding",
        "best_for": ["UI logic", "workflow systems", "game development"],
        "prompt": """Apply State-Machine Reasoning:

TASK: {query}
CONTEXT: {context}

1. IDENTIFY_STATES: All possible states
2. MAP_TRANSITIONS: State transitions + triggers
3. DEFINE_ACTIONS: onEnter/onExit/whileIn for each state
4. VALIDATE: Check for unreachable states, deadlocks
5. IMPLEMENT: Code the state machine"""
    },
    "chain_of_thought": {
        "category": "context",
        "description": "Step-by-step reasoning (foundational technique)",
        "best_for": ["complex reasoning", "logical deduction", "math problems"],
        "prompt": """Apply Chain-of-Thought reasoning:

TASK: {query}
CONTEXT: {context}

1. UNDERSTAND: Restate problem
2. BREAK_DOWN: Decompose into logical steps
3. REASON: Work through each step explicitly
4. CONCLUDE: Final answer with justification

Show all your work step-by-step!"""
    },
    # Additional coding frameworks (2026 expansion)
    "alphacodium": {
        "category": "code",
        "description": "Test-based multi-stage iterative code generation",
        "best_for": ["competitive programming", "complex algorithms", "code contests"],
        "prompt": """Apply AlphaCodium test-based iterative flow:

TASK: {query}
CONTEXT: {context}

PHASE 1 - PRE-PROCESSING (Natural Language):
1. PROBLEM_REFLECTION: Understand problem in depth, identify edge cases
2. PUBLIC_TEST_REASONING: Explain why each example works
3. GENERATE_POSSIBLE_SOLUTIONS: Brainstorm 2-3 approaches
4. RANK_SOLUTIONS: Pick best approach based on constraints
5. GENERATE_AI_TESTS: Create additional test cases

PHASE 2 - CODE_ITERATIONS:
6. GENERATE_INITIAL_CODE: Modular code in YAML format
7. ITERATE_ON_PUBLIC_TESTS: Run public tests, fix failures
8. ITERATE_ON_AI_TESTS: Run AI-generated tests, fix failures
9. RANK_SOLUTIONS: If multiple candidates, pick best

Return refined, test-verified code."""
    },
    "codechain": {
        "category": "code",
        "description": "Chain of self-revisions guided by sub-modules",
        "best_for": ["modular code generation", "incremental refinement", "complex implementations"],
        "prompt": """Apply CodeChain sub-module-based self-revision:

TASK: {query}
CONTEXT: {context}

1. DECOMPOSE: Break into sub-modules/functions (identify 3-5 core components)
2. GENERATE_SUB_MODULES: Implement each sub-module independently
3. CHAIN_REVISIONS: For each module:
   - Generate initial version
   - Compare with representative examples from previous iterations
   - Self-revise based on patterns learned
4. INTEGRATE: Combine revised sub-modules
5. GLOBAL_REVISION: Review and refine the integrated solution"""
    },
    "evol_instruct": {
        "category": "code",
        "description": "Evolutionary instruction complexity for code",
        "best_for": ["challenging code problems", "constraint-based coding", "code debugging challenges"],
        "prompt": """Apply Evol-Instruct evolutionary complexity:

TASK: {query}
CONTEXT: {context}

EVOLVE the instruction through:
1. ADD_CONSTRAINTS: Introduce time/space complexity requirements
2. ADD_DEBUGGING: Inject intentional bugs to identify and fix
3. INCREASE_REASONING_DEPTH: Add layers of logic and edge cases
4. CONCRETIZE: Add specific examples and detailed requirements
5. INCREASE_BREADTH: Consider alternative approaches and trade-offs

Now solve the EVOLVED problem:
- Implement solution meeting all evolved constraints
- Debug and verify correctness
- Optimize for complexity requirements"""
    },
    "llmloop": {
        "category": "code",
        "description": "Automated iterative feedback loops for code+tests",
        "best_for": ["code quality assurance", "test generation", "production-ready code"],
        "prompt": """Apply LLMLOOP automated iterative refinement:

TASK: {query}
CONTEXT: {context}

LOOP 1 - COMPILATION_ERRORS:
- Generate initial code
- Attempt compilation
- Fix syntax and type errors
- Repeat until compiles cleanly

LOOP 2 - STATIC_ANALYSIS:
- Run linter/static analyzer
- Fix warnings and code smells
- Apply best practices

LOOP 3 - TEST_FAILURES:
- Generate comprehensive test cases
- Run tests, identify failures
- Fix failing tests iteratively

LOOP 4 - MUTATION_TESTING:
- Apply mutation analysis to tests
- Improve test quality and coverage
- Ensure robustness

LOOP 5 - FINAL_REFINEMENT:
- Code review checklist
- Documentation
- Performance optimization"""
    },
    "procoder": {
        "category": "code",
        "description": "Compiler-feedback-guided iterative refinement",
        "best_for": ["project-level code generation", "large codebase integration", "API usage"],
        "prompt": """Apply ProCoder compiler-guided refinement:

TASK: {query}
CONTEXT: {context}

1. INITIAL_GENERATION: Generate code based on requirements
2. COMPILER_FEEDBACK: Attempt compilation/execution
   - Collect errors and warnings
   - Extract context from error messages
3. CONTEXT_ALIGNMENT:
   - Identify mismatches (undefined variables, wrong APIs, import errors)
   - Search project for correct patterns and APIs
   - Extract relevant code snippets from codebase
4. ITERATIVE_FIXING:
   - Fix errors using extracted project context
   - Re-compile and collect new feedback
   - Repeat until code compiles and runs
5. INTEGRATION_VERIFY: Ensure code fits project architecture"""
    },
    "recode": {
        "category": "code",
        "description": "Multi-candidate validation with CFG-based debugging",
        "best_for": ["reliable code generation", "execution debugging", "high-stakes code"],
        "prompt": """Apply RECODE multi-candidate cross-validation:

TASK: {query}
CONTEXT: {context}

1. MULTI_CANDIDATE_GENERATION: Generate 3-5 candidate solutions
2. SELF_TEST_GENERATION: Create test cases for each candidate
3. CROSS_VALIDATION:
   - Run each candidate's tests on ALL candidates
   - Use majority voting to select most reliable tests
   - Identify most robust solution candidate
4. STATIC_PATTERN_EXTRACTION:
   - Analyze common patterns across passing candidates
   - Extract best practices
5. CFG_DEBUGGING (if tests fail):
   - Build Control Flow Graph
   - Trace execution path through failing test
   - Identify exact branching/loop error
   - Provide fine-grained feedback for fix
6. ITERATIVE_REFINEMENT: Apply CFG insights, regenerate, re-test
7. FINAL_SOLUTION: Return cross-validated, debugged code"""
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
                "tools": 40 + 14,  # 40 think_* + 14 utility tools
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
    logger.info("Tools: 40 think_* + 1 reason + 14 utility = 55 total")
    logger.info("=" * 60)

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
