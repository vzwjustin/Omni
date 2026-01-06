"""
Omni-Cortex MCP Server

Exposes 60 thinking framework tools + utility tools.
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

# Configure structlog to use stdlib logging (required for ChromaDB/LangChain compatibility)
# PrintLoggerFactory lacks .disabled attribute that these libraries expect
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
    logger_factory=structlog.stdlib.LoggerFactory(),
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
    VectorstoreSearchError,
    enhance_state_with_langchain,
    AVAILABLE_TOOLS,
    OmniCortexCallback,
)
from app.collection_manager import get_collection_manager
from app.core.router import HyperRouter
from app.core.vibe_dictionary import VIBE_DICTIONARY
from app.core.context_gateway import get_context_gateway, StructuredContext
from app.core.context_utils import (
    count_tokens,
    compress_content,
    detect_truncation,
    analyze_claude_md,
    generate_claude_md_template,
    inject_rules,
    RULE_PRESETS,
)

import os
# LEAN_MODE: Only expose essential tools (reason + utilities) to reduce MCP token overhead
# Set LEAN_MODE=false to expose all 55+ think_* tools individually
LEAN_MODE = os.environ.get("LEAN_MODE", "true").lower() in ("true", "1", "yes")

# Import MCP sampling and orchestrators
from app.core.sampling import (
    ClientSampler,
    SamplingNotSupportedError,
    call_llm_with_fallback,
    LANGCHAIN_LLM_ENABLED,
)
from app.orchestrators import FRAMEWORK_ORCHESTRATORS

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
    # =========================================================================
    # VERIFICATION FRAMEWORKS (2026 Expansion)
    # =========================================================================
    "self_consistency": {
        "category": "verification",
        "description": "Multi-sample voting for reliable answers",
        "best_for": ["ambiguous bugs", "tricky logic", "multiple plausible fixes"],
        "prompt": """Apply Self-Consistency:

TASK: {query}
CONTEXT: {context}

1. GENERATE: Create 3-7 independent solution paths (don't reuse reasoning)
2. NORMALIZE: Structure as (hypothesis -> fix -> expected evidence)
3. SCORE: Rate consistency, constraint fit, simplicity, testability
4. SELECT: Choose winner; keep runner-up if high risk
5. OUTPUT: Final solution + why it won + validation checks"""
    },
    "self_ask": {
        "category": "verification",
        "description": "Sub-question decomposition before solving",
        "best_for": ["unclear tickets", "missing requirements", "multi-part debugging"],
        "prompt": """Apply Self-Ask:

TASK: {query}
CONTEXT: {context}

1. GENERATE: List 5-12 sub-questions that must be answered
2. CLASSIFY: Mark each as must-know vs nice-to-know
3. ANSWER: Address must-know using context, tools, or docs
4. RECOMPOSE: Build final solution with stated assumptions
5. VALIDATE: Check against acceptance criteria"""
    },
    "rar": {
        "category": "verification",
        "description": "Rephrase-and-Respond for clarity",
        "best_for": ["vague prompts", "poorly written bug reports", "ambiguous requirements"],
        "prompt": """Apply Rephrase-and-Respond (RaR):

TASK: {query}
CONTEXT: {context}

1. REPHRASE: Write precise task spec (objective, constraints, acceptance criteria)
2. CONFIRM: Check for internal consistency
3. SOLVE: Implement strictly against rephrased spec
4. VERIFY: Map results to acceptance criteria"""
    },
    "verify_and_edit": {
        "category": "verification",
        "description": "Verify claims, edit only failures",
        "best_for": ["code review", "security guidance", "implementation plans", "surgical edits"],
        "prompt": """Apply Verify-and-Edit:

TASK: {query}
CONTEXT: {context}

1. DRAFT: Create initial solution
2. EXTRACT: Identify verifiable claims and risky assertions
3. VERIFY: Check each via context, tests, docs (mark assumptions)
4. EDIT: Fix ONLY failing sections; preserve good sections
5. FINALIZE: Produce verification ledger"""
    },
    "rarr": {
        "category": "verification",
        "description": "Research, Augment, Revise - evidence-driven revision",
        "best_for": ["external docs", "repo knowledge", "prove-it requirements"],
        "prompt": """Apply RARR (Research, Augment, Revise):

TASK: {query}
CONTEXT: {context}

1. DRAFT: Create initial output
2. GENERATE: Create 3-8 targeted evidence queries
3. RETRIEVE: Gather relevant evidence
4. REVISE: Update to align with evidence; remove unsupported claims
5. CITE: Provide anchors (file:line when possible)"""
    },
    "selfcheckgpt": {
        "category": "verification",
        "description": "Hallucination detection via sampling consistency",
        "best_for": ["high-stakes guidance", "unfamiliar libraries", "final pre-flight gate"],
        "prompt": """Apply SelfCheckGPT:

TASK: {query}
CONTEXT: {context}

1. IDENTIFY: Flag high-risk claims in draft
2. SAMPLE: Generate multiple re-answers focusing on risky claims
3. CHECK: Compare answers; flag disagreement hotspots
4. REPLACE: Disputed content with verified evidence or explicit uncertainty
5. OUTPUT: Final result + risk register"""
    },
    "metaqa": {
        "category": "verification",
        "description": "Metamorphic testing for reasoning reliability",
        "best_for": ["brittle reasoning", "edge cases", "policy consistency"],
        "prompt": """Apply MetaQA (Metamorphic QA):

TASK: {query}
CONTEXT: {context}

1. DEFINE: Invariants that must stay true
2. GENERATE: 3-10 transformed variants (rewording, constraint tweaks)
3. SOLVE: Answer each variant
4. CHECK: Identify contradictions between answers
5. PATCH: Fix solution to satisfy invariants across variants"""
    },
    "ragas": {
        "category": "verification",
        "description": "RAG Assessment - evaluate retrieval quality",
        "best_for": ["RAG pipelines", "retrieval quality", "source grounding"],
        "prompt": """Apply RAGAS (RAG Assessment):

TASK: {query}
CONTEXT: {context}

Evaluate across dimensions:
1. RELEVANCE: Are retrieved chunks relevant?
2. FAITHFULNESS: Does answer stick to sources?
3. COMPLETENESS: All aspects covered?
4. NOISE: Any irrelevant/misleading content?
5. DIAGNOSE: Failure modes + corrective actions"""
    },
    # =========================================================================
    # AGENT FRAMEWORKS (2026 Expansion)
    # =========================================================================
    "rewoo": {
        "category": "agent",
        "description": "Reasoning Without Observation - plan then execute",
        "best_for": ["multi-step tasks", "cost control", "plan-once-execute-clean"],
        "prompt": """Apply ReWOO (Reasoning Without Observation):

TASK: {query}
CONTEXT: {context}

1. PLAN: Create tool-free plan with expected observations
2. SCHEDULE: Convert to tool call schedule (what, when, why)
3. EXECUTE: Run tools in batches; collect observations
4. REVISE: Update plan only if observations contradict
5. FINALIZE: Result + checks + next actions"""
    },
    "lats": {
        "category": "agent",
        "description": "Language Agent Tree Search over action sequences",
        "best_for": ["complex repo changes", "multiple fix paths", "uncertain root cause"],
        "prompt": """Apply LATS (Language Agent Tree Search):

TASK: {query}
CONTEXT: {context}

1. PRIMITIVES: Define actions (edit, test, inspect, search)
2. EXPAND: Generate multiple action sequence branches
3. SCORE: Rate by likelihood, risk, effort, rollback ease
4. EXECUTE: Run best branch; backtrack if fails
5. FINALIZE: Chosen path + alternatives considered"""
    },
    "mrkl": {
        "category": "agent",
        "description": "Modular Reasoning with specialized modules",
        "best_for": ["big systems", "mixed domains", "tool-rich setups"],
        "prompt": """Apply MRKL (Modular Reasoning, Knowledge, Language):

TASK: {query}
CONTEXT: {context}

1. DECOMPOSE: Break into module tasks (Security, Perf, Test, Product)
2. ROUTE: Specify input/output/validation for each module
3. EXECUTE: Run modules with clear interfaces
4. RECONCILE: Resolve conflicting outputs
5. SYNTHESIZE: Combine into final decision + verification plan"""
    },
    "swe_agent": {
        "category": "agent",
        "description": "Repo-first execution loop - inspect/edit/run/iterate",
        "best_for": ["multi-file bugfixes", "CI failures", "make tests pass"],
        "prompt": """Apply SWE-Agent:

TASK: {query}
CONTEXT: {context}

1. INSPECT: Entry points, failing tests, logs, config
2. IDENTIFY: Minimal change set
3. PATCH: Apply changes in small increments
4. VERIFY: Run tests/lint/typecheck
5. ITERATE: Until green, then summarize + remaining risks"""
    },
    "toolformer": {
        "category": "agent",
        "description": "Smart tool selection policy",
        "best_for": ["router logic", "preventing pointless calls", "standardized prompts"],
        "prompt": """Apply Toolformer (Tool Selection Policy):

TASK: {query}
CONTEXT: {context}

1. IDENTIFY: What claims require external confirmation?
2. JUSTIFY: Does tool call materially reduce uncertainty?
3. OPTIMIZE: Specify tight inputs + expected outputs
4. INTEGRATE: Parse results, update confidence
5. DOCUMENT: Tool decision rationale"""
    },
    # =========================================================================
    # RAG FRAMEWORKS (2026 Expansion)
    # =========================================================================
    "self_rag": {
        "category": "rag",
        "description": "Self-triggered selective retrieval",
        "best_for": ["mixed knowledge tasks", "large corpora", "minimizing irrelevant retrieval"],
        "prompt": """Apply Self-RAG:

TASK: {query}
CONTEXT: {context}

1. DRAFT: Write with confidence tags (HIGH/MEDIUM/LOW)
2. IDENTIFY: Gaps in LOW-confidence segments
3. RETRIEVE: Fetch evidence only for uncertain parts
4. UPDATE: Revise only uncertain segments
5. CRITIQUE: Confirm groundedness; remove unsupported claims"""
    },
    "hyde": {
        "category": "rag",
        "description": "Hypothetical Document Embeddings for better retrieval",
        "best_for": ["fuzzy search", "unclear intent", "broad problems"],
        "prompt": """Apply HyDE (Hypothetical Document Embeddings):

TASK: {query}
CONTEXT: {context}

1. HYPOTHESIZE: Write ideal answer document
2. EXTRACT: Convert to strong retrieval queries
3. RETRIEVE: Find real documents/snippets
4. GROUND: Answer based on retrieved evidence
5. CITE: Evidence anchors for claims"""
    },
    "rag_fusion": {
        "category": "rag",
        "description": "Multi-query retrieval with rank fusion",
        "best_for": ["improving recall", "complex queries", "noisy corpora"],
        "prompt": """Apply RAG-Fusion:

TASK: {query}
CONTEXT: {context}

1. GENERATE: 3-8 diverse queries (synonyms, facets, constraints)
2. RETRIEVE: Top-K per query
3. FUSE: Dedupe + reciprocal rank merge
4. SYNTHESIZE: Answer using fused evidence
5. COVERAGE: Ensure all query facets addressed"""
    },
    "raptor": {
        "category": "rag",
        "description": "Hierarchical abstraction retrieval for large docs",
        "best_for": ["huge repos", "long design docs", "monorepos"],
        "prompt": """Apply RAPTOR (Recursive Abstraction Retrieval):

TASK: {query}
CONTEXT: {context}

1. HIERARCHY: Summaries at chunk/section/doc levels
2. RETRIEVE TOP-DOWN: High-level first, then drill down
3. GATHER: Supporting details from lower levels
4. SYNTHESIZE: Combine abstraction with specifics
5. ANCHOR: Both overview context and specific citations"""
    },
    "graphrag": {
        "category": "rag",
        "description": "Entity-relation grounding for dependencies",
        "best_for": ["architecture questions", "module relationships", "impact analysis"],
        "prompt": """Apply GraphRAG:

TASK: {query}
CONTEXT: {context}

1. EXTRACT: Entities (modules, APIs, tables, services)
2. MAP: Relations (calls, reads/writes, owns, triggers)
3. GRAPH: Build conceptual relation map
4. QUERY: Trace paths, find blast radius, identify dependencies
5. CITE: Show relationship chains supporting claims"""
    },
    # =========================================================================
    # CODE FRAMEWORKS (2026 Expansion - Additional)
    # =========================================================================
    "pal": {
        "category": "code",
        "description": "Program-Aided Language - code as reasoning substrate",
        "best_for": ["algorithms", "parsing", "numeric logic", "validation"],
        "prompt": """Apply PAL (Program-Aided Language):

TASK: {query}
CONTEXT: {context}

1. TRANSLATE: Convert reasoning to a small program
2. PSEUDOCODE: Start with pseudocode if unsure
3. IMPLEMENT: Write executable code
4. VALIDATE: Test with examples (normal + edge cases)
5. CONVERT: Translate verified code to final solution"""
    },
    "scratchpads": {
        "category": "code",
        "description": "Structured intermediate reasoning workspace",
        "best_for": ["multi-step fixes", "multi-constraint reasoning", "state tracking"],
        "prompt": """Apply Scratchpads:

TASK: {query}
CONTEXT: {context}

Maintain structured scratchpad:
- FACTS: Key known information
- CONSTRAINTS: Must-do and cannot-do
- PLAN: Ordered approach
- RISKS: Potential issues + mitigations
- CHECKS: Verification steps

Update as you work; present final with scratchpad summary."""
    },
    "parsel": {
        "category": "code",
        "description": "Compositional code generation from natural language specs",
        "best_for": ["complex functions", "dependency graphs", "spec-to-code", "modular systems"],
        "prompt": """Apply Parsel (Compositional Code Generation):

TASK: {query}
CONTEXT: {context}

1. DECOMPOSE: Break task into individual function specs
   - Name, description, inputs, outputs, dependencies
2. GRAPH: Build dependency order (no cycles)
3. BASE: Generate leaf functions first (no dependencies)
4. COMPOSE: Build dependent functions using generated ones
5. INTEGRATE: Combine into cohesive module with entry point"""
    },
    "docprompting": {
        "category": "code",
        "description": "Documentation-driven code generation",
        "best_for": ["API usage", "library integration", "following docs", "correct usage"],
        "prompt": """Apply DocPrompting (Documentation-Driven):

TASK: {query}
CONTEXT: {context}

1. IDENTIFY: List required APIs/libraries
2. RETRIEVE: Find relevant documentation + examples
3. EXTRACT: Note function signatures, usage patterns, idioms
4. GENERATE: Write code following doc patterns
5. VERIFY: Cross-check against doc for correct params, types, error handling"""
    },
}


def create_server() -> Server:
    """Create the MCP server with all tools."""
    server = Server("omni-cortex")

    # Initialize client sampler for multi-turn orchestration
    sampler = ClientSampler(server)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        tools = []

        # =======================================================================
        # LEAN_MODE (default): Ultra-lean - only 4 essential tools
        # Gemini handles context prep, file discovery, RAG, and framework selection
        # All 62 frameworks + full feature set available internally
        # =======================================================================
        # Set LEAN_MODE=false to expose all 77 tools (62 think_* + 15 utilities)
        # =======================================================================

        if LEAN_MODE:
            # ULTRA-LEAN: 4 tools - Gemini does the heavy lifting
            # Full feature set (62 frameworks, RAG, memory) available internally

            # 1. Context Gateway - Gemini prepares everything
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

            # 2. Smart reasoning - executes with auto-selected framework
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

            # 3. Code execution - run and test code
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

            # 4. Health check
            tools.append(Tool(
                name="health",
                description="Check server health, available frameworks, and capabilities",
                inputSchema={"type": "object", "properties": {}}
            ))

            # 5-9. Context optimization tools (merged from context-optimizer)
            tools.append(Tool(
                name="count_tokens",
                description="Count tokens in text using Claude's tokenizer",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to count tokens for"}
                    },
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
                    "properties": {
                        "text": {"type": "string", "description": "Text to check"}
                    },
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

        # =======================================================================
        # FULL MODE: All 77 tools exposed (62 think_* + 15 utilities)
        # =======================================================================

        # 62 Framework tools (think_*) - direct access to each framework
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

        # Context Gateway - Gemini-powered context preparation for Claude
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
        logger.debug("call_tool", name=name, args=list(arguments.keys()))

        manager = get_collection_manager()

        # Smart routing with HyperRouter (reason tool) - Returns structured brief
        if name == "reason":
            query = arguments.get("query", "")
            context = arguments.get("context", "")
            thread_id = arguments.get("thread_id")

            # Use singleton router imported from app.graph
            hyper_router = router

            # Generate structured brief using the new protocol
            try:
                router_output = await hyper_router.generate_structured_brief(
                    query=query,
                    context=context,
                    code_snippet=None,
                    ide_context=None,
                    file_list=None
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

            except Exception as e:
                # Fallback to simple template mode if structured brief fails
                import traceback
                import sys
                err_detail = traceback.format_exc()
                logger.warning(f"Structured brief generation failed: {e}\n{err_detail}")
                # Write to stderr for visibility
                print(f"BRIEF FAILED: {e}\n{err_detail}", file=sys.stderr)

                selected = hyper_router._check_vibe_dictionary(query)
                if not selected:
                    selected = hyper_router._heuristic_select(query, context)
                if not selected or selected not in FRAMEWORKS:
                    selected = "self_discover"

                fw_info = hyper_router.get_framework_info(selected)
                complexity = hyper_router.estimate_complexity(query, context if context != "None provided" else None)

                fw = FRAMEWORKS.get(selected, {"category": "unknown", "best_for": [], "prompt": "Apply your best reasoning to: {query}\n\nContext: {context}"})
                prompt = fw["prompt"].format(query=query, context=context or "None provided")

                output = f"# Framework: {selected}\n"
                output += f"Category: {fw_info.get('category', fw.get('category', 'unknown'))} | Complexity: {complexity:.2f}\n"
                output += f"Best for: {', '.join(fw_info.get('best_for', fw.get('best_for', [])))}\n"
                output += "\n---\n\n"
                output += prompt

            return [TextContent(type="text", text=output)]

        # Framework tools (think_*) - Try orchestrator, fallback to template mode
        if name.startswith("think_"):
            fw_name = name[6:]
            query = arguments.get("query", "")
            context = arguments.get("context", "")
            thread_id = arguments.get("thread_id")

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

        # List frameworks
        if name == "list_frameworks":
            output = f"# Omni-Cortex: {len(FRAMEWORKS)} Thinking Frameworks\n\n"
            for cat in ["strategy", "search", "iterative", "code", "context", "fast", "verification", "agent", "rag"]:
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
            try:
                thread_id = arguments.get("thread_id")
                query = arguments.get("query")
                answer = arguments.get("answer")
                framework = arguments.get("framework")

                if not all([thread_id, query, answer, framework]):
                    missing = [k for k, v in {"thread_id": thread_id, "query": query, "answer": answer, "framework": framework}.items() if not v]
                    return [TextContent(type="text", text=f"Missing required arguments: {', '.join(missing)}")]

                await save_to_langchain_memory(thread_id, query, answer, framework)
                return [TextContent(type="text", text="Context saved successfully")]
            except Exception as e:
                logger.error("save_context_failed", error=str(e))
                return [TextContent(type="text", text=f"Failed to save context: {str(e)}")]

        # RAG: search documentation
        if name == "search_documentation":
            query = arguments.get("query", "")
            k = arguments.get("k", 5)
            try:
                docs = search_vectorstore(query, k=k)
            except VectorstoreSearchError as exc:
                logger.error("search_documentation_failed: %s", exc)
                return [TextContent(type="text", text=f"Search failed: {exc}")]
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
            valid = ["strategy", "search", "iterative", "code", "context", "fast", "verification", "agent", "rag"]
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
            if LEAN_MODE:
                # Ultra-lean: 8 tools (4 core + 4 context utils)
                exposed_tools = 8
                note = "ULTRA-LEAN MODE: 8 tools exposed (prepare_context, reason, execute_code, health + count_tokens, compress_content, detect_truncation, manage_claude_md). Gemini handles context prep, 62 frameworks available internally."
            else:
                # Full mode: 62 think_* + 19 utilities = 81 tools
                exposed_tools = len(FRAMEWORKS) + 19
                note = "FULL MODE: All 81 tools exposed (62 think_* + 19 utilities)"
            return [TextContent(type="text", text=json.dumps({
                "status": "healthy",
                "mode": "ultra-lean" if LEAN_MODE else "full",
                "tools_exposed": exposed_tools,
                "frameworks_available": len(FRAMEWORKS),
                "gemini_context_gateway": LEAN_MODE,
                "collections": collections,
                "memory_enabled": True,
                "rag_enabled": True,
                "note": note
            }, indent=2))]

        # Context Gateway - Gemini-powered context preparation
        if name == "prepare_context":
            query = arguments.get("query", "")
            workspace_path = arguments.get("workspace_path")
            code_context = arguments.get("code_context")
            file_list = arguments.get("file_list")
            search_docs = arguments.get("search_docs", True)
            output_format = arguments.get("output_format", "prompt")

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
                    # Return rich, structured prompt ready for Claude
                    output = "# Context Prepared by Gemini\n\n"
                    output += context.to_claude_prompt()
                    return [TextContent(type="text", text=output)]

            except Exception as e:
                logger.error(f"Context gateway failed: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "error": str(e),
                    "hint": "Ensure GOOGLE_API_KEY is set for Gemini-powered context preparation"
                }, indent=2))]

        # Context optimization tools (merged from context-optimizer)
        if name == "count_tokens":
            text = arguments.get("text", "")
            tokens = count_tokens(text)
            return [TextContent(type="text", text=json.dumps({"tokens": tokens, "characters": len(text)}))]

        if name == "compress_content":
            content = arguments.get("content", "")
            target = arguments.get("target_reduction", 0.3)
            result = compress_content(content, target)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        if name == "detect_truncation":
            text = arguments.get("text", "")
            result = detect_truncation(text)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        if name == "manage_claude_md":
            action = arguments.get("action", "analyze")

            if action == "analyze":
                directory = arguments.get("directory", os.getcwd())
                result = analyze_claude_md(directory)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif action == "generate":
                project_type = arguments.get("project_type", "general")
                rules = arguments.get("rules", [])
                presets = arguments.get("presets", [])
                result = generate_claude_md_template(project_type, rules, presets)
                return [TextContent(type="text", text=result)]

            elif action == "inject":
                existing = arguments.get("existing_content", "")
                rules = arguments.get("rules", [])
                presets = arguments.get("presets", [])
                section = arguments.get("section", "Rules")
                result = inject_rules(existing, rules, section, presets)
                return [TextContent(type="text", text=result)]

            elif action == "list_presets":
                result = {name: {"rules": rules, "count": len(rules)} for name, rules in RULE_PRESETS.items()}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            return [TextContent(type="text", text=f"Unknown action: {action}")]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

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
