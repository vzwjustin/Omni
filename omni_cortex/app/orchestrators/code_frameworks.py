"""
Code Framework Orchestrators

Frameworks specialized for code generation, debugging, and testing.
"""

from typing import Dict, Any
from ..core.sampling import ClientSampler, extract_code_blocks, extract_score


async def program_of_thoughts(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Program of Thoughts: Step-by-step code reasoning
    """
    # Step 1: Understand the computation
    understand_prompt = f"""UNDERSTAND the computational problem:

{query}

Context: {context}

What's the input? What's the output? What transformations are needed?"""

    understanding = await sampler.request_sample(understand_prompt, temperature=0.5)

    # Step 2: Decompose into steps
    decompose_prompt = f"""DECOMPOSE into computational steps:

{understanding}

Break this into step-by-step operations. What needs to happen in sequence?"""

    steps = await sampler.request_sample(decompose_prompt, temperature=0.6)

    # Step 3: Write code with comments
    code_prompt = f"""Write code implementing each step:

{steps}

Provide clean, commented code for each step. Original problem: {query}"""

    code = await sampler.request_sample(code_prompt, temperature=0.5)

    # Step 4: Trace with sample input
    trace_prompt = f"""TRACE through the code with a sample input:

{code}

Walk through execution step-by-step. Verify correctness."""

    trace = await sampler.request_sample(trace_prompt, temperature=0.4)

    return {
        "final_answer": f"{code}\n\n---\n## Trace\n{trace}",
        "metadata": {"framework": "program_of_thoughts"}
    }


async def chain_of_verification(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Chain of Verification: Draft-Verify-Patch cycle
    """
    # Draft
    draft = await sampler.request_sample(
        f"Create initial solution:\n\n{query}\n\nContext: {context}",
        temperature=0.6
    )

    # Verify
    verify = await sampler.request_sample(
        f"""Verify this solution for issues:

{draft}

Check for:
- Security vulnerabilities (injection, XSS, etc.)
- Logic bugs
- Edge cases not handled
- Best practice violations

List all issues found.""",
        temperature=0.4
    )

    # Patch
    patched = await sampler.request_sample(
        f"""Fix all identified issues:

Issues: {verify}

Original code: {draft}

Provide corrected version.""",
        temperature=0.5
    )

    # Validate
    validate = await sampler.request_sample(
        f"""Final validation:

{patched}

Confirm: All issues fixed? No regressions? Production-ready?""",
        temperature=0.3
    )

    return {
        "final_answer": f"{patched}\n\n---\n## Validation\n{validate}",
        "metadata": {"framework": "chain_of_verification"}
    }


async def critic(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    CRITIC: Generate then critique
    """
    # Generate
    solution = await sampler.request_sample(
        f"Generate solution:\n\n{query}\n\nContext: {context}",
        temperature=0.6
    )

    # Critique
    critique = await sampler.request_sample(
        f"""Critique this solution:

{solution}

What works? What's missing? What could break? Be thorough.""",
        temperature=0.5
    )

    # Revise
    revised = await sampler.request_sample(
        f"""Address each criticism:

Critiques: {critique}

Original: {solution}

Provide improved version.""",
        temperature=0.5
    )

    # Final check
    final_check = await sampler.request_sample(
        f"Final review: {revised}\n\nAny remaining issues?",
        temperature=0.3
    )

    return {
        "final_answer": f"{revised}\n\n---\n## Review\n{final_check}",
        "metadata": {"framework": "critic"}
    }


async def chain_of_code(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Chain-of-Code: Code-based problem decomposition
    """
    steps = []

    # Decompose
    decompose = await sampler.request_sample(
        f"Break into code blocks:\n\n{query}\n\nContext: {context}\n\nList discrete code units needed.",
        temperature=0.6
    )
    steps.append(decompose)

    # Trace execution
    trace = await sampler.request_sample(
        f"Trace execution step-by-step:\n\n{decompose}\n\nWalk through what happens.",
        temperature=0.5
    )
    steps.append(trace)

    # Identify divergence
    identify = await sampler.request_sample(
        f"Where does logic diverge from intent?\n\n{trace}\n\nPinpoint the issue.",
        temperature=0.5
    )
    steps.append(identify)

    # Fix
    fix = await sampler.request_sample(
        f"Apply targeted fix:\n\n{identify}\n\nProvide corrected code with explanation.",
        temperature=0.5
    )
    steps.append(fix)

    # Verify
    verify = await sampler.request_sample(
        f"Trace corrected code:\n\n{fix}\n\nConfirm fix works.",
        temperature=0.4
    )

    return {
        "final_answer": f"{fix}\n\n---\n## Verification\n{verify}",
        "metadata": {"framework": "chain_of_code", "steps": len(steps)}
    }


async def self_debugging(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Self-Debugging: Mental execution trace before presenting
    """
    # Draft
    draft = await sampler.request_sample(
        f"Write initial solution:\n\n{query}\n\nContext: {context}",
        temperature=0.6
    )

    # Trace
    trace = await sampler.request_sample(
        f"Mentally execute with sample inputs:\n\n{draft}\n\nTrace through execution.",
        temperature=0.5
    )

    # Test edges
    edges = await sampler.request_sample(
        f"Test boundary conditions:\n\n{draft}\n\nTest: 0, 1, empty, null, max values. What breaks?",
        temperature=0.5
    )

    # Catch bugs
    catch = await sampler.request_sample(
        f"Identify potential bugs:\n\nTrace: {trace[:200]}...\nEdges: {edges[:200]}...\n\nWhat bugs exist?",
        temperature=0.5
    )

    # Fix and present
    fixed = await sampler.request_sample(
        f"Present corrected code:\n\nBugs found: {catch}\n\nOriginal: {draft}\n\nProvide debugged version.",
        temperature=0.5
    )

    return {
        "final_answer": fixed,
        "metadata": {"framework": "self_debugging"}
    }


async def tdd_prompting(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    TDD Prompting: Test-first development
    """
    # Write tests first
    tests = await sampler.request_sample(
        f"""Write test cases FIRST:

{query}

Context: {context}

Cover:
- Happy path
- Edge cases
- Error conditions

Provide comprehensive test suite.""",
        temperature=0.6
    )

    # Implement to pass tests
    implementation = await sampler.request_sample(
        f"Implement minimal code to pass these tests:\n\n{tests}\n\nOriginal requirement: {query}",
        temperature=0.5
    )

    # Refactor
    refactored = await sampler.request_sample(
        f"Refactor while keeping tests green:\n\n{implementation}\n\nClean up, improve readability.",
        temperature=0.5
    )

    # Verify
    verify = await sampler.request_sample(
        f"Verify all tests pass:\n\nTests: {tests[:300]}...\n\nCode: {refactored[:300]}...\n\nDo all tests pass?",
        temperature=0.3
    )

    return {
        "final_answer": f"{refactored}\n\n---\n## Tests\n{tests}",
        "metadata": {"framework": "tdd_prompting"}
    }


async def reverse_cot(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Reverse CoT: Backward reasoning from output delta
    """
    # Expected output
    expected = await sampler.request_sample(
        f"What SHOULD the output be?\n\n{query}\n\nContext: {context}\n\nDescribe expected behavior.",
        temperature=0.5
    )

    # Actual output
    actual = await sampler.request_sample(
        f"What IS the actual output?\n\n{context}\n\nDescribe current behavior.",
        temperature=0.5
    )

    # Delta
    delta = await sampler.request_sample(
        f"What's the difference?\n\nExpected: {expected}\nActual: {actual}\n\nPrecisely describe the delta.",
        temperature=0.5
    )

    # Backtrack
    backtrack = await sampler.request_sample(
        f"Work backward from the delta:\n\n{delta}\n\nWhat code change would cause this difference?",
        temperature=0.6
    )

    # Fix
    fix = await sampler.request_sample(
        f"Apply correction:\n\n{backtrack}\n\nProvide fixed code and verify expected output.",
        temperature=0.5
    )

    return {
        "final_answer": fix,
        "metadata": {"framework": "reverse_cot"}
    }


async def alphacodium(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    AlphaCodium: Test-based multi-stage iterative code generation
    """
    # Stage 1: Understand
    understand = await sampler.request_sample(
        f"""UNDERSTAND the problem:

{query}

Context: {context}

Parse:
- Input/output format
- Constraints
- Edge cases

Provide analysis.""",
        temperature=0.5
    )

    # Stage 2: Generate tests
    tests = await sampler.request_sample(
        f"""Generate test cases:

{understand}

Create:
- Public test cases
- Edge cases
- Large input cases

Provide test suite.""",
        temperature=0.6
    )

    # Stage 3: Iterative code generation
    max_iterations = 3
    solution = ""

    for i in range(max_iterations):
        # Generate solution
        solution = await sampler.request_sample(
            f"""Generate solution (iteration {i+1}):

Tests: {tests}

Problem: {query}

{f'Previous attempt had issues: {solution[:200]}' if solution else 'First attempt'}

Provide code.""",
            temperature=0.6
        )

        # Test
        test_result = await sampler.request_sample(
            f"Test solution:\n\n{solution}\n\nAgainst: {tests[:300]}...\n\nWhich tests pass/fail?",
            temperature=0.3
        )

        if "all pass" in test_result.lower() or "pass" in test_result.lower() and "fail" not in test_result.lower():
            break

    return {
        "final_answer": solution,
        "metadata": {"framework": "alphacodium", "iterations": i+1}
    }


async def codechain(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    CodeChain: Chain of self-revisions guided by sub-modules
    """
    # Identify modules
    modules = await sampler.request_sample(
        f"Identify independent sub-modules:\n\n{query}\n\nContext: {context}\n\nList modules with clear interfaces.",
        temperature=0.6
    )

    # Generate each module
    module_code = await sampler.request_sample(
        f"Generate each module:\n\n{modules}\n\nProvide code for each with clear interfaces.",
        temperature=0.6
    )

    # Integrate
    integrated = await sampler.request_sample(
        f"Integrate modules:\n\n{module_code}\n\nConnect them together.",
        temperature=0.5
    )

    # Self-critique and improve
    improved = await sampler.request_sample(
        f"Self-critique each module and improve:\n\n{integrated}\n\nWhat can be better?",
        temperature=0.5
    )

    # Validate
    validated = await sampler.request_sample(
        f"Test integrated solution:\n\n{improved}\n\nDoes it work end-to-end?",
        temperature=0.4
    )

    return {
        "final_answer": f"{improved}\n\n---\n## Validation\n{validated}",
        "metadata": {"framework": "codechain"}
    }


async def evol_instruct(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Evol-Instruct: Evolutionary instruction complexity
    """
    # Base solution
    base = await sampler.request_sample(
        f"Solve base problem:\n\n{query}\n\nContext: {context}",
        temperature=0.6
    )

    evolutions = [base]

    # Evolve 1: Add constraint
    evolved1 = await sampler.request_sample(
        f"""Evolve with additional constraint:

Base: {base}

Add: Performance constraint (O(n log n) or better)

Provide enhanced solution.""",
        temperature=0.6
    )
    evolutions.append(evolved1)

    # Evolve 2: Add edge cases
    evolved2 = await sampler.request_sample(
        f"""Further evolve with edge case handling:

Current: {evolved1[:300]}...

Add: Handle empty input, null, very large values

Provide robust solution.""",
        temperature=0.6
    )
    evolutions.append(evolved2)

    # Evolve 3: Add requirements
    evolved3 = await sampler.request_sample(
        f"""Final evolution:

Current: {evolved2[:300]}...

Add: Thread-safety, error handling, logging

Provide production-ready solution.""",
        temperature=0.6
    )

    return {
        "final_answer": evolved3,
        "metadata": {"framework": "evol_instruct", "evolutions": len(evolutions)}
    }


async def llmloop(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    LLMLoop: Automated iterative feedback loops for code+tests
    """
    max_iterations = 3

    for iteration in range(max_iterations):
        # Generate code + tests
        generated = await sampler.request_sample(
            f"""Generate code AND tests (iteration {iteration+1}):

{query}

Context: {context}

Provide both implementation and test suite.""",
            temperature=0.6
        )

        # Run tests (mental trace)
        test_results = await sampler.request_sample(
            f"Execute tests:\n\n{generated}\n\nMentally trace execution. What passes/fails?",
            temperature=0.4
        )

        # Analyze failures
        if "fail" not in test_results.lower():
            # All pass - do quality checks
            quality = await sampler.request_sample(
                f"Quality checks:\n\n{generated}\n\nCorrectness? Edges? Readability? Efficiency?",
                temperature=0.4
            )

            if all(word not in quality.lower() for word in ["issue", "problem", "missing", "poor"]):
                return {
                    "final_answer": generated,
                    "metadata": {"framework": "llmloop", "iterations": iteration+1}
                }

        # Fix issues
        analysis = await sampler.request_sample(
            f"Analyze failures:\n\n{test_results}\n\nWhy did tests fail?",
            temperature=0.5
        )

    return {
        "final_answer": generated,
        "metadata": {"framework": "llmloop", "iterations": max_iterations}
    }


async def procoder(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    ProCoder: Compiler-feedback-guided iterative refinement
    """
    # Analyze project context
    analysis = await sampler.request_sample(
        f"Analyze project context:\n\n{context}\n\nWhat types, APIs, patterns exist?",
        temperature=0.5
    )

    # Generate type-safe code
    code = await sampler.request_sample(
        f"""Generate type-safe code:

Project context: {analysis}

Task: {query}

Use project patterns and types.""",
        temperature=0.6
    )

    # Check for errors
    check = await sampler.request_sample(
        f"Check for errors:\n\n{code}\n\nType errors? Import issues? Linter warnings?",
        temperature=0.4
    )

    # Fix issues
    fixed = await sampler.request_sample(
        f"Fix compiler/linter feedback:\n\nIssues: {check}\n\nCode: {code}\n\nProvide corrected version.",
        temperature=0.5
    )

    # Verify integration
    integrated = await sampler.request_sample(
        f"Verify clean integration:\n\n{fixed}\n\nDoes it fit cleanly with existing code?",
        temperature=0.4
    )

    return {
        "final_answer": f"{fixed}\n\n---\n## Integration Check\n{integrated}",
        "metadata": {"framework": "procoder"}
    }


async def recode(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    RECODE: Multi-candidate validation with CFG-based debugging
    """
    # Generate 3 candidates
    candidates = []
    for i in range(3):
        candidate = await sampler.request_sample(
            f"Generate candidate solution #{i+1}:\n\n{query}\n\nContext: {context}",
            temperature=0.7 + i*0.1
        )
        candidates.append(candidate)

    # Validate each
    validations = []
    for i, candidate in enumerate(candidates):
        validation = await sampler.request_sample(
            f"Validate candidate #{i+1}:\n\n{candidate}\n\nTest independently. Correctness?",
            temperature=0.3
        )
        validations.append(validation)

    # Analyze control flow
    cfg_analysis = await sampler.request_sample(
        f"""Analyze control flow of each:

Candidate 1: {candidates[0][:200]}...
Candidate 2: {candidates[1][:200]}...
Candidate 3: {candidates[2][:200]}...

Check for control flow issues.""",
        temperature=0.4
    )

    # Vote
    vote = await sampler.request_sample(
        f"""Select best candidate:

Validations:
1. {validations[0][:150]}...
2. {validations[1][:150]}...
3. {validations[2][:150]}...

CFG: {cfg_analysis[:200]}...

Which is best based on correctness, robustness, clarity?""",
        temperature=0.4
    )

    # Extract winner number
    winner_idx = 0
    for i in range(3):
        if f"candidate #{i+1}" in vote.lower() or f"#{i+1}" in vote.lower() or str(i+1) in vote:
            winner_idx = i
            break

    return {
        "final_answer": f"{candidates[winner_idx]}\n\n---\n## Selection Reasoning\n{vote}",
        "metadata": {"framework": "recode", "candidates": 3}
    }


async def pal(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    PAL: Program-Aided Language (code as reasoning substrate)
    """
    # Translate to program
    translate = await sampler.request_sample(
        f"Translate reasoning to a program:\n\n{query}\n\nContext: {context}\n\nWhat code would solve this?",
        temperature=0.6
    )

    # Pseudocode first if complex
    pseudo = await sampler.request_sample(
        f"Start with pseudocode:\n\n{translate}\n\nWrite clear pseudocode steps.",
        temperature=0.5
    )

    # Implement
    implementation = await sampler.request_sample(
        f"Implement executable code:\n\n{pseudo}\n\nConvert to working code.",
        temperature=0.5
    )

    # Validate with examples
    validated = await sampler.request_sample(
        f"Test with examples:\n\n{implementation}\n\nTest normal + edge cases.",
        temperature=0.4
    )

    return {
        "final_answer": f"{implementation}\n\n---\n## Validation\n{validated}",
        "metadata": {"framework": "pal"}
    }


async def scratchpads(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Scratchpads: Structured intermediate reasoning workspace
    """
    # Build scratchpad
    scratchpad = {}

    # Facts
    scratchpad['facts'] = await sampler.request_sample(
        f"FACTS: Key known information\n\n{query}\n\nContext: {context}\n\nList facts.",
        temperature=0.5
    )

    # Constraints
    scratchpad['constraints'] = await sampler.request_sample(
        f"CONSTRAINTS: Must-do and cannot-do\n\n{query}\n\nList constraints.",
        temperature=0.5
    )

    # Plan
    scratchpad['plan'] = await sampler.request_sample(
        f"PLAN: Ordered approach\n\nFacts: {scratchpad['facts'][:150]}...\n\nConstraints: {scratchpad['constraints'][:150]}...\n\nWhat's the plan?",
        temperature=0.6
    )

    # Risks
    scratchpad['risks'] = await sampler.request_sample(
        f"RISKS: Potential issues + mitigations\n\nPlan: {scratchpad['plan'][:200]}...\n\nWhat could go wrong?",
        temperature=0.6
    )

    # Checks
    scratchpad['checks'] = await sampler.request_sample(
        f"CHECKS: Verification steps\n\n{scratchpad['plan'][:150]}...\n\nHow to verify correctness?",
        temperature=0.5
    )

    # Solve using scratchpad
    solution = await sampler.request_sample(
        f"""Solve using scratchpad:

FACTS: {scratchpad['facts'][:100]}...
CONSTRAINTS: {scratchpad['constraints'][:100]}...
PLAN: {scratchpad['plan'][:100]}...
RISKS: {scratchpad['risks'][:100]}...
CHECKS: {scratchpad['checks'][:100]}...

Original: {query}

Provide solution.""",
        temperature=0.5
    )

    return {
        "final_answer": f"{solution}\n\n---\n## Scratchpad Summary\n{chr(10).join(f'{k.upper()}: {v[:80]}...' for k, v in scratchpad.items())}",
        "metadata": {"framework": "scratchpads"}
    }


async def parsel(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Parsel: Compositional code from natural language specs
    """
    # Decompose into function specs
    specs = await sampler.request_sample(
        f"""Decompose into function specs:

{query}

Context: {context}

For each function:
- Name
- Description
- Inputs
- Outputs
- Dependencies

List all functions needed.""",
        temperature=0.6
    )

    # Build dependency graph
    graph = await sampler.request_sample(
        f"Build dependency order:\n\n{specs}\n\nOrder by dependencies (no cycles). Leaf functions first.",
        temperature=0.5
    )

    # Generate base functions
    base_funcs = await sampler.request_sample(
        f"Generate leaf functions first:\n\n{graph}\n\nImplement functions with no dependencies.",
        temperature=0.6
    )

    # Compose dependent functions
    composed = await sampler.request_sample(
        f"Build dependent functions:\n\nBase: {base_funcs}\n\nGraph: {graph[:200]}...\n\nImplement remaining functions.",
        temperature=0.6
    )

    # Integrate with entry point
    integrated = await sampler.request_sample(
        f"Integrate into cohesive module:\n\n{composed}\n\nProvide entry point and full integration.",
        temperature=0.5
    )

    return {
        "final_answer": integrated,
        "metadata": {"framework": "parsel"}
    }


async def docprompting(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    DocPrompting: Documentation-driven code generation
    """
    # Identify required APIs
    apis = await sampler.request_sample(
        f"Identify required APIs/libraries:\n\n{query}\n\nContext: {context}\n\nWhat APIs needed?",
        temperature=0.5
    )

    # Retrieve docs (simulated - would use RAG in practice)
    docs = await sampler.request_sample(
        f"Find relevant documentation:\n\n{apis}\n\nDescribe relevant docs, examples, patterns.",
        temperature=0.5
    )

    # Extract signatures and patterns
    signatures = await sampler.request_sample(
        f"Extract API details:\n\n{docs}\n\nFunction signatures, usage patterns, idioms.",
        temperature=0.5
    )

    # Generate following docs
    code = await sampler.request_sample(
        f"Generate code following docs:\n\nSignatures: {signatures}\n\nProblem: {query}\n\nUse doc patterns correctly.",
        temperature=0.6
    )

    # Verify against docs
    verified = await sampler.request_sample(
        f"Cross-check against docs:\n\n{code}\n\nDocs: {signatures[:200]}...\n\nCorrect params, types, error handling?",
        temperature=0.4
    )

    return {
        "final_answer": f"{code}\n\n---\n## Doc Verification\n{verified}",
        "metadata": {"framework": "docprompting"}
    }
