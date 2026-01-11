"""
Framework Factory: Eliminates copy-paste orchestrator code.

Define frameworks as configuration, execute them uniformly.
This reduces the 17 nearly-identical functions in code_frameworks.py
to simple declarative configurations.

Usage:
    config = PROGRAM_OF_THOUGHTS
    result = await execute_framework(config, sampler, query, context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.sampling import ClientSampler


@dataclass
class FrameworkStep:
    """A single step in a framework orchestration.

    Attributes:
        name: Unique identifier for this step (used in template substitution)
        prompt_template: Prompt string with placeholders like {query}, {context}, {previous}
        temperature: LLM temperature for this step (default 0.5)
        uses_previous: Whether to include previous step output as {previous}
        max_tokens: Optional max tokens for this step's response
    """

    name: str
    prompt_template: str
    temperature: float = 0.5
    uses_previous: bool = True
    max_tokens: int | None = None


@dataclass
class FrameworkConfig:
    """Configuration for a complete framework orchestration.

    Attributes:
        name: Framework identifier (e.g., "program_of_thoughts")
        steps: Ordered list of FrameworkStep objects
        final_template: Template for combining step outputs into final answer.
                       Has access to all step names as {step_name} plus {last_step}
        description: Human-readable description of what this framework does
        metadata_extras: Additional metadata to include in response
    """

    name: str
    steps: list[FrameworkStep]
    final_template: str = "{last_step}"
    description: str = ""
    metadata_extras: dict[str, Any] = field(default_factory=dict)


async def execute_framework(
    config: FrameworkConfig, sampler: ClientSampler, query: str, context: str
) -> dict[str, Any]:
    """Execute a framework based on its configuration.

    Args:
        config: The FrameworkConfig defining the orchestration
        sampler: ClientSampler for making LLM requests
        query: User's query/problem statement
        context: Relevant code context

    Returns:
        Dict with 'final_answer' and 'metadata' keys
    """
    step_outputs: dict[str, str] = {}
    last_output = ""

    for step in config.steps:
        # Build prompt with substitutions
        # Available placeholders: {query}, {context}, {previous}, and all previous step names
        format_args = {
            "query": query,
            "context": context,
            "previous": last_output if step.uses_previous else "",
            **step_outputs,
        }

        try:
            prompt = step.prompt_template.format(**format_args)
        except KeyError as e:
            # If a placeholder is missing, provide empty string
            missing_key = str(e).strip("'")
            format_args[missing_key] = ""
            prompt = step.prompt_template.format(**format_args)

        # Make the LLM request
        kwargs = {"temperature": step.temperature}
        if step.max_tokens:
            kwargs["max_tokens"] = step.max_tokens

        response = await sampler.request_sample(prompt, **kwargs)

        step_outputs[step.name] = response
        last_output = response

    # Format final answer using template
    final_format_args = {**step_outputs, "last_step": last_output}
    final = config.final_template.format(**final_format_args)

    # Build metadata
    metadata = {"framework": config.name, "steps": len(config.steps), **config.metadata_extras}

    return {"final_answer": final, "metadata": metadata}


# =============================================================================
# Framework Configurations
# =============================================================================

PROGRAM_OF_THOUGHTS = FrameworkConfig(
    name="program_of_thoughts",
    description="Step-by-step code reasoning with execution trace",
    steps=[
        FrameworkStep(
            name="understand",
            prompt_template="""UNDERSTAND the computational problem:

{query}

Context: {context}

What's the input? What's the output? What transformations are needed?""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="decompose",
            prompt_template="""DECOMPOSE into computational steps:

{previous}

Break this into step-by-step operations. What needs to happen in sequence?""",
            temperature=0.6,
        ),
        FrameworkStep(
            name="code",
            prompt_template="""Write code implementing each step:

{previous}

Provide clean, commented code for each step. Original problem: {query}""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="trace",
            prompt_template="""TRACE through the code with a sample input:

{previous}

Walk through execution step-by-step. Verify correctness.""",
            temperature=0.4,
        ),
    ],
    final_template="{code}\n\n---\n## Trace\n{trace}",
)


CHAIN_OF_VERIFICATION = FrameworkConfig(
    name="chain_of_verification",
    description="Draft-Verify-Patch cycle for robust solutions",
    steps=[
        FrameworkStep(
            name="draft",
            prompt_template="""Create initial solution:

{query}

Context: {context}""",
            temperature=0.6,
            uses_previous=False,
        ),
        FrameworkStep(
            name="verify",
            prompt_template="""Verify this solution for issues:

{previous}

Check for:
- Security vulnerabilities (injection, XSS, etc.)
- Logic bugs
- Edge cases not handled
- Best practice violations

List all issues found.""",
            temperature=0.4,
        ),
        FrameworkStep(
            name="patch",
            prompt_template="""Fix all identified issues:

Issues: {verify}

Original code: {draft}

Provide corrected version.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="validate",
            prompt_template="""Final validation:

{previous}

Confirm: All issues fixed? No regressions? Production-ready?""",
            temperature=0.3,
        ),
    ],
    final_template="{patch}\n\n---\n## Validation\n{validate}",
)


CRITIC = FrameworkConfig(
    name="critic",
    description="Generate then critique approach for iterative refinement",
    steps=[
        FrameworkStep(
            name="solution",
            prompt_template="""Generate solution:

{query}

Context: {context}""",
            temperature=0.6,
            uses_previous=False,
        ),
        FrameworkStep(
            name="critique",
            prompt_template="""Critique this solution:

{previous}

What works? What's missing? What could break? Be thorough.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="revised",
            prompt_template="""Address each criticism:

Critiques: {critique}

Original: {solution}

Provide improved version.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="final_check",
            prompt_template="""Final review: {previous}

Any remaining issues?""",
            temperature=0.3,
        ),
    ],
    final_template="{revised}\n\n---\n## Review\n{final_check}",
)


CHAIN_OF_CODE = FrameworkConfig(
    name="chain_of_code",
    description="Code-based problem decomposition with trace-based debugging",
    steps=[
        FrameworkStep(
            name="decompose",
            prompt_template="""Break into code blocks:

{query}

Context: {context}

List discrete code units needed.""",
            temperature=0.6,
            uses_previous=False,
        ),
        FrameworkStep(
            name="trace",
            prompt_template="""Trace execution step-by-step:

{previous}

Walk through what happens.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="identify",
            prompt_template="""Where does logic diverge from intent?

{previous}

Pinpoint the issue.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="fix",
            prompt_template="""Apply targeted fix:

{previous}

Provide corrected code with explanation.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="verify",
            prompt_template="""Trace corrected code:

{previous}

Confirm fix works.""",
            temperature=0.4,
        ),
    ],
    final_template="{fix}\n\n---\n## Verification\n{verify}",
)


SELF_DEBUGGING = FrameworkConfig(
    name="self_debugging",
    description="Mental execution trace before presenting solution",
    steps=[
        FrameworkStep(
            name="draft",
            prompt_template="""Write initial solution:

{query}

Context: {context}""",
            temperature=0.6,
            uses_previous=False,
        ),
        FrameworkStep(
            name="trace",
            prompt_template="""Mentally execute with sample inputs:

{previous}

Trace through execution.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="edges",
            prompt_template="""Test boundary conditions:

{draft}

Test: 0, 1, empty, null, max values. What breaks?""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="catch",
            prompt_template="""Identify potential bugs:

Trace: {trace}
Edges: {edges}

What bugs exist?""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="fixed",
            prompt_template="""Present corrected code:

Bugs found: {catch}

Original: {draft}

Provide debugged version.""",
            temperature=0.5,
        ),
    ],
    final_template="{fixed}",
)


TDD_PROMPTING = FrameworkConfig(
    name="tdd_prompting",
    description="Test-first development approach",
    steps=[
        FrameworkStep(
            name="tests",
            prompt_template="""Write test cases FIRST:

{query}

Context: {context}

Cover:
- Happy path
- Edge cases
- Error conditions

Provide comprehensive test suite.""",
            temperature=0.6,
            uses_previous=False,
        ),
        FrameworkStep(
            name="implementation",
            prompt_template="""Implement minimal code to pass these tests:

{previous}

Original requirement: {query}""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="refactored",
            prompt_template="""Refactor while keeping tests green:

{previous}

Clean up, improve readability.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="verify",
            prompt_template="""Verify all tests pass:

Tests: {tests}

Code: {refactored}

Do all tests pass?""",
            temperature=0.3,
        ),
    ],
    final_template="{refactored}\n\n---\n## Tests\n{tests}",
)


REVERSE_COT = FrameworkConfig(
    name="reverse_cot",
    description="Backward reasoning from expected output delta",
    steps=[
        FrameworkStep(
            name="expected",
            prompt_template="""What SHOULD the output be?

{query}

Context: {context}

Describe expected behavior.""",
            temperature=0.5,
            uses_previous=False,
        ),
        FrameworkStep(
            name="actual",
            prompt_template="""What IS the actual output?

{context}

Describe current behavior.""",
            temperature=0.5,
            uses_previous=False,
        ),
        FrameworkStep(
            name="delta",
            prompt_template="""What's the difference?

Expected: {expected}
Actual: {actual}

Precisely describe the delta.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="backtrack",
            prompt_template="""Work backward from the delta:

{previous}

What code change would cause this difference?""",
            temperature=0.6,
        ),
        FrameworkStep(
            name="fix",
            prompt_template="""Apply correction:

{previous}

Provide fixed code and verify expected output.""",
            temperature=0.5,
        ),
    ],
    final_template="{fix}",
)


CODECHAIN = FrameworkConfig(
    name="codechain",
    description="Chain of self-revisions guided by sub-modules",
    steps=[
        FrameworkStep(
            name="modules",
            prompt_template="""Identify independent sub-modules:

{query}

Context: {context}

List modules with clear interfaces.""",
            temperature=0.6,
            uses_previous=False,
        ),
        FrameworkStep(
            name="module_code",
            prompt_template="""Generate each module:

{previous}

Provide code for each with clear interfaces.""",
            temperature=0.6,
        ),
        FrameworkStep(
            name="integrated",
            prompt_template="""Integrate modules:

{previous}

Connect them together.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="improved",
            prompt_template="""Self-critique each module and improve:

{previous}

What can be better?""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="validated",
            prompt_template="""Test integrated solution:

{previous}

Does it work end-to-end?""",
            temperature=0.4,
        ),
    ],
    final_template="{improved}\n\n---\n## Validation\n{validated}",
)


EVOL_INSTRUCT = FrameworkConfig(
    name="evol_instruct",
    description="Evolutionary instruction complexity for robust solutions",
    steps=[
        FrameworkStep(
            name="base",
            prompt_template="""Solve base problem:

{query}

Context: {context}""",
            temperature=0.6,
            uses_previous=False,
        ),
        FrameworkStep(
            name="evolved1",
            prompt_template="""Evolve with additional constraint:

Base: {previous}

Add: Performance constraint (O(n log n) or better)

Provide enhanced solution.""",
            temperature=0.6,
        ),
        FrameworkStep(
            name="evolved2",
            prompt_template="""Further evolve with edge case handling:

Current: {previous}

Add: Handle empty input, null, very large values

Provide robust solution.""",
            temperature=0.6,
        ),
        FrameworkStep(
            name="evolved3",
            prompt_template="""Final evolution:

Current: {previous}

Add: Thread-safety, error handling, logging

Provide production-ready solution.""",
            temperature=0.6,
        ),
    ],
    final_template="{evolved3}",
    metadata_extras={"evolutions": 3},
)


PROCODER = FrameworkConfig(
    name="procoder",
    description="Compiler-feedback-guided iterative refinement",
    steps=[
        FrameworkStep(
            name="analysis",
            prompt_template="""Analyze project context:

{context}

What types, APIs, patterns exist?""",
            temperature=0.5,
            uses_previous=False,
        ),
        FrameworkStep(
            name="code",
            prompt_template="""Generate type-safe code:

Project context: {previous}

Task: {query}

Use project patterns and types.""",
            temperature=0.6,
        ),
        FrameworkStep(
            name="check",
            prompt_template="""Check for errors:

{previous}

Type errors? Import issues? Linter warnings?""",
            temperature=0.4,
        ),
        FrameworkStep(
            name="fixed",
            prompt_template="""Fix compiler/linter feedback:

Issues: {check}

Code: {code}

Provide corrected version.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="integrated",
            prompt_template="""Verify clean integration:

{previous}

Does it fit cleanly with existing code?""",
            temperature=0.4,
        ),
    ],
    final_template="{fixed}\n\n---\n## Integration Check\n{integrated}",
)


PAL = FrameworkConfig(
    name="pal",
    description="Program-Aided Language - code as reasoning substrate",
    steps=[
        FrameworkStep(
            name="translate",
            prompt_template="""Translate reasoning to a program:

{query}

Context: {context}

What code would solve this?""",
            temperature=0.6,
            uses_previous=False,
        ),
        FrameworkStep(
            name="pseudo",
            prompt_template="""Start with pseudocode:

{previous}

Write clear pseudocode steps.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="implementation",
            prompt_template="""Implement executable code:

{previous}

Convert to working code.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="validated",
            prompt_template="""Test with examples:

{previous}

Test normal + edge cases.""",
            temperature=0.4,
        ),
    ],
    final_template="{implementation}\n\n---\n## Validation\n{validated}",
)


PARSEL = FrameworkConfig(
    name="parsel",
    description="Compositional code from natural language specs",
    steps=[
        FrameworkStep(
            name="specs",
            prompt_template="""Decompose into function specs:

{query}

Context: {context}

For each function:
- Name
- Description
- Inputs
- Outputs
- Dependencies

List all functions needed.""",
            temperature=0.6,
            uses_previous=False,
        ),
        FrameworkStep(
            name="graph",
            prompt_template="""Build dependency order:

{previous}

Order by dependencies (no cycles). Leaf functions first.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="base_funcs",
            prompt_template="""Generate leaf functions first:

{previous}

Implement functions with no dependencies.""",
            temperature=0.6,
        ),
        FrameworkStep(
            name="composed",
            prompt_template="""Build dependent functions:

Base: {base_funcs}

Graph: {graph}

Implement remaining functions.""",
            temperature=0.6,
        ),
        FrameworkStep(
            name="integrated",
            prompt_template="""Integrate into cohesive module:

{previous}

Provide entry point and full integration.""",
            temperature=0.5,
        ),
    ],
    final_template="{integrated}",
)


DOCPROMPTING = FrameworkConfig(
    name="docprompting",
    description="Documentation-driven code generation",
    steps=[
        FrameworkStep(
            name="apis",
            prompt_template="""Identify required APIs/libraries:

{query}

Context: {context}

What APIs needed?""",
            temperature=0.5,
            uses_previous=False,
        ),
        FrameworkStep(
            name="docs",
            prompt_template="""Find relevant documentation:

{previous}

Describe relevant docs, examples, patterns.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="signatures",
            prompt_template="""Extract API details:

{previous}

Function signatures, usage patterns, idioms.""",
            temperature=0.5,
        ),
        FrameworkStep(
            name="code",
            prompt_template="""Generate code following docs:

Signatures: {signatures}

Problem: {query}

Use doc patterns correctly.""",
            temperature=0.6,
        ),
        FrameworkStep(
            name="verified",
            prompt_template="""Cross-check against docs:

{previous}

Docs: {signatures}

Correct params, types, error handling?""",
            temperature=0.4,
        ),
    ],
    final_template="{code}\n\n---\n## Doc Verification\n{verified}",
)


# =============================================================================
# Registry of all framework configurations
# =============================================================================

FRAMEWORK_CONFIGS: dict[str, FrameworkConfig] = {
    "program_of_thoughts": PROGRAM_OF_THOUGHTS,
    "chain_of_verification": CHAIN_OF_VERIFICATION,
    "critic": CRITIC,
    "chain_of_code": CHAIN_OF_CODE,
    "self_debugging": SELF_DEBUGGING,
    "tdd_prompting": TDD_PROMPTING,
    "reverse_cot": REVERSE_COT,
    "codechain": CODECHAIN,
    "evol_instruct": EVOL_INSTRUCT,
    "procoder": PROCODER,
    "pal": PAL,
    "parsel": PARSEL,
    "docprompting": DOCPROMPTING,
}


async def run_framework(
    framework_name: str, sampler: ClientSampler, query: str, context: str
) -> dict[str, Any]:
    """Convenience function to run a framework by name.

    Args:
        framework_name: Name of the framework (must exist in FRAMEWORK_CONFIGS)
        sampler: ClientSampler for making LLM requests
        query: User's query/problem statement
        context: Relevant code context

    Returns:
        Dict with 'final_answer' and 'metadata' keys

    Raises:
        KeyError: If framework_name not found in FRAMEWORK_CONFIGS
    """
    if framework_name not in FRAMEWORK_CONFIGS:
        available = ", ".join(FRAMEWORK_CONFIGS.keys())
        raise KeyError(f"Framework '{framework_name}' not found. Available: {available}")

    config = FRAMEWORK_CONFIGS[framework_name]
    return await execute_framework(config, sampler, query, context)


def get_available_frameworks() -> list[str]:
    """Return list of all available framework names."""
    return list(FRAMEWORK_CONFIGS.keys())


def get_framework_description(framework_name: str) -> str:
    """Get the description of a framework."""
    if framework_name not in FRAMEWORK_CONFIGS:
        raise KeyError(f"Framework '{framework_name}' not found")
    return FRAMEWORK_CONFIGS[framework_name].description
