"""
Verify-and-Edit Framework: Real Implementation

Verification with correction loop:
1. Generate solution
2. Verify correctness with checks
3. Identify what needs editing
4. Edit to fix issues
5. Re-verify
6. Repeat until verified
"""

import asyncio
import structlog
from dataclasses import dataclass

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("verify_and_edit")

MAX_EDIT_ROUNDS = 3


@dataclass
class EditRound:
    """One verify-edit round."""
    round_num: int
    content: str
    verification_checks: list[str]
    passed_checks: list[str]
    failed_checks: list[str]
    edits_made: list[str]


async def _generate_initial_content(query: str, code_context: str, state: GraphState) -> str:
    """Generate initial content."""
    prompt = f"""Generate solution.

PROBLEM: {query}
CONTEXT: {code_context}

SOLUTION:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    return response


async def _create_verification_checks(query: str, state: GraphState) -> list[str]:
    """Create verification checks."""
    prompt = f"""Create verification checks for this solution.

PROBLEM: {query}

What should we verify? List 4-5 checks:

CHECK_1:
CHECK_2:
CHECK_3:
CHECK_4:
CHECK_5:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    
    checks = []
    for line in response.split("\n"):
        if line.startswith("CHECK_"):
            check = line.split(":", 1)[-1].strip()
            if check:
                checks.append(check)
    
    return checks


async def _run_verification(content: str, checks: list[str], query: str, state: GraphState) -> tuple[list[str], list[str]]:
    """Run verification checks."""
    checks_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(checks))
    
    prompt = f"""Verify this solution against checks.

SOLUTION:
{content[:800]}

CHECKS TO VERIFY:
{checks_text}

PROBLEM: {query}

For each check, state PASS or FAIL:

CHECK_1_RESULT:
CHECK_2_RESULT:
CHECK_3_RESULT:
CHECK_4_RESULT:
CHECK_5_RESULT:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    passed = []
    failed = []
    
    for i, check in enumerate(checks, 1):
        check_key = f"CHECK_{i}_RESULT"
        for line in response.split("\n"):
            if check_key in line:
                if "PASS" in line.upper():
                    passed.append(check)
                elif "FAIL" in line.upper():
                    failed.append(check)
                break
    
    return passed, failed


async def _generate_edits(content: str, failed_checks: list[str], query: str, state: GraphState) -> tuple[str, list[str]]:
    """Generate edits to fix failed checks."""
    failed_text = "\n".join(f"- {c}" for c in failed_checks)
    
    prompt = f"""Edit solution to fix failed checks.

CURRENT SOLUTION:
{content[:800]}

FAILED CHECKS:
{failed_text}

PROBLEM: {query}

Provide edited solution and list changes made:

EDITS:
[List changes]

EDITED SOLUTION:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    
    # Extract edits and solution
    edits = []
    if "EDITS:" in response:
        edits_section = response.split("EDITS:")[1].split("EDITED SOLUTION:")[0]
        for line in edits_section.split("\n"):
            line = line.strip()
            if line and line.startswith(("-", "*", "•")):
                edits.append(line[1:].strip())
    
    if "EDITED SOLUTION:" in response:
        edited = response.split("EDITED SOLUTION:")[-1].strip()
    else:
        edited = response.strip()
    
    return edited, edits


@quiet_star
async def verify_and_edit_node(state: GraphState) -> GraphState:
    """
    Verify-and-Edit - REAL IMPLEMENTATION
    
    Verification loop with corrections:
    - Generate solution
    - Create verification checks
    - Run verification
    - Edit to fix failures
    - Re-verify until passing
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("verify_and_edit_start", query_preview=query[:50])
    
    # Initial content
    current_content = await _generate_initial_content(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="verify_and_edit",
        thought="Generated initial solution",
        action="generate"
    )
    
    # Create verification checks
    checks = await _create_verification_checks(query, state)
    
    add_reasoning_step(
        state=state,
        framework="verify_and_edit",
        thought=f"Created {len(checks)} verification checks",
        action="create_checks"
    )
    
    rounds = []
    
    for round_num in range(1, MAX_EDIT_ROUNDS + 1):
        logger.info("verify_edit_round", round=round_num)
        
        # Verify
        passed, failed = await _run_verification(current_content, checks, query, state)
        
        add_reasoning_step(
            state=state,
            framework="verify_and_edit",
            thought=f"Round {round_num}: {len(passed)}/{len(checks)} checks passed",
            action="verify"
        )
        
        if not failed:
            # All checks passed
            rounds.append(EditRound(
                round_num=round_num,
                content=current_content,
                verification_checks=checks,
                passed_checks=passed,
                failed_checks=[],
                edits_made=[]
            ))
            logger.info("verification_passed", rounds=round_num)
            break
        
        # Edit to fix failures
        edited_content, edits = await _generate_edits(current_content, failed, query, state)
        
        rounds.append(EditRound(
            round_num=round_num,
            content=current_content,
            verification_checks=checks,
            passed_checks=passed,
            failed_checks=failed,
            edits_made=edits
        ))
        
        add_reasoning_step(
            state=state,
            framework="verify_and_edit",
            thought=f"Applied {len(edits)} edits to fix {len(failed)} failures",
            action="edit"
        )
        
        current_content = edited_content
    
    # Format rounds
    rounds_viz = "\n\n".join([
        f"### Round {r.round_num}\n"
        f"**Passed**: {len(r.passed_checks)}/{len(r.verification_checks)}\n"
        f"**Failed**: {', '.join(r.failed_checks) if r.failed_checks else 'None'}\n"
        f"**Edits**: {len(r.edits_made)}"
        for r in rounds
    ])
    
    all_passed = len(rounds[-1].failed_checks) == 0
    
    final_answer = f"""# Verify-and-Edit Analysis

## Verification Checks
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(checks))}

## Edit Rounds
{rounds_viz}

## Final Solution ({"✓ Verified" if all_passed else "⚠ Partial verification"})
{current_content}

## Statistics
- Edit rounds: {len(rounds)}
- Final checks passed: {len(rounds[-1].passed_checks)}/{len(checks)}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = 1.0 if all_passed else 0.7
    
    logger.info("verify_and_edit_complete", rounds=len(rounds), all_passed=all_passed)
    
    return state
