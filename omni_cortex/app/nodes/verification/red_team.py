"""
Red Team Framework: Real Implementation

Adversarial testing and critique:
1. Generate solution
2. Red team attacks it (find weaknesses)
3. Document vulnerabilities
4. Generate hardened version
5. Re-attack to verify
6. Iterate until robust
"""

from dataclasses import dataclass

import structlog

from ...state import GraphState
from ..common import (
    add_reasoning_step,
    call_deep_reasoner,
    call_fast_synthesizer,
    prepare_context_with_gemini,
    quiet_star,
)

logger = structlog.get_logger("red_team")

MAX_ATTACK_ROUNDS = 3


@dataclass
class Attack:
    """A red team attack."""

    round_num: int
    attack_vector: str
    vulnerability_found: str
    severity: str
    exploitable: bool


async def _generate_initial_solution(query: str, code_context: str, state: GraphState) -> str:
    """Generate initial solution to be attacked."""
    prompt = f"""Generate solution.

PROBLEM: {query}
CONTEXT: {code_context}

SOLUTION:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    return response


async def _red_team_attack(
    solution: str, round_num: int, previous_attacks: list[Attack], query: str, state: GraphState
) -> list[Attack]:
    """Red team attacks the solution."""

    prev_attacks_text = ""
    if previous_attacks:
        prev_attacks_text = "\n\nPREVIOUS ATTACKS:\n"
        prev_attacks_text += "\n".join(
            f"- {a.attack_vector}: {a.vulnerability_found}" for a in previous_attacks[-3:]
        )

    prompt = f"""Red team attack this solution. Find 3-4 weaknesses/vulnerabilities.

PROBLEM: {query}

SOLUTION TO ATTACK:
{solution[:800]}
{prev_attacks_text}

Identify new attack vectors:

ATTACK_1_VECTOR: [Type of attack]
ATTACK_1_VULNERABILITY: [What's vulnerable]
ATTACK_1_SEVERITY: [low/medium/high/critical]

ATTACK_2_VECTOR: [Different attack]
ATTACK_2_VULNERABILITY: [What's vulnerable]
ATTACK_2_SEVERITY: [low/medium/high/critical]

ATTACK_3_VECTOR: [Another attack]
ATTACK_3_VULNERABILITY: [What's vulnerable]
ATTACK_3_SEVERITY: [low/medium/high/critical]
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=768)

    attacks = []
    current = {"vector": "", "vuln": "", "severity": "medium"}

    for line in response.split("\n"):
        if "_VECTOR:" in line:
            if current["vector"]:
                attacks.append(
                    Attack(
                        round_num=round_num,
                        attack_vector=current["vector"],
                        vulnerability_found=current["vuln"],
                        severity=current["severity"],
                        exploitable=True,
                    )
                )
            current = {"vector": line.split(":", 1)[-1].strip(), "vuln": "", "severity": "medium"}
        elif "_VULNERABILITY:" in line:
            current["vuln"] = line.split(":", 1)[-1].strip()
        elif "_SEVERITY:" in line:
            current["severity"] = line.split(":", 1)[-1].strip().lower()

    if current["vector"]:
        attacks.append(
            Attack(
                round_num=round_num,
                attack_vector=current["vector"],
                vulnerability_found=current["vuln"],
                severity=current["severity"],
                exploitable=True,
            )
        )

    return attacks


async def _harden_solution(
    solution: str, attacks: list[Attack], query: str, state: GraphState
) -> str:
    """Harden solution against attacks."""

    attacks_text = "\n".join(
        [f"- {a.attack_vector} ({a.severity}): {a.vulnerability_found}" for a in attacks]
    )

    prompt = f"""Harden this solution against red team attacks.

ORIGINAL SOLUTION:
{solution[:800]}

VULNERABILITIES FOUND:
{attacks_text}

PROBLEM: {query}

Generate hardened version that addresses these vulnerabilities:

HARDENED SOLUTION:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    return response


async def _verify_hardening(
    hardened: str, original_attacks: list[Attack], query: str, state: GraphState
) -> tuple[bool, str]:
    """Verify hardening worked."""

    attacks_text = "\n".join(f"- {a.attack_vector}" for a in original_attacks)

    prompt = f"""Verify hardening against these attacks.

HARDENED SOLUTION:
{hardened[:800]}

ORIGINAL ATTACKS:
{attacks_text}

PROBLEM: {query}

Are the vulnerabilities fixed?

SECURE: [yes/no]
VERIFICATION:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)

    is_secure = "yes" in response.lower()
    return is_secure, response.strip()


@quiet_star
async def red_team_node(state: GraphState) -> GraphState:
    """
    Red Team - REAL IMPLEMENTATION

    Adversarial testing:
    - Generates solution
    - Red team attacks it
    - Documents vulnerabilities
    - Hardens solution
    - Re-attacks to verify
    - Iterates until robust
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("red_team_start", query_preview=query[:50])

    # Initial solution
    current_solution = await _generate_initial_solution(query, code_context, state)

    add_reasoning_step(
        state=state, framework="red_team", thought="Generated initial solution", action="generate"
    )

    all_attacks = []

    for round_num in range(1, MAX_ATTACK_ROUNDS + 1):
        logger.info("red_team_round", round=round_num)

        # Attack
        attacks = await _red_team_attack(current_solution, round_num, all_attacks, query, state)

        if not attacks:
            # No vulnerabilities found
            add_reasoning_step(
                state=state,
                framework="red_team",
                thought=f"Round {round_num}: No vulnerabilities found",
                action="attack",
            )
            break

        all_attacks.extend(attacks)

        critical_count = sum(1 for a in attacks if a.severity == "critical")

        add_reasoning_step(
            state=state,
            framework="red_team",
            thought=f"Round {round_num}: Found {len(attacks)} vulnerabilities ({critical_count} critical)",
            action="attack",
        )

        # Harden
        hardened = await _harden_solution(current_solution, attacks, query, state)

        add_reasoning_step(
            state=state,
            framework="red_team",
            thought=f"Hardened solution against {len(attacks)} attacks",
            action="harden",
        )

        # Verify
        is_secure, verification = await _verify_hardening(hardened, attacks, query, state)

        current_solution = hardened

        if is_secure:
            logger.info("red_team_secure", rounds=round_num)
            break

    # Count by severity
    severity_counts = {}
    for attack in all_attacks:
        severity_counts[attack.severity] = severity_counts.get(attack.severity, 0) + 1

    # Format attacks log
    attacks_log = "\n\n".join(
        [
            f"### Round {a.round_num} Attack: {a.attack_vector}\n"
            f"**Severity**: {a.severity.upper()}\n"
            f"**Vulnerability**: {a.vulnerability_found}\n"
            f"**Exploitable**: {'Yes' if a.exploitable else 'No'}"
            for a in all_attacks
        ]
    )

    final_answer = f"""# Red Team Analysis

## Attack Surface Explored
- Total attacks: {len(all_attacks)}
- By severity: {", ".join(f"{s}: {c}" for s, c in sorted(severity_counts.items()))}

## Attack Log
{attacks_log}

## Hardened Solution
{current_solution}

## Security Posture
- Rounds of hardening: {round_num}
- Vulnerabilities addressed: {len(all_attacks)}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = 0.9 if len(all_attacks) > 5 else 0.7

    logger.info("red_team_complete", attacks=len(all_attacks))

    return state
