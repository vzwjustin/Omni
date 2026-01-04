"""
Threat Modeling / Red-Teaming: Security-Focused Code Review

AI assumes attacker mindset to find vulnerabilities in code.
Identifies security issues like SQLi, XSS, authentication bypasses, etc.
(Headless Mode: Returns Reasoning Protocol for Client Execution)
"""

import logging
from ...state import GraphState
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
    call_fast_synthesizer # Kept for import compatibility
)

logger = logging.getLogger(__name__)

@quiet_star
async def red_team_node(state: GraphState) -> GraphState:
    """
    Teaming: Security-Focused Code Review
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Teaming Protocol

I have selected the **Teaming** framework for this task.
Security-Focused Code Review

## Use Case
Security audits, penetration testing, vulnerability scanning

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Teaming** using your internal context:

### Framework Steps
1. RECONNAISSANCE: Understand the code's attack surface
2. THREAT_MODEL: Identify potential attack vectors
3. EXPLOIT: Find specific vulnerabilities
4. ASSESS: Rate severity and impact
5. PATCH: Provide secure fixes

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Teaming process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="red_team",
        thought="Generated Teaming protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
