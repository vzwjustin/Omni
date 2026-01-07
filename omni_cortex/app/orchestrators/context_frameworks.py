"""
Context Framework Orchestrators

Frameworks for understanding, exploration, and contextual reasoning.
"""

from typing import Dict, Any
from ..core.sampling import ClientSampler


async def chain_of_note(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Chain of Note: Research and note-taking approach
    """
    notes = []

    # Note 1: Observations
    obs = await sampler.request_sample(
        f"NOTE 1 - Observations: What do you see?\n\n{query}\n\nContext: {context}\n\nList key observations.",
        temperature=0.6
    )
    notes.append(("Observations", obs))

    # Note 2: Connections
    conn = await sampler.request_sample(
        f"NOTE 2 - Connections: How do pieces relate?\n\nObservations: {obs}\n\nHow do these connect?",
        temperature=0.6
    )
    notes.append(("Connections", conn))

    # Note 3: Inferences
    inf = await sampler.request_sample(
        f"NOTE 3 - Inferences: What can you conclude?\n\nConnections: {conn}\n\nWhat conclusions follow?",
        temperature=0.6
    )
    notes.append(("Inferences", inf))

    # Note 4: Synthesis
    synth = await sampler.request_sample(
        f"""NOTE 4 - Synthesis: Complete answer

Notes so far:
{chr(10).join(f'{label}: {text[:CONTENT.QUERY_LOG]}...' for label, text in notes)}

Original question: {query}

Provide complete synthesized answer.""",
        temperature=0.5
    )

    return {
        "final_answer": synth,
        "metadata": {"framework": "chain_of_note", "notes": len(notes)}
    }


async def step_back(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Step-Back: Abstract principles first, then apply
    """
    # Step back to abstract principles
    abstract = await sampler.request_sample(
        f"""STEP BACK: What category is this? What principles apply?

{query}

Context: {context}

Think at a higher level: What's the general class of problem? What principles govern it?""",
        temperature=0.6
    )

    # Identify key constraints and trade-offs
    constraints = await sampler.request_sample(
        f"Key constraints and trade-offs:\n\n{abstract}\n\nWhat are the fundamental constraints? Known trade-offs?",
        temperature=0.6
    )

    # Apply principles to concrete solution
    applied = await sampler.request_sample(
        f"""APPLY abstract principles:

Principles: {abstract[:CONTENT.ERROR_PREVIEW]}...
Constraints: {constraints[:CONTENT.ERROR_PREVIEW]}...

Concrete problem: {query}
Context: {context}

Map principles to specific solution.""",
        temperature=0.5
    )

    # Verify follows principles
    verified = await sampler.request_sample(
        f"VERIFY solution follows principles:\n\n{applied}\n\nPrinciples: {abstract[:150]}...\n\nDoes it align?",
        temperature=0.4
    )

    return {
        "final_answer": f"{applied}\n\n---\n## Verification\n{verified}",
        "metadata": {"framework": "step_back"}
    }


async def analogical(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Analogical: Find and adapt similar solutions
    """
    # Find analogies
    analogies = await sampler.request_sample(
        f"""Find 2-3 similar problems that have been solved:

{query}

Context: {context}

What analogous problems exist? How were they solved?""",
        temperature=0.7
    )

    # Select best analogy
    best = await sampler.request_sample(
        f"Select best analogy:\n\n{analogies}\n\nWhich maps best to our problem?",
        temperature=0.5
    )

    # Map analogy to problem
    mapping = await sampler.request_sample(
        f"MAP the analogy:\n\nBest analogy: {best}\n\nOur problem: {query}\n\nHow does it map? What transfers?",
        temperature=0.6
    )

    # Adapt
    adapted = await sampler.request_sample(
        f"ADAPT for differences:\n\nMapping: {mapping}\n\nWhat's different? How to adapt the solution?",
        temperature=0.6
    )

    # Implement
    implemented = await sampler.request_sample(
        f"IMPLEMENT adapted solution:\n\n{adapted}\n\nOriginal: {query}\n\nContext: {context}\n\nComplete solution:",
        temperature=0.5
    )

    return {
        "final_answer": implemented,
        "metadata": {"framework": "analogical"}
    }


async def red_team(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Red-Team: Adversarial security analysis (STRIDE, OWASP)
    """
    threats = {}

    # STRIDE analysis
    stride_prompt = f"""Security analysis using STRIDE:

{query}

Context: {context}

Check each:
- SPOOFING: Identity issues?
- TAMPERING: Data integrity issues?
- REPUDIATION: Audit issues?
- INFO DISCLOSURE: Leakage issues?
- DENIAL OF SERVICE: Availability issues?
- ELEVATION OF PRIVILEGE: Access control issues?

List findings for each."""

    stride = await sampler.request_sample(stride_prompt, temperature=0.6)
    threats['stride'] = stride

    # OWASP Top 10 check
    owasp = await sampler.request_sample(
        f"OWASP Top 10 check:\n\n{context}\n\nCheck for: Injection, broken auth, XSS, broken access control, etc.\n\nList vulnerabilities.",
        temperature=0.6
    )
    threats['owasp'] = owasp

    # Fixes for each finding
    fixes = await sampler.request_sample(
        f"""Provide fixes:

STRIDE findings: {stride[:300]}...
OWASP findings: {owasp[:300]}...

For each vulnerability, provide specific fix.""",
        temperature=0.5
    )

    return {
        "final_answer": fixes,
        "metadata": {"framework": "red_team", "threat_models": ["STRIDE", "OWASP"]}
    }


async def state_machine(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    State-Machine: Formal FSM design before coding
    """
    # Enumerate states
    states = await sampler.request_sample(
        f"STATES: Enumerate all possible states\n\n{query}\n\nContext: {context}\n\nList all states.",
        temperature=0.6
    )

    # Define transitions
    transitions = await sampler.request_sample(
        f"TRANSITIONS: What triggers state changes?\n\nStates: {states}\n\nFor each state, what events cause transitions?",
        temperature=0.6
    )

    # Guards
    guards = await sampler.request_sample(
        f"GUARDS: What conditions must be met?\n\nTransitions: {transitions[:300]}...\n\nWhat conditions guard transitions?",
        temperature=0.6
    )

    # Actions
    actions = await sampler.request_sample(
        f"ACTIONS: What happens on transition?\n\nGuards: {guards[:CONTENT.ERROR_PREVIEW]}...\n\nWhat actions execute?",
        temperature=0.6
    )

    # Implement
    implementation = await sampler.request_sample(
        f"""IMPLEMENT state machine:

States: {states[:150]}...
Transitions: {transitions[:150]}...
Guards: {guards[:150]}...
Actions: {actions[:150]}...

Original: {query}

Code the state machine with clear structure.""",
        temperature=0.5
    )

    return {
        "final_answer": implementation,
        "metadata": {"framework": "state_machine"}
    }


async def chain_of_thought(sampler: ClientSampler, query: str, context: str) -> Dict[str, Any]:
    """
    Chain-of-Thought: Step-by-step reasoning chain
    """
    # Build reasoning chain step by step
    reasoning_chain = await sampler.request_sample(
        f"""Think step by step:

{query}

Context: {context}

Show your complete reasoning process:
STEP 1: [First logical step]
STEP 2: [Building on step 1]
STEP 3: [Continue reasoning]
...
CONCLUSION: [Final answer based on chain]

Provide detailed chain of thought.""",
        temperature=0.6
    )

    return {
        "final_answer": reasoning_chain,
        "metadata": {"framework": "chain_of_thought"}
    }
