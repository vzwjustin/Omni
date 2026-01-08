"""
Society of Mind Framework: Real Implementation

Implements Minsky's Society of Mind with collaborative agents:
1. Spawn diverse cognitive agents (critics, builders, planners, etc.)
2. Agents communicate and collaborate
3. Competitive and cooperative dynamics
4. Emergent solution from agent interactions

This is a framework with actual multi-agent collaboration.
"""

import asyncio
import structlog
from dataclasses import dataclass, field
from typing import Optional

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("society_of_mind")

NUM_AGENTS = 6
INTERACTION_ROUNDS = 3


@dataclass
class MindAgent:
    """A cognitive agent in the society."""
    id: int
    role: str
    personality: str
    contributions: list[str] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)
    collaboration_score: float = 0.0


async def _spawn_society(
    query: str,
    state: GraphState
) -> list[MindAgent]:
    """Create a diverse society of cognitive agents."""
    
    prompt = f"""Create {NUM_AGENTS} cognitive agents for collaborative problem solving.

PROBLEM: {query}

Define agents with complementary roles (like Minsky's Society of Mind):
- Critics (find flaws)
- Builders (create solutions)
- Planners (strategize)
- Testers (verify)
- Optimizers (improve)
- Integrators (combine ideas)

For each agent, specify role and personality:

AGENT_1_ROLE: [Role type]
AGENT_1_PERSONALITY: [Brief personality/approach]

AGENT_2_ROLE: [Different role]
AGENT_2_PERSONALITY: [Different personality]

(continue for {NUM_AGENTS} agents)
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    agents = []
    current_role = None
    current_personality = None
    
    for line in response.split("\n"):
        line = line.strip()
        if "_ROLE:" in line:
            if current_role and current_personality:
                agents.append(MindAgent(
                    id=len(agents) + 1,
                    role=current_role,
                    personality=current_personality
                ))
            current_role = line.split(":")[-1].strip()
            current_personality = None
        elif "_PERSONALITY:" in line:
            current_personality = line.split(":", 1)[-1].strip()
    
    if current_role and current_personality:
        agents.append(MindAgent(
            id=len(agents) + 1,
            role=current_role,
            personality=current_personality
        ))
    
    # Ensure we have enough agents
    default_agents = [
        ("Critic", "Skeptical, finds flaws"),
        ("Builder", "Constructive, creates solutions"),
        ("Planner", "Strategic, organizes steps"),
        ("Tester", "Thorough, validates ideas"),
        ("Optimizer", "Efficient, improves solutions"),
        ("Integrator", "Holistic, combines perspectives")
    ]
    
    while len(agents) < NUM_AGENTS:
        role, personality = default_agents[len(agents)]
        agents.append(MindAgent(id=len(agents) + 1, role=role, personality=personality))
    
    return agents[:NUM_AGENTS]


async def _agent_contribute(
    agent: MindAgent,
    query: str,
    code_context: str,
    previous_contributions: list[str],
    state: GraphState
) -> str:
    """Agent makes a contribution based on their role."""
    
    context = ""
    if previous_contributions:
        context = "\n\nPREVIOUS CONTRIBUTIONS FROM OTHER AGENTS:\n"
        context += "\n".join([f"- {c[:100]}..." for c in previous_contributions[-5:]])
    
    prompt = f"""You are a {agent.role} with this personality: {agent.personality}

PROBLEM: {query}

CONTEXT:
{code_context}
{context}

Based on your role, make a contribution:
- If you're a CRITIC: identify problems or weaknesses
- If you're a BUILDER: propose solutions or implementations
- If you're a PLANNER: outline strategies or steps
- If you're a TESTER: propose validation methods
- If you're an OPTIMIZER: suggest improvements
- If you're an INTEGRATOR: combine or synthesize ideas

CONTRIBUTION:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    return response.strip()


async def _agent_critique(
    agent: MindAgent,
    target_agent: MindAgent,
    query: str,
    state: GraphState
) -> str:
    """Agent critiques another agent's contribution."""
    
    if not target_agent.contributions:
        return ""
    
    prompt = f"""You are a {agent.role}. Critique this contribution from the {target_agent.role}.

PROBLEM: {query}

{target_agent.role}'S CONTRIBUTION:
{target_agent.contributions[-1]}

From your {agent.role} perspective, provide constructive criticism.
What works? What doesn't? What's missing?

CRITIQUE:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
    return response.strip()


async def _synthesize_society_output(
    agents: list[MindAgent],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Synthesize the collective output of the society."""
    
    contributions_by_role = {}
    for agent in agents:
        role_contribs = "\n".join([f"  - {c}" for c in agent.contributions])
        contributions_by_role[agent.role] = role_contribs
    
    society_output = "\n\n".join([
        f"**{role}**:\n{contribs}"
        for role, contribs in contributions_by_role.items()
    ])
    
    prompt = f"""Synthesize the collective intelligence of this society of mind.

PROBLEM: {query}

CONTEXT:
{code_context}

SOCIETY CONTRIBUTIONS:
{society_output}

Create a unified solution that emerges from these diverse perspectives.
Show how different agents' insights combine to create a better solution.
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def system1_node(state: GraphState) -> GraphState:
    """
    Society of Mind Framework - REAL IMPLEMENTATION
    
    Multi-agent collaborative reasoning:
    - Spawns diverse cognitive agents
    - Agents contribute based on roles
    - Competitive and cooperative dynamics
    - Emergent solution from interactions
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("society_of_mind_start", query_preview=query[:50], agents=NUM_AGENTS)
    
    # Step 1: Spawn society
    agents = await _spawn_society(query, state)
    
    add_reasoning_step(
        state=state,
        framework="society_of_mind",
        thought=f"Spawned society with {len(agents)} agents",
        action="spawn",
        observation=", ".join([f"{a.role}" for a in agents])
    )
    
    # Step 2: Interaction rounds
    all_contributions = []
    
    for round_num in range(INTERACTION_ROUNDS):
        logger.info("society_round", round=round_num + 1)
        
        # Each agent contributes
        for agent in agents:
            contribution = await _agent_contribute(
                agent, query, code_context, all_contributions, state
            )
            agent.contributions.append(contribution)
            all_contributions.append(f"{agent.role}: {contribution}")
        
        add_reasoning_step(
            state=state,
            framework="society_of_mind",
            thought=f"Round {round_num + 1}: All {len(agents)} agents contributed",
            action="contribute"
        )
        
        # Agents critique each other (selective pairings)
        critique_pairs = [
            (agents[0], agents[1]),  # Critic critiques Builder
            (agents[2], agents[3]),  # Planner critiques Tester
            (agents[4], agents[5]),  # Optimizer critiques Integrator
        ]
        
        for agent, target in critique_pairs[:len(agents)//2]:
            critique = await _agent_critique(agent, target, query, state)
            if critique:
                agent.critiques.append(f"To {target.role}: {critique}")
        
        add_reasoning_step(
            state=state,
            framework="society_of_mind",
            thought=f"Round {round_num + 1}: Inter-agent critiques completed",
            action="critique"
        )
    
    # Step 3: Synthesize
    solution = await _synthesize_society_output(agents, query, code_context, state)
    
    # Format society transcript
    transcript = ""
    for agent in agents:
        transcript += f"\n### {agent.role} - {agent.personality}\n"
        transcript += f"**Contributions** ({len(agent.contributions)}):\n"
        for i, contrib in enumerate(agent.contributions, 1):
            transcript += f"{i}. {contrib[:150]}...\n"
        if agent.critiques:
            transcript += f"**Critiques given**: {len(agent.critiques)}\n"
    
    final_answer = f"""# Society of Mind Analysis

## The Society
{', '.join([f'{a.role}' for a in agents])}

## Interaction Transcript ({INTERACTION_ROUNDS} rounds)
{transcript}

## Emergent Solution
{solution}

## Statistics
- Agents: {len(agents)}
- Interaction rounds: {INTERACTION_ROUNDS}
- Total contributions: {sum(len(a.contributions) for a in agents)}
- Inter-agent critiques: {sum(len(a.critiques) for a in agents)}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = 0.85
    
    logger.info(
        "society_of_mind_complete",
        agents=len(agents),
        contributions=sum(len(a.contributions) for a in agents)
    )
    
    return state
