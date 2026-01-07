"""
Debate Framework: Real Implementation

Implements multi-agent debate for problem solving:
1. Generate diverse initial positions
2. Agents argue for their positions
3. Agents critique opposing positions
4. Synthesis of strongest arguments
5. Final verdict based on debate

This is a REAL framework with simulated multi-agent debate.
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

logger = structlog.get_logger("debate")

NUM_AGENTS = 3
NUM_ROUNDS = 2


@dataclass
class Agent:
    """A debate agent with a position."""
    id: int
    name: str
    position: str
    arguments: list[str]
    rebuttals: list[str]
    final_score: float = 0.0


async def _generate_positions(
    query: str,
    code_context: str,
    state: GraphState
) -> list[Agent]:
    """Generate diverse initial positions from different agents."""
    
    prompt = f"""Generate {NUM_AGENTS} different expert positions on this problem.

PROBLEM: {query}

CONTEXT:
{code_context}

Create {NUM_AGENTS} distinct expert personas with different approaches:

Respond in this format:
AGENT_1_NAME: [Expert type, e.g., "Pragmatic Engineer"]
AGENT_1_POSITION: [Their main approach/solution]

AGENT_2_NAME: [Different expert type]
AGENT_2_POSITION: [Different approach]

AGENT_3_NAME: [Third expert type]
AGENT_3_POSITION: [Third approach]
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    
    agents = []
    current_name = None
    current_position = None
    
    for line in response.split("\n"):
        line = line.strip()
        if "_NAME:" in line:
            current_name = line.split(":")[-1].strip()
        elif "_POSITION:" in line:
            current_position = line.split(":", 1)[-1].strip()
            if current_name and current_position:
                agents.append(Agent(
                    id=len(agents) + 1,
                    name=current_name,
                    position=current_position,
                    arguments=[],
                    rebuttals=[]
                ))
                current_name = None
                current_position = None
    
    # Ensure we have agents
    while len(agents) < NUM_AGENTS:
        agents.append(Agent(
            id=len(agents) + 1,
            name=f"Expert {len(agents) + 1}",
            position=f"Alternative approach {len(agents) + 1}",
            arguments=[],
            rebuttals=[]
        ))
    
    return agents[:NUM_AGENTS]


async def _agent_argue(
    agent: Agent,
    query: str,
    code_context: str,
    other_positions: list[str],
    state: GraphState
) -> list[str]:
    """Agent presents arguments for their position."""
    
    others = "\n".join([f"- {p}" for p in other_positions])
    
    prompt = f"""You are {agent.name}. Argue for your position.

PROBLEM: {query}

YOUR POSITION: {agent.position}

OTHER POSITIONS:
{others}

CONTEXT:
{code_context}

Present 3 strong arguments for YOUR position. Be persuasive and specific.

ARG_1: [First argument]
ARG_2: [Second argument]
ARG_3: [Third argument]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    arguments = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("ARG_"):
            arg = line.split(":", 1)[-1].strip()
            if arg:
                arguments.append(arg)
    
    return arguments[:3]


async def _agent_rebut(
    agent: Agent,
    target_agent: Agent,
    query: str,
    state: GraphState
) -> str:
    """Agent rebuts another agent's position."""
    
    target_args = "\n".join([f"- {a}" for a in target_agent.arguments])
    
    prompt = f"""You are {agent.name}. Rebut {target_agent.name}'s position.

PROBLEM: {query}

YOUR POSITION: {agent.position}

{target_agent.name}'S POSITION: {target_agent.position}
Their arguments:
{target_args}

Provide a concise but strong rebuttal. Point out weaknesses in their approach.

REBUTTAL:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
    return response.replace("REBUTTAL:", "").strip()


async def _judge_debate(
    agents: list[Agent],
    query: str,
    code_context: str,
    state: GraphState
) -> tuple[Agent, str]:
    """Judge the debate and determine winner with synthesis."""
    
    debate_summary = ""
    for agent in agents:
        debate_summary += f"\n## {agent.name}\n"
        debate_summary += f"**Position**: {agent.position}\n"
        debate_summary += f"**Arguments**: {'; '.join(agent.arguments)}\n"
        debate_summary += f"**Rebuttals given**: {'; '.join(agent.rebuttals)}\n"
    
    prompt = f"""Judge this debate and synthesize the best solution.

PROBLEM: {query}

CONTEXT:
{code_context}

DEBATE SUMMARY:
{debate_summary}

Evaluate each position and provide:
1. WINNER: [Agent name] - who had the strongest case
2. WINNER_SCORE: [0.0-1.0]
3. SYNTHESIS: [Combined solution taking best from all positions]

Respond in this format:
WINNER: [name]
WINNER_SCORE: [score]
SYNTHESIS: [your synthesized solution]
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    
    winner_name = agents[0].name
    winner_score = 0.7
    synthesis = response
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("WINNER:") and "SCORE" not in line:
            winner_name = line.split(":")[-1].strip()
        elif line.startswith("WINNER_SCORE:"):
            try:
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    winner_score = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
        elif line.startswith("SYNTHESIS:"):
            synthesis = line.split(":", 1)[-1].strip()
    
    # Find winning agent
    winner = agents[0]
    for agent in agents:
        if agent.name.lower() in winner_name.lower():
            winner = agent
            break
    
    winner.final_score = winner_score
    
    return winner, synthesis


@quiet_star
async def multi_agent_debate_node(state: GraphState) -> GraphState:
    """
    Debate Framework - REAL IMPLEMENTATION
    
    Multi-agent debate:
    - Multiple expert personas
    - Argument presentation
    - Rebuttals and counter-arguments
    - Judged synthesis
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("debate_start", query_preview=query[:50], agents=NUM_AGENTS)
    
    # Generate diverse positions
    agents = await _generate_positions(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="debate",
        thought=f"Initialized {len(agents)} debate agents",
        action="setup",
        observation=", ".join([a.name for a in agents])
    )
    
    # Debate rounds
    for round_num in range(NUM_ROUNDS):
        logger.info("debate_round", round=round_num + 1)
        
        # Each agent presents arguments
        for agent in agents:
            other_positions = [a.position for a in agents if a.id != agent.id]
            agent.arguments = await _agent_argue(
                agent, query, code_context, other_positions, state
            )
        
        add_reasoning_step(
            state=state,
            framework="debate",
            thought=f"Round {round_num + 1}: All agents presented arguments",
            action="argue"
        )
        
        # Each agent rebuts others
        for agent in agents:
            for target in agents:
                if target.id != agent.id:
                    rebuttal = await _agent_rebut(agent, target, query, state)
                    agent.rebuttals.append(f"To {target.name}: {rebuttal}")
        
        add_reasoning_step(
            state=state,
            framework="debate",
            thought=f"Round {round_num + 1}: Rebuttals exchanged",
            action="rebut"
        )
    
    # Judge debate
    winner, synthesis = await _judge_debate(agents, query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="debate",
        thought=f"Winner: {winner.name} (score: {winner.final_score:.2f})",
        action="judge",
        score=winner.final_score
    )
    
    # Format debate transcript
    transcript = ""
    for agent in agents:
        transcript += f"\n### {agent.name}\n"
        transcript += f"**Position**: {agent.position}\n\n"
        transcript += "**Arguments**:\n"
        for i, arg in enumerate(agent.arguments, 1):
            transcript += f"{i}. {arg}\n"
        transcript += "\n**Rebuttals**:\n"
        for reb in agent.rebuttals:
            transcript += f"- {reb}\n"
    
    final_answer = f"""# Debate Analysis

## Participants
{', '.join([f'{a.name}' for a in agents])}

## Debate Transcript
{transcript}

## Verdict
**Winner**: {winner.name} (Score: {winner.final_score:.2f})

## Synthesized Solution
{synthesis}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = winner.final_score
    
    logger.info("debate_complete", winner=winner.name, score=winner.final_score)
    
    return state
