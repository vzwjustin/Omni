"""
Multi-Agent Debate: Proponent vs Critic

Implements adversarial debate between agents arguing
for different implementations of a feature or solution.
"""

from typing import Optional
from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context
)
from ...core.config import settings


@quiet_star
async def multi_agent_debate_node(state: GraphState) -> GraphState:
    """
    Multi-Agent Debate: Proponent vs Critic.
    
    Adversarial process:
    1. PROPOSE: Proponent presents implementation
    2. CRITIQUE: Critic identifies weaknesses
    3. DEFEND: Proponent addresses critiques
    4. COUNTER: Critic counters or concedes
    5. Continue until consensus or max rounds
    6. JUDGE: Neutral judge decides winner
    
    Best for: Design decisions, trade-offs, architecture choices
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    max_rounds = min(settings.debate_max_rounds, 5)
    proponent_args: list[str] = []
    critic_args: list[str] = []
    
    # =========================================================================
    # Initial Proposal (Proponent)
    # =========================================================================
    
    proposal_prompt = f"""You are the PROPONENT in a technical debate.

TASK/QUESTION:
{query}

CONTEXT:
{code_context}

Present your PROPOSED SOLUTION:

**SOLUTION OVERVIEW**
[High-level description]

**KEY DESIGN DECISIONS**
1. [Decision 1 and rationale]
2. [Decision 2 and rationale]
3. [Decision 3 and rationale]

**IMPLEMENTATION APPROACH**
[How you would implement it]

**BENEFITS**
- [Benefit 1]
- [Benefit 2]
- [Benefit 3]

Argue convincingly for your approach."""

    proposal_response, _ = await call_deep_reasoner(
        prompt=proposal_prompt,
        state=state,
        system="You are the PROPONENT. Advocate strongly for your solution.",
        temperature=0.7
    )
    
    proponent_args.append(proposal_response)
    
    add_reasoning_step(
        state=state,
        framework="multi_agent_debate",
        thought="Proponent presented initial proposal",
        action="proposal",
        observation=proposal_response[:200]
    )
    
    # =========================================================================
    # Debate Rounds
    # =========================================================================
    
    for round_num in range(max_rounds):
        # CRITIQUE (Critic's turn)
        critique_prompt = f"""You are the CRITIC in a technical debate.

ORIGINAL TASK:
{query}

PROPONENT'S ARGUMENT:
{proponent_args[-1]}

{"PREVIOUS DEBATE:" if round_num > 0 else ""}
{_format_debate_history(proponent_args[:-1], critic_args) if round_num > 0 else ""}

Provide your CRITIQUE:

**WEAKNESSES IDENTIFIED**
1. [Weakness 1]: [Why it's a problem]
2. [Weakness 2]: [Why it's a problem]
3. [Weakness 3]: [Why it's a problem]

**RISKS & CONCERNS**
- [Risk 1]
- [Risk 2]

**ALTERNATIVE APPROACH** (if applicable)
[Suggest a different approach if you think it's better]

**QUESTIONS FOR PROPONENT**
1. [Question that challenges their approach]
2. [Another challenging question]

Be rigorous but fair."""

        critique_response, _ = await call_deep_reasoner(
            prompt=critique_prompt,
            state=state,
            system="You are the CRITIC. Find weaknesses and challenge assumptions.",
            temperature=0.6
        )
        
        critic_args.append(critique_response)
        
        add_reasoning_step(
            state=state,
            framework="multi_agent_debate",
            thought=f"Round {round_num + 1}: Critic challenged proposal",
            action="critique",
            observation=f"Identified weaknesses in proposal"
        )
        
        # DEFENSE (Proponent's response)
        defense_prompt = f"""You are the PROPONENT defending your solution.

ORIGINAL TASK:
{query}

YOUR PROPOSAL:
{proponent_args[-1]}

CRITIC'S CHALLENGES:
{critique_response}

Provide your DEFENSE:

**ADDRESSING WEAKNESSES**
1. [How you address weakness 1]
2. [How you address weakness 2]
3. [How you address weakness 3]

**RISK MITIGATIONS**
[How you would mitigate the identified risks]

**ANSWERING QUESTIONS**
1. [Answer to question 1]
2. [Answer to question 2]

**REFINED PROPOSAL** (if needed)
[Any adjustments to your original proposal based on valid critiques]

Defend strongly but concede valid points."""

        defense_response, _ = await call_deep_reasoner(
            prompt=defense_prompt,
            state=state,
            system="You are the PROPONENT defending. Concede valid points, defend strong ones.",
            temperature=0.6
        )
        
        proponent_args.append(defense_response)
        
        add_reasoning_step(
            state=state,
            framework="multi_agent_debate",
            thought=f"Round {round_num + 1}: Proponent defended",
            action="defense",
            observation="Addressed critiques"
        )
        
        # Check for consensus
        if _check_consensus(defense_response, critique_response):
            state["consensus_reached"] = True
            break
    
    # Store debate history
    state["proponent_arguments"] = proponent_args
    state["critic_arguments"] = critic_args
    state["debate_round"] = len(critic_args)
    
    # =========================================================================
    # Judge's Decision
    # =========================================================================
    
    judge_prompt = f"""You are a NEUTRAL JUDGE evaluating a technical debate.

TASK:
{query}

CONTEXT:
{code_context}

COMPLETE DEBATE:
{_format_full_debate(proponent_args, critic_args)}

As the judge, provide:

**DEBATE SUMMARY**
[Brief summary of key points from both sides]

**VALID PROPONENT POINTS**
[List the proponent's valid arguments]

**VALID CRITIC POINTS**
[List the critic's valid concerns]

**FINAL VERDICT**
[Which approach should be taken and why]

**RECOMMENDED SOLUTION**
[The solution that incorporates the best of both sides]

**IMPLEMENTATION**
```
[Code if applicable]
```

Be fair and objective."""

    judge_response, _ = await call_deep_reasoner(
        prompt=judge_prompt,
        state=state,
        system="You are the NEUTRAL JUDGE. Be fair and objective.",
        temperature=0.4,
        max_tokens=4000
    )
    
    add_reasoning_step(
        state=state,
        framework="multi_agent_debate",
        thought="Judge rendered final verdict",
        action="judgment",
        observation=f"Debate concluded after {len(critic_args)} rounds"
    )
    
    # Extract code
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, judge_response, re.DOTALL)
    
    # Update final state
    state["final_answer"] = judge_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.85 if state.get("consensus_reached") else 0.75
    
    return state


def _format_debate_history(proponent_args: list[str], critic_args: list[str]) -> str:
    """Format previous debate exchanges."""
    history = []
    for i, (p, c) in enumerate(zip(proponent_args, critic_args)):
        history.append(f"Round {i+1} Proponent: {p[:300]}...")
        history.append(f"Round {i+1} Critic: {c[:300]}...")
    return "\n\n".join(history)


def _format_full_debate(proponent_args: list[str], critic_args: list[str]) -> str:
    """Format complete debate for judge."""
    debate = []
    for i, p in enumerate(proponent_args):
        debate.append(f"=== PROPONENT (Round {i+1}) ===\n{p}\n")
        if i < len(critic_args):
            debate.append(f"=== CRITIC (Round {i+1}) ===\n{critic_args[i]}\n")
    return "\n".join(debate)


def _check_consensus(defense: str, critique: str) -> bool:
    """Check if consensus has been reached."""
    consensus_indicators = [
        "agree", "concede", "valid point", "you're right",
        "accept", "incorporate", "good point"
    ]
    
    defense_lower = defense.lower()
    critique_lower = critique.lower()
    
    defense_concessions = sum(1 for ind in consensus_indicators if ind in defense_lower)
    
    return defense_concessions >= 2
