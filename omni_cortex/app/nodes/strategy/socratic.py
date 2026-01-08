"""
Socratic Dialogue Framework: Real Implementation

Implements Socratic method with question-answer agents:
1. Questioner agent poses probing questions
2. Answerer agent provides responses
3. Critic agent evaluates reasoning quality
4. Iterative refinement through questioning
5. Truth emerges through dialectic

This is a REAL framework with actual multi-agent Socratic dialogue.
"""

import asyncio
import re
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

logger = structlog.get_logger("socratic")

NUM_EXCHANGES = 4


@dataclass
class Exchange:
    """A question-answer exchange in the dialogue."""
    round_num: int
    question: str
    answer: str
    critique: str
    quality_score: float = 0.0


async def _questioner_ask(
    query: str,
    code_context: str,
    previous_exchanges: list[Exchange],
    state: GraphState
) -> str:
    """Questioner agent poses a probing question."""
    
    dialogue_so_far = ""
    if previous_exchanges:
        dialogue_so_far = "\n\nDIALOGUE SO FAR:\n"
        for ex in previous_exchanges:
            dialogue_so_far += f"\nQ: {ex.question}\nA: {ex.answer}\n"
    
    prompt = f"""You are a Socratic questioner. Pose a probing question to deepen understanding.

PROBLEM: {query}

CONTEXT:
{code_context}
{dialogue_so_far}

Ask a question that:
- Challenges assumptions
- Probes deeper into reasoning
- Exposes hidden complexities
- Tests understanding
- Reveals contradictions or gaps

QUESTION:
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
    return response.strip()


async def _answerer_respond(
    question: str,
    query: str,
    code_context: str,
    previous_exchanges: list[Exchange],
    state: GraphState
) -> str:
    """Answerer agent provides thoughtful response."""
    
    dialogue_context = ""
    if previous_exchanges:
        dialogue_context = "\n\nPREVIOUS DIALOGUE:\n"
        for ex in previous_exchanges[-2:]:
            dialogue_context += f"Q: {ex.question}\nA: {ex.answer}\n"
    
    prompt = f"""You are answering Socratic questions to develop understanding.

ORIGINAL PROBLEM: {query}

CONTEXT:
{code_context}
{dialogue_context}

CURRENT QUESTION: {question}

Provide a thoughtful, well-reasoned answer. Be willing to:
- Acknowledge gaps in understanding
- Refine previous answers
- Explore implications
- Consider counterarguments

ANSWER:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=512)
    return response.strip()


async def _critic_evaluate(
    exchange: Exchange,
    query: str,
    state: GraphState
) -> tuple[str, float]:
    """Critic agent evaluates the quality of reasoning."""
    
    prompt = f"""You are a Socratic critic. Evaluate this exchange.

ORIGINAL PROBLEM: {query}

QUESTION: {exchange.question}

ANSWER: {exchange.answer}

Evaluate:
1. Does the answer address the question directly?
2. Is the reasoning sound?
3. Are assumptions acknowledged?
4. Does it advance understanding?

Provide critique and quality score (0.0-1.0).

CRITIQUE: [Your evaluation]
QUALITY: [0.0-1.0]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    
    critique = response
    quality = 0.7
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("CRITIQUE:"):
            critique = line.split(":", 1)[-1].strip()
        elif line.startswith("QUALITY:"):
            try:
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    quality = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
    
    return critique, quality


async def _synthesize_insights(
    exchanges: list[Exchange],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Synthesize insights from the Socratic dialogue."""
    
    dialogue = "\n\n".join([
        f"**Round {ex.round_num}**\n"
        f"Q: {ex.question}\n"
        f"A: {ex.answer}\n"
        f"Critique: {ex.critique} (Quality: {ex.quality_score:.2f})"
        for ex in exchanges
    ])
    
    prompt = f"""Synthesize the truth that emerged from this Socratic dialogue.

ORIGINAL PROBLEM: {query}

CONTEXT:
{code_context}

SOCRATIC DIALOGUE:
{dialogue}

What understanding emerged through this dialectic process?
What insights were uncovered through questioning?
Provide a comprehensive solution informed by the dialogue.
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def self_ask_node(state: GraphState) -> GraphState:
    """
    Socratic Dialogue Framework - REAL IMPLEMENTATION
    
    Multi-agent Socratic method:
    - Questioner poses probing questions
    - Answerer provides reasoned responses
    - Critic evaluates reasoning quality
    - Truth emerges through dialectic
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("socratic_start", query_preview=query[:50], exchanges=NUM_EXCHANGES)
    
    exchanges: list[Exchange] = []
    
    # Socratic dialogue loop
    for round_num in range(1, NUM_EXCHANGES + 1):
        logger.info("socratic_round", round=round_num)
        
        # Questioner asks
        question = await _questioner_ask(query, code_context, exchanges, state)
        
        add_reasoning_step(
            state=state,
            framework="socratic",
            thought=f"Round {round_num}: Question posed",
            action="question",
            observation=question[:100] + "..."
        )
        
        # Answerer responds
        answer = await _answerer_respond(question, query, code_context, exchanges, state)
        
        add_reasoning_step(
            state=state,
            framework="socratic",
            thought=f"Round {round_num}: Answer provided",
            action="answer"
        )
        
        # Create exchange
        exchange = Exchange(
            round_num=round_num,
            question=question,
            answer=answer,
            critique=""
        )
        
        # Critic evaluates
        critique, quality = await _critic_evaluate(exchange, query, state)
        exchange.critique = critique
        exchange.quality_score = quality
        
        exchanges.append(exchange)
        
        add_reasoning_step(
            state=state,
            framework="socratic",
            thought=f"Round {round_num}: Quality {quality:.2f}",
            action="critique",
            score=quality
        )
    
    # Synthesize insights
    solution = await _synthesize_insights(exchanges, query, code_context, state)
    
    avg_quality = sum(ex.quality_score for ex in exchanges) / len(exchanges)
    
    # Format dialogue
    dialogue_transcript = "\n\n".join([
        f"### Exchange {ex.round_num}\n"
        f"**Questioner**: {ex.question}\n\n"
        f"**Answerer**: {ex.answer}\n\n"
        f"**Critic**: {ex.critique}\n"
        f"**Quality**: {ex.quality_score:.2f}"
        for ex in exchanges
    ])
    
    final_answer = f"""# Socratic Dialogue Analysis

## Dialectic Process ({NUM_EXCHANGES} exchanges)
{dialogue_transcript}

## Emergent Understanding
{solution}

## Statistics
- Exchanges: {len(exchanges)}
- Average reasoning quality: {avg_quality:.2f}
- Highest quality exchange: Round {max(exchanges, key=lambda e: e.quality_score).round_num}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = avg_quality
    
    logger.info(
        "socratic_complete",
        exchanges=len(exchanges),
        avg_quality=avg_quality
    )
    
    return state
