"""
Self-RAG: Self-Triggered Retrieval

Retrieve only when needed; self-critique triggers retrieval
to fill knowledge gaps. Selective, not blanket retrieval.
(Headless Mode: Returns Reasoning Protocol for Client Execution)
"""

import logging
from ...state import GraphState
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
)

logger = logging.getLogger(__name__)


@quiet_star
async def self_rag_node(state: GraphState) -> GraphState:
    """
    Framework: Self-RAG
    Self-triggered selective retrieval.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# Self-RAG Protocol

I have selected the **Self-RAG** framework for this task.
Retrieve only when needed, not as a reflex.

## Use Case
Mixed "I know some, need some" tasks, large corpora, minimizing irrelevant retrieval

## Task
{query}

## Execution Protocol (Client-Side)

Selective retrieval based on confidence.

### Framework Steps
1. **DRAFT WITH CONFIDENCE**: Write answer with confidence tags
   - HIGH: Confident, no retrieval needed
   - MEDIUM: Mostly sure, verify if possible
   - LOW: Uncertain, retrieval required
2. **IDENTIFY GAPS**: For LOW-confidence segments:
   - What specific information is missing?
   - What would resolve the uncertainty?
3. **GENERATE QUERIES**: Create targeted retrieval queries
4. **RETRIEVE**: Fetch evidence for low-confidence parts only
5. **UPDATE**: Revise only the uncertain segments
6. **SELF-CRITIQUE**: Confirm groundedness:
   - Is each claim now supported?
   - Remove anything still unsupported
7. **FINALIZE**: Deliver grounded answer with evidence anchors

## Code Context
{code_context}

**Draft, identify confidence gaps, retrieve selectively, verify groundedness.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="self_rag",
        thought="Generated Self-RAG protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
