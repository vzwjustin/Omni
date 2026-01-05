"""
RAPTOR: Recursive Abstraction Retrieval

Build hierarchical summaries to enable retrieval across long
documents/codebases. Prevents getting lost in chunks.
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
async def raptor_node(state: GraphState) -> GraphState:
    """
    Framework: RAPTOR
    Hierarchical abstraction retrieval.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# RAPTOR Protocol

I have selected the **RAPTOR** (Recursive Abstraction Retrieval) framework.
Hierarchical retrieval for long documents and large codebases.

## Use Case
Huge repos, long design docs, monorepos with repeated patterns

## Task
{query}

## Execution Protocol (Client-Side)

Prevent "lost in chunk land" with hierarchical thinking.

### Framework Steps
1. **BUILD HIERARCHY**: Create conceptual summaries at multiple levels:
   - CHUNK level: Individual code blocks/paragraphs
   - SECTION level: Module/file summaries
   - DOC level: Component/system overviews
2. **RETRIEVE TOP-DOWN**: Start at high level:
   - Which sections are relevant?
   - Which files within those sections?
   - Which specific chunks matter?
3. **DRILL DOWN**: Gather supporting details from lower levels
4. **SYNTHESIZE**: Combine abstraction with specifics
5. **ANCHOR**: Provide both:
   - High-level summary context
   - Specific snippet citations
6. **VERIFY**: Ensure details support the abstractions

## Code Context
{code_context}

**Think hierarchically: overview first, then drill into relevant details.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="raptor",
        thought="Generated RAPTOR protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
