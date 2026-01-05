"""
GraphRAG: Entity-Relation Grounding

Ground answers using entity-relation structure. Understand
how components relate through explicit dependency graphs.
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
async def graphrag_node(state: GraphState) -> GraphState:
    """
    Framework: GraphRAG
    Entity-relation grounding for dependencies.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# GraphRAG Protocol

I have selected the **GraphRAG** framework for this task.
Entity-relation grounding for understanding dependencies and relationships.

## Use Case
Architecture questions, "how do modules relate", dependency analysis, impact assessments

## Task
{query}

## Execution Protocol (Client-Side)

Make relationships explicit for impact analysis.

### Framework Steps
1. **EXTRACT ENTITIES**: Identify key components:
   - Modules/Classes
   - APIs/Endpoints
   - Tables/Stores
   - Services/Workers
   - Config items
2. **MAP RELATIONS**: Document how entities connect:
   - calls/imports
   - reads/writes
   - owns/contains
   - triggers/listens
3. **BUILD GRAPH**: Create conceptual relation map:
   - Entity: [Relations]
   - Identify clusters/layers
4. **QUERY GRAPH**: Answer the question using graph structure:
   - Trace paths between entities
   - Find blast radius of changes
   - Identify dependencies
5. **GROUND ANSWER**: Base conclusions on graph evidence
6. **CITE PATHS**: Show the relationship chains that support claims

## Code Context
{code_context}

**Extract entities, map relations, query the graph for your answer.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="graphrag",
        thought="Generated GraphRAG protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
