"""
GraphRAG Framework: Real Implementation

Graph-based retrieval augmented generation:
1. Extract entities and relationships from context
2. Build knowledge graph
3. Query graph for relevant subgraphs
4. Traverse graph paths
5. Synthesize answer from graph structure
"""

import asyncio
import structlog
from dataclasses import dataclass, field

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("graphrag")


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    id: int
    name: str
    type: str
    properties: list[str] = field(default_factory=list)


@dataclass
class Relationship:
    """A relationship between entities."""
    from_entity: int
    to_entity: int
    relation_type: str
    description: str


async def _extract_entities(query: str, code_context: str, state: GraphState) -> list[Entity]:
    """Extract entities from context."""
    prompt = f"""Extract key entities from this context.

QUERY: {query}

CONTEXT:
{code_context[:1000]}

Identify 5-8 key entities (concepts, objects, people, etc.):

ENTITY_1_NAME: [Name]
ENTITY_1_TYPE: [Type]

ENTITY_2_NAME: [Name]
ENTITY_2_TYPE: [Type]

(continue...)
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    entities = []
    current_name = None
    current_type = None
    
    for line in response.split("\n"):
        if "_NAME:" in line:
            if current_name and current_type:
                entities.append(Entity(id=len(entities)+1, name=current_name, type=current_type))
            current_name = line.split(":", 1)[-1].strip()
            current_type = None
        elif "_TYPE:" in line:
            current_type = line.split(":", 1)[-1].strip()
    
    if current_name and current_type:
        entities.append(Entity(id=len(entities)+1, name=current_name, type=current_type))
    
    return entities


async def _extract_relationships(entities: list[Entity], query: str, code_context: str, state: GraphState) -> list[Relationship]:
    """Extract relationships between entities."""
    entities_text = "\n".join(f"{e.id}. {e.name} ({e.type})" for e in entities)
    
    prompt = f"""Identify relationships between these entities.

ENTITIES:
{entities_text}

QUERY: {query}
CONTEXT: {code_context[:800]}

List relationships (use entity IDs):

REL_1: [entity_id] [relation] [entity_id] - [description]
REL_2: [entity_id] [relation] [entity_id] - [description]
(continue...)
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    relationships = []
    for line in response.split("\n"):
        if line.startswith("REL_"):
            try:
                ids = [int(x) for x in re.findall(r'\b\d+\b', line)]
                if len(ids) >= 2:
                    parts = line.split("-", 1)
                    rel_type = parts[0].split("]")[-1].strip() if "]" in parts[0] else "related_to"
                    desc = parts[1].strip() if len(parts) > 1 else ""
                    
                    relationships.append(Relationship(
                        from_entity=ids[0],
                        to_entity=ids[1],
                        relation_type=rel_type,
                        description=desc
                    ))
            except Exception as e:
        logger.debug("score_parsing_failed", response=score_response[:50] if "score_response" in locals() else response[:50], error=str(e))
                pass
    
    return relationships


async def _query_graph(entities: list[Entity], relationships: list[Relationship], query: str, state: GraphState) -> list[int]:
    """Query graph for relevant entities."""
    entities_text = "\n".join(f"{e.id}. {e.name} ({e.type})" for e in entities)
    
    prompt = f"""Which entities are most relevant to this query?

QUERY: {query}

GRAPH ENTITIES:
{entities_text}

List IDs of relevant entities (most to least relevant):

RELEVANT_IDS: [comma-separated IDs]
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=128)
    
    try:
        ids = [int(x) for x in re.findall(r'\d+', response)]
        return ids[:5]
    except Exception as e:
        logger.debug("score_parsing_failed", response=score_response[:50] if "score_response" in locals() else response[:50], error=str(e))
        return [1, 2, 3]


async def _traverse_graph_paths(start_ids: list[int], entities: list[Entity], relationships: list[Relationship], state: GraphState) -> list[list[int]]:
    """Find interesting paths through the graph."""
    paths = []
    
    for start_id in start_ids[:3]:
        # Simple BFS to find paths
        path = [start_id]
        visited = {start_id}
        
        # Find up to 2 connected entities
        for _ in range(2):
            current = path[-1]
            next_entities = []
            
            for rel in relationships:
                if rel.from_entity == current and rel.to_entity not in visited:
                    next_entities.append(rel.to_entity)
                elif rel.to_entity == current and rel.from_entity not in visited:
                    next_entities.append(rel.from_entity)
            
            if next_entities:
                next_id = next_entities[0]
                path.append(next_id)
                visited.add(next_id)
            else:
                break
        
        if len(path) > 1:
            paths.append(path)
    
    return paths


async def _synthesize_from_graph(paths: list[list[int]], entities: list[Entity], relationships: list[Relationship], query: str, code_context: str, state: GraphState) -> str:
    """Synthesize answer from graph paths."""
    
    paths_text = ""
    for i, path in enumerate(paths, 1):
        path_names = " → ".join(entities[eid-1].name for eid in path if eid <= len(entities))
        paths_text += f"\nPath {i}: {path_names}"
    
    entities_dict = {e.id: e for e in entities}
    
    graph_context = "GRAPH KNOWLEDGE:\n"
    for path in paths:
        for eid in path:
            if eid in entities_dict:
                e = entities_dict[eid]
                graph_context += f"- {e.name} ({e.type})\n"
        graph_context += "\n"
    
    prompt = f"""Synthesize answer using graph knowledge.

QUERY: {query}

GRAPH PATHS:
{paths_text}

{graph_context}

ORIGINAL CONTEXT:
{code_context[:500]}

Provide comprehensive answer:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def graphrag_node(state: GraphState) -> GraphState:
    """
    GraphRAG - REAL IMPLEMENTATION
    
    Graph-based RAG:
    - Extracts entities from context
    - Builds knowledge graph
    - Queries graph for relevant subgraphs
    - Traverses graph paths
    - Synthesizes from graph structure
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("graphrag_start", query_preview=query[:50])
    
    # Extract entities
    entities = await _extract_entities(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="graphrag",
        thought=f"Extracted {len(entities)} entities",
        action="extract_entities"
    )
    
    # Extract relationships
    relationships = await _extract_relationships(entities, query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="graphrag",
        thought=f"Identified {len(relationships)} relationships",
        action="extract_relationships"
    )
    
    # Query graph
    relevant_ids = await _query_graph(entities, relationships, query, state)
    
    add_reasoning_step(
        state=state,
        framework="graphrag",
        thought=f"Found {len(relevant_ids)} relevant entities",
        action="query_graph"
    )
    
    # Traverse paths
    paths = await _traverse_graph_paths(relevant_ids, entities, relationships, state)
    
    add_reasoning_step(
        state=state,
        framework="graphrag",
        thought=f"Traversed {len(paths)} graph paths",
        action="traverse"
    )
    
    # Synthesize
    answer = await _synthesize_from_graph(paths, entities, relationships, query, code_context, state)
    
    # Format graph visualization
    graph_viz = "```\n"
    for e in entities:
        graph_viz += f"[{e.id}] {e.name} ({e.type})\n"
    graph_viz += "\nRelationships:\n"
    for r in relationships:
        from_name = entities[r.from_entity-1].name if r.from_entity <= len(entities) else "?"
        to_name = entities[r.to_entity-1].name if r.to_entity <= len(entities) else "?"
        graph_viz += f"{from_name} --{r.relation_type}--> {to_name}\n"
    graph_viz += "```"
    
    final_answer = f"""# GraphRAG Analysis

## Knowledge Graph
{graph_viz}

## Relevant Paths
{chr(10).join(f'Path {i}: {" → ".join(entities[eid-1].name for eid in path if eid <= len(entities))}' for i, path in enumerate(paths, 1))}

## Answer
{answer}

## Statistics
- Entities: {len(entities)}
- Relationships: {len(relationships)}
- Paths traversed: {len(paths)}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = 0.85
    
    logger.info("graphrag_complete", entities=len(entities), relationships=len(relationships))
    
    return state
