"""
Graph of Thoughts Framework: Real Implementation

Creates a graph of interconnected thoughts with:
1. Generate multiple thought nodes
2. Identify relationships/dependencies
3. Score thought quality
4. Find best path through graph
5. Synthesize from optimal path
"""

import asyncio
import re
import structlog
from dataclasses import dataclass, field
from typing import Optional

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    process_reward_model,
    add_reasoning_step,
    format_code_context,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("graph_of_thoughts")

NUM_NODES = 8


@dataclass
class ThoughtNode:
    """A node in the thought graph."""
    id: int
    content: str
    score: float = 0.0
    dependencies: list[int] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)


async def _generate_thought_nodes(
    query: str,
    code_context: str,
    state: GraphState
) -> list[ThoughtNode]:
    """Generate diverse thought nodes."""
    
    prompt = f"""Generate {NUM_NODES} different thoughts/approaches for this problem.

PROBLEM: {query}
CONTEXT: {code_context}

Create diverse thoughts covering different aspects:

THOUGHT_1: [First thought]
THOUGHT_2: [Second thought]
THOUGHT_3: [Third thought]
THOUGHT_4: [Fourth thought]
THOUGHT_5: [Fifth thought]
THOUGHT_6: [Sixth thought]
THOUGHT_7: [Seventh thought]
THOUGHT_8: [Eighth thought]
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)
    
    nodes = []
    for line in response.split("\n"):
        if line.startswith("THOUGHT_"):
            content = line.split(":", 1)[-1].strip()
            if content:
                nodes.append(ThoughtNode(id=len(nodes) + 1, content=content))
    
    while len(nodes) < NUM_NODES:
        nodes.append(ThoughtNode(id=len(nodes) + 1, content=f"Fallback thought {len(nodes) + 1}"))
    
    return nodes[:NUM_NODES]


async def _identify_dependencies(
    nodes: list[ThoughtNode],
    query: str,
    state: GraphState
) -> None:
    """Identify which thoughts depend on others."""
    
    nodes_text = "\n".join([f"{n.id}. {n.content}" for n in nodes])
    
    prompt = f"""Identify dependencies between these thoughts.

PROBLEM: {query}

THOUGHTS:
{nodes_text}

For each thought, which other thoughts does it depend on or build upon?

Format: THOUGHT_X_DEPENDS_ON: [list of IDs]

DEPENDENCIES:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    for line in response.split("\n"):
        if "DEPENDS_ON:" in line:
            try:
                thought_id = int(re.search(r'THOUGHT_(\d+)', line).group(1))
                deps = [int(d) for d in re.findall(r'\d+', line.split("DEPENDS_ON:")[-1])]
                
                if thought_id <= len(nodes):
                    nodes[thought_id - 1].dependencies = deps
                    for dep_id in deps:
                        if dep_id <= len(nodes):
                            nodes[dep_id - 1].successors.append(thought_id)
            except Exception as e:
                logger.debug("dep_parsing_failed", error=str(e))
                pass


async def _score_nodes(
    nodes: list[ThoughtNode],
    query: str,
    code_context: str,
    state: GraphState
) -> None:
    """Score quality of each thought node."""
    
    tasks = [
        process_reward_model(
            step=node.content,
            context=code_context,
            goal=query
        )
        for node in nodes
    ]
    
    scores = await asyncio.gather(*tasks)
    
    for node, score in zip(nodes, scores):
        node.score = score


def _find_best_path(nodes: list[ThoughtNode]) -> list[ThoughtNode]:
    """Find highest-scoring path through graph."""
    
    # Simple: start with highest-scoring node and follow successors
    sorted_nodes = sorted(nodes, key=lambda n: n.score, reverse=True)
    
    path = [sorted_nodes[0]]
    current = sorted_nodes[0]
    
    visited = {current.id}
    
    while current.successors:
        # Pick highest-scoring unvisited successor
        candidates = [n for n in nodes if n.id in current.successors and n.id not in visited]
        if not candidates:
            break
        
        next_node = max(candidates, key=lambda n: n.score)
        path.append(next_node)
        visited.add(next_node.id)
        current = next_node
    
    return path


async def _synthesize_from_path(
    path: list[ThoughtNode],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Synthesize solution from thought path."""
    
    path_text = "\n".join([f"{i+1}. {node.content}" for i, node in enumerate(path)])
    
    prompt = f"""Synthesize solution from this path through the thought graph.

PROBLEM: {query}

THOUGHT PATH (highest quality):
{path_text}

CONTEXT: {code_context}

SOLUTION:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def graph_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Graph of Thoughts - REAL IMPLEMENTATION
    
    Creates interconnected thought graph:
    - Multiple thought nodes
    - Dependency relationships
    - Quality scoring
    - Optimal path finding
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("graph_of_thoughts_start", query_preview=query[:50])
    
    # Generate nodes
    nodes = await _generate_thought_nodes(query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="graph_of_thoughts",
        thought=f"Generated {len(nodes)} thought nodes",
        action="generate"
    )
    
    # Identify dependencies
    await _identify_dependencies(nodes, query, state)
    
    add_reasoning_step(
        state=state,
        framework="graph_of_thoughts",
        thought="Identified thought dependencies",
        action="connect"
    )
    
    # Score nodes
    await _score_nodes(nodes, query, code_context, state)
    
    best_node = max(nodes, key=lambda n: n.score)
    
    add_reasoning_step(
        state=state,
        framework="graph_of_thoughts",
        thought=f"Scored nodes, best: {best_node.score:.2f}",
        action="score",
        score=best_node.score
    )
    
    # Find best path
    path = _find_best_path(nodes)
    
    add_reasoning_step(
        state=state,
        framework="graph_of_thoughts",
        thought=f"Found optimal path with {len(path)} nodes",
        action="path_find"
    )
    
    # Synthesize
    solution = await _synthesize_from_path(path, query, code_context, state)
    
    # Visualize graph
    graph_viz = "```\n"
    for node in nodes:
        deps = f" <- [{','.join(str(d) for d in node.dependencies)}]" if node.dependencies else ""
        graph_viz += f"[{node.id}] (score: {node.score:.2f}) {node.content[:40]}...{deps}\n"
    graph_viz += "```"
    
    path_viz = " → ".join([str(n.id) for n in path])
    
    final_answer = f"""# Graph of Thoughts Analysis

## Thought Graph
{graph_viz}

## Optimal Path
{path_viz}

## Path Thoughts
{chr(10).join(f'{i+1}. {node.content}' for i, node in enumerate(path))}

## Solution
{solution}

## Statistics
- Nodes: {len(nodes)}
- Path length: {len(path)}
- Average score: {sum(n.score for n in nodes) / len(nodes):.2f}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = sum(n.score for n in path) / len(path) if path else 0.5
    
    logger.info("graph_of_thoughts_complete", nodes=len(nodes), path_length=len(path))
    
    return state
