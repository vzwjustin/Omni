"""
LATS (Language Agent Tree Search): Real Implementation

Combines tree search with language agents:
1. Generate multiple action candidates
2. Self-evaluate each action
3. Select best action
4. Execute and get observation
5. Repeat with backtracking
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

logger = structlog.get_logger("lats")

MAX_DEPTH = 4
NUM_ACTIONS = 3


@dataclass
class ActionNode:
    """Node in the action tree."""
    action: str
    observation: str
    value: float
    depth: int
    parent: Optional["ActionNode"] = None
    children: list["ActionNode"] = field(default_factory=list)
    
    def path(self) -> list[str]:
        if self.parent is None:
            return [self.action]
        return self.parent.path() + [self.action]


async def _generate_action_candidates(
    query: str,
    code_context: str,
    current_path: list[str],
    state: GraphState
) -> list[str]:
    """Generate candidate actions."""
    
    path_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(current_path))
    
    prompt = f"""Generate {NUM_ACTIONS} possible next actions.

PROBLEM: {query}
CONTEXT: {code_context}

ACTIONS SO FAR:
{path_text if path_text else 'None'}

What are {NUM_ACTIONS} different actions you could take next?

ACTION_1:
ACTION_2:
ACTION_3:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=384)
    
    actions = []
    for line in response.split("\n"):
        if line.startswith("ACTION_"):
            action = line.split(":", 1)[-1].strip()
            if action:
                actions.append(action)
    
    return actions[:NUM_ACTIONS]


async def _self_evaluate_action(
    action: str,
    query: str,
    code_context: str,
    state: GraphState
) -> float:
    """Agent evaluates action value."""
    
    prompt = f"""Evaluate how valuable this action is (0.0-1.0).

PROBLEM: {query}
ACTION: {action}

How likely is this action to lead to a solution?

VALUE:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=32)
    
    try:
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
    except Exception as e:
        logger.debug("score_parsing_failed", response=score_response[:50] if "score_response" in locals() else response[:50], error=str(e))
        pass
    
    return 0.5


async def _execute_action(
    action: str,
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Execute action and get observation."""
    
    prompt = f"""Execute this action and describe the observation.

PROBLEM: {query}
ACTION: {action}
CONTEXT: {code_context}

What do you observe after taking this action?

OBSERVATION:
"""
    
    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=256)
    return response.strip()


async def _synthesize_solution(
    best_path: list[ActionNode],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Synthesize solution from action path."""
    
    path_text = "\n".join([
        f"{i+1}. Action: {node.action}\n   Observation: {node.observation}\n   Value: {node.value:.2f}"
        for i, node in enumerate(best_path)
    ])
    
    prompt = f"""Based on this action sequence, provide the final solution.

PROBLEM: {query}

ACTION SEQUENCE:
{path_text}

CONTEXT: {code_context}

SOLUTION:
"""
    
    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1536)
    return response


@quiet_star
async def lats_node(state: GraphState) -> GraphState:
    """
    LATS (Language Agent Tree Search) - REAL IMPLEMENTATION
    
    Tree search with language agents:
    - Generates action candidates
    - Self-evaluates each
    - Selects best path
    - Combines search with LLM reasoning
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("lats_start", query_preview=query[:50])
    
    # Root node
    root = ActionNode(action="START", observation="Initial state", value=1.0, depth=0)
    all_nodes = [root]
    
    # BFS expansion
    frontier = [root]
    
    for depth in range(MAX_DEPTH):
        if not frontier:
            break
        
        new_frontier = []
        
        for node in frontier:
            # Generate candidates
            candidates = await _generate_action_candidates(
                query, code_context, node.path(), state
            )
            
            # Evaluate and expand
            for action in candidates:
                value = await _self_evaluate_action(action, query, code_context, state)
                observation = await _execute_action(action, query, code_context, state)
                
                child = ActionNode(
                    action=action,
                    observation=observation,
                    value=value,
                    depth=depth + 1,
                    parent=node
                )
                node.children.append(child)
                all_nodes.append(child)
                new_frontier.append(child)
            
            add_reasoning_step(
                state=state,
                framework="lats",
                thought=f"Expanded node at depth {depth}: {len(candidates)} children",
                action="expand"
            )
        
        # Prune: keep only top nodes
        new_frontier.sort(key=lambda n: n.value, reverse=True)
        frontier = new_frontier[:NUM_ACTIONS]
        
        logger.info("lats_depth", depth=depth + 1, nodes=len(all_nodes))
    
    # Find best path
    leaf_nodes = [n for n in all_nodes if not n.children]
    if not leaf_nodes:
        leaf_nodes = [all_nodes[-1]]
    
    best_leaf = max(leaf_nodes, key=lambda n: n.value)
    best_path = []
    node = best_leaf
    while node.parent is not None:
        best_path.insert(0, node)
        node = node.parent
    
    add_reasoning_step(
        state=state,
        framework="lats",
        thought=f"Selected best path: {len(best_path)} actions",
        action="select",
        score=best_leaf.value
    )
    
    # Synthesize
    solution = await _synthesize_solution(best_path, query, code_context, state)
    
    # Format tree
    tree_viz = "```\n"
    for node in all_nodes[:20]:  # Show first 20
        indent = "  " * node.depth
        tree_viz += f"{indent}[{node.value:.2f}] {node.action[:40]}...\n"
    tree_viz += "```"
    
    final_answer = f"""# LATS Analysis

## Search Tree (partial)
{tree_viz}

## Best Path
{chr(10).join(f'{i+1}. {node.action} (value: {node.value:.2f})' for i, node in enumerate(best_path))}

## Solution
{solution}

## Statistics
- Nodes explored: {len(all_nodes)}
- Max depth: {max(n.depth for n in all_nodes)}
- Best path value: {best_leaf.value:.2f}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = best_leaf.value
    
    logger.info("lats_complete", nodes=len(all_nodes), best_value=best_leaf.value)
    
    return state
