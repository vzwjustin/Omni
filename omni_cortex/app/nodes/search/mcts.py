"""
Monte Carlo Tree Search (MCTS) Framework: Real Implementation

Implements genuine MCTS for problem solving:
1. Selection: Choose promising node using UCB1
2. Expansion: Add new child nodes
3. Simulation: Rollout to estimate value
4. Backpropagation: Update node statistics

This is a REAL framework with actual tree search, not a prompt template.
"""

import asyncio
import math
import random
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

logger = structlog.get_logger("mcts")

NUM_SIMULATIONS = 12
EXPLORATION_CONSTANT = 1.414
MAX_DEPTH = 5


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    thought: str
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    is_terminal: bool = False
    
    def ucb1(self, parent_visits: int) -> float:
        """Calculate UCB1 score for node selection."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = EXPLORATION_CONSTANT * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self) -> Optional["MCTSNode"]:
        """Select best child using UCB1."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1(self.visits))
    
    def path(self) -> list[str]:
        """Get path from root to this node."""
        if self.parent is None:
            return [self.thought]
        return self.parent.path() + [self.thought]


async def _expand_node(
    node: MCTSNode,
    query: str,
    code_context: str,
    state: GraphState
) -> list[MCTSNode]:
    """Expand a node by generating possible next steps."""
    
    path_so_far = "\n".join([f"Step {i+1}: {t}" for i, t in enumerate(node.path())])
    
    prompt = f"""Generate 3 different possible next steps for this problem-solving path.

PROBLEM: {query}

PATH SO FAR:
{path_so_far}

CONTEXT:
{code_context}

Generate 3 DIFFERENT next steps. Each should be a distinct approach.

Respond in this format:
STEP_1: [First possible next step]
STEP_2: [Second possible next step]
STEP_3: [Third possible next step]
TERMINAL: [yes/no - is this path ready for final answer?]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=512)
    
    children = []
    is_terminal = False
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("STEP_"):
            content = line.split(":", 1)[-1].strip()
            if content:
                child = MCTSNode(
                    thought=content,
                    parent=node,
                    depth=node.depth + 1,
                    is_terminal=(node.depth + 1 >= MAX_DEPTH)
                )
                children.append(child)
        elif line.startswith("TERMINAL:"):
            is_terminal = "yes" in line.lower()
    
    for child in children:
        if is_terminal:
            child.is_terminal = True
    
    node.children = children
    return children


async def _simulate(
    node: MCTSNode,
    query: str,
    code_context: str,
    state: GraphState
) -> float:
    """Simulate/rollout from node to estimate value."""
    
    path_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(node.path())])
    
    prompt = f"""Evaluate how promising this reasoning path is for solving the problem.

PROBLEM: {query}

REASONING PATH:
{path_text}

CONTEXT:
{code_context}

Rate this path from 0.0 to 1.0:
- 1.0: Excellent path, likely leads to correct solution
- 0.7: Good path, making progress
- 0.5: Neutral, unclear if helpful
- 0.3: Weak path, might be going wrong direction
- 0.0: Bad path, clearly wrong

Respond with ONLY a decimal number between 0.0 and 1.0.
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=32)
    
    try:
        match = re.search(r'(\d+\.?\d*)', response.strip())
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        pass
    
    return 0.5


def _backpropagate(node: MCTSNode, value: float):
    """Backpropagate simulation result up the tree."""
    current = node
    while current is not None:
        current.visits += 1
        current.value += value
        current = current.parent


async def _select_and_expand(
    root: MCTSNode,
    query: str,
    code_context: str,
    state: GraphState
) -> MCTSNode:
    """Selection phase: traverse tree to find node to expand."""
    node = root
    
    while node.children and not node.is_terminal:
        node = node.best_child()
        if node is None:
            break
    
    if not node.is_terminal and not node.children:
        await _expand_node(node, query, code_context, state)
        if node.children:
            node = random.choice(node.children)
    
    return node


async def _generate_solution(
    best_path: list[str],
    query: str,
    code_context: str,
    state: GraphState
) -> str:
    """Generate final solution from best path."""
    
    path_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(best_path)])
    
    prompt = f"""Based on this MCTS-discovered reasoning path, provide the complete solution.

PROBLEM: {query}

BEST REASONING PATH (discovered via Monte Carlo Tree Search):
{path_text}

CONTEXT:
{code_context}

Provide the complete solution following this path:
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def mcts_rstar_node(state: GraphState) -> GraphState:
    """
    Monte Carlo Tree Search Framework - REAL IMPLEMENTATION
    
    Executes genuine MCTS:
    - Selection using UCB1
    - Expansion of promising nodes
    - Simulation/rollout for value estimation
    - Backpropagation of results
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(

        query=query,

        state=state

    )
    
    logger.info("mcts_start", query_preview=query[:50], simulations=NUM_SIMULATIONS)
    
    # Initialize root
    root = MCTSNode(thought=f"Solve: {query[:100]}...", depth=0)
    
    # Expand root first
    await _expand_node(root, query, code_context, state)
    
    add_reasoning_step(
        state=state,
        framework="mcts",
        thought=f"Initialized MCTS tree with {len(root.children)} initial branches",
        action="initialize"
    )
    
    # Run MCTS simulations
    for sim in range(NUM_SIMULATIONS):
        # Selection + Expansion
        leaf = await _select_and_expand(root, query, code_context, state)
        
        # Simulation
        value = await _simulate(leaf, query, code_context, state)
        
        # Backpropagation
        _backpropagate(leaf, value)
        
        if (sim + 1) % 4 == 0:
            add_reasoning_step(
                state=state,
                framework="mcts",
                thought=f"Completed {sim + 1}/{NUM_SIMULATIONS} simulations",
                action="simulate",
                score=root.value / max(root.visits, 1)
            )
        
        logger.debug("mcts_simulation", sim=sim + 1, leaf_depth=leaf.depth, value=value)
    
    # Find best path
    def get_best_path(node: MCTSNode) -> list[str]:
        path = [node.thought]
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            path.append(node.thought)
        return path
    
    best_path = get_best_path(root)
    
    # Count total nodes
    def count_nodes(node: MCTSNode) -> int:
        return 1 + sum(count_nodes(c) for c in node.children)
    
    total_nodes = count_nodes(root)
    
    add_reasoning_step(
        state=state,
        framework="mcts",
        thought=f"Selected best path with {len(best_path)} steps",
        action="select",
        score=root.value / max(root.visits, 1)
    )
    
    # Generate solution
    solution = await _generate_solution(best_path, query, code_context, state)
    
    # Format tree statistics
    def format_tree_stats(node: MCTSNode, indent: int = 0) -> str:
        prefix = "  " * indent
        avg_value = node.value / max(node.visits, 1)
        result = f"{prefix}[V:{node.visits} Q:{avg_value:.2f}] {node.thought[:50]}...\n"
        for child in sorted(node.children, key=lambda c: c.visits, reverse=True)[:3]:
            result += format_tree_stats(child, indent + 1)
        return result
    
    tree_stats = format_tree_stats(root)
    
    final_answer = f"""# Monte Carlo Tree Search Analysis

## Search Statistics
- Simulations run: {NUM_SIMULATIONS}
- Total nodes explored: {total_nodes}
- Root visits: {root.visits}
- Average value: {root.value / max(root.visits, 1):.3f}

## Tree Structure (top branches)
```
{tree_stats}
```

## Best Path Found
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(best_path))}

## Solution
{solution}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = root.value / max(root.visits, 1)
    
    logger.info("mcts_complete", nodes=total_nodes, best_value=root.value / max(root.visits, 1))
    
    return state
