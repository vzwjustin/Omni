"""
rStar-Code MCTS: Monte Carlo Tree Search for Code Patches

Implements MCTS-style search for exploring code modification
space, with simulated rollouts and PRM-based pruning.
"""

import asyncio
import random
import math
from typing import Optional, Any
from dataclasses import dataclass, field
from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    process_reward_model,
    add_reasoning_step,
    format_code_context
)
from ...core.config import settings


@dataclass
class MCTSNode:
    """A node in the Monte Carlo search tree."""
    id: str
    parent_id: Optional[str]
    state_description: str
    code_state: Optional[str]
    action_taken: str
    visits: int = 0
    total_value: float = 0.0
    children: list[str] = field(default_factory=list)
    is_terminal: bool = False
    
    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_value / self.visits
        exploration = math.sqrt(2 * math.log(self.visits + 1) / self.visits)
        return exploitation + exploration
    
    @property
    def average_value(self) -> float:
        """Average value from rollouts."""
        return self.total_value / self.visits if self.visits > 0 else 0.0


class MCTSTree:
    """Monte Carlo Tree for code patch exploration."""
    
    def __init__(self):
        self.nodes: dict[str, MCTSNode] = {}
        self.node_counter = 0
    
    def create_root(self, state_description: str, code_state: str) -> MCTSNode:
        """Create the root node."""
        node = MCTSNode(
            id="root",
            parent_id=None,
            state_description=state_description,
            code_state=code_state,
            action_taken="initial_state"
        )
        self.nodes["root"] = node
        return node
    
    def add_child(
        self,
        parent_id: str,
        state_description: str,
        code_state: str,
        action: str
    ) -> MCTSNode:
        """Add a child node."""
        self.node_counter += 1
        node_id = f"node_{self.node_counter}"
        
        node = MCTSNode(
            id=node_id,
            parent_id=parent_id,
            state_description=state_description,
            code_state=code_state,
            action_taken=action
        )
        self.nodes[node_id] = node
        self.nodes[parent_id].children.append(node_id)
        
        return node
    
    def select_best_leaf(self) -> MCTSNode:
        """Select best leaf node using UCB."""
        current = self.nodes["root"]
        
        while current.children:
            children = [self.nodes[cid] for cid in current.children]
            current = max(children, key=lambda n: n.ucb_score)
        
        return current
    
    def backpropagate(self, node_id: str, value: float) -> None:
        """Backpropagate value up the tree."""
        current_id = node_id
        while current_id is not None:
            node = self.nodes[current_id]
            node.visits += 1
            node.total_value += value
            current_id = node.parent_id
    
    def get_best_path(self) -> list[MCTSNode]:
        """Get the best path from root to best leaf."""
        path = [self.nodes["root"]]
        current = self.nodes["root"]
        
        while current.children:
            children = [self.nodes[cid] for cid in current.children]
            best_child = max(children, key=lambda n: n.average_value)
            path.append(best_child)
            current = best_child
        
        return path


@quiet_star
async def mcts_rstar_node(state: GraphState) -> GraphState:
    """
    rStar-Code MCTS: Monte Carlo Tree Search for Code.
    
    Four-phase MCTS:
    1. SELECT: Use UCB to find promising leaf
    2. EXPAND: Generate child nodes (code modifications)
    3. SIMULATE: Evaluate modifications with PRM
    4. BACKPROPAGATE: Update values up the tree
    
    Best for: Complex bugs, multi-step solutions, optimization
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    max_rollouts = min(settings.mcts_max_rollouts, 30)
    
    # Initialize search tree
    tree = MCTSTree()
    root = tree.create_root(
        state_description=query,
        code_state=state.get("code_snippet", "")
    )
    
    add_reasoning_step(
        state=state,
        framework="mcts_rstar",
        thought="Initialized MCTS search tree",
        action="initialization",
        observation=f"Starting search with max {max_rollouts} rollouts"
    )
    
    # =========================================================================
    # Main MCTS Loop
    # =========================================================================
    
    for rollout in range(max_rollouts):
        # PHASE 1: SELECT - Find best leaf using UCB
        leaf = tree.select_best_leaf()
        
        # PHASE 2: EXPAND - Generate possible modifications
        if not leaf.is_terminal and leaf.visits > 0:
            expand_prompt = f"""You are exploring code modifications via MCTS.

GOAL: {query}

CURRENT STATE:
{leaf.state_description}

CURRENT CODE:
```
{leaf.code_state or 'No code yet'}
```

ACTION THAT LED HERE:
{leaf.action_taken}

Generate 2-3 DIFFERENT possible next modifications. For each:
1. [ACTION]: One-line description of the change
2. [RATIONALE]: Why this might help
3. [CODE]: The modified code snippet

Be creative but grounded. Explore different approaches."""

            expand_response, _ = await call_fast_synthesizer(
                prompt=expand_prompt,
                state=state,
                max_tokens=2000,
                temperature=0.8
            )
            
            # Parse and create child nodes
            children = _parse_mcts_expansions(expand_response, leaf.code_state)
            
            for child_data in children[:3]:  # Limit branching factor
                tree.add_child(
                    parent_id=leaf.id,
                    state_description=child_data["description"],
                    code_state=child_data["code"],
                    action=child_data["action"]
                )
            
            # Select one child to simulate
            if children:
                leaf = tree.nodes[tree.nodes[leaf.id].children[0]]
        
        # PHASE 3: SIMULATE - Evaluate with PRM
        prm_score = await process_reward_model(
            step=leaf.action_taken,
            context=code_context,
            goal=query,
            previous_steps=[tree.nodes[pid].action_taken for pid in _get_ancestors(tree, leaf.id)]
        )
        
        # PHASE 4: BACKPROPAGATE - Update tree values
        tree.backpropagate(leaf.id, prm_score)
        
        # Record progress periodically
        if rollout % 10 == 9:
            add_reasoning_step(
                state=state,
                framework="mcts_rstar",
                thought=f"Completed {rollout + 1} MCTS rollouts",
                action="search_progress",
                observation=f"Best path score: {tree.nodes['root'].average_value:.2f}",
                score=tree.nodes["root"].average_value
            )
    
    # =========================================================================
    # Extract Best Solution
    # =========================================================================
    
    best_path = tree.get_best_path()
    
    # Summarize the search result
    summary_prompt = f"""Summarize the MCTS search results.

GOAL: {query}

ORIGINAL CODE:
```
{state.get('code_snippet', 'No original code')}
```

BEST PATH FOUND ({len(best_path)} steps):
{"\n".join(f"{i+1}. {n.action_taken} (score: {n.average_value:.2f})" for i, n in enumerate(best_path))}

FINAL CODE STATE:
```
{best_path[-1].code_state if best_path[-1].code_state else 'No code generated'}
```

Provide:
1. **SUMMARY**: What the search discovered
2. **FINAL SOLUTION**: Polished implementation
3. **CODE**: Clean, final code
4. **CONFIDENCE**: How confident in this solution"""

    summary_response, _ = await call_deep_reasoner(
        prompt=summary_prompt,
        state=state,
        system="You are rStar-Code summarizing MCTS search results.",
        temperature=0.5
    )
    
    add_reasoning_step(
        state=state,
        framework="mcts_rstar",
        thought=f"Completed MCTS with {max_rollouts} rollouts",
        action="search_complete",
        observation=f"Best path length: {len(best_path)}, Final score: {best_path[-1].average_value:.2f}",
        score=best_path[-1].average_value
    )
    
    # Extract code if present
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, summary_response, re.DOTALL)
    
    # Store search tree in state for reference
    state["search_tree"] = {
        "total_nodes": len(tree.nodes),
        "best_path_length": len(best_path),
        "rollouts": max_rollouts
    }
    state["best_path"] = [n.action_taken for n in best_path]
    state["prm_scores"] = [n.average_value for n in best_path]
    state["cumulative_prm_score"] = best_path[-1].average_value
    
    # Update final state
    state["final_answer"] = summary_response
    state["final_code"] = best_path[-1].code_state or ("\n\n".join([m.strip() for m in matches]) if matches else None)
    state["confidence_score"] = best_path[-1].average_value
    
    return state


def _parse_mcts_expansions(response: str, current_code: str) -> list[dict]:
    """Parse expansion response into child node data."""
    children = []
    
    # Simple parsing - look for numbered items
    import re
    items = re.split(r'\n\d+\.', response)
    
    for item in items[1:]:  # Skip first empty split
        action_match = re.search(r'\[ACTION\][:\s]*(.+?)(?:\n|$)', item, re.IGNORECASE)
        code_match = re.search(r'```(?:\w+)?\n(.*?)```', item, re.DOTALL)
        
        children.append({
            "action": action_match.group(1).strip() if action_match else "Modify code",
            "description": item[:200].strip(),
            "code": code_match.group(1).strip() if code_match else current_code
        })
    
    # Fallback if parsing failed
    if not children:
        children = [{"action": "Continue analysis", "description": response[:200], "code": current_code}]
    
    return children


def _get_ancestors(tree: MCTSTree, node_id: str) -> list[str]:
    """Get all ancestor node IDs."""
    ancestors = []
    current_id = tree.nodes[node_id].parent_id
    
    while current_id is not None:
        ancestors.append(current_id)
        current_id = tree.nodes[current_id].parent_id
    
    return ancestors
