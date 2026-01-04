"""
Tree of Thoughts (ToT): BFS/DFS Exploration

Classic tree-based search for algorithmic brainstorming
and problem-solving with explicit thought branching.
"""

import asyncio
from typing import Optional, Any
from dataclasses import dataclass, field
from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    process_reward_model,
    batch_score_steps,
    add_reasoning_step,
    format_code_context
)


@dataclass
class ThoughtNode:
    """A node in the thought tree."""
    id: str
    thought: str
    parent_id: Optional[str]
    depth: int
    score: float = 0.0
    children: list[str] = field(default_factory=list)
    is_solution: bool = False


class ThoughtTree:
    """Tree structure for exploring thoughts via BFS/DFS."""
    
    def __init__(self, max_depth: int = 4, branching_factor: int = 3):
        self.nodes: dict[str, ThoughtNode] = {}
        self.node_counter = 0
        self.max_depth = max_depth
        self.branching_factor = branching_factor
    
    def add_node(
        self,
        thought: str,
        parent_id: Optional[str] = None,
        score: float = 0.0
    ) -> ThoughtNode:
        """Add a thought node to the tree."""
        self.node_counter += 1
        node_id = f"thought_{self.node_counter}"
        depth = 0 if parent_id is None else self.nodes[parent_id].depth + 1
        
        node = ThoughtNode(
            id=node_id,
            thought=thought,
            parent_id=parent_id,
            depth=depth,
            score=score
        )
        
        self.nodes[node_id] = node
        if parent_id:
            self.nodes[parent_id].children.append(node_id)
        
        return node
    
    def get_best_path(self) -> list[ThoughtNode]:
        """Get the path to the highest-scoring leaf."""
        if not self.nodes:
            return []
        
        # Find best leaf
        leaves = [n for n in self.nodes.values() if not n.children]
        if not leaves:
            return []
        
        best_leaf = max(leaves, key=lambda n: n.score)
        
        # Trace path from root
        path = []
        current = best_leaf
        while current:
            path.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        
        return list(reversed(path))
    
    def get_frontier(self) -> list[ThoughtNode]:
        """Get leaf nodes for expansion."""
        return [
            n for n in self.nodes.values()
            if not n.children and n.depth < self.max_depth
        ]


@quiet_star
async def tree_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Tree of Thoughts: BFS/DFS Thought Exploration.
    
    Process:
    1. Generate initial thought branches
    2. Score each branch with PRM
    3. Expand best branches (BFS)
    4. Continue until solution found or max depth
    
    Best for: Algorithms, optimization, problem solving
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    tree = ThoughtTree(max_depth=4, branching_factor=3)
    
    # =========================================================================
    # Generate Initial Thoughts (Depth 0)
    # =========================================================================
    
    initial_prompt = f"""You are exploring solutions using Tree of Thoughts.

PROBLEM: {query}

CONTEXT:
{code_context}

Generate 3 DIFFERENT initial approaches to this problem.
Each approach should represent a fundamentally different strategy.

Format:
APPROACH 1: [Strategy name]
[2-3 sentences describing this approach and why it might work]

APPROACH 2: [Strategy name]
[2-3 sentences describing this approach and why it might work]

APPROACH 3: [Strategy name]
[2-3 sentences describing this approach and why it might work]

Think creatively - explore diverse solution spaces."""

    initial_response, _ = await call_fast_synthesizer(
        prompt=initial_prompt,
        state=state,
        max_tokens=1000,
        temperature=0.8
    )
    
    # Parse and create initial branches
    approaches = _parse_approaches(initial_response)
    
    for approach in approaches[:3]:
        tree.add_node(approach["description"], parent_id=None)
    
    add_reasoning_step(
        state=state,
        framework="tree_of_thoughts",
        thought=f"Generated {len(approaches)} initial approaches",
        action="initial_branching",
        observation=", ".join([a["name"] for a in approaches[:3]])
    )
    
    # =========================================================================
    # BFS Expansion Loop
    # =========================================================================
    
    for depth in range(tree.max_depth):
        frontier = tree.get_frontier()
        if not frontier:
            break
        
        # Score current frontier
        thoughts = [n.thought for n in frontier]
        scores = await batch_score_steps(thoughts, code_context, query)
        
        for node, score in zip(frontier, scores):
            node.score = score
        
        # Select top nodes for expansion (beam search)
        beam_width = 2
        frontier.sort(key=lambda n: n.score, reverse=True)
        to_expand = frontier[:beam_width]
        
        # Expand each selected node
        for node in to_expand:
            expand_prompt = f"""Continue developing this approach.

PROBLEM: {query}

CURRENT APPROACH:
{node.thought}

PATH SO FAR:
{_get_path_text(tree, node.id)}

Generate 2-3 next steps or sub-thoughts that develop this approach further.
Each should be a concrete step toward a solution.

If you've reached a complete solution, mark it as [SOLUTION].

Format:
STEP 1: [Description of next step or refinement]
STEP 2: [Alternative continuation]
STEP 3: [Another option, or [SOLUTION] if complete]"""

            expand_response, _ = await call_fast_synthesizer(
                prompt=expand_prompt,
                state=state,
                max_tokens=800,
                temperature=0.7
            )
            
            # Parse and add children
            steps = _parse_steps(expand_response)
            for step in steps[:3]:
                child = tree.add_node(step["content"], parent_id=node.id)
                if step["is_solution"]:
                    child.is_solution = True
                    child.score = 1.0
        
        add_reasoning_step(
            state=state,
            framework="tree_of_thoughts",
            thought=f"Expanded depth {depth + 1}",
            action="tree_expansion",
            observation=f"Tree now has {len(tree.nodes)} nodes"
        )
        
        # Check for solution
        solutions = [n for n in tree.nodes.values() if n.is_solution]
        if solutions:
            break
    
    # =========================================================================
    # Extract Best Solution
    # =========================================================================
    
    best_path = tree.get_best_path()
    
    synthesis_prompt = f"""Synthesize the solution from Tree of Thoughts exploration.

PROBLEM: {query}

CONTEXT:
{code_context}

BEST PATH FOUND:
{"\n".join(f"{i+1}. {n.thought} [Score: {n.score:.2f}]" for i, n in enumerate(best_path))}

Based on this exploration, provide:
1. **SOLUTION**: Clear explanation of the chosen approach
2. **IMPLEMENTATION**: Step-by-step how to implement it
3. **CODE** (if applicable): Working implementation
4. **JUSTIFICATION**: Why this approach is best"""

    synthesis_response, _ = await call_deep_reasoner(
        prompt=synthesis_prompt,
        state=state,
        system="You are Tree of Thoughts synthesizing the best solution.",
        temperature=0.5
    )
    
    add_reasoning_step(
        state=state,
        framework="tree_of_thoughts",
        thought="Synthesized solution from best path",
        action="synthesis",
        observation=f"Final path length: {len(best_path)}",
        score=best_path[-1].score if best_path else 0.5
    )
    
    # Extract code if present
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, synthesis_response, re.DOTALL)
    
    # Store search info
    state["search_tree"] = {
        "total_nodes": len(tree.nodes),
        "max_depth_reached": max(n.depth for n in tree.nodes.values()) if tree.nodes else 0,
        "solutions_found": len([n for n in tree.nodes.values() if n.is_solution])
    }
    state["best_path"] = [n.thought for n in best_path]
    state["prm_scores"] = [n.score for n in best_path]
    
    # Update final state
    state["final_answer"] = synthesis_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = best_path[-1].score if best_path else 0.5
    
    return state


def _parse_approaches(response: str) -> list[dict]:
    """Parse initial approaches from response."""
    import re
    approaches = []
    
    pattern = r'APPROACH\s*\d+:\s*\[?([^\]\n]+)\]?\n(.*?)(?=APPROACH|\Z)'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    for name, description in matches:
        approaches.append({
            "name": name.strip(),
            "description": f"{name.strip()}: {description.strip()}"
        })
    
    if not approaches:
        # Fallback parsing
        lines = response.strip().split('\n')
        approaches = [{"name": "Default", "description": response[:300]}]
    
    return approaches


def _parse_steps(response: str) -> list[dict]:
    """Parse expansion steps from response."""
    import re
    steps = []
    
    pattern = r'STEP\s*\d+:\s*(.*?)(?=STEP|\Z)'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    for content in matches:
        is_solution = "[SOLUTION]" in content.upper()
        steps.append({
            "content": content.replace("[SOLUTION]", "").strip(),
            "is_solution": is_solution
        })
    
    if not steps:
        steps = [{"content": response[:200], "is_solution": False}]
    
    return steps


def _get_path_text(tree: ThoughtTree, node_id: str) -> str:
    """Get text representation of path to node."""
    path = []
    current_id = node_id
    
    while current_id:
        node = tree.nodes[current_id]
        path.append(f"- {node.thought[:100]}...")
        current_id = node.parent_id
    
    return "\n".join(reversed(path))
