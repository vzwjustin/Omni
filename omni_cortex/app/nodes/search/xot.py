"""
Everything-of-Thought (XoT): MCTS + Fast Thought Generation

Combines Monte Carlo Tree Search with high-speed thought
generation for efficient exploration of large solution spaces.
"""

import asyncio
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


@dataclass
class XoTNode:
    """Node in the XoT search tree."""
    id: str
    thought: str
    code_state: Optional[str]
    parent_id: Optional[str]
    visits: int = 0
    value: float = 0.0
    children: list[str] = field(default_factory=list)
    generation_time: float = 0.0
    
    @property
    def ucb(self) -> float:
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + math.sqrt(2 * math.log(self.visits + 1) / self.visits)


class XoTTree:
    """XoT tree with fast generation cache."""
    
    def __init__(self):
        self.nodes: dict[str, XoTNode] = {}
        self.counter = 0
        self.thought_cache: dict[str, list[str]] = {}  # Cache for fast thought retrieval
    
    def add_node(
        self,
        thought: str,
        code_state: Optional[str] = None,
        parent_id: Optional[str] = None
    ) -> XoTNode:
        self.counter += 1
        node_id = f"xot_{self.counter}"
        
        node = XoTNode(
            id=node_id,
            thought=thought,
            code_state=code_state,
            parent_id=parent_id
        )
        self.nodes[node_id] = node
        
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node_id)
        
        return node
    
    def select_leaf(self) -> XoTNode:
        current = list(self.nodes.values())[0]  # Root
        while current.children:
            children = [self.nodes[cid] for cid in current.children]
            current = max(children, key=lambda n: n.ucb)
        return current
    
    def backprop(self, node_id: str, value: float):
        while node_id:
            node = self.nodes[node_id]
            node.visits += 1
            node.value += value
            node_id = node.parent_id


@quiet_star
async def everything_of_thought_node(state: GraphState) -> GraphState:
    """
    Everything-of-Thought (XoT): MCTS + Fast Thought Generator.
    
    Key innovation: Uses fast model (GPT-5.2) to rapidly generate
    candidate thoughts, then deep model (Claude) for verification.
    
    Process:
    1. Fast thought generation (parallel, cached)
    2. MCTS selection and expansion
    3. Deep verification of promising paths
    4. Solution synthesis
    
    Best for: Complex refactoring, large changes, migration tasks
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    tree = XoTTree()
    
    # =========================================================================
    # Phase 1: Fast Thought Generation (Batch)
    # =========================================================================
    
    # Generate many candidate thoughts quickly
    fast_gen_prompt = f"""Generate 5 diverse initial approaches rapidly.

PROBLEM: {query}

CONTEXT:
{code_context}

Output 5 different approaches, one per line:
1. [Approach 1]
2. [Approach 2]
3. [Approach 3]
4. [Approach 4]
5. [Approach 5]

Be creative and varied. Speed over depth."""

    fast_response, _ = await call_fast_synthesizer(
        prompt=fast_gen_prompt,
        state=state,
        max_tokens=500,
        temperature=0.9
    )
    
    # Parse and create initial nodes
    initial_thoughts = _parse_fast_thoughts(fast_response)
    
    # Create root with problem statement
    root = tree.add_node(
        thought=f"Root: {query[:100]}",
        code_state=state.get("code_snippet")
    )
    
    # Add initial thoughts as children of root
    for thought in initial_thoughts[:5]:
        tree.add_node(thought=thought, parent_id=root.id)
    
    # Cache these for reuse
    tree.thought_cache[query[:50]] = initial_thoughts
    
    add_reasoning_step(
        state=state,
        framework="everything_of_thought",
        thought=f"Fast-generated {len(initial_thoughts)} initial approaches",
        action="fast_generation",
        observation="Cached thoughts for rapid expansion"
    )
    
    # =========================================================================
    # Phase 2: MCTS with Fast Expansion
    # =========================================================================
    
    max_iterations = 20
    
    for iteration in range(max_iterations):
        # SELECT best leaf using UCB
        leaf = tree.select_leaf()
        
        # EXPAND with fast thought generation
        if leaf.visits > 0:
            expand_prompt = f"""Quickly generate 3 next steps.

Current: {leaf.thought}
Goal: {query}

Three options:
1. [Next step 1]
2. [Next step 2]
3. [Next step 3]"""

            expand_response, _ = await call_fast_synthesizer(
                prompt=expand_prompt,
                state=state,
                max_tokens=300,
                temperature=0.8
            )
            
            expansions = _parse_fast_thoughts(expand_response)
            for exp in expansions[:3]:
                tree.add_node(thought=exp, parent_id=leaf.id)
            
            # Select first child for simulation
            if leaf.children:
                leaf = tree.nodes[leaf.children[0]]
        
        # SIMULATE: Quick PRM evaluation
        score = await process_reward_model(
            step=leaf.thought,
            context=code_context,
            goal=query
        )
        
        # BACKPROPAGATE
        tree.backprop(leaf.id, score)
    
    add_reasoning_step(
        state=state,
        framework="everything_of_thought",
        thought=f"Completed {max_iterations} MCTS iterations",
        action="mcts_search",
        observation=f"Tree has {len(tree.nodes)} nodes"
    )
    
    # =========================================================================
    # Phase 3: Deep Verification of Best Path
    # =========================================================================
    
    # Find best path
    best_nodes = sorted(tree.nodes.values(), key=lambda n: n.value / max(n.visits, 1), reverse=True)[:3]
    best_paths_text = "\n".join([f"- {n.thought} (score: {n.value/max(n.visits,1):.2f})" for n in best_nodes])
    
    # Deep verification with Claude
    verify_prompt = f"""Deeply verify and refine the best approaches found.

PROBLEM: {query}

CONTEXT:
{code_context}

BEST CANDIDATES FROM SEARCH:
{best_paths_text}

For each candidate:
1. Analyze its CORRECTNESS
2. Identify potential ISSUES
3. Suggest IMPROVEMENTS

Then provide the BEST OVERALL SOLUTION with:
- Complete implementation plan
- Code (if applicable)
- Verification steps"""

    verify_response, _ = await call_deep_reasoner(
        prompt=verify_prompt,
        state=state,
        system="You are XoT performing deep verification of search results.",
        temperature=0.4,
        max_tokens=4000
    )
    
    add_reasoning_step(
        state=state,
        framework="everything_of_thought",
        thought="Deep verification of top candidates",
        action="deep_verification",
        observation="Used Claude for thorough analysis"
    )
    
    # =========================================================================
    # Phase 4: Solution Synthesis
    # =========================================================================
    
    synthesis_prompt = f"""Synthesize the final solution from XoT exploration.

PROBLEM: {query}

VERIFICATION ANALYSIS:
{verify_response}

CONTEXT:
{code_context}

Provide:
1. **FINAL SOLUTION**: The recommended approach
2. **IMPLEMENTATION**: Complete code or detailed steps
3. **CONFIDENCE**: Your confidence level and reasoning
4. **ALTERNATIVES**: Brief mention of viable alternatives if primary fails"""

    synthesis_response, _ = await call_deep_reasoner(
        prompt=synthesis_prompt,
        state=state,
        system="You are XoT synthesizing the final solution.",
        temperature=0.5
    )
    
    # Extract code
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, synthesis_response, re.DOTALL)
    
    # Calculate final score
    final_score = best_nodes[0].value / max(best_nodes[0].visits, 1) if best_nodes else 0.5
    
    add_reasoning_step(
        state=state,
        framework="everything_of_thought",
        thought="Synthesized final solution from XoT",
        action="synthesis",
        observation=f"Final confidence: {final_score:.2f}",
        score=final_score
    )
    
    # Store search info
    state["search_tree"] = {
        "total_nodes": len(tree.nodes),
        "iterations": max_iterations,
        "cached_thoughts": len(tree.thought_cache)
    }
    
    # Update final state
    state["final_answer"] = synthesis_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = min(0.95, final_score + 0.1)  # Boost for verification
    
    return state


def _parse_fast_thoughts(response: str) -> list[str]:
    """Parse numbered thoughts from fast response."""
    import re
    
    # Match numbered items
    pattern = r'\d+\.\s*\[?([^\]\n]+)\]?'
    matches = re.findall(pattern, response)
    
    if matches:
        return [m.strip() for m in matches if m.strip()]
    
    # Fallback: split by newlines
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[:5] if lines else ["Analyze the problem", "Propose solution"]
