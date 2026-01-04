"""
Graph of Thoughts (GoT): Non-Linear Thought Processing

Implements graph-based reasoning with merge and aggregate
operations for refactoring and complex code restructuring.
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
    add_reasoning_step,
    format_code_context
)


@dataclass
class GraphNode:
    """A node in the thought graph."""
    id: str
    thought: str
    node_type: str  # "initial", "branch", "merge", "aggregate", "solution"
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class ThoughtGraph:
    """Graph structure for non-linear thought exploration."""
    
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.node_counter = 0
    
    def add_node(
        self,
        thought: str,
        node_type: str,
        parents: Optional[list[str]] = None,
        score: float = 0.0,
        metadata: Optional[dict] = None
    ) -> GraphNode:
        """Add a node to the graph."""
        self.node_counter += 1
        node_id = f"g_{self.node_counter}"
        
        node = GraphNode(
            id=node_id,
            thought=thought,
            node_type=node_type,
            parents=parents or [],
            score=score,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        
        # Update parent references
        for parent_id in node.parents:
            if parent_id in self.nodes:
                self.nodes[parent_id].children.append(node_id)
        
        return node
    
    def merge_nodes(self, node_ids: list[str], merge_thought: str) -> GraphNode:
        """Create a merge node from multiple parent nodes."""
        return self.add_node(
            thought=merge_thought,
            node_type="merge",
            parents=node_ids,
            metadata={"merged_from": node_ids}
        )
    
    def aggregate_nodes(self, node_ids: list[str], aggregate_thought: str) -> GraphNode:
        """Create an aggregate node combining insights."""
        return self.add_node(
            thought=aggregate_thought,
            node_type="aggregate",
            parents=node_ids,
            metadata={"aggregated_from": node_ids}
        )
    
    def get_leaves(self) -> list[GraphNode]:
        """Get all leaf nodes (no children)."""
        return [n for n in self.nodes.values() if not n.children]
    
    def get_by_type(self, node_type: str) -> list[GraphNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]
    
    def get_best_solution(self) -> Optional[GraphNode]:
        """Get the highest-scoring solution node."""
        solutions = self.get_by_type("solution")
        if not solutions:
            solutions = self.get_leaves()
        
        return max(solutions, key=lambda n: n.score) if solutions else None


@quiet_star
async def graph_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Graph of Thoughts: Non-Linear Thinking with Merge/Aggregate.
    
    Process:
    1. Generate parallel initial thoughts
    2. Branch each thought into sub-explorations
    3. Merge compatible thoughts
    4. Aggregate insights into comprehensive solution
    
    Best for: Refactoring, code restructuring, complex dependencies
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    graph = ThoughtGraph()
    
    # =========================================================================
    # Phase 1: Generate Initial Parallel Thoughts
    # =========================================================================
    
    initial_prompt = f"""You are using Graph of Thoughts for non-linear problem solving.

PROBLEM: {query}

CONTEXT:
{code_context}

Generate 4 INDEPENDENT initial thoughts about this problem.
Each thought should analyze a DIFFERENT ASPECT:

THOUGHT 1 - STRUCTURAL: What are the structural issues/opportunities?
THOUGHT 2 - BEHAVIORAL: What behavior needs to change or be added?
THOUGHT 3 - DEPENDENCIES: What are the dependency relationships?
THOUGHT 4 - PATTERNS: What patterns or anti-patterns are present?

Format:
THOUGHT 1 [STRUCTURAL]: [Analysis]
THOUGHT 2 [BEHAVIORAL]: [Analysis]
THOUGHT 3 [DEPENDENCIES]: [Analysis]
THOUGHT 4 [PATTERNS]: [Analysis]"""

    initial_response, _ = await call_fast_synthesizer(
        prompt=initial_prompt,
        state=state,
        max_tokens=1500,
        temperature=0.7
    )
    
    initial_thoughts = _parse_graph_thoughts(initial_response)
    initial_nodes = []
    
    for thought in initial_thoughts:
        node = graph.add_node(
            thought=thought["content"],
            node_type="initial",
            metadata={"aspect": thought["aspect"]}
        )
        initial_nodes.append(node)
    
    add_reasoning_step(
        state=state,
        framework="graph_of_thoughts",
        thought=f"Generated {len(initial_nodes)} parallel initial thoughts",
        action="parallel_generation",
        observation=", ".join([n.metadata.get("aspect", "unknown") for n in initial_nodes])
    )
    
    # =========================================================================
    # Phase 2: Branch Each Thought
    # =========================================================================
    
    for initial_node in initial_nodes:
        branch_prompt = f"""Develop this thought further with 2 branches.

ORIGINAL THOUGHT ({initial_node.metadata.get('aspect', 'unknown')}):
{initial_node.thought}

PROBLEM CONTEXT:
{query}

Generate 2 different directions to develop this thought:

BRANCH A: [More conservative/incremental approach]
BRANCH B: [More aggressive/transformative approach]"""

        branch_response, _ = await call_fast_synthesizer(
            prompt=branch_prompt,
            state=state,
            max_tokens=800,
            temperature=0.7
        )
        
        branches = _parse_branches(branch_response)
        for branch in branches[:2]:
            graph.add_node(
                thought=branch["content"],
                node_type="branch",
                parents=[initial_node.id],
                metadata={"approach": branch["approach"]}
            )
    
    add_reasoning_step(
        state=state,
        framework="graph_of_thoughts",
        thought="Branched initial thoughts into sub-explorations",
        action="branching",
        observation=f"Graph now has {len(graph.nodes)} nodes"
    )
    
    # =========================================================================
    # Phase 3: Merge Compatible Thoughts
    # =========================================================================
    
    # Find pairs of branches that might merge well
    leaves = graph.get_leaves()
    
    # Group leaves by their root aspect
    aspect_groups: dict[str, list[GraphNode]] = {}
    for leaf in leaves:
        # Find the root node
        parent_id = leaf.parents[0] if leaf.parents else None
        if parent_id and parent_id in graph.nodes:
            parent = graph.nodes[parent_id]
            aspect = parent.metadata.get("aspect", "unknown")
            if aspect not in aspect_groups:
                aspect_groups[aspect] = []
            aspect_groups[aspect].append(leaf)
    
    # Merge structural + dependencies (complementary)
    merge_candidates = []
    structural_leaves = aspect_groups.get("STRUCTURAL", [])
    deps_leaves = aspect_groups.get("DEPENDENCIES", [])
    
    if structural_leaves and deps_leaves:
        merge_candidates.append((structural_leaves[0], deps_leaves[0]))
    
    behavioral_leaves = aspect_groups.get("BEHAVIORAL", [])
    patterns_leaves = aspect_groups.get("PATTERNS", [])
    
    if behavioral_leaves and patterns_leaves:
        merge_candidates.append((behavioral_leaves[0], patterns_leaves[0]))
    
    merge_nodes = []
    for node_a, node_b in merge_candidates:
        merge_prompt = f"""Merge these two complementary thoughts into a unified insight.

THOUGHT A:
{node_a.thought}

THOUGHT B:
{node_b.thought}

PROBLEM:
{query}

Create a SYNTHESIZED thought that:
1. Combines key insights from both
2. Resolves any conflicts
3. Identifies synergies

MERGED INSIGHT: [Your synthesis]"""

        merge_response, _ = await call_fast_synthesizer(
            prompt=merge_prompt,
            state=state,
            max_tokens=600,
            temperature=0.6
        )
        
        merged = graph.merge_nodes(
            [node_a.id, node_b.id],
            merge_response.split("MERGED INSIGHT:")[-1].strip() if "MERGED INSIGHT:" in merge_response else merge_response
        )
        merge_nodes.append(merged)
    
    add_reasoning_step(
        state=state,
        framework="graph_of_thoughts",
        thought=f"Created {len(merge_nodes)} merge nodes",
        action="merging",
        observation="Combined complementary aspects"
    )
    
    # =========================================================================
    # Phase 4: Aggregate into Final Solution
    # =========================================================================
    
    # Get all current leaves (including merges)
    final_leaves = graph.get_leaves()
    leaf_thoughts = [n.thought for n in final_leaves]
    
    aggregate_prompt = f"""Create the final aggregated solution from Graph of Thoughts.

PROBLEM: {query}

CONTEXT:
{code_context}

EXPLORED THOUGHTS:
{"\n".join(f"- {t[:200]}..." for t in leaf_thoughts)}

Create a comprehensive FINAL SOLUTION that:
1. Aggregates all valuable insights
2. Presents a cohesive implementation plan
3. Addresses all aspects (structural, behavioral, dependencies, patterns)

Provide:
**AGGREGATED SOLUTION**:
[Comprehensive solution description]

**IMPLEMENTATION PLAN**:
[Ordered list of specific changes]

**CODE** (if applicable):
[Implementation code]

**RATIONALE**:
[Why this aggregation is optimal]"""

    aggregate_response, _ = await call_deep_reasoner(
        prompt=aggregate_prompt,
        state=state,
        system="You are Graph of Thoughts creating the final aggregated solution.",
        temperature=0.5,
        max_tokens=4000
    )
    
    # Score the final solution
    solution_score = await process_reward_model(
        step=aggregate_response[:500],
        context=code_context,
        goal=query
    )
    
    solution_node = graph.add_node(
        thought=aggregate_response,
        node_type="solution",
        parents=[n.id for n in final_leaves],
        score=solution_score
    )
    
    add_reasoning_step(
        state=state,
        framework="graph_of_thoughts",
        thought="Aggregated all thoughts into final solution",
        action="aggregation",
        observation=f"Solution score: {solution_score:.2f}",
        score=solution_score
    )
    
    # Extract code if present
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, aggregate_response, re.DOTALL)
    
    # Store graph info
    state["search_tree"] = {
        "total_nodes": len(graph.nodes),
        "initial_thoughts": len(initial_nodes),
        "merge_nodes": len(merge_nodes),
        "node_types": {t: len(graph.get_by_type(t)) for t in ["initial", "branch", "merge", "aggregate", "solution"]}
    }
    
    # Update final state
    state["final_answer"] = aggregate_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = solution_score
    
    return state


def _parse_graph_thoughts(response: str) -> list[dict]:
    """Parse initial thoughts from response."""
    import re
    thoughts = []
    
    pattern = r'THOUGHT\s*\d+\s*\[([^\]]+)\]:\s*(.*?)(?=THOUGHT\s*\d+|\Z)'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    for aspect, content in matches:
        thoughts.append({
            "aspect": aspect.strip().upper(),
            "content": content.strip()
        })
    
    if not thoughts:
        thoughts = [{"aspect": "GENERAL", "content": response[:300]}]
    
    return thoughts


def _parse_branches(response: str) -> list[dict]:
    """Parse branches from response."""
    import re
    branches = []
    
    pattern = r'BRANCH\s*([AB]):\s*(.*?)(?=BRANCH|\Z)'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    for approach, content in matches:
        branches.append({
            "approach": "conservative" if approach == "A" else "transformative",
            "content": content.strip()
        })
    
    if not branches:
        branches = [{"approach": "default", "content": response[:200]}]
    
    return branches
