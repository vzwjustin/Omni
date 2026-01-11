"""
Tree of Thoughts Framework: Real Implementation

Implements genuine multi-path exploration with backtracking:
1. Generate multiple initial thoughts/approaches
2. Evaluate each thought using Process Reward Model
3. Expand the most promising paths
4. Prune low-scoring branches
5. Select the best complete solution

This is a framework with actual tree search, not a prompt template.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import structlog

from ...state import GraphState
from ..common import (
    add_reasoning_step,
    call_deep_reasoner,
    call_fast_synthesizer,
    prepare_context_with_gemini,
    process_reward_model,
    quiet_star,
)

logger = structlog.get_logger("tree_of_thoughts")

MAX_DEPTH = 4
BRANCHING_FACTOR = 3
BEAM_WIDTH = 2  # Keep top N paths at each level


@dataclass
class ThoughtNode:
    """A node in the thought tree."""

    thought: str
    score: float = 0.0
    depth: int = 0
    parent: Optional["ThoughtNode"] = None
    children: list["ThoughtNode"] = field(default_factory=list)
    is_complete: bool = False

    def path(self) -> list[str]:
        """Get the path from root to this node."""
        if self.parent is None:
            return [self.thought]
        return self.parent.path() + [self.thought]

    def path_score(self) -> float:
        """Get average score along path."""
        path_nodes = []
        node = self
        while node is not None:
            path_nodes.append(node.score)
            node = node.parent
        return sum(path_nodes) / len(path_nodes) if path_nodes else 0.0


async def _generate_initial_thoughts(
    query: str, code_context: str, state: GraphState
) -> list[ThoughtNode]:
    """Generate initial divergent approaches to the problem."""

    prompt = f"""You are solving a problem using Tree of Thoughts. Generate {BRANCHING_FACTOR} DIFFERENT initial approaches.

PROBLEM:
{query}

CONTEXT:
{code_context}

Generate {BRANCHING_FACTOR} distinctly different approaches. Each should be a viable starting point.

Respond in this EXACT format:
APPROACH_1: [First approach - describe the strategy and first concrete step]
APPROACH_2: [Second approach - must be meaningfully different from #1]
APPROACH_3: [Third approach - must be different from both #1 and #2]
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=1024)

    thoughts = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("APPROACH_"):
            content = line.split(":", 1)[-1].strip()
            if content:
                thoughts.append(ThoughtNode(thought=content, depth=0))

    # Fallback if parsing fails
    if len(thoughts) < BRANCHING_FACTOR:
        thoughts.append(ThoughtNode(thought=f"Direct approach: {response[:200]}", depth=0))

    return thoughts[:BRANCHING_FACTOR]


async def _expand_thought(
    node: ThoughtNode, query: str, code_context: str, state: GraphState
) -> list[ThoughtNode]:
    """Expand a thought node into child nodes."""

    path_so_far = "\n".join([f"Step {i + 1}: {t}" for i, t in enumerate(node.path())])

    prompt = f"""Continue this reasoning path with {BRANCHING_FACTOR} different next steps.

PROBLEM:
{query}

REASONING SO FAR:
{path_so_far}

CONTEXT:
{code_context}

Generate {BRANCHING_FACTOR} different ways to continue. Each should be a logical next step.

Respond in this EXACT format:
NEXT_1: [First possible next step]
NEXT_2: [Second possible next step - different approach]
NEXT_3: [Third possible next step - different from both]
IS_COMPLETE: [yes/no - is this path ready for final solution?]
"""

    response, _ = await call_fast_synthesizer(prompt, state, max_tokens=768)

    children = []
    is_complete = False

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("NEXT_"):
            content = line.split(":", 1)[-1].strip()
            if content:
                child = ThoughtNode(thought=content, depth=node.depth + 1, parent=node)
                children.append(child)
        elif line.startswith("IS_COMPLETE:"):
            is_complete = "yes" in line.lower()

    # Mark nodes as complete if at max depth or explicitly complete
    for child in children:
        if child.depth >= MAX_DEPTH - 1 or is_complete:
            child.is_complete = True

    node.children = children
    return children


async def _score_thought(
    node: ThoughtNode, query: str, code_context: str, state: GraphState
) -> float:
    """Score a thought using the Process Reward Model."""

    "\n".join(node.path())

    score = await process_reward_model(
        step=node.thought,
        context=f"{query}\n\n{code_context}",
        goal=query,
        previous_steps=node.path()[:-1] if len(node.path()) > 1 else None,
    )

    node.score = score
    return score


async def _generate_solution(
    best_path: list[str], query: str, code_context: str, state: GraphState
) -> str:
    """Generate final solution from the best reasoning path."""

    path_text = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(best_path)])

    prompt = f"""Based on this reasoning path, provide the complete solution.

PROBLEM:
{query}

REASONING PATH (best path from tree search):
{path_text}

CONTEXT:
{code_context}

Provide:
1. SOLUTION: The complete solution based on the reasoning path
2. IMPLEMENTATION: Specific code or steps to implement
3. RATIONALE: Why this path was best
"""

    response, _ = await call_deep_reasoner(prompt, state, max_tokens=2048)
    return response


@quiet_star
async def tree_of_thoughts_node(state: GraphState) -> GraphState:  # noqa: PLR0915
    """
    Tree of Thoughts Framework - REAL IMPLEMENTATION

    Executes genuine tree search with:
    - Multiple initial approaches (branching)
    - Evaluation of each branch (scoring)
    - Expansion of promising paths (exploration)
    - Pruning of poor paths (beam search)
    - Selection of best solution (exploitation)
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    logger.info("tree_of_thoughts_start", query_preview=query[:50])

    # Step 1: Generate initial thoughts
    frontier = await _generate_initial_thoughts(query, code_context, state)

    add_reasoning_step(
        state=state,
        framework="tree_of_thoughts",
        thought=f"Generated {len(frontier)} initial approaches",
        action="branch",
        observation=", ".join([n.thought[:50] + "..." for n in frontier]),
    )

    # Step 2: Score initial thoughts
    score_tasks = [_score_thought(n, query, code_context, state) for n in frontier]
    await asyncio.gather(*score_tasks)

    all_nodes = list(frontier)
    completed_paths: list[ThoughtNode] = []

    # Step 3: Tree search with beam pruning
    for depth in range(MAX_DEPTH):
        if not frontier:
            break

        # Sort by score and keep top BEAM_WIDTH
        frontier.sort(key=lambda n: n.score, reverse=True)
        frontier = frontier[:BEAM_WIDTH]

        add_reasoning_step(
            state=state,
            framework="tree_of_thoughts",
            thought=f"Depth {depth + 1}: Exploring {len(frontier)} paths",
            action="prune",
            observation=f"Best score: {frontier[0].score:.2f}" if frontier else "No paths",
        )

        # Separate complete and incomplete nodes
        complete = [n for n in frontier if n.is_complete]
        incomplete = [n for n in frontier if not n.is_complete]

        completed_paths.extend(complete)

        if not incomplete:
            break

        # Expand incomplete nodes
        new_frontier = []
        for node in incomplete:
            children = await _expand_thought(node, query, code_context, state)

            # Score children
            for child in children:
                await _score_thought(child, query, code_context, state)

            all_nodes.extend(children)
            new_frontier.extend(children)

        frontier = new_frontier

        logger.info(
            "tree_of_thoughts_depth",
            depth=depth + 1,
            nodes_explored=len(all_nodes),
            frontier_size=len(frontier),
        )

    # Add remaining frontier to completed
    completed_paths.extend(frontier)

    # Step 4: Select best path
    if completed_paths:
        completed_paths.sort(key=lambda n: n.path_score(), reverse=True)
        best_node = completed_paths[0]
        best_path = best_node.path()
    else:
        # Fallback to highest-scoring node
        all_nodes.sort(key=lambda n: n.score, reverse=True)
        best_node = all_nodes[0] if all_nodes else None
        best_path = best_node.path() if best_node else ["No valid path found"]

    add_reasoning_step(
        state=state,
        framework="tree_of_thoughts",
        thought=f"Selected best path with {len(best_path)} steps",
        action="select",
        score=best_node.path_score() if best_node else 0.0,
    )

    # Step 5: Generate solution from best path
    solution = await _generate_solution(best_path, query, code_context, state)

    # Format tree visualization
    def format_tree(nodes: list[ThoughtNode], indent: int = 0) -> str:
        result = ""
        for node in nodes:
            prefix = "  " * indent + ("├─ " if indent > 0 else "")
            score_str = f"[{node.score:.2f}]"
            result += f"{prefix}{score_str} {node.thought[:60]}...\n"
            if node.children:
                result += format_tree(node.children, indent + 1)
        return result

    root_nodes = [n for n in all_nodes if n.parent is None]
    tree_viz = format_tree(root_nodes)

    final_answer = f"""# Tree of Thoughts Analysis

## Exploration Tree
```
{tree_viz}
```

## Best Path (Score: {best_node.path_score():.2f if best_node else 0:.2f})
{chr(10).join(f"{i + 1}. {step}" for i, step in enumerate(best_path))}

## Solution
{solution}

## Statistics
- Nodes explored: {len(all_nodes)}
- Max depth reached: {max(n.depth for n in all_nodes) + 1 if all_nodes else 0}
- Paths completed: {len(completed_paths)}
"""

    state["final_answer"] = final_answer
    state["confidence_score"] = best_node.path_score() if best_node else 0.5

    logger.info(
        "tree_of_thoughts_complete",
        nodes_explored=len(all_nodes),
        best_score=best_node.path_score() if best_node else 0.0,
    )

    return state
