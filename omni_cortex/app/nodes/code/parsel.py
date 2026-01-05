"""
Parsel: Compositional Code Generation from Natural Language Specs

Builds a dependency graph of functions from specs, then generates
each function with its dependencies satisfied.
(Headless Mode: Returns Reasoning Protocol for Client Execution)

Based on: "Parsel: A (De-)compositional Framework for Algorithmic Reasoning with Language Models"
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
async def parsel_node(state: GraphState) -> GraphState:
    """
    Parsel: Compositional Code Generation from Natural Language Specs

    Decomposes complex coding tasks into a dependency graph of functions,
    each with natural language specs, then generates code bottom-up.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# Parsel Protocol: Compositional Code Generation

I have selected the **Parsel** framework for this task.
Compositional code generation by decomposing specs into a dependency graph.

## Task
{query}

## Execution Protocol

Follow these steps to generate compositional, well-structured code:

### Step 1: Decompose into Function Specs
Break down the task into individual functions. For each function, write:
- **Name**: A clear, descriptive function name
- **Spec**: Natural language description of what it does
- **Inputs**: What parameters it takes
- **Outputs**: What it returns
- **Dependencies**: Which other functions it calls

Example format:
```
fn: calculate_total
spec: "Calculate the total price including tax and discounts"
inputs: items (list), tax_rate (float), discount (float)
outputs: total (float)
deps: [apply_discount, calculate_tax]
```

### Step 2: Build Dependency Graph
Arrange functions by dependency order (leaves first):
- Identify functions with no dependencies (base functions)
- Order remaining functions so dependencies come before dependents
- Verify no circular dependencies exist

### Step 3: Generate Base Functions
For each function with no dependencies:
- Generate implementation based on its spec
- Include docstring matching the spec
- Add type hints

### Step 4: Generate Dependent Functions
For each function in dependency order:
- Reference already-generated dependencies
- Compose using the dependency implementations
- Ensure interfaces match

### Step 5: Integrate and Test
- Combine all functions into cohesive module
- Add any necessary imports
- Create integration point (main function or class)

## Code Context
{code_context}

**Start by listing all function specs with their dependencies, then build the dependency graph.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="parsel",
        thought="Generated Parsel protocol for compositional code generation",
        action="handoff",
        observation="Prompt generated with dependency graph approach"
    )

    return state
