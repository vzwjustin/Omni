"""
Parsel: Compositional Code Generation from Natural Language Specs

Builds a dependency graph of functions from specs, then generates
each function with its dependencies satisfied.
(Headless Mode: Returns Reasoning Protocol for Client Execution)

Based on: "Parsel: A (De-)compositional Framework for Algorithmic Reasoning with Language Models"
"""

import logging
from ...state import GraphState
from ...collection_manager import get_collection_manager
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
)

logger = logging.getLogger(__name__)


def _search_code_examples(query: str, task_type: str = "code_generation") -> str:
    """Search instruction knowledge base for similar code examples."""
    try:
        manager = get_collection_manager()
        results = manager.search_instruction_knowledge(query, k=3, task_type=task_type, language="python")

        if not results:
            return ""

        examples = []
        for i, doc in enumerate(results, 1):
            examples.append(f"Code Example {i}:\n{doc.page_content[:500]}")

        return "\n\n".join(examples)
    except Exception as e:
        # Gracefully degrade if no API key or collection empty
        logger.debug("instruction_knowledge_search_skipped", error=str(e))
        return ""


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

    # Search for similar code examples
    code_examples = _search_code_examples(query, task_type="code_generation")
    if code_examples:
        logger.info("instruction_knowledge_examples_found", query_preview=query[:50])

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
{f'''
## 💡 Similar Code Examples from 12K+ Knowledge Base
{code_examples}
''' if code_examples else ''}
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
