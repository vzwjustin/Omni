"""
DocPrompting: Documentation-Driven Code Generation

Retrieves relevant documentation and examples, then generates code
that follows the patterns and conventions from the docs.
(Headless Mode: Returns Reasoning Protocol for Client Execution)

Based on: "DocPrompting: Generating Code by Retrieving the Docs"
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
async def docprompting_node(state: GraphState) -> GraphState:
    """
    DocPrompting: Documentation-Driven Code Generation

    Uses documentation examples and API references as the primary
    guide for generating correct, idiomatic code.
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    prompt = f"""# DocPrompting Protocol: Documentation-Driven Code Generation

I have selected the **DocPrompting** framework for this task.
Generate code by first retrieving and analyzing relevant documentation.

## Task
{query}

## Execution Protocol

Follow these steps to generate documentation-grounded code:

### Step 1: Identify Required APIs/Libraries
List all libraries, frameworks, or APIs needed for this task:
- Core language features
- Third-party libraries
- Internal APIs/modules

### Step 2: Retrieve Documentation
For each API/library identified:
- Find the official documentation or source
- Locate relevant function signatures
- Extract usage examples from docs
- Note any version-specific behavior

Format your findings:
```
Library: <name>
Relevant Functions:
  - function_name(params) -> return_type
    Description: <from docs>
    Example: <from docs>
```

### Step 3: Analyze Documentation Patterns
From the retrieved docs, identify:
- Recommended usage patterns
- Common idioms and conventions
- Error handling approaches
- Best practices mentioned

### Step 4: Generate Doc-Aligned Code
Write code that:
- Follows patterns from documentation examples
- Uses correct function signatures from docs
- Handles errors as docs recommend
- Includes docstrings referencing the source docs

### Step 5: Verify Against Docs
Cross-check your generated code:
- Parameter types match documentation
- Return types match documentation
- Edge cases handled as docs specify
- Deprecation warnings addressed

## Code Context
{code_context}

**Start by listing the APIs/libraries needed and their key documentation points.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="docprompting",
        thought="Generated DocPrompting protocol for documentation-driven code generation",
        action="handoff",
        observation="Prompt generated with doc-retrieval approach"
    )

    return state
