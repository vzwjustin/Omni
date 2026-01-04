"""
CRITIC: External Tool Verification

Uses vector store documentation search and real API validation
to verify code correctness and API usage.
"""

from typing import Optional
from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context
)
from ...langchain_integration import search_vectorstore
from ..langchain_tools import call_langchain_tool


@quiet_star
async def critic_node(state: GraphState) -> GraphState:
    """
    CRITIC: External Tool Verification.
    
    Process:
    1. EXTRACT: Identify APIs/libraries used in code
    2. LOOKUP: Query documentation via vector store
    3. COMPARE: Check if usage matches documentation
    4. CRITIQUE: Identify mismatches and issues
    5. CORRECT: Provide fixed version
    
    Best for: API usage verification, library integration, external services
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    # =========================================================================
    # Phase 1: EXTRACT APIs/Libraries Used
    # =========================================================================
    
    extract_prompt = f"""Analyze this code/task and identify all external APIs and libraries used.

TASK: {query}

CODE:
{code_context}

List ALL:
1. **LIBRARIES/PACKAGES**: External libraries imported
2. **API CALLS**: HTTP/REST/GraphQL calls made
3. **FUNCTIONS USED**: Key functions from each library
4. **DATA FORMATS**: JSON, XML, etc. being processed

Format:
LIBRARY: [name]
FUNCTIONS: [function1, function2, ...]
USAGE: [how it's being used]"""

    extract_response, _ = await call_fast_synthesizer(
        prompt=extract_prompt,
        state=state,
        max_tokens=800
    )
    
    add_reasoning_step(
        state=state,
        framework="critic",
        thought="Extracted APIs and libraries from code",
        action="extraction",
        observation=extract_response[:200]
    )
    
    # =========================================================================
    # Phase 2: LOOKUP Documentation (Enhanced)
    # =========================================================================
    
    # Use enhanced search tools for precise function/class lookup
    docs_found = []
    
    # Parse extract_response for function and class names
    import re
    functions = re.findall(r'Function:\s*([\w.]+)', extract_response)
    classes = re.findall(r'Class:\s*([\w.]+)', extract_response)
    libraries = re.findall(r'Library:\s*([\w.]+)', extract_response)
    
    # Search for specific functions
    for func in functions[:3]:  # Limit to top 3
        try:
            result = await call_langchain_tool("search_function_implementation", func, state)
            if result and "No function" not in result:
                docs_found.append(f"## Function: {func}\n{result[:1000]}")
        except Exception:
            pass
    
    # Search for specific classes
    for cls in classes[:3]:  # Limit to top 3
        try:
            result = await call_langchain_tool("search_class_implementation", cls, state)
            if result and "No class" not in result:
                docs_found.append(f"## Class: {cls}\n{result[:1000]}")
        except Exception:
            pass
    
    # Fallback to documentation search for libraries
    if not docs_found and libraries:
        for lib in libraries[:2]:
            try:
                result = await call_langchain_tool("search_documentation_only", f"{lib} API usage", state)
                if result:
                    docs_found.append(f"## Library: {lib}\n{result[:800]}")
            except Exception:
                pass
    
    # Final fallback to legacy vector search
    if not docs_found:
        docs = search_vectorstore(query + "\n" + code_context, k=3)
        if docs:
            for d in docs:
                meta = d.metadata or {}
                path = meta.get("path", "unknown")
                snippet = d.page_content[:800]
                docs_found.append(f"### {path}\n{snippet}")
    
    lookup_results = "## Documentation Lookup Results (Enhanced Search)\n\n"
    if docs_found:
        lookup_results += "\n\n".join(docs_found)
    else:
        lookup_results += "*No specific documentation found. Using general knowledge.*\n"
    
    add_reasoning_step(
        state=state,
        framework="critic",
        thought=f"Looked up documentation for {len(docs_found)} libraries",
        action="doc_lookup",
        observation=f"Found docs for: {', '.join(docs_found) if docs_found else 'None'}"
    )
    
    # =========================================================================
    # Phase 3: COMPARE Usage with Documentation
    # =========================================================================
    
    compare_prompt = f"""Compare the code's API usage against documentation.

ORIGINAL CODE/TASK:
{query}

CODE:
{code_context}

EXTRACTED USAGE:
{extract_response}

DOCUMENTATION:
{lookup_results}

For each API/function used, verify:
1. Is the function signature correct?
2. Are required parameters provided?
3. Are optional parameters used correctly?
4. Is the return value handled properly?
5. Is error handling appropriate?

**VERIFICATION RESULTS**:

For each API:
```
API: [name]
CORRECT: [YES/NO/PARTIAL]
ISSUE: [description if any]
DOCUMENTATION SAYS: [what docs say]
CODE DOES: [what code does]
```"""

    compare_response, _ = await call_deep_reasoner(
        prompt=compare_prompt,
        state=state,
        system="Compare API usage against documentation precisely.",
        temperature=0.4
    )
    
    add_reasoning_step(
        state=state,
        framework="critic",
        thought="Compared API usage against documentation",
        action="comparison",
        observation="Identified any mismatches"
    )
    
    # =========================================================================
    # Phase 4: CRITIQUE and CORRECT
    # =========================================================================
    
    critique_prompt = f"""Provide final critique and corrected code.

ORIGINAL TASK: {query}

ORIGINAL CODE:
{code_context}

COMPARISON RESULTS:
{compare_response}

Provide:

**CRITIQUE SUMMARY**
[List of issues found]

**BEST PRACTICES RECOMMENDATIONS**
[Suggestions for improvement]

**CORRECTED CODE**
```
[Fixed code with proper API usage]
```

**EXPLANATION OF FIXES**
[What was changed and why]"""

    critique_response, _ = await call_deep_reasoner(
        prompt=critique_prompt,
        state=state,
        system="Critique the code and provide corrections.",
        temperature=0.5,
        max_tokens=3000
    )
    
    add_reasoning_step(
        state=state,
        framework="critic",
        thought="Generated critique and corrections",
        action="critique",
        observation="Provided corrected code with explanations"
    )
    
    # Extract corrected code
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, critique_response, re.DOTALL)
    
    # Determine confidence based on issues found
    has_issues = "NO" in compare_response.upper() or "ISSUE" in compare_response.upper()
    
    # Update final state
    state["final_answer"] = critique_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.7 if has_issues else 0.9
    
    return state


def _detect_language(code: str, query: str) -> str:
    """Detect programming language from code and query."""
    combined = (code + " " + query).lower()
    
    indicators = {
        "python": ["def ", "import ", ".py", "python", "pandas", "numpy", "django", "flask"],
        "javascript": ["function ", "const ", "let ", "var ", "=>", "javascript", "js", "node", "react"],
        "typescript": ["typescript", "interface ", ": string", ": number", ".ts"],
        "java": ["public class", "public static void", ".java", "java"],
        "go": ["func ", "package main", "golang", ".go"],
        "rust": ["fn ", "let mut", "rust", ".rs"],
    }
    
    scores = {}
    for lang, keywords in indicators.items():
        scores[lang] = sum(1 for kw in keywords if kw in combined)
    
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    
    return "python"  # Default


async def lookup_api_documentation(library: str, function: str) -> str:
    """
    Lookup API documentation via vector store. Returns best matching snippets.
    """
    query = f"{library} {function} API usage"
    docs = search_vectorstore(query, k=3)
    if not docs:
        return f"No documentation found for {library}.{function}"
    formatted = []
    for d in docs:
        meta = d.metadata or {}
        path = meta.get("path", "unknown")
        formatted.append(f"{path}: {d.page_content[:500]}")
    return "\n\n".join(formatted)
