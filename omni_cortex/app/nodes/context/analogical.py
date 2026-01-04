"""
Analogical Prompting: Analogy-Based Problem Solving

Uses analogies to find novel solutions by comparing
the current problem to similar solved problems.
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


# Analogy patterns for common software engineering problems
# These are checked first before creative analogy generation
ANALOGY_PATTERNS = {
    "caching": {
        "signal": ["cache", "memoize", "store", "retrieve", "lookup", "fast"],
        "analogy": "Like a library's reference desk keeping frequently requested books nearby"
    },
    "pub_sub": {
        "signal": ["event", "subscribe", "publish", "notify", "listener", "observer"],
        "analogy": "Like a newspaper subscription - publishers don't know subscribers, but all get updates"
    },
    "rate_limiting": {
        "signal": ["throttle", "limit", "rate", "quota", "slow", "control"],
        "analogy": "Like a bouncer at a club controlling how many people enter per minute"
    },
    "circuit_breaker": {
        "signal": ["fail", "retry", "fallback", "resilient", "timeout", "circuit"],
        "analogy": "Like an electrical circuit breaker that stops current flow when overloaded"
    },
    "pooling": {
        "signal": ["pool", "reuse", "connection", "resource", "expensive", "allocate"],
        "analogy": "Like a car rental agency - cars are shared and reused rather than each person owning one"
    },
    "saga": {
        "signal": ["transaction", "compensate", "rollback", "distributed", "atomic", "saga"],
        "analogy": "Like a travel booking - if flight fails, hotel and car reservations are cancelled"
    },
    "sharding": {
        "signal": ["shard", "partition", "distribute", "scale", "split", "horizontal"],
        "analogy": "Like dividing a library into sections by author's last name initial"
    },
    "backpressure": {
        "signal": ["backpressure", "overflow", "buffer", "queue", "producer", "consumer"],
        "analogy": "Like a dam controlling water flow to prevent downstream flooding"
    }
}


@quiet_star
async def analogical_node(state: GraphState) -> GraphState:
    """
    Analogical Prompting: Analogy-Based Problem Solving.
    
    Process:
    1. ABSTRACT: Extract the essence of the problem
    2. SEARCH: Find similar problems/patterns
    3. MAP: Map the analogy to the current problem
    4. SOLVE: Apply the analogous solution
    
    Best for: Creative solutions, finding patterns, novel approaches
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )
    
    # =========================================================================
    # Phase 1: ABSTRACT - Extract Problem Essence
    # =========================================================================
    
    abstract_prompt = f"""Extract the ESSENCE of this problem in abstract terms.

PROBLEM:
{query}

CONTEXT:
{code_context}

Describe the problem abstractly:

**CORE CHALLENGE**
[What is the fundamental issue, stripped of specifics?]

**KEY CHARACTERISTICS**
- [Characteristic 1 - e.g., "needs to handle many-to-one relationship"]
- [Characteristic 2 - e.g., "requires consistency under concurrent access"]
- [Characteristic 3]

**CONSTRAINTS**
- [What limitations must be respected]

**GOAL IN ABSTRACT TERMS**
[What are we trying to achieve, generically?]

Think about this problem as a TYPE of problem, not just this specific instance."""

    abstract_response, _ = await call_fast_synthesizer(
        prompt=abstract_prompt,
        state=state,
        max_tokens=600
    )
    
    add_reasoning_step(
        state=state,
        framework="analogical",
        thought="Abstracted problem to its essence",
        action="abstraction",
        observation=abstract_response[:200]
    )
    
    # =========================================================================
    # Phase 2: SEARCH - Find Analogies
    # =========================================================================
    
    # Check for known patterns
    matched_patterns = []
    combined_text = (query + " " + (abstract_response or "")).lower()
    
    for pattern_name, pattern_info in ANALOGY_PATTERNS.items():
        signal_matches = sum(1 for s in pattern_info["signal"] if s in combined_text)
        if signal_matches >= 2:
            matched_patterns.append({
                "name": pattern_name,
                "matches": signal_matches,
                "info": pattern_info
            })
    
    # Sort by match strength
    matched_patterns.sort(key=lambda x: x["matches"], reverse=True)
    
    pattern_hints = ""
    if matched_patterns:
        pattern_hints = "\n\nPOTENTIAL PATTERN MATCHES:\n"
        for p in matched_patterns[:2]:
            pattern_hints += f"- {p['name']}: {p['info']['analogy']}\n"
    
    search_prompt = f"""Find analogies from other domains that match this problem.

ABSTRACT PROBLEM:
{abstract_response}

{pattern_hints}

Generate 2-3 ANALOGIES from different domains:

**ANALOGY 1: [Domain/Field]**
In [domain], this is like: [concrete example]
How it was solved there: [solution approach]
Key insight: [what makes this work]

**ANALOGY 2: [Domain/Field]**
In [domain], this is like: [concrete example]
How it was solved there: [solution approach]
Key insight: [what makes this work]

**ANALOGY 3: [Domain/Field]** (optional)
In [domain], this is like: [concrete example]
How it was solved there: [solution approach]
Key insight: [what makes this work]

Think broadly - consider patterns from physics, biology, economics, etc."""

    analogies_response, _ = await call_deep_reasoner(
        prompt=search_prompt,
        state=state,
        system="Find creative, insightful analogies from diverse domains.",
        temperature=0.8
    )
    
    add_reasoning_step(
        state=state,
        framework="analogical",
        thought="Found analogies from other domains",
        action="analogy_search",
        observation=f"Found patterns from {len(matched_patterns)} known patterns + creative analogies"
    )
    
    # =========================================================================
    # Phase 3: MAP - Map Analogies to Problem
    # =========================================================================
    
    map_prompt = f"""Map the most promising analogy to our specific problem.

ORIGINAL PROBLEM:
{query}

ABSTRACT FORM:
{abstract_response}

ANALOGIES FOUND:
{analogies_response}

Select the BEST analogy and create a detailed mapping:

**SELECTED ANALOGY**
[Which analogy fits best and why]

**MAPPING TABLE**
| Analogy Element | Our Problem Element |
|-----------------|---------------------|
| [Element A]     | [Our equivalent A]  |
| [Element B]     | [Our equivalent B]  |
| [Solution X]    | [Our approach X]    |

**INSIGHTS FROM MAPPING**
[What does this analogy reveal about our problem?]

**POTENTIAL SOLUTION APPROACH**
Based on this analogy, we should:
1. [Mapped step 1]
2. [Mapped step 2]
3. [Mapped step 3]

Be specific about how the analogy translates."""

    mapping_response, _ = await call_deep_reasoner(
        prompt=map_prompt,
        state=state,
        system="Create a precise mapping between analogy and problem.",
        temperature=0.6
    )
    
    add_reasoning_step(
        state=state,
        framework="analogical",
        thought="Mapped analogy to specific problem",
        action="mapping",
        observation="Created element-by-element mapping"
    )
    
    # =========================================================================
    # Phase 4: SOLVE - Apply Analogous Solution
    # =========================================================================
    
    solve_prompt = f"""Apply the analogous solution to solve the original problem.

ORIGINAL PROBLEM:
{query}

CONTEXT:
{code_context}

ANALOGY MAPPING:
{mapping_response}

Implement the solution based on the analogy:

**SOLUTION OVERVIEW**
[How the analogous approach solves our problem]

**IMPLEMENTATION**
```
[Code that implements the mapped solution]
```

**WHY THIS WORKS**
[Explain why the analogous approach is effective here]

**EDGE CASES**
[Where the analogy might break down and how to handle it]

Translate the analogy into concrete code/solution."""

    solve_response, _ = await call_deep_reasoner(
        prompt=solve_prompt,
        state=state,
        system="Implement a solution inspired by the analogy.",
        temperature=0.6,
        max_tokens=3000
    )
    
    add_reasoning_step(
        state=state,
        framework="analogical",
        thought="Applied analogous solution",
        action="solution",
        observation="Implemented analogy-inspired approach"
    )
    
    # Extract code
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(code_pattern, solve_response, re.DOTALL)
    
    # Store analogy artifacts
    state["working_memory"]["abstraction"] = abstract_response
    state["working_memory"]["analogies"] = analogies_response
    state["working_memory"]["mapping"] = mapping_response
    
    # Update final state
    state["final_answer"] = solve_response
    state["final_code"] = "\n\n".join([m.strip() for m in matches]) if matches else None
    state["confidence_score"] = 0.75  # Creative solutions have moderate confidence
    
    return state
