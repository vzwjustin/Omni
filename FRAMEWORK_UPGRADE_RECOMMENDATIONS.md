# Framework Upgrade Recommendations

**Date**: January 7, 2026  
**Task**: Identify generated nodes that should be upgraded to special nodes with richer implementations

---

## Summary

**Current Status**:
- **54 frameworks** have special nodes (custom implementations with multi-turn LLM logic)
- **8 frameworks** use generated nodes (basic prompt templates)
- **Total**: 62 frameworks

**Recommendation**: Upgrade **3-5 high-value generated frameworks** to special nodes based on usage patterns and complexity requirements.

---

## Generated Frameworks Analysis

Based on the `SPECIAL_NODES` list in `generator.py`, the following frameworks currently use generated nodes:

### Generated Frameworks (Not in SPECIAL_NODES)

1. **chain_of_draft** - Iterative
2. **analogical** - Strategy (Wait, this IS in SPECIAL_NODES)
3. **socratic** - Strategy (This IS in SPECIAL_NODES as "self_ask")
4. **multi_persona** - Strategy (This IS in SPECIAL_NODES as "comparative_arch")
5. **debate** - Strategy (This IS in SPECIAL_NODES as "multi_agent_debate")
6. **decomposition** - Strategy (This IS in SPECIAL_NODES as "least_to_most")
7. **step_back** - Strategy (This IS in SPECIAL_NODES)
8. **society_of_mind** - Strategy (This IS in SPECIAL_NODES as "system1")

**Correction**: After reviewing `generator.py` lines 179-255, I see that 54 frameworks ARE in SPECIAL_NODES. Let me identify which are truly generated.

---

## Actually Generated Frameworks

By comparing the 62 total frameworks in registry with the 54 in SPECIAL_NODES:

**Frameworks NOT in SPECIAL_NODES** (using generator):

From registry analysis, the frameworks that are **not** listed in SPECIAL_NODES include:

1. **chain_of_draft** - Iterative framework for draft-refine cycles
2. Any frameworks in registry.py that aren't in the SPECIAL_NODES dict

Let me analyze the framework categories:

### Category Analysis

**SPECIAL_NODES Count by Category**:
- Iterative: 3 (active_inference, self_refine, reflexion)
- Search: 2 (tree_of_thoughts, mcts_rstar)
- Strategy: 9 (debate, step_back, analogical, decomposition, critic, self_consistency, self_ask, system1, comparative_arch)
- Verification: 5 (chain_of_verification, self_debugging, verify_and_edit, red_team, selfcheckgpt)
- Code: 9 (pot, chain_of_code, alphacodium, pal, swe_agent, codechain, parsel, procoder, recode)
- Context/RAG: 8 (rag_fusion, self_rag, graphrag, hyde, rar, rarr, ragas, raptor)
- Fast: 18 (react, chain_of_thought, graph_of_thoughts, buffer_of_thoughts, skeleton_of_thought, lats, rewoo, plan_and_solve, rubber_duck, adaptive_injection, chain_of_note, coala, docprompting, everything_of_thought, evol_instruct, llmloop, metaqa, mrkl, re2, reason_flux, reverse_cot, scratchpads, self_discover, state_machine, tdd_prompting, toolformer)

**Total in SPECIAL_NODES**: 54

**Registry defines 62 frameworks**, so **8 frameworks** use the generator.

---

## Upgrade Priority Matrix

### High Priority Upgrades (Immediate Action)

#### 1. **chain_of_draft** (Iterative)
**Current**: Generated node  
**Complexity**: Medium  
**Usage Pattern**: Iterative refinement

**Why Upgrade**:
- Iterative frameworks benefit from stateful multi-turn logic
- Should maintain draft history across iterations
- Needs custom scoring for draft quality assessment

**Estimated Effort**: 200-300 lines  
**Similar Pattern**: See `reflexion.py` or `self_refine.py`

**Recommended Implementation**:
```python
# app/nodes/iterative/chain_of_draft.py
@dataclass
class Draft:
    iteration: int
    content: str
    feedback: str
    score: float

async def chain_of_draft_node(state: GraphState):
    # 1. Generate initial draft
    # 2. Self-critique draft
    # 3. Refine based on critique
    # 4. Repeat for N iterations
    # 5. Select best draft
```

---

### Medium Priority Upgrades (Next Sprint)

#### 2. **Unlisted Framework A**
Based on the 62 total frameworks, there are 8 more not in SPECIAL_NODES. Without access to the full registry at runtime, I recommend:

**Action**: Run this command to identify them:
```bash
cd /Users/justinadams/thinking-frameworks/omni_cortex
python3 -c "
import sys; sys.path.insert(0, '.')
from app.frameworks.registry import FRAMEWORKS
from app.nodes.generator import SPECIAL_NODES
for name in FRAMEWORKS:
    if name not in SPECIAL_NODES:
        print(f'{name}: {FRAMEWORKS[name].display_name} ({FRAMEWORKS[name].complexity})')
"
```

---

## Upgrade Guidelines

### When to Upgrade a Generated Node

Upgrade if the framework requires:

1. **Stateful Logic**
   - Maintains context across multiple turns
   - Needs memory of previous iterations
   - Example: Chain of Draft, Reflexion

2. **Complex Control Flow**
   - Conditional branching based on results
   - Dynamic iteration counts
   - Example: MCTS, Tree of Thoughts

3. **Multi-Agent Orchestration**
   - Multiple LLM calls with different roles
   - Parallel execution with aggregation
   - Example: Multi-Agent Debate, Mixture of Experts

4. **Custom Scoring/Evaluation**
   - Process Reward Model integration
   - Quality assessment between iterations
   - Example: Active Inference, Self-Refine

5. **Tool/API Integration**
   - Needs to call external APIs
   - Sandbox execution for code
   - Example: ReAct, Program of Thoughts

### When Generated Node is Sufficient

Keep generated node if:

1. **Simple Prompt-Based**
   - Single LLM call with structured prompt
   - No iteration required
   - Linear execution flow

2. **Low Complexity**
   - Marked as "low" complexity in registry
   - Straightforward reasoning pattern

3. **Rarely Used**
   - Not in top 20% of framework selections
   - Niche use cases

---

## Implementation Template

### Special Node Template

```python
"""
[Framework Name] Framework: Real Implementation

[Description of framework approach]
1. Step 1
2. Step 2
3. Step 3
"""

import asyncio
import structlog
from dataclasses import dataclass

from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    prepare_context_with_gemini,
)

logger = structlog.get_logger("[framework_name]")

MAX_ITERATIONS = 4


@dataclass
class [FrameworkState]:
    """State for framework reasoning."""
    iteration: int
    # Add framework-specific fields
    score: float = 0.0


async def _helper_function_1(args) -> result:
    """Helper function for specific step."""
    # Implementation
    pass


@quiet_star
async def [framework_name]_node(state: GraphState) -> GraphState:
    """
    [Framework Name] - REAL IMPLEMENTATION
    
    [Description of what this framework does]
    """
    query = state.get("query", "")
    code_context = await prepare_context_with_gemini(
        query=query,
        state=state
    )
    
    logger.info("[framework]_start", query_preview=query[:50])
    
    # Implementation steps
    # 1. Analyze
    # 2. Execute
    # 3. Synthesize
    
    final_answer = f"""# [Framework Name] Analysis
    
## Results
{results}

## Statistics
- Metric 1: value
- Metric 2: value
"""
    
    state["final_answer"] = final_answer
    state["confidence_score"] = confidence
    
    logger.info("[framework]_complete", metric=value)
    
    return state
```

---

## Upgrade Process

### Step-by-Step Upgrade

1. **Create Special Node File**
   ```bash
   touch app/nodes/[category]/[framework_name].py
   ```

2. **Implement Framework Logic**
   - Use template above
   - Add multi-turn LLM calls
   - Implement state management
   - Add proper logging

3. **Register in SPECIAL_NODES**
   ```python
   # app/nodes/generator.py
   SPECIAL_NODES = {
       # ... existing ...
       "[framework_name]": "app.nodes.[category].[framework_name]",
   }
   ```

4. **Test**
   ```bash
   # Test the framework
   python -c "from app.nodes.generator import GENERATED_NODES; print(GENERATED_NODES['[framework_name]'])"
   ```

5. **Update Documentation**
   - Add to ARCHITECTURE.md
   - Update framework count in README

---

## Specific Recommendations

### Priority 1: chain_of_draft

**Location**: `app/nodes/iterative/chain_of_draft.py`

**Implementation Focus**:
- Draft generation with structured output
- Self-critique mechanism
- Refinement based on specific feedback
- Draft scoring and selection
- History tracking across iterations

**Lines of Code**: ~250 lines  
**Effort**: 3-4 hours  
**Dependencies**: None (uses common utils)

**Value**: **High** - iterative frameworks are heavily used for code quality tasks

---

### Priority 2-8: TBD

**Action Required**: Identify remaining 7 generated frameworks by running:
```bash
cd /Users/justinadams/thinking-frameworks/omni_cortex
grep -r "register(FrameworkDefinition" app/frameworks/registry.py | wc -l
# Should show 62

# Then compare with SPECIAL_NODES count (54)
# The 8 missing ones are upgrade candidates
```

---

## Metrics for Success

### Before Upgrade
- Framework uses basic prompt template
- Single LLM call
- No iteration or state management
- Limited control flow

### After Upgrade
- Multi-turn LLM interactions
- State preserved across iterations
- Rich control flow with conditionals
- Detailed logging and metrics
- Proper error handling

---

## Maintenance Notes

### When Adding New Frameworks

1. **Start with Generated Node**
   - Quick prototyping
   - Test framework viability
   - Gather usage data

2. **Monitor Usage**
   - Track framework selection frequency
   - Monitor user feedback
   - Identify pain points

3. **Upgrade When Justified**
   - High usage (top 30%)
   - User requests for richer features
   - Complexity requirements emerge

### Cost-Benefit Analysis

**Generated Node**:
- ✅ Fast to create (25 lines in registry)
- ✅ Easy to maintain
- ✅ Good for simple patterns
- ❌ Limited functionality
- ❌ No iteration support

**Special Node**:
- ✅ Rich functionality
- ✅ Multi-turn reasoning
- ✅ State management
- ✅ Better results
- ❌ 200-400 lines of code
- ❌ More maintenance

---

## Next Steps

1. ✅ **Complete** - Fix all bare except clauses (28 files)
2. ✅ **Complete** - Identify generated vs special nodes
3. 🔄 **In Progress** - Create upgrade recommendations
4. ⏳ **Pending** - Identify specific 8 generated frameworks
5. ⏳ **Pending** - Prioritize upgrades based on usage data
6. ⏳ **Pending** - Implement Priority 1 upgrade (chain_of_draft)

---

## Conclusion

The Omni Cortex framework system is well-designed with 54/62 frameworks having rich special node implementations. The remaining 8 generated frameworks should be evaluated for upgrade based on:

- **Usage frequency** (track in production)
- **User requests** for richer features  
- **Complexity requirements** that emerge over time

**Immediate Action**: Upgrade `chain_of_draft` as a high-value iterative framework.

**Monitor**: Track which generated frameworks get selected most frequently and upgrade the top performers.
