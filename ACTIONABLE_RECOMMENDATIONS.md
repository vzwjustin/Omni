# Actionable Recommendations for Omni Cortex
**Generated**: 2026-01-05
**Priority-Ordered Action Plan**

---

## Quick Start: What to Do Next

If you only have time for **3 things**, do these:

1. **Add Basic Test Suite** (2-3 hours)
   - Create `tests/test_router.py` with basic routing tests
   - Create `tests/test_frameworks.py` with smoke tests for all 62 frameworks
   - Set up pytest configuration

2. **Fix Silent Error Handling** (1-2 hours)
   - Search codebase for `except Exception: pass`
   - Replace with proper logging and error propagation
   - Add custom exception types

3. **Pin Dependencies** (15 minutes)
   - Update `requirements.txt` with exact versions
   - Test that installation still works
   - Document any breaking changes

---

## Priority 0: Critical (Do First)

### C1: Implement Basic Test Suite

**Why**: Near-zero test coverage means any change could break the system

**Steps**:
```bash
# 1. Create test directory structure
mkdir -p tests/{unit,integration,e2e}
touch tests/__init__.py
touch tests/conftest.py

# 2. Install pytest
pip install pytest pytest-asyncio pytest-cov

# 3. Create pytest.ini
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
EOF

# 4. Create basic router test
cat > tests/unit/test_router.py << 'EOF'
import pytest
from omni_cortex.app.core.router import HyperRouter

@pytest.mark.asyncio
async def test_category_routing_debug():
    router = HyperRouter()
    category, confidence = router._route_to_category("this is broken and needs debugging")
    assert category == "debug"
    assert confidence > 0.5

@pytest.mark.asyncio
async def test_vibe_dictionary_match():
    router = HyperRouter()
    framework = router._check_vibe_dictionary("spaghetti code needs refactoring")
    assert framework == "graph_of_thoughts"

@pytest.mark.asyncio
async def test_select_framework_chain():
    router = HyperRouter()
    chain, reasoning, category = await router.select_framework_chain(
        "fix this complex bug in my async handler"
    )
    assert category == "debug"
    assert len(chain) > 0
    assert isinstance(chain, list)
EOF

# 5. Create framework smoke tests
cat > tests/unit/test_frameworks.py << 'EOF'
import pytest
from omni_cortex.app.state import GraphState
from omni_cortex.app.graph import FRAMEWORK_NODES

@pytest.mark.asyncio
@pytest.mark.parametrize("framework_name", list(FRAMEWORK_NODES.keys()))
async def test_framework_basic_execution(framework_name):
    """Smoke test: ensure every framework can execute without crashing"""
    framework_fn = FRAMEWORK_NODES[framework_name]
    state = GraphState(
        query="test query",
        code_snippet="def test(): pass",
        file_list=[],
        working_memory={},
        reasoning_steps=[]
    )
    result = await framework_fn(state)
    assert "final_answer" in result or "final_code" in result
    assert result.get("confidence_score", 0) > 0
EOF

# 6. Run tests
pytest tests/ -v --cov=omni_cortex --cov-report=html
```

**Expected Outcome**:
- Basic test coverage (target: 20-30% initially)
- Confidence that core functionality works
- Foundation for future tests

**Time Estimate**: 2-3 hours

---

### C2: Fix Silent Error Handling

**Why**: Silent failures hide bugs and make debugging impossible

**Steps**:
```bash
# 1. Find all silent exception handlers
grep -r "except Exception:" omni_cortex/app/ | grep "pass"

# 2. Create custom exceptions
cat > omni_cortex/app/exceptions.py << 'EOF'
"""Custom exceptions for Omni Cortex."""

class OmniCortexError(Exception):
    """Base exception for Omni Cortex."""
    pass

class RoutingError(OmniCortexError):
    """Error during framework routing."""
    pass

class FrameworkExecutionError(OmniCortexError):
    """Error during framework execution."""
    pass

class VectorStoreError(OmniCortexError):
    """Error with vector store operations."""
    pass

class MemoryError(OmniCortexError):
    """Error with memory operations."""
    pass
EOF

# 3. Replace silent handlers (example)
# BEFORE:
# try:
#     result = some_operation()
# except Exception as e:
#     pass

# AFTER:
# try:
#     result = some_operation()
# except Exception as e:
#     logger.error("operation_failed", error=str(e), exc_info=True)
#     raise FrameworkExecutionError(f"Failed to execute: {e}") from e
```

**Files to Fix** (in order of priority):
1. `app/core/router.py` - Lines 1081, 1152, 1297, 1331
2. `app/langchain_integration.py` - Lines 333, 348, 398, 658, 686
3. `app/collection_manager.py` - Lines 52, 75, 121, 190, 260, 312
4. `app/nodes/common.py` - Lines 159, 238

**Expected Outcome**:
- Errors are logged and visible
- Stack traces preserved
- Easier debugging

**Time Estimate**: 1-2 hours

---

### C3: Pin Dependencies

**Why**: Unpinned dependencies can introduce breaking changes

**Steps**:
```bash
# 1. Generate current locked versions
pip freeze | grep -E "(mcp|langgraph|langchain|chromadb|anthropic|openai)" > requirements.lock

# 2. Update requirements.txt
cat > omni_cortex/requirements.txt << 'EOF'
# Core MCP
mcp[cli]==1.0.0

# LangChain ecosystem
langgraph==0.2.45
langchain==0.3.7
langchain-core==0.3.15
langchain-community==0.3.5
langchain-anthropic==0.2.4
langchain-openai==0.2.5
langchain-huggingface==0.1.2

# Vector stores
chromadb==0.5.23

# LLM providers
anthropic==0.40.0
openai==1.58.1

# Utilities
pydantic==2.10.3
pydantic-settings==2.6.1
python-dotenv==1.0.1
structlog==24.4.0

# Development
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-cov==6.0.0
EOF

# 3. Test installation
pip install -r omni_cortex/requirements.txt

# 4. Run smoke tests
pytest tests/unit/test_frameworks.py -k "test_framework_basic_execution[active_inference]"
```

**Expected Outcome**:
- Reproducible builds
- No surprise breaking changes
- Clear upgrade path

**Time Estimate**: 15-30 minutes

---

## Priority 1: High (Do Soon)

### H1: Add Input Validation

**Implementation**:
```python
# omni_cortex/app/validation.py
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    code_snippet: Optional[str] = Field(None, max_length=100000)  # 100KB limit
    thread_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]{1,64}$')

    @field_validator('code_snippet')
    def validate_code_snippet(cls, v):
        if v and len(v.encode('utf-8')) > 100000:
            raise ValueError('Code snippet exceeds 100KB limit')
        return v

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    k: int = Field(default=5, ge=1, le=100)
    collection_names: list[str] = Field(default_factory=list)
```

**Where to Add**:
- `server/main.py` - Validate all MCP tool inputs
- `app/core/router.py` - Validate routing inputs
- `app/langchain_integration.py` - Validate search parameters

**Time Estimate**: 1-2 hours

---

### H2: Add LLM Call Timeouts

**Implementation**:
```python
# In app/nodes/common.py

def call_deep_reasoner(
    prompt: str,
    state: GraphState,
    max_tokens: int = 4000,
    temperature: float = 0.7,
    timeout: float = 30.0  # NEW: 30 second timeout
) -> str:
    """Call deep reasoning model with timeout."""
    try:
        llm = _get_llm_client(
            model_type="deep",
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Add timeout wrapper
        import asyncio
        result = asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=timeout
        )

        return result.content
    except asyncio.TimeoutError:
        logger.error("llm_call_timeout", prompt_length=len(prompt), timeout=timeout)
        raise FrameworkExecutionError(f"LLM call timed out after {timeout}s")
```

**Time Estimate**: 1 hour

---

### H3: Fix Framework Count Documentation - COMPLETED

**Status**: ✅ RESOLVED

All documentation files have been updated to correctly state 62 frameworks:
1. `/home/user/Omni/README.md` - ✅ Already says 62 frameworks
2. `/home/user/Omni/omni_cortex/README.md` - ✅ Already says 62 frameworks
3. `/home/user/Omni/omni_cortex/FRAMEWORKS.md` - ✅ Lists all 62 frameworks
4. `/home/user/Omni/omni_cortex/ARCHITECTURE.md` - ✅ Updated to 62 frameworks
5. `/home/user/Omni/omni_cortex/MCP_SETUP.md` - ✅ Updated to 62 frameworks
6. `/home/user/Omni/omni_cortex/CLAUDE.md` - ✅ Updated to 62 frameworks
7. `/home/user/Omni/omni_cortex/CODEBASE_ISSUES.md` - ✅ Updated to 62 frameworks
8. `/home/user/Omni/omni_cortex/INCOMPLETE_CODE_AND_DOCS_AUDIT.md` - ✅ Updated to 62 frameworks

---

### H4: Decide on PoT Sandbox

**Option A: Integrate the Sandbox**
```python
# In app/nodes/code/pot.py
async def program_of_thoughts_node(state: GraphState) -> GraphState:
    # ... existing prompt generation ...

    # NEW: Actually execute the sandbox if code is provided
    if state.get("code_snippet"):
        try:
            execution_result = await _safe_execute(state["code_snippet"])
            state["working_memory"]["pot_execution"] = execution_result
            # Include execution result in prompt
        except Exception as e:
            logger.error("pot_execution_failed", error=str(e))

    # ... rest of function ...
```

**Option B: Remove Dead Code**
```bash
# Delete lines 33-176 in app/nodes/code/pot.py
# Keep only the protocol generator
```

**Recommendation**: Option A (integrate) - The sandbox is well-implemented and adds value

**Time Estimate**: 1-2 hours (Option A) or 15 minutes (Option B)

---

## Priority 2: Medium (Technical Debt)

### M1: Deduplicate Framework Definitions

**Current State**: Framework definitions exist in TWO places:
1. `server/main.py` - Lines 67-1041 (inline FRAMEWORKS dict)
2. `router.py` - Framework metadata

**Solution**:
```python
# 1. Create single source of truth
# omni_cortex/app/framework_registry.py
from typing import TypedDict

class FrameworkMetadata(TypedDict):
    name: str
    category: str
    description: str
    use_cases: list[str]
    protocol: str  # Full protocol text

FRAMEWORK_REGISTRY: dict[str, FrameworkMetadata] = {
    "active_inference": {
        "name": "Active Inference",
        "category": "iterative",
        "description": "Root cause analysis through hypothesis testing",
        "use_cases": ["Bug hunting", "Impossible bugs", "Root cause analysis"],
        "protocol": """# Active Inference Protocol
        ...
        """
    },
    # ... all 62 frameworks
}

# 2. Update server/main.py to import from registry
from omni_cortex.app.framework_registry import FRAMEWORK_REGISTRY

@server.list_tools()
async def list_tools() -> list[Tool]:
    tools = []
    for name, metadata in FRAMEWORK_REGISTRY.items():
        tools.append(Tool(
            name=f"think_{name}",
            description=f"[{metadata['category'].upper()}] {metadata['description']}",
            inputSchema={...}
        ))
    return tools

# 3. Update router.py to import from registry
from omni_cortex.app.framework_registry import FRAMEWORK_REGISTRY

class HyperRouter:
    FRAMEWORKS = {name: meta["description"] for name, meta in FRAMEWORK_REGISTRY.items()}
```

**Time Estimate**: 3-4 hours

---

### M2: Refactor Router

**Split router.py into modules**:
```
omni_cortex/app/core/
├── router/
│   ├── __init__.py
│   ├── category_router.py      # Category routing logic
│   ├── vibe_dictionary.py      # VIBE_DICTIONARY constant
│   ├── specialist_agents.py    # Specialist prompts and logic
│   ├── chain_patterns.py       # Pre-defined chains
│   └── hyper_router.py         # Main HyperRouter class
```

**Time Estimate**: 4-5 hours

---

### M3: Enhance Generic Frameworks

**Target**: 10 frameworks with generic steps

**Template for Enhancement** (example: `active_inf.py`):
```python
# BEFORE:
prompt = f"""# Active Inference Protocol
...
### Framework Steps
1. Analyze the Problem
2. Apply Framework Principles
3. Generate Solution
"""

# AFTER:
prompt = f"""# Active Inference Protocol
...
### Framework Steps
1. FORMULATE HYPOTHESIS: What could cause this behavior?
   - List 3-5 plausible hypotheses
   - Rank by likelihood

2. DESIGN TESTS: How to confirm/reject each hypothesis?
   - Minimal reproducible test cases
   - Expected vs actual outcomes

3. EXECUTE INFERENCE LOOP:
   - Run test for hypothesis #1
   - Update belief distribution
   - If inconclusive, test hypothesis #2
   - Iterate until confident

4. VERIFY SOLUTION:
   - Confirm fix addresses root cause
   - No regression in other areas
"""
```

**Frameworks to Enhance**:
1. `reason_flux.py`
2. `self_discover.py`
3. `bot.py`
4. `mcts_rstar.py`
5. `active_inf.py`
6. `debate.py`
7. `system1.py`
8. `analogical.py`
9. `step_back.py`

**Time Estimate**: 30 minutes per framework (5 hours total)

---

### M4: Optimize Vibe Matching

**Current Performance**: O(frameworks × vibes × query_length)

**Option A: Pre-compile Regex Patterns**
```python
import re

class HyperRouter:
    def __init__(self):
        # Pre-compile regex patterns for faster matching
        self._vibe_patterns = {}
        for framework, vibes in self.VIBE_DICTIONARY.items():
            # Combine vibes into single regex
            escaped_vibes = [re.escape(vibe) for vibe in vibes]
            pattern = re.compile(
                r'\b(' + '|'.join(escaped_vibes) + r')\b',
                re.IGNORECASE
            )
            self._vibe_patterns[framework] = pattern

    def _check_vibe_dictionary(self, query: str) -> Optional[str]:
        scores = {}
        for framework, pattern in self._vibe_patterns.items():
            matches = pattern.findall(query.lower())
            if matches:
                # Score based on match count and length
                total_score = sum(len(m.split()) for m in matches)
                scores[framework] = total_score
        # ... rest of logic
```

**Option B: Use Trie Data Structure**
```python
from pygtrie import StringTrie

class HyperRouter:
    def __init__(self):
        # Build trie for fast prefix matching
        self._vibe_trie = StringTrie()
        for framework, vibes in self.VIBE_DICTIONARY.items():
            for vibe in vibes:
                self._vibe_trie[vibe] = (framework, len(vibe.split()))
```

**Recommendation**: Option A (regex) - Simpler and sufficient

**Time Estimate**: 2 hours

---

## Priority 3: Low (Nice to Have)

### L1: Add Performance Monitoring

**Implementation**:
```python
# omni_cortex/app/monitoring.py
import time
from contextlib import contextmanager
from collections import defaultdict
import structlog

logger = structlog.get_logger()

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    @contextmanager
    def measure(self, operation: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.metrics[operation].append(duration)
            logger.info("performance_metric", operation=operation, duration_ms=duration*1000)

    def get_stats(self, operation: str):
        values = self.metrics[operation]
        if not values:
            return None
        return {
            "count": len(values),
            "avg_ms": sum(values) / len(values) * 1000,
            "min_ms": min(values) * 1000,
            "max_ms": max(values) * 1000
        }

monitor = PerformanceMonitor()

# Usage:
async def select_framework_chain(self, query: str, ...):
    with monitor.measure("routing_decision"):
        category, confidence = self._route_to_category(query)
        # ... rest of routing
```

**Time Estimate**: 2-3 hours

---

### L2: Add Health Checks

**Implementation**:
```python
# omni_cortex/server/health.py
from typing import Dict, Any

async def health_check() -> Dict[str, Any]:
    """Comprehensive health check."""
    results = {
        "status": "healthy",
        "checks": {}
    }

    # Check ChromaDB
    try:
        from omni_cortex.app.langchain_integration import get_vectorstore
        vs = get_vectorstore()
        vs.similarity_search("test", k=1)
        results["checks"]["chromadb"] = {"status": "ok"}
    except Exception as e:
        results["checks"]["chromadb"] = {"status": "error", "error": str(e)}
        results["status"] = "degraded"

    # Check LLM provider (if not pass-through)
    try:
        from omni_cortex.app.core.config import settings
        if settings.llm_provider != "pass-through":
            from omni_cortex.app.nodes.common import get_chat_model
            llm = get_chat_model("fast")
            # Quick test call
            results["checks"]["llm_provider"] = {"status": "ok"}
    except Exception as e:
        results["checks"]["llm_provider"] = {"status": "error", "error": str(e)}

    # Check memory store
    try:
        from omni_cortex.app.langchain_integration import get_memory
        await get_memory("health_check")
        results["checks"]["memory_store"] = {"status": "ok"}
    except Exception as e:
        results["checks"]["memory_store"] = {"status": "error", "error": str(e)}
        results["status"] = "degraded"

    return results

# Add to MCP server
@server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    if name == "health_check":
        health = await health_check()
        return [TextContent(type="text", text=json.dumps(health, indent=2))]
```

**Time Estimate**: 1-2 hours

---

## Quick Wins (30 minutes each)

### QW1: Add .gitignore Entries
```bash
echo "/data/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo ".coverage" >> .gitignore
echo "htmlcov/" >> .gitignore
echo ".env" >> .gitignore
```

### QW2: Add Pre-commit Hooks
```bash
pip install pre-commit
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
EOF
pre-commit install
```

### QW3: Add Type Checking
```bash
pip install mypy
cat > mypy.ini << 'EOF'
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False  # Start permissive
ignore_missing_imports = True
EOF
mypy omni_cortex/app/core/
```

---

## Execution Plan: First Week

### Day 1: Critical Foundations
- [ ] Set up pytest (C1)
- [ ] Create basic test suite (C1)
- [ ] Pin dependencies (C3)

### Day 2: Error Handling
- [ ] Create custom exceptions (C2)
- [ ] Fix router.py error handling (C2)
- [ ] Fix langchain_integration.py error handling (C2)

### Day 3: Input Validation
- [ ] Add validation module (H1)
- [ ] Add LLM timeouts (H2)
- [ ] Add input validation to MCP tools (H1)

### Day 4: Documentation
- [ ] Fix framework count (H3)
- [ ] Update README (H3)
- [ ] Synchronize all docs (H3)

### Day 5: Code Cleanup
- [ ] Decide on PoT sandbox (H4)
- [ ] Remove or integrate dead code (H4)
- [ ] Add .gitignore entries (QW1)

---

## Success Metrics

| Metric | Current | Target (1 month) |
|--------|---------|------------------|
| Test Coverage | 0% | 50% |
| Silent Error Handlers | 30+ | 0 |
| Unpinned Dependencies | 10+ | 0 |
| Framework Count Docs Match | No | Yes |
| Input Validation | None | Comprehensive |
| Dead Code | Yes | No |
| Performance Monitoring | No | Basic |

---

## Resources Needed

**Tools**:
- pytest, pytest-asyncio, pytest-cov
- mypy (type checking)
- black, isort (formatting)
- pre-commit (git hooks)

**Time Investment**:
- **Week 1**: 20-25 hours (Critical + High priority)
- **Week 2-3**: 15-20 hours (Medium priority)
- **Week 4**: 10 hours (Low priority + polish)

**Total**: ~50-60 hours for comprehensive improvements

---

*End of Recommendations*
