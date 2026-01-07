# Omni-Cortex Code Review Report

**Date:** 2026-01-06
**Reviewer:** Senior Developer (Automated Analysis)
**Severity Scale:** CRITICAL > HIGH > MEDIUM > LOW

---

## Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Modules | 54 | - |
| Tested Modules | 9 | **17% coverage** |
| Untested LOC | 9,675 | **55% of codebase** |
| Mock Invocations | 161 | Hiding real bugs |
| Silent Exception Handlers | 5+ | Data loss risk |
| Duplicate Config Systems | 3 | Maintenance nightmare |

**Overall Grade: C-** - Solid architecture buried under technical debt.

---

## Critical Issues

### 1. Test Coverage Crisis

**Severity: CRITICAL**

The test suite covers only 17% of modules. Critical path components have zero tests:

| Module | Lines | Coverage | Risk |
|--------|-------|----------|------|
| `core/router.py` | 462 | 0% | Framework selection logic untested |
| `graph.py` | 297 | 0% | Workflow orchestration untested |
| `frameworks/registry.py` | 1,869 | 0% | 62 framework definitions untested |
| `collection_manager.py` | 583 | 0% | RAG system untested |
| `orchestrators/*.py` | 3,500+ | 0% | All 9 orchestrators untested |

**Impact:** Regressions will reach production undetected.

**Evidence:**
```
/omni_cortex/tests/
├── unit/
│   ├── test_sandbox.py       (559 lines)
│   ├── test_mcp_tools.py     (855 lines)
│   ├── test_framework_factory.py (420 lines)
│   ├── test_validation.py    (438 lines)
│   ├── test_resilient_sampler.py (366 lines)
│   ├── test_state.py         (325 lines)
│   ├── test_memory.py        (287 lines)
│   ├── test_refactor_smoke.py (236 lines)
│   └── test_correlation.py   (90 lines)
└── (NO integration test directory)
```

**Recommendation:** Prioritize tests for `router.py`, `graph.py`, and `collection_manager.py`.

---

### 2. Mock Overuse Hiding Real Bugs

**Severity: CRITICAL**

161 mock invocations return happy-path responses:

```python
# conftest.py - Global mocks affecting ALL tests
@pytest.fixture
def mock_deep_reasoner():
    with patch("app.nodes.common.call_deep_reasoner") as mock:
        mock.return_value = ("Mock response", 100)  # Always succeeds
```

**Never tested:**
- LLM API timeouts
- Rate limit errors
- Malformed API responses
- Token limit exceeded
- Network failures

**Recommendation:** Add integration tests with real API calls (even if limited) and failure injection tests.

---

### 3. Silent Exception Swallowing

**Severity: HIGH**

Exceptions are caught and silently ignored:

**File:** `app/core/sampling.py:397,403`
```python
try:
    return json.loads(match.group(1))
except json.JSONDecodeError:
    pass  # Silent failure - no logging, no metrics
```

**File:** `app/vector_schema.py:234`
```python
except SyntaxError as e:
    pass  # Returns empty imports silently
```

**Impact:** Data corruption and failures go undetected until downstream effects manifest.

**Recommendation:** Add `logger.debug()` or `logger.warning()` to all exception handlers.

---

## High Priority Issues

### 4. Duplicate Configuration Systems

**Severity: HIGH**

Three separate configuration sources create confusion:

| File | Purpose | Status |
|------|---------|--------|
| `core/settings.py` | Pydantic settings with thread-safe singleton | **Current** |
| `core/config.py` | Legacy Settings class + dead ModelConfig stub | **Deprecated** |
| `core/constants.py` | Centralized constants (underutilized) | **Partial** |

**Evidence of confusion:**
```python
# Some modules use old config
from .core.config import settings

# Other modules use new settings
from .core.settings import get_settings
```

**Dead code in config.py:**
```python
class ModelConfig:
    """Stub - raises NotImplementedError for all methods"""
    async def call_deep_reasoner(self, prompt: str, **kwargs):
        raise NotImplementedError("Use the MCP tools")
```

This `ModelConfig` is imported in `nodes/common.py:16` but never used.

**Recommendation:**
1. Delete `core/config.py`
2. Migrate all imports to `core/settings.py`
3. Move remaining constants to `core/constants.py`

---

### 5. Inconsistent Logging

**Severity: HIGH**

Mixed logging approaches across the codebase:

| Pattern | File Count | Location |
|---------|------------|----------|
| `structlog.get_logger()` | 27 | App modules |
| `logging.getLogger()` | 3 | Server handlers |
| `print()` | 5+ | Test files |

**Deprecated API usage:**
```python
# ingest_repo.py:69
logger.warn("no_docs_found")  # Deprecated since Python 3.2
```

Should be: `logger.warning("no_docs_found")`

**Debug print in production:**
```python
# frameworks/__init__.py:51
print(f"Total frameworks: {count()}")  # Should use logger
```

**Recommendation:** Standardize on `structlog` across all modules.

---

### 6. Duplicate Constants

**Severity: MEDIUM**

Same constants defined in multiple files with variations:

**DEFAULT_PATTERNS:**
```python
# ingest_repo.py:23-29
DEFAULT_PATTERNS = [
    "**/*.py", "**/*.md", "**/*.txt",
    "**/*.yaml", "**/*.yml",
]

# enhanced_ingestion.py:28-35
DEFAULT_PATTERNS = [
    "**/*.py", "**/*.md", "**/*.txt",
    "**/*.yaml", "**/*.yml", "**/*.json"  # Extra: .json
]
```

**SKIP_DIRS:**
```python
# ingest_repo.py:32
SKIP_DIRS = {"data", "venv", ".venv", "__pycache__", ".git", "node_modules", ".mcp"}

# enhanced_ingestion.py:38
SKIP_DIRS = {"data", "venv", ".venv", "__pycache__", ".git", "node_modules", ".mcp", ".pytest_cache"}
```

**VIBE_DICTIONARY:** Defined in both `vibe_dictionary.py` and `registry.py`

**Recommendation:** Consolidate all constants into `core/constants.py`.

---

### 7. Broad Exception Catching

**Severity: MEDIUM**

40+ instances of generic exception handling:

```python
except Exception as e:
    logger.error(...)
    # continue execution
```

Custom error hierarchy exists but is underutilized:

```python
# core/errors.py - Available but rarely used
class OmniCortexError(Exception): ...
class RoutingError(OmniCortexError): ...
class FrameworkNotFoundError(OmniCortexError): ...
class EmbeddingError(OmniCortexError): ...
class ConfigurationError(OmniCortexError): ...
```

**Recommendation:** Replace generic `Exception` catches with specific custom errors where appropriate.

---

### 8. Magic Numbers

**Severity: MEDIUM**

Hardcoded values scattered across modules:

```python
# nodes/common.py:30-37
DEFAULT_DEEP_REASONING_TOKENS = 4096
DEFAULT_FAST_SYNTHESIS_TOKENS = 2048
DEFAULT_PRM_TOKENS = 10
DEFAULT_PRM_TEMP = 0.1

# router.py:84
min(scores[best] / 5.0, 1.0)  # What is 5.0?

# router.py:76
word_count >= 2  # Magic threshold
```

`constants.py` has organized dataclasses but they're incomplete:

```python
@dataclass(frozen=True)
class FrameworkLimits:
    MAX_REASONING_DEPTH: int = 10
    MCTS_MAX_ROLLOUTS: int = 50
    # Token limits missing!
```

**Recommendation:** Move all magic numbers to `constants.py` dataclasses.

---

## Low Priority Issues

### 9. Unused Imports

**File:** `app/nodes/common.py:16`
```python
from ..core.config import model_config, settings
# model_config is imported but never used
```

### 10. Async Testing Gap

34 modules use async/await but only 11 async tests exist:
- `ResilientSampler`: 9 tests
- `FrameworkFactory`: 2 tests
- Everything else: 0

Race conditions and deadlocks are untested.

---

## File-by-File Issues

| File | Line | Issue | Severity |
|------|------|-------|----------|
| `nodes/common.py` | 16 | Unused `model_config` import | LOW |
| `core/sampling.py` | 397 | Silent `JSONDecodeError` | HIGH |
| `core/sampling.py` | 403 | Silent `JSONDecodeError` | HIGH |
| `vector_schema.py` | 234 | Silent `SyntaxError` | MEDIUM |
| `ingest_repo.py` | 69 | Deprecated `logger.warn()` | LOW |
| `frameworks/__init__.py` | 51 | Debug `print()` in production | LOW |
| `core/config.py` | 54-67 | Dead `ModelConfig` class | LOW |

---

## Recommended Action Plan

### Phase 1: Critical (This Week)

1. **Add tests for critical path**
   - `test_router.py` - Framework selection logic
   - `test_graph.py` - Workflow orchestration
   - `test_collection_manager.py` - RAG operations

2. **Fix silent exception handlers**
   - Add logging to `sampling.py:397,403`
   - Add logging to `vector_schema.py:234`

### Phase 2: High Priority (This Sprint)

3. **Consolidate configuration**
   - Delete `core/config.py`
   - Migrate all imports to `core/settings.py`

4. **Standardize logging**
   - Update 3 handler files to use `structlog`
   - Fix deprecated `logger.warn()` call
   - Remove debug `print()` statement

### Phase 3: Medium Priority (Next Sprint)

5. **Consolidate constants**
   - Move `DEFAULT_PATTERNS` to `constants.py`
   - Move `SKIP_DIRS` to `constants.py`
   - Single source for `VIBE_DICTIONARY`

6. **Add magic numbers to constants**
   - Token limits
   - Confidence thresholds
   - Scoring divisors

### Phase 4: Ongoing

7. **Improve exception handling**
   - Replace generic `Exception` with custom errors
   - Add error metrics/monitoring

8. **Add async tests**
   - Test concurrent operations
   - Test timeout behavior
   - Test error recovery

---

## Appendix: Test Coverage by Category

| Category | Modules | Tested | Coverage |
|----------|---------|--------|----------|
| Core Routing | 4 | 0 | 0% |
| Data/Retrieval | 5 | 1 | 20% |
| Framework Orchestration | 10 | 1 | 10% |
| Graph/Execution | 1 | 0 | 0% |
| Context/Analysis | 6 | 0 | 0% |
| Models/LLM | 2 | 0 | 0% |
| Prompts | 2 | 0 | 0% |
| Ingestion | 5 | 0 | 0% |
| Schemas | 3 | 1 | 33% |
| Callbacks | 2 | 0 | 0% |

---

## Appendix: Modules Requiring Tests (Priority Order)

### Tier 1 - Critical
- `app/core/router.py` (462 lines)
- `app/graph.py` (297 lines)
- `app/collection_manager.py` (583 lines)
- `app/frameworks/registry.py` (1,869 lines)

### Tier 2 - High
- `app/core/routing/*.py` (1,735 lines)
- `app/orchestrators/*.py` (3,500+ lines)
- `app/core/context_gateway.py` (453 lines)
- `app/models/*.py` (~400 lines)

### Tier 3 - Medium
- `app/retrieval/*.py` (~500 lines)
- `app/nodes/common.py` (645 lines)
- `app/core/context_utils.py` (481 lines)
- `app/prompts/*.py` (~300 lines)

---

*Report generated by automated code analysis. Manual review recommended for nuanced issues.*
