# Omni-Cortex Enhancement Roadmap

**Generated:** 2026-01-10  
**Current State:** Level 1 (Basic Setup) - 115 lint errors remaining

## Priority 1: Critical Infrastructure

### 1.1 Rate Limiting Integration ⚠️
- **Status:** Rate limiter exists (`app/core/rate_limiter.py`) but only used in `server/main.py`
- **Impact:** High - prevents API abuse, protects against runaway costs
- **Action:**
  - Integrate rate limiter into all LLM-calling nodes
  - Add rate limiting to Context Gateway operations
  - Add per-user/per-session rate limits
  - Track rate limit metrics in Prometheus
- **Files:**
  - `app/nodes/*/*.py` (62 framework nodes)
  - `app/core/context_gateway.py`
  - `server/handlers/*.py`

### 1.2 Circuit Breaker Coverage 🔥
- **Status:** Circuit breaker exists but incomplete coverage
- **Impact:** High - prevents cascade failures, improves resilience
- **Action:**
  - Wrap all external API calls (LLM providers, embeddings)
  - Add circuit breakers to ChromaDB operations
  - Add circuit breakers to file discovery operations
  - Implement fallback strategies for each protected operation
- **Files:**
  - `app/retrieval/embeddings.py`
  - `app/collection_manager.py`
  - `app/nodes/common.py`

### 1.3 Prometheus Metrics Implementation 📊
- **Status:** Stubs exist when `prometheus_client` not installed
- **Impact:** Medium-High - no observability without metrics
- **Action:**
  - Make `prometheus_client` required dependency
  - Remove stub implementations from `app/core/metrics.py`
  - Add metrics endpoint to MCP server
  - Instrument all 62 frameworks with timing/success metrics
  - Add Context Gateway metrics (cache hit rate, query analysis time)
- **Dependencies:** `prometheus_client>=0.20.0`

### 1.4 Error Handling Refinement 🛡️
- **Status:** 62 files with broad `except Exception` handlers
- **Impact:** Medium - masks bugs, makes debugging harder
- **Action:**
  - Replace `except Exception` with specific exception types
  - Use custom exceptions from `app/core/errors.py` (32 types available)
  - Add correlation IDs to all error logs
  - Implement structured error responses for MCP
- **Example:**
  ```python
  # Before
  except Exception as e:
      logger.error("search_failed", error=str(e))
  
  # After
  except (RAGError, EmbeddingError) as e:
      logger.error("search_failed", error=str(e), correlation_id=get_correlation_id())
  except ValueError as e:
      logger.error("search_validation_failed", error=str(e))
  ```

## Priority 2: Code Quality

### 2.1 Complexity Reduction 🔧
- **Status:** 49 violations (23 too-many-branches, 14 complex-structure, 12 too-many-statements)
- **Impact:** Medium - hard to maintain, test, debug
- **Action:**
  - Refactor top 10 most complex functions
  - Extract helper functions for complex branches
  - Use strategy pattern for framework selection logic
  - Add complexity checks to pre-commit hooks
- **Target:** Max complexity 15 → 10
- **Files:**
  - `app/frameworks/registry.py` (2791 lines)
  - `app/core/context_gateway.py` (1514 lines)
  - `app/core/router.py` (654 lines)

### 2.2 Caching Enhancement 💾
- **Status:** Only 1 `@lru_cache` decorator in entire codebase
- **Impact:** Medium - repeated computation wastes resources
- **Action:**
  - Add caching to framework registry lookups
  - Cache vibe matching results
  - Cache file discovery results (with TTL)
  - Add Redis support for distributed caching
- **Candidates:**
  ```python
  # app/frameworks/registry.py
  @lru_cache(maxsize=128)
  def find_by_name(name: str) -> FrameworkDefinition | None
  
  # app/core/vibe_dictionary.py
  @lru_cache(maxsize=256)
  def match_vibes(query: str) -> list[tuple[str, float]]
  
  # app/core/router.py
  @lru_cache(maxsize=64)
  def _get_specialist_prompt_template(category: str) -> tuple
  ```

### 2.3 Type Hints Coverage 📝
- **Status:** 58 files import typing, but coverage incomplete
- **Impact:** Low-Medium - harder to catch bugs at dev time
- **Action:**
  - Add type hints to all public functions
  - Enable `mypy` strict mode in pre-commit
  - Add `py.typed` marker for library users
  - Use `typing.Protocol` for interfaces
- **Target:** 100% coverage for public APIs

### 2.4 Lint Error Resolution ✨
- **Status:** 115 remaining errors (down from 4860)
- **Impact:** Low - mostly style, some real issues
- **Action:**
  - Fix remaining too-many-branches (23)
  - Fix unused-method-argument (15)
  - Fix redefined-outer-name (12)
  - Fix too-many-statements (12)
- **Target:** 0 errors

## Priority 3: Testing & Validation

### 3.1 Test Coverage Expansion 🧪
- **Status:** 40 test files, 50% minimum coverage, tests failing to collect
- **Impact:** High - can't verify correctness
- **Action:**
  - Fix test collection errors
  - Add integration tests for Context Gateway
  - Add tests for all 62 frameworks (smoke tests)
  - Add property-based tests with Hypothesis
  - Add load tests for rate limiter
  - Add chaos tests for circuit breakers
- **Target:** 80% coverage

### 3.2 Contract Testing 📋
- **Status:** No contract tests for MCP protocol
- **Impact:** Medium - breaking changes to MCP interface
- **Action:**
  - Add OpenAPI/JSON Schema validation for MCP requests
  - Add Pact-style contract tests
  - Validate all tool schemas against MCP spec
  - Test LEAN_MODE vs full mode compatibility

### 3.3 Performance Benchmarks ⚡
- **Status:** No performance tests
- **Impact:** Medium - don't know if optimizations work
- **Action:**
  - Benchmark Context Gateway operations
  - Benchmark framework routing decisions
  - Benchmark embedding operations
  - Add performance regression tests to CI

## Priority 4: Security Hardening

### 4.1 Input Validation 🔒
- **Status:** Basic validation in rate limiter, incomplete elsewhere
- **Impact:** High - injection attacks, DoS
- **Action:**
  - Add Pydantic models for all MCP tool inputs
  - Validate file paths prevent directory traversal
  - Sanitize code execution inputs
  - Add max depth limits for recursive operations
- **Example:**
  ```python
  from pydantic import BaseModel, Field, validator
  
  class PrepareContextArgs(BaseModel):
      query: str = Field(..., min_length=1, max_length=1000)
      workspace_path: str | None = Field(None, max_length=500)
      
      @validator("workspace_path")
      def validate_path(cls, v):
          if v and (".." in v or v.startswith("/")):
              raise ValueError("Invalid path")
          return v
  ```

### 4.2 Secrets Management 🔐
- **Status:** Using environment variables, no rotation
- **Impact:** Medium - secrets in logs, no rotation
- **Action:**
  - Integrate with HashiCorp Vault or AWS Secrets Manager
  - Add secrets rotation support
  - Implement API key scoping (per-user, per-tool)
  - Add audit logging for secret access
  - Enhance log scrubbing (`app/core/log_scrubbing.py`)

### 4.3 Dependency Security 🛡️
- **Status:** Dependabot enabled, 1 moderate vulnerability detected
- **Impact:** Medium - supply chain attacks
- **Action:**
  - Fix detected GitHub vulnerability
  - Enable Trivy scanning in CI (already configured)
  - Add license compatibility checks
  - Pin all transitive dependencies

## Priority 5: Performance Optimization

### 5.1 Async Optimization 🚀
- **Status:** Many nodes are async but some block
- **Impact:** Medium - slower than necessary
- **Action:**
  - Audit all `await` calls for blocking I/O
  - Use `asyncio.gather()` for parallel operations
  - Add connection pooling for ChromaDB
  - Use async file I/O for large files
- **Targets:**
  - File discovery: parallelize directory traversal
  - Documentation search: parallel web requests
  - Multi-repo operations: concurrent git clones

### 5.2 Memory Optimization 💾
- **Status:** No memory profiling
- **Impact:** Medium - OOM on large codebases
- **Action:**
  - Profile memory usage with `memory_profiler`
  - Add streaming for large file operations
  - Implement chunk-based processing for embeddings
  - Add memory limits to sandbox execution
- **Target:** Support 100k+ file repositories

### 5.3 Database Optimization 🗄️
- **Status:** ChromaDB works but no tuning
- **Impact:** Low-Medium - slow on large collections
- **Action:**
  - Add connection pooling
  - Tune batch sizes for ingestion
  - Add index optimization
  - Consider PostgreSQL pgvector for production

## Priority 6: Features & UX

### 6.1 Multi-Tenancy Support 👥
- **Status:** Single-user system
- **Impact:** Medium - can't deploy for teams
- **Action:**
  - Add user authentication/authorization
  - Implement per-user rate limits
  - Add usage tracking per user
  - Implement data isolation

### 6.2 Observability Dashboard 📊
- **Status:** Logs only, no visualization
- **Impact:** Medium - hard to understand system behavior
- **Action:**
  - Add Grafana dashboards for metrics
  - Add Jaeger for distributed tracing
  - Add ELK stack for log aggregation
  - Create health check dashboard

### 6.3 Framework Recommendation Engine 🤖
- **Status:** Basic vibe matching, rule-based routing
- **Impact:** Low-Medium - could be smarter
- **Action:**
  - Train ML model on historical framework performance
  - Add collaborative filtering (similar queries)
  - Add A/B testing for framework selection
  - Track and learn from framework success rates

### 6.4 Context Gateway Enhancements 🧠
- **Status:** Feature-complete but could be smarter
- **Impact:** Low-Medium - better context = better results
- **Action:**
  - Add semantic code search (beyond text matching)
  - Add code graph analysis (call graphs, dependency graphs)
  - Add historical context (previous queries, successful solutions)
  - Add workspace understanding (project structure, conventions)

## Priority 7: Developer Experience

### 7.1 Local Development Improvements 🛠️
- **Status:** Docker works, local dev setup complex
- **Impact:** Low - slows onboarding
- **Action:**
  - Create dev container with all tools
  - Add `make` targets for common tasks
  - Add hot-reload for code changes
  - Improve error messages in setup scripts

### 7.2 Documentation Expansion 📚
- **Status:** Good architecture docs, missing guides
- **Impact:** Low-Medium - harder to contribute
- **Action:**
  - Add framework authoring guide
  - Add MCP integration examples
  - Add troubleshooting guide
  - Add performance tuning guide
  - Document all configuration options

### 7.3 CLI Tools 🖥️
- **Status:** Basic scripts, no unified CLI
- **Impact:** Low - manual operations tedious
- **Action:**
  - Create `omni` CLI with Click/Typer
  - Add commands: `omni ingest`, `omni search`, `omni stats`
  - Add interactive framework selection
  - Add configuration validation

## Quick Wins (Low Effort, High Impact)

1. ✅ **Fix remaining lint errors** (2-3 hours)
2. ✅ **Add Prometheus metrics** (4-6 hours) 
3. ✅ **Integrate rate limiter everywhere** (4-6 hours)
4. ✅ **Add caching decorators** (2-4 hours)
5. ✅ **Fix test collection** (1-2 hours)
6. ✅ **Document all environment variables** (1-2 hours)
7. ✅ **Add health check endpoint** (1 hour)
8. ✅ **Fix GitHub vulnerability** (30 minutes)

## Tracking

- **GitHub Issues:** Create issues for each Priority 1-2 item
- **Milestones:**
  - v0.2.0: Priority 1 complete (Level 2 agent readiness)
  - v0.3.0: Priority 2 complete (Level 3 agent readiness)
  - v1.0.0: Production-ready (all priorities complete)
