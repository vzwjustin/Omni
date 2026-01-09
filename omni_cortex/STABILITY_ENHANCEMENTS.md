# Stability Enhancement Plan
**Date**: January 9, 2026
**Analyzed with**: Enhanced Context Preparation + Chain of Verification
**Priority**: High - Production Reliability

---

## Executive Summary

Comprehensive stability analysis using the newly integrated enhanced features identified **4 critical** and **5 moderate** stability issues that could impact production reliability.

**Analysis Method**: Used `prepare_context` with enhanced features → `reason` with chain_of_verification framework

**Status**: Circuit breakers are correctly implemented ✅, but cache and async patterns need hardening.

---

## 🔴 Critical Issues

### 1. Cache Thundering Herd Problem
**Severity**: CRITICAL
**Impact**: Multiple concurrent requests regenerate same expired cache entry
**Location**: `app/core/context/context_cache.py:261-300`

**Problem**:
```python
async def get(self, cache_key: str, allow_stale: bool = False):
    # Line 279: Check if key exists
    if cache_key not in self._cache:
        return None

    # Line 289: Get entry
    entry = self._cache[cache_key]

    # Line 293: If expired, return None (or stale)
    if entry.is_expired:
        if allow_stale:
            return entry  # stale
        return None  # triggers regeneration
```

**Race Condition**:
1. 10 concurrent requests check cache at t=0
2. All see same expired entry
3. All return None (or stale)
4. All 10 trigger expensive Gemini API calls simultaneously
5. Results in 10x cost, potential rate limiting

**Solution**: Add pending request tracking
```python
class ContextCache:
    def __init__(self):
        self._pending_regenerations: Dict[str, asyncio.Lock] = {}
        self._pending_lock = asyncio.Lock()

    async def get_or_generate(self, cache_key, generator_func):
        # Check cache first
        entry = await self.get(cache_key)
        if entry and not entry.is_expired:
            return entry.value

        # Check if regeneration already pending
        async with self._pending_lock:
            if cache_key not in self._pending_regenerations:
                self._pending_regenerations[cache_key] = asyncio.Lock()

        # Only first request generates, others wait
        async with self._pending_regenerations[cache_key]:
            # Double-check cache (might have been populated)
            entry = await self.get(cache_key)
            if entry and not entry.is_expired:
                return entry.value

            # Generate and cache
            value = await generator_func()
            await self.set(cache_key, value, cache_type)

            # Cleanup pending lock
            async with self._pending_lock:
                del self._pending_regenerations[cache_key]

            return value
```

**Files to Modify**:
- `app/core/context/context_cache.py` - Add pending request tracking
- `app/core/context_gateway.py` - Use new `get_or_generate` pattern

---

### 2. Missing asyncio.Lock for Async State Mutations
**Severity**: CRITICAL (under high concurrency)
**Impact**: Race conditions in cache statistics and metrics
**Location**: Multiple locations

**Problem - Cache Stats** (`app/core/context/context_cache.py:330-360`):
```python
def _track_cache_hit(self, cache_type: str, is_stale: bool = False, age_seconds: float = 0):
    # Line 332: NOT thread-safe for async
    if cache_type not in self._cache_stats["hits"]:
        self._cache_stats["hits"][cache_type] = 0
    self._cache_stats["hits"][cache_type] += 1  # RACE CONDITION
```

Two concurrent tasks can:
1. Both read current count = 5
2. Both increment to 6
3. Both write 6 (lost update)
4. Result: Count is 6 instead of 7

**Problem - Relevance Tracker** (`app/core/context/relevance_tracker.py`):
Similar issues with session state mutations.

**Solution**: Use asyncio.Lock for all async state mutations
```python
class ContextCache:
    def __init__(self):
        self._stats_lock = asyncio.Lock()

    async def _track_cache_hit_async(self, cache_type: str, ...):
        async with self._stats_lock:
            if cache_type not in self._cache_stats["hits"]:
                self._cache_stats["hits"][cache_type] = 0
            self._cache_stats["hits"][cache_type] += 1
```

**Files to Modify**:
- `app/core/context/context_cache.py` - Add async locks for stats
- `app/core/context/relevance_tracker.py` - Add async locks for session state
- `app/core/context/gateway_metrics.py` - Verify thread-safety

---

### 3. Unhandled Exceptions in watchdog File System Observer
**Severity**: HIGH
**Impact**: File system watcher can crash silently
**Location**: `app/core/context/context_cache.py:32-57`

**Problem**:
```python
class WorkspaceChangeHandler(FileSystemEventHandler):
    def on_any_event(self, event: FileSystemEvent) -> None:
        # Line 40: No try/except
        if event.is_directory:
            return

        # Line 46-47: Can raise exceptions
        if any(part.startswith('.') for part in Path(event.src_path).parts):
            return

        # Line 57: No exception handling
        self.cache._mark_workspace_for_invalidation(self.workspace_path)
```

**Risks**:
- `Path(event.src_path)` can fail for invalid paths
- `_mark_workspace_for_invalidation` could raise
- watchdog stops delivering events after exception
- Cache invalidation stops working silently

**Solution**: Add comprehensive exception handling
```python
def on_any_event(self, event: FileSystemEvent) -> None:
    try:
        if event.is_directory:
            return

        # Ignore hidden files
        try:
            if any(part.startswith('.') for part in Path(event.src_path).parts):
                return
        except Exception as e:
            logger.warning("path_parsing_error", path=event.src_path, error=str(e))
            return

        # Invalidate cache
        try:
            self.cache._mark_workspace_for_invalidation(self.workspace_path)
        except Exception as e:
            logger.error("cache_invalidation_error", workspace=self.workspace_path, error=str(e))

    except Exception as e:
        # Catch-all to prevent watchdog from stopping
        logger.error("file_system_event_error", event=event, error=str(e))
```

**Files to Modify**:
- `app/core/context/context_cache.py:32-57` - Add exception handling

---

### 4. LRU Cache Eviction Not Thread/Async Safe
**Severity**: HIGH
**Impact**: Cache size tracking can become inconsistent
**Location**: `app/core/context/context_cache.py:379-426`

**Problem**:
```python
async def set(self, cache_key: str, value: Any, cache_type: str, workspace_path: Optional[str] = None):
    # Line 405-415: Eviction logic with race conditions
    while len(self._cache) >= self._max_entries and self._cache:
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        removed = self._cache.pop(oldest_key)
        self._current_size_bytes -= removed.size_bytes  # RACE CONDITION
```

Two concurrent `set` calls can:
1. Both check `len(self._cache) >= self._max_entries` (true)
2. Both get the oldest_key (same key!)
3. First pop succeeds, second pop raises KeyError
4. Size tracking becomes inconsistent

**Solution**: Add lock around eviction
```python
async def set(self, cache_key: str, ...):
    async with self._cache_lock:  # NEW
        # Evict if needed
        while len(self._cache) >= self._max_entries and self._cache:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            removed = self._cache.pop(oldest_key, None)  # Use pop with default
            if removed:
                self._current_size_bytes -= removed.size_bytes
```

**Files to Modify**:
- `app/core/context/context_cache.py` - Add cache_lock for eviction

---

## ⚠️ Moderate Issues

### 5. Missing Resource Cleanup in FileDiscoverer
**Severity**: MODERATE
**Impact**: File handles may leak on exceptions
**Location**: `app/core/context/file_discoverer.py`

**Problem**: If Gemini API call fails mid-discovery, temporary file handles aren't guaranteed to be closed.

**Solution**: Use `async with` or ensure cleanup in `finally` blocks.

---

### 6. No Timeout on Gemini API Calls
**Severity**: MODERATE
**Impact**: Hung requests can block indefinitely
**Location**: All Gemini API interactions

**Recommendation**: Add timeout wrapper:
```python
async def with_timeout(coro, timeout_seconds=30):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise LLMError("Gemini API timeout")
```

---

### 7. Singleton Double-Check Locking Pattern Correct But Can Be Improved
**Severity**: LOW
**Impact**: Minor efficiency issue, not a correctness issue
**Location**: All `get_*()` singleton functions

**Current Pattern (Correct)**:
```python
def get_context_gateway() -> ContextGateway:
    global _gateway
    if _gateway is not None:  # Fast path
        return _gateway

    with _gateway_lock:  # Slow path
        if _gateway is None:
            _gateway = ContextGateway()
    return _gateway
```

**This is correct!** The threading.Lock prevents race conditions during initialization.

---

### 8. Memory Cleanup for Long-Running Sessions
**Severity**: LOW
**Impact**: Memory can grow over time
**Location**: `app/core/context/relevance_tracker.py`

**Recommendation**: Add LRU eviction for old sessions
```python
# Current: Sessions never removed
self._sessions: Dict[str, ContextUsageSession] = {}

# Better: Use MaxDict or implement LRU
from collections import OrderedDict
self._sessions = OrderedDict()  # Limit to 100 most recent
```

---

### 9. No Rate Limiting on Gemini API Calls
**Severity**: LOW (depends on usage)
**Impact**: Could hit API rate limits
**Location**: All Gemini interactions

**Recommendation**: Add rate limiter:
```python
from app.core.rate_limiter import RateLimiter

gemini_limiter = RateLimiter(
    max_calls=100,  # calls per minute
    window_seconds=60
)

async def call_gemini(...):
    await gemini_limiter.acquire()
    return await genai.generate(...)
```

---

## 📊 Priority Matrix

| Issue | Severity | Likelihood | Impact | Priority |
|-------|----------|------------|--------|----------|
| 1. Thundering Herd | Critical | High | Cost 10x | P0 |
| 2. Async Lock Missing | Critical | High | Data corruption | P0 |
| 3. Watchdog Exceptions | High | Medium | Silent failure | P1 |
| 4. Cache Eviction Race | High | Medium | Inconsistency | P1 |
| 5. File Handle Leaks | Moderate | Low | Slow leak | P2 |
| 6. Gemini Timeouts | Moderate | Medium | Hung requests | P2 |
| 7. Singleton Pattern | Low | N/A | None | P3 |
| 8. Session Memory | Low | Low | Slow growth | P3 |
| 9. Rate Limiting | Low | Low | API errors | P3 |

---

## 🛠️ Implementation Plan

### Phase 1: Critical Fixes (P0) - Immediate
**Timeline**: 1-2 hours

1. ✅ Add thundering herd protection to cache
2. ✅ Add asyncio.Lock to async state mutations
3. ✅ Test under concurrent load

**Files to modify**:
- `app/core/context/context_cache.py`
- `app/core/context/relevance_tracker.py`
- `app/core/context/gateway_metrics.py`

### Phase 2: High Priority (P1) - Same Day
**Timeline**: 2-3 hours

1. ✅ Add exception handling to watchdog
2. ✅ Fix cache eviction race condition
3. ✅ Add unit tests for concurrency

**Files to modify**:
- `app/core/context/context_cache.py`

### Phase 3: Moderate Priority (P2) - Next Sprint
**Timeline**: 1 day

1. ✅ Add resource cleanup guarantees
2. ✅ Add timeout wrappers
3. ✅ Add integration tests

### Phase 4: Low Priority (P3) - Future
**Timeline**: As needed

1. Session memory management
2. Rate limiting
3. Performance optimization

---

## ✅ Circuit Breaker Implementation - Already Correct!

**Analysis Result**: The circuit breaker implementation is **excellent** ✅

```python
class CircuitBreaker:
    def __init__(self, ...):
        self._lock = threading.Lock()  # ✓ Correct

    def _record_success(self):
        with self._lock:  # ✓ All state mutations protected
            self._last_success_time = datetime.now()
            self._success_count += 1
```

**Verified**:
- ✅ All state mutations use `with self._lock:`
- ✅ threading.Lock is correct (not asyncio.Lock) because it's accessed from multiple asyncio tasks
- ✅ No race conditions in state transitions
- ✅ Proper exception handling

**No changes needed for circuit breaker!**

---

## 🧪 Testing Strategy

### Concurrency Tests Needed
```python
async def test_thundering_herd():
    """Test that only one request regenerates expired cache."""
    cache = ContextCache()
    call_count = 0

    async def expensive_generator():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)
        return "result"

    # 10 concurrent requests for same expired key
    tasks = [
        cache.get_or_generate("key1", expensive_generator)
        for _ in range(10)
    ]
    results = await asyncio.gather(*tasks)

    # Should only call generator once
    assert call_count == 1
    assert all(r == "result" for r in results)
```

### Race Condition Tests
```python
async def test_cache_stats_race():
    """Test that cache stats are correctly incremented under concurrency."""
    cache = ContextCache()

    async def increment_hit():
        await cache._track_cache_hit_async("test_type")

    # 100 concurrent increments
    await asyncio.gather(*[increment_hit() for _ in range(100)])

    # Should be exactly 100
    assert cache._cache_stats["hits"]["test_type"] == 100
```

---

## 📚 References

**Analysis Tools Used**:
- ✅ Enhanced Context Preparation (newly integrated)
- ✅ Chain of Verification reasoning framework
- ✅ Code pattern analysis

**Patterns Applied**:
- Double-checked locking (already correct in singletons)
- Pending request tracking (new for cache)
- asyncio.Lock for async mutations (new)
- Exception boundaries (new for watchdog)

---

## 📝 Summary

**Good News** ✅:
- Circuit breakers are perfectly implemented
- Singleton patterns are correct
- Error handling is mostly good
- Recent integrations are solid

**Needs Attention** ⚠️:
- Cache thundering herd (P0)
- Async state mutation locks (P0)
- Watchdog exception handling (P1)
- Cache eviction races (P1)

**Estimated Total Effort**: 1-2 days for critical + high priority fixes

**Risk Level After Fixes**: LOW - System will be production-hardened

---

**Status**: ANALYSIS COMPLETE - READY FOR IMPLEMENTATION

**Next Step**: Implement Phase 1 (P0) fixes immediately
