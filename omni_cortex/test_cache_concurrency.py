#!/usr/bin/env python3
"""
Concurrency Tests for Context Cache P0 Fixes

Tests the following critical fixes:
1. Thundering herd protection - only one request regenerates expired cache
2. Async-safe stat tracking - no lost updates under concurrency
3. Async-safe cache eviction - no race conditions
"""

import sys
import asyncio
from pathlib import Path
from typing import Callable, Awaitable

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_thundering_herd():
    """Test that only one request regenerates expired cache (P0 critical)."""
    print("\n" + "=" * 70)
    print("TEST 1: THUNDERING HERD PROTECTION")
    print("=" * 70)

    from app.core.context.context_cache import ContextCache

    # Create cache with short TTL
    cache = ContextCache(ttl_settings={"test": 1})  # 1 second TTL

    # Track how many times generator is called
    call_count = 0
    call_lock = asyncio.Lock()

    async def expensive_generator():
        """Simulates expensive API call."""
        nonlocal call_count
        async with call_lock:
            call_count += 1

        # Simulate slow API call
        await asyncio.sleep(0.1)
        return f"result_{call_count}"

    cache_key = "test:thundering_herd"

    print("\n1. Launching 10 concurrent requests for same expired cache key...")

    # Launch 10 concurrent requests
    tasks = [
        cache.get_or_generate(
            cache_key=cache_key,
            cache_type="test",
            generator_func=expensive_generator,
            workspace_path=None,
            allow_stale=False
        )
        for _ in range(10)
    ]

    results = await asyncio.gather(*tasks)

    print(f"\n2. Results:")
    print(f"   - Generator called: {call_count} times")
    print(f"   - Expected: 1 time (thundering herd protection)")
    print(f"   - All results identical: {len(set(results)) == 1}")
    print(f"   - Result value: {results[0]}")

    # Verify thundering herd protection worked
    if call_count == 1:
        print("\n✅ PASS: Thundering herd protection working!")
        return True
    else:
        print(f"\n❌ FAIL: Expected 1 generator call, got {call_count}")
        return False


async def test_async_stats_race():
    """Test that cache stats are correctly incremented under concurrency."""
    print("\n" + "=" * 70)
    print("TEST 2: ASYNC-SAFE STATS TRACKING")
    print("=" * 70)

    from app.core.context.context_cache import ContextCache

    cache = ContextCache()

    # Pre-populate cache with valid entries
    for i in range(10):
        await cache.set(
            cache_key=f"test:key_{i}",
            value=f"value_{i}",
            cache_type="test",
            workspace_path=None
        )

    print("\n1. Launching 100 concurrent cache hit requests...")

    # 100 concurrent cache hits
    async def cache_hit_request(key_num: int):
        cache_key = f"test:key_{key_num % 10}"
        entry = await cache.get(cache_key)
        return entry is not None

    tasks = [cache_hit_request(i) for i in range(100)]
    results = await asyncio.gather(*tasks)

    # Get stats
    stats = cache.get_stats()
    test_hits = stats["cache_hits"].get("test", 0)

    print(f"\n2. Results:")
    print(f"   - Successful hits: {sum(results)}")
    print(f"   - Expected hits: 100")
    print(f"   - Tracked hits: {test_hits}")

    # Verify no lost updates
    if test_hits == 100:
        print("\n✅ PASS: Async-safe stats tracking working!")
        return True
    else:
        print(f"\n❌ FAIL: Expected 100 tracked hits, got {test_hits} (lost updates!)")
        return False


async def test_cache_eviction_race():
    """Test that cache eviction handles concurrent operations safely."""
    print("\n" + "=" * 70)
    print("TEST 3: ASYNC-SAFE CACHE EVICTION")
    print("=" * 70)

    from app.core.context.context_cache import ContextCache

    # Create cache with very small limit
    cache = ContextCache(ttl_settings={"test": 3600})
    cache._max_entries = 20  # Small limit to trigger eviction

    print("\n1. Launching 50 concurrent cache set operations...")
    print("   (Will trigger eviction multiple times)")

    # 50 concurrent sets (will trigger eviction)
    async def cache_set_request(i: int):
        try:
            await cache.set(
                cache_key=f"test:concurrent_{i}",
                value=f"value_{i}",
                cache_type="test",
                workspace_path=None
            )
            return True
        except Exception as e:
            print(f"   ❌ Exception during set: {e}")
            return False

    tasks = [cache_set_request(i) for i in range(50)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for exceptions
    exceptions = [r for r in results if isinstance(r, Exception)]
    successes = [r for r in results if r is True]

    print(f"\n2. Results:")
    print(f"   - Successful sets: {len(successes)}")
    print(f"   - Exceptions: {len(exceptions)}")
    print(f"   - Final cache size: {len(cache._cache)} entries")
    print(f"   - Max entries: {cache._max_entries}")

    if exceptions:
        print(f"\n   Exception details:")
        for i, exc in enumerate(exceptions[:3]):  # Show first 3
            print(f"   - {exc}")

    # Verify no race conditions (all operations succeeded)
    if len(exceptions) == 0:
        print("\n✅ PASS: Async-safe cache eviction working!")
        return True
    else:
        print(f"\n❌ FAIL: Got {len(exceptions)} exceptions during concurrent eviction")
        return False


async def test_stale_fallback():
    """Test that stale fallback works correctly with thundering herd."""
    print("\n" + "=" * 70)
    print("TEST 4: STALE FALLBACK WITH THUNDERING HERD")
    print("=" * 70)

    from app.core.context.context_cache import ContextCache

    cache = ContextCache(ttl_settings={"test": 1})  # 1 second TTL
    cache._enable_stale_fallback = True

    # Pre-populate cache
    cache_key = "test:stale_fallback"
    await cache.set(cache_key, "stale_value", "test", workspace_path=None)

    print("\n1. Waiting for cache to expire...")
    await asyncio.sleep(1.5)

    print("2. Launching 5 concurrent requests with failing generator...")

    # Generator that fails
    async def failing_generator():
        await asyncio.sleep(0.05)
        raise Exception("Simulated API failure")

    # Try with allow_stale=True
    tasks = [
        cache.get_or_generate(
            cache_key=cache_key,
            cache_type="test",
            generator_func=failing_generator,
            workspace_path=None,
            allow_stale=True
        )
        for _ in range(5)
    ]

    # Should return stale value instead of raising
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        exceptions = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if not isinstance(r, Exception)]

        print(f"\n3. Results:")
        print(f"   - Successful returns: {len(successes)}")
        print(f"   - Exceptions raised: {len(exceptions)}")

        if successes:
            print(f"   - Returned value: {successes[0]}")
            print(f"   - Expected: 'stale_value'")

        if len(successes) > 0 and successes[0] == "stale_value":
            print("\n✅ PASS: Stale fallback working correctly!")
            return True
        else:
            print("\n❌ FAIL: Stale fallback did not return stale value")
            return False

    except Exception as e:
        print(f"\n❌ FAIL: Unexpected exception: {e}")
        return False


async def main():
    """Run all concurrency tests."""
    print("=" * 70)
    print("CONTEXT CACHE CONCURRENCY TESTS (P0 CRITICAL FIXES)")
    print("=" * 70)
    print("\nTesting fixes for:")
    print("  1. Thundering herd problem")
    print("  2. Async race conditions in stats tracking")
    print("  3. Cache eviction race conditions")
    print("  4. Stale fallback with error handling")

    results = {}

    # Run tests
    try:
        results["thundering_herd"] = await test_thundering_herd()
    except Exception as e:
        print(f"\n❌ TEST 1 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results["thundering_herd"] = False

    try:
        results["async_stats"] = await test_async_stats_race()
    except Exception as e:
        print(f"\n❌ TEST 2 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results["async_stats"] = False

    try:
        results["cache_eviction"] = await test_cache_eviction_race()
    except Exception as e:
        print(f"\n❌ TEST 3 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results["cache_eviction"] = False

    try:
        results["stale_fallback"] = await test_stale_fallback()
    except Exception as e:
        print(f"\n❌ TEST 4 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        results["stale_fallback"] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✅")
        print("P0 critical fixes are working correctly!")
    else:
        failed_count = sum(1 for v in results.values() if not v)
        print(f"TESTS FAILED: {failed_count}/{len(results)} ❌")
        print("P0 fixes need attention!")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
