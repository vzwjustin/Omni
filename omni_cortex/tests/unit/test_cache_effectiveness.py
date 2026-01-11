"""
Unit tests for cache effectiveness tracking.

Tests cache hit/miss metrics, token savings calculation, and dashboard functionality.
"""

from datetime import datetime

import pytest

from app.core.context.context_cache import ContextCache
from app.core.context.enhanced_models import CacheEntry


@pytest.fixture
def cache():
    """Create a test cache instance."""
    ttl_settings = {
        "query_analysis": 3600,
        "file_discovery": 1800,
        "documentation": 86400,
    }
    return ContextCache(ttl_settings=ttl_settings)


@pytest.mark.asyncio
async def test_cache_hit_tracking(cache):
    """Test that cache hits are tracked correctly."""
    # Initial stats should be empty
    stats = cache.get_stats()
    assert stats["cache_hits"] == {}
    assert stats["cache_misses"] == {}

    # Simulate a cache miss
    await cache._track_cache_miss_async("query_analysis")

    stats = cache.get_stats()
    assert stats["cache_misses"]["query_analysis"] == 1
    assert stats["cache_hits"].get("query_analysis", 0) == 0

    # Simulate a cache hit
    await cache._track_cache_hit_async("query_analysis", is_stale=False, age_seconds=100.0)

    stats = cache.get_stats()
    assert stats["cache_hits"]["query_analysis"] == 1
    assert stats["cache_misses"]["query_analysis"] == 1


@pytest.mark.asyncio
async def test_token_savings_calculation(cache):
    """Test that token savings are calculated correctly."""
    # Track hits for different cache types
    await cache._track_cache_hit_async("query_analysis", is_stale=False, age_seconds=100.0)
    await cache._track_cache_hit_async("file_discovery", is_stale=False, age_seconds=200.0)
    await cache._track_cache_hit_async("documentation", is_stale=False, age_seconds=300.0)

    stats = cache.get_stats()

    # Check that tokens were saved
    assert stats["tokens_saved"]["query_analysis"] == 500  # Expected estimate
    assert stats["tokens_saved"]["file_discovery"] == 2000  # Expected estimate
    assert stats["tokens_saved"]["documentation"] == 1500  # Expected estimate
    assert stats["total_tokens_saved"] == 4000  # Sum of all


@pytest.mark.asyncio
async def test_stale_hit_tracking(cache):
    """Test that stale cache hits are tracked separately."""
    # Track regular hit
    await cache._track_cache_hit_async("query_analysis", is_stale=False, age_seconds=100.0)

    # Track stale hit
    await cache._track_cache_hit_async("query_analysis", is_stale=True, age_seconds=5000.0)

    stats = cache.get_stats()

    # Regular hits should be tracked
    assert stats["cache_hits"]["query_analysis"] == 1

    # Stale hits should be tracked separately
    assert stats["stale_hits"]["query_analysis"] == 1


def test_invalidation_tracking(cache):
    """Test that cache invalidations are tracked by reason."""
    # Track different invalidation reasons
    cache._track_invalidation("workspace_change", 5)
    cache._track_invalidation("ttl_expired", 3)
    cache._track_invalidation("size_limit", 2)
    cache._track_invalidation("manual", 1)

    stats = cache.get_stats()

    assert stats["invalidations"]["workspace_change"] == 5
    assert stats["invalidations"]["ttl_expired"] == 3
    assert stats["invalidations"]["size_limit"] == 2
    assert stats["invalidations"]["manual"] == 1


@pytest.mark.asyncio
async def test_hit_rate_calculation(cache):
    """Test that hit rates are calculated correctly."""
    # Simulate some hits and misses
    await cache._track_cache_hit_async("query_analysis", is_stale=False, age_seconds=100.0)
    await cache._track_cache_hit_async("query_analysis", is_stale=False, age_seconds=200.0)
    await cache._track_cache_hit_async("query_analysis", is_stale=False, age_seconds=300.0)
    await cache._track_cache_miss_async("query_analysis")

    stats = cache.get_stats()

    # Hit rate should be 3/4 = 0.75
    assert stats["hit_rates"]["query_analysis"] == 0.75


@pytest.mark.asyncio
async def test_effectiveness_dashboard(cache):
    """Test that effectiveness dashboard returns formatted data."""
    # Add some cache activity
    await cache._track_cache_hit_async("query_analysis", is_stale=False, age_seconds=100.0)
    await cache._track_cache_hit_async("file_discovery", is_stale=False, age_seconds=200.0)
    await cache._track_cache_miss_async("query_analysis")
    cache._track_invalidation("workspace_change", 2)

    dashboard = cache.get_effectiveness_dashboard()

    # Check summary section
    assert "summary" in dashboard
    assert "overall_hit_rate" in dashboard["summary"]
    assert "total_tokens_saved" in dashboard["summary"]

    # Check by_cache_type section
    assert "by_cache_type" in dashboard
    assert "query_analysis" in dashboard["by_cache_type"]
    assert "file_discovery" in dashboard["by_cache_type"]

    # Check cache_health section
    assert "cache_health" in dashboard
    assert "size_utilization" in dashboard["cache_health"]

    # Check invalidations section
    assert "invalidations" in dashboard
    assert dashboard["invalidations"]["workspace_changes"] == 2


def test_clear_tracks_invalidation(cache):
    """Test that clearing cache tracks manual invalidation."""
    # Add some entries
    cache._cache["key1"] = CacheEntry(
        value="test1",
        created_at=datetime.now(),
        ttl_seconds=3600,
        cache_type="query_analysis",
        workspace_fingerprint="fp1",
        query_hash="qh1"
    )
    cache._cache["key2"] = CacheEntry(
        value="test2",
        created_at=datetime.now(),
        ttl_seconds=3600,
        cache_type="file_discovery",
        workspace_fingerprint="fp2",
        query_hash="qh2"
    )

    # Clear cache
    cache.clear()

    stats = cache.get_stats()

    # Should track manual invalidation
    assert stats["invalidations"]["manual"] == 2
    assert len(cache._cache) == 0


@pytest.mark.asyncio
async def test_multiple_cache_types(cache):
    """Test tracking across multiple cache types."""
    # Track activity for all cache types
    for cache_type in ["query_analysis", "file_discovery", "documentation"]:
        await cache._track_cache_hit_async(cache_type, is_stale=False, age_seconds=100.0)
        await cache._track_cache_miss_async(cache_type)

    stats = cache.get_stats()

    # All cache types should have metrics
    for cache_type in ["query_analysis", "file_discovery", "documentation"]:
        assert stats["cache_hits"][cache_type] == 1
        assert stats["cache_misses"][cache_type] == 1
        assert stats["hit_rates"][cache_type] == 0.5


def test_zero_division_in_hit_rate(cache):
    """Test that hit rate calculation handles zero requests."""
    stats = cache.get_stats()

    # With no requests, hit rate should be 0.0
    for cache_type in ["query_analysis", "file_discovery", "documentation"]:
        assert stats["hit_rates"][cache_type] == 0.0


@pytest.mark.asyncio
async def test_dashboard_formatting(cache):
    """Test that dashboard values are properly formatted."""
    # Add some activity
    await cache._track_cache_hit_async("query_analysis", is_stale=False, age_seconds=100.0)
    await cache._track_cache_miss_async("query_analysis")

    dashboard = cache.get_effectiveness_dashboard()

    # Check that percentages are formatted
    assert "%" in dashboard["summary"]["overall_hit_rate"]
    assert "%" in dashboard["by_cache_type"]["query_analysis"]["hit_rate"]

    # Check that numbers have commas
    assert "," in dashboard["summary"]["total_tokens_saved"] or dashboard["summary"]["total_tokens_saved"] == "500"
