"""
Context Cache System for Intelligent Caching

Implements intelligent caching with:
- Query similarity-based cache keys
- Workspace fingerprint for invalidation
- Separate TTL handling for different cache types
- Stale cache fallback for resilience
"""

import asyncio
import hashlib
import re
import threading
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..metrics import (
    record_context_cache_access,
    record_context_cache_invalidation,
    update_context_cache_stats,
)
from ..settings import get_settings
from .enhanced_models import CacheEntry

logger = structlog.get_logger("context_cache")


class WorkspaceChangeHandler(FileSystemEventHandler):
    """File system event handler for workspace change detection."""

    def __init__(self, cache: "ContextCache", workspace_path: str):
        self.cache = cache
        self.workspace_path = workspace_path
        super().__init__()

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Handle any file system event with comprehensive exception handling."""
        try:
            if event.is_directory:
                return

            # Ignore hidden files and common temporary files
            try:
                path = Path(event.src_path)
                if any(part.startswith(".") for part in path.parts):
                    return
            except Exception as e:
                logger.warning(
                    "path_parsing_error",
                    path=event.src_path,
                    error=str(e),
                    workspace=self.workspace_path,
                )
                return

            # Invalidate cache for this workspace
            try:
                logger.debug(
                    "workspace_file_changed",
                    path=event.src_path,
                    event_type=event.event_type,
                    workspace=self.workspace_path,
                )
                # Use asyncio-safe method to schedule invalidation
                self.cache._mark_workspace_for_invalidation(self.workspace_path)
            except Exception as e:
                logger.error(
                    "cache_invalidation_error",
                    workspace=self.workspace_path,
                    path=event.src_path,
                    error=str(e),
                )

        except Exception as e:
            # Catch-all to prevent watchdog from stopping
            logger.error(
                "file_system_event_error",
                event=str(event),
                workspace=self.workspace_path,
                error=str(e),
            )


class ContextCache:
    """
    Intelligent context cache with TTL management.

    Features:
    - Query similarity-based cache keys
    - Workspace fingerprint for invalidation
    - Separate TTL for different cache types
    - Stale cache fallback for resilience
    - File system watching for automatic invalidation
    """

    def __init__(self, ttl_settings: dict[str, int] | None = None):
        """
        Initialize context cache.

        Args:
            ttl_settings: Optional TTL settings by cache type.
                         Defaults to settings from configuration.
        """
        settings = get_settings()

        # TTL settings (in seconds)
        self._ttl_settings = ttl_settings or {
            "query_analysis": settings.cache_query_analysis_ttl,
            "file_discovery": settings.cache_file_discovery_ttl,
            "documentation": settings.cache_documentation_ttl,
        }

        # Cache storage
        self._cache: dict[str, CacheEntry] = {}

        # Workspace fingerprints for invalidation
        self._workspace_fingerprints: dict[str, str] = {}

        # File system watchers
        self._observers: dict[str, Observer] = {}

        # Workspaces marked for invalidation (needs lock protection)
        # Using threading.Lock because accessed from watchdog thread (not async)
        self._invalidation_queue: list[str] = []
        self._invalidation_lock = threading.Lock()  # Protects invalidation queue

        # Thundering herd protection: Prevent multiple concurrent regenerations
        self._pending_regenerations: dict[str, asyncio.Lock] = {}
        self._pending_lock = asyncio.Lock()

        # Async-safe locks for state mutations
        self._cache_lock = asyncio.Lock()  # For cache operations
        self._stats_lock = asyncio.Lock()  # For stats updates

        # Cache size limits
        self._max_entries = settings.cache_max_entries
        self._max_size_bytes = settings.cache_max_size_mb * 1024 * 1024
        self._current_size_bytes = 0

        # Feature flags
        self._enable_stale_fallback = settings.enable_stale_cache_fallback

        # Cache effectiveness tracking
        self._cache_stats = {
            "hits": {},  # hits by cache_type
            "misses": {},  # misses by cache_type
            "stale_hits": {},  # stale hits by cache_type
            "tokens_saved": {},  # tokens saved by cache_type
            "total_tokens_saved": 0,
            "invalidations": {
                "workspace_change": 0,
                "ttl_expired": 0,
                "size_limit": 0,
                "manual": 0,
            },
        }

        logger.info(
            "context_cache_initialized",
            ttl_settings=self._ttl_settings,
            max_entries=self._max_entries,
            max_size_mb=settings.cache_max_size_mb,
            stale_fallback=self._enable_stale_fallback,
        )

    def generate_cache_key(
        self, query: str, workspace_path: str | None = None, cache_type: str = "query_analysis"
    ) -> str:
        """
        Generate cache key based on query similarity and workspace.

        Args:
            query: The user query
            workspace_path: Optional workspace path
            cache_type: Type of cache entry

        Returns:
            Cache key string
        """
        # Compute query similarity hash
        query_hash = self._compute_query_similarity_hash(query)

        # Compute workspace fingerprint if provided
        workspace_hash = ""
        if workspace_path:
            workspace_hash = self._compute_workspace_fingerprint(workspace_path)

        # Combine into cache key
        cache_key = f"{cache_type}:{query_hash}:{workspace_hash}"
        return cache_key

    def _compute_query_similarity_hash(self, query: str) -> str:
        """
        Compute similarity-based hash for query.

        Uses normalized query text to allow similar queries to share cache.

        Args:
            query: The user query

        Returns:
            Hash string representing query similarity
        """
        # Normalize query for similarity matching
        normalized = query.lower().strip()

        # Remove common variations that don't affect intent
        normalized = re.sub(r"\s+", " ", normalized)  # Normalize whitespace
        normalized = re.sub(r"[^\w\s]", "", normalized)  # Remove punctuation

        # Extract key terms (simple keyword extraction)
        # This allows similar queries to share cache
        words = normalized.split()

        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by"}
        key_terms = [w for w in words if w not in stop_words]

        # Sort terms for order-independence
        key_terms.sort()

        # Create hash from key terms
        term_string = " ".join(key_terms)
        query_hash = hashlib.sha256(term_string.encode()).hexdigest()[:16]

        return query_hash

    def _compute_workspace_fingerprint(self, workspace_path: str) -> str:
        """
        Compute workspace fingerprint for cache invalidation.

        Uses file modification times and structure to detect changes.

        Args:
            workspace_path: Path to workspace

        Returns:
            Fingerprint hash string
        """
        # Check if we have a cached fingerprint
        if workspace_path in self._workspace_fingerprints:
            return self._workspace_fingerprints[workspace_path]

        try:
            workspace = Path(workspace_path)
            if not workspace.exists():
                return "nonexistent"

            # Collect file modification times for key files
            # Focus on source files to avoid cache invalidation from logs, etc.
            source_extensions = {".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h"}

            file_info = []
            for ext in source_extensions:
                for file_path in workspace.rglob(f"*{ext}"):
                    # Skip hidden directories and common ignore patterns
                    if any(part.startswith(".") for part in file_path.parts):
                        continue
                    if "node_modules" in file_path.parts or "__pycache__" in file_path.parts:
                        continue

                    try:
                        stat = file_path.stat()
                        file_info.append(f"{file_path.name}:{stat.st_mtime}")
                    except (OSError, PermissionError):
                        continue

            # Create fingerprint from file info
            if not file_info:
                # No source files found, use directory mtime
                fingerprint = f"empty:{workspace.stat().st_mtime}"
            else:
                # Sort for consistency
                file_info.sort()
                # Take first 100 files to avoid huge fingerprints
                fingerprint_data = "|".join(file_info[:100])
                fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

            # Cache the fingerprint
            self._workspace_fingerprints[workspace_path] = fingerprint

            return fingerprint

        except Exception as e:
            logger.warning("workspace_fingerprint_failed", workspace=workspace_path, error=str(e))
            return "error"

    async def get(self, cache_key: str, allow_stale: bool = False) -> CacheEntry | None:
        """
        Get entry from cache.

        Args:
            cache_key: Cache key to lookup
            allow_stale: If True, return expired entries (for fallback)

        Returns:
            CacheEntry if found and valid, None otherwise
        """
        # Process any pending invalidations
        self._process_invalidation_queue()

        if cache_key not in self._cache:
            # Track miss for unknown type (we don't know what was requested)
            await self._track_cache_miss_async("unknown")

            logger.debug("cache_miss", cache_key=cache_key)
            return None

        entry = self._cache[cache_key]
        cache_type = entry.cache_type

        # Check if expired
        if entry.is_expired:
            if allow_stale and self._enable_stale_fallback:
                # Track stale hit
                await self._track_cache_hit_async(
                    cache_type, is_stale=True, age_seconds=entry.age.total_seconds()
                )

                logger.info(
                    "cache_stale_hit",
                    cache_key=cache_key,
                    age_seconds=entry.age.total_seconds(),
                    cache_type=entry.cache_type,
                )
                return entry
            else:
                # Track miss (expired)
                await self._track_cache_miss_async(cache_type)

                logger.debug(
                    "cache_expired", cache_key=cache_key, age_seconds=entry.age.total_seconds()
                )
                # Remove expired entry
                self._remove_entry(cache_key)
                self._track_invalidation("ttl_expired", 1)
                return None

        # Track hit
        await self._track_cache_hit_async(
            cache_type, is_stale=False, age_seconds=entry.age.total_seconds()
        )

        logger.info(
            "cache_hit",
            cache_key=cache_key,
            age_seconds=entry.age.total_seconds(),
            cache_type=entry.cache_type,
        )
        return entry

    async def get_or_generate(
        self,
        cache_key: str,
        cache_type: str,
        generator_func: Callable[[], Awaitable[Any]],
        workspace_path: str | None = None,
        allow_stale: bool = False,
    ) -> Any:
        """
        Get from cache or generate if missing/expired (with thundering herd protection).

        This method ensures that only one concurrent request regenerates an expired
        or missing cache entry. Other concurrent requests wait for the first to complete.

        Args:
            cache_key: Cache key to lookup
            cache_type: Type of cache entry (for TTL selection)
            generator_func: Async function to generate value if not cached
            workspace_path: Optional workspace path for fingerprinting
            allow_stale: If True, return expired entries while regenerating

        Returns:
            Cached or freshly generated value
        """
        # First, check cache without lock (fast path)
        entry = await self.get(cache_key, allow_stale=allow_stale)
        if entry and not entry.is_expired:
            return entry.value

        # Cache miss or expired - acquire lock for regeneration
        async with self._pending_lock:
            # Create lock for this specific cache key if it doesn't exist
            if cache_key not in self._pending_regenerations:
                self._pending_regenerations[cache_key] = asyncio.Lock()
            key_lock = self._pending_regenerations[cache_key]

        # Only first request generates, others wait
        async with key_lock:
            # Double-check cache (might have been populated by another request)
            entry = await self.get(cache_key, allow_stale=False)
            if entry and not entry.is_expired:
                # Another request already generated it
                logger.debug(
                    "cache_regeneration_avoided", cache_key=cache_key, cache_type=cache_type
                )
                return entry.value

            # Generate new value
            logger.info("cache_regenerating", cache_key=cache_key, cache_type=cache_type)

            try:
                value = await generator_func()

                # Cache the new value
                await self.set(cache_key, value, cache_type, workspace_path)

                logger.info("cache_regenerated", cache_key=cache_key, cache_type=cache_type)

                return value

            except Exception as e:
                # If generation fails and we have stale data, return it
                if allow_stale:
                    stale_entry = await self.get(cache_key, allow_stale=True)
                    if stale_entry:
                        logger.warning(
                            "cache_regeneration_failed_using_stale",
                            cache_key=cache_key,
                            cache_type=cache_type,
                            error=str(e),
                        )
                        return stale_entry.value

                # No stale data or stale not allowed, raise the error
                logger.error(
                    "cache_regeneration_failed",
                    cache_key=cache_key,
                    cache_type=cache_type,
                    error=str(e),
                )
                raise

            finally:
                # Cleanup: Remove the lock for this key
                async with self._pending_lock:
                    if cache_key in self._pending_regenerations:
                        del self._pending_regenerations[cache_key]

    async def set(
        self, cache_key: str, value: Any, cache_type: str, workspace_path: str | None = None
    ) -> None:
        """
        Store entry in cache.

        Args:
            cache_key: Cache key
            value: Value to cache
            cache_type: Type of cache entry (for TTL selection)
            workspace_path: Optional workspace path for fingerprinting
        """
        # Get TTL for this cache type
        ttl_seconds = self._ttl_settings.get(cache_type, 3600)

        # Compute workspace fingerprint
        workspace_fingerprint = ""
        if workspace_path:
            workspace_fingerprint = self._compute_workspace_fingerprint(workspace_path)

        # Extract query hash from cache key
        parts = cache_key.split(":")
        query_hash = parts[1] if len(parts) > 1 else ""

        # Create cache entry
        entry = CacheEntry(
            value=value,
            created_at=datetime.now(),
            ttl_seconds=ttl_seconds,
            cache_type=cache_type,
            workspace_fingerprint=workspace_fingerprint,
            query_hash=query_hash,
        )

        # Check cache size limits
        await self._enforce_size_limits()

        # Store entry
        self._cache[cache_key] = entry

        # Estimate entry size (rough approximation)
        entry_size = len(str(value))
        self._current_size_bytes += entry_size

        logger.info(
            "cache_set",
            cache_key=cache_key,
            cache_type=cache_type,
            ttl_seconds=ttl_seconds,
            size_bytes=entry_size,
            total_entries=len(self._cache),
        )

        # Start watching workspace if not already watching
        if workspace_path and workspace_path not in self._observers:
            self._start_watching_workspace(workspace_path)

    async def invalidate_workspace(self, workspace_path: str) -> int:
        """
        Invalidate all cache entries for a workspace.

        Args:
            workspace_path: Path to workspace

        Returns:
            Number of entries invalidated
        """
        # Recompute workspace fingerprint
        old_fingerprint = self._workspace_fingerprints.get(workspace_path)

        # Clear cached fingerprint to force recomputation
        if workspace_path in self._workspace_fingerprints:
            del self._workspace_fingerprints[workspace_path]

        new_fingerprint = self._compute_workspace_fingerprint(workspace_path)

        # Find and remove entries with old fingerprint
        keys_to_remove = []
        for key, entry in self._cache.items():
            if entry.workspace_fingerprint == old_fingerprint:
                keys_to_remove.append(key)

        # Remove entries
        for key in keys_to_remove:
            self._remove_entry(key)

        # Track invalidation
        if keys_to_remove:
            self._track_invalidation("workspace_change", len(keys_to_remove))

        logger.info(
            "workspace_invalidated",
            workspace=workspace_path,
            entries_removed=len(keys_to_remove),
            old_fingerprint=old_fingerprint,
            new_fingerprint=new_fingerprint,
        )

        return len(keys_to_remove)

    def _mark_workspace_for_invalidation(self, workspace_path: str) -> None:
        """
        Mark workspace for invalidation (thread-safe with lock protection).

        Args:
            workspace_path: Path to workspace
        """
        with self._invalidation_lock:
            if workspace_path not in self._invalidation_queue:
                self._invalidation_queue.append(workspace_path)

    def _process_invalidation_queue(self) -> None:
        """Process pending workspace invalidations (thread-safe)."""
        while True:
            # Pop one item with lock held (brief critical section)
            with self._invalidation_lock:
                if not self._invalidation_queue:
                    break
                workspace_path = self._invalidation_queue.pop(0)

            # Process outside lock (longer operation)
            try:
                # Compute new fingerprint
                if workspace_path in self._workspace_fingerprints:
                    del self._workspace_fingerprints[workspace_path]
                self._compute_workspace_fingerprint(workspace_path)
            except Exception as e:
                logger.warning(
                    "invalidation_queue_processing_failed", workspace=workspace_path, error=str(e)
                )

    def _start_watching_workspace(self, workspace_path: str) -> None:
        """
        Start watching workspace for file changes.

        Args:
            workspace_path: Path to workspace
        """
        try:
            workspace = Path(workspace_path)
            if not workspace.exists() or not workspace.is_dir():
                return

            # Create event handler
            event_handler = WorkspaceChangeHandler(self, workspace_path)

            # Create observer
            observer = Observer()
            observer.schedule(event_handler, str(workspace), recursive=True)
            observer.start()

            self._observers[workspace_path] = observer

            logger.info("workspace_watching_started", workspace=workspace_path)

        except Exception as e:
            logger.warning("workspace_watching_failed", workspace=workspace_path, error=str(e))

    def _stop_watching_workspace(self, workspace_path: str) -> None:
        """
        Stop watching workspace.

        Args:
            workspace_path: Path to workspace
        """
        if workspace_path in self._observers:
            try:
                observer = self._observers[workspace_path]
                observer.stop()
                observer.join(timeout=1.0)
                del self._observers[workspace_path]
                logger.info("workspace_watching_stopped", workspace=workspace_path)
            except Exception as e:
                logger.warning(
                    "workspace_watching_stop_failed", workspace=workspace_path, error=str(e)
                )

    def _remove_entry(self, cache_key: str) -> None:
        """
        Remove entry from cache.

        Args:
            cache_key: Cache key to remove
        """
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            # Estimate size
            entry_size = len(str(entry.value))
            self._current_size_bytes -= entry_size
            del self._cache[cache_key]

    async def _enforce_size_limits(self) -> None:
        """Enforce cache size limits by removing oldest entries (async-safe)."""
        async with self._cache_lock:
            # Check entry count limit
            if len(self._cache) >= self._max_entries:
                # Remove oldest 10% of entries
                num_to_remove = max(1, len(self._cache) // 10)

                # Sort by creation time
                sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].created_at)

                for i in range(num_to_remove):
                    if i < len(sorted_entries):  # Safety check
                        cache_key = sorted_entries[i][0]
                        # Check key still exists (another task might have removed it)
                        if cache_key in self._cache:
                            self._remove_entry(cache_key)

                # Track invalidation
                self._track_invalidation("size_limit", num_to_remove)

                logger.info(
                    "cache_size_limit_enforced", entries_removed=num_to_remove, reason="max_entries"
                )

            # Check size limit
            if self._current_size_bytes >= self._max_size_bytes:
                # Remove oldest entries until under limit
                sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].created_at)

                removed = 0
                for cache_key, entry in sorted_entries:
                    if self._current_size_bytes < self._max_size_bytes * 0.8:
                        break
                    # Check key still exists
                    if cache_key in self._cache:
                        self._remove_entry(cache_key)
                        removed += 1

                # Track invalidation
                if removed > 0:
                    self._track_invalidation("size_limit", removed)

                logger.info(
                    "cache_size_limit_enforced", entries_removed=removed, reason="max_size_bytes"
                )

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics including effectiveness metrics.

        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self._cache)
        entries_by_type = {}
        expired_count = 0

        for entry in self._cache.values():
            cache_type = entry.cache_type
            entries_by_type[cache_type] = entries_by_type.get(cache_type, 0) + 1
            if entry.is_expired:
                expired_count += 1

        # Calculate hit rates
        hit_rates = {}
        for cache_type in self._ttl_settings.keys():
            hits = self._cache_stats["hits"].get(cache_type, 0)
            misses = self._cache_stats["misses"].get(cache_type, 0)
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0
            hit_rates[cache_type] = hit_rate

        # Update Prometheus metrics
        update_context_cache_stats(
            total_entries=total_entries,
            entries_by_type=entries_by_type,
            size_bytes=self._current_size_bytes,
            hit_rates=hit_rates,
        )

        return {
            "total_entries": total_entries,
            "entries_by_type": entries_by_type,
            "expired_entries": expired_count,
            "size_bytes": self._current_size_bytes,
            "size_mb": self._current_size_bytes / (1024 * 1024),
            "watched_workspaces": len(self._observers),
            "max_entries": self._max_entries,
            "max_size_mb": self._max_size_bytes / (1024 * 1024),
            # Cache effectiveness metrics
            "cache_hits": dict(self._cache_stats["hits"]),
            "cache_misses": dict(self._cache_stats["misses"]),
            "stale_hits": dict(self._cache_stats["stale_hits"]),
            "hit_rates": hit_rates,
            "tokens_saved": dict(self._cache_stats["tokens_saved"]),
            "total_tokens_saved": self._cache_stats["total_tokens_saved"],
            "invalidations": dict(self._cache_stats["invalidations"]),
        }

    def get_effectiveness_dashboard(self) -> dict[str, Any]:
        """
        Get cache effectiveness metrics formatted for dashboard display.

        Returns:
            Dictionary with dashboard-ready cache effectiveness metrics
        """
        stats = self.get_stats()

        # Calculate overall metrics
        total_hits = sum(stats["cache_hits"].values())
        total_misses = sum(stats["cache_misses"].values())
        total_requests = total_hits + total_misses
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

        # Calculate average tokens saved per hit
        avg_tokens_per_hit = stats["total_tokens_saved"] / total_hits if total_hits > 0 else 0

        # Format for dashboard
        dashboard = {
            "summary": {
                "overall_hit_rate": f"{overall_hit_rate:.1%}",
                "total_requests": total_requests,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "total_tokens_saved": f"{stats['total_tokens_saved']:,}",
                "avg_tokens_per_hit": f"{avg_tokens_per_hit:.0f}",
            },
            "by_cache_type": {},
            "cache_health": {
                "size_utilization": f"{(stats['size_bytes'] / self._max_size_bytes * 100):.1f}%",
                "entry_utilization": f"{(stats['total_entries'] / self._max_entries * 100):.1f}%",
                "expired_entries": stats["expired_entries"],
                "watched_workspaces": stats["watched_workspaces"],
            },
            "invalidations": {
                "workspace_changes": stats["invalidations"].get("workspace_change", 0),
                "ttl_expired": stats["invalidations"].get("ttl_expired", 0),
                "size_limits": stats["invalidations"].get("size_limit", 0),
                "manual": stats["invalidations"].get("manual", 0),
            },
        }

        # Add per-cache-type metrics
        for cache_type in self._ttl_settings.keys():
            hits = stats["cache_hits"].get(cache_type, 0)
            misses = stats["cache_misses"].get(cache_type, 0)
            stale_hits = stats["stale_hits"].get(cache_type, 0)
            tokens_saved = stats["tokens_saved"].get(cache_type, 0)
            hit_rate = stats["hit_rates"].get(cache_type, 0.0)

            dashboard["by_cache_type"][cache_type] = {
                "hit_rate": f"{hit_rate:.1%}",
                "hits": hits,
                "misses": misses,
                "stale_hits": stale_hits,
                "tokens_saved": f"{tokens_saved:,}",
                "ttl_seconds": self._ttl_settings[cache_type],
            }

        return dashboard

    def clear(self) -> None:
        """Clear all cache entries."""
        num_entries = len(self._cache)
        self._cache.clear()
        self._workspace_fingerprints.clear()
        self._current_size_bytes = 0

        # Track manual invalidation
        if num_entries > 0:
            self._track_invalidation("manual", num_entries)

        logger.info("cache_cleared", entries_removed=num_entries)

    async def _track_cache_hit_async(
        self, cache_type: str, is_stale: bool, age_seconds: float
    ) -> None:
        """
        Track a cache hit for effectiveness metrics (async-safe).

        Args:
            cache_type: Type of cache entry
            is_stale: Whether this was a stale cache hit
            age_seconds: Age of cache entry in seconds
        """
        async with self._stats_lock:
            if is_stale:
                self._cache_stats["stale_hits"][cache_type] = (
                    self._cache_stats["stale_hits"].get(cache_type, 0) + 1
                )
            else:
                self._cache_stats["hits"][cache_type] = (
                    self._cache_stats["hits"].get(cache_type, 0) + 1
                )

            # Estimate tokens saved based on cache type
            # These are rough estimates based on typical Gemini API usage
            tokens_saved_estimates = {
                "query_analysis": 500,  # Query analysis typically uses ~500 tokens
                "file_discovery": 2000,  # File discovery with summaries ~2000 tokens
                "documentation": 1500,  # Documentation search ~1500 tokens
            }
            tokens_saved = tokens_saved_estimates.get(cache_type, 1000)

            self._cache_stats["tokens_saved"][cache_type] = (
                self._cache_stats["tokens_saved"].get(cache_type, 0) + tokens_saved
            )
            self._cache_stats["total_tokens_saved"] += tokens_saved

        # Record to Prometheus
        record_context_cache_access(
            cache_type=cache_type,
            hit=True,
            tokens_saved=tokens_saved,
            cache_age_seconds=age_seconds,
            is_stale=is_stale,
        )

    async def _track_cache_miss_async(self, cache_type: str) -> None:
        """
        Track a cache miss for effectiveness metrics (async-safe).

        Args:
            cache_type: Type of cache entry
        """
        async with self._stats_lock:
            self._cache_stats["misses"][cache_type] = (
                self._cache_stats["misses"].get(cache_type, 0) + 1
            )

        # Record to Prometheus
        record_context_cache_access(
            cache_type=cache_type, hit=False, tokens_saved=0, cache_age_seconds=0.0, is_stale=False
        )

    def _track_invalidation(self, reason: str, count: int) -> None:
        """
        Track cache invalidation for effectiveness metrics.

        Args:
            reason: Reason for invalidation
            count: Number of entries invalidated
        """
        self._cache_stats["invalidations"][reason] = (
            self._cache_stats["invalidations"].get(reason, 0) + count
        )

        # Record to Prometheus
        record_context_cache_invalidation(reason=reason, count=count)

    def shutdown(self) -> None:
        """Shutdown cache and stop all watchers."""
        # Stop all observers
        for workspace_path in list(self._observers.keys()):
            self._stop_watching_workspace(workspace_path)

        logger.info("context_cache_shutdown")


# Global singleton
_cache: ContextCache | None = None


def get_context_cache() -> ContextCache:
    """Get the global context cache singleton."""
    global _cache
    if _cache is None:
        _cache = ContextCache()
    return _cache


def reset_context_cache() -> None:
    """Reset context cache singleton (useful for testing)."""
    global _cache
    if _cache is not None:
        _cache.shutdown()
    _cache = None
