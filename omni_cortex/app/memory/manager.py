"""
Memory Manager - Thread-Safe Memory Store

Provides LRU-evicting memory store with async-safe access.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from .omni_memory import OmniCortexMemory
from ..core.constants import LIMITS

# Global memory store (keyed by thread_id) with simple LRU eviction
_memory_store: OrderedDict[str, OmniCortexMemory] = OrderedDict()
_memory_store_lock = asyncio.Lock()

MAX_MEMORY_THREADS = LIMITS.MAX_MEMORY_THREADS


async def get_memory(thread_id: str) -> OmniCortexMemory:
    """Get or create memory for a thread with thread-safe access."""
    async with _memory_store_lock:
        if thread_id in _memory_store:
            _memory_store.move_to_end(thread_id)
            return _memory_store[thread_id]

        # Evict oldest if over capacity
        if len(_memory_store) >= MAX_MEMORY_THREADS:
            _memory_store.popitem(last=False)

        mem = OmniCortexMemory(thread_id)
        _memory_store[thread_id] = mem
        return mem


def get_memory_store_lock() -> asyncio.Lock:
    """Get the memory store lock for external use."""
    return _memory_store_lock


def get_memory_store() -> OrderedDict[str, OmniCortexMemory]:
    """Get the memory store for external use (use with lock)."""
    return _memory_store
