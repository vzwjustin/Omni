"""
Memory Manager - Thread-Safe Memory Store

Provides LRU-evicting memory store with async-safe access.

Thread Safety Model:
--------------------
This module uses a hybrid locking strategy:

1. threading.Lock (_memory_store_lock): Protects access to the _memory_store dict.
   Used because asyncio.Lock() cannot be created at module level (no event loop yet)
   and because we need cross-thread protection for the singleton pattern.

2. The operations inside get_memory() are synchronous (dict access, OrderedDict
   operations, object instantiation), so threading.Lock is appropriate here.

3. If async operations were needed inside the lock, we would need to use
   asyncio.Lock with lazy initialization. But for this use case, threading.Lock
   works correctly and is simpler.

Note: Python's GIL provides some protection, but explicit locking ensures
correctness regardless of GIL behavior and makes the intent clear.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from ..core.constants import LIMITS
from .omni_memory import OmniCortexMemory

# Global memory store (keyed by thread_id) with simple LRU eviction
_memory_store: OrderedDict[str, OmniCortexMemory] = OrderedDict()
# threading.Lock for module-level singleton protection - works across threads
# and can be created at module load time (unlike asyncio.Lock which needs an event loop)
_memory_store_lock = threading.Lock()

MAX_MEMORY_THREADS = LIMITS.MAX_MEMORY_THREADS


async def get_memory(thread_id: str) -> OmniCortexMemory:
    """
    Get or create memory for a thread with thread-safe access.

    Uses threading.Lock (not asyncio.Lock) because:
    - All operations inside are synchronous (dict access, object creation)
    - threading.Lock can be created at module level without an event loop
    - threading.Lock provides true cross-thread protection

    The function is async for interface consistency with the rest of the codebase,
    but the critical section contains only synchronous operations.
    """
    with _memory_store_lock:
        if thread_id in _memory_store:
            _memory_store.move_to_end(thread_id)
            return _memory_store[thread_id]

        # Evict oldest if over capacity
        if len(_memory_store) >= MAX_MEMORY_THREADS:
            # Clear memory before eviction to ensure references are dropped
            evicted_id, evicted_mem = _memory_store.popitem(last=False)
            evicted_mem.clear()  # Explicit cleanup of messages and history

        mem = OmniCortexMemory(thread_id)
        _memory_store[thread_id] = mem
        return mem


def get_memory_store_lock() -> threading.Lock:
    """Get the memory store lock for external use (use with `with` statement)."""
    return _memory_store_lock


def get_memory_store() -> OrderedDict[str, OmniCortexMemory]:
    """Get the memory store for external use (use with lock)."""
    return _memory_store
