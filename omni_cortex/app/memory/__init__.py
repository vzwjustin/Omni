"""
Memory Systems for Omni-Cortex

Provides conversation memory and thread management.
"""

from .omni_memory import OmniCortexMemory
from .manager import get_memory, MAX_MEMORY_THREADS

__all__ = ["OmniCortexMemory", "get_memory", "MAX_MEMORY_THREADS"]
