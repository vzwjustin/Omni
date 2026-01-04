"""Nodes package - All cognitive framework implementations."""

from .common import (
    quiet_star,
    process_reward_model,
    optimize_prompt,
    call_deep_reasoner,
    call_fast_synthesizer,
    CODE_BLOCK_PATTERN,
    DEFAULT_DEEP_REASONING_TOKENS,
    DEFAULT_FAST_SYNTHESIS_TOKENS,
    DEFAULT_DEEP_REASONING_TEMP,
    DEFAULT_FAST_SYNTHESIS_TEMP
)

__all__ = [
    "quiet_star",
    "process_reward_model", 
    "optimize_prompt",
    "call_deep_reasoner",
    "call_fast_synthesizer",
    "CODE_BLOCK_PATTERN",
    "DEFAULT_DEEP_REASONING_TOKENS",
    "DEFAULT_FAST_SYNTHESIS_TOKENS",
    "DEFAULT_DEEP_REASONING_TEMP",
    "DEFAULT_FAST_SYNTHESIS_TEMP"
]
