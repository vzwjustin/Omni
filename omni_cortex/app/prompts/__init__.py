"""
Prompt Templates for Omni-Cortex

Provides reusable prompt templates and output parsers.
"""

from .parsers import (
    FrameworkSelection,
    ReasoningOutput,
    framework_parser,
    reasoning_parser,
)
from .templates import (
    CODE_GENERATION_TEMPLATE,
    FRAMEWORK_SELECTION_TEMPLATE,
    REASONING_TEMPLATE,
)

__all__ = [
    "FRAMEWORK_SELECTION_TEMPLATE",
    "REASONING_TEMPLATE",
    "CODE_GENERATION_TEMPLATE",
    "ReasoningOutput",
    "FrameworkSelection",
    "reasoning_parser",
    "framework_parser",
]
