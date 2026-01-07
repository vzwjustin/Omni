"""
Prompt Templates for Omni-Cortex

Provides reusable prompt templates and output parsers.
"""

from .templates import (
    FRAMEWORK_SELECTION_TEMPLATE,
    REASONING_TEMPLATE,
    CODE_GENERATION_TEMPLATE,
)
from .parsers import (
    ReasoningOutput,
    FrameworkSelection,
    reasoning_parser,
    framework_parser,
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
