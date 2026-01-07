"""
Output Parsers for Omni-Cortex

Structured output parsing using Pydantic models.
"""

from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class ReasoningOutput(BaseModel):
    """Structured output for reasoning results."""
    framework_used: str = Field(description="The reasoning framework that was used")
    thought_process: str = Field(description="The step-by-step reasoning")
    answer: str = Field(description="The final answer or solution")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
    code: Optional[str] = Field(default=None, description="Generated code if applicable")


class FrameworkSelection(BaseModel):
    """Structured output for framework selection."""
    selected_framework: str = Field(description="The chosen framework")
    reasoning: str = Field(description="Why this framework was selected")
    task_type: str = Field(description="Identified task type")
    estimated_complexity: float = Field(description="Complexity estimate 0-1", ge=0, le=1)


# Parser instances
reasoning_parser = PydanticOutputParser(pydantic_object=ReasoningOutput)
framework_parser = PydanticOutputParser(pydantic_object=FrameworkSelection)
