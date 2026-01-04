"""
Pydantic Schemas for Omni-Cortex MCP Server

Defines request/response models for reasoning operations.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ReasoningRequest(BaseModel):
    """Input schema for reasoning requests from IDE agents."""
    
    query: str = Field(
        ...,
        description="The main task or question to reason about",
        examples=["Debug this null pointer exception", "Design a REST API for user management"]
    )
    code_snippet: Optional[str] = Field(
        default=None,
        description="Relevant code context for the task"
    )
    file_list: Optional[list[str]] = Field(
        default_factory=list,
        description="List of file paths relevant to the task"
    )
    ide_context: Optional[str] = Field(
        default=None,
        description="Additional IDE context (language, framework, project type)"
    )
    preferred_framework: Optional[str] = Field(
        default=None,
        description="Force a specific framework (e.g., 'mcts', 'debate', 'reason_flux')"
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum reasoning iterations for iterative frameworks"
    )


class ReasoningStep(BaseModel):
    """A single step in the reasoning trace."""
    
    step_number: int
    framework_node: str
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ReasoningResponse(BaseModel):
    """Output schema for reasoning results."""
    
    strategy_executed: str = Field(
        ...,
        description="Name of the framework that was executed"
    )
    reasoning_trace: list[ReasoningStep] = Field(
        default_factory=list,
        description="Step-by-step trace of the reasoning process"
    )
    final_code: Optional[str] = Field(
        default=None,
        description="Generated or modified code output"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="Natural language answer or explanation"
    )
    complexity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated complexity of the task (0=trivial, 1=extremely complex)"
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Model confidence in the solution"
    )
    tokens_used: int = Field(
        default=0,
        description="Total tokens consumed during reasoning"
    )


class FrameworkInfo(BaseModel):
    """Information about an available reasoning framework."""
    
    name: str
    category: str
    description: str
    best_for: list[str]
    complexity: str  # "low", "medium", "high"


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    version: str
    frameworks_loaded: int
    uptime_seconds: float
