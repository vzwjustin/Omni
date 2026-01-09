"""
Enhanced Data Models for Context Gateway Enhancements

New data models to support:
- Intelligent caching with TTL management
- Streaming progress events
- Multi-repository discovery
- Enhanced documentation grounding with source attribution
- Comprehensive metrics and monitoring
- Token budget management
- Advanced resilience patterns
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Callable, Union
from enum import Enum


# =============================================================================
# Cache System Models
# =============================================================================

@dataclass
class CacheEntry:
    """Entry in the context cache with TTL and metadata."""
    value: Any
    created_at: datetime
    ttl_seconds: int
    cache_type: str  # "query_analysis", "file_discovery", "documentation"
    workspace_fingerprint: str
    query_hash: str
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age(self) -> timedelta:
        """Get age of cache entry."""
        return datetime.now() - self.created_at


@dataclass
class CacheMetadata:
    """Metadata about cache usage for a context preparation."""
    cache_key: str
    cache_hit: bool
    cache_age: timedelta
    cache_type: str  # "query_analysis", "file_discovery", "documentation"
    workspace_fingerprint: str
    is_stale_fallback: bool = False  # True if served stale cache due to API failure


# =============================================================================
# Streaming Progress Models
# =============================================================================

class ProgressStatus(Enum):
    """Status values for progress events."""
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressEvent:
    """Progress event for streaming context preparation."""
    component: str  # "query_analysis", "file_discovery", "doc_search", "code_search"
    status: ProgressStatus
    progress: float  # 0.0 to 1.0
    data: Any  # Component-specific data
    timestamp: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[float] = None  # seconds remaining
    message: Optional[str] = None  # Human-readable status message


# =============================================================================
# Multi-Repository Models
# =============================================================================

@dataclass
class RepoInfo:
    """Information about a discovered repository."""
    path: str
    name: str
    git_root: str
    ignore_patterns: List[str] = field(default_factory=list)
    access_permissions: Dict[str, bool] = field(default_factory=dict)
    last_commit: Optional[str] = None
    branch: Optional[str] = None
    is_accessible: bool = True
    error_message: Optional[str] = None


@dataclass
class CrossRepoDependency:
    """Cross-repository dependency relationship."""
    source_repo: str
    target_repo: str
    dependency_type: str  # "import", "api_call", "config_reference"
    source_file: str
    target_file: Optional[str] = None
    confidence: float = 1.0  # 0.0 to 1.0


# =============================================================================
# Enhanced Documentation Models
# =============================================================================

@dataclass
class SourceAttribution:
    """Source attribution for documentation results."""
    url: str
    title: str
    domain: str
    authority_score: float  # 0.0 to 1.0
    publication_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    is_official: bool = False  # True for official documentation
    grounding_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedDocumentationContext:
    """Enhanced documentation context with source attribution."""
    source: str  # URL or doc name
    title: str
    snippet: str  # Relevant excerpt
    relevance_score: float
    attribution: Optional[SourceAttribution] = None
    merge_source: str = "web"  # "web", "local", "merged"


# =============================================================================
# Metrics and Monitoring Models
# =============================================================================

@dataclass
class ComponentMetrics:
    """Metrics for a single component execution."""
    component_name: str
    execution_time: float  # seconds
    api_calls_made: int
    tokens_consumed: int
    success: bool
    error_message: Optional[str] = None
    fallback_used: bool = False
    cache_hit: bool = False


@dataclass
class RelevanceMetrics:
    """Relevance tracking metrics for context elements."""
    avg_file_usage_rate: float = 0.0  # Average usage rate for files
    avg_doc_usage_rate: float = 0.0  # Average usage rate for documentation
    high_value_elements: int = 0  # Count of high-value elements (usage_rate > 0.7)
    low_value_elements: int = 0  # Count of low-value elements (usage_rate < 0.2)
    optimizations_applied: int = 0  # Number of relevance score optimizations applied
    historical_data_available: bool = False  # Whether historical data was used


@dataclass
class QualityMetrics:
    """Quality metrics for context preparation."""
    overall_quality_score: float  # 0.0 to 1.0
    component_quality_scores: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    completeness_score: float = 0.0  # How complete the context is
    relevance_distribution: List[float] = field(default_factory=list)  # Relevance scores of all items
    relevance_metrics: Optional[RelevanceMetrics] = None  # Relevance tracking metrics


@dataclass
class ContextGatewayMetrics:
    """Comprehensive metrics for context gateway operation."""
    total_execution_time: float
    component_metrics: List[ComponentMetrics] = field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None
    cache_metrics: Dict[str, Any] = field(default_factory=dict)  # hit_rate, tokens_saved, etc.
    token_usage: Dict[str, int] = field(default_factory=dict)  # by component
    api_call_counts: Dict[str, int] = field(default_factory=dict)  # by model/provider


# =============================================================================
# Token Budget Management Models
# =============================================================================

@dataclass
class TokenBudgetAllocation:
    """Token budget allocation across components."""
    total_budget: int
    query_analysis: int
    file_discovery: int
    documentation_search: int
    code_search: int
    context_assembly: int
    reserve: int  # Emergency reserve for critical operations


@dataclass
class TokenBudgetUsage:
    """Actual token usage vs allocated budget."""
    allocated_budget: int
    actual_usage: int
    utilization_percentage: float
    component_allocation: Dict[str, int] = field(default_factory=dict)
    component_usage: Dict[str, int] = field(default_factory=dict)
    optimization_applied: bool = False
    optimization_details: List[str] = field(default_factory=list)


# =============================================================================
# Circuit Breaker and Resilience Models
# =============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerStatus:
    """Current status of a circuit breaker."""
    state: CircuitBreakerState
    failure_count: int
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None


class ComponentStatus(Enum):
    """Status of individual components."""
    SUCCESS = "success"
    PARTIAL = "partial"      # Succeeded with warnings
    FALLBACK = "fallback"    # Used fallback method
    FAILED = "failed"


@dataclass
class ComponentStatusInfo:
    """Detailed status information for a component."""
    status: ComponentStatus
    execution_time: float
    error_message: Optional[str] = None
    fallback_method: Optional[str] = None
    api_calls_made: int = 0
    tokens_consumed: int = 0
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Enhanced Context Models
# =============================================================================

@dataclass
class EnhancedFileContext:
    """Enhanced file context with repository information."""
    path: str
    relevance_score: float  # 0-1
    summary: str  # Gemini-generated summary
    key_elements: List[str] = field(default_factory=list)  # functions, classes, etc.
    line_count: int = 0
    size_kb: float = 0
    repository: Optional[str] = None  # Repository name if multi-repo
    last_modified: Optional[datetime] = None
    git_blame_info: Optional[Dict[str, Any]] = None


@dataclass
class EnhancedStructuredContext:
    """Enhanced structured context with all new features."""
    # Original fields from StructuredContext
    task_type: str
    task_summary: str
    complexity: str
    relevant_files: List[EnhancedFileContext] = field(default_factory=list)
    entry_point: Optional[str] = None
    documentation: List[EnhancedDocumentationContext] = field(default_factory=list)
    code_search: List[Any] = field(default_factory=list)  # Keep original for compatibility
    recommended_framework: str = "reason_flux"
    framework_reason: str = ""
    chain_suggestion: Optional[List[str]] = None
    execution_steps: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    potential_blockers: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    token_budget: int = 50000
    actual_tokens: int = 0
    
    # New enhancement fields
    cache_metadata: Optional[CacheMetadata] = None
    quality_metrics: Optional[QualityMetrics] = None
    repository_info: List[RepoInfo] = field(default_factory=list)
    source_attributions: List[SourceAttribution] = field(default_factory=list)
    token_budget_usage: Optional[TokenBudgetUsage] = None
    component_status: Dict[str, ComponentStatusInfo] = field(default_factory=dict)
    gateway_metrics: Optional[ContextGatewayMetrics] = None
    cross_repo_dependencies: List[CrossRepoDependency] = field(default_factory=list)
    
    def to_claude_prompt_enhanced(self) -> str:
        """Enhanced prompt with quality indicators and source links."""
        sections = []
        
        # Task Section with quality indicators
        quality_indicator = ""
        if self.quality_metrics:
            score = self.quality_metrics.overall_quality_score
            quality_bar = "█" * int(score * 5) + "░" * (5 - int(score * 5))
            quality_indicator = f" | Quality: [{quality_bar}] {score:.2f}"
        
        sections.append(f"""## Task Analysis{quality_indicator}
**Type**: {self.task_type} | **Complexity**: {self.complexity}
**Summary**: {self.task_summary}""")
        
        # Repository Information
        if self.repository_info:
            repo_lines = ["## Repository Information"]
            for repo in self.repository_info:
                status = "✓" if repo.is_accessible else "✗"
                repo_lines.append(f"- {status} `{repo.name}` ({repo.path})")
                if repo.last_commit:
                    repo_lines.append(f"  Latest: {repo.last_commit[:8]}")
            sections.append("\n".join(repo_lines))
        
        # Enhanced Files Section
        if self.relevant_files:
            file_lines = ["## Relevant Files"]
            if self.entry_point:
                file_lines.append(f"**Start here**: `{self.entry_point}`\n")
            for f in self.relevant_files[:10]:
                score_bar = "█" * int(f.relevance_score * 5) + "░" * (5 - int(f.relevance_score * 5))
                repo_label = f"[{f.repository}] " if f.repository else ""
                file_lines.append(f"- {repo_label}`{f.path}` [{score_bar}] - {f.summary}")
                if f.key_elements:
                    file_lines.append(f"  Key: {', '.join(f.key_elements[:5])}")
            sections.append("\n".join(file_lines))
        
        # Enhanced Documentation Section with Attribution
        if self.documentation:
            doc_lines = ["## Pre-Fetched Documentation"]
            for doc in self.documentation[:5]:
                authority_indicator = ""
                if doc.attribution:
                    if doc.attribution.is_official:
                        authority_indicator = " 🏛️"
                    elif doc.attribution.authority_score > 0.8:
                        authority_indicator = " ⭐"
                
                doc_lines.append(f"### {doc.title}{authority_indicator}")
                doc_lines.append(f"*Source: [{doc.source}]({doc.source})*")
                doc_lines.append(f"```\n{doc.snippet}\n```")
            sections.append("\n".join(doc_lines))
        
        # Component Status Section
        if self.component_status:
            status_lines = ["## Component Status"]
            for component, status_info in self.component_status.items():
                status_emoji = {
                    ComponentStatus.SUCCESS: "✅",
                    ComponentStatus.PARTIAL: "⚠️",
                    ComponentStatus.FALLBACK: "🔄",
                    ComponentStatus.FAILED: "❌"
                }
                emoji = status_emoji.get(status_info.status, "❓")
                status_lines.append(f"- {emoji} {component}: {status_info.status.value}")
                if status_info.fallback_method:
                    status_lines.append(f"  Fallback: {status_info.fallback_method}")
            sections.append("\n".join(status_lines))
        
        # Framework Section
        sections.append(f"""## Recommended Approach
**Framework**: `{self.recommended_framework}`
**Reason**: {self.framework_reason}""")
        if self.chain_suggestion:
            sections.append(f"**Chain**: {' -> '.join(self.chain_suggestion)}")
        
        # Execution Plan
        if self.execution_steps:
            steps = ["## Execution Plan"]
            for i, step in enumerate(self.execution_steps, 1):
                steps.append(f"{i}. {step}")
            sections.append("\n".join(steps))
        
        # Success Criteria
        if self.success_criteria:
            criteria = ["## Success Criteria"]
            for c in self.success_criteria:
                criteria.append(f"- [ ] {c}")
            sections.append("\n".join(criteria))
        
        # Token Budget Information
        if self.token_budget_usage:
            budget_lines = ["## Token Budget"]
            usage = self.token_budget_usage
            utilization_bar = "█" * int(usage.utilization_percentage / 20) + "░" * (5 - int(usage.utilization_percentage / 20))
            budget_lines.append(f"**Usage**: [{utilization_bar}] {usage.utilization_percentage:.1f}% ({usage.actual_usage:,}/{usage.allocated_budget:,} tokens)")
            if usage.optimization_applied:
                budget_lines.append("**Optimizations Applied**: " + ", ".join(usage.optimization_details))
            sections.append("\n".join(budget_lines))
        
        return "\n\n".join(sections)
    
    def to_detailed_json(self) -> Dict[str, Any]:
        """Detailed JSON with all metadata for debugging."""
        return {
            # Original fields
            "task_type": self.task_type,
            "task_summary": self.task_summary,
            "complexity": self.complexity,
            "recommended_framework": self.recommended_framework,
            "framework_reason": self.framework_reason,
            
            # Enhanced fields
            "cache_metadata": {
                "cache_hit": self.cache_metadata.cache_hit if self.cache_metadata else False,
                "cache_age_seconds": self.cache_metadata.cache_age.total_seconds() if self.cache_metadata else 0,
                "is_stale_fallback": self.cache_metadata.is_stale_fallback if self.cache_metadata else False,
            } if self.cache_metadata else None,
            
            "quality_metrics": {
                "overall_score": self.quality_metrics.overall_quality_score,
                "completeness": self.quality_metrics.completeness_score,
                "component_scores": self.quality_metrics.component_quality_scores,
            } if self.quality_metrics else None,
            
            "repositories": [
                {
                    "name": repo.name,
                    "path": repo.path,
                    "accessible": repo.is_accessible,
                    "last_commit": repo.last_commit,
                }
                for repo in self.repository_info
            ],
            
            "component_status": {
                name: {
                    "status": status.status.value,
                    "execution_time": status.execution_time,
                    "api_calls": status.api_calls_made,
                    "tokens": status.tokens_consumed,
                    "fallback": status.fallback_method,
                }
                for name, status in self.component_status.items()
            },
            
            "token_budget": {
                "allocated": self.token_budget_usage.allocated_budget if self.token_budget_usage else self.token_budget,
                "used": self.token_budget_usage.actual_usage if self.token_budget_usage else self.actual_tokens,
                "utilization": self.token_budget_usage.utilization_percentage if self.token_budget_usage else 0,
            },
            
            "files_count": len(self.relevant_files),
            "docs_count": len(self.documentation),
            "repos_count": len(self.repository_info),
        }

