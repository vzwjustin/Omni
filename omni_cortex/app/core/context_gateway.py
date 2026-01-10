"""
Context Gateway: Gemini-Powered Context Optimization for Claude

This module acts as a preprocessing layer that uses Gemini Flash to:
1. Analyze queries and understand intent
2. Discover relevant files in the workspace
3. Search web for documentation/APIs
4. Score and rank file relevance
5. Structure rich context for Claude

Architecture:
    User Query -> Gemini Flash (cheap, fast) -> Structured Context -> Claude (expensive, powerful)

Gemini does the "egg hunting" so Claude can focus on deep reasoning.

This module uses composition with specialized components:
- QueryAnalyzer: Query understanding and task analysis
- FileDiscoverer: Workspace file discovery and ranking
- DocumentationSearcher: Web and knowledge base documentation search
- CodeSearcher: Codebase search via grep/git
"""

import asyncio
import re
import threading
import time
import structlog
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional, List, Dict, Any

from .constants import CONTENT, LLM
from .context import (
    QueryAnalyzer,
    FileDiscoverer,
    DocumentationSearcher,
    CodeSearcher,
    ContextCache,
    get_context_cache,
    CacheMetadata,
    EnhancedFileContext,
    EnhancedDocumentationContext,
    ComponentStatus,
    QualityMetrics,
    MultiRepoFileDiscoverer,
)
from .context.fallback_analysis import (
    get_fallback_analyzer,
    ComponentFallbackMethods,
)
from .settings import get_settings

logger = structlog.get_logger("context_gateway")


# =============================================================================
# Dataclasses - Keep in gateway for cross-module compatibility
# =============================================================================

@dataclass
class FileContext:
    """Context about a discovered file."""
    path: str
    relevance_score: float  # 0-1
    summary: str  # Gemini-generated summary
    key_elements: List[str] = field(default_factory=list)  # functions, classes, etc.
    line_count: int = 0
    size_kb: float = 0


@dataclass
class DocumentationContext:
    """Context from web documentation lookup."""
    source: str  # URL or doc name
    title: str
    snippet: str  # Relevant excerpt
    relevance_score: float


@dataclass
class CodeSearchContext:
    """Context from code search (grep/git)."""
    search_type: str  # grep, git_log, git_blame
    query: str
    results: str  # Command output
    file_count: int = 0
    match_count: int = 0


@dataclass
class LibraryDocContext:
    """Context from library documentation (Context7)."""
    library: str
    library_id: str
    snippet: str
    source: str


@dataclass
class RepoContext:
    """Context from repository analysis (Greptile)."""
    context_type: str  # search, pr, review, custom
    title: str
    summary: str
    relevance_score: float
    source: str


@dataclass
class StructuredContext:
    """
    Rich, structured context packet for Claude.

    Everything Claude needs, pre-organized and ready for deep reasoning.
    No egg hunting required.
    """
    # Task Understanding
    task_type: str  # debug, implement, refactor, architect, etc.
    task_summary: str  # Clear description of what needs to be done
    complexity: str  # low, medium, high, very_high

    # Relevant Files (paths + summaries, not full contents)
    relevant_files: List[FileContext] = field(default_factory=list)
    entry_point: Optional[str] = None  # Where to start

    # Documentation (pre-fetched snippets)
    documentation: List[DocumentationContext] = field(default_factory=list)

    # Code Search Results (grep/git)
    code_search: List[CodeSearchContext] = field(default_factory=list)

    # Framework Recommendation
    recommended_framework: str = "reason_flux"
    framework_reason: str = ""
    chain_suggestion: Optional[List[str]] = None  # For complex tasks

    # Execution Plan
    execution_steps: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    potential_blockers: List[str] = field(default_factory=list)

    # Additional Context
    related_patterns: List[str] = field(default_factory=list)  # Similar code patterns
    dependencies: List[str] = field(default_factory=list)  # External deps to consider

    # Token Budget (Gemini optimizes to stay under this)
    token_budget: int = LLM.CONTEXT_TOKEN_BUDGET  # Max tokens for Claude prompt
    actual_tokens: int = 0  # Actual token count after generation

    def to_claude_prompt(self) -> str:
        """Format as rich context prompt for Claude."""
        sections = []

        # Task Section
        sections.append(f"""## Task Analysis
**Type**: {self.task_type} | **Complexity**: {self.complexity}
**Summary**: {self.task_summary}""")

        # Repository Information (multi-repo)
        repository_info = getattr(self, "repository_info", None)
        if repository_info:
            repo_lines = ["## Repository Information"]
            for repo in repository_info:
                name = getattr(repo, "name", None)
                path = getattr(repo, "path", None)
                if not name and not path:
                    continue
                line = f"- {name or 'unknown'}"
                if path:
                    line += f" ({path})"
                branch = getattr(repo, "branch", None)
                if branch:
                    line += f" [{branch}]"
                last_commit = getattr(repo, "last_commit", None)
                if last_commit:
                    line += f" @ {last_commit[:8]}"
                if getattr(repo, "is_accessible", True) is False:
                    line += " [inaccessible]"
                repo_lines.append(line)
            if len(repo_lines) > 1:
                sections.append("\n".join(repo_lines))

        cross_repo_dependencies = getattr(self, "cross_repo_dependencies", None)
        if cross_repo_dependencies:
            dep_lines = ["## Cross-Repo Dependencies"]
            for dep in cross_repo_dependencies[:10]:
                dep_lines.append(
                    f"- {dep.source_repo} -> {dep.target_repo} ({dep.dependency_type})"
                )
            sections.append("\n".join(dep_lines))

        # Files Section
        if self.relevant_files:
            file_lines = ["## Relevant Files"]
            if self.entry_point:
                file_lines.append(f"**Start here**: `{self.entry_point}`\n")
            for f in self.relevant_files[:10]:  # Top 10
                score_bar = "█" * int(f.relevance_score * 5) + "░" * (5 - int(f.relevance_score * 5))
                repo_label = f"[{f.repository}] " if getattr(f, "repository", None) else ""
                file_lines.append(f"- {repo_label}`{f.path}` [{score_bar}] - {f.summary}")
                if f.key_elements:
                    file_lines.append(f"  Key: {', '.join(f.key_elements[:5])}")
            sections.append("\n".join(file_lines))

        # Documentation Section
        if self.documentation:
            doc_lines = ["## Pre-Fetched Documentation"]
            for doc in self.documentation[:5]:
                authority_indicator = ""
                attribution = getattr(doc, "attribution", None)
                if attribution:
                    if getattr(attribution, "is_official", False):
                        authority_indicator = " [official]"
                    elif getattr(attribution, "authority_score", 0) >= 0.8:
                        authority_indicator = " [high-authority]"
                doc_lines.append(f"### {doc.title}{authority_indicator}")
                merge_source = getattr(doc, "merge_source", None)
                if merge_source:
                    doc_lines.append(f"*Source ({merge_source}): {doc.source}*")
                else:
                    doc_lines.append(f"*Source: {doc.source}*")
                doc_lines.append(f"```\n{doc.snippet}\n```")
            sections.append("\n".join(doc_lines))

        source_attributions = getattr(self, "source_attributions", None)
        if source_attributions:
            source_lines = ["## Source Attributions"]
            for attr in source_attributions[:5]:
                title = getattr(attr, "title", None) or getattr(attr, "url", "")
                url = getattr(attr, "url", "")
                line = f"- {title}" if title else "-"
                if url:
                    line += f" ({url})"
                domain = getattr(attr, "domain", None)
                if domain:
                    line += f" [{domain}]"
                if getattr(attr, "is_official", False):
                    line += " [official]"
                source_lines.append(line)
            sections.append("\n".join(source_lines))

        # Code Search Section
        if self.code_search:
            search_lines = ["## Code Search Results"]
            for search in self.code_search[:3]:
                search_lines.append(f"### {search.search_type.upper()}: {search.query}")
                search_lines.append(f"*Files: {search.file_count} | Matches: {search.match_count}*")
                search_lines.append(f"```\n{search.results[:CONTENT.SNIPPET_MAX]}\n```")
            sections.append("\n".join(search_lines))

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

        # Potential Blockers
        if self.potential_blockers:
            blockers = ["## Potential Blockers"]
            for b in self.potential_blockers:
                blockers.append(f"- {b}")
            sections.append("\n".join(blockers))

        return "\n\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        relevant_files = []
        for f in self.relevant_files:
            file_entry = {
                "path": f.path,
                "relevance_score": f.relevance_score,
                "summary": f.summary,
                "key_elements": f.key_elements,
            }
            repository = getattr(f, "repository", None)
            if repository:
                file_entry["repository"] = repository
            relevant_files.append(file_entry)

        documentation = []
        for d in self.documentation:
            doc_entry = {
                "source": d.source,
                "title": d.title,
                "snippet": d.snippet,
            }
            merge_source = getattr(d, "merge_source", None)
            if merge_source:
                doc_entry["merge_source"] = merge_source
            documentation.append(doc_entry)

        return {
            "task_type": self.task_type,
            "task_summary": self.task_summary,
            "complexity": self.complexity,
            "relevant_files": relevant_files,
            "entry_point": self.entry_point,
            "documentation": documentation,
            "recommended_framework": self.recommended_framework,
            "framework_reason": self.framework_reason,
            "chain_suggestion": self.chain_suggestion,
            "execution_steps": self.execution_steps,
            "success_criteria": self.success_criteria,
            "potential_blockers": self.potential_blockers,
        }


@dataclass
class EnhancedStructuredContext(StructuredContext):
    """
    Enhanced version of StructuredContext with quality metrics and diagnostics.

    Includes all fields from StructuredContext plus:
    - Cache metadata (hit/miss, age, staleness)
    - Quality metrics (relevance scores, coverage)
    - Token budget usage and transparency
    - Component status and errors
    - Repository information for multi-repo contexts
    """
    # Cache information
    cache_metadata: Optional[CacheMetadata] = None

    # Quality metrics
    quality_metrics: Optional[QualityMetrics] = None

    # Token budget transparency
    token_budget_usage: Optional[Any] = None  # TokenBudgetUsage - avoid circular import

    # Component status tracking
    component_status: Dict[str, ComponentStatus] = field(default_factory=dict)

    # Multi-repository information
    repository_info: List[Any] = field(default_factory=list)  # List[RepoInfo] - avoid circular import
    cross_repo_dependencies: List[Any] = field(default_factory=list)
    source_attributions: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enhanced fields."""
        base_dict = super().to_dict()

        # Add enhanced fields
        if self.cache_metadata:
            base_dict["cache_metadata"] = {
                "cache_hit": self.cache_metadata.cache_hit,
                "cache_age_seconds": self.cache_metadata.cache_age.total_seconds() if self.cache_metadata.cache_age else 0,
                "is_stale_fallback": self.cache_metadata.is_stale_fallback,
            }

        if self.quality_metrics:
            base_dict["quality_metrics"] = {
                "avg_file_relevance": self.quality_metrics.avg_file_relevance,
                "avg_doc_relevance": self.quality_metrics.avg_doc_relevance,
                "context_coverage_score": self.quality_metrics.context_coverage_score,
                "diversity_score": self.quality_metrics.diversity_score,
            }

        if self.token_budget_usage:
            base_dict["token_budget_usage"] = {
                "allocated": self.token_budget_usage.allocated_tokens,
                "used": self.token_budget_usage.used_tokens,
                "utilization": f"{self.token_budget_usage.utilization_percentage:.1f}%",
                "within_budget": self.token_budget_usage.within_budget,
            }

        if self.component_status:
            base_dict["component_status"] = {
                name: status.value for name, status in self.component_status.items()
            }
        if self.repository_info:
            base_dict["repository_info"] = [
                {
                    "name": repo.name,
                    "path": repo.path,
                    "branch": repo.branch,
                    "last_commit": repo.last_commit,
                    "is_accessible": repo.is_accessible,
                }
                for repo in self.repository_info
            ]
        if self.cross_repo_dependencies:
            base_dict["cross_repo_dependencies"] = [
                {
                    "source_repo": dep.source_repo,
                    "target_repo": dep.target_repo,
                    "dependency_type": dep.dependency_type,
                    "source_file": dep.source_file,
                    "target_file": dep.target_file,
                    "confidence": dep.confidence,
                }
                for dep in self.cross_repo_dependencies
            ]
        if self.source_attributions:
            base_dict["source_attributions"] = [
                {
                    "url": attr.url,
                    "title": attr.title,
                    "domain": attr.domain,
                    "authority_score": attr.authority_score,
                    "is_official": attr.is_official,
                }
                for attr in self.source_attributions
            ]

        return base_dict


# =============================================================================
# ContextGateway - Now uses composition with specialized components
# =============================================================================

class ContextGateway:
    """
    Gemini-powered context optimization layer.

    Uses composition with specialized components:
    - QueryAnalyzer: Analyzes queries to understand intent
    - FileDiscoverer: Discovers relevant files in workspace
    - DocumentationSearcher: Fetches documentation from web/knowledge base
    - CodeSearcher: Searches codebase via grep/git

    Usage:
        gateway = ContextGateway()
        context = await gateway.prepare_context(
            query="Fix the authentication bug in the login flow",
            workspace_path="/path/to/project"
        )
        # context.to_claude_prompt() returns rich, structured context
    """

    def __init__(
        self,
        query_analyzer: Optional[QueryAnalyzer] = None,
        file_discoverer: Optional[FileDiscoverer] = None,
        doc_searcher: Optional[DocumentationSearcher] = None,
        code_searcher: Optional[CodeSearcher] = None,
        cache: Optional[ContextCache] = None,
    ):
        """
        Initialize ContextGateway with optional dependency injection.

        Args:
            query_analyzer: Custom QueryAnalyzer instance (for testing)
            file_discoverer: Custom FileDiscoverer instance (for testing)
            doc_searcher: Custom DocumentationSearcher instance (for testing)
            code_searcher: Custom CodeSearcher instance (for testing)
            cache: Custom ContextCache instance (for testing)
        """
        # Core components (dependency injection pattern)
        self._query_analyzer = query_analyzer or QueryAnalyzer()
        self._file_discoverer = file_discoverer or FileDiscoverer()
        self._doc_searcher = doc_searcher or DocumentationSearcher()
        self._code_searcher = code_searcher or CodeSearcher()
        self._cache = cache or get_context_cache()
        self._settings = get_settings()
        self._enable_cache = self._settings.enable_context_cache
        self._multi_repo_discoverer: Optional[MultiRepoFileDiscoverer] = None
        self._enhanced_doc_searcher: Optional[Any] = None

        # Enhanced components (lazy-loaded to avoid circular imports and support testing)
        self._circuit_breakers: Dict[str, Any] = {}
        self._metrics: Optional[Any] = None
        self._budget_integration: Optional[Any] = None
        self._relevance_tracker: Optional[Any] = None

        # Initialize enhanced components if enabled
        if self._settings.enable_circuit_breaker:
            self._init_circuit_breakers()

        if self._settings.enable_enhanced_metrics:
            self._init_metrics()

        if self._settings.enable_dynamic_token_budget:
            self._init_budget_integration()

        if self._settings.enable_relevance_tracking:
            self._init_relevance_tracking()

    def _init_circuit_breakers(self) -> None:
        """Initialize circuit breakers for each component."""
        from .context import get_circuit_breaker
        self._circuit_breakers = {
            "query_analysis": get_circuit_breaker("query_analysis"),
            "file_discovery": get_circuit_breaker("file_discovery"),
            "doc_search": get_circuit_breaker("doc_search"),
            "code_search": get_circuit_breaker("code_search"),
        }

    def _init_metrics(self) -> None:
        """Initialize gateway metrics collector."""
        from .context import get_gateway_metrics
        self._metrics = get_gateway_metrics()

    def _init_budget_integration(self) -> None:
        """Initialize token budget integration."""
        from .context import get_budget_integration
        self._budget_integration = get_budget_integration()

    def _init_relevance_tracking(self) -> None:
        """Initialize relevance tracker."""
        from .context import get_relevance_tracker
        self._relevance_tracker = get_relevance_tracker()

    def _get_multi_repo_discoverer(self) -> MultiRepoFileDiscoverer:
        """Lazily initialize multi-repo discoverer."""
        if self._multi_repo_discoverer is None:
            self._multi_repo_discoverer = MultiRepoFileDiscoverer()
        return self._multi_repo_discoverer

    def _get_doc_searcher(self, enable_source_attribution: bool) -> DocumentationSearcher:
        """Select documentation searcher based on attribution settings."""
        if enable_source_attribution:
            if self._enhanced_doc_searcher is None:
                from .context.doc_searcher import EnhancedDocumentationSearcher
                self._enhanced_doc_searcher = EnhancedDocumentationSearcher()
            return self._enhanced_doc_searcher
        return self._doc_searcher

    @staticmethod
    def _normalize_file_discovery_result(
        file_result: Any
    ) -> tuple[List[Any], List[Any], List[Any]]:
        """Normalize file discovery result to (files, repos, dependencies)."""
        if isinstance(file_result, dict) and "files" in file_result:
            return (
                file_result.get("files", []) or [],
                file_result.get("repositories", []) or [],
                file_result.get("dependencies", []) or [],
            )
        return file_result or [], [], []

    @staticmethod
    def _coerce_enhanced_file(file_item: Any) -> EnhancedFileContext:
        """Normalize file items to EnhancedFileContext for downstream consumers."""
        if isinstance(file_item, EnhancedFileContext):
            return file_item
        return EnhancedFileContext(
            path=file_item.path,
            relevance_score=file_item.relevance_score,
            summary=file_item.summary,
            key_elements=getattr(file_item, "key_elements", []),
            line_count=getattr(file_item, "line_count", 0),
            size_kb=getattr(file_item, "size_kb", 0),
            repository=getattr(file_item, "repository", None),
            last_modified=getattr(file_item, "last_modified", None),
            git_blame_info=getattr(file_item, "git_blame_info", None),
        )

    def _fallback_analyze(self, query: str) -> Dict[str, Any]:
        """
        Fallback analyzer using enhanced pattern matching when Gemini is unavailable.

        Uses the EnhancedFallbackAnalyzer for improved task detection.

        Args:
            query: The user's query string

        Returns:
            Dictionary with task_type, summary, complexity, framework, and framework_reason
        """
        fallback_analyzer = get_fallback_analyzer()
        return fallback_analyzer.analyze(query)

    async def prepare_context(
        self,
        query: str,
        workspace_path: Optional[str] = None,
        code_context: Optional[str] = None,
        file_list: Optional[List[str]] = None,
        search_docs: bool = True,
        max_files: int = 15,
        enable_multi_repo: Optional[bool] = None,
        enable_source_attribution: Optional[bool] = None,
    ) -> EnhancedStructuredContext:
        """
        Prepare rich, structured context for Claude.

        Gemini does all the hunting:
        1. Analyzes the query to understand task type and complexity
        2. Discovers relevant files in workspace
        3. Fetches documentation if needed
        4. Optimizes content with token budget management
        5. Structures everything for Claude with quality metrics

        Args:
            query: The user's request
            workspace_path: Path to the workspace/project
            code_context: Any code snippets provided
            file_list: Pre-specified files to consider
            search_docs: Whether to search web for documentation
            max_files: Maximum files to include in context
            enable_multi_repo: Override multi-repo discovery settings
            enable_source_attribution: Override documentation source attribution settings

        Returns:
            EnhancedStructuredContext ready for Claude with quality metrics
        """
        # Start timer for metrics
        start_time = time.time()

        logger.info("context_gateway_start", query=query[:CONTENT.QUERY_LOG])

        # Initialize tracking variables
        cache_metadata: Optional[CacheMetadata] = None
        is_stale_fallback = False
        component_status: Dict[str, ComponentStatus] = {}
        relevance_session_id: Optional[str] = None
        repository_info: List[Any] = []
        cross_repo_dependencies: List[Any] = []
        source_attributions: List[Any] = []

        use_multi_repo = self._settings.enable_multi_repo_discovery if enable_multi_repo is None else enable_multi_repo
        use_source_attribution = self._settings.enable_source_attribution if enable_source_attribution is None else enable_source_attribution
        use_doc_prioritization = self._settings.enable_documentation_prioritization
        if file_list or not workspace_path:
            use_multi_repo = False

        doc_searcher = self._get_doc_searcher(use_source_attribution)
        doc_search_method = (
            doc_searcher.search_web_with_attribution
            if use_source_attribution and hasattr(doc_searcher, "search_web_with_attribution")
            else doc_searcher.search_web
        )

        async def _run_doc_search() -> List[Any]:
            doc_task_type = "general"
            if use_doc_prioritization:
                doc_task_type = self._fallback_analyze(query).get("task_type", "general")

            if hasattr(doc_searcher, "search_with_fallback"):
                return await doc_searcher.search_with_fallback(query, doc_task_type)

            results = await doc_search_method(query)
            if not use_doc_prioritization or not hasattr(doc_searcher, "search_knowledge_base"):
                return results

            local_results = []
            try:
                local_results = await doc_searcher.search_knowledge_base(query, doc_task_type)
            except Exception as e:
                logger.warning("knowledge_base_merge_failed", error=str(e))

            if local_results:
                merged = list(results or []) + list(local_results)
                merged.sort(key=lambda d: getattr(d, "relevance_score", 0), reverse=True)
                return merged

            return results

        # Start relevance tracking session if enabled
        if self._relevance_tracker:
            relevance_session_id = self._relevance_tracker.start_session(query)

        # Phase 0: Cache Lookup (if enabled)
        if self._enable_cache and workspace_path:
            # Try to get cached results for each component
            query_cache_key = self._cache.generate_cache_key(
                query, workspace_path, "query_analysis"
            )
            file_cache_key = self._cache.generate_cache_key(
                query, workspace_path, "file_discovery"
            )
            doc_cache_key = self._cache.generate_cache_key(
                query, workspace_path, "documentation"
            )
            
            # Check cache for query analysis
            cached_query_analysis = await self._cache.get(query_cache_key)
            
            # Check cache for file discovery
            cached_file_result = await self._cache.get(file_cache_key)
            
            # Check cache for documentation
            cached_doc_result = await self._cache.get(doc_cache_key)
            
            # If we have all cached results, use them
            if cached_query_analysis and cached_file_result and cached_doc_result:
                logger.info(
                    "cache_full_hit",
                    query_age=cached_query_analysis.age.total_seconds(),
                    file_age=cached_file_result.age.total_seconds(),
                    doc_age=cached_doc_result.age.total_seconds()
                )
                
                # Build context from cache
                query_analysis = cached_query_analysis.value
                file_result = cached_file_result.value
                doc_result = cached_doc_result.value
                
                # Still need to run code search (not cached as it's fast)
                try:
                    if workspace_path:
                        code_result = await self._code_searcher.search(query, workspace_path)
                    else:
                        code_result = None
                except Exception as e:
                    logger.warning("code_search_failed", error=str(e))
                    code_result = None
                
                # Create cache metadata
                cache_metadata = CacheMetadata(
                    cache_key=query_cache_key,
                    cache_hit=True,
                    cache_age=cached_query_analysis.age,
                    cache_type="full",
                    workspace_fingerprint=cached_query_analysis.workspace_fingerprint,
                    is_stale_fallback=False
                )
                
                # Skip to context assembly
                file_items, repository_info, cross_repo_dependencies = self._normalize_file_discovery_result(file_result)
                converted_files = [self._coerce_enhanced_file(f) for f in file_items]
                doc_contexts = doc_result
                source_attributions = [
                    doc.attribution
                    for doc in doc_contexts
                    if getattr(doc, "attribution", None)
                ]
                code_search_results = []
                if code_result:
                    code_search_results = [
                        CodeSearchContext(
                            search_type=c.search_type,
                            query=c.query,
                            results=c.results,
                            file_count=c.file_count,
                            match_count=c.match_count,
                        )
                        for c in code_result
                    ]
                
                # Jump to context assembly
                context = EnhancedStructuredContext(
                    task_type=query_analysis.get("task_type", "general"),
                    task_summary=query_analysis.get("summary", query),
                    complexity=query_analysis.get("complexity", "medium"),
                    relevant_files=converted_files,
                    entry_point=query_analysis.get("entry_point"),
                    documentation=doc_contexts,
                    code_search=code_search_results,
                    recommended_framework=query_analysis.get("framework", "reason_flux"),
                    framework_reason=query_analysis.get("framework_reason", "General-purpose reasoning"),
                    chain_suggestion=query_analysis.get("chain"),
                    execution_steps=query_analysis.get("steps", []),
                    success_criteria=query_analysis.get("success_criteria", []),
                    potential_blockers=query_analysis.get("blockers", []),
                    related_patterns=query_analysis.get("patterns", []),
                    dependencies=query_analysis.get("dependencies", []),
                    cache_metadata=cache_metadata,
                    component_status=component_status,
                    repository_info=repository_info,
                    cross_repo_dependencies=cross_repo_dependencies,
                    source_attributions=source_attributions,
                )
                
                logger.info(
                    "context_gateway_complete_cached",
                    task_type=context.task_type,
                    files=len(context.relevant_files),
                    docs=len(context.documentation),
                    code_searches=len(context.code_search),
                    framework=context.recommended_framework,
                )
                
                return context

        # Phase 1: Discovery & Search (Parallel with named tasks)
        # Run discovery tasks with explicit naming for cleaner result handling

        async def _run_file_discovery() -> Any:
            if use_multi_repo:
                discoverer = self._get_multi_repo_discoverer()
                files, repos, deps = await discoverer.discover_multi_repo(
                    query=query,
                    workspace_path=workspace_path,
                    max_files=max_files,
                )
                return {"files": files, "repositories": repos, "dependencies": deps}
            return await self._file_discoverer.discover(query, workspace_path, file_list, max_files)

        # Create named coroutines for parallel execution
        async def _discover_files():
            # Check cache first
            if self._enable_cache and workspace_path:
                file_cache_key = self._cache.generate_cache_key(
                    query, workspace_path, "file_discovery"
                )
                cached = await self._cache.get(file_cache_key, allow_stale=True)
                if cached and not cached.is_expired:
                    logger.info("cache_hit_file_discovery", age=cached.age.total_seconds())
                    return cached.value
                elif cached and cached.is_expired:
                    # Try fresh discovery, fallback to stale cache on error
                    try:
                        result = await _run_file_discovery()
                        # Cache the result
                        await self._cache.set(file_cache_key, result, "file_discovery", workspace_path)
                        return result
                    except Exception as e:
                        logger.warning("file_discovery_failed_using_stale_cache", error=str(e))
                        nonlocal is_stale_fallback
                        is_stale_fallback = True
                        return cached.value
            
            # No cache, do fresh discovery with circuit breaker protection
            comp_start = time.time()
            try:
                if self._circuit_breakers and "file_discovery" in self._circuit_breakers:
                    result = await self._circuit_breakers["file_discovery"].call(
                        _run_file_discovery
                    )
                    component_status["file_discovery"] = ComponentStatus.SUCCESS
                else:
                    result = await _run_file_discovery()
                    component_status["file_discovery"] = ComponentStatus.SUCCESS

                # Record metrics
                if self._metrics:
                    duration = time.time() - comp_start
                    self._metrics.record_component_performance("file_discovery", duration, success=True)

                # Cache the result
                if self._enable_cache and workspace_path:
                    file_cache_key = self._cache.generate_cache_key(
                        query, workspace_path, "file_discovery"
                    )
                    await self._cache.set(file_cache_key, result, "file_discovery", workspace_path)

                return result
            except Exception as e:
                duration = time.time() - comp_start
                if self._metrics:
                    self._metrics.record_component_performance("file_discovery", duration, success=False)
                component_status["file_discovery"] = ComponentStatus.FAILED
                logger.error("file_discovery_error", error=str(e))
                raise
        
        async def _search_docs():
            if not search_docs:
                return None
            
            # Check cache first
            if self._enable_cache and workspace_path:
                doc_cache_key = self._cache.generate_cache_key(
                    query, workspace_path, "documentation"
                )
                cached = await self._cache.get(doc_cache_key, allow_stale=True)
                if cached and not cached.is_expired:
                    logger.info("cache_hit_documentation", age=cached.age.total_seconds())
                    return cached.value
                elif cached and cached.is_expired:
                    # Try fresh search, fallback to stale cache on error
                    try:
                        result = await _run_doc_search()
                        # Cache the result
                        await self._cache.set(doc_cache_key, result, "documentation", workspace_path)
                        return result
                    except Exception as e:
                        logger.warning("doc_search_failed_using_stale_cache", error=str(e))
                        nonlocal is_stale_fallback
                        is_stale_fallback = True
                        return cached.value
            
            # No cache, do fresh search with circuit breaker protection
            comp_start = time.time()
            try:
                if self._circuit_breakers and "doc_search" in self._circuit_breakers:
                    result = await self._circuit_breakers["doc_search"].call(
                        _run_doc_search
                    )
                    component_status["doc_search"] = ComponentStatus.SUCCESS
                else:
                    result = await _run_doc_search()
                    component_status["doc_search"] = ComponentStatus.SUCCESS

                # Record metrics
                if self._metrics:
                    duration = time.time() - comp_start
                    self._metrics.record_component_performance("doc_search", duration, success=True)

                # Cache the result
                if self._enable_cache and workspace_path:
                    doc_cache_key = self._cache.generate_cache_key(
                        query, workspace_path, "documentation"
                    )
                    await self._cache.set(doc_cache_key, result, "documentation", workspace_path)

                return result
            except Exception as e:
                duration = time.time() - comp_start
                if self._metrics:
                    self._metrics.record_component_performance("doc_search", duration, success=False)
                component_status["doc_search"] = ComponentStatus.FAILED
                logger.error("doc_search_error", error=str(e))
                raise
        
        async def _search_code():
            # Code search is fast, don't cache it
            if not workspace_path:
                return None

            comp_start = time.time()
            try:
                if self._circuit_breakers and "code_search" in self._circuit_breakers:
                    result = await self._circuit_breakers["code_search"].call(
                        self._code_searcher.search, query, workspace_path
                    )
                    component_status["code_search"] = ComponentStatus.SUCCESS
                else:
                    result = await self._code_searcher.search(query, workspace_path)
                    component_status["code_search"] = ComponentStatus.SUCCESS

                # Record metrics
                if self._metrics:
                    duration = time.time() - comp_start
                    self._metrics.record_component_performance("code_search", duration, success=True)

                return result
            except Exception as e:
                duration = time.time() - comp_start
                if self._metrics:
                    self._metrics.record_component_performance("code_search", duration, success=False)
                component_status["code_search"] = ComponentStatus.FAILED
                logger.error("code_search_error", error=str(e))
                raise
        
        # Execute all discovery tasks in parallel
        file_result, doc_result, code_result = await asyncio.gather(
            _discover_files(),
            _search_docs(),
            _search_code(),
            return_exceptions=True,
        )
        
        # Process file discovery results
        converted_files = []
        if isinstance(file_result, Exception):
            logger.warning("file_discovery_failed", error=str(file_result))
        elif file_result:
            file_items, repository_info, cross_repo_dependencies = self._normalize_file_discovery_result(file_result)
            converted_files = [self._coerce_enhanced_file(f) for f in file_items]

        # Process documentation search results
        doc_contexts = []
        source_attributions = []
        if isinstance(doc_result, Exception):
            logger.warning("doc_search_failed", error=str(doc_result))
        elif doc_result:
            doc_contexts = list(doc_result)
            for doc in doc_result:
                attribution = getattr(doc, "attribution", None)
                if attribution:
                    source_attributions.append(attribution)

        # Process code search results
        code_search_results = []
        if isinstance(code_result, Exception):
            logger.warning("code_search_failed", error=str(code_result))
        elif code_result:
            code_search_results = [
                CodeSearchContext(
                    search_type=c.search_type,
                    query=c.query,
                    results=c.results,
                    file_count=c.file_count,
                    match_count=c.match_count,
                )
                for c in code_result
            ]

        # Phase 2: Analysis (Sequential - uses discovery context)
        # Construct documentation context string for the analyzer
        docs_context_str = ""
        if doc_contexts:
            docs_context_str += "DOCUMENTATION FOUND:\n"
            for d in doc_contexts[:3]:
                docs_context_str += f"- {d.title} ({d.source}): {d.snippet[:200]}...\n"
        
        if converted_files:
            docs_context_str += "\nRELEVANT FILES:\n"
            for f in converted_files[:5]:
                docs_context_str += f"- {f.path}: {f.summary}\n"

        # Check cache for query analysis
        query_analysis = None
        if self._enable_cache and workspace_path:
            query_cache_key = self._cache.generate_cache_key(
                query, workspace_path, "query_analysis"
            )
            cached = await self._cache.get(query_cache_key, allow_stale=True)
            if cached and not cached.is_expired:
                logger.info("cache_hit_query_analysis", age=cached.age.total_seconds())
                query_analysis = cached.value
                
                # Create cache metadata
                cache_metadata = CacheMetadata(
                    cache_key=query_cache_key,
                    cache_hit=True,
                    cache_age=cached.age,
                    cache_type="query_analysis",
                    workspace_fingerprint=cached.workspace_fingerprint,
                    is_stale_fallback=is_stale_fallback
                )
            elif cached and cached.is_expired:
                # Try fresh analysis with circuit breaker, fallback to stale cache on error
                comp_start = time.time()
                try:
                    if self._circuit_breakers and "query_analysis" in self._circuit_breakers:
                        query_analysis = await self._circuit_breakers["query_analysis"].call(
                            self._query_analyzer.analyze,
                            query,
                            code_context,
                            documentation_context=docs_context_str if docs_context_str else None
                        )
                    else:
                        query_analysis = await self._query_analyzer.analyze(
                            query,
                            code_context,
                            documentation_context=docs_context_str if docs_context_str else None
                        )

                    # Record metrics
                    if self._metrics:
                        duration = time.time() - comp_start
                        self._metrics.record_component_performance("query_analysis", duration, success=True)

                    component_status["query_analysis"] = ComponentStatus.SUCCESS

                    # Cache the result
                    await self._cache.set(query_cache_key, query_analysis, "query_analysis", workspace_path)
                    
                    # Create cache metadata
                    cache_metadata = CacheMetadata(
                        cache_key=query_cache_key,
                        cache_hit=False,
                        cache_age=timedelta(0),
                        cache_type="query_analysis",
                        workspace_fingerprint=self._cache._compute_workspace_fingerprint(workspace_path),
                        is_stale_fallback=False
                    )
                except Exception as e:
                    logger.warning("query_analysis_failed_using_stale_cache", error=str(e))
                    query_analysis = cached.value
                    is_stale_fallback = True
                    
                    # Create cache metadata
                    cache_metadata = CacheMetadata(
                        cache_key=query_cache_key,
                        cache_hit=True,
                        cache_age=cached.age,
                        cache_type="query_analysis",
                        workspace_fingerprint=cached.workspace_fingerprint,
                        is_stale_fallback=True
                    )
        
        # If no cached analysis, do fresh analysis
        if query_analysis is None:
            comp_start = time.time()
            try:
                if self._circuit_breakers and "query_analysis" in self._circuit_breakers:
                    query_analysis = await self._circuit_breakers["query_analysis"].call(
                        self._query_analyzer.analyze,
                        query,
                        code_context,
                        documentation_context=docs_context_str if docs_context_str else None
                    )
                else:
                    query_analysis = await self._query_analyzer.analyze(
                        query,
                        code_context,
                        documentation_context=docs_context_str if docs_context_str else None
                    )

                # Record metrics
                if self._metrics:
                    duration = time.time() - comp_start
                    self._metrics.record_component_performance("query_analysis", duration, success=True)

                component_status["query_analysis"] = ComponentStatus.SUCCESS

                # Cache the result
                if self._enable_cache and workspace_path:
                    query_cache_key = self._cache.generate_cache_key(
                        query, workspace_path, "query_analysis"
                    )
                    await self._cache.set(query_cache_key, query_analysis, "query_analysis", workspace_path)

                    # Create cache metadata
                    cache_metadata = CacheMetadata(
                        cache_key=query_cache_key,
                        cache_hit=False,
                        cache_age=timedelta(0),
                        cache_type="query_analysis",
                        workspace_fingerprint=self._cache._compute_workspace_fingerprint(workspace_path),
                        is_stale_fallback=False
                    )
            except Exception as e:
                duration = time.time() - comp_start
                if self._metrics:
                    self._metrics.record_component_performance("query_analysis", duration, success=False)
                component_status["query_analysis"] = ComponentStatus.FALLBACK
                logger.warning("query_analysis_failed_using_fallback", error=str(e))
                query_analysis = self._fallback_analyze(query)

        # Phase 3: Token Budget Optimization (if enabled)
        task_type = query_analysis.get("task_type", "general")
        complexity = query_analysis.get("complexity", "medium")
        token_budget_usage = None

        # Convert to enhanced models for budget optimization
        enhanced_files = [
            self._coerce_enhanced_file(f)
            for f in converted_files
        ]

        enhanced_docs = []
        for d in doc_contexts:
            if isinstance(d, EnhancedDocumentationContext):
                enhanced_docs.append(d)
            else:
                enhanced_docs.append(EnhancedDocumentationContext(
                    source=d.source,
                    title=d.title,
                    snippet=d.snippet,
                    relevance_score=d.relevance_score,
                    attribution=getattr(d, "attribution", None),
                    merge_source=getattr(d, "merge_source", "web"),
                ))

        # Apply token budget optimization if enabled
        if self._budget_integration:
            try:
                logger.info("token_budget_optimization_start", files=len(enhanced_files), docs=len(enhanced_docs))
                optimized_files, optimized_docs, optimized_code, token_budget_usage = \
                    await self._budget_integration.optimize_context_for_budget(
                        query, task_type, complexity,
                        enhanced_files, enhanced_docs, code_search_results
                    )
                # Use optimized content
                converted_files = [self._coerce_enhanced_file(f) for f in optimized_files]
                doc_contexts = list(optimized_docs)
                code_search_results = optimized_code
                logger.info("token_budget_optimization_complete", optimized_files=len(converted_files), optimized_docs=len(doc_contexts))
            except Exception as e:
                logger.warning("token_budget_optimization_failed", error=str(e))
                # Continue with unoptimized content

        # Refresh source attributions after any optimization
        source_attributions = [
            doc.attribution
            for doc in doc_contexts
            if getattr(doc, "attribution", None)
        ]

        # Phase 4: Relevance Tracking (if enabled)
        if self._relevance_tracker and relevance_session_id:
            try:
                # Track provided files
                for file in converted_files:
                    self._relevance_tracker.track_element_provided(
                        relevance_session_id, "file", file.path, file.relevance_score
                    )
                # Track provided docs
                for doc in doc_contexts:
                    self._relevance_tracker.track_element_provided(
                        relevance_session_id, "doc", doc.source, doc.relevance_score
                    )
            except Exception as e:
                logger.warning("relevance_tracking_failed", error=str(e))

        # Phase 5: Calculate Quality Metrics
        quality_metrics = None
        if converted_files or doc_contexts:
            try:
                avg_file_relevance = sum(f.relevance_score for f in converted_files) / len(converted_files) if converted_files else 0
                avg_doc_relevance = sum(d.relevance_score for d in doc_contexts) / len(doc_contexts) if doc_contexts else 0
                context_coverage_score = min(1.0, (len(converted_files) / max_files) * 0.7 + (len(doc_contexts) / 5) * 0.3)
                diversity_score = len(set(f.path.split('/')[0] for f in converted_files if '/' in f.path)) / max(1, len(converted_files))

                quality_metrics = QualityMetrics(
                    overall_quality_score=context_coverage_score,
                    avg_file_relevance=avg_file_relevance,
                    avg_doc_relevance=avg_doc_relevance,
                    context_coverage_score=context_coverage_score,
                    diversity_score=diversity_score,
                )
            except Exception as e:
                logger.warning("quality_metrics_calculation_failed", error=str(e))

        # Phase 6: Build Enhanced Structured Context
        context = EnhancedStructuredContext(
            task_type=task_type,
            task_summary=query_analysis.get("summary", query),
            complexity=complexity,
            relevant_files=converted_files,
            entry_point=query_analysis.get("entry_point"),
            documentation=doc_contexts,
            code_search=code_search_results,
            recommended_framework=query_analysis.get("framework", "reason_flux"),
            framework_reason=query_analysis.get("framework_reason", "General-purpose reasoning"),
            chain_suggestion=query_analysis.get("chain"),
            execution_steps=query_analysis.get("steps", []),
            success_criteria=query_analysis.get("success_criteria", []),
            potential_blockers=query_analysis.get("blockers", []),
            related_patterns=query_analysis.get("patterns", []),
            dependencies=query_analysis.get("dependencies", []),
            # Enhanced fields
            cache_metadata=cache_metadata,
            quality_metrics=quality_metrics,
            token_budget_usage=token_budget_usage,
            component_status=component_status,
            repository_info=repository_info,
            cross_repo_dependencies=cross_repo_dependencies,
            source_attributions=source_attributions,
        )

        # Phase 7: Record Final Metrics
        total_duration = time.time() - start_time
        if self._metrics:
            try:
                # Record overall gateway performance
                self._metrics.record_component_performance("context_gateway", total_duration, success=True)

                # Record context quality if available
                if quality_metrics:
                    relevance_scores = [f.relevance_score for f in converted_files] + [d.relevance_score for d in doc_contexts]
                    self._metrics.record_context_quality(
                        quality_score=quality_metrics.context_coverage_score,
                        relevance_scores=relevance_scores
                    )

                # Record cache effectiveness
                if cache_metadata:
                    # Estimate tokens saved from cache
                    tokens_saved = len(converted_files) * 200 + len(doc_contexts) * 300
                    if hasattr(self._metrics, "record_cache_operation"):
                        self._metrics.record_cache_operation(
                            operation="hit" if cache_metadata.cache_hit else "miss",
                            hit=cache_metadata.cache_hit,
                            tokens_saved=tokens_saved if cache_metadata.cache_hit else 0
                        )
            except Exception as e:
                logger.warning("final_metrics_recording_failed", error=str(e))

        logger.info(
            "context_gateway_complete",
            task_type=context.task_type,
            files=len(context.relevant_files),
            docs=len(context.documentation),
            code_searches=len(context.code_search),
            framework=context.recommended_framework,
            cache_hit=cache_metadata.cache_hit if cache_metadata else False,
            stale_fallback=is_stale_fallback,
            duration_seconds=f"{total_duration:.2f}",
            quality_score=quality_metrics.context_coverage_score if quality_metrics else None,
            budget_within=token_budget_usage.within_budget if token_budget_usage else None,
        )

        return context

    async def quick_analyze(self, query: str) -> Dict[str, Any]:
        """
        Quick analysis without file discovery or doc search.

        Use for fast routing decisions.
        """
        return await self._query_analyzer.analyze(query)


# =============================================================================
# Global singleton with thread-safe initialization
# =============================================================================

_gateway: Optional[ContextGateway] = None
_gateway_lock = threading.Lock()


def get_context_gateway() -> ContextGateway:
    """Get the global ContextGateway singleton (thread-safe)."""
    global _gateway

    # Fast path: already initialized
    if _gateway is not None:
        return _gateway

    # Thread-safe initialization with double-check locking
    with _gateway_lock:
        if _gateway is None:
            _gateway = ContextGateway()
    return _gateway


async def prepare_context_for_claude(
    query: str,
    workspace_path: Optional[str] = None,
    code_context: Optional[str] = None,
    file_list: Optional[List[str]] = None,
    search_docs: bool = True,
    enable_multi_repo: Optional[bool] = None,
    enable_source_attribution: Optional[bool] = None,
) -> EnhancedStructuredContext:
    """
    Convenience function to prepare context with all enhancements.

    Returns EnhancedStructuredContext with quality metrics, cache info,
    token budget usage, and component status.

    Example:
        context = await prepare_context_for_claude(
            query="Fix the auth bug",
            workspace_path="/my/project"
        )
        claude_prompt = context.to_claude_prompt()
        print(f"Quality: {context.quality_metrics.context_coverage_score}")
    """
    gateway = get_context_gateway()
    return await gateway.prepare_context(
        query=query,
        workspace_path=workspace_path,
        code_context=code_context,
        file_list=file_list,
        search_docs=search_docs,
        enable_multi_repo=enable_multi_repo,
        enable_source_attribution=enable_source_attribution,
    )
