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
import structlog
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .constants import CONTENT, LLM
from .context import (
    QueryAnalyzer,
    FileDiscoverer,
    DocumentationSearcher,
    CodeSearcher,
)

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

        # Files Section
        if self.relevant_files:
            file_lines = ["## Relevant Files"]
            if self.entry_point:
                file_lines.append(f"**Start here**: `{self.entry_point}`\n")
            for f in self.relevant_files[:10]:  # Top 10
                score_bar = "█" * int(f.relevance_score * 5) + "░" * (5 - int(f.relevance_score * 5))
                file_lines.append(f"- `{f.path}` [{score_bar}] - {f.summary}")
                if f.key_elements:
                    file_lines.append(f"  Key: {', '.join(f.key_elements[:5])}")
            sections.append("\n".join(file_lines))

        # Documentation Section
        if self.documentation:
            doc_lines = ["## Pre-Fetched Documentation"]
            for doc in self.documentation[:5]:
                doc_lines.append(f"### {doc.title}")
                doc_lines.append(f"*Source: {doc.source}*")
                doc_lines.append(f"```\n{doc.snippet}\n```")
            sections.append("\n".join(doc_lines))

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
        return {
            "task_type": self.task_type,
            "task_summary": self.task_summary,
            "complexity": self.complexity,
            "relevant_files": [
                {
                    "path": f.path,
                    "relevance_score": f.relevance_score,
                    "summary": f.summary,
                    "key_elements": f.key_elements,
                }
                for f in self.relevant_files
            ],
            "entry_point": self.entry_point,
            "documentation": [
                {
                    "source": d.source,
                    "title": d.title,
                    "snippet": d.snippet,
                }
                for d in self.documentation
            ],
            "recommended_framework": self.recommended_framework,
            "framework_reason": self.framework_reason,
            "chain_suggestion": self.chain_suggestion,
            "execution_steps": self.execution_steps,
            "success_criteria": self.success_criteria,
            "potential_blockers": self.potential_blockers,
        }


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
    ):
        """
        Initialize ContextGateway with optional dependency injection.
        
        Args:
            query_analyzer: Custom QueryAnalyzer instance (for testing)
            file_discoverer: Custom FileDiscoverer instance (for testing)
            doc_searcher: Custom DocumentationSearcher instance (for testing)
            code_searcher: Custom CodeSearcher instance (for testing)
        """
        # Use provided instances or create defaults (dependency injection pattern)
        self._query_analyzer = query_analyzer or QueryAnalyzer()
        self._file_discoverer = file_discoverer or FileDiscoverer()
        self._doc_searcher = doc_searcher or DocumentationSearcher()
        self._code_searcher = code_searcher or CodeSearcher()

    def _fallback_analyze(self, query: str) -> Dict[str, Any]:
        """
        Fallback analyzer using pattern matching when Gemini is unavailable.

        Uses regex patterns to detect task types based on keywords in the query.
        This provides graceful degradation when the LLM-based analyzer fails.

        Args:
            query: The user's query string

        Returns:
            Dictionary with task_type, summary, complexity, framework, and framework_reason
        """
        query_lower = query.lower()

        # Pattern definitions: (regex_pattern, task_type, framework)
        patterns = [
            (r'\b(debug|error|fix|bug|crash|exception|traceback|issue)\b',
             "debug", "self_debugging"),
            (r'\b(implement|add|create|build|new|feature|develop)\b',
             "implement", "reason_flux"),
            (r'\b(refactor|clean|improve|optimize|restructure|simplify)\b',
             "refactor", "chain_of_verification"),
            (r'\b(explain|how|what|why|understand|describe|clarify)\b',
             "explain", "chain_of_note"),
        ]

        # Check patterns in order of priority
        for pattern, task_type, framework in patterns:
            if re.search(pattern, query_lower):
                framework_reasons = {
                    "debug": "Pattern-based routing: debugging task detected",
                    "implement": "Pattern-based routing: implementation task detected",
                    "refactor": "Pattern-based routing: refactoring task detected",
                    "explain": "Pattern-based routing: explanation task detected",
                }
                return {
                    "task_type": task_type,
                    "summary": query,
                    "complexity": "medium",
                    "framework": framework,
                    "framework_reason": framework_reasons.get(task_type, "Fallback pattern matching"),
                }

        # Default fallback
        return {
            "task_type": "general",
            "summary": query,
            "complexity": "medium",
            "framework": "reason_flux",
            "framework_reason": "Fallback: no specific pattern matched, using general-purpose reasoning",
        }

    async def prepare_context(
        self,
        query: str,
        workspace_path: Optional[str] = None,
        code_context: Optional[str] = None,
        file_list: Optional[List[str]] = None,
        search_docs: bool = True,
        max_files: int = 15,
    ) -> StructuredContext:
        """
        Prepare rich, structured context for Claude.

        Gemini does all the hunting:
        1. Analyzes the query to understand task type and complexity
        2. Discovers relevant files in workspace
        3. Fetches documentation if needed
        4. Structures everything for Claude

        Args:
            query: The user's request
            workspace_path: Path to the workspace/project
            code_context: Any code snippets provided
            file_list: Pre-specified files to consider
            search_docs: Whether to search web for documentation
            max_files: Maximum files to include in context

        Returns:
            StructuredContext ready for Claude
        """
        logger.info("context_gateway_start", query=query[:CONTENT.QUERY_LOG])

        # Phase 1: Discovery & Search (Parallel with named tasks)
        # Run discovery tasks with explicit naming for cleaner result handling
        
        # Create named coroutines for parallel execution
        async def _discover_files():
            return await self._file_discoverer.discover(query, workspace_path, file_list, max_files)
        
        async def _search_docs():
            if search_docs:
                return await self._doc_searcher.search_web(query)
            return None
        
        async def _search_code():
            if workspace_path:
                return await self._code_searcher.search(query, workspace_path)
            return None
        
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
            converted_files = [
                FileContext(
                    path=f.path,
                    relevance_score=f.relevance_score,
                    summary=f.summary,
                    key_elements=f.key_elements,
                    line_count=f.line_count,
                    size_kb=f.size_kb,
                )
                for f in file_result
            ]

        # Process documentation search results
        doc_contexts = []
        if isinstance(doc_result, Exception):
            logger.warning("doc_search_failed", error=str(doc_result))
        elif doc_result:
            doc_contexts = [
                DocumentationContext(
                    source=d.source,
                    title=d.title,
                    snippet=d.snippet,
                    relevance_score=d.relevance_score,
                )
                for d in doc_result
            ]

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

        try:
            query_analysis = await self._query_analyzer.analyze(
                query,
                code_context,
                documentation_context=docs_context_str if docs_context_str else None
            )
        except Exception as e:
            logger.warning("query_analysis_failed_using_fallback", error=str(e))
            query_analysis = self._fallback_analyze(query)

        # Build structured context
        context = StructuredContext(
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
        )

        logger.info(
            "context_gateway_complete",
            task_type=context.task_type,
            files=len(context.relevant_files),
            docs=len(context.documentation),
            code_searches=len(context.code_search),
            framework=context.recommended_framework,
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
) -> StructuredContext:
    """
    Convenience function to prepare context.

    Example:
        context = await prepare_context_for_claude(
            query="Fix the auth bug",
            workspace_path="/my/project"
        )
        claude_prompt = context.to_claude_prompt()
    """
    gateway = get_context_gateway()
    return await gateway.prepare_context(
        query=query,
        workspace_path=workspace_path,
        code_context=code_context,
        file_list=file_list,
        search_docs=search_docs,
    )
