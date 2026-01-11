"""
Streaming Context Gateway for Real-Time Progress Updates

Extends ContextGateway to provide streaming progress events during context preparation.
Supports cancellation and completion time estimation.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

# Import ContextGateway - this is safe because context_gateway imports from context,
# but streaming_gateway is imported at the END of context/__init__.py after all other imports
from ..context_gateway import ContextGateway
from ..settings import get_settings
from .code_searcher import CodeSearchContext, CodeSearcher
from .context_cache import ContextCache
from .doc_searcher import DocumentationContext, DocumentationSearcher
from .enhanced_models import (
    ComponentMetrics,
    ComponentStatus,
    ComponentStatusInfo,
    ContextGatewayMetrics,
    EnhancedStructuredContext,
    ProgressEvent,
    ProgressStatus,
)
from .file_discoverer import FileContext, FileDiscoverer
from .query_analyzer import QueryAnalyzer

logger = structlog.get_logger("streaming_gateway")


class PerformanceTracker:
    """Tracks historical performance for completion time estimation."""

    def __init__(self):
        self._history: list[dict[str, Any]] = []
        self._max_history = 100

    def record_execution(self, component: str, workspace_size: int, execution_time: float) -> None:
        """Record a component execution for historical tracking."""
        self._history.append(
            {
                "component": component,
                "workspace_size": workspace_size,
                "execution_time": execution_time,
                "timestamp": datetime.now(),
            }
        )

        # Keep only recent history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def estimate_completion_time(self, component: str, workspace_size: int) -> float:
        """
        Estimate completion time for a component based on historical data.

        Args:
            component: Component name
            workspace_size: Size of workspace (number of files)

        Returns:
            Estimated time in seconds
        """
        # Filter history for this component
        component_history = [h for h in self._history if h["component"] == component]

        if not component_history:
            # No history, use defaults based on component type
            defaults = {
                "query_analysis": 2.0,
                "file_discovery": 5.0,
                "doc_search": 3.0,
                "code_search": 2.0,
            }
            return defaults.get(component, 3.0)

        # Use weighted average based on workspace size similarity
        total_weight = 0.0
        weighted_time = 0.0

        for h in component_history[-20:]:  # Use last 20 executions
            # Weight by workspace size similarity
            size_diff = abs(h["workspace_size"] - workspace_size)
            weight = 1.0 / (1.0 + size_diff / 100.0)  # Decay with size difference

            weighted_time += h["execution_time"] * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_time / total_weight
        else:
            return 3.0  # Default fallback


class StreamingContextGateway(ContextGateway):
    """
    Streaming version of ContextGateway with real-time progress updates.

    Features:
    - Real-time progress events for each component
    - Cancellation support with cleanup
    - Completion time estimation
    - Partial context on cancellation
    """

    def __init__(
        self,
        query_analyzer: QueryAnalyzer | None = None,
        file_discoverer: FileDiscoverer | None = None,
        doc_searcher: DocumentationSearcher | None = None,
        code_searcher: CodeSearcher | None = None,
        cache: ContextCache | None = None,
    ):
        """Initialize streaming context gateway."""
        super().__init__(
            query_analyzer=query_analyzer,
            file_discoverer=file_discoverer,
            doc_searcher=doc_searcher,
            code_searcher=code_searcher,
            cache=cache,
        )
        self._performance_tracker = PerformanceTracker()
        self._settings = get_settings()

    def _emit_progress(
        self,
        component: str,
        status: ProgressStatus,
        progress: float,
        data: Any = None,
        message: str | None = None,
        estimated_completion: float | None = None,
        callback: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        """
        Emit a progress event.

        Args:
            component: Component name
            status: Progress status
            progress: Progress value (0.0 to 1.0)
            data: Component-specific data
            message: Human-readable status message
            estimated_completion: Estimated seconds remaining
            callback: Progress callback function
        """
        if callback:
            event = ProgressEvent(
                component=component,
                status=status,
                progress=progress,
                data=data,
                timestamp=datetime.now(),
                estimated_completion=estimated_completion,
                message=message,
            )
            try:
                callback(event)
            except Exception as e:
                logger.warning("progress_callback_failed", component=component, error=str(e))

    async def prepare_context_streaming(  # noqa: C901, PLR0912, PLR0915
        self,
        query: str,
        progress_callback: Callable[[ProgressEvent], None],
        cancellation_token: asyncio.Event,
        workspace_path: str | None = None,
        code_context: str | None = None,
        file_list: list[str] | None = None,
        search_docs: bool = True,
        max_files: int = 15,
    ) -> EnhancedStructuredContext:
        """
        Prepare context with streaming progress updates.

        Args:
            query: The user's request
            progress_callback: Callback function for progress events
            cancellation_token: Event to signal cancellation
            workspace_path: Path to the workspace/project
            code_context: Any code snippets provided
            file_list: Pre-specified files to consider
            search_docs: Whether to search web for documentation
            max_files: Maximum files to include in context

        Returns:
            EnhancedStructuredContext with all metadata
        """
        logger.info("streaming_context_gateway_start", query=query[:100])

        start_time = time.time()
        component_metrics: list[ComponentMetrics] = []
        component_status: dict[str, ComponentStatusInfo] = {}

        # Calculate workspace size for estimation
        workspace_size = 0
        if workspace_path:
            try:
                workspace = Path(workspace_path)
                if workspace.exists():
                    workspace_size = sum(1 for _ in workspace.rglob("*.py"))
            except Exception:
                workspace_size = 100  # Default estimate

        # Estimate completion times for each component
        query_analysis_est = self._performance_tracker.estimate_completion_time(
            "query_analysis", workspace_size
        )
        file_discovery_est = self._performance_tracker.estimate_completion_time(
            "file_discovery", workspace_size
        )
        doc_search_est = self._performance_tracker.estimate_completion_time(
            "doc_search", workspace_size
        )
        code_search_est = self._performance_tracker.estimate_completion_time(
            "code_search", workspace_size
        )

        total_estimated = query_analysis_est + max(
            file_discovery_est, doc_search_est, code_search_est
        )

        # Emit initial progress event
        self._emit_progress(
            component="overall",
            status=ProgressStatus.STARTED,
            progress=0.0,
            message="Starting context preparation",
            estimated_completion=total_estimated,
            callback=progress_callback,
        )

        # Check for cancellation
        if cancellation_token.is_set():
            logger.info("context_preparation_cancelled_before_start")
            return self._create_partial_context(query, [], [], [], component_status)

        # Phase 1: Parallel Discovery with Streaming
        # Create tasks for parallel execution with progress tracking

        async def _discover_files_streaming():
            """File discovery with progress updates."""
            comp_start = time.time()
            self._emit_progress(
                component="file_discovery",
                status=ProgressStatus.STARTED,
                progress=0.0,
                message="Discovering relevant files",
                estimated_completion=file_discovery_est,
                callback=progress_callback,
            )

            try:
                # Check cache first
                if self._enable_cache and workspace_path:
                    file_cache_key = self._cache.generate_cache_key(
                        query, workspace_path, "file_discovery"
                    )
                    cached = await self._cache.get(file_cache_key, allow_stale=True)
                    if cached and not cached.is_expired:
                        self._emit_progress(
                            component="file_discovery",
                            status=ProgressStatus.COMPLETED,
                            progress=1.0,
                            data={"cache_hit": True},
                            message="Files loaded from cache",
                            callback=progress_callback,
                        )
                        comp_time = time.time() - comp_start
                        component_metrics.append(
                            ComponentMetrics(
                                component_name="file_discovery",
                                execution_time=comp_time,
                                api_calls_made=0,
                                tokens_consumed=0,
                                success=True,
                                cache_hit=True,
                            )
                        )
                        component_status["file_discovery"] = ComponentStatusInfo(
                            status=ComponentStatus.SUCCESS,
                            execution_time=comp_time,
                            api_calls_made=0,
                            tokens_consumed=0,
                        )
                        return cached.value

                # Check for cancellation
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("File discovery cancelled")

                # Emit progress during discovery
                self._emit_progress(
                    component="file_discovery",
                    status=ProgressStatus.PROGRESS,
                    progress=0.3,
                    message="Analyzing workspace structure",
                    callback=progress_callback,
                )

                # Perform discovery
                result = await self._file_discoverer.discover(
                    query, workspace_path, file_list, max_files
                )

                # Cache the result
                if self._enable_cache and workspace_path:
                    await self._cache.set(file_cache_key, result, "file_discovery", workspace_path)

                comp_time = time.time() - comp_start
                self._performance_tracker.record_execution(
                    "file_discovery", workspace_size, comp_time
                )

                # Estimate token usage for file discovery API call
                # Formula: prompt tokens (query + file listing) + response tokens (JSON output)
                estimated_prompt_tokens = (
                    len(query) // 4 + workspace_size // 4
                )  # ~4 chars per token
                estimated_response_tokens = len(result) * 50  # ~50 tokens per file result
                tokens_consumed = estimated_prompt_tokens + estimated_response_tokens

                self._emit_progress(
                    component="file_discovery",
                    status=ProgressStatus.COMPLETED,
                    progress=1.0,
                    data={"files_found": len(result)},
                    message=f"Found {len(result)} relevant files",
                    callback=progress_callback,
                )

                component_metrics.append(
                    ComponentMetrics(
                        component_name="file_discovery",
                        execution_time=comp_time,
                        api_calls_made=1,
                        tokens_consumed=tokens_consumed,
                        success=True,
                        cache_hit=False,
                    )
                )
                component_status["file_discovery"] = ComponentStatusInfo(
                    status=ComponentStatus.SUCCESS,
                    execution_time=comp_time,
                    api_calls_made=1,
                    tokens_consumed=tokens_consumed,
                )

                return result

            except asyncio.CancelledError:
                comp_time = time.time() - comp_start
                self._emit_progress(
                    component="file_discovery",
                    status=ProgressStatus.CANCELLED,
                    progress=0.5,
                    message="File discovery cancelled",
                    callback=progress_callback,
                )
                component_status["file_discovery"] = ComponentStatusInfo(
                    status=ComponentStatus.FAILED,
                    execution_time=comp_time,
                    error_message="Cancelled by user",
                )
                raise
            except Exception as e:
                comp_time = time.time() - comp_start
                logger.warning("file_discovery_failed", error=str(e))
                self._emit_progress(
                    component="file_discovery",
                    status=ProgressStatus.FAILED,
                    progress=0.0,
                    message=f"File discovery failed: {str(e)}",
                    callback=progress_callback,
                )
                component_status["file_discovery"] = ComponentStatusInfo(
                    status=ComponentStatus.FAILED, execution_time=comp_time, error_message=str(e)
                )
                return []

        async def _search_docs_streaming():
            """Documentation search with progress updates."""
            if not search_docs:
                return None

            comp_start = time.time()
            self._emit_progress(
                component="doc_search",
                status=ProgressStatus.STARTED,
                progress=0.0,
                message="Searching documentation",
                estimated_completion=doc_search_est,
                callback=progress_callback,
            )

            try:
                # Check cache first
                if self._enable_cache and workspace_path:
                    doc_cache_key = self._cache.generate_cache_key(
                        query, workspace_path, "documentation"
                    )
                    cached = await self._cache.get(doc_cache_key, allow_stale=True)
                    if cached and not cached.is_expired:
                        self._emit_progress(
                            component="doc_search",
                            status=ProgressStatus.COMPLETED,
                            progress=1.0,
                            data={"cache_hit": True},
                            message="Documentation loaded from cache",
                            callback=progress_callback,
                        )
                        comp_time = time.time() - comp_start
                        component_metrics.append(
                            ComponentMetrics(
                                component_name="doc_search",
                                execution_time=comp_time,
                                api_calls_made=0,
                                tokens_consumed=0,
                                success=True,
                                cache_hit=True,
                            )
                        )
                        component_status["doc_search"] = ComponentStatusInfo(
                            status=ComponentStatus.SUCCESS,
                            execution_time=comp_time,
                            api_calls_made=0,
                            tokens_consumed=0,
                        )
                        return cached.value

                # Check for cancellation
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Documentation search cancelled")

                # Emit progress
                self._emit_progress(
                    component="doc_search",
                    status=ProgressStatus.PROGRESS,
                    progress=0.5,
                    message="Fetching documentation snippets",
                    callback=progress_callback,
                )

                # Perform search
                result = await self._doc_searcher.search_web(query)

                # Cache the result
                if self._enable_cache and workspace_path:
                    await self._cache.set(doc_cache_key, result, "documentation", workspace_path)

                comp_time = time.time() - comp_start
                self._performance_tracker.record_execution("doc_search", workspace_size, comp_time)

                # Estimate token usage for documentation search
                # Formula: query tokens + doc context tokens + response tokens
                estimated_prompt_tokens = len(query) // 4 + workspace_size // 4
                doc_count = len(result) if result else 0
                estimated_response_tokens = doc_count * 100  # ~100 tokens per doc snippet
                tokens_consumed = estimated_prompt_tokens + estimated_response_tokens

                self._emit_progress(
                    component="doc_search",
                    status=ProgressStatus.COMPLETED,
                    progress=1.0,
                    data={"docs_found": doc_count},
                    message=f"Found {doc_count} documentation snippets",
                    callback=progress_callback,
                )

                component_metrics.append(
                    ComponentMetrics(
                        component_name="doc_search",
                        execution_time=comp_time,
                        api_calls_made=1,
                        tokens_consumed=tokens_consumed,
                        success=True,
                        cache_hit=False,
                    )
                )
                component_status["doc_search"] = ComponentStatusInfo(
                    status=ComponentStatus.SUCCESS,
                    execution_time=comp_time,
                    api_calls_made=1,
                    tokens_consumed=tokens_consumed,
                )

                return result

            except asyncio.CancelledError:
                comp_time = time.time() - comp_start
                self._emit_progress(
                    component="doc_search",
                    status=ProgressStatus.CANCELLED,
                    progress=0.5,
                    message="Documentation search cancelled",
                    callback=progress_callback,
                )
                component_status["doc_search"] = ComponentStatusInfo(
                    status=ComponentStatus.FAILED,
                    execution_time=comp_time,
                    error_message="Cancelled by user",
                )
                raise
            except Exception as e:
                comp_time = time.time() - comp_start
                logger.warning("doc_search_failed", error=str(e))
                self._emit_progress(
                    component="doc_search",
                    status=ProgressStatus.FAILED,
                    progress=0.0,
                    message=f"Documentation search failed: {str(e)}",
                    callback=progress_callback,
                )
                component_status["doc_search"] = ComponentStatusInfo(
                    status=ComponentStatus.FAILED, execution_time=comp_time, error_message=str(e)
                )
                return None

        async def _search_code_streaming():
            """Code search with progress updates."""
            if not workspace_path:
                return None

            comp_start = time.time()
            self._emit_progress(
                component="code_search",
                status=ProgressStatus.STARTED,
                progress=0.0,
                message="Searching codebase",
                estimated_completion=code_search_est,
                callback=progress_callback,
            )

            try:
                # Check for cancellation
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Code search cancelled")

                # Perform search
                result = await self._code_searcher.search(query, workspace_path)

                comp_time = time.time() - comp_start
                self._performance_tracker.record_execution("code_search", workspace_size, comp_time)

                # Estimate token usage for code search
                # Code search is typically local (ripgrep), but may use LLM for analysis
                # Estimate conservatively based on query and results
                search_count = len(result) if result else 0
                estimated_tokens = (
                    len(query) // 4 + search_count * 30
                )  # ~30 tokens per search result
                tokens_consumed = estimated_tokens

                self._emit_progress(
                    component="code_search",
                    status=ProgressStatus.COMPLETED,
                    progress=1.0,
                    data={"searches": search_count},
                    message=f"Completed {search_count} code searches",
                    callback=progress_callback,
                )

                component_metrics.append(
                    ComponentMetrics(
                        component_name="code_search",
                        execution_time=comp_time,
                        api_calls_made=1,
                        tokens_consumed=tokens_consumed,
                        success=True,
                        cache_hit=False,
                    )
                )
                component_status["code_search"] = ComponentStatusInfo(
                    status=ComponentStatus.SUCCESS,
                    execution_time=comp_time,
                    api_calls_made=1,
                    tokens_consumed=tokens_consumed,
                )

                return result

            except asyncio.CancelledError:
                comp_time = time.time() - comp_start
                self._emit_progress(
                    component="code_search",
                    status=ProgressStatus.CANCELLED,
                    progress=0.5,
                    message="Code search cancelled",
                    callback=progress_callback,
                )
                component_status["code_search"] = ComponentStatusInfo(
                    status=ComponentStatus.FAILED,
                    execution_time=comp_time,
                    error_message="Cancelled by user",
                )
                raise
            except Exception as e:
                comp_time = time.time() - comp_start
                logger.warning("code_search_failed", error=str(e))
                self._emit_progress(
                    component="code_search",
                    status=ProgressStatus.FAILED,
                    progress=0.0,
                    message=f"Code search failed: {str(e)}",
                    callback=progress_callback,
                )
                component_status["code_search"] = ComponentStatusInfo(
                    status=ComponentStatus.FAILED, execution_time=comp_time, error_message=str(e)
                )
                return None

        # Execute discovery tasks in parallel with cancellation support
        try:
            file_result, doc_result, code_result = await asyncio.gather(
                _discover_files_streaming(),
                _search_docs_streaming(),
                _search_code_streaming(),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            logger.info("context_preparation_cancelled_during_discovery")
            # Return partial context with what we have so far
            return self._create_partial_context(query, [], [], [], component_status)

        # Check for cancellation after discovery
        if cancellation_token.is_set():
            logger.info("context_preparation_cancelled_after_discovery")
            # Convert results to proper format
            converted_files = self._convert_file_results(file_result)
            doc_contexts = self._convert_doc_results(doc_result)
            code_search_results = self._convert_code_results(code_result)
            return self._create_partial_context(
                query, converted_files, doc_contexts, code_search_results, component_status
            )

        # Process results
        converted_files = self._convert_file_results(file_result)
        doc_contexts = self._convert_doc_results(doc_result)
        code_search_results = self._convert_code_results(code_result)

        # Phase 2: Query Analysis with Streaming
        comp_start = time.time()
        self._emit_progress(
            component="query_analysis",
            status=ProgressStatus.STARTED,
            progress=0.0,
            message="Analyzing query and planning execution",
            estimated_completion=query_analysis_est,
            callback=progress_callback,
        )

        try:
            # Check for cancellation
            if cancellation_token.is_set():
                raise asyncio.CancelledError("Query analysis cancelled")

            # Build documentation context string
            docs_context_str = ""
            if doc_contexts:
                docs_context_str += "DOCUMENTATION FOUND:\n"
                for d in doc_contexts[:3]:
                    docs_context_str += f"- {d.title} ({d.source}): {d.snippet[:200]}...\n"

            if converted_files:
                docs_context_str += "\nRELEVANT FILES:\n"
                for f in converted_files[:5]:
                    docs_context_str += f"- {f.path}: {f.summary}\n"

            # Check cache
            query_analysis = None
            if self._enable_cache and workspace_path:
                query_cache_key = self._cache.generate_cache_key(
                    query, workspace_path, "query_analysis"
                )
                cached = await self._cache.get(query_cache_key, allow_stale=True)
                if cached and not cached.is_expired:
                    query_analysis = cached.value
                    self._emit_progress(
                        component="query_analysis",
                        status=ProgressStatus.COMPLETED,
                        progress=1.0,
                        data={"cache_hit": True},
                        message="Query analysis loaded from cache",
                        callback=progress_callback,
                    )

            # Perform analysis if not cached
            if query_analysis is None:
                self._emit_progress(
                    component="query_analysis",
                    status=ProgressStatus.PROGRESS,
                    progress=0.5,
                    message="Understanding task requirements",
                    callback=progress_callback,
                )

                query_analysis = await self._query_analyzer.analyze(
                    query,
                    code_context,
                    documentation_context=docs_context_str if docs_context_str else None,
                )

                # Cache the result
                if self._enable_cache and workspace_path:
                    await self._cache.set(
                        query_cache_key, query_analysis, "query_analysis", workspace_path
                    )

            comp_time = time.time() - comp_start
            self._performance_tracker.record_execution("query_analysis", workspace_size, comp_time)

            # Estimate token usage for query analysis
            # Formula: query + code context + documentation context + response (structured JSON)
            estimated_prompt_tokens = len(query) // 4
            if code_context:
                estimated_prompt_tokens += len(code_context) // 4
            if docs_context_str:
                estimated_prompt_tokens += len(docs_context_str) // 4
            estimated_response_tokens = 200  # Structured JSON response with task analysis
            tokens_consumed = estimated_prompt_tokens + estimated_response_tokens

            self._emit_progress(
                component="query_analysis",
                status=ProgressStatus.COMPLETED,
                progress=1.0,
                data={"task_type": query_analysis.get("task_type", "general")},
                message=f"Task identified: {query_analysis.get('task_type', 'general')}",
                callback=progress_callback,
            )

            component_metrics.append(
                ComponentMetrics(
                    component_name="query_analysis",
                    execution_time=comp_time,
                    api_calls_made=1,
                    tokens_consumed=tokens_consumed,
                    success=True,
                    cache_hit=False,
                )
            )
            component_status["query_analysis"] = ComponentStatusInfo(
                status=ComponentStatus.SUCCESS,
                execution_time=comp_time,
                api_calls_made=1,
                tokens_consumed=tokens_consumed,
            )

        except asyncio.CancelledError:
            comp_time = time.time() - comp_start
            self._emit_progress(
                component="query_analysis",
                status=ProgressStatus.CANCELLED,
                progress=0.5,
                message="Query analysis cancelled",
                callback=progress_callback,
            )
            component_status["query_analysis"] = ComponentStatusInfo(
                status=ComponentStatus.FAILED,
                execution_time=comp_time,
                error_message="Cancelled by user",
            )
            # Return partial context
            return self._create_partial_context(
                query, converted_files, doc_contexts, code_search_results, component_status
            )
        except Exception as e:
            comp_time = time.time() - comp_start
            logger.warning("query_analysis_failed", error=str(e))
            self._emit_progress(
                component="query_analysis",
                status=ProgressStatus.FAILED,
                progress=0.0,
                message=f"Query analysis failed: {str(e)}",
                callback=progress_callback,
            )
            component_status["query_analysis"] = ComponentStatusInfo(
                status=ComponentStatus.FAILED,
                execution_time=comp_time,
                error_message=str(e),
                fallback_method="pattern_based",
            )
            # Use fallback
            query_analysis = self._fallback_analyze(query)

        # Build enhanced structured context
        total_time = time.time() - start_time

        gateway_metrics = ContextGatewayMetrics(
            total_execution_time=total_time,
            component_metrics=component_metrics,
            token_usage={},
            api_call_counts={},
        )

        context = EnhancedStructuredContext(
            task_type=query_analysis.get("task_type", "general"),
            task_summary=query_analysis.get("summary", query),
            complexity=query_analysis.get("complexity", "medium"),
            relevant_files=[],  # Will be populated below
            entry_point=query_analysis.get("entry_point"),
            documentation=[],  # Will be populated below
            code_search=code_search_results,
            recommended_framework=query_analysis.get("framework", "reason_flux"),
            framework_reason=query_analysis.get("framework_reason", "General-purpose reasoning"),
            chain_suggestion=query_analysis.get("chain"),
            execution_steps=query_analysis.get("steps", []),
            success_criteria=query_analysis.get("success_criteria", []),
            potential_blockers=query_analysis.get("blockers", []),
            related_patterns=query_analysis.get("patterns", []),
            dependencies=query_analysis.get("dependencies", []),
            component_status=component_status,
            gateway_metrics=gateway_metrics,
        )

        # Convert file contexts to enhanced format
        from .enhanced_models import EnhancedFileContext

        context.relevant_files = [
            EnhancedFileContext(
                path=f.path,
                relevance_score=f.relevance_score,
                summary=f.summary,
                key_elements=f.key_elements,
                line_count=f.line_count,
                size_kb=f.size_kb,
            )
            for f in converted_files
        ]

        # Convert documentation contexts to enhanced format
        from .enhanced_models import EnhancedDocumentationContext

        context.documentation = [
            EnhancedDocumentationContext(
                source=d.source,
                title=d.title,
                snippet=d.snippet,
                relevance_score=d.relevance_score,
            )
            for d in doc_contexts
        ]

        # Emit final progress event
        self._emit_progress(
            component="overall",
            status=ProgressStatus.COMPLETED,
            progress=1.0,
            message="Context preparation complete",
            callback=progress_callback,
        )

        logger.info(
            "streaming_context_gateway_complete",
            task_type=context.task_type,
            files=len(context.relevant_files),
            docs=len(context.documentation),
            code_searches=len(context.code_search),
            framework=context.recommended_framework,
            total_time=total_time,
        )

        return context

    def _convert_file_results(self, file_result: Any) -> list[FileContext]:
        """Convert file discovery results to FileContext list."""
        if isinstance(file_result, Exception):
            logger.warning("file_result_is_exception", error=str(file_result))
            return []
        if not file_result:
            return []

        return [
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

    def _convert_doc_results(self, doc_result: Any) -> list[DocumentationContext]:
        """Convert documentation search results to DocumentationContext list."""
        if isinstance(doc_result, Exception):
            logger.warning("doc_result_is_exception", error=str(doc_result))
            return []
        if not doc_result:
            return []

        return [
            DocumentationContext(
                source=d.source,
                title=d.title,
                snippet=d.snippet,
                relevance_score=d.relevance_score,
            )
            for d in doc_result
        ]

    def _convert_code_results(self, code_result: Any) -> list[CodeSearchContext]:
        """Convert code search results to CodeSearchContext list."""
        if isinstance(code_result, Exception):
            logger.warning("code_result_is_exception", error=str(code_result))
            return []
        if not code_result:
            return []

        return [
            CodeSearchContext(
                search_type=c.search_type,
                query=c.query,
                results=c.results,
                file_count=c.file_count,
                match_count=c.match_count,
            )
            for c in code_result
        ]

    def _create_partial_context(
        self,
        query: str,
        files: list[FileContext],
        docs: list[DocumentationContext],
        code_searches: list[CodeSearchContext],
        component_status: dict[str, ComponentStatusInfo],
    ) -> EnhancedStructuredContext:
        """
        Create partial context when cancelled or failed.

        Args:
            query: Original query
            files: Discovered files (may be empty)
            docs: Documentation results (may be empty)
            code_searches: Code search results (may be empty)
            component_status: Status of each component

        Returns:
            Partial EnhancedStructuredContext
        """
        from .enhanced_models import EnhancedDocumentationContext, EnhancedFileContext

        # Use fallback analysis
        fallback_analysis = self._fallback_analyze(query)

        context = EnhancedStructuredContext(
            task_type=fallback_analysis.get("task_type", "general"),
            task_summary=fallback_analysis.get("summary", query),
            complexity=fallback_analysis.get("complexity", "medium"),
            relevant_files=[
                EnhancedFileContext(
                    path=f.path,
                    relevance_score=f.relevance_score,
                    summary=f.summary,
                    key_elements=f.key_elements,
                    line_count=f.line_count,
                    size_kb=f.size_kb,
                )
                for f in files
            ],
            documentation=[
                EnhancedDocumentationContext(
                    source=d.source,
                    title=d.title,
                    snippet=d.snippet,
                    relevance_score=d.relevance_score,
                )
                for d in docs
            ],
            code_search=code_searches,
            recommended_framework=fallback_analysis.get("framework", "reason_flux"),
            framework_reason=fallback_analysis.get(
                "framework_reason", "Partial context due to cancellation"
            ),
            component_status=component_status,
        )

        return context


# Global singleton
_streaming_gateway: StreamingContextGateway | None = None


def get_streaming_context_gateway() -> StreamingContextGateway:
    """Get the global StreamingContextGateway singleton."""
    global _streaming_gateway
    if _streaming_gateway is None:
        _streaming_gateway = StreamingContextGateway()
    return _streaming_gateway
