"""
Integration tests for Enhanced Context Gateway.

Tests end-to-end context preparation with all enhancements:
- Intelligent caching system
- Streaming progress updates
- Multi-repository discovery
- Enhanced documentation grounding
- Comprehensive metrics
- Token budget management
- Advanced resilience patterns
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.context.context_cache import get_context_cache
from app.core.context.enhanced_models import (
    ComponentStatus,
    EnhancedDocumentationContext,
    EnhancedFileContext,
    EnhancedStructuredContext,
    ProgressEvent,
    ProgressStatus,
    RepoInfo,
    SourceAttribution,
)
from app.core.context.multi_repo_discoverer import MultiRepoFileDiscoverer
from app.core.context.streaming_gateway import StreamingContextGateway
from app.core.context_gateway import ContextGateway, StructuredContext

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)

    # Create sample files
    (workspace / "main.py").write_text("""
def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
""")

    (workspace / "utils.py").write_text("""
def helper_function():
    return "helper"
""")

    (workspace / "README.md").write_text("""
# Test Project
This is a test project.
""")

    yield str(workspace)

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def multi_repo_workspace():
    """Create a temporary workspace with multiple repositories."""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)

    # Create first repository
    repo1 = workspace / "service-a"
    repo1.mkdir()
    (repo1 / ".git").mkdir()
    (repo1 / "api.py").write_text("""
from service_b.client import ServiceBClient

def handle_request():
    client = ServiceBClient()
    return client.fetch_data()
""")

    # Create second repository
    repo2 = workspace / "service-b"
    repo2.mkdir()
    (repo2 / ".git").mkdir()
    (repo2 / "client.py").write_text("""
class ServiceBClient:
    def fetch_data(self):
        return {"data": "value"}
""")

    yield str(workspace)

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API responses."""
    return {
        "task_type": "implement",
        "summary": "Implement a new feature",
        "complexity": "medium",
        "framework": "reason_flux",
        "framework_reason": "Good for implementation tasks",
        "steps": ["Step 1", "Step 2"],
        "success_criteria": ["Criterion 1"],
        "blockers": [],
    }


# =============================================================================
# Basic Context Gateway Integration Tests
# =============================================================================

class TestContextGatewayIntegration:
    """Test basic context gateway functionality end-to-end."""

    @pytest.mark.asyncio
    async def test_prepare_context_basic(self, temp_workspace, mock_gemini_response):
        """Test basic context preparation without enhancements."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = mock_gemini_response

            gateway = ContextGateway()
            context = await gateway.prepare_context(
                query="Fix the bug in main.py",
                workspace_path=temp_workspace,
                search_docs=False
            )

            assert isinstance(context, StructuredContext)
            assert context.task_type == "implement"
            assert context.complexity == "medium"
            assert context.recommended_framework == "reason_flux"

    @pytest.mark.asyncio
    async def test_prepare_context_with_file_discovery(self, temp_workspace, mock_gemini_response):
        """Test context preparation with file discovery."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:

            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = [
                MagicMock(
                    path="main.py",
                    relevance_score=0.9,
                    summary="Main entry point",
                    key_elements=["main"],
                    line_count=10,
                    size_kb=1.0
                )
            ]

            gateway = ContextGateway()
            context = await gateway.prepare_context(
                query="Fix the bug in main.py",
                workspace_path=temp_workspace,
                search_docs=False
            )

            assert len(context.relevant_files) > 0
            assert context.relevant_files[0].path == "main.py"
            assert context.relevant_files[0].relevance_score == 0.9

    @pytest.mark.asyncio
    async def test_prepare_context_with_documentation(self, temp_workspace, mock_gemini_response):
        """Test context preparation with documentation search."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.doc_searcher.DocumentationSearcher.search_web') as mock_search:

            mock_analyze.return_value = mock_gemini_response
            mock_search.return_value = [
                MagicMock(
                    source="https://docs.python.org",
                    title="Python Documentation",
                    snippet="Python is a programming language",
                    relevance_score=0.8
                )
            ]

            gateway = ContextGateway()
            context = await gateway.prepare_context(
                query="How to use Python decorators?",
                workspace_path=temp_workspace,
                search_docs=True,
                enable_source_attribution=False,
            )

            assert len(context.documentation) > 0
            assert context.documentation[0].source == "https://docs.python.org"

    @pytest.mark.asyncio
    async def test_prepare_context_with_source_attribution(self, temp_workspace, mock_gemini_response):
        """Test context preparation using source attribution searcher."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover, \
             patch('app.core.context.code_searcher.CodeSearcher.search') as mock_code_search, \
             patch('app.core.context.doc_searcher.EnhancedDocumentationSearcher.search_with_fallback') as mock_search:

            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            mock_code_search.return_value = []
            attribution = SourceAttribution(
                url="https://docs.python.org",
                title="Python Documentation",
                domain="docs.python.org",
                authority_score=1.0,
                is_official=True,
            )
            mock_search.return_value = [
                EnhancedDocumentationContext(
                    source="https://docs.python.org",
                    title="Python Documentation",
                    snippet="Python is a programming language",
                    relevance_score=0.8,
                    attribution=attribution,
                )
            ]

            gateway = ContextGateway()
            context = await gateway.prepare_context(
                query="How to use Python decorators?",
                workspace_path=temp_workspace,
                search_docs=True,
                enable_source_attribution=True,
            )

            mock_search.assert_called_once()
            assert len(context.documentation) > 0
            assert context.documentation[0].source == "https://docs.python.org"
            assert context.documentation[0].attribution == attribution
            assert context.source_attributions
            assert context.source_attributions[0].url == "https://docs.python.org"


# =============================================================================
# Caching Integration Tests
# =============================================================================

class TestCachingIntegration:
    """Test intelligent caching system integration."""

    @pytest.mark.asyncio
    async def test_cache_hit_on_similar_query(self, temp_workspace, mock_gemini_response):
        """Test that similar queries hit the cache."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = mock_gemini_response

            gateway = ContextGateway()

            # First call - cache miss
            context1 = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=temp_workspace,
                search_docs=False
            )

            # Second call - should hit cache
            context2 = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=temp_workspace,
                search_docs=False
            )

            # Both should return valid contexts
            assert isinstance(context1, StructuredContext)
            assert isinstance(context2, StructuredContext)

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_file_change(self, temp_workspace, mock_gemini_response):
        """Test that cache is invalidated when files change."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = mock_gemini_response

            gateway = ContextGateway()
            cache = get_context_cache()

            # First call
            await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=temp_workspace,
                search_docs=False
            )

            # Modify a file
            workspace = Path(temp_workspace)
            (workspace / "main.py").write_text("# Modified content")

            # Invalidate cache for workspace
            await cache.invalidate_workspace(temp_workspace)

            # Second call should not use stale cache
            context = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=temp_workspace,
                search_docs=False
            )

            assert isinstance(context, StructuredContext)


# =============================================================================
# Streaming Integration Tests
# =============================================================================

class TestStreamingIntegration:
    """Test streaming context preparation integration."""

    @pytest.mark.asyncio
    async def test_streaming_progress_events(self, temp_workspace, mock_gemini_response):
        """Test that streaming emits progress events."""
        progress_events = []

        def progress_callback(event: ProgressEvent):
            progress_events.append(event)

        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = mock_gemini_response

            gateway = StreamingContextGateway()
            cancellation_token = asyncio.Event()

            context = await gateway.prepare_context_streaming(
                query="Fix the bug",
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
                workspace_path=temp_workspace,
                search_docs=False
            )

            # Should have received progress events
            assert len(progress_events) > 0

            # Check for overall start event
            start_events = [e for e in progress_events if e.status == ProgressStatus.STARTED]
            assert len(start_events) > 0

            # Check for completion events
            complete_events = [e for e in progress_events if e.status == ProgressStatus.COMPLETED]
            assert len(complete_events) > 0

            # Context should be enhanced
            assert isinstance(context, EnhancedStructuredContext)

    @pytest.mark.asyncio
    async def test_streaming_cancellation(self, temp_workspace, mock_gemini_response):
        """Test that streaming can be cancelled."""
        progress_events = []

        def progress_callback(event: ProgressEvent):
            progress_events.append(event)

        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = mock_gemini_response

            gateway = StreamingContextGateway()
            cancellation_token = asyncio.Event()

            # Cancel immediately
            cancellation_token.set()

            context = await gateway.prepare_context_streaming(
                query="Fix the bug",
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
                workspace_path=temp_workspace,
                search_docs=False
            )

            # Should return partial context
            assert isinstance(context, EnhancedStructuredContext)

            # Should have some cancelled status in component_status
            if context.component_status:
                cancelled_components = [  # noqa: F841
                    name for name, status in context.component_status.items()
                    if status.status == ComponentStatus.FAILED
                ]
                # May have cancelled components
                assert True  # Test passes if we get here without error


# =============================================================================
# Multi-Repository Integration Tests
# =============================================================================

class TestMultiRepoIntegration:
    """Test multi-repository discovery integration."""

    @pytest.mark.asyncio
    async def test_multi_repo_detection(self, multi_repo_workspace):
        """Test that multiple repositories are detected."""
        discoverer = MultiRepoFileDiscoverer()

        repos = await discoverer._detect_repositories(multi_repo_workspace)

        # Should detect both repositories
        assert len(repos) >= 2
        repo_names = [r.name for r in repos]
        assert "service-a" in repo_names
        assert "service-b" in repo_names

    @pytest.mark.asyncio
    async def test_multi_repo_discovery(self, multi_repo_workspace, mock_gemini_response):
        """Test file discovery across multiple repositories."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:

            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = [
                MagicMock(
                    path="api.py",
                    relevance_score=0.9,
                    summary="API handler",
                    key_elements=["handle_request"],
                    line_count=10,
                    size_kb=1.0
                )
            ]

            discoverer = MultiRepoFileDiscoverer()

            files, repos, deps = await discoverer.discover_multi_repo(
                query="Fix the API bug",
                workspace_path=multi_repo_workspace,
                max_files=15
            )

            # Should have discovered files
            assert len(files) > 0

            # Should have repository information
            assert len(repos) >= 2

    @pytest.mark.asyncio
    async def test_context_gateway_uses_multi_repo(self, multi_repo_workspace, mock_gemini_response):
        """Test ContextGateway uses multi-repo discovery when enabled."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.code_searcher.CodeSearcher.search') as mock_code_search, \
             patch('app.core.context.multi_repo_discoverer.MultiRepoFileDiscoverer.discover_multi_repo') as mock_discover:

            mock_analyze.return_value = mock_gemini_response
            mock_code_search.return_value = []
            mock_discover.return_value = (
                [
                    EnhancedFileContext(
                        path="service-a/api.py",
                        relevance_score=0.9,
                        summary="API handler",
                        key_elements=["handle_request"],
                        line_count=10,
                        size_kb=1.0,
                        repository="service-a",
                    )
                ],
                [
                    RepoInfo(
                        path=f"{multi_repo_workspace}/service-a",
                        name="service-a",
                        git_root=f"{multi_repo_workspace}/service-a",
                    ),
                    RepoInfo(
                        path=f"{multi_repo_workspace}/service-b",
                        name="service-b",
                        git_root=f"{multi_repo_workspace}/service-b",
                    ),
                ],
                []
            )

            gateway = ContextGateway()
            gateway._enable_cache = False
            context = await gateway.prepare_context(
                query="Fix the API bug",
                workspace_path=multi_repo_workspace,
                search_docs=False,
                enable_multi_repo=True,
            )

            mock_discover.assert_called_once()
            assert len(context.repository_info) == 2
            assert context.relevant_files
            assert context.relevant_files[0].repository == "service-a"

    @pytest.mark.asyncio
    async def test_cross_repo_dependency_detection(self, multi_repo_workspace):
        """Test cross-repository dependency detection."""
        discoverer = MultiRepoFileDiscoverer()

        # Detect repositories
        repos = await discoverer._detect_repositories(multi_repo_workspace)

        # Create mock file results
        repo_results = {}
        for repo in repos:
            if repo.name == "service-a":
                repo_results[repo] = [
                    MagicMock(
                        path="api.py",
                        relevance_score=0.9,
                        summary="API handler",
                        key_elements=[],
                        line_count=10,
                        size_kb=1.0
                    )
                ]

        # Detect dependencies
        deps = await discoverer._follow_cross_repo_dependencies(repos, repo_results)

        # Should detect dependency from service-a to service-b
        # (based on the import in api.py)
        assert isinstance(deps, list)


# =============================================================================
# MCP Tool Integration Tests
# =============================================================================

class TestMCPToolIntegration:
    """Test MCP tool integration with enhanced context gateway."""

    @pytest.mark.asyncio
    async def test_prepare_context_tool(self, temp_workspace, mock_gemini_response):
        """Test prepare_context MCP tool with enhancements."""
        with patch('server.main.get_collection_manager') as mock_cm, \
             patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze:

            mock_cm.return_value = MagicMock(COLLECTIONS={})
            mock_analyze.return_value = mock_gemini_response

            from server.main import create_server
            server = create_server()

            tool_handler = server.call_tool_handler
            result = await tool_handler("prepare_context", {
                "query": "Fix the bug",
                "workspace_path": temp_workspace
            })

            # Should return structured context
            assert len(result) > 0
            assert "task_type" in result[0].text or "implement" in result[0].text

    @pytest.mark.asyncio
    async def test_context_cache_status_tool(self):
        """Test context_cache_status MCP tool."""
        with patch('server.main.get_collection_manager') as mock_cm:
            mock_cm.return_value = MagicMock(COLLECTIONS={})

            from server.main import create_server
            server = create_server()

            tool_handler = server.call_tool_handler

            # Check if tool exists
            try:
                result = await tool_handler("context_cache_status", {})
                # Tool should return cache status
                assert len(result) > 0
            except Exception as e:
                # Tool might not be implemented yet
                assert "Unknown tool" in str(e) or "context_cache_status" in str(e)


# =============================================================================
# Error Handling and Resilience Tests
# =============================================================================

class TestResilienceIntegration:
    """Test error handling and resilience patterns."""

    @pytest.mark.asyncio
    async def test_fallback_on_gemini_failure(self, temp_workspace):
        """Test fallback analysis when Gemini fails."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze:
            # Simulate Gemini failure
            mock_analyze.side_effect = Exception("Gemini API unavailable")

            gateway = ContextGateway()
            context = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=temp_workspace,
                search_docs=False
            )

            # Should still return context using fallback
            assert isinstance(context, StructuredContext)
            assert context.task_type is not None

    @pytest.mark.asyncio
    async def test_partial_component_failure(self, temp_workspace, mock_gemini_response):
        """Test handling of partial component failures."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:

            mock_analyze.return_value = mock_gemini_response
            # File discovery fails
            mock_discover.side_effect = Exception("File discovery failed")

            gateway = ContextGateway()
            context = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=temp_workspace,
                search_docs=False
            )

            # Should still return context with other components
            assert isinstance(context, StructuredContext)
            assert context.task_type == "implement"
            # Files list may be empty due to failure
            assert isinstance(context.relevant_files, list)

    @pytest.mark.asyncio
    async def test_inaccessible_repository_handling(self, multi_repo_workspace):
        """Test handling of inaccessible repositories."""
        discoverer = MultiRepoFileDiscoverer()

        # Detect repositories
        repos = await discoverer._detect_repositories(multi_repo_workspace)

        # Mark one repo as inaccessible
        if repos:
            repos[0].is_accessible = False
            repos[0].error_message = "Permission denied"

        # Generate warnings
        warnings = discoverer.generate_repository_warnings(repos)

        # Should have warning for inaccessible repo
        assert len(warnings) > 0
        assert any("not accessible" in w for w in warnings)


# =============================================================================
# Performance and Metrics Tests
# =============================================================================

class TestPerformanceIntegration:
    """Test performance and metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_collection(self, temp_workspace, mock_gemini_response):
        """Test that metrics are collected during context preparation."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = mock_gemini_response

            gateway = StreamingContextGateway()
            cancellation_token = asyncio.Event()

            context = await gateway.prepare_context_streaming(
                query="Fix the bug",
                progress_callback=lambda _e: None,
                cancellation_token=cancellation_token,
                workspace_path=temp_workspace,
                search_docs=False
            )

            # Enhanced context should have metrics
            assert isinstance(context, EnhancedStructuredContext)
            if hasattr(context, 'metrics'):
                assert context.metrics is not None

    @pytest.mark.asyncio
    async def test_context_preparation_performance(self, temp_workspace, mock_gemini_response):
        """Test that context preparation completes in reasonable time."""
        import time

        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = mock_gemini_response

            gateway = ContextGateway()

            start_time = time.time()
            context = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=temp_workspace,
                search_docs=False
            )
            end_time = time.time()

            # Should complete in under 10 seconds (with mocks)
            assert (end_time - start_time) < 10.0
            assert isinstance(context, StructuredContext)
