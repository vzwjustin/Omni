"""
Performance tests for Enhanced Context Gateway.

Tests performance characteristics and optimization:
- Benchmark context preparation performance
- Validate streaming doesn't degrade performance
- Test multi-repository scaling
- Measure cache effectiveness
- Token budget optimization validation
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from app.core.context_gateway import ContextGateway, StructuredContext
from app.core.context.streaming_gateway import StreamingContextGateway
from app.core.context.multi_repo_discoverer import MultiRepoFileDiscoverer
from app.core.context.enhanced_models import ProgressEvent
from app.core.context.context_cache import get_context_cache


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_workspace():
    """Create a small workspace (5 files)."""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)
    
    for i in range(5):
        (workspace / f"file{i}.py").write_text(f"""
def function_{i}():
    return {i}
""")
    
    yield str(workspace)
    shutil.rmtree(temp_dir)


@pytest.fixture
def medium_workspace():
    """Create a medium workspace (20 files)."""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)
    
    for i in range(20):
        (workspace / f"file{i}.py").write_text(f"""
def function_{i}():
    return {i}

class Class{i}:
    def method(self):
        return {i}
""")
    
    yield str(workspace)
    shutil.rmtree(temp_dir)


@pytest.fixture
def large_workspace():
    """Create a large workspace (50 files)."""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)
    
    # Create directory structure
    for dir_idx in range(5):
        dir_path = workspace / f"module{dir_idx}"
        dir_path.mkdir()
        
        for file_idx in range(10):
            (dir_path / f"file{file_idx}.py").write_text(f"""
def function_{dir_idx}_{file_idx}():
    return {dir_idx * 10 + file_idx}

class Class{dir_idx}{file_idx}:
    def method(self):
        return {dir_idx * 10 + file_idx}
""")
    
    yield str(workspace)
    shutil.rmtree(temp_dir)


@pytest.fixture
def multi_repo_workspace_scaled():
    """Create a workspace with multiple repositories (scaled)."""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)
    
    # Create 3 repositories
    for repo_idx in range(3):
        repo = workspace / f"service-{repo_idx}"
        repo.mkdir()
        (repo / ".git").mkdir()
        
        # Create files in each repo
        for file_idx in range(5):
            (repo / f"module{file_idx}.py").write_text(f"""
from service_{(repo_idx + 1) % 3}.client import Client

def handler_{file_idx}():
    client = Client()
    return client.fetch()
""")
    
    yield str(workspace)
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
        "steps": ["Step 1", "Step 2", "Step 3"],
        "success_criteria": ["Criterion 1", "Criterion 2"],
        "blockers": ["Blocker 1"],
        "patterns": ["Pattern 1"],
        "dependencies": ["Dependency 1"],
    }


# =============================================================================
# Basic Performance Benchmarks
# =============================================================================

class TestBasicPerformance:
    """Test basic performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_small_workspace_performance(self, small_workspace, mock_gemini_response):
        """Benchmark context preparation for small workspace."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            gateway = ContextGateway()
            
            start_time = time.time()
            context = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=small_workspace,
                search_docs=False
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Should complete quickly for small workspace
            assert duration < 5.0, f"Small workspace took {duration:.2f}s (expected < 5s)"
            assert isinstance(context, StructuredContext)
    
    @pytest.mark.asyncio
    async def test_medium_workspace_performance(self, medium_workspace, mock_gemini_response):
        """Benchmark context preparation for medium workspace."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            gateway = ContextGateway()
            
            start_time = time.time()
            context = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=medium_workspace,
                search_docs=False
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Should complete reasonably for medium workspace
            assert duration < 10.0, f"Medium workspace took {duration:.2f}s (expected < 10s)"
            assert isinstance(context, StructuredContext)
    
    @pytest.mark.asyncio
    async def test_large_workspace_performance(self, large_workspace, mock_gemini_response):
        """Benchmark context preparation for large workspace."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            gateway = ContextGateway()
            
            start_time = time.time()
            context = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=large_workspace,
                search_docs=False
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Should complete within reasonable time for large workspace
            assert duration < 15.0, f"Large workspace took {duration:.2f}s (expected < 15s)"
            assert isinstance(context, StructuredContext)


# =============================================================================
# Streaming Performance Tests
# =============================================================================

class TestStreamingPerformance:
    """Test that streaming doesn't degrade performance."""
    
    @pytest.mark.asyncio
    async def test_streaming_vs_non_streaming_performance(self, medium_workspace, mock_gemini_response):
        """Compare streaming vs non-streaming performance."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            # Test non-streaming
            gateway = ContextGateway()
            start_time = time.time()
            context1 = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=medium_workspace,
                search_docs=False
            )
            non_streaming_duration = time.time() - start_time
            
            # Test streaming
            streaming_gateway = StreamingContextGateway()
            cancellation_token = asyncio.Event()
            
            start_time = time.time()
            context2 = await streaming_gateway.prepare_context_streaming(
                query="Fix the bug",
                progress_callback=lambda e: None,
                cancellation_token=cancellation_token,
                workspace_path=medium_workspace,
                search_docs=False
            )
            streaming_duration = time.time() - start_time
            
            # Streaming should not be significantly slower (allow 50% overhead)
            overhead_ratio = streaming_duration / non_streaming_duration if non_streaming_duration > 0 else 1.0
            assert overhead_ratio < 1.5, f"Streaming overhead: {overhead_ratio:.2f}x (expected < 1.5x)"
            
            assert isinstance(context1, StructuredContext)
            assert context2 is not None
    
    @pytest.mark.asyncio
    async def test_streaming_progress_overhead(self, medium_workspace, mock_gemini_response):
        """Test overhead of progress event emission."""
        progress_count = 0
        
        def progress_callback(event: ProgressEvent):
            nonlocal progress_count
            progress_count += 1
        
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            gateway = StreamingContextGateway()
            cancellation_token = asyncio.Event()
            
            start_time = time.time()
            context = await gateway.prepare_context_streaming(
                query="Fix the bug",
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
                workspace_path=medium_workspace,
                search_docs=False
            )
            duration = time.time() - start_time
            
            # Should emit multiple progress events
            assert progress_count > 0, "No progress events emitted"
            
            # Should still complete quickly
            assert duration < 10.0, f"Streaming took {duration:.2f}s (expected < 10s)"
            assert context is not None


# =============================================================================
# Multi-Repository Scaling Tests
# =============================================================================

class TestMultiRepoScaling:
    """Test multi-repository scaling characteristics."""
    
    @pytest.mark.asyncio
    async def test_multi_repo_parallel_analysis(self, multi_repo_workspace_scaled, mock_gemini_response):
        """Test that multi-repo analysis scales with parallelization."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            discoverer = MultiRepoFileDiscoverer()
            
            start_time = time.time()
            files, repos, deps = await discoverer.discover_multi_repo(
                query="Fix the API bug",
                workspace_path=multi_repo_workspace_scaled,
                max_files=15
            )
            duration = time.time() - start_time
            
            # Should complete in reasonable time despite multiple repos
            # Parallel execution should keep time similar to single repo
            assert duration < 15.0, f"Multi-repo discovery took {duration:.2f}s (expected < 15s)"
            
            # Should have detected multiple repos
            assert len(repos) >= 3, f"Expected 3 repos, found {len(repos)}"
    
    @pytest.mark.asyncio
    async def test_repo_count_scaling(self, mock_gemini_response):
        """Test scaling with increasing number of repositories."""
        durations = []
        repo_counts = [1, 2, 3]
        
        for count in repo_counts:
            # Create workspace with N repos
            temp_dir = tempfile.mkdtemp()
            workspace = Path(temp_dir)
            
            for i in range(count):
                repo = workspace / f"repo-{i}"
                repo.mkdir()
                (repo / ".git").mkdir()
                (repo / "file.py").write_text(f"# Repo {i}")
            
            with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
                 patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
                
                mock_analyze.return_value = mock_gemini_response
                mock_discover.return_value = []
                
                discoverer = MultiRepoFileDiscoverer()
                
                start_time = time.time()
                files, repos, deps = await discoverer.discover_multi_repo(
                    query="Fix bug",
                    workspace_path=str(workspace),
                    max_files=15
                )
                duration = time.time() - start_time
                durations.append(duration)
            
            shutil.rmtree(temp_dir)
        
        # Scaling should be sub-linear due to parallelization
        # 3 repos should not take 3x as long as 1 repo
        if len(durations) >= 3 and durations[0] > 0:
            scaling_factor = durations[2] / durations[0]
            assert scaling_factor < 2.5, f"Scaling factor: {scaling_factor:.2f}x (expected < 2.5x)"


# =============================================================================
# Cache Performance Tests
# =============================================================================

class TestCachePerformance:
    """Test cache effectiveness and performance impact."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, medium_workspace, mock_gemini_response):
        """Test that cache hits are significantly faster."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            gateway = ContextGateway()
            
            # First call - cache miss
            start_time = time.time()
            context1 = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=medium_workspace,
                search_docs=False
            )
            miss_duration = time.time() - start_time
            
            # Second call - cache hit
            start_time = time.time()
            context2 = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=medium_workspace,
                search_docs=False
            )
            hit_duration = time.time() - start_time
            
            # Cache hit should be faster (or at least not slower)
            # Allow some variance due to test environment
            assert hit_duration <= miss_duration * 1.2, \
                f"Cache hit ({hit_duration:.2f}s) slower than miss ({miss_duration:.2f}s)"
            
            assert isinstance(context1, StructuredContext)
            assert isinstance(context2, StructuredContext)
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness_metrics(self, medium_workspace, mock_gemini_response):
        """Test cache effectiveness tracking."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            gateway = ContextGateway()
            cache = get_context_cache()
            
            # Clear cache
            cache._cache.clear()
            
            # Make multiple calls
            for i in range(5):
                await gateway.prepare_context(
                    query="Fix the bug",
                    workspace_path=medium_workspace,
                    search_docs=False
                )
            
            # Cache should have entries
            assert len(cache._cache) > 0, "Cache should have entries after multiple calls"


# =============================================================================
# Token Budget Performance Tests
# =============================================================================

class TestTokenBudgetPerformance:
    """Test token budget management performance."""
    
    @pytest.mark.asyncio
    async def test_token_budget_optimization_overhead(self, medium_workspace, mock_gemini_response):
        """Test overhead of token budget optimization."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = [
                MagicMock(
                    path=f"file{i}.py",
                    relevance_score=0.9 - (i * 0.1),
                    summary=f"File {i} summary",
                    key_elements=[f"func{i}"],
                    line_count=100,
                    size_kb=5.0
                )
                for i in range(20)
            ]
            
            gateway = ContextGateway()
            
            start_time = time.time()
            context = await gateway.prepare_context(
                query="Fix the bug",
                workspace_path=medium_workspace,
                search_docs=False,
                max_files=10  # Limit files to test budget optimization
            )
            duration = time.time() - start_time
            
            # Should complete quickly even with budget optimization
            assert duration < 10.0, f"Token budget optimization took {duration:.2f}s (expected < 10s)"
            
            # Should have limited files based on max_files
            assert len(context.relevant_files) <= 10


# =============================================================================
# Concurrent Request Performance Tests
# =============================================================================

class TestConcurrentPerformance:
    """Test performance under concurrent requests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_context_preparation(self, medium_workspace, mock_gemini_response):
        """Test handling of concurrent context preparation requests."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            gateway = ContextGateway()
            
            # Create multiple concurrent requests
            async def prepare_context_task(query_id: int):
                return await gateway.prepare_context(
                    query=f"Fix bug {query_id}",
                    workspace_path=medium_workspace,
                    search_docs=False
                )
            
            start_time = time.time()
            results = await asyncio.gather(*[
                prepare_context_task(i) for i in range(5)
            ])
            duration = time.time() - start_time
            
            # Should handle concurrent requests efficiently
            # 5 concurrent requests should not take 5x as long
            assert duration < 15.0, f"Concurrent requests took {duration:.2f}s (expected < 15s)"
            
            # All requests should succeed
            assert len(results) == 5
            assert all(isinstance(r, StructuredContext) for r in results)


# =============================================================================
# Memory Usage Tests
# =============================================================================

class TestMemoryPerformance:
    """Test memory usage characteristics."""
    
    @pytest.mark.asyncio
    async def test_cache_memory_bounds(self, medium_workspace, mock_gemini_response):
        """Test that cache respects memory bounds."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            gateway = ContextGateway()
            cache = get_context_cache()
            
            # Clear cache
            cache._cache.clear()
            
            # Make many unique requests
            for i in range(100):
                await gateway.prepare_context(
                    query=f"Fix bug number {i}",
                    workspace_path=medium_workspace,
                    search_docs=False
                )
            
            # Cache should not grow unbounded
            # (Actual limit depends on implementation)
            cache_size = len(cache._cache)
            assert cache_size < 200, f"Cache size {cache_size} may be unbounded"


# =============================================================================
# Performance Regression Tests
# =============================================================================

class TestPerformanceRegression:
    """Test for performance regressions."""
    
    @pytest.mark.asyncio
    async def test_baseline_performance(self, medium_workspace, mock_gemini_response):
        """Establish baseline performance metrics."""
        with patch('app.core.context.query_analyzer.QueryAnalyzer.analyze') as mock_analyze, \
             patch('app.core.context.file_discoverer.FileDiscoverer.discover') as mock_discover:
            
            mock_analyze.return_value = mock_gemini_response
            mock_discover.return_value = []
            
            gateway = ContextGateway()
            
            # Run multiple iterations to get average
            durations = []
            for _ in range(3):
                start_time = time.time()
                await gateway.prepare_context(
                    query="Fix the bug",
                    workspace_path=medium_workspace,
                    search_docs=False
                )
                durations.append(time.time() - start_time)
            
            avg_duration = sum(durations) / len(durations)
            
            # Baseline should be under 10 seconds
            assert avg_duration < 10.0, f"Baseline performance: {avg_duration:.2f}s (expected < 10s)"
            
            # Log baseline for future comparison
            print(f"\nBaseline performance: {avg_duration:.2f}s (min: {min(durations):.2f}s, max: {max(durations):.2f}s)")
