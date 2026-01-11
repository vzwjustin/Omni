"""
Unit tests for EnhancedDocumentationSearcher.

Tests the enhanced documentation searcher with source attribution,
intelligent merging, and authority-based prioritization.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.context.doc_searcher import (
    DocumentationContext,
    EnhancedDocumentationSearcher,
)
from app.core.context.enhanced_models import EnhancedDocumentationContext, SourceAttribution


class TestEnhancedDocumentationSearcher:
    """Unit tests for EnhancedDocumentationSearcher."""

    def test_calculate_authority_score_official(self):
        """Test authority score calculation for official documentation."""
        searcher = EnhancedDocumentationSearcher()

        # Official domains should get score of 1.0
        assert searcher._calculate_authority_score("docs.python.org") == 1.0
        assert searcher._calculate_authority_score("reactjs.org") == 1.0
        assert searcher._calculate_authority_score("golang.org") == 1.0

    def test_calculate_authority_score_high_authority(self):
        """Test authority score calculation for high authority domains."""
        searcher = EnhancedDocumentationSearcher()

        # High authority domains should get score of 0.85
        assert searcher._calculate_authority_score("stackoverflow.com") == 0.85
        assert searcher._calculate_authority_score("github.com") == 0.85

    def test_calculate_authority_score_documentation_pattern(self):
        """Test authority score for domains with documentation patterns."""
        searcher = EnhancedDocumentationSearcher()

        # Domains with 'docs.' pattern should get 0.75
        assert searcher._calculate_authority_score("docs.example.com") == 0.75
        assert searcher._calculate_authority_score("documentation.example.com") == 0.75
        assert searcher._calculate_authority_score("developer.example.com") == 0.75

    def test_calculate_authority_score_github_repo(self):
        """Test authority score for GitHub repositories."""
        searcher = EnhancedDocumentationSearcher()

        # GitHub repos (not docs) should get 0.7
        assert searcher._calculate_authority_score("github.com/user/repo") == 0.7

    def test_calculate_authority_score_default(self):
        """Test authority score for unknown domains."""
        searcher = EnhancedDocumentationSearcher()

        # Unknown domains should get default score of 0.5
        assert searcher._calculate_authority_score("example.com") == 0.5
        assert searcher._calculate_authority_score("random-blog.net") == 0.5

    def test_extract_grounding_metadata_no_metadata(self):
        """Test grounding metadata extraction when no metadata is present."""
        searcher = EnhancedDocumentationSearcher()

        # Mock response without grounding metadata
        mock_response = Mock()
        mock_response.candidates = []

        attributions = searcher._extract_grounding_metadata(mock_response)
        assert attributions == []

    def test_extract_grounding_metadata_with_web_chunks(self):
        """Test grounding metadata extraction with web chunks."""
        searcher = EnhancedDocumentationSearcher()

        # Mock response with grounding chunks
        mock_web = Mock()
        mock_web.uri = "https://docs.python.org/3/library/asyncio.html"
        mock_web.title = "asyncio — Asynchronous I/O"

        mock_chunk = Mock()
        mock_chunk.web = mock_web

        mock_metadata = Mock()
        mock_metadata.grounding_chunks = [mock_chunk]

        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_metadata

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        attributions = searcher._extract_grounding_metadata(mock_response)

        assert len(attributions) == 1
        assert attributions[0].url == "https://docs.python.org/3/library/asyncio.html"
        assert attributions[0].title == "asyncio — Asynchronous I/O"
        assert attributions[0].domain == "docs.python.org"
        assert attributions[0].authority_score == 1.0  # Official Python docs
        assert attributions[0].is_official is True

    @pytest.mark.asyncio
    async def test_merge_web_and_local_results_deduplication(self):
        """Test that merging deduplicates similar content."""
        searcher = EnhancedDocumentationSearcher()

        # Create web results
        web_results = [
            EnhancedDocumentationContext(
                source="https://docs.python.org/3/library/asyncio.html",
                title="asyncio Documentation",
                snippet="asyncio is a library to write concurrent code using the async/await syntax.",
                relevance_score=0.9,
                attribution=SourceAttribution(
                    url="https://docs.python.org/3/library/asyncio.html",
                    title="asyncio Documentation",
                    domain="docs.python.org",
                    authority_score=1.0,
                    is_official=True
                ),
                merge_source="web"
            )
        ]

        # Create local results with similar content (should be deduplicated)
        local_results = [
            DocumentationContext(
                source="Local Knowledge Base",
                title="asyncio Guide",
                snippet="asyncio is a library to write concurrent code using the async/await syntax.",
                relevance_score=0.8
            )
        ]

        merged = await searcher._merge_web_and_local_results(web_results, local_results)

        # Should only have one result due to deduplication
        assert len(merged) == 1
        assert merged[0].source == "https://docs.python.org/3/library/asyncio.html"

    @pytest.mark.asyncio
    async def test_merge_web_and_local_results_different_content(self):
        """Test that merging keeps different content."""
        searcher = EnhancedDocumentationSearcher()

        # Create web results
        web_results = [
            EnhancedDocumentationContext(
                source="https://docs.python.org/3/library/asyncio.html",
                title="asyncio Documentation",
                snippet="asyncio is a library to write concurrent code using the async/await syntax.",
                relevance_score=0.9,
                attribution=SourceAttribution(
                    url="https://docs.python.org/3/library/asyncio.html",
                    title="asyncio Documentation",
                    domain="docs.python.org",
                    authority_score=1.0,
                    is_official=True
                ),
                merge_source="web"
            )
        ]

        # Create local results with different content
        local_results = [
            DocumentationContext(
                source="Local Knowledge Base",
                title="Threading Guide",
                snippet="Threading is a different approach to concurrency using OS threads.",
                relevance_score=0.7
            )
        ]

        merged = await searcher._merge_web_and_local_results(web_results, local_results)

        # Should have both results
        assert len(merged) == 2
        # Official docs should be first due to higher authority
        assert merged[0].source == "https://docs.python.org/3/library/asyncio.html"

    def test_prioritize_by_authority_official_first(self):
        """Test that official documentation is prioritized first."""
        searcher = EnhancedDocumentationSearcher()

        results = [
            EnhancedDocumentationContext(
                source="https://stackoverflow.com/questions/123",
                title="Stack Overflow Answer",
                snippet="Some answer",
                relevance_score=0.95,
                attribution=SourceAttribution(
                    url="https://stackoverflow.com/questions/123",
                    title="Stack Overflow Answer",
                    domain="stackoverflow.com",
                    authority_score=0.85,
                    is_official=False
                ),
                merge_source="web"
            ),
            EnhancedDocumentationContext(
                source="https://docs.python.org/3/",
                title="Python Documentation",
                snippet="Official docs",
                relevance_score=0.8,
                attribution=SourceAttribution(
                    url="https://docs.python.org/3/",
                    title="Python Documentation",
                    domain="docs.python.org",
                    authority_score=1.0,
                    is_official=True
                ),
                merge_source="web"
            )
        ]

        prioritized = searcher._prioritize_by_authority(results)

        # Official docs should be first despite lower relevance score
        assert len(prioritized) == 2
        assert prioritized[0].source == "https://docs.python.org/3/"
        assert prioritized[1].source == "https://stackoverflow.com/questions/123"

    def test_prioritize_by_authority_categories(self):
        """Test that results are categorized correctly by authority."""
        searcher = EnhancedDocumentationSearcher()

        results = [
            # Other source
            EnhancedDocumentationContext(
                source="https://random-blog.com/post",
                title="Blog Post",
                snippet="Some content",
                relevance_score=0.9,
                attribution=SourceAttribution(
                    url="https://random-blog.com/post",
                    title="Blog Post",
                    domain="random-blog.com",
                    authority_score=0.5,
                    is_official=False
                ),
                merge_source="web"
            ),
            # High authority
            EnhancedDocumentationContext(
                source="https://github.com/user/repo",
                title="GitHub Repo",
                snippet="Repo content",
                relevance_score=0.85,
                attribution=SourceAttribution(
                    url="https://github.com/user/repo",
                    title="GitHub Repo",
                    domain="github.com",
                    authority_score=0.85,
                    is_official=False
                ),
                merge_source="web"
            ),
            # Official
            EnhancedDocumentationContext(
                source="https://reactjs.org/docs",
                title="React Docs",
                snippet="Official React docs",
                relevance_score=0.7,
                attribution=SourceAttribution(
                    url="https://reactjs.org/docs",
                    title="React Docs",
                    domain="reactjs.org",
                    authority_score=1.0,
                    is_official=True
                ),
                merge_source="web"
            )
        ]

        prioritized = searcher._prioritize_by_authority(results)

        # Order should be: official, high authority, other
        assert len(prioritized) == 3
        assert prioritized[0].source == "https://reactjs.org/docs"
        assert prioritized[1].source == "https://github.com/user/repo"
        assert prioritized[2].source == "https://random-blog.com/post"

    @pytest.mark.asyncio
    async def test_search_with_fallback_web_success(self):
        """Test search with fallback when web search succeeds."""
        searcher = EnhancedDocumentationSearcher()

        # Mock successful web search
        web_result = EnhancedDocumentationContext(
            source="https://docs.python.org/3/",
            title="Python Docs",
            snippet="Python documentation",
            relevance_score=0.9,
            attribution=SourceAttribution(
                url="https://docs.python.org/3/",
                title="Python Docs",
                domain="docs.python.org",
                authority_score=1.0,
                is_official=True
            ),
            merge_source="web"
        )

        with (
            patch.object(searcher, 'search_web_with_attribution', new_callable=AsyncMock) as mock_web,
            patch.object(searcher, 'search_knowledge_base', new_callable=AsyncMock) as mock_local,
        ):
            mock_web.return_value = [web_result]
            mock_local.return_value = []

            results = await searcher.search_with_fallback("test query", "implement")

            assert len(results) > 0
            assert results[0].source == "https://docs.python.org/3/"
            # Should not have fallback warning
            assert "[Note: Web search unavailable" not in results[0].snippet

    @pytest.mark.asyncio
    async def test_search_with_fallback_web_failure(self):
        """Test search with fallback when web search fails."""
        searcher = EnhancedDocumentationSearcher()

        # Mock local result
        local_result = DocumentationContext(
            source="Local Knowledge Base",
            title="Local Doc",
            snippet="Local documentation",
            relevance_score=0.8
        )

        with (
            patch.object(searcher, 'search_web_with_attribution', new_callable=AsyncMock) as mock_web,
            patch.object(searcher, 'search_knowledge_base', new_callable=AsyncMock) as mock_local,
        ):
            # Web search fails
            mock_web.side_effect = Exception("API unavailable")
            mock_local.return_value = [local_result]

            results = await searcher.search_with_fallback("test query", "implement")

            assert len(results) > 0
            # Should have fallback warning
            assert "[Note: Web search unavailable" in results[0].snippet


if __name__ == "__main__":
    # Run basic tests
    test = TestEnhancedDocumentationSearcher()
    test.test_calculate_authority_score_official()
    test.test_calculate_authority_score_high_authority()
    test.test_calculate_authority_score_documentation_pattern()
    test.test_calculate_authority_score_github_repo()
    test.test_calculate_authority_score_default()
    test.test_extract_grounding_metadata_no_metadata()
    test.test_extract_grounding_metadata_with_web_chunks()
    test.test_prioritize_by_authority_official_first()
    test.test_prioritize_by_authority_categories()
    print("✅ All unit tests passed!")
