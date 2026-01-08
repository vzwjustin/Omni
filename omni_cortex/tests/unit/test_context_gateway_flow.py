import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.context_gateway import ContextGateway, FileContext, DocumentationContext, CodeSearchContext

@pytest.fixture
def mock_components():
    with patch("app.core.context_gateway.QueryAnalyzer") as MockQueryAnalyzer, \
         patch("app.core.context_gateway.FileDiscoverer") as MockFileDiscoverer, \
         patch("app.core.context_gateway.DocumentationSearcher") as MockDocSearcher, \
         patch("app.core.context_gateway.CodeSearcher") as MockCodeSearcher:
        
        yield {
            "query_analyzer": MockQueryAnalyzer.return_value,
            "file_discoverer": MockFileDiscoverer.return_value,
            "doc_searcher": MockDocSearcher.return_value,
            "code_searcher": MockCodeSearcher.return_value,
        }

@pytest.mark.asyncio
async def test_prepare_context_flow(mock_components):
    """
    Verify that prepare_context runs discovery FIRST, then passes results to analysis.
    """
    gateway = ContextGateway()
    
    # Setup mock returns
    mock_components["file_discoverer"].discover = AsyncMock(return_value=[
        FileContext(path="test.py", relevance_score=0.9, summary="Test file", key_elements=[])
    ])
    
    mock_components["doc_searcher"].search_web = AsyncMock(return_value=[
        DocumentationContext(source="http://example.com", title="Example Docs", snippet="This is a snippet", relevance_score=0.8)
    ])
    
    mock_components["code_searcher"].search = AsyncMock(return_value=[
        CodeSearchContext(search_type="grep", query="test", results="found match", file_count=1, match_count=1)
    ])
    
    mock_components["query_analyzer"].analyze = AsyncMock(return_value={
        "task_type": "implement",
        "summary": "Implement feature",
        "complexity": "medium",
        "framework": "reason_flux"
    })

    # Execute
    query = "Implement feature X"
    await gateway.prepare_context(query, workspace_path="/tmp", search_docs=True)

    # Verification 1: Discovery called
    mock_components["file_discoverer"].discover.assert_called_once()
    mock_components["doc_searcher"].search_web.assert_called_once()
    mock_components["code_searcher"].search.assert_called_once()

    # Verification 2: Analysis called AFTER discovery with context
    mock_components["query_analyzer"].analyze.assert_called_once()
    
    # Check arguments passed to analyze
    call_args = mock_components["query_analyzer"].analyze.call_args
    _, kwargs = call_args
    
    # Verify documentation_context was constructed and passed
    assert "documentation_context" in kwargs
    doc_context = kwargs["documentation_context"]
    
    # Verify content of documentation_context
    assert "DOCUMENTATION FOUND:" in doc_context
    assert "Example Docs (http://example.com): This is a snippet" in doc_context
    assert "RELEVANT FILES:" in doc_context
    assert "test.py: Test file" in doc_context

@pytest.mark.asyncio
async def test_prepare_context_no_docs(mock_components):
    """Verify flow when search_docs is False."""
    gateway = ContextGateway()
    
    mock_components["file_discoverer"].discover = AsyncMock(return_value=[])
    mock_components["query_analyzer"].analyze = AsyncMock(return_value={})

    await gateway.prepare_context("query", search_docs=False)

    mock_components["doc_searcher"].search_web.assert_not_called()
    mock_components["query_analyzer"].analyze.assert_called_once()
