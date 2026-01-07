"""
Context Module: Specialized Context Gathering Components

This module provides focused, single-responsibility classes for context preparation:

- QueryAnalyzer: Analyzes queries using Gemini to understand intent
- FileDiscoverer: Discovers relevant files in workspace using Gemini scoring
- DocumentationSearcher: Searches web and knowledge base for documentation
- CodeSearcher: Searches codebase using grep/ripgrep and git

Usage:
    from app.core.context import (
        QueryAnalyzer,
        FileDiscoverer,
        DocumentationSearcher,
        CodeSearcher,
    )

    analyzer = QueryAnalyzer()
    analysis = await analyzer.analyze("Fix the authentication bug")
"""

from .query_analyzer import QueryAnalyzer
from .file_discoverer import FileDiscoverer, FileContext
from .doc_searcher import DocumentationSearcher, DocumentationContext
from .code_searcher import CodeSearcher, CodeSearchContext

__all__ = [
    # Classes
    "QueryAnalyzer",
    "FileDiscoverer",
    "DocumentationSearcher",
    "CodeSearcher",
    # Dataclasses
    "FileContext",
    "DocumentationContext",
    "CodeSearchContext",
]
