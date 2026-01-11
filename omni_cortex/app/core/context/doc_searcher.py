"""
Documentation Searcher: Web and Knowledge Base Documentation Lookup

Searches for relevant documentation using:
- Google Search via Gemini grounding
- ChromaDB knowledge base (learnings, debugging patterns, framework docs)
"""

import asyncio
from dataclasses import dataclass
from urllib.parse import urlparse

import structlog

from ..constants import CONTENT, SEARCH
from ..correlation import get_correlation_id
from ..errors import LLMError, OmniCortexError, ProviderNotConfiguredError, RAGError
from ..settings import get_settings
from .enhanced_models import EnhancedDocumentationContext, SourceAttribution

# Try to import Google AI (new package with thinking mode)
try:
    from google import genai
    from google.genai import types

    GOOGLE_AI_AVAILABLE = True
    NEW_GENAI_PACKAGE = True
except ImportError:
    NEW_GENAI_PACKAGE = False
    # Fallback to deprecated package
    try:
        import google.generativeai as genai
        from google.generativeai.types import Tool

        GenerativeModel = genai.GenerativeModel
        GOOGLE_AI_AVAILABLE = True
    except ImportError:
        GOOGLE_AI_AVAILABLE = False
        genai = None
        GenerativeModel = None
        Tool = None

# Import RAG systems
try:
    from ...collection_manager import get_collection_manager

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    get_collection_manager = None

logger = structlog.get_logger("context.doc_searcher")


@dataclass
class DocumentationContext:
    """Context from web documentation lookup."""

    source: str  # URL or doc name
    title: str
    snippet: str  # Relevant excerpt
    relevance_score: float


# Official documentation domains for authority scoring
OFFICIAL_DOMAINS = {
    "python.org",
    "docs.python.org",
    "nodejs.org",
    "docs.npmjs.com",
    "reactjs.org",
    "react.dev",
    "vuejs.org",
    "angular.io",
    "rust-lang.org",
    "golang.org",
    "go.dev",
    "docs.microsoft.com",
    "developer.mozilla.org",
    "github.com/docs",
    "docs.github.com",
    "cloud.google.com/docs",
    "docs.aws.amazon.com",
    "kubernetes.io/docs",
    "docker.com/docs",
}

# High authority domains
HIGH_AUTHORITY_DOMAINS = {
    "stackoverflow.com",
    "github.com",
    "medium.com",
    "dev.to",
    "hackernoon.com",
}


class DocumentationSearcher:
    """
    Searches web and knowledge base for documentation.

    Two search modes:
    1. Web search - Uses Gemini with Google Search grounding
    2. Knowledge base - Searches ChromaDB for past learnings and patterns
    """

    def __init__(self):
        self.settings = get_settings()
        self._search_model = None

    def _get_search_model(self):
        """Get model with Google Search grounding for documentation lookup."""
        if self._search_model is None:
            if not GOOGLE_AI_AVAILABLE:
                raise ProviderNotConfiguredError(
                    "google-generativeai not installed",
                    details={"provider": "google", "package": "google-generativeai"},
                )

            api_key = self.settings.google_api_key
            if not api_key:
                raise ProviderNotConfiguredError(
                    "GOOGLE_API_KEY not configured",
                    details={"provider": "google", "env_var": "GOOGLE_API_KEY"},
                )

            if NEW_GENAI_PACKAGE:
                # New google-genai package uses Client API
                # Note: Google Search grounding may work differently in new package
                self._search_model = genai.Client(api_key=api_key)
            else:
                # Legacy google.generativeai package
                genai.configure(api_key=api_key)

                # Google Search grounding tool (legacy API)
                google_search_tool = Tool.from_google_search_retrieval(
                    google_search_retrieval={"disable_attribution": False}
                )

                self._search_model = GenerativeModel(
                    model_name="gemini-2.0-flash", tools=[google_search_tool]
                )
        return self._search_model

    async def search_web(self, query: str) -> list[DocumentationContext]:
        """
        Search web for relevant documentation using Gemini with Google Search.

        Args:
            query: The user's request

        Returns:
            List of DocumentationContext from web search
        """
        try:
            model = self._get_search_model()

            prompt = f"""Search for relevant documentation, API references, or technical guides for:

{query}

Find official documentation, tutorials, or authoritative sources that would help with this task.
For each relevant source, provide:
- The URL/source
- A brief title
- The most relevant snippet or information

Focus on actionable, technical content."""

            if NEW_GENAI_PACKAGE:
                # New google-genai package uses Client.models.generate_content
                # Use Google Search Grounding tool
                google_search_tool = types.Tool(google_search=types.GoogleSearch())

                response = await asyncio.to_thread(
                    model.models.generate_content,
                    model=self.settings.routing_model or "gemini-3-flash-preview",
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.3, tools=[google_search_tool]),
                )
                text = response.text if hasattr(response, "text") else str(response)

                # Check for grounding metadata if available
                source_urls = []
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                        # Extract URLs from grounding metadata
                        metadata = candidate.grounding_metadata
                        if hasattr(metadata, "grounding_chunks"):
                            for chunk in metadata.grounding_chunks:
                                if hasattr(chunk, "web"):
                                    source_urls.append(chunk.web.uri)

            else:
                # Legacy google.generativeai package
                # Model is already configured with tools in _get_search_model
                response = await asyncio.to_thread(
                    model.generate_content, prompt, generation_config={"temperature": 0.3}
                )
                text = response.text

                # Attempt to extract sources from legacy response if available
                # (Note: Legacy API structure might differ, this is a best-effort check)
                source_urls = []
                if hasattr(response, "candidates") and response.candidates:
                    # Legacy extraction logic would go here if needed
                    pass

            # Parse response into documentation contexts
            docs = []

            # For now, return as single context if there's content
            if text and len(text) > 50:
                # Format sources if available
                source_label = "Google Search (Gemini Grounded)"
                if source_urls:
                    # Deduplicate and limit URLs
                    unique_urls = list(dict.fromkeys(source_urls))[:3]
                    source_label += f" - Sources: {', '.join(unique_urls)}"

                docs.append(
                    DocumentationContext(
                        source=source_label,
                        title="Documentation Search Results",
                        snippet=text[: CONTENT.SNIPPET_MAX],
                        relevance_score=0.8,
                    )
                )

            logger.debug("web_search_complete", docs_found=len(docs))
            return docs
        except (LLMError, ProviderNotConfiguredError):
            raise  # Re-raise custom LLM errors
        except Exception as e:
            # Graceful degradation: Web documentation search is optional enhancement.
            # Failures should not block the main reasoning workflow. We log and
            # return empty results, allowing the system to proceed without web docs.
            logger.warning("doc_search_failed", error=str(e), error_type=type(e).__name__)
            return []

    async def search_knowledge_base(self, query: str, task_type: str) -> list[DocumentationContext]:  # noqa: C901, PLR0912
        """
        Search ChromaDB collections for relevant framework docs, learnings, and patterns.

        Args:
            query: User's query
            task_type: Type of task (debug, implement, refactor, etc.)

        Returns:
            List of relevant documentation from ChromaDB
        """
        if not CHROMA_AVAILABLE:
            logger.debug("chroma_unavailable", reason="import_failed")
            return []

        try:
            manager = get_collection_manager()
            results = []

            # Search learnings for similar past solutions
            try:
                learnings = await asyncio.to_thread(
                    manager.search_learnings, query, k=SEARCH.K_STANDARD, min_rating=0.7
                )
                for learning in learnings:
                    if "solution" in learning and "problem" in learning:
                        results.append(
                            DocumentationContext(
                                source=f"Past Learning (Framework: {learning.get('framework', 'unknown')})",
                                title=learning.get("problem", "")[: CONTENT.QUERY_LOG],
                                snippet=learning.get("solution", "")[: CONTENT.SNIPPET_EXTENDED],
                                relevance_score=learning.get("rating", 0.8),
                            )
                        )
            except RAGError as e:
                logger.debug("learnings_search_failed", error=str(e), error_type="RAGError")
            except Exception as e:
                # Graceful degradation: Individual collection searches are independent.
                # A failure in learnings search should not prevent other collections
                # from being searched. Log and continue to next collection.
                logger.debug("learnings_search_failed", error=str(e), error_type=type(e).__name__)

            # For debug tasks, search debugging knowledge
            if task_type == "debug":
                try:
                    debug_docs = await asyncio.to_thread(
                        manager.search_debugging_knowledge, query, k=SEARCH.K_STANDARD
                    )
                    for doc in debug_docs:
                        if hasattr(doc, "page_content"):
                            results.append(
                                DocumentationContext(
                                    source="Debugging Knowledge Base",
                                    title="Similar Bug Fix",
                                    snippet=doc.page_content[: CONTENT.SNIPPET_EXTENDED],
                                    relevance_score=0.75,
                                )
                            )
                except RAGError as e:
                    logger.debug(
                        "debug_knowledge_search_failed", error=str(e), error_type="RAGError"
                    )
                except Exception as e:
                    # Graceful degradation: Debug knowledge search is supplementary.
                    # Failure here should not block other collection searches or
                    # the overall knowledge base lookup. Log and continue.
                    logger.debug(
                        "debug_knowledge_search_failed", error=str(e), error_type=type(e).__name__
                    )

            # Search framework documentation
            try:
                framework_docs = await asyncio.to_thread(
                    manager.search_frameworks, query, k=SEARCH.K_STANDARD
                )
                for doc in framework_docs:
                    if hasattr(doc, "page_content"):
                        metadata = doc.metadata or {}
                        results.append(
                            DocumentationContext(
                                source=f"Framework Docs: {metadata.get('framework_name', 'unknown')}",
                                title=metadata.get(
                                    "function", metadata.get("class", "Documentation")
                                ),
                                snippet=doc.page_content[: CONTENT.SNIPPET_EXTENDED],
                                relevance_score=0.7,
                            )
                        )
            except RAGError as e:
                logger.debug("framework_docs_search_failed", error=str(e), error_type="RAGError")
            except Exception as e:
                # Graceful degradation: Framework docs search is one of several
                # knowledge sources. Failure should not prevent returning results
                # from other successful collection searches. Log and continue.
                logger.debug(
                    "framework_docs_search_failed", error=str(e), error_type=type(e).__name__
                )

            logger.info("knowledge_base_search_complete", results=len(results))
            return results[:5]  # Cap at 5 results

        except RAGError as e:
            logger.error("knowledge_base_search_failed", error=str(e), error_type="RAGError")
            return []
        except Exception as e:
            # Non-graceful: This outer exception handler catches failures in the
            # collection manager initialization or other critical infrastructure.
            # Unlike individual collection search failures, this indicates a
            # fundamental problem that should be surfaced to the caller.
            logger.error(
                "knowledge_base_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            raise OmniCortexError(f"Knowledge base search failed: {e}") from e


class EnhancedDocumentationSearcher(DocumentationSearcher):
    """
    Enhanced documentation searcher with source attribution and intelligent merging.

    Extends DocumentationSearcher to provide:
    - Source attribution extraction from Gemini grounding metadata
    - Intelligent merging of web and local results
    - Documentation prioritization by authority
    - Clickable link formatting
    """

    def __init__(self):
        super().__init__()
        self.logger = structlog.get_logger("context.enhanced_doc_searcher")

    def _extract_grounding_metadata(self, response) -> list[SourceAttribution]:
        """
        Extract source attribution from Gemini grounding metadata.

        Args:
            response: Gemini API response with grounding metadata

        Returns:
            List of SourceAttribution objects
        """
        attributions = []

        try:
            if not hasattr(response, "candidates") or not response.candidates:
                return attributions

            candidate = response.candidates[0]

            # Check for grounding metadata
            if not hasattr(candidate, "grounding_metadata") or not candidate.grounding_metadata:
                return attributions

            metadata = candidate.grounding_metadata

            # Extract grounding chunks (new google-genai package)
            if hasattr(metadata, "grounding_chunks"):
                for chunk in metadata.grounding_chunks:
                    if hasattr(chunk, "web") and chunk.web:
                        url = chunk.web.uri if hasattr(chunk.web, "uri") else str(chunk.web)
                        title = chunk.web.title if hasattr(chunk.web, "title") else "Documentation"

                        # Parse domain for authority scoring
                        domain = urlparse(url).netloc.lower()
                        authority_score = self._calculate_authority_score(domain)
                        is_official = domain in OFFICIAL_DOMAINS

                        attributions.append(
                            SourceAttribution(
                                url=url,
                                title=title,
                                domain=domain,
                                authority_score=authority_score,
                                is_official=is_official,
                                grounding_metadata={
                                    "chunk_index": len(attributions),
                                    "source_type": "web_grounding",
                                },
                            )
                        )

            # Extract search entry points (alternative structure)
            if hasattr(metadata, "search_entry_point"):
                entry_point = metadata.search_entry_point
                if hasattr(entry_point, "rendered_content"):
                    # This might contain additional source information
                    pass

            # Extract grounding supports (legacy structure)
            if hasattr(metadata, "grounding_supports"):
                for support in metadata.grounding_supports:
                    if hasattr(support, "source") and hasattr(support.source, "uri"):
                        url = support.source.uri
                        domain = urlparse(url).netloc.lower()
                        authority_score = self._calculate_authority_score(domain)
                        is_official = domain in OFFICIAL_DOMAINS

                        attributions.append(
                            SourceAttribution(
                                url=url,
                                title=support.source.title
                                if hasattr(support.source, "title")
                                else "Documentation",
                                domain=domain,
                                authority_score=authority_score,
                                is_official=is_official,
                                grounding_metadata={
                                    "support_index": len(attributions),
                                    "source_type": "grounding_support",
                                },
                            )
                        )

            self.logger.debug("extracted_grounding_metadata", attribution_count=len(attributions))

        except Exception as e:
            self.logger.warning("grounding_metadata_extraction_failed", error=str(e))

        return attributions

    def _calculate_authority_score(self, domain: str) -> float:
        """
        Calculate authority score for a domain.

        Args:
            domain: Domain name (e.g., 'docs.python.org')

        Returns:
            Authority score from 0.0 to 1.0
        """
        # Official documentation gets highest score
        if domain in OFFICIAL_DOMAINS:
            return 1.0

        # High authority domains
        if domain in HIGH_AUTHORITY_DOMAINS:
            return 0.85

        # Check for common documentation patterns
        if any(pattern in domain for pattern in ["docs.", "documentation.", "developer."]):
            return 0.75

        # GitHub repositories
        if "github.com" in domain:
            return 0.7

        # Default score for other domains
        return 0.5

    async def search_web_with_attribution(self, query: str) -> list[EnhancedDocumentationContext]:
        """
        Search web for documentation with full source attribution.

        Args:
            query: The user's request

        Returns:
            List of EnhancedDocumentationContext with attribution
        """
        try:
            model = self._get_search_model()

            prompt = f"""Search for relevant documentation, API references, or technical guides for:

{query}

Find official documentation, tutorials, or authoritative sources that would help with this task.
For each relevant source, provide:
- The URL/source
- A brief title
- The most relevant snippet or information

Focus on actionable, technical content."""

            if NEW_GENAI_PACKAGE:
                # New google-genai package uses Client.models.generate_content
                google_search_tool = types.Tool(google_search=types.GoogleSearch())

                response = await asyncio.to_thread(
                    model.models.generate_content,
                    model=self.settings.routing_model or "gemini-2.0-flash-exp",
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.3, tools=[google_search_tool]),
                )
                text = response.text if hasattr(response, "text") else str(response)

                # Extract source attributions from grounding metadata
                attributions = self._extract_grounding_metadata(response)

            else:
                # Legacy google.generativeai package
                response = await asyncio.to_thread(
                    model.generate_content, prompt, generation_config={"temperature": 0.3}
                )
                text = response.text

                # Extract attributions from legacy response
                attributions = self._extract_grounding_metadata(response)

            # Parse response into enhanced documentation contexts
            docs = []

            if text and len(text) > 50:
                # If we have attributions, create separate contexts for each
                if attributions:
                    # Split content by sources if possible
                    for i, attribution in enumerate(attributions[:5]):  # Limit to top 5
                        docs.append(
                            EnhancedDocumentationContext(
                                source=attribution.url,
                                title=attribution.title,
                                snippet=text[: CONTENT.SNIPPET_MAX]
                                if i == 0
                                else text[: CONTENT.SNIPPET_EXTENDED],
                                relevance_score=attribution.authority_score,
                                attribution=attribution,
                                merge_source="web",
                            )
                        )
                else:
                    # No attributions, create single context
                    docs.append(
                        EnhancedDocumentationContext(
                            source="Google Search (Gemini Grounded)",
                            title="Documentation Search Results",
                            snippet=text[: CONTENT.SNIPPET_MAX],
                            relevance_score=0.8,
                            attribution=None,
                            merge_source="web",
                        )
                    )

            self.logger.debug(
                "web_search_with_attribution_complete",
                docs_found=len(docs),
                attributions=len(attributions),
            )
            return docs

        except (LLMError, ProviderNotConfiguredError):
            raise
        except Exception as e:
            self.logger.warning(
                "doc_search_with_attribution_failed", error=str(e), error_type=type(e).__name__
            )
            return []

    async def _merge_web_and_local_results(
        self,
        web_results: list[EnhancedDocumentationContext],
        local_results: list[DocumentationContext],
    ) -> list[EnhancedDocumentationContext]:
        """
        Intelligently merge web and local documentation results.

        Deduplicates based on content similarity and prioritizes by authority.

        Args:
            web_results: Results from web search with attribution
            local_results: Results from ChromaDB knowledge base

        Returns:
            Merged and deduplicated list of EnhancedDocumentationContext
        """
        merged = []
        seen_snippets = set()

        # Convert local results to enhanced format
        enhanced_local = []
        for local in local_results:
            enhanced_local.append(
                EnhancedDocumentationContext(
                    source=local.source,
                    title=local.title,
                    snippet=local.snippet,
                    relevance_score=local.relevance_score,
                    attribution=None,
                    merge_source="local",
                )
            )

        # Combine all results
        all_results = web_results + enhanced_local

        # Deduplicate based on snippet similarity
        for doc in all_results:
            # Create a normalized snippet for comparison
            normalized = doc.snippet.lower().strip()[:200]

            # Check for duplicates
            is_duplicate = False
            for seen in seen_snippets:
                # Simple similarity check - if 80% of words match, consider duplicate
                seen_words = set(seen.split())
                doc_words = set(normalized.split())
                if len(seen_words) > 0:
                    overlap = len(seen_words & doc_words) / len(seen_words)
                    if overlap > 0.8:
                        is_duplicate = True
                        break

            if not is_duplicate:
                seen_snippets.add(normalized)
                merged.append(doc)

        # Sort by authority and relevance
        merged.sort(
            key=lambda d: (
                d.attribution.authority_score if d.attribution else d.relevance_score,
                d.relevance_score,
            ),
            reverse=True,
        )

        self.logger.debug(
            "merged_results",
            web_count=len(web_results),
            local_count=len(local_results),
            merged_count=len(merged),
        )

        return merged

    def _prioritize_by_authority(
        self, results: list[EnhancedDocumentationContext]
    ) -> list[EnhancedDocumentationContext]:
        """
        Prioritize documentation results by authority score.

        Official documentation comes first, followed by high-authority sources,
        then other sources sorted by relevance.

        Args:
            results: List of documentation contexts to prioritize

        Returns:
            Prioritized list
        """
        # Separate into categories
        official = []
        high_authority = []
        other = []

        for doc in results:
            if doc.attribution and doc.attribution.is_official:
                official.append(doc)
            elif doc.attribution and doc.attribution.authority_score >= 0.8:
                high_authority.append(doc)
            else:
                other.append(doc)

        # Sort each category by relevance score
        official.sort(key=lambda d: d.relevance_score, reverse=True)
        high_authority.sort(key=lambda d: d.relevance_score, reverse=True)
        other.sort(key=lambda d: d.relevance_score, reverse=True)

        # Combine in priority order
        prioritized = official + high_authority + other

        self.logger.debug(
            "prioritized_by_authority",
            official_count=len(official),
            high_authority_count=len(high_authority),
            other_count=len(other),
        )

        return prioritized

    async def search_with_fallback(
        self, query: str, task_type: str
    ) -> list[EnhancedDocumentationContext]:
        """
        Search with automatic fallback to ChromaDB-only if web search fails.

        Args:
            query: User's query
            task_type: Type of task (debug, implement, etc.)

        Returns:
            List of enhanced documentation contexts
        """
        web_results = []
        web_search_failed = False

        # Try web search first
        try:
            web_results = await self.search_web_with_attribution(query)
        except Exception as e:
            self.logger.warning("web_search_failed_falling_back", error=str(e))
            web_search_failed = True

        # Always search local knowledge base
        local_results = await self.search_knowledge_base(query, task_type)

        # Merge results
        merged = await self._merge_web_and_local_results(web_results, local_results)

        # Prioritize by authority
        prioritized = self._prioritize_by_authority(merged)

        # Add warning if web search failed
        if web_search_failed and prioritized and prioritized[0].snippet:
            prioritized[0].snippet = (
                f"[Note: Web search unavailable, showing local results only]\n\n"
                f"{prioritized[0].snippet}"
            )

        self.logger.info(
            "search_with_fallback_complete",
            web_results=len(web_results),
            local_results=len(local_results),
            final_count=len(prioritized),
            web_failed=web_search_failed,
        )

        return prioritized[:10]  # Return top 10 results
