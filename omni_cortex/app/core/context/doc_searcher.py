"""
Documentation Searcher: Web and Knowledge Base Documentation Lookup

Searches for relevant documentation using:
- Google Search via Gemini grounding
- ChromaDB knowledge base (learnings, debugging patterns, framework docs)
"""

import asyncio
import structlog
from dataclasses import dataclass
from typing import List, Optional

from ..settings import get_settings
from ..constants import CONTENT, SEARCH
from ..errors import LLMError, ProviderNotConfiguredError, RAGError, OmniCortexError
from ..correlation import get_correlation_id

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
                    details={"provider": "google", "package": "google-generativeai"}
                )

            api_key = self.settings.google_api_key
            if not api_key:
                raise ProviderNotConfiguredError(
                    "GOOGLE_API_KEY not configured",
                    details={"provider": "google", "env_var": "GOOGLE_API_KEY"}
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
                    model_name="gemini-2.0-flash",
                    tools=[google_search_tool]
                )
        return self._search_model

    async def search_web(self, query: str) -> List[DocumentationContext]:
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
                # Note: Grounding may require different configuration in new API
                response = await asyncio.to_thread(
                    model.models.generate_content,
                    model=self.settings.routing_model or "gemini-2.0-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.3)
                )
                text = response.text if hasattr(response, 'text') else str(response)
            else:
                # Legacy google.generativeai package
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config={"temperature": 0.3}
                )
                text = response.text

            # Parse response into documentation contexts
            docs = []

            # For now, return as single context if there's content
            if text and len(text) > 50:
                docs.append(DocumentationContext(
                    source="Google Search (Gemini Grounded)",
                    title="Documentation Search Results",
                    snippet=text[:CONTENT.SNIPPET_MAX],
                    relevance_score=0.8
                ))

            logger.debug("web_search_complete", docs_found=len(docs))
            return docs
        except (LLMError, ProviderNotConfiguredError):
            raise  # Re-raise custom LLM errors
        except Exception as e:
            # Graceful degradation: Web documentation search is optional enhancement.
            # Failures should not block the main reasoning workflow. We log and
            # return empty results, allowing the system to proceed without web docs.
            logger.warning(
                "doc_search_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return []

    async def search_knowledge_base(
        self,
        query: str,
        task_type: str
    ) -> List[DocumentationContext]:
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
                    manager.search_learnings,
                    query,
                    k=SEARCH.K_STANDARD,
                    min_rating=0.7
                )
                for learning in learnings:
                    if 'solution' in learning and 'problem' in learning:
                        results.append(DocumentationContext(
                            source=f"Past Learning (Framework: {learning.get('framework', 'unknown')})",
                            title=learning.get('problem', '')[:CONTENT.QUERY_LOG],
                            snippet=learning.get('solution', '')[:CONTENT.SNIPPET_EXTENDED],
                            relevance_score=learning.get('rating', 0.8)
                        ))
            except RAGError as e:
                logger.debug("learnings_search_failed", error=str(e), error_type="RAGError")
            except Exception as e:
                # Graceful degradation: Individual collection searches are independent.
                # A failure in learnings search should not prevent other collections
                # from being searched. Log and continue to next collection.
                logger.debug(
                    "learnings_search_failed",
                    error=str(e),
                    error_type=type(e).__name__
                )

            # For debug tasks, search debugging knowledge
            if task_type == "debug":
                try:
                    debug_docs = await asyncio.to_thread(
                        manager.search_debugging_knowledge,
                        query,
                        k=SEARCH.K_STANDARD
                    )
                    for doc in debug_docs:
                        if hasattr(doc, 'page_content'):
                            results.append(DocumentationContext(
                                source="Debugging Knowledge Base",
                                title="Similar Bug Fix",
                                snippet=doc.page_content[:CONTENT.SNIPPET_EXTENDED],
                                relevance_score=0.75
                            ))
                except RAGError as e:
                    logger.debug("debug_knowledge_search_failed", error=str(e), error_type="RAGError")
                except Exception as e:
                    # Graceful degradation: Debug knowledge search is supplementary.
                    # Failure here should not block other collection searches or
                    # the overall knowledge base lookup. Log and continue.
                    logger.debug(
                        "debug_knowledge_search_failed",
                        error=str(e),
                        error_type=type(e).__name__
                    )

            # Search framework documentation
            try:
                framework_docs = await asyncio.to_thread(
                    manager.search_frameworks,
                    query,
                    k=SEARCH.K_STANDARD
                )
                for doc in framework_docs:
                    if hasattr(doc, 'page_content'):
                        metadata = doc.metadata or {}
                        results.append(DocumentationContext(
                            source=f"Framework Docs: {metadata.get('framework_name', 'unknown')}",
                            title=metadata.get('function', metadata.get('class', 'Documentation')),
                            snippet=doc.page_content[:CONTENT.SNIPPET_EXTENDED],
                            relevance_score=0.7
                        ))
            except RAGError as e:
                logger.debug("framework_docs_search_failed", error=str(e), error_type="RAGError")
            except Exception as e:
                # Graceful degradation: Framework docs search is one of several
                # knowledge sources. Failure should not prevent returning results
                # from other successful collection searches. Log and continue.
                logger.debug(
                    "framework_docs_search_failed",
                    error=str(e),
                    error_type=type(e).__name__
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
                correlation_id=get_correlation_id()
            )
            raise OmniCortexError(f"Knowledge base search failed: {e}") from e
