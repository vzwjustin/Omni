"""
Embedding Providers for Omni-Cortex

Supports: Gemini (free), OpenAI, OpenRouter, HuggingFace (local)
"""

import hashlib
import time
from collections import OrderedDict
from typing import Any

import structlog

from ..core.settings import get_settings

logger = structlog.get_logger("embeddings")

# Cache configuration
EMBEDDING_CACHE_MAX_SIZE = 1000


class GeminiEmbeddings:
    """LangChain-compatible wrapper for Gemini embeddings (FREE tier)."""

    def __init__(self, api_key: str):
        # Try new google-genai package first, fall back to deprecated
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._use_new_api = True
        except ImportError:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._use_new_api = False
        self.model = "text-embedding-004"

        # LRU cache for embed_query (OrderedDict-based)
        self._cache: OrderedDict[str, list] = OrderedDict()
        self._cache_max_size = EMBEDDING_CACHE_MAX_SIZE

    def _get_cache_key(self, text: str) -> str:
        """Generate a hash key for cache lookup."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def embed_documents(self, texts: list) -> list:
        """Embed a list of documents."""
        if not texts:
            return []

        start_time = time.perf_counter()
        results = []
        # Batch in groups of 100 (Gemini API limit)
        for i in range(0, len(texts), 100):
            batch = texts[i:i+100]
            if self._use_new_api:
                result = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                )
                # New API returns EmbedContentResponse with embeddings list
                for emb in result.embeddings:
                    results.append(emb.values)
            else:
                result = self._genai.embed_content(
                    model=f"models/{self.model}",
                    content=batch,
                    task_type="retrieval_document"
                )
                if isinstance(result['embedding'][0], list):
                    results.extend(result['embedding'])
                else:
                    results.append(result['embedding'])

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info("embedding_complete", provider="gemini", texts=len(texts), duration_ms=round(duration_ms, 2))
        return results

    def embed_query(self, text: str) -> list:
        """Embed a single query with LRU caching."""
        hash_key = self._get_cache_key(text)

        # Check cache first
        if hash_key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(hash_key)
            logger.debug("embedding_cache_hit", text_hash=hash_key)
            return self._cache[hash_key]

        logger.debug("embedding_cache_miss", text_hash=hash_key)

        # Cache miss - compute embedding
        start_time = time.perf_counter()

        if self._use_new_api:
            result = self._client.models.embed_content(
                model=self.model,
                contents=text,
            )
            embedding = result.embeddings[0].values
        else:
            result = self._genai.embed_content(
                model=f"models/{self.model}",
                content=text,
                task_type="retrieval_query"
            )
            embedding = result['embedding']

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info("embedding_complete", provider="gemini", texts=1, duration_ms=round(duration_ms, 2))

        # Add to cache with LRU eviction
        self._cache[hash_key] = embedding
        if len(self._cache) > self._cache_max_size:
            # Remove oldest (first) item
            self._cache.popitem(last=False)

        return embedding


def get_embeddings() -> Any:
    """
    Get the appropriate embedding model based on provider configuration.

    Configuration (via environment variables):
        EMBEDDING_PROVIDER: openai, huggingface, gemini, or openrouter
        EMBEDDING_MODEL: Model name (default: text-embedding-3-small for OpenAI)

    Returns:
        An embedding model instance compatible with LangChain
    """
    from langchain_openai import OpenAIEmbeddings

    # Cache settings instance to avoid repeated calls
    settings = get_settings()

    # Use dedicated embedding provider if set
    provider = getattr(settings, 'embedding_provider', None)
    if not provider or provider == "":
        provider = settings.llm_provider.lower()
    else:
        provider = provider.lower()

    model = getattr(settings, 'embedding_model', 'text-embedding-3-small')

    # Gemini embeddings (FREE up to 1500 req/day)
    if provider in ("google", "gemini"):
        if settings.google_api_key:
            try:
                logger.info("embeddings_init", provider="gemini", model="text-embedding-004")
                return GeminiEmbeddings(api_key=settings.google_api_key)
            except ImportError:
                logger.warning("gemini_embeddings_failed", error="google-generativeai not installed")
        else:
            logger.debug("gemini_embeddings_skipped", reason="GOOGLE_API_KEY not set")

    # OpenAI embeddings
    if provider == "openai" and settings.openai_api_key:
        logger.info("embeddings_init", provider="openai", model=model)
        return OpenAIEmbeddings(model=model, api_key=settings.openai_api_key)

    # OpenRouter embeddings (OpenAI-compatible endpoint)
    if provider == "openrouter" and settings.openrouter_api_key:
        logger.info("embeddings_init", provider="openrouter", model=model)
        return OpenAIEmbeddings(
            model=model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    # HuggingFace embeddings - local, no API key required
    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            hf_model = model if model != "text-embedding-3-small" else "sentence-transformers/all-MiniLM-L6-v2"
            logger.info("embeddings_init", provider="huggingface", model=hf_model)
            return HuggingFaceEmbeddings(model_name=hf_model)
        except ImportError:
            logger.error("embeddings_init_failed", error="langchain-huggingface not installed")

    # Fallback chain: try OpenRouter -> OpenAI -> HuggingFace
    if settings.openrouter_api_key:
        logger.info("embeddings_fallback", provider="openrouter", model=model)
        return OpenAIEmbeddings(
            model=model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    if settings.openai_api_key:
        logger.info("embeddings_fallback", provider="openai", model=model)
        return OpenAIEmbeddings(model=model, api_key=settings.openai_api_key)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("embeddings_fallback", provider="huggingface", model=hf_model)
        return HuggingFaceEmbeddings(model_name=hf_model)
    except ImportError as e:
        logger.debug("huggingface_fallback_unavailable", error=str(e))

    raise ValueError(
        "No embedding provider available. Set one of:\n"
        "- EMBEDDING_PROVIDER=openrouter + OPENROUTER_API_KEY\n"
        "- EMBEDDING_PROVIDER=openai + OPENAI_API_KEY\n"
        "- EMBEDDING_PROVIDER=huggingface (local, install langchain-huggingface)"
    )
