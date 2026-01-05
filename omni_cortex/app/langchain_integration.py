"""
LangChain Integration for Omni-Cortex

Provides LangChain components that complement reasoning frameworks:
- Memory systems for conversation context
- Tools for LLM-powered tool use
- Retrieval for RAG-enhanced reasoning
- Callbacks for monitoring
- Output parsers for structured responses
"""

import asyncio
import re
import os
from typing import Any, Optional, List, Dict
from collections import OrderedDict
from dataclasses import dataclass, field

# LangChain 1.0+ imports
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
import structlog

from .core.config import settings
from .state import GraphState
from .core.config import model_config

# Note: _safe_execute is imported at runtime in execute_code() to avoid circular import

logger = structlog.get_logger("langchain-integration")


# =============================================================================
# Memory Systems (LangChain 1.0+ compatible)
# =============================================================================

class OmniCortexMemory:
    """
    Unified memory system using LangChain 1.0+ message types.

    Provides:
    - Conversation buffer for recent exchanges (list of messages)
    - Framework history tracking
    - Simple and lightweight for pass-through mode
    """

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id

        # Short-term: Recent conversation history as messages
        self.messages: List[BaseMessage] = []

        # Track framework usage
        self.framework_history: List[str] = []

        # Max messages to keep in buffer
        self.max_messages = 20

    def add_exchange(self, query: str, answer: str, framework: str) -> None:
        """Add a query-answer exchange to memory."""
        self.messages.append(HumanMessage(content=query))
        self.messages.append(AIMessage(content=answer))

        # Trim to max size
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        self.framework_history.append(framework)
        logger.info("memory_updated", thread_id=self.thread_id, framework=framework)

    def get_context(self) -> Dict[str, Any]:
        """Get full memory context for prompting."""
        return {
            "chat_history": self.messages,
            "framework_history": self.framework_history
        }

    def clear(self) -> None:
        """Clear all memory."""
        self.messages = []
        self.framework_history = []


# Global memory store (keyed by thread_id) with simple LRU eviction
_memory_store: "OrderedDict[str, OmniCortexMemory]" = OrderedDict()
_memory_store_lock = asyncio.Lock()
MAX_MEMORY_THREADS = 100


async def get_memory(thread_id: str) -> OmniCortexMemory:
    """Get or create memory for a thread with thread-safe access."""
    async with _memory_store_lock:
        if thread_id in _memory_store:
            _memory_store.move_to_end(thread_id)
            return _memory_store[thread_id]
        
        # Evict oldest if over capacity
        if len(_memory_store) >= MAX_MEMORY_THREADS:
            _memory_store.popitem(last=False)
        
        mem = OmniCortexMemory(thread_id)
        _memory_store[thread_id] = mem
        return mem


# =============================================================================
# Tools for LLM Use
# =============================================================================

class VectorstoreSearchError(RuntimeError):
    """Raised when a vectorstore search fails (not a no-results case)."""

@tool
async def search_documentation(query: str) -> str:
    """Search the indexed documentation/code via vector store."""
    logger.info("tool_called", tool="search_documentation", query=query)
    try:
        docs = await search_vectorstore_async(query, k=5)
    except VectorstoreSearchError as exc:
        logger.error("search_documentation_failed", error=str(exc))
        return f"Search failed: {exc}"
    except Exception as exc:
        logger.error("search_documentation_failed", error=str(exc))
        return f"Search failed: {exc}"
    if not docs:
        return "No results found in the indexed corpus. Try re-ingesting or refining the query."
    formatted = []
    for d in docs:
        meta = d.metadata or {}
        path = meta.get("path", "unknown")
        formatted.append(f"### {path}\n{d.page_content[:1500]}")
    return "\n\n".join(formatted)


@tool
async def execute_code(code: str, language: str = "python") -> dict:
    """Execute code in the PoT sandbox (_safe_execute)."""
    # Import at runtime to avoid circular import
    from .nodes.code.pot import _safe_execute

    logger.info("tool_called", tool="execute_code", language=language)
    if language.lower() != "python":
        return {"success": False, "output": "", "error": f"Sandbox only supports python (requested: {language})"}
    result = await _safe_execute(code)
    # Ensure dict structure for callers
    return {
        "success": bool(result.get("success")),
        "output": result.get("output", "") or "",
        "error": result.get("error", "")
    }


@tool
async def save_learning(
    query: str,
    answer: str,
    framework: str,
    success_rating: float = 1.0,
    problem_type: str = "general"
) -> str:
    """
    Save a successful solution to the learnings database for future reference.
    
    Args:
        query: The original problem/question that was solved
        answer: The successful solution/answer
        framework: Which reasoning framework was used (e.g., 'active_inference')
        success_rating: Quality rating from 0.0 to 1.0 (default: 1.0)
        problem_type: Category like 'debugging', 'optimization', 'refactoring', etc.
    
    Returns:
        Confirmation message
    """
    from .collection_manager import get_collection_manager
    
    logger.info(
        "tool_called",
        tool="save_learning",
        framework=framework,
        problem_type=problem_type
    )
    
    manager = get_collection_manager()
    success = manager.add_learning(
        query=query,
        answer=answer,
        framework_used=framework,
        success_rating=success_rating,
        problem_type=problem_type
    )
    
    if success:
        return f"✅ Learning saved! Future queries will benefit from this {framework} solution."
    else:
        return "⚠️ Failed to save learning. Check logs for details."


@tool
async def retrieve_context(query: str, thread_id: Optional[str] = None) -> str:
    """Retrieve recent chat and framework history as lightweight context.

    Args:
        query: The query to contextualize
        thread_id: Optional thread ID to filter context to specific session
    """
    logger.info("tool_called", tool="retrieve_context", query=query, thread_id=thread_id)

    # Use lock for thread-safe access to _memory_store
    async with _memory_store_lock:
        # If thread_id is provided, only get context from that specific thread
        if thread_id and thread_id in _memory_store:
            mem = _memory_store[thread_id]
            if mem.messages:
                # Take last 6 messages (3 exchanges)
                recent = mem.messages[-6:]
                history = "\n".join(str(m.content) for m in recent)
                return f"Recent context (thread {thread_id}):\n\n{history}"
            return "No prior context available for this thread."

        # If no thread_id provided, only return context from the most recently accessed thread
        # to avoid leaking context between sessions
        if _memory_store:
            # OrderedDict keeps insertion/access order - get the most recent
            # Convert to list for atomic iteration (safe under lock)
            most_recent_thread_id = list(_memory_store.keys())[-1]
            mem = _memory_store[most_recent_thread_id]
            if mem.messages:
                recent = mem.messages[-6:]
                history = "\n".join(str(m.content) for m in recent)
                return f"Recent context:\n\n{history}"

    return "No prior context available."


# Import enhanced search tools
try:
    from .enhanced_search_tools import ENHANCED_SEARCH_TOOLS
    _enhanced_tools = ENHANCED_SEARCH_TOOLS
except ImportError:
    _enhanced_tools = []

# Export available tools for MCP
AVAILABLE_TOOLS = [search_documentation, execute_code, retrieve_context, save_learning] + _enhanced_tools


# =============================================================================
# Retrieval (RAG) Helpers
# =============================================================================

_vectorstore: Optional[Chroma] = None
_vectorstore_lock = asyncio.Lock()
import threading
_vectorstore_threading_lock = threading.Lock()


def _get_embeddings() -> Any:
    """
    Get the appropriate embedding model based on provider configuration.

    Configuration (via environment variables):
        EMBEDDING_PROVIDER: openai, huggingface, or openrouter (default: uses LLM_PROVIDER)
        EMBEDDING_MODEL: Model name (default: text-embedding-3-small for OpenAI)
        OPENAI_API_KEY: Required for OpenAI embeddings
        OPENROUTER_API_KEY: For OpenRouter (uses OpenAI-compatible endpoint)

    Supports:
    - openai: Uses OpenAI's embedding API
    - openrouter: Uses OpenRouter's OpenAI-compatible embedding endpoint
    - huggingface: Uses HuggingFace sentence-transformers (local, no API key needed)

    Returns:
        An embedding model instance compatible with LangChain
    """
    # Use dedicated embedding provider if set, otherwise fall back to LLM provider
    provider = getattr(settings, 'embedding_provider', None)
    if not provider or provider == "":
        provider = settings.llm_provider.lower()
    else:
        provider = provider.lower()

    model = getattr(settings, 'embedding_model', 'text-embedding-3-small')

    # OpenAI embeddings
    if provider == "openai" and settings.openai_api_key:
        logger.info("embeddings_init", provider="openai", model=model)
        return OpenAIEmbeddings(
            model=model,
            api_key=settings.openai_api_key
        )

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

    # Fallback chain: try OpenRouter → OpenAI → HuggingFace
    if settings.openrouter_api_key:
        logger.info("embeddings_fallback", provider="openrouter", model=model)
        return OpenAIEmbeddings(
            model=model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    if settings.openai_api_key:
        logger.info("embeddings_fallback", provider="openai", model=model)
        return OpenAIEmbeddings(
            model=model,
            api_key=settings.openai_api_key
        )

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("embeddings_fallback", provider="huggingface", model=hf_model)
        return HuggingFaceEmbeddings(model_name=hf_model)
    except ImportError:
        pass

    raise ValueError(
        "No embedding provider available. Set one of:\n"
        "- EMBEDDING_PROVIDER=openrouter + OPENROUTER_API_KEY\n"
        "- EMBEDDING_PROVIDER=openai + OPENAI_API_KEY\n"
        "- EMBEDDING_PROVIDER=huggingface (local, install langchain-huggingface)"
    )


def get_vectorstore() -> Optional[Chroma]:
    """Get or initialize a persistent Chroma vector store (thread-safe)."""
    global _vectorstore

    # Fast path: already initialized
    if _vectorstore is not None:
        return _vectorstore

    # Thread-safe initialization
    with _vectorstore_threading_lock:
        # Double-check after acquiring lock
        if _vectorstore is not None:
            return _vectorstore

        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "/app/data/chroma")
        os.makedirs(persist_dir, exist_ok=True)

        try:
            embeddings = _get_embeddings()
            _vectorstore = Chroma(
                collection_name="omni-cortex-context",
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            return _vectorstore
        except Exception as e:
            logger.error("vectorstore_init_failed", error=str(e))
            return None


def add_documents(texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> int:
    """Ingest documents into the vector store (legacy method)."""
    vs = get_vectorstore()
    if not vs:
        return 0
    metadatas = metadatas or [{} for _ in texts]
    try:
        vs.add_texts(texts=texts, metadatas=metadatas)
        # Note: persist() is no longer needed in Chroma 0.4+ with persist_directory
        return len(texts)
    except Exception as e:
        logger.error("vectorstore_add_failed", error=str(e))
        return 0


def add_documents_with_metadata(texts: List[str], metadatas: List[Dict[str, Any]], collection_name: str = "omni-cortex-enhanced") -> int:
    """Add documents with rich metadata to a specific collection."""
    from .collection_manager import get_collection_manager
    
    manager = get_collection_manager()
    
    # Route documents to appropriate collections based on metadata
    collections_docs: Dict[str, tuple[List[str], List[Dict[str, Any]]]] = {}
    
    for text, metadata in zip(texts, metadatas):
        coll_name = manager.route_to_collection(metadata)
        if coll_name not in collections_docs:
            collections_docs[coll_name] = ([], [])
        collections_docs[coll_name][0].append(text)
        collections_docs[coll_name][1].append(metadata)
    
    # Add to each collection
    total_added = 0
    for coll_name, (coll_texts, coll_metas) in collections_docs.items():
        added = manager.add_documents(coll_texts, coll_metas, coll_name)
        total_added += added
        logger.info("documents_routed", collection=coll_name, count=added)
    
    return total_added


def get_vectorstore_by_collection(collection_name: str) -> Optional[Chroma]:
    """Get a specific collection from the manager."""
    from .collection_manager import get_collection_manager
    return get_collection_manager().get_collection(collection_name)


async def search_vectorstore_async(query: str, k: int = 5) -> List[Document]:
    """Search the vector store for relevant documents (async, non-blocking)."""
    return await asyncio.to_thread(_search_vectorstore_sync, query, k)


def _search_vectorstore_sync(query: str, k: int = 5) -> List[Document]:
    """Synchronous vectorstore search (internal use)."""
    from .collection_manager import get_collection_manager
    manager = get_collection_manager()
    return manager.search(
        query,
        collection_names=["frameworks", "documentation", "utilities"],
        k=k,
        raise_on_error=True
    )


def search_vectorstore(query: str, k: int = 5) -> List[Document]:
    """Search the vector store for relevant documents.

    Note: This is synchronous for backwards compatibility.
    Prefer search_vectorstore_async() in async contexts to avoid blocking.
    """
    try:
        return _search_vectorstore_sync(query, k)
    except Exception as e:
        logger.error("vectorstore_search_failed", error=str(e))
        raise VectorstoreSearchError(f"Vectorstore search failed: {e}") from e


# =============================================================================
# Callbacks for Monitoring
# =============================================================================

class OmniCortexCallback(BaseCallbackHandler):
    """
    Custom callback handler for tracking LLM usage, timing, and errors.
    """
    
    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.total_tokens = 0
        self.llm_calls = 0
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Track LLM call start."""
        self.llm_calls += 1
        logger.info(
            "llm_call_start",
            thread_id=self.thread_id,
            call_number=self.llm_calls,
            prompt_count=len(prompts)
        )
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Track LLM call completion and token usage."""
        # Guard against None response
        if response is None:
            logger.warning("on_llm_end_null_response", thread_id=self.thread_id)
            return

        # Handle both object and dict response types (LangChain 1.0+ compatibility)
        llm_output = None
        if isinstance(response, dict):
            llm_output = response.get('llm_output')
        elif hasattr(response, 'llm_output'):
            llm_output = response.llm_output

        if llm_output:
            tokens = llm_output.get('token_usage', {}) if isinstance(llm_output, dict) else {}
            total = tokens.get('total_tokens', 0)
            self.total_tokens += total
            logger.info(
                "llm_call_end",
                thread_id=self.thread_id,
                tokens=total,
                cumulative_tokens=self.total_tokens
            )
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Track LLM errors."""
        logger.error(
            "llm_call_error",
            thread_id=self.thread_id,
            error=str(error)
        )
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Track tool usage."""
        logger.info(
            "tool_start",
            thread_id=self.thread_id,
            tool=serialized.get("name", "unknown"),
            input=input_str[:100]
        )
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Track tool completion."""
        logger.info("tool_end", thread_id=self.thread_id, output_length=len(output))


# =============================================================================
# Prompt Templates
# =============================================================================

# Framework selection prompt
FRAMEWORK_SELECTION_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are an AI routing system that selects the optimal reasoning framework.
    
Available frameworks: {frameworks}

Task characteristics:
- Type: {task_type}
- Complexity: {complexity}
- Has code: {has_code}

Previous frameworks used: {framework_history}"""),
    ("human", "{query}")
])


# General reasoning prompt
REASONING_TEMPLATE = PromptTemplate(
    input_variables=["framework", "query", "context", "chat_history"],
    template="""You are using the {framework} reasoning framework.

Previous conversation:
{chat_history}

Current task: {query}

Context:
{context}

Apply the {framework} methodology to solve this problem."""
)


# Code generation prompt
CODE_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are a code generation expert using {framework}.

Requirements:
- Write clean, well-commented code
- Follow best practices
- Handle edge cases
- Include error handling"""),
    ("human", """Task: {query}

Existing code:
{code_context}

Generate improved code:""")
])


# =============================================================================
# Output Parsers
# =============================================================================

class ReasoningOutput(BaseModel):
    """Structured output for reasoning results."""
    framework_used: str = Field(description="The reasoning framework that was used")
    thought_process: str = Field(description="The step-by-step reasoning")
    answer: str = Field(description="The final answer or solution")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
    code: Optional[str] = Field(default=None, description="Generated code if applicable")


class FrameworkSelection(BaseModel):
    """Structured output for framework selection."""
    selected_framework: str = Field(description="The chosen framework")
    reasoning: str = Field(description="Why this framework was selected")
    task_type: str = Field(description="Identified task type")
    estimated_complexity: float = Field(description="Complexity estimate 0-1", ge=0, le=1)


# Parser instances
reasoning_parser = PydanticOutputParser(pydantic_object=ReasoningOutput)
framework_parser = PydanticOutputParser(pydantic_object=FrameworkSelection)


# =============================================================================
# LangChain Chat Models
# =============================================================================

def get_routing_model() -> Any:
    """
    Get Gemini model specifically for routing/task analysis.

    This always uses Gemini regardless of LLM_PROVIDER setting.
    Used by HyperRouter._gemini_analyze_task() to offload thinking from Claude Code.

    Features:
    - Google Search grounding for fresh docs/APIs
    - Native SDK for full feature access
    """
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY required for routing. Gemini does task analysis so Claude Code can focus on execution.")

    import google.generativeai as genai

    # Configure the API
    genai.configure(api_key=settings.google_api_key)

    # Use fast model for routing - Gemini 3 Flash is ideal
    model_name = os.getenv("ROUTING_MODEL", "gemini-2.0-flash")
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    # Create model with Google Search grounding enabled
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=genai.GenerationConfig(
            temperature=0.7,
            max_output_tokens=4096,
        ),
        tools=[genai.Tool(google_search=genai.GoogleSearch())]
    )

    return GeminiRoutingWrapper(model)


class GeminiRoutingWrapper:
    """
    Wrapper to make native Gemini SDK compatible with LangChain-style ainvoke().
    Enables Google Search grounding for fresh context.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    async def ainvoke(self, prompt: str) -> Any:
        """Async invoke with search grounding."""
        import asyncio

        # Run sync generate_content in thread pool
        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt
        )

        # Return wrapper with .content attribute for compatibility
        return GeminiResponse(response)


class GeminiResponse:
    """Response wrapper for compatibility with existing code."""

    def __init__(self, response: Any) -> None:
        self._response = response

    @property
    def content(self) -> str:
        """Extract text content from Gemini response."""
        try:
            return self._response.text
        except Exception:
            # Fallback for different response formats
            if hasattr(self._response, 'candidates') and self._response.candidates:
                parts = self._response.candidates[0].content.parts
                return "".join(p.text for p in parts if hasattr(p, 'text'))
            return str(self._response)


def get_chat_model(model_type: str = "deep", enable_thinking: bool = False) -> Any:
    """
    Get configured LangChain chat model.

    Args:
        model_type: "deep" for reasoning or "fast" for synthesis
        enable_thinking: Enable extended thinking/reasoning mode (Gemini only)

    Supports: google, anthropic, openai, openrouter
    """
    model_name = settings.deep_reasoning_model if model_type == "deep" else settings.fast_synthesis_model
    temperature = 0.7 if model_type == "deep" else 0.5

    # Remove provider prefix if present (e.g., "google/gemini-3" -> "gemini-3")
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Use configured model - thinking mode is enabled via prompting style
        # For thinking-like behavior, we increase temperature and tokens
        max_tokens = 8192 if enable_thinking else 4096
        temp = max(0.7, temperature) if enable_thinking else temperature

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.google_api_key,
            temperature=temp,
            max_output_tokens=max_tokens
        )
    elif settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=model_name,
            api_key=settings.anthropic_api_key,
            temperature=temperature
        )
    elif settings.llm_provider == "openrouter":
        # OpenRouter needs full model path
        full_model = settings.deep_reasoning_model if model_type == "deep" else settings.fast_synthesis_model
        return ChatOpenAI(
            model=full_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=temperature
        )
    else:
        # OpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=temperature
        )


# =============================================================================
# Helper Functions
# =============================================================================

async def enhance_state_with_langchain(state: GraphState, thread_id: str) -> GraphState:
    """
    Enhance GraphState with LangChain memory, context, and RAG-retrieved documents.

    This is the central enrichment point - ALL frameworks benefit from:
    1. Chat history (conversation memory)
    2. Framework history (what worked before)
    3. RAG context (relevant code/docs from vector store)
    """
    memory = await get_memory(thread_id)
    context = memory.get_context()

    # Add conversation memory to working memory
    state["working_memory"]["chat_history"] = context["chat_history"]
    state["working_memory"]["framework_history"] = context["framework_history"]

    # ==========================================================================
    # AUTOMATIC RAG PRE-FETCH: Enrich ALL frameworks with relevant context
    # Only runs if embeddings are available (requires API key or HuggingFace)
    # ==========================================================================
    query = state.get("query", "")
    code_snippet = state.get("code_snippet", "")

    rag_context = []

    # Check if RAG is available (need either OpenAI key or HuggingFace installed)
    rag_available = False
    try:
        if settings.openai_api_key:
            rag_available = True
        else:
            # Check if HuggingFace is available as fallback
            from langchain_huggingface import HuggingFaceEmbeddings
            rag_available = True
    except ImportError:
        pass

    if not rag_available:
        logger.debug("rag_prefetch_skipped", reason="no embedding provider available")
        state["working_memory"]["rag_context"] = []
        state["working_memory"]["rag_context_formatted"] = ""
        return state

    # Search with query
    if query:
        try:
            from .collection_manager import get_collection_manager
            manager = get_collection_manager()

            # Search across all relevant collections
            query_docs = manager.search(
                query,
                collection_names=["frameworks", "documentation", "utilities"],
                k=3
            )
            for doc in query_docs:
                meta = doc.metadata or {}
                rag_context.append({
                    "source": meta.get("path", "unknown"),
                    "type": meta.get("chunk_type", "unknown"),
                    "content": doc.page_content[:1500],
                    "relevance": "query_match"
                })
            logger.info("rag_prefetch_query", docs_found=len(query_docs))
        except Exception as e:
            logger.warning("rag_prefetch_query_failed", error=str(e))

    # Search with code context if provided
    if code_snippet and len(code_snippet) > 50:
        try:
            from .collection_manager import get_collection_manager
            manager = get_collection_manager()

            # Extract key terms from code for better matching
            code_query = code_snippet[:500]  # First 500 chars for embedding
            code_docs = manager.search(
                code_query,
                collection_names=["frameworks", "utilities"],
                k=2
            )
            for doc in code_docs:
                meta = doc.metadata or {}
                # Avoid duplicates
                source = meta.get("path", "unknown")
                if not any(r["source"] == source for r in rag_context):
                    rag_context.append({
                        "source": source,
                        "type": meta.get("chunk_type", "unknown"),
                        "content": doc.page_content[:1000],
                        "relevance": "code_match"
                    })
            logger.info("rag_prefetch_code", docs_found=len(code_docs))
        except Exception as e:
            logger.warning("rag_prefetch_code_failed", error=str(e))

    # Store RAG context for frameworks to use
    state["working_memory"]["rag_context"] = rag_context
    state["working_memory"]["rag_context_formatted"] = _format_rag_context(rag_context)

    logger.info(
        "state_enriched",
        thread_id=thread_id,
        chat_messages=len(context["chat_history"]),
        rag_docs=len(rag_context)
    )

    return state


def _format_rag_context(rag_context: list) -> str:
    """Format RAG context for injection into prompts."""
    if not rag_context:
        return ""

    parts = ["## Relevant Context from Codebase\n"]
    for item in rag_context:
        # Use .get() for safe dictionary access to prevent KeyError
        source = item.get('source', 'unknown')
        item_type = item.get('type', 'unknown')
        content = item.get('content', '')
        parts.append(f"### {source} ({item_type})")
        parts.append(f"```\n{content}\n```\n")

    return "\n".join(parts)


async def save_to_langchain_memory(
    thread_id: str,
    query: str,
    answer: str,
    framework: str
) -> None:
    """Save interaction to LangChain memory."""
    memory = await get_memory(thread_id)
    memory.add_exchange(query, answer, framework)
