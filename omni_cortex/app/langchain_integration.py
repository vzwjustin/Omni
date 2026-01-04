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

    def __init__(self, thread_id: str):
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
MAX_MEMORY_THREADS = 100


def get_memory(thread_id: str) -> OmniCortexMemory:
    """Get or create memory for a thread."""
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

@tool
async def search_documentation(query: str) -> str:
    """Search the indexed documentation/code via vector store."""
    logger.info("tool_called", tool="search_documentation", query=query)
    docs = search_vectorstore(query, k=5)
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
async def retrieve_context(query: str) -> str:
    """Retrieve recent chat and framework history as lightweight context."""
    logger.info("tool_called", tool="retrieve_context", query=query)
    # Pull most recent memory if available
    history_snippets = []
    for mem in _memory_store.values():
        if mem.messages:
            # Take last 6 messages (3 exchanges)
            recent = mem.messages[-6:]
            history_snippets.append("\n".join(str(m.content) for m in recent))
    if not history_snippets:
        return "No prior context available."
    return "Recent context:\n\n" + "\n\n---\n\n".join(history_snippets)


# Import enhanced search tools
try:
    from .enhanced_search_tools import ENHANCED_SEARCH_TOOLS
    _enhanced_tools = ENHANCED_SEARCH_TOOLS
except ImportError:
    _enhanced_tools = []

# Export available tools for MCP
AVAILABLE_TOOLS = [search_documentation, execute_code, retrieve_context] + _enhanced_tools


# =============================================================================
# Retrieval (RAG) Helpers
# =============================================================================

_vectorstore: Optional[Chroma] = None


def get_vectorstore() -> Optional[Chroma]:
    """Get or initialize a persistent Chroma vector store."""
    global _vectorstore
    if _vectorstore:
        return _vectorstore
    
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "/app/data/chroma")
    os.makedirs(persist_dir, exist_ok=True)
    
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.openai_api_key or settings.openrouter_api_key
        )
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


def get_vectorstore_by_collection(collection_name: str):
    """Get a specific collection from the manager."""
    from .collection_manager import get_collection_manager
    return get_collection_manager().get_collection(collection_name)


def search_vectorstore(query: str, k: int = 5) -> List[Document]:
    """Search the vector store for relevant documents."""
    vs = get_vectorstore()
    if not vs:
        return []
    try:
        return vs.similarity_search(query, k=k)
    except Exception as e:
        logger.error("vectorstore_search_failed", error=str(e))
        return []


# =============================================================================
# Callbacks for Monitoring
# =============================================================================

class OmniCortexCallback(BaseCallbackHandler):
    """
    Custom callback handler for tracking LLM usage, timing, and errors.
    """
    
    def __init__(self, thread_id: str):
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
        if hasattr(response, 'llm_output') and response.llm_output:
            tokens = response.llm_output.get('token_usage', {})
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

def get_chat_model(model_type: str = "deep") -> Any:
    """
    Get configured LangChain chat model.
    
    Args:
        model_type: "deep" for reasoning or "fast" for synthesis
    """
    if settings.llm_provider == "anthropic":
        model_name = settings.deep_reasoning_model if model_type == "deep" else settings.fast_synthesis_model
        # Remove provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[1]
        
        return ChatAnthropic(
            model=model_name,
            api_key=settings.anthropic_api_key,
            temperature=0.7 if model_type == "deep" else 0.5
        )
    else:
        # OpenAI or OpenRouter
        model_name = settings.deep_reasoning_model if model_type == "deep" else settings.fast_synthesis_model
        
        kwargs = {
            "model": model_name,
            "temperature": 0.7 if model_type == "deep" else 0.5
        }
        
        if settings.llm_provider == "openrouter":
            kwargs["api_key"] = settings.openrouter_api_key
            kwargs["base_url"] = settings.openrouter_base_url
        else:
            kwargs["api_key"] = settings.openai_api_key
        
        return ChatOpenAI(**kwargs)


# =============================================================================
# Helper Functions
# =============================================================================

def enhance_state_with_langchain(state: GraphState, thread_id: str) -> GraphState:
    """
    Enhance GraphState with LangChain memory and context.
    """
    memory = get_memory(thread_id)
    context = memory.get_context()
    
    # Add to working memory
    state["working_memory"]["chat_history"] = context["chat_history"]
    state["working_memory"]["framework_history"] = context["framework_history"]
    
    return state


def save_to_langchain_memory(
    thread_id: str,
    query: str,
    answer: str,
    framework: str
) -> None:
    """Save interaction to LangChain memory."""
    memory = get_memory(thread_id)
    memory.add_exchange(query, answer, framework)
