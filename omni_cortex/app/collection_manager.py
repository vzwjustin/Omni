"""
Multi-Collection Manager for Specialized Retrieval

Manages multiple Chroma collections for different content types,
enabling precise retrieval based on context.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import structlog

from .core.config import settings

logger = structlog.get_logger("collection-manager")


class CollectionManager:
    """Manages multiple specialized Chroma collections."""
    
    COLLECTIONS = {
        "frameworks": "Framework implementations and reasoning nodes",
        "documentation": "Markdown docs, READMEs, guides",
        "configs": "Configuration files and environment settings",
        "utilities": "Utility functions and helpers",
        "tests": "Test files and fixtures",
        "integrations": "LangChain/LangGraph integration code"
    }
    
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "/app/data/chroma")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self._embedding_function = None
        self._collections: Dict[str, Chroma] = {}
    
    def get_embedding_function(self):
        """Lazy initialization of embedding function."""
        if self._embedding_function is None:
            api_key = settings.openai_api_key or settings.openrouter_api_key
            if not api_key:
                logger.error("embedding_init_failed", error="No API key configured (OPENAI_API_KEY or OPENROUTER_API_KEY)")
                return None
            try:
                self._embedding_function = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    api_key=api_key
                )
            except Exception as e:
                logger.error("embedding_init_failed", error=str(e))
                return None
        return self._embedding_function
    
    def get_collection(self, collection_name: str) -> Optional[Chroma]:
        """Get or create a collection."""
        if collection_name in self._collections:
            return self._collections[collection_name]
        
        if collection_name not in self.COLLECTIONS:
            logger.warning("unknown_collection", name=collection_name)
            return None
        
        try:
            collection = Chroma(
                collection_name=f"omni-cortex-{collection_name}",
                persist_directory=self.persist_dir,
                embedding_function=self.get_embedding_function()
            )
            self._collections[collection_name] = collection
            logger.info("collection_loaded", name=collection_name)
            return collection
        except Exception as e:
            logger.error("collection_load_failed", name=collection_name, error=str(e))
            return None
    
    def search(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search across one or more collections.
        
        Args:
            query: Search query
            collection_names: Collections to search (default: all)
            k: Number of results per collection
            filter_dict: Metadata filters (e.g., {"category": "framework"})
        """
        if collection_names is None:
            collection_names = list(self.COLLECTIONS.keys())
        
        all_results = []
        
        for coll_name in collection_names:
            collection = self.get_collection(coll_name)
            if not collection:
                continue
            
            try:
                if filter_dict:
                    results = collection.similarity_search(
                        query,
                        k=k,
                        filter=filter_dict
                    )
                else:
                    results = collection.similarity_search(query, k=k)
                
                all_results.extend(results)
                logger.debug("search_complete", collection=coll_name, results=len(results))
            except Exception as e:
                logger.error("search_failed", collection=coll_name, error=str(e))
        
        # Sort by relevance (if scores available) and deduplicate
        return self._deduplicate_results(all_results)
    
    def search_frameworks(
        self,
        query: str,
        framework_name: Optional[str] = None,
        framework_category: Optional[str] = None,
        k: int = 5
    ) -> List[Document]:
        """Search specifically in framework code."""
        filter_dict = {}
        if framework_name:
            filter_dict["framework_name"] = framework_name
        if framework_category:
            filter_dict["framework_category"] = framework_category
        
        return self.search(
            query,
            collection_names=["frameworks"],
            k=k,
            filter_dict=filter_dict if filter_dict else None
        )
    
    def search_documentation(self, query: str, k: int = 5) -> List[Document]:
        """Search specifically in documentation."""
        return self.search(query, collection_names=["documentation"], k=k)
    
    def search_by_function(self, function_name: str, k: int = 3) -> List[Document]:
        """Find specific function implementations."""
        return self.search(
            function_name,
            collection_names=["frameworks", "utilities"],
            k=k,
            filter_dict={"chunk_type": "function"}
        )
    
    def search_by_class(self, class_name: str, k: int = 3) -> List[Document]:
        """Find specific class implementations."""
        return self.search(
            class_name,
            collection_names=["frameworks", "utilities"],
            k=k,
            filter_dict={"chunk_type": "class"}
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        collection_name: str = "frameworks"
    ) -> int:
        """Add documents to a specific collection."""
        collection = self.get_collection(collection_name)
        if not collection:
            return 0
        
        try:
            collection.add_texts(texts=texts, metadatas=metadatas)
            # Note: persist() is no longer needed in Chroma 0.4+ with persist_directory
            logger.info("documents_added", collection=collection_name, count=len(texts))
            return len(texts)
        except Exception as e:
            logger.error("add_documents_failed", collection=collection_name, error=str(e))
            return 0
    
    def route_to_collection(self, metadata: Dict[str, Any]) -> str:
        """Determine which collection a document belongs to based on metadata."""
        category = metadata.get("category", "")
        file_type = metadata.get("file_type", "")
        
        if category == "framework":
            return "frameworks"
        elif category == "documentation":
            return "documentation"
        elif category == "config":
            return "configs"
        elif category == "test":
            return "tests"
        elif category in ["integration", "server"]:
            return "integrations"
        else:
            return "utilities"
    
    @staticmethod
    def _deduplicate_results(results: List[Document]) -> List[Document]:
        """Remove duplicate results based on content."""
        seen = set()
        unique = []
        
        for doc in results:
            # Use path + chunk_index as unique key
            metadata = doc.metadata or {}
            key = f"{metadata.get('path', '')}:{metadata.get('chunk_index', 0)}"
            
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        
        return unique


# Global collection manager instance
_collection_manager: Optional[CollectionManager] = None


def get_collection_manager() -> CollectionManager:
    """Get or create the global collection manager."""
    global _collection_manager
    if _collection_manager is None:
        _collection_manager = CollectionManager()
    return _collection_manager
