"""
OmniCortexMemory - Unified Memory System

Provides short-term conversation memory using LangChain 1.0+ message types.
"""

import threading
from typing import Any, List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import structlog

logger = structlog.get_logger("memory")


class OmniCortexMemory:
    """
    Unified memory system using LangChain 1.0+ message types.

    Provides:
    - Conversation buffer for recent exchanges (list of messages)
    - Framework history tracking
    - Simple and lightweight for pass-through mode
    """

    def __init__(self, thread_id: str, max_messages: int = 20) -> None:
        self.thread_id = thread_id
        self.messages: List[BaseMessage] = []
        self.framework_history: List[str] = []
        self.max_messages = max_messages
        self._lock = threading.Lock()  # Protect concurrent modifications

    def add_exchange(self, query: str, answer: str, framework: str) -> None:
        """Add a query-answer exchange to memory (thread-safe)."""
        with self._lock:
            self.messages.append(HumanMessage(content=query))
            self.messages.append(AIMessage(content=answer))

            # Trim to max size
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]

            self.framework_history.append(framework)
            # Trim framework history to match message limit (prevent unbounded growth)
            if len(self.framework_history) > self.max_messages:
                self.framework_history = self.framework_history[-self.max_messages:]
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

    def __repr__(self) -> str:
        return (
            f"OmniCortexMemory(thread={self.thread_id!r}, "
            f"messages={len(self.messages)}, "
            f"frameworks={len(self.framework_history)})"
        )
