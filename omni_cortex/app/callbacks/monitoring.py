"""
Monitoring Callback for Omni-Cortex

Tracks LLM usage, timing, tokens, and errors.
"""

from typing import Any, List, Dict

from langchain_core.callbacks import BaseCallbackHandler
import structlog

from ..core.constants import CONTENT

logger = structlog.get_logger("callbacks")


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
            input=input_str[:CONTENT.QUERY_LOG]
        )

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Track tool completion."""
        logger.info("tool_end", thread_id=self.thread_id, output_length=len(output))
