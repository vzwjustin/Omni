"""
Token Reduction Integration Module

Combines TOON serialization and LLMLingua-2 compression for optimal token usage.
This module provides utilities to reduce AI token costs by 30-80% through:

1. TOON (Token-Oriented Object Notation): 20-60% reduction for structured data
2. LLMLingua-2: 50-80% reduction for natural language prompts

Usage:
    # Automatic compression based on settings
    compressed = reduce_tokens(content, content_type="prompt")

    # Manual TOON serialization
    toon_data = serialize_to_toon(data_dict)

    # Manual LLMLingua compression
    compressed_prompt = compress_prompt(prompt_text, rate=0.5)
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from .context_utils import count_tokens
from .llmlingua_compressor import get_compressor
from .llmlingua_compressor import is_available as llmlingua_available
from .settings import get_settings
from .toon import TOONDecoder, TOONEncoder, get_token_savings

logger = structlog.get_logger(__name__)


class TokenReductionManager:
    """Manages token reduction strategies across the system."""

    def __init__(self):
        """Initialize token reduction manager."""
        self.settings = get_settings()
        self._toon_encoder: TOONEncoder | None = None
        self._toon_decoder: TOONDecoder | None = None
        self._llmlingua_available = llmlingua_available()

    @property
    def toon_encoder(self) -> TOONEncoder:
        """Get or create TOON encoder."""
        if self._toon_encoder is None:
            self._toon_encoder = TOONEncoder(
                delimiter=self.settings.toon_delimiter, threshold=self.settings.toon_array_threshold
            )
        return self._toon_encoder

    @property
    def toon_decoder(self) -> TOONDecoder:
        """Get or create TOON decoder."""
        if self._toon_decoder is None:
            self._toon_decoder = TOONDecoder(delimiter=self.settings.toon_delimiter)
        return self._toon_decoder

    def serialize_to_toon(self, data: Any) -> str:
        """
        Serialize data to TOON format.

        Args:
            data: Python object to serialize

        Returns:
            TOON-formatted string
        """
        if not self.settings.enable_toon_serialization:
            return json.dumps(data)

        try:
            return self.toon_encoder.encode(data)
        except Exception as e:
            logger.warning("TOON serialization failed, falling back to JSON", error=str(e))
            return json.dumps(data)

    def deserialize_from_toon(self, toon_str: str) -> Any:
        """
        Deserialize TOON format to Python object.

        Args:
            toon_str: TOON-formatted string

        Returns:
            Python object
        """
        if not self.settings.enable_toon_serialization:
            return json.loads(toon_str)

        try:
            return self.toon_decoder.decode(toon_str)
        except Exception as e:
            logger.warning("TOON deserialization failed, falling back to JSON", error=str(e))
            return json.loads(toon_str)

    def compress_prompt(
        self, prompt: str, rate: float | None = None, min_tokens: int | None = None
    ) -> dict[str, Any]:
        """
        Compress prompt using LLMLingua-2.

        Args:
            prompt: Prompt text to compress
            rate: Compression rate (None = use settings default)
            min_tokens: Only compress if prompt exceeds this token count

        Returns:
            Dict with compressed_prompt, original_tokens, compressed_tokens, etc.
        """
        if not self.settings.enable_llmlingua_compression:
            return {
                "compressed_prompt": prompt,
                "compressed": False,
                "reason": "LLMLingua compression disabled",
            }

        if not self._llmlingua_available:
            return {
                "compressed_prompt": prompt,
                "compressed": False,
                "reason": "LLMLingua not available",
            }

        # Check minimum token threshold
        min_tokens = min_tokens or self.settings.compression_min_tokens
        token_count = count_tokens(prompt)

        if token_count < min_tokens:
            return {
                "compressed_prompt": prompt,
                "compressed": False,
                "reason": f"Below minimum threshold ({token_count} < {min_tokens})",
            }

        try:
            compressor = get_compressor(
                model_name=self.settings.llmlingua_model_name, device=self.settings.llmlingua_device
            )

            rate = rate or self.settings.llmlingua_compression_rate

            result = compressor.compress(prompt, rate=rate)

            logger.info(
                "Prompt compressed with LLMLingua",
                original_tokens=result.get("origin_tokens", -1),
                compressed_tokens=result.get("compressed_tokens", -1),
                ratio=result.get("ratio", rate),
            )

            return result

        except Exception as e:
            logger.error("LLMLingua compression failed", error=str(e))
            return {"compressed_prompt": prompt, "compressed": False, "error": str(e)}

    def compress_context(
        self, instruction: str, context: str, rate: float | None = None
    ) -> dict[str, Any]:
        """
        Compress context while preserving instruction (useful for RAG).

        Args:
            instruction: User instruction to preserve
            context: Context to compress
            rate: Compression rate

        Returns:
            Dict with compressed result
        """
        if not self.settings.enable_llmlingua_compression or not self._llmlingua_available:
            return {"compressed_prompt": f"{instruction}\n\n{context}", "compressed": False}

        try:
            compressor = get_compressor(
                model_name=self.settings.llmlingua_model_name, device=self.settings.llmlingua_device
            )

            rate = rate or self.settings.llmlingua_compression_rate

            result = compressor.compress_with_instruction(
                instruction=instruction, context=context, rate=rate
            )

            return result

        except Exception as e:
            logger.error("Context compression failed", error=str(e))
            return {
                "compressed_prompt": f"{instruction}\n\n{context}",
                "compressed": False,
                "error": str(e),
            }

    def optimize_structured_data(self, data: dict | list) -> str:
        """
        Optimize structured data using TOON serialization.

        Args:
            data: Dictionary or list to optimize

        Returns:
            TOON-formatted string
        """
        return self.serialize_to_toon(data)

    def get_format_comparison(self, data: dict | list) -> dict[str, Any]:
        """
        Compare JSON vs TOON format efficiency.

        Args:
            data: Data to compare

        Returns:
            Dict with comparison statistics
        """
        json_str = json.dumps(data)
        toon_str = self.serialize_to_toon(data)

        json_tokens = count_tokens(json_str)
        toon_tokens = count_tokens(toon_str)

        savings = get_token_savings(json_str, toon_str)

        return {
            "json": {"chars": len(json_str), "tokens": json_tokens, "format": "json"},
            "toon": {"chars": len(toon_str), "tokens": toon_tokens, "format": "toon"},
            "savings": savings,
            "recommendation": "toon" if toon_tokens < json_tokens else "json",
        }


# Global singleton
_manager: TokenReductionManager | None = None


def get_manager() -> TokenReductionManager:
    """Get or create global token reduction manager."""
    global _manager
    if _manager is None:
        _manager = TokenReductionManager()
    return _manager


def serialize_to_toon(data: Any) -> str:
    """Convenience function to serialize data to TOON format."""
    return get_manager().serialize_to_toon(data)


def deserialize_from_toon(toon_str: str) -> Any:
    """Convenience function to deserialize TOON format."""
    return get_manager().deserialize_from_toon(toon_str)


def compress_prompt(prompt: str, rate: float | None = None) -> str:
    """Convenience function to compress prompt with LLMLingua-2."""
    result = get_manager().compress_prompt(prompt, rate=rate)
    return result.get("compressed_prompt", prompt)


def reduce_tokens(
    content: str, content_type: str = "text", auto_detect: bool = True
) -> dict[str, Any]:
    """
    Automatically reduce tokens based on content type and settings.

    Args:
        content: Content to reduce
        content_type: Type of content ("prompt", "context", "structured", "text")
        auto_detect: Auto-detect best strategy

    Returns:
        Dict with reduced content and metadata
    """
    manager = get_manager()
    settings = get_settings()

    # Try to parse as JSON for structured data
    if auto_detect or content_type == "structured":
        try:
            data = json.loads(content)
            if isinstance(data, (dict, list)):
                toon_str = manager.serialize_to_toon(data)
                return {
                    "content": toon_str,
                    "method": "toon",
                    "original_tokens": count_tokens(content),
                    "reduced_tokens": count_tokens(toon_str),
                    "reduction_percent": (1 - count_tokens(toon_str) / count_tokens(content)) * 100
                    if count_tokens(content) > 0
                    else 0.0,
                }
        except json.JSONDecodeError:
            pass

    # Use LLMLingua for prompts/context
    if content_type in ("prompt", "context") and settings.enable_llmlingua_compression:
        result = manager.compress_prompt(content)
        if result.get("compressed", False):
            return {
                "content": result["compressed_prompt"],
                "method": "llmlingua",
                "original_tokens": result.get("origin_tokens", -1),
                "reduced_tokens": result.get("compressed_tokens", -1),
                "reduction_percent": (1 - result.get("ratio", 1.0)) * 100,
            }

    # No reduction applied
    return {
        "content": content,
        "method": "none",
        "original_tokens": count_tokens(content),
        "reduced_tokens": count_tokens(content),
        "reduction_percent": 0,
    }


def get_reduction_stats(original: str, reduced: str, method: str) -> dict[str, Any]:
    """
    Calculate token reduction statistics.

    Args:
        original: Original content
        reduced: Reduced content
        method: Method used (toon, llmlingua, none)

    Returns:
        Statistics dictionary
    """
    original_tokens = count_tokens(original)
    reduced_tokens = count_tokens(reduced)

    return {
        "method": method,
        "original_chars": len(original),
        "reduced_chars": len(reduced),
        "original_tokens": original_tokens,
        "reduced_tokens": reduced_tokens,
        "tokens_saved": original_tokens - reduced_tokens,
        "reduction_percent": ((original_tokens - reduced_tokens) / original_tokens * 100)
        if original_tokens > 0
        else 0,
        "compression_ratio": reduced_tokens / original_tokens if original_tokens > 0 else 1.0,
    }
