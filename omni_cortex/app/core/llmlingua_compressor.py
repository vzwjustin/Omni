"""
LLMLingua-2 Prompt Compression Module

Microsoft's LLMLingua-2 is a prompt compression method that can reduce tokens by 50-80%
while maintaining semantic meaning through BERT-based token classification.

Key Features:
- Task-agnostic compression (works across various use cases)
- 3x-6x faster than LLMLingua v1
- Trained via data distillation from GPT-4
- Supports up to 20x compression with minimal performance loss
- Better handling of out-of-domain data

References:
- https://github.com/microsoft/LLMLingua
- https://llmlingua.com/llmlingua2.html
- https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/
"""

import os
from typing import Dict, List, Optional, Union, Any
import structlog

logger = structlog.get_logger(__name__)


class LLMLinguaCompressor:
    """Wrapper for Microsoft's LLMLingua-2 prompt compression."""

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize LLMLingua-2 compressor.

        Args:
            model_name: HuggingFace model identifier for LLMLingua-2
            device: Device to run model on ("cpu", "cuda", "mps")
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/llmlingua")
        self._compressor = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization to avoid loading model on import."""
        if self._initialized:
            return

        try:
            from llmlingua import PromptCompressor

            logger.info(
                "Initializing LLMLingua-2 compressor",
                model=self.model_name,
                device=self.device
            )

            self._compressor = PromptCompressor(
                model_name=self.model_name,
                device_map=self.device,
                model_config={"revision": "main"},
                open_api_config={}
            )

            self._initialized = True
            logger.info("LLMLingua-2 compressor initialized successfully")

        except ImportError as e:
            logger.warning(
                "LLMLingua not installed, compression unavailable",
                error=str(e)
            )
            raise ImportError(
                "llmlingua package not installed. Install with: pip install llmlingua"
            ) from e
        except Exception as e:
            logger.error("Failed to initialize LLMLingua-2 compressor", error=str(e))
            raise

    def compress(
        self,
        prompt: str,
        rate: float = 0.5,
        target_token: Optional[int] = None,
        force_tokens: Optional[List[str]] = None,
        force_reserve_digit: bool = False,
        drop_consecutive: bool = True,
        chunk_end_tokens: Optional[List[str]] = None,
        return_word_label: bool = False
    ) -> Dict[str, Any]:
        """
        Compress prompt using LLMLingua-2.

        Args:
            prompt: Original prompt text to compress
            rate: Target compression rate (0.5 = 50% of original tokens)
            target_token: Explicit target token count (overrides rate)
            force_tokens: Tokens that must be preserved (e.g., ['\n', '.', '!', '?'])
            force_reserve_digit: Force preservation of digit tokens
            drop_consecutive: Drop consecutive identical tokens
            chunk_end_tokens: Tokens that mark chunk boundaries (e.g., ['.', '\n'])
            return_word_label: Include word-level compression labels in output

        Returns:
            Dictionary with compressed_prompt, origin_tokens, compressed_tokens,
            ratio, and optionally word_labels
        """
        self._lazy_init()

        if not self._compressor:
            logger.warning("Compressor not available, returning original prompt")
            return {
                "compressed_prompt": prompt,
                "origin_tokens": -1,
                "compressed_tokens": -1,
                "ratio": 1.0,
                "compressed": False,
                "error": "Compressor not initialized"
            }

        try:
            # Set default force tokens if not provided
            if force_tokens is None:
                force_tokens = ['\n', '.', '!', '?', ',']

            # Set default chunk end tokens
            if chunk_end_tokens is None:
                chunk_end_tokens = ['.', '\n']

            logger.debug(
                "Compressing prompt",
                original_length=len(prompt),
                target_rate=rate,
                target_tokens=target_token
            )

            # Use LLMLingua-2 compression
            result = self._compressor.compress_prompt_llmlingua2(
                prompt,
                rate=rate,
                target_token=target_token,
                force_tokens=force_tokens,
                force_reserve_digit=force_reserve_digit,
                drop_consecutive=drop_consecutive,
                chunk_end_tokens=chunk_end_tokens,
                return_word_label=return_word_label
            )

            # Add metadata
            result["compressed"] = True
            result["ratio"] = result.get("ratio", rate)

            logger.info(
                "Prompt compressed successfully",
                original_tokens=result.get("origin_tokens", -1),
                compressed_tokens=result.get("compressed_tokens", -1),
                ratio=result["ratio"]
            )

            return result

        except Exception as e:
            logger.error("Prompt compression failed", error=str(e))
            return {
                "compressed_prompt": prompt,
                "origin_tokens": -1,
                "compressed_tokens": -1,
                "ratio": 1.0,
                "compressed": False,
                "error": str(e)
            }

    def compress_with_instruction(
        self,
        instruction: str,
        context: str,
        rate: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compress context while preserving instruction.

        This is useful for RAG scenarios where you want to compress retrieved
        context but keep the user's instruction intact.

        Args:
            instruction: User instruction to preserve
            context: Context to compress
            rate: Target compression rate for context
            **kwargs: Additional arguments for compress()

        Returns:
            Dictionary with compressed result and reassembled prompt
        """
        self._lazy_init()

        try:
            # Compress only the context
            context_result = self.compress(context, rate=rate, **kwargs)

            # Reassemble with original instruction
            compressed_prompt = f"{instruction}\n\n{context_result['compressed_prompt']}"

            return {
                "compressed_prompt": compressed_prompt,
                "original_instruction": instruction,
                "compressed_context": context_result["compressed_prompt"],
                "origin_tokens": context_result.get("origin_tokens", -1),
                "compressed_tokens": context_result.get("compressed_tokens", -1),
                "ratio": context_result.get("ratio", rate),
                "compressed": True
            }

        except Exception as e:
            logger.error("Instruction-context compression failed", error=str(e))
            return {
                "compressed_prompt": f"{instruction}\n\n{context}",
                "compressed": False,
                "error": str(e)
            }

    def compress_multiple(
        self,
        prompts: List[str],
        rate: float = 0.5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Compress multiple prompts in batch.

        Args:
            prompts: List of prompts to compress
            rate: Target compression rate
            **kwargs: Additional arguments for compress()

        Returns:
            List of compression results
        """
        self._lazy_init()

        results = []
        for i, prompt in enumerate(prompts):
            logger.debug(f"Compressing prompt {i+1}/{len(prompts)}")
            result = self.compress(prompt, rate=rate, **kwargs)
            results.append(result)

        return results

    def get_compression_stats(
        self,
        original_prompt: str,
        compressed_result: Dict[str, Any]
    ) -> Dict[str, Union[int, float]]:
        """
        Calculate detailed compression statistics.

        Args:
            original_prompt: Original prompt text
            compressed_result: Result from compress()

        Returns:
            Dictionary with detailed statistics
        """
        original_chars = len(original_prompt)
        compressed_chars = len(compressed_result.get("compressed_prompt", ""))

        origin_tokens = compressed_result.get("origin_tokens", -1)
        compressed_tokens = compressed_result.get("compressed_tokens", -1)

        return {
            "original_chars": original_chars,
            "compressed_chars": compressed_chars,
            "char_reduction": original_chars - compressed_chars,
            "char_reduction_percent": ((original_chars - compressed_chars) / original_chars * 100)
            if original_chars > 0 else 0,
            "origin_tokens": origin_tokens,
            "compressed_tokens": compressed_tokens,
            "token_reduction": origin_tokens - compressed_tokens if origin_tokens > 0 else -1,
            "token_reduction_percent": ((origin_tokens - compressed_tokens) / origin_tokens * 100)
            if origin_tokens > 0 else 0,
            "compression_ratio": compressed_result.get("ratio", 1.0)
        }


# Global singleton instance (lazy-loaded)
_compressor_instance: Optional[LLMLinguaCompressor] = None


def get_compressor(
    model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    device: str = "cpu"
) -> LLMLinguaCompressor:
    """
    Get or create global LLMLingua compressor instance.

    Args:
        model_name: HuggingFace model identifier
        device: Device to run model on

    Returns:
        LLMLingua compressor instance
    """
    global _compressor_instance

    if _compressor_instance is None:
        _compressor_instance = LLMLinguaCompressor(
            model_name=model_name,
            device=device
        )

    return _compressor_instance


def compress_prompt(
    prompt: str,
    rate: float = 0.5,
    **kwargs
) -> str:
    """
    Convenience function to compress a prompt and return only the compressed text.

    Args:
        prompt: Original prompt
        rate: Compression rate
        **kwargs: Additional arguments for compress()

    Returns:
        Compressed prompt text
    """
    compressor = get_compressor()
    result = compressor.compress(prompt, rate=rate, **kwargs)
    return result.get("compressed_prompt", prompt)


def is_available() -> bool:
    """
    Check if LLMLingua is available.

    Returns:
        True if llmlingua package is installed
    """
    try:
        import llmlingua  # noqa: F401
        return True
    except ImportError:
        return False
