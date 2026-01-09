"""
MCP Sampling Infrastructure

Enables the MCP server to request completions from the client
(Claude Desktop/Claude Code) without making external API calls. This allows
multi-turn orchestration where the server coordinates reasoning
but the client does all the actual inference locally.

Note: Sampling requires the client to support the sampling capability.
Claude Code CLI support is tracked at: https://github.com/anthropics/claude-code/issues/1785

Configuration:
    ENABLE_MCP_SAMPLING=true  - Attempt MCP sampling (default: false)
    USE_LANGCHAIN_LLM=true    - Fall back to LangChain direct API calls (default: false)

When both are false, template mode is used (server returns prompts, client executes).
"""

import asyncio
import re
import time
from typing import Optional
import structlog

from .settings import get_settings
from .errors import LLMError, SamplerTimeout, SamplerCircuitOpen

# Import from canonical location to avoid duplication
from ..nodes.common import extract_code_blocks

logger = structlog.get_logger("sampling")

# Get settings
_settings = get_settings()

# Check if MCP sampling is explicitly enabled
# Default to False since Claude Code doesn't support it yet
SAMPLING_ENABLED = _settings.enable_mcp_sampling

# Check if LangChain LLM fallback is enabled
# When true, uses direct API calls via LangChain when sampling fails
# Requires API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY)
LANGCHAIN_LLM_ENABLED = _settings.use_langchain_llm


class SamplingNotSupportedError(Exception):
    """Raised when the MCP client doesn't support sampling."""
    pass


class ClientSampler:
    """
    Handles requesting samples from the MCP client.

    The client (Claude Desktop/Claude Code) executes these locally - no external APIs.
    """

    def __init__(self, server, context=None):
        """
        Args:
            server: MCP Server instance (passed during initialization)
            context: Optional Context object from FastMCP (provides session access)
        """
        self.server = server
        self.context = context
        self._sampling_supported = None  # Cached capability check

    async def request_sample(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Request a completion from the MCP client.

        The client executes this using its local model (no API call).

        Args:
            prompt: The prompt to send to the client
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system instruction

        Returns:
            The client's response as a string

        Raises:
            SamplingNotSupportedError: If the client doesn't support sampling
        """
        # Check if sampling is enabled (default: disabled for Claude Code compatibility)
        if not SAMPLING_ENABLED:
            raise SamplingNotSupportedError(
                "MCP sampling disabled. Set ENABLE_MCP_SAMPLING=true to enable. "
                "Note: Claude Code doesn't support sampling yet (Issue #1785)."
            )

        from mcp.types import SamplingMessage, TextContent

        # Build message with proper TextContent object (not dict)
        messages = [
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=prompt)
            )
        ]

        # Get session from context or server
        session = None
        try:
            if self.context and hasattr(self.context, 'session'):
                session = self.context.session
            elif hasattr(self.server, 'request_context') and self.server.request_context:
                session = self.server.request_context.session
        except Exception as e:
            logger.warning("failed_to_get_session", error=str(e))

        if not session:
            raise SamplingNotSupportedError(
                "No active session found. The MCP client may not support sampling."
            )

        # Check if session has create_message capability
        if not hasattr(session, 'create_message'):
            raise SamplingNotSupportedError(
                "Session doesn't have create_message method. Client may not support sampling."
            )

        # Try the sampling call with timeout protection
        try:
            # Use a timeout to detect clients that don't respond to sampling
            result = await asyncio.wait_for(
                session.create_message(
                    messages=messages,
                    max_tokens=max_tokens,
                ),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            raise SamplingNotSupportedError(
                "Sampling request timed out. Client may not support sampling."
            )
        except AttributeError as e:
            raise SamplingNotSupportedError(
                f"Session doesn't support create_message: {e}"
            )
        except NotImplementedError as e:
            raise SamplingNotSupportedError(
                f"Client doesn't implement sampling: {e}"
            )
        except Exception as e:
            error_str = str(e)
            if "sampling" in error_str.lower() or "not supported" in error_str.lower():
                raise SamplingNotSupportedError(f"Sampling not supported: {e}")
            if "timeout" in error_str.lower():
                raise SamplingNotSupportedError(f"Sampling timed out: {e}")
            # Re-raise other errors
            logger.error("sampling_request_failed", error=error_str)
            raise

        # Extract text from response
        if hasattr(result, 'content'):
            if isinstance(result.content, str):
                return result.content
            elif isinstance(result.content, dict) and 'text' in result.content:
                return result.content['text']
            elif hasattr(result.content, 'text'):
                return result.content.text

        # Fallback: convert to string
        return str(result)


class ResilientSampler:
    """Wrapper around ClientSampler with timeout, retry, and circuit breaker."""

    def __init__(
        self,
        sampler: ClientSampler,
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_threshold: int = 5,
        circuit_reset_time: float = 60.0
    ):
        """
        Initialize a resilient sampler wrapper.

        Args:
            sampler: The underlying ClientSampler instance
            timeout: Default timeout in seconds for requests
            max_retries: Maximum number of retry attempts
            circuit_threshold: Number of failures before circuit opens
            circuit_reset_time: Seconds to wait before allowing retry after circuit opens
        """
        self._sampler = sampler
        self._timeout = timeout
        self._max_retries = max_retries
        self._failure_count = 0
        self._circuit_open_until: Optional[float] = None
        self._circuit_threshold = circuit_threshold
        self._circuit_reset_time = circuit_reset_time

    async def request_sample(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> str:
        """
        Request a sample with timeout, retry, and circuit breaker protection.

        Args:
            prompt: The prompt to send to the client
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system instruction
            timeout: Override the default timeout for this request

        Returns:
            The client's response as a string

        Raises:
            SamplerCircuitOpen: If circuit breaker is open
            SamplerTimeout: If all retries timed out
            LLMError: If all retries failed for other reasons
        """
        # Check circuit breaker
        if self._circuit_open_until and time.time() < self._circuit_open_until:
            remaining = self._circuit_open_until - time.time()
            logger.warning(
                "circuit_breaker_open",
                remaining_seconds=remaining,
                failure_count=self._failure_count
            )
            raise SamplerCircuitOpen(
                f"Circuit breaker open, retry after {remaining:.0f}s"
            )

        effective_timeout = timeout or self._timeout
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                logger.debug(
                    "sampler_request_attempt",
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    timeout=effective_timeout
                )

                result = await asyncio.wait_for(
                    self._sampler.request_sample(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt
                    ),
                    timeout=effective_timeout
                )

                # Success - reset failure count
                self._failure_count = 0
                self._circuit_open_until = None
                logger.debug("sampler_request_success", attempt=attempt + 1)
                return result

            except asyncio.TimeoutError:
                last_error = SamplerTimeout(
                    f"Request timed out after {effective_timeout}s (attempt {attempt + 1}/{self._max_retries})"
                )
                logger.warning(
                    "sampler_timeout",
                    attempt=attempt + 1,
                    timeout=effective_timeout
                )

            except SamplingNotSupportedError:
                # Don't retry for unsupported sampling - propagate immediately
                raise

            except Exception as e:
                last_error = e
                logger.warning(
                    "sampler_request_error",
                    attempt=attempt + 1,
                    error=str(e),
                    error_type=type(e).__name__
                )

            # Record failure
            self._failure_count += 1

            # Check if circuit breaker should open
            if self._failure_count >= self._circuit_threshold:
                self._circuit_open_until = time.time() + self._circuit_reset_time
                logger.error(
                    "circuit_breaker_opened",
                    failure_count=self._failure_count,
                    reset_after_seconds=self._circuit_reset_time
                )

            # Exponential backoff before next retry
            if attempt < self._max_retries - 1:
                backoff_time = 2 ** attempt  # 1s, 2s, 4s...
                logger.debug("sampler_backoff", seconds=backoff_time)
                await asyncio.sleep(backoff_time)

        # All retries exhausted
        if last_error:
            if isinstance(last_error, SamplerTimeout):
                raise last_error
            raise LLMError(f"Sampler failed after {self._max_retries} retries: {last_error}")
        raise LLMError(f"Sampler failed after {self._max_retries} retries")

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker state."""
        self._failure_count = 0
        self._circuit_open_until = None
        logger.info("circuit_breaker_reset")

    @property
    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is currently open."""
        if self._circuit_open_until is None:
            return False
        return time.time() < self._circuit_open_until

    @property
    def failure_count(self) -> int:
        """Get the current failure count."""
        return self._failure_count


def extract_score(text: str, default: float = 0.5) -> float:
    """
    Extract a numeric score from LLM output.

    Handles various formats:
    - "Score: 8/10"
    - "8.5"
    - "Rating: 7 out of 10"
    - etc.

    Returns:
        Score normalized to 0.0-1.0 range
    """
    # Try to find score/rating patterns
    patterns = [
        r'(?:score|rating):\s*(-?\d+(?:\.\d+)?)\s*/\s*(\d+)',  # "Score: 8/10" or "-5/10"
        r'(?:score|rating):\s*(-?\d+(?:\.\d+)?)',  # "Score: 8.5" or "-5"
        r'(-?\d+(?:\.\d+)?)\s*/\s*(\d+)',  # "8/10"
        r'(-?\d+(?:\.\d+)?)\s+out of\s+(\d+)',  # "8 out of 10"
        r'^(-?\d+(?:\.\d+)?)$',  # Just "8.5"
    ]

    text_lower = text.lower().strip()

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                # Format: X/Y or X out of Y
                numerator = float(groups[0])
                denominator = float(groups[1])
                return min(1.0, max(0.0, numerator / denominator))
            elif len(groups) == 1:
                # Format: X (assume out of 10)
                score = float(groups[0])
                if score <= 1.0:
                    return score  # Already normalized
                else:
                    return min(1.0, max(0.0, score / 10.0))

    # Couldn't parse - return default
    return default


def extract_json_object(text: str) -> Optional[dict]:
    """Attempt to extract a JSON object from text."""
    import json

    # Try to find JSON in markdown code blocks first
    json_pattern = r"```json\n(.*?)```"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            # Intentional fallback: code block content wasn't valid JSON,
            # continue to try parsing full text or regex extraction
            logger.debug("json_code_block_parse_failed", error=str(e), text_preview=text[:100])

    # Try to parse the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Intentional fallback: full text isn't valid JSON,
        # continue to try regex-based extraction
        logger.debug("json_full_text_parse_failed", error=str(e), text_preview=text[:100])

    # Try to find JSON object anywhere in text
    json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_obj_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError as e:
            # Intentional fallback: this regex match wasn't valid JSON,
            # continue to try next match in the list
            logger.debug("json_regex_match_parse_failed", error=str(e), match_preview=match[:50])
            continue

    return None


# =============================================================================
# LangChain LLM Fallback
# =============================================================================

async def call_llm_with_fallback(
    prompt: str,
    sampler: Optional[ClientSampler] = None,
    max_tokens: int = 4000,
    temperature: float = 0.7
) -> str:
    """
    Try MCP sampling first, fall back to LangChain direct API if enabled.

    Priority:
    1. MCP Sampling (if ENABLE_MCP_SAMPLING=true and client supports it)
    2. LangChain direct API (if USE_LANGCHAIN_LLM=true)
    3. Raise SamplingNotSupportedError (caller should use template mode)

    Args:
        prompt: The prompt to send
        sampler: Optional ClientSampler instance for MCP sampling
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature

    Returns:
        LLM response text

    Raises:
        SamplingNotSupportedError: If neither sampling nor LangChain fallback available
    """
    # Try MCP sampling first (if enabled and sampler provided)
    if SAMPLING_ENABLED and sampler:
        try:
            return await sampler.request_sample(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except SamplingNotSupportedError:
            logger.info("mcp_sampling_failed_trying_fallback")
            # Fall through to LangChain fallback

    # Try LangChain direct API (if enabled)
    if LANGCHAIN_LLM_ENABLED:
        try:
            logger.info("using_langchain_llm_fallback")
            from ..nodes.common import call_deep_reasoner

            # Create minimal state for the LLM call
            state = {
                "tokens_used": 0,
                "working_memory": {},
                "quiet_thoughts": []
            }

            response, tokens = await call_deep_reasoner(
                prompt=prompt,
                state=state,
                max_tokens=max_tokens,
                temperature=temperature
            )

            logger.info("langchain_llm_response", tokens=tokens)
            return response

        except Exception as e:
            logger.error("langchain_llm_fallback_failed", error=str(e))
            raise SamplingNotSupportedError(
                f"LangChain LLM fallback failed: {e}. "
                "Check that LLM_PROVIDER and API key are configured."
            )

    # Neither sampling nor LangChain available
    raise SamplingNotSupportedError(
        "No LLM backend available. Options:\n"
        "1. Set ENABLE_MCP_SAMPLING=true (requires client support)\n"
        "2. Set USE_LANGCHAIN_LLM=true + API key (direct API calls)\n"
        "3. Use template mode (default - server returns prompts)"
    )
