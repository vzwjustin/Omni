"""
MCP Sampling Infrastructure

Enables the MCP server to request completions from the client
(Claude Desktop/Claude Code) without making external API calls. This allows
multi-turn orchestration where the server coordinates reasoning
but the client does all the actual inference locally.

Note: Sampling requires the client to support the sampling capability.
Claude Code CLI support is tracked at: https://github.com/anthropics/claude-code/issues/1785

Configuration:
    Set ENABLE_MCP_SAMPLING=true to attempt MCP sampling.
    Default is false (use template mode for Claude Code compatibility).
"""

import os
import re
from typing import Optional
import structlog

logger = structlog.get_logger("sampling")

# Check if MCP sampling is explicitly enabled
# Default to False since Claude Code doesn't support it yet
SAMPLING_ENABLED = os.getenv("ENABLE_MCP_SAMPLING", "false").lower() == "true"


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
        import asyncio
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
        r'(?:score|rating):\s*(\d+(?:\.\d+)?)\s*/\s*(\d+)',  # "Score: 8/10"
        r'(?:score|rating):\s*(\d+(?:\.\d+)?)',  # "Score: 8.5"
        r'(\d+(?:\.\d+)?)\s*/\s*(\d+)',  # "8/10"
        r'(\d+(?:\.\d+)?)\s+out of\s+(\d+)',  # "8 out of 10"
        r'^(\d+(?:\.\d+)?)$',  # Just "8.5"
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


def extract_code_blocks(text: str) -> list[str]:
    """Extract code blocks from markdown-formatted text."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


def extract_json_object(text: str) -> Optional[dict]:
    """Attempt to extract a JSON object from text."""
    import json

    # Try to find JSON in markdown code blocks first
    json_pattern = r"```json\n(.*?)```"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to parse the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object anywhere in text
    json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_obj_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None
