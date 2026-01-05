"""
MCP Sampling Infrastructure

Enables the MCP server to request completions from the client
(Claude Desktop) without making external API calls. This allows
multi-turn orchestration where the server coordinates reasoning
but the client does all the actual inference locally.
"""

import re
from typing import Optional


class ClientSampler:
    """
    Handles requesting samples from the MCP client.

    The client (Claude Desktop) executes these locally - no external APIs.
    """

    def __init__(self, server):
        """
        Args:
            server: MCP Server instance (passed during initialization)
        """
        self.server = server

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
        """
        from mcp.types import SamplingMessage, CreateMessageRequest

        messages = [SamplingMessage(role="user", content={"type": "text", "text": prompt})]

        request = CreateMessageRequest(
            messages=messages,
            modelPreferences={
                "hints": [{"name": "claude-3-5-sonnet"}],  # Preference hint
                "costPriority": 0.5,
                "speedPriority": 0.5,
                "intelligencePriority": 1.0
            },
            systemPrompt=system_prompt or "",
            maxTokens=max_tokens,
            temperature=temperature,
            includeContext="thisServer"
        )

        # Request sampling from client
        result = await self.server.request_sampling(request)

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
