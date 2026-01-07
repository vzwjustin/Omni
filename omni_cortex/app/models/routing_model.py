"""
Routing Model for Omni-Cortex

Gemini-based model specifically for task routing and analysis.
Always uses Gemini regardless of LLM_PROVIDER setting.
"""

import asyncio
from typing import Any

import structlog

from ..core.settings import get_settings

logger = structlog.get_logger("routing_model")


class GeminiRoutingWrapper:
    """
    Wrapper to make native Gemini SDK compatible with LangChain-style ainvoke().
    Enables Google Search grounding for fresh context.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    async def ainvoke(self, prompt: str) -> Any:
        """Async invoke with search grounding."""
        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt
        )
        return GeminiResponse(response)

    def __repr__(self) -> str:
        model_name = getattr(self.model, 'model_name', 'unknown')
        return f"GeminiRoutingWrapper(model={model_name})"


class GeminiResponse:
    """Response wrapper for compatibility with existing code."""

    def __init__(self, response: Any) -> None:
        self._response = response

    @property
    def content(self) -> str:
        """Extract text content from Gemini response."""
        try:
            return self._response.text
        except (AttributeError, ValueError) as e:
            # Fallback for different response formats (blocked responses, etc.)
            logger.debug("gemini_response_fallback", error=str(e))
            if hasattr(self._response, 'candidates') and self._response.candidates:
                parts = self._response.candidates[0].content.parts
                return "".join(p.text for p in parts if hasattr(p, 'text'))
            return str(self._response)

    def __repr__(self) -> str:
        try:
            c = self.content
            preview = (c[:50] + "...") if len(c) > 50 else c
        except Exception:
            preview = "<unavailable>"
        return f"GeminiResponse({preview!r})"


def get_routing_model() -> GeminiRoutingWrapper:
    """
    Get Gemini model specifically for routing/task analysis.

    This always uses Gemini regardless of LLM_PROVIDER setting.
    Used by HyperRouter._gemini_analyze_task() to offload thinking from Claude Code.

    Features:
    - Google Search grounding for fresh docs/APIs
    - Native SDK for full feature access
    """
    settings = get_settings()
    if not settings.google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY required for routing. "
            "Gemini does task analysis so Claude Code can focus on execution."
        )

    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)

    # Use fast model for routing - Gemini 3 Flash is ideal
    model_name = settings.routing_model
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
