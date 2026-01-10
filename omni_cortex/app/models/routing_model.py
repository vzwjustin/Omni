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
    Uses deprecated google.generativeai package (fallback).
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


class GeminiRoutingWrapperNew:
    """
    Wrapper using new google-genai package.
    Enables Google Search grounding for fresh context.
    """

    def __init__(self, client: Any, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    async def ainvoke(self, prompt: str) -> Any:
        """Async invoke with search grounding."""
        from google.genai import types

        # Google Search grounding tool
        google_search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=4096,
                tools=[google_search_tool]
            )
        )
        return GeminiResponse(response)

    def __repr__(self) -> str:
        return f"GeminiRoutingWrapperNew(model={self._model_name})"


class GeminiResponse:
    """Response wrapper for compatibility with existing code."""

    def __init__(self, response: Any) -> None:
        self._response = response

    @property
    def content(self) -> str:
        """Extract text content from Gemini response.

        Defensive implementation that handles various failure modes:
        - Missing/None response
        - Blocked responses (no text attribute)
        - Malformed candidates structure
        - Non-string return values
        """
        if self._response is None:
            return "<no response>"

        # Try direct .text access first (most common case)
        try:
            text = self._response.text
            if isinstance(text, str):
                return text
            # Handle non-string text (shouldn't happen but be safe)
            return str(text) if text is not None else "<empty>"
        except (AttributeError, ValueError, TypeError) as e:
            logger.debug("gemini_response_text_fallback", error=str(e)[:100])
        except Exception as e:
            # Catch unexpected errors from .text property
            logger.debug("gemini_response_text_unexpected", error=str(e)[:100])

        # Try candidates fallback
        try:
            if hasattr(self._response, 'candidates') and self._response.candidates:
                candidate = self._response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    text_parts = [p.text for p in parts if hasattr(p, 'text') and p.text]
                    if text_parts:
                        return "".join(text_parts)
        except (IndexError, AttributeError, TypeError) as e:
            logger.debug("gemini_response_candidates_fallback", error=str(e)[:100])
        except Exception as e:
            logger.debug("gemini_response_candidates_unexpected", error=str(e)[:100])

        # Last resort: stringify the response
        try:
            return str(self._response)
        except Exception:
            return "<unreadable response>"

    def __repr__(self) -> str:
        """Safe repr that never throws."""
        try:
            c = self.content
            # Defensive: ensure c is a string before slicing
            if not isinstance(c, str):
                c = str(c) if c is not None else "<none>"
            preview = (c[:50] + "...") if len(c) > 50 else c
        except Exception as e:
            # This should never happen given content's defensive implementation,
            # but belt-and-suspenders for __repr__
            logger.debug("gemini_response_repr_fallback", error=str(e)[:100])
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

    # Try new google-genai package first
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.google_api_key)

        # Use fast model for routing - Gemini 3 Flash is ideal
        model_name = settings.routing_model
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        # Create a wrapper that uses the new client API
        return GeminiRoutingWrapperNew(client, model_name)

    except ImportError:
        # Fallback to deprecated package
        import google.generativeai as genai

        genai.configure(api_key=settings.google_api_key)

        model_name = settings.routing_model
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=4096,
            ),
            tools=[genai.Tool(google_search=genai.GoogleSearch())]
        )

        return GeminiRoutingWrapper(model)
