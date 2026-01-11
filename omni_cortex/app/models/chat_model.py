"""
Chat Model Factory for Omni-Cortex

Provides multi-provider chat model initialization.
Supports: Google, Anthropic, OpenAI, OpenRouter
"""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from ..core.settings import get_settings


def get_chat_model(model_type: str = "deep", enable_thinking: bool = False) -> Any:
    """
    Get configured LangChain chat model.

    Args:
        model_type: "deep" for reasoning or "fast" for synthesis
        enable_thinking: Enable extended thinking/reasoning mode (Gemini only)

    Supports: google, anthropic, openai, openrouter
    """
    settings = get_settings()
    model_name = (
        settings.deep_reasoning_model if model_type == "deep" else settings.fast_synthesis_model
    )
    temperature = 0.7 if model_type == "deep" else 0.5

    # Remove provider prefix if present (e.g., "google/gemini-3" -> "gemini-3")
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # For thinking-like behavior, increase temperature and tokens
        max_tokens = 8192 if enable_thinking else 4096
        temp = max(0.7, temperature) if enable_thinking else temperature

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.google_api_key,
            temperature=temp,
            max_output_tokens=max_tokens,
        )

    elif settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=model_name, api_key=settings.anthropic_api_key, temperature=temperature
        )

    elif settings.llm_provider == "openrouter":
        # OpenRouter needs full model path
        full_model = (
            settings.deep_reasoning_model if model_type == "deep" else settings.fast_synthesis_model
        )
        return ChatOpenAI(
            model=full_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=temperature,
        )

    else:
        # OpenAI
        return ChatOpenAI(
            model=model_name, api_key=settings.openai_api_key, temperature=temperature
        )
