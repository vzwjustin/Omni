"""
Common Utilities for Reasoning Frameworks

Shared components used across all framework nodes:
- @quiet_star decorator: Forces internal thought prefix
- @with_rate_limit decorator: Rate limiting for framework nodes
- @with_circuit_breaker decorator: Circuit breaker for external calls
- process_reward_model: PRM scoring for search algorithms
- optimize_prompt: DSPy-style prompt optimization
- LLM client wrappers
- ContextGateway integration for Gemini preprocessing
"""

import asyncio
import difflib
import functools
import re
import warnings
from collections.abc import Callable
from typing import Any

import structlog

from ..core.constants import CONTENT
from ..core.context_gateway import get_context_gateway
from ..core.errors import (
    CircuitBreakerOpenError,
    LLMError,
    ProviderNotConfiguredError,
)
from ..core.rate_limiter import get_rate_limiter
from ..core.settings import get_settings
from ..nodes.langchain_tools import (
    call_langchain_tool,
    format_tool_descriptions,
    get_available_tools_for_framework,
)
from ..state import GraphState

logger = structlog.get_logger("common")


# Common default values for LLM calls
DEFAULT_DEEP_REASONING_TOKENS = 4096
DEFAULT_FAST_SYNTHESIS_TOKENS = 2048
DEFAULT_DEEP_REASONING_TEMP = 0.7
DEFAULT_FAST_SYNTHESIS_TEMP = 0.5
DEFAULT_PRM_TOKENS = 10
DEFAULT_PRM_TEMP = 0.1
DEFAULT_OPTIMIZATION_TOKENS = 2000
DEFAULT_OPTIMIZATION_TEMP = 0.3

# Granular token limits for specific use cases
TOKENS_SCORE_PARSING = 32  # Parse numerical scores (0.0-1.0)
TOKENS_SHORT_RESPONSE = 64  # Very short responses (satisfaction, quality)
TOKENS_QUESTION = 256  # Generate questions or critiques
TOKENS_ANALYSIS = 512  # Quick analysis or evaluation
TOKENS_DETAILED = 768  # Detailed solutions or reasoning
TOKENS_COMPREHENSIVE = 1024  # Comprehensive analysis
TOKENS_EXTENDED = 1536  # Extended reasoning
TOKENS_FULL = 2048  # Full synthesis or final answers

# Memory bounding configuration moved to state_utils.py


# =============================================================================
# Quiet-STaR Decorator
# =============================================================================


def quiet_star(func: Callable) -> Callable:
    """
    Quiet-STaR decorator: Enforces <quiet_thought> internal monologue.

    Wraps framework nodes to ensure every LLM generation includes
    an internal thinking block before the actual output.

    Usage:
        @quiet_star
        async def my_framework_node(state: GraphState) -> GraphState:
            ...
    """

    @functools.wraps(func)
    async def wrapper(state: GraphState, *args, **kwargs) -> GraphState:
        # Add quiet thought instruction to the system context
        quiet_instruction = (
            "Before responding, you MUST include a <quiet_thought> block "
            "containing your internal reasoning process. This helps ensure "
            "careful, step-by-step thinking. Format:\n"
            "<quiet_thought>\n"
            "[Your internal reasoning here]\n"
            "</quiet_thought>\n"
            "[Your actual response here]"
        )

        # Ensure working_memory exists before accessing
        if "working_memory" not in state or state["working_memory"] is None:
            state["working_memory"] = {}

        # Store the instruction for LLM calls within this context
        state["working_memory"]["quiet_star_enabled"] = True
        state["working_memory"]["quiet_instruction"] = quiet_instruction

        # Execute the wrapped function
        result = await func(state, *args, **kwargs)

        return result

    return wrapper


def extract_quiet_thought(response: str) -> tuple[str, str]:
    """
    Extract quiet thought and actual response from LLM output.

    Returns: (quiet_thought, actual_response)
    """
    pattern = r"<quiet_thought>(.*?)</quiet_thought>"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        quiet_thought = match.group(1).strip()
        actual_response = re.sub(pattern, "", response, flags=re.DOTALL).strip()
        return quiet_thought, actual_response

    return "", response


# =============================================================================
# Rate Limiting Decorator
# =============================================================================


def with_rate_limit(tool_name: str | None = None) -> Callable:
    """
    Rate limiting decorator for framework nodes and operations.

    Checks rate limits before executing the wrapped function.
    Uses tool_name for rate limit categorization (llm, search, memory, utility).

    Usage:
        @with_rate_limit("think_chain_of_thought")
        async def chain_of_thought_node(state: GraphState) -> GraphState:
            ...

    Args:
        tool_name: Name of the tool for rate limit tracking. If None, derives from function name.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(state: GraphState, *args, **kwargs) -> GraphState:
            # Determine tool name
            effective_tool_name = tool_name or func.__name__.replace("_node", "")

            # Check rate limit
            rate_limiter = await get_rate_limiter()
            allowed, error_msg = await rate_limiter.check_rate_limit(effective_tool_name)

            if not allowed:
                logger.warning(
                    "rate_limit_blocked",
                    tool=effective_tool_name,
                    function=func.__name__,
                    error=error_msg,
                )
                # Store error in state instead of raising (graceful degradation)
                state["routing_error"] = error_msg
                state["confidence_score"] = 0.0
                return state

            # Execute the wrapped function
            result = await func(state, *args, **kwargs)
            return result

        return wrapper

    return decorator


# =============================================================================
# Circuit Breaker Decorator
# =============================================================================


def with_circuit_breaker(operation_name: str, fallback_value: Any = None) -> Callable:
    """
    Circuit breaker decorator for external operations (LLM calls, embeddings, DB).

    Protects against cascade failures by failing fast when operations are unhealthy.
    Uses the circuit breaker from app/core/context/circuit_breaker.py.

    Usage:
        @with_circuit_breaker("openai_api", fallback_value="API unavailable")
        async def call_openai(prompt: str) -> str:
            ...

    Args:
        operation_name: Name of the operation for circuit breaker tracking
        fallback_value: Value to return when circuit is open (if None, raises error)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            from ..core.context.circuit_breaker import get_circuit_breaker

            breaker = get_circuit_breaker(operation_name)

            try:
                result = await breaker.call(func, *args, **kwargs)
                return result
            except CircuitBreakerOpenError as e:
                logger.error(
                    "circuit_breaker_open",
                    operation=operation_name,
                    function=func.__name__,
                    error=str(e),
                )
                if fallback_value is not None:
                    logger.info(
                        "circuit_breaker_fallback",
                        operation=operation_name,
                        fallback=str(fallback_value)[:100],
                    )
                    return fallback_value
                raise

        return wrapper

    return decorator


# =============================================================================
# Process Reward Model (PRM)
# =============================================================================


async def process_reward_model(
    step: str, context: str, goal: str, previous_steps: list[str] | None = None
) -> float:
    """
    Process Reward Model: Score a reasoning step on 0-1 scale.

    Used by MCTS, ToT, and other search algorithms to evaluate
    the quality of intermediate reasoning steps.

    Args:
        step: The current reasoning step to evaluate
        context: Task context (code, query, etc.)
        goal: The end goal we're trying to achieve
        previous_steps: Steps taken before this one

    Returns:
        Score from 0.0 (poor) to 1.0 (excellent)
    """
    if not get_settings().enable_prm_scoring:
        return 0.5  # Default neutral score when PRM is disabled

    previous_context = ""
    if previous_steps:
        previous_context = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(previous_steps))

    prompt = f"""You are a Process Reward Model evaluating reasoning steps.

GOAL: {goal}

CONTEXT:
{context}

PREVIOUS STEPS:
{previous_context if previous_context else "None"}

CURRENT STEP TO EVALUATE:
{step}

Rate this step on a scale from 0.0 to 1.0:
- 1.0: Excellent step, directly advances toward the goal
- 0.8: Good step, makes meaningful progress
- 0.6: Acceptable step, somewhat useful
- 0.4: Mediocre step, minor value
- 0.2: Poor step, potentially misleading
- 0.0: Harmful step, moves away from goal

Respond with ONLY a single decimal number between 0.0 and 1.0."""

    try:
        response, _ = await call_fast_synthesizer(
            prompt=prompt,
            state=None,  # PRM doesn't need state context
            max_tokens=DEFAULT_PRM_TOKENS,
            temperature=DEFAULT_PRM_TEMP,
        )

        # Parse the score - extract first decimal number from response
        # Handles cases like "0.8", "0.8 because...", "I rate this 0.75", etc.
        match = re.search(r"(\d+\.?\d*)", response.strip())
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        else:
            logger.warning(
                "prm_score_no_number", response=response[: CONTENT.QUERY_LOG] if response else ""
            )
            return 0.5
    except ValueError as e:
        # Failed to parse float from response
        logger.warning(
            "prm_score_parsing_failed",
            response=response[: CONTENT.QUERY_LOG] if response else "",
            error=str(e),
        )
        return 0.5
    except (LLMError, ProviderNotConfiguredError) as e:
        # LLM provider error - log and return default
        logger.error("prm_scoring_failed", error=str(e), error_type=type(e).__name__)
        return 0.5  # Default on error
    except Exception as e:
        # Broad catch intentional: LLM calls can fail with unpredictable errors
        # (network issues, provider-specific exceptions, serialization errors, etc.)
        # We log and wrap in LLMError to provide consistent error handling upstream.
        logger.error("prm_scoring_failed", error=str(e), error_type=type(e).__name__)
        raise LLMError(f"PRM scoring failed: {e}") from e


async def batch_score_steps(steps: list[str], context: str, goal: str) -> list[float]:
    """Score multiple steps in parallel for efficiency."""
    tasks = [process_reward_model(step, context, goal) for step in steps]
    return await asyncio.gather(*tasks)


# =============================================================================
# DSPy-Style Prompt Optimization
# =============================================================================


async def optimize_prompt(
    task_description: str, base_prompt: str, examples: list[dict] | None = None
) -> str:
    """
    DSPy-style prompt optimization.

    Uses the LLM to rewrite a prompt for a specific task,
    incorporating best practices and task-specific optimizations.

    Args:
        task_description: Description of what we're trying to accomplish
        base_prompt: The original prompt template
        examples: Optional few-shot examples to include

    Returns:
        Optimized prompt string
    """
    if not get_settings().enable_dspy_optimization:
        return base_prompt

    examples_text = ""
    if examples:
        examples_text = "\n\nEXAMPLES:\n"
        for i, ex in enumerate(examples):
            examples_text += f"Example {i + 1}:\n"
            examples_text += f"  Input: {ex.get('input', 'N/A')}\n"
            examples_text += f"  Output: {ex.get('output', 'N/A')}\n"

    optimization_prompt = f"""You are a prompt engineering expert. Optimize the following prompt for the given task.

TASK DESCRIPTION:
{task_description}

BASE PROMPT:
{base_prompt}
{examples_text}

OPTIMIZATION GUIDELINES:
1. Make instructions clearer and more specific
2. Add relevant formatting requirements
3. Include edge case handling
4. Ensure output format is well-defined
5. Remove ambiguity
6. Keep the core intent intact

Return ONLY the optimized prompt, no explanations."""

    try:
        optimized, _ = await call_fast_synthesizer(
            prompt=optimization_prompt,
            state=None,
            max_tokens=DEFAULT_OPTIMIZATION_TOKENS,
            temperature=DEFAULT_OPTIMIZATION_TEMP,
        )
        return optimized.strip()
    except (LLMError, ProviderNotConfiguredError) as e:
        # LLM provider error during optimization; fall back to original
        logger.warning(
            "prompt_optimization_failed",
            error=str(e),
            error_type=type(e).__name__,
            task=task_description[: CONTENT.QUERY_PREVIEW],
        )
        return base_prompt
    except Exception as e:
        # Broad catch intentional: LLM calls can fail with unpredictable errors
        # (network issues, provider-specific exceptions, serialization errors, etc.)
        # Optimization failure is non-critical - we gracefully fall back to base_prompt.
        logger.warning(
            "prompt_optimization_failed",
            error=str(e),
            error_type=type(e).__name__,
            task=task_description[: CONTENT.QUERY_PREVIEW],
        )
        return base_prompt


# =============================================================================
# LLM Client Wrappers
# =============================================================================


def _create_google_client(settings, model_name: str, temperature: float, max_tokens: int) -> Any:
    if not settings.google_api_key:
        raise ProviderNotConfiguredError(
            "LLM_PROVIDER=google but GOOGLE_API_KEY is not set",
            details={"provider": "google", "env_var": "GOOGLE_API_KEY"},
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def _create_anthropic_client(settings, model_name: str, temperature: float, max_tokens: int) -> Any:
    if not settings.anthropic_api_key:
        raise ProviderNotConfiguredError(
            "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set",
            details={"provider": "anthropic", "env_var": "ANTHROPIC_API_KEY"},
        )
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model_name,
        api_key=settings.anthropic_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _create_openai_client(settings, model_name: str, temperature: float, max_tokens: int) -> Any:
    if not settings.openai_api_key:
        raise ProviderNotConfiguredError(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is not set",
            details={"provider": "openai", "env_var": "OPENAI_API_KEY"},
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _create_openrouter_client(
    settings, model_name: str, temperature: float, max_tokens: int
) -> Any:
    if not settings.openrouter_api_key:
        raise ProviderNotConfiguredError(
            "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set",
            details={"provider": "openrouter", "env_var": "OPENROUTER_API_KEY"},
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# Provider Registry
PROVIDER_FACTORIES = {
    "google": _create_google_client,
    "anthropic": _create_anthropic_client,
    "openai": _create_openai_client,
    "openrouter": _create_openrouter_client,
}


def _get_llm_client(model_type: str, temperature: float, max_tokens: int) -> Any:
    """
    Create and return an LLM client based on provider configuration.

    Centralizes all provider selection logic using a strategy pattern.

    Args:
        model_type: Either "deep_reasoning" or "fast_synthesis" to select the model
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens for response

    Returns:
        Configured LangChain chat client (ChatAnthropic, ChatOpenAI, or ChatGoogleGenerativeAI)

    Raises:
        ProviderNotConfiguredError: If no valid provider is configured or pass-through mode is active
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    # Handle explicit pass-through mode
    if provider == "pass-through":
        raise ProviderNotConfiguredError(
            "Pass-through mode is configured (LLM_PROVIDER=pass-through). "
            "Internal LLM calls are disabled. Set LLM_PROVIDER to 'anthropic', 'openai', or 'openrouter' "
            "and provide the corresponding API key to enable internal reasoning.",
            details={"provider": "pass-through", "mode": "disabled"},
        )

    # Determine effective provider: explicit setting takes priority, then auto-detect
    effective_provider = None
    if provider in PROVIDER_FACTORIES:
        effective_provider = provider
    elif settings.google_api_key:
        effective_provider = "google"
    elif settings.anthropic_api_key:
        effective_provider = "anthropic"
    elif settings.openai_api_key:
        effective_provider = "openai"
    elif settings.openrouter_api_key:
        effective_provider = "openrouter"
    else:
        raise ProviderNotConfiguredError(
            "No LLM provider configured. Set LLM_PROVIDER to 'google', 'anthropic', 'openai', or 'openrouter' "
            "with the corresponding API key, or provide any API key for auto-detection.",
            details={"provider": "none", "hint": "Set LLM_PROVIDER and corresponding API key"},
        )

    factory = PROVIDER_FACTORIES.get(effective_provider)
    if not factory:
        raise ProviderNotConfiguredError(
            f"Unknown provider: {effective_provider}", details={"provider": effective_provider}
        )

    # Select the appropriate model name based on type
    if model_type == "deep_reasoning":
        base_model = settings.deep_reasoning_model
    else:
        base_model = settings.fast_synthesis_model

    # Helper to strip provider prefix from model name (e.g., "anthropic/claude-3" -> "claude-3")
    # OpenRouter needs the full path, others might prefer stripped
    model_name = base_model
    if effective_provider != "openrouter" and "/" in base_model:
        model_name = base_model.split("/")[-1]

    return factory(settings, model_name, temperature, max_tokens)


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using a character-based heuristic.

    This function assumes approximately 4 characters per token, which is a common
    heuristic for English text with BPE tokenizers (like GPT-4's).

    Args:
        text (str): The input text to estimate.

    Returns:
        int: The estimated number of tokens. Returns 0 for empty strings.

    Note:
        This is not a replacement for a real tokenizer (like `tiktoken`).
        Use this only for rough logging or bounding, not for precise billing
        or context window management.
    """
    if not text:
        return 0
    return len(text) // 4


@with_circuit_breaker("llm_deep_reasoning", fallback_value=("LLM service unavailable", 0))
async def _invoke_deep_reasoner(client: Any, message: str) -> Any:
    """Circuit-breaker protected LLM invocation for deep reasoning."""
    return await asyncio.to_thread(client.invoke, message)


async def call_deep_reasoner(
    prompt: str,
    state: GraphState,
    system: str | None = None,
    max_tokens: int = DEFAULT_DEEP_REASONING_TOKENS,
    temperature: float = DEFAULT_DEEP_REASONING_TEMP,
) -> tuple[str, int]:
    """
    Wrapper for deep reasoning model (Claude 4.5 Sonnet).

    Handles Quiet-STaR integration and token tracking.
    Protected by circuit breaker for fault tolerance.
    """
    callback = state.get("working_memory", {}).get("langchain_callback") if state else None
    if callback and hasattr(callback, "on_llm_start"):
        try:
            callback.on_llm_start({"name": "call_deep_reasoner"}, [prompt])
        except Exception as e:
            # Broad catch intentional: Callback failures should not crash the LLM call.
            # Third-party callbacks may raise any exception; we log and continue.
            logger.warning(
                "callback_on_llm_start_failed", error=str(e), error_type=type(e).__name__
            )

    # Check if Quiet-STaR is enabled
    if state and state.get("working_memory", {}).get("quiet_star_enabled"):
        quiet_instruction = state.get("working_memory", {}).get("quiet_instruction", "")
        system = quiet_instruction + "\n\n" + system if system else quiet_instruction

    # Get the LLM client using the centralized helper
    client = _get_llm_client(
        model_type="deep_reasoning", temperature=temperature, max_tokens=max_tokens
    )

    # Make the LLM call (circuit breaker protected)
    message = prompt if not system else f"{system}\n\n{prompt}"
    try:
        lc_response = await _invoke_deep_reasoner(client, message)
    except CircuitBreakerOpenError:
        logger.error("deep_reasoner_circuit_open", prompt_preview=prompt[:100])
        return "LLM service temporarily unavailable due to repeated failures", 0
    # Handle different response formats (Google AI returns list, others return string)
    content = lc_response.content if hasattr(lc_response, "content") else str(lc_response)
    if isinstance(content, list):
        # Google AI format: [{'type': 'text', 'text': '...'}]
        text = content[0].get("text", str(content)) if content else ""
    else:
        text = content
    tokens = _estimate_tokens(text)

    # Extract and store quiet thought if present
    if state:
        quiet_thought, actual_response = extract_quiet_thought(text)
        if quiet_thought:
            # Ensure quiet_thoughts list exists before appending
            if "quiet_thoughts" not in state or state["quiet_thoughts"] is None:
                state["quiet_thoughts"] = []
            state["quiet_thoughts"].append(quiet_thought)
            text = actual_response
        state["tokens_used"] = state.get("tokens_used", 0) + tokens

    if callback and hasattr(callback, "on_llm_end"):
        try:
            callback.on_llm_end({"llm_output": {"token_usage": {"total_tokens": tokens}}})
        except Exception as e:
            # Broad catch intentional: Callback failures should not crash the LLM call.
            # Third-party callbacks may raise any exception; we log and continue.
            logger.warning("callback_on_llm_end_failed", error=str(e), error_type=type(e).__name__)

    return text, tokens


@with_circuit_breaker("llm_fast_synthesis", fallback_value=("LLM service unavailable", 0))
async def _invoke_fast_synthesizer(client: Any, message: str) -> Any:
    """Circuit-breaker protected LLM invocation for fast synthesis."""
    return await asyncio.to_thread(client.invoke, message)


async def call_fast_synthesizer(
    prompt: str,
    state: GraphState | None = None,
    system: str | None = None,
    max_tokens: int = DEFAULT_FAST_SYNTHESIS_TOKENS,
    temperature: float = DEFAULT_FAST_SYNTHESIS_TEMP,
) -> tuple[str, int]:
    """
    Wrapper for fast synthesis model.

    Used for quick operations, thought generation, and synthesis.
    Protected by circuit breaker for fault tolerance.
    """
    callback = None
    if state:
        callback = state.get("working_memory", {}).get("langchain_callback")
        if callback and hasattr(callback, "on_llm_start"):
            try:
                callback.on_llm_start({"name": "call_fast_synthesizer"}, [prompt])
            except Exception as e:
                # Broad catch intentional: Callback failures should not crash the LLM call.
                # Third-party callbacks may raise any exception; we log and continue.
                logger.warning(
                    "callback_on_llm_start_failed", error=str(e), error_type=type(e).__name__
                )

    # Get the LLM client using the centralized helper
    client = _get_llm_client(
        model_type="fast_synthesis", temperature=temperature, max_tokens=max_tokens
    )

    # Make the LLM call (circuit breaker protected)
    message = prompt if not system else f"{system}\n\n{prompt}"
    try:
        lc_response = await _invoke_fast_synthesizer(client, message)
    except CircuitBreakerOpenError:
        logger.error("fast_synthesizer_circuit_open", prompt_preview=prompt[:100])
        return "LLM service temporarily unavailable due to repeated failures", 0
    # Handle different response formats (Google AI returns list, others return string)
    content = lc_response.content if hasattr(lc_response, "content") else str(lc_response)
    if isinstance(content, list):
        # Google AI format: [{'type': 'text', 'text': '...'}]
        text = content[0].get("text", str(content)) if content else ""
    else:
        text = content
    tokens = _estimate_tokens(text)

    if state:
        state["tokens_used"] = state.get("tokens_used", 0) + tokens
    if callback and hasattr(callback, "on_llm_end"):
        try:
            callback.on_llm_end({"llm_output": {"token_usage": {"total_tokens": tokens}}})
        except Exception as e:
            # Broad catch intentional: Callback failures should not crash the LLM call.
            # Third-party callbacks may raise any exception; we log and continue.
            logger.warning("callback_on_llm_end_failed", error=str(e), error_type=type(e).__name__)

    return text, tokens


# =============================================================================
# Utility Functions
# =============================================================================


def add_reasoning_step(
    state: GraphState,
    framework: str,
    thought: str,
    action: str | None = None,
    observation: str | None = None,
    score: float | None = None,
) -> None:
    """
    Add a reasoning step to the state trace with memory bounding.

    Maintains a rolling window of reasoning steps to prevent OOM in long loops.
    """
    # Ensure reasoning_steps exists
    if "reasoning_steps" not in state or state["reasoning_steps"] is None:
        state["reasoning_steps"] = []

    # Use a persistent step counter in state if available, else derive
    if "step_counter" not in state:
        state["step_counter"] = len(state["reasoning_steps"])

    state["step_counter"] += 1
    step_num = state["step_counter"]

    new_step = {
        "step_number": step_num,
        "framework_node": framework,
        "thought": thought,
        "action": action,
        "observation": observation,
        "score": score,
    }

    state["reasoning_steps"].append(new_step)

    # Implement rolling window memory bounding
    # Keep first 5 steps (context) + remaining recent memory
    # Use settings for configurable memory bound
    settings = get_settings()
    memory_bound = settings.reasoning_memory_bound
    if len(state["reasoning_steps"]) > memory_bound:
        initial_context = state["reasoning_steps"][:5]
        recent_memory = state["reasoning_steps"][-(memory_bound - 5) :]

        state["reasoning_steps"] = initial_context + recent_memory

        # Insert truncation marker
        state["reasoning_steps"].insert(
            5,
            {
                "step_number": -1,
                "framework_node": "system",
                "thought": f"... {step_num - memory_bound} intermediate steps truncated for memory efficiency ...",
                "action": "truncate_memory",
                "observation": None,
                "score": None,
            },
        )


def extract_code_blocks(text: str) -> list[str]:
    """Extract code blocks from markdown-formatted text."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


# Reusable regex pattern for code block extraction
CODE_BLOCK_PATTERN = r"```(?:\w+)?\n(.*?)```"


async def prepare_context_with_gemini(query: str, state: GraphState) -> str:
    """
    Use Gemini (via ContextGateway) to preprocess and structure context for Claude.

    This is the proper Gemini→Claude flow:
    1. Gemini analyzes query, discovers files, fetches docs
    2. Returns StructuredContext with rich preprocessing
    3. Claude uses this context for deep reasoning
    """
    # Use singleton gateway (thread-safe)
    gateway = get_context_gateway()

    # Prepare structured context via Gemini
    structured_context = await gateway.prepare_context(
        query=query,
        code_context=state.get("code_snippet"),
        file_list=state.get("file_list"),
        search_docs=True,
        max_files=15,
    )

    # Convert to Claude-ready prompt
    return structured_context.to_claude_prompt()


def format_code_context(
    code_snippet: str | None,
    file_list: list[str] | None,
    ide_context: str | None,
    state: GraphState | None = None,
) -> str:
    """
    DEPRECATED: Use prepare_context_with_gemini() instead for proper Gemini→Claude flow.

    This is a fallback for simple context formatting without Gemini preprocessing.

    .. deprecated:: 1.0.0
        Use :func:`prepare_context_with_gemini` instead.
    """
    warnings.warn(
        "format_code_context is deprecated, use prepare_context_with_gemini() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    parts = []

    # Include RAG context if available (auto-fetched during routing)
    if state:
        rag_formatted = state.get("working_memory", {}).get("rag_context_formatted", "")
        if rag_formatted:
            parts.append(rag_formatted)

        # Include Past Learnings (Episodic Memory)
        episodic_memory = state.get("episodic_memory", [])
        if episodic_memory:
            learnings_text = "## 🧠 Past Learnings (Similar Issues Solved Before):\n"
            for i, memory in enumerate(episodic_memory, 1):
                # Ensure memory is a dict (robustness)
                if isinstance(memory, dict):
                    learnings_text += f"\n{i}. **{memory.get('framework', 'Unknown Framework')}**: {memory.get('problem', 'No problem description')}\n"
                    learnings_text += (
                        f"   Solution: {memory.get('solution', 'No solution provided')[:300]}...\n"
                    )
            parts.append(learnings_text)

    if code_snippet:
        parts.append(f"CODE:\n```\n{code_snippet}\n```")

    if file_list:
        parts.append(f"FILES: {', '.join(file_list)}")

    if ide_context:
        parts.append(f"IDE CONTEXT: {ide_context}")

    return "\n\n".join(parts) if parts else "No code context provided."


def get_rag_context(state: GraphState) -> str:
    """Get pre-fetched RAG context from state (populated during routing)."""
    return state.get("working_memory", {}).get("rag_context_formatted", "")


def generate_code_diff(original_code: str, updated_code: str) -> str:
    """Generate a unified diff between original and updated code."""
    diff = difflib.unified_diff(original_code.splitlines(), updated_code.splitlines(), lineterm="")
    return "\n".join(diff)


# =============================================================================
# Tool Helpers
# =============================================================================


async def run_tool(tool_name: str, tool_input: Any, state: GraphState) -> Any:
    """Proxy to LangChain tool execution for framework nodes."""
    return await call_langchain_tool(tool_name, tool_input, state)


def list_tools_for_framework(framework_name: str, state: GraphState) -> list[str]:
    """List recommended tools for a framework."""
    return get_available_tools_for_framework(framework_name, state)


def tool_descriptions() -> str:
    """Formatted tool descriptions for prompts."""
    return format_tool_descriptions()
