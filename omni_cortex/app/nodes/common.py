"""
Common Utilities for Reasoning Frameworks

Shared components used across all framework nodes:
- @quiet_star decorator: Forces internal thought prefix
- process_reward_model: PRM scoring for search algorithms
- optimize_prompt: DSPy-style prompt optimization
- LLM client wrappers
"""

import asyncio
import functools
import re
from typing import Callable, Optional, Any
from ..core.config import model_config, settings
from ..state import GraphState
from ..nodes.langchain_tools import (
    call_langchain_tool,
    get_available_tools_for_framework,
    format_tool_descriptions,
)


# Common default values for LLM calls
DEFAULT_DEEP_REASONING_TOKENS = 4096
DEFAULT_FAST_SYNTHESIS_TOKENS = 2048
DEFAULT_DEEP_REASONING_TEMP = 0.7
DEFAULT_FAST_SYNTHESIS_TEMP = 0.5
DEFAULT_PRM_TOKENS = 10
DEFAULT_PRM_TEMP = 0.1
DEFAULT_OPTIMIZATION_TOKENS = 2000
DEFAULT_OPTIMIZATION_TEMP = 0.3


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
# Process Reward Model (PRM)
# =============================================================================

async def process_reward_model(
    step: str,
    context: str,
    goal: str,
    previous_steps: Optional[list[str]] = None
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
    if not settings.enable_prm_scoring:
        return 0.5  # Default neutral score when PRM is disabled
    
    previous_context = ""
    if previous_steps:
        previous_context = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(previous_steps))
    
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
            temperature=DEFAULT_PRM_TEMP
        )
        
        # Parse the score
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        # ValueError from float parsing or other LLM/network errors
        return 0.5  # Default on error


async def batch_score_steps(
    steps: list[str],
    context: str,
    goal: str
) -> list[float]:
    """Score multiple steps in parallel for efficiency."""
    tasks = [
        process_reward_model(step, context, goal)
        for step in steps
    ]
    return await asyncio.gather(*tasks)


# =============================================================================
# DSPy-Style Prompt Optimization
# =============================================================================

async def optimize_prompt(
    task_description: str,
    base_prompt: str,
    examples: Optional[list[dict]] = None
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
    if not settings.enable_dspy_optimization:
        return base_prompt
    
    examples_text = ""
    if examples:
        examples_text = "\n\nEXAMPLES:\n"
        for i, ex in enumerate(examples):
            examples_text += f"Example {i+1}:\n"
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
            temperature=DEFAULT_OPTIMIZATION_TEMP
        )
        return optimized.strip()
    except Exception as e:
        # LLM or network error during optimization; fall back to original
        return base_prompt


# =============================================================================
# LLM Client Wrappers
# =============================================================================

async def call_deep_reasoner(
    prompt: str,
    state: GraphState,
    system: Optional[str] = None,
    max_tokens: int = DEFAULT_DEEP_REASONING_TOKENS,
    temperature: float = DEFAULT_DEEP_REASONING_TEMP
) -> tuple[str, int]:
    """
    Wrapper for deep reasoning model (Claude 4.5 Sonnet).
    
    Handles Quiet-STaR integration and token tracking.
    """
    callback = state.get("working_memory", {}).get("langchain_callback")
    if callback:
        callback.on_llm_start({"name": "call_deep_reasoner"}, [prompt])
    # Check if Quiet-STaR is enabled
    if state and state.get("working_memory", {}).get("quiet_star_enabled"):
        quiet_instruction = state.get("working_memory", {}).get("quiet_instruction", "")
        if system:
            system = quiet_instruction + "\n\n" + system
        else:
            system = quiet_instruction
    
    # Provider-specific handling using LangChain
    # Priority: 1) explicit provider setting, 2) auto-detect from available keys
    provider = settings.llm_provider.lower()

    # Handle explicit pass-through mode
    if provider == "pass-through":
        raise ValueError(
            "Pass-through mode is configured (LLM_PROVIDER=pass-through). "
            "Internal LLM calls are disabled. Set LLM_PROVIDER to 'anthropic', 'openai', or 'openrouter' "
            "and provide the corresponding API key to enable internal reasoning."
        )

    # Explicit provider selection (preferred)
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        model_name = settings.deep_reasoning_model.split("/")[-1] if "/" in settings.deep_reasoning_model else settings.deep_reasoning_model
        from langchain_anthropic import ChatAnthropic
        client = ChatAnthropic(
            model=model_name,
            api_key=settings.anthropic_api_key,
            temperature=temperature
        )
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        from langchain_openai import ChatOpenAI
        model_name = settings.deep_reasoning_model.split("/")[-1] if "/" in settings.deep_reasoning_model else settings.deep_reasoning_model
        client = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=temperature
        )
    elif provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set")
        from langchain_openai import ChatOpenAI
        model_name = settings.deep_reasoning_model
        client = ChatOpenAI(
            model=model_name,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=temperature
        )
    # Auto-detect from available keys (fallback)
    elif settings.anthropic_api_key:
        model_name = settings.deep_reasoning_model.split("/")[-1] if "/" in settings.deep_reasoning_model else settings.deep_reasoning_model
        from langchain_anthropic import ChatAnthropic
        client = ChatAnthropic(
            model=model_name,
            api_key=settings.anthropic_api_key,
            temperature=temperature
        )
    elif settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        model_name = settings.deep_reasoning_model.split("/")[-1] if "/" in settings.deep_reasoning_model else settings.deep_reasoning_model
        client = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=temperature
        )
    elif settings.openrouter_api_key:
        from langchain_openai import ChatOpenAI
        model_name = settings.deep_reasoning_model
        client = ChatOpenAI(
            model=model_name,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=temperature
        )
    else:
        raise ValueError(
            f"No LLM provider configured. Either set LLM_PROVIDER to 'anthropic', 'openai', or 'openrouter' "
            f"with the corresponding API key, or provide any API key for auto-detection."
        )

    # Make the LLM call
    lc_response = await asyncio.to_thread(
        client.invoke,
        prompt if not system else f"{system}\n\n{prompt}"
    )
    text = lc_response.content if hasattr(lc_response, "content") else str(lc_response)
    tokens = len(text) // 4  # rough estimate
    
    # Extract and store quiet thought if present
    if state:
        quiet_thought, actual_response = extract_quiet_thought(text)
        if quiet_thought:
            state["quiet_thoughts"].append(quiet_thought)
            text = actual_response
        state["tokens_used"] = state.get("tokens_used", 0) + tokens
    
    if callback:
        callback.on_llm_end({"llm_output": {"token_usage": {"total_tokens": tokens}}})
    
    return text, tokens


async def call_fast_synthesizer(
    prompt: str,
    state: Optional[GraphState] = None,
    system: Optional[str] = None,
    max_tokens: int = DEFAULT_FAST_SYNTHESIS_TOKENS,
    temperature: float = DEFAULT_FAST_SYNTHESIS_TEMP
) -> tuple[str, int]:
    """
    Wrapper for fast synthesis model.

    Used for quick operations, thought generation, and synthesis.
    """
    callback = None
    if state:
        callback = state.get("working_memory", {}).get("langchain_callback")
        if callback:
            callback.on_llm_start({"name": "call_fast_synthesizer"}, [prompt])

    # Provider-specific handling using LangChain
    # Priority: 1) explicit provider setting, 2) auto-detect from available keys
    provider = settings.llm_provider.lower()

    # Handle explicit pass-through mode
    if provider == "pass-through":
        raise ValueError(
            "Pass-through mode is configured (LLM_PROVIDER=pass-through). "
            "Internal LLM calls are disabled. Set LLM_PROVIDER to 'anthropic', 'openai', or 'openrouter' "
            "and provide the corresponding API key to enable internal reasoning."
        )

    # Explicit provider selection (preferred)
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        model_name = settings.fast_synthesis_model.split("/")[-1] if "/" in settings.fast_synthesis_model else settings.fast_synthesis_model
        from langchain_anthropic import ChatAnthropic
        client = ChatAnthropic(
            model=model_name,
            api_key=settings.anthropic_api_key,
            temperature=temperature
        )
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        from langchain_openai import ChatOpenAI
        model_name = settings.fast_synthesis_model.split("/")[-1] if "/" in settings.fast_synthesis_model else settings.fast_synthesis_model
        client = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=temperature
        )
    elif provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set")
        from langchain_openai import ChatOpenAI
        model_name = settings.fast_synthesis_model
        client = ChatOpenAI(
            model=model_name,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=temperature
        )
    # Auto-detect from available keys (fallback)
    elif settings.anthropic_api_key:
        model_name = settings.fast_synthesis_model.split("/")[-1] if "/" in settings.fast_synthesis_model else settings.fast_synthesis_model
        from langchain_anthropic import ChatAnthropic
        client = ChatAnthropic(
            model=model_name,
            api_key=settings.anthropic_api_key,
            temperature=temperature
        )
    elif settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        model_name = settings.fast_synthesis_model.split("/")[-1] if "/" in settings.fast_synthesis_model else settings.fast_synthesis_model
        client = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=temperature
        )
    elif settings.openrouter_api_key:
        from langchain_openai import ChatOpenAI
        model_name = settings.fast_synthesis_model
        client = ChatOpenAI(
            model=model_name,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=temperature
        )
    else:
        raise ValueError(
            f"No LLM provider configured. Either set LLM_PROVIDER to 'anthropic', 'openai', or 'openrouter' "
            f"with the corresponding API key, or provide any API key for auto-detection."
        )

    # Make the LLM call
    lc_response = await asyncio.to_thread(
        client.invoke,
        prompt if not system else f"{system}\n\n{prompt}"
    )
    text = lc_response.content if hasattr(lc_response, "content") else str(lc_response)
    tokens = len(text) // 4  # rough estimate

    if state:
        state["tokens_used"] = state.get("tokens_used", 0) + tokens
    if callback:
        callback.on_llm_end({"llm_output": {"token_usage": {"total_tokens": tokens}}})

    return text, tokens


# =============================================================================
# Utility Functions
# =============================================================================

def add_reasoning_step(
    state: GraphState,
    framework: str,
    thought: str,
    action: Optional[str] = None,
    observation: Optional[str] = None,
    score: Optional[float] = None
) -> None:
    """Add a reasoning step to the state trace."""
    step_num = len(state["reasoning_steps"]) + 1
    state["reasoning_steps"].append({
        "step_number": step_num,
        "framework_node": framework,
        "thought": thought,
        "action": action,
        "observation": observation,
        "score": score
    })


def extract_code_blocks(text: str) -> list[str]:
    """Extract code blocks from markdown-formatted text."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


# Reusable regex pattern for code block extraction
CODE_BLOCK_PATTERN = r"```(?:\w+)?\n(.*?)```"


def format_code_context(
    code_snippet: Optional[str],
    file_list: Optional[list[str]],
    ide_context: Optional[str]
) -> str:
    """Format code context for LLM prompts."""
    parts = []
    
    if code_snippet:
        parts.append(f"CODE:\n```\n{code_snippet}\n```")
    
    if file_list:
        parts.append(f"FILES: {', '.join(file_list)}")
    
    if ide_context:
        parts.append(f"IDE CONTEXT: {ide_context}")
    
    return "\n\n".join(parts) if parts else "No code context provided."


def generate_code_diff(original_code: str, updated_code: str) -> str:
    """Generate a unified diff between original and updated code."""
    import difflib
    
    diff = difflib.unified_diff(
        original_code.splitlines(),
        updated_code.splitlines(),
        lineterm=""
    )
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
