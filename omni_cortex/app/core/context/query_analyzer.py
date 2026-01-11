"""
Query Analyzer: Gemini-Powered Query Understanding

Analyzes user queries to understand:
- Task type (debug, implement, refactor, etc.)
- Complexity estimation
- Recommended framework
- Execution plan
"""

import asyncio
import json
import re
from typing import Any

import structlog

from ..constants import CONTENT, LIMITS
from ..correlation import get_correlation_id
from ..errors import LLMError, ProviderNotConfiguredError
from ..settings import get_settings
from .thinking_mode_optimizer import (
    get_thinking_mode_optimizer,
)

# Try to import Google AI (new package with thinking mode)
try:
    from google import genai
    from google.genai import types

    GOOGLE_AI_AVAILABLE = True
except ImportError:
    # Fallback to deprecated package
    try:
        import google.generativeai as genai

        types = None
        GOOGLE_AI_AVAILABLE = True
    except ImportError:
        GOOGLE_AI_AVAILABLE = False
        genai = None
        types = None

logger = structlog.get_logger("context.query_analyzer")


def _sanitize_prompt_input(text: str, max_length: int = 50000) -> str:
    """
    Sanitize user input before interpolating into LLM prompts.

    Prevents prompt injection by:
    1. Truncating to max_length to prevent context flooding
    2. Unicode normalization to prevent lookalike bypasses
    3. Case-insensitive pattern detection
    4. Escaping control sequences that could hijack prompt structure
    5. Removing null bytes and other dangerous characters

    Args:
        text: Raw user input
        max_length: Maximum allowed length

    Returns:
        Sanitized string safe for prompt interpolation
    """
    import unicodedata

    if not text:
        return ""

    # Truncate first to avoid processing huge inputs
    text = text[:max_length]

    # Unicode normalization to prevent lookalike character bypasses
    # NFKC normalization converts compatibility characters to canonical form
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width characters that could be used to bypass patterns
    # Zero-width space, zero-width joiner, zero-width non-joiner, etc.
    zero_width_chars = [
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u2060",  # Word joiner
        "\ufeff",  # Zero-width no-break space
    ]
    for char in zero_width_chars:
        text = text.replace(char, "")

    # Remove null bytes and other control characters (except newlines/tabs)
    text = "".join(
        c for c in text if c == "\n" or c == "\t" or (ord(c) >= 32 and ord(c) < 127) or ord(c) > 127
    )

    # Escape common prompt injection patterns (case-insensitive)
    # These sequences could be used to break out of the prompt structure
    injection_patterns = [
        (r"```", "` ` `"),  # Break code blocks
        (r"query\s*:", "[QUERY]"),  # Prevent fake section headers
        (r"code\s+context\s*:", "[CODE CONTEXT]"),
        (r"documentation\s+context\s*:", "[DOCUMENTATION CONTEXT]"),
        (r"respond\s+in\s+json", "[Respond in JSON]"),
        (r"be\s+specific", "[Be specific]"),
        (r"ignore\s+(previous|all|above)", "[IGNORE]"),  # Ignore instructions
        (r"system\s*:", "[SYSTEM]"),  # System role injection
        (r"assistant\s*:", "[ASSISTANT]"),  # Assistant role injection
        (r"<\|.*?\|>", ""),  # Special tokens used by some models
    ]

    for pattern, replacement in injection_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Remove consecutive newlines (limit to 2) to prevent structure manipulation
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


class QueryAnalyzer:
    """
    Analyzes queries using Gemini to understand intent.

    Extracts:
    - Task type (debug, implement, refactor, architect, etc.)
    - Complexity estimation
    - Entry point suggestions
    - Framework recommendations
    - Execution steps
    - Success criteria
    - Potential blockers
    """

    def __init__(self):
        self.settings = get_settings()
        self._model = None

    def _get_client(self):
        """Get or create Gemini client for analysis with thinking mode."""
        if self._model is None:
            if not GOOGLE_AI_AVAILABLE:
                raise ProviderNotConfiguredError(
                    "google-genai not installed",
                    details={"provider": "google", "package": "google-genai"},
                )

            api_key = self.settings.google_api_key
            if not api_key:
                raise ProviderNotConfiguredError(
                    "GOOGLE_API_KEY not configured",
                    details={"provider": "google", "env_var": "GOOGLE_API_KEY"},
                )

            # Use new client API with thinking support
            if types:  # New package
                self._model = genai.Client(api_key=api_key)
            else:  # Fallback to old package
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(
                    self.settings.routing_model or "gemini-2.0-flash"
                )
        return self._model

    async def analyze(
        self,
        query: str,
        code_context: str | None = None,
        documentation_context: str | None = None,
        available_budget: int | None = None,
    ) -> dict[str, Any]:
        """
        Analyze a query to understand intent and plan execution.

        Args:
            query: The user's request
            code_context: Optional code snippets for context
            documentation_context: Optional documentation/URL context found by search
            available_budget: Optional available token budget for adaptive thinking mode

        Returns:
            Dictionary with analysis results:
            - task_type: Type of task
            - summary: Clear description
            - complexity: Estimated complexity
            - entry_point: Suggested starting point
            - framework: Recommended reasoning framework
            - framework_reason: Why this framework
            - chain: Optional framework chain for complex tasks
            - steps: Execution steps
            - success_criteria: Success criteria
            - blockers: Potential blockers
            - patterns: Code patterns to look for
            - dependencies: External dependencies
            - thinking_mode_used: Whether thinking mode was used
            - thinking_level: Thinking level used (if applicable)
        """
        client = self._get_client()

        # Sanitize all user-provided inputs before prompt interpolation
        safe_query = _sanitize_prompt_input(query, max_length=10000)
        safe_code_context = (
            _sanitize_prompt_input(code_context, max_length=CONTENT.SNIPPET_MAX)
            if code_context
            else ""
        )
        safe_doc_context = (
            _sanitize_prompt_input(documentation_context, max_length=10000)
            if documentation_context
            else ""
        )

        prompt = f"""Analyze this coding task and provide structured analysis.

QUERY: {safe_query}

{f"CODE CONTEXT:{chr(10)}{safe_code_context}" if safe_code_context else ""}

{f"DOCUMENTATION CONTEXT:{chr(10)}{safe_doc_context}" if safe_doc_context else ""}

Respond in JSON format:
{{
    "task_type": "debug|implement|refactor|architect|test|review|explain|optimize",
    "summary": "Clear 1-2 sentence description of what needs to be done",
    "complexity": "low|medium|high|very_high",
    "entry_point": "suggested file or function to start with, or null",
    "framework": "best framework from: reason_flux, active_inference, self_debugging, mcts_rstar, alphacodium, plan_and_solve, multi_agent_debate, chain_of_verification, swe_agent, tree_of_thoughts",
    "framework_reason": "Why this framework is best for this task",
    "chain": ["framework1", "framework2"] or null if single framework sufficient,
    "steps": ["Step 1: ...", "Step 2: ..."],
    "success_criteria": ["Criterion 1", "Criterion 2"],
    "blockers": ["Potential issue 1"] or [],
    "patterns": ["Pattern to look for in code"],
    "dependencies": ["External deps to consider"]
}}

Be specific and actionable. Focus on what Claude needs to execute effectively."""

        try:
            # Get thinking mode optimizer
            optimizer = get_thinking_mode_optimizer()

            # First, do a quick complexity estimation (without thinking mode)
            # This helps us decide whether to use thinking mode for the full analysis
            quick_complexity = self._estimate_complexity_from_query(query)

            # Decide on thinking mode usage
            budget = available_budget or 50000  # Default budget
            thinking_decision = optimizer.decide_thinking_mode(
                query=query,
                complexity=quick_complexity,
                available_budget=budget,
                task_type=None,  # We don't know task type yet
            )

            logger.debug(
                "thinking_mode_decision",
                use_thinking=thinking_decision.use_thinking_mode,
                level=thinking_decision.thinking_level.value,
                reason=thinking_decision.reason,
                complexity=quick_complexity,
            )

            # Use new API with adaptive thinking mode if available
            if types:  # New google-genai package
                model = self.settings.routing_model or "gemini-3-flash-preview"

                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    ),
                ]

                # Check if model supports thinking mode
                supports_thinking = optimizer.should_use_thinking_for_model(model)

                # Configure thinking mode based on decision
                if supports_thinking and thinking_decision.use_thinking_mode:
                    # Enable adaptive thinking mode
                    config = types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level=thinking_decision.thinking_level.value,
                        ),
                        temperature=0.3,
                        response_mime_type="application/json",
                    )
                    logger.info(
                        "using_adaptive_thinking_mode",
                        level=thinking_decision.thinking_level.value,
                        model=model,
                    )
                else:
                    # Standard config without thinking mode
                    config = types.GenerateContentConfig(
                        temperature=0.3, response_mime_type="application/json"
                    )
                    if supports_thinking and not thinking_decision.use_thinking_mode:
                        logger.info(
                            "thinking_mode_disabled",
                            reason=thinking_decision.reason,
                            model=model,
                        )

                # Track start time for metrics
                start_time = asyncio.get_event_loop().time()

                # Use non-streaming API (simpler and works with async)
                # Wrap with timeout to prevent hanging on slow/unresponsive API
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.models.generate_content,
                        model=model,
                        contents=contents,
                        config=config,
                    ),
                    timeout=LIMITS.LLM_TIMEOUT,
                )

                # Track execution time
                execution_time = asyncio.get_event_loop().time() - start_time

                # With response_mime_type="application/json", text IS valid JSON
                result = json.loads(response.text)

                # Record thinking mode metrics
                if thinking_decision.use_thinking_mode:
                    # Estimate tokens used (rough approximation)
                    tokens_used = len(prompt.split()) + len(response.text.split())

                    # Get actual complexity from result
                    actual_complexity = result.get("complexity", quick_complexity)

                    # Estimate reasoning quality based on result completeness
                    quality_score = self._estimate_reasoning_quality(result)

                    optimizer.record_metrics(
                        thinking_level=thinking_decision.thinking_level,
                        tokens_used=tokens_used,
                        execution_time=execution_time,
                        complexity=actual_complexity,
                        budget_available=budget,
                        reasoning_quality_score=quality_score,
                        fallback_used=False,
                    )

                # Add thinking mode metadata to result
                result["thinking_mode_used"] = thinking_decision.use_thinking_mode
                result["thinking_level"] = thinking_decision.thinking_level.value

            else:  # Fallback to old package
                # Old package also supports response_mime_type in generation_config
                start_time = asyncio.get_event_loop().time()

                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.generate_content,
                        prompt,
                        generation_config={
                            "temperature": 0.3,
                            "response_mime_type": "application/json",
                        },
                    ),
                    timeout=LIMITS.LLM_TIMEOUT,
                )

                execution_time = asyncio.get_event_loop().time() - start_time
                result = json.loads(response.text)

                # Old package doesn't support thinking mode
                result["thinking_mode_used"] = False
                result["thinking_level"] = "none"

            logger.debug(
                "query_analysis_complete",
                task_type=result.get("task_type"),
                complexity=result.get("complexity"),
                thinking_mode=result.get("thinking_mode_used", False),
            )
            return result
        except (LLMError, ProviderNotConfiguredError):
            raise  # Re-raise custom LLM errors
        except Exception as e:
            # Graceful degradation: Query analysis failures should not block the
            # overall workflow. We convert errors to LLMError with helpful context
            # so callers can decide whether to proceed with reduced functionality.
            error_msg = str(e).lower()

            # Check if this is a thinking mode specific error
            is_thinking_error = (
                "thinking" in error_msg
                or "thinking_config" in error_msg
                or "thinking_level" in error_msg
            )

            # If thinking mode failed and we were using it, try fallback without thinking
            if is_thinking_error and thinking_decision.use_thinking_mode:
                logger.warning(
                    "thinking_mode_unavailable_fallback",
                    error=str(e),
                    original_level=thinking_decision.thinking_level.value,
                )

                # Record fallback metrics
                optimizer.record_metrics(
                    thinking_level=thinking_decision.thinking_level,
                    tokens_used=0,
                    execution_time=0,
                    complexity=quick_complexity,
                    budget_available=budget,
                    reasoning_quality_score=0.0,
                    fallback_used=True,
                    fallback_reason="Thinking mode unavailable",
                )

                # Retry without thinking mode
                try:
                    if types:  # New google-genai package
                        model = self.settings.routing_model or "gemini-3-flash-preview"

                        contents = [
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=prompt)],
                            ),
                        ]

                        # Standard config without thinking mode
                        config = types.GenerateContentConfig(
                            temperature=0.3, response_mime_type="application/json"
                        )

                        start_time = asyncio.get_event_loop().time()

                        response = await asyncio.wait_for(
                            asyncio.to_thread(
                                client.models.generate_content,
                                model=model,
                                contents=contents,
                                config=config,
                            ),
                            timeout=LIMITS.LLM_TIMEOUT,
                        )

                        execution_time = asyncio.get_event_loop().time() - start_time
                        result = json.loads(response.text)

                        # Add fallback metadata
                        result["thinking_mode_used"] = False
                        result["thinking_level"] = "none"
                        result["thinking_mode_fallback"] = True
                        result["thinking_mode_fallback_reason"] = "Thinking mode unavailable"

                        logger.info(
                            "thinking_mode_fallback_success",
                            execution_time=execution_time,
                        )

                        return result
                    else:
                        # Old package fallback
                        start_time = asyncio.get_event_loop().time()

                        response = await asyncio.wait_for(
                            asyncio.to_thread(
                                client.generate_content,
                                prompt,
                                generation_config={
                                    "temperature": 0.3,
                                    "response_mime_type": "application/json",
                                },
                            ),
                            timeout=LIMITS.LLM_TIMEOUT,
                        )

                        execution_time = asyncio.get_event_loop().time() - start_time
                        result = json.loads(response.text)

                        result["thinking_mode_used"] = False
                        result["thinking_level"] = "none"
                        result["thinking_mode_fallback"] = True
                        result["thinking_mode_fallback_reason"] = "Thinking mode unavailable"

                        return result

                except Exception as fallback_error:
                    logger.error(
                        "thinking_mode_fallback_failed",
                        error=str(fallback_error),
                    )
                    # Continue to general error handling below

            # Detect specific API errors and provide helpful messages
            if "insufficient" in error_msg or "quota" in error_msg or "billing" in error_msg:
                logger.warning(
                    "gemini_billing_issue",
                    error="Insufficient funds or quota exceeded",
                    error_type=type(e).__name__,
                    hint="Add credits to your Google Cloud account for Gemini API access.",
                )
                raise LLMError(
                    "Gemini API billing issue: insufficient funds or quota exceeded. "
                    "Context preparation requires a valid GOOGLE_API_KEY with credits."
                ) from e
            elif "api_key" in error_msg or "unauthorized" in error_msg or "invalid" in error_msg:
                logger.warning(
                    "gemini_auth_issue",
                    error="API key invalid or unauthorized",
                    error_type=type(e).__name__,
                    hint="Check your GOOGLE_API_KEY is valid and has Gemini API enabled.",
                )
                raise LLMError(
                    "Gemini API auth issue: API key invalid or unauthorized. "
                    "Set a valid GOOGLE_API_KEY with Gemini API enabled."
                ) from e
            else:
                logger.error(
                    "query_analysis_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    correlation_id=get_correlation_id(),
                )
                raise LLMError(f"Query analysis failed: {e}") from e

    def _estimate_complexity_from_query(self, query: str) -> str:
        """
        Quickly estimate complexity from query text without LLM call.

        This is a heuristic-based estimation used to decide on thinking mode.

        Args:
            query: The user's query

        Returns:
            Complexity level: "low", "medium", "high", or "very_high"
        """
        query_lower = query.lower()

        # Very high complexity indicators
        very_high_indicators = [
            "architecture",
            "design system",
            "refactor entire",
            "migrate",
            "performance optimization",
            "security audit",
            "distributed system",
            "microservices",
            "scale",
            "multi-repo",
            "cross-service",
        ]
        if any(indicator in query_lower for indicator in very_high_indicators):
            return "very_high"

        # High complexity indicators
        high_indicators = [
            "debug",
            "fix bug",
            "root cause",
            "investigate",
            "analyze",
            "optimize",
            "refactor",
            "redesign",
            "complex",
            "multiple",
        ]
        if any(indicator in query_lower for indicator in high_indicators):
            return "high"

        # Low complexity indicators
        low_indicators = [
            "add",
            "create simple",
            "basic",
            "quick",
            "small change",
            "update",
            "modify",
            "change",
            "simple",
        ]
        if any(indicator in query_lower for indicator in low_indicators):
            return "low"

        # Default to medium
        return "medium"

    def _estimate_reasoning_quality(self, result: dict[str, Any]) -> float:
        """
        Estimate reasoning quality from analysis result.

        Quality is based on completeness and specificity of the analysis.

        Args:
            result: The analysis result dictionary

        Returns:
            Quality score from 0.0 to 1.0
        """
        score = 0.0

        # Check for required fields (0.3 points)
        required_fields = ["task_type", "summary", "complexity", "framework"]
        present_required = sum(1 for field in required_fields if result.get(field))
        score += (present_required / len(required_fields)) * 0.3

        # Check for detailed fields (0.4 points)
        detailed_fields = ["steps", "success_criteria", "blockers", "patterns"]
        present_detailed = sum(
            1 for field in detailed_fields if result.get(field) and len(result[field]) > 0
        )
        score += (present_detailed / len(detailed_fields)) * 0.4

        # Check for specificity (0.3 points)
        # Longer, more detailed responses indicate better reasoning
        summary_length = len(result.get("summary", ""))
        if summary_length > 100:
            score += 0.15
        elif summary_length > 50:
            score += 0.10
        elif summary_length > 20:
            score += 0.05

        steps_count = len(result.get("steps", []))
        if steps_count >= 5:
            score += 0.15
        elif steps_count >= 3:
            score += 0.10
        elif steps_count >= 1:
            score += 0.05

        return min(score, 1.0)

    def check_thinking_mode_availability(self) -> dict[str, Any]:
        """
        Check if thinking mode is available for the configured model.

        Returns:
            Dictionary with availability information:
            - available: bool - Whether thinking mode is available
            - model: str - The model being checked
            - reason: str - Explanation of availability status
            - supported_levels: List[str] - Supported thinking levels
        """
        try:
            client = self._get_client()
            model = self.settings.routing_model or "gemini-3-flash-preview"

            # Check if model supports thinking mode
            optimizer = get_thinking_mode_optimizer()
            supports_thinking = optimizer.should_use_thinking_for_model(model)

            if not supports_thinking:
                return {
                    "available": False,
                    "model": model,
                    "reason": f"Model '{model}' does not support thinking mode",
                    "supported_levels": [],
                }

            # Check if adaptive thinking mode is enabled
            if not self.settings.enable_adaptive_thinking_mode:
                return {
                    "available": False,
                    "model": model,
                    "reason": "Adaptive thinking mode disabled in settings",
                    "supported_levels": ["LOW", "MEDIUM", "HIGH"],
                }

            # Try a simple test call with thinking mode
            if types:  # New google-genai package
                try:
                    test_prompt = "Test thinking mode availability"
                    contents = [
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=test_prompt)],
                        ),
                    ]

                    config = types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="LOW",
                        ),
                        temperature=0.3,
                        max_output_tokens=10,
                    )

                    # Quick test call
                    client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )

                    return {
                        "available": True,
                        "model": model,
                        "reason": "Thinking mode test successful",
                        "supported_levels": ["LOW", "MEDIUM", "HIGH"],
                    }

                except Exception as e:
                    error_msg = str(e).lower()
                    if "thinking" in error_msg:
                        return {
                            "available": False,
                            "model": model,
                            "reason": f"Thinking mode not available: {str(e)}",
                            "supported_levels": [],
                        }
                    else:
                        # Other error, might not be thinking-related
                        return {
                            "available": True,
                            "model": model,
                            "reason": "Model supports thinking mode (test inconclusive)",
                            "supported_levels": ["LOW", "MEDIUM", "HIGH"],
                        }
            else:
                # Old package doesn't support thinking mode
                return {
                    "available": False,
                    "model": model,
                    "reason": "Old google-generativeai package doesn't support thinking mode",
                    "supported_levels": [],
                }

        except Exception as e:
            logger.error(
                "thinking_mode_availability_check_failed",
                error=str(e),
            )
            return {
                "available": False,
                "model": "unknown",
                "reason": f"Availability check failed: {str(e)}",
                "supported_levels": [],
            }
