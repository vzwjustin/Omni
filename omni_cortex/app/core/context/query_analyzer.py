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
import structlog
from typing import Dict, Any, Optional

from ..settings import get_settings
from ..constants import CONTENT
from ..errors import LLMError, ProviderNotConfiguredError
from ..correlation import get_correlation_id

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
    2. Escaping control sequences that could hijack prompt structure
    3. Removing null bytes and other dangerous characters
    
    Args:
        text: Raw user input
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string safe for prompt interpolation
    """
    if not text:
        return ""
    
    # Truncate first to avoid processing huge inputs
    text = text[:max_length]
    
    # Remove null bytes and other control characters (except newlines/tabs)
    text = "".join(c for c in text if c == '\n' or c == '\t' or (ord(c) >= 32 and ord(c) < 127) or ord(c) > 127)
    
    # Escape common prompt injection patterns
    # These sequences could be used to break out of the prompt structure
    injection_patterns = [
        ("```", "` ` `"),  # Break code blocks
        ("QUERY:", "[QUERY]"),  # Prevent fake section headers
        ("CODE CONTEXT:", "[CODE CONTEXT]"),
        ("DOCUMENTATION CONTEXT:", "[DOCUMENTATION CONTEXT]"),
        ("Respond in JSON", "[Respond in JSON]"),
        ("Be specific", "[Be specific]"),
    ]
    
    for pattern, replacement in injection_patterns:
        text = text.replace(pattern, replacement)
    
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
                    details={"provider": "google", "package": "google-genai"}
                )

            api_key = self.settings.google_api_key
            if not api_key:
                raise ProviderNotConfiguredError(
                    "GOOGLE_API_KEY not configured",
                    details={"provider": "google", "env_var": "GOOGLE_API_KEY"}
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
        code_context: Optional[str] = None,
        documentation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a query to understand intent and plan execution.

        Args:
            query: The user's request
            code_context: Optional code snippets for context
            documentation_context: Optional documentation/URL context found by search

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
        """
        client = self._get_client()

        # Sanitize all user-provided inputs before prompt interpolation
        safe_query = _sanitize_prompt_input(query, max_length=10000)
        safe_code_context = _sanitize_prompt_input(code_context, max_length=CONTENT.SNIPPET_MAX) if code_context else ""
        safe_doc_context = _sanitize_prompt_input(documentation_context, max_length=10000) if documentation_context else ""

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
            # Use new API with thinking mode if available
            if types:  # New google-genai package
                model = self.settings.routing_model or "gemini-3-flash-preview"
                
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    ),
                ]
                
                # Enable thinking config for models that support it:
                # - Models with "thinking" in name (legacy experimental)
                # - Gemini 3 models (all support thinking mode)
                is_thinking_model = "thinking" in model.lower() or "gemini-3" in model.lower()
                
                if is_thinking_model:
                    # Enable HIGH thinking mode for deep analysis
                    config = types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="HIGH",
                        ),
                        temperature=0.3,
                        response_mime_type="application/json"
                    )
                else:
                    # Standard config for non-thinking models
                    config = types.GenerateContentConfig(
                        temperature=0.3,
                        response_mime_type="application/json"
                    )
                
                # Use non-streaming API (simpler and works with async)
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                )

                # With response_mime_type="application/json", text IS valid JSON
                result = json.loads(response.text)
                
            else:  # Fallback to old package
                # Old package also supports response_mime_type in generation_config
                response = await asyncio.to_thread(
                    client.generate_content,
                    prompt,
                    generation_config={
                        "temperature": 0.3,
                        "response_mime_type": "application/json"
                    }
                )
                result = json.loads(response.text)

            logger.debug(
                "query_analysis_complete",
                task_type=result.get("task_type"),
                complexity=result.get("complexity")
            )
            return result
        except (LLMError, ProviderNotConfiguredError):
            raise  # Re-raise custom LLM errors
        except Exception as e:
            # Graceful degradation: Query analysis failures should not block the
            # overall workflow. We convert errors to LLMError with helpful context
            # so callers can decide whether to proceed with reduced functionality.
            error_msg = str(e).lower()

            # Detect specific API errors and provide helpful messages
            if "insufficient" in error_msg or "quota" in error_msg or "billing" in error_msg:
                logger.warning(
                    "gemini_billing_issue",
                    error="Insufficient funds or quota exceeded",
                    error_type=type(e).__name__,
                    hint="Add credits to your Google Cloud account for Gemini API access."
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
                    hint="Check your GOOGLE_API_KEY is valid and has Gemini API enabled."
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
                    correlation_id=get_correlation_id()
                )
                raise LLMError(f"Query analysis failed: {e}") from e
