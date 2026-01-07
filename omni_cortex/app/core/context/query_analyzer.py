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
        code_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a query to understand intent and plan execution.

        Args:
            query: The user's request
            code_context: Optional code snippets for context

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

        prompt = f"""Analyze this coding task and provide structured analysis.

QUERY: {query}

{f"CODE CONTEXT:{chr(10)}{code_context[:CONTENT.SNIPPET_MAX]}" if code_context else ""}

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
                model = self.settings.routing_model or "gemini-2.0-flash-thinking-exp-01-21"
                
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    ),
                ]
                
                # Enable HIGH thinking mode for deep analysis
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="HIGH",
                    ),
                    temperature=0.3,
                    response_mime_type="application/json"
                )
                
                # Stream response to capture thinking
                full_response = ""
                thinking_blocks = []
                
                async def _stream():
                    nonlocal full_response, thinking_blocks
                    async for chunk in client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=config,
                    ):
                        if hasattr(chunk, 'thinking'):
                            thinking_blocks.append(chunk.thinking)
                        if hasattr(chunk, 'text'):
                            full_response += chunk.text
                
                await _stream()
                
                # Log thinking process
                if thinking_blocks:
                    logger.debug("gemini_thinking", thoughts=len(thinking_blocks))
                
                text = full_response
            else:  # Fallback to old package
                response = await asyncio.to_thread(
                    client.generate_content,
                    prompt,
                    generation_config={"temperature": 0.3}
                )
                text = response.text

            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                result = json.loads(json_match.group())
                logger.debug(
                    "query_analysis_complete",
                    task_type=result.get("task_type"),
                    complexity=result.get("complexity")
                )
                return result
            return {}
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
