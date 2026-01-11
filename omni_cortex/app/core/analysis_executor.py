"""
Analysis Executor: Gemini-Powered Code Analysis Execution

Unlike the brief generator which returns templates for Claude to follow,
this module actually EXECUTES the analysis using Gemini and returns
specific findings with file locations, issues, and recommendations.

This is the missing piece that makes the MCP tools actually useful -
Gemini does the analysis work, not just orchestration.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from .constants import CONTENT, LIMITS
from .errors import LLMError, ProviderNotConfiguredError
from .settings import get_settings

# Try to import Google AI
try:
    from google import genai
    from google.genai import types

    GOOGLE_AI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai

        types = None
        GOOGLE_AI_AVAILABLE = True
    except ImportError:
        GOOGLE_AI_AVAILABLE = False
        genai = None
        types = None

logger = structlog.get_logger("analysis_executor")


@dataclass
class AnalysisFinding:
    """A specific finding from code analysis."""

    severity: str  # critical, high, medium, low
    category: str  # error_handling, async, race_condition, etc.
    file_path: str
    line_number: int | None
    code_snippet: str
    issue: str
    recommendation: str
    effort: str = "medium"  # low, medium, high


@dataclass
class AnalysisResult:
    """Complete analysis result with findings and summary."""

    query: str
    framework_used: str
    total_files_analyzed: int
    findings: list[AnalysisFinding] = field(default_factory=list)
    summary: str = ""
    execution_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "framework_used": self.framework_used,
            "total_files_analyzed": self.total_files_analyzed,
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "code_snippet": f.code_snippet,
                    "issue": f.issue,
                    "recommendation": f.recommendation,
                    "effort": f.effort,
                }
                for f in self.findings
            ],
            "summary": self.summary,
            "execution_time_ms": self.execution_time_ms,
        }

    def to_markdown(self) -> str:
        """Format as markdown for display."""
        lines = []

        # Header
        lines.append("## Analysis Results")
        lines.append(
            f"**Framework**: {self.framework_used} | **Files Analyzed**: {self.total_files_analyzed}"
        )
        lines.append("")

        # Summary
        if self.summary:
            lines.append("### Summary")
            lines.append(self.summary)
            lines.append("")

        # Group findings by severity
        by_severity = {"critical": [], "high": [], "medium": [], "low": []}
        for finding in self.findings:
            severity = finding.severity.lower()
            if severity in by_severity:
                by_severity[severity].append(finding)
            else:
                by_severity["medium"].append(finding)

        # Output findings by severity
        severity_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

        for severity in ["critical", "high", "medium", "low"]:
            findings = by_severity[severity]
            if not findings:
                continue

            icon = severity_icons[severity]
            lines.append(f"### {icon} {severity.upper()} ({len(findings)})")
            lines.append("")

            for i, f in enumerate(findings, 1):
                location = f"`{f.file_path}"
                if f.line_number:
                    location += f":{f.line_number}"
                location += "`"

                lines.append(f"**{i}. {f.category}** - {location}")
                lines.append(f"- **Issue**: {f.issue}")
                if f.code_snippet:
                    lines.append("```")
                    lines.append(f"{f.code_snippet[:300]}")
                    lines.append("```")
                lines.append(f"- **Fix**: {f.recommendation}")
                lines.append(f"- **Effort**: {f.effort}")
                lines.append("")

        if not self.findings:
            lines.append("No issues found.")

        return "\n".join(lines)


class AnalysisExecutor:
    """
    Executes actual code analysis using Gemini.

    This is the key difference from the brief generator:
    - Brief generator: Returns templates/plans for Claude
    - Analysis executor: Actually analyzes code and returns findings
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = None

    def _get_client(self):
        """Get or create Gemini client."""
        if self._client is None:
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

            if types:  # New package
                self._client = genai.Client(api_key=api_key)
            else:  # Old package
                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel(
                    self.settings.routing_model or "gemini-2.0-flash"
                )

        return self._client

    async def execute_analysis(
        self,
        query: str,
        context: str | None = None,
        file_contents: dict[str, str] | None = None,
        framework: str = "chain_of_verification",
    ) -> AnalysisResult:
        """
        Execute actual code analysis using Gemini.

        Args:
            query: What to analyze (e.g., "stability improvements")
            context: Prepared context from ContextGateway
            file_contents: Dict of {file_path: file_content} to analyze
            framework: Reasoning framework to apply

        Returns:
            AnalysisResult with specific findings
        """
        import time

        start_time = time.monotonic()

        client = self._get_client()

        # Build the analysis prompt
        prompt = self._build_analysis_prompt(query, context, file_contents, framework)

        try:
            if types:  # New google-genai package
                model = self.settings.routing_model or "gemini-2.0-flash"

                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    ),
                ]

                config = types.GenerateContentConfig(
                    temperature=0.2,  # Lower temp for more precise analysis
                    response_mime_type="application/json",
                )

                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.models.generate_content,
                        model=model,
                        contents=contents,
                        config=config,
                    ),
                    timeout=LIMITS.LLM_TIMEOUT * 2,  # Allow more time for analysis
                )

                result_data = json.loads(response.text)

            else:  # Old package
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.generate_content,
                        prompt,
                        generation_config={
                            "temperature": 0.2,
                            "response_mime_type": "application/json",
                        },
                    ),
                    timeout=LIMITS.LLM_TIMEOUT * 2,
                )

                result_data = json.loads(response.text)

            # Parse findings and validate file paths against provided files
            provided_paths = set(file_contents.keys()) if file_contents else set()
            findings = []
            hallucinated_count = 0

            # Handle case where Gemini returns list directly instead of {"findings": [...]}
            if isinstance(result_data, list):
                raw_findings = result_data
            else:
                raw_findings = result_data.get("findings", [])

            for f in raw_findings:
                file_path = f.get("file_path", "unknown")

                # Validate: only accept findings for files we actually provided
                if provided_paths and file_path not in provided_paths:
                    # Normalize paths and check for exact basename match with full path verification
                    # This prevents "malicious_utils.py" from matching "utils.py"
                    import os
                    normalized_reported = os.path.normpath(file_path)
                    matched = False
                    for p in provided_paths:
                        normalized_provided = os.path.normpath(p)
                        # Allow match if normalized paths are equal OR if one is a suffix with path separator
                        if normalized_reported == normalized_provided:
                            matched = True
                            break
                        # Check if reported path ends with /provided_path (proper suffix match)
                        if normalized_reported.endswith(os.sep + normalized_provided):
                            matched = True
                            break
                        if normalized_provided.endswith(os.sep + normalized_reported):
                            matched = True
                            break

                    if not matched:
                        hallucinated_count += 1
                        logger.warning(
                            "hallucinated_file_path_filtered",
                            reported_path=file_path,
                            provided_paths=list(provided_paths)[:5],
                        )
                        continue  # Skip this finding - hallucinated path

                findings.append(
                    AnalysisFinding(
                        severity=f.get("severity", "medium"),
                        category=f.get("category", "unknown"),
                        file_path=file_path,
                        line_number=f.get("line_number"),
                        code_snippet=f.get("code_snippet", ""),
                        issue=f.get("issue", ""),
                        recommendation=f.get("recommendation", ""),
                        effort=f.get("effort", "medium"),
                    )
                )

            if hallucinated_count > 0:
                logger.info(
                    "hallucinated_findings_filtered",
                    count=hallucinated_count,
                    remaining=len(findings),
                )

            execution_time = int((time.monotonic() - start_time) * 1000)

            # Extract summary (handle list case)
            summary = ""
            if isinstance(result_data, dict):
                summary = result_data.get("summary", "")

            return AnalysisResult(
                query=query,
                framework_used=framework,
                total_files_analyzed=len(file_contents) if file_contents else 0,
                findings=findings,
                summary=summary,
                execution_time_ms=execution_time,
            )

        except asyncio.TimeoutError as e:
            raise LLMError("Analysis timed out - try with fewer files or a simpler query") from e
        except json.JSONDecodeError as e:
            logger.error("analysis_json_parse_error", error=str(e))
            raise LLMError(f"Failed to parse analysis response: {e}") from e
        except Exception as e:
            logger.error("analysis_execution_failed", error=str(e))
            raise LLMError(f"Analysis failed: {e}") from e

    def _build_analysis_prompt(
        self,
        query: str,
        context: str | None,
        file_contents: dict[str, str] | None,
        framework: str,
    ) -> str:
        """Build the analysis prompt for Gemini."""

        # Framework-specific instructions
        framework_instructions = self._get_framework_instructions(framework)

        prompt_parts = [
            f"""You are a senior software engineer performing a thorough code analysis.

## TASK
{query}

## ANALYSIS APPROACH
{framework_instructions}

## INSTRUCTIONS
1. Analyze ONLY the files provided below - do not invent or hallucinate file paths
2. Use EXACT file paths from the "FILES TO ANALYZE" section headers
3. Identify SPECIFIC issues with exact file paths and line numbers
4. For each issue, provide a concrete fix recommendation
5. Prioritize by severity (critical > high > medium > low)
6. Be specific - vague findings are not helpful
7. CRITICAL: If a file path is not in the provided files list, DO NOT report findings for it

## RESPONSE FORMAT
Respond with a JSON object:
{{
    "summary": "Brief overall assessment (2-3 sentences)",
    "findings": [
        {{
            "severity": "critical|high|medium|low",
            "category": "error_handling|async_await|race_condition|timeout|validation|resource_cleanup|graceful_degradation|security|other",
            "file_path": "path/to/file.py",
            "line_number": 123,
            "code_snippet": "the problematic code (max 200 chars)",
            "issue": "Clear description of what's wrong",
            "recommendation": "Specific fix with code example if helpful",
            "effort": "low|medium|high"
        }}
    ]
}}

Focus on ACTIONABLE findings. Skip obvious or nitpick issues.
"""
        ]

        # Add context if provided
        if context:
            prompt_parts.append(f"\n## CONTEXT\n{context[: CONTENT.SNIPPET_MAX]}")

        # Add file contents with explicit path list
        if file_contents:
            file_list = list(file_contents.keys())
            prompt_parts.append(f"\n## FILES TO ANALYZE\nValid file paths: {file_list}\n")
            prompt_parts.append("IMPORTANT: Only report findings for files in the above list.\n")
            for path, content in file_contents.items():
                # Truncate large files but keep enough for analysis
                # 15000 chars (~4k tokens) per file - enough for most files while staying under limits
                truncated = content[:15000] if len(content) > 15000 else content
                prompt_parts.append(f"\n### {path}\n```\n{truncated}\n```\n")
        else:
            prompt_parts.append("\n## NO FILES PROVIDED\nNo files were provided for analysis. Return empty findings list.\n")

        return "".join(prompt_parts)

    def _get_framework_instructions(self, framework: str) -> str:
        """Get analysis instructions based on the framework."""
        instructions = {
            "chain_of_verification": """
Apply Chain of Verification:
1. Identify potential issues in the code
2. Verify each issue is actually a problem (not intended behavior)
3. Cross-check with related code to confirm
4. Only report verified issues with evidence
""",
            "self_debugging": """
Apply Self-Debugging approach:
1. Trace execution paths looking for failure points
2. Identify error handling gaps
3. Find edge cases that aren't handled
4. Check resource cleanup paths
""",
            "active_inference": """
Apply Active Inference:
1. Model the system's expected behavior
2. Identify deviations from expected patterns
3. Find areas of uncertainty/ambiguity
4. Suggest ways to reduce uncertainty
""",
            "mcts_rstar": """
Apply systematic exploration:
1. Explore all code branches
2. Identify unexplored error paths
3. Find dead code and unreachable states
4. Check boundary conditions
""",
        }

        return instructions.get(framework, instructions["chain_of_verification"])


# Singleton instance
_executor: AnalysisExecutor | None = None


def get_analysis_executor() -> AnalysisExecutor:
    """Get the global AnalysisExecutor singleton."""
    global _executor
    if _executor is None:
        _executor = AnalysisExecutor()
    return _executor


async def execute_code_analysis(
    query: str,
    workspace_path: str | None = None,
    context: str | None = None,
    file_list: list[str] | None = None,
    framework: str = "chain_of_verification",
    max_files: int = 10,
) -> AnalysisResult:
    """
    Convenience function to execute code analysis with auto file discovery.

    Args:
        query: What to analyze
        workspace_path: Path to workspace for file discovery
        context: Pre-prepared context (skips file discovery)
        file_list: Specific files to analyze
        framework: Reasoning framework to use
        max_files: Maximum files to analyze

    Returns:
        AnalysisResult with specific findings
    """
    executor = get_analysis_executor()

    file_contents = {}
    prepared_context = context

    # If workspace provided, discover and read relevant files
    if workspace_path and not context:
        try:
            from .context_gateway import get_context_gateway

            gateway = get_context_gateway()
            structured_context = await gateway.prepare_context(
                query=query,
                workspace_path=workspace_path,
                file_list=file_list,
                search_docs=False,  # Skip docs for code analysis
                max_files=max_files,
            )

            prepared_context = structured_context.to_claude_prompt()

            # Read actual file contents for analysis
            import os

            for file_ctx in structured_context.relevant_files[:max_files]:
                file_path = file_ctx.path
                full_path = (
                    os.path.join(workspace_path, file_path)
                    if not os.path.isabs(file_path)
                    else file_path
                )

                if os.path.exists(full_path):
                    try:
                        with open(full_path, encoding="utf-8", errors="ignore") as f:
                            file_contents[file_path] = f.read()
                    except Exception as e:
                        logger.warning("file_read_error", path=file_path, error=str(e))

        except Exception as e:
            logger.warning("context_preparation_failed", error=str(e))

    # If specific files provided, read them
    if file_list and not file_contents:
        import os

        base_path = workspace_path or os.getcwd()

        for file_path in file_list[:max_files]:
            full_path = (
                os.path.join(base_path, file_path) if not os.path.isabs(file_path) else file_path
            )

            if os.path.exists(full_path):
                try:
                    with open(full_path, encoding="utf-8", errors="ignore") as f:
                        file_contents[file_path] = f.read()
                except Exception as e:
                    logger.warning("file_read_error", path=file_path, error=str(e))

    return await executor.execute_analysis(
        query=query,
        context=prepared_context,
        file_contents=file_contents,
        framework=framework,
    )
