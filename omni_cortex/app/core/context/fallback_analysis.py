"""
Enhanced Fallback Analysis System

Provides improved pattern-based task detection and component-specific
fallback methods when Gemini API is unavailable. Includes fallback
quality indicators to help users understand the limitations.

This system ensures graceful degradation while maintaining useful
functionality even when AI services are down.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from .enhanced_models import (
    ComponentStatus,
    ComponentStatusInfo,
    EnhancedDocumentationContext,
    EnhancedFileContext,
)

logger = structlog.get_logger("fallback_analysis")


@dataclass
class FallbackQualityIndicator:
    """Quality indicator for fallback results."""

    method: str  # "pattern_matching", "extension_based", "simple_listing", etc.
    confidence: float  # 0.0 to 1.0
    limitations: list[str]  # What this fallback cannot do
    recommendations: list[str]  # How to improve results


class EnhancedFallbackAnalyzer:
    """
    Enhanced fallback analyzer with improved pattern matching.

    Provides better task type detection and framework recommendations
    when Gemini API is unavailable.
    """

    # Enhanced pattern definitions with priority and confidence scores
    TASK_PATTERNS = [
        # Debug patterns (highest priority for error-related queries)
        {
            "pattern": r"\b(debug|error|fix|bug|crash|exception|traceback|stack\s*trace|"
            r"failure|failed|broken|not\s*working|issue|problem)\b",
            "task_type": "debug",
            "framework": "self_debugging",
            "confidence": 0.9,
            "keywords": ["debug", "error", "fix", "bug", "crash"],
        },
        # Security patterns
        {
            "pattern": r"\b(security|vulnerability|exploit|injection|xss|csrf|"
            r"authentication|authorization|sanitize|validate)\b",
            "task_type": "security_review",
            "framework": "red_team",
            "confidence": 0.85,
            "keywords": ["security", "vulnerability", "exploit"],
        },
        # Implementation patterns
        {
            "pattern": r"\b(implement|add|create|build|new|feature|develop|write|"
            r"generate|construct)\b",
            "task_type": "implement",
            "framework": "reason_flux",
            "confidence": 0.8,
            "keywords": ["implement", "add", "create", "build"],
        },
        # Refactoring patterns
        {
            "pattern": r"\b(refactor|clean|improve|optimize|restructure|simplify|"
            r"reorganize|modernize)\b",
            "task_type": "refactor",
            "framework": "chain_of_verification",
            "confidence": 0.8,
            "keywords": ["refactor", "clean", "improve", "optimize"],
        },
        # Testing patterns
        {
            "pattern": r"\b(test|testing|unit\s*test|integration\s*test|e2e|"
            r"coverage|assertion|mock)\b",
            "task_type": "testing",
            "framework": "chain_of_verification",
            "confidence": 0.85,
            "keywords": ["test", "testing", "coverage"],
        },
        # Architecture patterns
        {
            "pattern": r"\b(architect|design|structure|pattern|system|"
            r"scalability|performance|architecture)\b",
            "task_type": "architect",
            "framework": "step_back",
            "confidence": 0.75,
            "keywords": ["architect", "design", "structure"],
        },
        # Documentation patterns
        {
            "pattern": r"\b(document|documentation|comment|docstring|readme|"
            r"explain|describe|clarify)\b",
            "task_type": "document",
            "framework": "chain_of_note",
            "confidence": 0.8,
            "keywords": ["document", "explain", "describe"],
        },
        # Code review patterns
        {
            "pattern": r"\b(review|analyze|assess|evaluate|check|inspect|audit)\b",
            "task_type": "review",
            "framework": "chain_of_verification",
            "confidence": 0.75,
            "keywords": ["review", "analyze", "assess"],
        },
    ]

    def analyze(self, query: str, code_context: str | None = None) -> dict[str, Any]:
        """
        Analyze query using enhanced pattern matching.

        Args:
            query: User's query
            code_context: Optional code context

        Returns:
            Analysis dictionary with task_type, framework, confidence, etc.
        """
        query_lower = query.lower()

        # Find all matching patterns
        matches = []
        for pattern_def in self.TASK_PATTERNS:
            if re.search(pattern_def["pattern"], query_lower):
                matches.append(pattern_def)

        # Use highest confidence match, or default
        if matches:
            # Sort by confidence (highest first)
            matches.sort(key=lambda x: x["confidence"], reverse=True)
            best_match = matches[0]

            task_type = best_match["task_type"]
            framework = best_match["framework"]
            confidence = best_match["confidence"]

            # Adjust complexity based on query length and code context
            complexity = self._estimate_complexity(query, code_context)

            # Generate execution steps based on task type
            steps = self._generate_execution_steps(task_type, query)

            logger.info(
                "fallback_analysis_match",
                task_type=task_type,
                framework=framework,
                confidence=confidence,
                pattern_count=len(matches),
            )

            return {
                "task_type": task_type,
                "summary": query,
                "complexity": complexity,
                "framework": framework,
                "framework_reason": f"Pattern-based routing: {task_type} task detected (confidence: {confidence:.0%})",
                "confidence": confidence,
                "steps": steps,
                "success_criteria": self._generate_success_criteria(task_type),
                "blockers": self._generate_potential_blockers(task_type),
                "fallback_quality": FallbackQualityIndicator(
                    method="enhanced_pattern_matching",
                    confidence=confidence,
                    limitations=[
                        "Cannot analyze code semantics",
                        "Limited context understanding",
                        "No file relevance scoring",
                    ],
                    recommendations=[
                        "Provide more specific keywords in query",
                        "Include code context for better analysis",
                        "Retry when Gemini API is available",
                    ],
                ),
            }

        # No pattern matched - use default
        logger.warning("fallback_analysis_no_match", query=query[:100])

        return {
            "task_type": "general",
            "summary": query,
            "complexity": "medium",
            "framework": "reason_flux",
            "framework_reason": "Fallback: no specific pattern matched, using general-purpose reasoning",
            "confidence": 0.5,
            "steps": ["Analyze the query", "Identify relevant code", "Propose solution"],
            "success_criteria": ["Query is addressed", "Solution is provided"],
            "blockers": ["Limited context without Gemini analysis"],
            "fallback_quality": FallbackQualityIndicator(
                method="default_fallback",
                confidence=0.5,
                limitations=[
                    "No task-specific analysis",
                    "Generic framework selection",
                    "Limited execution planning",
                ],
                recommendations=[
                    "Use more specific keywords",
                    "Retry when Gemini API is available",
                ],
            ),
        }

    def _estimate_complexity(self, query: str, code_context: str | None) -> str:
        """Estimate task complexity based on query and context."""
        # Simple heuristics for complexity estimation
        query_length = len(query.split())

        # Check for complexity indicators
        high_complexity_indicators = [
            "multiple",
            "complex",
            "entire",
            "system",
            "architecture",
            "refactor",
            "migrate",
            "redesign",
            "scale",
        ]

        low_complexity_indicators = [
            "simple",
            "quick",
            "small",
            "minor",
            "typo",
            "fix",
        ]

        query_lower = query.lower()

        # High complexity
        if any(ind in query_lower for ind in high_complexity_indicators):
            return "high"

        # Low complexity
        if any(ind in query_lower for ind in low_complexity_indicators):
            return "low"

        # Medium complexity based on length
        if query_length > 20 or (code_context and len(code_context) > 500):
            return "high"
        elif query_length < 5:
            return "low"

        return "medium"

    def _generate_execution_steps(self, task_type: str, query: str) -> list[str]:
        """Generate execution steps based on task type."""
        steps_map = {
            "debug": [
                "Reproduce the error",
                "Analyze error messages and stack traces",
                "Identify root cause",
                "Implement fix",
                "Verify fix resolves the issue",
            ],
            "implement": [
                "Understand requirements",
                "Design solution approach",
                "Implement core functionality",
                "Add error handling",
                "Test implementation",
            ],
            "refactor": [
                "Analyze current code structure",
                "Identify improvement opportunities",
                "Plan refactoring approach",
                "Refactor incrementally",
                "Verify functionality preserved",
            ],
            "testing": [
                "Identify test scenarios",
                "Write test cases",
                "Implement tests",
                "Run tests and verify coverage",
                "Fix any failing tests",
            ],
            "architect": [
                "Understand system requirements",
                "Design high-level architecture",
                "Identify components and interfaces",
                "Document design decisions",
                "Validate design against requirements",
            ],
            "document": [
                "Understand code functionality",
                "Write clear documentation",
                "Add code comments",
                "Create usage examples",
                "Review for clarity",
            ],
            "review": [
                "Read and understand code",
                "Check for bugs and issues",
                "Evaluate code quality",
                "Provide constructive feedback",
                "Suggest improvements",
            ],
            "security_review": [
                "Identify security-sensitive areas",
                "Check for common vulnerabilities",
                "Review authentication/authorization",
                "Test input validation",
                "Document security findings",
            ],
        }

        return steps_map.get(
            task_type,
            [
                "Analyze the query",
                "Identify relevant code",
                "Propose solution",
                "Implement changes",
                "Verify results",
            ],
        )

    def _generate_success_criteria(self, task_type: str) -> list[str]:
        """Generate success criteria based on task type."""
        criteria_map = {
            "debug": [
                "Error is resolved",
                "Tests pass",
                "No regressions introduced",
            ],
            "implement": [
                "Feature works as specified",
                "Code is tested",
                "Documentation is updated",
            ],
            "refactor": [
                "Code is cleaner and more maintainable",
                "All tests still pass",
                "Functionality is preserved",
            ],
            "testing": [
                "Tests cover key scenarios",
                "All tests pass",
                "Coverage meets requirements",
            ],
            "architect": [
                "Design addresses requirements",
                "Architecture is scalable",
                "Design is documented",
            ],
            "document": [
                "Documentation is clear and complete",
                "Examples are provided",
                "Code is well-commented",
            ],
            "review": [
                "Code is thoroughly reviewed",
                "Issues are identified",
                "Feedback is actionable",
            ],
            "security_review": [
                "Security vulnerabilities are identified",
                "Risks are documented",
                "Mitigations are proposed",
            ],
        }

        return criteria_map.get(
            task_type,
            [
                "Query is addressed",
                "Solution is provided",
                "Changes are tested",
            ],
        )

    def _generate_potential_blockers(self, task_type: str) -> list[str]:
        """Generate potential blockers based on task type."""
        blockers_map = {
            "debug": [
                "Error is intermittent or hard to reproduce",
                "Root cause is in external dependency",
                "Insufficient logging or error information",
            ],
            "implement": [
                "Requirements are unclear",
                "Dependencies are missing",
                "Breaking changes required",
            ],
            "refactor": [
                "Extensive test coverage needed",
                "Breaking changes to public API",
                "Large codebase complexity",
            ],
            "testing": [
                "Code is hard to test",
                "External dependencies need mocking",
                "Test infrastructure is missing",
            ],
            "architect": [
                "Requirements are incomplete",
                "Constraints are unclear",
                "Stakeholder alignment needed",
            ],
            "document": [
                "Code is complex or unclear",
                "Domain knowledge required",
                "Documentation standards undefined",
            ],
            "review": [
                "Large amount of code to review",
                "Context is missing",
                "Code quality is very poor",
            ],
            "security_review": [
                "Security expertise required",
                "Threat model is unclear",
                "Compliance requirements unknown",
            ],
        }

        return blockers_map.get(
            task_type,
            [
                "Limited context without Gemini analysis",
                "Complex dependencies",
                "Unclear requirements",
            ],
        )


class ComponentFallbackMethods:
    """
    Component-specific fallback methods for when Gemini API is unavailable.

    Each component has a fallback that provides basic functionality
    without AI assistance.
    """

    @staticmethod
    def fallback_file_discovery(
        workspace_path: str,
        file_list: list[str] | None = None,
        max_files: int = 15,
    ) -> tuple[list[EnhancedFileContext], ComponentStatusInfo]:
        """
        Fallback file discovery using extension-based relevance scoring.

        Args:
            workspace_path: Path to workspace
            file_list: Optional pre-specified files
            max_files: Maximum files to return

        Returns:
            Tuple of (file contexts, status info)
        """
        logger.info("fallback_file_discovery", workspace_path=workspace_path)

        # Extension-based relevance scores
        extension_scores = {
            # High relevance - source code
            ".py": 0.9,
            ".js": 0.9,
            ".ts": 0.9,
            ".jsx": 0.85,
            ".tsx": 0.85,
            ".java": 0.85,
            ".cpp": 0.85,
            ".c": 0.85,
            ".go": 0.85,
            ".rs": 0.85,
            ".rb": 0.8,
            ".php": 0.8,
            ".swift": 0.8,
            ".kt": 0.8,
            # Medium relevance - config and data
            ".json": 0.7,
            ".yaml": 0.7,
            ".yml": 0.7,
            ".toml": 0.7,
            ".xml": 0.6,
            ".ini": 0.6,
            ".conf": 0.6,
            ".env": 0.6,
            # Lower relevance - documentation
            ".md": 0.5,
            ".txt": 0.4,
            ".rst": 0.5,
            # Build and dependency files
            "package.json": 0.8,
            "requirements.txt": 0.8,
            "Cargo.toml": 0.8,
            "pom.xml": 0.8,
            "build.gradle": 0.8,
            "Makefile": 0.7,
        }

        files = []

        if file_list:
            # Use provided file list
            for file_path in file_list[:max_files]:
                ext = Path(file_path).suffix.lower()
                name = Path(file_path).name

                # Get relevance score
                relevance = extension_scores.get(name, extension_scores.get(ext, 0.3))

                files.append(
                    EnhancedFileContext(
                        path=file_path,
                        relevance_score=relevance,
                        summary=f"File: {name} (fallback: no AI analysis available)",
                        key_elements=[],
                    )
                )
        else:
            # Discover files in workspace
            try:
                workspace = Path(workspace_path)

                # Common directories to skip
                skip_dirs = {
                    ".git",
                    ".svn",
                    "node_modules",
                    "__pycache__",
                    ".pytest_cache",
                    "venv",
                    ".venv",
                    "env",
                    ".env",
                    "dist",
                    "build",
                    "target",
                    ".idea",
                    ".vscode",
                    ".DS_Store",
                }

                # Walk directory tree
                for root, dirs, filenames in os.walk(workspace):
                    # Filter out skip directories
                    dirs[:] = [d for d in dirs if d not in skip_dirs]

                    for filename in filenames:
                        file_path = Path(root) / filename
                        rel_path = file_path.relative_to(workspace)

                        ext = file_path.suffix.lower()

                        # Get relevance score
                        relevance = extension_scores.get(filename, extension_scores.get(ext, 0.3))

                        # Skip very low relevance files
                        if relevance < 0.3:
                            continue

                        files.append(
                            EnhancedFileContext(
                                path=str(rel_path),
                                relevance_score=relevance,
                                summary=f"File: {filename} (fallback: extension-based scoring)",
                                key_elements=[],
                                size_kb=file_path.stat().st_size / 1024,
                            )
                        )

                        # Stop if we have enough files
                        if len(files) >= max_files * 2:  # Get more, then sort
                            break

                    if len(files) >= max_files * 2:
                        break

                # Sort by relevance and take top N
                files.sort(key=lambda f: f.relevance_score, reverse=True)
                files = files[:max_files]

            except Exception as e:
                logger.error("fallback_file_discovery_error", error=str(e))
                files = []

        status = ComponentStatusInfo(
            status=ComponentStatus.FALLBACK,
            execution_time=0.0,
            fallback_method="extension_based_scoring",
            warnings=[
                "Using extension-based file scoring (no AI analysis)",
                "File summaries and key elements unavailable",
                "Relevance scores are approximate",
            ],
        )

        logger.info("fallback_file_discovery_complete", file_count=len(files))

        return files, status

    @staticmethod
    def fallback_documentation_search(
        query: str,
    ) -> tuple[list[EnhancedDocumentationContext], ComponentStatusInfo]:
        """
        Fallback documentation search (returns empty results with status).

        Args:
            query: Search query

        Returns:
            Tuple of (empty list, status info)
        """
        logger.warning("fallback_documentation_search", query=query[:100])

        status = ComponentStatusInfo(
            status=ComponentStatus.FALLBACK,
            execution_time=0.0,
            fallback_method="no_documentation_available",
            warnings=[
                "Documentation search unavailable without Gemini API",
                "Web search and grounding require AI service",
                "Consider using local documentation or manual search",
            ],
        )

        return [], status

    @staticmethod
    def fallback_code_search(
        query: str,
        workspace_path: str,
    ) -> tuple[list[Any], ComponentStatusInfo]:
        """
        Fallback code search using simple grep (if available).

        Args:
            query: Search query
            workspace_path: Path to workspace

        Returns:
            Tuple of (search results, status info)
        """
        logger.info("fallback_code_search", query=query[:100])

        # Simple keyword extraction (no AI)
        keywords = [word for word in query.split() if len(word) > 3]

        status = ComponentStatusInfo(
            status=ComponentStatus.FALLBACK,
            execution_time=0.0,
            fallback_method="simple_keyword_extraction",
            warnings=[
                "Using simple keyword extraction (no AI analysis)",
                "Search terms may not be optimal",
                "Consider manual code search",
            ],
        )

        return [], status


# =============================================================================
# Global singleton
# =============================================================================

_fallback_analyzer: EnhancedFallbackAnalyzer | None = None


def get_fallback_analyzer() -> EnhancedFallbackAnalyzer:
    """Get the global fallback analyzer singleton."""
    global _fallback_analyzer
    if _fallback_analyzer is None:
        _fallback_analyzer = EnhancedFallbackAnalyzer()
    return _fallback_analyzer
