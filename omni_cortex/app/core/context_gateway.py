"""
Context Gateway: Gemini-Powered Context Optimization for Claude

This module acts as a preprocessing layer that uses Gemini Flash to:
1. Analyze queries and understand intent
2. Discover relevant files in the workspace
3. Search web for documentation/APIs
4. Score and rank file relevance
5. Structure rich context for Claude

Architecture:
    User Query → Gemini Flash (cheap, fast) → Structured Context → Claude (expensive, powerful)

Gemini does the "egg hunting" so Claude can focus on deep reasoning.
"""

import asyncio
import json
import os
import re
import structlog
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from .settings import get_settings

logger = structlog.get_logger("context_gateway")

# Try to import Google AI
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None


@dataclass
class FileContext:
    """Context about a discovered file."""
    path: str
    relevance_score: float  # 0-1
    summary: str  # Gemini-generated summary
    key_elements: List[str] = field(default_factory=list)  # functions, classes, etc.
    line_count: int = 0
    size_kb: float = 0


@dataclass
class DocumentationContext:
    """Context from web documentation lookup."""
    source: str  # URL or doc name
    title: str
    snippet: str  # Relevant excerpt
    relevance_score: float


@dataclass
class StructuredContext:
    """
    Rich, structured context packet for Claude.

    Everything Claude needs, pre-organized and ready for deep reasoning.
    No egg hunting required.
    """
    # Task Understanding
    task_type: str  # debug, implement, refactor, architect, etc.
    task_summary: str  # Clear description of what needs to be done
    complexity: str  # low, medium, high, very_high

    # Relevant Files (paths + summaries, not full contents)
    relevant_files: List[FileContext] = field(default_factory=list)
    entry_point: Optional[str] = None  # Where to start

    # Documentation (pre-fetched snippets)
    documentation: List[DocumentationContext] = field(default_factory=list)

    # Framework Recommendation
    recommended_framework: str = "reason_flux"
    framework_reason: str = ""
    chain_suggestion: Optional[List[str]] = None  # For complex tasks

    # Execution Plan
    execution_steps: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    potential_blockers: List[str] = field(default_factory=list)

    # Additional Context
    related_patterns: List[str] = field(default_factory=list)  # Similar code patterns
    dependencies: List[str] = field(default_factory=list)  # External deps to consider

    def to_claude_prompt(self) -> str:
        """Format as rich context prompt for Claude."""
        sections = []

        # Task Section
        sections.append(f"""## Task Analysis
**Type**: {self.task_type} | **Complexity**: {self.complexity}
**Summary**: {self.task_summary}""")

        # Files Section
        if self.relevant_files:
            file_lines = ["## Relevant Files"]
            if self.entry_point:
                file_lines.append(f"**Start here**: `{self.entry_point}`\n")
            for f in self.relevant_files[:10]:  # Top 10
                score_bar = "█" * int(f.relevance_score * 5) + "░" * (5 - int(f.relevance_score * 5))
                file_lines.append(f"- `{f.path}` [{score_bar}] - {f.summary}")
                if f.key_elements:
                    file_lines.append(f"  Key: {', '.join(f.key_elements[:5])}")
            sections.append("\n".join(file_lines))

        # Documentation Section
        if self.documentation:
            doc_lines = ["## Pre-Fetched Documentation"]
            for doc in self.documentation[:5]:
                doc_lines.append(f"### {doc.title}")
                doc_lines.append(f"*Source: {doc.source}*")
                doc_lines.append(f"```\n{doc.snippet}\n```")
            sections.append("\n".join(doc_lines))

        # Framework Section
        sections.append(f"""## Recommended Approach
**Framework**: `{self.recommended_framework}`
**Reason**: {self.framework_reason}""")
        if self.chain_suggestion:
            sections.append(f"**Chain**: {' → '.join(self.chain_suggestion)}")

        # Execution Plan
        if self.execution_steps:
            steps = ["## Execution Plan"]
            for i, step in enumerate(self.execution_steps, 1):
                steps.append(f"{i}. {step}")
            sections.append("\n".join(steps))

        # Success Criteria
        if self.success_criteria:
            criteria = ["## Success Criteria"]
            for c in self.success_criteria:
                criteria.append(f"- [ ] {c}")
            sections.append("\n".join(criteria))

        # Potential Blockers
        if self.potential_blockers:
            blockers = ["## ⚠️ Potential Blockers"]
            for b in self.potential_blockers:
                blockers.append(f"- {b}")
            sections.append("\n".join(blockers))

        return "\n\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_type": self.task_type,
            "task_summary": self.task_summary,
            "complexity": self.complexity,
            "relevant_files": [
                {
                    "path": f.path,
                    "relevance_score": f.relevance_score,
                    "summary": f.summary,
                    "key_elements": f.key_elements,
                }
                for f in self.relevant_files
            ],
            "entry_point": self.entry_point,
            "documentation": [
                {
                    "source": d.source,
                    "title": d.title,
                    "snippet": d.snippet,
                }
                for d in self.documentation
            ],
            "recommended_framework": self.recommended_framework,
            "framework_reason": self.framework_reason,
            "chain_suggestion": self.chain_suggestion,
            "execution_steps": self.execution_steps,
            "success_criteria": self.success_criteria,
            "potential_blockers": self.potential_blockers,
        }


class ContextGateway:
    """
    Gemini-powered context optimization layer.

    Does the heavy lifting of:
    - Analyzing queries to understand intent
    - Discovering relevant files in workspace
    - Fetching documentation from the web
    - Scoring and ranking relevance
    - Structuring everything for Claude

    Usage:
        gateway = ContextGateway()
        context = await gateway.prepare_context(
            query="Fix the authentication bug in the login flow",
            workspace_path="/path/to/project"
        )
        # context.to_claude_prompt() returns rich, structured context
    """

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._search_model = None

    def _get_model(self):
        """Get or create Gemini model for analysis."""
        if self._model is None:
            if not GOOGLE_AI_AVAILABLE:
                raise RuntimeError("google-generativeai not installed")

            api_key = self.settings.google_api_key
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY not configured")

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                self.settings.routing_model or "gemini-2.0-flash"
            )
        return self._model

    def _get_search_model(self):
        """Get model with Google Search grounding for documentation lookup."""
        if self._search_model is None:
            if not GOOGLE_AI_AVAILABLE:
                raise RuntimeError("google-generativeai not installed")

            api_key = self.settings.google_api_key
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY not configured")

            genai.configure(api_key=api_key)

            # Use grounded model for web search
            from google.generativeai import GenerativeModel
            from google.generativeai.types import Tool

            # Google Search grounding tool
            google_search_tool = Tool.from_google_search_retrieval(
                google_search_retrieval={"disable_attribution": False}
            )

            self._search_model = GenerativeModel(
                model_name="gemini-3-flash-preview",
                tools=[google_search_tool]
            )
        return self._search_model

    async def prepare_context(
        self,
        query: str,
        workspace_path: Optional[str] = None,
        code_context: Optional[str] = None,
        file_list: Optional[List[str]] = None,
        search_docs: bool = True,
        max_files: int = 15,
    ) -> StructuredContext:
        """
        Prepare rich, structured context for Claude.

        Gemini does all the hunting:
        1. Analyzes the query to understand task type and complexity
        2. Discovers relevant files in workspace
        3. Fetches documentation if needed
        4. Structures everything for Claude

        Args:
            query: The user's request
            workspace_path: Path to the workspace/project
            code_context: Any code snippets provided
            file_list: Pre-specified files to consider
            search_docs: Whether to search web for documentation
            max_files: Maximum files to include in context

        Returns:
            StructuredContext ready for Claude
        """
        logger.info("context_gateway_start", query=query[:100])

        # Run analysis tasks in parallel
        tasks = [
            self._analyze_query(query, code_context),
            self._discover_files(query, workspace_path, file_list, max_files),
        ]

        if search_docs:
            tasks.append(self._search_documentation(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse results
        query_analysis = results[0] if not isinstance(results[0], Exception) else {}
        file_contexts = results[1] if not isinstance(results[1], Exception) else []
        doc_contexts = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []

        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("context_task_failed", task=i, error=str(result))

        # Build structured context
        context = StructuredContext(
            task_type=query_analysis.get("task_type", "general"),
            task_summary=query_analysis.get("summary", query),
            complexity=query_analysis.get("complexity", "medium"),
            relevant_files=file_contexts,
            entry_point=query_analysis.get("entry_point"),
            documentation=doc_contexts,
            recommended_framework=query_analysis.get("framework", "reason_flux"),
            framework_reason=query_analysis.get("framework_reason", "General-purpose reasoning"),
            chain_suggestion=query_analysis.get("chain"),
            execution_steps=query_analysis.get("steps", []),
            success_criteria=query_analysis.get("success_criteria", []),
            potential_blockers=query_analysis.get("blockers", []),
            related_patterns=query_analysis.get("patterns", []),
            dependencies=query_analysis.get("dependencies", []),
        )

        logger.info(
            "context_gateway_complete",
            task_type=context.task_type,
            files=len(context.relevant_files),
            docs=len(context.documentation),
            framework=context.recommended_framework,
        )

        return context

    async def _analyze_query(
        self,
        query: str,
        code_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use Gemini to deeply analyze the query."""
        model = self._get_model()

        prompt = f"""Analyze this coding task and provide structured analysis.

QUERY: {query}

{f"CODE CONTEXT:{chr(10)}{code_context[:2000]}" if code_context else ""}

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
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.3}
            )

            # Extract JSON from response
            text = response.text
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except Exception as e:
            logger.error("query_analysis_failed", error=str(e))
            return {}

    async def _discover_files(
        self,
        query: str,
        workspace_path: Optional[str],
        file_list: Optional[List[str]],
        max_files: int
    ) -> List[FileContext]:
        """Discover and rank relevant files using Gemini."""
        if not workspace_path and not file_list:
            return []

        model = self._get_model()

        # Get file listing
        files_to_analyze = []
        if file_list:
            files_to_analyze = file_list[:50]  # Cap at 50
        elif workspace_path:
            files_to_analyze = await self._list_workspace_files(workspace_path)

        if not files_to_analyze:
            return []

        # Have Gemini score relevance
        file_listing = "\n".join(files_to_analyze[:100])  # Cap listing

        prompt = f"""Given this query and file listing, identify the most relevant files.

QUERY: {query}

FILES:
{file_listing}

For each relevant file, respond in JSON array format:
[
    {{
        "path": "path/to/file.py",
        "relevance_score": 0.95,
        "summary": "Main entry point for authentication logic",
        "key_elements": ["login()", "validate_token()", "User class"]
    }}
]

Only include files that are actually relevant (score > 0.5).
Order by relevance score descending.
Maximum {max_files} files."""

        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.2}
            )

            text = response.text
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                files_data = json.loads(json_match.group())
                return [
                    FileContext(
                        path=f["path"],
                        relevance_score=f.get("relevance_score", 0.5),
                        summary=f.get("summary", ""),
                        key_elements=f.get("key_elements", []),
                    )
                    for f in files_data[:max_files]
                ]
            return []
        except Exception as e:
            logger.error("file_discovery_failed", error=str(e))
            return []

    async def _list_workspace_files(self, workspace_path: str) -> List[str]:
        """List relevant files in workspace (excludes common non-code files)."""
        exclude_patterns = {
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            ".pytest_cache", ".mypy_cache", "dist", "build", ".egg-info",
            ".tox", "htmlcov", ".coverage", "*.pyc", "*.pyo"
        }

        include_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
            ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
            ".kt", ".scala", ".sql", ".sh", ".yaml", ".yml", ".json",
            ".toml", ".md", ".rst", ".dockerfile"
        }

        files = []
        try:
            workspace = Path(workspace_path)
            for path in workspace.rglob("*"):
                if path.is_file():
                    # Check exclusions
                    if any(ex in str(path) for ex in exclude_patterns):
                        continue
                    # Check extensions
                    if path.suffix.lower() in include_extensions or path.name.lower() in {"dockerfile", "makefile"}:
                        rel_path = str(path.relative_to(workspace))
                        files.append(rel_path)
                        if len(files) >= 200:  # Cap at 200 files
                            break
        except Exception as e:
            logger.error("workspace_listing_failed", error=str(e))

        return files

    async def _search_documentation(self, query: str) -> List[DocumentationContext]:
        """Search web for relevant documentation using Gemini with Google Search."""
        try:
            model = self._get_search_model()

            prompt = f"""Search for relevant documentation, API references, or technical guides for:

{query}

Find official documentation, tutorials, or authoritative sources that would help with this task.
For each relevant source, provide:
- The URL/source
- A brief title
- The most relevant snippet or information

Focus on actionable, technical content."""

            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.3}
            )

            # Parse response into documentation contexts
            # The grounded response includes search results
            docs = []
            text = response.text

            # Try to extract structured data, or parse the text
            # For now, return as single context if there's content
            if text and len(text) > 50:
                docs.append(DocumentationContext(
                    source="Google Search (Gemini Grounded)",
                    title="Documentation Search Results",
                    snippet=text[:2000],  # Cap snippet size
                    relevance_score=0.8
                ))

            return docs
        except Exception as e:
            logger.warning("doc_search_failed", error=str(e))
            return []

    async def quick_analyze(self, query: str) -> Dict[str, Any]:
        """
        Quick analysis without file discovery or doc search.

        Use for fast routing decisions.
        """
        return await self._analyze_query(query)


# Global singleton
_gateway: Optional[ContextGateway] = None


def get_context_gateway() -> ContextGateway:
    """Get the global ContextGateway singleton."""
    global _gateway
    if _gateway is None:
        _gateway = ContextGateway()
    return _gateway


async def prepare_context_for_claude(
    query: str,
    workspace_path: Optional[str] = None,
    code_context: Optional[str] = None,
    file_list: Optional[List[str]] = None,
    search_docs: bool = True,
) -> StructuredContext:
    """
    Convenience function to prepare context.

    Example:
        context = await prepare_context_for_claude(
            query="Fix the auth bug",
            workspace_path="/my/project"
        )
        claude_prompt = context.to_claude_prompt()
    """
    gateway = get_context_gateway()
    return await gateway.prepare_context(
        query=query,
        workspace_path=workspace_path,
        code_context=code_context,
        file_list=file_list,
        search_docs=search_docs,
    )
