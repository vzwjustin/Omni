"""
File Discoverer: Gemini-Powered File Relevance Scoring

Discovers and ranks relevant files in a workspace using:
- File listing with smart filtering
- LLM-based relevance scoring
- Key element extraction

Refactored to offload blocking I/O to thread pool and use scandir for performance.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from ..constants import WORKSPACE
from ..correlation import get_correlation_id
from ..errors import ContextRetrievalError, LLMError, ProviderNotConfiguredError
from ..settings import get_settings

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

logger = structlog.get_logger("context.file_discoverer")


@dataclass
class FileContext:
    """Context about a discovered file."""

    path: str
    relevance_score: float  # 0-1
    summary: str  # Gemini-generated summary
    key_elements: list[str] = field(default_factory=list)  # functions, classes, etc.
    line_count: int = 0
    size_kb: float = 0


class FileDiscoverer:
    """
    Discovers relevant files in workspace using Gemini scoring.

    Process:
    1. List files in workspace (filtered by extension/exclusions)
    2. Send file listing to Gemini for relevance analysis
    3. Return ranked list of FileContext objects
    """

    def __init__(self):
        self.settings = get_settings()
        self._model = None

    def _get_client(self):
        """Get or create Gemini client for analysis."""
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

            self._model = genai.Client(api_key=api_key)
        return self._model

    async def discover(
        self,
        query: str,
        workspace_path: str | None = None,
        file_list: list[str] | None = None,
        max_files: int = 15,
    ) -> list[FileContext]:
        """
        Discover and rank relevant files.

        Args:
            query: The user's request
            workspace_path: Path to the workspace/project
            file_list: Pre-specified files to consider
            max_files: Maximum files to return

        Returns:
            List of FileContext objects, ranked by relevance
        """
        if not workspace_path and not file_list:
            return []

        client = self._get_client()
        model_name = self.settings.routing_model or "gemini-3-flash-preview"

        # Get file listing
        files_to_analyze = []
        if file_list:
            files_to_analyze = file_list[: WORKSPACE.MAX_FILES_LIST]
        elif workspace_path:
            files_to_analyze = await self._list_workspace_files(workspace_path)

        if not files_to_analyze:
            return []

        # Have Gemini score relevance
        file_listing = "\n".join(files_to_analyze[: WORKSPACE.MAX_FILES_ANALYZE])

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
                client.models.generate_content,
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2, response_mime_type="application/json"
                ),
            )

            # With JSON mode, response.text should be valid JSON
            files_data = json.loads(response.text)

            # Ensure it's a list (handle case where model might wrap in object)
            if isinstance(files_data, dict) and "files" in files_data:
                files_data = files_data["files"]
            elif isinstance(files_data, dict):
                # Try to find any list value
                for val in files_data.values():
                    if isinstance(val, list):
                        files_data = val
                        break

            if not isinstance(files_data, list):
                logger.warning("file_discovery_invalid_format", format=type(files_data))
                return []

            result = [
                FileContext(
                    path=f["path"],
                    relevance_score=f.get("relevance_score", 0.5),
                    summary=f.get("summary", ""),
                    key_elements=f.get("key_elements", []),
                )
                for f in files_data[:max_files]
            ]
            logger.debug(
                "file_discovery_complete",
                files_found=len(result),
                top_file=result[0].path if result else None,
            )
            return result
        except (LLMError, ProviderNotConfiguredError):
            raise  # Re-raise custom LLM errors
        except Exception as e:
            # Graceful degradation: File discovery is best-effort. If parsing fails
            # or an unexpected error occurs, we log and raise a structured LLMError
            # rather than crashing, allowing callers to handle discovery failures.
            logger.error(
                "file_discovery_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            raise LLMError(f"File discovery failed: {e}") from e

    async def _list_workspace_files(self, workspace_path: str) -> list[str]:
        """
        List relevant files in workspace asynchronously.
        Offloads blocking I/O to a thread to prevent event loop starvation.
        """
        try:
            return await asyncio.to_thread(self._sync_list_files, workspace_path)
        except Exception as e:
            # Differentiate catastrophic failure from partial failure
            logger.error(
                "workspace_listing_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            # Raise structured error for retry/handling logic upstream
            raise ContextRetrievalError(f"Failed to access workspace: {e}") from e

    def _sync_list_files(self, workspace_path: str) -> list[str]:
        """
        Synchronous implementation of file listing using os.scandir for performance.
        Strictly separated for thread safety.

        Security: Validates all paths to prevent path traversal attacks.
        """
        exclude_patterns = set(WORKSPACE.EXCLUDE_DIRS)
        include_extensions = set(WORKSPACE.CODE_EXTENSIONS)
        special_filenames = {"dockerfile", "makefile"}
        files = []

        # Security: Normalize and validate workspace path
        try:
            workspace_path_resolved = Path(workspace_path).resolve()
            if not workspace_path_resolved.exists():
                logger.warning("workspace_path_not_found", path=workspace_path)
                return []
            if not workspace_path_resolved.is_dir():
                logger.warning("workspace_path_not_directory", path=workspace_path)
                return []
        except (OSError, ValueError) as e:
            logger.warning("workspace_path_invalid", path=workspace_path, error=str(e))
            return []

        try:
            # Use os.walk which uses scandir internally (faster than glob)
            # Top-down iteration allows us to modify dirnames in-place to prune traversal
            for root, dirs, filenames in os.walk(str(workspace_path_resolved), topdown=True):
                # Modify dirs in-place to skip excluded directories
                # This prevents descending into excluded directories entirely
                dirs[:] = [
                    d
                    for d in dirs
                    if d not in exclude_patterns and not any(ex in d for ex in exclude_patterns)
                ]

                for filename in filenames:
                    # Check extensions or special names
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in include_extensions or filename.lower() in special_filenames:
                        full_path = Path(root) / filename

                        # Security: Validate file is within workspace (prevent symlink attacks)
                        try:
                            full_path_resolved = full_path.resolve()
                            full_path_resolved.relative_to(workspace_path_resolved)
                        except (ValueError, OSError):
                            # File is outside workspace or inaccessible - skip it
                            logger.debug(
                                "skipping_file_outside_workspace",
                                file=str(full_path),
                                workspace=str(workspace_path_resolved),
                            )
                            continue

                        rel_path = os.path.relpath(
                            str(full_path_resolved), str(workspace_path_resolved)
                        )
                        files.append(rel_path)

                        if len(files) >= WORKSPACE.MAX_FILES_SCAN:
                            return files

        except OSError:
            # Raise to caller (async wrapper) if root dir is bad
            raise

        return files
