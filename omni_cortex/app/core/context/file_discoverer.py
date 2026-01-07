"""
File Discoverer: Gemini-Powered File Relevance Scoring

Discovers and ranks relevant files in a workspace using:
- File listing with smart filtering
- LLM-based relevance scoring
- Key element extraction
"""

import asyncio
import json
import re
import structlog
from pathlib import Path
from typing import List, Optional

from ..settings import get_settings
from ..constants import CONTENT, WORKSPACE
from ..errors import LLMError, ProviderNotConfiguredError, OmniCortexError
from ..correlation import get_correlation_id

# Try to import Google AI
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None

logger = structlog.get_logger("context.file_discoverer")


# Import dataclass from parent - will be available after refactor
# For now, define inline to avoid circular imports
from dataclasses import dataclass, field


@dataclass
class FileContext:
    """Context about a discovered file."""
    path: str
    relevance_score: float  # 0-1
    summary: str  # Gemini-generated summary
    key_elements: List[str] = field(default_factory=list)  # functions, classes, etc.
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

    def _get_model(self):
        """Get or create Gemini model for analysis."""
        if self._model is None:
            if not GOOGLE_AI_AVAILABLE:
                raise ProviderNotConfiguredError(
                    "google-generativeai not installed",
                    details={"provider": "google", "package": "google-generativeai"}
                )

            api_key = self.settings.google_api_key
            if not api_key:
                raise ProviderNotConfiguredError(
                    "GOOGLE_API_KEY not configured",
                    details={"provider": "google", "env_var": "GOOGLE_API_KEY"}
                )

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                self.settings.routing_model or "gemini-2.0-flash"
            )
        return self._model

    async def discover(
        self,
        query: str,
        workspace_path: Optional[str] = None,
        file_list: Optional[List[str]] = None,
        max_files: int = 15
    ) -> List[FileContext]:
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

        model = self._get_model()

        # Get file listing
        files_to_analyze = []
        if file_list:
            files_to_analyze = file_list[:WORKSPACE.MAX_FILES_LIST]
        elif workspace_path:
            files_to_analyze = await self._list_workspace_files(workspace_path)

        if not files_to_analyze:
            return []

        # Have Gemini score relevance
        file_listing = "\n".join(files_to_analyze[:WORKSPACE.MAX_FILES_ANALYZE])

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
                    top_file=result[0].path if result else None
                )
                return result
            return []
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
                correlation_id=get_correlation_id()
            )
            raise LLMError(f"File discovery failed: {e}") from e

    async def _list_workspace_files(self, workspace_path: str) -> List[str]:
        """
        List relevant files in workspace.

        Excludes common non-code directories and filters by extension.
        """
        exclude_patterns = set(WORKSPACE.EXCLUDE_DIRS)
        include_extensions = set(WORKSPACE.CODE_EXTENSIONS)

        # Also include dockerfile extension check
        special_filenames = {"dockerfile", "makefile"}

        files = []
        try:
            workspace = Path(workspace_path)
            for path in workspace.rglob("*"):
                if path.is_file():
                    # Check exclusions
                    if any(ex in str(path) for ex in exclude_patterns):
                        continue
                    # Check extensions or special names
                    if (path.suffix.lower() in include_extensions or
                            path.name.lower() in special_filenames):
                        rel_path = str(path.relative_to(workspace))
                        files.append(rel_path)
                        if len(files) >= WORKSPACE.MAX_FILES_SCAN:
                            break
        except OmniCortexError:
            raise  # Re-raise custom errors
        except Exception as e:
            # Graceful degradation: File listing is best-effort for discovery.
            # Filesystem errors (permissions, broken symlinks, etc.) should not
            # crash the entire discovery process - we return whatever files we
            # successfully collected before the error occurred.
            logger.error(
                "workspace_listing_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id()
            )
            # File system errors are not fatal - return what we have

        logger.debug("workspace_files_listed", count=len(files))
        return files
