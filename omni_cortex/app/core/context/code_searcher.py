"""
Code Searcher: Codebase Search via grep/ripgrep and git

Searches the codebase using:
- ripgrep (preferred) or grep for pattern matching
- git log for commit history search
"""

import asyncio
import os
import structlog
from dataclasses import dataclass
from typing import List, Optional

from ..settings import get_settings
from ..constants import CONTENT, SEARCH
from ..errors import LLMError, ProviderNotConfiguredError, OmniCortexError
from ..correlation import get_correlation_id

# Try to import Google AI for query extraction
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None

logger = structlog.get_logger("context.code_searcher")


@dataclass
class CodeSearchContext:
    """Context from code search (grep/git)."""
    search_type: str  # grep, git_log, git_blame
    query: str
    results: str  # Command output
    file_count: int = 0
    match_count: int = 0


class CodeSearcher:
    """
    Searches codebase using grep/ripgrep and git.

    Features:
    - Uses ripgrep if available (faster), falls back to grep
    - Extracts search terms from query using Gemini
    - Searches git log for relevant commits
    """

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._search_cmd = None

    def _get_model(self):
        """Get or create Gemini model for query extraction."""
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

    def _detect_search_command(self) -> Optional[str]:
        """Detect available search command (ripgrep or grep)."""
        if self._search_cmd is not None:
            return self._search_cmd

        # Try ripgrep first (faster)
        if os.system("which rg > /dev/null 2>&1") == 0:
            self._search_cmd = "rg"
        elif os.system("which grep > /dev/null 2>&1") == 0:
            self._search_cmd = "grep"
        else:
            self._search_cmd = ""  # Empty string means none available

        return self._search_cmd if self._search_cmd else None

    async def search(
        self,
        query: str,
        workspace_path: str,
        search_queries: Optional[List[str]] = None
    ) -> List[CodeSearchContext]:
        """
        Search codebase using grep/ripgrep or git commands.

        Args:
            query: User's original query
            workspace_path: Path to workspace
            search_queries: Specific search queries (extracted by Gemini if not provided)

        Returns:
            List of code search results
        """
        if not workspace_path or not os.path.exists(workspace_path):
            return []

        results = []

        # If no specific queries, let Gemini extract them from the query
        if not search_queries:
            search_queries = await self._extract_search_queries(query)

        # Detect search command
        search_cmd = self._detect_search_command()

        if search_cmd and search_queries:
            # Run grep/rg searches
            grep_results = await self._run_grep_searches(
                search_cmd, search_queries, workspace_path
            )
            results.extend(grep_results)

        # Also try git log if this is a git repo
        git_results = await self._search_git_log(query, workspace_path)
        if git_results:
            results.append(git_results)

        logger.info(
            "code_search_complete",
            queries=len(search_queries) if search_queries else 0,
            results=len(results)
        )
        return results

    async def _extract_search_queries(self, query: str) -> List[str]:
        """Extract specific search terms from the user's query."""
        try:
            model = self._get_model()
            prompt = f"""Extract 1-3 specific code search queries from this task:

{query}

Return ONLY the search terms, one per line. Focus on:
- Function/class names mentioned
- Error messages or strings
- Technical keywords
- File patterns

Example output:
authenticate
login_required
JWT"""

            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.2}
            )
            search_queries = [
                line.strip()
                for line in response.text.split('\n')
                if line.strip()
            ][:SEARCH.SEARCH_QUERIES_MAX]
            logger.debug("search_queries_extracted", count=len(search_queries))
            return search_queries
        except (LLMError, ProviderNotConfiguredError) as e:
            logger.warning("search_query_extraction_failed", error=str(e), error_type=type(e).__name__)
            return []
        except Exception as e:
            logger.warning("search_query_extraction_failed", error=str(e))
            return []

    async def _run_grep_searches(
        self,
        search_cmd: str,
        search_queries: List[str],
        workspace_path: str
    ) -> List[CodeSearchContext]:
        """Run grep/ripgrep searches for each query."""
        results = []

        for search_term in search_queries[:SEARCH.SEARCH_QUERIES_MAX]:
            try:
                if search_cmd == "rg":
                    cmd = [
                        "rg",
                        "--no-heading",
                        "--line-number",
                        "--context", "2",
                        "--max-count", str(SEARCH.GREP_MAX_COUNT),
                        "--type-not", "lock",
                        search_term,
                        workspace_path
                    ]
                else:
                    cmd = [
                        "grep",
                        "-r",
                        "-n",
                        "-C", "2",
                        "--exclude=*.lock",
                        "--exclude-dir=node_modules",
                        "--exclude-dir=.git",
                        search_term,
                        workspace_path
                    ]

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

                if proc.returncode == 0 and stdout:
                    output = stdout.decode('utf-8', errors='ignore')
                    lines = output.split('\n')
                    file_count = len(set(line.split(':')[0] for line in lines if ':' in line))

                    results.append(CodeSearchContext(
                        search_type="grep",
                        query=search_term,
                        results=output[:CONTENT.COMMAND_OUTPUT],
                        file_count=file_count,
                        match_count=len([l for l in lines if search_term in l])
                    ))

            except asyncio.TimeoutError:
                logger.warning("code_search_timeout", query=search_term)
            except OmniCortexError:
                raise  # Re-raise custom errors
            except Exception as e:
                logger.error(
                    "code_search_failed",
                    query=search_term,
                    error=str(e),
                    correlation_id=get_correlation_id()
                )
                # Code search errors are non-fatal - continue with other searches

        return results

    async def _search_git_log(
        self,
        query: str,
        workspace_path: str
    ) -> Optional[CodeSearchContext]:
        """Search git log for relevant commits."""
        git_dir = os.path.join(workspace_path, ".git")
        if not os.path.exists(git_dir):
            return None

        try:
            cmd = [
                "git",
                "-C", workspace_path,
                "log",
                "--all",
                "--oneline",
                "--grep", query[:CONTENT.QUERY_LOG],
                f"-{SEARCH.GIT_LOG_MAX}"
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)

            if stdout:
                output = stdout.decode('utf-8', errors='ignore')
                if output.strip():
                    return CodeSearchContext(
                        search_type="git_log",
                        query=query[:CONTENT.QUERY_LOG],
                        results=output,
                        file_count=0,
                        match_count=len(output.split('\n'))
                    )
        except asyncio.TimeoutError:
            logger.debug("git_log_search_timeout")
        except OmniCortexError:
            raise  # Re-raise custom errors
        except Exception as e:
            logger.debug("git_log_search_skipped", error=str(e))

        return None
