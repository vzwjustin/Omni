"""
Multi-Repository File Discoverer

Extends FileDiscoverer to handle multiple repositories in a workspace:
- Detects multiple git repositories
- Analyzes repositories in parallel
- Follows cross-repository dependencies
- Handles repository access failures gracefully
"""

import asyncio
import os
import re
import subprocess
import structlog
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field

from .file_discoverer import FileDiscoverer, FileContext
from .enhanced_models import RepoInfo, CrossRepoDependency, EnhancedFileContext
from ..settings import get_settings
from ..errors import ContextRetrievalError
from ..correlation import get_correlation_id

logger = structlog.get_logger("multi_repo_discoverer")


class MultiRepoFileDiscoverer(FileDiscoverer):
    """
    Multi-repository file discoverer.
    
    Extends FileDiscoverer to handle workspaces with multiple repositories:
    - Detects all git repositories in workspace
    - Runs parallel Gemini analysis per repository
    - Follows cross-repository dependencies
    - Handles inaccessible repositories gracefully
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
    
    async def discover_multi_repo(
        self,
        query: str,
        workspace_path: str,
        max_files: int = 15
    ) -> Tuple[List[EnhancedFileContext], List[RepoInfo], List[CrossRepoDependency]]:
        """
        Discover files across multiple repositories.
        
        Args:
            query: The user's request
            workspace_path: Path to the workspace (may contain multiple repos)
            max_files: Maximum files to return across all repositories
        
        Returns:
            Tuple of (enhanced_file_contexts, repo_info_list, cross_repo_dependencies)
        """
        logger.info("multi_repo_discovery_start", workspace=workspace_path)
        
        # Phase 1: Detect repositories
        repos = await self._detect_repositories(workspace_path)
        
        if not repos:
            logger.info("no_repositories_detected", workspace=workspace_path)
            # Fall back to single-repo discovery
            files = await self.discover(query, workspace_path, max_files=max_files)
            enhanced_files = [
                EnhancedFileContext(
                    path=f.path,
                    relevance_score=f.relevance_score,
                    summary=f.summary,
                    key_elements=f.key_elements,
                    line_count=f.line_count,
                    size_kb=f.size_kb,
                    repository=None
                )
                for f in files
            ]
            return enhanced_files, [], []
        
        logger.info("repositories_detected", count=len(repos), repos=[r.name for r in repos])
        
        # Phase 2: Analyze repositories in parallel
        repo_results = await self._analyze_repositories_parallel(query, repos, max_files)
        
        # Phase 3: Follow cross-repository dependencies
        cross_repo_deps = await self._follow_cross_repo_dependencies(repos, repo_results)
        
        # Phase 4: Merge and rank results
        all_files = []
        for repo, files in repo_results.items():
            for f in files:
                enhanced_file = EnhancedFileContext(
                    path=f.path,
                    relevance_score=f.relevance_score,
                    summary=f.summary,
                    key_elements=f.key_elements,
                    line_count=f.line_count,
                    size_kb=f.size_kb,
                    repository=repo.name
                )
                all_files.append(enhanced_file)
        
        # Sort by relevance and limit
        all_files.sort(key=lambda f: f.relevance_score, reverse=True)
        all_files = all_files[:max_files]
        
        logger.info(
            "multi_repo_discovery_complete",
            total_files=len(all_files),
            repos=len(repos),
            dependencies=len(cross_repo_deps)
        )
        
        return all_files, repos, cross_repo_deps
    
    def generate_repository_warnings(
        self,
        repos: List[RepoInfo]
    ) -> List[str]:
        """
        Generate warning messages for inaccessible or problematic repositories.
        
        Args:
            repos: List of repository information
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        for repo in repos:
            if not repo.is_accessible:
                warning = f"⚠️ Repository '{repo.name}' at {repo.path} is not accessible"
                if repo.error_message:
                    warning += f": {repo.error_message}"
                warnings.append(warning)
            elif not repo.access_permissions.get("read", False):
                warnings.append(
                    f"⚠️ Repository '{repo.name}' has limited read permissions"
                )
            elif not repo.last_commit:
                warnings.append(
                    f"⚠️ Repository '{repo.name}' may not be a valid git repository"
                )
        
        return warnings
    
    async def _detect_repositories(self, workspace_path: str) -> List[RepoInfo]:
        """
        Detect all git repositories in workspace.
        
        Args:
            workspace_path: Path to workspace
        
        Returns:
            List of RepoInfo objects
        """
        try:
            return await asyncio.to_thread(self._sync_detect_repositories, workspace_path)
        except Exception as e:
            logger.error(
                "repository_detection_failed",
                workspace=workspace_path,
                error=str(e),
                correlation_id=get_correlation_id()
            )
            return []
    
    def _sync_detect_repositories(self, workspace_path: str) -> List[RepoInfo]:
        """
        Synchronous repository detection.
        
        Searches for .git directories to identify repositories.
        """
        repos = []
        workspace = Path(workspace_path)
        
        if not workspace.exists():
            return repos
        
        # Check if workspace itself is a git repo
        if (workspace / '.git').exists():
            repo_info = self._create_repo_info(workspace_path, workspace_path)
            if repo_info:
                repos.append(repo_info)
        
        # Search for nested repositories
        # Limit depth to avoid excessive scanning
        max_depth = 3
        for root, dirs, files in os.walk(workspace_path):
            # Calculate current depth
            depth = root[len(workspace_path):].count(os.sep)
            if depth >= max_depth:
                dirs[:] = []  # Don't descend further
                continue
            
            # Check if this directory is a git repo
            if '.git' in dirs:
                repo_path = root
                # Skip if this is the workspace root (already checked)
                if repo_path != workspace_path:
                    repo_info = self._create_repo_info(repo_path, workspace_path)
                    if repo_info:
                        repos.append(repo_info)
                
                # Don't descend into this repo's subdirectories
                dirs[:] = []
        
        return repos
    
    def _create_repo_info(self, repo_path: str, workspace_path: str) -> Optional[RepoInfo]:
        """
        Create RepoInfo object for a repository.

        Args:
            repo_path: Path to repository
            workspace_path: Path to workspace root

        Returns:
            RepoInfo object or None if repo is inaccessible
        """
        try:
            # Validate and normalize paths to prevent path traversal
            repo = Path(repo_path).resolve()
            workspace = Path(workspace_path).resolve()

            # Security: Ensure repo_path is within workspace_path
            # Prevents path traversal attacks
            try:
                repo.relative_to(workspace)
            except ValueError:
                logger.warning(
                    "repo_path_outside_workspace",
                    repo_path=str(repo),
                    workspace_path=str(workspace)
                )
                return None

            # Convert back to string for subprocess calls
            repo_path_validated = str(repo)

            # Get repository name
            name = repo.name

            # Get git root (should be repo_path)
            git_root = repo_path_validated

            # Load .gitignore patterns
            ignore_patterns = self._load_gitignore_patterns(repo_path_validated)

            # Check access permissions
            access_permissions = {
                "read": os.access(repo_path_validated, os.R_OK),
                "write": os.access(repo_path_validated, os.W_OK),
                "execute": os.access(repo_path_validated, os.X_OK),
            }

            # Get last commit info
            last_commit = None
            branch = None
            try:
                # Get current branch
                result = subprocess.run(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    cwd=repo_path_validated,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    branch = result.stdout.strip()

                # Get last commit hash
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    cwd=repo_path_validated,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    last_commit = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            is_accessible = access_permissions["read"] and access_permissions["execute"]

            return RepoInfo(
                path=repo_path_validated,
                name=name,
                git_root=git_root,
                ignore_patterns=ignore_patterns,
                access_permissions=access_permissions,
                last_commit=last_commit,
                branch=branch,
                is_accessible=is_accessible,
                error_message=None if is_accessible else "Repository not accessible"
            )
            
        except Exception as e:
            logger.warning(
                "repo_info_creation_failed",
                repo_path=repo_path,
                error=str(e)
            )
            return RepoInfo(
                path=repo_path,
                name=Path(repo_path).name,
                git_root=repo_path,
                ignore_patterns=[],
                access_permissions={},
                last_commit=None,
                branch=None,
                is_accessible=False,
                error_message=str(e)
            )
    
    def _load_gitignore_patterns(self, repo_path: str) -> List[str]:
        """
        Load .gitignore patterns from repository.
        
        Args:
            repo_path: Path to repository
        
        Returns:
            List of ignore patterns
        """
        patterns = []
        gitignore_path = Path(repo_path) / '.gitignore'
        
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith('#'):
                            patterns.append(line)
            except Exception as e:
                logger.warning(
                    "gitignore_load_failed",
                    repo_path=repo_path,
                    error=str(e)
                )
        
        return patterns
    
    async def _analyze_repositories_parallel(
        self,
        query: str,
        repos: List[RepoInfo],
        max_files_per_repo: int
    ) -> Dict[RepoInfo, List[FileContext]]:
        """
        Analyze multiple repositories in parallel.
        
        Args:
            query: The user's request
            repos: List of repositories to analyze
            max_files_per_repo: Maximum files per repository
        
        Returns:
            Dictionary mapping RepoInfo to discovered files
        """
        # Create analysis tasks for accessible repositories
        tasks = []
        accessible_repos = []
        inaccessible_repos = []
        
        for repo in repos:
            if repo.is_accessible:
                tasks.append(self._analyze_repository(repo, query, max_files_per_repo))
                accessible_repos.append(repo)
            else:
                inaccessible_repos.append(repo)
                logger.warning(
                    "skipping_inaccessible_repo",
                    repo=repo.name,
                    path=repo.path,
                    error=repo.error_message
                )
        
        # Log summary of repository access
        if inaccessible_repos:
            logger.warning(
                "inaccessible_repositories_detected",
                count=len(inaccessible_repos),
                repos=[r.name for r in inaccessible_repos],
                accessible_count=len(accessible_repos)
            )
        
        if not tasks:
            logger.error("no_accessible_repositories", total_repos=len(repos))
            return {}
        
        # Run analyses in parallel with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0  # 60 second timeout for all repo analyses
            )
        except asyncio.TimeoutError:
            logger.error(
                "repo_analysis_timeout",
                repos=[r.name for r in accessible_repos]
            )
            # Return empty results for all repos
            return {repo: [] for repo in accessible_repos}
        
        # Build result dictionary with error handling
        repo_results = {}
        for repo, result in zip(accessible_repos, results):
            if isinstance(result, Exception):
                logger.error(
                    "repo_analysis_failed",
                    repo=repo.name,
                    path=repo.path,
                    error=str(result),
                    error_type=type(result).__name__,
                    correlation_id=get_correlation_id()
                )
                # Mark repo as failed but continue with others
                repo_results[repo] = []
                # Update repo info to reflect failure
                repo.is_accessible = False
                repo.error_message = f"Analysis failed: {str(result)}"
            else:
                repo_results[repo] = result
        
        # Log success summary
        successful_repos = [r for r, files in repo_results.items() if files]
        logger.info(
            "repo_analysis_summary",
            total_repos=len(repos),
            accessible=len(accessible_repos),
            inaccessible=len(inaccessible_repos),
            successful=len(successful_repos),
            failed=len(accessible_repos) - len(successful_repos)
        )
        
        return repo_results
    
    async def _analyze_repository(
        self,
        repo: RepoInfo,
        query: str,
        max_files: int
    ) -> List[FileContext]:
        """
        Analyze a single repository.
        
        Args:
            repo: Repository information
            query: The user's request
            max_files: Maximum files to return
        
        Returns:
            List of FileContext objects
        """
        try:
            # Use parent class discover method for this repository
            files = await self.discover(
                query=query,
                workspace_path=repo.path,
                max_files=max_files
            )
            
            # Filter files based on .gitignore patterns
            if repo.ignore_patterns:
                files = self._filter_by_gitignore(files, repo)
            
            logger.info(
                "repo_analysis_complete",
                repo=repo.name,
                files_found=len(files)
            )
            
            return files
            
        except Exception as e:
            logger.error(
                "repo_analysis_error",
                repo=repo.name,
                error=str(e),
                correlation_id=get_correlation_id()
            )
            raise
    
    def _filter_by_gitignore(
        self,
        files: List[FileContext],
        repo: RepoInfo
    ) -> List[FileContext]:
        """
        Filter files based on repository's .gitignore patterns.
        
        Args:
            files: List of file contexts
            repo: Repository information with ignore patterns
        
        Returns:
            Filtered list of file contexts
        """
        if not repo.ignore_patterns:
            return files
        
        filtered_files = []
        for file_ctx in files:
            # Check if file matches any ignore pattern
            should_ignore = False
            for pattern in repo.ignore_patterns:
                if self._matches_gitignore_pattern(file_ctx.path, pattern):
                    should_ignore = True
                    logger.debug(
                        "file_ignored_by_gitignore",
                        file=file_ctx.path,
                        pattern=pattern,
                        repo=repo.name
                    )
                    break
            
            if not should_ignore:
                filtered_files.append(file_ctx)
        
        if len(filtered_files) < len(files):
            logger.info(
                "files_filtered_by_gitignore",
                repo=repo.name,
                original_count=len(files),
                filtered_count=len(filtered_files),
                removed=len(files) - len(filtered_files)
            )
        
        return filtered_files
    
    def _matches_gitignore_pattern(self, file_path: str, pattern: str) -> bool:
        """
        Check if file path matches a .gitignore pattern.
        
        Implements basic .gitignore pattern matching:
        - * matches any string except /
        - ** matches any string including /
        - / at start means root of repo
        - / at end means directory only
        
        Args:
            file_path: Relative file path
            pattern: .gitignore pattern
        
        Returns:
            True if file matches pattern
        """
        # Remove leading/trailing whitespace
        pattern = pattern.strip()
        
        # Skip empty patterns
        if not pattern:
            return False
        
        # Handle negation (! prefix) - we don't support this yet
        if pattern.startswith('!'):
            return False
        
        # Handle directory-only patterns (trailing /)
        is_dir_only = pattern.endswith('/')
        if is_dir_only:
            pattern = pattern[:-1]
            # For simplicity, we'll skip directory-only patterns
            # since we're working with files
            return False
        
        # Handle root-relative patterns (leading /)
        if pattern.startswith('/'):
            pattern = pattern[1:]
            # Match from root
            return self._glob_match(file_path, pattern)
        
        # Handle ** (match any directory depth)
        if '**' in pattern:
            # Convert ** to regex
            regex_pattern = pattern.replace('**', '.*')
            regex_pattern = regex_pattern.replace('*', '[^/]*')
            regex_pattern = '^' + regex_pattern + '$'
            return bool(re.match(regex_pattern, file_path))
        
        # Handle * (match within directory)
        if '*' in pattern:
            # Check if pattern matches filename or any parent directory
            parts = file_path.split('/')
            for i in range(len(parts)):
                subpath = '/'.join(parts[i:])
                if self._glob_match(subpath, pattern):
                    return True
            return False
        
        # Exact match or substring match
        return pattern in file_path or file_path.endswith(pattern)
    
    def _glob_match(self, text: str, pattern: str) -> bool:
        """
        Simple glob pattern matching.
        
        Args:
            text: Text to match
            pattern: Glob pattern with * wildcards
        
        Returns:
            True if text matches pattern
        """
        # Convert glob to regex
        regex_pattern = pattern.replace('*', '[^/]*')
        regex_pattern = '^' + regex_pattern + '$'
        return bool(re.match(regex_pattern, text))
    
    async def _follow_cross_repo_dependencies(
        self,
        repos: List[RepoInfo],
        repo_results: Dict[RepoInfo, List[FileContext]]
    ) -> List[CrossRepoDependency]:
        """
        Follow cross-repository dependencies.
        
        Analyzes import statements and API calls to detect dependencies
        between repositories.
        
        Args:
            repos: List of all repositories
            repo_results: Discovered files per repository
        
        Returns:
            List of CrossRepoDependency objects
        """
        dependencies = []
        
        # Build repository name to RepoInfo mapping
        repo_by_name = {repo.name: repo for repo in repos}
        
        # Analyze each file for cross-repo references
        for repo, files in repo_results.items():
            for file_ctx in files:
                file_path = Path(repo.path) / file_ctx.path
                
                # Skip if file doesn't exist or isn't accessible
                if not file_path.exists():
                    continue
                
                try:
                    # Read file content
                    content = await asyncio.to_thread(self._read_file_safe, file_path)
                    if not content:
                        continue
                    
                    # Detect cross-repo dependencies
                    file_deps = self._detect_cross_repo_references(
                        content,
                        file_ctx.path,
                        repo,
                        repo_by_name
                    )
                    dependencies.extend(file_deps)
                    
                except Exception as e:
                    logger.debug(
                        "dependency_analysis_failed",
                        file=file_ctx.path,
                        repo=repo.name,
                        error=str(e)
                    )
        
        logger.info(
            "cross_repo_dependencies_detected",
            count=len(dependencies)
        )
        
        return dependencies
    
    def _read_file_safe(self, file_path: Path, max_size_kb: int = 500) -> Optional[str]:
        """
        Safely read file content with size limit.
        
        Args:
            file_path: Path to file
            max_size_kb: Maximum file size to read in KB
        
        Returns:
            File content or None if too large/unreadable
        """
        try:
            # Check file size
            size_kb = file_path.stat().st_size / 1024
            if size_kb > max_size_kb:
                return None
            
            # Read content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return None
    
    def _detect_cross_repo_references(
        self,
        content: str,
        file_path: str,
        source_repo: RepoInfo,
        repo_by_name: Dict[str, RepoInfo]
    ) -> List[CrossRepoDependency]:
        """
        Detect cross-repository references in file content.
        
        Looks for:
        - Import statements referencing other repos
        - API calls to other services
        - Configuration references
        
        Args:
            content: File content
            file_path: Relative path to file
            source_repo: Source repository
            repo_by_name: Mapping of repo names to RepoInfo
        
        Returns:
            List of detected dependencies
        """
        dependencies = []
        seen_deps = set()  # Deduplicate dependencies
        
        # Pattern 1: Python imports from other repos
        # Example: from other_service.api import handler
        import_patterns = [
            r'from\s+([a-zA-Z0-9_]+)\..*?import',  # Python
            r'import\s+([a-zA-Z0-9_]+)',  # Python simple import
            r'require\(["\']([a-zA-Z0-9_/-]+)["\']\)',  # Node.js
            r'import\s+.*?from\s+["\']([a-zA-Z0-9_/-]+)["\']',  # ES6
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                module_name = match.group(1).split('/')[0]  # Get root module
                module_name = module_name.replace('-', '_')  # Normalize
                
                if module_name in repo_by_name and module_name != source_repo.name:
                    dep_key = (source_repo.name, module_name, "import")
                    if dep_key not in seen_deps:
                        dependencies.append(CrossRepoDependency(
                            source_repo=source_repo.name,
                            target_repo=module_name,
                            dependency_type="import",
                            source_file=file_path,
                            target_file=None,
                            confidence=0.9
                        ))
                        seen_deps.add(dep_key)
        
        # Pattern 2: API calls to other services
        # Example: requests.get("http://other-service/api/...")
        api_patterns = [
            r'(?:requests\.|http\.|fetch\(|axios\.).*?["\']https?://([a-zA-Z0-9_-]+)',
            r'["\']https?://([a-zA-Z0-9_-]+)(?::\d+)?/api',
            r'API_URL\s*=\s*["\']https?://([a-zA-Z0-9_-]+)',
        ]
        
        for pattern in api_patterns:
            for match in re.finditer(pattern, content):
                service_name = match.group(1)
                # Try to match service name to repository name
                for repo_name in repo_by_name:
                    normalized_repo = repo_name.replace('_', '-')
                    normalized_service = service_name.replace('_', '-')
                    
                    if (normalized_repo in normalized_service or 
                        normalized_service in normalized_repo):
                        if repo_name != source_repo.name:
                            dep_key = (source_repo.name, repo_name, "api_call")
                            if dep_key not in seen_deps:
                                dependencies.append(CrossRepoDependency(
                                    source_repo=source_repo.name,
                                    target_repo=repo_name,
                                    dependency_type="api_call",
                                    source_file=file_path,
                                    target_file=None,
                                    confidence=0.7
                                ))
                                seen_deps.add(dep_key)
        
        # Pattern 3: Configuration references
        # Example: SERVICE_URL = "other-service:8080"
        config_patterns = [
            r'[A-Z_]+\s*=\s*["\']([a-zA-Z0-9_-]+):',
            r'host:\s*["\']([a-zA-Z0-9_-]+)["\']',
            r'service:\s*["\']([a-zA-Z0-9_-]+)["\']',
        ]
        
        for pattern in config_patterns:
            for match in re.finditer(pattern, content):
                service_name = match.group(1)
                for repo_name in repo_by_name:
                    normalized_repo = repo_name.replace('_', '-')
                    normalized_service = service_name.replace('_', '-')
                    
                    if (normalized_repo in normalized_service or 
                        normalized_service in normalized_repo):
                        if repo_name != source_repo.name:
                            dep_key = (source_repo.name, repo_name, "config_reference")
                            if dep_key not in seen_deps:
                                dependencies.append(CrossRepoDependency(
                                    source_repo=source_repo.name,
                                    target_repo=repo_name,
                                    dependency_type="config_reference",
                                    source_file=file_path,
                                    target_file=None,
                                    confidence=0.6
                                ))
                                seen_deps.add(dep_key)
        
        return dependencies
    
    def build_dependency_graph(
        self,
        dependencies: List[CrossRepoDependency]
    ) -> Dict[str, Dict[str, List[CrossRepoDependency]]]:
        """
        Build a dependency graph from cross-repository dependencies.
        
        Args:
            dependencies: List of cross-repository dependencies
        
        Returns:
            Nested dictionary: {source_repo: {target_repo: [dependencies]}}
        """
        graph: Dict[str, Dict[str, List[CrossRepoDependency]]] = {}
        
        for dep in dependencies:
            if dep.source_repo not in graph:
                graph[dep.source_repo] = {}
            
            if dep.target_repo not in graph[dep.source_repo]:
                graph[dep.source_repo][dep.target_repo] = []
            
            graph[dep.source_repo][dep.target_repo].append(dep)
        
        return graph
    
    def get_transitive_dependencies(
        self,
        repo_name: str,
        graph: Dict[str, Dict[str, List[CrossRepoDependency]]],
        visited: Optional[Set[str]] = None
    ) -> Set[str]:
        """
        Get all transitive dependencies for a repository.
        
        Args:
            repo_name: Repository to get dependencies for
            graph: Dependency graph
            visited: Set of already visited repos (for cycle detection)
        
        Returns:
            Set of all dependent repository names
        """
        if visited is None:
            visited = set()
        
        if repo_name in visited:
            return set()  # Cycle detected
        
        visited.add(repo_name)
        dependencies = set()
        
        if repo_name in graph:
            for target_repo in graph[repo_name]:
                dependencies.add(target_repo)
                # Recursively get transitive dependencies
                transitive = self.get_transitive_dependencies(
                    target_repo,
                    graph,
                    visited.copy()
                )
                dependencies.update(transitive)
        
        return dependencies
