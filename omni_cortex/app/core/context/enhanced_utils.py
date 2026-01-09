"""
Enhanced Utilities for Context Gateway Enhancements

Utility functions and classes to support the enhanced context gateway functionality:
- Cache key generation and similarity hashing
- Workspace fingerprinting
- Token counting and budget calculations
- Quality scoring algorithms
- Circuit breaker utilities
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import asdict

from ..constants import CACHE, TOKEN_BUDGET, MULTI_REPO
from ..settings import get_settings


class CacheKeyGenerator:
    """Generates cache keys for context gateway operations."""
    
    @staticmethod
    def generate_query_similarity_hash(query: str) -> str:
        """
        Generate a similarity-based hash for query caching.
        
        Normalizes the query by:
        - Converting to lowercase
        - Removing common stop words
        - Extracting key terms
        - Creating a stable hash
        
        Args:
            query: The user's query string
            
        Returns:
            Hex string hash for cache key generation
        """
        # Normalize query
        normalized = query.lower().strip()
        
        # Remove common stop words that don't affect intent
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        # Extract meaningful words
        words = re.findall(r'\b\w+\b', normalized)
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Sort words to ensure consistent ordering
        meaningful_words.sort()
        
        # Create hash from meaningful words
        content = ' '.join(meaningful_words)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def generate_workspace_fingerprint(workspace_path: str) -> str:
        """
        Generate a fingerprint for workspace state.
        
        Creates a hash based on:
        - Directory structure (up to configured depth)
        - File modification times for key files
        - Git commit hash if available
        
        Args:
            workspace_path: Path to the workspace
            
        Returns:
            Hex string fingerprint of workspace state
        """
        if not os.path.exists(workspace_path):
            return "nonexistent"
        
        fingerprint_data = []
        
        # Add git commit hash if available
        git_dir = os.path.join(workspace_path, '.git')
        if os.path.exists(git_dir):
            try:
                head_file = os.path.join(git_dir, 'HEAD')
                if os.path.exists(head_file):
                    with open(head_file, 'r') as f:
                        head_content = f.read().strip()
                        if head_content.startswith('ref: '):
                            # Get commit hash from ref
                            ref_path = os.path.join(git_dir, head_content[5:])
                            if os.path.exists(ref_path):
                                with open(ref_path, 'r') as ref_f:
                                    commit_hash = ref_f.read().strip()
                                    fingerprint_data.append(f"git:{commit_hash}")
                        else:
                            # Direct commit hash
                            fingerprint_data.append(f"git:{head_content}")
            except (OSError, IOError):
                pass  # Git info not available
        
        # Add directory structure and key file timestamps
        try:
            for root, dirs, files in os.walk(workspace_path):
                # Limit depth to avoid excessive computation
                depth = root[len(workspace_path):].count(os.sep)
                if depth >= CACHE.WORKSPACE_FINGERPRINT_DEPTH:
                    dirs.clear()  # Don't descend further
                    continue
                
                # Skip common ignore directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                    '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build'
                }]
                
                # Add directory to fingerprint
                rel_root = os.path.relpath(root, workspace_path)
                if rel_root != '.':
                    fingerprint_data.append(f"dir:{rel_root}")
                
                # Add key files with timestamps
                for file in files:
                    if file.startswith('.'):
                        continue
                    
                    # Focus on key files that indicate project changes
                    key_extensions = {'.py', '.js', '.ts', '.go', '.rs', '.java', '.cpp', '.c'}
                    key_files = {'package.json', 'requirements.txt', 'Cargo.toml', 'go.mod', 'pom.xml'}
                    
                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file)
                    
                    if ext.lower() in key_extensions or file.lower() in key_files:
                        try:
                            mtime = os.path.getmtime(file_path)
                            rel_path = os.path.relpath(file_path, workspace_path)
                            fingerprint_data.append(f"file:{rel_path}:{int(mtime)}")
                        except OSError:
                            continue
        
        except OSError:
            # Fallback to basic timestamp
            try:
                mtime = os.path.getmtime(workspace_path)
                fingerprint_data.append(f"workspace_mtime:{int(mtime)}")
            except OSError:
                fingerprint_data.append(f"workspace_error:{int(time.time())}")
        
        # Create stable hash
        content = '|'.join(sorted(fingerprint_data))
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def generate_cache_key(
        query: str,
        workspace_path: Optional[str],
        cache_type: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a complete cache key.
        
        Args:
            query: The user's query
            workspace_path: Path to workspace (optional)
            cache_type: Type of cache entry
            additional_context: Additional context for key generation
            
        Returns:
            Complete cache key string
        """
        components = [
            f"type:{cache_type}",
            f"query:{CacheKeyGenerator.generate_query_similarity_hash(query)}"
        ]
        
        if workspace_path:
            workspace_fp = CacheKeyGenerator.generate_workspace_fingerprint(workspace_path)
            components.append(f"workspace:{workspace_fp}")
        
        if additional_context:
            # Sort keys for consistent ordering
            context_items = sorted(additional_context.items())
            context_str = json.dumps(context_items, sort_keys=True)
            context_hash = hashlib.sha256(context_str.encode('utf-8')).hexdigest()[:8]
            components.append(f"context:{context_hash}")
        
        return "|".join(components)


class TokenBudgetCalculator:
    """Calculates and manages token budgets for context preparation."""
    
    @staticmethod
    def calculate_base_budget(complexity: str, task_type: str) -> int:
        """
        Calculate base token budget based on complexity and task type.
        
        Args:
            complexity: Task complexity (low, medium, high, very_high)
            task_type: Type of task (debug, implement, etc.)
            
        Returns:
            Base token budget
        """
        # Base budgets by complexity
        base_budgets = {
            "low": TOKEN_BUDGET.LOW_COMPLEXITY_BUDGET,
            "medium": TOKEN_BUDGET.MEDIUM_COMPLEXITY_BUDGET,
            "high": TOKEN_BUDGET.HIGH_COMPLEXITY_BUDGET,
            "very_high": TOKEN_BUDGET.VERY_HIGH_COMPLEXITY_BUDGET,
        }
        
        base_budget = base_budgets.get(complexity, TOKEN_BUDGET.MEDIUM_COMPLEXITY_BUDGET)
        
        # Task type multipliers
        task_multipliers = {
            "debug": 1.2,      # Debugging needs more context
            "architect": 1.3,   # Architecture needs comprehensive view
            "refactor": 1.1,    # Refactoring needs good understanding
            "implement": 1.0,   # Standard implementation
            "explain": 0.8,     # Explanation can be more focused
            "test": 0.9,        # Testing needs targeted context
        }
        
        multiplier = task_multipliers.get(task_type, 1.0)
        return int(base_budget * multiplier)
    
    @staticmethod
    def allocate_budget(total_budget: int) -> Dict[str, int]:
        """
        Allocate total budget across components.
        
        Args:
            total_budget: Total available token budget
            
        Returns:
            Dictionary mapping component names to allocated tokens
        """
        return {
            "query_analysis": int(total_budget * TOKEN_BUDGET.QUERY_ANALYSIS_PERCENT),
            "file_discovery": int(total_budget * TOKEN_BUDGET.FILE_DISCOVERY_PERCENT),
            "documentation_search": int(total_budget * TOKEN_BUDGET.DOCUMENTATION_PERCENT),
            "code_search": int(total_budget * TOKEN_BUDGET.CODE_SEARCH_PERCENT),
            "context_assembly": int(total_budget * TOKEN_BUDGET.ASSEMBLY_PERCENT),
            "reserve": int(total_budget * TOKEN_BUDGET.RESERVE_PERCENT),
        }
    
    @staticmethod
    def estimate_token_count(text: str) -> int:
        """
        Estimate token count for text (rough approximation).
        
        Uses a simple heuristic: ~4 characters per token for English text.
        This is approximate but sufficient for budget planning.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Simple heuristic: 4 characters per token on average
        # This accounts for spaces, punctuation, and typical English word lengths
        return max(1, len(text) // 4)


class QualityScorer:
    """Calculates quality scores for context preparation results."""
    
    @staticmethod
    def calculate_file_relevance_quality(file_contexts: List[Any]) -> float:
        """
        Calculate quality score for file relevance results.
        
        Args:
            file_contexts: List of FileContext objects
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        if not file_contexts:
            return 0.0
        
        # Factors for quality assessment
        scores = []
        
        # 1. Relevance score distribution (prefer high relevance)
        relevance_scores = [f.relevance_score for f in file_contexts]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        scores.append(min(1.0, avg_relevance * 1.2))  # Boost high relevance
        
        # 2. Coverage (prefer diverse file types)
        extensions = set()
        for f in file_contexts:
            _, ext = os.path.splitext(f.path)
            if ext:
                extensions.add(ext.lower())
        
        # More diverse extensions = better coverage
        coverage_score = min(1.0, len(extensions) / 5.0)  # Normalize to 5 extensions
        scores.append(coverage_score)
        
        # 3. Summary quality (prefer non-empty summaries)
        summary_quality = sum(1 for f in file_contexts if f.summary and len(f.summary) > 10)
        summary_score = summary_quality / len(file_contexts)
        scores.append(summary_score)
        
        # 4. Key elements extraction (prefer files with identified elements)
        elements_quality = sum(1 for f in file_contexts if f.key_elements)
        elements_score = elements_quality / len(file_contexts)
        scores.append(elements_score)
        
        # Weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # Relevance is most important
        return sum(score * weight for score, weight in zip(scores, weights))
    
    @staticmethod
    def calculate_documentation_quality(doc_contexts: List[Any]) -> float:
        """
        Calculate quality score for documentation results.
        
        Args:
            doc_contexts: List of DocumentationContext objects
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        if not doc_contexts:
            return 0.0
        
        scores = []
        
        # 1. Relevance scores
        relevance_scores = [d.relevance_score for d in doc_contexts]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        scores.append(avg_relevance)
        
        # 2. Source diversity (prefer multiple sources)
        sources = set(getattr(d, 'source', '') for d in doc_contexts)
        source_diversity = min(1.0, len(sources) / 3.0)  # Normalize to 3 sources
        scores.append(source_diversity)
        
        # 3. Content quality (prefer substantial snippets)
        content_quality = sum(1 for d in doc_contexts if len(d.snippet) > 100)
        content_score = content_quality / len(doc_contexts)
        scores.append(content_score)
        
        # 4. Attribution quality (if available)
        attribution_score = 0.0
        if hasattr(doc_contexts[0], 'attribution'):
            attributed_count = sum(1 for d in doc_contexts if d.attribution is not None)
            attribution_score = attributed_count / len(doc_contexts)
        scores.append(attribution_score)
        
        # Weighted average
        weights = [0.4, 0.2, 0.2, 0.2]
        return sum(score * weight for score, weight in zip(scores, weights))
    
    @staticmethod
    def calculate_overall_quality(
        file_quality: float,
        doc_quality: float,
        code_search_quality: float,
        component_success_rate: float
    ) -> float:
        """
        Calculate overall context quality score.
        
        Args:
            file_quality: File discovery quality score
            doc_quality: Documentation search quality score
            code_search_quality: Code search quality score
            component_success_rate: Rate of successful component execution
            
        Returns:
            Overall quality score from 0.0 to 1.0
        """
        # Component quality scores
        component_scores = [file_quality, doc_quality, code_search_quality]
        avg_component_quality = sum(component_scores) / len(component_scores)
        
        # Combine with success rate
        # Success rate is critical - if components fail, quality suffers significantly
        overall_quality = avg_component_quality * (0.5 + 0.5 * component_success_rate)
        
        return min(1.0, max(0.0, overall_quality))


class CircuitBreakerUtils:
    """Utilities for circuit breaker functionality."""
    
    @staticmethod
    def calculate_backoff_delay(
        attempt: int,
        initial_delay: float = 1.0,
        max_delay: float = 300.0,
        multiplier: float = 2.0,
        jitter_factor: float = 0.1
    ) -> float:
        """
        Calculate exponential backoff delay with jitter.
        
        Args:
            attempt: Attempt number (0-based)
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            multiplier: Exponential multiplier
            jitter_factor: Jitter factor (0.0 to 1.0)
            
        Returns:
            Delay in seconds
        """
        import random
        
        # Exponential backoff
        delay = initial_delay * (multiplier ** attempt)
        delay = min(delay, max_delay)
        
        # Add jitter to prevent thundering herd
        jitter = delay * jitter_factor * (2 * random.random() - 1)  # ±jitter_factor
        delay += jitter
        
        return max(0.1, delay)  # Minimum 100ms delay
    
    @staticmethod
    def should_attempt_recovery(
        last_failure_time: datetime,
        recovery_timeout: int
    ) -> bool:
        """
        Check if circuit breaker should attempt recovery.
        
        Args:
            last_failure_time: Time of last failure
            recovery_timeout: Recovery timeout in seconds
            
        Returns:
            True if recovery should be attempted
        """
        if not last_failure_time:
            return True
        
        time_since_failure = datetime.now() - last_failure_time
        return time_since_failure.total_seconds() >= recovery_timeout


class MultiRepoUtils:
    """Utilities for multi-repository operations."""
    
    @staticmethod
    def detect_repositories(workspace_path: str) -> List[str]:
        """
        Detect git repositories in workspace.
        
        Args:
            workspace_path: Path to search for repositories
            
        Returns:
            List of repository root paths
        """
        repositories = []
        
        try:
            for root, dirs, files in os.walk(workspace_path):
                # Limit search depth
                depth = root[len(workspace_path):].count(os.sep)
                if depth >= MULTI_REPO.MAX_REPO_DEPTH:
                    dirs.clear()
                    continue
                
                # Check if this directory is a git repository
                if '.git' in dirs:
                    repositories.append(root)
                    # Don't search inside this repository for nested repos
                    dirs.clear()
                
                # Stop if we've found enough repositories
                if len(repositories) >= MULTI_REPO.MAX_REPOSITORIES:
                    break
        
        except OSError:
            pass  # Handle permission errors gracefully
        
        return repositories
    
    @staticmethod
    def extract_repo_name(repo_path: str) -> str:
        """
        Extract repository name from path.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Repository name
        """
        return os.path.basename(repo_path) or "root"
    
    @staticmethod
    def is_repository_accessible(repo_path: str) -> Tuple[bool, Optional[str]]:
        """
        Check if repository is accessible.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Tuple of (is_accessible, error_message)
        """
        try:
            # Check basic read access
            if not os.path.exists(repo_path):
                return False, "Repository path does not exist"
            
            if not os.access(repo_path, os.R_OK):
                return False, "No read permission for repository"
            
            # Check git repository validity
            git_dir = os.path.join(repo_path, '.git')
            if not os.path.exists(git_dir):
                return False, "Not a git repository"
            
            return True, None
        
        except Exception as e:
            return False, f"Access check failed: {str(e)}"