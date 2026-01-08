"""
Complexity Analysis Service

Estimates task complexity based on query, code context, and file lists.
Used to determine routing strategy and model selection.
"""

import re
from typing import Optional, List


class ComplexityEstimator:
    """Service for estimating task complexity."""

    def estimate(
        self,
        query: str,
        code_snippet: Optional[str] = None,
        file_list: Optional[List[str]] = None
    ) -> float:
        """
        Estimate task complexity on 0-1 scale.
        
        Args:
            query: The user's query
            code_snippet: Optional code context
            file_list: Optional list of files involved
            
        Returns:
            Float between 0.0 and 1.0 representing complexity
        """
        complexity = 0.3

        # Factor 1: Query length
        word_count = len(query.split())
        if word_count > 50:
            complexity += 0.15
        if word_count > 100:
            complexity += 0.1

        # Factor 2: Code context size
        if code_snippet:
            lines = code_snippet.count('\n') + 1
            if lines > 50:
                complexity += 0.1
            if lines > 200:
                complexity += 0.15

        # Factor 3: Scope (file count)
        if file_list and len(file_list) > 5:
            complexity += 0.1

        # Factor 4: Keyword indicators
        indicators = [
            r"complex", r"difficult", r"tricky", 
            r"interdependent", r"legacy", r"distributed",
            r"architecture", r"refactor", r"rewrite"
        ]
        
        for ind in indicators:
            if re.search(ind, query, re.IGNORECASE):
                complexity += 0.05

        return min(complexity, 1.0)