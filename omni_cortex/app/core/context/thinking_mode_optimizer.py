"""
Thinking Mode Optimizer: Adaptive Gemini Thinking Mode Usage

This module provides intelligent thinking mode optimization for Gemini models:
- Complexity-based thinking mode activation
- Token budget consideration
- Quality metrics tracking
- Graceful fallback handling

Thinking mode is a Gemini feature that enables deeper reasoning at the cost of
additional tokens. This optimizer ensures thinking mode is used effectively.
"""

import structlog
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from ..settings import get_settings

logger = structlog.get_logger("context.thinking_mode_optimizer")


class ThinkingLevel(Enum):
    """Thinking mode levels supported by Gemini."""
    NONE = "none"          # No thinking mode
    LOW = "LOW"            # Basic reasoning
    MEDIUM = "MEDIUM"      # Moderate reasoning
    HIGH = "HIGH"          # Deep reasoning


@dataclass
class ThinkingModeMetrics:
    """Metrics for thinking mode usage and quality."""
    thinking_level: ThinkingLevel
    tokens_used: int
    reasoning_quality_score: float  # 0.0 to 1.0
    execution_time: float  # seconds
    complexity_detected: str  # low, medium, high, very_high
    budget_available: int
    budget_sufficient: bool
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ThinkingModeDecision:
    """Decision about thinking mode usage."""
    use_thinking_mode: bool
    thinking_level: ThinkingLevel
    reason: str
    estimated_token_cost: int
    complexity: str
    budget_available: int


class ThinkingModeOptimizer:
    """
    Optimizes Gemini thinking mode usage based on complexity and budget.
    
    Thinking mode provides deeper reasoning but costs more tokens. This optimizer
    ensures thinking mode is used when it provides the most value.
    
    Decision Logic:
    1. Check if thinking mode is enabled in settings
    2. Detect query complexity
    3. Check available token budget
    4. Decide on thinking level (NONE, LOW, MEDIUM, HIGH)
    5. Track quality metrics for continuous improvement
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._metrics_history: List[ThinkingModeMetrics] = []
        
        # Complexity thresholds for thinking mode activation
        self._complexity_thresholds = {
            "low": ThinkingLevel.NONE,
            "medium": ThinkingLevel.LOW,
            "high": ThinkingLevel.MEDIUM,
            "very_high": ThinkingLevel.HIGH,
        }
        
        # Token cost multipliers for thinking levels
        self._token_multipliers = {
            ThinkingLevel.NONE: 1.0,
            ThinkingLevel.LOW: 1.3,
            ThinkingLevel.MEDIUM: 1.6,
            ThinkingLevel.HIGH: 2.0,
        }
    
    def decide_thinking_mode(
        self,
        query: str,
        complexity: str,
        available_budget: int,
        task_type: Optional[str] = None,
    ) -> ThinkingModeDecision:
        """
        Decide whether to use thinking mode and at what level.
        
        Args:
            query: The user's query
            complexity: Detected complexity (low, medium, high, very_high)
            available_budget: Available token budget
            task_type: Optional task type for additional context
            
        Returns:
            ThinkingModeDecision with recommendation
        """
        # Check if adaptive thinking mode is enabled
        if not self.settings.enable_adaptive_thinking_mode:
            return ThinkingModeDecision(
                use_thinking_mode=False,
                thinking_level=ThinkingLevel.NONE,
                reason="Adaptive thinking mode disabled in settings",
                estimated_token_cost=0,
                complexity=complexity,
                budget_available=available_budget,
            )
        
        # Get complexity threshold from settings
        threshold_complexity = self.settings.thinking_mode_complexity_threshold
        
        # Check if complexity meets threshold
        complexity_order = ["low", "medium", "high", "very_high"]
        if complexity_order.index(complexity) < complexity_order.index(threshold_complexity):
            return ThinkingModeDecision(
                use_thinking_mode=False,
                thinking_level=ThinkingLevel.NONE,
                reason=f"Complexity '{complexity}' below threshold '{threshold_complexity}'",
                estimated_token_cost=0,
                complexity=complexity,
                budget_available=available_budget,
            )
        
        # Determine thinking level based on complexity
        recommended_level = self._complexity_thresholds.get(complexity, ThinkingLevel.NONE)
        
        # Estimate token cost
        base_tokens = len(query.split()) * 1.3  # Rough estimate
        estimated_cost = int(base_tokens * self._token_multipliers[recommended_level])
        
        # Check if budget is sufficient
        token_threshold = self.settings.thinking_mode_token_threshold
        if available_budget < token_threshold:
            # Budget too low, downgrade thinking level
            if available_budget < token_threshold * 0.5:
                # Very low budget, disable thinking mode
                return ThinkingModeDecision(
                    use_thinking_mode=False,
                    thinking_level=ThinkingLevel.NONE,
                    reason=f"Insufficient budget ({available_budget} < {token_threshold})",
                    estimated_token_cost=0,
                    complexity=complexity,
                    budget_available=available_budget,
                )
            else:
                # Moderate budget, use lower thinking level
                downgraded_level = self._downgrade_thinking_level(recommended_level)
                estimated_cost = int(base_tokens * self._token_multipliers[downgraded_level])
                return ThinkingModeDecision(
                    use_thinking_mode=downgraded_level != ThinkingLevel.NONE,
                    thinking_level=downgraded_level,
                    reason=f"Budget limited, downgraded from {recommended_level.value} to {downgraded_level.value}",
                    estimated_token_cost=estimated_cost,
                    complexity=complexity,
                    budget_available=available_budget,
                )
        
        # Check for task types that benefit most from thinking mode
        high_value_tasks = ["debug", "architect", "refactor", "optimize"]
        if task_type in high_value_tasks and recommended_level == ThinkingLevel.LOW:
            # Upgrade to MEDIUM for high-value tasks
            recommended_level = ThinkingLevel.MEDIUM
            estimated_cost = int(base_tokens * self._token_multipliers[recommended_level])
            reason = f"Upgraded to {recommended_level.value} for high-value task '{task_type}'"
        else:
            reason = f"Complexity '{complexity}' recommends {recommended_level.value}"
        
        return ThinkingModeDecision(
            use_thinking_mode=recommended_level != ThinkingLevel.NONE,
            thinking_level=recommended_level,
            reason=reason,
            estimated_token_cost=estimated_cost,
            complexity=complexity,
            budget_available=available_budget,
        )
    
    def _downgrade_thinking_level(self, level: ThinkingLevel) -> ThinkingLevel:
        """Downgrade thinking level by one step."""
        downgrade_map = {
            ThinkingLevel.HIGH: ThinkingLevel.MEDIUM,
            ThinkingLevel.MEDIUM: ThinkingLevel.LOW,
            ThinkingLevel.LOW: ThinkingLevel.NONE,
            ThinkingLevel.NONE: ThinkingLevel.NONE,
        }
        return downgrade_map.get(level, ThinkingLevel.NONE)
    
    def record_metrics(
        self,
        thinking_level: ThinkingLevel,
        tokens_used: int,
        execution_time: float,
        complexity: str,
        budget_available: int,
        reasoning_quality_score: float = 0.5,
        fallback_used: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> ThinkingModeMetrics:
        """
        Record metrics for a thinking mode execution.
        
        Args:
            thinking_level: The thinking level used
            tokens_used: Actual tokens consumed
            execution_time: Execution time in seconds
            complexity: Detected complexity
            budget_available: Available budget at decision time
            reasoning_quality_score: Quality score (0.0 to 1.0)
            fallback_used: Whether fallback was used
            fallback_reason: Reason for fallback if used
            
        Returns:
            ThinkingModeMetrics object
        """
        metrics = ThinkingModeMetrics(
            thinking_level=thinking_level,
            tokens_used=tokens_used,
            reasoning_quality_score=reasoning_quality_score,
            execution_time=execution_time,
            complexity_detected=complexity,
            budget_available=budget_available,
            budget_sufficient=tokens_used <= budget_available,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )
        
        # Store in history (keep last 100)
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 100:
            self._metrics_history.pop(0)
        
        logger.info(
            "thinking_mode_metrics_recorded",
            level=thinking_level.value,
            tokens=tokens_used,
            quality=reasoning_quality_score,
            complexity=complexity,
            fallback=fallback_used,
        )
        
        return metrics
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """
        Get quality statistics from metrics history.
        
        Returns:
            Dictionary with quality statistics
        """
        if not self._metrics_history:
            return {
                "total_executions": 0,
                "avg_quality_score": 0.0,
                "avg_tokens_used": 0,
                "fallback_rate": 0.0,
            }
        
        total = len(self._metrics_history)
        avg_quality = sum(m.reasoning_quality_score for m in self._metrics_history) / total
        avg_tokens = sum(m.tokens_used for m in self._metrics_history) / total
        fallback_count = sum(1 for m in self._metrics_history if m.fallback_used)
        fallback_rate = fallback_count / total
        
        # Quality by thinking level
        quality_by_level = {}
        for level in ThinkingLevel:
            level_metrics = [m for m in self._metrics_history if m.thinking_level == level]
            if level_metrics:
                quality_by_level[level.value] = {
                    "count": len(level_metrics),
                    "avg_quality": sum(m.reasoning_quality_score for m in level_metrics) / len(level_metrics),
                    "avg_tokens": sum(m.tokens_used for m in level_metrics) / len(level_metrics),
                }
        
        return {
            "total_executions": total,
            "avg_quality_score": avg_quality,
            "avg_tokens_used": avg_tokens,
            "fallback_rate": fallback_rate,
            "quality_by_level": quality_by_level,
        }
    
    def should_use_thinking_for_model(self, model_name: str) -> bool:
        """
        Check if a model supports thinking mode.
        
        Args:
            model_name: The model name
            
        Returns:
            True if model supports thinking mode
        """
        # Models with "thinking" in name (legacy experimental)
        if "thinking" in model_name.lower():
            return True
        
        # Gemini 3 models all support thinking mode
        if "gemini-3" in model_name.lower() or "gemini-2.0" in model_name.lower():
            return True
        
        return False


# Global singleton
_optimizer: Optional[ThinkingModeOptimizer] = None


def get_thinking_mode_optimizer() -> ThinkingModeOptimizer:
    """Get the global ThinkingModeOptimizer singleton."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ThinkingModeOptimizer()
    return _optimizer
