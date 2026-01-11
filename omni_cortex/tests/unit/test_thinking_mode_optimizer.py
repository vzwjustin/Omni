"""
Unit tests for Thinking Mode Optimizer.

Tests the adaptive thinking mode optimization including:
- Complexity-based thinking mode activation
- Token budget consideration
- Quality metrics tracking
- Graceful fallback handling
"""

from unittest.mock import patch

import pytest

from app.core.context.thinking_mode_optimizer import (
    ThinkingLevel,
    ThinkingModeDecision,
    ThinkingModeMetrics,
    ThinkingModeOptimizer,
    get_thinking_mode_optimizer,
)


class TestThinkingModeOptimizer:
    """Test suite for ThinkingModeOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = ThinkingModeOptimizer()
        assert optimizer is not None
        assert optimizer.settings is not None
        assert len(optimizer._metrics_history) == 0

    def test_singleton_pattern(self):
        """Test that get_thinking_mode_optimizer returns singleton."""
        optimizer1 = get_thinking_mode_optimizer()
        optimizer2 = get_thinking_mode_optimizer()
        assert optimizer1 is optimizer2

    def test_decide_thinking_mode_low_complexity(self):
        """Test that low complexity doesn't trigger thinking mode."""
        optimizer = ThinkingModeOptimizer()

        decision = optimizer.decide_thinking_mode(
            query="Add a simple function",
            complexity="low",
            available_budget=50000,
            task_type="implement",
        )

        assert isinstance(decision, ThinkingModeDecision)
        assert not decision.use_thinking_mode
        assert decision.thinking_level == ThinkingLevel.NONE
        assert decision.complexity == "low"

    def test_decide_thinking_mode_high_complexity(self):
        """Test that high complexity triggers thinking mode with sufficient budget."""
        optimizer = ThinkingModeOptimizer()

        decision = optimizer.decide_thinking_mode(
            query="Debug complex authentication issue",
            complexity="high",
            available_budget=80000,
            task_type="debug",
        )

        assert isinstance(decision, ThinkingModeDecision)
        # Should use thinking mode for high complexity with good budget
        assert decision.thinking_level in [ThinkingLevel.MEDIUM, ThinkingLevel.HIGH]
        assert decision.complexity == "high"

    def test_decide_thinking_mode_budget_constraint(self):
        """Test that low budget prevents thinking mode."""
        optimizer = ThinkingModeOptimizer()

        decision = optimizer.decide_thinking_mode(
            query="Debug complex issue",
            complexity="high",
            available_budget=5000,  # Very low budget
            task_type="debug",
        )

        assert isinstance(decision, ThinkingModeDecision)
        # Should either disable or downgrade thinking mode due to budget
        assert decision.thinking_level in [ThinkingLevel.NONE, ThinkingLevel.LOW]

    def test_decide_thinking_mode_very_high_complexity(self):
        """Test that very high complexity uses HIGH thinking level."""
        optimizer = ThinkingModeOptimizer()

        decision = optimizer.decide_thinking_mode(
            query="Refactor entire microservices architecture",
            complexity="very_high",
            available_budget=100000,
            task_type="architect",
        )

        assert isinstance(decision, ThinkingModeDecision)
        assert decision.use_thinking_mode
        assert decision.thinking_level == ThinkingLevel.HIGH
        assert decision.complexity == "very_high"

    def test_decide_thinking_mode_disabled_in_settings(self):
        """Test that disabled setting prevents thinking mode."""
        optimizer = ThinkingModeOptimizer()

        # Mock settings to disable adaptive thinking mode
        with patch.object(optimizer.settings, 'enable_adaptive_thinking_mode', False):
            decision = optimizer.decide_thinking_mode(
                query="Any query",
                complexity="high",
                available_budget=80000,
            )

            assert not decision.use_thinking_mode
            assert decision.thinking_level == ThinkingLevel.NONE
            assert "disabled" in decision.reason.lower()

    def test_record_metrics(self):
        """Test metrics recording."""
        optimizer = ThinkingModeOptimizer()

        metrics = optimizer.record_metrics(
            thinking_level=ThinkingLevel.HIGH,
            tokens_used=5000,
            execution_time=2.5,
            complexity="high",
            budget_available=80000,
            reasoning_quality_score=0.85,
            fallback_used=False,
        )

        assert isinstance(metrics, ThinkingModeMetrics)
        assert metrics.thinking_level == ThinkingLevel.HIGH
        assert metrics.tokens_used == 5000
        assert metrics.execution_time == 2.5
        assert metrics.complexity_detected == "high"
        assert metrics.reasoning_quality_score == 0.85
        assert not metrics.fallback_used
        assert len(optimizer._metrics_history) == 1

    def test_record_metrics_with_fallback(self):
        """Test metrics recording with fallback."""
        optimizer = ThinkingModeOptimizer()

        metrics = optimizer.record_metrics(
            thinking_level=ThinkingLevel.MEDIUM,
            tokens_used=0,
            execution_time=0,
            complexity="medium",
            budget_available=50000,
            reasoning_quality_score=0.0,
            fallback_used=True,
            fallback_reason="Thinking mode unavailable",
        )

        assert metrics.fallback_used
        assert metrics.fallback_reason == "Thinking mode unavailable"

    def test_get_quality_statistics_empty(self):
        """Test quality statistics with no history."""
        optimizer = ThinkingModeOptimizer()

        stats = optimizer.get_quality_statistics()

        assert stats["total_executions"] == 0
        assert stats["avg_quality_score"] == 0.0
        assert stats["avg_tokens_used"] == 0
        assert stats["fallback_rate"] == 0.0

    def test_get_quality_statistics_with_data(self):
        """Test quality statistics with recorded metrics."""
        optimizer = ThinkingModeOptimizer()

        # Record multiple metrics
        optimizer.record_metrics(
            thinking_level=ThinkingLevel.HIGH,
            tokens_used=5000,
            execution_time=2.5,
            complexity="high",
            budget_available=80000,
            reasoning_quality_score=0.85,
            fallback_used=False,
        )

        optimizer.record_metrics(
            thinking_level=ThinkingLevel.MEDIUM,
            tokens_used=3000,
            execution_time=1.8,
            complexity="medium",
            budget_available=50000,
            reasoning_quality_score=0.75,
            fallback_used=False,
        )

        optimizer.record_metrics(
            thinking_level=ThinkingLevel.LOW,
            tokens_used=0,
            execution_time=0,
            complexity="low",
            budget_available=20000,
            reasoning_quality_score=0.0,
            fallback_used=True,
        )

        stats = optimizer.get_quality_statistics()

        assert stats["total_executions"] == 3
        assert 0.0 < stats["avg_quality_score"] < 1.0
        assert stats["avg_tokens_used"] > 0
        assert stats["fallback_rate"] == pytest.approx(1/3)
        assert "quality_by_level" in stats

    def test_should_use_thinking_for_model_gemini_3(self):
        """Test model support detection for Gemini 3."""
        optimizer = ThinkingModeOptimizer()

        assert optimizer.should_use_thinking_for_model("gemini-3-flash-preview")
        assert optimizer.should_use_thinking_for_model("gemini-3-pro")

    def test_should_use_thinking_for_model_gemini_2(self):
        """Test model support detection for Gemini 2.0."""
        optimizer = ThinkingModeOptimizer()

        assert optimizer.should_use_thinking_for_model("gemini-2.0-flash")
        assert optimizer.should_use_thinking_for_model("gemini-2.0-pro")

    def test_should_use_thinking_for_model_thinking_exp(self):
        """Test model support detection for thinking experimental models."""
        optimizer = ThinkingModeOptimizer()

        assert optimizer.should_use_thinking_for_model("gemini-thinking-exp")

    def test_should_use_thinking_for_model_old_models(self):
        """Test model support detection for older models."""
        optimizer = ThinkingModeOptimizer()

        assert not optimizer.should_use_thinking_for_model("gemini-1.5-pro")
        assert not optimizer.should_use_thinking_for_model("gemini-1.0-pro")

    def test_downgrade_thinking_level(self):
        """Test thinking level downgrade logic."""
        optimizer = ThinkingModeOptimizer()

        assert optimizer._downgrade_thinking_level(ThinkingLevel.HIGH) == ThinkingLevel.MEDIUM
        assert optimizer._downgrade_thinking_level(ThinkingLevel.MEDIUM) == ThinkingLevel.LOW
        assert optimizer._downgrade_thinking_level(ThinkingLevel.LOW) == ThinkingLevel.NONE
        assert optimizer._downgrade_thinking_level(ThinkingLevel.NONE) == ThinkingLevel.NONE

    def test_metrics_history_limit(self):
        """Test that metrics history is limited to 100 entries."""
        optimizer = ThinkingModeOptimizer()

        # Record 150 metrics
        for _ in range(150):
            optimizer.record_metrics(
                thinking_level=ThinkingLevel.LOW,
                tokens_used=1000,
                execution_time=1.0,
                complexity="low",
                budget_available=50000,
                reasoning_quality_score=0.5,
            )

        # Should only keep last 100
        assert len(optimizer._metrics_history) == 100

    def test_high_value_task_upgrade(self):
        """Test that high-value tasks get upgraded thinking level."""
        optimizer = ThinkingModeOptimizer()

        # Debug task with medium complexity should get upgraded
        decision = optimizer.decide_thinking_mode(
            query="Debug authentication issue",
            complexity="medium",
            available_budget=80000,
            task_type="debug",
        )

        # Should use at least LOW thinking mode for debug tasks
        assert decision.thinking_level in [ThinkingLevel.LOW, ThinkingLevel.MEDIUM]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
