#!/usr/bin/env python3
"""
Quick test for thinking mode optimization implementation.
"""

import asyncio
from app.core.context.thinking_mode_optimizer import (
    ThinkingModeOptimizer,
    ThinkingLevel,
    get_thinking_mode_optimizer,
)


def test_thinking_mode_decision():
    """Test thinking mode decision logic."""
    optimizer = get_thinking_mode_optimizer()
    
    # Test 1: Low complexity, should not use thinking mode
    decision = optimizer.decide_thinking_mode(
        query="Add a simple function",
        complexity="low",
        available_budget=50000,
        task_type="implement",
    )
    print(f"Test 1 - Low complexity:")
    print(f"  Use thinking: {decision.use_thinking_mode}")
    print(f"  Level: {decision.thinking_level.value}")
    print(f"  Reason: {decision.reason}")
    print()
    
    # Test 2: High complexity with good budget, should use thinking mode
    decision = optimizer.decide_thinking_mode(
        query="Debug complex authentication issue",
        complexity="high",
        available_budget=80000,
        task_type="debug",
    )
    print(f"Test 2 - High complexity with budget:")
    print(f"  Use thinking: {decision.use_thinking_mode}")
    print(f"  Level: {decision.thinking_level.value}")
    print(f"  Reason: {decision.reason}")
    print()
    
    # Test 3: High complexity with low budget, should downgrade
    decision = optimizer.decide_thinking_mode(
        query="Debug complex authentication issue",
        complexity="high",
        available_budget=5000,
        task_type="debug",
    )
    print(f"Test 3 - High complexity with low budget:")
    print(f"  Use thinking: {decision.use_thinking_mode}")
    print(f"  Level: {decision.thinking_level.value}")
    print(f"  Reason: {decision.reason}")
    print()
    
    # Test 4: Very high complexity
    decision = optimizer.decide_thinking_mode(
        query="Refactor entire microservices architecture",
        complexity="very_high",
        available_budget=100000,
        task_type="architect",
    )
    print(f"Test 4 - Very high complexity:")
    print(f"  Use thinking: {decision.use_thinking_mode}")
    print(f"  Level: {decision.thinking_level.value}")
    print(f"  Reason: {decision.reason}")
    print()


def test_metrics_recording():
    """Test metrics recording."""
    optimizer = get_thinking_mode_optimizer()
    
    # Record some metrics
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
    
    # Get statistics
    stats = optimizer.get_quality_statistics()
    print("Metrics Statistics:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Avg quality score: {stats['avg_quality_score']:.2f}")
    print(f"  Avg tokens used: {stats['avg_tokens_used']:.0f}")
    print(f"  Fallback rate: {stats['fallback_rate']:.2%}")
    print()


def test_model_support():
    """Test model support detection."""
    optimizer = get_thinking_mode_optimizer()
    
    models = [
        "gemini-3-flash-preview",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-thinking-exp",
    ]
    
    print("Model Support:")
    for model in models:
        supports = optimizer.should_use_thinking_for_model(model)
        print(f"  {model}: {'✓' if supports else '✗'}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Thinking Mode Optimization")
    print("=" * 60)
    print()
    
    test_thinking_mode_decision()
    test_metrics_recording()
    test_model_support()
    
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
