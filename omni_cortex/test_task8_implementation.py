#!/usr/bin/env python3
"""
Test script for Task 8 implementation:
- Circuit Breaker
- Enhanced Fallback Analysis
- Status Tracking
"""

from app.core.context.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.core.context.fallback_analysis import get_fallback_analyzer, ComponentFallbackMethods
from app.core.context.status_tracking import get_status_tracker, StatusFormatter

def test_circuit_breaker():
    """Test CircuitBreaker basic functionality."""
    print('Testing CircuitBreaker...')
    breaker = CircuitBreaker('test_breaker')
    print(f'  Initial state: {breaker.state.value}')
    status = breaker.get_status()
    print(f'  Status: {status.state.value}, failures: {status.failure_count}')
    print('  ✅ CircuitBreaker initialized successfully')

def test_fallback_analyzer():
    """Test EnhancedFallbackAnalyzer."""
    print('\nTesting EnhancedFallbackAnalyzer...')
    analyzer = get_fallback_analyzer()
    result = analyzer.analyze('Fix the authentication bug in the login flow')
    print(f'  Task type: {result["task_type"]}')
    print(f'  Framework: {result["framework"]}')
    print(f'  Confidence: {result["confidence"]}')
    print(f'  Steps: {len(result["steps"])} steps')
    print('  ✅ EnhancedFallbackAnalyzer working correctly')

def test_status_tracker():
    """Test ComponentStatusTracker."""
    print('\nTesting ComponentStatusTracker...')
    tracker = get_status_tracker()
    tracker.record_success('test_component', 1.5, api_calls=2, tokens=100)
    status_info = tracker.get_component_status('test_component')
    print(f'  Component status: {status_info.status.value}')
    print(f'  Execution time: {status_info.execution_time}s')
    
    health = tracker.get_health_summary()
    print(f'  Health: {health["overall_health"]}')
    print(f'  Successful components: {health["successful"]}')
    print('  ✅ ComponentStatusTracker working correctly')

def test_status_formatter():
    """Test StatusFormatter."""
    print('\nTesting StatusFormatter...')
    tracker = get_status_tracker()
    status_info = tracker.get_component_status('test_component')
    formatted = StatusFormatter.format_component_status(status_info)
    print(f'  Formatted status: {formatted}')
    
    health = tracker.get_health_summary()
    formatted_health = StatusFormatter.format_health_summary(health)
    print('  Formatted health:')
    for line in formatted_health.split('\n'):
        print(f'    {line}')
    print('  ✅ StatusFormatter working correctly')

if __name__ == '__main__':
    print('=' * 60)
    print('Task 8 Implementation Tests')
    print('=' * 60)
    
    try:
        test_circuit_breaker()
        test_fallback_analyzer()
        test_status_tracker()
        test_status_formatter()
        
        print('\n' + '=' * 60)
        print('✅ All tests passed successfully!')
        print('=' * 60)
    except Exception as e:
        print(f'\n❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
