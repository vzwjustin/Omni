#!/usr/bin/env python3
"""
Simple test script for ContextGatewayMetrics
"""

import sys
sys.path.insert(0, '.')

from app.core.context.gateway_metrics import ContextGatewayMetrics, get_gateway_metrics

def test_basic_functionality():
    """Test basic metrics functionality."""
    print("Testing ContextGatewayMetrics...")
    
    # Create metrics instance
    metrics = ContextGatewayMetrics(retention_days=7)
    print("✓ Created ContextGatewayMetrics instance")
    
    # Test API call recording
    metrics.record_api_call(
        component="query_analyzer",
        model="gemini-flash-2.0",
        provider="google",
        tokens=500,
        duration=1.5,
        success=True,
        thinking_mode=True
    )
    print("✓ Recorded API call")
    
    # Test component performance recording
    metrics.record_component_performance(
        component="file_discoverer",
        duration=2.3,
        success=True,
        api_calls=2,
        tokens=1000,
        cache_hit=False,
        fallback_used=False
    )
    print("✓ Recorded component performance")
    
    # Test context quality recording
    quality_metrics = metrics.record_context_quality(
        quality_score=0.85,
        task_type="debug",
        component_scores={
            "query_analyzer": 0.9,
            "file_discoverer": 0.8,
            "doc_searcher": 0.85
        },
        relevance_scores=[0.9, 0.8, 0.7, 0.85],
        completeness_score=0.9
    )
    print(f"✓ Recorded context quality: {quality_metrics.overall_quality_score}")
    
    # Test session recording
    metrics.record_session(
        duration=5.5,
        success=True,
        complexity="medium"
    )
    print("✓ Recorded session")
    
    # Test summaries
    api_summary = metrics.get_api_call_summary()
    print(f"✓ API call summary: {api_summary['total_calls']} calls, {api_summary['total_tokens']} tokens")
    
    component_summary = metrics.get_component_performance_summary()
    print(f"✓ Component performance summary: {len(component_summary)} components")
    
    quality_summary = metrics.get_quality_summary()
    print(f"✓ Quality summary: avg={quality_summary['avg_quality']:.2f}")
    
    token_summary = metrics.get_token_usage_summary()
    print(f"✓ Token usage summary: {token_summary['total_tokens']} total tokens")
    
    session_summary = metrics.get_session_summary()
    print(f"✓ Session summary: {session_summary['total_sessions']} sessions")
    
    # Test comprehensive dashboard
    dashboard = metrics.get_comprehensive_dashboard()
    print(f"✓ Comprehensive dashboard generated")
    print(f"  - Overview: {dashboard['overview']['total_sessions']} sessions")
    print(f"  - Success rate: {dashboard['overview']['success_rate']}")
    
    # Test singleton
    singleton = get_gateway_metrics()
    print("✓ Got singleton instance")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
