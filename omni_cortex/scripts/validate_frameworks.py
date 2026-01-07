#!/usr/bin/env python3
"""
Validate Framework Registry

Checks all framework definitions for completeness and correctness.
Run this before committing changes to the framework registry.

Usage:
    python scripts/validate_frameworks.py
    python -m scripts.validate_frameworks  # Alternative
"""

import sys
sys.path.insert(0, '.')

from app.frameworks.validation import FrameworkValidator

if __name__ == "__main__":
    print("🔍 Validating Omni Cortex Framework Registry...")
    print("=" * 60)
    
    validator = FrameworkValidator()
    issues = validator.validate_all()
    validator.print_report()
    
    summary = validator.get_summary()
    
    print("\n" + "=" * 60)
    if summary["error"] > 0:
        print("❌ Validation FAILED - fix errors before committing")
        sys.exit(1)
    elif summary["warning"] > 0:
        print("⚠️  Validation passed with warnings - consider addressing them")
        sys.exit(0)
    else:
        print("✅ Validation PASSED - all frameworks are properly defined")
        sys.exit(0)
