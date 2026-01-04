"""
Chain of Verification (CoVe): Draft-Verify-Patch

Generates code, then systematically verifies for
security, bugs, and best practices, then patches.
"""

from typing import Optional
from ...state import GraphState
from ..common import (
    quiet_star,
    call_deep_reasoner,
    call_fast_synthesizer,
    add_reasoning_step,
    format_code_context,
    run_tool,
)


# Verification categories
VERIFICATION_CHECKS = [
    {
        "category": "security",
        "checks": [
            "SQL injection vulnerabilities",
            "XSS (cross-site scripting) risks",
            "CSRF protection",
            "Input validation/sanitization",
            "Authentication/authorization issues",
            "Sensitive data exposure",
            "Insecure dependencies"
        ]
    },
    {
        "category": "bugs",
        "checks": [
            "Null/undefined reference possibilities",
            "Off-by-one errors",
            "Race conditions",
            "Memory leaks",
            "Infinite loops",
            "Unhandled exceptions",
            "Type errors"
        ]
    },
    {
        "category": "best_practices",
        "checks": [
            "Code organization and readability",
            "Error handling completeness",
            "Logging and observability",
            "Performance considerations",
            "Testability",
            "Documentation",
            "DRY violations"
        ]
    }
]


@quiet_star
async def chain_of_verification_node(state: GraphState) -> GraphState:
    """
    Chain of Verification: Draft-Verify-Patch.
    
    Process:
    1. DRAFT: Generate initial code solution
    2. VERIFY: Run systematic verification checks
    3. IDENTIFY: List all issues found
    4. PATCH: Fix each issue systematically
    5. VALIDATE: Confirm fixes don't introduce new issues
    
    Best for: Security review, code validation, quality assurance
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    # Optional: retrieve security and best practices context
    try:
        # Use enhanced search for security best practices
        doc_context = await run_tool("search_by_category", {"query": f"security best practices {query}", "category": "documentation", "k": 3}, state)
    except Exception as e:
        # Silently continue - doc context is optional enhancement, not critical
        doc_context = ""
    
    # =========================================================================
    # Phase 1: DRAFT Initial Solution
    # =========================================================================
    
    draft_prompt = f"""Generate an initial code solution.

TASK: {query}

CONTEXT:
{code_context}

Create a working implementation. Focus on functionality first.
We will verify and improve it in subsequent steps.

```
[Your initial code]
```

Include comments explaining key decisions."""

    draft_response, _ = await call_deep_reasoner(
        prompt=draft_prompt,
        state=state,
        system="Generate functional code. Focus on correctness.",
        temperature=0.6
    )
    
    # Extract draft code
    import re
    code_pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(code_pattern, draft_response, re.DOTALL)
    draft_code = match.group(1).strip() if match else draft_response

    draft_lines = len(draft_code.split('\n'))
    add_reasoning_step(
        state=state,
        framework="chain_of_verification",
        thought="Generated initial draft code",
        action="draft",
        observation=f"Draft has {draft_lines} lines"
    )
    
    # =========================================================================
    # Phase 2: VERIFY with Systematic Checks
    # =========================================================================
    
    all_issues: list[dict] = []
    
    for check_category in VERIFICATION_CHECKS:
        category = check_category["category"]
        checks = check_category["checks"]

        checks_formatted = "\n".join(f"- {check}" for check in checks)
        verify_prompt = f"""Verify this code for {category.upper()} issues.

CODE TO VERIFY:
```
{draft_code}
```

ORIGINAL TASK: {query}

DOC CONTEXT:
{doc_context}

Check for these specific issues:
{checks_formatted}

For EACH issue found:
```
ISSUE: [Issue name]
SEVERITY: [HIGH/MEDIUM/LOW]
LOCATION: [Where in the code]
DESCRIPTION: [What's wrong]
FIX: [How to fix it]
```

If no issues found for a check, explicitly state "No issues found for: [check]"."""

        verify_response, _ = await call_fast_synthesizer(
            prompt=verify_prompt,
            state=state,
            max_tokens=1500
        )
        
        # Parse issues
        issues = _parse_issues(verify_response, category)
        all_issues.extend(issues)
        
        add_reasoning_step(
            state=state,
            framework="chain_of_verification",
            thought=f"Verified {category}: Found {len(issues)} issues",
            action=f"verify_{category}",
            observation=f"{len(issues)} {category} issues identified"
        )
    
    # Store verification results
    # Count total checks performed across all categories
    total_checks = sum(len(cat["checks"]) for cat in VERIFICATION_CHECKS)
    state["verification_checks"] = all_issues
    state["passed_checks"] = max(0, total_checks - len(all_issues))
    state["failed_checks"] = len(all_issues)
    
    # =========================================================================
    # Phase 3: PATCH Issues
    # =========================================================================
    
    if all_issues:
        # Group by severity
        high_severity = [i for i in all_issues if i["severity"] == "HIGH"]
        medium_severity = [i for i in all_issues if i["severity"] == "MEDIUM"]
        low_severity = [i for i in all_issues if i["severity"] == "LOW"]
        
        issues_summary = "ISSUES TO FIX:\n\n"
        
        if high_severity:
            issues_summary += "HIGH SEVERITY:\n"
            for i in high_severity:
                issues_summary += f"- {i['issue']}: {i['description']}\n  Fix: {i['fix']}\n"
        
        if medium_severity:
            issues_summary += "\nMEDIUM SEVERITY:\n"
            for i in medium_severity:
                issues_summary += f"- {i['issue']}: {i['description']}\n  Fix: {i['fix']}\n"
        
        if low_severity:
            issues_summary += "\nLOW SEVERITY:\n"
            for i in low_severity:
                issues_summary += f"- {i['issue']}: {i['description']}\n  Fix: {i['fix']}\n"
        
        patch_prompt = f"""Patch all identified issues in the code.

ORIGINAL CODE:
```
{draft_code}
```

{issues_summary}

Provide the FULLY PATCHED code that addresses ALL issues.
Add comments marking where fixes were applied.

```
[Patched code]
```

After the code, briefly explain each patch made."""

        patch_response, _ = await call_deep_reasoner(
            prompt=patch_prompt,
            state=state,
            system="Fix all issues systematically. Maintain functionality.",
            temperature=0.4,
            max_tokens=4000
        )
        
        match = re.search(code_pattern, patch_response, re.DOTALL)
        patched_code = match.group(1).strip() if match else draft_code
        
        add_reasoning_step(
            state=state,
            framework="chain_of_verification",
            thought=f"Patched {len(all_issues)} issues",
            action="patch",
            observation=f"Fixed {len(high_severity)} high, {len(medium_severity)} medium, {len(low_severity)} low severity issues"
        )
    else:
        patched_code = draft_code
        patch_response = "No issues found. Original code is clean."
        
        add_reasoning_step(
            state=state,
            framework="chain_of_verification",
            thought="No issues found - code passed all checks",
            action="patch_skip",
            observation="Draft code passed verification"
        )
    
    # =========================================================================
    # Phase 4: VALIDATE Patches
    # =========================================================================
    
    validate_prompt = f"""Validate the patched code.

PATCHED CODE:
```
{patched_code}
```

ORIGINAL TASK: {query}

Confirm:
1. All original issues are fixed
2. No new issues were introduced
3. Original functionality is preserved

**VALIDATION RESULT**: [PASS/FAIL]

**REMAINING CONCERNS**: [Any remaining issues or none]

**FINAL RECOMMENDATION**: [Ready for use / Needs more work]"""

    validate_response, _ = await call_fast_synthesizer(
        prompt=validate_prompt,
        state=state,
        max_tokens=600
    )
    
    validation_passed = "PASS" in validate_response.upper() and "FAIL" not in validate_response.upper()
    
    add_reasoning_step(
        state=state,
        framework="chain_of_verification",
        thought="Validated patched code",
        action="validate",
        observation=f"Validation: {'PASSED' if validation_passed else 'NEEDS ATTENTION'}",
        score=0.9 if validation_passed else 0.6
    )
    
    # Generate final summary
    summary = f"""## Chain of Verification Results

### Original Task
{query}

### Verification Summary
- Total checks: {sum(len(cat['checks']) for cat in VERIFICATION_CHECKS)}
- Issues found: {len(all_issues)}
- High severity: {len([i for i in all_issues if i['severity'] == 'HIGH'])}
- Medium severity: {len([i for i in all_issues if i['severity'] == 'MEDIUM'])}
- Low severity: {len([i for i in all_issues if i['severity'] == 'LOW'])}

### Patched Code
```
{patched_code}
```

### Validation
{validate_response}
"""
    
    # Update final state
    state["final_answer"] = summary
    state["final_code"] = patched_code
    state["confidence_score"] = 0.9 if validation_passed and len(all_issues) == 0 else (0.8 if validation_passed else 0.6)
    
    return state


def _parse_issues(response: str, category: str) -> list[dict]:
    """Parse identified issues from verification response."""
    import re
    issues = []
    
    # Look for issue blocks
    issue_pattern = r'ISSUE:\s*([^\n]+)\nSEVERITY:\s*([^\n]+)\nLOCATION:\s*([^\n]+)\nDESCRIPTION:\s*([^\n]+)\nFIX:\s*([^\n]+)'
    matches = re.findall(issue_pattern, response, re.IGNORECASE)
    
    for match in matches:
        issue_name, severity, location, description, fix = match
        severity = severity.strip().upper()
        if severity not in ["HIGH", "MEDIUM", "LOW"]:
            severity = "MEDIUM"
        
        issues.append({
            "category": category,
            "issue": issue_name.strip(),
            "severity": severity,
            "location": location.strip(),
            "description": description.strip(),
            "fix": fix.strip()
        })
    
    return issues
