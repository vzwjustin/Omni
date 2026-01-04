"""
Threat Modeling / Red-Teaming: Security-Focused Code Review

AI assumes attacker mindset to find vulnerabilities in code.
Identifies security issues like SQLi, XSS, authentication bypasses, etc.
"""

import asyncio
from typing import Optional, List, Dict
from ...state import GraphState
from ..common import (
    quiet_star,
    call_fast_synthesizer,
    call_deep_reasoner,
    add_reasoning_step,
    format_code_context,
    extract_code_blocks,
)


@quiet_star
async def red_team_node(state: GraphState) -> GraphState:
    """
    Red-Teaming: Adversarial Security Analysis.

    Process:
    1. RECONNAISSANCE: Understand the code's attack surface
    2. THREAT_MODEL: Identify potential attack vectors
    3. EXPLOIT: Find specific vulnerabilities
    4. ASSESS: Rate severity and impact
    5. PATCH: Provide secure fixes

    Best for: Security audits, penetration testing, vulnerability scanning
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # =========================================================================
    # Phase 1: RECONNAISSANCE - Understand Attack Surface
    # =========================================================================

    recon_prompt = f"""Perform reconnaissance to understand the attack surface.

CODE TO AUDIT:
{code_context}

AUDIT REQUEST: {query}

As a security researcher, identify:
1. **Entry points**: Where does external data enter? (user input, APIs, files, etc.)
2. **Data flow**: How does data move through the system?
3. **Sensitive operations**: Authentication, authorization, database queries, file I/O
4. **Trust boundaries**: Where does untrusted data become trusted?
5. **External dependencies**: Third-party libraries, APIs, services

Map the attack surface."""

    recon_response, _ = await call_deep_reasoner(
        prompt=recon_prompt,
        state=state,
        system="Analyze code from a security researcher perspective.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="red_team",
        thought="Mapped attack surface and entry points",
        action="reconnaissance",
        observation=recon_response[:200]
    )

    # =========================================================================
    # Phase 2: THREAT MODEL - Identify Attack Vectors
    # =========================================================================

    threat_model_prompt = f"""Build a threat model identifying potential attack vectors.

ATTACK SURFACE:
{recon_response}

CODE:
{code_context}

For each entry point and sensitive operation, identify threats using STRIDE:
- **Spoofing**: Can identity be faked?
- **Tampering**: Can data be modified?
- **Repudiation**: Can actions be denied?
- **Information Disclosure**: Can sensitive data leak?
- **Denial of Service**: Can availability be disrupted?
- **Elevation of Privilege**: Can access be escalated?

Also consider OWASP Top 10:
- Injection (SQL, Command, XSS, etc.)
- Broken Authentication
- Sensitive Data Exposure
- XML External Entities (XXE)
- Broken Access Control
- Security Misconfiguration
- Cross-Site Scripting (XSS)
- Insecure Deserialization
- Using Components with Known Vulnerabilities
- Insufficient Logging & Monitoring

List specific attack vectors for this code."""

    threat_model_response, _ = await call_deep_reasoner(
        prompt=threat_model_prompt,
        state=state,
        system="Build comprehensive threat models.",
        temperature=0.7
    )

    add_reasoning_step(
        state=state,
        framework="red_team",
        thought="Built threat model with attack vectors",
        action="threat_modeling",
        observation=threat_model_response[:200]
    )

    # =========================================================================
    # Phase 3: EXPLOIT - Find Specific Vulnerabilities
    # =========================================================================

    exploit_prompt = f"""Identify specific vulnerabilities and potential exploits.

THREAT MODEL:
{threat_model_response}

CODE:
{code_context}

For each identified threat vector, find concrete vulnerabilities:

Format each as:
**Vulnerability #N: [Type]**
- Location: [file:line or code snippet]
- Description: [what's vulnerable]
- Exploit scenario: [how an attacker could exploit it]
- Example payload: [malicious input that would work]
- Impact: [what damage could be done]

Be specific with code locations and exploit details."""

    exploit_response, _ = await call_deep_reasoner(
        prompt=exploit_prompt,
        state=state,
        system="Identify specific exploitable vulnerabilities.",
        temperature=0.6
    )

    add_reasoning_step(
        state=state,
        framework="red_team",
        thought="Identified specific exploitable vulnerabilities",
        action="vulnerability_discovery",
        observation=exploit_response[:200]
    )

    # =========================================================================
    # Phase 4: ASSESS - Risk Assessment
    # =========================================================================

    assess_prompt = f"""Assess the severity and risk of each vulnerability.

VULNERABILITIES FOUND:
{exploit_response}

For each vulnerability, provide CVSS-style assessment:
- **Severity**: Critical / High / Medium / Low
- **Exploitability**: How easy to exploit? (1-10)
- **Impact**: Confidentiality, Integrity, Availability impact (High/Medium/Low)
- **Attack Vector**: Network / Adjacent / Local / Physical
- **Attack Complexity**: Low / High
- **Privileges Required**: None / Low / High
- **User Interaction**: None / Required

Rank vulnerabilities by risk (severity × likelihood)."""

    assess_response, _ = await call_fast_synthesizer(
        prompt=assess_prompt,
        state=state,
        max_tokens=1200
    )

    add_reasoning_step(
        state=state,
        framework="red_team",
        thought="Assessed vulnerability severity and risk",
        action="risk_assessment",
        observation=assess_response[:200]
    )

    # =========================================================================
    # Phase 5: PATCH - Provide Secure Fixes
    # =========================================================================

    patch_prompt = f"""Provide secure fixes for all vulnerabilities.

VULNERABILITIES:
{exploit_response}

ORIGINAL CODE:
{code_context}

For each vulnerability, provide:
1. **Secure Code**: Fixed version with vulnerability patched
2. **Mitigation**: What the fix does to prevent the attack
3. **Best Practice**: General security principle applied

Provide complete patched code if possible:

```python
# Patched secure code
```

Include security comments explaining the fixes."""

    patch_response, _ = await call_deep_reasoner(
        prompt=patch_prompt,
        state=state,
        system="Generate secure code fixes for vulnerabilities.",
        temperature=0.4
    )

    patched_code_blocks = extract_code_blocks(patch_response)
    patched_code = patched_code_blocks[0] if patched_code_blocks else ""

    add_reasoning_step(
        state=state,
        framework="red_team",
        thought="Generated secure patches for vulnerabilities",
        action="patching",
        observation=f"Created secure version with {patched_code.count('security')} security improvements"
    )

    # =========================================================================
    # Final Report
    # =========================================================================

    # Count vulnerabilities
    vuln_count = exploit_response.lower().count("vulnerability #")

    final_answer = f"""# 🔴 Red Team Security Assessment

## Code Under Review
{query}

## Executive Summary
Found **{vuln_count} potential vulnerabilities** requiring attention.

---

## 1. Attack Surface Analysis
{recon_response}

---

## 2. Threat Model
{threat_model_response}

---

## 3. Vulnerabilities Discovered
{exploit_response}

---

## 4. Risk Assessment
{assess_response}

---

## 5. Secure Patches

{patch_response}

### Patched Code
```python
{patched_code if patched_code else "See specific fixes above"}
```

---

## Recommendations
1. Apply all patches immediately for Critical/High severity issues
2. Implement input validation and sanitization at all entry points
3. Use parameterized queries to prevent injection attacks
4. Apply principle of least privilege
5. Implement comprehensive logging and monitoring
6. Regular security audits and penetration testing

**Security is a continuous process. Re-assess after changes.**
"""

    # Store security findings
    state["working_memory"]["red_team_vulns_found"] = vuln_count
    state["working_memory"]["red_team_threat_model"] = threat_model_response
    state["working_memory"]["red_team_patches"] = patched_code

    # Update final state
    state["final_answer"] = final_answer
    if patched_code:
        state["final_code"] = patched_code
    state["confidence_score"] = 0.88

    return state
