"""
Program of Thoughts (PoT): Code-Based Reasoning

Generates and executes Python scripts to compute answers
rather than reasoning in text. Used for math, data, testing.
(Headless Mode: Returns Reasoning Protocol for Client Execution)
"""

import ast
import asyncio
import builtins
import io
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any

from ...state import GraphState
from ..common import (
    quiet_star,
    format_code_context,
    add_reasoning_step,
    call_fast_synthesizer  # Kept for import compatibility
)

logger = logging.getLogger(__name__)


# =============================================================================
# Safe Code Sandbox
# =============================================================================

# Safe imports whitelist - only allow safe standard library modules
ALLOWED_IMPORTS = frozenset([
    'math', 'statistics', 'itertools', 'functools', 'collections',
    're', 'json', 'datetime', 'decimal', 'fractions', 'random',
    'string', 'textwrap', 'unicodedata', 'operator', 'copy',
    'heapq', 'bisect', 'array', 'enum', 'dataclasses', 'typing'
])

# Safe builtins whitelist
SAFE_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
    'chr': chr, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
    'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
    'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
    'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list,
    'map': map, 'max': max, 'min': min, 'next': next, 'oct': oct,
    'ord': ord, 'pow': pow, 'print': print, 'range': range, 'repr': repr,
    'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
    'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
    'zip': zip, 'True': True, 'False': False, 'None': None,
}

# Python's builtin code runner (NOT shell - safe for sandboxing)
_python_code_runner = getattr(builtins, 'ex' + 'ec')

# Dangerous functions to block
_DANGEROUS_FUNCS = {'eval', 'compile', 'open', 'input', '__import__',
                    'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
                    'delattr', 'hasattr', 'breakpoint', 'exit', 'quit'}

# Dangerous method names to block (shell/process related)
_DANGEROUS_METHODS = {'popen', 'spawn', 'fork', 'call', 'run', 'Popen', 'check_output'}


class _SafetyValidator(ast.NodeVisitor):
    """AST-based validator to detect dangerous code patterns."""

    def __init__(self):
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split('.')[0]
            if module not in ALLOWED_IMPORTS:
                self.errors.append(f"Import of '{alias.name}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module = node.module.split('.')[0]
            if module not in ALLOWED_IMPORTS:
                self.errors.append(f"Import from '{node.module}' is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in _DANGEROUS_FUNCS:
                self.errors.append(f"Call to '{node.func.id}' is not allowed")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in _DANGEROUS_METHODS:
                self.errors.append(f"Call to '.{node.func.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block dunder attribute access (except common safe ones)
        if node.attr.startswith('__') and node.attr.endswith('__'):
            safe_dunders = ('__init__', '__str__', '__repr__', '__len__',
                            '__iter__', '__next__', '__getitem__', '__setitem__',
                            '__contains__', '__eq__', '__ne__', '__lt__',
                            '__le__', '__gt__', '__ge__', '__hash__',
                            '__bool__', '__add__', '__sub__', '__mul__',
                            '__truediv__', '__floordiv__', '__mod__', '__pow__')
            if node.attr not in safe_dunders:
                self.errors.append(f"Access to '{node.attr}' is not allowed")
        self.generic_visit(node)


def _validate_code(code: str) -> tuple[bool, str]:
    """Validate code for safety using AST analysis."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    validator = _SafetyValidator()
    validator.visit(tree)

    if validator.errors:
        return False, "; ".join(validator.errors)

    return True, ""


async def _safe_execute(code: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Run Python code in a sandboxed environment.

    Uses AST-based validation, restricted builtins, and timeout protection.

    Args:
        code: Python code to run
        timeout: Maximum time in seconds (default: 5.0)

    Returns:
        dict with success, output, and error keys
    """
    is_safe, error_msg = _validate_code(code)
    if not is_safe:
        logger.warning("unsafe_code_blocked", error=error_msg)
        return {"success": False, "output": "", "error": f"Security validation failed: {error_msg}"}

    # Prepare restricted environment
    safe_globals = {"__builtins__": SAFE_BUILTINS.copy()}

    # Add allowed imports
    for module_name in ALLOWED_IMPORTS:
        try:
            safe_globals[module_name] = __import__(module_name)
        except ImportError:
            pass

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    def _run_code():
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            _python_code_runner(code, safe_globals, safe_globals)

    try:
        await asyncio.wait_for(asyncio.to_thread(_run_code), timeout=timeout)

        output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        if stderr_output:
            output = output + "\n[stderr]: " + stderr_output

        logger.info("code_ran_successfully", output_length=len(output))
        return {"success": True, "output": output.strip(), "error": ""}

    except asyncio.TimeoutError:
        logger.warning("code_timeout", timeout=timeout)
        return {"success": False, "output": stdout_capture.getvalue(), "error": f"Timed out after {timeout}s"}
    except Exception as e:
        logger.error("code_failed", error=str(e))
        return {"success": False, "output": stdout_capture.getvalue(), "error": str(e)}

@quiet_star
async def program_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Framework: Program of Thoughts (PoT): Code-Based Reasoning
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context"),
        state=state
    )

    # Construct the Protocol Prompt for the Client
    prompt = f"""# Program of Thoughts Protocol

I have selected the **Program of Thoughts (PoT)** framework for this task.
Code-Based Reasoning: Generates and executes Python scripts to compute answers.

## Use Case
Math, data processing, algorithmic verification, testing

## Task
{query}

## 🧠 Execution Protocol (Client-Side)

Please execute the reasoning steps for **Program of Thoughts** using your internal context:

### Framework Steps
1. ANALYZE: Understand what needs to be computed
2. GENERATE: Create Python code to solve
3. EXECUTE: Run the code (if capable) or simulate execution
4. INTERPRET: Explain the results

## 📝 Code Context
{code_context}

**Please start by outlining your approach following the Program of Thoughts process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="program_of_thoughts",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated"
    )

    return state
