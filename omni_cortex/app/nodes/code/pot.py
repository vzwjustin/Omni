"""
Program of Thoughts (PoT): Code-Based Reasoning

Generates and executes Python scripts to compute answers
rather than reasoning in text. Used for math, data, testing.
Returns Reasoning Protocol for Client Execution
"""

import ast
import asyncio
import builtins
import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import structlog

from ...state import GraphState
from ..common import (
    add_reasoning_step,
    prepare_context_with_gemini,
    quiet_star,
)
from ..example_utilities import search_code_examples

logger = structlog.get_logger(__name__)


# =============================================================================
# Safe Code Sandbox
# =============================================================================

# Safe imports whitelist - only allow safe standard library modules
ALLOWED_IMPORTS = frozenset(
    [
        "math",
        "statistics",
        "itertools",
        "functools",
        "collections",
        "re",
        "json",
        "datetime",
        "decimal",
        "fractions",
        "random",
        "string",
        "textwrap",
        "unicodedata",
        "operator",
        "copy",
        "heapq",
        "bisect",
        "array",
        "enum",
        "dataclasses",
        "typing",
    ]
)

# Safe builtins whitelist
# SECURITY: Intentionally EXCLUDES:
# - type(): can create classes with arbitrary methods
# - isinstance/issubclass: type introspection can leak info
# - getattr/setattr/delattr/hasattr: attribute manipulation
# - eval/exec/compile: code execution
# - open/__import__: file/module access (raw __import__ excluded, wrapped version added below)
SAFE_BUILTINS = {
    # Math/logic
    "abs": abs,
    "all": all,
    "any": any,
    "divmod": divmod,
    "pow": pow,
    "round": round,
    "sum": sum,
    "min": min,
    "max": max,
    # Type constructors (safe - can't define methods)
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "tuple": tuple,
    # Iteration
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "reversed": reversed,
    "sorted": sorted,
    "iter": iter,
    "next": next,
    "slice": slice,
    # String/repr
    "chr": chr,
    "ord": ord,
    "repr": repr,
    "format": format,
    "bin": bin,
    "hex": hex,
    "oct": oct,
    "hash": hash,
    # Length
    "len": len,
    # I/O (sandboxed)
    "print": print,
    # Constants
    "True": True,
    "False": False,
    "None": None,
    # Internal Python builtins needed for exec()
    # __build_class__ is required for 'class' keyword to work
    "__build_class__": builtins.__build_class__,
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Safe import wrapper that only allows imports from ALLOWED_IMPORTS.

    This is needed because exec() uses __import__ internally when it encounters
    'import' statements in the code being executed.
    """
    # Get the root module name (e.g., 'math' from 'math.sin')
    root_module = name.split(".")[0]

    if root_module not in ALLOWED_IMPORTS:
        raise ImportError(
            f"Import of '{name}' is not allowed. Allowed imports: {', '.join(sorted(ALLOWED_IMPORTS))}"
        )

    # Use the real __import__ for allowed modules
    return builtins.__import__(name, globals, locals, fromlist, level)


# exec() for sandboxed code execution
# This is intentionally using exec() - the sandbox security comes from:
# 1. AST validation blocking dangerous patterns
# 2. Restricted builtins (no type(), getattr(), etc.)
# 3. Timeout protection
# 4. Restricted globals/locals
_python_code_runner = exec

# Dangerous functions to block
_DANGEROUS_FUNCS = {
    "eval",
    "compile",
    "open",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "breakpoint",
    "exit",
    "quit",
    "type",
}  # type() allows dynamic class creation - sandbox escape vector

# Dangerous method names to block (shell/process/introspection related)
_DANGEROUS_METHODS = {
    # Process execution
    "popen",
    "spawn",
    "fork",
    "call",
    "run",
    "Popen",
    "check_output",
    "check_call",
    "system",
    "execv",
    "execve",
    "spawnl",
    "spawnle",
    # File operations
    "read",
    "write",
    "readlines",
    "writelines",
    # Dangerous introspection
    "mro",
    "__subclasses__",
    "__bases__",
    "__class__",
    # Code execution
    "exec",
    "eval",
    "compile",
}


class _SafetyValidator(ast.NodeVisitor):
    """AST-based validator to detect dangerous code patterns."""

    def __init__(self):
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module not in ALLOWED_IMPORTS:
                self.errors.append(f"Import of '{alias.name}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module = node.module.split(".")[0]
            if module not in ALLOWED_IMPORTS:
                self.errors.append(f"Import from '{node.module}' is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in _DANGEROUS_FUNCS:
                self.errors.append(f"Call to '{node.func.id}' is not allowed")
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _DANGEROUS_METHODS:
            self.errors.append(f"Call to '.{node.func.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block dunder attribute access (except common safe ones)
        if node.attr.startswith("__") and node.attr.endswith("__"):
            safe_dunders = (
                "__init__",
                "__str__",
                "__repr__",
                "__len__",
                "__iter__",
                "__next__",
                "__getitem__",
                "__setitem__",
                "__contains__",
                "__eq__",
                "__ne__",
                "__lt__",
                "__le__",
                "__gt__",
                "__ge__",
                "__hash__",
                "__bool__",
                "__add__",
                "__sub__",
                "__mul__",
                "__truediv__",
                "__floordiv__",
                "__mod__",
                "__pow__",
            )
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


async def _safe_execute(code: str, timeout: float = 5.0) -> dict[str, Any]:
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

    # Prepare restricted environment with safe builtins
    restricted_builtins = SAFE_BUILTINS.copy()
    # Add the safe import wrapper so 'import' statements work for allowed modules
    restricted_builtins["__import__"] = _safe_import
    safe_globals = {
        "__builtins__": restricted_builtins,
        "__name__": "__main__",  # Required for class definitions
    }

    # Add allowed imports (skip if not available on system - not all are required)
    for module_name in ALLOWED_IMPORTS:
        try:
            safe_globals[module_name] = __import__(module_name)
        except ImportError as e:
            # Module not installed - acceptable, not all modules required
            logger.debug("optional_sandbox_module_unavailable", module=module_name, error=str(e))

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
        return {
            "success": False,
            "output": stdout_capture.getvalue(),
            "error": f"Timed out after {timeout}s",
        }
    except Exception as e:
        logger.error("code_failed", error=str(e))
        return {"success": False, "output": stdout_capture.getvalue(), "error": str(e)}


@quiet_star
async def program_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Framework: Program of Thoughts (PoT): Code-Based Reasoning
    """
    query = state.get("query", "")
    # Use Gemini to preprocess context via ContextGateway

    code_context = await prepare_context_with_gemini(query=query, state=state)

    # Search for similar code examples
    code_examples = search_code_examples(query, task_type="code_generation")
    if code_examples:
        logger.info("pot_enhanced", query_preview=query[:50])

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
{
        f'''
## 💡 Similar Code Examples from 12K+ Knowledge Base
{code_examples}
'''
        if code_examples
        else ""
    }
**Please start by outlining your approach following the Program of Thoughts process.**
"""

    state["final_answer"] = prompt
    state["confidence_score"] = 1.0

    add_reasoning_step(
        state=state,
        framework="program_of_thoughts",
        thought="Generated Framework protocol for client execution",
        action="handoff",
        observation="Prompt generated",
    )

    return state
