"""
Program of Thoughts (PoT): Code-Based Reasoning

Generates and executes Python scripts to compute answers
rather than reasoning in text. Used for math, data, testing.
"""

import ast
import asyncio
import io
import sys
import traceback
from typing import Optional, Any, Set
from contextlib import redirect_stdout, redirect_stderr
from ...state import GraphState
from ..common import (
    quiet_star,
    call_fast_synthesizer,
    call_deep_reasoner,
    add_reasoning_step,
    format_code_context,
    run_tool,
)


# Safe execution environment
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

ALLOWED_IMPORTS = ['math', 'statistics', 'itertools', 'functools', 'collections', 're', 'json', 'datetime']


@quiet_star
async def program_of_thoughts_node(state: GraphState) -> GraphState:
    """
    Program of Thoughts: Code-Based Computation.
    
    Process:
    1. ANALYZE: Understand what needs to be computed
    2. GENERATE: Create Python code to solve
    3. EXECUTE: Run the code safely
    4. INTERPRET: Explain the results
    
    Best for: Math, data processing, algorithmic verification, testing
    """
    query = state["query"]
    code_context = format_code_context(
        state.get("code_snippet"),
        state.get("file_list"),
        state.get("ide_context")
    )
    
    # =========================================================================
    # Phase 1: ANALYZE the Computation
    # =========================================================================
    
    analyze_prompt = f"""Analyze what computation is needed.

TASK: {query}

CONTEXT:
{code_context}

Identify:
1. **COMPUTATION TYPE**: What type of computation? (math, data processing, algorithm, test)
2. **INPUTS**: What data/values are inputs?
3. **EXPECTED OUTPUT**: What should the result look like?
4. **APPROACH**: What algorithm/method should be used?

Be specific about the computational approach."""

    analyze_response, _ = await call_fast_synthesizer(
        prompt=analyze_prompt,
        state=state,
        max_tokens=600
    )
    
    add_reasoning_step(
        state=state,
        framework="program_of_thoughts",
        thought="Analyzed computation requirements",
        action="analysis",
        observation=analyze_response[:200]
    )
    
    # =========================================================================
    # Phase 2: GENERATE Python Code
    # =========================================================================
    
    generate_prompt = f"""Generate Python code to solve this computational task.

TASK: {query}

ANALYSIS:
{analyze_response}

CONTEXT:
{code_context}

CONSTRAINTS:
- Use ONLY these imports: {', '.join(ALLOWED_IMPORTS)}
- No file I/O, network, or system calls
- Code must be self-contained
- Print the final result clearly

Generate EXECUTABLE Python code:

```python
# Your code here
# End with print() statements showing results
```

Make the code clean, commented, and correct."""

    generate_response, _ = await call_deep_reasoner(
        prompt=generate_prompt,
        state=state,
        system="Generate clean, correct, executable Python code.",
        temperature=0.4
    )
    
    # Extract code
    import re
    code_pattern = r"```python\n(.*?)```"
    match = re.search(code_pattern, generate_response, re.DOTALL)
    
    if not match:
        code_pattern = r"```\n(.*?)```"
        match = re.search(code_pattern, generate_response, re.DOTALL)
    
    generated_code = match.group(1).strip() if match else ""

    code_lines = len(generated_code.split('\n'))
    add_reasoning_step(
        state=state,
        framework="program_of_thoughts",
        thought="Generated Python code for computation",
        action="code_generation",
        observation=f"Generated {code_lines} lines of code"
    )
    
    # =========================================================================
    # Phase 3: EXECUTE the Code (via LangChain tool)
    execution_result = await run_tool("execute_code", {"code": generated_code, "language": "python"}, state)
    
    add_reasoning_step(
        state=state,
        framework="program_of_thoughts",
        thought="Executed generated code",
        action="execution",
        observation=f"Success: {execution_result['success']}, Output: {execution_result['output'][:100] if execution_result['output'] else 'None'}"
    )
    
    # Retry if failed
    if not execution_result['success'] and execution_result['error']:
        fix_prompt = f"""Fix the Python code error.

ORIGINAL CODE:
```python
{generated_code}
```

ERROR:
{execution_result['error']}

Provide FIXED code that addresses this error.

```python
# Fixed code
```"""

        fix_response, _ = await call_fast_synthesizer(
            prompt=fix_prompt,
            state=state,
            max_tokens=1500
        )
        
        match = re.search(r"```python\n(.*?)```", fix_response, re.DOTALL)
        if match:
            generated_code = match.group(1).strip()
            execution_result = await run_tool("execute_code", {"code": generated_code, "language": "python"}, state)
            
            add_reasoning_step(
                state=state,
                framework="program_of_thoughts",
                thought="Retried with fixed code",
                action="retry_execution",
                observation=f"Success: {execution_result['success']}"
            )
    
    # =========================================================================
    # Phase 4: INTERPRET Results
    # =========================================================================
    
    interpret_prompt = f"""Interpret the program execution results.

ORIGINAL TASK: {query}

CODE EXECUTED:
```python
{generated_code}
```

EXECUTION OUTPUT:
{execution_result['output'] if execution_result['success'] else f"Error: {execution_result['error']}"}

Provide:
1. **RESULT**: The answer to the original task
2. **EXPLANATION**: How the code solved it
3. **VERIFICATION**: Why we can trust this result
4. **FINAL ANSWER**: Clear, direct answer to the task"""

    interpret_response, _ = await call_deep_reasoner(
        prompt=interpret_prompt,
        state=state,
        system="Interpret the computational results clearly.",
        temperature=0.4
    )
    
    add_reasoning_step(
        state=state,
        framework="program_of_thoughts",
        thought="Interpreted execution results",
        action="interpretation",
        observation="Generated final answer from computation"
    )
    
    # Store execution info
    state["working_memory"]["pot_code"] = generated_code
    state["working_memory"]["pot_output"] = execution_result['output']
    state["working_memory"]["pot_success"] = execution_result['success']
    
    # Update final state
    state["final_answer"] = interpret_response
    state["final_code"] = generated_code
    state["confidence_score"] = 0.9 if execution_result['success'] else 0.5
    
    return state


class _SafetyValidator(ast.NodeVisitor):
    """AST visitor to detect dangerous code patterns that must be BLOCKED."""

    # Modules that are BLOCKED from being imported (security risk)
    BLOCKED_IMPORTS: Set[str] = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests',
        'urllib', 'ctypes', 'multiprocessing', 'threading',
        'signal', 'pty', 'fcntl', 'resource', 'pwd', 'grp', 'crypt',
        'tempfile', 'glob', 'pathlib', 'builtins', 'importlib', 'code',
    }

    # Function calls that are BLOCKED (security risk)
    BLOCKED_CALLS: Set[str] = {
        '__import__', 'exec', 'eval', 'compile', 'open', 'file',
        'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
        'setattr', 'getattr', 'delattr', 'hasattr', 'object',
        'breakpoint', 'exit', 'quit', 'help', 'license', 'credits',
    }

    # Attribute names that are BLOCKED (security risk)
    BLOCKED_ATTRS: Set[str] = {
        '__builtins__', '__class__', '__mro__', '__globals__', '__code__',
        '__dict__', '__loader__', '__spec__', '__subclasses__', '__bases__',
        '__name__', '__qualname__', '__module__', '__func__', '__self__',
        '__closure__', '__annotations__', '__kwdefaults__', '__defaults__',
    }

    def __init__(self, allowed_imports: Set[str]):
        self.allowed_imports = allowed_imports
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split('.')[0]
            if module in self.BLOCKED_IMPORTS:
                self.violations.append(f"Blocked import: {alias.name}")
            elif module not in self.allowed_imports:
                self.violations.append(f"Disallowed import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module = node.module.split('.')[0]
            if module in self.BLOCKED_IMPORTS:
                self.violations.append(f"Blocked import from: {node.module}")
            elif module not in self.allowed_imports:
                self.violations.append(f"Disallowed import from: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.BLOCKED_CALLS:
                self.violations.append(f"Blocked call: {node.func.id}()")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.BLOCKED_CALLS:
                self.violations.append(f"Blocked call: .{node.func.attr}()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.BLOCKED_ATTRS:
            self.violations.append(f"Blocked attribute access: .{node.attr}")
        self.generic_visit(node)


async def _safe_execute(code: str, timeout: float = 5.0) -> dict:
    """
    Safely execute Python code in a sandboxed environment.

    Uses AST parsing to detect dangerous patterns (not bypassable string matching).

    Returns: {"success": bool, "output": str, "error": str}
    """
    try:
        # Parse code to AST for safety validation
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "success": False,
                "output": "",
                "error": f"Syntax error: {e}"
            }

        # Validate AST for dangerous patterns
        allowed = set(ALLOWED_IMPORTS)
        validator = _SafetyValidator(allowed)
        validator.visit(tree)

        if validator.violations:
            return {
                "success": False,
                "output": "",
                "error": f"Security violation: {'; '.join(validator.violations)}"
            }

        # Prepare safe globals
        safe_globals = {"__builtins__": SAFE_BUILTINS.copy()}
        
        # Add allowed imports
        for module_name in ALLOWED_IMPORTS:
            try:
                safe_globals[module_name] = __import__(module_name)
            except ImportError:
                pass
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Execute with timeout simulation (asyncio-friendly)
        # Use safe_globals for both globals and locals to properly capture variables
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, safe_globals, safe_globals)
        
        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        
        return {
            "success": True,
            "output": output + (f"\nStderr: {error_output}" if error_output else ""),
            "error": ""
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }
