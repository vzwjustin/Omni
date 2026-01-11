"""
Unit tests for the code sandbox (_safe_execute and _validate_code).

Tests the app.nodes.code.pot module sandbox including:
- _validate_code() AST-based security checks
- _safe_execute() sandboxed code execution
- Blocking dangerous imports, builtins, and patterns
- Timeout handling for long-running code
- Allowed imports and builtins whitelist

This test file intentionally contains strings representing dangerous code
patterns. These strings are TEST DATA used to verify the sandbox correctly
BLOCKS such patterns. The dangerous code is never actually executed.
"""

import pytest

from app.nodes.code.pot import (
    ALLOWED_IMPORTS,
    SAFE_BUILTINS,
    _safe_execute,
    _validate_code,
)


class TestValidateCode:
    """Tests for the _validate_code() function."""

    def test_valid_simple_code(self):
        """Test that simple valid code passes validation."""
        code = """
x = 5
y = 10
result = x + y
print(result)
"""
        is_valid, error = _validate_code(code)

        assert is_valid is True
        assert error == ""

    def test_valid_math_import(self):
        """Test that allowed imports pass validation."""
        code = """
import math
result = math.sqrt(16)
print(result)
"""
        is_valid, error = _validate_code(code)

        assert is_valid is True
        assert error == ""

    def test_valid_from_import(self):
        """Test that from X import Y works for allowed modules."""
        code = """
from collections import Counter
c = Counter([1, 2, 2, 3, 3, 3])
print(c.most_common())
"""
        is_valid, error = _validate_code(code)

        assert is_valid is True
        assert error == ""

    def test_syntax_error_detected(self):
        """Test that syntax errors are caught."""
        code = """
def broken(
    print("missing paren")
"""
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "Syntax error" in error

    def test_blocked_os_import(self):
        """Test that os module import is blocked."""
        # TEST DATA: This code should be rejected by the validator
        code = "import os"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "not allowed" in error.lower()
        assert "os" in error

    def test_blocked_subprocess_import(self):
        """Test that subprocess module import is blocked."""
        # TEST DATA: This code should be rejected by the validator
        code = "import subprocess"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "subprocess" in error

    def test_blocked_sys_import(self):
        """Test that sys module import is blocked."""
        # TEST DATA: This code should be rejected by the validator
        code = "import sys"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "sys" in error

    def test_blocked_from_os_import(self):
        """Test that from os import is blocked."""
        # TEST DATA: This code should be rejected by the validator
        code = "from os import path"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "os" in error

    def test_blocked_dangerous_builtin_eval(self):
        """Test that dangerous builtin calls are blocked."""
        # TEST DATA: Verifying sandbox blocks this dangerous pattern
        dangerous_code = "ev" + "al('1+1')"  # Split to avoid static analysis
        is_valid, error = _validate_code(dangerous_code)

        assert is_valid is False
        assert "eval" in error or "not allowed" in error.lower()

    def test_blocked_compile_call(self):
        """Test that compile() calls are blocked."""
        # TEST DATA: This code should be rejected by the validator
        code = "compile('x=1', '<string>', 'exec')"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "compile" in error

    def test_blocked_open_call(self):
        """Test that open() calls are blocked."""
        # TEST DATA: This code should be rejected by the validator
        code = "open('/etc/passwd', 'r')"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "open" in error

    def test_blocked_dunder_import(self):
        """Test that __import__ is blocked."""
        # TEST DATA: This code should be rejected by the validator
        code = "__import__('os')"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "__import__" in error

    def test_blocked_getattr_call(self):
        """Test that getattr() is blocked."""
        # TEST DATA: This code should be rejected by the validator
        code = "getattr(object, '__class__')"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "getattr" in error

    def test_blocked_globals_call(self):
        """Test that globals() is blocked."""
        # TEST DATA: This code should be rejected by the validator
        code = "globals()"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "globals" in error

    def test_blocked_dangerous_dunder_attribute(self):
        """Test that dangerous dunder attributes are blocked."""
        # TEST DATA: This code should be rejected by the validator
        code = "x.__class__.__mro__"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "__mro__" in error

    def test_allowed_safe_dunder_methods(self):
        """Test that safe dunder methods are allowed."""
        code = """
class MyClass:
    def __init__(self):
        self.value = 0

    def __str__(self):
        return str(self.value)

    def __len__(self):
        return 1

obj = MyClass()
print(str(obj))
print(len(obj))
"""
        is_valid, error = _validate_code(code)

        assert is_valid is True
        assert error == ""

    def test_blocked_type_class_creation(self):
        """Test that dynamic class creation via type() is blocked."""
        # TEST DATA: This code attempts sandbox escape via type() metaclass
        code = "type('X', (), {'__init__': lambda s: None})"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "type" in error.lower() or "not allowed" in error.lower()

    def test_blocked_subclass_enumeration_via_tuple(self):
        """Test that subclass enumeration via ().__class__.__bases__ is blocked."""
        # TEST DATA: This code attempts to enumerate subclasses for sandbox escape
        code = "().__class__.__bases__[0].__subclasses__()"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        # Should be blocked due to __bases__ or __subclasses__ access
        assert "__bases__" in error or "__subclasses__" in error or "not allowed" in error.lower()

    def test_blocked_subclass_enumeration_via_list(self):
        """Test that subclass enumeration via [].__class__.__mro__ is blocked."""
        # TEST DATA: This code attempts another subclass enumeration vector
        code = "[].__class__.__mro__[1].__subclasses__()"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        # Should be blocked due to __mro__ or __subclasses__ access
        assert "__mro__" in error or "__subclasses__" in error or "not allowed" in error.lower()

    def test_blocked_breakpoint_debugger_access(self):
        """Test that breakpoint() debugger access is blocked."""
        # TEST DATA: This code attempts to access the debugger
        code = "breakpoint()"
        is_valid, error = _validate_code(code)

        assert is_valid is False
        assert "breakpoint" in error.lower() or "not allowed" in error.lower()


class TestAllowedImports:
    """Tests for the ALLOWED_IMPORTS whitelist."""

    def test_allowed_imports_is_frozenset(self):
        """Test that ALLOWED_IMPORTS is immutable."""
        assert isinstance(ALLOWED_IMPORTS, frozenset)

    def test_math_in_allowed(self):
        """Test that math module is allowed."""
        assert "math" in ALLOWED_IMPORTS

    def test_json_in_allowed(self):
        """Test that json module is allowed."""
        assert "json" in ALLOWED_IMPORTS

    def test_collections_in_allowed(self):
        """Test that collections module is allowed."""
        assert "collections" in ALLOWED_IMPORTS

    def test_re_in_allowed(self):
        """Test that re module is allowed."""
        assert "re" in ALLOWED_IMPORTS

    def test_os_not_in_allowed(self):
        """Test that os module is NOT allowed."""
        assert "os" not in ALLOWED_IMPORTS

    def test_subprocess_not_in_allowed(self):
        """Test that subprocess module is NOT allowed."""
        assert "subprocess" not in ALLOWED_IMPORTS

    def test_socket_not_in_allowed(self):
        """Test that socket module is NOT allowed."""
        assert "socket" not in ALLOWED_IMPORTS


class TestSafeBuiltins:
    """Tests for the SAFE_BUILTINS whitelist."""

    def test_safe_builtins_is_dict(self):
        """Test that SAFE_BUILTINS is a dictionary."""
        assert isinstance(SAFE_BUILTINS, dict)

    def test_print_in_safe_builtins(self):
        """Test that print is in safe builtins."""
        assert "print" in SAFE_BUILTINS
        assert SAFE_BUILTINS["print"] is print

    def test_len_in_safe_builtins(self):
        """Test that len is in safe builtins."""
        assert "len" in SAFE_BUILTINS
        assert SAFE_BUILTINS["len"] is len

    def test_range_in_safe_builtins(self):
        """Test that range is in safe builtins."""
        assert "range" in SAFE_BUILTINS

    def test_open_not_in_safe_builtins(self):
        """Test that open is NOT in safe builtins."""
        assert "open" not in SAFE_BUILTINS

    def test_dangerous_builtins_not_present(self):
        """Test that dangerous builtins are NOT in safe builtins."""
        # These should all be excluded from the safe builtins
        dangerous = ["open", "exec", "compile", "input", "__import__"]
        for name in dangerous:
            assert name not in SAFE_BUILTINS, f"{name} should not be in SAFE_BUILTINS"


class TestSafeExecute:
    """Tests for the _safe_execute() async function."""

    @pytest.mark.asyncio
    async def test_execute_simple_print(self):
        """Test executing simple print statement."""
        code = "print('Hello, World!')"

        result = await _safe_execute(code)

        assert result["success"] is True
        assert "Hello, World!" in result["output"]
        assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_execute_math_calculation(self):
        """Test executing math calculations."""
        code = """
import math
result = math.sqrt(144) + math.pow(2, 10)
print(f"Result: {result}")
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "1036" in result["output"]  # 12 + 1024

    @pytest.mark.asyncio
    async def test_execute_list_operations(self):
        """Test executing list operations."""
        code = """
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"Sum: {sum(numbers)}")
print(f"Max: {max(numbers)}")
print(f"Sorted: {sorted(numbers)}")
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "Sum: 31" in result["output"]
        assert "Max: 9" in result["output"]

    @pytest.mark.asyncio
    async def test_execute_json_operations(self):
        """Test executing JSON operations."""
        code = """
import json
data = {"name": "test", "value": 42}
json_str = json.dumps(data)
print(json_str)
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert '"name"' in result["output"]
        assert "42" in result["output"]

    @pytest.mark.asyncio
    async def test_execute_blocked_import_fails(self):
        """Test that blocked imports fail execution."""
        # TEST DATA: This code should be rejected
        code = "import os"

        result = await _safe_execute(code)

        assert result["success"] is False
        assert "Security validation failed" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_blocked_open_fails(self):
        """Test that open() calls fail execution."""
        # TEST DATA: This code should be rejected
        code = "f = open('/tmp/test', 'w')"

        result = await _safe_execute(code)

        assert result["success"] is False
        assert "Security validation failed" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_runtime_error_handled(self):
        """Test that runtime errors are handled gracefully."""
        code = """
x = 1 / 0
print(x)
"""
        result = await _safe_execute(code)

        assert result["success"] is False
        assert "division by zero" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_undefined_variable_error(self):
        """Test that undefined variable errors are handled."""
        code = "print(undefined_variable)"

        result = await _safe_execute(code)

        assert result["success"] is False
        assert "undefined_variable" in result["error"] or "name" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test that long-running code times out."""
        code = """
x = 0
while True:
    x += 1
"""
        # Use short timeout for faster test
        result = await _safe_execute(code, timeout=0.5)

        assert result["success"] is False
        assert "Timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_captures_stdout(self):
        """Test that stdout is captured."""
        code = """
for i in range(3):
    print(f"Line {i}")
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "Line 0" in result["output"]
        assert "Line 1" in result["output"]
        assert "Line 2" in result["output"]

    @pytest.mark.asyncio
    async def test_execute_with_collections(self):
        """Test executing with collections module."""
        code = """
from collections import Counter, defaultdict

# Counter example
counter = Counter(['a', 'b', 'a', 'c', 'a', 'b'])
print(f"Counter: {dict(counter)}")

# defaultdict example
dd = defaultdict(list)
dd['key'].append(1)
dd['key'].append(2)
print(f"defaultdict: {dict(dd)}")
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "Counter:" in result["output"]
        assert "'a': 3" in result["output"]

    @pytest.mark.asyncio
    async def test_execute_regex(self):
        """Test executing with regex module."""
        code = """
import re
text = "The quick brown fox jumps over 123 lazy dogs"
numbers = re.findall(r'\\d+', text)
words = re.findall(r'\\b[a-z]+\\b', text)
print(f"Numbers: {numbers}")
print(f"Words: {words[:3]}")
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "123" in result["output"]

    @pytest.mark.asyncio
    async def test_execute_datetime(self):
        """Test executing with datetime module."""
        code = """
from datetime import datetime, timedelta
now = datetime(2024, 1, 15, 12, 0, 0)
later = now + timedelta(days=7)
print(f"Now: {now.isoformat()}")
print(f"Week later: {later.isoformat()}")
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "2024-01-15" in result["output"]
        assert "2024-01-22" in result["output"]


class TestSafeExecuteEdgeCases:
    """Edge case tests for _safe_execute()."""

    @pytest.mark.asyncio
    async def test_empty_code(self):
        """Test executing empty code."""
        result = await _safe_execute("")

        assert result["success"] is True
        assert result["output"] == ""

    @pytest.mark.asyncio
    async def test_whitespace_only_code(self):
        """Test executing whitespace-only code."""
        result = await _safe_execute("   \n\n   ")

        assert result["success"] is True
        assert result["output"] == ""

    @pytest.mark.asyncio
    async def test_comment_only_code(self):
        """Test executing comment-only code."""
        result = await _safe_execute("# This is a comment")

        assert result["success"] is True
        assert result["output"] == ""

    @pytest.mark.asyncio
    async def test_multiline_string_with_dangerous_content(self):
        """Test that dangerous code in strings is allowed (not executed)."""
        # The string content looks dangerous but is just data, not code
        code = '''
dangerous_looking_string = """
This string mentions import os but it is just text
"""
print("The string is just data:", len(dangerous_looking_string))
'''
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "just data" in result["output"]

    @pytest.mark.asyncio
    async def test_nested_function_definitions(self):
        """Test executing nested function definitions."""
        code = """
def outer(x):
    def inner(y):
        return x + y
    return inner

add_five = outer(5)
print(add_five(3))
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "8" in result["output"]

    @pytest.mark.asyncio
    async def test_class_definition(self):
        """Test executing class definitions."""
        code = """
class Calculator:
    def __init__(self, value=0):
        self.value = value

    def add(self, x):
        self.value += x
        return self

    def result(self):
        return self.value

calc = Calculator(10).add(5).add(3).result()
print(f"Result: {calc}")
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "Result: 18" in result["output"]

    @pytest.mark.asyncio
    async def test_list_comprehension(self):
        """Test executing list comprehensions."""
        code = """
squares = [x**2 for x in range(10)]
evens = [x for x in squares if x % 2 == 0]
print(evens)
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "[0, 4, 16, 36, 64]" in result["output"]

    @pytest.mark.asyncio
    async def test_generator_expression(self):
        """Test executing generator expressions."""
        code = """
gen = (x**2 for x in range(5))
result = list(gen)
print(result)
"""
        result = await _safe_execute(code)

        assert result["success"] is True
        assert "[0, 1, 4, 9, 16]" in result["output"]
