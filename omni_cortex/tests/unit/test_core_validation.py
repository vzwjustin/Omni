"""
Tests for app/core/validation.py - Input validation and injection attack prevention.

This test suite verifies that the validation module properly blocks:
- Script injection attacks
- XSS attempts
- Directory traversal
- Invalid input formats
- Malformed data
"""

import pytest

from app.core.validation import (
    ValidationError,
    sanitize_code_snippet,
    sanitize_context,
    sanitize_file_path,
    sanitize_query,
    validate_boolean,
    validate_float,
    validate_framework_name,
    validate_integer,
    validate_thread_id,
)


class TestSanitizeQuery:
    """Tests for sanitize_query function - blocks injection attacks."""

    def test_clean_input_passes(self):
        """Clean natural language query should pass validation."""
        clean_query = "How do I implement a binary search algorithm?"
        result = sanitize_query(clean_query)
        assert result == clean_query

    def test_multiline_query_passes(self):
        """Multiline queries should be allowed."""
        query = """I need help with:
        1. Database optimization
        2. Query performance
        3. Index creation"""
        result = sanitize_query(query)
        assert "Database optimization" in result

    def test_technical_terms_pass(self):
        """Technical terms should not trigger false positives."""
        query = "Explain React's useEffect and useState hooks"
        result = sanitize_query(query)
        assert result == query

    def test_script_tag_blocked(self):
        """Script tags should be blocked."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_query("<script>alert('XSS')</script>")
        assert "suspicious pattern" in str(exc_info.value).lower()

    def test_script_tag_case_insensitive(self):
        """Script tags should be blocked regardless of case."""
        with pytest.raises(ValidationError):
            sanitize_query("<SCRIPT>alert('XSS')</SCRIPT>")
        with pytest.raises(ValidationError):
            sanitize_query("<ScRiPt>alert('XSS')</ScRiPt>")

    def test_script_tag_with_spaces(self):
        """Script tags with spaces should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("<  script  >alert('XSS')</script>")

    def test_javascript_protocol_blocked(self):
        """JavaScript protocol should be blocked."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_query("Click here: javascript:alert('XSS')")
        assert "suspicious pattern" in str(exc_info.value).lower()

    def test_javascript_protocol_case_insensitive(self):
        """JavaScript protocol should be blocked regardless of case."""
        with pytest.raises(ValidationError):
            sanitize_query("JAVASCRIPT:alert(1)")
        with pytest.raises(ValidationError):
            sanitize_query("JaVaScRiPt:alert(1)")

    def test_event_handlers_blocked(self):
        """Event handlers like onclick, onerror should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("<img onclick='alert(1)'>")
        with pytest.raises(ValidationError):
            sanitize_query("<img onerror='alert(1)'>")
        with pytest.raises(ValidationError):
            sanitize_query("<div onload='malicious()'>")

    def test_hex_escapes_blocked(self):
        """Hex escape sequences should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("\\x3cscript\\x3e")

    def test_unicode_escapes_blocked(self):
        """Unicode escape sequences should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("\\u003cscript\\u003e")

    def test_eval_blocked(self):
        """eval() calls should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("eval('malicious code')")

    def test_exec_blocked(self):
        """exec() calls should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("exec('malicious code')")

    def test_import_blocked(self):
        """__import__ should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("__import__('os').system('rm -rf /')")

    def test_iframe_blocked(self):
        """iframe tags should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("<iframe src='evil.com'></iframe>")

    def test_embed_blocked(self):
        """embed tags should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("<embed src='evil.swf'>")

    def test_object_blocked(self):
        """object tags should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("<object data='evil.pdf'>")

    def test_empty_input_blocked(self):
        """Empty input should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_query("")
        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only_blocked(self):
        """Whitespace-only input should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_query("   \n\t   ")
        assert "empty" in str(exc_info.value).lower()

    def test_max_length_enforced(self):
        """Input exceeding max length should be rejected."""
        long_query = "x" * 10001
        with pytest.raises(ValidationError) as exc_info:
            sanitize_query(long_query)
        assert "maximum length" in str(exc_info.value).lower()

    def test_non_string_rejected(self):
        """Non-string input should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_query(12345)
        assert "must be a string" in str(exc_info.value).lower()

    def test_whitespace_trimmed(self):
        """Leading and trailing whitespace should be trimmed."""
        query = "  How do I use Docker?  "
        result = sanitize_query(query)
        assert result == "How do I use Docker?"


class TestSanitizeCodeSnippet:
    """Tests for sanitize_code_snippet - allows code patterns but enforces limits."""

    def test_none_returns_none(self):
        """None input should return None."""
        result = sanitize_code_snippet(None)
        assert result is None

    def test_valid_python_code_passes(self):
        """Valid Python code should pass validation."""
        code = """
def hello():
    print("Hello, World!")
"""
        result = sanitize_code_snippet(code)
        assert "def hello" in result

    def test_javascript_code_passes(self):
        """JavaScript code should pass (allow_code=True)."""
        code = """
function hello() {
    console.log("Hello");
}
"""
        result = sanitize_code_snippet(code)
        assert "function hello" in result

    def test_eval_allowed_in_code(self):
        """eval is allowed in code snippets (legitimate use case)."""
        code = "result = eval('2 + 2')"
        result = sanitize_code_snippet(code)
        assert result == code

    def test_max_length_enforced(self):
        """Code exceeding max length should be rejected."""
        long_code = "x" * 50001
        with pytest.raises(ValidationError) as exc_info:
            sanitize_code_snippet(long_code)
        assert "maximum length" in str(exc_info.value).lower()

    def test_empty_code_rejected(self):
        """Empty code snippet should be rejected."""
        with pytest.raises(ValidationError):
            sanitize_code_snippet("")


class TestSanitizeContext:
    """Tests for sanitize_context - allows code in context."""

    def test_none_returns_none(self):
        """None input should return None."""
        result = sanitize_context(None)
        assert result is None

    def test_large_context_passes(self):
        """Large context within limits should pass."""
        context = "x" * 99000
        result = sanitize_context(context)
        assert len(result) == 99000

    def test_max_length_enforced(self):
        """Context exceeding max length should be rejected."""
        huge_context = "x" * 100001
        with pytest.raises(ValidationError) as exc_info:
            sanitize_context(huge_context)
        assert "maximum length" in str(exc_info.value).lower()

    def test_code_patterns_allowed(self):
        """Code patterns should be allowed in context."""
        context = """
        function evaluate() {
            return eval('1 + 1');
        }
        """
        result = sanitize_context(context)
        assert "eval" in result


class TestSanitizeFilePath:
    """Tests for sanitize_file_path - prevents directory traversal."""

    def test_valid_relative_path_passes(self):
        """Valid relative path should pass."""
        path = "src/main.py"
        result = sanitize_file_path(path)
        assert result == path

    def test_nested_relative_path_passes(self):
        """Nested relative path should pass."""
        path = "app/core/validation.py"
        result = sanitize_file_path(path)
        assert result == path

    def test_directory_traversal_blocked(self):
        """Directory traversal with .. should be blocked."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_file_path("../../../etc/passwd")
        assert "traversal" in str(exc_info.value).lower()

    def test_directory_traversal_in_middle_blocked(self):
        """Directory traversal in middle of path should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_file_path("app/../../../etc/passwd")

    def test_absolute_path_blocked(self):
        """Absolute paths should be blocked."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_file_path("/etc/passwd")
        assert "absolute" in str(exc_info.value).lower()

    def test_null_byte_blocked(self):
        """Null bytes in path should be blocked."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_file_path("file\x00.py")
        assert "null" in str(exc_info.value).lower()

    def test_windows_absolute_path_blocked(self):
        """Windows absolute paths should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_file_path("/C:/Windows/System32")


class TestValidateThreadId:
    """Tests for validate_thread_id - validates thread ID format."""

    def test_none_returns_none(self):
        """None input should return None."""
        result = validate_thread_id(None)
        assert result is None

    def test_valid_alphanumeric_passes(self):
        """Valid alphanumeric thread ID should pass."""
        thread_id = "thread123abc"
        result = validate_thread_id(thread_id)
        assert result == thread_id

    def test_hyphens_allowed(self):
        """Hyphens should be allowed."""
        thread_id = "thread-123-abc"
        result = validate_thread_id(thread_id)
        assert result == thread_id

    def test_underscores_allowed(self):
        """Underscores should be allowed."""
        thread_id = "thread_123_abc"
        result = validate_thread_id(thread_id)
        assert result == thread_id

    def test_mixed_format_allowed(self):
        """Mixed hyphens, underscores, and alphanumeric should be allowed."""
        thread_id = "user-123_session-abc_2024"
        result = validate_thread_id(thread_id)
        assert result == thread_id

    def test_invalid_characters_blocked(self):
        """Invalid characters should be blocked."""
        with pytest.raises(ValidationError) as exc_info:
            validate_thread_id("thread@123")
        assert "invalid characters" in str(exc_info.value).lower()

    def test_spaces_blocked(self):
        """Spaces should be blocked."""
        with pytest.raises(ValidationError):
            validate_thread_id("thread 123")

    def test_special_chars_blocked(self):
        """Special characters should be blocked."""
        with pytest.raises(ValidationError):
            validate_thread_id("thread#123")
        with pytest.raises(ValidationError):
            validate_thread_id("thread$123")
        with pytest.raises(ValidationError):
            validate_thread_id("thread%123")

    def test_directory_traversal_blocked(self):
        """Directory traversal should be blocked."""
        with pytest.raises(ValidationError):
            validate_thread_id("../../../etc/passwd")

    def test_max_length_enforced(self):
        """Thread ID exceeding max length should be rejected."""
        long_id = "x" * 257
        with pytest.raises(ValidationError) as exc_info:
            validate_thread_id(long_id)
        assert "too long" in str(exc_info.value).lower()

    def test_max_length_boundary(self):
        """Thread ID at exactly max length should pass."""
        thread_id = "x" * 256
        result = validate_thread_id(thread_id)
        assert result == thread_id

    def test_non_string_rejected(self):
        """Non-string input should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_thread_id(12345)
        assert "must be a string" in str(exc_info.value).lower()


class TestValidateFrameworkName:
    """Tests for validate_framework_name - validates framework names."""

    def test_none_returns_none(self):
        """None input should return None."""
        result = validate_framework_name(None)
        assert result is None

    def test_valid_framework_name_passes(self):
        """Valid framework name should pass."""
        name = "chain_of_thought"
        result = validate_framework_name(name)
        assert result == name

    def test_lowercase_alphanumeric_passes(self):
        """Lowercase alphanumeric with underscores should pass."""
        name = "active_inference_v2"
        result = validate_framework_name(name)
        assert result == name

    def test_uppercase_blocked(self):
        """Uppercase letters should be blocked."""
        with pytest.raises(ValidationError) as exc_info:
            validate_framework_name("ChainOfThought")
        assert "invalid characters" in str(exc_info.value).lower()

    def test_hyphens_blocked(self):
        """Hyphens should be blocked in framework names."""
        with pytest.raises(ValidationError):
            validate_framework_name("chain-of-thought")

    def test_spaces_blocked(self):
        """Spaces should be blocked."""
        with pytest.raises(ValidationError):
            validate_framework_name("chain of thought")

    def test_max_length_enforced(self):
        """Framework name exceeding max length (100) should be rejected."""
        long_name = "x" * 101
        with pytest.raises(ValidationError) as exc_info:
            validate_framework_name(long_name)
        assert "too long" in str(exc_info.value).lower()

    def test_starting_with_number_blocked(self):
        """Framework name starting with number should be blocked."""
        with pytest.raises(ValidationError) as exc_info:
            validate_framework_name("123_framework")
        assert "invalid characters" in str(exc_info.value).lower()

    def test_non_string_rejected(self):
        """Non-string input should be rejected."""
        with pytest.raises(ValidationError):
            validate_framework_name(123)


class TestValidateBoolean:
    """Tests for validate_boolean - validates and converts boolean values."""

    def test_true_returns_true(self):
        """True should return True."""
        result = validate_boolean(True)
        assert result is True

    def test_false_returns_false(self):
        """False should return False."""
        result = validate_boolean(False)
        assert result is False

    def test_string_true_converts(self):
        """String 'true' should convert to True."""
        assert validate_boolean("true") is True
        assert validate_boolean("TRUE") is True
        assert validate_boolean("True") is True

    def test_string_false_converts(self):
        """String 'false' should convert to False."""
        assert validate_boolean("false") is False
        assert validate_boolean("FALSE") is False
        assert validate_boolean("False") is False

    def test_string_yes_converts_to_true(self):
        """String 'yes' should convert to True."""
        assert validate_boolean("yes") is True
        assert validate_boolean("YES") is True

    def test_string_no_converts_to_false(self):
        """String 'no' should convert to False."""
        assert validate_boolean("no") is False
        assert validate_boolean("NO") is False

    def test_string_on_converts_to_true(self):
        """String 'on' should convert to True."""
        assert validate_boolean("on") is True
        assert validate_boolean("ON") is True

    def test_string_off_converts_to_false(self):
        """String 'off' should convert to False."""
        assert validate_boolean("off") is False
        assert validate_boolean("OFF") is False

    def test_string_1_converts_to_true(self):
        """String '1' should convert to True."""
        assert validate_boolean("1") is True

    def test_string_0_converts_to_false(self):
        """String '0' should convert to False."""
        assert validate_boolean("0") is False

    def test_integer_1_converts_to_true(self):
        """Integer 1 should convert to True."""
        assert validate_boolean(1) is True

    def test_integer_0_converts_to_false(self):
        """Integer 0 should convert to False."""
        assert validate_boolean(0) is False

    def test_invalid_string_rejected(self):
        """Invalid string should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_boolean("maybe")
        assert "must be a boolean" in str(exc_info.value).lower()

    def test_invalid_integer_rejected(self):
        """Integers other than 0 or 1 should be rejected."""
        with pytest.raises(ValidationError):
            validate_boolean(2)
        with pytest.raises(ValidationError):
            validate_boolean(-1)

    def test_none_rejected(self):
        """None should be rejected."""
        with pytest.raises(ValidationError):
            validate_boolean(None)

    def test_custom_field_name_in_error(self):
        """Custom field name should appear in error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_boolean("invalid", field_name="enable_logging")
        assert "enable_logging" in str(exc_info.value)


class TestValidateInteger:
    """Tests for validate_integer - validates integer values with ranges."""

    def test_valid_integer_passes(self):
        """Valid integer should pass."""
        result = validate_integer(42)
        assert result == 42

    def test_string_integer_converts(self):
        """String representation of integer should convert."""
        result = validate_integer("42")
        assert result == 42

    def test_negative_integer_passes(self):
        """Negative integer should pass if no min specified."""
        result = validate_integer(-10)
        assert result == -10

    def test_min_value_enforced(self):
        """Minimum value should be enforced."""
        with pytest.raises(ValidationError) as exc_info:
            validate_integer(5, min_value=10)
        assert ">=" in str(exc_info.value)

    def test_max_value_enforced(self):
        """Maximum value should be enforced."""
        with pytest.raises(ValidationError) as exc_info:
            validate_integer(100, max_value=50)
        assert "<=" in str(exc_info.value)

    def test_range_enforced(self):
        """Both min and max should be enforced."""
        result = validate_integer(50, min_value=0, max_value=100)
        assert result == 50

        with pytest.raises(ValidationError):
            validate_integer(-1, min_value=0, max_value=100)

        with pytest.raises(ValidationError):
            validate_integer(101, min_value=0, max_value=100)

    def test_float_rejected(self):
        """Float should be rejected."""
        with pytest.raises(ValidationError):
            validate_integer(42.5)

    def test_non_numeric_string_rejected(self):
        """Non-numeric string should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_integer("not a number")
        assert "must be an integer" in str(exc_info.value).lower()

    def test_custom_field_name_in_error(self):
        """Custom field name should appear in error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_integer("invalid", field_name="port_number")
        assert "port_number" in str(exc_info.value)


class TestValidateFloat:
    """Tests for validate_float - validates float values with ranges."""

    def test_valid_float_passes(self):
        """Valid float should pass."""
        result = validate_float(3.14)
        assert result == 3.14

    def test_integer_converts_to_float(self):
        """Integer should convert to float."""
        result = validate_float(42)
        assert result == 42.0
        assert isinstance(result, float)

    def test_string_float_converts(self):
        """String representation of float should convert."""
        result = validate_float("3.14")
        assert result == 3.14

    def test_negative_float_passes(self):
        """Negative float should pass if no min specified."""
        result = validate_float(-10.5)
        assert result == -10.5

    def test_min_value_enforced(self):
        """Minimum value should be enforced."""
        with pytest.raises(ValidationError) as exc_info:
            validate_float(0.5, min_value=1.0)
        assert ">=" in str(exc_info.value)

    def test_max_value_enforced(self):
        """Maximum value should be enforced."""
        with pytest.raises(ValidationError) as exc_info:
            validate_float(10.5, max_value=5.0)
        assert "<=" in str(exc_info.value)

    def test_range_enforced(self):
        """Both min and max should be enforced."""
        result = validate_float(0.5, min_value=0.0, max_value=1.0)
        assert result == 0.5

        with pytest.raises(ValidationError):
            validate_float(-0.1, min_value=0.0, max_value=1.0)

        with pytest.raises(ValidationError):
            validate_float(1.1, min_value=0.0, max_value=1.0)

    def test_non_numeric_string_rejected(self):
        """Non-numeric string should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_float("not a number")
        assert "must be a number" in str(exc_info.value).lower()

    def test_custom_field_name_in_error(self):
        """Custom field name should appear in error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_float("invalid", field_name="temperature")
        assert "temperature" in str(exc_info.value)


class TestSecurityEdgeCases:
    """Tests for security edge cases and advanced attack vectors."""

    def test_nested_script_tags(self):
        """Nested script tags should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("<div><script>alert(1)</script></div>")

    def test_svg_with_script(self):
        """SVG with embedded script should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("<svg onload=alert(1)>")

    def test_data_uri_with_javascript(self):
        """Data URIs with JavaScript should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("data:text/html,<script>alert(1)</script>")

    def test_comment_obfuscation(self):
        """Script tags with HTML comments should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("<!-- <script>alert(1)</script> -->")

    def test_multiple_injection_attempts(self):
        """Multiple injection patterns should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("<script>eval(atob('YWxlcnQoMSk='))</script>")

    def test_polyglot_injection(self):
        """Polyglot injection attempts should be blocked."""
        with pytest.raises(ValidationError):
            sanitize_query("javascript:/*--></title></style></textarea></script></xmp>")

    def test_very_long_attack_string(self):
        """Very long attack strings should be rejected."""
        long_attack = "<script>" + "a" * 20000 + "</script>"
        with pytest.raises(ValidationError):
            sanitize_query(long_attack)
