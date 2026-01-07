"""Tests for input validation utilities."""
import pytest
from server.handlers.validation import (
    ValidationError,
    validate_thread_id,
    validate_query,
    validate_path,
    validate_context,
    validate_code,
    validate_text,
    validate_positive_int,
    validate_framework_name,
    validate_category,
    validate_action,
    validate_file_list,
    validate_string_list,
    validate_boolean,
    validate_float,
)


class TestValidateThreadId:
    """Tests for validate_thread_id function."""

    def test_valid_thread_id(self):
        """Valid thread_id with alphanumeric, dash, and underscore."""
        assert validate_thread_id("abc-123_XYZ") == "abc-123_XYZ"

    def test_valid_simple_id(self):
        """Simple alphanumeric thread_id."""
        assert validate_thread_id("thread123") == "thread123"

    def test_empty_thread_id_returns_none(self):
        """Empty thread_id returns None when not required."""
        assert validate_thread_id("") is None
        assert validate_thread_id(None) is None

    def test_empty_thread_id_raises_when_required(self):
        """Empty thread_id raises ValidationError when required."""
        with pytest.raises(ValidationError, match="thread_id is required"):
            validate_thread_id("", required=True)

    def test_invalid_chars_raises(self):
        """Path traversal characters are rejected."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_thread_id("thread/../../../etc/passwd")

    def test_special_chars_raise(self):
        """Special characters are rejected."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_thread_id("thread@123")
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_thread_id("thread#id")
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_thread_id("thread id")  # space

    def test_too_long_raises(self):
        """Thread_id exceeding max length raises ValidationError."""
        with pytest.raises(ValidationError, match="too long"):
            validate_thread_id("x" * 300)

    def test_max_length_allowed(self):
        """Thread_id at exactly max length (256) is allowed."""
        long_id = "a" * 256
        assert validate_thread_id(long_id) == long_id

    def test_non_string_raises(self):
        """Non-string thread_id raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_thread_id(12345)


class TestValidateQuery:
    """Tests for validate_query function."""

    def test_valid_query(self):
        """Valid query is returned unchanged."""
        query = "How do I implement a binary search?"
        assert validate_query(query) == query

    def test_empty_query_raises(self):
        """Empty query raises ValidationError when required (default)."""
        with pytest.raises(ValidationError, match="query is required"):
            validate_query("")

    def test_empty_query_returns_empty_when_not_required(self):
        """Empty query returns empty string when not required."""
        assert validate_query("", required=False) == ""

    def test_too_long_raises(self):
        """Query exceeding max length raises ValidationError."""
        with pytest.raises(ValidationError, match="too long"):
            validate_query("x" * 60000)

    def test_custom_max_length(self):
        """Custom max_length is respected."""
        with pytest.raises(ValidationError, match="max 100 chars"):
            validate_query("x" * 101, max_length=100)

    def test_non_string_raises(self):
        """Non-string query raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_query(12345)


class TestValidatePath:
    """Tests for validate_path function."""

    def test_valid_path(self):
        """Valid file path is returned unchanged."""
        path = "/home/user/project/file.py"
        assert validate_path(path) == path

    def test_empty_path_returns_none(self):
        """Empty path returns None when not required."""
        assert validate_path("") is None
        assert validate_path(None) is None

    def test_empty_path_raises_when_required(self):
        """Empty path raises ValidationError when required."""
        with pytest.raises(ValidationError, match="is required"):
            validate_path("", required=True)

    def test_path_traversal_raises(self):
        """Path traversal sequences are rejected."""
        with pytest.raises(ValidationError, match="path traversal"):
            validate_path("/etc/../../../etc/passwd")
        with pytest.raises(ValidationError, match="path traversal"):
            validate_path("../../../etc/passwd")

    def test_null_bytes_raise(self):
        """Null bytes in path are rejected."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_path("/path/to\x00/file")

    def test_too_long_raises(self):
        """Path exceeding max length raises ValidationError."""
        with pytest.raises(ValidationError, match="too long"):
            validate_path("/" + "a" * 5000)

    def test_non_string_raises(self):
        """Non-string path raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_path(12345, required=True)


class TestValidateContext:
    """Tests for validate_context function."""

    def test_valid_context(self):
        """Valid context is returned unchanged."""
        context = "def foo(): pass"
        assert validate_context(context) == context

    def test_empty_context_returns_empty(self):
        """Empty context returns empty string."""
        assert validate_context("") == ""
        assert validate_context(None) == ""

    def test_too_long_raises(self):
        """Context exceeding max length raises ValidationError."""
        with pytest.raises(ValidationError, match="too long"):
            validate_context("x" * 200000)

    def test_custom_max_length(self):
        """Custom max_length is respected."""
        with pytest.raises(ValidationError, match="max 500 chars"):
            validate_context("x" * 501, max_length=500)

    def test_non_string_raises(self):
        """Non-string context raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_context(["code", "snippet"])


class TestValidateCode:
    """Tests for validate_code function."""

    def test_valid_code(self):
        """Valid code is returned unchanged."""
        code = "print('hello')"
        assert validate_code(code) == code

    def test_empty_code_raises(self):
        """Empty code raises ValidationError."""
        with pytest.raises(ValidationError, match="code is required"):
            validate_code("")

    def test_too_long_raises(self):
        """Code exceeding max length raises ValidationError."""
        with pytest.raises(ValidationError, match="too long"):
            validate_code("x" * 200000)


class TestValidateText:
    """Tests for validate_text function."""

    def test_valid_text(self):
        """Valid text is returned unchanged."""
        text = "Some text content"
        assert validate_text(text) == text

    def test_empty_text_raises_when_required(self):
        """Empty text raises ValidationError when required."""
        with pytest.raises(ValidationError, match="is required"):
            validate_text("")

    def test_empty_text_returns_empty_when_not_required(self):
        """Empty text returns empty string when not required."""
        assert validate_text("", required=False) == ""

    def test_custom_param_name_in_error(self):
        """Custom parameter name appears in error message."""
        with pytest.raises(ValidationError, match="description is required"):
            validate_text("", param_name="description")


class TestValidatePositiveInt:
    """Tests for validate_positive_int function."""

    def test_valid_int(self):
        """Valid positive integer is returned."""
        assert validate_positive_int(5, "count", 10) == 5

    def test_none_returns_default(self):
        """None returns the default value."""
        assert validate_positive_int(None, "count", 10) == 10

    def test_zero_raises(self):
        """Zero raises ValidationError (must be positive)."""
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_int(0, "count", 10)

    def test_negative_raises(self):
        """Negative value raises ValidationError."""
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_int(-5, "count", 10)

    def test_too_large_raises(self):
        """Value exceeding max_value raises ValidationError."""
        with pytest.raises(ValidationError, match="too large"):
            validate_positive_int(1500, "count", 10, max_value=1000)

    def test_boolean_raises(self):
        """Boolean value raises ValidationError."""
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_int(True, "count", 10)

    def test_non_int_raises(self):
        """Non-integer value raises ValidationError."""
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_int("5", "count", 10)


class TestValidateFrameworkName:
    """Tests for validate_framework_name function."""

    def test_valid_name(self):
        """Valid framework name is returned."""
        assert validate_framework_name("chain_of_thought") == "chain_of_thought"

    def test_empty_raises(self):
        """Empty name raises ValidationError."""
        with pytest.raises(ValidationError, match="is required"):
            validate_framework_name("")

    def test_too_long_raises(self):
        """Name exceeding max length raises ValidationError."""
        with pytest.raises(ValidationError, match="too long"):
            validate_framework_name("x" * 101)

    def test_invalid_start_raises(self):
        """Name starting with number raises ValidationError."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_framework_name("123_framework")

    def test_special_chars_raise(self):
        """Name with special characters raises ValidationError."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_framework_name("chain-of-thought")  # dash not allowed


class TestValidateCategory:
    """Tests for validate_category function."""

    def test_valid_category(self):
        """Valid category is returned."""
        valid = ["strategy", "search", "iterative"]
        assert validate_category("strategy", valid) == "strategy"

    def test_invalid_category_raises(self):
        """Invalid category raises ValidationError."""
        valid = ["strategy", "search", "iterative"]
        with pytest.raises(ValidationError, match="Invalid category"):
            validate_category("unknown", valid)

    def test_empty_category_raises(self):
        """Empty category raises ValidationError."""
        with pytest.raises(ValidationError, match="is required"):
            validate_category("", ["a", "b"])


class TestValidateAction:
    """Tests for validate_action function."""

    def test_valid_action(self):
        """Valid action is returned."""
        valid = ["create", "update", "delete"]
        assert validate_action("create", valid) == "create"

    def test_invalid_action_raises(self):
        """Invalid action raises ValidationError."""
        valid = ["create", "update", "delete"]
        with pytest.raises(ValidationError, match="Invalid action"):
            validate_action("destroy", valid)


class TestValidateFileList:
    """Tests for validate_file_list function."""

    def test_valid_file_list(self):
        """Valid file list is returned."""
        files = ["/path/to/file1.py", "/path/to/file2.py"]
        assert validate_file_list(files) == files

    def test_empty_list_returns_none(self):
        """Empty list returns None."""
        assert validate_file_list([]) is None
        assert validate_file_list(None) is None

    def test_too_many_files_raises(self):
        """Too many files raises ValidationError."""
        files = [f"/path/to/file{i}.py" for i in range(150)]
        with pytest.raises(ValidationError, match="too long"):
            validate_file_list(files)

    def test_non_list_raises(self):
        """Non-list raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a list"):
            validate_file_list("/path/to/file.py")

    def test_non_string_item_raises(self):
        """Non-string item in list raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_file_list(["/path/to/file.py", 123])

    def test_path_traversal_in_item_raises(self):
        """Path traversal in item raises ValidationError."""
        with pytest.raises(ValidationError, match="path traversal"):
            validate_file_list(["/path/../../../etc/passwd"])


class TestValidateStringList:
    """Tests for validate_string_list function."""

    def test_valid_list(self):
        """Valid string list is returned."""
        items = ["item1", "item2", "item3"]
        assert validate_string_list(items, "tags") == items

    def test_empty_list_returns_none(self):
        """Empty list returns None."""
        assert validate_string_list([], "tags") is None

    def test_too_many_items_raises(self):
        """Too many items raises ValidationError."""
        items = [f"item{i}" for i in range(150)]
        with pytest.raises(ValidationError, match="too long"):
            validate_string_list(items, "tags")

    def test_item_too_long_raises(self):
        """Item exceeding max length raises ValidationError."""
        items = ["short", "x" * 1500]
        with pytest.raises(ValidationError, match="too long"):
            validate_string_list(items, "tags")


class TestValidateBoolean:
    """Tests for validate_boolean function."""

    def test_true_value(self):
        """True is returned as True."""
        assert validate_boolean(True, "flag", False) is True

    def test_false_value(self):
        """False is returned as False."""
        assert validate_boolean(False, "flag", True) is False

    def test_none_returns_default(self):
        """None returns the default value."""
        assert validate_boolean(None, "flag", True) is True
        assert validate_boolean(None, "flag", False) is False

    def test_non_bool_raises(self):
        """Non-boolean value raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a boolean"):
            validate_boolean("true", "flag", False)
        with pytest.raises(ValidationError, match="must be a boolean"):
            validate_boolean(1, "flag", False)


class TestValidateFloat:
    """Tests for validate_float function."""

    def test_valid_float(self):
        """Valid float within range is returned."""
        assert validate_float(0.5, "temperature", 0.7) == 0.5

    def test_int_converted_to_float(self):
        """Integer is converted to float."""
        result = validate_float(1, "temperature", 0.7)
        assert result == 1.0
        assert isinstance(result, float)

    def test_none_returns_default(self):
        """None returns the default value."""
        assert validate_float(None, "temperature", 0.7) == 0.7

    def test_below_min_raises(self):
        """Value below min raises ValidationError."""
        with pytest.raises(ValidationError, match="must be between"):
            validate_float(-0.5, "temperature", 0.7)

    def test_above_max_raises(self):
        """Value above max raises ValidationError."""
        with pytest.raises(ValidationError, match="must be between"):
            validate_float(1.5, "temperature", 0.7)

    def test_custom_range(self):
        """Custom min/max range is respected."""
        assert validate_float(50.0, "score", 0.0, min_value=0.0, max_value=100.0) == 50.0
        with pytest.raises(ValidationError, match="must be between"):
            validate_float(150.0, "score", 0.0, min_value=0.0, max_value=100.0)

    def test_non_number_raises(self):
        """Non-numeric value raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate_float("0.5", "temperature", 0.7)
