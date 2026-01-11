"""
Unit tests for TOON (Token-Oriented Object Notation) serialization.

Tests the TOON encoder/decoder and token reduction capabilities.
"""

import json

import pytest

from app.core.toon import (
    TOONDecoder,
    TOONEncoder,
    from_toon,
    get_compression_ratio,
    get_token_savings,
    to_toon,
)


class TestTOONEncoder:
    """Test TOON encoding functionality."""

    def test_encode_simple_array(self):
        """Test encoding a simple array of uniform objects."""
        data = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "SF"}
        ]

        encoder = TOONEncoder()
        result = encoder.encode(data)

        # Should produce tabular format
        assert "{name|age|city}" in result
        assert "Alice|30|NYC" in result
        assert "Bob|25|SF" in result

    def test_encode_preserves_data(self):
        """Test that encoding-decoding round trip preserves data."""
        data = [
            {"id": 1, "name": "Test", "value": 42.5},
            {"id": 2, "name": "Example", "value": 10.0}
        ]

        encoder = TOONEncoder()
        decoder = TOONDecoder()

        toon_str = encoder.encode(data)
        decoded = decoder.decode(toon_str)

        assert len(decoded) == 2
        assert decoded[0]["name"] == "Test"
        assert decoded[1]["value"] == 10.0

    def test_encode_with_escaped_delimiter(self):
        """Test that delimiter in strings is properly escaped."""
        data = [
            {"name": "Alice|Bob", "value": "test|case"}
        ]

        encoder = TOONEncoder(delimiter="|")
        result = encoder.encode(data)

        # Delimiter should be escaped
        assert "Alice\\|Bob" in result
        assert "test\\|case" in result

    def test_encode_empty_array(self):
        """Test encoding empty array."""
        encoder = TOONEncoder()
        result = encoder.encode([])
        assert result == "[]"

    def test_encode_primitive_array(self):
        """Test encoding array of primitives."""
        encoder = TOONEncoder()
        result = encoder.encode([1, 2, 3, 4, 5])
        assert result == "[1,2,3,4,5]"

    def test_encode_nested_object(self):
        """Test encoding nested objects."""
        data = {
            "name": "Test",
            "nested": {
                "key": "value"
            }
        }

        encoder = TOONEncoder()
        result = encoder.encode(data)
        # Should handle nested structure
        assert "name" in result
        assert "Test" in result


class TestTOONDecoder:
    """Test TOON decoding functionality."""

    def test_decode_tabular_array(self):
        """Test decoding tabular TOON format."""
        toon_str = """{name|age|city}
Alice|30|NYC
Bob|25|SF"""

        decoder = TOONDecoder()
        result = decoder.decode(toon_str)

        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[0]["age"] == 30
        assert result[0]["city"] == "NYC"
        assert result[1]["name"] == "Bob"

    def test_decode_with_escaped_delimiter(self):
        """Test decoding with escaped delimiters."""
        toon_str = """{name|value}
Alice\\|Bob|test\\|case"""

        decoder = TOONDecoder(delimiter="|")
        result = decoder.decode(toon_str)

        assert len(result) == 1
        assert result[0]["name"] == "Alice|Bob"
        assert result[0]["value"] == "test|case"

    def test_decode_null_values(self):
        """Test decoding null values."""
        toon_str = """{name|value}
Alice|null
Bob|42"""

        decoder = TOONDecoder()
        result = decoder.decode(toon_str)

        assert result[0]["value"] is None
        assert result[1]["value"] == 42

    def test_decode_boolean_values(self):
        """Test decoding boolean values."""
        toon_str = """{name|active}
Alice|true
Bob|false"""

        decoder = TOONDecoder()
        result = decoder.decode(toon_str)

        assert result[0]["active"] is True
        assert result[1]["active"] is False


class TestTOONConvenienceFunctions:
    """Test convenience functions."""

    def test_to_toon_from_toon_roundtrip(self):
        """Test roundtrip with convenience functions."""
        data = [
            {"id": 1, "name": "Test"},
            {"id": 2, "name": "Example"}
        ]

        toon_str = to_toon(data)
        decoded = from_toon(toon_str)

        assert decoded == data

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        json_str = json.dumps([
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ])

        toon_str = to_toon(json.loads(json_str))
        ratio = get_compression_ratio(json_str, toon_str)

        # TOON should be more compact
        assert ratio < 1.0
        assert len(toon_str) < len(json_str)

    def test_token_savings(self):
        """Test token savings statistics."""
        data = [{"name": f"User{i}", "value": i} for i in range(10)]
        json_str = json.dumps(data)
        toon_str = to_toon(data)

        stats = get_token_savings(json_str, toon_str)

        assert "original_chars" in stats
        assert "compressed_chars" in stats
        assert "reduction_percent" in stats
        assert stats["saved_chars"] > 0
        assert stats["reduction_percent"] > 0


class TestTOONEdgeCases:
    """Test edge cases and error handling."""

    def test_mixed_array(self):
        """Test array with mixed structures."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "city": "NYC"}  # Different keys
        ]

        encoder = TOONEncoder()
        # Should fall back to standard format for non-uniform arrays
        result = encoder.encode(data)
        assert result is not None

    def test_array_below_threshold(self):
        """Test array below tabular threshold."""
        data = [{"name": "Alice", "age": 30}]  # Only 1 item

        encoder = TOONEncoder(threshold=2)
        result = encoder.encode(data)

        # Should not use tabular format
        assert "{name|age}" not in result

    def test_custom_delimiter(self):
        """Test using custom delimiter."""
        data = [{"a": 1, "b": 2}]

        encoder = TOONEncoder(delimiter=";", threshold=1)
        result = encoder.encode(data)

        assert "{a;b}" in result
        assert "1;2" in result


@pytest.mark.parametrize("data", [
    [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
    [{"id": i, "value": i * 10} for i in range(5)],
    {"key": "value", "number": 42},
    [1, 2, 3, 4, 5],
])
def test_toon_roundtrip_various_data(data):
    """Test TOON roundtrip with various data structures."""
    toon_str = to_toon(data)
    decoded = from_toon(toon_str)

    # Handle potential type differences (int vs float)
    if isinstance(decoded, list) and decoded:
        if isinstance(decoded[0], dict):
            # For dictionaries, compare values
            for orig, dec in zip(data if isinstance(data, list) else [data], decoded):
                for key in orig:
                    assert orig[key] == dec[key] or (
                        isinstance(orig[key], (int, float)) and
                        isinstance(dec[key], (int, float)) and
                        float(orig[key]) == float(dec[key])
                    )
        else:
            # For primitive arrays
            assert decoded == data
    else:
        assert decoded == data
