"""
Unit tests for token reduction integration.

Tests the integration of TOON and LLMLingua-2 for token optimization.
"""

import json
from unittest.mock import Mock, patch

import pytest

from app.core.token_reduction import (
    TokenReductionManager,
    deserialize_from_toon,
    get_manager,
    get_reduction_stats,
    reduce_tokens,
    serialize_to_toon,
)


class TestTokenReductionManager:
    """Test TokenReductionManager functionality."""

    def test_manager_singleton(self):
        """Test that get_manager returns singleton instance."""
        manager1 = get_manager()
        manager2 = get_manager()
        assert manager1 is manager2

    def test_serialize_to_toon(self):
        """Test TOON serialization through manager."""
        manager = get_manager()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]

        result = manager.serialize_to_toon(data)
        assert "{name|age}" in result
        assert "Alice|30" in result

    def test_deserialize_from_toon(self):
        """Test TOON deserialization through manager."""
        manager = get_manager()
        toon_str = """{name|age}
Alice|30
Bob|25"""

        result = manager.deserialize_from_toon(toon_str)
        assert len(result) == 2
        assert result[0]["name"] == "Alice"

    def test_get_format_comparison(self):
        """Test format comparison functionality."""
        manager = get_manager()
        data = [{"name": f"User{i}", "value": i} for i in range(10)]

        comparison = manager.get_format_comparison(data)

        assert "json" in comparison
        assert "toon" in comparison
        assert "savings" in comparison
        assert "recommendation" in comparison

        # TOON should be more efficient for uniform arrays
        assert comparison["toon"]["tokens"] <= comparison["json"]["tokens"]


class TestCompressPrompt:
    """Test prompt compression functionality."""

    @patch('app.core.token_reduction.llmlingua_available')
    @patch('app.core.token_reduction.get_compressor')
    def test_compress_prompt_success(self, mock_get_compressor, mock_available):
        """Test successful prompt compression."""
        mock_available.return_value = True

        mock_compressor = Mock()
        mock_compressor.compress.return_value = {
            "compressed_prompt": "compressed text",
            "origin_tokens": 100,
            "compressed_tokens": 50,
            "ratio": 0.5,
            "compressed": True
        }
        mock_get_compressor.return_value = mock_compressor

        manager = TokenReductionManager()
        manager._llmlingua_available = True

        result = manager.compress_prompt("This is a long prompt that should be compressed.")

        assert result["compressed"] is True
        assert result["compressed_tokens"] == 50
        assert result["ratio"] == 0.5

    def test_compress_prompt_disabled(self):
        """Test prompt compression when disabled."""
        with patch('app.core.token_reduction.get_settings') as mock_settings:
            mock_settings.return_value.enable_llmlingua_compression = False

            manager = TokenReductionManager()
            result = manager.compress_prompt("test prompt")

            assert result["compressed"] is False
            assert "reason" in result

    def test_compress_prompt_below_threshold(self):
        """Test prompt compression below minimum token threshold."""
        with patch('app.core.token_reduction.count_tokens', return_value=100):
            manager = TokenReductionManager()
            result = manager.compress_prompt("short", min_tokens=5000)

            assert result["compressed"] is False
            assert "Below minimum threshold" in result["reason"]


class TestReduceTokens:
    """Test automatic token reduction."""

    def test_reduce_tokens_json_data(self):
        """Test token reduction for JSON data."""
        json_data = json.dumps([
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ])

        result = reduce_tokens(json_data, content_type="structured")

        assert "content" in result
        assert "method" in result
        assert result["method"] == "toon"
        assert result["reduction_percent"] > 0

    def test_reduce_tokens_plain_text(self):
        """Test token reduction for plain text."""
        text = "This is plain text that cannot be compressed as JSON."

        result = reduce_tokens(text, content_type="text")

        # Should return original if no compression method applies
        assert result["content"] == text or result["method"] in ["llmlingua", "none"]

    def test_reduce_tokens_auto_detect(self):
        """Test auto-detection of best compression method."""
        json_data = json.dumps({"key": "value"})

        result = reduce_tokens(json_data, auto_detect=True)

        assert "method" in result
        assert "reduction_percent" in result


class TestReductionStats:
    """Test token reduction statistics."""

    def test_get_reduction_stats(self):
        """Test calculation of reduction statistics."""
        original = "This is a long text that will be compressed into something shorter."
        reduced = "Short text."

        stats = get_reduction_stats(original, reduced, method="llmlingua")

        assert "method" in stats
        assert stats["method"] == "llmlingua"
        assert "original_chars" in stats
        assert "reduced_chars" in stats
        assert "tokens_saved" in stats
        assert stats["reduction_percent"] > 0
        assert stats["original_chars"] > stats["reduced_chars"]

    def test_reduction_stats_no_reduction(self):
        """Test stats when no reduction occurred."""
        text = "Same text"

        stats = get_reduction_stats(text, text, method="none")

        assert stats["tokens_saved"] == 0
        assert stats["reduction_percent"] == 0
        assert stats["compression_ratio"] == 1.0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_serialize_to_toon_convenience(self):
        """Test serialize_to_toon convenience function."""
        data = [{"name": "Test"}]
        result = serialize_to_toon(data)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_deserialize_from_toon_convenience(self):
        """Test deserialize_from_toon convenience function."""
        toon_str = """{name}
Test"""

        result = deserialize_from_toon(toon_str)
        assert isinstance(result, list)
        assert len(result) > 0


class TestErrorHandling:
    """Test error handling in token reduction."""

    def test_serialize_invalid_json(self):
        """Test handling of serialization errors."""
        manager = get_manager()

        # Should fall back gracefully
        with patch('app.core.toon.TOONEncoder.encode', side_effect=Exception("Test error")):
            result = manager.serialize_to_toon({"key": "value"})
            # Should return JSON as fallback
            assert "key" in result

    def test_compress_prompt_error_handling(self):
        """Test error handling in prompt compression."""
        manager = TokenReductionManager()
        manager._llmlingua_available = True

        with patch('app.core.token_reduction.get_compressor', side_effect=Exception("Test error")):
            result = manager.compress_prompt("test")

            assert result["compressed"] is False
            assert "error" in result


@pytest.mark.parametrize("data,expected_method", [
    ([{"name": "Alice"}], "toon"),
    ("plain text", "none"),
])
def test_reduce_tokens_various_inputs(data, expected_method):
    """Test reduce_tokens with various input types."""
    if isinstance(data, str) and expected_method == "toon":
        data = json.dumps(data)

    result = reduce_tokens(data if isinstance(data, str) else json.dumps(data))

    assert result["method"] in ["toon", "llmlingua", "none"]


class TestIntegrationWithSettings:
    """Test integration with settings."""

    @patch('app.core.token_reduction.get_settings')
    def test_toon_enabled_setting(self, mock_get_settings):
        """Test TOON serialization respects settings."""
        mock_settings = Mock()
        mock_settings.enable_toon_serialization = False
        mock_get_settings.return_value = mock_settings

        manager = TokenReductionManager()
        data = [{"name": "Test"}]
        result = manager.serialize_to_toon(data)

        # Should use JSON when TOON is disabled
        assert result.startswith("[")

    @patch('app.core.token_reduction.get_settings')
    def test_llmlingua_settings(self, mock_get_settings):
        """Test LLMLingua compression respects settings."""
        mock_settings = Mock()
        mock_settings.enable_llmlingua_compression = True
        mock_settings.llmlingua_compression_rate = 0.6
        mock_settings.llmlingua_model_name = "test-model"
        mock_settings.llmlingua_device = "cpu"
        mock_settings.compression_min_tokens = 1000
        mock_get_settings.return_value = mock_settings

        manager = TokenReductionManager()

        assert manager.settings.enable_llmlingua_compression is True
        assert manager.settings.llmlingua_compression_rate == 0.6
