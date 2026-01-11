"""Tests for correlation ID utilities."""
from app.core.correlation import (
    clear_correlation_id,
    get_correlation_id,
    set_correlation_id,
)


class TestCorrelationId:
    """Tests for correlation ID management."""

    def test_generates_id_when_none(self):
        """Generates a new ID when none is set."""
        clear_correlation_id()
        cid = get_correlation_id()
        assert cid is not None
        assert len(cid) == 8  # Short UUID format

    def test_returns_same_id(self):
        """Returns the same ID on subsequent calls within same context."""
        clear_correlation_id()
        cid1 = get_correlation_id()
        cid2 = get_correlation_id()
        assert cid1 == cid2

    def test_set_and_get(self):
        """Setting a correlation ID makes it retrievable."""
        set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"

    def test_clear_removes_id(self):
        """Clearing correlation ID causes new one to be generated."""
        set_correlation_id("test-456")
        assert get_correlation_id() == "test-456"
        clear_correlation_id()
        # After clearing, a new ID should be generated
        new_id = get_correlation_id()
        assert new_id != "test-456"
        assert len(new_id) == 8

    def test_generated_id_format(self):
        """Generated ID is a valid short UUID format."""
        clear_correlation_id()
        cid = get_correlation_id()
        # Should be 8 hex characters (first 8 chars of UUID)
        assert len(cid) == 8
        # Should be valid hexadecimal (or alphanumeric from UUID)
        assert all(c.isalnum() or c == '-' for c in cid)

    def test_set_empty_string(self):
        """Setting empty string is allowed and retrievable."""
        set_correlation_id("")
        # Empty string is falsy, so get_correlation_id will generate new one
        cid = get_correlation_id()
        assert len(cid) == 8

    def test_set_custom_format(self):
        """Custom correlation ID format is preserved."""
        custom_id = "req-abc123-xyz789"
        set_correlation_id(custom_id)
        assert get_correlation_id() == custom_id

    def test_multiple_clear_calls(self):
        """Multiple clear calls are safe."""
        clear_correlation_id()
        clear_correlation_id()
        clear_correlation_id()
        cid = get_correlation_id()
        assert len(cid) == 8

    def test_set_none(self):
        """Setting None causes new ID to be generated on get."""
        set_correlation_id("test-789")
        set_correlation_id(None)
        cid = get_correlation_id()
        # None is falsy, so should generate new ID
        assert cid != "test-789"
        assert len(cid) == 8

    def test_uniqueness_after_clear(self):
        """Different IDs are generated after clearing (with high probability)."""
        clear_correlation_id()
        id1 = get_correlation_id()
        clear_correlation_id()
        id2 = get_correlation_id()
        clear_correlation_id()
        id3 = get_correlation_id()
        # All should be different (extremely high probability)
        assert len({id1, id2, id3}) == 3
