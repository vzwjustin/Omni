"""Tests for ResilientSampler with timeout, retry, and circuit breaker."""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.sampling import (
    ResilientSampler,
    ClientSampler,
    SamplingNotSupportedError,
    extract_score,
    extract_json_object,
)
from app.nodes.common import extract_code_blocks
from app.core.errors import SamplerTimeout, SamplerCircuitOpen, LLMError


class TestResilientSampler:
    """Tests for ResilientSampler wrapper class."""

    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Successful request returns response and resets failure count."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(return_value="response")

        resilient = ResilientSampler(mock_sampler)
        result = await resilient.request_sample("prompt")

        assert result == "response"
        assert resilient.failure_count == 0
        mock_sampler.request_sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        """Request that times out raises SamplerTimeout after retries."""
        mock_sampler = MagicMock(spec=ClientSampler)

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(10)
            return "response"

        mock_sampler.request_sample = slow_response

        resilient = ResilientSampler(mock_sampler, timeout=0.1, max_retries=1)
        with pytest.raises(SamplerTimeout):
            await resilient.request_sample("prompt")

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Sampler retries on transient failures."""
        mock_sampler = MagicMock(spec=ClientSampler)
        call_count = 0

        async def flaky_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            return "success"

        mock_sampler.request_sample = flaky_response

        resilient = ResilientSampler(mock_sampler, max_retries=3, timeout=5.0)
        result = await resilient.request_sample("prompt")

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_threshold(self):
        """Circuit breaker opens after reaching failure threshold."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(side_effect=Exception("Always fails"))

        resilient = ResilientSampler(
            mock_sampler,
            max_retries=2,
            circuit_threshold=4,
            circuit_reset_time=1.0,
            timeout=0.5
        )

        # First call will fail after 2 retries, adding 2 to failure count
        with pytest.raises(LLMError):
            await resilient.request_sample("prompt")

        # Second call will fail after 2 more retries, hitting threshold
        with pytest.raises(LLMError):
            await resilient.request_sample("prompt")

        # Now circuit should be open
        assert resilient.is_circuit_open
        assert resilient.failure_count >= 4

    @pytest.mark.asyncio
    async def test_circuit_open_raises_immediately(self):
        """When circuit is open, requests fail immediately."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(side_effect=Exception("Fail"))

        resilient = ResilientSampler(
            mock_sampler,
            max_retries=1,
            circuit_threshold=1,
            circuit_reset_time=60.0,
            timeout=0.5
        )

        # Trigger circuit open
        with pytest.raises(LLMError):
            await resilient.request_sample("prompt")

        # Circuit should now be open
        assert resilient.is_circuit_open

        # Next call should fail immediately with SamplerCircuitOpen
        with pytest.raises(SamplerCircuitOpen):
            await resilient.request_sample("prompt")

    @pytest.mark.asyncio
    async def test_sampling_not_supported_propagates_immediately(self):
        """SamplingNotSupportedError is not retried."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(
            side_effect=SamplingNotSupportedError("Not supported")
        )

        resilient = ResilientSampler(mock_sampler, max_retries=3)

        with pytest.raises(SamplingNotSupportedError):
            await resilient.request_sample("prompt")

        # Should only be called once (no retries)
        mock_sampler.request_sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker(self):
        """Manual circuit breaker reset works."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(side_effect=Exception("Fail"))

        resilient = ResilientSampler(
            mock_sampler,
            max_retries=1,
            circuit_threshold=1,
            timeout=0.5
        )

        # Trigger circuit open
        with pytest.raises(LLMError):
            await resilient.request_sample("prompt")

        assert resilient.is_circuit_open

        # Reset circuit
        resilient.reset_circuit_breaker()

        assert not resilient.is_circuit_open
        assert resilient.failure_count == 0

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        """Successful request resets the failure count."""
        mock_sampler = MagicMock(spec=ClientSampler)
        call_count = 0

        async def eventually_succeeds(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First call fails")
            return "success"

        mock_sampler.request_sample = eventually_succeeds

        resilient = ResilientSampler(mock_sampler, max_retries=2, timeout=5.0)
        result = await resilient.request_sample("prompt")

        assert result == "success"
        assert resilient.failure_count == 0  # Reset on success

    @pytest.mark.asyncio
    async def test_timeout_override(self):
        """Per-request timeout override works."""
        mock_sampler = MagicMock(spec=ClientSampler)

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.5)
            return "response"

        mock_sampler.request_sample = slow_response

        resilient = ResilientSampler(mock_sampler, timeout=5.0, max_retries=1)

        # Override with short timeout should fail
        with pytest.raises(SamplerTimeout):
            await resilient.request_sample("prompt", timeout=0.1)

    def test_is_circuit_open_property(self):
        """is_circuit_open property correctly reflects state."""
        mock_sampler = MagicMock(spec=ClientSampler)
        resilient = ResilientSampler(mock_sampler)

        # Initially not open
        assert not resilient.is_circuit_open

        # Manually set circuit open time in the future
        resilient._circuit_open_until = time.time() + 60
        assert resilient.is_circuit_open

        # Set to past time
        resilient._circuit_open_until = time.time() - 1
        assert not resilient.is_circuit_open

    def test_failure_count_property(self):
        """failure_count property returns current count."""
        mock_sampler = MagicMock(spec=ClientSampler)
        resilient = ResilientSampler(mock_sampler)

        assert resilient.failure_count == 0

        resilient._failure_count = 5
        assert resilient.failure_count == 5


class TestExtractScore:
    """Tests for extract_score helper function."""

    def test_score_with_slash(self):
        """Extracts score from 'X/Y' format."""
        assert extract_score("Score: 8/10") == 0.8
        assert extract_score("Rating: 7/10") == 0.7

    def test_out_of_format(self):
        """Extracts score from 'X out of Y' format."""
        assert extract_score("8 out of 10") == 0.8
        assert extract_score("Rating: 9 out of 10") == 0.9

    def test_decimal_score(self):
        """Extracts decimal score."""
        assert extract_score("Score: 8.5") == 0.85
        assert extract_score("Rating: 7.5/10") == 0.75

    def test_normalized_score(self):
        """Already normalized score (0-1) is preserved."""
        assert extract_score("0.85") == 0.85
        assert extract_score("Score: 0.7") == 0.7

    def test_plain_number(self):
        """Plain number is normalized from 0-10 scale."""
        assert extract_score("8") == 0.8
        assert extract_score("5") == 0.5

    def test_no_score_returns_default(self):
        """Returns default when no score found."""
        assert extract_score("No score here") == 0.5
        assert extract_score("Just some text", default=0.3) == 0.3

    def test_clamped_to_range(self):
        """Score is clamped to 0.0-1.0 range."""
        assert extract_score("Score: 15/10") == 1.0
        assert extract_score("Score: -5/10") == 0.0


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks helper function."""

    def test_single_code_block(self):
        """Extracts single code block."""
        text = """Here is some code:
```python
def hello():
    print("Hello")
```
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert "def hello():" in blocks[0]

    def test_multiple_code_blocks(self):
        """Extracts multiple code blocks."""
        text = """
```python
code1()
```

```javascript
code2();
```
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2
        assert "code1()" in blocks[0]
        assert "code2();" in blocks[1]

    def test_no_language_specified(self):
        """Extracts code block without language specification."""
        text = """
```
plain code
```
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert "plain code" in blocks[0]

    def test_no_code_blocks(self):
        """Returns empty list when no code blocks found."""
        text = "Just regular text without code"
        blocks = extract_code_blocks(text)
        assert blocks == []

    def test_strips_whitespace(self):
        """Strips leading/trailing whitespace from code blocks."""
        text = """
```python

  code()

```
"""
        blocks = extract_code_blocks(text)
        assert blocks[0] == "code()"


class TestExtractJsonObject:
    """Tests for extract_json_object helper function."""

    def test_json_in_code_block(self):
        """Extracts JSON from markdown code block."""
        text = """
```json
{"key": "value", "number": 42}
```
"""
        result = extract_json_object(text)
        assert result == {"key": "value", "number": 42}

    def test_raw_json(self):
        """Extracts raw JSON string."""
        text = '{"key": "value"}'
        result = extract_json_object(text)
        assert result == {"key": "value"}

    def test_json_in_text(self):
        """Extracts JSON embedded in other text."""
        text = 'The result is {"status": "ok", "count": 5} for this query.'
        result = extract_json_object(text)
        assert result["status"] == "ok"
        assert result["count"] == 5

    def test_no_json_returns_none(self):
        """Returns None when no valid JSON found."""
        text = "No JSON here at all"
        assert extract_json_object(text) is None

    def test_invalid_json_returns_none(self):
        """Returns None for invalid JSON."""
        text = "{key: value}"  # Missing quotes
        assert extract_json_object(text) is None

    def test_nested_json(self):
        """Extracts nested JSON objects."""
        text = '{"outer": {"inner": "value"}}'
        result = extract_json_object(text)
        assert result == {"outer": {"inner": "value"}}
