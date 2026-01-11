"""
Unit tests for StreamingContextGateway

Tests streaming context preparation with progress events and cancellation.
"""

import asyncio
import contextlib
from datetime import datetime

import pytest

from app.core.context.enhanced_models import ProgressEvent, ProgressStatus
from app.core.context.streaming_gateway import PerformanceTracker, StreamingContextGateway


class TestPerformanceTracker:
    """Test PerformanceTracker for completion time estimation."""

    def test_record_and_estimate(self):
        """Test recording executions and estimating completion time."""
        tracker = PerformanceTracker()

        # Record some executions
        tracker.record_execution("file_discovery", 100, 5.0)
        tracker.record_execution("file_discovery", 120, 6.0)
        tracker.record_execution("file_discovery", 90, 4.5)

        # Estimate for similar workspace size
        estimate = tracker.estimate_completion_time("file_discovery", 100)

        # Should be close to recorded times
        assert 4.0 <= estimate <= 7.0

    def test_estimate_without_history(self):
        """Test estimation when no history exists."""
        tracker = PerformanceTracker()

        # Should return default estimate
        estimate = tracker.estimate_completion_time("file_discovery", 100)
        assert estimate > 0

    def test_history_limit(self):
        """Test that history is limited to max size."""
        tracker = PerformanceTracker()

        # Record more than max history
        for _ in range(150):
            tracker.record_execution("test_component", 100, 1.0)

        # Should only keep last 100
        assert len(tracker._history) == 100


class TestStreamingContextGateway:
    """Test StreamingContextGateway functionality."""

    @pytest.mark.asyncio
    async def test_progress_events_emitted(self):
        """Test that progress events are emitted during context preparation."""
        gateway = StreamingContextGateway()
        events: list[ProgressEvent] = []

        def progress_callback(event: ProgressEvent):
            events.append(event)

        cancellation_token = asyncio.Event()

        # This will likely fail due to missing Gemini API, but we can check events
        with contextlib.suppress(Exception):
            await gateway.prepare_context_streaming(
                query="Test query",
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
                workspace_path=None,  # No workspace to avoid file system operations
                search_docs=False,  # Disable doc search
            )

        # Should have emitted at least the initial event
        assert len(events) > 0
        assert events[0].component == "overall"
        assert events[0].status == ProgressStatus.STARTED

    @pytest.mark.asyncio
    async def test_cancellation_before_start(self):
        """Test cancellation before context preparation starts."""
        gateway = StreamingContextGateway()
        events: list[ProgressEvent] = []

        def progress_callback(event: ProgressEvent):
            events.append(event)

        # Set cancellation token before starting
        cancellation_token = asyncio.Event()
        cancellation_token.set()

        # Should return partial context immediately
        context = await gateway.prepare_context_streaming(
            query="Test query",
            progress_callback=progress_callback,
            cancellation_token=cancellation_token,
            workspace_path=None,
            search_docs=False,
        )

        # Should have basic context
        assert context is not None
        assert context.task_summary == "Test query"

        # Should have emitted start event
        assert len(events) >= 1
        assert events[0].status == ProgressStatus.STARTED

    @pytest.mark.asyncio
    async def test_partial_context_creation(self):
        """Test that partial context is created correctly."""
        gateway = StreamingContextGateway()

        # Create partial context with empty data
        partial = gateway._create_partial_context(
            query="Test query",
            files=[],
            docs=[],
            code_searches=[],
            component_status={}
        )

        # Should have basic structure
        assert partial is not None
        assert partial.task_summary == "Test query"
        assert partial.task_type in ["debug", "implement", "refactor", "explain", "general"]
        assert len(partial.relevant_files) == 0
        assert len(partial.documentation) == 0

    def test_emit_progress(self):
        """Test progress event emission."""
        gateway = StreamingContextGateway()
        events: list[ProgressEvent] = []

        def callback(event: ProgressEvent):
            events.append(event)

        # Emit a progress event
        gateway._emit_progress(
            component="test_component",
            status=ProgressStatus.PROGRESS,
            progress=0.5,
            data={"test": "data"},
            message="Test message",
            estimated_completion=5.0,
            callback=callback
        )

        # Should have emitted one event
        assert len(events) == 1
        event = events[0]
        assert event.component == "test_component"
        assert event.status == ProgressStatus.PROGRESS
        assert event.progress == 0.5
        assert event.data == {"test": "data"}
        assert event.message == "Test message"
        assert event.estimated_completion == 5.0
        assert isinstance(event.timestamp, datetime)

    def test_emit_progress_with_exception(self):
        """Test that progress emission handles callback exceptions gracefully."""
        gateway = StreamingContextGateway()

        def failing_callback(event: ProgressEvent):
            raise ValueError("Callback failed")

        # Should not raise exception
        gateway._emit_progress(
            component="test",
            status=ProgressStatus.STARTED,
            progress=0.0,
            callback=failing_callback
        )

        # Test passes if no exception is raised
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
