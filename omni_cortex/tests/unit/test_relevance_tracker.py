"""
Unit tests for Context Relevance Tracker

Tests the relevance tracking functionality that monitors which context
elements are most valuable for different task types.
"""

import pytest
from datetime import datetime, timedelta
from app.core.context.relevance_tracker import (
    RelevanceTracker,
    ElementUsage,
    ContextUsageSession,
)


class TestRelevanceTracker:
    """Test suite for RelevanceTracker."""
    
    def test_record_context_preparation(self):
        """Test recording context preparation with files and docs."""
        tracker = RelevanceTracker()
        
        files = [
            {"path": "app/main.py", "relevance_score": 0.9},
            {"path": "app/utils.py", "relevance_score": 0.7},
        ]
        
        docs = [
            {"source": "https://docs.python.org/3/", "relevance_score": 0.8},
        ]
        
        code_search = [
            {"search_type": "grep", "query": "def authenticate"},
        ]
        
        tracker.record_context_preparation(
            session_id="test_session_1",
            query="Fix authentication bug",
            files=files,
            documentation=docs,
            code_search=code_search,
            task_type="debug",
            complexity="medium"
        )
        
        # Verify elements were tracked
        assert len(tracker._element_usage) == 4
        assert "file:app/main.py" in tracker._element_usage
        assert "file:app/utils.py" in tracker._element_usage
        assert "doc:https://docs.python.org/3/" in tracker._element_usage
        
        # Verify session was recorded
        assert len(tracker._sessions) == 1
        session = tracker._sessions[0]
        assert session.session_id == "test_session_1"
        assert session.task_type == "debug"
        assert len(session.included_elements) == 4
    
    def test_record_solution_usage(self):
        """Test detecting element usage in solution text."""
        tracker = RelevanceTracker()
        
        # Record context preparation
        files = [
            {"path": "app/main.py", "relevance_score": 0.9},
            {"path": "app/utils.py", "relevance_score": 0.7},
        ]
        
        tracker.record_context_preparation(
            session_id="test_session_2",
            query="Fix bug",
            files=files,
            documentation=[],
            code_search=[],
            task_type="debug"
        )
        
        # Record solution that references main.py but not utils.py
        solution = """
        I fixed the bug in app/main.py by updating the authentication logic.
        Here's the change:
        
        ```python
        # app/main.py
        def authenticate(user):
            return validate_credentials(user)
        ```
        """
        
        usage_counts = tracker.record_solution_usage("test_session_2", solution)
        
        # Verify usage detection
        assert "file:app/main.py" in usage_counts
        assert usage_counts["file:app/main.py"] > 0
        
        # Verify usage statistics
        main_usage = tracker._element_usage["file:app/main.py"]
        assert main_usage.times_used == 1
        assert main_usage.times_included == 1
        assert main_usage.usage_rate == 1.0
        
        utils_usage = tracker._element_usage["file:app/utils.py"]
        assert utils_usage.times_used == 0
        assert utils_usage.times_included == 1
        assert utils_usage.usage_rate == 0.0
    
    def test_get_element_statistics(self):
        """Test retrieving element statistics."""
        tracker = RelevanceTracker()
        
        # Create some usage data
        tracker._element_usage = {
            "file:high_value.py": ElementUsage(
                element_id="file:high_value.py",
                element_type="file",
                times_included=10,
                times_used=9,
                usage_rate=0.9
            ),
            "file:low_value.py": ElementUsage(
                element_id="file:low_value.py",
                element_type="file",
                times_included=10,
                times_used=1,
                usage_rate=0.1
            ),
            "doc:some_doc": ElementUsage(
                element_id="doc:some_doc",
                element_type="documentation",
                times_included=5,
                times_used=4,
                usage_rate=0.8
            ),
        }
        
        # Get all file statistics
        file_stats = tracker.get_element_statistics(element_type="file")
        assert len(file_stats) == 2
        assert file_stats[0].element_id == "file:high_value.py"  # Sorted by usage rate
        assert file_stats[1].element_id == "file:low_value.py"
        
        # Get high-value elements only
        high_value = tracker.get_element_statistics(min_usage_rate=0.5)
        assert len(high_value) == 2
        assert all(e.usage_rate >= 0.5 for e in high_value)
    
    def test_get_relevance_feedback(self):
        """Test getting relevance feedback for optimization."""
        tracker = RelevanceTracker()
        
        # Create diverse usage data
        tracker._element_usage = {
            "file:high1.py": ElementUsage(
                element_id="file:high1.py",
                element_type="file",
                times_included=10,
                times_used=9,
                usage_rate=0.9,
                avg_relevance_score=0.8
            ),
            "file:low1.py": ElementUsage(
                element_id="file:low1.py",
                element_type="file",
                times_included=10,
                times_used=1,
                usage_rate=0.1,
                avg_relevance_score=0.6
            ),
            "doc:high_doc": ElementUsage(
                element_id="doc:high_doc",
                element_type="documentation",
                times_included=5,
                times_used=5,
                usage_rate=1.0
            ),
        }
        
        feedback = tracker.get_relevance_feedback(top_n=10)
        
        # Verify feedback structure
        assert "high_value_files" in feedback
        assert "low_value_files" in feedback
        assert "high_value_docs" in feedback
        assert "overall_stats" in feedback
        
        # Verify high-value files
        assert len(feedback["high_value_files"]) == 1
        assert feedback["high_value_files"][0]["path"] == "high1.py"
        assert feedback["high_value_files"][0]["usage_rate"] == 0.9
        
        # Verify low-value files
        assert len(feedback["low_value_files"]) == 1
        assert feedback["low_value_files"][0]["path"] == "low1.py"
        
        # Verify overall stats
        assert feedback["overall_stats"]["total_elements"] == 3
        assert feedback["overall_stats"]["high_value_count"] == 2
    
    def test_optimize_relevance_scores(self):
        """Test relevance score optimization based on historical usage."""
        tracker = RelevanceTracker()
        
        # Create historical usage data
        tracker._element_usage = {
            "file:high_value.py": ElementUsage(
                element_id="file:high_value.py",
                element_type="file",
                times_included=10,
                times_used=9,
                usage_rate=0.9
            ),
            "file:low_value.py": ElementUsage(
                element_id="file:low_value.py",
                element_type="file",
                times_included=10,
                times_used=1,
                usage_rate=0.1
            ),
        }
        
        # Files with initial scores
        files = [
            {"path": "high_value.py", "relevance_score": 0.6},
            {"path": "low_value.py", "relevance_score": 0.6},
            {"path": "new_file.py", "relevance_score": 0.5},
        ]
        
        optimized = tracker.optimize_relevance_scores(files)
        
        # Verify high-value file got boosted
        high_value_file = next(f for f in optimized if f["path"] == "high_value.py")
        assert high_value_file["relevance_score"] > 0.6
        assert high_value_file["score_adjustment"] == "boosted"
        
        # Verify low-value file got reduced
        low_value_file = next(f for f in optimized if f["path"] == "low_value.py")
        assert low_value_file["relevance_score"] < 0.6
        assert low_value_file["score_adjustment"] == "reduced"
        
        # Verify new file has no history
        new_file = next(f for f in optimized if f["path"] == "new_file.py")
        assert new_file["score_adjustment"] == "no_history"
    
    def test_cleanup_old_data(self):
        """Test cleanup of old tracking data."""
        tracker = RelevanceTracker(max_history_days=7)
        
        # Create old session
        old_session = ContextUsageSession(
            session_id="old_session",
            query="old query",
            timestamp=datetime.now() - timedelta(days=10)
        )
        tracker._sessions.append(old_session)
        
        # Create recent session
        recent_session = ContextUsageSession(
            session_id="recent_session",
            query="recent query",
            timestamp=datetime.now()
        )
        tracker._sessions.append(recent_session)
        
        # Create old element usage
        tracker._element_usage["file:old.py"] = ElementUsage(
            element_id="file:old.py",
            element_type="file",
            last_included=datetime.now() - timedelta(days=10)
        )
        
        # Create recent element usage
        tracker._element_usage["file:recent.py"] = ElementUsage(
            element_id="file:recent.py",
            element_type="file",
            last_included=datetime.now()
        )
        
        # Run cleanup
        removed_count = tracker.cleanup_old_data()
        
        # Verify old data was removed
        assert removed_count == 1
        assert len(tracker._sessions) == 1
        assert tracker._sessions[0].session_id == "recent_session"
        assert "file:old.py" not in tracker._element_usage
        assert "file:recent.py" in tracker._element_usage
    
    def test_get_summary_statistics(self):
        """Test getting summary statistics."""
        tracker = RelevanceTracker()
        
        # Create diverse data
        tracker._sessions = [
            ContextUsageSession("s1", "q1", datetime.now(), task_type="debug"),
            ContextUsageSession("s2", "q2", datetime.now(), task_type="implement"),
        ]
        
        tracker._element_usage = {
            "file:f1.py": ElementUsage("file:f1.py", "file", times_included=5, times_used=4, usage_rate=0.8),
            "file:f2.py": ElementUsage("file:f2.py", "file", times_included=5, times_used=1, usage_rate=0.2),
            "doc:d1": ElementUsage("doc:d1", "documentation", times_included=3, times_used=3, usage_rate=1.0),
        }
        
        tracker._usage_by_task_type = {
            "debug": {},
            "implement": {},
        }
        
        stats = tracker.get_summary_statistics()
        
        # Verify statistics
        assert stats["total_sessions"] == 2
        assert stats["total_elements_tracked"] == 3
        assert stats["files_tracked"] == 2
        assert stats["docs_tracked"] == 1
        assert stats["high_value_files"] == 1  # f1.py with 0.8 usage rate
        assert stats["low_value_files"] == 0  # f2.py only included 5 times (< 3 threshold)
        assert "debug" in stats["task_types_tracked"]
        assert "implement" in stats["task_types_tracked"]
    
    def test_task_type_specific_tracking(self):
        """Test tracking usage by task type."""
        tracker = RelevanceTracker()
        
        # Record context for debug task
        files = [{"path": "app/main.py", "relevance_score": 0.9}]
        tracker.record_context_preparation(
            session_id="debug_session",
            query="Fix bug",
            files=files,
            documentation=[],
            code_search=[],
            task_type="debug"
        )
        
        # Record solution usage
        solution = "Fixed the bug in app/main.py"
        tracker.record_solution_usage("debug_session", solution)
        
        # Verify task-specific tracking
        assert "debug" in tracker._usage_by_task_type
        assert "file:app/main.py" in tracker._usage_by_task_type["debug"]
        
        debug_usage = tracker._usage_by_task_type["debug"]["file:app/main.py"]
        assert debug_usage.times_used == 1
        assert debug_usage.usage_rate == 1.0
        
        # Get task-specific statistics
        debug_stats = tracker.get_element_statistics(task_type="debug")
        assert len(debug_stats) == 1
        assert debug_stats[0].element_id == "file:app/main.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
