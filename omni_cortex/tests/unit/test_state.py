"""
Unit tests for GraphState creation and validation.

Tests the app.state module including:
- GraphState TypedDict structure
- create_initial_state() factory function
- MemoryStore operations
"""

import pytest
from typing import get_type_hints

from app.state import GraphState, create_initial_state, MemoryStore


class TestCreateInitialState:
    """Tests for the create_initial_state() factory function."""

    def test_minimal_state_creation(self):
        """Test creating state with only required query parameter."""
        state = create_initial_state(query="Test query")

        assert state["query"] == "Test query"
        assert state["code_snippet"] is None
        assert state["file_list"] == []
        assert state["ide_context"] is None
        assert state["preferred_framework"] is None
        assert state["max_iterations"] == 5

    def test_full_state_creation(self):
        """Test creating state with all parameters."""
        state = create_initial_state(
            query="Debug this code",
            code_snippet="def foo(): pass",
            file_list=["main.py", "utils.py"],
            ide_context="Line 10: TypeError",
            preferred_framework="active_inference",
            max_iterations=10,
        )

        assert state["query"] == "Debug this code"
        assert state["code_snippet"] == "def foo(): pass"
        assert state["file_list"] == ["main.py", "utils.py"]
        assert state["ide_context"] == "Line 10: TypeError"
        assert state["preferred_framework"] == "active_inference"
        assert state["max_iterations"] == 10

    def test_default_routing_values(self):
        """Test that routing fields have correct defaults."""
        state = create_initial_state(query="Test")

        assert state["selected_framework"] == ""
        assert state["framework_chain"] == []
        assert state["routing_category"] == "unknown"
        assert state["task_type"] == "unknown"
        assert state["complexity_estimate"] == 0.5

    def test_default_memory_values(self):
        """Test that memory fields have correct defaults."""
        state = create_initial_state(query="Test")

        assert state["working_memory"] == {}
        assert state["reasoning_steps"] == []
        assert state["episodic_memory"] == []

    def test_default_output_values(self):
        """Test that output fields have correct defaults."""
        state = create_initial_state(query="Test")

        assert state["final_code"] is None
        assert state["final_answer"] is None
        assert state["confidence_score"] == 0.5
        assert state["tokens_used"] == 0

    def test_default_quiet_star_values(self):
        """Test that Quiet-STaR fields have correct defaults."""
        state = create_initial_state(query="Test")

        assert state["quiet_thoughts"] == []

    def test_default_error_handling(self):
        """Test that error field has correct default."""
        state = create_initial_state(query="Test")

        assert state["error"] is None

    def test_file_list_default_is_empty_list(self):
        """Test that file_list defaults to empty list, not None."""
        state = create_initial_state(query="Test", file_list=None)

        # Should be empty list, not None
        assert state["file_list"] == []
        assert isinstance(state["file_list"], list)


class TestGraphStateStructure:
    """Tests for GraphState TypedDict structure."""

    def test_state_is_dict_like(self, minimal_state):
        """Test that GraphState behaves like a dictionary."""
        # Can access keys
        assert "query" in minimal_state

        # Can iterate
        keys = list(minimal_state.keys())
        assert "query" in keys

        # Can get values
        values = list(minimal_state.values())
        assert len(values) > 0

    def test_state_contains_all_expected_fields(self, minimal_state):
        """Test that all expected fields are present in state."""
        expected_fields = [
            # Input fields
            "query", "code_snippet", "file_list", "ide_context",
            "preferred_framework", "max_iterations",
            # Routing fields
            "selected_framework", "framework_chain", "routing_category",
            "task_type", "complexity_estimate",
            # Memory fields
            "working_memory", "reasoning_steps", "episodic_memory",
            # Output fields
            "final_code", "final_answer", "confidence_score", "tokens_used",
            # Quiet-STaR
            "quiet_thoughts",
            # Error handling
            "error",
        ]

        for field in expected_fields:
            assert field in minimal_state, f"Missing field: {field}"

    def test_state_modification(self, minimal_state):
        """Test that state can be modified."""
        minimal_state["selected_framework"] = "tree_of_thoughts"
        minimal_state["confidence_score"] = 0.9
        minimal_state["tokens_used"] = 1000

        assert minimal_state["selected_framework"] == "tree_of_thoughts"
        assert minimal_state["confidence_score"] == 0.9
        assert minimal_state["tokens_used"] == 1000

    def test_reasoning_steps_append(self, minimal_state):
        """Test appending to reasoning_steps list."""
        step = {
            "framework": "active_inference",
            "thought": "Analyzing the problem",
            "action": "investigate",
            "observation": "Found potential issue",
        }

        minimal_state["reasoning_steps"].append(step)

        assert len(minimal_state["reasoning_steps"]) == 1
        assert minimal_state["reasoning_steps"][0]["framework"] == "active_inference"

    def test_working_memory_nested_update(self, minimal_state):
        """Test updating nested working_memory dictionary."""
        minimal_state["working_memory"]["chat_history"] = ["msg1", "msg2"]
        minimal_state["working_memory"]["framework_history"] = ["active_inference"]

        assert minimal_state["working_memory"]["chat_history"] == ["msg1", "msg2"]
        assert minimal_state["working_memory"]["framework_history"] == ["active_inference"]


class TestMemoryStore:
    """Tests for MemoryStore dataclass."""

    def test_memory_store_creation(self, memory_store):
        """Test creating a fresh MemoryStore."""
        assert memory_store.working == {}
        assert memory_store.episodic == []
        assert memory_store.thought_templates == []
        assert memory_store.total_queries == 0
        assert memory_store.successful_queries == 0
        assert memory_store.framework_usage == {}

    def test_add_episode(self, memory_store):
        """Test adding an episode to memory."""
        episode = {
            "query": "Fix the bug",
            "framework": "active_inference",
            "success": True,
        }

        memory_store.add_episode(episode)

        assert len(memory_store.episodic) == 1
        assert memory_store.episodic[0] == episode

    def test_add_episode_limit(self, memory_store):
        """Test that episodic memory is limited to 1000 episodes."""
        # Add 1010 episodes
        for i in range(1010):
            memory_store.add_episode({"id": i})

        # Should only keep last 1000
        assert len(memory_store.episodic) == 1000
        # First episode should be id=10 (0-9 were evicted)
        assert memory_store.episodic[0]["id"] == 10

    def test_add_thought_template(self, memory_store):
        """Test adding a thought template."""
        template = {
            "task_type": "debugging",
            "keywords": ["bug", "error", "fix"],
            "template": "First, identify the symptom...",
        }

        memory_store.add_thought_template(template)

        assert len(memory_store.thought_templates) == 1
        assert memory_store.thought_templates[0] == template

    def test_add_thought_template_limit(self, memory_store):
        """Test that thought templates are limited to 500."""
        for i in range(510):
            memory_store.add_thought_template({"id": i})

        assert len(memory_store.thought_templates) == 500
        # First template should be id=10
        assert memory_store.thought_templates[0]["id"] == 10

    def test_find_similar_templates_by_task_type(self, memory_store):
        """Test finding templates by task type."""
        memory_store.add_thought_template({
            "task_type": "debugging",
            "keywords": ["bug"],
            "content": "Debug template",
        })
        memory_store.add_thought_template({
            "task_type": "refactoring",
            "keywords": ["clean"],
            "content": "Refactor template",
        })

        matches = memory_store.find_similar_templates(
            task_type="debugging",
            keywords=[],
            limit=5,
        )

        assert len(matches) >= 1
        assert any(m["task_type"] == "debugging" for m in matches)

    def test_find_similar_templates_by_keywords(self, memory_store):
        """Test finding templates by keyword overlap."""
        memory_store.add_thought_template({
            "task_type": "other",
            "keywords": ["performance", "slow", "optimize"],
            "content": "Performance template",
        })

        matches = memory_store.find_similar_templates(
            task_type="something_else",
            keywords=["slow", "performance"],
            limit=5,
        )

        assert len(matches) == 1
        assert matches[0]["content"] == "Performance template"

    def test_find_similar_templates_sorted_by_overlap(self, memory_store):
        """Test that matches are sorted by keyword overlap."""
        memory_store.add_thought_template({
            "task_type": "other",
            "keywords": ["a"],
            "overlap": 1,
        })
        memory_store.add_thought_template({
            "task_type": "other",
            "keywords": ["a", "b", "c"],
            "overlap": 3,
        })
        memory_store.add_thought_template({
            "task_type": "other",
            "keywords": ["a", "b"],
            "overlap": 2,
        })

        matches = memory_store.find_similar_templates(
            task_type="none",
            keywords=["a", "b", "c"],
            limit=3,
        )

        # Should be sorted by overlap (3, 2, 1)
        assert matches[0]["overlap"] == 3
        assert matches[1]["overlap"] == 2
        assert matches[2]["overlap"] == 1

    def test_record_framework_usage(self, memory_store):
        """Test recording framework usage statistics."""
        memory_store.record_framework_usage("active_inference")
        memory_store.record_framework_usage("tree_of_thoughts")
        memory_store.record_framework_usage("active_inference")

        assert memory_store.framework_usage["active_inference"] == 2
        assert memory_store.framework_usage["tree_of_thoughts"] == 1

    def test_clear_working_memory(self, memory_store):
        """Test clearing working memory."""
        memory_store.working = {"key": "value", "other": 123}

        memory_store.clear_working_memory()

        assert memory_store.working == {}


class TestStateWithFixtures:
    """Tests using fixtures from conftest.py."""

    def test_full_state_fixture(self, full_state):
        """Test the full_state fixture has expected values."""
        assert "Debug" in full_state["query"]
        assert full_state["code_snippet"] is not None
        assert len(full_state["file_list"]) == 2
        assert full_state["preferred_framework"] == "active_inference"

    def test_state_with_code_fixture(self, state_with_code):
        """Test the state_with_code fixture."""
        assert "Optimize" in state_with_code["query"]
        assert "bubble_sort" in state_with_code["code_snippet"]
        assert "sorting.py" in state_with_code["file_list"][0]
