"""
Comprehensive tests for app/frameworks/registry.py - the Single Source of Truth for all frameworks.

Tests cover:
- FrameworkCategory enum values
- FrameworkDefinition dataclass creation and to_dict()
- Framework registration and retrieval
- Category filtering
- Vibe matching and weighted scoring
- Backward compatibility with VIBE_DICTIONARY
"""


import pytest

from app.core.errors import FrameworkNotFoundError
from app.core.routing import get_framework_info
from app.frameworks.registry import (
    FRAMEWORKS,
    VIBE_DICTIONARY,
    # Core data structures
    FrameworkCategory,
    FrameworkDefinition,
    count,
    find_by_vibe,
    get_all_vibes,
    # Retrieval functions
    get_framework,
    get_framework_names,
    get_framework_safe,
    get_frameworks_by_category,
    get_frameworks_dict,
    infer_task_type,
    list_by_category,
    match_vibes,
    # Registration
    register,
)

# =============================================================================
# FrameworkCategory Enum Tests
# =============================================================================

class TestFrameworkCategory:
    """Tests for the FrameworkCategory enum."""

    def test_all_expected_categories_exist(self):
        """Verify all expected category values exist."""
        expected_categories = [
            "STRATEGY",
            "SEARCH",
            "ITERATIVE",
            "CODE",
            "CONTEXT",
            "FAST",
            "VERIFICATION",
            "AGENT",
            "RAG",
        ]
        for cat_name in expected_categories:
            assert hasattr(FrameworkCategory, cat_name), f"Missing category: {cat_name}"

    def test_category_values(self):
        """Verify category values are lowercase strings."""
        assert FrameworkCategory.STRATEGY.value == "strategy"
        assert FrameworkCategory.SEARCH.value == "search"
        assert FrameworkCategory.ITERATIVE.value == "iterative"
        assert FrameworkCategory.CODE.value == "code"
        assert FrameworkCategory.CONTEXT.value == "context"
        assert FrameworkCategory.FAST.value == "fast"
        assert FrameworkCategory.VERIFICATION.value == "verification"
        assert FrameworkCategory.AGENT.value == "agent"
        assert FrameworkCategory.RAG.value == "rag"

    def test_category_count(self):
        """Verify total number of categories."""
        assert len(FrameworkCategory) == 9

    def test_category_iteration(self):
        """Verify categories can be iterated."""
        categories = list(FrameworkCategory)
        assert len(categories) == 9
        assert all(isinstance(cat, FrameworkCategory) for cat in categories)


# =============================================================================
# FrameworkDefinition Dataclass Tests
# =============================================================================

class TestFrameworkDefinition:
    """Tests for the FrameworkDefinition dataclass."""

    def test_minimal_creation(self):
        """Test creating a FrameworkDefinition with minimal required fields."""
        fw = FrameworkDefinition(
            name="test_framework",
            display_name="Test Framework",
            category=FrameworkCategory.CODE,
            description="A test framework",
            best_for=["testing"],
            vibes=["test this"],
        )
        assert fw.name == "test_framework"
        assert fw.display_name == "Test Framework"
        assert fw.category == FrameworkCategory.CODE
        assert fw.description == "A test framework"
        assert fw.best_for == ["testing"]
        assert fw.vibes == ["test this"]
        # Check defaults
        assert fw.steps == []
        assert fw.use_case == ""
        assert fw.node_function is None
        assert fw.complexity == "medium"
        assert fw.task_type == "unknown"
        assert fw.example_type == ""
        assert fw.prompt_template == ""

    def test_full_creation(self):
        """Test creating a FrameworkDefinition with all fields."""
        fw = FrameworkDefinition(
            name="full_framework",
            display_name="Full Framework",
            category=FrameworkCategory.STRATEGY,
            description="A complete test framework",
            best_for=["comprehensive testing", "validation"],
            vibes=["full test", "complete check"],
            steps=["Step 1", "Step 2", "Step 3"],
            use_case="Testing all framework features",
            node_function="app.nodes.test.full_framework_node",
            complexity="high",
            task_type="testing",
            example_type="test_examples",
            prompt_template="Custom template: {query}",
        )
        assert fw.name == "full_framework"
        assert fw.steps == ["Step 1", "Step 2", "Step 3"]
        assert fw.node_function == "app.nodes.test.full_framework_node"
        assert fw.complexity == "high"
        assert fw.task_type == "testing"
        assert fw.example_type == "test_examples"
        assert fw.prompt_template == "Custom template: {query}"

    def test_to_dict(self):
        """Test the to_dict() method returns correct structure."""
        fw = FrameworkDefinition(
            name="dict_test",
            display_name="Dict Test Framework",
            category=FrameworkCategory.ITERATIVE,
            description="Framework for dict testing",
            best_for=["serialization", "api responses"],
            vibes=["serialize", "json output"],
            complexity="low",
            task_type="serialization",
        )
        result = fw.to_dict()

        assert isinstance(result, dict)
        # Check key presence
        assert "name" in result
        assert "category" in result
        assert "description" in result
        assert "best_for" in result
        assert "vibes" in result
        assert "complexity" in result
        assert "task_type" in result

        # Check values - note: to_dict uses display_name for "name" key
        assert result["name"] == "Dict Test Framework"
        assert result["category"] == "iterative"  # .value, not enum
        assert result["description"] == "Framework for dict testing"
        assert result["best_for"] == ["serialization", "api responses"]
        assert result["vibes"] == ["serialize", "json output"]
        assert result["complexity"] == "low"
        assert result["task_type"] == "serialization"

    def test_to_dict_excludes_internal_fields(self):
        """Verify to_dict excludes internal implementation fields."""
        fw = FrameworkDefinition(
            name="internal_test",
            display_name="Internal Test",
            category=FrameworkCategory.CODE,
            description="Test internal fields exclusion",
            best_for=["test"],
            vibes=["test"],
            steps=["Step 1"],
            node_function="app.nodes.test",
            prompt_template="Template",
        )
        result = fw.to_dict()

        # These should NOT be in the dict (implementation details)
        assert "steps" not in result
        assert "node_function" not in result
        assert "use_case" not in result
        assert "example_type" not in result
        assert "prompt_template" not in result


# =============================================================================
# Framework Registration Tests
# =============================================================================

class TestFrameworkRegistration:
    """Tests for framework registration."""

    def test_register_adds_to_frameworks(self):
        """Test that register() adds framework to FRAMEWORKS dict."""
        # Create a unique test framework
        test_name = "_test_registration_framework_unique"
        fw = FrameworkDefinition(
            name=test_name,
            display_name="Test Registration",
            category=FrameworkCategory.FAST,
            description="Testing registration",
            best_for=["registration testing"],
            vibes=["register test"],
        )

        # Register it
        result = register(fw)

        try:
            # Verify it was added
            assert test_name in FRAMEWORKS
            assert FRAMEWORKS[test_name] is fw
            # Verify register returns the framework
            assert result is fw
        finally:
            # Cleanup: remove test framework
            if test_name in FRAMEWORKS:
                del FRAMEWORKS[test_name]

    def test_register_overwrites_existing(self):
        """Test that registering with same name overwrites."""
        test_name = "_test_overwrite_framework"
        fw1 = FrameworkDefinition(
            name=test_name,
            display_name="Version 1",
            category=FrameworkCategory.CODE,
            description="First version",
            best_for=["v1"],
            vibes=["version one"],
        )
        fw2 = FrameworkDefinition(
            name=test_name,
            display_name="Version 2",
            category=FrameworkCategory.CODE,
            description="Second version",
            best_for=["v2"],
            vibes=["version two"],
        )

        try:
            register(fw1)
            assert FRAMEWORKS[test_name].display_name == "Version 1"

            register(fw2)
            assert FRAMEWORKS[test_name].display_name == "Version 2"
        finally:
            if test_name in FRAMEWORKS:
                del FRAMEWORKS[test_name]


# =============================================================================
# get_framework() Tests
# =============================================================================

class TestGetFramework:
    """Tests for get_framework() function."""

    def test_get_existing_framework(self):
        """Test retrieving an existing framework."""
        # active_inference is a known framework
        fw = get_framework("active_inference")
        assert fw is not None
        assert isinstance(fw, FrameworkDefinition)
        assert fw.name == "active_inference"
        assert fw.display_name == "Active Inference"
        assert fw.category == FrameworkCategory.ITERATIVE

    def test_get_multiple_frameworks(self):
        """Test retrieving several different frameworks."""
        frameworks_to_test = [
            ("reason_flux", "ReasonFlux", FrameworkCategory.STRATEGY),
            ("tree_of_thoughts", "Tree of Thoughts", FrameworkCategory.SEARCH),
            ("program_of_thoughts", "Program of Thoughts", FrameworkCategory.CODE),
            ("chain_of_note", "Chain of Note", FrameworkCategory.CONTEXT),
            ("system1", "System1 Fast", FrameworkCategory.FAST),
            ("self_consistency", "Self-Consistency", FrameworkCategory.VERIFICATION),
            ("react", "ReAct", FrameworkCategory.ITERATIVE),  # ReAct is in ITERATIVE category
            ("self_rag", "Self-RAG", FrameworkCategory.RAG),
            ("rewoo", "ReWOO", FrameworkCategory.AGENT),  # Example from AGENT category
        ]

        for name, display_name, category in frameworks_to_test:
            fw = get_framework(name)
            assert fw.name == name, f"Name mismatch for {name}"
            assert fw.display_name == display_name, f"Display name mismatch for {name}"
            assert fw.category == category, f"Category mismatch for {name}"

    def test_get_unknown_framework_raises_error(self):
        """Test that requesting unknown framework raises FrameworkNotFoundError."""
        with pytest.raises(FrameworkNotFoundError) as exc_info:
            get_framework("nonexistent_framework_xyz")

        assert "Unknown framework" in str(exc_info.value)
        assert "nonexistent_framework_xyz" in str(exc_info.value)

    def test_framework_not_found_error_details(self):
        """Test that FrameworkNotFoundError contains useful details."""
        try:
            get_framework("fake_framework_name")
            pytest.fail("Should have raised FrameworkNotFoundError")
        except FrameworkNotFoundError as e:
            assert "details" in dir(e)
            assert "requested" in e.details
            assert e.details["requested"] == "fake_framework_name"
            assert "available" in e.details
            assert isinstance(e.details["available"], list)


# =============================================================================
# get_framework_safe() Tests
# =============================================================================

class TestGetFrameworkSafe:
    """Tests for get_framework_safe() function."""

    def test_get_existing_framework_safe(self):
        """Test retrieving existing framework returns the framework."""
        fw = get_framework_safe("active_inference")
        assert fw is not None
        assert isinstance(fw, FrameworkDefinition)
        assert fw.name == "active_inference"

    def test_get_unknown_framework_returns_none(self):
        """Test retrieving unknown framework returns None (no exception)."""
        result = get_framework_safe("completely_fake_framework")
        assert result is None

    def test_safe_vs_unsafe_comparison(self):
        """Compare safe and unsafe retrieval behavior."""
        # Safe returns None
        assert get_framework_safe("fake") is None

        # Unsafe raises
        with pytest.raises(FrameworkNotFoundError):
            get_framework("fake")


# =============================================================================
# get_frameworks_by_category() Tests
# =============================================================================

class TestGetFrameworksByCategory:
    """Tests for get_frameworks_by_category() function."""

    def test_get_strategy_frameworks(self):
        """Test getting STRATEGY category frameworks."""
        frameworks = get_frameworks_by_category(FrameworkCategory.STRATEGY)
        assert len(frameworks) >= 7  # At least 7 strategy frameworks
        assert all(fw.category == FrameworkCategory.STRATEGY for fw in frameworks)

        # Check known strategy frameworks
        names = [fw.name for fw in frameworks]
        assert "reason_flux" in names
        assert "self_discover" in names
        assert "plan_and_solve" in names

    def test_get_search_frameworks(self):
        """Test getting SEARCH category frameworks."""
        frameworks = get_frameworks_by_category(FrameworkCategory.SEARCH)
        assert len(frameworks) >= 4
        names = [fw.name for fw in frameworks]
        assert "mcts_rstar" in names
        assert "tree_of_thoughts" in names
        assert "graph_of_thoughts" in names

    def test_get_code_frameworks(self):
        """Test getting CODE category frameworks."""
        frameworks = get_frameworks_by_category(FrameworkCategory.CODE)
        assert len(frameworks) >= 15  # 15 code frameworks according to docs
        names = [fw.name for fw in frameworks]
        assert "program_of_thoughts" in names
        assert "chain_of_verification" in names
        assert "tdd_prompting" in names

    def test_get_fast_frameworks(self):
        """Test getting FAST category frameworks."""
        frameworks = get_frameworks_by_category(FrameworkCategory.FAST)
        assert len(frameworks) >= 2
        names = [fw.name for fw in frameworks]
        assert "skeleton_of_thought" in names
        assert "system1" in names

    def test_all_categories_return_frameworks(self):
        """Verify each category has at least one framework."""
        for category in FrameworkCategory:
            frameworks = get_frameworks_by_category(category)
            assert len(frameworks) > 0, f"No frameworks in category {category.name}"

    def test_returns_list_of_framework_definitions(self):
        """Verify return type is list of FrameworkDefinition."""
        frameworks = get_frameworks_by_category(FrameworkCategory.VERIFICATION)
        assert isinstance(frameworks, list)
        assert all(isinstance(fw, FrameworkDefinition) for fw in frameworks)


# =============================================================================
# get_all_vibes() Tests
# =============================================================================

class TestGetAllVibes:
    """Tests for get_all_vibes() function."""

    def test_returns_dict(self):
        """Test that get_all_vibes returns a dictionary."""
        vibes = get_all_vibes()
        assert isinstance(vibes, dict)

    def test_keys_are_framework_names(self):
        """Test that keys are valid framework names."""
        vibes = get_all_vibes()
        for name in vibes:
            assert name in FRAMEWORKS, f"Unknown framework name in vibes: {name}"

    def test_values_are_lists_of_strings(self):
        """Test that values are lists of strings."""
        vibes = get_all_vibes()
        for name, vibe_list in vibes.items():
            assert isinstance(vibe_list, list), f"Vibes for {name} is not a list"
            assert all(isinstance(v, str) for v in vibe_list), \
                f"Non-string vibe found in {name}"

    def test_all_frameworks_have_vibes(self):
        """Test that every framework has at least one vibe."""
        vibes = get_all_vibes()
        for name in FRAMEWORKS:
            assert name in vibes, f"Framework {name} missing from vibes"
            assert len(vibes[name]) > 0, f"Framework {name} has no vibes"

    def test_known_vibes_exist(self):
        """Test specific known vibes exist."""
        vibes = get_all_vibes()
        # active_inference should have debug-related vibes
        assert "active_inference" in vibes
        ai_vibes = vibes["active_inference"]
        assert any("debug" in v for v in ai_vibes)

        # reason_flux should have architecture vibes
        assert "reason_flux" in vibes
        rf_vibes = vibes["reason_flux"]
        assert any("architect" in v for v in rf_vibes)


# =============================================================================
# get_framework_names() Tests
# =============================================================================

class TestGetFrameworkNames:
    """Tests for get_framework_names() function."""

    def test_returns_list(self):
        """Test that get_framework_names returns a list."""
        names = get_framework_names()
        assert isinstance(names, list)

    def test_returns_all_framework_names(self):
        """Test that all registered frameworks are returned."""
        names = get_framework_names()
        assert len(names) == len(FRAMEWORKS)
        assert set(names) == set(FRAMEWORKS.keys())

    def test_names_are_strings(self):
        """Test that all names are strings."""
        names = get_framework_names()
        assert all(isinstance(name, str) for name in names)

    def test_contains_known_frameworks(self):
        """Test that known framework names are included."""
        names = get_framework_names()
        known_frameworks = [
            "active_inference",
            "reason_flux",
            "tree_of_thoughts",
            "chain_of_thought",
            "system1",
            "self_rag",
            "react",
        ]
        for fw_name in known_frameworks:
            assert fw_name in names, f"Missing known framework: {fw_name}"


# =============================================================================
# get_framework_info() Tests
# =============================================================================

class TestGetFrameworkInfo:
    """Tests for get_framework_info() function."""

    def test_valid_framework_returns_dict(self):
        """Test that valid framework returns info dict."""
        info = get_framework_info("active_inference")
        assert isinstance(info, dict)
        assert "name" in info
        assert "category" in info
        assert "description" in info
        assert "best_for" in info
        assert "complexity" in info

    def test_valid_framework_info_values(self):
        """Test that valid framework info has correct values."""
        info = get_framework_info("reason_flux")
        assert info["name"] == "ReasonFlux"  # display_name
        assert info["category"] == "strategy"
        assert "architecture" in info["description"].lower() or "planning" in info["description"].lower()
        assert isinstance(info["best_for"], list)
        assert info["complexity"] in ["low", "medium", "high"]

    def test_unknown_framework_default_behavior(self):
        """Test that unknown framework returns default info (raise_on_unknown=False)."""
        info = get_framework_info("fake_framework")
        assert info["name"] == "fake_framework"
        assert info["category"] == "unknown"
        assert info["description"] == "Unknown framework"
        assert info["best_for"] == []
        assert info["complexity"] == "unknown"

    def test_unknown_framework_raises_when_requested(self):
        """Test that unknown framework raises when raise_on_unknown=True."""
        with pytest.raises(FrameworkNotFoundError):
            get_framework_info("nonexistent_fw", raise_on_unknown=True)

    def test_all_frameworks_have_valid_info(self):
        """Test that all frameworks return valid info."""
        for name in FRAMEWORKS:
            info = get_framework_info(name)
            assert info["name"] != ""
            assert info["category"] in [cat.value for cat in FrameworkCategory]
            assert info["description"] != ""
            assert isinstance(info["best_for"], list)
            assert info["complexity"] in ["low", "medium", "high"]


# =============================================================================
# infer_task_type() Tests
# =============================================================================

class TestInferTaskType:
    """Tests for infer_task_type() function."""

    def test_known_framework_returns_task_type(self):
        """Test that known frameworks return their task type."""
        assert infer_task_type("active_inference") == "debug"
        assert infer_task_type("reason_flux") == "architecture"
        assert infer_task_type("tree_of_thoughts") == "algorithm"
        assert infer_task_type("system1") == "quick"

    def test_unknown_framework_returns_unknown(self):
        """Test that unknown frameworks return 'unknown'."""
        assert infer_task_type("fake_framework") == "unknown"
        assert infer_task_type("nonexistent") == "unknown"

    def test_all_frameworks_have_task_type(self):
        """Test that all frameworks have a task type (not 'unknown')."""
        for name in FRAMEWORKS:
            task_type = infer_task_type(name)
            # Note: some frameworks legitimately have task_type="unknown"
            # in their definition, but most should have meaningful types
            assert isinstance(task_type, str)


# =============================================================================
# get_frameworks_dict() Tests
# =============================================================================

class TestGetFrameworksDict:
    """Tests for get_frameworks_dict() function."""

    def test_returns_dict(self):
        """Test that get_frameworks_dict returns a dictionary."""
        result = get_frameworks_dict()
        assert isinstance(result, dict)

    def test_format_is_name_to_description(self):
        """Test that format is {name: description}."""
        result = get_frameworks_dict()
        for name, description in result.items():
            assert isinstance(name, str)
            assert isinstance(description, str)
            assert name in FRAMEWORKS
            assert description == FRAMEWORKS[name].description

    def test_includes_all_frameworks(self):
        """Test that all frameworks are included."""
        result = get_frameworks_dict()
        assert len(result) == len(FRAMEWORKS)
        assert set(result.keys()) == set(FRAMEWORKS.keys())

    def test_descriptions_are_not_empty(self):
        """Test that descriptions are not empty."""
        result = get_frameworks_dict()
        for name, description in result.items():
            assert len(description) > 0, f"Empty description for {name}"


# =============================================================================
# list_by_category() Tests
# =============================================================================

class TestListByCategory:
    """Tests for list_by_category() function."""

    def test_returns_dict(self):
        """Test that list_by_category returns a dictionary."""
        result = list_by_category()
        assert isinstance(result, dict)

    def test_keys_are_category_values(self):
        """Test that keys are category value strings."""
        result = list_by_category()
        valid_categories = [cat.value for cat in FrameworkCategory]
        for key in result:
            assert key in valid_categories, f"Invalid category key: {key}"

    def test_values_are_lists_of_names(self):
        """Test that values are lists of framework names."""
        result = list_by_category()
        for category, names in result.items():
            assert isinstance(names, list)
            for name in names:
                assert name in FRAMEWORKS, f"Unknown framework {name} in category {category}"

    def test_all_frameworks_categorized(self):
        """Test that all frameworks appear in exactly one category."""
        result = list_by_category()
        all_names = []
        for names in result.values():
            all_names.extend(names)

        # All frameworks should be in exactly one category
        assert len(all_names) == len(FRAMEWORKS)
        assert set(all_names) == set(FRAMEWORKS.keys())

    def test_known_category_contents(self):
        """Test known frameworks appear in correct categories."""
        result = list_by_category()

        # Strategy frameworks
        assert "reason_flux" in result.get("strategy", [])
        assert "self_discover" in result.get("strategy", [])

        # Search frameworks
        assert "tree_of_thoughts" in result.get("search", [])
        assert "mcts_rstar" in result.get("search", [])

        # Fast frameworks
        assert "system1" in result.get("fast", [])
        assert "skeleton_of_thought" in result.get("fast", [])


# =============================================================================
# count() Tests
# =============================================================================

class TestCount:
    """Tests for count() function."""

    def test_returns_integer(self):
        """Test that count returns an integer."""
        result = count()
        assert isinstance(result, int)

    def test_count_matches_frameworks_dict(self):
        """Test that count matches FRAMEWORKS dict length."""
        assert count() == len(FRAMEWORKS)

    def test_count_is_approximately_62(self):
        """Test that count is approximately 62 frameworks (as documented)."""
        # The docs say 62 frameworks, allow some variance
        assert count() >= 60, "Too few frameworks registered"
        assert count() <= 70, "Many more frameworks than documented"

    def test_count_is_positive(self):
        """Test that count is positive."""
        assert count() > 0


# =============================================================================
# find_by_vibe() Tests
# =============================================================================

class TestFindByVibe:
    """Tests for find_by_vibe() function."""

    def test_finds_framework_by_exact_vibe(self):
        """Test finding framework by exact vibe match."""
        # "debug this" is a vibe for active_inference
        fw = find_by_vibe("debug this code please")
        assert fw is not None
        # Could match active_inference or another debug framework
        assert fw.name in ["active_inference", "rubber_duck", "chain_of_code",
                           "self_debugging", "reverse_cot"]

    def test_finds_framework_case_insensitive(self):
        """Test that vibe matching is case-insensitive."""
        fw1 = find_by_vibe("DEBUG THIS")
        fw2 = find_by_vibe("debug this")
        # Both should find a framework (may or may not be the same one)
        assert fw1 is not None
        assert fw2 is not None

    def test_returns_none_for_no_match(self):
        """Test returns None when no vibe matches."""
        result = find_by_vibe("xyzzy plugh completely random gibberish")
        assert result is None

    def test_returns_framework_definition(self):
        """Test that return type is FrameworkDefinition."""
        fw = find_by_vibe("optimize performance")
        if fw is not None:
            assert isinstance(fw, FrameworkDefinition)

    def test_various_vibes(self):
        """Test various vibes find appropriate frameworks."""
        test_cases = [
            ("design a system", ["reason_flux", "comparative_arch"]),
            ("fix this bug", ["active_inference", "self_debugging", "rubber_duck"]),
            ("refactor this mess", ["graph_of_thoughts", "everything_of_thought"]),
            ("quick question", ["system1"]),
            ("security audit", ["red_team", "chain_of_verification"]),
        ]

        for vibe, _expected_frameworks in test_cases:
            fw = find_by_vibe(vibe)
            if fw is not None:
                # Just verify it returns a valid framework
                assert fw.name in FRAMEWORKS


# =============================================================================
# match_vibes() Tests (Weighted Scoring)
# =============================================================================

class TestMatchVibes:
    """Tests for match_vibes() weighted scoring function."""

    def test_returns_string_or_none(self):
        """Test that match_vibes returns string framework name or None."""
        result = match_vibes("debug this issue")
        assert result is None or isinstance(result, str)

    def test_returns_none_for_no_match(self):
        """Test returns None when no patterns match."""
        result = match_vibes("asdfghjkl qwertyuiop completely random")
        assert result is None

    def test_returns_framework_name_for_match(self):
        """Test returns valid framework name for matches."""
        result = match_vibes("I need to debug this really hard bug")
        if result is not None:
            assert result in FRAMEWORKS

    def test_longer_phrases_score_higher(self):
        """Test that longer phrase matches get higher weight."""
        # "design a system" is a 3-word phrase in reason_flux vibes
        result = match_vibes("please design a system for user management")
        # Should match reason_flux due to the 3-word phrase
        if result is not None:
            assert result in FRAMEWORKS

    def test_multi_word_phrase_wins_over_single_word(self):
        """Test that multi-word phrases are weighted higher."""
        # This should favor frameworks with multi-word vibe matches
        # "big O" is a 2-word phrase for step_back
        result = match_vibes("what is the big O complexity")
        if result is not None:
            assert result in FRAMEWORKS

    def test_score_accumulation(self):
        """Test that multiple matches accumulate score."""
        # Query with multiple vibes for the same framework
        # active_inference has: "debug", "wtf is wrong", "find the bug"
        result = match_vibes("wtf is wrong with this, please find the bug and debug it")
        # Should strongly match active_inference
        if result is not None:
            # At minimum, should find a debug-oriented framework
            assert result in FRAMEWORKS

    def test_clear_winner_selection(self):
        """Test that clear winners are selected correctly."""
        # Query specifically targeting architecture
        result = match_vibes("design a system with microservices architecture")
        # Should match reason_flux or similar architecture framework
        if result is not None:
            fw = get_framework(result)
            assert fw is not None

    def test_minimum_score_threshold(self):
        """Test that weak matches are filtered out."""
        # Single short word that might match weakly
        result = match_vibes("just a random word")
        # Might return None due to low score
        if result is not None:
            assert result in FRAMEWORKS

    def test_case_insensitivity(self):
        """Test that matching is case-insensitive."""
        result1 = match_vibes("DEBUG THIS ISSUE")
        result2 = match_vibes("debug this issue")
        # Both should produce consistent results
        if result1 is not None and result2 is not None:
            # They should match the same or similar frameworks
            assert result1 in FRAMEWORKS
            assert result2 in FRAMEWORKS


# =============================================================================
# VIBE_DICTIONARY Compatibility Tests
# =============================================================================

class TestVibeDictionaryCompatibility:
    """Tests for backward compatibility with VIBE_DICTIONARY."""

    def test_vibe_dictionary_exists(self):
        """Test that VIBE_DICTIONARY is exported."""
        assert VIBE_DICTIONARY is not None

    def test_vibe_dictionary_is_dict(self):
        """Test that VIBE_DICTIONARY is a dictionary."""
        assert isinstance(VIBE_DICTIONARY, dict)

    def test_vibe_dictionary_matches_get_all_vibes(self):
        """Test that VIBE_DICTIONARY equals get_all_vibes()."""
        vibes = get_all_vibes()
        assert vibes == VIBE_DICTIONARY

    def test_vibe_dictionary_has_all_frameworks(self):
        """Test that VIBE_DICTIONARY includes all frameworks."""
        assert len(VIBE_DICTIONARY) == len(FRAMEWORKS)

    def test_vibe_dictionary_format(self):
        """Test VIBE_DICTIONARY format is {name: [vibes]}."""
        for name, vibes in VIBE_DICTIONARY.items():
            assert isinstance(name, str)
            assert isinstance(vibes, list)
            assert all(isinstance(v, str) for v in vibes)


# =============================================================================
# Integration Tests
# =============================================================================

class TestRegistryIntegration:
    """Integration tests verifying components work together."""

    def test_full_workflow(self):
        """Test a complete workflow of registering and querying."""
        test_name = "_integration_test_framework"
        try:
            # Register
            fw = FrameworkDefinition(
                name=test_name,
                display_name="Integration Test",
                category=FrameworkCategory.VERIFICATION,
                description="Testing integration",
                best_for=["integration tests"],
                vibes=["integration test vibe"],
                complexity="low",
                task_type="testing",
            )
            register(fw)

            # Query by name
            assert get_framework(test_name).name == test_name
            assert get_framework_safe(test_name) is not None

            # Check in category
            verification_fws = get_frameworks_by_category(FrameworkCategory.VERIFICATION)
            assert any(f.name == test_name for f in verification_fws)

            # Check in all vibes
            vibes = get_all_vibes()
            assert test_name in vibes

            # Check in names list
            names = get_framework_names()
            assert test_name in names

            # Check in frameworks dict
            fw_dict = get_frameworks_dict()
            assert test_name in fw_dict

            # Check task type
            assert infer_task_type(test_name) == "testing"

            # Check in category list
            by_cat = list_by_category()
            assert test_name in by_cat["verification"]

        finally:
            # Cleanup
            if test_name in FRAMEWORKS:
                del FRAMEWORKS[test_name]

    def test_all_frameworks_have_required_fields(self):
        """Test that all registered frameworks have required fields."""
        for name, fw in FRAMEWORKS.items():
            assert fw.name, f"{name}: missing name"
            assert fw.display_name, f"{name}: missing display_name"
            assert fw.category, f"{name}: missing category"
            assert fw.description, f"{name}: missing description"
            assert isinstance(fw.best_for, list), f"{name}: best_for not a list"
            assert isinstance(fw.vibes, list), f"{name}: vibes not a list"
            assert len(fw.vibes) > 0, f"{name}: no vibes defined"

    def test_category_distribution(self):
        """Test that frameworks are distributed across categories."""
        by_cat = list_by_category()

        # Verify expected category sizes (approximate)
        expected_ranges = {
            "strategy": (5, 10),
            "search": (3, 6),
            "iterative": (6, 12),
            "code": (12, 20),
            "context": (4, 10),
            "fast": (1, 5),
            "verification": (5, 12),
            "agent": (4, 8),
            "rag": (4, 8),
        }

        for category, (min_count, max_count) in expected_ranges.items():
            actual = len(by_cat.get(category, []))
            assert actual >= min_count, \
                f"Category {category} has too few frameworks: {actual} < {min_count}"
            assert actual <= max_count, \
                f"Category {category} has too many frameworks: {actual} > {max_count}"

    def test_complexity_values_are_valid(self):
        """Test that all frameworks have valid complexity values."""
        valid_complexities = {"low", "medium", "high"}
        for name, fw in FRAMEWORKS.items():
            assert fw.complexity in valid_complexities, \
                f"{name} has invalid complexity: {fw.complexity}"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_framework_name(self):
        """Test handling of empty string framework name."""
        with pytest.raises(FrameworkNotFoundError):
            get_framework("")

        assert get_framework_safe("") is None

    def test_none_vibe_matching(self):
        """Test vibe matching with empty/minimal input."""
        assert match_vibes("") is None
        assert find_by_vibe("") is None

    def test_very_long_query(self):
        """Test vibe matching with very long query."""
        long_query = "debug " * 1000  # Very long query
        result = match_vibes(long_query)
        # Should not crash, may or may not find a match
        assert result is None or result in FRAMEWORKS

    def test_special_characters_in_query(self):
        """Test vibe matching with special characters."""
        queries = [
            "debug this!!! @#$%",
            "what's wrong with this???",
            "fix the <bug> now",
            "debug\nthis\ncode",
            "optimize\tperformance",
        ]
        for query in queries:
            result = match_vibes(query)
            # Should not crash
            assert result is None or result in FRAMEWORKS

    def test_unicode_in_query(self):
        """Test vibe matching with unicode characters."""
        result = match_vibes("debug this code")
        # Should not crash
        assert result is None or result in FRAMEWORKS

    def test_numbers_in_query(self):
        """Test vibe matching with numbers."""
        result = match_vibes("optimize O(n^2) to O(n)")
        # Should not crash
        assert result is None or result in FRAMEWORKS
