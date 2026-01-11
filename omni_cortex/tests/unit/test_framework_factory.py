"""Tests for framework factory and configuration-based orchestration."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.sampling import ClientSampler
from app.orchestrators.framework_factory import (
    CHAIN_OF_VERIFICATION,
    CRITIC,
    FRAMEWORK_CONFIGS,
    PROGRAM_OF_THOUGHTS,
    FrameworkConfig,
    FrameworkStep,
    execute_framework,
    get_available_frameworks,
    get_framework_description,
    run_framework,
)


class TestFrameworkStep:
    """Tests for FrameworkStep dataclass."""

    def test_basic_step(self):
        """Basic step creation with defaults."""
        step = FrameworkStep("analyze", "Analyze {query}")
        assert step.name == "analyze"
        assert step.prompt_template == "Analyze {query}"
        assert step.temperature == 0.5
        assert step.uses_previous is True
        assert step.max_tokens is None

    def test_step_with_custom_values(self):
        """Step with custom temperature and max_tokens."""
        step = FrameworkStep(
            name="generate",
            prompt_template="Generate code for {query}",
            temperature=0.8,
            uses_previous=False,
            max_tokens=2000
        )
        assert step.temperature == 0.8
        assert step.uses_previous is False
        assert step.max_tokens == 2000


class TestFrameworkConfig:
    """Tests for FrameworkConfig dataclass."""

    def test_basic_config(self):
        """Basic config creation with defaults."""
        steps = [FrameworkStep("step1", "prompt1")]
        config = FrameworkConfig(name="test", steps=steps)
        assert config.name == "test"
        assert config.final_template == "{last_step}"
        assert config.description == ""
        assert config.metadata_extras == {}

    def test_config_with_all_options(self):
        """Config with all options specified."""
        steps = [
            FrameworkStep("a", "prompt1"),
            FrameworkStep("b", "prompt2"),
        ]
        config = FrameworkConfig(
            name="full_test",
            steps=steps,
            final_template="{a}\n{b}",
            description="A test framework",
            metadata_extras={"version": "1.0"}
        )
        assert config.final_template == "{a}\n{b}"
        assert config.description == "A test framework"
        assert config.metadata_extras == {"version": "1.0"}


class TestExecuteFramework:
    """Tests for execute_framework function."""

    @pytest.mark.asyncio
    async def test_execute_simple_framework(self):
        """Execute a simple two-step framework."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(side_effect=["step1_output", "step2_output"])

        config = FrameworkConfig(
            name="test",
            steps=[
                FrameworkStep("a", "First step: {query}", temperature=0.5),
                FrameworkStep("b", "Second step with {previous}", temperature=0.5),
            ],
            final_template="{b}"
        )

        result = await execute_framework(config, mock_sampler, "test query", "test context")

        assert result["final_answer"] == "step2_output"
        assert result["metadata"]["framework"] == "test"
        assert result["metadata"]["steps"] == 2

    @pytest.mark.asyncio
    async def test_execute_with_context_substitution(self):
        """Framework correctly substitutes query and context."""
        mock_sampler = MagicMock(spec=ClientSampler)
        captured_prompts = []

        async def capture_prompt(prompt, **kwargs):
            captured_prompts.append(prompt)
            return "output"

        mock_sampler.request_sample = capture_prompt

        config = FrameworkConfig(
            name="test",
            steps=[
                FrameworkStep("step", "Query: {query}\nContext: {context}"),
            ]
        )

        await execute_framework(config, mock_sampler, "my query", "my context")

        assert "Query: my query" in captured_prompts[0]
        assert "Context: my context" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_execute_with_previous_output(self):
        """Framework correctly passes previous step output."""
        mock_sampler = MagicMock(spec=ClientSampler)
        captured_prompts = []

        async def capture_prompt(prompt, **kwargs):
            captured_prompts.append(prompt)
            return f"output_{len(captured_prompts)}"

        mock_sampler.request_sample = capture_prompt

        config = FrameworkConfig(
            name="test",
            steps=[
                FrameworkStep("first", "Initial: {query}"),
                FrameworkStep("second", "Previous was: {previous}"),
            ]
        )

        await execute_framework(config, mock_sampler, "query", "context")

        assert "Previous was: output_1" in captured_prompts[1]

    @pytest.mark.asyncio
    async def test_execute_uses_previous_false(self):
        """Step with uses_previous=False gets empty string."""
        mock_sampler = MagicMock(spec=ClientSampler)
        captured_prompts = []

        async def capture_prompt(prompt, **kwargs):
            captured_prompts.append(prompt)
            return "output"

        mock_sampler.request_sample = capture_prompt

        config = FrameworkConfig(
            name="test",
            steps=[
                FrameworkStep("first", "Step 1"),
                FrameworkStep("second", "Previous: [{previous}]", uses_previous=False),
            ]
        )

        await execute_framework(config, mock_sampler, "query", "context")

        assert "Previous: []" in captured_prompts[1]

    @pytest.mark.asyncio
    async def test_execute_step_references_named_output(self):
        """Steps can reference outputs by name."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(side_effect=["draft", "critique", "final"])

        config = FrameworkConfig(
            name="test",
            steps=[
                FrameworkStep("draft", "Create draft"),
                FrameworkStep("critique", "Critique draft"),
                FrameworkStep("revision", "Revise based on {critique}: {draft}"),
            ]
        )

        await execute_framework(config, mock_sampler, "query", "context")

        # The third call should have received the named outputs
        calls = mock_sampler.request_sample.call_args_list
        third_prompt = calls[2][0][0]
        assert "critique" in third_prompt
        assert "draft" in third_prompt

    @pytest.mark.asyncio
    async def test_execute_final_template(self):
        """Final template combines step outputs correctly."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(side_effect=["code", "tests"])

        config = FrameworkConfig(
            name="test",
            steps=[
                FrameworkStep("code", "Generate code"),
                FrameworkStep("tests", "Generate tests"),
            ],
            final_template="## Code\n{code}\n\n## Tests\n{tests}"
        )

        result = await execute_framework(config, mock_sampler, "query", "context")

        assert "## Code\ncode" in result["final_answer"]
        assert "## Tests\ntests" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_execute_metadata_extras(self):
        """Metadata extras are included in result."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(return_value="output")

        config = FrameworkConfig(
            name="test",
            steps=[FrameworkStep("step", "prompt")],
            metadata_extras={"evolutions": 3, "version": "2.0"}
        )

        result = await execute_framework(config, mock_sampler, "query", "context")

        assert result["metadata"]["evolutions"] == 3
        assert result["metadata"]["version"] == "2.0"

    @pytest.mark.asyncio
    async def test_execute_respects_temperature(self):
        """Framework passes correct temperature to sampler."""
        mock_sampler = MagicMock(spec=ClientSampler)
        captured_kwargs = []

        async def capture_kwargs(prompt, **kwargs):
            captured_kwargs.append(kwargs)
            return "output"

        mock_sampler.request_sample = capture_kwargs

        config = FrameworkConfig(
            name="test",
            steps=[
                FrameworkStep("step1", "prompt1", temperature=0.3),
                FrameworkStep("step2", "prompt2", temperature=0.9),
            ]
        )

        await execute_framework(config, mock_sampler, "query", "context")

        assert captured_kwargs[0]["temperature"] == 0.3
        assert captured_kwargs[1]["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_execute_respects_max_tokens(self):
        """Framework passes max_tokens when specified."""
        mock_sampler = MagicMock(spec=ClientSampler)
        captured_kwargs = []

        async def capture_kwargs(prompt, **kwargs):
            captured_kwargs.append(kwargs)
            return "output"

        mock_sampler.request_sample = capture_kwargs

        config = FrameworkConfig(
            name="test",
            steps=[
                FrameworkStep("step1", "prompt1"),
                FrameworkStep("step2", "prompt2", max_tokens=1000),
            ]
        )

        await execute_framework(config, mock_sampler, "query", "context")

        assert "max_tokens" not in captured_kwargs[0]
        assert captured_kwargs[1]["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_execute_handles_missing_placeholder(self):
        """Framework handles missing placeholders gracefully."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(return_value="output")

        config = FrameworkConfig(
            name="test",
            steps=[
                FrameworkStep("step", "Use {nonexistent_placeholder}"),
            ]
        )

        # Should not raise - missing placeholders get empty string
        result = await execute_framework(config, mock_sampler, "query", "context")
        assert result["final_answer"] == "output"


class TestRunFramework:
    """Tests for run_framework convenience function."""

    @pytest.mark.asyncio
    async def test_run_existing_framework(self):
        """Run an existing framework by name."""
        mock_sampler = MagicMock(spec=ClientSampler)
        mock_sampler.request_sample = AsyncMock(return_value="output")

        # PROGRAM_OF_THOUGHTS has 4 steps
        result = await run_framework(
            "program_of_thoughts",
            mock_sampler,
            "test query",
            "test context"
        )

        assert result["metadata"]["framework"] == "program_of_thoughts"
        assert result["metadata"]["steps"] == 4

    @pytest.mark.asyncio
    async def test_run_nonexistent_framework_raises(self):
        """Running nonexistent framework raises KeyError."""
        mock_sampler = MagicMock(spec=ClientSampler)

        with pytest.raises(KeyError, match="not found"):
            await run_framework(
                "nonexistent_framework",
                mock_sampler,
                "query",
                "context"
            )


class TestFrameworkRegistry:
    """Tests for framework registry functions."""

    def test_get_available_frameworks(self):
        """Get list of all available frameworks."""
        frameworks = get_available_frameworks()
        assert isinstance(frameworks, list)
        assert len(frameworks) > 0
        assert "program_of_thoughts" in frameworks
        assert "chain_of_verification" in frameworks
        assert "critic" in frameworks

    def test_get_framework_description(self):
        """Get description of a framework."""
        desc = get_framework_description("program_of_thoughts")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_description_nonexistent_raises(self):
        """Getting description of nonexistent framework raises."""
        with pytest.raises(KeyError, match="not found"):
            get_framework_description("nonexistent")


class TestPredefinedFrameworks:
    """Tests for predefined framework configurations."""

    def test_program_of_thoughts_config(self):
        """PROGRAM_OF_THOUGHTS has expected structure."""
        assert PROGRAM_OF_THOUGHTS.name == "program_of_thoughts"
        assert len(PROGRAM_OF_THOUGHTS.steps) == 4
        step_names = [s.name for s in PROGRAM_OF_THOUGHTS.steps]
        assert "understand" in step_names
        assert "decompose" in step_names
        assert "code" in step_names
        assert "trace" in step_names

    def test_chain_of_verification_config(self):
        """CHAIN_OF_VERIFICATION has expected structure."""
        assert CHAIN_OF_VERIFICATION.name == "chain_of_verification"
        assert len(CHAIN_OF_VERIFICATION.steps) == 4
        step_names = [s.name for s in CHAIN_OF_VERIFICATION.steps]
        assert "draft" in step_names
        assert "verify" in step_names
        assert "patch" in step_names
        assert "validate" in step_names

    def test_critic_config(self):
        """CRITIC has expected structure."""
        assert CRITIC.name == "critic"
        assert len(CRITIC.steps) == 4
        step_names = [s.name for s in CRITIC.steps]
        assert "solution" in step_names
        assert "critique" in step_names
        assert "revised" in step_names

    def test_all_frameworks_have_names(self):
        """All registered frameworks have non-empty names."""
        for name, config in FRAMEWORK_CONFIGS.items():
            assert name == config.name
            assert len(config.name) > 0

    def test_all_frameworks_have_steps(self):
        """All registered frameworks have at least one step."""
        for name, config in FRAMEWORK_CONFIGS.items():
            assert len(config.steps) > 0, f"{name} has no steps"

    def test_all_steps_have_valid_templates(self):
        """All framework steps have valid prompt templates."""
        for name, config in FRAMEWORK_CONFIGS.items():
            for step in config.steps:
                assert len(step.prompt_template) > 0
                # Templates should contain at least one of the standard placeholders
                # or reference a previous step
                assert (
                    "{query}" in step.prompt_template or
                    "{context}" in step.prompt_template or
                    "{previous}" in step.prompt_template or
                    any(s.name in step.prompt_template for s in config.steps)
                ), f"{name}.{step.name} has no valid placeholders"

    def test_framework_temperatures_in_range(self):
        """All framework temperatures are in valid range 0-1."""
        for name, config in FRAMEWORK_CONFIGS.items():
            for step in config.steps:
                assert 0.0 <= step.temperature <= 1.0, (
                    f"{name}.{step.name} has invalid temperature {step.temperature}"
                )
