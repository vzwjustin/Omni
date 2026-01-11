"""
MCP Reasoning Router Schemas

Implements the structured handoff protocol between Gemini (orchestration)
and Claude Code (execution) with evidence-gated pipelines.
"""

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# =============================================================================
# ENUMS
# =============================================================================


class TaskType(str, Enum):
    DEBUG = "DEBUG"
    IMPLEMENT = "IMPLEMENT"
    REFACTOR = "REFACTOR"
    IMPROVE = "IMPROVE"
    ADD_FEATURE = "ADD_FEATURE"
    DOCS = "DOCS"
    PERF = "PERF"
    SECURITY = "SECURITY"
    TESTING = "TESTING"
    RELEASE = "RELEASE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CostClass(str, Enum):
    LIGHT = "LIGHT"
    MEDIUM = "MEDIUM"
    HEAVY = "HEAVY"


class StageRole(str, Enum):
    SCOUT = "SCOUT"
    ARCHITECT = "ARCHITECT"
    OPERATOR = "OPERATOR"


class SignalType(str, Enum):
    STACK_TRACE = "STACK_TRACE"
    FAILING_TESTS = "FAILING_TESTS"
    REPRO_STEPS = "REPRO_STEPS"
    PERF_REGRESSION = "PERF_REGRESSION"
    API_CONTRACT_CHANGE = "API_CONTRACT_CHANGE"
    AMBIGUOUS_REQUIREMENTS = "AMBIGUOUS_REQUIREMENTS"
    MULTI_SERVICE = "MULTI_SERVICE"
    MIGRATION = "MIGRATION"
    DEPENDENCY_CONFLICT = "DEPENDENCY_CONFLICT"
    ENVIRONMENT_SPECIFIC = "ENVIRONMENT_SPECIFIC"
    SECURITY_RELEVANT = "SECURITY_RELEVANT"
    UI_ONLY = "UI_ONLY"
    DATA_INTEGRITY = "DATA_INTEGRITY"


class ConfidenceBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SourceType(str, Enum):
    FILE = "FILE"
    DIFF = "DIFF"
    LOG = "LOG"
    STACK_TRACE = "STACK_TRACE"
    TEST_OUTPUT = "TEST_OUTPUT"
    COMMAND_OUTPUT = "COMMAND_OUTPUT"
    USER_TEXT = "USER_TEXT"


class OutputType(str, Enum):
    FACTS = "FACTS"
    DERIVED = "DERIVED"
    ASSUMPTIONS = "ASSUMPTIONS"
    OPEN_QUESTIONS = "OPEN_QUESTIONS"
    DECISIONS = "DECISIONS"
    NEXT_ACTIONS = "NEXT_ACTIONS"
    VERIFICATION_PLAN = "VERIFICATION_PLAN"
    PATCH_PLAN = "PATCH_PLAN"


class GateAction(str, Enum):
    PROCEED = "PROCEED"
    DOWNGRADE_PIPELINE = "DOWNGRADE_PIPELINE"
    REQUEST_MORE_INPUT = "REQUEST_MORE_INPUT"
    SWITCH_FRAMEWORK = "SWITCH_FRAMEWORK"
    ABORT_UNSAFE = "ABORT_UNSAFE"


class FallbackAction(str, Enum):
    USE_SAFE_BASELINE = "USE_SAFE_BASELINE"
    REQUEST_MORE_INPUT = "REQUEST_MORE_INPUT"
    DOWNGRADE_TO_SINGLE_STAGE = "DOWNGRADE_TO_SINGLE_STAGE"


class InputPriority(str, Enum):
    MUST = "MUST"
    SHOULD = "SHOULD"
    NICE = "NICE"


# =============================================================================
# EVIDENCE & STATE SCHEMAS
# =============================================================================


class EvidenceExcerpt(BaseModel):
    """Verbatim evidence from source with relevance annotation."""

    source_type: SourceType
    ref: str = Field(..., min_length=1, max_length=240)
    range: str | None = Field(None, max_length=80)
    content: str = Field(..., min_length=1, max_length=2000)
    relevance: str = Field(..., min_length=5, max_length=220)


class StageState(BaseModel):
    """Structured output from each pipeline stage."""

    facts: list[str] = Field(default_factory=list)
    derived: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    verification_plan: list[str] = Field(default_factory=list)
    patch_plan: list[str] = Field(default_factory=list)


# =============================================================================
# FRAMEWORK DEFINITION
# =============================================================================


class TriggerCondition(BaseModel):
    """Signal that increases likelihood of using this framework."""

    signal: SignalType
    weight: int = Field(..., ge=1, le=10)
    notes: str = Field("", max_length=180)


class InputRequirement(BaseModel):
    """Required/optional input for framework."""

    name: str = Field(..., min_length=2, max_length=80)
    priority: InputPriority
    why: str = Field("", max_length=180)


class FrameworkDefinition(BaseModel):
    """Complete framework definition with triggers and contracts."""

    id: str = Field(..., pattern=r"^[a-z0-9_\-]{3,64}$")
    name: str = Field(..., min_length=3, max_length=80)
    description: str = Field(..., min_length=20, max_length=800)
    role_fit: list[StageRole] = Field(..., min_items=1)
    best_for: list[str] = Field(..., min_items=1)
    triggers: list[TriggerCondition] = Field(default_factory=list)
    anti_triggers: list[TriggerCondition] = Field(default_factory=list)
    inputs_required: list[InputRequirement] = Field(default_factory=list)
    cost_class: CostClass
    stop_conditions: list[str] = Field(..., min_items=1)
    outputs_emitted: list[OutputType] = Field(..., min_items=1)
    tags: list[str] = Field(default_factory=list)


# =============================================================================
# INTEGRITY GATE
# =============================================================================


class Confidence(BaseModel):
    """Confidence score with band."""

    band: ConfidenceBand
    score: float = Field(..., ge=0, le=1)


class AlignmentCheck(BaseModel):
    """Check if pipeline matches user goal."""

    matches_user_goal: bool
    notes: str = Field("", max_length=240)


class GateRecommendation(BaseModel):
    """Action recommendation from integrity gate."""

    action: GateAction
    notes: str = Field("", max_length=240)


class IntegrityGate(BaseModel):
    """Quality gate between stages - validates evidence and alignment."""

    top_facts: list[str] = Field(..., min_items=1, max_items=5)
    top_assumptions: list[str] = Field(default_factory=list, max_items=5)
    falsifiers: list[str] = Field(default_factory=list, max_items=5)
    alignment_check: AlignmentCheck
    confidence: Confidence
    recommendation: GateRecommendation


# =============================================================================
# CLAUDE CODE BRIEF (OUTPUT TO CLAUDE)
# =============================================================================


class RepoTargets(BaseModel):
    """Files and areas to focus on / avoid."""

    files: list[str] = Field(default_factory=list)
    areas: list[str] = Field(default_factory=list)
    do_not_touch: list[str] = Field(default_factory=list)


class Verification(BaseModel):
    """Commands and criteria to verify success."""

    commands: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class ClaudeCodeBrief(BaseModel):
    """
    Compact execution brief for Claude Code.

    Token budget: 700 target, 1100 hard cap.
    Evidence: max 6 excerpts, max 1800 chars each.
    """

    objective: str = Field(..., min_length=5, max_length=260)
    task_type: TaskType
    constraints: list[str] = Field(default_factory=list)
    repo_targets: RepoTargets
    execution_plan: list[str] = Field(..., min_items=1)
    verification: Verification
    stop_conditions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceExcerpt] = Field(default_factory=list, max_items=6)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert brief to Claude-friendly prompt format."""
        lines = [f"# Objective: {self.objective}", f"Task Type: {self.task_type.value}", ""]

        if self.constraints:
            lines.append("## Constraints")
            for c in self.constraints:
                lines.append(f"- {c}")
            lines.append("")

        if self.repo_targets.files or self.repo_targets.areas:
            lines.append("## Focus Areas")
            if self.repo_targets.files:
                lines.append("Files: " + ", ".join(self.repo_targets.files[:5]))
            if self.repo_targets.areas:
                lines.append("Areas: " + ", ".join(self.repo_targets.areas[:3]))
            if self.repo_targets.do_not_touch:
                lines.append("Do NOT touch: " + ", ".join(self.repo_targets.do_not_touch[:3]))
            lines.append("")

        lines.append("## Execution Plan")
        for i, step in enumerate(self.execution_plan, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

        if self.verification.commands or self.verification.acceptance_criteria:
            lines.append("## Verification")
            if self.verification.commands:
                lines.append("Commands: " + " && ".join(self.verification.commands[:3]))
            if self.verification.acceptance_criteria:
                lines.append("Criteria:")
                for c in self.verification.acceptance_criteria[:3]:
                    lines.append(f"  - {c}")
            lines.append("")

        if self.stop_conditions:
            lines.append("## Stop Conditions")
            for s in self.stop_conditions[:3]:
                lines.append(f"- {s}")
            lines.append("")

        if self.evidence:
            lines.append("## Evidence")
            for e in self.evidence[:3]:
                lines.append(f"[{e.source_type.value}] {e.ref}")
                lines.append(f"  {e.content[:500]}...")
                lines.append(f"  Relevance: {e.relevance}")
            lines.append("")

        if self.assumptions:
            lines.append("## Assumptions")
            for a in self.assumptions[:3]:
                lines.append(f"- {a}")
            lines.append("")

        if self.open_questions:
            lines.append("## Open Questions")
            for q in self.open_questions[:3]:
                lines.append(f"- {q}")

        return "\n".join(lines)

    def to_compact_prompt(self) -> str:
        """Token-optimized brief format - rich context, efficient format."""
        lines = [f"**{self.task_type.value}**: {self.objective}"]

        # Targets - single line
        targets = []
        if self.repo_targets.files:
            targets.append(f"files={','.join(self.repo_targets.files[:4])}")
        if self.repo_targets.areas:
            targets.append(f"areas={','.join(self.repo_targets.areas[:3])}")
        if self.repo_targets.do_not_touch:
            targets.append(f"avoid={','.join(self.repo_targets.do_not_touch[:2])}")
        if targets:
            lines.append(f"**Scope**: {' | '.join(targets)}")

        # Execution plan - numbered but compact
        if self.execution_plan:
            steps = [f"{i + 1}.{s}" for i, s in enumerate(self.execution_plan[:5])]
            lines.append(f"**Plan**: {' → '.join(steps)}")

        # Verification - single line
        if self.verification.commands or self.verification.acceptance_criteria:
            v_parts = []
            if self.verification.commands:
                v_parts.append(f"run: {' && '.join(self.verification.commands[:2])}")
            if self.verification.acceptance_criteria:
                v_parts.append(f"pass: {'; '.join(self.verification.acceptance_criteria[:2])}")
            lines.append(f"**Verify**: {' | '.join(v_parts)}")

        # Constraints - single line
        if self.constraints:
            lines.append(f"**Constraints**: {'; '.join(self.constraints[:3])}")

        # Stop conditions - single line
        if self.stop_conditions:
            lines.append(f"**Stop if**: {'; '.join(self.stop_conditions[:2])}")

        # Evidence - compact but full content
        if self.evidence:
            lines.append("**Evidence**:")
            for e in self.evidence[:4]:
                lines.append(f"  [{e.source_type.value}] {e.ref}: {e.content[:300]}")

        # Assumptions & questions - single lines
        if self.assumptions:
            lines.append(f"**Assumes**: {'; '.join(self.assumptions[:3])}")
        if self.open_questions:
            lines.append(f"**Questions**: {'; '.join(self.open_questions[:3])}")

        return "\n".join(lines)

    def to_surgical_prompt(self) -> str:
        """
        Token-efficient brief that preserves ALL information.

        Uses bullet points for efficiency - clear structure without prose.
        Claude gets everything it needs.
        """
        lines = [f"[{self.task_type.value}] {self.objective}"]

        # Targets with bullet points
        if self.repo_targets.files:
            lines.append(f"→ {' '.join(self.repo_targets.files)}")
        if self.repo_targets.areas:
            for area in self.repo_targets.areas:
                lines.append(f"  • {area}")
        if self.repo_targets.do_not_touch:
            for avoid in self.repo_targets.do_not_touch:
                lines.append(f"  ⊘ {avoid}")

        # ALL execution steps with numbers
        if self.execution_plan:
            lines.append("")
            for i, step in enumerate(self.execution_plan, 1):
                lines.append(f"{i}. {step}")

        # Verification with bullets
        if self.verification.commands or self.verification.acceptance_criteria:
            lines.append("")
            for cmd in self.verification.commands:
                lines.append(f"✓ {cmd}")
            for criteria in self.verification.acceptance_criteria:
                lines.append(f"• {criteria}")

        # ALL constraints with bullets
        if self.constraints:
            lines.append("")
            for c in self.constraints:
                lines.append(f"⚠ {c}")

        # ALL evidence with bullets
        if self.evidence:
            lines.append("")
            for e in self.evidence:
                lines.append(f"• [{e.source_type.value}] {e.ref}")
                lines.append(f"  → {e.relevance}")
                if len(e.content) <= 150:
                    lines.append(f"  {e.content}")

        # ALL assumptions with bullets
        if self.assumptions:
            lines.append("")
            for a in self.assumptions:
                lines.append(f"• {a}")

        # ALL open questions with bullets
        if self.open_questions:
            for q in self.open_questions:
                lines.append(f"? {q}")

        # ALL stop conditions with bullets
        if self.stop_conditions:
            lines.append("")
            for s in self.stop_conditions:
                lines.append(f"⛔ {s}")

        return "\n".join(lines)

    def token_estimate(self) -> int:
        """Estimate token count for the surgical prompt."""
        return len(self.to_surgical_prompt()) // 4


# =============================================================================
# PIPELINE STAGES
# =============================================================================


class StageInputs(BaseModel):
    """What a stage receives."""

    facts_only: bool = True
    derived_allowed: bool = True
    evidence: list[EvidenceExcerpt] = Field(default_factory=list, max_items=6)


class StageBudget(BaseModel):
    """Cost constraints for stage."""

    cost_class: CostClass
    notes: str = Field("", max_length=160)


class PipelineStage(BaseModel):
    """Single stage in the framework pipeline."""

    stage_role: StageRole
    framework_id: str = Field(..., pattern=r"^[a-z0-9_\-]{3,64}$")
    inputs: StageInputs
    expected_outputs: list[OutputType]
    budget: StageBudget


class SelectionRationale(BaseModel):
    """Explains why frameworks were chosen."""

    top_pick: str = Field(..., pattern=r"^[a-z0-9_\-]{3,64}$")
    runner_up: str = Field(..., pattern=r"^[a-z0-9_\-]{3,64}$")
    why_top_pick: str = Field(..., min_length=5, max_length=260)
    why_not_runner_up: str = Field("", max_length=260)
    confidence: Confidence


class PipelineFallback(BaseModel):
    """What to do if pipeline fails."""

    action: FallbackAction
    framework_id: str = Field(..., pattern=r"^[a-z0-9_\-]{3,64}$")
    notes: str = Field("", max_length=200)


class Pipeline(BaseModel):
    """1-3 stage framework pipeline."""

    max_frameworks: int = Field(3, ge=1, le=3)
    stages: list[PipelineStage] = Field(..., min_items=1, max_items=3)
    selection_rationale: SelectionRationale
    fallback: PipelineFallback


# =============================================================================
# DETECTED SIGNALS
# =============================================================================


class DetectedSignal(BaseModel):
    """Signal detected in the input."""

    type: SignalType
    evidence_refs: list[str] = Field(default_factory=list)
    notes: str = Field("", max_length=200)


# =============================================================================
# TASK PROFILE
# =============================================================================


class TaskProfile(BaseModel):
    """High-level task classification."""

    task_type: TaskType
    risk_level: RiskLevel
    primary_goal: str = Field(..., min_length=5, max_length=240)
    constraints: list[str] = Field(default_factory=list)


# =============================================================================
# ROUTER METADATA
# =============================================================================


class RouterMeta(BaseModel):
    """Routing request metadata."""

    request_id: str = Field(
        default_factory=lambda: f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    )
    timestamp_iso: str = Field(default_factory=lambda: datetime.now().isoformat())
    router_version: str = "1.0.0"


class Telemetry(BaseModel):
    """Performance and cost tracking."""

    routing_latency_ms: int = Field(0, ge=0, le=60000)
    inputs_token_estimate: int = Field(0, ge=0, le=200000)
    brief_token_estimate: int = Field(0, ge=0, le=5000)
    notes: str = Field("", max_length=200)


# =============================================================================
# GEMINI ROUTER OUTPUT (FULL RESPONSE)
# =============================================================================


class GeminiRouterOutput(BaseModel):
    """
    Complete output from Gemini router.

    Contains:
    - Task profile and detected signals
    - 1-3 stage pipeline with rationale
    - Integrity gate with confidence
    - Claude Code brief (the actual handoff)
    - Telemetry
    """

    router_meta: RouterMeta = Field(default_factory=RouterMeta)
    task_profile: TaskProfile
    detected_signals: list[DetectedSignal] = Field(default_factory=list)
    pipeline: Pipeline
    integrity_gate: IntegrityGate
    claude_code_brief: ClaudeCodeBrief
    telemetry: Telemetry = Field(default_factory=Telemetry)

    def get_brief_prompt(self) -> str:
        """Get the Claude Code brief as a formatted prompt (detailed)."""
        return self.claude_code_brief.to_prompt()

    def get_surgical_prompt(self) -> str:
        """Get ultra token-efficient brief for Claude Max optimization."""
        return self.claude_code_brief.to_surgical_prompt()

    def get_pipeline_summary(self) -> str:
        """Get a summary of the pipeline for logging."""
        stages = " → ".join([s.framework_id for s in self.pipeline.stages])
        conf = self.integrity_gate.confidence
        return f"[Pipeline: {stages}] [Confidence: {conf.band.value} {conf.score:.2f}]"


# =============================================================================
# DEFAULTS & CONSTANTS
# =============================================================================

BUDGETS = {
    "claude_brief_token_target": 700,
    "claude_brief_token_hard_cap": 1100,
    "evidence_excerpt_max_items": 6,
    "evidence_excerpt_max_chars_each": 1800,
    "derived_summary_max_chars_total": 2200,
}

DEFAULT_STOP_CONDITIONS = [
    "If required inputs are missing (e.g., repro steps, failing test output), request them before proceeding.",
    "If verification fails twice with unrelated errors, stop and report the blockers and proposed next diagnostic command.",
    "If changes would impact public API/contracts without explicit approval, stop and ask for confirmation.",
]

SAFE_BASELINE_PIPELINE = Pipeline(
    max_frameworks=2,
    stages=[
        PipelineStage(
            stage_role=StageRole.SCOUT,
            framework_id="self_discover",
            inputs=StageInputs(facts_only=True, derived_allowed=True),
            expected_outputs=[
                OutputType.FACTS,
                OutputType.ASSUMPTIONS,
                OutputType.OPEN_QUESTIONS,
                OutputType.NEXT_ACTIONS,
            ],
            budget=StageBudget(
                cost_class=CostClass.LIGHT, notes="Minimize overhead; aim for crisp ticketing."
            ),
        ),
        PipelineStage(
            stage_role=StageRole.OPERATOR,
            framework_id="chain_of_thought",
            inputs=StageInputs(facts_only=True, derived_allowed=False),
            expected_outputs=[
                OutputType.DECISIONS,
                OutputType.NEXT_ACTIONS,
                OutputType.VERIFICATION_PLAN,
            ],
            budget=StageBudget(
                cost_class=CostClass.MEDIUM, notes="Execute with verification-first behavior."
            ),
        ),
    ],
    selection_rationale=SelectionRationale(
        top_pick="self_discover",
        runner_up="chain_of_thought",
        why_top_pick="Safe baseline for unknown task patterns.",
        why_not_runner_up="Used as second stage for execution.",
        confidence=Confidence(band=ConfidenceBand.MEDIUM, score=0.6),
    ),
    fallback=PipelineFallback(
        action=FallbackAction.USE_SAFE_BASELINE,
        framework_id="chain_of_thought",
        notes="Fallback to general reasoning if specialized routing fails.",
    ),
)
