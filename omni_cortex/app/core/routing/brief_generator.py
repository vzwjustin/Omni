"""
Structured Brief Generator for Omni-Cortex

Generates GeminiRouterOutput with ClaudeCodeBrief for handoff protocol.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from ..constants import CONTENT
from .framework_registry import CATEGORIES
from .task_analysis import (
    enrich_evidence_from_chroma,
    gemini_analyze_task,
    save_task_analysis,
)

if TYPE_CHECKING:
    from ..router import HyperRouter
    from ..schemas import (
        ClaudeCodeBrief,
        DetectedSignal,
        GeminiRouterOutput,
        OutputType,
        RiskLevel,
        StageRole,
        TaskType,
    )

logger = structlog.get_logger("brief_generator")

# Claude Max token budget - keep briefs ultra-efficient
MAX_CLAUDE_TOKENS = 500


class StructuredBriefGenerator:
    """
    Generates structured GeminiRouterOutput with ClaudeCodeBrief.

    This is the main entry point for the structured handoff protocol.
    Gemini orchestrates, Claude executes.
    """

    def __init__(self, router: HyperRouter):
        """Initialize with parent router for framework selection."""
        self.router = router

    async def generate(  # noqa: PLR0912, PLR0915
        self,
        query: str,
        context: str | None = None,
        code_snippet: str | None = None,
        ide_context: str | None = None,
        file_list: list[str] | None = None,
    ) -> GeminiRouterOutput:
        """Generate a structured GeminiRouterOutput with ClaudeCodeBrief."""
        from ..schemas import (
            DEFAULT_STOP_CONDITIONS,
            AlignmentCheck,
            ClaudeCodeBrief,
            Confidence,
            ConfidenceBand,
            CostClass,
            EvidenceExcerpt,
            FallbackAction,
            GateAction,
            GateRecommendation,
            GeminiRouterOutput,
            IntegrityGate,
            Pipeline,
            PipelineFallback,
            PipelineStage,
            RepoTargets,
            RouterMeta,
            SelectionRationale,
            SourceType,
            StageBudget,
            StageInputs,
            StageRole,
            TaskProfile,
            Telemetry,
            Verification,
        )

        start_time = time.monotonic()  # monotonic for elapsed time measurement

        # Detect signals from input
        detected_signals = self._detect_signals(query, context, code_snippet)

        # Route to category and select frameworks
        try:
            framework_chain, reasoning, category = await self.router.select_framework_chain(
                query, code_snippet, ide_context
            )
        except Exception as e:
            # Graceful degradation: Framework selection failures should not block brief generation.
            # We fall back to safe baseline frameworks to ensure the user always gets a response.
            logger.warning(
                "framework_selection_failed",
                error=str(e)[: CONTENT.QUERY_LOG],
                error_type=type(e).__name__,
                fallback="self_discover->chain_of_thought",
            )
            framework_chain = ["self_discover", "chain_of_thought"]
            reasoning = "Safe baseline fallback"
            category = "exploration"

        # Determine task type from signals and query
        task_type = self._detect_task_type(query, detected_signals)
        risk_level = self._assess_risk(query, context, detected_signals)

        # Build pipeline stages (max 3)
        stages = []
        role_sequence = [StageRole.SCOUT, StageRole.ARCHITECT, StageRole.OPERATOR]
        for i, fw in enumerate(framework_chain[:3]):
            stage_role = role_sequence[min(i, 2)]
            cost = CostClass.LIGHT if i == 0 else (CostClass.MEDIUM if i == 1 else CostClass.HEAVY)

            stages.append(
                PipelineStage(
                    stage_role=stage_role,
                    framework_id=fw,
                    inputs=StageInputs(facts_only=(i > 0), derived_allowed=(i < 2), evidence=[]),
                    expected_outputs=self._get_expected_outputs(stage_role),
                    budget=StageBudget(cost_class=cost, notes=f"Stage {i + 1}: {fw}"),
                )
            )

        # Calculate confidence
        confidence_score = self._calculate_confidence(detected_signals, framework_chain, category)
        confidence_band = (
            ConfidenceBand.HIGH
            if confidence_score >= 0.75
            else ConfidenceBand.MEDIUM
            if confidence_score >= 0.5
            else ConfidenceBand.LOW
        )

        # Build pipeline
        pipeline = Pipeline(
            max_frameworks=min(3, len(framework_chain)),
            stages=stages,
            selection_rationale=SelectionRationale(
                top_pick=framework_chain[0] if framework_chain else "self_discover",
                runner_up=framework_chain[1] if len(framework_chain) > 1 else "chain_of_thought",
                why_top_pick=reasoning[:260],
                why_not_runner_up="Used as subsequent stage"
                if len(framework_chain) > 1
                else "Alternative approach",
                confidence=Confidence(band=confidence_band, score=confidence_score),
            ),
            fallback=PipelineFallback(
                action=FallbackAction.USE_SAFE_BASELINE,
                framework_id="self_discover",
                notes="Fallback to self_discover if pipeline fails",
            ),
        )

        # Build integrity gate
        facts = self._extract_facts(query, context, code_snippet)
        assumptions = self._extract_assumptions(query, context)

        integrity_gate = IntegrityGate(
            top_facts=facts[:5] if facts else ["Query received: " + query[: CONTENT.QUERY_LOG]],
            top_assumptions=assumptions[:5],
            falsifiers=self._generate_falsifiers(query, detected_signals),
            alignment_check=AlignmentCheck(
                matches_user_goal=True,
                notes=f"Pipeline matches detected task type: {task_type.value}",
            ),
            confidence=Confidence(band=confidence_band, score=confidence_score),
            recommendation=GateRecommendation(
                action=GateAction.PROCEED
                if confidence_score >= 0.5
                else GateAction.REQUEST_MORE_INPUT,
                notes="Proceed with pipeline"
                if confidence_score >= 0.5
                else "Low confidence - may need more context",
            ),
        )

        # Build evidence excerpts
        evidence = []
        if code_snippet:
            evidence.append(
                EvidenceExcerpt(
                    source_type=SourceType.FILE,
                    ref="code_snippet",
                    content=code_snippet[:1800],
                    relevance="User-provided code context",
                )
            )
        if context and len(context) > 10:
            evidence.append(
                EvidenceExcerpt(
                    source_type=SourceType.USER_TEXT,
                    ref="context",
                    content=context[:1800],
                    relevance="User-provided context",
                )
            )

        # Use Gemini for rich task analysis
        gemini_analysis = await gemini_analyze_task(query, context, framework_chain, category)

        # Use Gemini's analysis if available, fallback to static templates
        if gemini_analysis.get("execution_plan"):
            execution_plan = gemini_analysis["execution_plan"][:5]
        else:
            execution_plan = self._generate_execution_plan(query, framework_chain, task_type)

        if gemini_analysis.get("focus_areas"):
            focus_areas = gemini_analysis["focus_areas"][:5]
        else:
            focus_areas = self._extract_areas(query, context)

        if gemini_analysis.get("assumptions"):
            rich_assumptions = gemini_analysis["assumptions"][:5]
        else:
            rich_assumptions = assumptions[:5]

        if gemini_analysis.get("questions"):
            open_questions = gemini_analysis["questions"][:5]
        else:
            open_questions = self._generate_open_questions(query, detected_signals)

        # Add prior knowledge as evidence if available
        if gemini_analysis.get("prior_knowledge") and gemini_analysis["prior_knowledge"] != "none":
            evidence.append(
                EvidenceExcerpt(
                    source_type=SourceType.USER_TEXT,
                    ref="prior_knowledge",
                    content=gemini_analysis["prior_knowledge"][: CONTENT.SNIPPET_SHORT],
                    relevance="Relevant insight from similar past problems",
                )
            )

        # Enrich with Chroma RAG context
        chroma_evidence = await enrich_evidence_from_chroma(
            query=query,
            category=category,
            framework_chain=framework_chain,
            task_type=task_type.value if hasattr(task_type, "value") else str(task_type),
        )
        evidence.extend(chroma_evidence)

        claude_brief = ClaudeCodeBrief(
            objective=self._generate_objective(query, task_type),
            task_type=task_type,
            constraints=self._extract_constraints(query, context),
            repo_targets=RepoTargets(
                files=file_list[:10] if file_list else [], areas=focus_areas, do_not_touch=[]
            ),
            execution_plan=execution_plan,
            verification=Verification(
                commands=self._suggest_verification_commands(task_type),
                acceptance_criteria=self._generate_acceptance_criteria(query, task_type),
            ),
            stop_conditions=DEFAULT_STOP_CONDITIONS[:3],
            evidence=evidence[:6],
            assumptions=rich_assumptions,
            open_questions=open_questions,
        )

        # Enforce token budget for Claude Max efficiency
        claude_brief = self._enforce_token_budget(claude_brief)

        # Save analysis to Chroma for future reference
        await save_task_analysis(query, gemini_analysis, framework_chain, category)

        # Calculate telemetry - use surgical prompt for accurate Claude token estimate
        routing_latency = int((time.monotonic() - start_time) * 1000)
        inputs_estimate = len(query) + len(context or "") + len(code_snippet or "")
        brief_estimate = claude_brief.token_estimate()  # Uses surgical prompt

        # Build full output
        output = GeminiRouterOutput(
            router_meta=RouterMeta(),
            task_profile=TaskProfile(
                task_type=task_type,
                risk_level=risk_level,
                primary_goal=query[:240],
                constraints=claude_brief.constraints,
            ),
            detected_signals=detected_signals,
            pipeline=pipeline,
            integrity_gate=integrity_gate,
            claude_code_brief=claude_brief,
            telemetry=Telemetry(
                routing_latency_ms=routing_latency,
                inputs_token_estimate=inputs_estimate // 4,
                brief_token_estimate=brief_estimate // 4,
                notes=f"Pipeline: {' -> '.join(framework_chain[:3])}",
            ),
        )

        return output

    def _detect_signals(
        self, query: str, context: str | None, code_snippet: str | None
    ) -> list[DetectedSignal]:
        """Detect signals from input to guide framework selection."""
        from ..schemas import DetectedSignal, SignalType

        signals = []
        combined = f"{query} {context or ''} {code_snippet or ''}".lower()

        signal_patterns = {
            SignalType.STACK_TRACE: [
                "traceback",
                "exception",
                "error at line",
                "stack trace",
                "at line",
            ],
            SignalType.FAILING_TESTS: [
                "test fail",
                "assertion error",
                "expected",
                "actual",
                "npm test",
                "pytest",
            ],
            SignalType.REPRO_STEPS: ["to reproduce", "steps:", "1.", "when i", "after running"],
            SignalType.PERF_REGRESSION: [
                "slow",
                "performance",
                "latency",
                "timeout",
                "memory leak",
            ],
            SignalType.API_CONTRACT_CHANGE: ["api", "endpoint", "breaking change", "deprecat"],
            SignalType.AMBIGUOUS_REQUIREMENTS: [
                "unclear",
                "not sure",
                "maybe",
                "should i",
                "which approach",
            ],
            SignalType.MULTI_SERVICE: [
                "microservice",
                "service a",
                "service b",
                "cross-service",
                "distributed",
            ],
            SignalType.MIGRATION: ["migrate", "upgrade", "v2", "legacy", "deprecate"],
            SignalType.DEPENDENCY_CONFLICT: [
                "dependency",
                "version conflict",
                "incompatible",
                "peer dep",
            ],
            SignalType.ENVIRONMENT_SPECIFIC: [
                "only in prod",
                "works locally",
                "docker",
                "kubernetes",
            ],
            SignalType.SECURITY_RELEVANT: [
                "security",
                "vulnerability",
                "auth",
                "injection",
                "xss",
                "csrf",
            ],
            SignalType.UI_ONLY: ["css", "layout", "style", "ui", "frontend", "display"],
            SignalType.DATA_INTEGRITY: ["data loss", "corrupt", "integrity", "consistency"],
        }

        for signal_type, patterns in signal_patterns.items():
            if any(p in combined for p in patterns):
                signals.append(
                    DetectedSignal(
                        type=signal_type,
                        evidence_refs=["query", "context"],
                        notes="Detected from input patterns",
                    )
                )

        return signals

    def _detect_task_type(self, query: str, signals: list[DetectedSignal]) -> TaskType:  # noqa: PLR0911
        """Determine task type from query and signals."""
        from ..schemas import SignalType, TaskType

        query_lower = query.lower()

        # Check signals first
        signal_types = {s.type for s in signals}
        if SignalType.STACK_TRACE in signal_types or SignalType.FAILING_TESTS in signal_types:
            return TaskType.DEBUG
        if SignalType.PERF_REGRESSION in signal_types:
            return TaskType.PERF
        if SignalType.SECURITY_RELEVANT in signal_types:
            return TaskType.SECURITY

        # Check query patterns
        if any(w in query_lower for w in ["fix", "bug", "error", "broken", "not working", "debug"]):
            return TaskType.DEBUG
        if any(w in query_lower for w in ["add", "create", "new feature", "implement"]):
            return TaskType.ADD_FEATURE
        if any(w in query_lower for w in ["refactor", "clean", "restructure", "reorganize"]):
            return TaskType.REFACTOR
        if any(w in query_lower for w in ["improve", "optimize", "enhance", "better"]):
            return TaskType.IMPROVE
        if any(w in query_lower for w in ["document", "readme", "docs", "comment"]):
            return TaskType.DOCS
        if any(w in query_lower for w in ["test", "coverage", "spec"]):
            return TaskType.TESTING
        if any(w in query_lower for w in ["deploy", "release", "publish"]):
            return TaskType.RELEASE

        return TaskType.IMPLEMENT

    def _assess_risk(
        self, query: str, context: str | None, signals: list[DetectedSignal]
    ) -> RiskLevel:
        """Assess risk level of the task."""
        from ..schemas import RiskLevel, SignalType

        signal_types = {s.type for s in signals}

        # Critical indicators
        if SignalType.SECURITY_RELEVANT in signal_types:
            return RiskLevel.CRITICAL
        if SignalType.DATA_INTEGRITY in signal_types:
            return RiskLevel.HIGH

        # High risk indicators
        combined = f"{query} {context or ''}".lower()
        if any(w in combined for w in ["production", "live", "customer", "payment", "auth"]):
            return RiskLevel.HIGH

        if SignalType.API_CONTRACT_CHANGE in signal_types or SignalType.MIGRATION in signal_types:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _get_expected_outputs(self, role: StageRole) -> list[OutputType]:
        """Get expected outputs for a stage role."""
        from ..schemas import OutputType, StageRole

        if role == StageRole.SCOUT:
            return [OutputType.FACTS, OutputType.ASSUMPTIONS, OutputType.OPEN_QUESTIONS]
        elif role == StageRole.ARCHITECT:
            return [OutputType.DECISIONS, OutputType.NEXT_ACTIONS, OutputType.PATCH_PLAN]
        else:  # OPERATOR
            return [OutputType.PATCH_PLAN, OutputType.VERIFICATION_PLAN, OutputType.NEXT_ACTIONS]

    def _calculate_confidence(
        self, signals: list[DetectedSignal], framework_chain: list[str], category: str
    ) -> float:
        """Calculate confidence score for the routing decision."""
        base = 0.5

        # More signals = more context = higher confidence
        base += min(len(signals) * 0.1, 0.2)

        # Known category = higher confidence
        if category in CATEGORIES:
            base += 0.1

        # Framework in category = higher confidence
        if framework_chain and category in CATEGORIES:
            cat_frameworks = CATEGORIES[category].get("frameworks", [])
            if framework_chain[0] in cat_frameworks:
                base += 0.1

        return min(base, 1.0)

    def _extract_facts(
        self, query: str, context: str | None, code_snippet: str | None
    ) -> list[str]:
        """Extract factual statements from input."""
        facts = [f"User query: {query[: CONTENT.QUERY_LOG]}"]

        if context:
            facts.append(f"Context provided: {len(context)} chars")
        if code_snippet:
            facts.append(f"Code snippet provided: {len(code_snippet)} chars")

        return facts

    def _extract_assumptions(self, query: str, context: str | None) -> list[str]:
        """Extract assumptions from input."""
        assumptions = []

        if not context:
            assumptions.append("No additional context provided - may need more information")

        if "?" in query:
            assumptions.append("User is seeking guidance, not just execution")

        return assumptions if assumptions else ["Standard implementation approach is acceptable"]

    def _generate_falsifiers(self, _query: str, signals: list[DetectedSignal]) -> list[str]:
        """Generate conditions that would invalidate the approach."""
        falsifiers = [
            "If the root cause is in a different module than identified",
            "If the fix requires breaking API changes",
        ]

        if any(s.type.value == "FAILING_TESTS" for s in signals):
            falsifiers.append("If tests fail for unrelated reasons after fix")

        return falsifiers[:5]

    def _generate_objective(self, query: str, task_type: TaskType) -> str:
        """Generate a clear objective statement."""
        from ..schemas import TaskType

        prefixes = {
            TaskType.DEBUG: "Fix: ",
            TaskType.IMPLEMENT: "Implement: ",
            TaskType.REFACTOR: "Refactor: ",
            TaskType.IMPROVE: "Improve: ",
            TaskType.ADD_FEATURE: "Add feature: ",
            TaskType.DOCS: "Document: ",
            TaskType.PERF: "Optimize: ",
            TaskType.SECURITY: "Secure: ",
            TaskType.TESTING: "Test: ",
            TaskType.RELEASE: "Release: ",
        }

        prefix = prefixes.get(task_type, "")
        return f"{prefix}{query[: CONTENT.ERROR_PREVIEW]}"

    def _extract_constraints(self, query: str, context: str | None) -> list[str]:
        """Extract constraints from input."""
        constraints = []
        combined = f"{query} {context or ''}".lower()

        if "don't" in combined or "do not" in combined:
            constraints.append("Respect explicit restrictions in query")
        if "existing" in combined or "current" in combined:
            constraints.append("Preserve existing functionality")
        if "api" in combined or "contract" in combined:
            constraints.append("Do not break public API contracts")

        return constraints if constraints else ["Follow existing code conventions"]

    def _extract_areas(self, query: str, context: str | None) -> list[str]:
        """Extract code areas to focus on."""
        areas = []
        combined = f"{query} {context or ''}".lower()

        # Common area patterns
        if "auth" in combined:
            areas.append("authentication")
        if "api" in combined or "endpoint" in combined:
            areas.append("api")
        if "database" in combined or "sql" in combined:
            areas.append("database")
        if "test" in combined:
            areas.append("tests")
        if "ui" in combined or "frontend" in combined:
            areas.append("frontend")

        return areas[:5]

    def _generate_execution_plan(
        self, _query: str, framework_chain: list[str], task_type: TaskType
    ) -> list[str]:
        """Generate step-by-step execution plan."""
        from ..schemas import TaskType

        if task_type == TaskType.DEBUG:
            return [
                "Identify the error location from logs/stack trace",
                "Understand the expected vs actual behavior",
                "Trace the data flow to find root cause",
                "Implement minimal fix without side effects",
                "Verify fix with targeted tests",
            ]
        elif task_type in (TaskType.IMPLEMENT, TaskType.ADD_FEATURE):
            return [
                "Review existing patterns in the codebase",
                "Design the implementation approach",
                "Implement core functionality",
                "Add error handling and edge cases",
                "Write tests and verify",
            ]
        elif task_type == TaskType.REFACTOR:
            return [
                "Understand current implementation",
                "Identify refactoring targets",
                "Apply changes incrementally",
                "Verify behavior is preserved",
                "Update tests if needed",
            ]
        else:
            return [
                f"Apply {framework_chain[0] if framework_chain else 'reasoning'} approach",
                "Gather relevant context",
                "Formulate solution",
                "Implement changes",
                "Verify results",
            ]

    def _suggest_verification_commands(self, task_type: TaskType) -> list[str]:
        """Suggest verification commands based on task type."""
        from ..schemas import TaskType

        base = ["npm test", "npm run lint"]

        if task_type in (TaskType.DEBUG, TaskType.TESTING):
            return ["npm test -- --coverage", "npm run lint"]
        elif task_type == TaskType.PERF:
            return ["npm run benchmark", "npm test"]
        elif task_type == TaskType.SECURITY:
            return ["npm audit", "npm run lint:security"]

        return base

    def _generate_acceptance_criteria(self, _query: str, task_type: TaskType) -> list[str]:
        """Generate acceptance criteria."""
        from ..schemas import TaskType

        criteria = ["All existing tests pass"]

        if task_type == TaskType.DEBUG:
            criteria.append("The reported issue is resolved")
        elif task_type == TaskType.ADD_FEATURE:
            criteria.append("New feature works as specified")
        elif task_type == TaskType.REFACTOR:
            criteria.append("Behavior is unchanged")

        criteria.append("No new linting errors")
        return criteria[:4]

    def _generate_open_questions(self, query: str, signals: list[DetectedSignal]) -> list[str]:
        """Generate open questions that may need clarification."""
        questions = []

        if "?" not in query:
            questions.append("Are there any constraints not mentioned?")

        if not signals:
            questions.append("Can you provide more context or examples?")

        return questions[:3]

    def _enforce_token_budget(self, brief: ClaudeCodeBrief) -> ClaudeCodeBrief:
        """
        Log token usage for monitoring - NO content removal.

        Claude gets ALL information. Efficiency comes from formatting.
        This just tracks usage for optimization insights.
        """
        token_count = brief.token_estimate()

        if token_count > MAX_CLAUDE_TOKENS:
            logger.info(
                "token_budget_exceeded",
                tokens=token_count,
                budget=MAX_CLAUDE_TOKENS,
                overage=token_count - MAX_CLAUDE_TOKENS,
                note="Content preserved - Claude gets full context",
            )
        else:
            logger.debug("token_budget_ok", tokens=token_count, budget=MAX_CLAUDE_TOKENS)

        # Never trim - return as-is
        return brief
