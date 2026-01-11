"""
Task Analysis for Omni-Cortex

Gemini-powered task analysis for intelligent routing.
Offloads thinking from Claude to cheaper Gemini.
"""

from typing import Any

import structlog

from ..constants import CONTENT

logger = structlog.get_logger("task_analysis")


async def gemini_analyze_task(
    query: str, context: str | None, framework_chain: list[str], category: str
) -> dict[str, Any]:
    """
    Use Gemini to generate rich task analysis.

    Returns specific execution plan, focus areas, assumptions, questions.
    This offloads thinking from Claude to cheaper Gemini.
    """
    try:
        from ...langchain_integration import get_routing_model

        # Pull relevant learnings from Chroma (category-aware)
        prior_learnings = await get_relevant_learnings(query, category)

        prompt = f"""Analyze this coding task and provide a detailed execution plan.

TASK: {query}
{f"CONTEXT: {context}" if context else ""}
FRAMEWORK CHAIN: {" -> ".join(framework_chain)}
CATEGORY: {category}

{f"## Similar Past Solutions:{chr(10)}{prior_learnings}" if prior_learnings else ""}

Generate a SPECIFIC execution plan (not generic steps). Include:
1. Exact files/areas to investigate
2. Specific code patterns to look for
3. Concrete implementation steps
4. Edge cases to handle
5. Verification approach

Respond in this EXACT format:
EXECUTION_PLAN:
- Step 1: [specific action]
- Step 2: [specific action]
- Step 3: [specific action]
- Step 4: [specific action]
- Step 5: [specific action]

FOCUS_AREAS: [comma-separated list of specific files/modules]

KEY_ASSUMPTIONS:
- [assumption 1]
- [assumption 2]

QUESTIONS_FOR_USER:
- [question if context unclear]

PRIOR_KNOWLEDGE: [relevant insights from similar problems, or "none"]
"""

        # Use dedicated Gemini routing model (always Gemini, regardless of LLM_PROVIDER)
        llm = get_routing_model()
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = content[0].get("text", str(content)) if content else ""

        # Parse response
        result = {
            "execution_plan": [],
            "focus_areas": [],
            "assumptions": [],
            "questions": [],
            "prior_knowledge": "",
        }

        current_section = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("EXECUTION_PLAN:"):
                current_section = "plan"
            elif line.startswith("FOCUS_AREAS:"):
                areas = line.replace("FOCUS_AREAS:", "").strip()
                result["focus_areas"] = [a.strip() for a in areas.split(",") if a.strip()]
                current_section = None
            elif line.startswith("KEY_ASSUMPTIONS:"):
                current_section = "assumptions"
            elif line.startswith("QUESTIONS_FOR_USER:"):
                current_section = "questions"
            elif line.startswith("PRIOR_KNOWLEDGE:"):
                result["prior_knowledge"] = line.replace("PRIOR_KNOWLEDGE:", "").strip()
                current_section = None
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "plan":
                    # Remove "Step N:" prefix if present
                    item = item.split(":", 1)[-1].strip() if ":" in item else item
                    result["execution_plan"].append(item)
                elif current_section == "assumptions":
                    result["assumptions"].append(item)
                elif current_section == "questions":
                    result["questions"].append(item)

        return result

    except Exception as e:
        # Intentional fallback: Gemini analysis is optional enrichment.
        # If it fails (network, API limits, parsing), we return empty defaults
        # and let the system proceed without pre-analysis. Claude can still
        # handle the task; it just won't have the upfront execution plan.
        logger.warning(
            "gemini_analysis_failed", error=str(e)[: CONTENT.QUERY_LOG], error_type=type(e).__name__
        )
        return {
            "execution_plan": [],
            "focus_areas": [],
            "assumptions": [],
            "questions": [],
            "prior_knowledge": "",
        }


async def get_relevant_learnings(query: str, category: str = "") -> str:
    """Pull relevant prior learnings and training knowledge from Chroma."""
    try:
        from ...collection_manager import get_collection_manager

        manager = get_collection_manager()

        all_context = []

        # 1. Search learnings (successful past solutions)
        try:
            learnings = manager.search_learnings(query, k=2)
            for item in learnings:
                all_context.append(
                    f"[PRIOR_SOLUTION|{item.get('framework', 'unknown')}] "
                    f"{item.get('solution', '')[: CONTENT.ERROR_PREVIEW]}"
                )
        except Exception as e:
            # Intentional fallback: Learnings search is optional context enrichment.
            # Failures (Chroma unavailable, collection missing) degrade gracefully.
            logger.debug(
                "learnings_search_failed",
                error=str(e)[: CONTENT.QUERY_LOG],
                error_type=type(e).__name__,
            )

        # 2. Search debugging knowledge (for debug tasks)
        debug_keywords = ["bug", "error", "fix", "debug"]
        if category in ("debug", "verification") or any(w in query.lower() for w in debug_keywords):
            try:
                docs = manager.search(query, collection_names=["debugging_knowledge"], k=2)
                for doc in docs:
                    all_context.append(
                        f"[DEBUG_PATTERN] {doc.page_content[: CONTENT.ERROR_PREVIEW]}"
                    )
            except Exception as e:
                # Intentional fallback: Debug knowledge is optional enrichment.
                # Collection may not exist or be empty - continue without it.
                logger.debug(
                    "debug_knowledge_search_failed",
                    error=str(e)[: CONTENT.QUERY_LOG],
                    error_type=type(e).__name__,
                )

        # 3. Search reasoning knowledge (for complex tasks)
        if category in ("architecture", "exploration", "refactor"):
            try:
                docs = manager.search(query, collection_names=["reasoning_knowledge"], k=2)
                for doc in docs:
                    all_context.append(
                        f"[REASONING_EXAMPLE] {doc.page_content[: CONTENT.ERROR_PREVIEW]}"
                    )
            except Exception as e:
                # Intentional fallback: Reasoning knowledge is optional enrichment.
                # Collection may not exist or be empty - continue without it.
                logger.debug(
                    "reasoning_knowledge_search_failed",
                    error=str(e)[: CONTENT.QUERY_LOG],
                    error_type=type(e).__name__,
                )

        # 4. Search instruction knowledge (for implementation tasks)
        if category in ("code_gen", "agent"):
            try:
                docs = manager.search(query, collection_names=["instruction_knowledge"], k=2)
                for doc in docs:
                    all_context.append(
                        f"[INSTRUCTION_EXAMPLE] {doc.page_content[: CONTENT.ERROR_PREVIEW]}"
                    )
            except Exception as e:
                # Intentional fallback: Instruction knowledge is optional enrichment.
                # Collection may not exist or be empty - continue without it.
                logger.debug(
                    "instruction_knowledge_search_failed",
                    error=str(e)[: CONTENT.QUERY_LOG],
                    error_type=type(e).__name__,
                )

        return "\n".join(all_context[:5]) if all_context else ""

    except Exception as e:
        # Intentional fallback: All learnings retrieval is optional enrichment.
        # If ChromaDB is unavailable or manager fails to initialize, we return
        # empty string and let the system proceed without prior knowledge context.
        logger.debug(
            "get_relevant_learnings_failed",
            error=str(e)[: CONTENT.QUERY_LOG],
            error_type=type(e).__name__,
        )
        return ""


async def enrich_evidence_from_chroma(
    query: str, category: str, framework_chain: list[str], task_type: str
) -> list[Any]:
    """
    Pull rich, actionable context from Chroma to give Claude more to work with.

    Gemini does this work upfront so Claude can execute immediately.
    Returns list of EvidenceExcerpt objects.
    """
    from ..schemas import EvidenceExcerpt, SourceType

    evidence = []

    try:
        from ...collection_manager import get_collection_manager

        manager = get_collection_manager()

        # 1. Get relevant code documentation/examples
        try:
            docs = manager.search(query, collection_names=["documentation"], k=2)
            for doc in docs:
                if doc.page_content and len(doc.page_content) > 50:
                    evidence.append(
                        EvidenceExcerpt(
                            source_type=SourceType.FILE,
                            ref=doc.metadata.get("source", "documentation")[:80],
                            content=doc.page_content[:1500],
                            relevance="Relevant code/documentation from codebase",
                        )
                    )
        except Exception as e:
            # Intentional fallback: Documentation search is optional enrichment.
            # Collection may not exist - proceed without documentation evidence.
            logger.debug(
                "enrich_evidence_docs_failed",
                error=str(e)[: CONTENT.QUERY_LOG],
                error_type=type(e).__name__,
            )

        # 2. Get prior successful solutions for similar tasks
        try:
            learnings = manager.search_learnings(query, k=2)
            for learn in learnings:
                solution = learn.get("solution", "")
                if solution and len(solution) > 30:
                    evidence.append(
                        EvidenceExcerpt(
                            source_type=SourceType.USER_TEXT,
                            ref=f"prior_solution_{learn.get('framework', 'unknown')}",
                            content=solution[:1200],
                            relevance=f"Prior successful solution using {learn.get('framework', 'unknown')}",
                        )
                    )
        except Exception as e:
            # Intentional fallback: Prior learnings are optional enrichment.
            # If search fails, proceed without prior solution evidence.
            logger.debug(
                "enrich_evidence_learnings_failed",
                error=str(e)[: CONTENT.QUERY_LOG],
                error_type=type(e).__name__,
            )

        # 3. Debug-specific: Get debugging patterns and known fixes
        if task_type in ("debug", "fix") or category == "debug":
            try:
                debug_docs = manager.search_debugging_knowledge(query, k=2)
                for doc in debug_docs:
                    if doc.page_content:
                        evidence.append(
                            EvidenceExcerpt(
                                source_type=SourceType.USER_TEXT,
                                ref="debug_pattern",
                                content=doc.page_content[:1200],
                                relevance="Known debugging pattern or fix approach",
                            )
                        )
            except Exception as e:
                # Intentional fallback: Debug patterns are optional enrichment.
                # Collection may not exist - proceed without debug evidence.
                logger.debug(
                    "enrich_evidence_debug_failed",
                    error=str(e)[: CONTENT.QUERY_LOG],
                    error_type=type(e).__name__,
                )

        # 4. Architecture/design: Get reasoning examples
        if category in ("architecture", "exploration", "refactor"):
            try:
                reasoning_docs = manager.search_reasoning_knowledge(query, k=1)
                for doc in reasoning_docs:
                    if doc.page_content:
                        evidence.append(
                            EvidenceExcerpt(
                                source_type=SourceType.USER_TEXT,
                                ref="reasoning_example",
                                content=doc.page_content[:1200],
                                relevance="Example reasoning approach for similar problem",
                            )
                        )
            except Exception as e:
                # Intentional fallback: Reasoning examples are optional enrichment.
                # Collection may not exist - proceed without reasoning evidence.
                logger.debug(
                    "enrich_evidence_reasoning_failed",
                    error=str(e)[: CONTENT.QUERY_LOG],
                    error_type=type(e).__name__,
                )

        # 5. Get framework-specific examples if using known framework
        if framework_chain:
            primary_fw = framework_chain[0]
            try:
                fw_docs = manager.search_frameworks(f"{primary_fw} example usage", k=1)
                for doc in fw_docs:
                    if doc.page_content and len(doc.page_content) > 50:
                        evidence.append(
                            EvidenceExcerpt(
                                source_type=SourceType.FILE,
                                ref=f"framework_{primary_fw}",
                                content=doc.page_content[:800],
                                relevance=f"Example of {primary_fw} framework usage",
                            )
                        )
            except Exception as e:
                # Intentional fallback: Framework examples are optional enrichment.
                # Collection may not exist - proceed without framework evidence.
                logger.debug(
                    "enrich_evidence_framework_failed",
                    error=str(e)[: CONTENT.QUERY_LOG],
                    error_type=type(e).__name__,
                )

    except Exception as e:
        # Intentional fallback: All evidence enrichment is optional.
        # If ChromaDB is unavailable or manager fails to initialize,
        # we return whatever evidence we collected so far (possibly empty).
        logger.debug(
            "enrich_evidence_failed", error=str(e)[: CONTENT.QUERY_LOG], error_type=type(e).__name__
        )

    return evidence[:4]  # Max 4 enriched evidence items


async def save_task_analysis(
    query: str, analysis: dict[str, Any], framework_chain: list[str], category: str
) -> None:
    """Save task analysis to Chroma for future reference."""
    try:
        from ...collection_manager import get_collection_manager

        manager = get_collection_manager()

        # Format analysis as text
        content = f"Task: {query}\n"
        content += f"Chain: {' -> '.join(framework_chain)}\n"
        content += f"Plan: {'; '.join(analysis.get('execution_plan', []))}\n"
        if analysis.get("prior_knowledge"):
            content += f"Insight: {analysis['prior_knowledge']}"

        metadata = {
            "type": "task_analysis",
            "category": category,
            "framework": framework_chain[0] if framework_chain else "unknown",
            "chain_length": len(framework_chain),
        }

        manager.add_documents([content], [metadata], "learnings")

    except Exception as e:
        # Intentional fallback: Saving task analysis is fire-and-forget.
        # If ChromaDB is unavailable, we silently continue - the analysis
        # was already used for the current request, persistence is optional.
        logger.debug(
            "save_task_analysis_failed",
            error=str(e)[: CONTENT.QUERY_LOG],
            error_type=type(e).__name__,
        )
