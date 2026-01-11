"""
Framework Orchestrators

All 62 frameworks implemented as real multi-turn orchestrations
using MCP client sampling (no external API calls).
"""

# Import all orchestrators
from .agent_frameworks import (
    lats,
    mrkl,
    rewoo,
    swe_agent,
    toolformer,
)
from .code_frameworks import (
    alphacodium,
    chain_of_code,
    chain_of_verification,
    codechain,
    critic,
    docprompting,
    evol_instruct,
    llmloop,
    pal,
    parsel,
    procoder,
    program_of_thoughts,
    recode,
    reverse_cot,
    scratchpads,
    self_debugging,
    tdd_prompting,
)
from .context_frameworks import (
    analogical,
    chain_of_note,
    chain_of_thought,
    red_team,
    state_machine,
    step_back,
)
from .fast_frameworks import (
    skeleton_of_thought,
    system1,
)

# Framework Factory (declarative framework definitions)
from .framework_factory import (
    CHAIN_OF_CODE,
    CHAIN_OF_VERIFICATION,
    CODECHAIN,
    CRITIC,
    DOCPROMPTING,
    EVOL_INSTRUCT,
    FRAMEWORK_CONFIGS,
    PAL,
    PARSEL,
    PROCODER,
    # Individual configs
    PROGRAM_OF_THOUGHTS,
    REVERSE_COT,
    SELF_DEBUGGING,
    TDD_PROMPTING,
    FrameworkConfig,
    FrameworkStep,
    execute_framework,
    get_available_frameworks,
    get_framework_description,
    run_framework,
)
from .iterative_frameworks import (
    active_inference,
    adaptive_injection,
    multi_agent_debate,
    re2,
    react,
    reflexion,
    rubber_duck,
    self_refine,
)
from .rag_frameworks import (
    graphrag,
    hyde,
    rag_fusion,
    raptor,
    self_rag,
)
from .search_frameworks import (
    everything_of_thought,
    graph_of_thoughts,
    mcts_rstar,
    tree_of_thoughts,
)
from .strategy_frameworks import (
    buffer_of_thoughts,
    coala,
    comparative_arch,
    least_to_most,
    plan_and_solve,
    reason_flux,
    self_discover,
)
from .verification_frameworks import (
    metaqa,
    ragas,
    rar,
    rarr,
    self_ask,
    self_consistency,
    selfcheckgpt,
    verify_and_edit,
)

# Framework registry mapping framework names to orchestrator functions
FRAMEWORK_ORCHESTRATORS = {
    # Strategy (7)
    "reason_flux": reason_flux,
    "self_discover": self_discover,
    "buffer_of_thoughts": buffer_of_thoughts,
    "coala": coala,
    "least_to_most": least_to_most,
    "comparative_arch": comparative_arch,
    "plan_and_solve": plan_and_solve,
    # Search (4)
    "tree_of_thoughts": tree_of_thoughts,
    "graph_of_thoughts": graph_of_thoughts,
    "mcts_rstar": mcts_rstar,
    "everything_of_thought": everything_of_thought,
    # Iterative (8)
    "active_inference": active_inference,
    "multi_agent_debate": multi_agent_debate,
    "adaptive_injection": adaptive_injection,
    "re2": re2,
    "rubber_duck": rubber_duck,
    "react": react,
    "reflexion": reflexion,
    "self_refine": self_refine,
    # Code (17)
    "program_of_thoughts": program_of_thoughts,
    "chain_of_verification": chain_of_verification,
    "critic": critic,
    "chain_of_code": chain_of_code,
    "self_debugging": self_debugging,
    "tdd_prompting": tdd_prompting,
    "reverse_cot": reverse_cot,
    "alphacodium": alphacodium,
    "codechain": codechain,
    "evol_instruct": evol_instruct,
    "llmloop": llmloop,
    "procoder": procoder,
    "recode": recode,
    "pal": pal,
    "scratchpads": scratchpads,
    "parsel": parsel,
    "docprompting": docprompting,
    # Context (6)
    "chain_of_note": chain_of_note,
    "step_back": step_back,
    "analogical": analogical,
    "red_team": red_team,
    "state_machine": state_machine,
    "chain_of_thought": chain_of_thought,
    # Fast (2)
    "skeleton_of_thought": skeleton_of_thought,
    "system1": system1,
    # Verification (8)
    "self_consistency": self_consistency,
    "self_ask": self_ask,
    "rar": rar,
    "verify_and_edit": verify_and_edit,
    "rarr": rarr,
    "selfcheckgpt": selfcheckgpt,
    "metaqa": metaqa,
    "ragas": ragas,
    # Agent (5)
    "rewoo": rewoo,
    "lats": lats,
    "mrkl": mrkl,
    "swe_agent": swe_agent,
    "toolformer": toolformer,
    # RAG (5)
    "self_rag": self_rag,
    "hyde": hyde,
    "rag_fusion": rag_fusion,
    "raptor": raptor,
    "graphrag": graphrag,
}


__all__ = [
    "FRAMEWORK_ORCHESTRATORS",
    # Framework Factory
    "FrameworkStep",
    "FrameworkConfig",
    "execute_framework",
    "run_framework",
    "get_available_frameworks",
    "get_framework_description",
    "FRAMEWORK_CONFIGS",
    "PROGRAM_OF_THOUGHTS",
    "CHAIN_OF_VERIFICATION",
    "CRITIC",
    "CHAIN_OF_CODE",
    "SELF_DEBUGGING",
    "TDD_PROMPTING",
    "REVERSE_COT",
    "CODECHAIN",
    "EVOL_INSTRUCT",
    "PROCODER",
    "PAL",
    "PARSEL",
    "DOCPROMPTING",
    # Strategy
    "reason_flux",
    "self_discover",
    "buffer_of_thoughts",
    "coala",
    "least_to_most",
    "comparative_arch",
    "plan_and_solve",
    # Search
    "tree_of_thoughts",
    "graph_of_thoughts",
    "mcts_rstar",
    "everything_of_thought",
    # Iterative
    "active_inference",
    "multi_agent_debate",
    "adaptive_injection",
    "re2",
    "rubber_duck",
    "react",
    "reflexion",
    "self_refine",
    # Code
    "program_of_thoughts",
    "chain_of_verification",
    "critic",
    "chain_of_code",
    "self_debugging",
    "tdd_prompting",
    "reverse_cot",
    "alphacodium",
    "codechain",
    "evol_instruct",
    "llmloop",
    "procoder",
    "recode",
    "pal",
    "scratchpads",
    "parsel",
    "docprompting",
    # Context
    "chain_of_note",
    "step_back",
    "analogical",
    "red_team",
    "state_machine",
    "chain_of_thought",
    # Fast
    "skeleton_of_thought",
    "system1",
    # Verification
    "self_consistency",
    "self_ask",
    "rar",
    "verify_and_edit",
    "rarr",
    "selfcheckgpt",
    "metaqa",
    "ragas",
    # Agent
    "rewoo",
    "lats",
    "mrkl",
    "swe_agent",
    "toolformer",
    # RAG
    "self_rag",
    "hyde",
    "rag_fusion",
    "raptor",
    "graphrag",
]
