# Incomplete Code & Docs Audit (Chain-of-Note)

**Date**: 2026-01-04  
**Framework used**: `chain_of_note` (selected via omni MCP `reason`)  
**Scope**:
- Find faults / incomplete code paths (stubs, TODO/FIXME, placeholder logic)
- Identify missing functions/import mismatches
- Identify documentation gaps/mismatches

---

## NOTE 1 — Observations (evidence)

### 1) RESOLVED: All 62 frameworks now properly documented and exposed

**Status: FIXED**
- `README.md`: Updated to correctly document 62 thinking frameworks
- Docker logs and server implementation now correctly expose all 62 frameworks
- `server/main.py`: `FRAMEWORKS = {...}` contains **62 entries**
- `list_tools()` creates `think_*` tools for all 62 frameworks
- `list_frameworks` tool description: `List all 62 thinking frameworks by category`.

---

### 2) RESOLVED: Router/Graph and MCP now expose all 62 frameworks

**Evidence (graph registry)**
- `app/graph.py` imports and registers all 62 framework nodes, including:
  - `least_to_most`, `comparative_arch`, `plan_and_solve`
  - `rubber_duck`, `tdd_prompting`, `chain_of_thought`
  - plus many additional “code” frameworks (`alphacodium`, `codechain`, etc.)

**Evidence (router metadata)**
- `app/core/router.py` includes those frameworks in:
  - `FRAMEWORKS` descriptions
  - `VIBE_DICTIONARY`
  - `get_framework_info()` framework metadata map

**Net effect**
- The router, graph, and MCP server all consistently expose all 62 frameworks via `think_*` tools.

---

### 3) Some “incomplete code” grep hits are prompt examples (false positives)

**Evidence**
- `app/nodes/strategy/least_to_most.py` contains `pass`, but only inside the prompt template example code block.
- `app/nodes/context/state_machine.py` contains `pass`, but only inside the prompt template example code block.

**Conclusion**
- These are not runtime stubs.

---

### 4) Silent exception handling can hide real wiring failures

**Evidence**
- `app/nodes/context/chain_of_note.py`:
  - Tool calls (`retrieve_context`, `search_documentation_only`) are wrapped in `try/except`.
  - `search_documentation_only` failure branch uses `pass` (swallows error).

- `app/nodes/code/critic.py`:
  - Several `except Exception: pass` blocks around tool lookups (`search_function_implementation`, `search_class_implementation`, etc.).

**Risk**
- Missing tool registrations, RAG initialization failures, or tool API changes can be masked.

---

### 5) “Pass-through” messaging vs internal LLM wrappers

**Evidence**
- `app/core/config.py` contains a stub `ModelConfig` raising `NotImplementedError` for LLM calls.
- `app/nodes/common.py` contains real LangChain-based LLM wrappers:
  - `call_deep_reasoner`
  - `call_fast_synthesizer`
  - These raise if no provider is configured (e.g., missing keys).

**Risk**
- Architectural ambiguity: docs suggest pass-through/no keys, but framework nodes can still try to invoke LLM calls internally.

---

## NOTE 2 — Connections (how pieces relate)

- **RESOLVED**: All components are now consistent:
  - **Docs** correctly advertise **62 frameworks**
  - **Router/Graph** support all **62 frameworks**
  - **MCP tool surface** exposes all **62 frameworks**

Users can access all documented frameworks including `think_rubber_duck`, `think_tdd_prompting`, etc.

---

## NOTE 3 — Inferences (what’s likely incomplete)

- The repository is in a **half-migrated state**:
  - Many framework nodes exist and appear wired into LangGraph
  - Router metadata covers a wider set
  - MCP server exposure is restricted to a smaller set

- Silent exception handling likely exists to keep frameworks “best-effort”, but it also hides real runtime problems.

---

## NOTE 4 — Synthesis (prioritized punch list)

### P0 — RESOLVED: All 62 frameworks now consistently exposed and documented

**Solution implemented (Option A):**
- `server/main.py FRAMEWORKS` includes all 62 frameworks present in the graph/router
- `list_tools()` exposes `think_*` for all 62 frameworks
- `list_frameworks` output correctly lists all 62 frameworks
- All documentation updated to reflect 62 frameworks

---

### P1 — Make silent failures observable

- Keep best-effort behavior, but log tool failures:
  - Replace `except Exception: pass` with
    - `logger.warning(...)` (or debug)
    - and continue gracefully

---

### P2 — Clarify pass-through vs internal LLM usage

- Either:
  - fully commit to pass-through (avoid internal LLM wrappers), or
  - document that internal LLM wrappers can run when keys are provided.

---

## Current runtime verification snapshot

From Docker container logs:
- Auto-ingestion ran and populated embeddings (example run showed `added=67 total=67`).
- Server reports: `Frameworks: 62` and tools count `77` (62 think_* + 1 reason + 14 utility).

---

## Status

**RESOLVED**: All 62 frameworks are now exposed as MCP `think_*` tools and documentation has been updated accordingly.
