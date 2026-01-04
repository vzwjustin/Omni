# Incomplete Code & Docs Audit (Chain-of-Note)

**Date**: 2026-01-04  
**Framework used**: `chain_of_note` (selected via omni MCP `reason`)  
**Scope**:
- Find faults / incomplete code paths (stubs, TODO/FIXME, placeholder logic)
- Identify missing functions/import mismatches
- Identify documentation gaps/mismatches

---

## NOTE 1 — Observations (evidence)

### 1) Docs claim 40 frameworks, runtime exposes 20

**Evidence (docs)**
- `README.md`:
  - Claims: `40 thinking frameworks for your IDE` (line 3)
  - Lists `## 40 Frameworks` table (line 49+) with many frameworks not exposed as MCP tools.

**Evidence (runtime)**
- Docker logs (container startup):
  - `Frameworks: 20 thinking frameworks`
  - `Tools: 20 think_* + 1 reason + 14 utility = 35 total`

**Evidence (server implementation)**
- `server/main.py`:
  - `FRAMEWORKS = {...}` contains **20 entries**.
  - `list_tools()` creates `think_*` tools by iterating `FRAMEWORKS.items()` → only **20 `think_*` tools** are exposed.
  - `list_frameworks` tool description: `List all 20 thinking frameworks by category`.

---

### 2) Router/Graph support more frameworks than MCP exposes

**Evidence (graph registry)**
- `app/graph.py` imports and registers a broader set of framework nodes (effectively the “40 frameworks” universe), including:
  - `least_to_most`, `comparative_arch`, `plan_and_solve`
  - `rubber_duck`, `tdd_prompting`, `chain_of_thought`
  - plus many additional “code” frameworks (`alphacodium`, `codechain`, etc.)

**Evidence (router metadata)**
- `app/core/router.py` includes those frameworks in:
  - `FRAMEWORKS` descriptions
  - `VIBE_DICTIONARY`
  - `get_framework_info()` framework metadata map

**Net effect**
- The router and graph can refer to / support frameworks that the MCP server never exposes via `think_*` tools.

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

- There is a **triangle mismatch**:
  - **Docs** advertise **40 frameworks**
  - **Router/Graph** appear to support **~40 frameworks**
  - **MCP tool surface** currently exposes **20 frameworks**

This creates a “missing functionality” experience: users read docs and expect tools like `think_rubber_duck` / `think_tdd_prompting`, but those tools are not actually present.

---

## NOTE 3 — Inferences (what’s likely incomplete)

- The repository is in a **half-migrated state**:
  - Many framework nodes exist and appear wired into LangGraph
  - Router metadata covers a wider set
  - MCP server exposure is restricted to a smaller set

- Silent exception handling likely exists to keep frameworks “best-effort”, but it also hides real runtime problems.

---

## NOTE 4 — Synthesis (prioritized punch list)

### P0 — Decide: expose 40 frameworks or document 20

Pick one clear, consistent stance:

- **Option A (make README true):**
  - Expand `server/main.py FRAMEWORKS` to include all frameworks present in the graph/router.
  - Ensure `list_tools()` exposes `think_*` for all frameworks.
  - Verify `list_frameworks` output aligns.

- **Option B (make server truth the docs):**
  - Update `README.md` to document **20 frameworks** only.
  - Optionally document the rest as “internal/experimental”, or remove them.

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
- Server reports: `Frameworks: 20` and tools count `35`.

---

## Recommended next step

Confirm which direction you want:
- **A)** expose all frameworks as MCP `think_*` tools, or
- **B)** update docs + expectations to match the 20-tool surface.
