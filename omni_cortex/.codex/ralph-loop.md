---
iteration: 4
max_iterations: 6
completion_promise: "FIXED"
started_at: "2026-01-05T00:37:12Z"
---

Debug task: Distinguish vectorstore errors from true "no results" in search_documentation.

Context:
- Failing behavior: search_vectorstore() returns [] on exception, so callers display "No results found" even when the vectorstore failed.
- Files: app/langchain_integration.py, server/main.py, scripts/test_mcp_search.py
- Constraints: Keep normal success/no-results behavior unchanged; provide explicit error messaging on failures.

Acceptance criteria:
- search_vectorstore raises a specific error on failure with a clear message.
- search_documentation tools (app/langchain_integration and server/main) catch the error and return a "Search failed" response.
- scripts/test_mcp_search.py handles the error path explicitly.
- No behavioral change when there are legitimate zero results.

Loop rules:
- Keep this prompt unchanged each iteration.
- Run the smallest check that validates the hypothesis.
- Record an iteration log.
- Output <promise>FIXED</promise> only when all criteria are true.
- Stop after 6 iterations and summarize if still failing.
