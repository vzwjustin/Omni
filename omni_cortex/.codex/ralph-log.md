# Ralph Loop Log

- Created: 2026-01-05T00:37:12Z
- State file: .codex/ralph-loop.md
- Max iterations: 6
- Completion promise: FIXED

Iteration 1:
- Signal: search_vectorstore returns [] on exception, causing "No results" even on vectorstore failure.
- Hypothesis: raising a dedicated VectorstoreSearchError and handling it in tools will surface failures without changing no-results behavior.
- Change: add VectorstoreSearchError, raise it in search_vectorstore, catch in search_documentation (app/langchain_integration and server/main), handle in scripts/test_mcp_search.py.
- Verification: python3 -m py_compile app/langchain_integration.py server/main.py scripts/test_mcp_search.py (imports unavailable for runtime test).
- Next: verify runtime behavior when embeddings are configured.

Iteration 2:
- Signal: CollectionManager.search swallows collection load/search errors, so search_vectorstore may still return [] without raising.
- Hypothesis: tracking collection failures and raising when no collections can be searched will distinguish failures from true no-results.
- Change: add raise_on_error to CollectionManager.search, raise when all collections unavailable, and pass raise_on_error=True from search_vectorstore.
- Verification: python3 -m py_compile app/collection_manager.py app/langchain_integration.py server/main.py scripts/test_mcp_search.py; docker run search_documentation.ainvoke with API keys present (returns results) and with keys empty (returns "Search failed: ...").
- Next: none.

Iteration 3:
- Signal: Direct MCP call returned "Logger._log() got an unexpected keyword argument 'error'" from server/main logging.
- Hypothesis: stdlib logger in server/main cannot accept structlog-style kwargs.
- Change: replace logger.error(...) call with stdlib formatting.
- Verification: python3 -m py_compile server/main.py; MCP stdio client call now returns results without logging error.
- Next: rebuild local image and restart server when convenient.
