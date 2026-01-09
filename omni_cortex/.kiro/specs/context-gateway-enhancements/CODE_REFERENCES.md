# Code References for Context Gateway Enhancement

## File Locations and Key Code Sections

### 1. Main Context Gateway (app/core/context_gateway.py)

**ContextGateway Class**
- Lines 1-50: Module docstring and imports
- Lines 100-200: StructuredContext dataclass definition
- Lines 250-350: ContextGateway.__init__() and dependency injection
- Lines 360-420: _fallback_analyze() - Pattern-based fallback
- Lines 430-550: prepare_context() - Main orchestration method
- Lines 560-580: quick_analyze() - Fast analysis without discovery
- Lines 600-650: Global singleton pattern with thread-safe initialization

**Key Methods**:
- `prepare_context()`: Main entry point, orchestrates all components
- `_fallback_analyze()`: Pattern-based task type detection
- `get_context_gateway()`: Thread-safe singleton accessor

**Enhancement Points**:
- Add `_cache` dict for query caching
- Add `_streaming_queue` for progress events
- Modify `prepare_context()` to support streaming
- Add multi-repo detection logic

---

### 2. Query Analyzer (app/core/context/query_analyzer.py)

**QueryAnalyzer Class**
- Lines 1-50: Module docstring and imports
- Lines 100-150: _sanitize_prompt_input() - Prompt injection prevention
- Lines 160-200: _get_client() - Gemini client initialization
- Lines 210-350: analyze() - Main analysis method with thinking mode
- Lines 360-420: Error handling with specific detection

**Key Features**:
- Thinking mode support (HIGH level for complex queries)
- Prompt injection prevention via sanitization
- Graceful fallback on API errors
- JSON response parsing

**Enhancement Points**:
- Add thinking mode complexity detection
- Add reasoning quality metrics logging
- Implement token budget adaptation

---

### 3. File Discoverer (app/core/context/file_discoverer.py)

**FileDiscoverer Class**
- Lines 1-50: Module docstring and imports
- Lines 100-150: _get_client() - Gemini client initialization
- Lines 160-250: discover() - Main discovery method
- Lines 260-350: _list_workspace_files() - Async file listing
- Lines 360-420: _sync_list_files() - Synchronous file listing with os.walk

**Key Features**:
- Async I/O with thread pool offloading
- Gemini-based relevance scoring
- File filtering by extension
- Exclusion pattern support

**Enhancement Points**:
- Add multi-repo detection (multiple .git directories)
- Add repository labeling in results
- Add cross-repo dependency following
- Implement streaming file discovery

---

### 4. Documentation Searcher (app/core/context/doc_searcher.py)

**DocumentationSearcher Class**
- Lines 1-50: Module docstring and imports
- Lines 100-150: _get_search_model() - Gemini with Google Search grounding
- Lines 160-280: search_web() - Web search with grounding metadata
- Lines 290-400: search_knowledge_base() - ChromaDB collection search
- Lines 410-500: Collection-specific search methods

**Key Features**:
- Google Search grounding for web results
- Multiple ChromaDB collection support
- Graceful degradation per collection
- Source attribution from grounding metadata

**Enhancement Points**:
- Extract and preserve source URLs from grounding
- Add publication date tracking
- Implement result ranking by authority
- Add source link formatting

---

### 5. Code Searcher (app/core/context/code_searcher.py)

**CodeSearcher Class**
- Lines 1-50: Module docstring and imports
- Lines 100-150: _get_client() - Gemini client for query extraction
- Lines 160-200: _detect_search_command() - ripgrep vs grep detection
- Lines 210-350: search() - Main search orchestration
- Lines 360-450: _extract_search_queries() - Gemini-based query extraction
- Lines 460-550: _run_grep_searches() - ripgrep/grep execution
- Lines 560-620: _search_git_log() - Git log search

**Key Features**:
- Automatic search command detection
- Gemini-based search term extraction
- Parallel grep/ripgrep execution
- Git log integration

**Enhancement Points**:
- Add search result summarization
- Implement pattern extraction
- Add code relationship detection

---

### 6. MCP Server Integration (server/main.py)

**prepare_context Tool**
- Lines 400-450: Tool definition in list_tools()
- Lines 500-550: Tool handler routing in call_tool()
- Lines 560-600: handle_prepare_context() call

**reason Tool**
- Lines 350-380: Tool definition
- Lines 480-500: Tool handler routing

**Key Features**:
- LEAN_MODE support (8 core tools)
- Tool input validation
- Rate limiting integration
- Correlation ID tracking

**Enhancement Points**:
- Add streaming support to tool definition
- Add progress event handling
- Implement cancellation support

---

### 7. Reason Handler (server/handlers/reason_handler.py)

**handle_reason() Function**
- Lines 1-50: Module docstring and imports
- Lines 60-100: Input validation
- Lines 110-150: Auto-context feature (prepare_context call)
- Lines 160-250: Structured brief generation
- Lines 260-300: Fallback mode handling

**Key Features**:
- Auto-context preparation if no context provided
- Graceful fallback to template mode
- Audit logging

**Enhancement Points**:
- Add context quality tracking
- Implement context element relevance tracking
- Add cache hit logging

---

### 8. State Management (app/state.py)

**GraphState TypedDict**
- Lines 1-50: Module docstring
- Lines 60-150: InputState, ReasoningState, OutputState definitions
- Lines 160-250: GraphState flattened definition
- Lines 260-350: MemoryStore dataclass

**Key Features**:
- Typed state management
- Episodic memory support
- Thought template storage

**Enhancement Points**:
- Add context preparation state fields
- Add streaming progress fields
- Add cache metadata fields

---

### 9. Settings (app/core/settings.py)

**OmniCortexSettings Class**
- Lines 1-50: Module docstring
- Lines 60-150: API key configuration
- Lines 160-200: Model configuration
- Lines 210-250: Feature flags
- Lines 260-300: Validators

**Key Configuration**:
- `google_api_key`: Gemini API key
- `routing_model`: Model selection
- `lean_mode`: Tool exposure mode
- `chroma_persist_dir`: Vector DB location

**Enhancement Points**:
- Add cache TTL settings
- Add streaming configuration
- Add metrics configuration

---

### 10. Constants (app/core/constants.py)

**Content Limits**
- Lines 20-50: SNIPPET_* constants
- Lines 60-80: COMMAND_OUTPUT, OBSERVATION_LIMIT

**Search Limits**
- Lines 100-130: K_* constants
- Lines 140-160: GREP_MAX_COUNT, SEARCH_QUERIES_MAX

**Workspace Limits**
- Lines 180-220: MAX_FILES_*, EXCLUDE_DIRS, CODE_EXTENSIONS

**Enhancement Points**:
- Add cache size limits
- Add streaming buffer sizes
- Add timeout constants

---

### 11. Error Handling (app/core/errors.py)

**Error Hierarchy**
- Lines 1-50: OmniCortexError base class
- Lines 60-100: Routing errors
- Lines 110-150: Execution errors
- Lines 160-200: Memory errors
- Lines 210-250: RAG errors
- Lines 260-300: LLM errors

**Key Errors**:
- `ProviderNotConfiguredError`: Missing API key
- `LLMError`: API call failed
- `ContextRetrievalError`: File discovery failed
- `RAGError`: Vector store failed

**Enhancement Points**:
- Add circuit breaker error
- Add cache error types
- Add streaming error types

---

### 12. Metrics (app/core/metrics.py)

**Framework Metrics**
- Lines 50-100: FRAMEWORK_EXECUTIONS, FRAMEWORK_DURATION, FRAMEWORK_TOKENS
- Lines 110-150: FRAMEWORK_CONFIDENCE

**Router Metrics**
- Lines 160-200: ROUTER_DECISIONS, ROUTER_DURATION, ROUTER_CHAIN_LENGTH

**MCP Metrics**
- Lines 210-250: MCP_REQUESTS, MCP_REQUEST_DURATION

**Enhancement Points**:
- Add context gateway metrics
- Add cache metrics
- Add component performance metrics

---

### 13. Rate Limiter (app/core/rate_limiter.py)

**RateLimiter Class**
- Lines 1-50: Module docstring
- Lines 60-150: TokenBucket implementation
- Lines 160-250: RateLimiter class
- Lines 260-350: check_rate_limit() method
- Lines 360-400: validate_input_size() method

**Key Features**:
- Token bucket algorithm
- Per-category rate limiting
- Global rate limiting
- Input size validation

**Enhancement Points**:
- Add context gateway rate limits
- Add cache operation limits
- Add streaming operation limits

---

### 14. Collection Manager (app/collection_manager.py)

**CollectionManager Class**
- Lines 1-50: Module docstring
- Lines 60-150: COLLECTIONS definition
- Lines 160-250: __init__() and get_embedding_function()
- Lines 260-350: get_collection() with thread-safe caching
- Lines 360-450: search() method with error handling
- Lines 460-550: Collection-specific search methods

**Key Features**:
- Multi-collection management
- Thread-safe singleton pattern
- Graceful degradation per collection
- Deduplication of results

**Enhancement Points**:
- Add cache collection
- Add context quality collection
- Add metrics collection

---

### 15. Graph Orchestration (app/graph.py)

**Graph Creation**
- Lines 1-50: Module docstring
- Lines 60-150: route_node() - Framework selection
- Lines 160-250: execute_framework_node() - Framework execution
- Lines 260-350: should_continue() - Conditional routing
- Lines 360-400: retry_with_backoff() - Retry logic
- Lines 410-500: create_reasoning_graph() - Graph assembly

**Key Features**:
- LangGraph workflow
- Retry logic with exponential backoff
- Pipeline execution support
- Checkpointing support

**Enhancement Points**:
- Add context preparation node
- Add streaming support
- Add cache management node

---

## Integration Points for Enhancements

### For Caching (Requirement 1)
**Files to Modify**:
1. `app/core/context_gateway.py` - Add cache dict and TTL management
2. `app/core/constants.py` - Add cache TTL constants
3. `app/core/metrics.py` - Add cache metrics
4. `app/collection_manager.py` - Add cache collection

**Key Methods**:
- `ContextGateway.prepare_context()` - Check cache before discovery
- `ContextGateway._generate_cache_key()` - NEW: Query similarity hashing
- `ContextGateway._invalidate_cache()` - NEW: File change detection

### For Streaming (Requirement 3)
**Files to Modify**:
1. `app/core/context_gateway.py` - Add streaming support
2. `server/main.py` - Add streaming tool definition
3. `server/handlers/reason_handler.py` - Handle streaming responses

**Key Methods**:
- `ContextGateway.prepare_context_streaming()` - NEW: Streaming version
- `ContextGateway._emit_progress()` - NEW: Progress event emission

### For Multi-Repo (Requirement 4)
**Files to Modify**:
1. `app/core/context/file_discoverer.py` - Multi-repo detection
2. `app/core/context_gateway.py` - Multi-repo orchestration

**Key Methods**:
- `FileDiscoverer._detect_repositories()` - NEW: Find .git directories
- `FileDiscoverer._analyze_repository()` - NEW: Per-repo analysis

### For Metrics (Requirement 6)
**Files to Modify**:
1. `app/core/metrics.py` - Add context gateway metrics
2. `app/core/context_gateway.py` - Record metrics
3. `server/handlers/reason_handler.py` - Track context usage

**Key Metrics**:
- `context_gateway_api_calls_total` - Gemini calls per component
- `context_gateway_tokens_used` - Tokens per component
- `context_gateway_component_duration` - Timing per component

