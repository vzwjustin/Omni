# Context Gateway Analysis - Quick Reference

## System Overview

**Purpose**: Gemini-powered preprocessing layer that prepares rich context for Claude

**Architecture**: 4 parallel components orchestrated by ContextGateway
- QueryAnalyzer: Intent analysis
- FileDiscoverer: File ranking
- DocumentationSearcher: Web + knowledge base search
- CodeSearcher: Codebase search

**Output**: StructuredContext with task analysis, files, docs, code search, framework recommendation

---

## Key Files and Locations

### Core Gateway
- `app/core/context_gateway.py` - Main orchestrator
- `app/core/context/query_analyzer.py` - Query analysis
- `app/core/context/file_discoverer.py` - File discovery
- `app/core/context/doc_searcher.py` - Documentation search
- `app/core/context/code_searcher.py` - Code search

### MCP Integration
- `server/main.py` - MCP server (prepare_context tool)
- `server/handlers/reason_handler.py` - reason tool with auto-context

### State & Configuration
- `app/state.py` - GraphState definition
- `app/core/settings.py` - Configuration
- `app/core/constants.py` - Magic numbers
- `app/core/errors.py` - Error taxonomy

### Monitoring & Resilience
- `app/core/metrics.py` - Prometheus metrics
- `app/core/rate_limiter.py` - Rate limiting
- `app/collection_manager.py` - ChromaDB management

---

## Gemini Integration Details

### Models Used
- **Primary**: gemini-3-flash-preview
- **Thinking Mode**: Available on Gemini 3 models
- **Temperature**: 0.2-0.3 (deterministic)

### API Packages
- `google-genai>=0.3.0` (NEW - thinking mode support)
- `google-generativeai>=0.8.0` (DEPRECATED fallback)

### Thinking Mode Usage
```python
# QueryAnalyzer uses HIGH thinking mode for complex queries
config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
    temperature=0.3,
    response_mime_type="application/json"
)
```

### Error Handling
- Detects billing issues, auth errors, network failures
- Falls back to pattern-based analysis if Gemini unavailable
- Continues with available components on partial failures

---

## MCP Tool Integration

### prepare_context Tool
**Input**:
- query (required): Task description
- workspace_path (optional): Project directory
- code_context (optional): Code snippets
- file_list (optional): Pre-specified files
- search_docs (optional): Search web (default: true)
- output_format (optional): "prompt" or "json"

**Output**: StructuredContext formatted as markdown or JSON

### reason Tool Auto-Context
- If no context provided, automatically calls prepare_context
- Bridges gap between context preparation and reasoning
- Transparent to user

---

## Data Flow

### prepare_context() Flow
```
1. Parallel execution (asyncio.gather):
   - FileDiscoverer.discover() → List[FileContext]
   - DocumentationSearcher.search_web() → List[DocumentationContext]
   - CodeSearcher.search() → List[CodeSearchContext]

2. Sequential analysis (uses discovery results):
   - QueryAnalyzer.analyze() → Dict with task_type, framework, steps

3. Assembly:
   - Combine all results into StructuredContext
   - Format as Claude prompt or JSON
```

### Error Handling Flow
```
Component fails
  ↓
Caught as Exception in asyncio.gather()
  ↓
Logged as warning
  ↓
Converted to empty result ([], None, etc.)
  ↓
Continue with other components
  ↓
Return partial StructuredContext
```

---

## Configuration Points

### Environment Variables
```
GOOGLE_API_KEY=...              # Required for Gemini
OPENAI_API_KEY=...              # For embeddings
LEAN_MODE=true                  # 8 tools vs 81
ROUTING_MODEL=gemini-3-flash    # Model selection
CHROMA_PERSIST_DIR=/app/data    # Vector DB location
```

### Settings (app/core/settings.py)
- API keys for all providers
- Model selection
- Rate limits per tool category
- Feature flags
- Paths and timeouts

### Constants (app/core/constants.py)
- Content limits (SNIPPET_SHORT, SNIPPET_MAX, etc.)
- Search limits (K_STANDARD, GREP_MAX_COUNT, etc.)
- Workspace limits (MAX_FILES_SCAN, EXCLUDE_DIRS, etc.)
- Resource limits (timeouts, batch sizes)

---

## Performance Characteristics

### Parallel Execution
- File discovery, doc search, code search run concurrently
- Total time ≈ max(component_times), not sum
- Typical: 2-5 seconds for full context preparation

### Token Budget
- Fixed 50,000 token limit
- Files limited to top 10
- Docs limited to top 5
- Code search limited to top 3

### Caching (Current)
- Collection manager caches Chroma instances
- Embedding function cached after first init
- No query-level caching yet

---

## Resilience Patterns

### Pattern 1: Component Isolation
- Each component wrapped in try/except
- Failures don't block other components
- Partial results returned

### Pattern 2: Fallback Analysis
- Pattern-based task type detection
- Regex matching on keywords
- Works without Gemini API

### Pattern 3: Graceful Degradation
- Web search fails → ChromaDB-only results
- File discovery fails → Simple file listing
- Code search fails → Continue without code context

### Pattern 4: Retry Logic
- Exponential backoff: 100ms, 200ms, 400ms
- Max 3 retries
- Used in graph.py for routing failures

---

## Enhancement Opportunities (From Requirements)

### 1. Intelligent Caching
- Query similarity-based cache keys
- Separate TTLs: query (1h), files (30m), docs (24h)
- Workspace fingerprint for invalidation

### 2. Streaming Context
- Progress events for each component
- Real-time file discovery streaming
- Cancellation support

### 3. Multi-Repo Support
- Detect multiple .git directories
- Parallel Gemini analysis per repo
- Repository labels in results

### 4. Enhanced Documentation
- Extract source URLs from Gemini grounding
- Preserve publication dates
- Merge web + ChromaDB results

### 5. Quality Metrics
- Track Gemini API usage per component
- Monitor context element relevance
- Cache effectiveness metrics

### 6. Token Budget Management
- Dynamic allocation based on complexity
- Prioritize summaries over full content
- Gemini-based result ranking

### 7. Advanced Fallbacks
- Circuit breaker for Gemini API
- Exponential backoff with jitter
- Detailed service status indicators

---

## Testing Approach

### Validation Patterns
- Input size validation (rate_limiter.py)
- Schema validation (handlers/validation.py)
- Error type detection (errors.py)

### Available Test Scripts
- `python scripts/verify_learning_offline.py`
- `python scripts/test_mcp_search.py`
- `python -m scripts.debug_search`
- `python scripts/validate_frameworks.py`

### Testing Context Gateway
1. Mock Gemini responses
2. Test fallback analysis
3. Verify parallel execution
4. Check error handling
5. Validate output format

---

## Deployment

### Docker
```bash
docker-compose build
docker-compose up -d
docker-compose logs -f
```

### Local
```bash
pip install -r requirements.txt
python -m server.main
```

### Configuration
```bash
cp .env.example .env
# Edit .env with API keys
```

---

## Key Insights for Enhancement

### Strengths to Preserve
1. Composition pattern (clean separation)
2. Graceful degradation (works without Gemini)
3. Parallel execution (efficient)
4. Thread safety (proper locking)
5. Error isolation (partial failures OK)

### Integration Points
1. MCP server (prepare_context tool)
2. Framework selection (HyperRouter)
3. Memory system (LangChain)
4. RAG system (ChromaDB)

### Where to Add Features
- **Caching**: ContextGateway.__init__() and prepare_context()
- **Streaming**: prepare_context() with asyncio.Queue
- **Multi-repo**: FileDiscoverer._sync_list_files()
- **Metrics**: app/core/metrics.py
- **Circuit breaker**: LLMError handling in components

