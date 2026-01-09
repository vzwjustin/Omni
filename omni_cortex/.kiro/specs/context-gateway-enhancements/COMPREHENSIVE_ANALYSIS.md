# Comprehensive Context Gateway System Analysis

## Executive Summary

The Omni-Cortex Context Gateway is a sophisticated Gemini-powered preprocessing layer that orchestrates four specialized components to prepare rich, structured context for Claude. The system uses a composition pattern with QueryAnalyzer, FileDiscoverer, DocumentationSearcher, and CodeSearcher working in parallel to gather and structure information before Claude performs deep reasoning.

**Key Architecture Principle**: Gemini Flash (cheap, fast) does the "egg hunting" so Claude (expensive, powerful) can focus on deep reasoning.

---

## 1. CONTEXT GATEWAY ARCHITECTURE

### 1.1 High-Level Flow

```
User Query
    ↓
[ContextGateway.prepare_context()]
    ├─→ [QueryAnalyzer] - Analyzes intent, task type, complexity
    ├─→ [FileDiscoverer] - Discovers relevant files with scoring
    ├─→ [DocumentationSearcher] - Searches web + ChromaDB
    └─→ [CodeSearcher] - Searches codebase via grep/git
    ↓
[StructuredContext] - Rich context packet
    ↓
Claude - Deep reasoning on prepared context
```

### 1.2 Core Components

#### ContextGateway (app/core/context_gateway.py)
- **Responsibility**: Orchestrates all context preparation
- **Pattern**: Composition with dependency injection
- **Key Methods**:
  - `prepare_context()` - Main entry point, runs all components in parallel
  - `quick_analyze()` - Fast query analysis without file discovery
  - `_fallback_analyze()` - Pattern-based fallback when Gemini unavailable

#### QueryAnalyzer (app/core/context/query_analyzer.py)
- **Responsibility**: Analyzes queries to understand intent
- **Gemini Integration**: Uses Gemini 3 Flash with thinking mode support
- **Outputs**:
  - task_type: debug, implement, refactor, architect, test, review, explain, optimize
  - complexity: low, medium, high, very_high
  - framework: Recommended reasoning framework
  - execution_steps: Planned steps
  - success_criteria: What success looks like
  - blockers: Potential issues

#### FileDiscoverer (app/core/context/file_discoverer.py)
- **Responsibility**: Discovers and ranks relevant files
- **Process**:
  1. Lists workspace files (filtered by extension/exclusions)
  2. Sends file listing to Gemini for relevance scoring
  3. Returns ranked FileContext objects
- **Outputs**: List of FileContext with path, relevance_score, summary, key_elements

#### DocumentationSearcher (app/core/context/doc_searcher.py)
- **Responsibility**: Searches web and knowledge base
- **Dual Mode**:
  1. Web search via Gemini with Google Search grounding
  2. Knowledge base search via ChromaDB collections
- **Collections Searched**:
  - learnings: Past successful solutions
  - debugging_knowledge: Bug-fix patterns
  - reasoning_knowledge: Chain-of-thought examples
  - framework_docs: Framework documentation

#### CodeSearcher (app/core/context/code_searcher.py)
- **Responsibility**: Searches codebase
- **Methods**:
  1. Extracts search terms from query using Gemini
  2. Runs ripgrep (preferred) or grep for pattern matching
  3. Searches git log for relevant commits
- **Outputs**: CodeSearchContext with search_type, query, results, file_count, match_count

### 1.3 Data Models

#### StructuredContext (Main Output)
```python
@dataclass
class StructuredContext:
    # Task Understanding
    task_type: str
    task_summary: str
    complexity: str
    
    # Relevant Files
    relevant_files: List[FileContext]
    entry_point: Optional[str]
    
    # Documentation
    documentation: List[DocumentationContext]
    
    # Code Search Results
    code_search: List[CodeSearchContext]
    
    # Framework Recommendation
    recommended_framework: str
    framework_reason: str
    chain_suggestion: Optional[List[str]]
    
    # Execution Plan
    execution_steps: List[str]
    success_criteria: List[str]
    potential_blockers: List[str]
    
    # Token Budget
    token_budget: int
    actual_tokens: int
```

#### FileContext
```python
@dataclass
class FileContext:
    path: str
    relevance_score: float  # 0-1
    summary: str
    key_elements: List[str]  # functions, classes, etc.
    line_count: int
    size_kb: float
```

---

## 2. GEMINI INTEGRATION

### 2.1 API Patterns

#### Package Support
- **Primary**: `google-genai>=0.3.0` (NEW - supports thinking mode)
- **Fallback**: `google-generativeai>=0.8.0` (DEPRECATED)

#### Model Selection
- **Routing Model**: `gemini-3-flash-preview` (default)
- **Thinking Mode**: Available on Gemini 3 models
- **Temperature**: 0.2-0.3 for deterministic analysis

### 2.2 Thinking Mode Implementation

#### QueryAnalyzer Thinking Mode
```python
# NEW google-genai package
config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        thinking_level="HIGH",  # For complex queries
    ),
    temperature=0.3,
    response_mime_type="application/json"
)
```

#### Fallback Strategy
- Detects model support: "thinking" in name or "gemini-3" in name
- Falls back to standard config if thinking unavailable
- Logs reasoning quality metrics

### 2.3 Error Handling

#### Graceful Degradation Hierarchy
1. **Gemini Available**: Full analysis with thinking mode
2. **Gemini Unavailable**: Pattern-based fallback analysis
3. **Partial Failures**: Continue with available components

#### Error Types
- `ProviderNotConfiguredError`: Missing GOOGLE_API_KEY
- `LLMError`: API call failed
- Specific detection for:
  - Billing issues (insufficient funds/quota)
  - Auth issues (invalid API key)
  - Network errors

#### Fallback Analysis (Pattern-Based)
```python
def _fallback_analyze(query: str) -> Dict[str, Any]:
    # Uses regex patterns to detect task types
    patterns = [
        (r'\b(debug|error|fix|bug)\b', "debug", "self_debugging"),
        (r'\b(implement|add|create|build)\b', "implement", "reason_flux"),
        (r'\b(refactor|clean|improve)\b', "refactor", "chain_of_verification"),
        (r'\b(explain|how|what|why)\b', "explain", "chain_of_note"),
    ]
```

### 2.4 Input Sanitization

#### Prompt Injection Prevention
```python
def _sanitize_prompt_input(text: str, max_length: int = 50000) -> str:
    # 1. Truncate to prevent context flooding
    # 2. Remove null bytes and control characters
    # 3. Escape prompt injection patterns:
    #    - "```" → "` ` `" (break code blocks)
    #    - "QUERY:" → "[QUERY]" (prevent fake headers)
    #    - "Respond in JSON" → "[Respond in JSON]"
```

---

## 3. MCP SERVER INTEGRATION

### 3.1 Tool Exposure

#### LEAN_MODE (Default: True)
- **8 Core Tools**:
  1. `prepare_context` - Gemini context preparation
  2. `reason` - Smart framework routing
  3. `execute_code` - Sandboxed execution
  4. `health` - Server health check
  5. `count_tokens` - Token counting
  6. `compress_content` - Content compression
  7. `detect_truncation` - Truncation detection
  8. `manage_claude_md` - CLAUDE.md management

#### FULL_MODE (LEAN_MODE=false)
- All 8 core tools PLUS
- 62 `think_*` framework tools
- 11 search/discovery tools
- 2 memory tools

### 3.2 prepare_context Tool Handler

#### Input Schema
```python
{
    "query": str,  # Required: task description
    "workspace_path": str,  # Optional: project directory
    "code_context": str,  # Optional: code snippets
    "file_list": List[str],  # Optional: pre-specified files
    "search_docs": bool,  # Optional: search web (default: true)
    "output_format": str  # Optional: "prompt" or "json"
}
```

#### Output Format
- **prompt**: Formatted markdown for Claude
- **json**: Raw StructuredContext as JSON

### 3.3 reason Tool Integration

#### Auto-Context Feature
```python
# In reason_handler.py
if not context or context == "None provided":
    gateway = get_context_gateway()
    structured_context = await gateway.prepare_context(
        query=query,
        workspace_path=arguments.get("workspace_path"),
        code_context=arguments.get("code_context"),
        file_list=arguments.get("file_list"),
        search_docs=True,
    )
    context = structured_context.to_claude_prompt()
```

---

## 4. FRAMEWORK ORCHESTRATION

### 4.1 HyperRouter Integration

#### Router Flow
```
prepare_context() output
    ↓
[HyperRouter.route()]
    ├─ Analyzes task_type, complexity
    ├─ Checks vibe_dictionary
    ├─ Selects framework(s)
    └─ Returns framework_chain
    ↓
[execute_framework_node()]
    ├─ Single framework execution
    └─ OR Pipeline execution (multiple frameworks)
```

#### Framework Selection Signals
- Task type (debug, implement, refactor, etc.)
- Complexity estimate (low, medium, high, very_high)
- Vibe patterns (keywords, patterns)
- User preference (if specified)

### 4.2 Pipeline Execution

#### Multi-Framework Chains
```python
# Example: Complex debugging task
framework_chain = [
    "self_debugging",      # Step 1: Analyze the problem
    "tree_of_thoughts",    # Step 2: Explore solutions
    "chain_of_verification" # Step 3: Verify solution
]
```

#### Intermediate Result Handling
- Each framework receives output of previous
- Results stored in `reasoning_steps`
- Token usage aggregated
- Metrics recorded per framework

---

## 5. DATA MODELS AND STATE

### 5.1 GraphState (app/state.py)

#### Core Fields
```python
class GraphState(TypedDict):
    # Input
    query: str
    code_snippet: Optional[str]
    file_list: List[str]
    ide_context: Optional[str]
    preferred_framework: Optional[str]
    max_iterations: int
    
    # Routing
    selected_framework: str
    framework_chain: List[str]
    routing_category: str
    task_type: str
    complexity_estimate: float
    
    # Working Memory
    working_memory: Dict[str, Any]
    reasoning_steps: List[Dict[str, Any]]
    step_counter: int
    
    # Episodic Memory
    episodic_memory: List[Dict[str, Any]]
    
    # Output
    final_code: Optional[str]
    final_answer: Optional[str]
    confidence_score: float
    tokens_used: int
    
    # Quiet-STaR
    quiet_thoughts: List[str]
    
    # Error Handling
    error: Optional[str]
```

### 5.2 MemoryStore (app/state.py)

#### Persistent Storage
```python
@dataclass
class MemoryStore:
    working: dict[str, Any]  # Current session
    episodic: list[dict[str, Any]]  # Historical patterns
    thought_templates: list[dict[str, Any]]  # Successful reasoning
    total_queries: int
    successful_queries: int
    framework_usage: dict[str, int]
```

#### Methods
- `add_episode()` - Store completed reasoning
- `add_thought_template()` - Store successful patterns
- `find_similar_templates()` - Retrieve similar patterns
- `record_framework_usage()` - Track framework usage
- `clear_working_memory()` - Reset for new session

---

## 6. CONFIGURATION AND SETTINGS

### 6.1 OmniCortexSettings (app/core/settings.py)

#### API Keys
```python
google_api_key: Optional[str]  # GOOGLE_API_KEY
anthropic_api_key: Optional[str]  # ANTHROPIC_API_KEY
openai_api_key: Optional[str]  # OPENAI_API_KEY
openrouter_api_key: Optional[str]  # OPENROUTER_API_KEY
```

#### Model Configuration
```python
routing_model: str = "gemini-3-flash-preview"
deep_reasoning_model: str = "gemini-3-flash-preview"
fast_synthesis_model: str = "gemini-3-flash-preview"
llm_provider: str = "google"  # google, anthropic, openai, openrouter
```

#### Context Gateway Settings
```python
embedding_provider: str = "openrouter"
embedding_model: str = "text-embedding-3-small"
chroma_persist_dir: Path = Path("/app/data/chroma")
rag_default_k: int = 5
```

#### Feature Flags
```python
lean_mode: bool = True  # 8 tools vs 81 tools
enable_auto_ingest: bool = True
enable_dspy_optimization: bool = True
enable_prm_scoring: bool = True
use_langchain_llm: bool = False
```

### 6.2 Constants (app/core/constants.py)

#### Content Limits
```python
SNIPPET_SHORT: int = 500
SNIPPET_STANDARD: int = 1000
SNIPPET_MAX: int = 2000
COMMAND_OUTPUT: int = 3000
```

#### Search Limits
```python
K_STANDARD: int = 3
K_DEFAULT: int = 5
GREP_MAX_COUNT: int = 10
SEARCH_QUERIES_MAX: int = 3
```

#### Workspace Limits
```python
MAX_FILES_CONTEXT: int = 15
MAX_FILES_ANALYZE: int = 100
MAX_FILES_SCAN: int = 200
EXCLUDE_DIRS: Tuple = (__pycache__, .git, node_modules, ...)
CODE_EXTENSIONS: Tuple = (.py, .js, .ts, .go, .rs, ...)
```

---

## 7. ERROR HANDLING AND RESILIENCE

### 7.1 Error Taxonomy (app/core/errors.py)

#### Hierarchy
```
OmniCortexError (base)
├── RoutingError
│   ├── FrameworkNotFoundError
│   └── CategoryNotFoundError
├── ExecutionError
│   ├── SandboxSecurityError
│   └── SandboxTimeoutError
├── MemoryError
│   └── ThreadNotFoundError
├── RAGError
│   ├── CollectionNotFoundError
│   ├── EmbeddingError
│   └── ContextRetrievalError
└── LLMError
    ├── ProviderNotConfiguredError
    ├── RateLimitError
    └── SamplerTimeout
```

### 7.2 Graceful Degradation Patterns

#### Pattern 1: Component Failure Isolation
```python
# In prepare_context()
file_result, doc_result, code_result = await asyncio.gather(
    _discover_files(),
    _search_docs(),
    _search_code(),
    return_exceptions=True,  # Don't fail entire operation
)

# Process results, handling exceptions individually
if isinstance(file_result, Exception):
    logger.warning("file_discovery_failed", error=str(file_result))
    converted_files = []
else:
    converted_files = [...]
```

#### Pattern 2: Fallback Analysis
```python
try:
    query_analysis = await self._query_analyzer.analyze(query)
except Exception as e:
    logger.warning("query_analysis_failed_using_fallback", error=str(e))
    query_analysis = self._fallback_analyze(query)
```

#### Pattern 3: Partial Results
```python
# Continue with available components
if not collection:
    logger.debug("chroma_unavailable")
    return []  # Return empty, don't fail

# Individual collection search failures don't block others
try:
    learnings = await manager.search_learnings(query)
except RAGError as e:
    logger.debug("learnings_search_failed", error=str(e))
    # Continue to next collection
```

### 7.3 Retry Logic

#### Exponential Backoff (app/graph.py)
```python
MAX_RETRIES = 3
BASE_BACKOFF_MS = 100

# Backoff sequence: 100ms, 200ms, 400ms
backoff_ms = BASE_BACKOFF_MS * (2 ** retry_count)
await asyncio.sleep(backoff_ms / 1000.0)
```

---

## 8. PERFORMANCE AND CACHING

### 8.1 Existing Caching Mechanisms

#### Collection Manager Caching
```python
# Thread-safe singleton caching
_collections: Dict[str, Chroma] = {}

# Fast path: already cached
if collection_name in self._collections:
    return self._collections[collection_name]
```

#### Embedding Function Caching
```python
# Lazy initialization with thread-safe locking
_embedding_function: Any = None
_embedding_lock = threading.Lock()

def get_embedding_function(self) -> Any:
    if self._embedding_function is not None:
        return self._embedding_function
    # Initialize once, reuse thereafter
```

### 8.2 Token Budget Management

#### Current Implementation
```python
token_budget: int = LLM.CONTEXT_TOKEN_BUDGET  # 50,000 tokens
actual_tokens: int = 0

# Tracked in StructuredContext
def to_claude_prompt(self) -> str:
    # Limits files to top 10
    for f in self.relevant_files[:10]:
        # Include summary, not full content
```

### 8.3 Parallel Execution

#### Concurrent Component Execution
```python
file_result, doc_result, code_result = await asyncio.gather(
    _discover_files(),
    _search_docs(),
    _search_code(),
    return_exceptions=True,
)
```

#### Benefits
- File discovery doesn't wait for doc search
- Code search runs independently
- Total time = max(component_times), not sum

---

## 9. TESTING AND VALIDATION

### 9.1 Validation Patterns

#### Input Validation
```python
# In rate_limiter.py
def validate_input_size(self, arguments: Dict, tool_name: str) -> tuple[bool, str]:
    total_size = sum(len(v.encode("utf-8")) for v in arguments.values() if isinstance(v, str))
    if total_size > self.config.max_input_size:
        return False, f"Input size exceeds limit"
    return True, ""
```

#### Schema Validation
```python
# In handlers/validation.py
def validate_query(query: str, required: bool = False) -> str:
    if required and not query:
        raise ValidationError("query is required")
    if len(query) > CONTENT.QUERY_LOG:
        query = query[:CONTENT.QUERY_LOG]
    return query
```

### 9.2 Available Test Scripts

- `python scripts/verify_learning_offline.py` - Learning flow validation
- `python scripts/test_mcp_search.py` - ChromaDB search testing
- `python -m scripts.debug_search` - Chroma/OpenAI diagnostics
- `python scripts/validate_frameworks.py` - Framework validation

---

## 10. DEPENDENCIES AND EXTERNAL SERVICES

### 10.1 LLM Providers

#### Google Gemini
- **Package**: `google-genai>=0.3.0` (primary), `google-generativeai>=0.8.0` (fallback)
- **Models**: gemini-3-flash-preview, gemini-2.0-flash
- **Features**: Thinking mode, Google Search grounding
- **Cost**: Cheapest option for context preparation

#### Anthropic Claude
- **Package**: `anthropic>=0.40.0`
- **Role**: Deep reasoning (not used by context gateway)
- **Integration**: Via LangChain

#### OpenAI
- **Package**: `openai>=1.50.0`
- **Role**: Embeddings (via OpenRouter)
- **Integration**: Via LangChain

### 10.2 Vector Database

#### ChromaDB
- **Package**: `chromadb>=0.5.3`
- **Role**: Persistent knowledge base
- **Collections**:
  - frameworks: Framework implementations
  - documentation: Markdown docs
  - configs: Configuration files
  - utilities: Helper functions
  - tests: Test files
  - integrations: LangChain/LangGraph code
  - learnings: Successful solutions
  - debugging_knowledge: Bug-fix patterns
  - reasoning_knowledge: Chain-of-thought examples
  - instruction_knowledge: Task completion examples

#### Embeddings
- **Provider**: OpenRouter (default)
- **Model**: text-embedding-3-small
- **Fallback**: HuggingFace, Gemini

### 10.3 Orchestration

#### LangGraph
- **Package**: `langgraph>=0.2.0`
- **Role**: Workflow orchestration
- **Features**: State management, checkpointing, conditional edges

#### LangChain
- **Package**: `langchain>=0.3.0`
- **Role**: Memory management, RAG integration
- **Components**: ConversationBufferMemory, Chroma integration

### 10.4 Monitoring

#### Prometheus
- **Package**: `prometheus-client>=0.20.0`
- **Metrics**:
  - Framework execution counts, duration, tokens
  - Router decisions and chain lengths
  - MCP request metrics
  - Rate limit rejections
  - Cache hits/misses
  - RAG search metrics

---

## 11. CURRENT LIMITATIONS AND GAPS

### 11.1 Not Yet Implemented (From Requirements)

1. **Intelligent Context Caching** - No query similarity-based caching
2. **Streaming Context Preparation** - No progress events/streaming
3. **Multi-Repository Discovery** - Single repo only
4. **Enhanced Documentation Grounding** - Limited source attribution
5. **Context Quality Metrics** - Basic logging only
6. **Intelligent Token Budget Management** - Fixed limits, no dynamic optimization
7. **Advanced Fallback Mechanisms** - Pattern-based only, no circuit breaker

### 11.2 Known Constraints

- **File Discovery**: Max 200 files scanned, 100 analyzed
- **Documentation**: Max 5 results returned
- **Code Search**: Max 10 grep results per query
- **Token Budget**: Fixed 50,000 token limit
- **Thinking Mode**: Only on Gemini 3 models
- **Workspace Listing**: Synchronous I/O (offloaded to thread pool)

---

## 12. INTEGRATION POINTS FOR ENHANCEMENTS

### 12.1 Where to Add Caching

**File**: `app/core/context_gateway.py`
- Add `_cache` dict to ContextGateway.__init__()
- Implement cache key generation based on query similarity
- Add TTL management per component

### 12.2 Where to Add Streaming

**File**: `app/core/context_gateway.py`
- Modify `prepare_context()` to yield progress events
- Use asyncio.Queue for progress communication
- Add cancellation token support

### 12.3 Where to Add Multi-Repo Support

**File**: `app/core/context/file_discoverer.py`
- Detect multiple .git directories
- Run parallel Gemini analysis per repo
- Label results with repo name

### 12.4 Where to Add Metrics

**File**: `app/core/metrics.py`
- Add context gateway-specific metrics
- Track component performance
- Monitor cache effectiveness

---

## 13. DEPLOYMENT AND RUNTIME

### 13.1 Docker Deployment

```bash
docker-compose build
docker-compose up -d
docker-compose logs -f
```

### 13.2 Local Development

```bash
pip install -r requirements.txt
python -m server.main
```

### 13.3 Environment Configuration

```bash
# Copy template
cp .env.example .env

# Required for context gateway
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # For embeddings

# Optional
LEAN_MODE=true  # 8 tools vs 81
PRODUCTION_LOGGING=false
```

---

## 14. SUMMARY OF KEY INSIGHTS

### Architecture Strengths
1. **Composition Pattern**: Clean separation of concerns
2. **Graceful Degradation**: Works without Gemini via fallbacks
3. **Parallel Execution**: Efficient use of async/await
4. **Thread Safety**: Proper locking for singletons
5. **Error Isolation**: Component failures don't block others

### Integration Points
1. **MCP Server**: Via `prepare_context` and `reason` tools
2. **Framework Selection**: Via HyperRouter
3. **Memory**: Via LangChain integration
4. **RAG**: Via ChromaDB collections

### Enhancement Opportunities
1. Query similarity-based caching (semantic hashing)
2. Streaming progress events (asyncio.Queue)
3. Multi-repo support (parallel git detection)
4. Dynamic token budget allocation
5. Circuit breaker for Gemini API
6. Advanced metrics and observability

