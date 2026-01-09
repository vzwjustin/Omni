# Design Document

## Overview

This design document outlines enhancements to the Omni-Cortex Context Gateway system based on comprehensive codebase analysis. The Context Gateway is a sophisticated Gemini-powered preprocessing layer that orchestrates four specialized components (QueryAnalyzer, FileDiscoverer, DocumentationSearcher, CodeSearcher) to prepare rich, structured context for Claude's deep reasoning.

**Current Architecture Strengths:**
- Composition pattern with clean separation of concerns
- Graceful degradation via pattern-based fallbacks
- Parallel execution of discovery components using asyncio.gather()
- Thread-safe singleton patterns with proper locking
- Component failure isolation (exceptions don't cascade)

**Enhancement Strategy:**
Build upon the existing solid architecture to add intelligent caching, streaming progress, multi-repository support, enhanced documentation grounding, comprehensive metrics, dynamic token budget management, and advanced resilience patterns.

## Architecture

### Current System Architecture

```
User Query → MCP prepare_context Tool
    ↓
ContextGateway.prepare_context()
    ├─→ [QueryAnalyzer] - Gemini analysis with thinking mode
    ├─→ [FileDiscoverer] - Gemini-powered file relevance scoring  
    ├─→ [DocumentationSearcher] - Web search + ChromaDB
    └─→ [CodeSearcher] - Gemini query extraction + grep/git
    ↓ (asyncio.gather - parallel execution)
StructuredContext Assembly
    ↓
Claude Prompt or JSON Output
```

### Enhanced Architecture

```
User Query → Enhanced MCP Tools (prepare_context + prepare_context_streaming)
    ↓
Enhanced ContextGateway
    ├─→ [Context Cache] - Query similarity-based caching
    ├─→ [Progress Emitter] - Real-time streaming events
    ├─→ [Multi-Repo Detector] - Multiple repository discovery
    ├─→ [Enhanced QueryAnalyzer] - Adaptive thinking mode
    ├─→ [Enhanced FileDiscoverer] - Multi-repo file discovery
    ├─→ [Enhanced DocumentationSearcher] - Source attribution
    ├─→ [Enhanced CodeSearcher] - Pattern summarization
    ├─→ [Token Budget Manager] - Dynamic allocation
    ├─→ [Circuit Breaker] - Advanced resilience
    └─→ [Metrics Collector] - Comprehensive monitoring
    ↓
Enhanced StructuredContext with Quality Metrics
    ↓
Optimized Claude Prompt or Detailed JSON
```

## Components and Interfaces

### 1. Context Cache System

#### Interface
```python
class ContextCache:
    def __init__(self, ttl_settings: Dict[str, int]):
        self._cache: Dict[str, CacheEntry] = {}
        self._ttl_settings = ttl_settings
        self._workspace_fingerprints: Dict[str, str] = {}
    
    async def get(self, cache_key: str) -> Optional[CacheEntry]
    async def set(self, cache_key: str, value: Any, cache_type: str)
    async def invalidate_workspace(self, workspace_path: str)
    def generate_cache_key(self, query: str, workspace_path: str) -> str
    def _compute_query_similarity_hash(self, query: str) -> str
    def _compute_workspace_fingerprint(self, workspace_path: str) -> str
```

#### Implementation Strategy
- **Query Similarity**: Use semantic hashing based on query intent and keywords
- **Workspace Fingerprint**: Hash of file modification times and structure
- **TTL Management**: Separate TTLs for query analysis (1h), file discovery (30m), docs (24h)
- **Cache Invalidation**: File system watcher for workspace changes
- **Fallback Support**: Serve stale cache when Gemini API fails

### 2. Streaming Context Preparation

#### Interface
```python
class StreamingContextGateway(ContextGateway):
    async def prepare_context_streaming(
        self, 
        query: str,
        progress_callback: Callable[[ProgressEvent], None],
        cancellation_token: asyncio.Event,
        **kwargs
    ) -> AsyncIterator[Union[ProgressEvent, StructuredContext]]
    
    def _emit_progress(self, component: str, status: str, data: Any)
    async def _estimate_completion_time(self, workspace_size: int) -> float
```

#### Progress Event Types
```python
@dataclass
class ProgressEvent:
    component: str  # "query_analysis", "file_discovery", "doc_search", "code_search"
    status: str     # "started", "progress", "completed", "failed"
    progress: float # 0.0 to 1.0
    data: Any      # Component-specific data
    timestamp: datetime
    estimated_completion: Optional[float]
```

### 3. Multi-Repository Discovery

#### Interface
```python
class MultiRepoFileDiscoverer(FileDiscoverer):
    async def discover_multi_repo(
        self,
        query: str,
        workspace_path: str,
        max_files: int = 15
    ) -> List[FileContext]
    
    def _detect_repositories(self, workspace_path: str) -> List[RepoInfo]
    async def _analyze_repository(self, repo_info: RepoInfo, query: str) -> List[FileContext]
    def _follow_cross_repo_dependencies(self, repos: List[RepoInfo]) -> Dict[str, List[str]]
```

#### Repository Information
```python
@dataclass
class RepoInfo:
    path: str
    name: str
    git_root: str
    ignore_patterns: List[str]
    access_permissions: Dict[str, bool]
    last_commit: Optional[str]
```

### 4. Enhanced Documentation Grounding

#### Interface
```python
class EnhancedDocumentationSearcher(DocumentationSearcher):
    async def search_web_with_attribution(self, query: str) -> List[DocumentationContext]
    def _extract_grounding_metadata(self, response) -> List[SourceAttribution]
    def _merge_web_and_local_results(
        self, 
        web_results: List[DocumentationContext],
        local_results: List[DocumentationContext]
    ) -> List[DocumentationContext]
    def _prioritize_by_authority(self, results: List[DocumentationContext]) -> List[DocumentationContext]
```

#### Source Attribution
```python
@dataclass
class SourceAttribution:
    url: str
    title: str
    domain: str
    authority_score: float
    publication_date: Optional[datetime]
    last_updated: Optional[datetime]
```

### 5. Token Budget Manager

#### Interface
```python
class TokenBudgetManager:
    def __init__(self, base_budget: int = 50000):
        self.base_budget = base_budget
        self.complexity_multipliers = {"low": 0.7, "medium": 1.0, "high": 1.5, "very_high": 2.0}
    
    def calculate_budget(self, complexity: str, task_type: str) -> int
    def allocate_budget(self, total_budget: int) -> Dict[str, int]
    def optimize_content(self, context: StructuredContext, budget: int) -> StructuredContext
    def _prioritize_content(self, items: List[Any], budget: int) -> List[Any]
```

### 6. Circuit Breaker and Advanced Resilience

#### Interface
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any
    def _record_success(self)
    def _record_failure(self)
    def _should_attempt_reset(self) -> bool
```

### 7. Comprehensive Metrics System

#### Interface
```python
class ContextGatewayMetrics:
    def __init__(self):
        self.api_calls = Counter()
        self.tokens_used = Histogram()
        self.component_duration = Histogram()
        self.cache_hits = Counter()
        self.context_quality = Histogram()
    
    def record_api_call(self, component: str, model: str, tokens: int, duration: float)
    def record_cache_operation(self, operation: str, hit: bool, tokens_saved: int = 0)
    def record_context_quality(self, quality_score: float, relevance_scores: List[float])
    def record_component_performance(self, component: str, duration: float, success: bool)
```

## Data Models

### Enhanced StructuredContext

```python
@dataclass
class EnhancedStructuredContext(StructuredContext):
    # Existing fields from StructuredContext...
    
    # New enhancement fields
    cache_metadata: Optional[CacheMetadata] = None
    quality_metrics: Optional[QualityMetrics] = None
    repository_info: List[RepoInfo] = field(default_factory=list)
    source_attributions: List[SourceAttribution] = field(default_factory=list)
    token_budget_usage: Optional[TokenBudgetUsage] = None
    component_status: Dict[str, ComponentStatus] = field(default_factory=dict)
    
    def to_claude_prompt_enhanced(self) -> str:
        """Enhanced prompt with quality indicators and source links."""
        
    def to_detailed_json(self) -> Dict[str, Any]:
        """Detailed JSON with all metadata for debugging."""
```

### Supporting Data Models

```python
@dataclass
class CacheMetadata:
    cache_key: str
    cache_hit: bool
    cache_age: timedelta
    cache_type: str  # "query_analysis", "file_discovery", "documentation"
    workspace_fingerprint: str

@dataclass
class QualityMetrics:
    overall_quality_score: float  # 0.0 to 1.0
    component_quality_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    completeness_score: float
    relevance_distribution: List[float]

@dataclass
class TokenBudgetUsage:
    allocated_budget: int
    actual_usage: int
    utilization_percentage: float
    component_allocation: Dict[str, int]
    optimization_applied: bool

@dataclass
class ComponentStatus:
    status: str  # "success", "partial", "fallback", "failed"
    execution_time: float
    error_message: Optional[str] = None
    fallback_method: Optional[str] = None
    api_calls_made: int = 0
    tokens_consumed: int = 0
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Cache Consistency and Invalidation
*For any* query and workspace combination, when files are modified in the workspace, cached file discovery results should be invalidated while preserving unaffected cache entries
**Validates: Requirements 1.1, 1.3**

### Property 2: Cache TTL Enforcement
*For any* cached entry, the entry should be automatically expired and removed when its TTL is exceeded, with different TTL values correctly applied per cache type
**Validates: Requirements 1.5**

### Property 3: Stale Cache Fallback
*For any* Gemini API failure, if stale cached results exist, they should be served with appropriate warnings rather than failing the entire operation
**Validates: Requirements 1.4**

### Property 4: Thinking Mode Adaptation
*For any* query complexity level and token budget combination, thinking mode usage should be appropriately adapted (HIGH for complex queries with sufficient budget, standard otherwise)
**Validates: Requirements 2.1, 2.5**

### Property 5: Streaming Progress Completeness
*For any* context preparation operation, progress events should be emitted for all four components (query analysis, file discovery, documentation search, code search) with monotonically increasing progress values
**Validates: Requirements 3.1, 3.2, 3.3**

### Property 6: Cancellation Cleanup
*For any* context preparation that is cancelled, all active Gemini API calls should be cleaned up and a partial StructuredContext should be returned with component status indicators
**Validates: Requirements 3.4**

### Property 7: Multi-Repository Discovery
*For any* workspace containing multiple git repositories, each repository should be identified and analyzed in parallel, with results labeled by repository name
**Validates: Requirements 4.1, 4.3**

### Property 8: Cross-Repository Dependency Following
*For any* multi-repository workspace with import relationships, the system should follow dependency paths between repositories and include relevant files from all connected repositories
**Validates: Requirements 4.2**

### Property 9: Repository Access Resilience
*For any* multi-repository workspace where some repositories are inaccessible, the system should continue with available repositories and include clear warnings about inaccessible ones
**Validates: Requirements 4.4**

### Property 10: Source Attribution Preservation
*For any* documentation search using Google Search grounding, source URLs and metadata should be extracted from grounding responses and preserved in the final context
**Validates: Requirements 5.1, 5.2**

### Property 11: Documentation Source Prioritization
*For any* documentation search results, official documentation should be ranked higher than community content based on domain authority scores
**Validates: Requirements 5.5**

### Property 12: Comprehensive Metrics Recording
*For any* context preparation operation, metrics should be recorded for Gemini API calls, token usage, response times, and component performance across all four components
**Validates: Requirements 6.1, 6.5**

### Property 13: Cache Effectiveness Tracking
*For any* cache operation (hit or miss), appropriate metrics should be recorded including cache effectiveness and token savings achieved
**Validates: Requirements 6.3**

### Property 14: Token Budget Prioritization
*For any* limited token budget scenario, file summaries should be prioritized over full documentation snippets, and content should be ranked by relevance for inclusion
**Validates: Requirements 7.1**

### Property 15: Dynamic Budget Allocation
*For any* task complexity level, the system should dynamically allocate token budget with higher complexity tasks receiving increased file discovery depth and more detailed analysis
**Validates: Requirements 7.2**

### Property 16: Gemini-Based Content Ranking
*For any* documentation search returning many results, Gemini should be used to rank and filter snippets by relevance rather than using simple heuristics
**Validates: Requirements 7.3**

### Property 17: Pattern Summarization
*For any* code search producing extensive results, patterns should be summarized using intelligent analysis rather than including raw grep output
**Validates: Requirements 7.4**

### Property 18: Fallback Analysis Activation
*For any* Gemini API unavailability, the system should automatically switch to pattern-based fallback analysis for basic task type detection
**Validates: Requirements 8.1**

### Property 19: Component Fallback Isolation
*For any* individual component failure (file discovery, documentation search, code search), the system should use appropriate fallback methods for that component while continuing with others
**Validates: Requirements 8.2, 8.3**

### Property 20: Circuit Breaker Behavior
*For any* sequence of Gemini API failures exceeding the threshold, the circuit breaker should open and subsequent calls should fail fast until the recovery timeout expires
**Validates: Requirements 8.5**

### Property 21: Partial Failure Status Indication
*For any* context preparation with partial component failures, the StructuredContext should clearly indicate which components succeeded, which failed, and which used fallback methods
**Validates: Requirements 8.4**

## Error Handling

### Enhanced Error Taxonomy

Building on the existing error hierarchy in `app/core/errors.py`:

```python
# New error types for enhancements
class ContextCacheError(OmniCortexError):
    """Cache-related errors."""
    pass

class StreamingError(OmniCortexError):
    """Streaming operation errors."""
    pass

class MultiRepoError(OmniCortexError):
    """Multi-repository operation errors."""
    pass

class CircuitBreakerError(OmniCortexError):
    """Circuit breaker activation errors."""
    pass
```

### Resilience Patterns

#### 1. Enhanced Circuit Breaker Pattern
- **Failure Threshold**: 5 consecutive failures
- **Recovery Timeout**: 60 seconds
- **States**: CLOSED (normal), OPEN (failing fast), HALF_OPEN (testing recovery)
- **Metrics**: Track state transitions and recovery success rates

#### 2. Graceful Degradation Hierarchy
1. **Full Gemini Analysis**: All components with thinking mode
2. **Standard Gemini**: All components without thinking mode  
3. **Partial Gemini**: Some components with fallbacks
4. **Pattern-Based Fallback**: No Gemini, pattern matching only
5. **Minimal Context**: Basic file listing and simple analysis

#### 3. Component Isolation Enhancement
- Each component wrapped in individual circuit breakers
- Component failures don't affect others
- Partial results clearly marked in StructuredContext
- Fallback methods documented per component

## Testing Strategy

### Dual Testing Approach

**Unit Tests**: Verify specific examples, edge cases, and error conditions
- Cache key generation and similarity hashing
- TTL expiration and invalidation logic
- Circuit breaker state transitions
- Multi-repository detection algorithms
- Token budget allocation calculations

**Property Tests**: Verify universal properties across all inputs (minimum 100 iterations per test)
- Cache consistency across different query/workspace combinations
- Streaming progress event completeness and ordering
- Multi-repository discovery and labeling
- Fallback behavior under various failure scenarios
- Token budget optimization effectiveness

### Property Test Configuration

Each property test must reference its design document property:
- **Tag Format**: `Feature: context-gateway-enhancements, Property {number}: {property_text}`
- **Minimum Iterations**: 100 per property test
- **Test Data Generation**: Smart generators for queries, workspaces, and failure scenarios
- **Assertion Strategy**: Verify both functional correctness and performance characteristics

### Integration Testing

- **MCP Tool Integration**: Test enhanced prepare_context and new streaming tools
- **Gemini API Integration**: Test thinking mode adaptation and circuit breaker behavior
- **ChromaDB Integration**: Test enhanced documentation search and caching
- **File System Integration**: Test multi-repository discovery and cache invalidation

### Performance Testing

- **Cache Performance**: Measure cache hit rates and response time improvements
- **Streaming Performance**: Verify streaming doesn't significantly impact total time
- **Multi-Repo Performance**: Ensure parallel repository analysis scales appropriately
- **Token Budget Performance**: Verify optimization doesn't degrade context quality