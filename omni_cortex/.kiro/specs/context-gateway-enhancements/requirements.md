# Requirements Document

## Introduction

This specification defines enhancements to the existing Context Gateway system in Omni-Cortex. The Context Gateway is a Gemini-powered preprocessing layer that acts as the "entire backend" for context preparation. It uses Gemini Flash (cheap, fast) to do all the "egg hunting" so Claude (expensive, powerful) can focus on deep reasoning. The system currently orchestrates four specialized components in parallel to prepare rich, structured context.

**Current Architecture:**
```
User Query → Gemini Flash Context Gateway → Structured Context → Claude
```

**Current Flow:**
1. **Query Analysis** - Gemini analyzes query to understand task type, complexity, framework recommendation
2. **Parallel Discovery** - File discovery, web documentation search, and code search run concurrently  
3. **Context Structuring** - Results assembled into rich StructuredContext with execution plan
4. **Claude Handoff** - Formatted prompt with everything Claude needs for deep reasoning

## Glossary

- **Context_Gateway**: The Gemini-powered context optimization layer that orchestrates all backend context preparation
- **Query_Analyzer**: Gemini-powered component that analyzes queries using thinking mode for deep task understanding
- **File_Discoverer**: Gemini-powered component that lists workspace files and scores relevance with summaries
- **Documentation_Searcher**: Component using Gemini with Google Search grounding for web docs + ChromaDB knowledge base
- **Code_Searcher**: Component using Gemini to extract search terms, then grep/ripgrep and git log for code patterns
- **StructuredContext**: Rich context packet with task analysis, relevant files, documentation, execution plan, and framework recommendation
- **Context_Cache**: New intelligent caching layer to avoid expensive re-computation of Gemini analysis
- **Streaming_Context**: New real-time progress system for long-running context preparation
- **Multi_Repo_Discovery**: New capability to handle microservices and monorepo contexts
- **Context_Quality_Metrics**: New monitoring system for context preparation effectiveness

## Requirements

### Requirement 1: Intelligent Context Caching System

**User Story:** As a developer, I want the context gateway to cache expensive Gemini analysis results, so that similar queries don't require re-running file discovery, documentation search, and query analysis.

#### Acceptance Criteria

1. WHEN a query is processed, THE Context_Cache SHALL store Gemini analysis results with cache keys based on query similarity and workspace fingerprint
2. WHEN a similar query is received within cache TTL, THE Context_Gateway SHALL reuse cached file relevance scores and documentation results
3. WHEN workspace files are modified, THE Context_Cache SHALL invalidate cached file discovery results for affected directories
4. WHEN Gemini API calls fail, THE Context_Cache SHALL serve stale cached results as fallback with appropriate warnings
5. THE Context_Cache SHALL support separate TTL settings for query analysis (1 hour), file discovery (30 minutes), and documentation search (24 hours)

### Requirement 2: Gemini Thinking Mode Optimization

**User Story:** As a developer working on complex problems, I want the context gateway to leverage Gemini's thinking mode more effectively, so that query analysis and file relevance scoring are more accurate.

#### Acceptance Criteria

1. WHEN analyzing complex queries, THE Query_Analyzer SHALL use HIGH thinking mode for deeper task understanding and framework selection
2. WHEN scoring file relevance, THE File_Discoverer SHALL use thinking mode to reason about code relationships and dependencies
3. WHEN thinking mode is unavailable, THE Context_Gateway SHALL gracefully fallback to standard Gemini models
4. WHEN thinking mode analysis completes, THE Context_Gateway SHALL log reasoning quality metrics for monitoring
5. THE Context_Gateway SHALL adapt thinking mode usage based on query complexity and available token budget

### Requirement 3: Streaming Context Preparation

**User Story:** As a developer, I want to see real-time progress during context preparation, so that I understand what the system is doing and can cancel long-running operations.

#### Acceptance Criteria

1. WHEN context preparation begins, THE Context_Gateway SHALL emit progress events for each parallel component (file discovery, doc search, code search)
2. WHEN file discovery is running, THE Context_Gateway SHALL stream discovered files with relevance scores as Gemini processes them
3. WHEN documentation search finds results, THE Context_Gateway SHALL stream documentation snippets as they are retrieved
4. WHEN context preparation is cancelled, THE Context_Gateway SHALL clean up Gemini API calls and return partial StructuredContext
5. THE Context_Gateway SHALL provide estimated completion times based on workspace size and historical Gemini response times

### Requirement 4: Multi-Repository Context Discovery

**User Story:** As a developer working with microservices or monorepos, I want the context gateway to discover and analyze files across multiple related repositories, so that I get comprehensive context for cross-service issues.

#### Acceptance Criteria

1. WHEN workspace contains multiple git repositories, THE File_Discoverer SHALL identify each repository and run parallel Gemini analysis
2. WHEN analyzing cross-repository dependencies, THE Context_Gateway SHALL follow import paths and API calls between services
3. WHEN preparing context for multi-repo tasks, THE StructuredContext SHALL include relevant files from all repositories with repository labels
4. WHEN repository access fails, THE Context_Gateway SHALL continue with available repositories and include warnings in StructuredContext
5. THE Context_Gateway SHALL respect repository-specific .gitignore patterns and access permissions during file discovery

### Requirement 5: Enhanced Documentation Grounding

**User Story:** As a developer, I want better documentation search results with proper source attribution, so that I can trust and verify the information provided by the context gateway.

#### Acceptance Criteria

1. WHEN using Google Search grounding, THE Documentation_Searcher SHALL extract and preserve source URLs from Gemini grounding metadata
2. WHEN documentation snippets are included, THE StructuredContext SHALL include clickable source links and publication dates
3. WHEN ChromaDB knowledge base has relevant content, THE Documentation_Searcher SHALL merge web results with local knowledge
4. WHEN documentation search fails, THE Context_Gateway SHALL fallback to ChromaDB-only search with appropriate notifications
5. THE Documentation_Searcher SHALL prioritize official documentation over community content based on domain authority

### Requirement 6: Context Quality Metrics and Monitoring

**User Story:** As a system administrator, I want comprehensive metrics on context gateway performance and quality, so that I can monitor Gemini API usage and optimize the system.

#### Acceptance Criteria

1. WHEN context preparation completes, THE Context_Gateway SHALL record Gemini API call counts, tokens used, and response times for each component
2. WHEN StructuredContext is used by Claude, THE Context_Gateway SHALL track which context elements were most relevant to the final solution
3. WHEN cache hits occur, THE Context_Gateway SHALL record cache effectiveness metrics and token savings
4. WHEN Gemini API errors occur, THE Context_Gateway SHALL log detailed error metrics with correlation IDs for debugging
5. THE Context_Gateway SHALL expose Prometheus metrics for Gemini API usage, context quality scores, and component performance

### Requirement 7: Intelligent Token Budget Management

**User Story:** As a developer, I want the context gateway to automatically optimize context based on available token budget, so that Claude gets the best possible information within limits.

#### Acceptance Criteria

1. WHEN token budget is limited, THE Context_Gateway SHALL prioritize file summaries over full documentation snippets
2. WHEN preparing context for complex tasks, THE Context_Gateway SHALL increase file discovery depth and use more detailed Gemini analysis
3. WHEN documentation search returns many results, THE Context_Gateway SHALL use Gemini to rank and filter the most relevant snippets
4. WHEN code search produces extensive results, THE Context_Gateway SHALL summarize patterns rather than including raw grep output
5. THE StructuredContext SHALL include actual token count and budget utilization metrics for transparency

### Requirement 8: Fallback and Resilience Improvements

**User Story:** As a developer, I want the context gateway to handle Gemini API failures gracefully, so that I can still get useful context even when the AI service is unavailable.

#### Acceptance Criteria

1. WHEN Gemini API is unavailable, THE Context_Gateway SHALL use pattern-based fallback analysis for basic task type detection
2. WHEN file discovery fails, THE Context_Gateway SHALL fallback to simple file listing with extension-based relevance scoring
3. WHEN documentation search fails, THE Context_Gateway SHALL provide ChromaDB-only results with clear service status indicators
4. WHEN partial failures occur, THE StructuredContext SHALL clearly indicate which components succeeded and which used fallback methods
5. THE Context_Gateway SHALL implement exponential backoff for Gemini API retries with circuit breaker patterns