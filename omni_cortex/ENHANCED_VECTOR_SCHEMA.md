# Enhanced Vector Database Schema

## Overview

The Omni-Cortex vector database has been upgraded with a **production-grade schema** featuring:

- **Rich structured metadata** for precise retrieval
- **Intelligent code chunking** (functions, classes, docstrings)
- **Multiple specialized collections** for different content types
- **Semantic code analysis** with AST parsing
- **6 enhanced search tools** for specialized queries

## Architecture

### Collections

The database is organized into **6 specialized collections**:

| Collection | Purpose | Content Types |
|------------|---------|---------------|
| `frameworks` | Reasoning framework implementations | Framework nodes, core logic |
| `documentation` | Project documentation | README, guides, markdown docs |
| `configs` | Configuration files | YAML, JSON, .env files |
| `utilities` | Helper functions and utilities | Common utilities, state management |
| `tests` | Test files and fixtures | Unit tests, integration tests |
| `integrations` | External integrations | LangChain, LangGraph, MCP server code |

### Metadata Schema

Each document chunk includes **20+ metadata fields**:

```python
{
    # Core identification
    "path": "app/nodes/iterative/active_inf.py",
    "file_name": "active_inf.py",
    "file_type": ".py",
    "category": "framework",
    
    # Chunking info
    "chunk_type": "function",  # full_file, function, class, docstring, section
    "chunk_index": 0,
    "total_chunks": 5,
    
    # Code structure (Python files)
    "function_name": "active_inference_node",
    "class_name": "ActiveInference",
    "imports": "typing,GraphState,quiet_star",
    "decorators": "quiet_star,async",
    
    # Location
    "line_start": 45,
    "line_end": 120,
    "char_count": 1523,
    
    # Framework context
    "framework_name": "active_inference",
    "framework_category": "iterative",
    "module_path": "app.nodes.iterative.active_inf",
    
    # Semantic tags
    "tags": "async,langchain_integration,decorated",
    "has_todo": false,
    "has_fixme": false,
    "complexity_score": 0.35
}
```

## Intelligent Chunking

### Python Files

**AST-based extraction**:
- Individual **functions** with signatures + docstrings
- **Classes** with method lists and docstrings
- **Import statements** tracked for dependency analysis
- **Decorators** preserved for pattern recognition
- **Complexity scoring** for prioritization

### Markdown Files

**Section-based chunking**:
- Split by headers (`#`, `##`, etc.)
- Preserves context within sections
- Detects code blocks, links, annotations

### Configuration Files

Stored as **complete documents** with structured metadata for easy reference.

## Enhanced Search Tools (MCP-Exposed)

All 6 new search tools are available via MCP alongside the original 3:

### 1. `search_frameworks_by_name`
Search within a specific framework's implementation.

```json
{
    "framework_name": "active_inference",
    "query": "hypothesis generation",
    "k": 3
}
```

### 2. `search_by_category`
Search within a specific code category.

```json
{
    "query": "memory management",
    "category": "integration",  // framework, documentation, config, utility, test, integration
    "k": 5
}
```

### 3. `search_function_implementation`
Find specific function implementations by name.

```json
{
    "function_name": "call_deep_reasoner",
    "k": 3
}
```

### 4. `search_class_implementation`
Find specific class implementations by name.

```json
{
    "class_name": "HyperRouter",
    "k": 3
}
```

### 5. `search_documentation_only`
Search only markdown documentation files.

```json
{
    "query": "installation instructions",
    "k": 5
}
```

### 6. `search_with_framework_context`
Search within a framework category.

```json
{
    "query": "error handling",
    "framework_category": "iterative",  // strategy, search, iterative, code, context, fast
    "k": 5
}
```

## Usage

### Initial Ingestion (Enhanced)

Run the enhanced ingestion pipeline:

```bash
python -m app.enhanced_ingestion
```

This will:
1. Scan the repository
2. Extract code structure with AST
3. Generate rich metadata
4. Chunk intelligently
5. Route to specialized collections
6. Index with embeddings

**Statistics**: ~200 files → ~800+ chunks with full metadata

### Querying from Code

```python
from app.collection_manager import get_collection_manager

manager = get_collection_manager()

# Search specific framework
docs = manager.search_frameworks(
    "hypothesis testing logic",
    framework_name="active_inference",
    k=3
)

# Search by function
docs = manager.search_by_function("route_node", k=2)

# Search with filters
docs = manager.search(
    "memory persistence",
    collection_names=["integrations"],
    filter_dict={"has_todo": False}
)
```

### From MCP Clients (IDEs)

```json
{
    "tool": "search_function_implementation",
    "arguments": {
        "function_name": "execute_framework_node",
        "k": 2
    }
}
```

## Benefits

### For Reasoning Frameworks

- **Precise retrieval** of relevant code patterns
- **Framework-specific search** for similar implementations
- **Function-level context** for better code understanding
- **Dependency tracking** via import metadata

### For Development

- **Find implementations** by function/class name instantly
- **Cross-reference** related frameworks and utilities
- **Documentation search** separate from code search
- **Complexity-aware** ranking (simpler examples first)

### For RAG Quality

- **Chunk-level metadata** prevents context mixing
- **Semantic tags** for relevance filtering
- **Multi-collection search** with automatic routing
- **Structure preservation** (functions keep their docstrings)

## Migration from Legacy Schema

The enhanced schema is **backward compatible**. Legacy search still works via `search_documentation` tool.

To migrate existing data:

```bash
# Clear old simple schema
rm -rf /app/data/chroma/*

# Run enhanced ingestion
python -m app.enhanced_ingestion
```

Or run both in parallel (collections are separate).

## Performance

- **Ingestion**: ~5-10 seconds for full repository
- **Search latency**: <100ms per query
- **Storage**: ~50MB for 800 chunks with embeddings
- **Embedding model**: OpenAI `text-embedding-3-large` (3072 dims)

## Future Enhancements

- [ ] Incremental updates (file watching)
- [ ] Cross-reference graph (imports → usage)
- [ ] Temporal versioning (track changes over time)
- [ ] User feedback loop (relevance scoring)
- [ ] Multi-lingual support (beyond Python)

---

**The enhanced schema transforms Omni-Cortex from simple document search to a semantic code intelligence system.**
