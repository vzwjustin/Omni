# Token Reduction Technologies Guide

## Overview

Omni-Cortex implements smart token reduction that **removes formatting overhead while preserving 100% of semantic content**:

1. **TOON (Token-Oriented Object Notation)**: Removes redundant labels and formatting (20-40% savings, LOSSLESS)
2. **LLMLingua-2**: Optional lossy compression for secondary content (50-80% reduction)

## The Smart Strategy: Remove Fluff, Keep Details

### What TOON Removes (Formatting Overhead):
- ❌ Repeated labels: "path:", "score:", "summary:"
- ❌ Redundant markdown syntax: bullet points, excessive spacing
- ❌ Verbose JSON structure: brackets, quotes, commas
- ❌ Column headers repeated per row

### What TOON Keeps (All Information):
- ✅ **Full file summaries** - every detail preserved
- ✅ **Complete code snippets** - nothing truncated
- ✅ **Entire documentation** - full text kept
- ✅ **All explanations** - zero information loss

### Example: Same Content, Less Overhead

**Before (verbose markdown):**
```markdown
## Relevant Files
- **Path**: /src/auth/login.py
  **Score**: 0.95
  **Summary**: Handles user authentication with JWT tokens, includes rate limiting and session management
  **Key Elements**: LoginHandler, validate_credentials, generate_token

- **Path**: /src/auth/middleware.py
  **Score**: 0.88
  **Summary**: Authentication middleware that validates tokens on each request
  **Key Elements**: AuthMiddleware, verify_token
```
**Tokens**: ~85 tokens

**After (TOON format):**
```toon
{path|score|summary|elements}
/src/auth/login.py|0.95|Handles user authentication with JWT tokens, includes rate limiting and session management|LoginHandler, validate_credentials, generate_token
/src/auth/middleware.py|0.88|Authentication middleware that validates tokens on each request|AuthMiddleware, verify_token
```
**Tokens**: ~55 tokens

**Savings**: 35% fewer tokens, **100% of the information preserved**

## ⚠️ What Gets Optimized for Claude

When Gemini sends context to Claude:

### ✅ TOON Optimization (Default: Enabled)
- Removes repeated labels and markdown overhead
- Keeps **ALL** summaries, snippets, and descriptions fully intact
- **Lossless**: Zero information loss, just less fluff
- **Result**: 20-35% token savings, extremely detailed briefs to Claude

### ❌ LLMLingua Compression (Default: Disabled)
- Would remove actual content words and semantic information
- **Lossy**: Reduces detail and may miss nuances
- **Not used** for Gemini → Claude pipeline
- Available as opt-in tool for other use cases only

## Architecture

### Token Reduction Flow

```
┌─────────────────────────────────────────────────────────┐
│  User Query                                             │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  Context Gateway (Gemini)                               │
│  - Discovers files                                      │
│  - Searches docs                                        │
│  - Ranks relevance                                      │
│  - Prepares FULL context                                │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ ❌ NO COMPRESSION HERE
                   │ Claude gets full context
                   │
┌──────────────────▼──────────────────────────────────────┐
│  Claude (MCP Client)                                    │
│  - Deep reasoning with full details                     │
│  - Framework selection                                  │
│  - Response generation                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ ✅ OPTIONAL: Compress response
                   │ for storage or client delivery
                   │
┌──────────────────▼──────────────────────────────────────┐
│  Response (to user/storage)                             │
└─────────────────────────────────────────────────────────┘
```

## Usage

### 1. TOON Serialization (for structured data)

**Best for**: Arrays of uniform objects, API responses, data tables

```python
from app.core.token_reduction import serialize_to_toon, deserialize_from_toon

# Example: User list
users = [
    {"name": "Alice", "age": 30, "city": "NYC"},
    {"name": "Bob", "age": 25, "city": "SF"},
    {"name": "Charlie", "age": 35, "city": "LA"}
]

# Serialize to TOON (saves 40-60% tokens)
toon_str = serialize_to_toon(users)
# Output:
# {name|age|city}
# Alice|30|NYC
# Bob|25|SF
# Charlie|35|LA

# Deserialize back
original_users = deserialize_from_toon(toon_str)
```

### 2. LLMLingua-2 Compression (for natural language)

**Best for**: Long context passages, documentation snippets, secondary information

**⚠️ IMPORTANT**: Only use when losing some detail is acceptable!

```python
from app.core.token_reduction import compress_prompt

# Example: Compressing documentation
long_docs = """
[Large documentation text with examples and explanations...]
"""

# Compress by 50%
result = compress_prompt(long_docs, rate=0.5)
compressed = result["compressed_prompt"]

# Stats
print(f"Original: {result['origin_tokens']} tokens")
print(f"Compressed: {result['compressed_tokens']} tokens")
print(f"Saved: {result['origin_tokens'] - result['compressed_tokens']} tokens")
```

### 3. MCP Tools (for users)

Users can access token reduction via MCP tools:

```python
# Tool: serialize_to_toon
{
  "data": "[{\"name\": \"Alice\", \"age\": 30}, ...]"
}

# Tool: compress_prompt
{
  "prompt": "Long text to compress...",
  "rate": 0.5  # 50% compression
}

# Tool: token_reduction_compare
{
  "content": "Your content here"
}
# Returns comparison of JSON vs TOON vs LLMLingua
```

## Configuration

### Environment Variables

```bash
# TOON Settings
ENABLE_TOON_SERIALIZATION=true     # Enable TOON (default: true)
TOON_DELIMITER="|"                 # Field separator
TOON_ARRAY_THRESHOLD=2             # Min array size for tabular format

# LLMLingua-2 Settings
ENABLE_LLMLINGUA_COMPRESSION=false  # Enable LLMLingua (default: false)
LLMLINGUA_COMPRESSION_RATE=0.5      # Default compression rate
LLMLINGUA_DEVICE=cpu                # cpu, cuda, or mps
COMPRESSION_MIN_TOKENS=5000         # Only compress prompts > 5000 tokens

# Auto-compression (KEEP DISABLED)
AUTO_COMPRESS_PROMPTS=false         # DO NOT ENABLE - breaks Claude context
AUTO_COMPRESS_CONTEXT=false         # DO NOT ENABLE - breaks Claude context
```

### Python Settings

```python
from app.core.settings import get_settings

settings = get_settings()
settings.enable_toon_serialization = True
settings.enable_llmlingua_compression = False  # Keep disabled unless needed
```

## Performance Impact

### TOON Serialization
- **Overhead**: Minimal (<10ms for typical arrays)
- **Benefit**: 20-60% token reduction
- **Best case**: Uniform arrays of 100+ objects (60% reduction)
- **Worst case**: Mixed structures (falls back to JSON)

### LLMLingua-2 Compression
- **Overhead**: Significant (100-500ms first run, 50-200ms subsequent)
- **Benefit**: 50-80% token reduction
- **Accuracy**: 95%+ semantic preservation (but some detail loss)
- **Model**: ~400MB BERT model (downloaded once, cached)

## Real-World Examples

### ✅ Good Use Cases

**1. API Response Serialization**
```python
# Large user dataset for API response
users = fetch_users(limit=1000)  # 1000 user records
response = serialize_to_toon(users)
# Saves 40% tokens in API payload
```

**2. Caching Context**
```python
# Store context for later retrieval
context = prepare_context(query)
cached_context = serialize_to_toon(context.to_dict())
cache.set(key, cached_context)
# Saves storage space and retrieval bandwidth
```

**3. Documentation Snippets**
```python
# Compress large doc excerpts for secondary reference
docs = fetch_documentation(topic)
if len(docs) > 10000:
    compressed_docs = compress_prompt(docs, rate=0.6)
```

### ❌ Bad Use Cases

**1. Main Claude Prompts**
```python
# ❌ DON'T DO THIS
context = gemini_gateway.prepare_context(query)
compressed = compress_prompt(context.to_claude_prompt())  # WRONG!
claude_response = call_claude(compressed)  # Claude gets degraded context
```

**2. Critical Code Context**
```python
# ❌ DON'T DO THIS
code_context = read_file("critical_module.py")
compressed_code = compress_prompt(code_context)  # Loses important details!
analyze(compressed_code)  # Analysis will be poor
```

**3. Small Prompts**
```python
# ❌ DON'T DO THIS
small_prompt = "Fix the bug in function foo()"
compressed = compress_prompt(small_prompt)  # Overhead > benefit
```

## Installation

### Required Dependencies

```bash
# TOON: No extra dependencies (built-in)

# LLMLingua-2: Requires ML libraries
pip install llmlingua>=0.2.0
pip install transformers>=4.30.0
pip install torch>=2.0.0
```

The first time LLMLingua-2 is used, it will download the BERT model (~400MB) to `~/.cache/llmlingua/`.

## Testing

Run tests to verify token reduction:

```bash
# TOON tests
pytest tests/unit/test_toon.py -v

# Token reduction integration tests
pytest tests/unit/test_token_reduction.py -v
```

## Monitoring

Token reduction statistics are logged:

```python
# TOON statistics
logger.info("toon_serialization",
    original_tokens=1000,
    compressed_tokens=400,
    reduction_percent=60.0
)

# LLMLingua statistics
logger.info("prompt_compressed",
    original_tokens=5000,
    compressed_tokens=2500,
    ratio=0.5
)
```

## Summary

| Technology | Best For | Token Savings | Overhead | Risk |
|-----------|----------|---------------|----------|------|
| **TOON** | Structured data, arrays | 20-60% | <10ms | Low (lossless) |
| **LLMLingua-2** | Long text, docs | 50-80% | 50-500ms | Medium (lossy) |

### Key Principles

1. **Claude gets full context**: Never compress Gemini → Claude pipeline
2. **Opt-in, not automatic**: Use compression intentionally, not by default
3. **Measure impact**: Always verify compression doesn't hurt quality
4. **Use TOON for data**: Structured data = TOON, not compression
5. **Reserve LLMLingua for secondary content**: Only compress nice-to-have info

## References

- [TOON Format Specification](https://scalevise.com/resources/toon-lightweight-json-for-ai-llm-systems/)
- [LLMLingua-2 Paper](https://llmlingua.com/llmlingua2.html)
- [Microsoft LLMLingua GitHub](https://github.com/microsoft/LLMLingua)
