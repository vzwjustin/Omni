# Bug-Fix Dataset Ingestion

Enrich Omni-Cortex with curated debugging knowledge from real-world bug-fix pairs.

## Supported Datasets

### 1. PyResBugs (5,007 bug-fix pairs)
- **Source**: [dessertlab/PyResBugs](https://github.com/dessertlab/PyResBugs)
- **Size**: ~5K pairs (manageable)
- **Format**: JSON with buggy code, fixed code, and descriptions
- **Quality**: Curated from real Python projects
- **Focus**: Residual bugs (production bugs that passed testing)

### 2. HaPy-Bug (793 bug-fix pairs)
- **Size**: ~800 pairs (small, high-quality)
- **Format**: Expert-annotated commits
- **Quality**: Line-level annotations by domain experts
- **Focus**: High-quality manual annotations

### 3. Learning-Fixes
- **Format**: Line-aligned bug-fix pairs
- **Splits**: Train/validation/test
- **Focus**: ML-ready bug-fix patterns

## Prerequisites

1. **API Key Required** (for embeddings):
   ```bash
   export OPENAI_API_KEY="sk-..."
   # OR
   export OPENROUTER_API_KEY="sk-or-..."
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Git** (for cloning datasets):
   ```bash
   git --version  # Ensure git is installed
   ```

## Usage

### Ingest All Datasets (Recommended)
```bash
cd /app  # Inside Docker container
python3 scripts/ingest_bug_fixes.py --all
```

This will:
- Clone PyResBugs, HaPy-Bug, and Learning-Fixes
- Parse all bug-fix pairs
- Generate embeddings
- Store in ChromaDB's `debugging_knowledge` collection

### Ingest Specific Dataset
```bash
# PyResBugs only
python3 scripts/ingest_bug_fixes.py --dataset pyresbugs

# HaPy-Bug only
python3 scripts/ingest_bug_fixes.py --dataset hapybug

# Learning-Fixes only
python3 scripts/ingest_bug_fixes.py --dataset learningfixes
```

### Use Local Dataset
If you've already downloaded a dataset:

```bash
python3 scripts/ingest_bug_fixes.py \
  --dataset pyresbugs \
  --local-path /path/to/PyResBugs
```

### Custom Repository URL
```bash
python3 scripts/ingest_bug_fixes.py \
  --dataset pyresbugs \
  --url https://github.com/your-fork/PyResBugs.git
```

## What Gets Stored

Each bug-fix pair is stored with:

**Text Content**:
```
Bug Description: [Natural language description]

Buggy Code:
```python
[Original buggy code]
```

Fixed Code:
```python
[Corrected code]
```

Fix Pattern: [Extracted pattern]
```

**Metadata**:
- `source`: Dataset name (pyresbugs/hapybug/learningfixes)
- `bug_type`: Error type (TypeError, AttributeError, etc.)
- `language`: Programming language (python)
- `category`: "debugging"
- `annotation_quality`: "expert" or "standard"
- `timestamp`: Ingestion timestamp

## Querying Debugging Knowledge

### From Python (MCP Server)
```python
from app.collection_manager import get_collection_manager

cm = get_collection_manager()

# Search for similar bugs
results = cm.search_debugging_knowledge(
    query="AttributeError: 'NoneType' object has no attribute 'get'",
    k=5,
    bug_type="AttributeError"
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

### From Framework Nodes
The debugging knowledge is automatically available to all reasoning frameworks through RAG.

## Expected Output

```
[INFO] Cloning https://github.com/dessertlab/PyResBugs.git...
[INFO] ✓ Cloned to /tmp/tmpXXX/pyresbugs
[INFO] Parsing pyresbugs...
[INFO] Found 127 JSON files in /tmp/tmpXXX/pyresbugs/bugs
[INFO] ✓ Parsed 5,007 bug-fix pairs
[INFO] Progress: 100/5007 (1%)
[INFO] Progress: 200/5007 (3%)
...
[INFO] Progress: 5007/5007 (100%)
[INFO] ✅ Successfully ingested 5,007 bug-fix pairs from pyresbugs!
```

## Data Persistence

All bug-fix pairs are stored in ChromaDB at:
```
/app/data/chroma/omni-cortex-debugging_knowledge/
```

This directory is mounted as a Docker volume, so data persists across container restarts.

## Troubleshooting

### "No API key found"
Ensure you've set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in your `.env` file or environment.

### "Failed to clone"
- Check your internet connection
- Verify the repository URL is accessible
- Try using `--local-path` with a manually downloaded dataset

### "No bug-fix pairs found"
The dataset structure may have changed. Check:
1. The repository's actual file structure
2. Update the parser functions in `ingest_bug_fixes.py`

### "Batch insert failed"
- Embedding API may be rate-limited
- Check your API key quota
- Try reducing batch size in the script (default: 100)

## Performance

| Dataset | Size | Est. Time* | Cost** |
|---------|------|-----------|--------|
| PyResBugs | 5,007 | ~15 min | ~$0.50 |
| HaPy-Bug | 793 | ~2 min | ~$0.08 |
| Learning-Fixes | Varies | ~5 min | ~$0.20 |
| **Total** | **~6K** | **~22 min** | **~$0.78** |

\* Using OpenAI text-embedding-3-small
\** Approximate, based on OpenAI pricing

## Next Steps

After ingestion, your MCP server can:
1. **Auto-suggest fixes** for runtime errors
2. **Provide debugging context** from similar past bugs
3. **Learn from patterns** in real-world bug fixes

## References

- [PyResBugs Paper](https://arxiv.org/html/2505.05777v1)
- [HaPy-Bug Paper](https://arxiv.org/html/2504.04810v1)
- [Learning-Fixes Site](https://sites.google.com/view/learning-fixes/data)
