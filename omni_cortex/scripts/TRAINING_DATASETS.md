# LLM Training Data Integration

Enrich Omni-Cortex with curated debugging, reasoning, and instruction datasets from HuggingFace Hub.

## 📊 Overview

Omni-Cortex can ingest **16,000+ training examples** across three categories:
- **Debugging**: Bug-fix pairs with error messages
- **Reasoning**: Chain-of-thought and step-by-step examples
- **Instructions**: Task completion and code generation examples

All data is stored as **vector embeddings** in ChromaDB for instant semantic search.

---

## 🐛 Debugging Datasets (10K+ examples)

### From GitHub (Local Ingestion)

#### 1. PyResBugs (5,007 pairs)
- **Source**: [dessertlab/PyResBugs](https://github.com/dessertlab/PyResBugs)
- **Description**: Production bugs from real Python projects
- **Format**: JSON with buggy code, fixed code, descriptions
- **Focus**: Residual bugs that passed testing

#### 2. HaPy-Bug (793 pairs)
- **Description**: Expert-annotated bug-fix commits
- **Quality**: Line-level annotations by domain experts
- **Format**: High-quality manual annotations

#### 3. Learning-Fixes (Varies)
- **Format**: Line-aligned bug-fix pairs
- **Splits**: Train/validation/test
- **Source**: [Learning-Fixes Site](https://sites.google.com/view/learning-fixes/data)

### From HuggingFace Hub (Direct Loading)

#### 4. Muennighoff/python-bugs (1-10K pairs)
- **HF**: [Muennighoff/python-bugs](https://huggingface.co/datasets/Muennighoff/python-bugs)
- **Size**: ~2.85 MB (small, curated)
- **Format**: JSON with bug types
- **Ingestion**: `--dataset python-bugs`

#### 5. alexjercan/bugnet (Medium)
- **HF**: [alexjercan/bugnet](https://huggingface.co/datasets/alexjercan/bugnet)
- **Description**: CodeNet competition bugs
- **Features**: Error messages, stderr output, change types
- **Ingestion**: `--dataset bugnet`

#### 6. HuggingFaceH4/Code-Feedback (Medium)
- **HF**: [HuggingFaceH4/Code-Feedback](https://huggingface.co/datasets/HuggingFaceH4/Code-Feedback)
- **Description**: Code review and feedback patterns
- **Ingestion**: `--dataset code-feedback`

---

## 🧠 Reasoning Datasets (6K+ examples)

#### 1. moremilk/General_Inquiry_Thinking-Chain-Of-Thought (6K)
- **HF**: [moremilk/General_Inquiry_Thinking-Chain-Of-Thought](https://huggingface.co/datasets/moremilk/General_Inquiry_Thinking-Chain-Of-Thought)
- **Description**: 6,000 Q&A pairs with full chain-of-thought reasoning
- **Format**: Question → Thinking process → Answer
- **Focus**: Step-by-step logic and deduction
- **Ingestion**: `--dataset chain-of-thought`

#### 2. AlekseyKorshuk/chain-of-thoughts-chatml (Varies)
- **HF**: [AlekseyKorshuk/chain-of-thoughts-chatml](https://huggingface.co/datasets/AlekseyKorshuk/chain-of-thoughts-chatml)
- **Format**: ChatML with reasoning steps
- **Ingestion**: `--dataset cot-chatml`

---

## 📝 Instruction Datasets (1.6M+ examples)

#### 1. nampdn-ai/tiny-codes (1.6M snippets, sampled to 10K)
- **HF**: [nampdn-ai/tiny-codes](https://huggingface.co/datasets/nampdn-ai/tiny-codes)
- **Description**: Synthetic, high-quality commented code
- **Languages**: Python, TypeScript, JavaScript, Ruby, Julia, Rust, C++, Bash, Java, C#, Go
- **Quality**: Textbook-style with clear comments
- **Ingestion**: `--dataset tiny-codes` (auto-samples 10K)

#### 2. HuggingFaceH4/helpful_instructions (Medium)
- **HF**: [HuggingFaceH4/helpful_instructions](https://huggingface.co/datasets/HuggingFaceH4/helpful_instructions)
- **Description**: Instruction-completion pairs for AI assistants
- **Format**: (instruction, completion) tuples
- **Ingestion**: `--dataset helpful-instructions`

---

## ⚙️ Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `datasets>=2.15.0` - HuggingFace datasets library
- `pandas>=2.0.0` - For CSV parsing

### 2. API Key (REQUIRED)
Embeddings require an API key for vector generation:

```bash
# In your .env file
OPENAI_API_KEY="sk-..."
# OR
OPENROUTER_API_KEY="sk-or-..."
```

**Without an API key, ingestion will not work.**

---

## 🚀 Usage

### Quick Start: Ingest Everything
```bash
python3 scripts/ingest_training_data.py --all
```

This ingests ALL datasets from HuggingFace Hub (~16K examples).

### Ingest by Category

**Debugging only** (6 datasets):
```bash
python3 scripts/ingest_training_data.py --category debugging
```

**Reasoning only** (2 datasets):
```bash
python3 scripts/ingest_training_data.py --category reasoning
```

**Instructions only** (2 datasets):
```bash
python3 scripts/ingest_training_data.py --category instruction
```

### Ingest Specific Dataset
```bash
# Single dataset
python3 scripts/ingest_training_data.py --dataset python-bugs
python3 scripts/ingest_training_data.py --dataset chain-of-thought
python3 scripts/ingest_training_data.py --dataset tiny-codes
```

### List Available Datasets
```bash
python3 scripts/ingest_training_data.py --list
```

### Legacy: GitHub Datasets (Bug-Fix Pairs)
For the original GitHub-based datasets (PyResBugs, HaPy-Bug, Learning-Fixes):
```bash
python3 scripts/ingest_bug_fixes.py --all
```

---

## 💾 What Gets Stored

### Debugging Examples
```
Bug Description: [Error message or description]

Buggy Code:
```python
[Original buggy code]
```

Fixed Code:
```python
[Corrected code]
```

Metadata: bug_type, language, source, timestamp
```

### Reasoning Examples
```
Question: [The problem or question]

Reasoning Process:
[Step-by-step chain-of-thought]

Answer: [Final answer]

Metadata: reasoning_type, source, timestamp
```

### Instruction Examples
```
Instruction: [The task or instruction]

Response:
```python
[Code solution]
```

Metadata: task_type, language, source, timestamp
```

---

## 🔍 Querying the Knowledge Base

### From Python (MCP Server)
```python
from app.collection_manager import get_collection_manager

cm = get_collection_manager()

# Search debugging knowledge
debug_results = cm.search_debugging_knowledge(
    query="AttributeError: 'NoneType' object has no attribute 'get'",
    k=5,
    bug_type="AttributeError"
)

# Search reasoning patterns
reasoning_results = cm.search_reasoning_knowledge(
    query="How to solve this problem step by step?",
    k=3
)

# Search instruction examples
instruction_results = cm.search_instruction_knowledge(
    query="Write a function to parse JSON",
    k=5,
    task_type="code_generation"
)
```

### From Framework Nodes
All knowledge is automatically available through RAG in reasoning frameworks.

---

## 📊 Expected Output

```
[INFO] ============================================================
[INFO] Dataset: python-bugs
[INFO] HuggingFace: Muennighoff/python-bugs
[INFO] Description: Curated Python bug collection (1-10K pairs)
[INFO] Size: small
[INFO] ============================================================
[INFO] Loading Muennighoff/python-bugs from HuggingFace Hub...
[INFO] ✓ Loaded 2847 examples
[INFO] ✓ Parsed 2847 examples
[INFO] Ingesting 2847 debugging examples from python-bugs...
[INFO] Progress: 100/2847 (3%)
[INFO] Progress: 200/2847 (7%)
...
[INFO] Progress: 2847/2847 (100%)
[INFO] ✅ Ingested 2847 debugging examples!
```

---

## 💰 Performance & Cost Estimates

| Category | Datasets | Examples | Time* | Cost** |
|----------|---------|----------|-------|--------|
| **Debugging (HF)** | 3 | ~5K | ~15 min | ~$0.60 |
| **Debugging (GitHub)** | 3 | ~6K | ~20 min | ~$0.80 |
| **Reasoning** | 2 | ~6K | ~15 min | ~$0.70 |
| **Instructions (sampled)** | 2 | ~12K | ~30 min | ~$1.40 |
| **Total (HF only)** | **7** | **~23K** | **~60 min** | **~$2.70** |
| **Total (All)** | **10** | **~29K** | **~80 min** | **~$3.50** |

\* Using OpenAI text-embedding-3-small
\** Approximate, based on OpenAI pricing ($0.02/1M tokens)

---

## 🗄️ Data Persistence

All embeddings are stored in ChromaDB at:
```
/app/data/chroma/omni-cortex-{collection_name}/
```

Collections:
- `omni-cortex-debugging_knowledge`
- `omni-cortex-reasoning_knowledge`
- `omni-cortex-instruction_knowledge`

These directories are mounted as Docker volumes, so data persists across container restarts.

---

## 🔧 Troubleshooting

### "No API key found"
Set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in your `.env` file.

### "datasets library not installed"
```bash
pip install datasets>=2.15.0
```

### "Failed to load dataset"
- Check your internet connection
- Verify the dataset exists on HuggingFace Hub
- Try a different dataset

### "No examples extracted"
The dataset schema may have changed. Check the parser function in `ingest_training_data.py` and adjust field names.

### Rate limiting
If you hit API rate limits:
1. Reduce batch size in the script (default: 100)
2. Add delays between batches
3. Use a higher-tier API plan

---

## 📚 Collections Summary

| Collection | Description | Source | Size |
|------------|-------------|--------|------|
| **debugging_knowledge** | Bug-fix pairs, error patterns | 6 HF datasets + 3 GitHub | ~10K |
| **reasoning_knowledge** | Chain-of-thought examples | 2 HF datasets | ~6K |
| **instruction_knowledge** | Task completion examples | 2 HF datasets | ~12K (sampled) |
| **frameworks** | Omni-Cortex framework code | Codebase ingestion | Varies |
| **documentation** | Markdown docs, READMEs | Codebase ingestion | Varies |
| **learnings** | Successful solutions | Runtime learning | Grows over time |

---

## 🎯 Next Steps

After ingestion:
1. **Restart MCP server** to load new embeddings
2. **Test queries** using the Python API
3. **Use in frameworks** - debugging knowledge auto-enriches Active Inference, Chain of Verification, etc.
4. **Monitor quality** - Check if retrieved examples are relevant

---

## 📖 References

**Debugging Datasets:**
- [PyResBugs Paper](https://arxiv.org/html/2505.05777v1)
- [HaPy-Bug Paper](https://arxiv.org/html/2504.04810v1)
- [Learning-Fixes Site](https://sites.google.com/view/learning-fixes/data)
- [Muennighoff/python-bugs](https://huggingface.co/datasets/Muennighoff/python-bugs)
- [alexjercan/bugnet](https://huggingface.co/datasets/alexjercan/bugnet)
- [HuggingFaceH4/Code-Feedback](https://huggingface.co/datasets/HuggingFaceH4/Code-Feedback)

**Reasoning Datasets:**
- [moremilk/General_Inquiry_Thinking-Chain-Of-Thought](https://huggingface.co/datasets/moremilk/General_Inquiry_Thinking-Chain-Of-Thought)
- [AlekseyKorshuk/chain-of-thoughts-chatml](https://huggingface.co/datasets/AlekseyKorshuk/chain-of-thoughts-chatml)

**Instruction Datasets:**
- [nampdn-ai/tiny-codes](https://huggingface.co/datasets/nampdn-ai/tiny-codes)
- [HuggingFaceH4/helpful_instructions](https://huggingface.co/datasets/HuggingFaceH4/helpful_instructions)
