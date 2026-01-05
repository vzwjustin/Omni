#!/usr/bin/env python3
"""
LLM Training Data Ingestion for Omni-Cortex
===========================================

Ingests debugging, reasoning, and instruction datasets from HuggingFace Hub
into ChromaDB collections for enhanced RAG capabilities.

Supported Categories:
- DEBUGGING: Bug-fix pairs, error handling patterns
- REASONING: Chain-of-thought, step-by-step examples
- INSTRUCTIONS: Task completion, code generation examples

Usage:
    # Ingest all datasets
    python3 scripts/ingest_training_data.py --all

    # Ingest by category
    python3 scripts/ingest_training_data.py --category debugging
    python3 scripts/ingest_training_data.py --category reasoning
    python3 scripts/ingest_training_data.py --category instructions

    # Ingest specific dataset
    python3 scripts/ingest_training_data.py --dataset python-bugs
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Add app to path
sys.path.append(os.getcwd())

from app.collection_manager import get_collection_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ingest_training_data")


# Dataset catalog with metadata
DATASETS = {
    # ===== DEBUGGING DATASETS =====
    "python-bugs": {
        "hf_name": "Muennighoff/python-bugs",
        "category": "debugging",
        "collection": "debugging_knowledge",
        "description": "Curated Python bug collection (1-10K pairs)",
        "size": "small",
        "parser": "parse_python_bugs"
    },
    "bugnet": {
        "hf_name": "alexjercan/bugnet",
        "category": "debugging",
        "collection": "debugging_knowledge",
        "description": "CodeNet competition bugs with error messages",
        "size": "medium",
        "parser": "parse_bugnet"
    },
    "code-feedback": {
        "hf_name": "HuggingFaceH4/Code-Feedback",
        "category": "debugging",
        "collection": "debugging_knowledge",
        "description": "Code review and feedback patterns",
        "size": "medium",
        "parser": "parse_code_feedback"
    },

    # ===== REASONING DATASETS =====
    "chain-of-thought": {
        "hf_name": "moremilk/General_Inquiry_Thinking-Chain-Of-Thought",
        "category": "reasoning",
        "collection": "reasoning_knowledge",
        "description": "6K Q&A pairs with step-by-step reasoning",
        "size": "small",
        "parser": "parse_cot_general"
    },
    "cot-chatml": {
        "hf_name": "AlekseyKorshuk/chain-of-thoughts-chatml",
        "category": "reasoning",
        "collection": "reasoning_knowledge",
        "description": "Chain-of-thought in ChatML format",
        "size": "small",
        "parser": "parse_cot_chatml"
    },

    # ===== INSTRUCTION DATASETS =====
    "tiny-codes": {
        "hf_name": "nampdn-ai/tiny-codes",
        "category": "instruction",
        "collection": "instruction_knowledge",
        "description": "1.6M high-quality commented code snippets",
        "size": "large",
        "parser": "parse_tiny_codes",
        "max_samples": 10000  # Limit for large datasets
    },
    "helpful-instructions": {
        "hf_name": "HuggingFaceH4/helpful_instructions",
        "category": "instruction",
        "collection": "instruction_knowledge",
        "description": "Instruction-completion pairs for assistants",
        "size": "medium",
        "parser": "parse_helpful_instructions"
    }
}


def load_hf_dataset(hf_name: str, max_samples: Optional[int] = None):
    """Load dataset from HuggingFace Hub with optional sampling."""
    try:
        from datasets import load_dataset

        logger.info(f"Loading {hf_name} from HuggingFace Hub...")

        # Load dataset
        dataset = load_dataset(hf_name, split="train", streaming=False)

        # Sample if needed
        if max_samples and len(dataset) > max_samples:
            logger.info(f"Sampling {max_samples} from {len(dataset)} total examples")
            dataset = dataset.shuffle(seed=42).select(range(max_samples))

        logger.info(f"✓ Loaded {len(dataset)} examples")
        return dataset

    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        return None
    except Exception as e:
        logger.error(f"Failed to load {hf_name}: {e}")
        return None


# ===== PARSERS FOR EACH DATASET =====

def parse_python_bugs(dataset) -> List[Dict[str, str]]:
    """Parse Muennighoff/python-bugs dataset."""
    examples = []
    for item in dataset:
        try:
            # Adapt to actual schema (may need adjustment)
            buggy = item.get("buggy_code") or item.get("incorrect") or ""
            fixed = item.get("fixed_code") or item.get("correct") or ""
            desc = item.get("description") or item.get("error_message") or ""

            if buggy and fixed:
                examples.append({
                    "buggy_code": buggy,
                    "fixed_code": fixed,
                    "description": desc,
                    "bug_type": item.get("bug_type", "unknown"),
                    "source": "python-bugs",
                    "language": "python"
                })
        except Exception as e:
            logger.debug(f"Skipped item: {e}")
            continue

    return examples


def parse_bugnet(dataset) -> List[Dict[str, str]]:
    """Parse alexjercan/bugnet dataset."""
    examples = []
    for item in dataset:
        try:
            fail_code = item.get("fail", "")
            pass_code = item.get("pass", "")
            error_msg = item.get("stderr", "") or item.get("error_message", "")

            if fail_code and pass_code:
                examples.append({
                    "buggy_code": fail_code,
                    "fixed_code": pass_code,
                    "description": f"Competition bug. Error: {error_msg}",
                    "bug_type": item.get("change_type", "unknown"),
                    "source": "bugnet",
                    "language": item.get("language", "python")
                })
        except Exception as e:
            logger.debug(f"Skipped item: {e}")
            continue

    return examples


def parse_code_feedback(dataset) -> List[Dict[str, str]]:
    """Parse HuggingFaceH4/Code-Feedback dataset."""
    examples = []
    for item in dataset:
        try:
            code = item.get("code") or item.get("prompt", "")
            feedback = item.get("feedback") or item.get("completion", "")

            if code and feedback:
                examples.append({
                    "buggy_code": code,
                    "fixed_code": feedback,  # Or improved version
                    "description": "Code review feedback",
                    "bug_type": "review",
                    "source": "code-feedback",
                    "language": "python"
                })
        except Exception as e:
            logger.debug(f"Skipped item: {e}")
            continue

    return examples


def parse_cot_general(dataset) -> List[Dict[str, str]]:
    """Parse moremilk/General_Inquiry_Thinking-Chain-Of-Thought."""
    examples = []
    for item in dataset:
        try:
            question = item.get("question") or item.get("prompt", "")
            thinking = item.get("thinking") or item.get("chain_of_thought", "")
            answer = item.get("answer") or item.get("response", "")

            if question and thinking:
                examples.append({
                    "question": question,
                    "reasoning": thinking,
                    "answer": answer,
                    "reasoning_type": "chain-of-thought",
                    "source": "cot-general",
                    "category": "reasoning"
                })
        except Exception as e:
            logger.debug(f"Skipped item: {e}")
            continue

    return examples


def parse_cot_chatml(dataset) -> List[Dict[str, str]]:
    """Parse AlekseyKorshuk/chain-of-thoughts-chatml."""
    examples = []
    for item in dataset:
        try:
            # ChatML format typically has messages array
            messages = item.get("messages", [])
            if not messages:
                continue

            question = ""
            reasoning = ""

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "user":
                    question = content
                elif role == "assistant":
                    reasoning = content

            if question and reasoning:
                examples.append({
                    "question": question,
                    "reasoning": reasoning,
                    "answer": "",  # Reasoning includes answer
                    "reasoning_type": "chain-of-thought",
                    "source": "cot-chatml",
                    "category": "reasoning"
                })
        except Exception as e:
            logger.debug(f"Skipped item: {e}")
            continue

    return examples


def parse_tiny_codes(dataset) -> List[Dict[str, str]]:
    """Parse nampdn-ai/tiny-codes."""
    examples = []
    for item in dataset:
        try:
            code = item.get("code") or item.get("content", "")
            language = item.get("language") or item.get("lang", "python")
            prompt = item.get("prompt") or item.get("description", "")

            if code:
                examples.append({
                    "instruction": prompt or f"Write {language} code",
                    "response": code,
                    "language": language,
                    "task_type": "code_generation",
                    "source": "tiny-codes",
                    "category": "instruction"
                })
        except Exception as e:
            logger.debug(f"Skipped item: {e}")
            continue

    return examples


def parse_helpful_instructions(dataset) -> List[Dict[str, str]]:
    """Parse HuggingFaceH4/helpful_instructions."""
    examples = []
    for item in dataset:
        try:
            instruction = item.get("instruction") or item.get("prompt", "")
            completion = item.get("completion") or item.get("response", "")

            if instruction and completion:
                examples.append({
                    "instruction": instruction,
                    "response": completion,
                    "language": "python",  # Assume Python for code tasks
                    "task_type": "general",
                    "source": "helpful-instructions",
                    "category": "instruction"
                })
        except Exception as e:
            logger.debug(f"Skipped item: {e}")
            continue

    return examples


def ingest_debugging_examples(examples: List[Dict[str, str]], source: str):
    """Ingest debugging examples into debugging_knowledge collection."""
    cm = get_collection_manager()

    texts = []
    metadatas = []
    batch_size = 100
    total = len(examples)
    ingested = 0

    logger.info(f"Ingesting {total} debugging examples from {source}...")

    for i, ex in enumerate(examples):
        combined_text = f"""
Bug Description: {ex.get('description', 'No description')}

Buggy Code:
```{ex.get('language', 'python')}
{ex.get('buggy_code', '')[:2000]}
```

Fixed Code:
```{ex.get('language', 'python')}
{ex.get('fixed_code', '')[:2000]}
```

Source: {ex.get('source', source)}
"""

        metadata = {
            "source": ex.get("source", source),
            "bug_type": ex.get("bug_type", "unknown"),
            "language": ex.get("language", "python"),
            "category": "debugging",
            "timestamp": datetime.utcnow().isoformat()
        }

        texts.append(combined_text)
        metadatas.append(metadata)

        if len(texts) >= batch_size or i == total - 1:
            count = cm.add_documents(texts, metadatas, "debugging_knowledge")
            ingested += count
            logger.info(f"Progress: {ingested}/{total} ({ingested*100//total}%)")
            texts = []
            metadatas = []

    logger.info(f"✅ Ingested {ingested} debugging examples!")


def ingest_reasoning_examples(examples: List[Dict[str, str]], source: str):
    """Ingest reasoning examples into reasoning_knowledge collection."""
    cm = get_collection_manager()

    texts = []
    metadatas = []
    batch_size = 100
    total = len(examples)
    ingested = 0

    logger.info(f"Ingesting {total} reasoning examples from {source}...")

    for i, ex in enumerate(examples):
        combined_text = f"""
Question: {ex.get('question', '')}

Reasoning Process:
{ex.get('reasoning', '')}

Answer: {ex.get('answer', '')}

Source: {ex.get('source', source)}
"""

        metadata = {
            "source": ex.get("source", source),
            "reasoning_type": ex.get("reasoning_type", "chain-of-thought"),
            "category": "reasoning",
            "timestamp": datetime.utcnow().isoformat()
        }

        texts.append(combined_text)
        metadatas.append(metadata)

        if len(texts) >= batch_size or i == total - 1:
            count = cm.add_documents(texts, metadatas, "reasoning_knowledge")
            ingested += count
            logger.info(f"Progress: {ingested}/{total} ({ingested*100//total}%)")
            texts = []
            metadatas = []

    logger.info(f"✅ Ingested {ingested} reasoning examples!")


def ingest_instruction_examples(examples: List[Dict[str, str]], source: str):
    """Ingest instruction examples into instruction_knowledge collection."""
    cm = get_collection_manager()

    texts = []
    metadatas = []
    batch_size = 100
    total = len(examples)
    ingested = 0

    logger.info(f"Ingesting {total} instruction examples from {source}...")

    for i, ex in enumerate(examples):
        combined_text = f"""
Instruction: {ex.get('instruction', '')}

Response:
```{ex.get('language', 'python')}
{ex.get('response', '')[:2000]}
```

Task Type: {ex.get('task_type', 'general')}
Source: {ex.get('source', source)}
"""

        metadata = {
            "source": ex.get("source", source),
            "task_type": ex.get("task_type", "general"),
            "language": ex.get("language", "python"),
            "category": "instruction",
            "timestamp": datetime.utcnow().isoformat()
        }

        texts.append(combined_text)
        metadatas.append(metadata)

        if len(texts) >= batch_size or i == total - 1:
            count = cm.add_documents(texts, metadatas, "instruction_knowledge")
            ingested += count
            logger.info(f"Progress: {ingested}/{total} ({ingested*100//total}%)")
            texts = []
            metadatas = []

    logger.info(f"✅ Ingested {ingested} instruction examples!")


def ingest_dataset(dataset_key: str):
    """Ingest a specific dataset."""
    if dataset_key not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_key}")
        logger.info(f"Available datasets: {', '.join(DATASETS.keys())}")
        return

    config = DATASETS[dataset_key]
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {dataset_key}")
    logger.info(f"HuggingFace: {config['hf_name']}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Size: {config['size']}")
    logger.info(f"{'='*60}")

    # Load dataset
    max_samples = config.get("max_samples")
    dataset = load_hf_dataset(config["hf_name"], max_samples)

    if not dataset:
        return

    # Parse dataset
    parser_name = config["parser"]
    parser_func = globals().get(parser_name)

    if not parser_func:
        logger.error(f"Parser function not found: {parser_name}")
        return

    examples = parser_func(dataset)
    logger.info(f"✓ Parsed {len(examples)} examples")

    if not examples:
        logger.warning("No examples extracted from dataset")
        return

    # Ingest based on category
    category = config["category"]
    if category == "debugging":
        ingest_debugging_examples(examples, dataset_key)
    elif category == "reasoning":
        ingest_reasoning_examples(examples, dataset_key)
    elif category == "instruction":
        ingest_instruction_examples(examples, dataset_key)


def main():
    parser = argparse.ArgumentParser(description="Ingest LLM training data from HuggingFace")

    parser.add_argument("--all", action="store_true", help="Ingest all datasets")
    parser.add_argument("--category", choices=["debugging", "reasoning", "instruction"],
                        help="Ingest all datasets in a category")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Ingest a specific dataset")
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    # Check for API keys
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("⚠️  No API key found (OPENAI_API_KEY or OPENROUTER_API_KEY)")
        logger.warning("Embeddings require an API key. Set it in your .env file.")
        return

    # List datasets
    if args.list:
        logger.info("\n📚 Available Datasets:\n")
        for key, config in DATASETS.items():
            logger.info(f"  {key:<20} - {config['description']}")
            logger.info(f"  {'':20}   Category: {config['category']}, Size: {config['size']}")
            logger.info("")
        return

    # Ingest datasets
    if args.all:
        logger.info("Starting ingestion of all datasets...")
        for key in DATASETS.keys():
            ingest_dataset(key)
    elif args.category:
        logger.info(f"Ingesting all {args.category} datasets...")
        for key, config in DATASETS.items():
            if config["category"] == args.category:
                ingest_dataset(key)
    elif args.dataset:
        ingest_dataset(args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
