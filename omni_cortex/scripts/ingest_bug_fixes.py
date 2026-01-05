#!/usr/bin/env python3
"""
Bug-Fix Dataset Ingestion for Omni-Cortex
==========================================

Ingests curated bug-fix pairs from multiple sources into the debugging_knowledge collection.

Supported Datasets:
- PyResBugs: 5,007 Python bugs with fixes and descriptions
- HaPy-Bug: 793 expert-annotated bug-fix commits
- Learning-Fixes: Line-aligned bug-fix pairs

Usage:
    python3 scripts/ingest_bug_fixes.py --all
    python3 scripts/ingest_bug_fixes.py --dataset pyresbugs
    python3 scripts/ingest_bug_fixes.py --dataset hapybug
    python3 scripts/ingest_bug_fixes.py --dataset learningfixes
"""

import argparse
import asyncio
import logging
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add app to path
sys.path.append(os.getcwd())

from app.collection_manager import get_collection_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ingest_bug_fixes")


async def download_git_repo(url: str, target_dir: Path) -> bool:
    """Clone a git repository."""
    try:
        import subprocess
        logger.info(f"Cloning {url}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target_dir)],
            check=True,
            capture_output=True
        )
        logger.info(f"✓ Cloned to {target_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to clone {url}: {e}")
        return False


def parse_pyresbugs(dataset_dir: Path) -> List[Dict[str, str]]:
    """
    Parse PyResBugs dataset.

    Expected structure:
    - bugs/ directory with JSON files
    - Each file contains bug metadata, buggy code, and fixed code
    """
    bug_fixes = []

    # Look for JSON files in common locations
    possible_paths = [
        dataset_dir / "bugs",
        dataset_dir / "data",
        dataset_dir / "dataset",
        dataset_dir
    ]

    for base_path in possible_paths:
        if not base_path.exists():
            continue

        json_files = list(base_path.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {base_path}")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, list):
                    for item in data:
                        bug_fix = extract_pyresbugs_item(item)
                        if bug_fix:
                            bug_fixes.append(bug_fix)
                else:
                    bug_fix = extract_pyresbugs_item(data)
                    if bug_fix:
                        bug_fixes.append(bug_fix)

            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")
                continue

    return bug_fixes


def extract_pyresbugs_item(item: Dict) -> Optional[Dict[str, str]]:
    """Extract bug-fix pair from PyResBugs item."""
    try:
        # Adapt to actual JSON structure (may need adjustment)
        buggy_code = item.get("buggy_code") or item.get("before") or item.get("bug")
        fixed_code = item.get("fixed_code") or item.get("after") or item.get("fix")
        description = item.get("description") or item.get("commit_message") or ""
        bug_type = item.get("bug_type") or item.get("category") or "unknown"

        if not buggy_code or not fixed_code:
            return None

        return {
            "buggy_code": buggy_code,
            "fixed_code": fixed_code,
            "description": description,
            "bug_type": bug_type,
            "source": "pyresbugs",
            "language": "python"
        }
    except Exception as e:
        logger.debug(f"Failed to extract item: {e}")
        return None


def parse_hapybug(dataset_dir: Path) -> List[Dict[str, str]]:
    """
    Parse HaPy-Bug dataset.

    Expected structure:
    - CSV or JSON files with bug-fix commits
    - Line-level annotations
    """
    bug_fixes = []

    # Look for CSV files
    csv_files = list(dataset_dir.rglob("*.csv"))
    if csv_files:
        try:
            import pandas as pd
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                logger.info(f"Found {len(df)} entries in {csv_file.name}")

                for _, row in df.iterrows():
                    bug_fix = {
                        "buggy_code": str(row.get("buggy_code", row.get("before", ""))),
                        "fixed_code": str(row.get("fixed_code", row.get("after", ""))),
                        "description": str(row.get("description", row.get("message", ""))),
                        "bug_type": str(row.get("bug_type", "unknown")),
                        "source": "hapybug",
                        "language": "python",
                        "annotation_quality": "expert"  # HaPy-Bug is expert-annotated
                    }
                    if bug_fix["buggy_code"] and bug_fix["fixed_code"]:
                        bug_fixes.append(bug_fix)
        except ImportError:
            logger.warning("pandas not installed, skipping CSV parsing")
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")

    # Also check for JSON
    json_files = list(dataset_dir.rglob("*.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        bug_fix = extract_pyresbugs_item(item)  # Similar structure
                        if bug_fix:
                            bug_fix["source"] = "hapybug"
                            bug_fix["annotation_quality"] = "expert"
                            bug_fixes.append(bug_fix)
        except Exception as e:
            logger.warning(f"Failed to parse JSON {json_file}: {e}")

    return bug_fixes


def parse_learningfixes(dataset_dir: Path) -> List[Dict[str, str]]:
    """
    Parse Learning-Fixes dataset.

    Expected structure:
    - train/val/test splits
    - Line-aligned bug-fix pairs
    """
    bug_fixes = []

    # Look for split directories
    splits = ["train", "val", "test", "validation"]

    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        logger.info(f"Processing {split} split...")

        # Look for paired files (buggy/fixed)
        buggy_files = list(split_dir.glob("*buggy*")) + list(split_dir.glob("*before*"))
        fixed_files = list(split_dir.glob("*fixed*")) + list(split_dir.glob("*after*"))

        # Try to pair them
        for buggy_file in buggy_files:
            # Find corresponding fixed file
            fixed_file = None
            for ff in fixed_files:
                if buggy_file.stem.replace("buggy", "fixed") == ff.stem or \
                   buggy_file.stem.replace("before", "after") == ff.stem:
                    fixed_file = ff
                    break

            if fixed_file and buggy_file.exists() and fixed_file.exists():
                try:
                    with open(buggy_file, 'r', encoding='utf-8') as f:
                        buggy_code = f.read()
                    with open(fixed_file, 'r', encoding='utf-8') as f:
                        fixed_code = f.read()

                    bug_fixes.append({
                        "buggy_code": buggy_code,
                        "fixed_code": fixed_code,
                        "description": f"Bug fix from {split} set",
                        "bug_type": "unknown",
                        "source": "learningfixes",
                        "language": "python",
                        "split": split
                    })
                except Exception as e:
                    logger.warning(f"Failed to read {buggy_file}: {e}")

    # Also check for JSON/JSONL
    json_files = list(dataset_dir.rglob("*.json")) + list(dataset_dir.rglob("*.jsonl"))
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                if json_file.suffix == ".jsonl":
                    for line in f:
                        item = json.loads(line)
                        bug_fix = extract_pyresbugs_item(item)
                        if bug_fix:
                            bug_fix["source"] = "learningfixes"
                            bug_fixes.append(bug_fix)
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            bug_fix = extract_pyresbugs_item(item)
                            if bug_fix:
                                bug_fix["source"] = "learningfixes"
                                bug_fixes.append(bug_fix)
        except Exception as e:
            logger.warning(f"Failed to parse {json_file}: {e}")

    return bug_fixes


async def ingest_dataset(dataset_name: str, repo_url: Optional[str] = None):
    """Main ingestion logic for a dataset."""

    # Dataset configurations
    DATASETS = {
        "pyresbugs": {
            "url": "https://github.com/dessertlab/PyResBugs.git",
            "parser": parse_pyresbugs
        },
        "hapybug": {
            "url": None,  # Need to find public repo
            "parser": parse_hapybug
        },
        "learningfixes": {
            "url": None,  # Need to find public repo
            "parser": parse_learningfixes
        }
    }

    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return

    config = DATASETS[dataset_name]
    url = repo_url or config["url"]

    if not url:
        logger.warning(f"No URL configured for {dataset_name}. Please provide --url")
        logger.warning(f"Or manually download and use --local-path")
        return

    # Create temp directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dataset_dir = temp_path / dataset_name

        # Download dataset
        if not await download_git_repo(url, dataset_dir):
            return

        # Parse dataset
        logger.info(f"Parsing {dataset_name}...")
        bug_fixes = config["parser"](dataset_dir)
        logger.info(f"✓ Parsed {len(bug_fixes)} bug-fix pairs")

        if not bug_fixes:
            logger.warning("No bug-fix pairs found. Check dataset structure.")
            return

        # Ingest into ChromaDB
        await ingest_bug_fixes(bug_fixes, dataset_name)


async def ingest_bug_fixes(bug_fixes: List[Dict[str, str]], source: str):
    """Ingest bug-fix pairs into the debugging_knowledge collection."""

    cm = get_collection_manager()
    collection = cm.get_collection("debugging_knowledge")

    if not collection:
        logger.error("Failed to get debugging_knowledge collection")
        return

    # Prepare documents for batch ingestion
    texts = []
    metadatas = []
    batch_size = 100
    total = len(bug_fixes)
    ingested = 0

    for i, bug_fix in enumerate(bug_fixes):
        # Create rich text combining buggy code, fixed code, and description
        combined_text = f"""
Bug Description: {bug_fix.get('description', 'No description')}

Buggy Code:
```python
{bug_fix['buggy_code'][:2000]}  # Limit to 2000 chars
```

Fixed Code:
```python
{bug_fix['fixed_code'][:2000]}  # Limit to 2000 chars
```

Fix Pattern: Code was modified to resolve {bug_fix.get('bug_type', 'unknown')} issue.
"""

        metadata = {
            "source": bug_fix.get("source", source),
            "bug_type": bug_fix.get("bug_type", "unknown"),
            "language": bug_fix.get("language", "python"),
            "category": "debugging",
            "timestamp": datetime.utcnow().isoformat(),
            "has_description": bool(bug_fix.get("description")),
            "annotation_quality": bug_fix.get("annotation_quality", "standard")
        }

        texts.append(combined_text)
        metadatas.append(metadata)

        # Batch insert
        if len(texts) >= batch_size or i == total - 1:
            try:
                count = cm.add_documents(
                    texts=texts,
                    metadatas=metadatas,
                    collection_name="debugging_knowledge"
                )
                ingested += count
                logger.info(f"Progress: {ingested}/{total} ({ingested*100//total}%)")
                texts = []
                metadatas = []
            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
                texts = []
                metadatas = []

    logger.info(f"✅ Successfully ingested {ingested} bug-fix pairs from {source}!")


async def ingest_from_local(dataset_name: str, local_path: str):
    """Ingest from a local directory."""

    PARSERS = {
        "pyresbugs": parse_pyresbugs,
        "hapybug": parse_hapybug,
        "learningfixes": parse_learningfixes
    }

    if dataset_name not in PARSERS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return

    dataset_dir = Path(local_path)
    if not dataset_dir.exists():
        logger.error(f"Directory not found: {local_path}")
        return

    logger.info(f"Parsing {dataset_name} from {local_path}...")
    bug_fixes = PARSERS[dataset_name](dataset_dir)
    logger.info(f"✓ Parsed {len(bug_fixes)} bug-fix pairs")

    if bug_fixes:
        await ingest_bug_fixes(bug_fixes, dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Ingest bug-fix datasets into Omni-Cortex")

    parser.add_argument("--all", action="store_true", help="Ingest all supported datasets")
    parser.add_argument("--dataset", choices=["pyresbugs", "hapybug", "learningfixes"],
                        help="Specific dataset to ingest")
    parser.add_argument("--url", help="Custom Git repository URL")
    parser.add_argument("--local-path", help="Path to local dataset directory")

    args = parser.parse_args()

    # Check for API keys
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("No API key found (OPENAI_API_KEY or OPENROUTER_API_KEY).")
        logger.warning("Skipping ingestion as embeddings require an API key.")
        logger.info("To enable ingestion, set OPENAI_API_KEY in your environment or .env file.")
        return

    if args.local_path:
        if not args.dataset:
            logger.error("--dataset is required when using --local-path")
            return
        asyncio.run(ingest_from_local(args.dataset, args.local_path))
    elif args.all:
        logger.info("Starting ingestion of all datasets...")
        for dataset in ["pyresbugs", "hapybug", "learningfixes"]:
            asyncio.run(ingest_dataset(dataset, args.url))
    elif args.dataset:
        asyncio.run(ingest_dataset(args.dataset, args.url))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
