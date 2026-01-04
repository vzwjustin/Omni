"""
Repository ingestion script for RAG.

Reads text/code files and ingests into the shared Chroma vector store
for retrieval-aware reasoning. Run manually when code/doc changes:

    python -m app.ingest_repo

Environment:
- CHROMA_PERSIST_DIR: optional, defaults to /app/data/chroma
"""

import os
from pathlib import Path
from typing import List
import structlog

from .langchain_integration import add_documents, get_vectorstore

logger = structlog.get_logger("ingest-repo")

# File globs to index
DEFAULT_PATTERNS = [
    "**/*.py",
    "**/*.md",
    "**/*.txt",
    "**/*.yaml",
    "**/*.yml",
]

# Directories to skip
SKIP_DIRS = {"data", "venv", ".venv", "__pycache__", ".git", "node_modules", ".mcp"}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & SKIP_DIRS)


def read_files(root: Path, patterns: List[str]) -> List[tuple[str, str]]:
    docs: List[tuple[str, str]] = []
    for pattern in patterns:
        for file in root.glob(pattern):
            if file.is_dir() or should_skip(file):
                continue
            try:
                text = file.read_text(encoding="utf-8")
            except Exception:
                continue
            rel = file.relative_to(root).as_posix()
            docs.append((rel, text))
    return docs


def main():
    repo_root = Path(__file__).resolve().parents[1]
    logger.info("ingest_start", root=str(repo_root))

    vs = get_vectorstore()
    if not vs:
        logger.error("vectorstore_unavailable")
        return

    docs = read_files(repo_root, DEFAULT_PATTERNS)
    if not docs:
        logger.warn("no_docs_found")
        return

    texts = [c for _, c in docs]
    metas = [{"path": p} for p, _ in docs]
    added = add_documents(texts, metas)
    logger.info("ingest_complete", added=added, total=len(texts))


if __name__ == "__main__":
    main()
