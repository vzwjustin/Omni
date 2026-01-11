"""
File watcher to auto-reingest the repository into Chroma when files change.

Usage (optional):
  ENABLE_AUTO_WATCH=true python -m app.ingest_watch

Environment:
  CHROMA_PERSIST_DIR (defaults to /app/data/chroma)
  WATCH_ROOT (defaults to repo root)

Designed to be lightweight: debounced, watches only relevant file types.
"""

import asyncio
import os
from pathlib import Path

import structlog
from watchfiles import awatch

from .ingest_repo import main as ingest_repo_main

logger = structlog.get_logger("ingest-watch")

# File globs to watch
WATCH_EXTENSIONS = {".py", ".md", ".txt", ".yaml", ".yml"}

# Debounce interval in seconds
DEBOUNCE_SECONDS = 2.0


def should_watch(path: Path) -> bool:
    # Ignore common noise dirs
    skip_dirs = {"data", "venv", ".venv", "__pycache__", ".git", "node_modules", ".mcp"}
    if any(part in skip_dirs for part in path.parts):
        return False
    return path.suffix.lower() in WATCH_EXTENSIONS


async def watch_and_ingest():
    repo_root = Path(os.getenv("WATCH_ROOT", Path(__file__).resolve().parents[1]))
    logger.info("watch_start", root=str(repo_root))
    async for changes in awatch(repo_root):
        filtered = [c for c in changes if should_watch(Path(c[1]))]
        if not filtered:
            continue
        logger.info("watch_change_detected", count=len(filtered))
        # Debounce simple sleep
        await asyncio.sleep(DEBOUNCE_SECONDS)
        try:
            # Wrap sync function with asyncio.to_thread to avoid blocking event loop
            await asyncio.to_thread(ingest_repo_main)
            logger.info("watch_ingest_complete", files=len(filtered))
        except Exception as e:
            logger.error("watch_ingest_failed", error=str(e))


def main():
    asyncio.run(watch_and_ingest())


if __name__ == "__main__":
    main()
