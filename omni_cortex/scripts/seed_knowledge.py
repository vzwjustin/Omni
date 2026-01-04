#!/usr/bin/env python3
"""
Knowledge Seeder for Omni-Cortex
================================

Fetches and ingests "Foundation Knowledge" from external sources (URLs, llms.txt, local files)
into the 'learnings' collection. This allows the system to start with a baseline of 
best practices rather than an empty memory.

Usage:
    python3 scripts/seed_knowledge.py --url https://example.com/llms.txt
    python3 scripts/seed_knowledge.py --file ./my_best_practices.md
"""

import argparse
import asyncio
import aiohttp
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional

# Add app to path
sys.path.append(os.getcwd())

from app.collection_manager import get_collection_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("seed_knowledge")

async def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    logger.info(f"Fetching {url}...")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"Failed to fetch URL: {response.status}")
            return await response.text()

def read_file(path: str) -> str:
    """Read content from a local file."""
    logger.info(f"Reading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_sections(text: str) -> List[Dict[str, str]]:
    """
    Parse content into logical sections.
    Assumes Markdown structure with headers.
    Returns list of dicts with 'title' and 'content'.
    """
    sections = []
    current_title = "General"
    current_content = []
    
    lines = text.splitlines()
    for line in lines:
        if line.strip().startswith("#"):
            # New section
            if current_content:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_content).strip()
                })
            
            # Clean header (remove # and trim)
            current_title = line.lstrip("#").strip()
            current_content = []
        else:
            current_content.append(line)
            
    # Add last section
    if current_content:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_content).strip()
        })
        
    return sections

async def seed_knowledge(source: str, is_url: bool = True):
    """Main seeding logic."""
    try:
        # 1. Get Content
        if is_url:
            content = await fetch_url(source)
        else:
            content = read_file(source)
            
        if not content:
            logger.error("No content retrieved.")
            return

        # 2. Parse into Chunks
        sections = parse_sections(content)
        logger.info(f"Parsed {len(sections)} sections.")
        
        # 3. Ingest into Learnings
        cm = get_collection_manager()
        count = 0
        
        for section in sections:
            # Skip empty or too short sections
            if len(section["content"]) < 50:
                continue
                
            # We treat the title as the "query" (what topic is this about?)
            # and the content as the "solution" (the best practice).
            topic = section["title"]
            best_practice = section["content"]
            
            success = cm.add_learning(
                query=f"Best practices for {topic}",
                answer=best_practice,
                framework_used="foundation_knowledge", # Special framework tag
                success_rating=1.0, # Trusted source
                problem_type="best_practice"
            )
            
            if success:
                count += 1
                logger.info(f"Seeded: {topic}")
            else:
                logger.warning(f"Failed to seed: {topic}")
                
        logger.info(f"✅ Successfully seeded {count} foundation knowledges!")
        
    except Exception as e:
        logger.error(f"Seeding failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Seed Omni-Cortex with foundation knowledge.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="URL to fetch (e.g. llms.txt)")
    group.add_argument("--file", help="Local file path")
    group.add_argument("--auto", action="store_true", help="Automatically fetch curated best practices (FastAPI, Pydantic, etc.)")
    
    args = parser.parse_args()
    
    # Check for API keys before proceeding
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("No API key found (OPENAI_API_KEY or OPENROUTER_API_KEY).")
        logger.warning("Skipping knowledge seeding as RAG features require embeddings.")
        logger.info("To enable seeding, set OPENAI_API_KEY in your environment or .env file.")
        return

    # Curated list of high-quality documentation (Strictly llms.txt standard)
    CURATED_SOURCES = [
        "https://docs.anthropic.com/llms.txt",                    # Anthropic Best Practices
        "https://docs.docker.com/llms.txt",                       # Docker/DevOps
        "https://python.langchain.com/llms.txt",                  # LangChain Python
        "https://sdk.vercel.ai/llms.txt",                         # Vercel AI SDK
        "https://docs.pydantic.dev/latest/llms.txt",              # Pydantic validation
        "https://docs.stripe.com/llms.txt",                       # Stripe (Payment patterns)
        "https://docs.pinecone.io/llms.txt",                      # Vector Database patterns
        "https://daisyui.com/llms.txt",                           # UI/Tailwind Components
    ]

    if args.auto:
        logger.info(f"Starting auto-seeding from {len(CURATED_SOURCES)} sources...")
        for url in CURATED_SOURCES:
            asyncio.run(seed_knowledge(url, is_url=True))
    elif args.url:
        asyncio.run(seed_knowledge(args.url, is_url=True))
    elif args.file:
        asyncio.run(seed_knowledge(args.file, is_url=False))

if __name__ == "__main__":
    main()
