"""
Enhanced Repository Ingestion with Rich Metadata

Uses structured schema and intelligent chunking for high-quality retrieval.
Replaces basic ingestion with production-grade document processing.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import structlog

from .vector_schema import (
    DocumentMetadata,
    FileCategory,
    ChunkType,
    CodeAnalyzer,
    MarkdownAnalyzer,
    categorize_file,
    extract_framework_info
)
from .langchain_integration import get_vectorstore_by_collection, add_documents_with_metadata

logger = structlog.get_logger("enhanced-ingestion")


# File patterns to ingest
DEFAULT_PATTERNS = [
    "**/*.py",
    "**/*.md",
    "**/*.txt",
    "**/*.yaml",
    "**/*.yml",
    "**/*.json"
]

# Directories to skip
SKIP_DIRS = {"data", "venv", ".venv", "__pycache__", ".git", "node_modules", ".mcp", ".pytest_cache"}


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    parts = set(path.parts)
    return bool(parts & SKIP_DIRS)


def process_python_file(file_path: Path, content: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Process Python file with AST analysis and intelligent chunking."""
    chunks = []
    relative_path = file_path.as_posix()
    
    # Extract imports
    imports = CodeAnalyzer.extract_imports(content)
    
    # Extract framework context
    framework_name, framework_category = extract_framework_info(file_path)
    
    # Determine category
    category = categorize_file(file_path)
    
    # Extract code structure (functions, classes)
    structure_chunks = CodeAnalyzer.extract_structure(content, relative_path)
    
    if structure_chunks:
        # Add structured chunks
        for chunk_content, metadata, line_start, line_end in structure_chunks:
            # Enhance metadata with context
            metadata.imports = imports
            metadata.framework_name = framework_name
            metadata.framework_category = framework_category
            metadata.module_path = _extract_module_path(file_path)
            metadata.category = category.value
            metadata.has_todo = 'TODO' in chunk_content
            metadata.has_fixme = 'FIXME' in chunk_content
            
            chunks.append((chunk_content, metadata.to_dict()))
    else:
        # Fallback: store entire file as one chunk
        metadata = DocumentMetadata(
            path=relative_path,
            file_name=file_path.name,
            file_type=file_path.suffix,
            category=category.value,
            chunk_type=ChunkType.FULL_FILE.value,
            imports=imports,
            framework_name=framework_name,
            framework_category=framework_category,
            module_path=_extract_module_path(file_path),
            char_count=len(content),
            has_todo='TODO' in content,
            has_fixme='FIXME' in content,
            tags=CodeAnalyzer._extract_tags(content)
        )
        chunks.append((content, metadata.to_dict()))
    
    return chunks


def process_markdown_file(file_path: Path, content: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Process Markdown file with section-based chunking."""
    chunks = []
    relative_path = file_path.as_posix()
    
    category = categorize_file(file_path)
    
    # Chunk by sections
    section_chunks = MarkdownAnalyzer.chunk_by_sections(content, relative_path)
    
    if len(section_chunks) > 1:
        # Multiple sections
        total_chunks = len(section_chunks)
        for idx, (chunk_content, metadata) in enumerate(section_chunks):
            metadata.category = category.value
            metadata.total_chunks = total_chunks
            metadata.chunk_index = idx
            metadata.has_todo = 'TODO' in chunk_content or 'FIXME' in chunk_content
            
            chunks.append((chunk_content, metadata.to_dict()))
    else:
        # Single section or small file
        metadata = DocumentMetadata(
            path=relative_path,
            file_name=file_path.name,
            file_type=file_path.suffix,
            category=category.value,
            chunk_type=ChunkType.FULL_FILE.value,
            char_count=len(content),
            has_todo='TODO' in content or 'FIXME' in content,
            tags=MarkdownAnalyzer._extract_section_tags(content)
        )
        chunks.append((content, metadata.to_dict()))
    
    return chunks


def process_config_file(file_path: Path, content: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Process configuration files (YAML, JSON, env)."""
    relative_path = file_path.as_posix()
    
    metadata = DocumentMetadata(
        path=relative_path,
        file_name=file_path.name,
        file_type=file_path.suffix,
        category=FileCategory.CONFIG.value,
        chunk_type=ChunkType.FULL_FILE.value,
        char_count=len(content),
        tags=['configuration']
    )
    
    return [(content, metadata.to_dict())]


def process_text_file(file_path: Path, content: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Process plain text files."""
    relative_path = file_path.as_posix()
    
    category = categorize_file(file_path)
    
    metadata = DocumentMetadata(
        path=relative_path,
        file_name=file_path.name,
        file_type=file_path.suffix,
        category=category.value,
        chunk_type=ChunkType.FULL_FILE.value,
        char_count=len(content)
    )
    
    return [(content, metadata.to_dict())]


def process_file(file_path: Path, root: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Process a single file with appropriate handler based on type."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("file_read_failed", path=str(file_path), error=str(e))
        return []
    
    # Make path relative to root
    relative_path = file_path.relative_to(root)
    
    # Route to appropriate processor
    if file_path.suffix == ".py":
        return process_python_file(relative_path, content)
    elif file_path.suffix == ".md":
        return process_markdown_file(relative_path, content)
    elif file_path.suffix in [".yaml", ".yml", ".json", ".env"]:
        return process_config_file(relative_path, content)
    else:
        return process_text_file(relative_path, content)


def ingest_repository(
    root_path: Path,
    patterns: List[str] = None,
    collection_name: str = "omni-cortex-enhanced"
) -> Dict[str, int]:
    """
    Ingest repository with enhanced metadata and chunking.
    
    Returns statistics about ingestion.
    """
    patterns = patterns or DEFAULT_PATTERNS
    
    logger.info("enhanced_ingest_start", root=str(root_path), collection=collection_name)
    
    # Collect all chunks
    all_chunks: List[Tuple[str, Dict[str, Any]]] = []
    file_count = 0
    
    for pattern in patterns:
        for file_path in root_path.glob(pattern):
            if file_path.is_dir() or should_skip(file_path):
                continue
            
            chunks = process_file(file_path, root_path)
            all_chunks.extend(chunks)
            file_count += 1
            
            if file_count % 10 == 0:
                logger.info("processing", files=file_count, chunks=len(all_chunks))
    
    if not all_chunks:
        logger.warning("no_documents_found")
        return {"files": 0, "chunks": 0, "added": 0}
    
    # Separate texts and metadatas
    texts = [chunk[0] for chunk in all_chunks]
    metadatas = [chunk[1] for chunk in all_chunks]
    
    # Add to vector store
    added = add_documents_with_metadata(texts, metadatas, collection_name)
    
    stats = {
        "files": file_count,
        "chunks": len(all_chunks),
        "added": added
    }
    
    logger.info("enhanced_ingest_complete", **stats)
    return stats


def _extract_module_path(file_path: Path) -> str:
    """Extract Python module path from file path."""
    # Convert file path to module path
    # e.g., app/nodes/iterative/active_inf.py -> app.nodes.iterative.active_inf
    parts = list(file_path.parts)
    
    # Remove file extension
    if parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
    
    # Join with dots
    return '.'.join(parts)


def main():
    """Main entry point for enhanced ingestion."""
    repo_root = Path(__file__).resolve().parents[1]
    
    stats = ingest_repository(repo_root)
    
    print(f"Enhanced ingestion complete:")
    print(f"  Files processed: {stats['files']}")
    print(f"  Chunks created: {stats['chunks']}")
    print(f"  Documents added: {stats['added']}")


if __name__ == "__main__":
    main()
