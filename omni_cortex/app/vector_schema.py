"""
Enhanced Vector Database Schema for Omni-Cortex

Provides structured metadata, intelligent chunking, and specialized collections
for high-quality semantic retrieval across the codebase.
"""

import ast
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger("vector-schema")


class FileCategory(str, Enum):
    """File categorization for specialized retrieval."""

    FRAMEWORK = "framework"
    DOCUMENTATION = "documentation"
    CONFIG = "config"
    UTILITY = "utility"
    TEST = "test"
    SCHEMA = "schema"
    SERVER = "server"
    INTEGRATION = "integration"


class ChunkType(str, Enum):
    """Type of content chunk."""

    FULL_FILE = "full_file"
    FUNCTION = "function"
    CLASS = "class"
    DOCSTRING = "docstring"
    SECTION = "section"
    CONFIG_BLOCK = "config_block"


@dataclass
class DocumentMetadata:
    """Rich metadata schema for vector store documents."""

    # Core identification
    path: str
    file_name: str
    file_type: str  # .py, .md, .yaml, etc.
    category: str  # FileCategory value

    # Chunking info
    chunk_type: str  # ChunkType value
    chunk_index: int = 0
    total_chunks: int = 1
    parent_document: str | None = None

    # Code structure (for Python files)
    function_name: str | None = None
    class_name: str | None = None
    imports: list[str] | None = None
    decorators: list[str] | None = None

    # Content characteristics
    line_start: int | None = None
    line_end: int | None = None
    char_count: int = 0
    complexity_score: float = 0.0

    # Framework/module context
    framework_name: str | None = None  # e.g., "active_inference"
    framework_category: str | None = None  # e.g., "iterative"
    module_path: str | None = None  # e.g., "app.nodes.iterative"

    # Semantic tags
    tags: list[str] | None = None
    has_todo: bool = False
    has_fixme: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for Chroma storage."""
        data = asdict(self)
        # Convert lists to comma-separated strings for Chroma compatibility
        # Convert lists to comma-separated strings for Chroma compatibility
        if isinstance(data.get("imports"), list):
            data["imports"] = ",".join(data["imports"])
        if isinstance(data.get("decorators"), list):
            data["decorators"] = ",".join(data["decorators"])
        if isinstance(data.get("tags"), list):
            data["tags"] = ",".join(data["tags"])
        # Remove None values
        return {k: v for k, v in data.items() if v is not None}


class CodeAnalyzer:
    """Analyze Python code structure for metadata extraction."""

    @staticmethod
    def extract_structure(
        code: str, file_path: str
    ) -> list[tuple[str, DocumentMetadata, int, int]]:
        """
        Extract code structure (functions, classes) with metadata.

        Returns: List of (content, metadata, line_start, line_end)
        """
        chunks = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning("ast_parse_failed", path=file_path, error=str(e))
            return chunks

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunk_content, metadata = CodeAnalyzer._extract_function(node, code, file_path)
                if chunk_content:
                    chunks.append(
                        (chunk_content, metadata, node.lineno, node.end_lineno or node.lineno)
                    )

            elif isinstance(node, ast.ClassDef):
                chunk_content, metadata = CodeAnalyzer._extract_class(node, code, file_path)
                if chunk_content:
                    chunks.append(
                        (chunk_content, metadata, node.lineno, node.end_lineno or node.lineno)
                    )

        return chunks

    @staticmethod
    def _extract_function(
        node: ast.FunctionDef, code: str, file_path: str
    ) -> tuple[str, DocumentMetadata]:
        """Extract function with rich metadata."""
        lines = code.split("\n")
        start = node.lineno - 1
        end = node.end_lineno if node.end_lineno else start + 1

        func_code = "\n".join(lines[start:end])

        # Extract decorators
        decorators = [
            d.id if isinstance(d, ast.Name) else ast.unparse(d) for d in node.decorator_list
        ]

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        metadata = DocumentMetadata(
            path=file_path,
            file_name=Path(file_path).name,
            file_type=".py",
            category=FileCategory.FRAMEWORK.value,
            chunk_type=ChunkType.FUNCTION.value,
            function_name=node.name,
            decorators=decorators,
            line_start=node.lineno,
            line_end=node.end_lineno,
            char_count=len(func_code),
            complexity_score=CodeAnalyzer._estimate_complexity(node),
            tags=CodeAnalyzer._extract_tags(func_code),
        )

        # Combine function signature, docstring, and body for better context
        content = f"Function: {node.name}\n\n{docstring}\n\n{func_code}"

        return content, metadata

    @staticmethod
    def _extract_class(
        node: ast.ClassDef, code: str, file_path: str
    ) -> tuple[str, DocumentMetadata]:
        """Extract class with rich metadata."""
        lines = code.split("\n")
        start = node.lineno - 1
        end = node.end_lineno if node.end_lineno else start + 1

        class_code = "\n".join(lines[start:end])
        docstring = ast.get_docstring(node) or ""

        # Extract method names
        methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]

        metadata = DocumentMetadata(
            path=file_path,
            file_name=Path(file_path).name,
            file_type=".py",
            category=FileCategory.FRAMEWORK.value,
            chunk_type=ChunkType.CLASS.value,
            class_name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno,
            char_count=len(class_code),
            tags=methods + CodeAnalyzer._extract_tags(class_code),
        )

        content = (
            f"Class: {node.name}\n\n{docstring}\n\nMethods: {', '.join(methods)}\n\n{class_code}"
        )

        return content, metadata

    @staticmethod
    def _estimate_complexity(node: ast.FunctionDef) -> float:
        """Estimate cyclomatic complexity (simplified)."""
        complexity = 1.0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
                complexity += 1
        return min(complexity / 10.0, 1.0)

    @staticmethod
    def _extract_tags(code: str) -> list[str]:
        """Extract semantic tags from code."""
        tags = []

        # Check for common patterns
        if "TODO" in code or "FIXME" in code:
            tags.append("needs_attention")
        if "async def" in code:
            tags.append("async")
        if "@tool" in code or "@quiet_star" in code:
            tags.append("decorated")
        if "LangChain" in code or "langchain" in code:
            tags.append("langchain_integration")
        if "LangGraph" in code or "langgraph" in code:
            tags.append("langgraph_workflow")
        if "Chroma" in code or "vectorstore" in code:
            tags.append("vector_store")

        return tags

    @staticmethod
    def extract_imports(code: str) -> list[str]:
        """Extract import statements."""
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.append(module)
        except SyntaxError as e:
            # Invalid Python - return empty imports (expected for broken files)
            logger.debug(
                "python_parse_failed_extracting_imports", error=str(e), error_type=type(e).__name__
            )
        return imports


class MarkdownAnalyzer:
    """Analyze Markdown documents for intelligent chunking."""

    @staticmethod
    def chunk_by_sections(content: str, file_path: str) -> list[tuple[str, DocumentMetadata]]:
        """Split markdown by headers with context preservation."""
        chunks = []

        # Split by headers
        sections = re.split(r"^(#{1,6}\s+.+)$", content, flags=re.MULTILINE)

        current_section = ""
        current_header = ""

        for _i, part in enumerate(sections):
            if re.match(r"^#{1,6}\s+", part):
                # Save previous section
                if current_section.strip():
                    chunks.append(
                        MarkdownAnalyzer._create_section_chunk(
                            current_header, current_section, file_path, len(chunks)
                        )
                    )
                current_header = part
                current_section = part + "\n"
            else:
                current_section += part

        # Add last section
        if current_section.strip():
            chunks.append(
                MarkdownAnalyzer._create_section_chunk(
                    current_header, current_section, file_path, len(chunks)
                )
            )

        return chunks

    @staticmethod
    def _create_section_chunk(
        header: str, content: str, file_path: str, index: int
    ) -> tuple[str, DocumentMetadata]:
        """Create chunk with metadata for a markdown section."""
        metadata = DocumentMetadata(
            path=file_path,
            file_name=Path(file_path).name,
            file_type=".md",
            category=FileCategory.DOCUMENTATION.value,
            chunk_type=ChunkType.SECTION.value,
            chunk_index=index,
            char_count=len(content),
            tags=MarkdownAnalyzer._extract_section_tags(content),
        )

        return content, metadata

    @staticmethod
    def _extract_section_tags(content: str) -> list[str]:
        """Extract semantic tags from markdown content."""
        tags = []

        if "```python" in content or "```py" in content:
            tags.append("code_example")
        if "##" in content:
            tags.append("subsections")
        if re.search(r"\[.*?\]\(.*?\)", content):
            tags.append("has_links")
        if re.search(r"TODO|FIXME|NOTE|WARNING", content):
            tags.append("annotations")

        return tags


def categorize_file(file_path: Path) -> FileCategory:
    """Determine file category based on path and name."""
    path_str = str(file_path).lower()
    name = file_path.name.lower()

    if "test" in path_str or name.startswith("test_"):
        return FileCategory.TEST
    elif "nodes" in path_str and any(
        cat in path_str for cat in ["strategy", "search", "iterative", "code", "context", "fast"]
    ):
        return FileCategory.FRAMEWORK
    elif file_path.suffix == ".md":
        return FileCategory.DOCUMENTATION
    elif file_path.suffix in [".yaml", ".yml", ".json", ".env"]:
        return FileCategory.CONFIG
    elif "server" in path_str or name == "main.py":
        return FileCategory.SERVER
    elif "langchain_integration" in name or "graph.py" in name:
        return FileCategory.INTEGRATION
    elif "schema" in name or "state" in name:
        return FileCategory.SCHEMA
    else:
        return FileCategory.UTILITY


def extract_framework_info(file_path: Path) -> tuple[str | None, str | None]:
    """Extract framework name and category from path."""
    parts = file_path.parts

    framework_categories = {
        "strategy": ["reason_flux", "self_discover", "buffer_of_thoughts", "coala"],
        "search": ["mcts_rstar", "tree_of_thoughts", "graph_of_thoughts", "everything_of_thought"],
        "iterative": ["active_inference", "multi_agent_debate", "adaptive_injection", "re2"],
        "code": ["program_of_thoughts", "chain_of_verification", "critic"],
        "context": ["chain_of_note", "step_back", "analogical"],
        "fast": ["skeleton_of_thought", "system1"],
    }

    for category, frameworks in framework_categories.items():
        if category in parts:
            # Try to extract framework name from filename
            name = file_path.stem
            for fw in frameworks:
                if fw in name or name in fw:
                    return fw, category

    return None, None
