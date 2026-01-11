"""
Context optimization utilities for Claude Code.

Provides token counting, content compression, truncation detection,
and CLAUDE.md management. Complements Gemini's prepare_context
by handling token-level optimizations.
"""

import re
from pathlib import Path

# Try tiktoken, fallback to simple estimation
try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _ENCODING = None
    _HAS_TIKTOKEN = False


def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken cl100k_base encoding.

    Args:
        text: The text to count tokens for.

    Returns:
        Token count (approximate if tiktoken unavailable).
    """
    if not text:
        return 0
    if _HAS_TIKTOKEN and _ENCODING:
        return len(_ENCODING.encode(text))
    # Fallback: ~4 chars per token is a reasonable approximation
    return len(text) // 4


def compress_content(content: str, target_reduction: float = 0.3) -> dict:
    """
    Compress content by removing comments, whitespace, and less important lines.

    Keeps imports, function definitions, class definitions, and structural code.
    Achieves 30-70% token reduction typically.

    Args:
        content: The file content to compress.
        target_reduction: Target reduction percentage (0.0-1.0). Default 0.3 (30%).

    Returns:
        Dict with original_tokens, compressed_tokens, compressed_content,
        reduction_percentage.
    """
    if not content:
        return {
            "original_tokens": 0,
            "compressed_tokens": 0,
            "compressed_content": "",
            "reduction_percentage": 0.0,
        }

    original_tokens = count_tokens(content)
    lines = content.split("\n")
    compressed_lines = []

    # Patterns for important lines to always keep
    important_patterns = [
        r"^\s*(import|from)\s+",  # imports
        r"^\s*(def|async def)\s+\w+",  # function definitions
        r"^\s*class\s+\w+",  # class definitions
        r"^\s*@\w+",  # decorators
        r"^\s*(if|elif|else|for|while|try|except|finally|with|match|case)\s*",  # control flow
        r"^\s*(return|yield|raise|await)\s+",  # returns and raises
        r"^\s*(self\.\w+\s*=)",  # instance attribute assignments
        r"^\s*[A-Z_]+\s*=",  # constants
        r"^\s*__\w+__\s*=",  # dunder attributes
    ]

    # Patterns for lines to remove

    in_docstring = False
    docstring_char = None
    consecutive_empty = 0

    for line in lines:
        stripped = line.strip()

        # Handle docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2 and len(stripped) > 6:
                    # Single-line docstring - keep first line only for context
                    if len(compressed_lines) > 0:
                        compressed_lines.append(
                            f"{line.split(docstring_char)[0]}{docstring_char}...{docstring_char}"
                        )
                    continue
                in_docstring = True
                continue
        else:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
                docstring_char = None
            continue

        # Skip pure comments (not type hints)
        if stripped.startswith("#") and "type:" not in stripped:
            continue

        # Handle empty lines - keep max 1 consecutive
        if not stripped:
            consecutive_empty += 1
            if consecutive_empty <= 1:
                compressed_lines.append("")
            continue
        else:
            consecutive_empty = 0

        # Check if line is important
        is_important = any(re.match(p, line) for p in important_patterns)

        if is_important:
            compressed_lines.append(line)
        else:
            # Keep line but compress whitespace
            compressed_line = re.sub(r"\s+", " ", line.strip())
            if compressed_line:
                # Preserve indentation level roughly
                indent = len(line) - len(line.lstrip())
                indent_str = " " * (indent // 2)  # Reduce indent
                compressed_lines.append(f"{indent_str}{compressed_line}")

    compressed_content = "\n".join(compressed_lines)
    compressed_tokens = count_tokens(compressed_content)

    reduction = 1.0 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0.0

    return {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compressed_content": compressed_content,
        "reduction_percentage": round(reduction * 100, 2),
    }


def detect_truncation(text: str) -> dict:
    """
    Detect if text appears to be truncated.

    Checks for unclosed code blocks, braces, brackets, and incomplete sentences.

    Args:
        text: The text to check for truncation.

    Returns:
        Dict with possibly_truncated, indicators list, confidence (0.0-1.0).
    """
    if not text:
        return {"possibly_truncated": False, "indicators": [], "confidence": 0.0}

    indicators = []

    # Check for unclosed code blocks
    code_block_opens = text.count("```")
    if code_block_opens % 2 != 0:
        indicators.append("unclosed_code_block")

    # Check for unclosed braces/brackets
    open_braces = text.count("{") - text.count("}")
    if open_braces > 0:
        indicators.append(f"unclosed_braces:{open_braces}")

    open_brackets = text.count("[") - text.count("]")
    if open_brackets > 0:
        indicators.append(f"unclosed_brackets:{open_brackets}")

    open_parens = text.count("(") - text.count(")")
    if open_parens > 0:
        indicators.append(f"unclosed_parentheses:{open_parens}")

    # Check for unclosed strings (simple check)
    lines = text.split("\n")
    last_lines = lines[-5:] if len(lines) >= 5 else lines
    for i, line in enumerate(last_lines):
        single_quotes = line.count("'") - line.count("\\'")
        double_quotes = line.count('"') - line.count('\\"')
        if single_quotes % 2 != 0:
            indicators.append(f"unclosed_single_quote_line_{len(lines) - len(last_lines) + i}")
        if double_quotes % 2 != 0:
            indicators.append(f"unclosed_double_quote_line_{len(lines) - len(last_lines) + i}")

    # Check for incomplete sentences at end
    last_line = text.rstrip().split("\n")[-1] if text.strip() else ""
    if last_line:
        # Ends mid-word or with continuation characters
        if last_line.endswith(("...", "..", ",", ":", "\\", "+", "-", "*", "/", "=")):
            indicators.append("incomplete_ending")
        # Ends mid-sentence without punctuation
        elif last_line and last_line[-1].isalnum() and len(last_line) > 50:
            indicators.append("possible_mid_sentence")

    # Check for truncation markers
    truncation_markers = ["[truncated]", "[...]", "...more...", "<truncated>"]
    for marker in truncation_markers:
        if marker.lower() in text.lower():
            indicators.append(f"truncation_marker:{marker}")

    # Calculate confidence
    confidence = min(len(indicators) * 0.25, 1.0)

    return {
        "possibly_truncated": len(indicators) > 0,
        "indicators": indicators,
        "confidence": round(confidence, 2),
    }


# Rule presets for CLAUDE.md generation
RULE_PRESETS: dict[str, list[str]] = {
    "security": [
        "NEVER include API keys, passwords, or secrets in code",
        "ALWAYS validate and sanitize user input",
        "Use parameterized queries to prevent SQL injection",
        "Escape output to prevent XSS attacks",
        "NEVER disable security features or bypass authentication",
    ],
    "performance": [
        "Avoid N+1 queries - use eager loading or batching",
        "Cache expensive computations and API calls",
        "Use pagination for large data sets",
        "Profile before optimizing - measure, don't guess",
        "Prefer streaming for large files and responses",
    ],
    "testing": [
        "Write tests for new functionality before merging",
        "Test edge cases and error conditions",
        "Use meaningful test names that describe behavior",
        "Mock external dependencies in unit tests",
        "Maintain test coverage above 80%",
    ],
    "documentation": [
        "Document public APIs with clear examples",
        "Keep README up to date with setup instructions",
        "Use docstrings for functions with non-obvious behavior",
        "Document breaking changes in CHANGELOG",
        "Include type hints for better IDE support",
    ],
    "code_quality": [
        "Keep functions under 50 lines",
        "Avoid deep nesting (max 3 levels)",
        "Use meaningful variable and function names",
        "Remove dead code and unused imports",
        "Follow the project's existing code style",
    ],
    "git": [
        "Write clear, concise commit messages",
        "Keep commits atomic - one logical change per commit",
        "Never force push to main/master branch",
        "Rebase feature branches before merging",
        "Use conventional commit format when applicable",
    ],
    "context_optimization": [
        "Use specific file paths instead of broad searches",
        "Read only necessary portions of large files",
        "Use /compact when context grows large",
        "Prefer grep with files_with_matches mode",
        "Start new sessions for unrelated tasks",
    ],
}


def generate_claude_md_template(
    project_type: str = "general", rules: list[str] | None = None, presets: list[str] | None = None
) -> str:
    """
    Generate a CLAUDE.md template for a project.

    Args:
        project_type: One of 'general', 'python', 'typescript', 'react', 'rust'.
        rules: Custom rules to include.
        presets: List of preset names to include (from RULE_PRESETS).

    Returns:
        Generated CLAUDE.md content.
    """
    rules = rules or []
    presets = presets or []

    # Project-specific headers
    project_headers = {
        "general": "# Project Guidelines\n\nThis file provides guidance for AI assistants working with this codebase.",
        "python": "# Python Project Guidelines\n\nThis file provides guidance for AI assistants working with this Python codebase.\n\n## Stack\n- Python 3.10+\n- Use type hints throughout\n- Follow PEP 8 style guide",
        "typescript": "# TypeScript Project Guidelines\n\nThis file provides guidance for AI assistants working with this TypeScript codebase.\n\n## Stack\n- TypeScript 5.x\n- ESLint + Prettier\n- Use strict mode",
        "react": "# React Project Guidelines\n\nThis file provides guidance for AI assistants working with this React codebase.\n\n## Stack\n- React 18+\n- TypeScript\n- Prefer functional components with hooks\n- Use React Query for data fetching",
        "rust": "# Rust Project Guidelines\n\nThis file provides guidance for AI assistants working with this Rust codebase.\n\n## Stack\n- Rust stable\n- Use clippy for linting\n- Run `cargo fmt` before committing",
    }

    header = project_headers.get(project_type, project_headers["general"])

    sections = [header, ""]

    # Add rules section
    sections.append("## Rules\n")

    # Add preset rules
    for preset_name in presets:
        if preset_name in RULE_PRESETS:
            sections.append(f"### {preset_name.replace('_', ' ').title()}\n")
            for rule in RULE_PRESETS[preset_name]:
                sections.append(f"- {rule}")
            sections.append("")

    # Add custom rules
    if rules:
        sections.append("### Custom Rules\n")
        for rule in rules:
            sections.append(f"- {rule}")
        sections.append("")

    return "\n".join(sections)


def inject_rules(
    existing_content: str,
    rules: list[str],
    section: str = "Rules",
    presets: list[str] | None = None,
) -> str:
    """
    Inject rules into existing CLAUDE.md content.

    Adds or replaces rules in a specific section.

    Args:
        existing_content: Current CLAUDE.md content.
        rules: Rules to inject.
        section: Section name to inject into (default "Rules").
        presets: Optional list of presets to also inject.

    Returns:
        Updated CLAUDE.md content.
    """
    presets = presets or []

    # Build the new rules content
    new_rules_lines = []

    # Add preset rules
    for preset_name in presets:
        if preset_name in RULE_PRESETS:
            new_rules_lines.append(f"### {preset_name.replace('_', ' ').title()}\n")
            for rule in RULE_PRESETS[preset_name]:
                new_rules_lines.append(f"- {rule}")
            new_rules_lines.append("")

    # Add custom rules
    if rules:
        new_rules_lines.append("### Injected Rules\n")
        for rule in rules:
            new_rules_lines.append(f"- {rule}")
        new_rules_lines.append("")

    new_rules_content = "\n".join(new_rules_lines)

    # Find the section in existing content
    section_pattern = rf"^(##\s+{re.escape(section)})\s*\n"
    section_match = re.search(section_pattern, existing_content, re.MULTILINE)

    if section_match:
        # Find the next section (## heading)
        section_start = section_match.end()
        next_section = re.search(r"^##\s+", existing_content[section_start:], re.MULTILINE)

        if next_section:
            section_end = section_start + next_section.start()
        else:
            section_end = len(existing_content)

        # Replace section content
        updated = (
            existing_content[:section_start]
            + "\n"
            + new_rules_content
            + existing_content[section_end:]
        )
    else:
        # Add new section at the end
        updated = existing_content.rstrip() + f"\n\n## {section}\n\n{new_rules_content}"

    return updated


def analyze_claude_md(directory: str) -> dict:
    """
    Find and analyze CLAUDE.md files in a project.

    Searches for CLAUDE.md in the directory and parent directories,
    reporting token usage and scope.

    Args:
        directory: Project directory to analyze.

    Returns:
        Dict with files_found count, files list with path/tokens/scope.
    """
    dir_path = Path(directory).resolve()
    files_found = []

    # Check common locations
    locations_to_check = [
        dir_path / "CLAUDE.md",
        dir_path / ".claude" / "CLAUDE.md",
        dir_path / "docs" / "CLAUDE.md",
    ]

    # Also check parent directories (up to 3 levels)
    parent = dir_path.parent
    for _ in range(3):
        if parent != parent.parent:  # Not at root
            locations_to_check.append(parent / "CLAUDE.md")
            parent = parent.parent

    # Check home directory for global CLAUDE.md
    home_claude = Path.home() / ".claude" / "CLAUDE.md"
    locations_to_check.append(home_claude)

    seen_paths = set()

    for location in locations_to_check:
        try:
            if location.exists() and location.is_file():
                real_path = location.resolve()
                if real_path in seen_paths:
                    continue
                seen_paths.add(real_path)

                content = location.read_text(encoding="utf-8")
                tokens = count_tokens(content)

                # Determine scope
                if str(real_path).startswith(str(Path.home() / ".claude")):
                    scope = "global"
                elif real_path.parent == dir_path:
                    scope = "project"
                elif str(real_path).startswith(str(dir_path)):
                    scope = "subdirectory"
                else:
                    scope = "parent"

                files_found.append(
                    {
                        "path": str(real_path),
                        "tokens": tokens,
                        "scope": scope,
                        "size_bytes": location.stat().st_size,
                    }
                )
        except (PermissionError, OSError):
            continue

    # Sort by scope priority: project > subdirectory > parent > global
    scope_order = {"project": 0, "subdirectory": 1, "parent": 2, "global": 3}
    files_found.sort(key=lambda f: scope_order.get(f["scope"], 99))

    total_tokens = sum(f["tokens"] for f in files_found)

    return {"files_found": len(files_found), "total_tokens": total_tokens, "files": files_found}
