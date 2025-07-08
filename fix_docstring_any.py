#!/usr/bin/env python3
"""Fix Any imports that are mistakenly in docstrings."""

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(
    "/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
)


def fix_docstring_any():
    """Fix Any imports that are in docstrings instead of proper imports."""

    # Get all files with F821 Any issues
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821", "--no-fix"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    any_files = []
    for line in result.stdout.split("\n"):
        if "F821" in line and "Undefined name `Any`" in line:
            file_path = line.split(":")[0]
            if file_path not in any_files:
                any_files.append(file_path)

    print(f"Found {len(any_files)} files with missing Any imports")

    for file_path in any_files:
        fix_file_docstring_any(Path(file_path))


def fix_file_docstring_any(file_path: Path):
    """Fix Any import in docstring and add proper import."""
    try:
        content = file_path.read_text()
        original_content = content

        # Look for the pattern of "from typing import Any" in docstring
        pattern = r'("""[^"]*?)\nfrom typing import Any\n([^"]*?""")'
        if re.search(pattern, content, re.DOTALL):
            # Remove from docstring
            content = re.sub(pattern, r"\1\n\2", content, flags=re.DOTALL)

            # Add proper import after __future__
            future_pattern = r"(from __future__ import annotations)\n"
            if re.search(future_pattern, content):
                content = re.sub(
                    future_pattern,
                    r"\1\n\nfrom typing import Any\n",
                    content,
                )
            else:
                # Add after docstring
                docstring_end = content.find('"""', content.find('"""') + 3) + 3
                if docstring_end > 2:
                    content = (
                        content[:docstring_end]
                        + "\n\nfrom typing import Any\n"
                        + content[docstring_end:]
                    )

        # Also handle triple single quote docstrings
        pattern_single = r"('''[^']*?)\nfrom typing import Any\n([^']*?''')"
        if re.search(pattern_single, content, re.DOTALL):
            content = re.sub(pattern_single, r"\1\n\2", content, flags=re.DOTALL)

            future_pattern = r"(from __future__ import annotations)\n"
            if re.search(future_pattern, content):
                content = re.sub(
                    future_pattern,
                    r"\1\n\nfrom typing import Any\n",
                    content,
                )

        if content != original_content:
            file_path.write_text(content)
            print(f"Fixed docstring Any import in {file_path}")

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")


if __name__ == "__main__":
    fix_docstring_any()

    # Check remaining issues
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821", "--no-fix"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    remaining = len(
        [
            line
            for line in result.stdout.split("\n")
            if "F821" in line and "Any" in line
        ],
    )
    print(f"\nRemaining Any F821 issues: {remaining}")
