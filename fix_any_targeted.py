#!/usr/bin/env python3
"""Fix missing Any imports in test files - targeted approach."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(
    "/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
)


def fix_any_targeted():
    """Fix missing Any imports in specific problematic files."""

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
        fix_file_any_targeted(Path(file_path))


def fix_file_any_targeted(file_path: Path):
    """Fix Any import in a specific file with targeted approach."""
    try:
        content = file_path.read_text()
        original_content = content

        # Check if file uses -> Any: but doesn't have proper import
        if "-> Any:" not in content:
            return  # No Any usage found

        # Check if file already has proper Any import
        if (
            "from typing import Any" in content
            and not content.startswith('"""')
            or "from typing import" in content
            and "Any" in content
        ):
            # Check if it's in a docstring vs real import
            lines = content.split("\n")
            has_real_import = False
            for line in lines:
                if line.strip().startswith("from typing import") and "Any" in line:
                    has_real_import = True
                    break
            if has_real_import:
                return  # Already has real import

        # Find the right place to add the import
        lines = content.split("\n")
        insert_position = 0

        # Skip past docstrings and __future__ imports
        in_docstring = False
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Handle docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    insert_position = i + 1
                else:
                    in_docstring = True
                continue

            if in_docstring:
                continue

            # Skip __future__ imports
            if stripped.startswith("from __future__"):
                insert_position = i + 1
                continue

            # Stop at first real import or code
            if stripped and not stripped.startswith("#"):
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_position = i
                break

        # Insert the import
        lines.insert(insert_position, "from typing import Any")
        content = "\n".join(lines)

        if content != original_content:
            file_path.write_text(content)
            print(f"Fixed Any import in {file_path}")

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")


if __name__ == "__main__":
    fix_any_targeted()

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
