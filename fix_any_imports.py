#!/usr/bin/env python3
"""Fix missing Any imports in test files."""

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(
    "/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
)


def fix_any_imports():
    """Fix missing Any imports by adding them to existing typing imports."""

    # Get all files with F821 Any issues
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821", "--no-fix"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    any_files = set()
    for line in result.stdout.split("\n"):
        if "F821" in line and "Undefined name `Any`" in line:
            file_path = line.split(":")[0]
            any_files.add(file_path)

    print(f"Found {len(any_files)} files with missing Any imports")

    for file_path in any_files:
        fix_file_any_import(Path(file_path))


def fix_file_any_import(file_path: Path):
    """Fix Any import in a specific file."""
    try:
        content = file_path.read_text()
        original_content = content

        # Check if Any is used but not imported
        if "-> Any:" not in content:
            return  # No Any usage found

        # Check if Any is already imported
        if re.search(r"from typing import.*\bAny\b", content):
            return  # Already imported

        # Add Any to existing typing import
        if "from typing import" in content:
            # Find the first typing import and add Any to it
            typing_import_pattern = r"from typing import ([^)]+)"
            match = re.search(typing_import_pattern, content)
            if match:
                imports = match.group(1).strip()
                if not imports.endswith(","):
                    imports += ","
                new_imports = f"from typing import {imports} Any"
                content = re.sub(typing_import_pattern, new_imports, content, count=1)
        else:
            # Add new typing import after __future__ imports or at the top
            lines = content.split("\n")
            insert_idx = 0

            # Find the best place to insert the import
            for i, line in enumerate(lines):
                if line.startswith("from __future__"):
                    insert_idx = i + 1
                elif (
                    line.strip()
                    and not line.startswith("#")
                    and not line.startswith('"""')
                    and "import" not in line
                ):
                    insert_idx = i
                    break

            lines.insert(insert_idx, "from typing import Any")
            content = "\n".join(lines)

        if content != original_content:
            file_path.write_text(content)
            print(f"Fixed Any import in {file_path}")

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")


if __name__ == "__main__":
    fix_any_imports()

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
