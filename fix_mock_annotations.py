#!/usr/bin/env python3
"""Fix Mock type annotations in test files."""

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(
    "/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
)


def fix_mock_annotations():
    """Fix Mock type annotations by analyzing context and using appropriate types."""

    # Fix Mock type annotations by replacing with Any for flexibility

    # Get all files with Mock annotation issues
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821", "--no-fix"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    mock_files = set()
    for line in result.stdout.split("\n"):
        if "F821" in line and "Undefined name `Mock`" in line:
            file_path = line.split(":")[0]
            mock_files.add(file_path)

    print(f"Found {len(mock_files)} files with Mock annotation issues")

    for file_path in mock_files:
        fix_file_mock_annotations(Path(file_path))


def fix_file_mock_annotations(file_path: Path):
    """Fix Mock annotations in a specific file."""
    try:
        content = file_path.read_text()
        original_content = content

        # Replace Mock with Any in function signatures
        patterns = [
            (r"-> Mock:", "-> Any:"),
            (r"def (\w+_strategy)\([^)]*\) -> Mock:", r"def \1(\g<0>) -> Any:"),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)

        # Simpler approach: just replace all "-> Mock:" with "-> Any:"
        content = content.replace("-> Mock:", "-> Any:")

        # Ensure Any is imported
        if "-> Any:" in content and "from typing import" in content:
            # Check if Any is already imported
            if re.search(r"from typing import.*\bAny\b", content):
                pass  # Already imported
            else:
                # Add Any to existing typing import
                content = re.sub(
                    r"from typing import ([^)]+)",
                    lambda m: f"from typing import {m.group(1).rstrip()}, Any"
                    if not m.group(1).rstrip().endswith(",")
                    else f"from typing import {m.group(1)} Any",
                    content,
                )
        elif "-> Any:" in content and "from typing import" not in content:
            # Add typing import at the top after __future__
            lines = content.split("\n")
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("from __future__"):
                    insert_idx = i + 1
                    break
                elif (
                    line.strip()
                    and not line.startswith("#")
                    and not line.startswith('"""')
                ):
                    insert_idx = i
                    break

            if insert_idx > 0:
                lines.insert(insert_idx, "from typing import Any")
                content = "\n".join(lines)

        if content != original_content:
            file_path.write_text(content)
            print(f"Fixed Mock annotations in {file_path}")

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")


if __name__ == "__main__":
    fix_mock_annotations()

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
            if "F821" in line and "Mock" in line
        ],
    )
    print(f"\nRemaining Mock F821 issues: {remaining}")
