#!/usr/bin/env python3
"""Fix Mock import issues in test files."""

import re
import subprocess
from pathlib import Path


def fix_mock_imports():
    """Fix Mock import issues in test files."""
    # Get files with Mock F821 errors
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821"],
        check=False,
        capture_output=True,
        text=True,
        cwd="/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
    )

    # Find files that need Mock import
    files_needing_mock = set()
    for line in result.stdout.split("\n"):
        if "F821" in line and "Undefined name `Mock`" in line:
            match = re.match(r"([^:]+):", line)
            if match:
                files_needing_mock.add(match.group(1))

    print(f"Found {len(files_needing_mock)} files needing Mock import")

    for filepath in files_needing_mock:
        path = Path(filepath)
        if not path.exists():
            continue

        try:
            content = path.read_text()
            lines = content.splitlines()

            # Check if Mock is already imported
            has_mock = any(
                "from unittest.mock import" in line and "Mock" in line for line in lines
            )
            if has_mock:
                continue

            # Find unittest.mock import line to modify
            for i, line in enumerate(lines):
                if (
                    "from unittest.mock import" in line
                    and not line.strip().endswith("\\")
                    and "Mock" not in line
                ):
                    # Simple case - add Mock to the import
                    if line.endswith(")"):
                        # Multi-line import - insert before closing paren
                        lines[i] = line.replace(")", ", Mock)")
                    else:
                        # Single line import - add Mock
                        lines[i] = line.rstrip() + ", Mock"

                    path.write_text("\n".join(lines) + "\n")
                    print(f"Fixed {filepath}")
                    break

        except Exception as e:
            print(f"Error fixing {filepath}: {e}")


if __name__ == "__main__":
    fix_mock_imports()
