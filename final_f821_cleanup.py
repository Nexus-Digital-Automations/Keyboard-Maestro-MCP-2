#!/usr/bin/env python3
"""Final cleanup of remaining F821 errors."""

import re
import subprocess
from pathlib import Path


def fix_remaining_files():
    """Fix the remaining F821 errors."""
    # Get current errors
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821"],
        check=False,
        capture_output=True,
        text=True,
        cwd="/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
    )

    # Parse errors by file
    file_errors = {}
    for line in result.stdout.split("\n"):
        if "F821" in line and "Undefined name" in line:
            match = re.match(r"([^:]+):\d+:\d+: F821 Undefined name `([^`]+)`", line)
            if match:
                filename, undefined_name = match.groups()
                if filename not in file_errors:
                    file_errors[filename] = set()
                file_errors[filename].add(undefined_name)

    # Fix each file
    for filepath, undefined_names in file_errors.items():
        path = Path(filepath)
        if not path.exists():
            continue

        try:
            content = path.read_text()
            lines = content.splitlines()
            modified = False

            # Check what imports are needed
            needs_context = "Context" in undefined_names
            needs_either = "Either" in undefined_names
            needs_path = "Path" in undefined_names

            # Check if imports already exist
            has_context = any("from fastmcp import Context" in line for line in lines)
            has_either = any(
                "from src.core.either import Either" in line for line in lines
            )
            has_path = any("from pathlib import Path" in line for line in lines)

            # Find where to insert imports
            insert_idx = None
            for i, line in enumerate(lines):
                if line.startswith("from __future__"):
                    insert_idx = i + 2  # After __future__ and blank line
                    break
                if line.strip() and (
                    line.startswith("import ") or line.startswith("from ")
                ):
                    insert_idx = i
                    break

            if insert_idx is None:
                # Find end of docstring
                for i, line in enumerate(lines):
                    if (
                        line.strip()
                        and not line.startswith("#")
                        and not ('"""' in line or "'''" in line)
                    ):
                        insert_idx = i
                        break

            # Add missing imports
            imports_to_add = []
            if needs_context and not has_context:
                imports_to_add.append("from fastmcp import Context")
            if needs_either and not has_either:
                imports_to_add.append("from src.core.either import Either")
            if needs_path and not has_path:
                imports_to_add.append("from pathlib import Path")

            if imports_to_add and insert_idx is not None:
                # Insert imports
                for imp in reversed(imports_to_add):  # Reverse to maintain order
                    lines.insert(insert_idx, imp)
                modified = True

            if modified:
                path.write_text("\n".join(lines) + "\n")
                print(f"Fixed {filepath}: {', '.join(undefined_names)}")

        except Exception as e:
            print(f"Error fixing {filepath}: {e}")


def main():
    """Main function."""
    print("Fixing remaining F821 errors...")
    fix_remaining_files()

    # Check final count
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821"],
        check=False,
        capture_output=True,
        text=True,
        cwd="/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
    )

    remaining = len([line for line in result.stdout.split("\n") if "F821" in line])
    print(f"\nFinal F821 error count: {remaining}")

    if remaining > 0:
        print("Remaining errors:")
        error_lines = [line for line in result.stdout.split("\n") if "F821" in line][
            :20
        ]
        for line in error_lines:
            print(line)
    else:
        print("✅ All F821 errors fixed!")


if __name__ == "__main__":
    main()
