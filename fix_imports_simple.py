#!/usr/bin/env python3
"""Simple fix for F821 errors by pattern matching."""

import re
import subprocess
from collections import defaultdict
from pathlib import Path


def run_ruff():
    """Get ruff F821 errors."""
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821"],
        check=False,
        capture_output=True,
        text=True,
        cwd="/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
    )
    return result.stdout


def parse_errors(output):
    """Parse ruff errors to get file -> undefined names mapping."""
    file_errors = defaultdict(set)

    for line in output.split("\n"):
        if "F821" in line and "Undefined name" in line:
            # Extract filename and undefined name
            match = re.match(r"([^:]+):\d+:\d+: F821 Undefined name `([^`]+)`", line)
            if match:
                filename, undefined_name = match.groups()
                file_errors[filename].add(undefined_name)

    return file_errors


def fix_file(filepath, undefined_names):
    """Fix a single file by adding missing imports."""
    path = Path(filepath)
    if not path.exists():
        return False

    try:
        content = path.read_text()
        lines = content.splitlines()

        # Find import section
        import_end = 0
        for i, line in enumerate(lines):
            if (
                line.strip()
                and not line.startswith("#")
                and not line.startswith('"""')
                and not line.startswith("from")
                and not line.startswith("import")
            ):
                import_end = i
                break

        # Categorize imports needed
        typing_imports = set()
        other_imports = []

        for name in undefined_names:
            if name in [
                "Any",
                "Callable",
                "Awaitable",
                "Optional",
                "Union",
                "List",
                "Dict",
                "Tuple",
                "Set",
                "TypeVar",
                "Generic",
                "Protocol",
                "ClassVar",
                "Final",
                "Literal",
                "overload",
                "TYPE_CHECKING",
            ]:
                typing_imports.add(name)
            elif name == "Context":
                other_imports.append("from fastmcp import Context")
            elif name == "Either":
                other_imports.append("from src.core.either import Either")
            elif name == "Path":
                other_imports.append("from pathlib import Path")

        # Check existing imports
        existing_typing = set()
        typing_line_idx = None

        for i, line in enumerate(lines):
            if line.startswith("from typing import"):
                typing_line_idx = i
                imports = line.split("import")[1].strip()
                existing_typing.update(imp.strip() for imp in imports.split(","))
                break

        # Update or add typing imports
        if typing_imports:
            all_typing = sorted(existing_typing | typing_imports)
            typing_line = f"from typing import {', '.join(all_typing)}"

            if typing_line_idx is not None:
                lines[typing_line_idx] = typing_line
            else:
                # Find where to insert typing import
                insert_idx = import_end
                for i, line in enumerate(lines):
                    if line.startswith("from __future__"):
                        insert_idx = i + 1
                        break
                lines.insert(insert_idx, typing_line)

        # Add other imports
        for imp in other_imports:
            if not any(imp in line for line in lines):
                lines.insert(import_end, imp)

        # Write back
        path.write_text("\n".join(lines) + "\n")
        return True

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    """Main function."""
    print("Getting F821 errors...")
    output = run_ruff()

    file_errors = parse_errors(output)
    print(f"Found {len(file_errors)} files with F821 errors")

    fixed = 0
    for filepath, undefined_names in file_errors.items():
        print(f"Fixing {filepath}: {sorted(undefined_names)}")
        if fix_file(filepath, undefined_names):
            fixed += 1

    print(f"\nFixed {fixed} files. Running ruff again...")

    # Check remaining errors
    new_output = run_ruff()
    remaining = len([line for line in new_output.split("\n") if "F821" in line])
    print(f"Remaining F821 errors: {remaining}")

    if remaining > 0:
        print("First 10 remaining errors:")
        error_lines = [line for line in new_output.split("\n") if "F821" in line][:10]
        for line in error_lines:
            print(line)


if __name__ == "__main__":
    main()
