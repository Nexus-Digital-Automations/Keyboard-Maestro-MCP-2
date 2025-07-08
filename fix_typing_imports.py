#!/usr/bin/env python3
"""Fix missing typing imports for F821 errors."""

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


def get_f821_errors():
    """Get all F821 errors from ruff."""
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821", "--output-format=json"],
        check=False,
        capture_output=True,
        text=True,
        cwd="/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
    )

    if result.returncode != 0:
        print("Error running ruff:", result.stderr)
        return []

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Error parsing ruff output")
        return []


def analyze_missing_imports():
    """Analyze which files need which imports."""
    errors = get_f821_errors()
    file_imports = defaultdict(set)

    for error in errors:
        if error["code"] == "F821":
            # Extract the undefined name from the message
            match = re.search(r"Undefined name `([^`]+)`", error["message"])
            if match:
                undefined_name = match.group(1)
                filename = error["filename"]

                # Map undefined names to their imports
                typing_imports = {
                    "Any": "Any",
                    "Callable": "Callable",
                    "Awaitable": "Awaitable",
                    "Optional": "Optional",
                    "Union": "Union",
                    "List": "List",
                    "Dict": "Dict",
                    "Tuple": "Tuple",
                    "Set": "Set",
                    "TypeVar": "TypeVar",
                    "Generic": "Generic",
                    "Protocol": "Protocol",
                    "ClassVar": "ClassVar",
                    "Final": "Final",
                    "Literal": "Literal",
                    "overload": "overload",
                    "TYPE_CHECKING": "TYPE_CHECKING",
                }

                if undefined_name in typing_imports:
                    file_imports[filename].add(typing_imports[undefined_name])
                elif undefined_name == "Context":
                    # Context is likely from MCP or another library
                    file_imports[filename].add("Context")
                elif undefined_name == "Either":
                    # Either is likely from a custom core module
                    file_imports[filename].add("Either")
                elif undefined_name == "Path":
                    # Path is from pathlib
                    file_imports[filename].add("Path")

    return file_imports


def fix_file_imports(filename, needed_imports):
    """Fix imports in a single file."""
    path = Path(filename)
    if not path.exists():
        print(f"File not found: {filename}")
        return False

    content = path.read_text()
    lines = content.splitlines()

    # Find where to insert imports
    insert_line = 0
    has_future_import = False

    for i, line in enumerate(lines):
        if line.startswith("from __future__ import"):
            has_future_import = True
            insert_line = i + 1
        elif line.startswith("import ") or line.startswith("from "):
            if not has_future_import:
                insert_line = i
                break
        elif line.strip() and not line.startswith("#"):
            # First non-comment, non-import line
            insert_line = i
            break

    # Check if typing import already exists
    existing_typing_imports = set()
    typing_import_line = None

    for i, line in enumerate(lines):
        if line.startswith("from typing import"):
            typing_import_line = i
            # Extract existing imports
            imports_part = line.split("import", 1)[1].strip()
            if imports_part.startswith("("):
                # Multi-line import
                imports_part = imports_part.strip("()")
                # Look for continuation lines
                j = i + 1
                while j < len(lines) and not lines[j].strip().endswith(")"):
                    imports_part += " " + lines[j].strip()
                    j += 1
                if j < len(lines):
                    imports_part += " " + lines[j].strip(")")

            # Parse imports
            for imp in imports_part.split(","):
                existing_typing_imports.add(imp.strip())
            break

    # Separate typing imports from other imports
    typing_imports = set()
    other_imports = set()

    for imp in needed_imports:
        if imp in [
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
            typing_imports.add(imp)
        else:
            other_imports.add(imp)

    # Add missing typing imports
    new_typing_imports = typing_imports - existing_typing_imports

    if new_typing_imports:
        all_typing_imports = sorted(existing_typing_imports | new_typing_imports)
        new_import_line = f"from typing import {', '.join(all_typing_imports)}"

        if typing_import_line is not None:
            # Replace existing typing import
            lines[typing_import_line] = new_import_line
        else:
            # Add new typing import
            lines.insert(insert_line, new_import_line)

    # Handle other imports (Context, Either, Path)
    for imp in other_imports:
        if imp == "Context":
            # Add Context import if not already present
            context_import = "from mcp.types import Context"
            if not any(context_import in line for line in lines):
                lines.insert(insert_line, context_import)
        elif imp == "Either":
            # Add Either import if not already present
            either_import = "from src.core.either import Either"
            if not any("Either" in line and "import" in line for line in lines):
                lines.insert(insert_line, either_import)
        elif imp == "Path":
            # Add Path import if not already present
            path_import = "from pathlib import Path"
            if not any(path_import in line for line in lines):
                lines.insert(insert_line, path_import)

    # Write back to file
    path.write_text("\n".join(lines) + "\n")
    return True


def main():
    """Main function to fix all typing imports."""
    print("Analyzing F821 errors...")
    file_imports = analyze_missing_imports()

    print(f"Found {len(file_imports)} files needing import fixes")

    fixed_files = 0
    for filename, imports in file_imports.items():
        print(f"Fixing {filename}: {', '.join(sorted(imports))}")
        if fix_file_imports(filename, imports):
            fixed_files += 1

    print(f"Fixed imports in {fixed_files} files")

    # Run ruff again to check remaining errors
    print("\nRunning ruff check again...")
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821"],
        check=False,
        capture_output=True,
        text=True,
        cwd="/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
    )

    if result.returncode == 0:
        print("✅ All F821 errors fixed!")
    else:
        remaining_errors = result.stdout.count("F821")
        print(f"❌ {remaining_errors} F821 errors remain")
        print("First few remaining errors:")
        print(result.stdout[:1000])


if __name__ == "__main__":
    main()
