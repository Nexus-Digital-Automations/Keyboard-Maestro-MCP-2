#!/usr/bin/env python3
"""Fix malformed imports in docstrings across all files."""

import re
import subprocess
from pathlib import Path


def fix_docstring_imports():
    """Fix imports that were incorrectly placed in docstrings."""
    # Get all Python files
    result = subprocess.run(
        ["/usr/bin/find", ".", "-name", "*.py", "-type", "f"],
        check=False,
        capture_output=True,
        text=True,
        cwd="/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
    )

    python_files = [
        f.strip()[2:] for f in result.stdout.split("\n") if f.strip().endswith(".py")
    ]

    fixed_count = 0

    for filepath in python_files:
        path = (
            Path(
                "/Users/jeremyparker/Desktop/Claude Coding Projects/"
                "Keyboard-Maestro-MCP",
            )
            / filepath
        )
        if not path.exists():
            continue

        try:
            content = path.read_text()
            original_content = content

            # Find imports in docstrings and move them
            patterns = [
                r'("""[^"]*?)from fastmcp\.utilities import Context([^"]*?""")',
                r'("""[^"]*?)from src\.core\.either import Either([^"]*?""")',
                r'("""[^"]*?)from pathlib import Path([^"]*?""")',
            ]

            imports_to_add = []

            for pattern in patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                for _match in matches:
                    # Extract the import
                    if "Context" in pattern or "Either" in pattern or "Path" in pattern:
                        imports_to_add.append("")
                        content = re.sub(pattern, r"\1\2", content, flags=re.DOTALL)

            # Also handle simpler cases where import is on its own line in docstring
            lines = content.splitlines()
            modified_lines = []
            in_docstring = False
            docstring_quote = None

            for _i, line in enumerate(lines):
                if not in_docstring:
                    if line.strip().startswith('"""') or line.strip().startswith("'''"):
                        in_docstring = True
                        docstring_quote = '"""' if '"""' in line else "'''"
                        modified_lines.append(line)
                        # Check if docstring ends on same line
                        if line.count(docstring_quote) >= 2:
                            in_docstring = False
                    else:
                        modified_lines.append(line)
                # We're in a docstring
                elif docstring_quote in line and not (
                    line.strip().startswith("from ")
                    or line.strip().startswith("import ")
                ):
                    in_docstring = False
                    modified_lines.append(line)
                elif (
                    line.strip().startswith("from fastmcp import Context")
                    or line.strip().startswith("from src.core.either import Either")
                    or line.strip().startswith("from pathlib import Path")
                ):
                    # Remove import from docstring
                    if "Context" in line:
                        imports_to_add.append("from fastmcp import Context")
                    elif "Either" in line:
                        imports_to_add.append("from src.core.either import Either")
                    elif "Path" in line:
                        imports_to_add.append("from pathlib import Path")
                    # Don't add this line
                    continue
                else:
                    modified_lines.append(line)

            if imports_to_add:
                content = "\n".join(modified_lines) + "\n"

                # Remove duplicates
                imports_to_add = list(set(imports_to_add))

                # Add imports in proper location
                lines = content.splitlines()
                insert_idx = None

                for i, line in enumerate(lines):
                    if line.startswith("from __future__"):
                        insert_idx = i + 2
                        break
                    if line.strip() and (
                        line.startswith("import ") or line.startswith("from ")
                    ):
                        insert_idx = i
                        break

                if insert_idx is not None:
                    for imp in reversed(imports_to_add):
                        # Check if import already exists
                        if not any(imp in existing_line for existing_line in lines):
                            lines.insert(insert_idx, imp)

                    content = "\n".join(lines) + "\n"

            # Write back if changed
            if content != original_content:
                path.write_text(content)
                fixed_count += 1
                print(f"Fixed {filepath}")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print(f"Fixed {fixed_count} files with docstring import issues")


def main():
    """Main function."""
    print("Fixing docstring import issues...")
    fix_docstring_imports()

    # Check final F821 count
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821"],
        check=False,
        capture_output=True,
        text=True,
        cwd="/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
    )

    remaining = len([line for line in result.stdout.split("\n") if "F821" in line])
    print(f"\nFinal F821 error count: {remaining}")

    if remaining < 50:  # Show all if less than 50
        error_lines = [line for line in result.stdout.split("\n") if "F821" in line]
        for line in error_lines:
            print(line)
    elif remaining > 0:
        print("First 20 remaining errors:")
        error_lines = [line for line in result.stdout.split("\n") if "F821" in line][
            :20
        ]
        for line in error_lines:
            print(line)


if __name__ == "__main__":
    main()
