#!/usr/bin/env python3
"""Fix test function type annotations specifically.

This script focuses on test functions which have a very predictable pattern:
- Test functions always return None
- Test fixture functions also typically return None or specific types
"""

import logging
import re
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP")

def fix_test_file_annotations(file_path: Path) -> tuple[bool, int]:
    """Fix type annotations in test files."""
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = 0

        # Fix test functions (always return None)
        test_function_pattern = re.compile(
            r'^(\s*)(async\s+def\s+test_\w+\s*\([^)]*\))\s*:\s*$',
            re.MULTILINE
        )

        def replace_test_function(match: Any) -> None:
            nonlocal changes_made
            indent = match.group(1)
            func_def = match.group(2)
            changes_made += 1
            return f"{indent}{func_def} -> None:"

        content = test_function_pattern.sub(replace_test_function, content)

        # Fix regular test functions (non-async)
        test_function_pattern2 = re.compile(
            r'^(\s*)(def\s+test_\w+\s*\([^)]*\))\s*:\s*$',
            re.MULTILINE
        )

        content = test_function_pattern2.sub(replace_test_function, content)

        # Fix fixture functions
        fixture_pattern = re.compile(
            r'^(\s*)(@pytest\.fixture[^\n]*\n\s*)(def\s+\w+\s*\([^)]*\))\s*:\s*$',
            re.MULTILINE
        )

        def replace_fixture_function(match: Any) -> Any:
            nonlocal changes_made
            indent = match.group(1)
            decorator = match.group(2)
            func_def = match.group(3)

            # Most fixtures return Any, but some are more specific
            return_type = "-> Any:" if 'mock_' in func_def else "-> Any:"

            changes_made += 1
            return f"{indent}{decorator}{func_def} {return_type}"

        content = fixture_pattern.sub(replace_fixture_function, content)

        # Fix setup/teardown methods
        setup_pattern = re.compile(
            r'^(\s*)(def\s+(?:setup_|teardown_|setUp|tearDown)\w*\s*\([^)]*\))\s*:\s*$',
            re.MULTILINE
        )

        def replace_setup_function(match: Any) -> None:
            nonlocal changes_made
            indent = match.group(1)
            func_def = match.group(2)
            changes_made += 1
            return f"{indent}{func_def} -> None:"

        content = setup_pattern.sub(replace_setup_function, content)

        # Check if we need to add typing imports
        if changes_made > 0:
            has_typing_import = any(import_line in content for import_line in [
                "from typing import", "import typing", "from __future__ import annotations"
            ])

            if not has_typing_import:
                # Find the best place to add typing import
                import_pattern = re.search(r'((?:^(?:from|import)\s+[^\n]+\n)+)', content, re.MULTILINE)
                if import_pattern:
                    imports_end = import_pattern.end()
                    content = content[:imports_end] + "from typing import Any\n" + content[imports_end:]
                else:
                    # Add at the beginning after docstring
                    docstring_end = 0
                    if content.startswith('"""') or content.startswith("'''"):
                        quote_type = '"""' if content.startswith('"""') else "'''"
                        end_pos = content.find(quote_type, 3)
                        if end_pos != -1:
                            docstring_end = end_pos + 3

                    imports = "\nfrom typing import Any\n"
                    content = content[:docstring_end] + imports + content[docstring_end:]

        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes_made

        return False, 0

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return False, 0

def main() -> None:
    """Main execution function."""
    logger.info("Starting test file type annotation fixing...")

    # Process test files
    test_files = list(PROJECT_ROOT.glob("tests/**/*.py"))

    files_modified = 0
    total_changes = 0

    for test_file in test_files:
        if test_file.is_file():
            try:
                modified, changes = fix_test_file_annotations(test_file)
                if modified:
                    files_modified += 1
                    total_changes += changes
                    logger.info(f"Fixed {changes} annotations in {test_file.relative_to(PROJECT_ROOT)}")
            except Exception as e:
                logger.error(f"Error processing {test_file}: {e}")

    logger.info("Test file annotation fix completed!")
    logger.info(f"Files modified: {files_modified}")
    logger.info(f"Total annotations added: {total_changes}")

if __name__ == "__main__":
    main()
