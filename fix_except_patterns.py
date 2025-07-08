#!/usr/bin/env python3
"""Bulk Exception Logging Fix Script.

This script systematically replaces bare except patterns with proper exception
handling and logging across the codebase.

Patterns fixed:
1. except: pass -> except SpecificException as e: logger.debug(...)
2. except: continue -> except SpecificException as e: logger.debug(...); continue
3. except: return None -> except SpecificException as e: logger.debug(...); return None
"""

import logging
import re
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for the project - target entire project including tests
PROJECT_ROOT = Path(
    "/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
)

# Common exception mappings for different contexts
EXCEPTION_MAPPINGS = {
    # File operations
    "file_ops": "(OSError, FileNotFoundError, PermissionError)",
    # Network/HTTP operations
    "network": "(requests.RequestException, ConnectionError, TimeoutError)",
    # JSON/data processing
    "json_ops": "(json.JSONDecodeError, KeyError, ValueError)",
    # Pickle operations
    "pickle_ops": "(pickle.PicklingError, pickle.UnpicklingError)",
    # Type conversions
    "type_ops": "(ValueError, TypeError)",
    # General I/O
    "io_ops": "(OSError, IOError, PermissionError)",
    # Import/module operations
    "import_ops": "(ImportError, ModuleNotFoundError)",
    # Generic fallback
    "generic": "(Exception)",
}


def detect_context(code_block: str) -> str:
    """Detect the context of the exception to choose appropriate exception types."""
    code_lower = code_block.lower()

    if any(
        keyword in code_lower
        for keyword in ["open(", "file", "path", ".read(", ".write("]
    ):
        return "file_ops"
    if any(keyword in code_lower for keyword in ["requests.", "http", "url", "fetch"]):
        return "network"
    if any(keyword in code_lower for keyword in ["json.", ".loads(", ".dumps("]):
        return "json_ops"
    if any(keyword in code_lower for keyword in ["pickle.", ".loads(", ".dumps("]):
        return "pickle_ops"
    if any(keyword in code_lower for keyword in ["int(", "float(", "str(", "bool("]):
        return "type_ops"
    if any(keyword in code_lower for keyword in ["import ", "from "]):
        return "import_ops"
    return "generic"


def generate_log_message(context: str, action: str) -> str:
    """Generate appropriate log message based on context."""
    messages = {
        "file_ops": f"File operation failed during {action}",
        "network": f"Network operation failed during {action}",
        "json_ops": f"JSON processing failed during {action}",
        "pickle_ops": f"Pickle operation failed during {action}",
        "type_ops": f"Type conversion failed during {action}",
        "io_ops": f"I/O operation failed during {action}",
        "import_ops": f"Import failed during {action}",
        "generic": f"Operation failed during {action}",
    }
    return messages.get(context, f"Operation failed during {action}")


def fix_except_patterns(file_path: Path) -> tuple[bool, int]:
    """Fix except patterns in a single file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        changes_made = 0

        # Pattern 1: except Exception: followed by pass (with optional comments)
        # This matches the actual S110 patterns found in ruff output
        # Handle multi-line comments between except and pass
        pattern1 = re.compile(
            r"(\s+)except\s+Exception\s*:\s*(?:\n\s*#[^\n]*)*\n(\s+)pass(?:\s*#[^\n]*)?\s*(?:\n|$)",
            re.MULTILINE,
        )

        def replace_except_exception_pass(match: re.Match[str]) -> str:
            nonlocal changes_made
            indent = match.group(1)
            pass_indent = match.group(2)

            # Get some context around the match
            start_pos = max(0, match.start() - 500)
            context_code = content[start_pos : match.start()]
            context_type = detect_context(context_code)
            exceptions = EXCEPTION_MAPPINGS[context_type]

            # Generate action description from function context
            func_match = re.search(r"def\s+(\w+)", context_code[::-1])
            action = func_match.group(1)[::-1] if func_match else "operation"

            log_msg = generate_log_message(context_type, action)

            changes_made += 1
            return (
                f"{indent}except {exceptions} as e:\n"
                f'{pass_indent}logger.debug(f"{log_msg}: {{e}}")\n'
            )

        content = pattern1.sub(replace_except_exception_pass, content)

        # Pattern 2: except: pass (bare except)
        # Handle multi-line comments between except and pass
        pattern2 = re.compile(
            r"(\s+)except\s*:\s*(?:\n\s*#[^\n]*)*\n(\s+)pass(?:\s*#[^\n]*)?\s*(?:\n|$)",
            re.MULTILINE,
        )

        def replace_bare_except_pass(match: re.Match[str]) -> str:
            nonlocal changes_made
            indent = match.group(1)
            pass_indent = match.group(2)

            # Get some context around the match
            start_pos = max(0, match.start() - 500)
            context_code = content[start_pos : match.start()]
            context_type = detect_context(context_code)
            exceptions = EXCEPTION_MAPPINGS[context_type]

            # Generate action description from function context
            func_match = re.search(r"def\s+(\w+)", context_code[::-1])
            action = func_match.group(1)[::-1] if func_match else "operation"

            log_msg = generate_log_message(context_type, action)

            changes_made += 1
            return (
                f"{indent}except {exceptions} as e:\n"
                f'{pass_indent}logger.debug(f"{log_msg}: {{e}}")\n'
            )

        content = pattern2.sub(replace_bare_except_pass, content)

        # Pattern 3: except: continue (bare except with continue)
        pattern3 = re.compile(
            r"(\s+)except\s*:\s*(?:\n\s*#[^\n]*)?\n(\s+)continue(?:\s*#[^\n]*)?\s*(?:\n|$)",
            re.MULTILINE,
        )

        def replace_except_continue(match: re.Match[str]) -> str:
            nonlocal changes_made
            indent = match.group(1)
            continue_indent = match.group(2)

            start_pos = max(0, match.start() - 500)
            context_code = content[start_pos : match.start()]
            context_type = detect_context(context_code)
            exceptions = EXCEPTION_MAPPINGS[context_type]

            func_match = re.search(r"def\s+(\w+)", context_code[::-1])
            action = func_match.group(1)[::-1] if func_match else "operation"

            log_msg = generate_log_message(context_type, action)

            changes_made += 1
            return (
                f"{indent}except {exceptions} as e:\n"
                f'{continue_indent}logger.debug(f"{log_msg}: {{e}}")\n'
                f"{continue_indent}continue\n"
            )

        content = pattern3.sub(replace_except_continue, content)

        # Pattern 4: except Exception: continue
        pattern4 = re.compile(
            r"(\s+)except\s+Exception\s*:\s*(?:\n\s*#[^\n]*)?\n(\s+)continue(?:\s*#[^\n]*)?\s*(?:\n|$)",
            re.MULTILINE,
        )

        def replace_except_exception_continue(match: re.Match[str]) -> str:
            nonlocal changes_made
            indent = match.group(1)
            continue_indent = match.group(2)

            start_pos = max(0, match.start() - 500)
            context_code = content[start_pos : match.start()]
            context_type = detect_context(context_code)
            exceptions = EXCEPTION_MAPPINGS[context_type]

            func_match = re.search(r"def\s+(\w+)", context_code[::-1])
            action = func_match.group(1)[::-1] if func_match else "operation"

            log_msg = generate_log_message(context_type, action)

            changes_made += 1
            return (
                f"{indent}except {exceptions} as e:\n"
                f'{continue_indent}logger.debug(f"{log_msg}: {{e}}")\n'
                f"{continue_indent}continue\n"
            )

        content = pattern4.sub(replace_except_exception_continue, content)

        # Only write if changes were made
        if changes_made > 0:
            # Check if logger import exists
            if "import logging" not in content and "from logging import" not in content:
                # Find imports section and add logger import
                import_pattern = re.search(
                    r"((?:^(?:from|import)\s+[^\n]+\n)+)",
                    content,
                    re.MULTILINE,
                )
                if import_pattern:
                    imports_end = import_pattern.end()
                    # Add logger setup after imports
                    logger_setup = (
                        "\nimport logging\n\nlogger = logging.getLogger(__name__)\n"
                    )
                    content = (
                        content[:imports_end] + logger_setup + content[imports_end:]
                    )
                else:
                    # Add at the beginning of the file after docstring
                    docstring_end = 0
                    if content.startswith('"""') or content.startswith("'''"):
                        quote_type = '"""' if content.startswith('"""') else "'''"
                        end_pos = content.find(quote_type, 3)
                        if end_pos != -1:
                            docstring_end = end_pos + 3

                    logger_setup = (
                        "\nimport logging\n\nlogger = logging.getLogger(__name__)\n"
                    )
                    content = (
                        content[:docstring_end] + logger_setup + content[docstring_end:]
                    )

            # Check if logger is defined but not imported properly
            elif "logger = logging.getLogger(__name__)" not in content:
                # Find imports section and add logger definition
                import_pattern = re.search(
                    r"((?:^(?:from|import)\s+[^\n]+\n)+)",
                    content,
                    re.MULTILINE,
                )
                if import_pattern:
                    imports_end = import_pattern.end()
                    logger_setup = "\nlogger = logging.getLogger(__name__)\n"
                    content = (
                        content[:imports_end] + logger_setup + content[imports_end:]
                    )

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return True, changes_made

        return False, 0

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return False, 0


def process_directory(directory: Path) -> tuple[int, int]:
    """Process all Python files in directory."""
    files_modified = 0
    total_changes = 0

    for py_file in directory.rglob("*.py"):
        if py_file.is_file():
            try:
                modified, changes = fix_except_patterns(py_file)
                if modified:
                    files_modified += 1
                    total_changes += changes
                    logger.info(
                        f"Fixed {changes} patterns in "
                        f"{py_file.relative_to(PROJECT_ROOT)}",
                    )
            except Exception as e:
                logger.error(f"Error processing {py_file}: {e}")

    return files_modified, total_changes


def main() -> None:
    """Main execution function."""
    logger.info("Starting bulk exception pattern fixing...")
    logger.info(f"Processing directory: {PROJECT_ROOT}")

    if not PROJECT_ROOT.exists():
        logger.error(f"Project root does not exist: {PROJECT_ROOT}")
        return

    files_modified, total_changes = process_directory(PROJECT_ROOT)

    logger.info("Bulk fix completed!")
    logger.info(f"Files modified: {files_modified}")
    logger.info(f"Total patterns fixed: {total_changes}")

    if files_modified > 0:
        logger.info("Remember to:")
        logger.info("1. Run tests to ensure functionality is preserved")
        logger.info("2. Review the changes for any context-specific adjustments needed")
        logger.info("3. Commit the changes with a descriptive message")


if __name__ == "__main__":
    main()
