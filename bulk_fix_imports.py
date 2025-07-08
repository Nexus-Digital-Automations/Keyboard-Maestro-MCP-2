#!/usr/bin/env python3
"""Bulk fix remaining F821 errors by pattern."""

import subprocess
from pathlib import Path

# Constants for magic value comparison fixes
DOCSTRING_SEARCH_LIMIT = 10  # Lines to search within for docstring patterns


def main():
    """Fix the remaining F821 errors."""
    # Fix server_tools files that have malformed imports
    server_tools_files = [
        "tests/server_tools/test_advanced_extensions_tools.py",
        "tests/server_tools/test_enterprise_features_tools.py",
        "tests/server_tools/test_foundation_tools_coverage.py",
        "tests/server_tools/test_high_impact_tools.py",
        "tests/server_tools/test_intelligent_automation_tools.py",
        "tests/server_tools/test_macro_creation_tools.py",
        "tests/server_tools/test_platform_expansion_tools.py",
    ]

    tools_files = [
        "tests/tools/test_accessibility_engine_tools.py",
        "tests/tools/test_analytics_engine_tools.py",
        "tests/tools/test_api_orchestration_tools.py",
        "tests/tools/test_computer_vision_tools_backup.py",
        "tests/tools/test_developer_toolkit_tools.py",
        "tests/tools/test_enterprise_sync_tools.py",
        "tests/tools/test_knowledge_management_tools.py",
        "tests/tools/test_macro_editor_tools.py",
        "tests/tools/test_natural_language_tools.py",
        "tests/tools/test_performance_monitor_tools.py",
        "tests/tools/test_plugin_ecosystem_tools.py",
        "tests/tools/test_quantum_ready_tools.py",
        "tests/tools/test_testing_automation_tools.py",
        "tests/tools/test_user_identity_tools.py",
        "tests/tools/test_visual_automation_tools.py",
        "tests/tools/test_voice_control_tools.py",
        "tests/tools/test_workflow_designer_tools.py",
        "tests/tools/test_workflow_intelligence_tools.py",
        "tests/tools/test_zero_trust_security_tools.py",
    ]

    # Fix files with misplaced imports in docstrings
    for filepath in server_tools_files:
        path = Path(filepath)
        if path.exists():
            content = path.read_text()

            # Fix malformed import in docstring
            if "from fastmcp import Context" in content and '"""' in content:
                # Move the import out of the docstring
                lines = content.splitlines()

                # Find and fix the docstring
                for i, line in enumerate(lines):
                    if (
                        "from fastmcp import Context" in line
                        and i < DOCSTRING_SEARCH_LIMIT
                    ):  # Within first 10 lines (likely in docstring)
                        # Remove the import from the docstring
                        lines[i] = line.replace(
                            "from fastmcp import Context",
                            "",
                        ).strip()
                        if not lines[i]:  # If line is now empty, remove it
                            lines.pop(i)
                        break

                # Add the import in the proper place
                if (
                    "from fastmcp import Context" not in content.split('"""')[2]
                ):  # Not in the imports section
                    # Find where to add the import
                    for i, line in enumerate(lines):
                        if line.startswith("from __future__"):
                            lines.insert(i + 2, "from fastmcp import Context")
                            break
                        if (
                            line.startswith("import ") or line.startswith("from ")
                        ) and "fastmcp.utilities" not in line:
                            lines.insert(i, "from fastmcp import Context")
                            break

                path.write_text("\n".join(lines) + "\n")
                print(f"Fixed {filepath}")

    # Fix tools files that need Both Context and Either imports
    for filepath in tools_files:
        path = Path(filepath)
        if path.exists():
            content = path.read_text()

            # Check if missing imports
            needs_context = (
                "Context" in content and "from fastmcp import Context" not in content
            )
            needs_either = (
                "Either" in content
                and "from src.core.either import Either" not in content
            )

            if needs_context or needs_either:
                lines = content.splitlines()

                # Fix malformed imports in docstring first
                for i, line in enumerate(lines):
                    if (
                        "from src.core.either import Either" in line
                        or "from fastmcp import Context" in line
                    ) and i < DOCSTRING_SEARCH_LIMIT:
                        lines[i] = (
                            line.replace("from src.core.either import Either", "")
                            .replace("from fastmcp import Context", "")
                            .strip()
                        )
                        if not lines[i]:
                            lines.pop(i)
                        break

                # Add imports in proper location
                insert_idx = None
                for i, line in enumerate(lines):
                    if line.startswith("from __future__"):
                        insert_idx = i + 2
                        break
                    if line.startswith("import ") or line.startswith("from "):
                        insert_idx = i
                        break

                if insert_idx is not None:
                    if needs_context:
                        lines.insert(insert_idx, "from fastmcp import Context")
                    if needs_either:
                        lines.insert(insert_idx, "from src.core.either import Either")

                    path.write_text("\n".join(lines) + "\n")
                    print(f"Fixed {filepath}")

    # Check remaining errors
    result = subprocess.run(
        ["/opt/homebrew/bin/ruff", "check", "--select=F821"],
        check=False,
        capture_output=True,
        text=True,
        cwd="/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP",
    )

    remaining = len([line for line in result.stdout.split("\n") if "F821" in line])
    print(f"\nRemaining F821 errors: {remaining}")

    if remaining > 0:
        print("First 10 remaining errors:")
        error_lines = [line for line in result.stdout.split("\n") if "F821" in line][
            :10
        ]
        for line in error_lines:
            print(line)


if __name__ == "__main__":
    main()
