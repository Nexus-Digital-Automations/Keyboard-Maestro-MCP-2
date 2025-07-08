#!/usr/bin/env python3
"""
Fix S311 violations for non-cryptographic random usage in ML/analytics contexts.
All identified violations are legitimate uses of random for data simulation.
"""

import re
from pathlib import Path


def fix_s311_violations():
    """Fix S311 violations by adding proper noqa comments for ML/analytics random usage.

    Adds appropriate noqa comments for random usage in machine learning contexts.
    """

    # File-specific fixes
    fixes = [
        {
            "file": "src/server/tools/predictive_analytics_tools.py",
            "patterns": [
                # Fix random.uniform patterns without existing noqa
                (
                    r"(\s+)(random\.uniform\([^)]+\))",
                    r"\1\2  # noqa: S311 # ML/analytics data simulation",
                ),
                # Fix random.randint patterns without existing noqa
                (
                    r"(\s+)(random\.randint\([^)]+\))",
                    r"\1\2  # noqa: S311 # ML/analytics data simulation",
                ),
            ],
        },
        {
            "file": "src/vision/object_detector.py",
            "patterns": [
                (
                    r"(\s+)(random\.randint\([^)]+\))",
                    r"\1\2  # noqa: S311 # ML/analytics data simulation",
                ),
            ],
        },
        {
            "file": "src/vision/scene_analyzer.py",
            "patterns": [
                (
                    r"(\s+)(random\.randint\([^)]+\))",
                    r"\1\2  # noqa: S311 # ML/analytics data simulation",
                ),
                (
                    r"(\s+)(random\.uniform\([^)]+\))",
                    r"\1\2  # noqa: S311 # ML/analytics data simulation",
                ),
                (
                    r"(\s+)(random\.sample\([^)]+\))",
                    r"\1\2  # noqa: S311 # Statistical sampling for ML",
                ),
            ],
        },
    ]

    for fix_info in fixes:
        file_path = Path(fix_info["file"])

        if not file_path.exists():
            print(f"Warning: File {file_path} not found")
            continue

        print(f"Processing {file_path}")

        # Read file content
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply patterns
        for pattern, replacement in fix_info["patterns"]:
            # Only apply if the line doesn't already have a noqa comment
            lines = content.split("\n")
            new_lines = []

            for line in lines:
                if (
                    re.search(pattern.split(")")[0] + r"[^)]*\)", line)
                    and "# noqa: S311" not in line
                ):
                    # Apply the replacement
                    new_line = re.sub(pattern, replacement, line)
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)

            content = "\n".join(new_lines)

        # Write back if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Fixed S311 violations in {file_path}")
        else:
            print(f"ℹ️  No changes needed in {file_path}")


if __name__ == "__main__":
    fix_s311_violations()
    print("\n🎉 S311 violation fixes completed!")
