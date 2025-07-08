#!/usr/bin/env python3
"""
Fix S311 violations by adding proper noqa comments for ML/analytics random usage.
Simple approach - target specific lines based on ruff output.
"""

from pathlib import Path


def fix_s311_simple():
    """Fix S311 violations by adding proper noqa comments."""

    # Specific line fixes based on ruff output
    fixes = [
        # predictive_analytics_tools.py fixes
        {
            "file": "src/server/tools/predictive_analytics_tools.py",
            "line_fixes": [
                (
                    1200,
                    "        random.uniform(10, 100)  "
                    "# noqa: S311 # ML/analytics data simulation",
                ),
                (
                    1216,
                    "        random.uniform(1.0, 5.0)  "
                    "# noqa: S311 # ML/analytics data simulation",
                ),
                (
                    1243,
                    "            random.uniform(20, 80)  "
                    "# noqa: S311 # ML/analytics data simulation",
                ),
                (
                    1248,
                    "            random.uniform(30, 90)  "
                    "# noqa: S311 # ML/analytics data simulation",
                ),
                (
                    1253,
                    "            random.uniform(40, 95)  "
                    "# noqa: S311 # ML/analytics data simulation",
                ),
                (
                    1258,
                    "            random.uniform(10, 60)  "
                    "# noqa: S311 # ML/analytics data simulation",
                ),
            ],
        },
        # object_detector.py fixes
        {
            "file": "src/vision/object_detector.py",
            "line_fixes": [
                (
                    332,
                    "            random.randint(1, 8),  "
                    "# noqa: S311 # ML/analytics data simulation",
                ),
            ],
        },
        # scene_analyzer.py fixes
        {
            "file": "src/vision/scene_analyzer.py",
            "line_fixes": [
                (
                    625,
                    "          palette_size = random.randint(  "
                    "# noqa: S311 # ML/analytics data simulation",
                ),
                (
                    642,
                    "          avg_saturation = random.uniform(  "
                    "# noqa: S311 # ML/analytics data simulation",
                ),
                (
                    645,
                    "          avg_brightness = random.uniform(  "
                    "# noqa: S311 # ML/analytics data simulation",
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
            lines = f.readlines()

        # Apply line-specific fixes
        changed = False
        for line_num, new_content in fix_info["line_fixes"]:
            if line_num <= len(lines):
                old_line = lines[line_num - 1].rstrip()
                if "# noqa: S311" not in old_line:
                    # Replace with new content
                    lines[line_num - 1] = new_content + "\n"
                    changed = True
                    print(f"  ✅ Fixed line {line_num}")
                else:
                    print(f"  ℹ️  Line {line_num} already has noqa comment")

        # Write back if changed
        if changed:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"✅ Updated {file_path}")
        else:
            print(f"ℹ️  No changes needed in {file_path}")


if __name__ == "__main__":
    fix_s311_simple()
    print("\n🎉 S311 violation fixes completed!")
