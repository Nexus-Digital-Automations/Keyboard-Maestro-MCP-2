#!/usr/bin/env python3
"""Drive KM editor to capture canonical action <dict> XML, one entry per type.

For every action in the Edit -> Insert Action -> By Name menu, this script
opens the search dialog, types the action name, presses return to insert,
reads `xml of action 1`, deletes the inserted action, and stores the
captured plist keyed by MacroActionType. The output JSON is the trusted
source for km_action_builder's template-based emitter.

USAGE
    python scripts/capture_km_action_templates.py \\
        --macro-id <SCRATCH_MACRO_UUID> \\
        [--out src/server/data/km_action_templates.json] \\
        [--categories Image Text Variables] \\
        [--limit 5]

REQUIREMENTS
    - Keyboard Maestro editor must be installable/runnable
    - Accessibility permission for osascript (System Settings > Privacy)
    - A scratch macro must already exist; pass its UUID as --macro-id.
      This script will append + delete actions in that macro, leaving it
      empty after a successful run.

SAFETY
    - The scratch macro is cleared at start and end.
    - Actions whose XML comes back as KM's "Invalid XML From AppleScript"
      Log placeholder are skipped (KM signals 'I didn't understand that
      action name').
    - Capture is idempotent: re-running merges with whatever is already
      in the output JSON; never overwrites an existing entry.

@stable
"""

from __future__ import annotations

import argparse
import json
import plistlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Categories that ship as submenus under Edit > Insert Action. Order matters
# only for output stability — captures are keyed by MacroActionType, not by
# discovery order.
CATEGORIES_DEFAULT = (
    "Application Control",
    "Clipboard",
    "Control Flow",
    "Debugger",
    "Execute",
    "File",
    "Front Browser Control",
    "Image",
    "Interface Control",
    "Keyboard Maestro",
    "Microsoft Edge Control",
    "MIDI",
    "Music Control",
    "Notifications",
    "Open",
    "QuickTime Player Control",
    "Safari Control",
    "Stream Deck Control",
    "Switchers",
    "System Control",
    "Text",
    "Variables",
    "Web",
)

_PLACEHOLDER_TEXT = "Invalid XML From AppleScript"


def _osascript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"osascript failed: {result.stderr.strip()}")
    return result.stdout.strip()


def list_category(category: str) -> list[str]:
    """Return the action labels listed under one Insert Action submenu."""
    quoted = category.replace('"', '\\"')
    raw = _osascript(
        f'''tell application "System Events"
            tell process "Keyboard Maestro"
                return name of every menu item of menu "{quoted}" of menu item "{quoted}" of menu "Insert Action" of menu item "Insert Action" of menu "Edit" of menu bar 1
            end tell
        end tell''',
    )
    # AppleScript returns a comma+space-separated list; entries prefixed with
    # "Help: " are accessibility-help twins that share a slot with the actual
    # action label, not separate actions.
    parts = [p.strip() for p in raw.split(", ")]
    return [p for p in parts if p and not p.startswith("Help:") and p != "missing value"]


def open_by_name(macro_id: str) -> None:
    _osascript(
        f'''tell application "Keyboard Maestro"
            activate
            editMacro "{macro_id}"
        end tell
        delay 0.4
        tell application "System Events"
            tell process "Keyboard Maestro"
                click menu item "By Name…" of menu "Insert Action" of menu item "Insert Action" of menu "Edit" of menu bar 1
            end tell
        end tell
        delay 0.5''',
    )


def insert_by_name(action_name: str) -> None:
    quoted = action_name.replace('"', '\\"')
    _osascript(
        f'''tell application "System Events"
            keystroke "{quoted}"
        end tell
        delay 0.3
        tell application "System Events"
            keystroke return
        end tell
        delay 0.4''',
    )


def read_first_action_xml(macro_id: str) -> str:
    return _osascript(
        f'''tell application "Keyboard Maestro"
            set m to first macro whose id is "{macro_id}"
            try
                return xml of action 1 of m
            on error
                return ""
            end try
        end tell''',
    )


def clear_macro(macro_id: str) -> None:
    _osascript(
        f'''tell application "Keyboard Maestro"
            set m to first macro whose id is "{macro_id}"
            repeat while (count of actions of m) > 0
                delete action 1 of m
            end repeat
        end tell''',
    )


def capture_one(macro_id: str, action_name: str) -> dict[str, Any] | None:
    """Insert one action by name, read its XML, delete it, return parsed plist."""
    open_by_name(macro_id)
    insert_by_name(action_name)
    raw = read_first_action_xml(macro_id)
    clear_macro(macro_id)
    if not raw:
        return None
    plist = plistlib.loads(raw.encode("utf-8"))
    plist.pop("ActionUID", None)
    text = plist.get("Text")
    if plist.get("MacroActionType") == "Log" and text == _PLACEHOLDER_TEXT:
        return None
    return plist


def serialize_dict_body(plist: dict[str, Any]) -> str:
    full = plistlib.dumps(plist, fmt=plistlib.FMT_XML, sort_keys=False).decode("utf-8")
    start = full.index("<dict>")
    end = full.rindex("</dict>") + len("</dict>")
    return full[start:end]


def run(out_path: Path, macro_id: str, categories: tuple[str, ...], limit: int) -> int:
    captured: dict[str, dict[str, Any]] = (
        json.loads(out_path.read_text()) if out_path.exists() else {}
    )
    new_count = 0
    today = time.strftime("%Y-%m-%d")
    clear_macro(macro_id)
    for category in categories:
        try:
            items = list_category(category)
        except RuntimeError as exc:
            print(f"  skip category {category!r}: {exc}", file=sys.stderr)
            continue
        for name in items:
            if limit and new_count >= limit:
                break
            print(f"  capturing: {category} > {name}")
            try:
                plist = capture_one(macro_id, name)
            except RuntimeError as exc:
                print(f"    error: {exc}", file=sys.stderr)
                continue
            if plist is None:
                continue
            mat = plist.get("MacroActionType")
            if not isinstance(mat, str) or mat in captured:
                continue
            captured[mat] = {
                "captured_at": today,
                "km_version": "11.x",
                "category_path": [category, name],
                "xml": serialize_dict_body(plist),
                "parameter_paths": {},
            }
            new_count += 1
        if limit and new_count >= limit:
            break
    out_path.write_text(json.dumps(captured, indent=2, sort_keys=True) + "\n")
    print(f"wrote {new_count} new entries; total in file: {len(captured)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--macro-id", required=True, help="Scratch macro UUID")
    parser.add_argument(
        "--out",
        default="src/server/data/km_action_templates.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORIES_DEFAULT),
        help="Submenu names to walk (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after capturing N new actions (default: 0 = no limit)",
    )
    args = parser.parse_args()
    return run(
        out_path=Path(args.out),
        macro_id=args.macro_id,
        categories=tuple(args.categories),
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())
