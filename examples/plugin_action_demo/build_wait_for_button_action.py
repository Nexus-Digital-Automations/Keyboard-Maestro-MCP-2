#!/usr/bin/env python3
"""Build a 'Wait for Button' Keyboard Maestro plug-in action.

Owner: Keyboard-Maestro-MCP-2 examples. Demonstrates ``km_build_plugin_action``
end-to-end by wrapping ``examples/ui_inspector/click_button_by_title.py``
(``wait_click`` mode) as a native KM plug-in. After running this script, copy
the emitted ``Wait for Button.kmactions`` folder into
``~/Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions/``
and the action appears in KM's editor sidebar under **Plug In → Third Party**.

Failure modes: missing PyObjC on the KM Engine's interpreter (action returns
exit code 5 at run time), CWD without write access for the output folder
(``OSError`` surfaces with file path).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.server.tools.plugin_action_tools import km_build_plugin_action  # noqa: E402

UI_INSPECTOR_SCRIPT = (
    PROJECT_ROOT / "examples" / "ui_inspector" / "click_button_by_title.py"
)

# Wrapper executed by KM. Reads the action's KMPARAM_* env vars, then invokes
# the embedded UI-Inspector CLI in wait_click mode and prints its result.
WRAPPER_SCRIPT = r"""#!/bin/bash
set -uo pipefail
APP="${KMPARAM_App:-}"
TITLE="${KMPARAM_Title:-}"
TIMEOUT_SECONDS="${KMPARAM_Timeout:-10}"
TIMEOUT_MS=$(/usr/bin/awk "BEGIN{print int(${TIMEOUT_SECONDS}*1000)}")
SCRIPT_PATH="$(dirname "$0")/ui_inspector.py"
exec /usr/bin/python3 "$SCRIPT_PATH" wait_click "$APP" "$TITLE" "$TIMEOUT_MS"
"""


async def main() -> int:
    output_dir = PROJECT_ROOT / "examples" / "plugin_action_demo" / "build"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = await km_build_plugin_action(
        output_dir=str(output_dir),
        name="Wait for Button",
        title="Wait for button '%Param%Title%' in %Param%App%",
        author="Keyboard-Maestro-MCP-2 examples",
        script_source=WRAPPER_SCRIPT,
        script_filename="run.sh",
        parameters=[
            {"Label": "App", "Type": "String", "Default": ""},
            {"Label": "Title", "Type": "String", "Default": ""},
            {"Label": "Timeout", "Type": "Calculation", "Default": "10"},
        ],
        results=["None", "Variable", "Window", "Briefly", "Clipboard"],
        keywords=["button", "wait", "click", "ax", "ui inspector"],
        help_text=(
            "Polls the front window of the named app until exactly one AXButton "
            "matches the title (case-insensitive substring), then clicks it. "
            "Returns 'OK: ...' or 'ERROR: ...' to the chosen result target."
        ),
        on_existing="replace",
    )
    if not result["success"]:
        print("ERROR:", result["error"], file=sys.stderr)
        return 1

    bundle = Path(result["data"]["bundle_path"])
    # Drop the UI-Inspector CLI alongside run.sh so the wrapper can find it.
    (bundle / "ui_inspector.py").write_bytes(UI_INSPECTOR_SCRIPT.read_bytes())

    print(f"Wrote {bundle}")
    print(result["metadata"]["install_hint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
