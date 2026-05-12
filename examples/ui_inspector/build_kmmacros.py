#!/usr/bin/env python3
"""Generate UI_Inspector.kmmacros from click_button_by_title.py.

Owner: Keyboard-Maestro-MCP-2 examples. Run after editing the click script
to refresh the importable macro bundle. The macro keeps the Python source
in a Keyboard Maestro variable (``BTN_CLICKER_SCRIPT``) and pipes it into
``/usr/bin/python3`` from two Execute Shell Script actions (``list``,
``click``), so the user imports a single self-contained ``.kmmacros``
file with no external script dependency.

Why a generator: the Python source contains ``<``, ``>``, ``&``, and
quote characters that need plist-XML escaping; doing that by hand is
brittle. ``plistlib.dumps`` handles all of it correctly.
"""
from __future__ import annotations

import plistlib
import time
from pathlib import Path

HERE = Path(__file__).parent
SCRIPT_PATH = HERE / "click_button_by_title.py"
OUTPUT_PATH = HERE / "UI_Inspector.kmmacros"

# Stable UIDs so re-imports update the existing macro instead of cloning it.
GROUP_UID = "F849EB6B-E6A7-47C6-AA62-0B9D77492EEA"
CLICK_MACRO_UID = "AA8BBA64-3741-4543-AA6E-171220BF1453"
WAIT_MACRO_UID = "B8BA5094-5B58-4173-9256-5752B4500DE5"
SCRIPT_VAR = "BTN_CLICKER_SCRIPT"


def km_epoch() -> float:
    """KM stores dates as seconds since 2001-01-01 (Cocoa epoch)."""
    return time.time() - 978307200.0


def shell_action(action_uid: int, text: str, output_variable: str) -> dict:
    return {
        "ActionUID": action_uid,
        "DisplayResultsInWindow": False,
        "IncludeStdErr": True,
        "MacroActionType": "ExecuteShellScript",
        "Path": "",
        "Text": text,
        "TimeOutAbortsMacro": True,
        "TrimResults": True,
        "TrimResultsNew": True,
        "UseText": True,
        "Variable": output_variable,
    }


def build_macro(script_source: str) -> dict:
    now = km_epoch()
    actions = [
        # 1. Stash the Python source in a variable both shell actions can read.
        {
            "ActionUID": 101,
            "MacroActionType": "SetVariableToText",
            "Text": script_source,
            "Variable": SCRIPT_VAR,
        },
        # 2. Ask the user which app to inspect.
        {
            "ActionUID": 102,
            "MacroActionType": "PromptForUserInput",
            "Prompt": "Enter the app name or bundle ID "
                      "(e.g. 'Safari' or 'com.apple.Safari'):",
            "Title": "UI Inspector — choose app",
            "DefaultButton": "OK",
            "Fields": [
                {
                    "Default": "",
                    "Label": "App",
                    "Variable": "App_Name",
                },
            ],
        },
        # 3. Activate it so AX queries hit the front window.
        {
            "ActionUID": 103,
            "Application": {
                "BundleIdentifier": "",
                "Name": "%Variable%App_Name%",
                "NewFile": "",
                "Path": "",
            },
            "MacroActionType": "ActivateApplication",
            "ReopenWindowOptions": "Normal",
        },
        # 4. Tiny wait — without this we sometimes scan the previous front window.
        {
            "ActionUID": 104,
            "MacroActionType": "Pause",
            "Time": "0.4",
            "TimeOutAbortsMacro": True,
        },
        # 5. List buttons. Pipe the embedded source to /usr/bin/python3 via stdin
        #    so we don't need to materialise a file on disk.
        shell_action(
            105,
            'printf "%s" "$KMVAR_' + SCRIPT_VAR + '" | '
            '/usr/bin/python3 - list "$KMVAR_App_Name"',
            "Button_List",
        ),
        # 6. Prompt with the list rendered in the body. KM's prompt body shows
        #    multi-line text; the user types the title to click (substring match).
        {
            "ActionUID": 106,
            "MacroActionType": "PromptForUserInput",
            "Prompt": "Buttons in %Variable%App_Name%:\n\n"
                      "%Variable%Button_List%\n\n"
                      "Type the title (case-insensitive substring match):",
            "Title": "UI Inspector — pick button",
            "DefaultButton": "Click",
            "Fields": [
                {
                    "Default": "",
                    "Label": "Title",
                    "Variable": "Button_Title",
                },
            ],
        },
        # 7. Click the selected button.
        shell_action(
            107,
            'printf "%s" "$KMVAR_' + SCRIPT_VAR + '" | '
            '/usr/bin/python3 - click "$KMVAR_App_Name" "$KMVAR_Button_Title"',
            "Click_Result",
        ),
        # 8. Show the outcome briefly so the user sees AXPress vs. fallback,
        #    or any ERROR: line emitted by the click step.
        {
            "ActionUID": 108,
            "DisplayKind": "Briefly",
            "MacroActionType": "DisplayText",
            "Text": "%Variable%Click_Result%",
        },
    ]
    return {
        "Actions": actions,
        "CreationDate": now,
        "ModificationDate": now,
        "Name": "Click Button by Title",
        "Triggers": [],
        "UID": CLICK_MACRO_UID,
    }


def build_wait_macro(script_source: str) -> dict:
    """Generate the 'Wait for Button' macro — prompts then polls until clickable."""
    now = km_epoch()
    actions = [
        {
            "ActionUID": 201,
            "MacroActionType": "SetVariableToText",
            "Text": script_source,
            "Variable": SCRIPT_VAR,
        },
        # One prompt with three fields keeps the macro usable as a trigger target —
        # quicker to drive than three sequential prompts.
        {
            "ActionUID": 202,
            "MacroActionType": "PromptForUserInput",
            "Prompt": "Wait for an AXButton in the front window, then click it.",
            "Title": "UI Inspector — wait for button",
            "DefaultButton": "Wait",
            "Fields": [
                {"Default": "", "Label": "App", "Variable": "App_Name"},
                {"Default": "", "Label": "Title", "Variable": "Button_Title"},
                {
                    "Default": "10",
                    "Label": "Timeout (seconds)",
                    "Variable": "Timeout_Seconds",
                },
            ],
        },
        {
            "ActionUID": 203,
            "Application": {
                "BundleIdentifier": "",
                "Name": "%Variable%App_Name%",
                "NewFile": "",
                "Path": "",
            },
            "MacroActionType": "ActivateApplication",
            "ReopenWindowOptions": "Normal",
        },
        # awk handles the seconds→ms conversion portably; POSIX $(( )) would
        # work in /bin/sh but awk avoids any shell-arithmetic surprise.
        shell_action(
            204,
            'TIMEOUT_MS=$(/usr/bin/awk '
            '"BEGIN{print int(${KMVAR_Timeout_Seconds:-10}*1000)}"); '
            'printf "%s" "$KMVAR_' + SCRIPT_VAR + '" | '
            '/usr/bin/python3 - wait_click '
            '"$KMVAR_App_Name" "$KMVAR_Button_Title" "$TIMEOUT_MS"',
            "Wait_Result",
        ),
        {
            "ActionUID": 205,
            "DisplayKind": "Briefly",
            "MacroActionType": "DisplayText",
            "Text": "%Variable%Wait_Result%",
        },
    ]
    return {
        "Actions": actions,
        "CreationDate": now,
        "ModificationDate": now,
        "Name": "Wait for Button",
        "Triggers": [],
        "UID": WAIT_MACRO_UID,
    }


def main() -> int:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    now = km_epoch()
    group = {
        "Activate": "Normal",
        "CreationDate": now,
        "CustomIconData": b"",
        "KeyCount": 0,
        "Macros": [build_macro(source), build_wait_macro(source)],
        "Name": "UI Inspector",
        "ToggleMacroUID": CLICK_MACRO_UID,
        "UID": GROUP_UID,
    }
    OUTPUT_PATH.write_bytes(plistlib.dumps([group], sort_keys=False))
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
