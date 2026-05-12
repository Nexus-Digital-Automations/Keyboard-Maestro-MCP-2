# UI Inspector — list & click buttons in any macOS app

A Keyboard Maestro macro that:

1. Asks for an app (name or bundle ID).
2. Brings it forward.
3. Enumerates every `AXButton` in the front window via macOS Accessibility.
4. Shows the list in a prompt.
5. Clicks the one whose title you type (case-insensitive substring match).

## Prerequisites

### 1. PyObjC

Install once into the system Python's user-site:

```bash
/usr/bin/python3 -m pip install --user \
    pyobjc-framework-Cocoa \
    pyobjc-framework-ApplicationServices \
    pyobjc-framework-Quartz
```

(The `Quartz` framework is only used by the click-via-position fallback for
the rare button that doesn't respond to `AXPress`.)

### 2. Accessibility permission

The macro runs Python from inside the Keyboard Maestro Engine. Grant the
**Engine** Accessibility access:

**System Settings → Privacy & Security → Accessibility →** enable
`Keyboard Maestro Engine`.

If the script exits with code 5 (`This process is not Accessibility-trusted`),
this is what's missing.

## Install

Double-click `UI_Inspector.kmmacros` in Finder — Keyboard Maestro will import
the group **UI Inspector** with two macros:

- **Click Button by Title** — one-shot: prompts, lists, clicks now.
- **Wait for Button** — polls the front window until the named button
  appears, then clicks. Useful as the Then-action of a "When app launches"
  trigger (e.g. auto-dismiss an "Update available" dialog).

## Use

### Click Button by Title

Trigger from the KM editor or a hotkey of your choice. You'll see two prompts:

1. **App** — e.g. `Safari`, `com.apple.Safari`, `Mail`, `Finder`.
2. **Title** — typed against the list shown in the prompt body. Substring
   match, case-insensitive. Ambiguous matches (multiple buttons match) abort
   with exit code 6 and the macro shows which buttons collided.

The result (success line or error) is displayed briefly via KM's Display Text.

### Wait for Button

One prompt collects App, Title, and a Timeout in seconds (default 10).
The CLI then polls the front window every 250 ms until the button appears
or the timeout elapses. If the app isn't running when the wait starts, the
loop will pick it up as soon as it launches and shows a front window.

CLI shape (used internally by the macro, also runnable standalone):

```bash
./click_button_by_title.py wait_click <app> <title> [timeout_ms=10000] [poll_ms=250]
```

Trigger recipe: pair this macro with a **When this application launches**
trigger pointed at the same app — Keyboard Maestro fires the macro, the
macro waits for the button, clicks it, done. No hand-tuned `Pause` chain.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | app not running (`list` / `click`; `wait_click` waits instead) |
| 3 | no front window (`list` / `click` only) |
| 4 | no buttons found / no title match |
| 5 | accessibility permission denied or click synthesis failed |
| 6 | ambiguous title match |
| 7 | `wait_click` timed out before the button appeared |
| 9 | invalid arguments |

## Files

- `click_button_by_title.py` — standalone runnable. Supports `list`,
  `click`, and `wait_click` modes. Test with `./click_button_by_title.py
  list Finder` from a Terminal that itself has Accessibility permission
  (or run from inside the macro).
- `UI_Inspector.kmmacros` — the importable bundle. One group, two macros
  (`Click Button by Title`, `Wait for Button`). The Python source is
  embedded in a KM variable (`BTN_CLICKER_SCRIPT`) and piped into
  `/usr/bin/python3` via stdin from each Execute Shell Script action.
- `build_kmmacros.py` — regenerates `UI_Inspector.kmmacros` from the
  standalone `.py`. Run after edits to the script.

## Known limitations

- **AXButton only.** No menu items, popups, or web/Electron content. The
  AX hierarchy of Electron/Catalyst apps is often shallow or empty — those
  apps usually need DOM-level automation instead.
- **Front window only.** Other windows of the same app are skipped.
- **Substring titles must be unambiguous.** If your target is "Save" but a
  "Save As…" button also exists, type enough to disambiguate ("Save…",
  `Save A`, etc.).
- **No verification after click.** The script reports AXPress success
  before the app has finished reacting; KM's next action runs immediately.
  Insert a Pause action if the next step depends on the click landing.
