# Plug-in Action Demo — Wait for Button

Demonstrates the `km_build_plugin_action` MCP tool by building a native
Keyboard Maestro **third-party plug-in action** out of the existing
`examples/ui_inspector/click_button_by_title.py` (`wait_click` mode).

Once installed, **Wait for Button** appears in the KM editor's actions sidebar
under **Plug In → Third Party**. Drop it into any macro, fill in three native
fields (App, Title, Timeout seconds), and KM hands the result string to a
variable.

## Build

```bash
uv run python examples/plugin_action_demo/build_wait_for_button_action.py
```

Output: `examples/plugin_action_demo/build/Wait for Button.kmactions/`.

The bundle contains:

| File | What it does |
|---|---|
| `Keyboard Maestro Action.plist` | Metadata (name, identifier, params, result mode). |
| `run.sh` | Wrapper KM executes. Reads `KMPARAM_*`, calls `ui_inspector.py`. |
| `ui_inspector.py` | A verbatim copy of `examples/ui_inspector/click_button_by_title.py`. |

## Install

1. Make sure PyObjC is installed in the system Python (the same prereq as
   the standalone macro):
   ```bash
   /usr/bin/python3 -m pip install --user \
       pyobjc-framework-Cocoa \
       pyobjc-framework-ApplicationServices \
       pyobjc-framework-Quartz
   ```
2. Grant Accessibility to **Keyboard Maestro Engine** in
   System Settings → Privacy & Security → Accessibility.
3. Copy the bundle into KM's Actions folder:
   ```bash
   mkdir -p "$HOME/Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions"
   cp -R "examples/plugin_action_demo/build/Wait for Button.kmactions" \
         "$HOME/Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions/"
   ```
4. Restart the Keyboard Maestro Engine (KM menu bar icon → Quit → relaunch),
   or use the MCP: `km_engine_control({"operation": "reload"})`.

## Use

In the KM editor, **Insert Action by Name** → type `Wait for Button`. It
appears under **Plug In → Third Party**. The native form shows:

- **App** — name or bundle ID (`Finder`, `com.apple.Safari`, …).
- **Title** — substring match against AXButton titles, case-insensitive.
- **Timeout** — seconds; the wrapper converts to ms.

Set **Save results to** → a KM variable. After the action runs, your
variable contains either `OK: AXPress on '<title>'` or `ERROR: …`.

## Rebuild

After editing `click_button_by_title.py` (the embedded CLI) or the wrapper
in `build_wait_for_button_action.py`, just re-run the build script — it
passes `on_existing="replace"` so the bundle is regenerated in place. KM
picks up changes after the next Engine restart.

## How this maps to the MCP tool

`build_wait_for_button_action.py` is ~50 lines and is the canonical example
of calling `km_build_plugin_action` from Python. Anywhere you'd hand-author
a `.kmactions` folder, you can instead build the spec dict and call the
tool — the plist serialisation, identifier validation, path-traversal
guard, and `chmod 0755` are all handled for you.
