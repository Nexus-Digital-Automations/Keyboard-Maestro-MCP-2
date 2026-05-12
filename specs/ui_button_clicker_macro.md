# UI Button Clicker — Keyboard Maestro macro

## Goal

A drop-in `.kmmacros` file the user imports into Keyboard Maestro. Given an app
name (or bundle ID) as a parameter, the macro:

1. Activates the target app.
2. Enumerates clickable AXButton elements in its front window.
3. Presents the list to the user via a KM popup.
4. AXPresses the selected button. Falls back to a Quartz mouse click at the
   button's screen position if AXPress is unsupported.

## Constraints

- macOS only. Uses system `/usr/bin/python3` (PyObjC pre-installed there).
- No new repository dependencies. No MCP server changes.
- Front window only.
- Title is the primary identifier; position is the fallback for unlabeled
  buttons.

## Acceptance criteria

- [x] One `.kmmacros` file under `examples/ui_inspector/` importable into KM.
- [x] Self-contained — no external script file the user must place. Python
      source is embedded in the macro as a KM variable, referenced by two
      Execute Shell Script actions (`list`, `click`).
- [x] A standalone `click_button_by_title.py` mirrors the embedded source so
      developers can test and iterate outside KM.
- [x] List output: one entry per line, format
      `<title or '<AXRole @ x,y>'>` — readable in a KM popup picker.
- [x] Click resolution: case-insensitive substring match on title. Ambiguous
      matches → exit code 6, error message names the matches.
- [x] Failure modes documented in the script's module docstring; each surfaces
      a distinct exit code so KM can branch on it.
- [x] `examples/ui_inspector/README.md` covers: prerequisites (Accessibility
      permission granted to KM Engine), import steps, parameter contract,
      troubleshooting.

## Out of scope

- AXMenuItem clicks (only AXButton in this iteration).
- Web views and Electron content (System Events / AX exposes them inconsistently;
  noted as a known limitation).
- Multiple windows (front window only).
- Registering as an MCP tool — that's a follow-on if needed.

## Exit codes (script)

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | app not running |
| 3 | no front window for app |
| 4 | no buttons found (list mode) / no title match (click mode) |
| 5 | accessibility permission denied / AX call failed |
| 6 | ambiguous click target (multiple buttons match the title substring) |
| 9 | invalid arguments |
