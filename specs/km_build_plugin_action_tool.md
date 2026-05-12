# Spec — `km_build_plugin_action` MCP tool

Owner: `src/server/tools/plugin_action_tools.py`. Companion demo at
`examples/plugin_action_demo/`.

## Goal

A new MCP tool that emits a Keyboard Maestro third-party plug-in action
bundle (`.kmactions` folder) from a JSON spec. The bundle, once copied into
`~/Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions/`,
appears in the KM editor under **Plug In → Third Party** as a first-class,
drag-droppable action with native parameter fields. The MCP server itself
makes no system writes — the caller installs the bundle.

## Inputs

| Field | Required | Notes |
|---|---|---|
| `output_dir` | yes | Absolute path; must resolve under CWD. |
| `name` | yes | Display name; sanitised for the folder basename. |
| `identifier` | yes | Reverse-DNS, lowercase, ≥2 dotted segments. |
| `author` | yes | String embedded in plist. |
| `script_source` | yes | Full script body. Caller includes shebang. |
| `script_filename` | no (`script`) | Filename inside the bundle; chmod 0755. |
| `parameters` | no | List of `{Label, Type, Default?, Menu?}`. |
| `results_type` | no (`Variable`) | KM enum: `None|Variable|Clipboard|TypedString|Window|Briefly`. |
| `timeout_seconds` | no (`60`) | Int in `[1, 3600]`. |
| `authentication` | no (`None`) | `None` or `Admin`. |
| `help_text` | no | Inline help shown in KM's editor. |
| `help_url` | no | `http(s)://…`. |
| `keywords` | no | List of non-empty strings, indexed by KM's action picker. |
| `icon_base64` | no | Base64-encoded PNG bytes. |
| `on_existing` | no (`error`) | `error` aborts, `replace` `rmtree`s first. |

`parameters[i].Type` enum: `String`, `Password`, `Calculation`, `PopupMenu`,
`Checkbox`, `Hidden`, `TokenString`, `TokenText`. `PopupMenu` requires
`Menu: list[str]` (joined with `\n` in the plist).

## Output envelope

**Success:**
```json
{
  "success": true,
  "data": {
    "bundle_path": "/abs/path/Name.kmactions",
    "plist_path": ".../Keyboard Maestro Action.plist",
    "script_path": ".../script",
    "plist_size_bytes": 1491,
    "parameter_count": 3,
    "identifier": "com.example.foo"
  },
  "metadata": {
    "tool": "km_build_plugin_action",
    "timestamp": "…ISO8601…",
    "duration_ms": 4,
    "install_hint": "Copy the bundle into ~/Library/Application Support/…"
  }
}
```

**Failure:** `{success: False, error: {code, message, recovery_suggestion, field?}, metadata: {...}}`. Codes used:

- `VALIDATION_FAILED` — bad identifier, bad enum, missing PopupMenu `Menu`, etc.
- `BUNDLE_EXISTS` — `on_existing="error"` and the folder already exists.
- `BUNDLE_REPLACE_FAILED` — `rmtree` of the existing bundle failed (perms, etc.).
- `BUNDLE_WRITE_FAILED` — `mkdir`/`plistlib.dump`/`write_text`/`chmod` raised OSError.

## Acceptance criteria

1. **Happy path round-trip.** Calling with name, identifier, author,
   script_source, and a mixed parameter list (String, Calculation,
   PopupMenu+Menu, Checkbox) writes a bundle whose
   `Keyboard Maestro Action.plist` round-trips through `plistlib.load` and
   contains the expected `Name`, `Identifier`, `Author`, `Script`,
   `ResultsType`, and `Parameters` (with `Menu` joined by `\n` and
   `Checkbox` default coerced to `"0"`/`"1"`).
2. **Script file written executable.** `bundle/<script_filename>` exists,
   `stat().st_mode & 0o777 == 0o755`, content equals `script_source` byte-
   for-byte (UTF-8).
3. **Validation rejects each defect** as `{success: False, error.code:
   "VALIDATION_FAILED"}` with `error.message` naming the failing field:
   - identifier `"Foo Bar"` → reverse-DNS message.
   - `results_type="Bogus"` → enum-list message.
   - `authentication="Root"` → enum-list message.
   - `parameters=[{"Label":"X","Type":"PopupMenu"}]` (no Menu) → param-index message.
   - `output_dir="/tmp"` from a CWD outside `/tmp` → path-traversal message.
   - `timeout_seconds=99999` → range message.
   - `script_filename="../oops"` → regex message.
4. **`on_existing` semantics.** Second call with `on_existing="error"`
   returns `BUNDLE_EXISTS`; second call with `on_existing="replace"`
   succeeds and the script content reflects the new `script_source`.
5. **Optional metadata** emitted iff supplied: `Help`, `HelpURL`,
   `KeyWords`, `Icon` (decoded PNG bytes) appear in the plist when
   provided; absent keys when arguments are `None`.
6. **No partial bundle on failure.** If validation fails, no folder is
   created. If the post-mkdir write step fails, the partially-written
   folder is left in place for inspection (operator forensics) — this is
   intentional; subsequent calls with `on_existing="replace"` clean it up.
7. **Tool auto-registers.** `ToolDiscovery` picks up the new function on
   server start without manual entry in `tool_config.py` (since the tool
   needs only default policy). Listing tools via the MCP shows
   `km_build_plugin_action` alongside the existing tools.
8. **Demo end-to-end.** `examples/plugin_action_demo/build_wait_for_button_action.py`
   runs to completion via `uv run python …` and emits a bundle with three
   parameters (`App`, `Title`, `Timeout`), a `run.sh` wrapper, and a copy
   of `click_button_by_title.py` next to it. Installing the bundle into
   KM's Actions folder makes **Wait for Button** appear under
   **Plug In → Third Party**.

## Non-goals (v1)

- No automatic install into `~/Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions/`.
- No zip packaging — the bundle is a plain folder, which `cp -R` handles.
- No KM Engine reload — caller invokes `km_engine_control` if desired.
- No editing of an existing bundle in place — `on_existing="replace"`
  rewrites the whole bundle.
