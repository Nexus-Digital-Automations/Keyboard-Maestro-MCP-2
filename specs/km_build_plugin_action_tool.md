# Spec — `km_build_plugin_action` MCP tool

Owner: `src/server/tools/plugin_action_tools.py`. Companion demo at
`examples/plugin_action_demo/`.

## Goal

A MCP tool that emits a Keyboard Maestro third-party plug-in action bundle
(`.kmactions` folder) from a JSON spec. The bundle, once copied into
`~/Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions/`,
appears in the KM editor under **Plug In → Third Party** as a first-class,
drag-droppable action with native parameter fields. The MCP server itself
makes no system writes — the caller installs the bundle.

## Schema source of truth

Plist keys were chosen by inspecting eight shipping third-party plug-ins
installed under `~/Library/Application Support/Keyboard Maestro/Keyboard
Maestro Actions/` (Choose File(s), Execute a Script in Terminal, Read/Click
XPath, Run AppleScript With Specified Parameters, Split Text, Set Finder
Label, Activate an Application by Name). Every key emitted by this tool is
present in at least one shipping plug-in. Keys absent from all shipping
plug-ins (`Identifier`, `Authentication`, `Timeout`) are not emitted — KM
ignored them, but they had no effect.

## Inputs

| Field | Required | Notes |
|---|---|---|
| `output_dir` | yes | Absolute path; must resolve under CWD. |
| `name` | yes | Sidebar label under Plug In > Third Party; also the bundle folder basename. |
| `title` | yes | Canvas display string. Supports `%Param%Label%` placeholders, e.g. `"Wait for '%Param%Title%' in %Param%App%"`. |
| `script_source` | yes | Full script body. Caller includes shebang. |
| `script_filename` | no (`script`) | Filename inside the bundle; chmod 0755. |
| `parameters` | no | List of `{Label, Type, Default?, Menu?}`. |
| `results` | no (`["Variable"]`) | List of allowed result targets; pipe-joined into the plist `Results` key. |
| `author` | no | Optional author string. |
| `help_text` | no | Inline help shown in KM's editor. |
| `help_url` | no | `http(s)://…`. |
| `keywords` | no | Non-empty strings, indexed by KM's action picker. |
| `icon_base64` | no | Base64-encoded PNG; written to `Icon.png` inside the bundle and referenced by filename in the plist. |
| `on_existing` | no (`error`) | `error` aborts, `replace` `rmtree`s first. |

`parameters[i].Type` enum: `String`, `Text`, `Password`, `Calculation`,
`PopupMenu`, `Checkbox`, `Hidden`, `TokenString`, `TokenText`. `PopupMenu`
requires `Menu: list[str]` (joined with `|` in the plist — verified against
Choose File(s)'s `'Folder...|Desktop|Home|Documents|...|custom path below'`).

`results` entries must each be one of: `None`, `Variable`, `Clipboard`,
`TypedString`, `Pasting`, `Typing`, `Window`, `Briefly`, `Token`,
`Asynchronously`. They are emitted to the plist as a pipe-joined string
(KM's wire format, e.g. `"Variable|Window|Briefly|Clipboard"`).

## Plist keys emitted

Always: `Name`, `Title`, `Script`, `Results`, `Parameters`.
Optional (only when caller supplies them): `Author`, `Help`, `HelpURL`,
`KeyWords`, `Icon` (filename `"Icon.png"`, with the decoded PNG written
alongside in the bundle).

## Output envelope

**Success:**
```json
{
  "success": true,
  "data": {
    "bundle_path": "/abs/path/Name.kmactions",
    "plist_path": ".../Keyboard Maestro Action.plist",
    "script_path": ".../script",
    "icon_path": ".../Icon.png",
    "plist_size_bytes": 1491,
    "parameter_count": 3,
    "results": ["Variable"]
  },
  "metadata": {
    "tool": "km_build_plugin_action",
    "timestamp": "…ISO8601…",
    "duration_ms": 4,
    "install_hint": "Copy the bundle into ~/Library/Application Support/…"
  }
}
```

`icon_path` is `null` when `icon_base64` was not provided.

**Failure:** `{success: False, error: {code, message, recovery_suggestion, field?}, metadata: {...}}`. Codes used:

- `VALIDATION_FAILED` — bad enum value, missing PopupMenu `Menu`, bad PNG magic, etc.
- `BUNDLE_EXISTS` — `on_existing="error"` and the folder already exists.
- `BUNDLE_REPLACE_FAILED` — `rmtree` of the existing bundle failed (perms, etc.).
- `BUNDLE_WRITE_FAILED` — `mkdir`/`plistlib.dump`/`write_text`/`chmod` raised OSError.

## Acceptance criteria

1. **Happy path round-trip.** Calling with `name`, `title`, `script_source`,
   `results=["Variable", "Window"]`, and a mixed parameter list (String,
   Calculation, PopupMenu+Menu, Checkbox) writes a bundle whose
   `Keyboard Maestro Action.plist` round-trips through `plistlib.load` and
   contains `Name`, `Title`, `Script`, `Results == "Variable|Window"`, and
   `Parameters` (with `Menu` joined by `|` and `Checkbox` default coerced
   to `"0"`/`"1"`). Keys `Identifier`, `Authentication`, `Timeout`,
   `ResultsType` are absent.
2. **Script file written executable.** `bundle/<script_filename>` exists,
   `stat().st_mode & 0o777 == 0o755`, content equals `script_source` byte-
   for-byte (UTF-8).
3. **Validation rejects each defect** as `{success: False, error.code:
   "VALIDATION_FAILED"}` with `error.message` naming the failing field:
   - `results=["Bogus"]` → unknown-target message.
   - `results=[]` → non-empty-list message.
   - `parameters=[{"Label":"X","Type":"PopupMenu"}]` (no Menu) → param-index message.
   - `output_dir="/tmp"` from a CWD outside `/tmp` → path-traversal message.
   - `script_filename="../oops"` → regex message.
   - `icon_base64="bm90IGEgcG5n"` (valid base64, not PNG) → PNG-magic message.
4. **`on_existing` semantics.** Second call with `on_existing="error"`
   returns `BUNDLE_EXISTS`; second call with `on_existing="replace"`
   succeeds and the script content reflects the new `script_source`.
5. **Optional metadata** emitted iff supplied: `Author`, `Help`, `HelpURL`,
   `KeyWords` appear in the plist when provided; absent keys when arguments
   are `None`. When `icon_base64` is provided, `Icon.png` exists in the
   bundle and the plist's `Icon` value is the string `"Icon.png"`.
6. **No partial bundle on failure.** If validation fails, no folder is
   created. If the post-mkdir write step fails, the partially-written
   folder is left in place for inspection (operator forensics) — this is
   intentional; subsequent calls with `on_existing="replace"` clean it up.
7. **Tool auto-registers.** `ToolDiscovery` picks up the function on
   server start without manual entry in `tool_config.py`. Listing tools
   via the MCP shows `km_build_plugin_action` alongside the existing tools.
8. **Demo end-to-end.** `examples/plugin_action_demo/build_wait_for_button_action.py`
   runs to completion and emits a bundle with three parameters (`App`,
   `Title`, `Timeout`), a `run.sh` wrapper, and a copy of
   `click_button_by_title.py` next to it. Installing the bundle into KM's
   Actions folder makes **Wait for Button** appear under **Plug In → Third
   Party** with the configured title rendering and result-target dropdown.

## Non-goals (v1)

- No automatic install into `~/Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions/`.
- No zip packaging — the bundle is a plain folder, which `cp -R` handles.
- No KM Engine reload — caller invokes `km_engine_control` if desired.
- No editing of an existing bundle in place — `on_existing="replace"`
  rewrites the whole bundle.
- No validation that `title`'s `%Param%X%` placeholders reference real
  parameter labels — KM tolerates missing references by rendering the
  placeholder literally, which is useful diagnostic feedback.
