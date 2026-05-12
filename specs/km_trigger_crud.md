# Spec — `km_trigger_crud` MCP tool

Owner: new file `src/server/tools/trigger_crud_tools.py`.
Companion km_client primitives in `src/integration/km_client.py`.

## Goal

A single MCP tool that performs full CRUD on the triggers of an existing
Keyboard Maestro macro, supporting **every KM trigger type** by round-tripping
the trigger's plist XML via Keyboard Maestro's AppleScript dictionary
(`xml of trigger N of macro …`). This is the same mechanism KM itself uses
to expose individual trigger configuration to scripts, and matches how
`km_action_builder` round-trips action XML today.

The tool is additive — it does **not** replace `km_trigger_manager`. That
tool stays as the friendly typed-API for hotkey/application triggers; this
one is the universal escape hatch for power users and for trigger types
not modelled by the typed API.

## Inputs

| Field | Required | Notes |
|---|---|---|
| `operation` | yes | `list \| get \| add \| update \| remove \| replace_all` |
| `macro_id` | yes | Macro name or UUID. Same identifier rules as other km tools. |
| `index` | conditional | 1-indexed position. Required for `get`, `update`, `remove`. |
| `xml` | conditional | Full trigger plist `<dict>…</dict>` XML. For `add`/`update`. |
| `trigger` | conditional | Structured dict; serialised to plist XML server-side. Alternative to `xml` for `add`/`update`. |
| `triggers` | conditional | List of trigger dicts or `{xml: "..."}`. For `replace_all`. |

`xml` and `trigger` are mutually exclusive per call. `trigger` is the
ergonomic path — caller passes a Python dict like
`{"MacroTriggerType": "HotKey", "FireType": "Pressed", "KeyCode": 49, "Modifiers": 256}`
and the server converts to a plist `<dict>` via `plistlib`.

### Trigger-type coverage

All KM trigger types are supported by virtue of the XML round-trip. The
known `MacroTriggerType` discriminators (from KM 11.x docs) are listed
non-exhaustively for caller convenience but **not** validated against an
allow-list — KM is the authority. Common values:

`HotKey`, `Time`, `TimeOfDay`, `Login`, `Wake`, `WirelessNetwork`,
`USBDeviceKey`, `FolderTrigger`, `PublicWebEntry`, `RemoteTrigger`,
`AppActivated`, `AppDeactivated`, `AppLaunched`, `AppQuit`, `EngineLaunch`,
`MacroPaletteEntry`, `ChangedKey`, `ChangedClipboard`, `MountedVolume`,
`UnmountedVolume`, `SystemSleep`, `SystemWake`, `Idle`.

If KM rejects unknown XML, the error envelope returns KM's `errMsg` verbatim
with code `KM_REJECTED_XML`.

## Output envelope

Success: `{"success": true, "data": {…op-specific…}}`.

Failure: `{"success": false, "error": {"code": "<CODE>", "message": "…",
"suggestion": "…", "field"?: "<which input>"}}`.

Per-op `data` shapes:

- `list` — `{macro_id, triggers: [{index, type, enabled, description, xml}]}`
- `get`  — `{macro_id, index, type, enabled, description, xml}`
- `add`  — `{macro_id, index, type}` (index = new 1-indexed position)
- `update` — `{macro_id, index, type}`
- `remove` — `{macro_id, index}`
- `replace_all` — `{macro_id, count}`

## Acceptance criteria

1. `operation="list"` returns one entry per trigger with `index`, `type`
   (= `MacroTriggerType` value), `enabled`, KM's `description` string,
   and the full `xml` of that trigger. Indices are 1-based and stable
   within the call.
2. `operation="get"` with a valid `index` returns the full XML of that
   single trigger; invalid index returns `INDEX_OUT_OF_RANGE`.
3. `operation="add"` with either `xml` or `trigger` appends a new trigger
   at the end of the macro's trigger list and returns the resulting 1-based
   `index`. Passing both `xml` and `trigger` returns `BOTH_XML_AND_TRIGGER`.
4. `operation="update"` replaces trigger at `index` with the supplied
   XML/dict. The KM-assigned UID is preserved when KM allows; otherwise
   the trigger is removed and re-inserted at the same index.
5. `operation="remove"` deletes trigger at `index`. Same out-of-range
   handling as `get`.
6. `operation="replace_all"` clears existing triggers then appends each
   element of `triggers` in order. On any per-element failure, prior
   inserts are kept (no rollback in v1) and the error envelope reports
   the failing index.
7. The MCP server is the only path that talks to AppleScript; no live-KM
   tests are required. Unit tests cover (a) the `trigger`-dict-to-plist
   serialiser, (b) the index/operation validation logic, (c) error
   envelope shape.
8. All inputs are validated and error envelopes have stable `code` values:
   `VALIDATION_ERROR`, `INDEX_OUT_OF_RANGE`, `BOTH_XML_AND_TRIGGER`,
   `MISSING_PAYLOAD`, `KM_REJECTED_XML`, `KME_UNREACHABLE`, `LIST_FAILED`,
   `ADD_FAILED`, `UPDATE_FAILED`, `REMOVE_FAILED`, `REPLACE_FAILED`.
9. AppleScript strings are escaped via the existing
   `KMClient._escape_applescript_string` helper. Trigger XML is
   transported through a `set xmlVar to «data ... »`-style here-string so
   embedded quotes survive the round-trip. (Implementation may use the
   same trick `km_action_builder` does — quote-by-concatenation if needed.)
10. Tool is registered in the dynamic tool registry alongside
    `km_trigger_manager` and shows up in `km_list_action_types`-style
    discovery. `ruff`, `mypy`, and `pytest` all pass.

## Out of scope (v1)

- No before-edit backup of the macro's prior trigger array.
- No rollback on partial `replace_all` failure.
- No live-KM integration tests.
- No `MacroTriggerType` allow-list — KM is the authority on what's valid.
- No UI in `km_trigger_manager` for the new operations; it stays as-is.
