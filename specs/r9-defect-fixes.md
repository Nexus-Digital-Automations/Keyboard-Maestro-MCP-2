# Round 9 follow-up: fix D2, D4, A6

Three defects surfaced in the R9 KM MCP smoke (see
`docs/km_mcp_audit_report.md` § "Round 9"). This spec captures
empirically-verified root causes and the minimal fixes.

## D2 — `km_create_macro template=window_manager` produces 0 actions

### Root cause (verified by probe in R9)

`_translate_window_manager` in `src/server/tools/creation_tools.py` passes
the caller's `operation` string straight through to the `<key>Action</key>`
plist field. KM 11's ManipulateWindow action requires a PascalCase enum
value (`MoveAndResize`, `Move`, `Resize`, `Minimize`, etc.). The template
docstring (line 560) advertises lowercase values (`move`, `resize`,
`arrange`), so callers following the docs supply `"move"` and KM
silently drops the action on import (macro lands with 0 actions).

R9 probe confirmation:

- `operation="MoveAndResize"` → macro lands with 1 action displayed as
  "Move and Resize Front Window".
- `operation="move"` (R9 D2 reproducer) → macro lands with 0 actions.

### Fix

In `_translate_window_manager`, normalize the documented lowercase
operation values to KM-canonical Action enums:

| Template input | KM Action |
|---|---|
| `move` | `MoveAndResize` (template is documented to take position + size, so the combined op is the right default) |
| `resize` | `Resize` |
| `arrange` | `MoveAndResize` (template's `arrangement` field is folded in by caller; the action is still a move+resize) |
| Anything else | passthrough (lets advanced callers use `Minimize`, `Zoom`, etc.) |

### Acceptance criteria

- `km_create_macro(template="window_manager", parameters={"operation": "move", "x": 100, "y": 100, "width": 800, "height": 600})` lands with **1 action** named "Move and Resize Front Window".
- Same for `operation="resize"`: 1 action named "Resize Front Window".
- Same for `operation="arrange"`: 1 action named "Move and Resize Front Window".
- `operation="MoveAndResize"` continues to work (passthrough).
- New unit test in `tests/test_tools/test_creation_tools.py` exercises the lowercase→PascalCase mapping at the translator level (no live KM needed).

## D4 — `km_action_builder.append action_type=execute_macro` rejected

### Root cause (verified by probe in R9)

KM 11's AppleScript verb `make new action with properties {xml:...}` —
which `append_macro_action_async` uses — rejects ANY ExecuteMacro plist
and substitutes the action with a `Log "Invalid XML From AppleScript"`
placeholder. Verified by probes with both shapes:

- Nested-dict shape (current emitter): rejected. KM substitutes Log.
- Bare-string shape (the legacy shape `km_control_flow` uses inside
  IfThenElse): also rejected via `make new action`. The control_flow
  path only works because it embeds the ExecuteMacro inside another
  action that goes through plistlib + .kmmacros import, never through
  `make new action`.
- Minimal `<dict><key>MacroActionType</key><string>ExecuteMacro</string></dict>`:
  also rejected.

So the issue is not the XML shape — it's the AppleScript verb. KM
restricts which MacroActionType values can be appended via the
scripting interface; ExecuteMacro is not on the allowlist.

### Fix

Route `execute_macro` append through the same export-edit-reimport
pipeline that `km_add_system_trigger` and `km_set_macro_triggers` use:

1. `fetch_macro_snapshot` to get the macro's plist + group context.
2. Append the ExecuteMacro action dict (with `Macro = {MacroName,
   MacroUID}` and `TargetingType = Specific`) to `plist.Actions`.
3. `rebuild_macro_via_reimport` to re-import with a fresh UID.

The UID rotation is unavoidable on this pipeline (already documented for
the trigger tools). The response surfaces `old_macro_id` and
`new_macro_id` plus the existing `uuid_changed: true` warning so callers
can update cross-macro references.

### Acceptance criteria

- `km_action_builder(operation="append", macro_id=<scratch>, action_type="execute_macro", action_config={"target_macro": "R9_Condition"})` succeeds and the appended action displays as `Execute Macro "R9_Condition"` (not the Log placeholder).
- Response includes `old_macro_id`, `new_macro_id`, `uuid_changed: true`, and the same warning text used by `km_add_system_trigger`.
- All other action types (pause / type_text / paste / set_variable / run_applescript / activate_application / manipulate_window / execute_shell_script / plug_in / paste_xml) continue to use the cheap `make new action` path — no UID rotation for them.
- `target_macro` that doesn't resolve still returns `EXECUTE_MACRO_TARGET_NOT_FOUND` (existing behavior preserved).
- Existing test `test_append_execute_macro_resolves_target_to_uid` is updated to reflect the new code path (mock `fetch_macro_snapshot` + `rebuild_macro_via_reimport` instead of `append_macro_action_async`).

## A6 — `km_refresh_action_templates` not exposed on MCP transport

### Root cause (verified by running discovery)

`src/server/tools/refresh_templates_tool.py` imports `Context` under
`if TYPE_CHECKING:`. `tool_registry._extract_tool_metadata` calls
`get_type_hints(func, include_extras=True)` which evaluates ALL parameter
annotations at runtime — including `ctx: Context | None`. Since `Context`
is not in the module globals at runtime, `get_type_hints` raises
`NameError: name 'Context' is not defined`. The discovery loop catches
the exception and silently drops the tool. Confirmed by running the
discovery in the project's venv:

```
ERROR:src.server.tool_registry:Failed to extract metadata for km_refresh_action_templates: name 'Context' is not defined
```

Every other tool module in `src/server/tools/` imports `Context` at
module level (e.g., `engine_tools.py:15 from fastmcp import Context`).
`refresh_templates_tool.py` is the only outlier.

### Fix

Move `from fastmcp import Context` out of the `TYPE_CHECKING` block in
`src/server/tools/refresh_templates_tool.py` so it's available when
`get_type_hints` resolves the annotation. Match the pattern used by the
other tool modules.

### Acceptance criteria

- Running `discover_tools()` returns `km_refresh_action_templates` in
  the dict (and no `name 'Context' is not defined` log line).
- After MCP host restart, `km_refresh_action_templates` appears on the
  transport surface alongside the other 28 `km_*` tools.
- (Live verification deferred to a separate session — the scrape takes
  over the KM editor and must be confirmed by the user before running.)

## Order of operations

1. Code changes (single commit per defect or one combined commit — combined is fine since all three are small).
2. `ruff check` (whole repo).
3. `mypy` (whole repo) — strict.
4. `pytest` (whole repo).
5. Verify D2 and D4 against live KM (D4 requires the spurious `IncludedVariables=["9999"]` defect on `run_applescript` to NOT be re-introduced; deferred D1 polish stays out of scope here).
6. Append a "Round 10" note to `docs/km_mcp_audit_report.md` once fixes ship.
