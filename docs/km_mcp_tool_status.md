# KM MCP Tool Status — 2026-05-14

Smoke-test snapshot of all 30 Keyboard Maestro MCP tools. "Worked
out-of-the-box" = passed the 2026-05-14 smoke without code changes.
"Fixed" = was broken before this session; commits cited below repaired it.
"Documented limitation" = part of the surface intentionally returns
`UNSUPPORTED_OPERATION` with a recovery suggestion.

> ⚠️ MCP server runs from `~/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP-2`
> (main branch). Fixes in `session/20260514-125257-36011` need a merge +
> Claude Code restart before they're live.

## Worked out-of-the-box (24)

- `km_list_macros`
- `km_list_action_types`
- `km_list_templates`
- `km_search_actions`
- `km_token_stats`
- `km_engine_control` — all 5 ops (status, reload, calculate, process_tokens, search_replace)
- `km_list_hotkey_triggers`
- `km_application_control` — all 5 ops (launch, quit, activate, list_running, get_state)
- `km_window_manager` — get_screens, get_info (move/resize/arrange succeed AppleScript-side; Finder-window-index drift is a target-app quirk, not a tool bug)
- `km_macro_group_manager` — all 5 ops
- `km_macro_editor` — all 5 ops
- `km_create_macro`
- `km_action_builder` — all 4 ops, every action_type
- `km_create_hotkey_trigger`
- `km_trigger_crud` — all 6 ops
- `km_trigger_manager` — `set_enabled` returns a documented KM-AppleScript limitation
- `km_variable_manager` — all 4 ops, every scope (global, local, instance, password)
- `km_execute_macro` — AppleScript + URL transports
- `km_notification_status`
- `km_dismiss_notifications`
- `km_create_plugin_action`
- `km_refresh_action_templates`
- `km_notifications` notification/hud/sound subtypes

## Fixed in this session

### `b83bba4` — surface-level smoke fixes

- `km_move_macro_to_group` — `create_group_if_missing=True` was ignored. Auto-create now runs **before** validation in `move_macro_to_group_async`.
- `km_notifications` alert — crashed with `AttributeError: 'KMError' has no attribute 'error_code'`. KMError now wrapped in MacroEngineError at the boundary.
- `km_token_processor` — regex split multi-segment tokens (`%Calculate%5*5%`). Extended the two-segment alternative to cover Calculate, ICUDateTime, JSONValue, XMLValue, AddressBook, AskForUserInput, CurrentMouse, Find, FoundImage, MIDI, Past, Path, Time, UUID, Wireless.

### `4255c55` — three previously-impossible tools now real

- `km_add_condition` — was always `UNSUPPORTED_OPERATION`. Now emits a canonical KM `IfThenElse` action plist via the shared emitter (`src/integration/km_if_then_else_xml.py`). Supports `variable`, `text`, `application`, `calculation` condition types and the full operator set; `action_on_true` / `action_on_false` become inner `ExecuteMacro` actions.
- `km_control_flow` if_then_else mode — previously raised `NotImplementedError`. Now dispatches to the same shared emitter; inner `actions_true` / `actions_false` lists are translated through `_build_action_xml`. Other modes (`for_loop`, `while_loop`, `switch_case`, `try_catch`) still raise `NotImplementedError`.
- `km_add_system_trigger` — KM 11 AppleScript hard-rejects non-HotKey trigger plists at `make new trigger`. Now uses an export-edit-reimport pipeline: fetch macro XML via `get xml of macro`, inject the trigger dict (`Login` / `EngineLaunch` / `WakeTrigger`), delete the original, import a modified `.kmmacros`. **Documented trade-off:** the macro's UUID changes (re-using the old UID would block on KM's "duplicate UID" GUI prompt). Response includes `old_macro_id` and `new_macro_id` so callers can rewrite cross-macro `ExecuteMacro` references.

## Documented limitations (partial coverage by design)

- `km_control_flow` — `for_loop`, `while_loop`, `switch_case`, `try_catch` still return `UNSUPPORTED_OPERATION`. Workaround: build the surrounding action XML and append via `km_action_builder(operation='append', action_type='paste_xml')`.
- `km_trigger_manager set_enabled` — per-trigger enable/disable is not exposed by KM 11 AppleScript; macro-level enable/disable is available via `km_macro_editor`.
- `km_window_manager` against Finder — move/resize/arrange AppleScript succeeds but Finder reuses window indices in surprising ways; `get_info` may poll the menubar instead of the moved window. Target-app issue, not a tool bug.
- `km_create_macro template="hotkey_action"` with `parameters` — returns `UNSUPPORTED_TEMPLATE`. Use `template="custom"` and attach the hotkey via `km_create_hotkey_trigger` after creation.
- `km_create_plugin_action` `output_dir` — must be under the MCP server CWD; relative paths outside the server root are blocked by the path-traversal guard.
- `km_refresh_action_templates` — `index` must be a UUID, not a name.

## How to reproduce

The full smoke harness is in:

- `specs/km_mcp_smoke_20260514.md` — round 1 (the 30-tool sweep)
- `specs/km_mcp_fix_broken_tools_20260514.md` — round 2 (the three implementations above)

Each spec lists the call shape, expected response, and the KM-side side-effect to verify.
