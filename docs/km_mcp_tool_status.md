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

### round 7 — switch_case + template parameters (2026-05-14, session 20260514-145029-69306)

- **`km_control_flow switch_case` — implemented.** New emitter `src/integration/km_switch_case_xml.py`. KM 11 Switch surface captured by inject-and-readback probe: 5 Source values (Variable / Clipboard / NamedClipboard / Calculation / Text) and 5 per-case ConditionType values (Is / IsNot / Contains / DoesNotContain / Otherwise). KM intentionally narrow — silently normalizes anything else to Clipboard / Contains. Default case is a sentinel CaseEntry with ConditionType=Otherwise (no separate plist key). Public surface adds `source: str` parameter (defaults "Variable"); existing `condition` carries the source value (variable name / calculation / text source); existing `cases` and `default_actions` honored. Live-verified against KM 11 for all 4 active source types.
- **`km_create_macro` template parameters — implemented atomically.** All 5 templates (`app_launcher`, `text_expansion`, `file_processor`, `window_manager`, `hotkey_action`) now bake their action sequence into the `.kmmacros` plist before import. Single KM round-trip per create — no N+1 appends. Three new action emitters added to `_build_action_xml`: `activate_application`, `manipulate_window`, `execute_shell_script` (each verified against KM-canonical templates already in `km_action_templates.json`). Hotkey trigger attaches via `km_create_hotkey_trigger` after import for `hotkey_action`. Live-verified all 5 templates create real macros with correct action+trigger counts.
- **Dead code removed.** `_apply_control_flow_to_macro`, `_generate_km_control_flow`, `_get_km_action_type`, `_generate_km_xml`, and the entire `src/integration/km_control_flow.py` module deleted (verified zero callers via `references_to`). All 5 emitter modes route directly via dedicated emitters now.

### round 6 — control-flow emitters + condition-key bug + UX wins (2026-05-14, session 20260514-145029-69306)

- **`km_control_flow` for_loop / while_loop / until_loop / try_catch — implemented.** Four new emitter modules (`src/integration/km_for_loop_xml.py`, `km_while_loop_xml.py`, `km_try_catch_xml.py`) plus four `_emit_*` dispatchers in `control_flow_tools.py`. KM-canonical XML captured against KM 11 by importing `.kmmacros` skeletons and reading back the normalized output (fixtures in `tests/fixtures/km_control_flow/`). for_loop supports all 13 KM editor collection types (Applications, Dictionaries, DictionaryKeys, Files, FinderSelection, FoundImages, JSON, LinesIn, PastClipboards, Range, SubstringsIn, Variables, Volumes) via a structured `collection_dict={'type': ..., ...}` parameter.
- **Variable + Text condition emitter bug — fixed.** `km_if_then_else_xml._variable_condition` was emitting `<key>ConditionResult</key>` for the comparison value; KM 11 expects `<key>VariableValue</key>` and silently drops the wrong key on import (then synthesises a placeholder). Same defect in `_text_condition` (`ConditionType=TextContents` should be `Text`; `TextContentsConditionType` should be `TextConditionType`; `ConditionResult` should be `TextValue`). Verified by KM-author-then-read-back probe. Affected every shipped if/then/else and add_condition call: the condition would compare against the wrong RHS at runtime.
- **`km_trigger_manager set_enabled` — verdict shipped.** Live probe (session 20260514-145029-69306) injected `<key>Disabled</key><true/>` into a HotKey trigger via `set xml of trigger` and read back: KM stripped the key. Triggers inherit the parent macro's enabled state; KM 11 stores no per-trigger enable bit. Error message rewritten to cite the probe and point to `km_macro_editor set_enabled` for macro-level toggle.
- **`km_refresh_action_templates` — accepts macro names.** Previously required a UUID; passing a name silently failed. Now resolves names via `_resolve_macro_uuid` and includes `resolved_macro_id` in the response.
- **`km_window_manager arrange` — surfaces stale bounds.** When the post-arrange `get_window_info` re-query fails, response now includes `window_info_source: "pre_operation"` so callers know the bounds reflect pre-arrange state.

### `b83bba4` — surface-level smoke fixes

- `km_move_macro_to_group` — `create_group_if_missing=True` was ignored. Auto-create now runs **before** validation in `move_macro_to_group_async`.
- `km_notifications` alert — crashed with `AttributeError: 'KMError' has no attribute 'error_code'`. KMError now wrapped in MacroEngineError at the boundary.
- `km_token_processor` — regex split multi-segment tokens (`%Calculate%5*5%`). Extended the two-segment alternative to cover Calculate, ICUDateTime, JSONValue, XMLValue, AddressBook, AskForUserInput, CurrentMouse, Find, FoundImage, MIDI, Past, Path, Time, UUID, Wireless.

### `4255c55` — three previously-impossible tools now real

- `km_add_condition` — was always `UNSUPPORTED_OPERATION`. Now emits a canonical KM `IfThenElse` action plist via the shared emitter (`src/integration/km_if_then_else_xml.py`). Supports `variable`, `text`, `application`, `calculation` condition types and the full operator set; `action_on_true` / `action_on_false` become inner `ExecuteMacro` actions.
- `km_control_flow` if_then_else mode — previously raised `NotImplementedError`. Now dispatches to the same shared emitter; inner `actions_true` / `actions_false` lists are translated through `_build_action_xml`. Other modes (`for_loop`, `while_loop`, `switch_case`, `try_catch`) still raise `NotImplementedError`.
- `km_add_system_trigger` — KM 11 AppleScript hard-rejects non-HotKey trigger plists at `make new trigger`. Now uses an export-edit-reimport pipeline: fetch macro XML via `get xml of macro`, inject the trigger dict (`Login` / `EngineLaunch` / `WakeTrigger`), delete the original, import a modified `.kmmacros`. **Documented trade-off:** the macro's UUID changes (re-using the old UID would block on KM's "duplicate UID" GUI prompt). Response includes `old_macro_id` and `new_macro_id` so callers can rewrite cross-macro `ExecuteMacro` references.

## Documented limitations (partial coverage by design)

- `km_trigger_manager set_enabled` — KM 11 trigger plists store no per-trigger enabled bit (verified 2026-05-14 by inject-and-read-back probe). Use `km_macro_editor set_enabled` for the parent macro's enabled state.
- `km_window_manager` against Finder — move/resize/arrange AppleScript succeeds but Finder reuses window indices in surprising ways; `get_info` may poll the menubar instead of the moved window. Target-app issue, not a tool bug.
- `km_create_plugin_action` `output_dir` — must be under the MCP server CWD; relative paths outside the server root are blocked by the path-traversal guard.

## How to reproduce

The full smoke harness is in:

- `specs/km_mcp_smoke_20260514.md` — round 1 (the 30-tool sweep)
- `specs/km_mcp_fix_broken_tools_20260514.md` — round 2 (the three implementations above)

Each spec lists the call shape, expected response, and the KM-side side-effect to verify.
