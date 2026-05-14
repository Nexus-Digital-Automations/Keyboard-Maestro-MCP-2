# KM MCP Tool Status — 2026-05-14

Smoke-test snapshot of all 30 Keyboard Maestro MCP tools. "Worked
out-of-the-box" = passed the 2026-05-14 smoke without code changes.
"Fixed" = was broken before this session; commits cited below repaired it.
"Documented limitation" = part of the surface intentionally returns
`UNSUPPORTED_OPERATION` with a recovery suggestion.

> ⚠️ MCP server runs from `~/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP-2`
> (main branch). The `session/20260514-125257-36011` merge is live in main as
> of `460edc5`; a Claude Code restart is required after pulling for the long-
> running MCP process to reload Python modules.
>
> ✅ **2026-05-14 re-verification (session `20260514-144807-63106`):** all six
> "Fixed in this session" tools below confirmed working against the running
> MCP server. One additional bug found and fixed — see `f23cd30` under
> "Fixed in this session" → `km_notifications` alert duration.

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

### `f23cd30` — alert duration honored (session 20260514-144807-63106)

- `km_notifications` alert — `duration` was silently dropped, so callers blocked
  for the full 30-second AppleScript timeout waiting for a user click. The
  AppleScript template now appends `giving up after N` (integer-rounded
  seconds) when `spec.duration` is set, so alerts auto-dismiss as documented.
  Live in source; takes effect after the next Claude Code restart reloads the
  MCP process.

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

## Next: post-restart smoke checklist (rounds 6 + 7)

The live MCP server runs from `~/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP-2` and caches Python at startup. The session-branch fixes need a **merge into main + Claude Code restart** before they're live. After restart, run the checks below in order. Each line cites the commit that introduced the change and the exact tool call to issue.

Sandbox: create or reuse macro group `KM MCP R6 Sandbox` and a scratch macro inside it (any UUID). Delete sandbox macros after each section.

### A. Round-6 control flow (commit `c159039`)

1. **for_loop** — one call per collection type the caller cares about. Minimum: `Range`, `LinesIn`, `Files`, `Variables`. Each call:
   ```
   km_control_flow(macro_identifier=<scratch>, control_type="for_loop",
     iterator="i", collection_dict={"type": "Range", "start": "1", "end": "5"},
     loop_actions=[{"type": "pause", "seconds": 0.1}])
   ```
   Expected: `success=true`, `data.macro_action_type="For"`, `data.collection_type` matches input. Read back via `km_action_builder list` — the action should display as "For Each i in Range 1 to 5".
2. **while_loop / until_loop** — one call each:
   ```
   km_control_flow(macro_identifier=<scratch>, control_type="while_loop",
     condition="MyVar", operator="equals", operand="yes",
     loop_actions=[{"type": "pause", "seconds": 0.1}])
   ```
   Expected: KM displays "While the following are true: MyVar is yes". Same shape for `until_loop` with `data.macro_action_type="Until"`.
3. **try_catch**:
   ```
   km_control_flow(macro_identifier=<scratch>, control_type="try_catch",
     try_actions=[{"type": "pause", "seconds": 0.1}],
     catch_actions=[{"type": "set_variable", "variable": "Caught", "text": "yes"}])
   ```
   To verify the trap fires, replace the try action with one that errors (e.g. `{"type": "execute_macro", "target_macro": "DoesNotExist"}`), `km_execute_macro` the wrapper, then `km_variable_manager get Caught` — should be `"yes"`.
4. **Variable / Text condition value preservation** (the bug that affected every shipped if/then/else and add_condition):
   ```
   km_add_condition(macro_identifier=<scratch>, condition_type="variable",
     operator="equals", operand="MyVar=ABCXYZ123")
   ```
   Read back the appended IfThenElse action via `km_action_builder list` — the condition should display "MyVar is ABCXYZ123" (NOT "MyVar is value", which was the pre-fix symptom). If you see "value" the round-6 fix didn't activate.
5. **set_enabled rejection message** — `km_trigger_manager set_enabled` should return error mentioning "KM 11 stores no per-trigger enabled bit" and pointing to `km_macro_editor set_enabled`.
6. **refresh_templates name coerce** — `km_refresh_action_templates(macro_id=<a name, not a UUID>, confirm=true, limit=1)` should succeed and report `data.resolved_macro_id`.
7. **arrange post-bounds flag** — `km_window_manager(operation="arrange", window_identifier="Finder", arrangement="left_half")` response includes `window_info_source` field with value `"post_operation"` or `"pre_operation"`.

### B. Round-7 switch + templates (commit `fd59947`)

8. **switch_case (Variable source + Otherwise)**:
   ```
   km_control_flow(macro_identifier=<scratch>, control_type="switch_case",
     source="Variable", condition="MyVar",
     cases=[
       {"condition_type": "Is", "test_value": "v1", "actions": [{"type": "pause", "seconds": 0.1}]},
       {"condition_type": "Contains", "test_value": "v2", "actions": [{"type": "pause", "seconds": 0.1}]},
     ],
     default_actions=[{"type": "set_variable", "variable": "FellThrough", "text": "yes"}])
   ```
   Expected: `data.case_count=3`, `data.has_otherwise=true`. KM displays "Switch on Variable MyVar" with three cases.
9. **switch_case other sources** — repeat call 8 with `source="Clipboard"` (drop `condition`), `source="Calculation"` with `condition="1+1"`, `source="Text"` with `condition="%CurrentUser%"`. All four should succeed.
10. **switch_case unsupported source** — `source="JSON"` should return `VALIDATION_ERROR` listing the 5 supported values; KM should NOT silently coerce. (If our pre-validation lets it through, KM will normalize to `Clipboard` — which is the bug we're guarding against.)
11. **Templates with parameters — all 5 atomic creates**:
    ```
    km_create_macro(name="T_AppLauncher", template="app_launcher",
      group_name="KM MCP R6 Sandbox",
      parameters={"app_name": "Finder", "bundle_id": "com.apple.finder"})
    ```
    Repeat for each template: `text_expansion` (`expansion_text`), `file_processor` (`script`), `window_manager` (`operation`, `x`, `y`, `width`, `height`), `hotkey_action` (`action`, `text`, `hotkey`, `modifiers`). Each should return `success=true` and the resulting macro should have exactly the expected action count via `km_action_builder list`. `hotkey_action` additionally needs `data.hotkey_attached=true` and the trigger should appear in `km_list_hotkey_triggers`.
12. **Template with unsupported inner action** — `km_create_macro(template="hotkey_action", parameters={"action": "wave_hands", ...})` should return `UNSUPPORTED_TEMPLATE_ACTION` naming the offender.
13. **f23cd30 alert-duration fix** (folded into the round-6 restart cycle) — `km_notifications(notification_type="alert", title="t", message="m", duration=5)` should return `success=true` without crashing.

### C. Regression sweep (anything we touched but didn't intend to change)

14. Re-run any tool that previously appeared in "Worked out-of-the-box" — focus on `km_engine_control` ops, `km_variable_manager` (all 4 ops), `km_macro_editor` (all 5 ops), `km_action_builder` (all 4 ops). Each should still pass — if anything regresses, the round-6/7 changes leaked into a shared code path.

### Sandbox cleanup

After each section, `km_macro_editor delete` for any sandbox macros. Leave the `KM MCP R6 Sandbox` group itself in place for future runs.

### Reporting

Use the format from prior rounds (round 4 in `docs/km_mcp_audit_report.md`): one line per check with PASS / PARTIAL / FAIL + a one-sentence note for anything not PASS. Append a "round 8" section to the audit report with the matrix.

## How to reproduce

The full smoke harness is in:

- `specs/km_mcp_smoke_20260514.md` — round 1 (the 30-tool sweep)
- `specs/km_mcp_fix_broken_tools_20260514.md` — round 2 (the three implementations above)

Each spec lists the call shape, expected response, and the KM-side side-effect to verify.
