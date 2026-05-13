# Spec — Full KM MCP smoke test (every tool, name + UUID, fix-inline)

## Goal

Exercise every `mcp__keyboard-maestro__km_*` tool exposed by this server
through live MCP calls. Every operation that takes a `macro_id` /
`group_id` / `identifier` is exercised twice: once by **name**, once by
**UUID**. Where the tool already mutates KM state, verify the side-effect
via a follow-up read call. Any failure → root-cause fix in source →
re-run that tool → commit. Push at end.

## Tools under test (27)

Group A — Macro CRUD / discovery
- `km_list_macros`
- `km_macro_group_manager` (list, create, rename, set_enabled, delete)
- `km_macro_editor` (create, rename, duplicate, set_enabled, delete)
- `km_create_macro` (templates)
- `km_list_templates`
- `km_move_macro_to_group`
- `km_execute_macro`

Group B — Triggers / hotkeys
- `km_trigger_crud` (list, get, add, update, remove, replace_all)
- `km_trigger_manager` (list, add, remove, clear, set_enabled)
- `km_create_hotkey_trigger`
- `km_list_hotkey_triggers`

Group C — Actions / conditions / control flow
- `km_list_action_types`
- `km_add_action`
- `km_action_builder` (list, append, delete, clear)
- `km_add_condition`
- `km_control_flow`

Group D — Engine / variables / tokens
- `km_variable_manager` (set/get/list/delete on global + password)
- `km_engine_control` (status, calculate, process_tokens, search_replace, reload)
- `km_token_processor`
- `km_token_stats`

Group E — UI / app / window / notifications
- `km_application_control` (list_running, get_state, launch, activate, quit)
- `km_window_manager` (get_screens, get_info, arrange, move, resize)
- `km_input_simulator` (read-only validation; do NOT actually type into editor)
- `km_interface_automation` (read-only validation; do NOT actually click)
- `km_notifications` (notification, alert is interactive-skip, hud, sound)
- `km_notification_status`
- `km_dismiss_notifications`

## Sandbox

- Group: `SmokeFullMatrix` (created fresh by the test)
- Macros: `SmokeMatrixA`, `SmokeMatrixB`, plus the carryover macros
  `SmokeRenamedByName` (934C561F-3B4B-4E10-8A2E-490CF9380FE0) and
  `SmokeDupByName` (CD137E1F-1545-4BA5-8340-A2F9FB51C515) from the prior session.
- Variables: `MCPSmoke_*`
- After run: keep group + macros for inspection (user-prior policy).

## Side-effect verification

After every mutation, the next call must read state back and assert the
expected change. Specifically:
- create → `km_list_macros` filter shows it; `km_macro_group_manager.list`
  shows the group if a group was created.
- rename → list shows the new name, no old name.
- set_enabled → list shows `enabled` reflecting the call.
- delete → list does not show it.
- trigger add → `km_trigger_crud list` shows count went up by 1.
- variable set → `km_variable_manager.get` returns the value.

## Definition of pass

For each tool x identifier-shape (name | UUID):
- `success: true` in the response,
- side-effect verification (where applicable) confirms intended state,
- no Python traceback strings in `error` envelopes,
- no `UNKNOWN_ERROR` codes.

Anything else = defect → fix in source → re-run.

## Fix policy

Per user answer: fix root cause, one commit per distinct defect, push
once at end (or after each commit if convenient). No workarounds, no
"try except pass" hides.

## Out of scope

- IoT, voice, smart-home, identity (not registered in this server build).
- `km_build_plugin_action` (already audited).
- Modal `km_notifications` alert (requires interactive button click).
- `km_input_simulator.type_text` and `km_interface_automation.click` —
  these will actually move the mouse / press keys on the user's machine.
  We exercise their validation paths only (e.g., missing-arg error path)
  unless the user explicitly opts in to live UI events.

## Output

- Punch list kept in-session (TaskList).
- Audit append to `docs/km_mcp_audit_report.md` at the end with the
  date-stamped matrix verdict.
