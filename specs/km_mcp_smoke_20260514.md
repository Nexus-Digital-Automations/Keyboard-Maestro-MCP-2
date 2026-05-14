# KM MCP Full Smoke Test — 2026-05-14

## Goal
Exercise every operation of every `mcp__keyboard-maestro__*` tool against a live KM engine, verify KM-side side effects, and fix any bugs found.

## Scope
30 tools total (per server instructions). All operations/modes covered. Verification via list/get after mutating ops.

## Cleanup
- Dedicated macro group: `MCP Smoke 20260514`
- All test macros created inside that group
- Delete the group at end of session (best-effort)
- Delete any tmp plugin bundle dirs created under `cache/smoke_plugin/`

## Tools to cover
1. km_list_macros — list, filter, sort, paginate
2. km_list_action_types — full, by-category, search, plug_in
3. km_list_templates
4. km_search_actions — empty, scored query, filters
5. km_refresh_action_templates — limit=1 smoke (slow)
6. km_create_macro — at least 2 templates (custom, hotkey_action)
7. km_macro_editor — create, rename, duplicate, set_enabled, delete
8. km_macro_group_manager — list, create, rename, set_enabled, delete
9. km_move_macro_to_group — move, create_group_if_missing
10. km_action_builder — list, append (pause, type_text, set_variable, plug_in), delete, clear
11. km_execute_macro — applescript, url
12. km_variable_manager — set, get, list, delete (global), password set/get behavior
13. km_engine_control — status, reload, calculate, process_tokens, search_replace
14. km_token_processor — text context, preview, with vars
15. km_token_stats
16. km_create_hotkey_trigger — basic, with modifiers
17. km_list_hotkey_triggers — all, filtered by macro
18. km_add_system_trigger — login, system_wake
19. km_trigger_crud — list, get, add, update, remove, replace_all
20. km_trigger_manager — list, add, set_enabled, remove, clear
21. km_add_condition — text, application
22. km_control_flow — if_then_else, for_loop
23. km_create_plugin_action — emit bundle to tmp dir
24. km_window_manager — get_screens, get_info, arrange, move, resize
25. km_application_control — list_running, get_state, launch, activate, quit
26. km_notifications — notification, hud, sound, alert (skip-blocking)
27. km_notification_status
28. km_dismiss_notifications

## Acceptance
- Every tool returns `success: true` for happy-path call(s), OR a documented expected-failure (e.g. KME unreachable for refresh templates).
- Side-effect-bearing ops verified via a follow-up read (list/get/status).
- Any bug → fix in src, re-run that tool, commit.
- Final summary: per-tool PASS/FAIL with one-line reason.
