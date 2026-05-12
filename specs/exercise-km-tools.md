# Exercise & Bug-Fix All KM MCP Tools

## Goal
Call every `mcp__keyboard-maestro__*` tool once with realistic inputs, log results, and fix any bug encountered (crashes, ValidationError, wrong returns, broken imports).

## Tools (22)
Engine/discovery: `km_engine_control`, `km_token_stats`, `km_list_action_types`, `km_list_templates`, `km_list_macros`, `km_list_hotkey_triggers`, `km_notification_status`
Variables/tokens: `km_variable_manager`, `km_token_processor`
Notifications: `km_notifications`, `km_dismiss_notifications`
Macro CRUD: `km_create_macro`, `km_add_action`, `km_add_condition`, `km_add_system_trigger`, `km_create_hotkey_trigger`, `km_control_flow`, `km_move_macro_to_group`, `km_macro_editor`, `km_execute_macro`
UI/window: `km_window_manager`, `km_interface_automation`

## Test Naming
- Macros: `mcp_test_*` (cleaned up at end)
- Variables: `mcp_test_var`
- Group: `Global Macro Group` (default)

## Acceptance
- Each tool exercised at least once
- Failures investigated → root cause identified → fix landed (or documented as KM-app limitation)
- Test artifacts deleted
- Lint clean on touched files

## Out of Scope
- New features, refactors, docs unless required to fix a bug
- Tools that require hardware not present (cameras etc.) — skip with note
