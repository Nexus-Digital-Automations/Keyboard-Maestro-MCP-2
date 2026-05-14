# Round 8 Smoke — Acceptance Criteria

Source: `docs/km_mcp_tool_status.md` § "Next: post-restart smoke checklist".

## Scope
Sections A (round-6 fixes), B (round-7 fixes), C (regression sweep).

## Sandbox
- Group: `KM MCP R6 Sandbox` (create if absent).
- One scratch macro per check; delete after each section.
- Group kept in place at end.

## Pass criteria (per check)

**A1. for_loop** — `success=true`, `data.macro_action_type="For"`, `data.collection_type` matches input. Run for Range, LinesIn, Files, Variables.
**A2. while_loop / until_loop** — `success=true`, `data.macro_action_type="While"` / `"Until"`.
**A3. try_catch** — `success=true`. Bonus: variable `Caught` becomes `"yes"` after executing wrapper with intentional error.
**A4. Variable condition value preservation** — after `km_add_condition(operand="MyVar=ABCXYZ123")`, `km_action_builder list` shows the operand value (not literal "value").
**A5. set_enabled rejection** — error message cites "KM 11 stores no per-trigger enabled bit" and points to `km_macro_editor set_enabled`.
**A6. refresh_templates name coerce** — call with macro name (not UUID) succeeds and response includes `data.resolved_macro_id`.
**A7. arrange window_info_source flag** — response includes `window_info_source` field.

**B8. switch_case Variable + Otherwise** — `data.case_count=3`, `data.has_otherwise=true`.
**B9. switch_case other sources** — Clipboard / Calculation / Text all succeed.
**B10. switch_case unsupported source** — `JSON` returns `VALIDATION_ERROR` listing 5 supported values.
**B11. Templates with parameters (×5)** — each `km_create_macro` returns `success=true`; hotkey_action also has `data.hotkey_attached=true`.
**B12. Template unsupported inner action** — `hotkey_action` with `action="wave_hands"` returns `UNSUPPORTED_TEMPLATE_ACTION`.
**B13. Alert duration fix** — `km_notifications(notification_type="alert", duration=5)` returns `success=true` without crash.

**C14. Regression sweep** — `km_engine_control` ops, `km_variable_manager` (4 scopes), `km_macro_editor` (5 ops), `km_action_builder` (4 ops) still pass.

## Reporting
Append a "Round 8" section to `docs/km_mcp_audit_report.md` with PASS/PARTIAL/FAIL matrix in prior-round format.
