# KM MCP Showcase Macro

## Goal

Probe the breadth of the Keyboard Maestro MCP server by authoring one
macro that exercises every shipping action emitter plus the full
control-flow surface in a single round trip. Leave it in place under a
dedicated group so the user can inspect it.

## Constraints

- Single macro, all actions appended through MCP tools (no manual edit).
- Lives in a new group `KM MCP Showcase` so cleanup is trivial.
- Bound to a hotkey so it's actually invokable.

## Acceptance Criteria

- [ ] Group `KM MCP Showcase` exists and is enabled.
- [ ] Macro `Showcase R13 — Full Tour` exists in that group with a
      hotkey trigger (any combination not already in conflict).
- [ ] Macro contains **≥ 25 actions** across **≥ 8 distinct
      MacroActionType** values, including at least one of each:
      `Pause`, `SetVariableToText`, `ExecuteAppleScript`,
      `ExecuteShellScript`, `Notification`, `If`, `For`, `Switch`,
      `TryCatch`.
- [ ] At least one nested control structure (e.g. `If` inside `For`,
      or `Switch` inside `TryCatch`).
- [ ] `km_action_builder list` returns the full sequence without
      truncation; final action count reported in the session summary.
