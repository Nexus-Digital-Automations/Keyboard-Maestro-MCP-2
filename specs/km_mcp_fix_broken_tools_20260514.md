# Fix Three Broken KM MCP Tools — 2026-05-14

Smoke test (`specs/km_mcp_smoke_20260514.md`) marked three tools as
unfixable; user reversed that decision with "Fix all three". This spec
captures the real plan.

## Tools

### 1. `km_add_condition`

**Old behaviour:** short-circuited with `UNSUPPORTED_OPERATION`.

**New behaviour:** emit a real KM `IfThenElse` action plist and append it
to the target macro via `append_macro_action_async`.

**Inputs translated:**
- `condition_type` ∈ {`variable`, `text`, `application`, `calculation`}
  (the original `system` and `logic` types are explicitly UNSUPPORTED —
  KM has no single-shot mapping for them).
- `operator` ∈ {`equals`, `contains`, `greater`, `less`, `regex`,
  `exists`} → mapped to the per-condition-type KM strings
  (`VariableConditionType=Is/Contains/...`, etc.).
- `operand` — comparison value (escaped).
- `case_sensitive`, `negate` — passed through where applicable.
- `action_on_true` / `action_on_false` — when non-None, treated as a
  macro NAME and emitted as an inner `ExecuteMacro` action inside
  `ThenActions` / `ElseActions`. When None, the array stays empty (still
  valid; user can append more via `km_action_builder`).
- `timeout_seconds` — set via `TimeOutAbortsMacro` (kept true by default).

**Acceptance:**
- Calling `km_add_condition(macro_id, "variable", "equals", "5", action_on_true="SomeMacro")`
  appends an IfThenElse action visible in the KM Editor with the right
  Variable condition and a nested ExecuteMacro inner action.
- Response includes the new action's index and the canonical KM
  `MacroActionType` (`IfThenElse`).
- Unsupported condition_type returns UNSUPPORTED_OPERATION with a clear
  recovery suggestion.

### 2. `km_control_flow` (if_then_else mode only)

**Old behaviour:** validated inputs, then raised NotImplementedError →
returned `UNSUPPORTED_OPERATION`.

**New behaviour:** for `control_type="if_then_else"`, build the same
IfThenElse plist via the shared helper (`src/integration/km_if_then_else_xml.py`)
and append it. The structured `actions_true: list[dict]` and
`actions_false: list[dict]` are translated into inner action `<dict>`s
using `_build_action_xml` from `action_builder_tools.py`.

Other control types (`for_loop`, `while_loop`, `switch_case`,
`try_catch`) continue to raise NotImplementedError. Out of scope.

**Acceptance:**
- `km_control_flow(macro_id, "if_then_else", condition="MyVar",
  operator="equals", operand="hello",
  actions_true=[{"type":"pause","seconds":1}])` appends an IfThenElse
  action with a Variable condition and a nested Pause action.

### 3. `km_add_system_trigger`

**Old behaviour:** UNSUPPORTED_OPERATION (KM 11 AppleScript rejects
non-HotKey trigger plists at `make new trigger`).

**New behaviour:** export → edit → reimport. Pipeline:

1. `tell application "Keyboard Maestro" to get xml of macro id "..."`
   — pull the existing macro plist.
2. Parse the plist; locate `Triggers` array; append the appropriate
   system-trigger dict:
   - `engine_launch` / `login` → `{ MacroTriggerType: "EngineLaunch",
     FireType: "ApplicationLaunched" }`
   - `system_wake` → `{ MacroTriggerType: "WakeTrigger" }`
3. Mint a **new** macro UID (the imported macro can't reuse the old UID
   without triggering KM's "duplicate UID — replace?" GUI prompt that
   would block the AppleScript).
4. Build a `.kmmacros` wrapper plist for the modified macro under the
   macro's current group.
5. Look up the old macro's group + UID first (because step 6 deletes it).
6. Delete the original macro.
7. `tell application "Keyboard Maestro" to open POSIX file "..."` to
   import the modified copy.
8. Wait for the new UID to appear.

**Documented limitation:** the macro's UUID changes. Any other macro
that referenced the old UID via `ExecuteMacro` must be updated
separately (out of scope; tool returns `old_macro_id` and `new_macro_id`
so the caller can handle that).

**Acceptance:**
- Calling on a `wake` kind appends a WakeTrigger to the target macro;
  inspection in the KM Editor shows "System wakes" as a trigger.
- Response includes `old_macro_id`, `new_macro_id`, `uuid_changed: true`
  with a warning string.
- Failure to export, parse, delete, or import returns a clear error
  with a recovery suggestion.

## Test plan

Manual smoke (next session, after MCP server restart on main):

1. `km_create_macro(name="ITE Smoke", template="custom", group_name="MCP Smoke Fix 20260514")`
2. `km_add_condition(macro_id="ITE Smoke", condition_type="variable",
   operator="equals", operand="42", action_on_true="ITE Smoke")` →
   verify IfThenElse action appears.
3. `km_control_flow(macro_id="ITE Smoke", control_type="if_then_else",
   condition="AppName", operator="contains", operand="Finder",
   actions_true=[{"type":"pause","seconds":1}])` → verify second
   IfThenElse appended with nested Pause.
4. `km_add_system_trigger(macro_id="ITE Smoke", trigger_kind="system_wake")`
   → verify wake trigger now attached, new UID returned.
5. Clean up the smoke group.

## Out of scope

- Multi-condition (`ConditionList` with > 1 entry) — single condition only.
- Cross-macro reference rewriting after UUID change in #3.
- `for_loop` / `while_loop` / `switch_case` / `try_catch` real XML.
