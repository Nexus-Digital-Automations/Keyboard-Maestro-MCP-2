# Spec — PR2: switch_case + km_create_macro template parameters

## Context

PR1 (commit `c159039`) shipped for_loop / while_loop / until_loop / try_catch. PR2 closes the remaining km_control_flow gap (`switch_case`) and the long-standing `km_create_macro` template-with-parameters gap (audit items 1 + 2). Both require capture-then-emit patterns proven in PR1.

## Workstream A — switch_case

### A1. Capture phase (gates the API surface)

Build a Switch action manually in KM 11 Editor with realistic structure, then dump via `osascript -e 'tell application "Keyboard Maestro" to get xml of action N of macro id ...'`. Save under `tests/fixtures/km_control_flow/`.

Capture targets:

- One Switch per Source type. KM-supported Source values discovered by inject-and-readback probe (PR1 used the same approach on for_loop's CollectionType, finding 13 valid values from C{Type}Collection.nib resources). Look for `C*Switch*.nib` or similar editor nibs in `/Applications/Keyboard Maestro.app/Contents/Resources/en.lproj/`.
- One Switch with multiple CaseEntries using different ConditionType per case (Is, IsNot, Contains, MatchesRegex, etc.). Probe each candidate by injection.
- One Switch with an Otherwise/Default case to determine how KM represents it (separate `OtherwiseActions` key, sentinel `ConditionType` like "Otherwise", or trailing CaseEntry).

### A2. Emitter

New module `src/integration/km_switch_case_xml.py`:

- `build_switch_case_xml(source, case_entries_xml, *, otherwise_actions_xml="")` → top-level Switch plist `<dict>`. Source is one of the captured strings.
- `build_case_entry(test_value, condition_type, actions_xml)` → one CaseEntry `<dict>` for the array.
- `SUPPORTED_SOURCES`, `SUPPORTED_CASE_CONDITIONS` tuples derived from the capture.
- `UnsupportedSwitchSource`, `UnsupportedCaseCondition` exceptions mirroring PR1's `UnsupportedCollectionType` shape.

### A3. Dispatcher

Add `_emit_switch_case` to `src/server/tools/control_flow_tools.py` mirroring `_emit_for_loop`:

- Validate `cases: list[dict]` non-empty; each entry needs `test_value`, `condition_type`, `actions`.
- Build CaseEntries via `_render_inner_actions` (existing) for inner actions per case, then `build_case_entry` per case.
- If `default_actions` non-empty, render and pass via the captured KM mechanism.
- Append via `append_macro_action_async`.

Add `switch_case` arm in `km_control_flow` BEFORE the legacy AST/Apply path. Once switch is on the new path, the `_apply_control_flow_to_macro` NotImplementedError, `_generate_km_control_flow`, `_get_km_action_type`, `_generate_km_xml`, and the dead `src/integration/km_control_flow.py` module become unreachable. Verify with `references_to`, then delete in this PR.

### A4. Validator

Update `_validate_control_flow_inputs`:

- For switch_case: require `cases` non-empty, each case has the three keys.
- Source validation: passed through to the emitter (error envelope normalized via the existing `_validation_failure` helper).

### A5. Tests

`tests/test_integration/test_control_flow_emitters.py` extension:

- `test_switch_case_xml_has_required_keys`: substring assertions on MacroActionType=Switch, CaseEntries, Source.
- `test_switch_case_supports_all_KM_sources`: iterate `SUPPORTED_SOURCES`, assert each emits without exception.
- `test_switch_case_with_otherwise`: confirm the captured default-case mechanism is honored.
- `test_switch_case_per_case_condition_types`: iterate `SUPPORTED_CASE_CONDITIONS`, build entries.

`tests/test_tools/test_control_flow_tools.py`:

- Replace the existing `test_switch_case_success` (mocks dead path) with the new emitter mock pattern.
- Add validation test for missing cases / malformed case entries.

### A6. Documentation

- `docs/km_mcp_tool_status.md`: move switch_case out of "Documented limitations".
- Append a "round 7" entry summarizing PR2.

## Workstream B — km_create_macro template parameters

### B1. Action emitters needed

Per PR1 spec table:

| Template | Action types | Already in `_build_action_xml` 6-set? |
|---|---|---|
| `app_launcher` | `activate_application` | NO — new emitter |
| `text_expansion` | `type_text` (or `paste`) | YES |
| `window_manager` | `manipulate_window` (Move/Resize/etc.) | NO — new emitter |
| `file_processor` | `execute_shell_script` or file action | NO — new emitter |
| `hotkey_action` | depends on user `parameters.actions` list | depends |

Three new emitters needed: `activate_application`, `manipulate_window`, `execute_shell_script`. Each capture-driven against KM 11 XML.

Capture by:
1. Drag the action onto a sandbox macro in KM Editor with realistic params.
2. Dump XML.
3. Save under `tests/fixtures/km_actions/<action_type>.xml`.
4. Mirror the shape exactly in `_build_action_xml`.

### B2. Atomic .kmmacros plist build

Extend `src/integration/kmmacros_import.py`:

- Add an `actions: list[dict[str, Any]] | None = None` parameter to `build_kmmacros_plist` and `create_empty_macro` (rename to `create_macro_with_actions` or keep the name and treat actions as optional).
- When actions is non-empty, embed each action `<dict>` inside the macro's `Actions` array of the plist BEFORE writing the file. KM imports the complete macro in one open call.
- Verify by import + read-back that KM accepts our atomic-build XML for each action emitter.

### B3. Templates wire-up

`src/server/tools/creation_tools.py` (lines 163–183):

- Replace blanket `UNSUPPORTED_TEMPLATE` with template-specific dispatch.
- Each template module in `src/creation/templates.py` already builds action dicts. Translate each dict → XML via `_build_action_xml`. Concatenate into one XML blob. Pass to `build_kmmacros_plist` (extended above).
- For `template="hotkey_action"` with parameters: bake actions atomically, then attach hotkey trigger via `km_create_hotkey_trigger` integration helper as a follow-up call (single round-trip, since trigger isn't an action).
- If a template asks for an action_type outside `_build_action_xml`'s extended set, return `UNSUPPORTED_TEMPLATE_ACTION` with the offending type listed.

### B4. Tests

`tests/test_integration/test_action_emitters.py` (new):

- `test_activate_application_emits_canonical_keys`: captured-fixture substring diff.
- `test_manipulate_window_emits_canonical_keys`: same.
- `test_execute_shell_script_emits_canonical_keys`: same.

`tests/test_tools/test_creation_tools.py` (extend or new):

- One success test per template (5 total) using mocked KM.
- One regression test that template→XML→KM round-trips for `window_manager` (most novel emitter).

### B5. Documentation

- `docs/km_mcp_tool_status.md`: move template-with-parameters out of "Documented limitations".
- Update the "round 7" entry.

## Out of scope for PR2

- The 146-action emitter for arbitrary `km_add_action` types (audit item 3). Stays gated by `XML_GENERATION_REJECTED`.
- KM-side limitations: process_tokens system-token expansion, Finder window indices, execute_macro by name lag.

## Acceptance criteria

PR2 ships when:

1. `km_control_flow control_type="switch_case"` returns `success: true` with a real Switch action visible in KM for each captured Source type and per-case ConditionType.
2. `km_create_macro template="<each of 5>"` with realistic `parameters` returns `success: true` with a real macro containing the expected action sequence; verified by `km_action_builder list`.
3. `_apply_control_flow_to_macro`, `_generate_km_control_flow`, `_get_km_action_type`, `_generate_km_xml`, and `src/integration/km_control_flow.py` are deleted; `references_to` returns empty for each.
4. `pytest tests/` shows fewer or equal failures vs PR1 baseline (no regressions).
5. `ruff check` clean on touched files (modulo pre-existing issues).
6. Live test against KM 11 confirms each path. Sandbox macros cleaned up after run.
7. `docs/km_mcp_tool_status.md` and `docs/km_mcp_audit_report.md` updated with round 7.
8. Branch pushed and PR URL returned.
