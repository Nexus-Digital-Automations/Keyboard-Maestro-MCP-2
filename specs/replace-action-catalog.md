# Spec: Truthful action catalog + plug-in support in km_action_builder

## Goal
`km_list_action_types` returns only action types that actually work; clients can both discover and insert third-party plug-ins via the existing MCP tools.

## Acceptance criteria

### AC-1: km_list_action_types is truthful
- AC-1.1: Calling `km_list_action_types()` with no args returns `success: True` and includes every installed plug-in from `~/Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions/` whose plist parses and has both `Name` and `Title`.
- AC-1.2: The response includes 6 built-in entries with identifiers exactly `pause`, `type_text`, `paste`, `set_variable`, `run_applescript`, `execute_macro`.
- AC-1.3: The response does NOT contain any of the old 146 identifiers (`"Type a String"`, `"Activate a Specific Application"`, `"If Then Else"`, etc.). Total `total_available` equals `6 + plugin_count`.
- AC-1.4: Plug-in entries have `category == "plug_in"` and an extra `plugin_metadata` block with keys `title`, `result_targets`, `parameter_types`, `bundle_path`.
- AC-1.5: `km_list_action_types(category="plug_in")` filters to plug-ins only. `km_list_action_types(search="MCP Smoke")` matches by identifier or description.
- AC-1.6: Response envelope retains the existing schema: `{success, data: {actions, summary, categories}, metadata: {timestamp, correlation_id, registry_version}}`.

### AC-2: km_action_builder.append handles plug_in
- AC-2.1: `km_action_builder(operation="append", macro_id="<m>", action_type="plug_in", action_config={"plugin_identifier": "MCP Smoke Plugin", "parameters": {"Message": "x"}, "result_target": "Variable", "variable_name": "V"})` returns `success: True` and the appended action exists in the macro.
- AC-2.2: Reading the macro back via `operation="list"` shows the new action with `MacroActionType=ExecutePlugIn` (NOT the corruption marker `Log "Invalid XML From AppleScript"`).
- AC-2.3: Missing `plugin_identifier` returns `UNSUPPORTED_ACTION_TYPE`. The error message lists `plug_in` among supported action types and documents the required `plugin_identifier` config field.
- AC-2.4: `result_target="Window"` emits `<key>DisplayResultsInWindow</key><true/>`. `result_target="Variable"` with `variable_name="V"` emits `<key>Variable</key><string>V</string>`. Other result_target values default to neither (KM uses bundle plist default).
- AC-2.5: Parameter dict order is preserved in the emitted XML (`<key>k1</key><string>v1</string><key>k2</key><string>v2</string>`...).
- AC-2.6: All parameter keys and values are XML-escaped (verified by injecting `<&>` characters).

### AC-3: End-to-end plug-in execution
- AC-3.1: Build pipeline: append plug_in action with `Message=hello`, then `km_execute_macro` the host macro, then read the variable named in `variable_name` — value equals `"Hello from MCP Smoke Plugin: hello"` (no surrounding whitespace).

### AC-4: Old broken surface gone
- AC-4.1: `km_add_action` is not exposed by the MCP server (not in the tool listing).
- AC-4.2: `from src.actions import ActionRegistry` raises `ImportError` (or the module no longer exists).
- AC-4.3: `from src.actions import ActionBuilder` raises `ImportError`.
- AC-4.4: No file under `src/` references `_XML_VERIFIED_ACTION_TYPES` or `XML_GENERATION_REJECTED`.

### AC-5: Quality gates
- AC-5.1: `ruff check src/` returns zero errors.
- AC-5.2: `pytest tests/` passes after deletion of tests for `ActionRegistry`/`ActionBuilder`.
- AC-5.3: `docs/km_mcp_audit_report.md` F17 entry includes "RESOLVED" with a brief pointer to the new architecture.
- AC-5.4: `README.md` tool table no longer lists `km_add_action`; `km_action_builder` row mentions plug-in support.
- AC-5.5: install_hint in `plugin_action_tools.py` says "Editor" (not "Engine").

## Non-goals
- Fixing per-type emitters for the 146 built-in KM action types. Those are deleted, not fixed. If users want more built-ins, add them one at a time to `_build_action_xml` in `action_builder_tools.py` after verifying the plist format with KM's own "Copy as XML".
- Adding write capability to the plug-in scanner (it's read-only).
- Modifying the live KM Actions folder during tests (tests use `KM_PLUGIN_ACTIONS_DIR` env override).

## Notes
- The bundle `Name` value is used as `PlugInIdentifier` in the macro action XML; KM matches by this string.
- `MCP Smoke Plugin` is installed from the prior smoke test and serves as the live verification fixture.
