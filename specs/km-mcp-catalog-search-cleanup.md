# Spec: KM MCP Catalog Expansion, Semantic Search, and Tool-Surface Cleanup

Plan: `/Users/jeremyparker/.claude/plans/what-s-missing-if-nifty-map.md` (approved).

Multi-phase execution. Each phase is a discrete commit with its own acceptance criteria.

---

## Phase 1 — Tracer: ClickAtFoundImage end-to-end

**Acceptance:**
- `src/server/data/km_action_templates.json` exists with one entry keyed `ClickAtFoundImage` containing real captured XML (not hand-written).
- `src/server/data/km_builtin_actions.json` exists with one entry for the same action: `MacroActionType`, `identifier`, `category`, `title`, `description`, `keywords`, `task_hints`, `parameters`, `builder_supported: true`, `result_targets`.
- `src/server/tools/action_builder/templates.py` has `load_templates()` and `render_action_xml(macro_action_type, params) -> str` using plistlib substitution (not regex string replacement).
- `_build_action_xml` in `action_builder_tools.py:62` consults templates when no hand-written emitter matches; legacy 6 emitters still hit first.
- `km_action_builder(operation="append", action_type="click_at_found_image", action_config={...})` returns `success: true` against a real macro on this machine.
- Macro execution does not produce `Invalid XML From AppleScript` placeholder.
- Existing tests in `tests/test_tools/test_action_builder_tools.py` still pass.

## Phase 2 — Template engine + paste_xml escape hatch

**Acceptance:**
- `templates.py` covers all currently-captured templates (still just one after Phase 1, more after Phase 3).
- `paste_xml` action_type accepted by `km_action_builder`. Validates: input parses as plist `<dict>`, top-level dict has `MacroActionType` key. Rejects with `VALIDATION_ERROR` otherwise.
- `tests/test_tools/test_action_templates.py` — round-trip for every loaded template; bad-XML rejection cases.

## Phase 3 — Scrape script

**Acceptance:**
- `scripts/capture_km_action_templates.py` runs end-to-end on this machine.
- Produces `src/server/data/km_action_templates.json` containing ≥120 entries (target ~150).
- Produces seeded `src/server/data/km_builtin_actions.json` with identifier + title + category populated from menu structure; `keywords`/`task_hints` empty (curation in a follow-up).
- Captured XML is normalized: no `ActionUID` or transient fields.
- Idempotent: re-running produces identical JSON byte-for-byte (modulo `captured_at` timestamp).

## Phase 4 — km_refresh_action_templates MCP tool

**Acceptance:**
- New MCP tool registered; requires `confirm: true`.
- Re-runs Phase 3 logic in-process; returns `{added: [...], removed: [...], changed: [...]}`.
- Safe-aborts if KM editor has unsaved scratch state.

## Phase 5 — km_search_actions

**Acceptance:**
- New MCP tool `km_search_actions(query, category?, builder_supported?, result_target?, min_score?, limit?)`.
- Empty query returns full catalog sorted by category then identifier.
- Non-empty query returns ranked results with `score`, `matched_fields`, full entry.
- Top-3 quality bar on 7 acceptance queries from plan (`wait for image to appear`, `open url in browser`, `set clipboard`, `move file`, `if window contains text`, `speak text`, `play sound`).
- `fastembed`, `rapidfuzz` deps in `pyproject.toml`. Embeddings cached under `src/server/data/.cache/`; cache dir gitignored.
- Fallback path: if `fastembed` model unavailable, search still works via rapidfuzz-only with `degraded: true` in response metadata.
- `km_list_action_types` retained as alias forwarding to `km_search_actions("")` with deprecation warning.

## Phase 6 — Plug-in metadata parser fix

**Acceptance:**
- `_scan_installed_plugins` (`plugin_action_tools.py:273`) extracts `KeyWords`/`Keywords` (both casings), `HelpURL`, `Author`.
- `_plugin_entry` (`action_tools.py:90`) surfaces `keywords` in response.
- Test: synthetic plist with KeyWords array → keywords present in entry.

## Phase 7 — Six tool merges (one commit per merge)

**Acceptance (each sub-phase):**
- Old tool name registered as deprecation alias that forwards and logs `logger.warning`.
- Behavior parity verified by a parameterized test: same arguments → same result envelope.
- README + audit doc updated.

7a. notifications → `km_notifications(operation)` 
7b. tokens → `km_token_manager(operation)` 
7c. plug-in builder rename → `km_create_plugin_action` 
7d. macro CRUD → `km_macro_editor` absorbs create/move 
7e. window split → `km_window_control` + `km_window_info` 
7f. trigger lifecycle → `km_trigger_lifecycle(kind, operation)`

## Phase 8 — Docs + final cleanup

**Acceptance:**
- README reflects 20-tool target surface.
- `docs/km_mcp_audit_report.md` updated.
- Tool-count assertion test: `len(set(target_names)) == 20`.
- All deprecation aliases test-covered.
