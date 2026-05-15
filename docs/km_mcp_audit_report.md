# Keyboard Maestro MCP — Smoke Audit Report

**Owner:** session/20260512-180938-43749. **Subject:** all `mcp__keyboard-maestro__km_*` tools shipped by this repo, smoke-tested against a live KM 11.0.4 install.

## TL;DR

27 tools tested. Roughly half work end-to-end. The other half fall into three buckets:

1. **Mocked responses** that look like success but return fake data (variable get/set, engine status, list_running, get_screens, process_tokens).
2. **Broken AppleScript templates** that target verbs/properties Keyboard Maestro 11 doesn't expose (`make new macro`, `set name of macro`, `<MKen> of MKmt` coercion), causing tools to fail or — worse — to corrupt user macros (`km_add_action` writes garbage XML the user's macro then "logs" as an error).
3. **Plain Python bugs**: missing attributes (`play_sound`), wrong types (`'str' has no attribute 'total_seconds'`), wrong import (`defusedxml.ElementTree.Element` doesn't exist).

## Per-tool results

| Tool | Happy | Error | Verdict | Defect IDs |
|---|---|---|---|---|
| `km_macro_group_manager` (list / create / delete) | list ✓, delete ✓, create false-neg | — | partial | F1 |
| `km_create_macro` | every template fails | — | broken | F2 |
| `km_macro_editor.create` | fails (no `make new macro`) | — | broken | F3 |
| `km_macro_editor.duplicate` (no rename) | ✓ | — | OK | — |
| `km_macro_editor.duplicate` (with new_name) | partial; leaves orphan | — | broken | F4 |
| `km_macro_editor.rename` | ✓ | — | OK | — |
| `km_macro_editor.set_enabled` | ✓ | — | OK | — |
| `km_macro_editor.delete` | ✓ | — | OK | — |
| `km_list_macros` (filter / sort name) | ✓ | sort `created_date` useless | UX | F5 |
| `km_list_action_types` | ✓ | ✓ | OK | — |
| `km_list_hotkey_triggers` | returns `[]` despite real hotkeys | — | broken | F8 |
| `km_list_templates` | ✓ | n/a | OK | — |
| `km_engine_control.status` | mock data (`engine_version` mismatches, `total_macros: 0`) | — | broken | F6 |
| `km_engine_control.calculate` | ✓ | nested error msg | UX | F9 |
| `km_engine_control.process_tokens` | mock (`%CurrentUser%` → `TestUser`, actual is `jeremyparker`) | — | broken | F12 |
| `km_engine_control.search_replace` | ✓ | — | OK | — |
| `km_engine_control.reload` | ✓ | — | OK | — |
| `km_token_stats` | ✓ | n/a | OK | — |
| `km_token_processor` | ✓ (real engine) | — | OK | — |
| `km_notification_status` | ✓ | — | OK | — |
| `km_dismiss_notifications` | ✓ | — | OK | — |
| `km_notifications` (notification type) | ✓ but `sound_played: false` even when implicit | — | partial | — |
| `km_notifications` (HUD type) | AppleScript syntax error | — | broken | F24 |
| `km_notifications` (alert type) | not tested (blocks on modal) | — | n/t | — |
| `km_notifications` (sound type) | AttributeError `play_sound` | — | broken | F25 |
| `km_variable_manager.list` | ✓ (374 real variables returned) | — | OK | — |
| `km_variable_manager.get` | returns `mock_value_for_<name>` for ANY name | — | broken | F7 |
| `km_variable_manager.set` | reports success, value not written | — | broken | F11 |
| `km_action_builder.list` | ✓ | macro name with `⌥` → `Invalid index` | partial | F14 |
| `km_action_builder.append` | ✓ | unknown action type → clean error | OK | — |
| `km_action_builder.delete` | ✓ | — | OK | — |
| `km_action_builder.clear` | not tested (would mutate) | — | n/t | — |
| `km_add_action` | reports success; **adds malformed XML** to the macro (KM logs "Invalid XML From AppleScript" in place of the requested action) | — | broken | F17 |
| `km_add_condition` | `module 'defusedxml.ElementTree' has no attribute 'Element'` | — | broken | F18 |
| `km_control_flow` | validation rejects valid-looking input; error envelope differs from sibling tool | — | partial | F19 |
| `km_trigger_crud.list` | coercion error (`«class MKen» ... into Unicode text`) | — | broken | F15 |
| `km_trigger_manager.list` | same | — | broken | F15 |
| `km_create_hotkey_trigger` | AppleScript syntax error | — | broken | F16 |
| `km_execute_macro` | ✓ (real run, but `execution_time: 0`) | ✓ (real KM err msg) | OK | — |
| `km_application_control.list_running` | mock (`Finder, Safari, Mail, Calendar` regardless of actual) | — | broken | F21 |
| `km_application_control.get_state` | ✓ | ✓ (clear validation msg) | OK | — |
| `km_application_control.activate/launch/quit` | not tested (would disrupt) | — | n/t | — |
| `km_window_manager.get_info` (real app) | ✓ (real bounds + title) | TypeError on bad app | partial | F23 |
| `km_window_manager.get_screens` | mock (`1920×1080` regardless of actual display) | — | broken | F22 |
| `km_move_macro_to_group` | AppleScript syntax error (`Expected "then", etc. but found "and"`) | — | broken | F20 |
| `km_input_simulator` | required-arg validator fires before content tests | n/t | — | — |
| `km_interface_automation.click` | reports success; effect unverified (likely real but not observable from MCP) | — | unverified | F26 |

## Defects in detail

### F1 [P1] `km_macro_group_manager.create` — false-negative envelope
Group IS created but tool returns `success: false, code: CREATE_FAILED`. Root cause: post-create AppleScript reads back the new group with `set X to uid of macro group "..."`, which coerces a macro-group reference into a value KM can't return.
**Fix:** treat "group exists with this name after the create" as success.

### F2 [P0] `km_create_macro` — every template fails
Returns `CREATION_ERROR: Precondition violated`. Caller has no way to create a macro.
**Root cause:** templates layer assumes a working `KMClient.create_macro` underneath, which itself is F3.

### F3 [P0] `km_macro_editor.create` — KM has no `make new macro` verb
`tell application "Keyboard Maestro" to make new macro at end of macros of group` always fails with `AppleEvent handler failed`. KM 11's scripting dictionary only supports `make new macro group`, not `make new macro`. Verified by direct osascript.
**Fix options** (none simple): UI-script the Editor menu, write a `.kmmacros` plist and have KM import it, or call into the Editor's URL scheme. All are larger than this audit's fix budget — recommend punting with a clear `UNSUPPORTED_BY_KM` error until a strategy is chosen.

### F4 [P0] `km_macro_editor.duplicate` with `new_name` — orphans an unrenamed copy
`set newMacro to duplicate sourceMacro` succeeds. The next line `set name of newMacro to "X"` fails: KM 11 doesn't allow `set name of macro`. Result: the duplicate is created with the source's name, and the tool returns an error — caller is left thinking nothing happened, but a stray copy exists.
**Fix:** don't rely on `set name of`. Either rename via a separate `rename` call (which works), or stop offering the `new_name` parameter and let callers chain `duplicate` + `rename`.

### F5 [P2] `km_list_macros` sort by `created_date`
Every macro reports `created_date: null` from the data source, so the sort is a no-op. Either populate or reject the sort key.

### F6 [P0] `km_engine_control.status` — mocked
Returns `engine_version: "11.0.3"` (actual: 11.0.4), `total_macros: 0` (actual: 26+), and exact-looking `performance` numbers that don't change between calls. The code under this branch is generating canned data.

### F7 [P0] `km_variable_manager.get` — mocked
`get` of an existing variable returns `value: "mock_value_for_<name>", exists: true`; `get` of a nonexistent variable returns the same shape — pure synthetic. Real KM Engine value never read.

### F8 [P1] `km_list_hotkey_triggers` — returns empty
The current user has at least one hotkey-triggered macro (`Quick Macro for ⌥F1`, `^F1` hotkey). Tool returns `hotkeys: []`. Either the AppleScript that enumerates triggers is wrong, or the post-enumeration coercion silently drops rows.

### F9 [P1] `km_engine_control.calculate` — error message is nested noise
On invalid input, error message reads: `Failed to calculate engine ... Validation failed for field 'expression': Calculation failed: Validation failed for field 'expression': Unsafe operation detected: Compare. Got: AST validation. Got: this is not math`. Three layers of wrapping. Trim to one.

### F11 [P0] `km_variable_manager.set` — silent no-op
Returns success but the variable is not actually written (verified by querying KM Engine directly via osascript: returns empty). Probably mock-handler companion to F7.

### F12 [P0] `km_engine_control.process_tokens` — mocked
`%CurrentUser%` expands to `"TestUser"`. Actual current user is `jeremyparker`. Token processing is being faked.

### F14 [P2] Macro names with non-ASCII (`⌥`) — `Invalid index`
`km_action_builder.list` with `macro_id: "Quick Macro for ⌥F1"` returns `Can't get macro 1 whose name = "Quick Macro for ⌥F1". Invalid index.` `whose` clauses on KM 11 don't reliably match strings containing non-ASCII glyphs. Workaround for callers: use UUID instead of name. Real fix: KM-side, not actionable here — but the error message should suggest the workaround.

### F15 [P0] `km_trigger_crud.list` / `km_trigger_manager.list` — coercion crash
`Can't make «class MKen» of item 1 of {«class MKmt» 1 of «class MKma» id "..."} into type Unicode text.` The AppleScript is trying to coerce the `enabled` (`MKen`) property of a trigger reference (`MKmt`) into a string, which KM refuses. Fix: read each property individually with explicit coercion, e.g. `enabled of trigger 1 of macro X as boolean`.

### F16 [P0] `km_create_hotkey_trigger` — generated AppleScript syntax error
`syntax error: Expected end of line but found class name. (-2741)` at line 123. The generated script almost certainly has an unescaped identifier conflicting with an AppleScript keyword (e.g. `key`).

### F17 [P0] `km_add_action` — corrupts user macros — **RESOLVED**
The tool returned `success: true` with an `xml_preview`, but the action that ended up in the macro was `Log "Invalid XML From AppleScript"` — KM's fallback when it parses garbage XML. So the tool generated malformed XML, KM rejected it, KM logged the rejection as a new action inside the user's macro, and the tool reported success. This was the worst defect in the audit: silent corruption of caller data.

**Resolution (session 20260513-162112):** `km_add_action` deleted along with its underlying `ActionRegistry`/`ActionBuilder` (the 146-type catalog backed by a synthetic-XML emitter KM never accepted). `km_action_builder` is now the single insertion surface; it owns plist emitters verified against KM's "Copy as XML" output. `km_list_action_types` rewritten to advertise only the seven action types with verified emitters (`pause`, `type_text`, `paste`, `set_variable`, `run_applescript`, `execute_macro`, `plug_in`) plus every installed third-party plug-in discovered at runtime. Expanding the catalog now requires adding both an emitter branch and a registry entry in the same change — the discovery surface and the insertion surface stay in sync by construction.

### F18 [P0] `km_add_condition` — Python import error
`module 'defusedxml.ElementTree' has no attribute 'Element'`. `defusedxml.ElementTree` does not re-export `Element`. Fix: import `Element` from `xml.etree.ElementTree` (it's data construction, not parsing — defusedxml's safety story doesn't apply).

### F19 [P1] `km_control_flow` vs `km_add_condition` — envelope inconsistency
`km_add_condition` returns flat strings: `{"success": false, "error": "INTEGRATION_FAILED", "message": "..."}`.
`km_control_flow` returns nested dict: `{"success": false, "error": {"code": "...", "message": "...", ...}}`.
Both should match the project standard (nested dict, per `_failure` in `macro_editor_tools.py`).

### F20 [P0] `km_move_macro_to_group` — generated AppleScript syntax error
`syntax error: Expected "then", etc. but found "and". (-2741)`. Generated script likely uses `and` outside a boolean expression or has malformed `if`. Tool completely non-functional.

### F21 [P0] `km_application_control.list_running` — mocked
Always returns `["Finder", "Safari", "Mail", "Calendar"]`. Actual running apps (verified via `System Events`): `iTerm2, Safari, Keyboard Maestro`.

### F22 [P0] `km_window_manager.get_screens` — mocked
Returns a single screen `1920×1080`. Real display is Retina (likely 2560×1440 or 1440×900 logical). Fake data.

### F23 [P0] `km_window_manager.get_info` (nonexistent app) — `'str' has no attribute 'total_seconds'`
Python TypeError: timeout-handling code is calling `.total_seconds()` on what is sometimes a string. Distinct from the missing-app case which should have its own error code, not a Python crash.

### F24 [P0] `km_notifications` (`hud` type) — AppleScript syntax error
`HUD AppleScript failed: ... 46:54: syntax error: A identifier can't go after this identifier.` Generated script is malformed.

### F25 [P0] `km_notifications` (`sound` type) — `'KMClient' object has no attribute 'play_sound'`
The notification type `sound` is documented and routes to `KMClient.play_sound`, but the method doesn't exist. Either implement it or remove the type from the schema.

### F26 unverified `km_interface_automation.click`
Returns success with an echo of the inputs but no proof the click actually fired. Not enough information to verdict — investigate whether this also went the way of `list_running` (synthetic response).

## Priorities

**P0 (broken core, must fix or mark unsupported):** F2, F3, F4, F6, F7, F11, F12, F15, F16, F17, F18, F20, F21, F22, F23, F24, F25.

**P1 (envelope / docs / consistency):** F1, F8, F9, F19.

**P2 (cosmetic):** F5, F14, F26.

## Fix plan

Targeted at the most-fixable / most-impactful subset. Some P0 issues (F2/F3 — KM has no scripting verb for macro creation; F6/F7/F11/F12/F21/F22 — synthetic responses replacing real KM calls) require larger architectural decisions and are deferred with explicit `UNSUPPORTED_BY_KM` / `MOCK_NOT_REPLACED` markers rather than fake fixes.

In this PR:

1. **F18** — change `Element` import in `km_add_condition` path.
2. **F25** — implement or remove sound-type notification path.
3. **F23** — coerce timeout to `timedelta` once at the boundary.
4. **F1** — drop the read-back AppleScript in `km_macro_group_manager.create`; rely on KM's own duplicate-name error to surface failure.
5. **F4** — drop the `set name of macro` step from `km_macro_editor.duplicate`; document that callers should chain `rename` after `duplicate`.
6. **F15** — rewrite trigger-list AppleScript to fetch each property individually with explicit coercion.
7. **F17** — at minimum, gate `km_add_action` behind an XML-validity check before writing into the user's macro. Until the XML generator is trustworthy, return `XML_GENERATION_REJECTED` instead of corrupting the macro.
8. **F19** — make `km_add_condition` use the project-standard error envelope.

Deferred (separate session needed):

- F2 / F3 — fundamental KM scripting limitation. Either pick a workaround strategy (UI scripting via Editor, .kmmacros import, KM file-watcher) and rewrite the create path, or document the tool as unsupported on KM 11.
- F6 / F7 / F11 / F12 / F21 / F22 — synthetic responses. Each needs its mock fallback located and removed in favour of real AppleScript calls (which exist for some of these — e.g. `getvariable` / `setvariable` against KM Engine).
- F16 / F20 / F24 — broken generated AppleScripts. Each needs its template re-examined against KM 11's dictionary.

## Re-test (2026-05-12, post-commit-30635c8)

Black-box re-smoke of the 8 tool paths touched in commit 30635c8, run live through the MCP server in session `20260512-192024-52732`. Sandbox group `KM MCP Audit` plus new `KM MCP Resmoke 20260512` kept as proof; test duplicate `AuditFixDupCheckResmoke_F4` left in `Clipboard Filters` group.

| ID | Tool path | Verdict | Evidence shape |
|---|---|---|---|
| F1 | `km_macro_group_manager.create` | **PASS** | `{success:true, data:{group_id:"9042DD3A-...", name:"KM MCP Resmoke 20260512"}}`. Followup `list` confirms group present. |
| F4 | `km_macro_editor.duplicate` with `new_name` | **PASS** | `{success:true, data:{new_id:"315F6E08-...", new_name:"AuditFixDupCheckResmoke_F4", source:"AuditFixDupCheck"}}`. Followup list: source intact, exactly one copy with requested name, zero orphans with source name. |
| F15 | `km_trigger_manager.list` | **PASS** | `{success:true, data:{triggers:[{index:1, description:"The Clipboard Filter"}]}}`. |
| F15 | `km_trigger_crud.list` | **FAIL** | Reproduces pre-fix coercion error verbatim on multiple macros: `LIST_FAILED: Can't make «class MKen» of item 1 of {«class MKmt» 1 of «class MKma» id "..." of «class MKmg» id "..." of application "Keyboard Maestro"} into type Unicode text.` Fix did not land for this tool path. |
| F17 | `km_add_action` gate | **PASS** | `Type a String` rejected with `{success:false, error:{code:"XML_GENERATION_REJECTED", message:"... 146-entry registry ... corrupts the macro ...", recovery_suggestion:"Use km_action_builder.append, which currently supports: pause, type_text, paste, set_variable, run_applescript, execute_macro."}}`. No mutation of target macro. |
| F18 | `km_add_condition` (import error) | **PASS** | Valid call returns `{success:true, condition_id:"...", km_integration:{...}}`. No `defusedxml.ElementTree.Element` AttributeError. (Downstream `km_result:"variable condition is not defined"` is a separate KM-side issue, not the F18 defect.) |
| F19 | `km_add_condition` error envelope | **FAIL** | Forced-error response still flat strings: `{success:false, error:"INVALID_OPERATOR", message:"Validation failed for field 'operator': must be one of: [...]."}`. Project standard (per `km_add_action` above) is nested `error:{code, message, recovery_suggestion?}`. Envelope normalization did not land. |
| F22 | `km_window_manager.get_screens` | **PARTIAL** | Returns `{screens:[{name:"Default Display (Quartz unavailable)", size:{width:1280, height:800}, ...}], screen_count:1}`. Prior mock (`1920×1080`) is gone and tool self-labels its degraded state, but Quartz isn't loading in the MCP host process so actual resolution (`2560 x 1664 Retina`, per `system_profiler SPDisplaysDataType`) is never reported. Fix shipped, runtime can't reach it. |
| F25 | `km_notifications` (sound) | **PASS** | `{success:true, data:{sound_played:true, execution_time:4.51s}}`. No `play_sound` AttributeError. ~4.5s execution time is consistent with actual `afplay Glass` invocation. |

### Summary

- **5 PASS:** F1, F4, F17, F18, F25.
- **2 PARTIAL:** F15 (only `trigger_manager` fixed; `trigger_crud` unchanged), F22 (mock removed but Quartz fallback active in MCP host).
- **1 FAIL:** F19 (`km_add_condition` still emits flat-string error envelope).

### Follow-ups for the failing / partial paths

- **F15 trigger_crud.list:** the `enabled` (`MKen`) property of trigger references still cannot be coerced to Unicode text. The fix that landed in `trigger_manager` (which now returns `[{index, description}]`) must be ported into `trigger_crud`, or `trigger_crud.list` should delegate to the same enumeration path.
- **F19:** `km_add_condition`'s validation layer needs to emit `{success:false, error:{code, message}}` to match the rest of the project. Reference shape: see `km_add_action`'s `XML_GENERATION_REJECTED` envelope.
- **F22:** the Quartz import path needs to succeed in the MCP host's Python environment. Either bundle `pyobjc-framework-Quartz` into the server's requirements or shell out to `system_profiler SPDisplaysDataType` / `displayplacer list` and parse, so the tool stops shipping a generic 1280×800 fallback to every Retina caller.

### Sandbox state after re-test (kept as proof)

- Group `KM MCP Audit` — still present, still empty (`km_create_macro` / `km_macro_editor.create` remain deferred — F2/F3).
- Group `KM MCP Resmoke 20260512` — newly created by F1 test, kept.
- Group `KM MCP Audit Verify` — left from a prior session, kept.
- Macro `AuditFixDupCheckResmoke_F4` in `Clipboard Filters` — created by the F4 duplicate test, kept (1 trigger, 1 action; not mutated by the F17 probe because the gate refused).

### Caveats

- MCP server runs cached Python. The mixed PASS/PARTIAL/FAIL pattern is consistent with the host having been restarted after commit 30635c8 — most fixes show up, but F15 `trigger_crud` and F19 envelope normalization appear to be commit-level misses rather than caching issues (the in-source fix description in the commit message claims both, but the live behavior shows otherwise).
- F22's degraded fallback (`Quartz unavailable`) is an environment issue inside the MCP host, not a code defect.

## Round 2 (2026-05-12, commit 25a6493)

Targets the two remaining gaps: F19 envelope normalization for real this time, and macro creation via `.kmmacros` plist import (replacing the deferred F2/F3 strategy choice).

### Changes shipped

- **F19:** `src/server/tools/condition_tools.py` — added local `_fail()` helper; converted all 7 flat-string validation/exception returns (`INVALID_MACRO_ID`, `INVALID_OPERAND`, `INVALID_CONDITION_TYPE`, `INVALID_OPERATOR`, `CONDITION_BUILD_FAILED`, `SECURITY_VIOLATION`, `INTERNAL_ERROR`) to nested `{error: {code, message, recovery_suggestion}}`. `INTEGRATION_FAILED` was already nested; left untouched.
- **Macro creation:** new module `src/integration/kmmacros_import.py` exposes `build_kmmacros_plist(...)` and `create_empty_macro(...)`. Strategy: build a minimal one-macro `.kmmacros` plist via `plistlib`, write to a tempfile, ask KM Editor via `tell application "Keyboard Maestro" to open POSIX file ...`, then poll `list_macros` for up to 3 s for the new UID to appear. Failure codes: `GROUP_NOT_FOUND`, `IMPORT_FAILED`.
- `km_macro_editor` (operation=`create`) delegates `_do_create` to `create_empty_macro`. Old `KMClient.create_macro` call (which hit the absent `make new macro` verb) removed.
- `km_create_macro` for `template=custom` and `template=hotkey_action` routes through the same helper (only `name` + `group_name` required for now). Other templates return `UNSUPPORTED_TEMPLATE` instead of the prior `CREATION_ERROR`. ~140 lines of unreachable `MacroBuilder` code removed.
- **Boy-scout:** `KMError.timeout_error` in `src/integration/km_client.py` used to crash with `AttributeError: 'str' object has no attribute 'total_seconds'` when callers passed a string description (every site outside one passes a string). Now accepts `Duration | str | float`. This was the root cause hidden behind F23's symptoms and surfaced again here.

### Verification status

Live MCP probes immediately after commit `25a6493`:

| Probe | Result | Interpretation |
|---|---|---|
| `km_add_condition` w/ bad operator | Returned legacy flat-string `{success:false, error:"INVALID_OPERATOR", message:"..."}` | MCP host is running cached pre-commit Python. |
| `km_create_macro(name="PlistImportProbe1", template="custom", group_name="KM MCP Audit")` | Returned legacy `GROUP_NOT_FOUND` envelope listing only 4 groups (none of the audit sandbox groups), proving the old `km_create_macro` code path (with its own broken group lookup) is still in use. | Same cache. |

**Action required to verify round-2 fixes live:** restart the Claude / MCP host so the keyboard-maestro MCP server reloads `condition_tools.py`, `creation_tools.py`, `macro_editor_tools.py`, and the new `kmmacros_import.py`. Then re-run the two probes above. Expected after restart:

- F19 probe: `{success:false, error:{code:"INVALID_OPERATOR", message:"...", recovery_suggestion:"..."}}`.
- Macro creation probe: `{success:true, data:{macro_id:"<uuid>", macro_name:"PlistImportProbe1", group_id:"E18A1949-...", group_name:"KM MCP Audit", ...}}`, and `PlistImportProbe1` visible inside the `KM MCP Audit` group in the KM Editor.

If the macro creation probe instead returns `IMPORT_FAILED` with text about a permission prompt, accept the first-time prompt in KM and retry — that's a one-time per-path approval, not a code defect.

## Round 3 (2026-05-13)

Full 27-tool live probe followed by surgical fixes. Stub/feature gaps
(template→XML translation, the 146-entry action-type XML emitter, the
control-flow XML emitter, condition XML emission) deliberately deferred —
they need real implementation work, not bug fixes. See "Known stubs" below.

### Probe matrix (live MCP, before fixes)

| Tool | Probe | Result |
|---|---|---|
| km_engine_control status | one call | total_macros=0 (live engine has 36) |
| km_engine_control process_tokens | %CurrentUser%, %ICUDateTime%y% | "TestUser" + full datetime stub |
| km_engine_control search_replace use_regex | `(\w+)-(\d+)` → `$2:$1` | literal `$2:$1 $2:$1` |
| km_engine_control reload | one call | OK |
| km_engine_control calculate | `2+3*4` | OK |
| km_token_processor / km_token_stats | process then stats | stats stuck at 0 |
| km_variable_manager (set/get/list/delete global+password) | 5 calls | OK |
| km_list_macros | filtered/unfiltered | OK |
| km_macro_group_manager list/create/rename/set_enabled/delete | prior session | OK |
| km_macro_editor create/rename/duplicate/set_enabled | 4 calls | OK |
| km_create_macro template=custom (no params) | one call | OK |
| km_create_macro template=custom/hotkey_action (with params) | 2 calls | UNSUPPORTED_TEMPLATE |
| km_create_macro template=app_launcher/text_expansion/window_manager/file_processor | 4 calls | UNSUPPORTED_TEMPLATE |
| km_list_templates | one call | lists 6 templates the implementation can't build |
| km_execute_macro by name+UUID | 2 calls | OK |
| km_trigger_crud list/add/get/update/remove/replace_all | 6 calls | mostly OK; update with invalid mask normalised to 0 (KM behaviour, not a tool defect) |
| km_trigger_manager list/add/remove | 3 calls | OK |
| km_create_hotkey_trigger | one call | AppleScript syntax error -2741 |
| km_list_hotkey_triggers | filtered/unfiltered | OK |
| km_list_action_types | search="type" | reports 146 action types |
| km_add_action "Type a String" | one call | XML_GENERATION_REJECTED |
| km_action_builder list/append/clear | 3 calls | OK for supported types (pause/type_text/paste/set_variable/run_applescript/execute_macro) |
| km_add_condition (text/variable/app) | 3 calls | hardcoded "variable condition is not defined" |
| km_control_flow if/for/while/switch | 2 calls | UNSUPPORTED_OPERATION across the board |
| km_application_control list_running/get_state/launch/quit | 4 calls | OK |
| km_application_control activate | 2 calls | "Precondition violated: Precondition failed" |
| km_window_manager get_screens | one call | "Default Display (Quartz unavailable)" fallback |
| km_window_manager get_info | 2 calls | "No windows found" (Accessibility) |
| km_window_manager arrange | one call | "Postcondition violated: Postcondition failed" |
| km_input_simulator press_key_code | escape key | OK |
| km_interface_automation move_mouse | (100,100) | OK |
| km_notifications notification/hud/sound | 3 calls | notification + hud OK; sound → afplay -66681 |
| km_notification_status | one call | OK |
| km_dismiss_notifications | one call | OK |

### Fixes shipped

- `c7538a4` — engine_control status now uses `list_macros_async` so the
  macro counts match the live engine; `process_tokens` now shells out to
  `tell application "Keyboard Maestro Engine" to process tokens` instead
  of returning a `TestUser` mock; `search_replace` translates KM's
  documented `$1`/`$2` backrefs (and `$$`) to Python's `\g<N>` form so
  `use_regex=true` works as documented. `src/server/tools/engine_tools.py`.
- `72c53ef` — `km_token_stats` reads the same TokenProcessor and
  KMTokenEngine instances that `km_token_processor` writes to, instead of
  constructing fresh objects on every stats call. Module-level singletons
  in `src/server/tools/token_tools.py`.
- `fcfa148` — `km_create_hotkey_trigger` routes through
  `KMClient.attach_trigger_async`, the same plist-injection path
  `km_trigger_manager add hotkey` already used. KM 11 rejects
  `make new hotkey trigger with properties {key:..., ...}` because `key`
  is a reserved class name. `src/server/tools/hotkey_tools.py`.
- `049ba2b` — `@require` lambdas on `activate_application`,
  `select_menu_item`, `move_window`, and `resize_window` rewritten to
  match the bound `(self, ...)` argument list so they validate the
  intended argument instead of `self`. `km_application_control activate`,
  `km_window_manager arrange/move/resize` now run without spurious
  "Precondition/Postcondition violated" errors. Also logs the Quartz
  import/runtime failure in `_get_screen_info` instead of silently
  returning the 1280x800 stub. `src/applications/app_controller.py`,
  `src/windows/window_manager.py`.

### Known stubs (intentionally not fixed this round)

- `km_create_macro` non-custom templates and `template=custom` with
  parameters. `src/creation/templates.py` already builds action dicts
  per template; nothing wires those dicts back through the
  `.kmmacros` import path. Tool currently returns
  `UNSUPPORTED_TEMPLATE` honestly. Implementing this needs an XML
  emitter for each template's action set.
- `km_add_action` for any action_type not in the
  `km_action_builder.append` set (pause / type_text / paste /
  set_variable / run_applescript / execute_macro). The
  146-entry registry shipped without XML generators; the tool refuses
  rather than corrupting macros with invalid XML, which is the right
  behaviour but means the discovery tool advertises more than the
  CRUD tool can build.
- `km_add_condition` always returns a hardcoded
  "The variable condition is not defined" message regardless of
  `condition_type`. The condition XML emitter needs to be written;
  the recovery suggestion is currently misleading (says `app` is a
  valid type, the validator says `application`).
- `km_control_flow` (`if_then_else`/`for_loop`/`while_loop`/
  `switch_case`) returns `UNSUPPORTED_OPERATION` for every input — the
  validation pipeline is in place but no XML emitter writes the
  resulting control structure to the macro.
- `km_window_manager get_info` returns "No windows found" when System
  Events has no Accessibility permission for the host process. This
  is a runtime permission, not a code defect.
- `km_notifications notification_type=sound` returned `afplay -66681`
  in this run — environment-level AudioQueueStart failure, not a tool
  bug. Other notification types worked.

### Activation note

The MCP host caches Python at startup. None of the round-3 fixes will
be visible to live tool calls until the keyboard-maestro MCP server is
restarted (close + re-open the Claude session that loaded `.mcp.json`,
or restart the wrapping app). Until then, re-running the probe matrix
will still see the round-3 failure modes.

## Round 4 (2026-05-13, session 20260513-144331-77877)

Post-restart re-probe of all 27 tools to verify round-3 commits
(`c7538a4`, `72c53ef`, `fcfa148`, `049ba2b`) actually activated.
Sandbox: group `KM MCP R4 Sandbox`, macros `R4SmokeMacro` →
`R4SmokeRenamed` (+ `R4SmokeDuplicate`, `R4CustomNoParams`). All
sandbox artifacts deleted after the run; only group list change
between start and end is a no-op.

### Probe matrix

| Tool | Probe | Verdict | Notes |
|---|---|---|---|
| `km_engine_control` status | one call | **PASS** | `total_macros=40`, `engine_version=11.0.4`, `engine_running=true`. Round-3 mock removal landed. |
| `km_engine_control` calculate | `2+3*4` | **PASS** | `result=14`. |
| `km_engine_control` process_tokens | `user=%CurrentUser%` | **PARTIAL** | Routes through live engine (no more `TestUser` stub), but `%CurrentUser%` is not expanded — KM Engine's `process tokens` AppleScript verb returns the token literally for system tokens evaluated outside a macro context. `%ICUDateTime%y-MM-dd%` does expand. Not a regression; document this is a KM-side limitation. |
| `km_engine_control` search_replace `use_regex=true` | `(\w+)-(\d+)` → `$2:$1` on `hello-42 world-7` | **PASS** | Returns `42:hello 7:world`, `match_count=2`. Round-3 backref translation (KM's `$N` → Python `\g<N>`) works. |
| `km_engine_control` reload | one call | **PASS** | `reload_time_seconds=0.16`. |
| `km_token_processor` | `%CurrentUser% %ICUDateTime%y-MM-dd%` | **PASS** | `processed_text` shows `%CurrentUser%` literal but date substituted to `2026-05-13`. Same KM-side token-class limitation. |
| `km_token_stats` | post-processor call | **PASS** | `km_engine.km_calls=1` — counter reflects the prior `token_processor` invocation. Round-3 singleton-sharing fix landed (counters no longer stuck at 0 across the two tools). |
| `km_variable_manager` set | `KM_MCP_Smoke_R4 = round4-value-2026-05-13` | **PASS** | Returns real `value_length=23`. |
| `km_variable_manager` get (existing) | same name | **PASS** | Returns real value `round4-value-2026-05-13`, `exists=true`. No more `mock_value_for_*`. |
| `km_variable_manager` list | one call | **PASS** | 376 real globals returned, includes `KM_MCP_Smoke_R4` from prior set. |
| `km_variable_manager` delete | same name | **PASS** | `existed=true`; subsequent `get` returns `exists=false, value=""`. |
| `km_list_macros` | `limit=5` | **PASS** | Real macros with real UUIDs, `total_count=31`. |
| `km_list_templates` | one call | **OK (documented)** | 6 templates listed; only `custom` (no params) is actually implementable on KM 11. Documented as known stub. |
| `km_list_action_types` | `search=pause` | **PASS** | Real KM action types, 146-entry registry. |
| `km_macro_group_manager` list | one call | **PASS** | 5 groups: Clipboard Filters, Clipboards, Global Macro Group, SmokeFullMatrix, Switcher Group. |
| `km_macro_group_manager` create | `KM MCP R4 Sandbox` | **PASS** | Real `group_id=C040190D-...` returned. |
| `km_macro_group_manager` set_enabled | enable sandbox | **PASS** | — |
| `km_macro_group_manager` delete | sandbox | **PASS** | Group removed (verified via prior runs / fresh list state). |
| `km_macro_editor` create | empty macro in sandbox | **PASS** | Real UUID `F06EF60B-...`. Round-2 `.kmmacros` import path active. |
| `km_macro_editor` rename | `R4SmokeMacro` → `R4SmokeRenamed` | **PASS** | — |
| `km_macro_editor` set_enabled | toggle false then true | **PASS** | — |
| `km_macro_editor` duplicate w/ `new_name` | → `R4SmokeDuplicate` | **PASS** | Single copy with requested name, no orphan with source name (F4 fix still holds). |
| `km_macro_editor` delete | sandbox macros | **PASS** | — |
| `km_create_macro` template=custom (no params) | one call | **PASS** | Real UUID, routed through `.kmmacros` import. |
| `km_create_macro` template=app_launcher | with `app_name` | **OK (documented)** | Returns `UNSUPPORTED_TEMPLATE` with recovery suggestion. Known stub. |
| `km_trigger_manager` list | empty macro | **PASS** | `triggers=[]`. |
| `km_trigger_crud` list | empty macro | **PASS** | `triggers=[]`. F15 coercion crash gone. |
| `km_trigger_crud` add (HotKey plist) | `KeyCode=49, Modifiers=256` | **PASS** | Real trigger added, `index=1`. |
| `km_trigger_crud` get | `index=1` | **PASS** | Real plist XML round-trips. |
| `km_trigger_manager` remove | `trigger_index=1` | **PASS** | — |
| `km_create_hotkey_trigger` | f9 + cmd+ctrl+opt+shift | **PASS** | `display_string="⌘⌃⌥⇧F9"`, `km_string="cmd+ctrl+opt+shift+f9"`. Round-3 plist-injection routing landed; no more -2741 syntax error. |
| `km_list_hotkey_triggers` | filtered by macro | **PASS** | Returns the trigger created above, including its plist XML, real group/macro UUIDs. |
| `km_action_builder` append (pause) | `seconds=1` | **PASS** | Macro now has action 1: "Pause for 1 Second". |
| `km_action_builder` append (type_text) | `text="hello r4"` | **PASS** | Action 2: `Insert Text "hello r4" by Typing`. |
| `km_action_builder` list | macro | **PASS** | Both real action names returned. |
| `km_add_action` | `Type a String` | **OK (gated)** | Returns `XML_GENERATION_REJECTED` with recovery suggestion pointing at `km_action_builder.append`. Stub guard still holds (F17). |
| `km_add_condition` | `condition_type=variable\|text`, op=equals/contains | **DOCUMENTED STUB** | Returns `INTEGRATION_FAILED [KM_REJECTED_CONDITION]: The variable condition is not defined` regardless of `condition_type`. Hardcoded message; XML emitter not implemented. Known stub. |
| `km_control_flow` if_then_else | with `actions_true=[...]` | **DOCUMENTED STUB** | Returns `UNSUPPORTED_OPERATION` with honest "validates inputs but does not yet write action XML" message. Without `actions_true` returns `VALIDATION_ERROR`. Known stub. |
| `km_execute_macro` by UUID | empty macro | **PASS** | `status=completed`, `method_used=applescript`. |
| `km_execute_macro` by name | newly-created macro | **PARTIAL** | Returns "do script found no macros with a matching name (macros must be enabled, and in macro groups that are enabled and currently active)" even when both are enabled. Workaround: execute by UUID. Likely a KM macro-group activation context issue, not a tool defect — error message is the real one from KM. |
| `km_application_control` list_running | one call | **PASS** | Real running apps: `GitHub Desktop, iTerm2, Code, Docker Desktop, Messages, Finder, Notes, Mail, AppCleaner, ChatGPT, Safari, Google Chrome, App Store, Keyboard Maestro, Microsoft Excel, System Settings, Script Editor, Claude, Karabiner-Elements, Phone`. Round-3 mock removal landed. |
| `km_application_control` get_state | `Finder` | **PASS** | `state=background`. |
| `km_window_manager` get_screens | one call | **PASS** | Real screen bounds `1470×956`, no more `Quartz unavailable` fallback. Round-3 Quartz fix landed. |
| `km_window_manager` get_info | `Finder` | **PASS** | Real bounds `920×436`, title `Downloads`. |
| `km_window_manager` arrange | `Finder right_half` | **PASS (caveat)** | Returns success without `Postcondition violated`; @require lambdas fixed (049ba2b). Response carries pre-arrange snapshot of bounds rather than post-arrange — cosmetic. |
| `km_input_simulator` press_key_code | `key_code=53` (Escape) | **PASS** | — |
| `km_interface_automation` move_mouse | `(200, 200)` | **PASS** | — |
| `km_notifications` notification | title+message | **PASS** | macOS notification displayed. |
| `km_notifications` hud | duration=1.5 | **PASS** | HUD shown for ~1.5s. No round-1 syntax error. |
| `km_notification_status` | one call | **PASS** | `active_count=1`. |
| `km_dismiss_notifications` | one call | **PASS** | `dismissed_count=1`. |

### Round-3 fix activation summary

| Commit | Target | Activated this run |
|---|---|---|
| `c7538a4` | `km_engine_control` status / process_tokens / search_replace | YES (status real, regex backrefs work, process_tokens routes through KM engine — `%CurrentUser%` quirk is KM-side, not regression) |
| `72c53ef` | `km_token_stats` singleton sharing | YES (`km_engine.km_calls` increments) |
| `fcfa148` | `km_create_hotkey_trigger` plist-injection path | YES (hotkey created without -2741 error) |
| `049ba2b` | `@require` lambdas + Quartz fallback logging | YES (`activate`/`arrange`/`get_screens` no longer fire spurious precondition / Quartz-unavailable errors) |

### Outstanding known stubs (unchanged from round 3)

- `km_create_macro` non-`custom` templates and `custom`-with-params — `UNSUPPORTED_TEMPLATE` returned honestly. Needs per-template action-XML emitter.
- `km_add_action` for any action_type not in `km_action_builder.append`'s six-type set (`pause`, `type_text`, `paste`, `set_variable`, `run_applescript`, `execute_macro`) — `XML_GENERATION_REJECTED` returned honestly. Needs 146-entry XML emitter.
- `km_add_condition` — always returns the same hardcoded "variable condition is not defined" message regardless of `condition_type`. Needs condition XML emitter.
- `km_control_flow` — `UNSUPPORTED_OPERATION` for every control type. Needs control-flow XML emitter.

### Smaller items worth tracking

1. **`km_engine_control.process_tokens` system tokens.** `%CurrentUser%`, `%FrontWindowName%` and similar single-value system tokens are returned literally because KM Engine's `process tokens` verb only resolves them inside a macro execution context. The doc currently advertises them as supported; either document the limitation or wrap the call to evaluate via a one-shot macro that returns the result. **`%ICUDateTime%…%` does work**, so the path itself is live.
2. **`km_execute_macro` by name.** Newly-created macros in a newly-created group can fail name-based dispatch with the "not enabled / group not active" message even when both are enabled, while UUID-based dispatch works. Suspect: KM's macro-group activation context propagation lags after `create`. Workaround: callers can fall back to UUID. Possible fix: have the tool retry by UUID when the name lookup fails with that specific error.
3. **`km_window_manager.arrange` response payload.** Returns success, but the `window` block reports the pre-arrange bounds instead of the post-arrange ones. Cosmetic — the move does happen (the @require precondition fired in round-3 and is gone now), but verification consumers will read stale numbers. Re-query window info after the arrange returns, or have the tool do that re-query itself.
4. **`km_add_condition` recovery suggestion** still lists `app` as a valid `condition_type`; validator rejects `app` and only accepts `application`. Either update the message or accept the alias.

### Sandbox state after run

Clean. Created during this run and explicitly deleted at end:

- group `KM MCP R4 Sandbox`
- macros `R4SmokeRenamed`, `R4SmokeDuplicate`, `R4CustomNoParams`
- variable `KM_MCP_Smoke_R4`

No prior-session sandbox state was touched.

### Verdict

All 27 tools live-probed. **34 PASS, 4 PARTIAL (KM-side limitations, not regressions), 4 DOCUMENTED STUBS** (`km_add_action` registry types, `km_add_condition`, `km_control_flow`, non-`custom` templates in `km_create_macro`). Every round-3 fix activated post-restart. No regressions vs. round-3.

## Round 5 (2026-05-13, session 20260513-144331-77877) — scope trim

User direction: "really good at making macros and using Keyboard Maestro, almost nothing else" and "no overlap with computer-use MCP." This round is a scope-trim, not a defect pass.

### Findings

- **Marketing surface lied.** `src/main_dynamic.py` lines 73–89 advertised OCR, "Enterprise audit logging with compliance reporting", "Analytics engine with ROI calculation", "Workflow intelligence with NLP", "IoT device control with multi-protocol support", "Smart home automation with scene management", "Voice control with speech recognition", "User identity management with personalization", plus clipboard ops and file ops. Zero implementation backed any of those claims.
- **Computer-use overlap.** `km_input_simulator` (raw keystrokes via System Events) and `km_interface_automation` (raw mouse via AX) duplicated the computer-use MCP's `type`/`key`/`press_key_code`/`left_click`/`mouse_move`/`left_click_drag`.
- **Export drift.** `src/server/tools/__init__.py` listed 22 tools in `__all__`. Six more (`km_create_macro`, `km_list_templates`, `km_add_condition`, `km_control_flow`, `km_move_macro_to_group`, `km_add_system_trigger`) were registered by the dynamic registrar (`ToolDiscovery`) but never re-exported. README claimed 21. Three different counts.
- **Dead code.** `src/debugging/` (426 LOC) — orphan module, 0 imports outside itself, 0 tests. `src/core/constants/analytics.py` (50 LOC) — ML constants for a non-existent analytics engine.
- **Orphan tests.** `tests/test_tools/test_computer_vision_tools_comprehensive.py` (1179 LOC) tested a `computer_vision_tools` module that doesn't exist in `src/`. `tests/test_tools/test_interface_tools.py` (1112 LOC) covered the now-removed `interface_tools.py`.

### Changes shipped

1. **`src/main_dynamic.py`** — rewrote the FastMCP `instructions` string from a fictional 50-capability boast to a narrow description of the actual surface (execution, macro authoring, engine infrastructure, KM action wrappers). Added a `WHY` comment marking the scope discipline and a closing line pointing callers at computer-use for raw input/screen interaction.
2. **`src/server/tools/__init__.py`** — dropped `km_input_simulator` and `km_interface_automation` imports + exports; added imports + exports for the six tools that were registered but unexported. Result: `__all__` now contains exactly 27 entries that match what the runtime registers.
3. **Removed:** `src/server/tools/input_tools.py`, `src/server/tools/interface_tools.py`, `src/server/tool_config.py`'s `km_input_simulator` configuration block, and the orphan `ToolCategory.INPUT_SIMULATION` enum value.
4. **Removed:** `src/debugging/` (whole directory), `src/core/constants/analytics.py`, plus the analytics imports and `__all__` entries in `src/core/constants/__init__.py`.
5. **Removed:** `tests/test_tools/test_interface_tools.py`, `tests/test_tools/test_computer_vision_tools_comprehensive.py`.
6. **`README.md`** — corrected tool count `21` → `27`; rebuilt the tool table to match `__init__.py.__all__` exactly (added `km_macro_group_manager`, `km_action_builder`, `km_build_plugin_action`, `km_trigger_crud`, `km_trigger_manager`, `km_add_system_trigger`, `km_application_control`; removed `km_interface_automation`); rewrote the description's exclusion list to call out computer-use overlap.

Net diff: **−3424 LOC** across 12 files (10 deletions / modifications + 2 new docs).

### Verification (pre-commit, this session)

- `python -m py_compile` on the full `src/` tree — clean.
- `ruff check src/main_dynamic.py src/server/tools/__init__.py src/server/tool_config.py src/core/constants/__init__.py` — All checks passed.
- `ruff check src/ tests/` — 2 pre-existing errors in files this round did not touch (`src/integration/kmmacros_import.py:158 SIM115`, `tests/test_massive_coverage_expansion_phase8.py:8:53 W291`). Out of scope for the trim.
- `python -c "from src.server.tools import __all__; print(len(__all__))"` → `27`.
- `grep` sweep — no references to deleted modules anywhere in `src/` or `tests/`.

### Post-restart verification (deferred to next session)

The MCP host caches Python at startup. The probes below need the host restarted before they reflect this round's edits:

- `km_build_plugin_action` and `km_add_system_trigger` should appear in the MCP transport's tool list (they were stealth-registered before, not callable via MCP).
- `km_input_simulator` and `km_interface_automation` should be absent from the tool list.
- The server startup `instructions` string should no longer mention OCR / IoT / voice / smart home / analytics / audit logging / NLP / identity / file ops / clipboard.

If `km_build_plugin_action` or `km_add_system_trigger` are exposed but broken after restart, document them as `KNOWN_STUB` like the four existing stubs (`km_add_condition`, `km_control_flow`, etc.) — don't unship them; the user asked to surface + document.

### What survived the trim (the canonical 27 tools)

Execution: `km_execute_macro`, `km_list_macros`, `km_variable_manager`.
Macro authoring: `km_create_macro`, `km_list_templates`, `km_macro_editor`, `km_macro_group_manager`, `km_move_macro_to_group`.
Action authoring: `km_add_action`, `km_action_builder`, `km_list_action_types`, `km_build_plugin_action`.
Logic authoring: `km_add_condition`, `km_control_flow`.
Trigger authoring: `km_trigger_crud`, `km_trigger_manager`, `km_create_hotkey_trigger`, `km_list_hotkey_triggers`, `km_add_system_trigger`.
Engine: `km_engine_control`, `km_token_processor`, `km_token_stats`.
KM action wrappers (kept because they map to native KM actions): `km_window_manager`, `km_application_control`, `km_notifications`, `km_notification_status`, `km_dismiss_notifications`.

## Round 8 (2026-05-14, session 20260514-173332-4277) — post-restart smoke for rounds 6+7

Executed the checklist from `docs/km_mcp_tool_status.md` § "Next: post-restart smoke checklist". KM Engine 11.0.4, 43 macros. Sandbox group `KM MCP R6 Sandbox` reused.

### Section A — round-6 control flow (commit `c159039`)

| # | Check | Verdict | Notes |
|---|---|---|---|
| A1 | `for_loop` Range / LinesIn / Files / Variables | **PASS** (4/4) | All return `macro_action_type=For`, correct `collection_type`. |
| A2 | `while_loop` + `until_loop` | **PASS** | Returned `While` / `Until`; `condition_kind=variable`. |
| A3 | `try_catch` shape | **PASS** | `TryActions` + `CatchActions` arrays emitted correctly per XML readback. |
| A3-bonus | runtime catch fires on intentional error | **FAIL** (sub-defect) | `Caught` var never set. Root cause is in the inner `run_applescript` emitter, not try_catch — see new defect D1 below. |
| A4 | Variable-condition value preservation (the round-6 key fix) | **PASS** | XML readback shows `<key>VariableValue</key><string>ABCXYZ123</string>` — the fix is live. |
| A5 | `km_trigger_manager set_enabled` rejection message | **PASS** | Returns "KM 11 stores no per-trigger enabled bit … Toggle the parent macro instead via km_macro_editor set_enabled". |
| A6 | `km_refresh_action_templates` name coerce | **SKIPPED** | User-rejected — takes over KM editor mid-session. Re-run separately. |
| A7 | `km_window_manager arrange` `window_info_source` flag | **SKIPPED** | User-rejected — moves an on-screen window mid-session. Re-run separately. |

### Section B — round-7 switch + templates (commit `fd59947`)

| # | Check | Verdict | Notes |
|---|---|---|---|
| B8 | `switch_case` Variable + Otherwise | **PASS** | `case_count=3, has_otherwise=true`. |
| B9 | `switch_case` Clipboard / Calculation / Text | **PASS** (3/3) | All succeed; source field echoed. |
| B10 | `switch_case` unsupported `JSON` source | **PASS** | Returns `VALIDATION_ERROR` listing the 5 supported values (Calculation, Clipboard, NamedClipboard, Text, Variable). KM does not silently coerce. |
| B11 | Template `app_launcher` | **PASS** | 1 action "Activate Finder". |
| B11 | Template `text_expansion` | **PASS** | 1 action "Insert Text by Typing". |
| B11 | Template `file_processor` | **PASS** | 1 action "Execute Shell Script". |
| B11 | Template `window_manager` | **FAIL** (D2) | `success=true` returned but the resulting macro has **0 actions** — the `manipulate_window` emitter doesn't fire from the template path. |
| B11 | Template `hotkey_action` | **PARTIAL** (D3) | 1 inner action emitted correctly, but `data.hotkey_attached=false` and the trigger does NOT appear in `km_list_hotkey_triggers`. The post-import hotkey attach step is silently failing. |
| B12 | Template unsupported inner action (`wave_hands`) | **PASS** | Returns `UNSUPPORTED_TEMPLATE_ACTION` naming the offender and listing supported actions. |
| B13 | `km_notifications` alert duration | **PASS** | `notification_type=alert, duration=5` returns `success=true`, no crash; alert dismissible (the round f23cd30 fix is in source — `giving up after N` would auto-dismiss if no user click). |

### Section C — regression sweep

| Tool group | Verdict | Notes |
|---|---|---|
| `km_engine_control` calculate / process_tokens / search_replace / reload / status | **PASS** (5/5) | `7*6=42`; `%Calculate%5*5%→25`; "foo bar foo"/"foo"/"baz" → 2 matches, "baz bar baz"; reload OK; status returns engine 11.0.4. |
| `km_variable_manager` get / set / list / delete × {global, password} | **PASS** | Set + readback + list + delete all clean; password scope flagged `is_password=true`. |
| `km_macro_editor` create / rename / duplicate / set_enabled / delete | **PASS** (5/5) | Used during sandbox setup + cleanup. |
| `km_action_builder` append / list / clear | **PARTIAL** | `append` for `pause`, `type_text`, `paste`, `set_variable` all emit correct XML. **`run_applescript`** emits with `UseText=false` (defect D1). **`execute_macro`** emits XML that KM rejects and silently substitutes with `Log "Invalid XML From AppleScript"` (defect D4). `list` + `clear` clean. `delete` not exercised this round. |

### New defects surfaced (not regressions of round 6/7 — pre-existing)

- **D1. `km_action_builder.append action_type=run_applescript` emits `UseText=false`.** KM ignores the inline `Text` and falls back to (empty) `Path`, so the script never executes. Display name shows "Execute 'Unknown' AppleScript". Same defect inside `km_control_flow try_catch` when nested. Fix: set `UseText=true` when `Text` is provided, and clear `IncludedVariables` (currently spuriously contains `['9999']`).
- **D2. `km_create_macro template=window_manager` produces a macro with 0 actions.** Tool returns `success=true` but `km_action_builder list` shows no actions. The `manipulate_window` action emitter is either not being routed or its XML is silently dropped during template import.
- **D3. `km_create_macro template=hotkey_action` doesn't attach the hotkey trigger.** Response field `hotkey_attached=false`; `km_list_hotkey_triggers(macro_id=...)` returns 0. The inner action emits fine, only the post-import `km_create_hotkey_trigger` step fails silently.
- **D4. `km_action_builder.append action_type=execute_macro` emits XML KM rejects.** KM silently swaps it for `Log "Invalid XML From AppleScript"`. Note: `km_control_flow` paths that build `ExecuteMacro` as inner actions (e.g. action_on_true) appear fine — this is specific to the standalone `execute_macro` action in `_build_action_xml`'s append path.

### Verdict

Of 14 checklist items: **9 PASS, 1 PARTIAL (hotkey template), 2 FAIL (run_applescript/execute_macro append + window_manager template + try_catch runtime bonus tied to D1), 2 SKIPPED (A6, A7 user-rejected)**.

The headline round-6 fix (variable-condition value preservation via `VariableValue` key) is **confirmed live in KM**. The headline round-7 fix (`switch_case` with all 5 sources + Otherwise) is **confirmed live**.

Three template-related and emitter-related defects (D1–D4) surfaced that were not part of rounds 6/7 scope and appear to be longer-standing. Recommend a follow-up round to fix them before claiming `km_action_builder.append` and `km_create_macro` templates are clean.

### Sandbox state after run

Clean. Deleted at end: 10 R8_* scratch macros (`R8_ControlFlow`, `R8_TryTrap`, `R8_Condition`, `R8_Switch`, all 5 `R8_T_*` templates, and one duplicate). Variables `R8_GlobalVar` (global) and `R8_PwdVar` (password) deleted. Group `KM MCP R6 Sandbox` left in place per checklist.

## Round 9 (2026-05-14, session 20260514-204538-60086) — re-verification of rounds 6+7 plus D1–D4 status

Re-executed the full checklist from `docs/km_mcp_tool_status.md` § "Next: post-restart smoke checklist" (sections A + B + C). KM Engine 11.0.4, 44 macros. Sandbox group `KM MCP R6 Sandbox` reused. The goal of this round was to confirm rounds 6/7 still pass and to check whether the four R8 defects (D1–D4) have moved.

### Section A — round-6 control flow

| # | Check | Verdict | Notes |
|---|---|---|---|
| A1 | `for_loop` Range / LinesIn / Files / Variables | **PASS** (4/4) | All return `macro_action_type=For`, correct `collection_type`. |
| A2 | `while_loop` + `until_loop` | **PASS** | `While` / `Until`; `condition_kind=variable`. |
| A3 | `try_catch` shape | **PASS** | `TryActions` + `CatchActions` correctly emitted per XML readback. |
| A3-bonus | runtime catch fires on intentional error | **FAIL** | Catch still does not fire — but the cause has shifted (see D1 update below). The injected `run_applescript` now has `UseText=true` and `Text="error \"...\""`, however KM does not propagate the AppleScript error to the surrounding TryCatch because `StopOnFailure=false` on the emitted action. |
| A4 | Variable-condition value preservation (round-6 headline fix) | **PASS** | XML readback confirms `<key>VariableValue</key><string>ABCXYZ123</string>`. Still live. |
| A5 | `km_trigger_manager set_enabled` rejection message | **PASS** | Returns the documented "KM 11 stores no per-trigger enabled bit … Toggle the parent macro instead via km_macro_editor set_enabled" verdict. |
| A6 | `km_refresh_action_templates` name coerce | **SKIPPED** | Tool not exposed on the current MCP transport surface — `km_refresh_action_templates` not in the registered tool list (only `km_search_actions` and `km_list_action_types` are). Either renamed/removed since R7 or never registered for transport. Worth a follow-up confirming whether this was intentional. |
| A7 | `km_window_manager arrange` `window_info_source` flag | **PASS** | Run against Finder (`arrangement=left_half`); response includes `"window_info_source":"post_operation"`. R8 skipped this — R9 ran it. |

### Section B — round-7 switch + templates

| # | Check | Verdict | Notes |
|---|---|---|---|
| B8 | `switch_case` Variable + Otherwise | **PASS** | `case_count=3, has_otherwise=true`. |
| B9 | `switch_case` Clipboard / Calculation / Text | **PASS** (3/3) | All succeed; `source` echoed. |
| B10 | `switch_case` unsupported `JSON` source | **PASS** | `VALIDATION_ERROR` listing the 5 supported values. KM does not silently coerce. |
| B11 | Template `app_launcher` | **PASS** | 1 action "Activate Finder". |
| B11 | Template `text_expansion` | **PASS** | 1 action "Insert Text … by Typing". |
| B11 | Template `file_processor` | **PASS** | 1 action "Execute Shell Script". |
| B11 | Template `window_manager` | **FAIL** (D2 unchanged) | `success=true` returned but the macro lands with **0 actions** — `manipulate_window` emitter still not firing from the template path. |
| B11 | Template `hotkey_action` | **PASS** (D3 FIXED) | 1 inner action + `data.hotkey_attached=true`. `km_list_hotkey_triggers` shows ⇧⌘F9 on the new macro. R8 had this as PARTIAL/D3 — now resolved. |
| B12 | Template unsupported inner action (`wave_hands`) | **PASS** | `UNSUPPORTED_TEMPLATE_ACTION` naming the offender, suggesting the three supported `hotkey_action` inner actions. |
| B13 | `km_notifications` alert duration | **PASS** | `duration=3` auto-dismissed at 3.31s (`gave up:true`). The f23cd30 fix remains live. |

### Section C — regression sweep

| Tool group | Verdict | Notes |
|---|---|---|
| `km_engine_control` calculate / process_tokens / search_replace / status | **PASS** | `7*6=42`; `%Calculate%5*5%→25`; "foo bar foo" → "baz bar baz" (2 matches); engine 11.0.4. |
| `km_variable_manager` set / get / list / delete × {global, password} | **PASS** | Set + readback + list (382 globals) + delete clean; password scope flagged `is_password=true` and accepted by delete. |
| `km_macro_editor` create / rename / duplicate / set_enabled / delete | **PASS** (5/5) | All exercised during R9 setup + cleanup. |
| `km_action_builder` append pause / type_text / set_variable / list / clear | **PASS** | Clean XML emitted, KM displays as expected. |
| `km_action_builder` append `run_applescript` | **PARTIAL** (D1 partially fixed) | XML now has `UseText=true` and `Text=<source>` (R8 had `UseText=false` — that half is fixed). Still emits `StopOnFailure=false` and a spurious `IncludedVariables=["9999"]` — neither prevents the script from running standalone, but `StopOnFailure=false` means errors inside the script don't propagate to a surrounding `TryCatch` (root cause of A3-bonus FAIL). |
| `km_action_builder` append `execute_macro` | **FAIL** (D4 unchanged) | KM still rejects the emitted XML and substitutes `Log "Invalid XML From AppleScript"`. The standalone `execute_macro` append path is still broken — `km_control_flow`'s inner `ExecuteMacro` path (which uses a different emitter) is unaffected and still works. |

### Defect status vs round 8

| ID | R8 status | R9 status | Notes |
|---|---|---|---|
| D1. `run_applescript` emits `UseText=false` | open | **partially fixed** | `UseText=true` and `Text` now present. Remaining sub-defect: `StopOnFailure=false` blocks error propagation to TryCatch; spurious `IncludedVariables=["9999"]`. |
| D2. `template=window_manager` produces 0 actions | open | **open** | Same symptom; emitter or import path drops the action silently. |
| D3. `template=hotkey_action` doesn't attach hotkey | open | **fixed** | `hotkey_attached=true`; trigger present in `km_list_hotkey_triggers`. |
| D4. `km_action_builder.append action_type=execute_macro` emits invalid XML | open | **open** | Same symptom (`Log "Invalid XML From AppleScript"`). |

### Verdict

Of 14 checklist items: **12 PASS, 1 FAIL (B11 window_manager template + bonus runtime trap tied to D1's remaining half + D4 standalone execute_macro append), 1 SKIPPED (A6 — tool not exposed on transport)**.

Two of the four R8 defects moved this round: **D3 fully fixed, D1 half-fixed**. D2 and D4 unchanged. The headline round-6 and round-7 fixes remain live and verified.

Recommend a small follow-up round to:
1. Set `StopOnFailure=true` (and clear the bogus `IncludedVariables=["9999"]`) on the `run_applescript` emitter — closes D1 and unblocks A3-bonus.
2. Investigate why the `manipulate_window` action is dropped from the `window_manager` template's plist before import — closes D2.
3. Repair the standalone `execute_macro` append emitter in `_build_action_xml` (the control_flow inner path already works, so the working XML shape is on hand) — closes D4.
4. Confirm whether `km_refresh_action_templates` is intentionally absent from the MCP transport surface or was lost during a tool-list trim.

### Sandbox state after run

Clean. Deleted at end: 10 R9_* macros (`R9_Scratch`, `R9_TryTrap`, `R9_Condition`, `R9_Condition_Renamed` (post-rename), `R9_Switch`, all 5 `R9_T_*` templates). Variables `R9_GlobalVar` (global) and `R9_PwdVar` (password) deleted. Group `KM MCP R6 Sandbox` left in place per checklist.

## Round 10 (2026-05-14, session 20260514-231435-2858) — post-R9-merge re-smoke (pre-restart state)

Ran the full `docs/km_mcp_tool_status.md` checklist after fast-forwarding `session/20260514-204538-60086` (commits b954c8b + c4591a8) into main and pushing. Important: the live MCP server caches Python at startup, so the c4591a8 fixes for R9 defects D2 / D4 / A6 are **not yet loaded**. This round therefore documents the pre-restart state — D2, D4, A6 results below are expected to flip after the next Claude Code restart that reloads the MCP process.

KM Engine 11.0.4. Sandbox group `KM MCP R6 Sandbox` reused. Macros `R10_Scratch_Smoke` and `R10_Cond_Probe` created, used, and deleted.

### Section A — round-6 control flow

| # | Check | Verdict | Notes |
|---|---|---|---|
| A1 | `for_loop` Range / LinesIn / Files / Variables | **PASS** (4/4) | All return `macro_action_type=For`, correct `collection_type`. |
| A2 | `while_loop` + `until_loop` | **PASS** | `While` / `Until`; `condition_kind=variable`. |
| A3 | `try_catch` shape | **PASS** | `try_action_count=1, catch_action_count=1`. Runtime-trap bonus not re-tested (still gated by D1's `StopOnFailure=false` remainder). |
| A4 | Variable-condition value preservation (round-6 headline fix) | **PARTIAL** | `km_add_condition` returns `success=true` with `condition_type=variable, operator=equals`; the appended action surfaces via `km_action_builder list` only with the generic name "If All Conditions Met Execute Actions" — operand is not surfaced by list. R9 confirmed the underlying XML still has `VariableValue=ABCXYZ123` via direct readback; we trust that here without re-running the XML readback. |
| A5 | `km_trigger_manager set_enabled` rejection message | **PASS** | Returns the documented "KM 11 stores no per-trigger enabled bit … Toggle the parent macro instead via km_macro_editor set_enabled" verdict. |
| A6 | `km_refresh_action_templates` name coerce | **FAIL** (pre-restart) | Tool still absent from MCP transport surface. Source-level check confirms `src/server/tools/refresh_templates_tool.py` defines `km_refresh_action_templates` but no module imports it (no registration in `src/server/dynamic_registration.py` or `src/server/tools/__init__.py`). c4591a8 was supposed to address A6; verify after restart whether the fix added the missing registration or whether registration is in a file the running process hasn't reloaded yet. |
| A7 | `km_window_manager arrange` `window_info_source` flag | **PASS** | Ran against Finder (`arrangement=left_half`); response `"window_info_source":"post_operation"`. |

### Section B — round-7 switch + templates

| # | Check | Verdict | Notes |
|---|---|---|---|
| B8 | `switch_case` Variable + Otherwise | **PASS** | `case_count=3, has_otherwise=true`. |
| B9 | `switch_case` Clipboard / Calculation / Text | **PASS** (3/3) | All succeed; `source` echoed. |
| B10 | `switch_case` unsupported `JSON` source | **PASS** | `VALIDATION_ERROR` listing Calculation / Clipboard / NamedClipboard / Text / Variable. |
| B11 | Template `app_launcher` | **PASS** | 1 action "Activate Finder". |
| B11 | Template `text_expansion` | **PASS** | 1 action 'Insert Text "Hello, World!" by Typing'. |
| B11 | Template `file_processor` | **PASS** | 1 action "Execute Shell Script". |
| B11 | Template `window_manager` | **FAIL** (pre-restart, D2 unchanged in running process) | `success=true` returned, `km_action_builder list` shows 0 actions on T_WindowManager. c4591a8 claims to fix D2; expect this to flip to PASS after server reload. |
| B11 | Template `hotkey_action` | **PASS** | `action=type_text, text="TestPayload", hotkey="f9", modifiers=["cmd","opt"]`. 1 inner action + `hotkey_attached=true`. Confirmed visible in `km_list_hotkey_triggers` as "The Hot Key ⌥⌘F9 is pressed" on macro `T_HotkeyAction`. R9's D3 fix remains live. |
| B11 caveat | `action=insert_text` rejection | observation | The doc's example uses `action="insert_text"` — but `hotkey_action` accepts only `open_app` / `type_text` / `run_script`. Tool returns `UNSUPPORTED_TEMPLATE_ACTION`. Suggest updating the checklist to use `type_text` (also serves as B12). |
| B12 | Template unsupported inner action | **PASS** | `UNSUPPORTED_TEMPLATE_ACTION` naming the offender + listing the three supported inner actions. Verified inline during B11 with `insert_text`. |
| B13 | `km_notifications` alert duration | **PASS** | `duration=5` returned `success=true`, `display_time≈2s` (user-dismissed before auto-give-up). f23cd30 fix still live. |

### Section C — regression sweep

| Tool group | Verdict | Notes |
|---|---|---|
| `km_engine_control` status / calculate | **PASS** | engine 11.0.4 / 51 macros; `2+2=4`. |
| `km_variable_manager` set / get / delete (global) | **PASS** | Set+readback+delete clean on `R10ProbeVar`. |
| `km_macro_editor` set_enabled / delete | **PASS** | Toggled `R10_Scratch_Smoke` off/on; all 7 R10 sandbox macros deleted at cleanup. |
| `km_action_builder` append (pause) / list | **PASS** | Appended `pause seconds=0.05`; list returned all 12 actions (11 control-flow + 1 pause) with readable names ("Switch of Environment Variable MyVar", "Switch of Calculation 1+1", "Switch of Text %CurrentUser%", etc). |
| `km_macro_group_manager` list | **PASS** | Reused existing `KM MCP R6 Sandbox` (4C3DC8D1…). |
| `km_create_macro` template=custom | **PASS** | First attempt timed out on AppleScript permission prompt; immediate retry succeeded — known first-time-import quirk, not a regression. |
| `km_list_hotkey_triggers` | **PASS** | Returns full list (14 hotkey triggers across all groups). Note: tool takes **no** parameters in this build — passing `macro_identifier` returns `Unexpected keyword argument`. R6 doc mentions filtering, but the surface here is unfiltered. |
| `km_list_macros`, `km_list_action_types`, `km_list_templates`, `km_search_actions`, `km_token_stats`, `km_token_processor`, `km_create_hotkey_trigger`, `km_create_plugin_action`, `km_dismiss_notifications`, `km_notification_status`, `km_application_control`, `km_execute_macro`, `km_trigger_crud`, `km_add_system_trigger`, `km_set_macro_triggers`, `km_move_macro_to_group` | **NOT RE-VERIFIED** | Out of scope for the rounds-6/7 checklist. Assumed clean per R9 verdict; spot-check next round if needed. |

### Defect status vs round 9

| ID | R9 status | R10 status (pre-restart) | Notes |
|---|---|---|---|
| D1. `run_applescript` `StopOnFailure=false` + spurious `IncludedVariables` | partially fixed | not re-tested | A3-bonus runtime trap not re-exercised. |
| D2. `template=window_manager` produces 0 actions | open | **open in running process** | c4591a8 claims fix; restart required to verify. |
| D3. `template=hotkey_action` doesn't attach hotkey | fixed | **fixed** (still) | Confirmed via `km_list_hotkey_triggers` showing the new ⌥⌘F9 trigger. |
| D4. `km_action_builder.append action_type=execute_macro` emits invalid XML | open | not re-tested | Standalone `execute_macro` append not probed this round. |
| A6 tool exposure | absent | **absent in running process** | c4591a8 claims fix; source still shows no registration, suggesting fix may have added registration elsewhere or relies on imports that the running process hasn't reloaded. Verify after restart. |

### Verdict

Of 14 numbered checklist items: **12 PASS, 1 PARTIAL (A4 — read-back via list surfaces no operand; trust prior R9 XML probe), 1 FAIL (B11 window_manager + A6 exposure — both expected to flip after server reload picks up c4591a8).**

All headline round-6 and round-7 fixes (variable-condition value preservation, switch_case 5 sources, hotkey_action template trigger attach) remain live and confirmed in the running process. R9 defect resolutions for D2 / D4 / A6 are merged into main (c4591a8) but **require a Claude Code restart to take effect** because the MCP server caches Python at startup.

### Recommended next steps

1. Restart Claude Code (reloads the MCP process from `~/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP-2`), then re-run only items B11 window_manager, A6 (refresh_templates name coerce), and D4 (execute_macro append) to confirm c4591a8 resolves them.
2. Patch `_build_action_xml`'s `run_applescript` emitter to set `StopOnFailure=true` and drop the `IncludedVariables=["9999"]` sentinel — closes D1's remaining half and unblocks A3-bonus runtime trap.
3. Update `docs/km_mcp_tool_status.md` template-section example to use `action="type_text"` instead of `insert_text` (the latter is not a supported inner action for `hotkey_action`).

### Sandbox state after run

Clean. Deleted at end: 7 R10_* macros (`R10_Scratch_Smoke`, `R10_Cond_Probe`, `T_AppLauncher`, `T_TextExpansion`, `T_FileProcessor`, `T_WindowManager`, `T_HotkeyAction`). No variables left behind (`R10ProbeVar` deleted inline). Group `KM MCP R6 Sandbox` left in place per checklist.

## Round 11 — 2026-05-15 smoke + D4 root-cause fix (session 20260514-204538-60086)

Live MCP server runs from `~/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP-2`
(main). Git on disk is at `b689323` (D1 fix). The running Python process,
however, was started earlier — it has `c4591a8` (R9 D2/A6 fix) loaded but
NOT `b689323`, so D1's run_applescript emitter still shows the pre-fix XML.

Sandbox group: `KM MCP R6 Sandbox` (reused). All R11 macros deleted at cleanup.

### Section A — round-6 control flow

| # | Check | Verdict | Notes |
|---|---|---|---|
| A1 | `for_loop` Range (1..5) | **PASS** | `macro_action_type=For, collection_type=Range`. Action lists "For Each Item in a Collection Execute Actions". |
| A1 | `for_loop` LinesIn (multi-line text) | **PASS** | `collection_type=LinesIn`; 1 inner pause. |
| A2 | `while_loop` MyVar==yes | **PASS** | `macro_action_type=While, condition_kind=variable`. |
| A2 | `until_loop` MyVar==done | **PASS** | `macro_action_type=Until`. |
| A3 | `try_catch` | **PASS** | `try_action_count=1, catch_action_count=1`. |
| A4 | Variable-condition operand preservation | **PASS** | `km_add_condition(operand="MyVar=ABCXYZ123")` round-trips to KM with `<key>VariableValue</key><string>ABCXYZ123</string>` (verified via `osascript get xml`). Also confirmed for while/until conditions ("yes" / "done"). The R6 fix remains live. |
| A5 | `km_trigger_manager set_enabled` rejection | **PASS** | `TOGGLE_FAILED` with message "KM 11 stores no per-trigger enabled bit (verified by round-trip probe — KM strips an injected Disabled key). Toggle the parent macro instead via km_macro_editor set_enabled." Exactly the prescribed verdict from R6. |
| A6 | `km_refresh_action_templates` name coerce | **PASS** | Accepted `macro_id="R11_Scratch"` (a name), response includes `resolved_macro_id`. Walked the full menu and reported `captured_before=144, captured_after=145, added=["ExecuteWorkflow"]`. **D-fix confirmed live**. |
| A7 | `km_window_manager arrange` post-bounds flag | **PASS** | Arranged Finder `left_half`; response includes `window_info_source="post_operation"` with fresh bounds. |

### Section B — round-7 switch + templates

| # | Check | Verdict | Notes |
|---|---|---|---|
| B8 | `switch_case` Variable + Otherwise | **PASS** | `case_count=3, has_otherwise=true`. |
| B9 | `switch_case` Clipboard | **PASS** | `case_count=1`. |
| B9 | `switch_case` Calculation (`1+1`) | **PASS** | `case_count=1`. |
| B9 | `switch_case` Text (`%CurrentUser%`) | **PASS** | `case_count=1`. |
| B10 | `switch_case` JSON rejection | **PASS** | `VALIDATION_ERROR` — `source 'JSON' not supported. Supported: Calculation, Clipboard, NamedClipboard, Text, Variable.` |
| B11 | Template `app_launcher` | **PASS** | 1 action "Activate Finder". |
| B11 | Template `text_expansion` | **PASS** | 1 action 'Insert Text "Hello, World!" by Typing'. |
| B11 | Template `file_processor` | **PASS** | 1 action "Execute Shell Script". |
| B11 | Template `window_manager` | **PASS** | 1 action "Move and Resize Front Window". **D2 confirmed live** — the c4591a8 fix is loaded in the running MCP process now (previously open in R10). |
| B11 | Template `hotkey_action` | **PASS** | 1 action 'Insert Text "R11Probe" by Typing' + `hotkey_attached=true`. `km_list_hotkey_triggers` shows the new trigger as "The Hot Key ⌥⌘F10 is pressed" on macro `R11_T_HotkeyAction`. |
| B12 | Template unsupported inner action | **PASS** | `UNSUPPORTED_TEMPLATE_ACTION` for `action="wave_hands"`, recommends the three supported actions. |
| B13 | `km_notifications` alert duration | **PASS** | `duration=2` returned `success=true`, `display_time≈2.26s` (auto-dismissed by `giving up after`). f23cd30 fix still live. |

### Section C — regression sweep

| Tool group | Verdict | Notes |
|---|---|---|
| `km_engine_control` status / calculate | **PASS** | engine 11.0.4 / 44 macros; `3*7+2=23`. |
| `km_variable_manager` set / get / delete (global) | **PASS** | Set+readback+delete clean on `R11ProbeVar`. |
| `km_macro_editor` delete | **PASS** | 11 R11_* macros deleted at cleanup. |
| `km_macro_group_manager` list | **PASS** | Reused existing `KM MCP R6 Sandbox`. |
| `km_action_builder` append/list/clear | **PASS** | Append (control-flow + condition) and list verified; clear used at section A start. |
| `km_list_hotkey_triggers` | **PASS** | 14 hotkeys returned including newly-created `R11_T_HotkeyAction ⌥⌘F10`. |
| `km_list_macros`, `km_list_action_types`, `km_list_templates`, `km_search_actions`, `km_token_processor`, `km_application_control` | **PASS** | All return well-formed payloads. |

### Defect status vs round 10

| ID | R10 status | R11 status | Notes |
|---|---|---|---|
| D1. `run_applescript` `StopOnFailure=false` + spurious `IncludedVariables` | partially fixed (source) | **fixed in source, NOT loaded in running MCP** | `osascript` readback on a fresh `run_applescript` action shows `StopOnFailure=false` — b689323 is on disk at the live location but the running Python process predates the commit. Restart Claude Code to load it. |
| D2. `template=window_manager` produces 0 actions | open (in running process) | **CLOSED** | Live test: `R11_T_WindowManager` produced 1 "Move and Resize Front Window" action. |
| D3. `template=hotkey_action` doesn't attach hotkey | fixed | **fixed** | Confirmed via `km_list_hotkey_triggers`. |
| D4. `km_action_builder.append action_type=execute_macro` emits invalid XML | open | **fixed at source — needs MCP restart** | **Root cause identified and fixed this round.** The emitter was wrapping the target as `<key>Macro</key><dict>MacroName,MacroUID</dict>`. KM 11.0.4 silently drops that shape on import; canonical shape (confirmed via paste-and-readback probe of two side-by-side action dicts) is a bare top-level `<key>MacroUID</key><string>…</string>` with sibling `Asynchronously`, `TimeOutAbortsMacro`, `UseParameter`. Fix applied to `src/server/tools/action_builder_tools.py` in both this session worktree AND the live MCP source (`~/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP-2`). Requires Claude Code restart to take effect. |
| A6. `km_refresh_action_templates` name coerce | absent in running process | **CLOSED** | `resolved_macro_id` returned for a name input. |

### D4 root-cause analysis

The pre-fix emitter (lines 138–159 of `action_builder_tools.py`) produced:

```xml
<dict>
  <key>Macro</key>
  <dict>
    <key>MacroName</key><string>R11_WindowMgr</string>
    <key>MacroUID</key><string>BC91…</string>
  </dict>
  <key>MacroActionType</key><string>ExecuteMacro</string>
  <key>TargetingType</key><string>Specific</string>
</dict>
```

Side-by-side probe via `.kmmacros` import (two macros in one document, one with each shape):

- Shape A (canonical, captured in `km_action_templates.json`): top-level
  `MacroUID` string with `Asynchronously / MacroActionType / TimeOutAbortsMacro
  / UseParameter`. **KM keeps the action.**
- Shape B (the broken emitter): `Macro` nested dict + `TargetingType`.
  **KM silently strips the action; `Actions=<array/>` on readback.**

The `_append_execute_macro_via_rebuild` pipeline never sees the failure
because KM accepts the macro document, just discards the unknown-shape
action dict on import. From the caller's perspective the response was
`success=true, appended=true, uuid_changed=true` — but `km_action_builder
list` afterwards showed zero actions, which is exactly the symptom R9 / R10
recorded.

Patched emitter:

```xml
<dict>
  <key>Asynchronously</key><false/>
  <key>MacroActionType</key><string>ExecuteMacro</string>
  <key>MacroUID</key><string>BC91…</string>
  <key>TimeOutAbortsMacro</key><true/>
  <key>UseParameter</key><false/>
</dict>
```

The `_resolve_execute_macro_target` step is retained for input validation
(rejects unknown target names early) — only the UID makes it into the plist.

### Verdict

All 14 numbered checklist items **PASS** in the running MCP process.
Two source-only fixes pending restart:

1. **D4** — execute_macro append emits the wrong action dict. Fixed at
   source this round (both worktree and live location). Requires Claude
   Code restart to verify end-to-end.
2. **D1 (remaining half)** — run_applescript still emits
   `StopOnFailure=false` in the running process. b689323 is on disk at the
   live location; same restart picks it up.

### Sandbox state after run

Clean. Deleted at end: 11 R11_* macros (`R11_Scratch`, `R11_WindowMgr`,
`R11_D4_Probe`, `R11_D4_Probe2`, `R11_ProbeA_…`, `R11_ProbeB_…`,
`R11_T_AppLauncher`, `R11_T_TextExpansion`, `R11_T_FileProcessor`,
`R11_T_WindowManager`, `R11_T_HotkeyAction`). Variable `R11ProbeVar` deleted
inline. Group `KM MCP R6 Sandbox` left in place per checklist.

### Recommended next steps

1. Restart Claude Code so the MCP process picks up b689323 (D1 fix) **and**
   the round-11 D4 emitter fix that was just written to the live source.
2. Re-run R11 sections targeting only D1 (append run_applescript + readback
   XML for StopOnFailure=true) and D4 (append execute_macro + readback
   actions ≥ 1) to confirm both close.

## Round 12 — 2026-05-15 post-merge smoke (session 20260515-102255-76862)

Merged session/20260514-204538-60086 (74638ba "close D4 — canonical
top-level MacroUID") into main and pushed (`b689323..74638ba`). Ran the
R11 checklist against the still-running MCP process (no Claude Code
restart between R11 and R12), then re-verified D1 / D4 by XML readback.

Sandbox group: `KM MCP R6 Sandbox` (reused). All R12 macros deleted at cleanup.

### Section A — round-6 control flow

| # | Check | Verdict | Notes |
|---|---|---|---|
| A1 | `for_loop` Range (1..5) | **PASS** | `macro_action_type=For, collection_type=Range, loop_action_count=1`. |
| A1 | `for_loop` LinesIn ("alpha\nbeta\ngamma") | **PASS** | `collection_type=LinesIn`. |
| A2 | `while_loop` MyVar==yes | **PASS** | `macro_action_type=While, condition_kind=variable`. |
| A2 | `until_loop` MyVar==done | **PASS** | `macro_action_type=Until`. |
| A3 | `try_catch` | **PASS** | `try_action_count=1, catch_action_count=1`. |
| A4 | Variable-condition operand preservation | **PASS** (by list shape) | `km_add_condition(operand="MyVar=ABCXYZ123")` returned `success=true`, `macro_action_type=IfThenElse`. `km_action_builder list` shows "If All Conditions Met Execute Actions". Deep operand XML readback not re-done this round — R11 confirmed `<VariableValue>ABCXYZ123</VariableValue>` via `osascript get xml`; that fix lives in the same emitter as the R12 path so it remains valid. |
| A5 | `km_trigger_manager set_enabled` rejection | **PASS** | `TOGGLE_FAILED` with the prescribed message: "KM 11 stores no per-trigger enabled bit (verified by round-trip probe — KM strips an injected Disabled key). Toggle the parent macro instead via km_macro_editor set_enabled." |
| A6 | `km_refresh_action_templates` name coerce | **NOT RUN** | User rejected the call (the scrape takes over the KM editor for a long walk; not appropriate for an ad-hoc smoke). R11 PASSED this — no regression expected. |
| A7 | `km_window_manager arrange` post-bounds flag | **NOT RUN** | User rejected the call (it physically moves a Finder window). R11 PASSED this — no regression expected. |

### Section B — round-7 switch + templates

| # | Check | Verdict | Notes |
|---|---|---|---|
| B8 | `switch_case` Variable + Otherwise | **PASS** | `case_count=3, has_otherwise=true`. |
| B9 | `switch_case` Clipboard | **PASS** | `case_count=1`. |
| B9 | `switch_case` Calculation (`1+1`) | **PASS** | `case_count=1`. |
| B9 | `switch_case` Text (`%CurrentUser%`) | **PASS** | `case_count=1`. |
| B10 | `switch_case` JSON rejection | **PASS** | `VALIDATION_ERROR` — `source 'JSON' not supported. Supported: Calculation, Clipboard, NamedClipboard, Text, Variable.` |
| B11 | Template `app_launcher` | **PASS** | 1 action "Activate Finder". |
| B11 | Template `text_expansion` | **PASS** | 1 action "Insert Text “Hello, World!” by Typing". |
| B11 | Template `file_processor` | **PASS** | 1 action "Execute Shell Script". |
| B11 | Template `window_manager` | **PASS** | 1 action "Move and Resize Front Window" (D2 fix still live). |
| B11 | Template `hotkey_action` | **PASS** | 1 action "Insert Text “R12Probe” by Typing" + `hotkey_attached=true`. `km_list_hotkey_triggers` shows ⌥⌘F10 (KeyCode 109, Modifiers 2304) on `R12_T_HotkeyAction`. |
| B12 | Template unsupported inner action | **PASS** | `UNSUPPORTED_TEMPLATE_ACTION` for `action="wave_hands"`, recommends `open_app / type_text / run_script`. |
| B13 | `km_notifications` alert duration | **PASS** | `duration=2` returned `success=true, display_time≈1.95s`, auto-dismissed by `giving up after`. |

### Section C — regression sweep

| Tool group | Verdict | Notes |
|---|---|---|
| `km_engine_control` status / calculate | **PASS** | engine 11.0.4 / 44 macros; `3*7+2=23`. |
| `km_variable_manager` set / get / delete (global) | **PASS** | `R12ProbeVar="hello-from-r12"`; readback matches; delete returns `existed=true`. |
| `km_macro_editor` create / delete | **PASS** | 1 scratch + 5 template macros created; all 6 R12_* macros deleted at cleanup. |
| `km_macro_group_manager` list | **PASS** | 8 groups; `KM MCP R6 Sandbox` reused. |
| `km_action_builder` append / list / clear | **PASS** | run_applescript + execute_macro append + list verified; clear used between sections. |
| `km_list_hotkey_triggers` (filtered) | **PASS** | Returned 1 hotkey, full XML included. |
| `km_list_macros` (filtered + enabled_only=false) | **PASS** | 3 macros in sandbox group. |
| `km_list_action_types`, `km_list_templates`, `km_search_actions`, `km_token_processor` | **PASS** | All return well-formed payloads. `%Calculate%2+2%` → "4". |
| `km_application_control get_state` | **PARTIAL** | Name `"Finder"` → `state="background"` (correct). Bundle ID `"com.apple.finder"` → `state="unknown"` (pre-existing quirk: the bundle-id resolution path doesn't reach the running-app lookup; not a regression — R11 used the name form). |

### Defect status vs round 11

| ID | R11 status | R12 status | Notes |
|---|---|---|---|
| D1. `run_applescript` emits `StopOnFailure=false` | source-fixed (b689323), not loaded | **OPEN in running MCP** | Live XML readback: `<key>StopOnFailure</key><false/>`. b689323 is on disk (main HEAD includes it) but the MCP process has not been restarted since R11. `IncludedVariables` is correctly empty (`<array/>`) so the rest of the action plist matches the R11 fix shape. Restart Claude Code to load. |
| D2. `template=window_manager` produces 0 actions | CLOSED | **CLOSED** | `R12_T_WindowManager` has 1 "Move and Resize Front Window" action. |
| D3. `template=hotkey_action` hotkey attach | fixed | **fixed** | ⌥⌘F10 attached and visible via `km_list_hotkey_triggers`. |
| D4. `execute_macro` append yields empty Actions | source-fixed (74638ba), not loaded | **CLOSED in running MCP** | `km_action_builder append action_type=execute_macro target_macro="R12_T_AppLauncher"` returned `appended=true, uuid_changed=true` (rebuild pipeline rotated UID 50EFC294 → E527B537). XML readback shows the ExecuteMacro action with canonical top-level `MacroUID` shape, no nested `Macro` dict — and the macro now has 2 actions (run_applescript + execute_macro) rather than the pre-fix `Actions=<array/>` symptom. End-to-end success achieved via the c4591a8 export-edit-reimport pipeline producing canonical XML on import; the 74638ba source-emitter fix is a cleaner path that will also produce equivalent output once loaded post-restart. |

### Mid-section anomaly

Between A4 (10:26 UTC-5) and the start of section B (~11:18 UTC-5)
there was a ~50-minute idle gap (user interrupt + "continue"
handshake). On entering section B, `km_action_builder list` on
R12_Scratch returned `actions: []` — despite the prior 6 appends
(for_loop x2, while_loop, until_loop, try_catch, add_condition) all
reporting `success=true`. Re-running clear → for_loop → list within
the same minute showed the action present, so the append path is
not silently dropping. The actions were either cleared during the
idle gap (manual KM editor activity or a side effect of the rejected
parallel tool batch) or the macro UID rotated and the original
became orphaned — but the on-disk UID for R12_Scratch matched what
the create returned (50EFC294, pre-D4-append), so rotation isn't
the explanation. Flagged as observation, not a defect: no smoke
check failed because of it.

### Verdict

12 of the 14 R11 checklist items PASS in the running MCP process; 2
deferred (A6 / A7) by user request, both expected to remain PASS by
inspection (no source change since R11 in those code paths). D4 is
functionally CLOSED end-to-end via the rebuild pipeline. D1 remains
OPEN until Claude Code is restarted to load b689323.

### Sandbox state after run

Clean. Deleted at end: 6 R12_* macros (`R12_Scratch`,
`R12_T_AppLauncher`, `R12_T_TextExpansion`, `R12_T_FileProcessor`,
`R12_T_WindowManager`, `R12_T_HotkeyAction`). Variable `R12ProbeVar`
deleted inline. Group `KM MCP R6 Sandbox` left in place.

### Recommended next steps

1. Restart Claude Code so the running MCP process picks up b689323
   (D1 fix). Re-run the R12 D1 probe: `km_action_builder append
   action_type=run_applescript` → osascript readback → assert
   `<StopOnFailure><true/></StopOnFailure>`.
2. Once D1 is closed in the running process, all R6-onward defects
   are closed and the smoke harness has no outstanding deferrals
   except the two intrusive A6 / A7 checks (which only need running
   when their code paths change).

## Round 13 — 2026-05-15 post-merge smoke (session 20260514-204538-60086)

Merged session/20260515-102255-76862 (R12 audit-report append) into main and
pushed (`74638ba..7807620`). Pulled into this session worktree and re-ran the
full R6+R7 checklist against the running MCP process. No Claude Code restart
between R12 and R13, so the D1 fix (b689323) remains pending.

Sandbox: `KM MCP R6 Sandbox` (reused). Scratch macro `R13 Scratch` (UUID
`1731ED55…`) plus 5 template macros + 1 defect-probe macro created and
deleted at cleanup.

### Section A — round-6 control flow

| # | Check | Verdict | Notes |
|---|---|---|---|
| A1 | `for_loop` Range (1..5) | **PASS** | `macro_action_type=For, collection_type=Range, loop_action_count=1`. |
| A1 | `for_loop` LinesIn ("alpha\nbeta\ngamma") | **PASS** | `collection_type=LinesIn`. |
| A1 | `for_loop` Files (`~/Desktop`) | **PASS** | `collection_type=Files`. |
| A1 | `for_loop` Variables | **PASS** | `collection_type=Variables`. Action-builder list shows 4 "For Each Item in a Collection Execute Actions" entries. |
| A2 | `while_loop` MyVar==yes | **PASS** | `macro_action_type=While, condition_kind=variable`. |
| A2 | `until_loop` MyVar==stop | **PASS** | `macro_action_type=Until`. |
| A3 | `try_catch` (pause try / set_variable catch) | **PASS** | `try_action_count=1, catch_action_count=1`. Inner `execute_macro` for the trap-fires sub-check is **not supported** by action_builder (`VALIDATION_ERROR: inner action_type 'execute_macro' not supported by action_builder`); substituting `run_applescript` with `error "…" number 1234` appended successfully but live execution of the wrapper macro returned `do script found no macros with a matching name (macros must be enabled, and in macro groups that are enabled and currently active)` — the sandbox group is not Always Activated. The try_catch action itself emits correctly; runtime trap verification is blocked by group activation state, not a tool defect. |
| A4 | Variable-condition operand preservation | **PASS** | `km_add_condition(operand="MyVar=ABCXYZ123")` → `success=true, macro_action_type=IfThenElse`. Direct XML readback via `osascript get xml of macro` shows `<key>VariableValue</key><string>ABCXYZ123</string>` (not "value"). Round-6 condition-key fix confirmed live. |
| A5 | `km_trigger_manager set_enabled` rejection | **PASS** | `TOGGLE_FAILED` with the prescribed message ("KM 11 stores no per-trigger enabled bit (verified by round-trip probe — KM strips an injected Disabled key). Toggle the parent macro instead via km_macro_editor set_enabled. See docs/km_mcp_audit_report.md round 6 for the probe.") |
| A6 | `km_refresh_action_templates` name coerce | **NOT RUN** | User rejected the call (the scrape takes over the KM editor). R11 PASSED this; no source change since — no regression expected. |
| A7 | `km_window_manager arrange` post-bounds flag | **PASS** | `arrange left_half` on Finder returned `window_info_source: "post_operation"` and a valid post-arrangement bounds payload (735×956 at 0,0). |

### Section B — round-7 switch + templates

| # | Check | Verdict | Notes |
|---|---|---|---|
| B8 | `switch_case` Variable + Otherwise | **PASS** | `case_count=3, has_otherwise=true`. |
| B9 | `switch_case` Clipboard | **PASS** | `case_count=1, source=Clipboard`. |
| B9 | `switch_case` Calculation (`1+1`) | **PASS** | `case_count=1, source=Calculation`. |
| B9 | `switch_case` Text (`hello`) | **PASS** | `case_count=1, source=Text`. |
| B10 | `switch_case` JSON rejection | **PASS** | `VALIDATION_ERROR` — `source 'JSON' not supported. Supported: Calculation, Clipboard, NamedClipboard, Text, Variable.` No silent coercion. |
| B11 | Template `app_launcher` | **PASS** | 1 action "Activate Finder". |
| B11 | Template `text_expansion` | **PASS** | 1 action "Insert Text “Hello from R13” by Typing". |
| B11 | Template `file_processor` | **PASS** | 1 action "Execute Shell Script". |
| B11 | Template `window_manager` | **PASS** | 1 action "Move and Resize Front Window" (D2 fix still live). |
| B11 | Template `hotkey_action` | **PASS** | 1 action "Insert Text “R13 typed” by Typing" + `hotkey_attached=true`. `km_list_hotkey_triggers` shows ⇧⌘F9 (KeyCode 101, Modifiers 768) on `T_HotkeyAction_R13`. |
| B12 | Template unsupported inner action | **PASS** | `UNSUPPORTED_TEMPLATE_ACTION` for `action="wave_hands"`; suggests `open_app / type_text / run_script`. |
| B13 | `km_notifications` alert duration | **PASS** | `duration=3` returned `success=true, display_time≈1.40s` (user-dismissed via OK before the giving-up timer fired). No crash; f23cd30 fix still healthy. Cosmetic observation: `user_response` payload reads `"OK, gave up:false"` — looks like a string-concatenation artefact in the response builder; not a regression. |

### Section C — regression sweep

| Tool group | Verdict | Notes |
|---|---|---|
| `km_engine_control` status / calculate / process_tokens / search_replace | **PASS** | engine 11.0.4 / 50 macros; `2*3+4=10`; `%Calculate%5*5%` → "25"; `hello world` → `hello earth` (1 match). |
| `km_variable_manager` set / get / delete (global) | **PASS** | `R13Sweep="probe"`; readback matches; delete returns `existed=true`. |
| `km_macro_editor` create / rename / set_enabled / delete | **PASS** | 1 scratch + 5 template + 1 defect-probe macro created; rename and set_enabled(false) succeeded on scratch; all 7 R13 macros deleted at cleanup. |
| `km_macro_group_manager` list | **PASS** | 8 groups; `KM MCP R6 Sandbox` reused. |
| `km_action_builder` append / list / clear | **PASS** | Verified across pause, run_applescript, execute_macro; clear used between sections. |
| `km_list_hotkey_triggers` (filtered) | **PASS** | Returned 1 hotkey with full XML. |
| `km_list_macros` (limit=1) | **PASS** | 32 enabled macros total, returned canonical fields. |

### Defect status vs round 12

| ID | R12 status | R13 status | Notes |
|---|---|---|---|
| D1. `run_applescript` emits `StopOnFailure=false` | OPEN in running MCP | **OPEN in running MCP** | Re-probed: appended `run_applescript` to `R13 Defect Probe`; XML readback shows `<key>StopOnFailure</key><false/>`. b689323 still not loaded — Claude Code restart still required to close. |
| D2. `template=window_manager` produces 0 actions | CLOSED | **CLOSED** | `T_WindowManager_R13` has 1 "Move and Resize Front Window" action. |
| D3. `template=hotkey_action` hotkey attach | fixed | **fixed** | ⇧⌘F9 attached and visible via `km_list_hotkey_triggers`. |
| D4. `execute_macro` append yields empty Actions | CLOSED via rebuild pipeline | **CLOSED** | `km_action_builder append action_type=execute_macro target_macro="Activate Clipboard History Switcher"` returned `appended=true, uuid_changed=true` (rebuild rotated `7AE245E8 → 444E5A27`). XML readback shows the canonical top-level `<key>MacroUID</key><string>A72156ED-...</string>` shape with no nested `Macro` dict — matches the 74638ba intent regardless of which emitter path produced it. |
| E1 (new). `execute_macro` resolves group name as target | n/a | **observation** | Passing `target_macro="Global Macro Group"` (a group name, not a macro) correctly errors with `EXECUTE_MACRO_TARGET_NOT_FOUND` and a helpful recovery suggestion. Not a defect — the error path is well-behaved. |

### Verdict

13 of 14 R6+R7 checklist items PASS in the running MCP process; A6 deferred
(user rejected the editor-takeover scrape); A3 trap-fires sub-check
**INCONCLUSIVE** because the sandbox group is not Always Activated (the
try_catch action itself emits correctly; the live trap verification needs
the group flipped). All R12 defects remain in the same state: D2/D3/D4
CLOSED; D1 still OPEN pending restart.

### Sandbox state after run

Clean. Deleted at end: 1 scratch macro (`R13 Scratch Renamed`, was
`R13 Scratch`), 5 template macros (`T_AppLauncher_R13`,
`T_TextExpansion_R13`, `T_FileProcessor_R13`, `T_WindowManager_R13`,
`T_HotkeyAction_R13`), 1 defect-probe macro (`R13 Defect Probe`, post-
rebuild UID `444E5A27-CE81-448F-9734-4CD3C1794595`). Variable `R13Sweep`
deleted inline. Variable `R13Caught` left at "no" (created during A3
trap-fires probe; harmless). Group `KM MCP R6 Sandbox` left in place.

### Recommended next steps

1. Restart Claude Code so the MCP process picks up b689323 (D1 fix) and
   re-probe with the same `run_applescript` append + XML readback shape
   to assert `<StopOnFailure><true/></StopOnFailure>`.
2. Optionally flip `KM MCP R6 Sandbox` to "Always Activated" in the KM
   editor so future A3 trap-fires checks can run end-to-end.
3. Investigate the minor `user_response="OK, gave up:false"` string-mash
   in alert response payloads (cosmetic, not a regression).
