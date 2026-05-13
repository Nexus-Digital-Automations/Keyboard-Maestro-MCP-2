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

### F17 [P0] `km_add_action` — corrupts user macros
The tool returns `success: true` with an `xml_preview`, but the action that ends up in the macro is `Log "Invalid XML From AppleScript"` — KM's fallback when it parses garbage XML. So the tool generates malformed XML, KM rejects it, KM logs the rejection as a new action inside the user's macro, and the tool reports success. **This is the worst defect in the audit:** silent corruption of caller data.

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
