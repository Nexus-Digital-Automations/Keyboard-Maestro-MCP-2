# Re-smoke of 8 fixes from commit 30635c8

## Scope

Re-test the 8 KM MCP tool paths fixed in commit 30635c8 via live MCP calls.
Update `docs/km_mcp_audit_report.md` in place with a Re-test column / section.
Use a sandbox group "KM MCP Audit" for mutations; do NOT clean it up — keep
as proof.

## Defects under verification

| ID | Tool path | Pre-fix verdict | Re-test acceptance criterion |
|---|---|---|---|
| F1 | `km_macro_group_manager.create` | false-negative (`CREATE_FAILED` though group existed) | Returns `success: true` AND group is listable by name |
| F4 | `km_macro_editor.duplicate` with `new_name` | orphaned unrenamed copy + tool error | Returns `success: true`; exactly one copy exists with the requested `new_name`; original untouched |
| F15 | `km_trigger_crud.list` (and `km_trigger_manager.list`) | coercion crash | Returns non-empty trigger list for a macro known to have triggers; no AppleScript coercion error |
| F17 | `km_add_action` | corrupted macros with "Invalid XML From AppleScript" log placeholder | Tool refuses corruption-causing inputs with a clear error; for permitted types (none of the unsafe registry entries), no garbage action is inserted into the target macro |
| F18 | `km_add_condition` | `defusedxml.ElementTree.Element` AttributeError | Does NOT raise the import-shaped error; returns a structured response (success or a non-Python error envelope) |
| F19 | `km_add_condition` error envelope | flat-string error shape | On failure, error is the project-standard nested-dict shape (`error.code` + `error.message`) |
| F22 | `km_window_manager.get_screens` | mock `1920×1080` regardless of hardware | Returns real Quartz data; reported resolution matches `system_profiler SPDisplaysDataType` for this machine |
| F25 | `km_notifications` (sound type) | `'KMClient' object has no attribute 'play_sound'` | Returns `success: true`; afplay invocation observable or response indicates sound played without AttributeError |

## Non-goals

- Re-test of the 19 other tools (deferred to a future audit).
- Validating fixes at the Python layer (already done in the prior session).
  This pass is strictly black-box through the MCP server.

## Output

- Update `docs/km_mcp_audit_report.md` in-place: add a "Re-test (2026-05-12)"
  section summarizing each of F1/F4/F15/F17/F18/F19/F22/F25 with PASS/FAIL/
  PARTIAL + the raw MCP response shape that justified the verdict.
- Keep the sandbox group `KM MCP Audit` after the run (do not delete).

## Risks / known caveats

- The MCP server caches Python. If the Claude host wasn't restarted since
  commit 30635c8, the fixes won't be live and every test will report the
  pre-fix failure. In that case, capture the evidence and call it out — do
  not try to remediate via Bash.
- F17 verification must NOT actually exercise the corruption path. Probe
  the gate only; abort if the test would mutate a real macro.
