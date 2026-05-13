# Round-4 Follow-up Fixes

Resolve the four follow-ups logged in round-4 audit (`docs/km_mcp_audit_report.md`). Per user direction: fix all four; #1 is doc-only.

## #1 `process_tokens` system tokens — DOC-ONLY

**Problem:** `km_engine_control.process_tokens` and `km_token_processor` advertise full token-set support, but KM's `process tokens` AppleScript verb returns `%CurrentUser%`, `%FrontWindowName%`, and similar single-value system tokens literally when invoked outside a macro execution context. `%ICUDateTime%`, `%Variable%...%`, `%Calculate%...%` do expand.

**Fix:** Narrow the advertised support in both tool descriptions. No code path change.

**Acceptance:**
- `km_engine_control.process_tokens`' docstring/description mentions the system-context token limitation and lists the affected token classes.
- `km_token_processor`'s docstring/description carries the same caveat.
- No behavior change at runtime; existing PASS probes still pass.

## #2 `km_execute_macro` by-name → UUID retry fallback

**Problem:** Freshly-created macros in a freshly-created group can fail name-based dispatch with `"do script found no macros with a matching name (macros must be enabled, and in macro groups that are enabled and currently active)"` even when both are enabled. UUID-based dispatch works immediately. The disconnect is KM's macro-group activation context propagation lag after `create`.

**Fix:** When `km_execute_macro` is called with a name (non-UUID `identifier`) and KM returns the "no macros with a matching name" error, look up the macro's UUID via the existing macro list path and retry once. If the second call also fails, return the original error.

**Acceptance:**
- Calling `km_execute_macro` immediately after `km_macro_editor.create` succeeds without manual UUID lookup by the caller.
- A name that legitimately doesn't exist still returns the original `EXECUTION_ERROR` (no infinite retry).
- UUID-based callers see no behavior change.
- Retry attempts logged at debug level.

## #3 `km_window_manager.arrange` post-arrange bounds

**Problem:** `arrange` returns success, but the `window` block in the response reports pre-arrange bounds (the move happens, only the echoed snapshot is stale).

**Fix:** After applying the arrangement, re-query the window's bounds and use those in the response payload.

**Acceptance:**
- Response `window.bounds` and `window.position`/`window.size` reflect the post-arrange state.
- No extra AppleScript round-trip when the arrange operation itself fails (we only re-query on success).
- `arrange` execution time still reported but `execution_time_ms` accounts only for the arrange call, not the re-query.

## #4 `km_add_condition` recovery suggestion text

**Problem:** Recovery suggestion advertises `app` as a valid `condition_type` while the validator accepts `application`.

**Fix:** Replace `app` with `application` in the recovery suggestion / docstring.

**Acceptance:**
- Recovery suggestion lists exactly the validator's accepted set: `text, variable, application, system, logic` (or whatever the validator actually accepts — check the source).
- No validator change.

## Non-goals

- No new tests for these — they are not on the critical-paths list (no payments/auth/billing/data-integrity/financial/security touch).
- No refactoring of unchanged code in touched files beyond cleaning obvious rot in modified blocks.

## Verification

After commits and MCP host restart:
- Re-probe `km_execute_macro` by name on a freshly-created macro (was the failing path in round-4).
- Re-probe `km_window_manager.arrange` and inspect the returned bounds.
- Trigger `km_add_condition` with an invalid `condition_type` and confirm the message says `application`, not `app`.
- `km_engine_control.process_tokens` / `km_token_processor` description text updated (visible in tool schema).
