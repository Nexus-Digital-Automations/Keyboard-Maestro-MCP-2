# Spec — Keyboard Maestro MCP smoke audit

## Goal

Smoke-test every `keyboard-maestro` MCP tool exposed by this project, produce
a punch list of every defect found (functional + UX + envelope-shape +
docstring), then fix them in priority order.

## Inputs

A live local Keyboard Maestro Editor + Engine (v11.0.4 confirmed running).
The MCP server is already connected (visible in the deferred-tool list).

## Scope — tools under test

Every tool prefixed `mcp__keyboard-maestro__km_*` (~27 tools). One happy-path
call + one obvious-error call per tool. Skip `km_build_plugin_action` (already
audited in a prior task and verified against live KM).

## Disposable test set

All write/CRUD operations run against a single disposable artifact set:

- Macro group: `KM MCP Audit`
- Macros inside: `Audit Sandbox A`, `Audit Sandbox B` (created/edited/deleted by tests)
- Variable namespace prefix: `MCPAudit_` (e.g. `MCPAudit_Counter`)

The group + leftover macros remain after the run for inspection (user opted
to keep them).

## Recording format

For each tool, capture in a markdown table:

| Field | Notes |
|---|---|
| Tool | `km_*` name |
| Happy call | Inputs used, outcome |
| Error call | Inputs used, outcome |
| Verdict | OK / BUG / UX |
| Defects | Free-text list |

The full audit log lives in `docs/km_mcp_audit_report.md` (created at the
end of the run, since it's a human-readable artifact the user asked for).

## Definition of "bug worth fixing"

User opted for "everything that's not perfect". Concretely:

1. **Functional** — wrong output, crash, violates own spec, ignores valid input.
2. **Envelope shape** — return shape inconsistent across happy/error paths or vs the in-project standard `{success, data|error, metadata}`.
3. **Validation** — accepts inputs the docstring forbids, or rejects inputs the docstring allows.
4. **UX / errors** — error messages without `recovery_suggestion`, generic codes like `UNKNOWN_ERROR`, no field name when a field is at fault.
5. **Docstring** — out-of-date param descriptions, missing failure modes, copy-paste errors.

Don't fix yet, do collect.

## Execution

1. Discover the full tool list (parse from deferred-tool registry).
2. Create the disposable group + macros via the MCP itself (tracer-code: if
   `km_macro_group_manager.create` is broken the audit halts before it
   starts, which is itself a finding).
3. Run smoke pair per tool, recording into the in-session punch list.
4. Produce the audit report at `docs/km_mcp_audit_report.md`.
5. Prioritise defects: P0 (broken core), P1 (misleading output), P2 (UX).
6. Fix in priority order, re-test each fixed tool.
7. Lint changed files, commit, push.

## Acceptance criteria

1. Every `km_*` tool tested with at least one happy + one error call.
2. Each tool has a verdict (OK / BUG / UX).
3. Audit report committed at `docs/km_mcp_audit_report.md`.
4. Every defect with priority P0/P1 has a fix or a documented reason it's
   deferred (e.g. requires external service).
5. Disposable test group `KM MCP Audit` exists after the run, with at least
   one macro inside.
6. Lint passes (`ruff check`) on every touched file.
7. Commit + push on this session branch.

## Non-goals

- No fuzzing / property tests — smoke only.
- No performance benchmarks.
- No changes to KM macros outside the `KM MCP Audit` group.
- No fixes for issues that require KM-side changes (Stairways' product) —
  those get documented and deferred.
