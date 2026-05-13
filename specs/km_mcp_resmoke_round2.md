# Re-smoke Round 2: F19 envelope + macro creation via .kmmacros import

## Decisions (from clarify)

- Macro creation strategy: build a minimal `.kmmacros` plist in a temp file
  and have Keyboard Maestro import it via
  `tell application "Keyboard Maestro" to open POSIX file ...`.
  Verify success by listing the parent group and finding the new macro by
  name/UID.
- F19 scope: surgical fix to `km_add_condition` only. Other tools' envelopes
  are out of scope.

## Acceptance criteria

### F19 — `km_add_condition` error envelope

- Every failure-path return from `km_add_condition` returns
  `{"success": False, "error": {"code": "...", "message": "...",
  "recovery_suggestion": "..."}}`.
- The previously-working success path is unchanged.
- The already-nested `INTEGRATION_FAILED` path is left alone (already
  conformant) or migrated to the shared helper for consistency.
- Verified live via MCP by calling `km_add_condition` with an invalid
  `operator` and confirming `response.error.code == "INVALID_OPERATOR"`
  (not the legacy `response.error == "INVALID_OPERATOR"`).

### Macro creation — `km_macro_editor.create` (primary)

- Given a `group_id` (name or UUID) and a `new_name`, returns
  `{"success": True, "data": {"macro_id": "<uuid>", "name": "<new_name>",
  "group_id": "<group_uuid>"}}`.
- A real macro with that name exists in the target group after the call,
  observable via `km_list_macros` and visible in the KM Editor.
- Failure modes return the project-standard nested envelope:
  - `KME_UNREACHABLE` if Keyboard Maestro Engine isn't running.
  - `GROUP_NOT_FOUND` if `group_id` doesn't resolve.
  - `IMPORT_FAILED` if KM's plist import returns an error or the
    expected macro is absent after import.
  - `VALIDATION_ERROR` for missing/empty `group_id` or `new_name`.

### Macro creation — `km_create_macro` (secondary)

- For `template="custom"` and `template="hotkey_action"` (no parameters),
  routes through the same plist import helper and returns
  `{"success": True, "data": {"macro_id": "...", ...}}`.
- Other templates may continue to error (template-specific action
  generation is deferred); they MUST return `UNSUPPORTED_TEMPLATE` with
  a clear message rather than the current `CREATION_ERROR: Precondition
  violated`.

### Non-goals

- Action/trigger population beyond an empty macro (callers chain
  `km_add_action` or `km_trigger_crud` after creation). Template-specific
  action generators are deferred.
- Fixing F15 `trigger_crud.list` or F22 Quartz environment — both out of
  scope for round 2.

## Implementation notes

- `.kmmacros` minimum schema: top-level `<array>` of group dicts, each
  with `Name`, `UID`, and `Macros` (array of macro dicts each with
  `Name`, `UID`, `Actions` array).
- Use `plistlib.dumps(...)` to build the plist; never string-concatenate
  XML (security and quoting risk).
- Resolve the target group's UID by calling `KMClient.list_groups` and
  matching name OR UID case-insensitively. If only a UID is given,
  fetch the matching `groupName` so the plist's `Name` field matches and
  KM merges into the existing group instead of creating a new one.
- Generate a fresh UUID for the new macro (`uuid.uuid4()` upper-cased
  with dashes) and return it.
- After `tell application "Keyboard Maestro" to open POSIX file ...`,
  poll `km_list_macros` for up to ~3 seconds looking for the new UID.
  If found → success. If not found → `IMPORT_FAILED` with the raw
  AppleScript output.
- Temp file: `tempfile.NamedTemporaryFile(suffix=".kmmacros",
  delete=False)`. Best-effort `os.unlink(...)` in a `finally` block.

## Caveats

- KM may surface a one-time "Allow incoming connections?" prompt the
  first time it imports from a path it hasn't seen. Document this in
  the error message if `IMPORT_FAILED` triggers and offer
  `recovery_suggestion` pointing to it.
- MCP host caches Python — verification step depends on the host
  restarting after this commit lands.
