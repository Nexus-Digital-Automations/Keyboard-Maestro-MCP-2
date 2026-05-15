# Quality gate pass

## Goal
Make `ruff check .`, `mypy .`, `pytest --collect-only`, and a secret scan all pass cleanly on `main`.

## Actions
1. **mypy** — Edit `pyproject.toml` `tool.mypy.overrides`. Append modules: `AppKit.*`, `ApplicationServices.*`, `Quartz.*`, `watchdog.*`, `rapidfuzz.*` with `ignore_missing_imports = true`. Single commit.
2. **pytest** — `git rm` the two broken test files:
   - `tests/test_tools/test_condition_tools.py` (imports nonexistent `_apply_operator`)
   - `tests/test_tools/test_macro_move_tools.py` (imports nonexistent `_check_group_exists`)
   Single commit.
3. **root file** — `git mv s110_issues.json logs/`. Single commit.
4. **secret scan** — `grep -REn '(api[_-]?key|password|secret|token|aws_access|private[_-]?key|BEGIN.*PRIVATE)\s*=\s*["\x27]' src/ scripts/`. Report findings.

After each commit, push to `origin/main`.

## Acceptance criteria
- `ruff check .` → exit 0, 0 findings.
- `mypy .` → exit 0, 0 errors.
- `pytest --collect-only` → exit 0, no collection errors. 1562+ tests collected.
- Project root no longer contains `s110_issues.json`.
- Secret scan reports no plausible credentials in `src/` or `scripts/`.
- All 4 commits on `origin/main`.
- Primary worktree's prior dirty state (km_action_templates.json + output/) restored intact.

## Non-goals
- Running the full test suite (only collection check).
- Fixing the runtime test failures behind the deleted files (those tests were testing removed helpers).
- Installing additional Python deps.
