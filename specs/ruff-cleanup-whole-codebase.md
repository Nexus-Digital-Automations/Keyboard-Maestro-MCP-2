# Ruff cleanup across whole codebase

## Goal
Resolve all 7 current ruff findings on `main`, regardless of whether the files were touched by this session.

## Findings to fix
1. **I001** (3 files) — unsorted imports
   - `tests/test_tools/test_action_search.py`
   - `tests/test_tools/test_action_templates.py`
   - `tests/test_tools/test_plugin_metadata.py`
2. **S607** — `scripts/capture_km_action_templates.py:80` — partial executable path
3. **SIM115** — `src/integration/kmmacros_import.py:193` — open() without context manager
4. **TC002** — `src/server/tools/refresh_templates_tool.py:26` — move third-party import into TYPE_CHECKING block
5. **W291** — `tests/test_massive_coverage_expansion_phase8.py:8` — trailing whitespace

## Approach
1. Stash uncommitted work in primary worktree (km_action_templates.json change + untracked output/).
2. Per fix-group, edit on `main` in the primary worktree, run `ruff check` on touched files to confirm 0 findings, commit, push.
   - Group A: I001 + W291 — auto-fixable, single commit via `ruff check --fix`.
   - Group B: TC002 — single targeted edit.
   - Group C: SIM115 — refactor open() to `with` block.
   - Group D: S607 — use full path to executable (likely `/usr/bin/osascript` or similar).
3. After each commit, push to `origin/main`.
4. Final `ruff check .` returns clean exit.
5. Restore stashed primary-worktree changes via `git stash pop`.

## Acceptance criteria
- `ruff check .` from project root exits 0 with zero findings.
- 4 commits land on `origin/main` (one per fix group).
- Primary worktree's prior uncommitted state (km_action_templates.json + output/) is restored intact.
- No changes to files outside the 7 listed findings.

## Non-goals
- Mypy errors.
- Unsafe-fix sweep.
- Reformatting / restyle of unrelated code.
