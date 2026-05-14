# Merge worktrees into main

## Goal
Consolidate work across 11 worktree branches into `main`, then clean up merged worktrees and local branches.

## Approach
1. Inventory each branch vs `main`: ahead/behind, files touched, already-merged status.
2. Present punch list; user selects which branches to merge and in what order.
3. For each selected branch (in chronological order from earliest commit date):
   - `git checkout main` in the primary worktree (~/Desktop/.../Keyboard-Maestro-MCP-2).
   - `git merge --no-ff <branch>`.
   - On conflict: abort, surface details, ask user.
   - On success: push `main` to `origin/main` immediately.
4. After all merges: delete worktrees for merged branches via `git worktree remove`, delete local branches via `git branch -d`.
5. Leave remote session branches, `cleanup/aggressive-scope-cut`, and the current session worktree intact.

## Acceptance criteria
- All user-selected branches are merged into `main` (verified via `git branch --merged main`).
- `origin/main` matches local `main` after final push.
- Worktrees for merged branches are gone; current session worktree untouched.
- No conflicts silently resolved — every conflict was surfaced.
- Working tree clean at end.

## Non-goals
- Touching `cleanup/aggressive-scope-cut` unless user opts in.
- Deleting remote branches.
- Force-pushing to `main`.
