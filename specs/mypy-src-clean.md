# Make `src/` mypy-clean

## Goal
Drive mypy errors in `src/` from 361 → 0. Keep existing strict settings (disallow_untyped_defs, warn_return_any, etc.).

## Baseline (per `mypy --explicit-package-bases src/`)
361 errors in 52 files. By category:

| Code | Count | Mechanical? |
|---|---|---|
| no-untyped-def | 63 | yes — add return types |
| assignment | 58 | no — type mismatches |
| return-value | 55 | no |
| arg-type | 49 | no |
| attr-defined | 29 | partial |
| unreachable | 26 | partial — often false positives |
| var-annotated | 14 | yes — annotate bare vars |
| union-attr | 12 | no |
| no-any-return | 12 | partial |
| str / return / dict-item / call-arg | ~33 | partial |
| Rest | ~10 | varies |

## Approach (commit per error-class)
1. **Config:** Add `explicit_package_bases = true` + `files = ["src"]` to `[tool.mypy]` so the gate scans src/. Single commit.
2. **Sweep A — no-untyped-def (63):** add `-> None` (or correct type) to functions/methods missing return annotations.
3. **Sweep B — var-annotated (14):** annotate bare collection/var initializers.
4. **Sweep C — unreachable (26):** review each; remove dead code where real, add `# type: ignore[unreachable]` where it's a false positive on intentional guards.
5. **Sweep D — import-not-found (1):** add module to mypy.overrides.
6. **Sweep E — no-any-return (12):** add `cast` or fix typing where return is genuinely `Any`.
7. **Design fixes — assignment / return-value / arg-type / attr-defined / union-attr / etc.:** group by file, fix per file, commit per file (or small file-group).

After each sweep: re-run mypy on src/, commit + push, update task description with new total.

## Acceptance criteria
- `mypy --explicit-package-bases src/` (or `mypy .` from project root with `files=['src']` set in pyproject) → 0 errors.
- All strict settings preserved.
- Each commit message records error count delta.
- No behavior changes (only annotations + dead-code removal + casts).

## Non-goals
- tests/, scripts/, examples/ — left untouched per user scope choice.
- Disabling strict flags.
