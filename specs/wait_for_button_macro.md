# Spec — Wait for Button (UI Inspector, macro 2)

Owner: `examples/ui_inspector/`. Companion to `specs/ui_button_clicker_macro.md`.

## Goal

Extend the UI Inspector bundle with a second macro and a third CLI mode
(`wait_click`) that polls the front window of a target app until a named
AXButton appears, then clicks it. Primary use case: as the Then-action of a
Keyboard Maestro trigger (e.g. "When this application launches"), so the user
gets a hands-free "wait for that button, then click it" without writing their
own `Pause` chain.

## Inputs

- **app** — name or bundle identifier, same as the existing `list` / `click`
  modes.
- **title** — case-insensitive substring match (same matcher as `click`).
- **timeout_ms** (optional, default `10000`) — give up after this many ms.
- **poll_ms** (optional, default `250`) — interval between AX scans.

CLI shape: `click_button_by_title.py wait_click <app> <title> [timeout_ms] [poll_ms]`.

## Acceptance criteria

1. **Already-present** — running `wait_click Finder "New Folder" 5000 250`
   while Finder's window is already showing the button returns exit `0` on
   the first poll and prints `OK: AXPress on 'New Folder'`. Wall time <300ms.
2. **Appears mid-wait** — if the button is not present at start but appears
   before the deadline, the macro clicks it and returns `0`. Verified
   indirectly: the loop reuses `buttons_in_front_window` per iteration, and
   the "already-present" case proves the success path; the "no-match
   timeout" case proves the loop actually iterates.
3. **App-not-running covered by wait** — if the target app is not running at
   the moment the wait begins, the script does not exit `2`. It catches
   `AppNotReady` and keeps polling until either the app launches (and shows
   a front window with the button) or the timeout expires.
4. **Timeout** — `wait_click Finder "DefinitelyNotAButton" 1200 200` returns
   exit `7` with a `timed out after 1200ms` message and a
   "Last-seen buttons:" hint listing up to 20 real Finder buttons.
5. **Ambiguity persists** — if two or more buttons match the title every
   time the loop polls, the script returns exit `6` at the deadline (rather
   than blindly clicking one). Single-match resolution wins as soon as it
   occurs.
6. **Bad timeout arg** — `wait_click Finder "X" notanumber` returns exit `9`
   and prints `timeout_ms must be an integer`. Zero or negative ints same.
7. **`Wait for Button` macro imports cleanly** — the regenerated
   `UI_Inspector.kmmacros` parses with `plistlib.load`, contains exactly one
   group with two macros (`Click Button by Title`, `Wait for Button`), and
   the wait macro has 5 actions in the expected order
   (SetVariableToText → PromptForUserInput → ActivateApplication →
   ExecuteShellScript → DisplayText).

## Exit code additions

Reuse the existing table from `specs/ui_button_clicker_macro.md`, adding:

| Code | Meaning |
|---|---|
| 7 | `wait_click` deadline reached with no single match |

Code `2` and `3` no longer apply to `wait_click` — those conditions become
retry signals inside the loop.

## Non-goals

- No verification after click (same as `click` mode).
- No "wait without clicking" mode — would be 5 lines of code, but adds a
  fourth CLI verb for a use case nobody has asked for. Add later if needed.
- No menu / popup / link support — that's a separate broader-controls task.
