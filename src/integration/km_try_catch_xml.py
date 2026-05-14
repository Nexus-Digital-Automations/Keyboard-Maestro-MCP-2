"""KM 11 ``Try/Catch`` action plist emitter.

Owner: km-mcp control-flow workstream (session 20260514-145029-69306).

KM has no Throw primitive — Try/Catch traps action failures (failing
AppleScript with `error N`, missing image, invalid macro reference,
etc.). Captured 2026-05-14 from ``tests/fixtures/km_control_flow/
try_catch.xml``; KM round-trips this shape verbatim including
the empty TryActions/CatchActions arrays.

@stable
"""

from __future__ import annotations


def build_try_catch_xml(
    try_actions_xml: str = "",
    catch_actions_xml: str = "",
    *,
    timeout_aborts: bool = True,
) -> str:
    """Wrap try/catch action lists into a TryCatch plist.

    Both arguments are concatenated action ``<dict>`` strings (KM expects
    each as an ``<array>`` item). Either may be empty — KM accepts an
    empty CatchActions silently (the failing try just aborts the macro).
    """
    return (
        "<dict>"
        "<key>CatchActions</key>"
        f"<array>{catch_actions_xml}</array>"
        "<key>MacroActionType</key>"
        "<string>TryCatch</string>"
        "<key>TimeOutAbortsMacro</key>"
        f"<{'true' if timeout_aborts else 'false'}/>"
        "<key>TryActions</key>"
        f"<array>{try_actions_xml}</array>"
        "</dict>"
    )
