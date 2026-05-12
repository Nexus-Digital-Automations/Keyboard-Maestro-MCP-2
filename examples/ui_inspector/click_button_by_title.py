#!/usr/bin/env python3
"""List or click AXButton elements in the front window of a macOS app.

Owner: Keyboard-Maestro-MCP-2 examples. Designed for use from
``examples/ui_inspector/UI_Inspector.kmmacros`` but runnable standalone for
testing and iteration.

Requires: ``/usr/bin/python3`` (system Python ships with PyObjC). Other
interpreters won't have ``ApplicationServices`` / ``AppKit`` and will exit 5.

Usage:
    click_button_by_title.py list  <app-name-or-bundle-id>
    click_button_by_title.py click <app-name-or-bundle-id> <title-substring>

Exit codes:
    0  success
    2  app not running
    3  no front window
    4  no buttons found / no title match
    5  accessibility permission denied or AX call failed
    6  ambiguous match (multiple buttons match the title substring)
    9  invalid arguments
"""

from __future__ import annotations

import sys

try:
    from AppKit import NSWorkspace
    from ApplicationServices import (
        AXIsProcessTrustedWithOptions,
        AXUIElementCopyAttributeValue,
        AXUIElementCreateApplication,
        AXUIElementPerformAction,
        AXValueGetValue,
        kAXChildrenAttribute,
        kAXDescriptionAttribute,
        kAXErrorSuccess,
        kAXFocusedWindowAttribute,
        kAXMainWindowAttribute,
        kAXPositionAttribute,
        kAXPressAction,
        kAXRoleAttribute,
        kAXSizeAttribute,
        kAXTitleAttribute,
        kAXValueCGPointType,
        kAXValueCGSizeType,
    )
except ImportError as exc:  # pragma: no cover — environment-dependent
    print(
        f"ERROR: PyObjC unavailable: {exc}. Use /usr/bin/python3.",
        file=sys.stderr,
    )
    sys.exit(5)


# Recursion guard: AX hierarchies can be deep (Electron, Catalyst) and
# occasionally cyclic if the app misbehaves. 30 is well past any real UI depth.
MAX_DEPTH = 30


def find_pid(app_id: str) -> int | None:
    workspace = NSWorkspace.sharedWorkspace()
    for app in workspace.runningApplications():
        name = app.localizedName() or ""
        bundle = app.bundleIdentifier() or ""
        if name == app_id or bundle == app_id:
            return int(app.processIdentifier())
    return None


def ax_attr(element, attr: str):
    """Return attribute value or None on any AX error."""
    err, value = AXUIElementCopyAttributeValue(element, attr, None)
    if err != kAXErrorSuccess:
        return None
    return value


def front_window(app_element):
    """Front window is ``AXFocusedWindow`` or, if missing, ``AXMainWindow``."""
    return ax_attr(app_element, kAXFocusedWindowAttribute) or ax_attr(
        app_element, kAXMainWindowAttribute,
    )


def cgpoint(ax_value):
    if ax_value is None:
        return None
    ok, point = AXValueGetValue(ax_value, kAXValueCGPointType, None)
    return (float(point.x), float(point.y)) if ok else None


def cgsize(ax_value):
    if ax_value is None:
        return None
    ok, size = AXValueGetValue(ax_value, kAXValueCGSizeType, None)
    return (float(size.width), float(size.height)) if ok else None


def label_for(button, position) -> str:
    """Human-readable identifier for a button.

    Title → AXDescription → ``<AXRole @ x,y>``. Position-coordinates fallback
    handles unlabelled icon buttons that would otherwise collide in the picker.
    """
    title = ax_attr(button, kAXTitleAttribute) or ax_attr(
        button, kAXDescriptionAttribute,
    )
    if title and str(title).strip():
        return str(title).strip()
    role = ax_attr(button, kAXRoleAttribute) or "AXButton"
    if position:
        return f"<{role} @ {int(position[0])},{int(position[1])}>"
    return f"<{role}>"


def walk_buttons(element, results: list[dict], depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        return
    role = ax_attr(element, kAXRoleAttribute)
    if role == "AXButton":
        position = cgpoint(ax_attr(element, kAXPositionAttribute))
        size = cgsize(ax_attr(element, kAXSizeAttribute))
        results.append(
            {
                "element": element,
                "label": label_for(element, position),
                "position": position,
                "size": size,
            },
        )
        return  # AXButtons don't contain other AXButtons in any normal UI
    children = ax_attr(element, kAXChildrenAttribute) or []
    for child in children:
        walk_buttons(child, results, depth + 1)


def require_accessibility() -> None:
    """Exit 5 if the host process isn't AX-trusted.

    KM's Engine usually already holds this; running from a fresh terminal does
    not. Failing early gives a clearer message than the cryptic ``AX -25204``.
    """
    if not AXIsProcessTrustedWithOptions(None):
        print(
            "ERROR: This process is not Accessibility-trusted. Grant access in "
            "System Settings → Privacy & Security → Accessibility, then retry.",
            file=sys.stderr,
        )
        sys.exit(5)


def buttons_in_front_window(app_id: str) -> list[dict]:
    pid = find_pid(app_id)
    if pid is None:
        print(f"ERROR: {app_id} is not running.", file=sys.stderr)
        sys.exit(2)
    app_element = AXUIElementCreateApplication(pid)
    window = front_window(app_element)
    if window is None:
        print(f"ERROR: {app_id} has no front window.", file=sys.stderr)
        sys.exit(3)
    results: list[dict] = []
    walk_buttons(window, results)
    return results


def cmd_list(app_id: str) -> int:
    buttons = buttons_in_front_window(app_id)
    if not buttons:
        print(f"ERROR: no buttons found in {app_id}'s front window.", file=sys.stderr)
        return 4
    for entry in buttons:
        print(entry["label"])
    return 0


def matches_title(label: str, query: str) -> bool:
    return query.casefold() in label.casefold()


def click_via_quartz(position, size) -> bool:
    """Synthesize a mouse click at the button's centre using Quartz events.

    Used only when AXPress is unsupported by the target (some custom controls).
    """
    if position is None or size is None:
        return False
    try:
        from Quartz import (
            CGEventCreateMouseEvent,
            CGEventPost,
            kCGEventLeftMouseDown,
            kCGEventLeftMouseUp,
            kCGHIDEventTap,
            kCGMouseButtonLeft,
        )
    except ImportError:
        return False
    centre = (position[0] + size[0] / 2.0, position[1] + size[1] / 2.0)
    down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, centre, kCGMouseButtonLeft)
    up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, centre, kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, down)
    CGEventPost(kCGHIDEventTap, up)
    return True


def cmd_click(app_id: str, query: str) -> int:
    buttons = buttons_in_front_window(app_id)
    matches = [b for b in buttons if matches_title(b["label"], query)]
    if not matches:
        print(
            f"ERROR: no button in {app_id} matches '{query}'. Available: "
            + ", ".join(b["label"] for b in buttons[:20]),
            file=sys.stderr,
        )
        return 4
    if len(matches) > 1:
        print(
            f"ERROR: '{query}' is ambiguous: "
            + ", ".join(b["label"] for b in matches),
            file=sys.stderr,
        )
        return 6
    target = matches[0]
    press_err = AXUIElementPerformAction(target["element"], kAXPressAction)
    if press_err == kAXErrorSuccess:
        print(f"OK: AXPress on '{target['label']}'")
        return 0
    if click_via_quartz(target["position"], target["size"]):
        print(f"OK: clicked '{target['label']}' at position fallback")
        return 0
    print(
        f"ERROR: AXPress failed (code {press_err}) and position fallback "
        f"could not synthesise a click for '{target['label']}'.",
        file=sys.stderr,
    )
    return 5


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 9
    mode = argv[1]
    app_id = argv[2]
    if mode == "list":
        require_accessibility()
        return cmd_list(app_id)
    if mode == "click":
        if len(argv) < 4:
            print("ERROR: click requires a button title argument.", file=sys.stderr)
            return 9
        require_accessibility()
        return cmd_click(app_id, argv[3])
    print(f"ERROR: unknown mode '{mode}'. Use 'list' or 'click'.", file=sys.stderr)
    return 9


if __name__ == "__main__":
    sys.exit(main(sys.argv))
