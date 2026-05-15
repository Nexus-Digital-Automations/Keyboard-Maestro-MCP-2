"""Critical-path tests for km_create_macro template rendering.

Hotkey embedding (D3) and template-action emission (D2) are data-integrity
paths: a wrong plist quietly produces an empty macro or a macro with no
trigger, both of which return ``success=True`` and mislead callers.
"""

import plistlib

import pytest
from src.integration.kmmacros_import import build_kmmacros_plist
from src.server.tools.creation_tools import _render_template_actions


def _macro_dict_from_plist(actions_xml: str, triggers_xml: str) -> dict:
    plist_bytes = build_kmmacros_plist(
        "TestGroup",
        "00000000-0000-0000-0000-000000000001",
        "TestMacro",
        "00000000-0000-0000-0000-000000000002",
        actions_xml,
        triggers_xml,
    )
    return plistlib.loads(plist_bytes)[0]["Macros"][0]


def test_hotkey_action_template_embeds_trigger_in_plist() -> None:
    """D3: hotkey_action used to call km_create_hotkey_trigger after the
    .kmmacros import, which raced KM's macro-group activation and silently
    no-op'd. Trigger must instead be embedded in the plist Triggers array
    atomically with the macro.
    """
    actions_xml, triggers_xml = _render_template_actions(
        "hotkey_action",
        {
            "action": "type_text",
            "text": "hi",
            "hotkey": "n",
            "modifiers": ["cmd", "shift"],
        },
    )
    assert triggers_xml, "hotkey_action with hotkey param must emit a trigger"

    macro = _macro_dict_from_plist(actions_xml, triggers_xml)
    triggers = macro["Triggers"]
    assert len(triggers) == 1
    trigger = triggers[0]
    assert trigger["MacroTriggerType"] == "HotKey"
    assert isinstance(trigger["KeyCode"], int)
    assert isinstance(trigger["Modifiers"], int)
    assert trigger["FireType"] == "Pressed"

    # The inner action (type_text) also landed.
    assert len(macro["Actions"]) == 1
    assert macro["Actions"][0]["MacroActionType"] == "InsertText"


def test_hotkey_action_without_hotkey_param_emits_no_trigger() -> None:
    actions_xml, triggers_xml = _render_template_actions(
        "hotkey_action",
        {"action": "type_text", "text": "hi"},
    )
    assert triggers_xml == ""
    assert actions_xml  # the inner action still lands


def test_window_manager_template_emits_manipulate_window_action() -> None:
    """D2: window_manager used to return success with an empty Actions
    array because KM rejected the emitted ManipulateWindow XML during
    import. After fix, the plist must contain exactly one
    ManipulateWindow action with the canonical geometry keys.
    """
    actions_xml, triggers_xml = _render_template_actions(
        "window_manager",
        {"x": "100", "y": "100", "width": "800", "height": "600"},
    )
    assert triggers_xml == ""

    macro = _macro_dict_from_plist(actions_xml, triggers_xml)
    actions = macro["Actions"]
    assert len(actions) == 1
    action = actions[0]
    assert action["MacroActionType"] == "ManipulateWindow"
    assert action["HorizontalExpression"] == "100"
    assert action["VerticalExpression"] == "100"
    assert action["WidthExpression"] == "800"
    assert action["HeightExpression"] == "600"


@pytest.mark.parametrize(
    ("operation_input", "expected_action"),
    [
        ("move", "MoveAndResize"),
        ("resize", "Resize"),
        ("arrange", "MoveAndResize"),
        ("MoveAndResize", "MoveAndResize"),
        ("Resize", "Resize"),
        ("Minimize", "Minimize"),  # passthrough for advanced KM enum values
    ],
)
def test_window_manager_operation_normalizes_to_km_pascal_case(
    operation_input: str, expected_action: str,
) -> None:
    """D2: callers follow the doc's lowercase aliases (move/resize/arrange)
    but KM's Action enum is PascalCase. Lowercase aliases must map to
    canonical names; unknown values pass through so advanced callers can
    reach Minimize/Zoom/etc.
    """
    actions_xml, _triggers_xml = _render_template_actions(
        "window_manager",
        {"operation": operation_input},
    )
    macro = _macro_dict_from_plist(actions_xml, "")
    assert macro["Actions"][0]["Action"] == expected_action
