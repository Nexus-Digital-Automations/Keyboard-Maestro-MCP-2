"""KM 11 ``While`` and ``Until`` loop action plist emitters.

Owner: km-mcp control-flow workstream (session 20260514-145029-69306).

Both action types share the same shape (Actions array + Conditions dict +
TimeOutAbortsMacro), differing only in ``MacroActionType``. Two separate
emitters per project standard (no bool flag params).

Captured 2026-05-14 — KM round-trips both shapes verbatim from
``tests/fixtures/km_control_flow/while.xml`` and ``until.xml``.
Conditions reuse ``km_if_then_else_xml.build_condition_dict``.

@stable
"""

from __future__ import annotations

from xml.sax.saxutils import escape as _xml_escape


def _escape(text: str) -> str:
    return _xml_escape(str(text), {'"': "&quot;"})


def build_while_loop_xml(
    condition_xml: str,
    actions_xml: str = "",
    *,
    match: str = "All",
    timeout_aborts: bool = True,
) -> str:
    """Wrap a condition + inner actions into a While action plist.

    ``condition_xml`` is one or more pre-rendered Condition ``<dict>``
    entries from ``build_condition_dict``. ``actions_xml`` is the
    concatenated inner action ``<dict>`` strings.
    """
    return _build_loop("While", condition_xml, actions_xml, match, timeout_aborts)


def build_until_loop_xml(
    condition_xml: str,
    actions_xml: str = "",
    *,
    match: str = "All",
    timeout_aborts: bool = True,
) -> str:
    """Wrap a condition + inner actions into an Until action plist.

    Until loops run the body until the condition first becomes true; the
    body always runs at least once. While loops run only while the
    condition is true.
    """
    return _build_loop("Until", condition_xml, actions_xml, match, timeout_aborts)


def _build_loop(
    macro_action_type: str,
    condition_xml: str,
    actions_xml: str,
    match: str,
    timeout_aborts: bool,
) -> str:
    return (
        "<dict>"
        "<key>Actions</key>"
        f"<array>{actions_xml}</array>"
        "<key>Conditions</key>"
        "<dict>"
        "<key>ConditionList</key>"
        f"<array>{condition_xml}</array>"
        "<key>ConditionListMatch</key>"
        f"<string>{_escape(match)}</string>"
        "</dict>"
        "<key>MacroActionType</key>"
        f"<string>{_escape(macro_action_type)}</string>"
        "<key>TimeOutAbortsMacro</key>"
        f"<{'true' if timeout_aborts else 'false'}/>"
        "</dict>"
    )
