"""KM 11 ``Switch/Case`` action plist emitter.

Owner: km-mcp control-flow workstream (session 20260514-145029-69306).

Surface captured 2026-05-14 by inject-and-readback probe against KM 11.4.
KM Switch is intentionally narrow:

- 5 Source types: ``Variable`` (extra ``Variable`` key for the var name),
  ``Clipboard`` (no extras), ``NamedClipboard`` (extras
  ``NamedClipboardName`` + ``RedundandDisplayName`` — typo is KM's),
  ``Calculation`` (extra ``Calculation`` key for the expression), and
  ``Text`` (extra ``Text`` key for the source string).
- 5 per-case ConditionType values: ``Is``, ``IsNot``, ``Contains``,
  ``DoesNotContain``, ``Otherwise``. KM silently normalizes anything else
  to ``Contains`` on import.
- The "default" case is a sentinel CaseEntry with
  ``ConditionType=Otherwise`` (no separate top-level OtherwiseActions key).

@stable
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape as _xml_escape


class UnsupportedSwitchSource(ValueError):
    """Raised when source is not one of KM 11's 5 supported Switch sources."""


class UnsupportedCaseCondition(ValueError):
    """Raised when a case's condition_type is not one of KM 11's 5 supported values."""


SUPPORTED_SOURCES: tuple[str, ...] = (
    "Calculation",
    "Clipboard",
    "NamedClipboard",
    "Text",
    "Variable",
)

SUPPORTED_CASE_CONDITIONS: tuple[str, ...] = (
    "Contains",
    "DoesNotContain",
    "Is",
    "IsNot",
    "Otherwise",
)


def escape(text: str) -> str:
    """XML-escape text for inclusion in a plist <string>."""
    return _xml_escape(str(text), {'"': "&quot;"})


def build_switch_case_xml(
    source: str,
    case_entries_xml: str,
    *,
    source_value: str = "",
    named_clipboard_name: str = "",
    named_clipboard_display: str = "",
) -> str:
    """Wrap a CaseEntries array into a Switch action plist.

    ``case_entries_xml`` is the concatenated CaseEntry ``<dict>`` strings
    from ``build_case_entry``. ``source_value`` carries the per-source extra:
    variable name for Variable, expression for Calculation, source text for
    Text, ignored for Clipboard. NamedClipboard uses the two dedicated args.
    """
    if source not in SUPPORTED_SOURCES:
        raise UnsupportedSwitchSource(
            f"source {source!r} not supported. "
            f"Supported: {sorted(SUPPORTED_SOURCES)}.",
        )
    extras = _source_extras(
        source, source_value, named_clipboard_name, named_clipboard_display,
    )
    return (
        "<dict>"
        f"{extras}"
        "<key>CaseEntries</key>"
        f"<array>{case_entries_xml}</array>"
        "<key>MacroActionType</key>"
        "<string>Switch</string>"
        "<key>Source</key>"
        f"<string>{escape(source)}</string>"
        "</dict>"
    )


def build_case_entry(
    condition_type: str,
    test_value: str,
    actions_xml: str = "",
) -> str:
    """Return one CaseEntry ``<dict>`` for the Switch CaseEntries array.

    ``test_value`` is ignored by KM for ``Otherwise`` but still required as
    a ``<string></string>`` for plist validity.
    """
    if condition_type not in SUPPORTED_CASE_CONDITIONS:
        raise UnsupportedCaseCondition(
            f"condition_type {condition_type!r} not supported. "
            f"Supported: {sorted(SUPPORTED_CASE_CONDITIONS)}.",
        )
    return (
        "<dict>"
        "<key>Actions</key>"
        f"<array>{actions_xml}</array>"
        "<key>ConditionType</key>"
        f"<string>{escape(condition_type)}</string>"
        "<key>TestValue</key>"
        f"<string>{escape(test_value)}</string>"
        "</dict>"
    )


def _source_extras(
    source: str,
    source_value: str,
    named_clipboard_name: str,
    named_clipboard_display: str,
) -> str:
    if source == "Variable":
        # KM expects the var name under <key>Variable</key>, not SourceVariable.
        return f"<key>Variable</key><string>{escape(source_value)}</string>"
    if source == "Calculation":
        return f"<key>Calculation</key><string>{escape(source_value)}</string>"
    if source == "Text":
        return f"<key>Text</key><string>{escape(source_value)}</string>"
    if source == "NamedClipboard":
        # KM's typo "Redundand" preserved — that's the actual plist key.
        return (
            f"<key>NamedClipboardName</key><string>{escape(named_clipboard_name)}</string>"
            f"<key>RedundandDisplayName</key><string>{escape(named_clipboard_display)}</string>"
        )
    return ""  # Clipboard


def render_cases(cases: list[dict[str, Any]], render_actions: Any) -> str:
    """Translate a ``cases`` list-of-dicts into a concatenated CaseEntries string.

    Each case must have ``condition_type`` and ``test_value``. ``actions``
    is an optional list rendered through the supplied ``render_actions``
    callable (typically ``_render_inner_actions`` from control_flow_tools).
    """
    pieces: list[str] = []
    for entry in cases:
        cond = str(entry.get("condition_type", ""))
        test = str(entry.get("test_value", ""))
        actions_xml = render_actions(entry.get("actions"))
        pieces.append(build_case_entry(cond, test, actions_xml))
    return "".join(pieces)
