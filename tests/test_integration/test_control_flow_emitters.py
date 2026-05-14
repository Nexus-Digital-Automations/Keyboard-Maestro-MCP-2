"""Substring-shape tests for the control-flow XML emitters.

The emitters were captured against KM 11's normalized output (see
``tests/fixtures/km_control_flow/``). These tests assert that each emitter
produces the canonical key shape KM accepts. They do NOT compare bytes —
KM canonicalizes whitespace and key ordering on import — only that the
required keys + MacroActionType strings are present in our output.

Owner: km-mcp control-flow workstream (session 20260514-145029-69306).
"""

from __future__ import annotations

from src.integration.km_for_loop_xml import (
    SUPPORTED_COLLECTION_TYPES,
    build_collection_dict,
    build_for_loop_xml,
)
from src.integration.km_if_then_else_xml import build_condition_dict
from src.integration.km_switch_case_xml import (
    SUPPORTED_CASE_CONDITIONS,
    SUPPORTED_SOURCES,
    build_case_entry,
    build_switch_case_xml,
)
from src.integration.km_try_catch_xml import build_try_catch_xml
from src.integration.km_while_loop_xml import (
    build_until_loop_xml,
    build_while_loop_xml,
)


def test_for_loop_xml_has_required_keys() -> None:
    coll = build_collection_dict("Range", start="1", end="5")
    xml = build_for_loop_xml("i", coll, "")
    assert "<key>MacroActionType</key><string>For</string>" in xml
    assert "<key>Variable</key><string>i</string>" in xml
    assert "<key>Collections</key>" in xml
    assert "<key>CollectionList</key>" in xml
    assert "<string>Range</string>" in xml


def test_for_loop_supports_all_13_collection_types() -> None:
    # KM 11's editor exposes exactly 13 collection types via
    # C{Type}Collection.nib resources. Each must be emitted without error.
    assert len(SUPPORTED_COLLECTION_TYPES) == 13
    for ct in SUPPORTED_COLLECTION_TYPES:
        snippet = build_collection_dict(ct)
        assert f"<string>{ct}</string>" in snippet


def test_while_and_until_use_distinct_action_types() -> None:
    cond = build_condition_dict("variable", "equals", "MyVar=yes")
    while_xml = build_while_loop_xml(cond, "")
    until_xml = build_until_loop_xml(cond, "")
    assert "<string>While</string>" in while_xml
    assert "<string>Until</string>" in until_xml
    assert "<key>Conditions</key>" in while_xml
    assert "<key>Conditions</key>" in until_xml
    # The two only differ in MacroActionType
    assert while_xml.replace("While", "Until") == until_xml


def test_variable_condition_uses_VariableValue_key() -> None:
    # Regression: KM 11 stores the comparison value under VariableValue,
    # not ConditionResult. The earlier emitter shipped the wrong key,
    # which silently dropped the RHS at import time.
    cond = build_condition_dict("variable", "equals", "MyVar=expected")
    assert "<key>VariableValue</key><string>expected</string>" in cond
    assert "<key>ConditionResult</key>" not in cond


def test_text_condition_uses_KM_native_keys() -> None:
    # Regression: ConditionType=Text (not TextContents),
    # TextConditionType (not TextContentsConditionType),
    # TextValue (not ConditionResult).
    cond = build_condition_dict("text", "contains", "%Variable%X%=hello")
    assert "<key>ConditionType</key><string>Text</string>" in cond
    assert "<key>TextConditionType</key>" in cond
    assert "<key>TextValue</key><string>hello</string>" in cond
    assert "TextContents" not in cond


def test_try_catch_xml_has_required_keys() -> None:
    xml = build_try_catch_xml("", "")
    assert "<key>MacroActionType</key><string>TryCatch</string>" in xml
    assert "<key>TryActions</key><array></array>" in xml
    assert "<key>CatchActions</key><array></array>" in xml


def test_switch_case_xml_has_required_keys() -> None:
    entry = build_case_entry("Is", "v1", "")
    xml = build_switch_case_xml("Variable", entry, source_value="MyVar")
    assert "<key>MacroActionType</key><string>Switch</string>" in xml
    assert "<key>Source</key><string>Variable</string>" in xml
    assert "<key>Variable</key><string>MyVar</string>" in xml
    assert "<key>CaseEntries</key>" in xml
    assert "<string>Is</string>" in xml
    assert "<string>v1</string>" in xml


def test_switch_case_supports_all_5_KM_sources() -> None:
    # KM 11 Switch accepts exactly 5 Source values (verified by
    # inject-and-readback probe; KM normalizes anything else to 'Clipboard').
    assert len(SUPPORTED_SOURCES) == 5
    for src in SUPPORTED_SOURCES:
        xml = build_switch_case_xml(src, "", source_value="x")
        assert f"<string>{src}</string>" in xml


def test_switch_case_supports_all_5_case_condition_types() -> None:
    # KM 11 Switch CaseEntry accepts exactly 5 ConditionType values
    # (Is, IsNot, Contains, DoesNotContain, Otherwise).
    assert len(SUPPORTED_CASE_CONDITIONS) == 5
    for cond in SUPPORTED_CASE_CONDITIONS:
        snippet = build_case_entry(cond, "v", "")
        assert f"<string>{cond}</string>" in snippet


def test_switch_case_default_is_otherwise_sentinel_entry() -> None:
    # KM has no separate OtherwiseActions key — the default case is a
    # CaseEntry with ConditionType=Otherwise.
    entry = build_case_entry("Otherwise", "", "")
    assert "<key>ConditionType</key><string>Otherwise</string>" in entry


def test_switch_case_named_clipboard_uses_typo_RedundandDisplayName() -> None:
    # KM's plist key really is "RedundandDisplayName" (typo preserved).
    xml = build_switch_case_xml(
        "NamedClipboard", "",
        named_clipboard_name="abc-123",
        named_clipboard_display="My Clip",
    )
    assert "<key>NamedClipboardName</key><string>abc-123</string>" in xml
    assert "<key>RedundandDisplayName</key><string>My Clip</string>" in xml
