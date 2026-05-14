"""Critical-path tests for the captured-XML template engine.

Bad template substitution corrupts macro contents silently — KM stores a Log
action with "Invalid XML From AppleScript" instead of the intended action.
These tests cover the data-integrity boundary between captured XML and what
gets handed to KM's `make new action`.
"""

import plistlib

import pytest
from src.server.tools._action_templates import (
    find_macro_action_type,
    load_catalog,
    load_templates,
    render_action_xml,
    validate_pasted_dict_xml,
)


def _parse_rendered(dict_xml: str) -> dict:
    body = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
        b'<plist version="1.0">' + dict_xml.encode("utf-8") + b"</plist>"
    )
    return plistlib.loads(body)


class TestCatalog:
    def test_catalog_loads_with_expected_shape(self) -> None:
        entries = load_catalog()
        assert entries, "catalog must not be empty"
        for entry in entries:
            assert isinstance(entry["identifier"], str)
            assert isinstance(entry["MacroActionType"], str)
            assert isinstance(entry["category"], str)
            assert isinstance(entry["keywords"], list)

    def test_find_known_identifier_returns_macro_action_type(self) -> None:
        assert find_macro_action_type("speak_text") == "SpeakText"

    def test_find_unknown_identifier_returns_none(self) -> None:
        assert find_macro_action_type("nope_not_real") is None


class TestTemplateRoundTrip:
    def test_every_loaded_template_round_trips_with_defaults(self) -> None:
        for macro_action_type in load_templates():
            rendered = render_action_xml(macro_action_type, {})
            assert rendered is not None, macro_action_type
            parsed = _parse_rendered(rendered)
            assert parsed.get("MacroActionType") == macro_action_type

    def test_parameter_override_lands_at_mapped_plist_key(self) -> None:
        rendered = render_action_xml("SpeakText", {"text": "Hello tracer"})
        assert rendered is not None
        parsed = _parse_rendered(rendered)
        assert parsed["Text"] == "Hello tracer"

    def test_unknown_parameter_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            render_action_xml("SpeakText", {"frobnitz": "x"})

    def test_unknown_macro_action_type_returns_none(self) -> None:
        assert render_action_xml("NotARealType", {}) is None


class TestPastedXmlValidation:
    _VALID = (
        "<dict><key>MacroActionType</key><string>SpeakText</string>"
        "<key>Text</key><string>hi</string></dict>"
    )

    def test_valid_dict_with_macro_action_type_returns_input(self) -> None:
        assert validate_pasted_dict_xml(self._VALID) == self._VALID

    def test_missing_macro_action_type_key_raises(self) -> None:
        bad = "<dict><key>Text</key><string>hi</string></dict>"
        with pytest.raises(ValueError, match="MacroActionType"):
            validate_pasted_dict_xml(bad)

    def test_malformed_xml_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="malformed"):
            validate_pasted_dict_xml("<dict>not closed")

    def test_non_dict_top_level_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_pasted_dict_xml("<array><string>x</string></array>")
