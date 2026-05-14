"""Substring-shape tests for the three new action_builder emitters.

The emitters were captured against KM 11's action_templates and verified
by KM-author / read-back probe. These tests assert the canonical key set
KM accepts. They do NOT compare bytes — KM canonicalizes whitespace and
key ordering on import — only that the required keys are present.

Owner: km-mcp template-with-parameters workstream (session 20260514-145029-69306).
"""

from __future__ import annotations

from src.server.tools.action_builder_tools import _build_action_xml


def test_activate_application_emits_canonical_keys() -> None:
    xml = _build_action_xml(
        "activate_application",
        {"app_name": "Finder", "bundle_id": "com.apple.finder"},
    )
    assert xml is not None
    assert "<key>MacroActionType</key><string>ActivateApplication</string>" in xml
    assert "<key>BundleIdentifier</key><string>com.apple.finder</string>" in xml
    assert "<key>Name</key><string>Finder</string>" in xml
    assert "<key>AllWindows</key><true/>" in xml


def test_activate_application_requires_app_identity() -> None:
    # Neither app_name nor bundle_id → emitter returns None.
    assert _build_action_xml("activate_application", {}) is None


def test_manipulate_window_emits_canonical_keys() -> None:
    xml = _build_action_xml(
        "manipulate_window",
        {"action": "MoveAndResize", "x": "100", "y": "100",
         "width": "800", "height": "600"},
    )
    assert xml is not None
    assert "<key>MacroActionType</key><string>ManipulateWindow</string>" in xml
    assert "<key>Action</key><string>MoveAndResize</string>" in xml
    assert "<key>HorizontalExpression</key><string>100</string>" in xml
    assert "<key>WidthExpression</key><string>800</string>" in xml
    assert "<key>Targeting</key><string>FrontWindow</string>" in xml


def test_execute_shell_script_emits_canonical_keys() -> None:
    xml = _build_action_xml("execute_shell_script", {"source": "echo hello"})
    assert xml is not None
    assert "<key>MacroActionType</key><string>ExecuteShellScript</string>" in xml
    assert "<key>Text</key><string>echo hello</string>" in xml
    assert "<key>UseText</key><true/>" in xml


def test_execute_shell_script_requires_source() -> None:
    assert _build_action_xml("execute_shell_script", {}) is None
