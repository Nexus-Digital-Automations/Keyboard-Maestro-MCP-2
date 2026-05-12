"""Critical-path tests for km_trigger_crud (data integrity for macro triggers).

Triggers fire automatically — a botched CRUD operation can mutate user
automations in ways that are hard to recover from. These tests exercise
the XML-payload validator, dispatch routing, and error envelopes against
a mocked km_client. Live KM is out of scope per the spec.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.core.either import Either
from src.integration.km_client import KMError
from src.server.tools.trigger_crud_tools import (
    _parse_trigger_type,
    _resolve_payload,
    _trigger_dict_to_xml,
    km_trigger_crud,
)

SAMPLE_TRIGGER_XML = (
    "<dict>"
    "<key>MacroTriggerType</key><string>HotKey</string>"
    "<key>FireType</key><string>Pressed</string>"
    "<key>KeyCode</key><integer>49</integer>"
    "<key>Modifiers</key><integer>256</integer>"
    "</dict>"
)


@pytest.fixture
def mock_client() -> Mock:
    client = Mock()
    client.check_connection = Mock(return_value=Either.right(True))
    client.list_macro_triggers_with_xml_async = AsyncMock(
        return_value=Either.right(
            [
                {
                    "index": 1,
                    "description": "⌃Space",
                    "enabled": True,
                    "xml": SAMPLE_TRIGGER_XML,
                },
            ],
        ),
    )
    client.get_macro_trigger_xml_async = AsyncMock(
        return_value=Either.right(SAMPLE_TRIGGER_XML),
    )
    client.append_macro_trigger_xml_async = AsyncMock(return_value=Either.right(True))
    client.update_macro_trigger_xml_async = AsyncMock(return_value=Either.right(True))
    client.remove_macro_trigger_async = AsyncMock(return_value=Either.right(True))
    client.clear_macro_triggers_async = AsyncMock(return_value=Either.right(True))
    return client


@pytest.fixture
def patched(mock_client: Mock) -> Any:
    with patch(
        "src.server.tools.trigger_crud_tools.get_km_client",
        return_value=mock_client,
    ):
        yield mock_client


def test_trigger_dict_to_xml_emits_bare_dict_element() -> None:
    payload = {"MacroTriggerType": "HotKey", "KeyCode": 49, "Modifiers": 256}
    xml_text = _trigger_dict_to_xml(payload)
    assert xml_text.startswith("<dict>") and xml_text.endswith("</dict>")
    assert "<key>MacroTriggerType</key>" in xml_text
    assert "<string>HotKey</string>" in xml_text


def test_parse_trigger_type_pulls_macro_trigger_type() -> None:
    assert _parse_trigger_type(SAMPLE_TRIGGER_XML) == "HotKey"
    assert _parse_trigger_type("<dict><key>Other</key></dict>") == "Unknown"


def test_resolve_payload_missing_returns_error_envelope() -> None:
    out = _resolve_payload(None, None)
    assert isinstance(out, dict)
    assert out["error"]["code"] == "MISSING_PAYLOAD"


def test_resolve_payload_both_xml_and_trigger_is_rejected() -> None:
    out = _resolve_payload("<dict></dict>", {"MacroTriggerType": "HotKey"})
    assert isinstance(out, dict)
    assert out["error"]["code"] == "BOTH_XML_AND_TRIGGER"


def test_resolve_payload_xml_without_dict_wrapper_is_rejected() -> None:
    out = _resolve_payload("HotKey", None)
    assert isinstance(out, dict)
    assert out["error"]["code"] == "VALIDATION_ERROR"
    assert out["error"]["field"] == "xml"


def test_resolve_payload_from_dict_round_trips_type() -> None:
    result = _resolve_payload(
        None,
        {"MacroTriggerType": "FolderTrigger", "Path": "/Users/me/Watched"},
    )
    assert not isinstance(result, dict) or "error" not in result
    assert isinstance(result, tuple)
    xml_text, summary = result
    assert "<key>MacroTriggerType</key>" in xml_text
    assert summary["type"] == "FolderTrigger"


@pytest.mark.asyncio
async def test_list_returns_typed_triggers(patched: Mock) -> None:
    out = await km_trigger_crud(operation="list", macro_id="MyMacro")
    assert out["success"] is True
    assert out["data"]["triggers"][0]["type"] == "HotKey"


@pytest.mark.asyncio
async def test_get_requires_index(patched: Mock) -> None:
    out = await km_trigger_crud(operation="get", macro_id="MyMacro")
    assert out["success"] is False
    assert out["error"]["code"] == "VALIDATION_ERROR"
    assert out["error"]["field"] == "index"


@pytest.mark.asyncio
async def test_get_returns_xml_and_type(patched: Mock) -> None:
    out = await km_trigger_crud(operation="get", macro_id="MyMacro", index=1)
    assert out["success"] is True
    assert out["data"]["type"] == "HotKey"
    assert "<key>MacroTriggerType</key>" in out["data"]["xml"]


@pytest.mark.asyncio
async def test_add_with_trigger_dict_appends_and_reports_index(patched: Mock) -> None:
    out = await km_trigger_crud(
        operation="add",
        macro_id="MyMacro",
        trigger={"MacroTriggerType": "AppActivated", "Application": {"Name": "Safari"}},
    )
    assert out["success"] is True
    assert out["data"]["type"] == "AppActivated"
    patched.append_macro_trigger_xml_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_rejects_both_payloads(patched: Mock) -> None:
    out = await km_trigger_crud(
        operation="add",
        macro_id="MyMacro",
        xml=SAMPLE_TRIGGER_XML,
        trigger={"MacroTriggerType": "HotKey"},
    )
    assert out["success"] is False
    assert out["error"]["code"] == "BOTH_XML_AND_TRIGGER"


@pytest.mark.asyncio
async def test_update_requires_index_and_payload(patched: Mock) -> None:
    no_index = await km_trigger_crud(operation="update", macro_id="MyMacro", xml=SAMPLE_TRIGGER_XML)
    assert no_index["error"]["code"] == "VALIDATION_ERROR"
    no_payload = await km_trigger_crud(operation="update", macro_id="MyMacro", index=1)
    assert no_payload["error"]["code"] == "MISSING_PAYLOAD"


@pytest.mark.asyncio
async def test_update_passes_xml_through(patched: Mock) -> None:
    out = await km_trigger_crud(
        operation="update",
        macro_id="MyMacro",
        index=1,
        xml=SAMPLE_TRIGGER_XML,
    )
    assert out["success"] is True
    patched.update_macro_trigger_xml_async.assert_awaited_once()
    call_args = patched.update_macro_trigger_xml_async.await_args
    assert call_args.args[1] == 1
    assert "MacroTriggerType" in call_args.args[2]


@pytest.mark.asyncio
async def test_remove_requires_index(patched: Mock) -> None:
    out = await km_trigger_crud(operation="remove", macro_id="MyMacro")
    assert out["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_remove_propagates_out_of_range(patched: Mock) -> None:
    patched.remove_macro_trigger_async = AsyncMock(
        return_value=Either.left(KMError.not_found_error("index out of range")),
    )
    out = await km_trigger_crud(operation="remove", macro_id="MyMacro", index=99)
    assert out["error"]["code"] == "INDEX_OUT_OF_RANGE"


@pytest.mark.asyncio
async def test_replace_all_requires_list(patched: Mock) -> None:
    out = await km_trigger_crud(operation="replace_all", macro_id="MyMacro")
    assert out["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_replace_all_clears_then_inserts(patched: Mock) -> None:
    out = await km_trigger_crud(
        operation="replace_all",
        macro_id="MyMacro",
        triggers=[
            {"MacroTriggerType": "HotKey", "KeyCode": 49, "Modifiers": 256},
            {"xml": SAMPLE_TRIGGER_XML},
        ],
    )
    assert out["success"] is True
    assert out["data"]["count"] == 2
    patched.clear_macro_triggers_async.assert_awaited_once()
    assert patched.append_macro_trigger_xml_async.await_count == 2


@pytest.mark.asyncio
async def test_replace_all_partial_failure_reports_failing_index(patched: Mock) -> None:
    patched.append_macro_trigger_xml_async = AsyncMock(
        side_effect=[
            Either.right(True),
            Either.left(KMError.execution_error("bad xml")),
        ],
    )
    out = await km_trigger_crud(
        operation="replace_all",
        macro_id="MyMacro",
        triggers=[{"MacroTriggerType": "HotKey"}, {"MacroTriggerType": "Bogus"}],
    )
    assert out["success"] is False
    assert out["error"]["code"] == "REPLACE_FAILED"
    assert out["error"]["field"] == "triggers[1]"


@pytest.mark.asyncio
async def test_engine_unreachable_short_circuits(mock_client: Mock) -> None:
    mock_client.check_connection = Mock(return_value=Either.right(False))
    with patch(
        "src.server.tools.trigger_crud_tools.get_km_client",
        return_value=mock_client,
    ):
        out = await km_trigger_crud(operation="list", macro_id="MyMacro")
    assert out["success"] is False
    assert out["error"]["code"] == "KME_UNREACHABLE"
