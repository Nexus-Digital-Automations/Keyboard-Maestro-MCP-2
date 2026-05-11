"""Critical-path tests for km_action_builder (data integrity for macro contents).

Bad action XML can silently corrupt macros. Tests cover validation,
type→XML mapping correctness for v1 action types, KM-engine disconnect,
and AppleScript error propagation.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.core.either import Either
from src.integration.km_client import KMError
from src.server.tools.action_builder_tools import _build_action_xml, km_action_builder


@pytest.fixture
def mock_km_client() -> Mock:
    client = Mock()
    client.check_connection = Mock(return_value=Either.right(True))
    client.list_macro_actions_async = AsyncMock(
        return_value=Either.right([{"index": 1, "name": "Pause", "enabled": True}]),
    )
    client.append_macro_action_async = AsyncMock(return_value=Either.right(True))
    client.delete_macro_action_async = AsyncMock(return_value=Either.right(True))
    client.clear_macro_actions_async = AsyncMock(return_value=Either.right(True))
    return client


@pytest.fixture
def patched_client(mock_km_client: Mock) -> Any:
    with patch(
        "src.server.tools.action_builder_tools.get_km_client",
        return_value=mock_km_client,
    ):
        yield mock_km_client


def test_xml_for_pause_includes_time() -> None:
    xml = _build_action_xml("pause", {"seconds": 2.5})
    assert xml is not None
    assert "<string>Pause</string>" in xml
    assert "<string>2.5</string>" in xml


def test_xml_for_type_text_uses_by_typing() -> None:
    xml = _build_action_xml("type_text", {"text": "hi"})
    assert xml is not None
    assert "<string>ByTyping</string>" in xml
    assert "<string>InsertText</string>" in xml


def test_xml_for_paste_uses_by_pasting() -> None:
    xml = _build_action_xml("paste", {"text": "hi"})
    assert xml is not None
    assert "<string>ByPasting</string>" in xml


def test_xml_for_set_variable_requires_variable_name() -> None:
    assert _build_action_xml("set_variable", {"text": "v"}) is None
    xml = _build_action_xml("set_variable", {"variable": "Foo", "text": "bar"})
    assert xml is not None
    assert "<string>Foo</string>" in xml
    assert "<string>bar</string>" in xml


def test_xml_escapes_dangerous_chars() -> None:
    xml = _build_action_xml("type_text", {"text": '<a>"&\'</a>'})
    assert xml is not None
    assert "&lt;a&gt;" in xml
    assert "&quot;" in xml
    assert "&amp;" in xml
    assert "&apos;" in xml


def test_xml_for_execute_macro_requires_target() -> None:
    assert _build_action_xml("execute_macro", {}) is None
    xml = _build_action_xml("execute_macro", {"target_macro": "Other"})
    assert xml is not None
    assert "<string>ExecuteMacro</string>" in xml


def test_xml_for_unknown_type_returns_none() -> None:
    assert _build_action_xml("bogus", {}) is None


@pytest.mark.asyncio
async def test_list_requires_macro_id(patched_client: Mock) -> None:
    result = await km_action_builder(operation="list")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_list_returns_actions(patched_client: Mock) -> None:
    result = await km_action_builder(operation="list", macro_id="M")
    assert result["success"] is True
    assert result["data"]["actions"][0]["name"] == "Pause"


@pytest.mark.asyncio
async def test_append_requires_macro_and_type(patched_client: Mock) -> None:
    result = await km_action_builder(operation="append", macro_id="M")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_append_rejects_unknown_type(patched_client: Mock) -> None:
    result = await km_action_builder(
        operation="append", macro_id="M", action_type="bogus", action_config={},
    )
    assert result["success"] is False
    assert result["error"]["code"] == "UNSUPPORTED_ACTION_TYPE"


@pytest.mark.asyncio
async def test_append_passes_xml_to_client(patched_client: Mock) -> None:
    result = await km_action_builder(
        operation="append", macro_id="M", action_type="pause", action_config={"seconds": 1},
    )
    assert result["success"] is True
    args = patched_client.append_macro_action_async.await_args.args
    assert args[0] == "M"
    assert "<string>Pause</string>" in args[1]


@pytest.mark.asyncio
async def test_delete_requires_both(patched_client: Mock) -> None:
    result = await km_action_builder(operation="delete", macro_id="M")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_delete_passes_index(patched_client: Mock) -> None:
    result = await km_action_builder(operation="delete", macro_id="M", action_index=3)
    assert result["success"] is True
    patched_client.delete_macro_action_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_clear_dispatches(patched_client: Mock) -> None:
    result = await km_action_builder(operation="clear", macro_id="M")
    assert result["success"] is True
    patched_client.clear_macro_actions_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_km_engine_unreachable_short_circuits(mock_km_client: Mock) -> None:
    mock_km_client.check_connection = Mock(return_value=Either.right(False))
    with patch(
        "src.server.tools.action_builder_tools.get_km_client",
        return_value=mock_km_client,
    ):
        result = await km_action_builder(operation="list", macro_id="M")
    assert result["success"] is False
    assert result["error"]["code"] == "KM_CONNECTION_FAILED"
    mock_km_client.list_macro_actions_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_underlying_applescript_error_propagates(mock_km_client: Mock) -> None:
    mock_km_client.append_macro_action_async = AsyncMock(
        return_value=Either.left(KMError.execution_error("macro not found")),
    )
    with patch(
        "src.server.tools.action_builder_tools.get_km_client",
        return_value=mock_km_client,
    ):
        result = await km_action_builder(
            operation="append", macro_id="missing",
            action_type="pause", action_config={"seconds": 1},
        )
    assert result["success"] is False
    assert result["error"]["code"] == "APPEND_FAILED"
    assert "macro not found" in result["error"]["message"]
