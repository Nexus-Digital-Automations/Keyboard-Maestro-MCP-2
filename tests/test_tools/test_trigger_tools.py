"""Critical-path tests for km_trigger_manager (data integrity for macro triggers).

Triggers fire automatically — bad attach/remove behavior can cause endless
fire loops or silently break user automations. These tests cover the same
shape as test_macro_editor_tools.py / test_macro_group_tools.py:
validation, happy-path argument passing, KM-engine-disconnected short-
circuit, and AppleScript error propagation.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.core.either import Either
from src.integration.km_client import KMError
from src.server.tools.trigger_tools import km_trigger_manager


@pytest.fixture
def mock_km_client() -> Mock:
    client = Mock()
    client.check_connection = Mock(return_value=Either.right(True))
    client.list_macro_triggers_async = AsyncMock(
        return_value=Either.right([{"index": 1, "description": "F1", "enabled": True}]),
    )
    client.attach_trigger_async = AsyncMock(return_value=Either.right(True))
    client.remove_macro_trigger_async = AsyncMock(return_value=Either.right(True))
    client.clear_macro_triggers_async = AsyncMock(return_value=Either.right(True))
    client.set_trigger_enabled_async = AsyncMock(return_value=Either.right(True))
    return client


@pytest.fixture
def patched_client(mock_km_client: Mock) -> Any:
    with patch(
        "src.server.tools.trigger_tools.get_km_client",
        return_value=mock_km_client,
    ):
        yield mock_km_client


@pytest.mark.asyncio
async def test_list_requires_macro_id(patched_client: Mock) -> None:
    result = await km_trigger_manager(operation="list")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_list_returns_triggers(patched_client: Mock) -> None:
    result = await km_trigger_manager(operation="list", macro_id="My Macro")
    assert result["success"] is True
    assert result["data"]["triggers"][0]["description"] == "F1"


@pytest.mark.asyncio
async def test_add_requires_macro_and_type(patched_client: Mock) -> None:
    result = await km_trigger_manager(operation="add", macro_id="X")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_add_passes_type_and_config(patched_client: Mock) -> None:
    result = await km_trigger_manager(
        operation="add",
        macro_id="X",
        trigger_type="hotkey",
        config={"key": "f", "modifiers": ["command"]},
    )
    assert result["success"] is True
    patched_client.attach_trigger_async.assert_awaited_once_with(
        "X", "hotkey", {"key": "f", "modifiers": ["command"]},
    )


@pytest.mark.asyncio
async def test_remove_requires_both(patched_client: Mock) -> None:
    result = await km_trigger_manager(operation="remove", macro_id="X")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_remove_passes_index(patched_client: Mock) -> None:
    result = await km_trigger_manager(operation="remove", macro_id="X", trigger_index=2)
    assert result["success"] is True
    patched_client.remove_macro_trigger_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_clear_requires_macro(patched_client: Mock) -> None:
    result = await km_trigger_manager(operation="clear")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_clear_dispatches(patched_client: Mock) -> None:
    result = await km_trigger_manager(operation="clear", macro_id="X")
    assert result["success"] is True
    patched_client.clear_macro_triggers_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_enabled_requires_all_three(patched_client: Mock) -> None:
    result = await km_trigger_manager(operation="set_enabled", macro_id="X", trigger_index=1)
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_set_enabled_passes_flag(patched_client: Mock) -> None:
    result = await km_trigger_manager(
        operation="set_enabled", macro_id="X", trigger_index=1, enabled=False,
    )
    assert result["success"] is True
    assert result["data"]["enabled"] is False


@pytest.mark.asyncio
async def test_km_engine_unreachable_short_circuits(mock_km_client: Mock) -> None:
    mock_km_client.check_connection = Mock(return_value=Either.right(False))
    with patch(
        "src.server.tools.trigger_tools.get_km_client",
        return_value=mock_km_client,
    ):
        result = await km_trigger_manager(operation="list", macro_id="X")
    assert result["success"] is False
    assert result["error"]["code"] == "KM_CONNECTION_FAILED"
    mock_km_client.list_macro_triggers_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_underlying_applescript_error_propagates(mock_km_client: Mock) -> None:
    mock_km_client.attach_trigger_async = AsyncMock(
        return_value=Either.left(KMError.execution_error("unsupported trigger type")),
    )
    with patch(
        "src.server.tools.trigger_tools.get_km_client",
        return_value=mock_km_client,
    ):
        result = await km_trigger_manager(
            operation="add", macro_id="X", trigger_type="bogus", config={},
        )
    assert result["success"] is False
    assert result["error"]["code"] == "ADD_FAILED"
    assert "unsupported" in result["error"]["message"]
