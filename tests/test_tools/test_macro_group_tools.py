"""Critical-path tests for km_macro_group_manager (data integrity for macro library structure)."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.core.either import Either
from src.integration.km_client import KMError
from src.server.tools.macro_group_tools import km_macro_group_manager


@pytest.fixture
def mock_km_client() -> Mock:
    client = Mock()
    client.check_connection = Mock(return_value=Either.right(True))
    client.list_groups_async = AsyncMock(
        return_value=Either.right([{"groupName": "G1", "groupID": "uid-1", "enabled": True}]),
    )
    client.create_group_async = AsyncMock(
        return_value=Either.right({"group_id": "uid-new", "name": "New Group"}),
    )
    client.delete_group_async = AsyncMock(return_value=Either.right(True))
    client.rename_group_async = AsyncMock(return_value=Either.right(True))
    client.set_group_enabled_async = AsyncMock(return_value=Either.right(True))
    return client


@pytest.fixture
def patched_client(mock_km_client: Mock) -> Any:
    with patch(
        "src.server.tools.macro_group_tools.get_km_client",
        return_value=mock_km_client,
    ):
        yield mock_km_client


@pytest.mark.asyncio
async def test_list_returns_groups(patched_client: Mock) -> None:
    result = await km_macro_group_manager(operation="list")
    assert result["success"] is True
    assert result["data"]["groups"][0]["groupName"] == "G1"


@pytest.mark.asyncio
async def test_create_requires_new_name(patched_client: Mock) -> None:
    result = await km_macro_group_manager(operation="create")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_create_passes_name(patched_client: Mock) -> None:
    result = await km_macro_group_manager(
        operation="create", new_name="My Group",
    )
    assert result["success"] is True
    patched_client.create_group_async.assert_awaited_once_with("My Group")


@pytest.mark.asyncio
async def test_delete_requires_group_id(patched_client: Mock) -> None:
    result = await km_macro_group_manager(operation="delete")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_delete_passes_group_id(patched_client: Mock) -> None:
    result = await km_macro_group_manager(operation="delete", group_id="G1")
    assert result["success"] is True
    patched_client.delete_group_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_rename_requires_both(patched_client: Mock) -> None:
    result = await km_macro_group_manager(operation="rename", group_id="G1")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_rename_passes_old_and_new(patched_client: Mock) -> None:
    result = await km_macro_group_manager(
        operation="rename", group_id="Old", new_name="New",
    )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_set_enabled_requires_both(patched_client: Mock) -> None:
    result = await km_macro_group_manager(operation="set_enabled", group_id="G1")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_set_enabled_passes_flag(patched_client: Mock) -> None:
    result = await km_macro_group_manager(
        operation="set_enabled", group_id="G1", enabled=True,
    )
    assert result["success"] is True
    assert result["data"]["enabled"] is True


@pytest.mark.asyncio
async def test_km_engine_unreachable_short_circuits(mock_km_client: Mock) -> None:
    mock_km_client.check_connection = Mock(return_value=Either.right(False))
    with patch(
        "src.server.tools.macro_group_tools.get_km_client",
        return_value=mock_km_client,
    ):
        result = await km_macro_group_manager(operation="list")
    assert result["success"] is False
    assert result["error"]["code"] == "KM_CONNECTION_FAILED"
    mock_km_client.list_groups_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_underlying_applescript_error_propagates(mock_km_client: Mock) -> None:
    mock_km_client.delete_group_async = AsyncMock(
        return_value=Either.left(KMError.not_found_error("no such group")),
    )
    with patch(
        "src.server.tools.macro_group_tools.get_km_client",
        return_value=mock_km_client,
    ):
        result = await km_macro_group_manager(operation="delete", group_id="missing")
    assert result["success"] is False
    assert result["error"]["code"] == "DELETE_FAILED"
    assert "no such group" in result["error"]["message"]
