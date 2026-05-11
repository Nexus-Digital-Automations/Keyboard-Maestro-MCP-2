"""Critical-path tests for km_macro_editor (data integrity for user's macro library)."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.core.either import Either
from src.integration.km_client import KMError
from src.server.tools.macro_editor_tools import km_macro_editor


@pytest.fixture
def mock_km_client() -> Mock:
    client = Mock()
    client.check_connection = Mock(return_value=Either.right(True))
    client.create_macro = Mock(return_value=Either.right({"macro_id": "new-uuid", "success": True}))
    client.delete_macro_async = AsyncMock(return_value=Either.right(True))
    client.rename_macro_async = AsyncMock(return_value=Either.right(True))
    client.duplicate_macro_async = AsyncMock(
        return_value=Either.right({"new_name": "Copy of X", "source": "X"}),
    )
    client.set_macro_enabled_async = AsyncMock(return_value=Either.right(True))
    return client


@pytest.fixture
def patched_client(mock_km_client: Mock) -> Any:
    with patch(
        "src.server.tools.macro_editor_tools.get_km_client",
        return_value=mock_km_client,
    ):
        yield mock_km_client


@pytest.mark.asyncio
async def test_create_requires_new_name_and_group(patched_client: Mock) -> None:
    result = await km_macro_editor(operation="create")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_create_calls_km_client_with_name_and_group(patched_client: Mock) -> None:
    result = await km_macro_editor(
        operation="create",
        new_name="Reload Config",
        group_id="Global Macro Group",
    )
    assert result["success"] is True
    patched_client.create_macro.assert_called_once_with(
        {"name": "Reload Config", "group": "Global Macro Group"},
    )


@pytest.mark.asyncio
async def test_delete_passes_macro_id(patched_client: Mock) -> None:
    result = await km_macro_editor(operation="delete", macro_id="My Macro")
    assert result["success"] is True
    patched_client.delete_macro_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_requires_macro_id(patched_client: Mock) -> None:
    result = await km_macro_editor(operation="delete")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_rename_requires_both_fields(patched_client: Mock) -> None:
    result = await km_macro_editor(operation="rename", macro_id="X")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_rename_passes_old_and_new(patched_client: Mock) -> None:
    result = await km_macro_editor(
        operation="rename", macro_id="Old", new_name="New",
    )
    assert result["success"] is True
    assert result["data"]["new_name"] == "New"


@pytest.mark.asyncio
async def test_duplicate_allows_optional_new_name(patched_client: Mock) -> None:
    result = await km_macro_editor(operation="duplicate", macro_id="Src")
    assert result["success"] is True
    patched_client.duplicate_macro_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_enabled_requires_both(patched_client: Mock) -> None:
    result = await km_macro_editor(operation="set_enabled", macro_id="X")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_set_enabled_passes_flag(patched_client: Mock) -> None:
    result = await km_macro_editor(
        operation="set_enabled", macro_id="X", enabled=False,
    )
    assert result["success"] is True
    assert result["data"]["enabled"] is False


@pytest.mark.asyncio
async def test_km_engine_unreachable_short_circuits(mock_km_client: Mock) -> None:
    mock_km_client.check_connection = Mock(return_value=Either.right(False))
    with patch(
        "src.server.tools.macro_editor_tools.get_km_client",
        return_value=mock_km_client,
    ):
        result = await km_macro_editor(operation="delete", macro_id="X")
    assert result["success"] is False
    assert result["error"]["code"] == "KM_CONNECTION_FAILED"
    mock_km_client.delete_macro_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_underlying_applescript_error_propagates(mock_km_client: Mock) -> None:
    mock_km_client.delete_macro_async = AsyncMock(
        return_value=Either.left(KMError.not_found_error("no such macro")),
    )
    with patch(
        "src.server.tools.macro_editor_tools.get_km_client",
        return_value=mock_km_client,
    ):
        result = await km_macro_editor(operation="delete", macro_id="missing")
    assert result["success"] is False
    assert result["error"]["code"] == "DELETE_FAILED"
    assert "no such macro" in result["error"]["message"]
