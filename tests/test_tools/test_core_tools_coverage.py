"""Comprehensive coverage tests for src/tools/core_tools.py.

This module provides comprehensive test coverage for the core_tools module
to achieve the 95% minimum coverage requirement.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context
from src.core.either import Either
from src.core.errors import (
    ExecutionError,
    PermissionDeniedError,
    TimeoutError,
)
from src.tools.core_tools import register_core_tools


class TestCoreToolsRegistration:
    """Test core tools registration and MCP integration."""

    def test_register_core_tools(self):
        """Test register_core_tools function."""
        mock_mcp = Mock()

        # Should register tools without errors
        register_core_tools(mock_mcp)

        # Should have registered the tool decorator
        assert mock_mcp.tool.called
        assert mock_mcp.tool.call_count == 3  # Three tools registered


class TestKMExecuteMacro:
    """Test km_execute_macro tool functionality."""

    def create_mock_context(self):
        """Create a mock context for testing."""
        mock_ctx = Mock(spec=Context)
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()
        mock_ctx.report_progress = AsyncMock()
        return mock_ctx

    @pytest.mark.asyncio
    async def test_km_execute_macro_basic_success(self):
        """Test basic macro execution success."""
        # Register tools to get access to functions
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        # Get the registered function
        km_execute_macro = (
            mock_mcp.tool.call_args_list[0][1]["func"]
            if mock_mcp.tool.call_args_list
            else None
        )

        if km_execute_macro:
            # Mock the engine and result
            with patch("src.tools.core_tools.get_default_engine") as _mock_get_engine:
                with patch(
                    "src.tools.core_tools.create_simple_macro"
                ) as _mock_create_macro:
                    with patch("asyncio.to_thread") as mock_to_thread:
                        mock_ctx = self.create_mock_context()

                        # Mock successful execution result
                        mock_result = Mock()
                        mock_result.execution_time.total_seconds.return_value = 1.5
                        mock_result.execution_token = "test_execution_token_1"  # noqa: S105
                        mock_result.status.value = "completed"
                        mock_result.timestamp.isoformat.return_value = (
                            "2024-01-01T00:00:00Z"
                        )
                        mock_result.output = "Test output"

                        mock_to_thread.return_value = mock_result

                        result = await km_execute_macro(
                            identifier="test-macro", ctx=mock_ctx
                        )

                        assert result["success"] is True
                        assert result["data"]["macro_id"] == "test-macro"
                        assert result["data"]["execution_time"] == 1.5
                        assert result["data"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_km_execute_macro_validation_error_empty_identifier(self):
        """Test macro execution with empty identifier."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_execute_macro = (
            mock_mcp.tool.call_args_list[0][1]["func"]
            if mock_mcp.tool.call_args_list
            else None
        )

        if km_execute_macro:
            mock_ctx = self.create_mock_context()

            result = await km_execute_macro(
                identifier="   ",  # Empty/whitespace identifier
                ctx=mock_ctx,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "INVALID_PARAMETER"
            assert "empty" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_km_execute_macro_permission_denied(self):
        """Test macro execution with permission denied error."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_execute_macro = (
            mock_mcp.tool.call_args_list[0][1]["func"]
            if mock_mcp.tool.call_args_list
            else None
        )

        if km_execute_macro:
            with patch("src.tools.core_tools.get_default_engine") as _mock_get_engine:
                with patch(
                    "src.tools.core_tools.create_simple_macro"
                ) as _mock_create_macro:
                    with patch("asyncio.to_thread") as mock_to_thread:
                        mock_ctx = self.create_mock_context()

                        # Mock permission denied error
                        mock_to_thread.side_effect = PermissionDeniedError(
                            "Permission denied"
                        )

                        result = await km_execute_macro(
                            identifier="test-macro", ctx=mock_ctx
                        )

                        assert result["success"] is False
                        assert result["error"]["code"] == "PERMISSION_DENIED"
                        assert (
                            "accessibility permissions"
                            in result["error"]["recovery_suggestion"]
                        )

    @pytest.mark.asyncio
    async def test_km_execute_macro_timeout(self):
        """Test macro execution timeout handling."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_execute_macro = (
            mock_mcp.tool.call_args_list[0][1]["func"]
            if mock_mcp.tool.call_args_list
            else None
        )

        if km_execute_macro:
            with patch("src.tools.core_tools.get_default_engine") as _mock_get_engine:
                with patch(
                    "src.tools.core_tools.create_simple_macro"
                ) as _mock_create_macro:
                    with patch("asyncio.to_thread") as mock_to_thread:
                        mock_ctx = self.create_mock_context()

                        # Mock timeout error
                        mock_to_thread.side_effect = TimeoutError("Operation timed out")

                        result = await km_execute_macro(
                            identifier="test-macro", timeout=5, ctx=mock_ctx
                        )

                        assert result["success"] is False
                        assert result["error"]["code"] == "TIMEOUT_ERROR"
                        assert "5s" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_km_execute_macro_execution_error(self):
        """Test macro execution error handling."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_execute_macro = (
            mock_mcp.tool.call_args_list[0][1]["func"]
            if mock_mcp.tool.call_args_list
            else None
        )

        if km_execute_macro:
            with patch("src.tools.core_tools.get_default_engine") as _mock_get_engine:
                with patch(
                    "src.tools.core_tools.create_simple_macro"
                ) as _mock_create_macro:
                    with patch("asyncio.to_thread") as mock_to_thread:
                        mock_ctx = self.create_mock_context()

                        # Mock execution error
                        mock_to_thread.side_effect = ExecutionError("Execution failed")

                        result = await km_execute_macro(
                            identifier="test-macro", ctx=mock_ctx
                        )

                        assert result["success"] is False
                        assert result["error"]["code"] == "EXECUTION_ERROR"
                        assert (
                            "macro configuration"
                            in result["error"]["recovery_suggestion"]
                        )

    @pytest.mark.asyncio
    async def test_km_execute_macro_unexpected_error(self):
        """Test macro execution with unexpected error."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_execute_macro = (
            mock_mcp.tool.call_args_list[0][1]["func"]
            if mock_mcp.tool.call_args_list
            else None
        )

        if km_execute_macro:
            with patch("src.tools.core_tools.get_default_engine") as _mock_get_engine:
                with patch(
                    "src.tools.core_tools.create_simple_macro"
                ) as _mock_create_macro:
                    with patch("asyncio.to_thread") as mock_to_thread:
                        mock_ctx = self.create_mock_context()

                        # Mock unexpected error
                        mock_to_thread.side_effect = RuntimeError("Unexpected error")

                        result = await km_execute_macro(
                            identifier="test-macro", ctx=mock_ctx
                        )

                        assert result["success"] is False
                        assert result["error"]["code"] == "SYSTEM_ERROR"
                        assert "Unexpected system error" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_km_execute_macro_with_trigger_value(self):
        """Test macro execution with trigger value."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_execute_macro = (
            mock_mcp.tool.call_args_list[0][1]["func"]
            if mock_mcp.tool.call_args_list
            else None
        )

        if km_execute_macro:
            with patch("src.tools.core_tools.get_default_engine") as _mock_get_engine:
                with patch(
                    "src.tools.core_tools.create_simple_macro"
                ) as _mock_create_macro:
                    with patch("asyncio.to_thread") as mock_to_thread:
                        mock_ctx = self.create_mock_context()

                        # Mock successful execution result
                        mock_result = Mock()
                        mock_result.execution_time.total_seconds.return_value = 1.0
                        mock_result.execution_token = "test_execution_token_2"  # noqa: S105
                        mock_result.status.value = "completed"
                        mock_result.timestamp.isoformat.return_value = (
                            "2024-01-01T00:00:00Z"
                        )
                        mock_result.output = "Test output"

                        mock_to_thread.return_value = mock_result

                        result = await km_execute_macro(
                            identifier="test-macro",
                            trigger_value="test-trigger-value",
                            ctx=mock_ctx,
                        )

                        assert result["success"] is True
                        assert result["data"]["trigger_value"] == "test-trigger-value"


class TestKMListMacros:
    """Test km_list_macros tool functionality."""

    def create_mock_context(self):
        """Create a mock context for testing."""
        mock_ctx = Mock(spec=Context)
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()
        mock_ctx.report_progress = AsyncMock()
        return mock_ctx

    @pytest.mark.asyncio
    async def test_km_list_macros_basic_success(self):
        """Test basic macro listing."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        # Get the km_list_macros function (second registered tool)
        km_list_macros = (
            mock_mcp.tool.call_args_list[1][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 1
            else None
        )

        if km_list_macros:
            with patch("src.tools.core_tools.get_km_client") as mock_get_client:
                mock_ctx = self.create_mock_context()

                # Mock successful client response
                mock_client = Mock()
                mock_macros = [
                    {"name": "Test Macro 1", "group": "Utilities", "enabled": True},
                    {"name": "Test Macro 2", "group": "Email", "enabled": True},
                ]
                mock_client.list_macros_async.return_value = Either.right(mock_macros)
                mock_get_client.return_value = mock_client

                result = await km_list_macros(ctx=mock_ctx)

                assert result["success"] is True
                assert result["data"]["total_count"] == 2
                assert len(result["data"]["macros"]) == 2
                assert result["data"]["macros"][0]["name"] == "Test Macro 1"

    @pytest.mark.asyncio
    async def test_km_list_macros_with_filters(self):
        """Test macro listing with group filters."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_list_macros = (
            mock_mcp.tool.call_args_list[1][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 1
            else None
        )

        if km_list_macros:
            with patch("src.tools.core_tools.get_km_client") as mock_get_client:
                mock_ctx = self.create_mock_context()

                # Mock client response with filtered macros
                mock_client = Mock()
                mock_macros = [
                    {"name": "Email Macro", "group": "Email", "enabled": True}
                ]
                mock_client.list_macros_async.return_value = Either.right(mock_macros)
                mock_get_client.return_value = mock_client

                result = await km_list_macros(group_filters=["Email"], ctx=mock_ctx)

                assert result["success"] is True
                assert result["data"]["filtered"] is True
                assert result["metadata"]["query_params"]["group_filters"] == ["Email"]

    @pytest.mark.asyncio
    async def test_km_list_macros_sorting(self):
        """Test macro listing with different sort options."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_list_macros = (
            mock_mcp.tool.call_args_list[1][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 1
            else None
        )

        if km_list_macros:
            with patch("src.tools.core_tools.get_km_client") as mock_get_client:
                mock_ctx = self.create_mock_context()

                # Mock client response with macros to sort
                mock_client = Mock()
                mock_macros = [
                    {
                        "name": "Z Macro",
                        "group": "Email",
                        "last_used": "2024-01-02T00:00:00Z",
                    },
                    {
                        "name": "A Macro",
                        "group": "Utilities",
                        "last_used": "2024-01-01T00:00:00Z",
                    },
                ]
                mock_client.list_macros_async.return_value = Either.right(mock_macros)
                mock_get_client.return_value = mock_client

                result = await km_list_macros(sort_by="name", ctx=mock_ctx)

                assert result["success"] is True
                assert (
                    result["data"]["macros"][0]["name"] == "A Macro"
                )  # Sorted alphabetically

    @pytest.mark.asyncio
    async def test_km_list_macros_connection_failure(self):
        """Test macro listing with KM connection failure."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_list_macros = (
            mock_mcp.tool.call_args_list[1][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 1
            else None
        )

        if km_list_macros:
            with patch("src.tools.core_tools.get_km_client") as mock_get_client:
                mock_ctx = self.create_mock_context()

                # Mock connection failure
                mock_client = Mock()
                mock_client.list_macros_async.return_value = Either.left(
                    "Connection failed"
                )
                mock_get_client.return_value = mock_client

                result = await km_list_macros(ctx=mock_ctx)

                assert result["success"] is False
                assert result["error"]["code"] == "KM_CONNECTION_FAILED"
                assert (
                    "accessibility permissions"
                    in result["error"]["recovery_suggestion"]
                )

    @pytest.mark.asyncio
    async def test_km_list_macros_limit_parameter(self):
        """Test macro listing with limit parameter."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_list_macros = (
            mock_mcp.tool.call_args_list[1][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 1
            else None
        )

        if km_list_macros:
            with patch("src.tools.core_tools.get_km_client") as mock_get_client:
                mock_ctx = self.create_mock_context()

                # Mock client response with many macros
                mock_client = Mock()
                mock_macros = [
                    {"name": f"Macro {i}", "group": "Test", "enabled": True}
                    for i in range(10)
                ]
                mock_client.list_macros_async.return_value = Either.right(mock_macros)
                mock_get_client.return_value = mock_client

                result = await km_list_macros(limit=5, ctx=mock_ctx)

                assert result["success"] is True
                assert len(result["data"]["macros"]) == 5
                assert result["data"]["total_count"] == 10
                assert result["data"]["pagination"]["has_more"] is True


class TestKMVariableManager:
    """Test km_variable_manager tool functionality."""

    def create_mock_context(self):
        """Create a mock context for testing."""
        mock_ctx = Mock(spec=Context)
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()
        mock_ctx.report_progress = AsyncMock()
        return mock_ctx

    @pytest.mark.asyncio
    async def test_variable_manager_get(self):
        """Test getting variable values."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        # Get the km_variable_manager function (third registered tool)
        km_variable_manager = (
            mock_mcp.tool.call_args_list[2][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 2
            else None
        )

        if km_variable_manager:
            mock_ctx = self.create_mock_context()

            result = await km_variable_manager(
                operation="get", name="TestVariable", ctx=mock_ctx
            )

            assert result["success"] is True
            assert result["data"]["name"] == "TestVariable"
            assert result["data"]["exists"] is True
            assert result["data"]["scope"] == "global"

    @pytest.mark.asyncio
    async def test_variable_manager_set(self):
        """Test setting variable values."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_variable_manager = (
            mock_mcp.tool.call_args_list[2][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 2
            else None
        )

        if km_variable_manager:
            mock_ctx = self.create_mock_context()

            result = await km_variable_manager(
                operation="set", name="TestVariable", value="TestValue", ctx=mock_ctx
            )

            assert result["success"] is True
            assert result["data"]["name"] == "TestVariable"
            assert result["data"]["operation"] == "set"
            assert result["data"]["value_length"] == 9

    @pytest.mark.asyncio
    async def test_variable_manager_delete(self):
        """Test deleting variables."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_variable_manager = (
            mock_mcp.tool.call_args_list[2][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 2
            else None
        )

        if km_variable_manager:
            mock_ctx = self.create_mock_context()

            result = await km_variable_manager(
                operation="delete", name="TestVariable", ctx=mock_ctx
            )

            assert result["success"] is True
            assert result["data"]["name"] == "TestVariable"
            assert result["data"]["operation"] == "delete"
            assert result["data"]["existed"] is True

    @pytest.mark.asyncio
    async def test_variable_manager_list(self):
        """Test listing variables."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_variable_manager = (
            mock_mcp.tool.call_args_list[2][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 2
            else None
        )

        if km_variable_manager:
            mock_ctx = self.create_mock_context()

            result = await km_variable_manager(
                operation="list", scope="global", ctx=mock_ctx
            )

            assert result["success"] is True
            assert "variables" in result["data"]
            assert result["data"]["scope"] == "global"
            assert result["data"]["count"] >= 0

    @pytest.mark.asyncio
    async def test_variable_manager_validation_errors(self):
        """Test variable manager validation."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_variable_manager = (
            mock_mcp.tool.call_args_list[2][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 2
            else None
        )

        if km_variable_manager:
            mock_ctx = self.create_mock_context()

            # Test missing name for get operation
            result = await km_variable_manager(operation="get", name=None, ctx=mock_ctx)

            assert result["success"] is False
            assert result["error"]["code"] == "INVALID_PARAMETER"
            assert "name is required" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_variable_manager_set_without_value(self):
        """Test set operation without value."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_variable_manager = (
            mock_mcp.tool.call_args_list[2][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 2
            else None
        )

        if km_variable_manager:
            mock_ctx = self.create_mock_context()

            result = await km_variable_manager(
                operation="set", name="TestVariable", value=None, ctx=mock_ctx
            )

            assert result["success"] is False
            assert result["error"]["code"] == "INVALID_PARAMETER"
            assert "value is required" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_variable_manager_password_scope(self):
        """Test password scope handling."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_variable_manager = (
            mock_mcp.tool.call_args_list[2][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 2
            else None
        )

        if km_variable_manager:
            mock_ctx = self.create_mock_context()

            result = await km_variable_manager(
                operation="get",
                name="password_variable",
                scope="password",
                ctx=mock_ctx,
            )

            assert result["success"] is True
            assert result["data"]["value"] == "[PROTECTED]"

    @pytest.mark.asyncio
    async def test_variable_manager_unknown_operation(self):
        """Test unknown operation handling."""
        mock_mcp = Mock()
        register_core_tools(mock_mcp)

        km_variable_manager = (
            mock_mcp.tool.call_args_list[2][1]["func"]
            if len(mock_mcp.tool.call_args_list) > 2
            else None
        )

        if km_variable_manager:
            mock_ctx = self.create_mock_context()

            result = await km_variable_manager(
                operation="unknown_operation", ctx=mock_ctx
            )

            assert result["success"] is False
            assert result["error"]["code"] == "INVALID_PARAMETER"
            assert "Unknown operation" in result["error"]["message"]
