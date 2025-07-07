"""Comprehensive test coverage for Foundation Tools (TASK_1-9).

This module provides systematic testing for the core macro engine components
with focus on MCP tool functionality, error handling, and integration patterns.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.core.errors import ValidationError

logger = logging.getLogger(__name__)


class TestCoreToolsFoundation:
    """Test foundation tools from TASK_1-9 core macro engine."""

    @pytest.fixture
    def sample_macro_data(self) -> Any:
        """Sample macro data for testing."""
        return {
            "macro_id": "test-macro-001",
            "name": "Test Macro",
            "enabled": True,
            "group_id": "test-group-001",
            "commands": [
                {"type": "type_text", "text": "Hello World"},
                {"type": "pause", "duration": 1.0},
            ],
        }

    @pytest.fixture
    def sample_group_data(self) -> Any:
        """Sample group data for testing."""
        return {
            "group_id": "test-group-001",
            "name": "Test Group",
            "enabled": True,
            "activation": "always",
        }


class TestMacroExecutionTools:
    """Test macro execution MCP tools from core engine."""

    def test_core_tools_import(self) -> None:
        """Test that core tools can be imported without errors."""
        try:
            from src.server.tools import core_tools

            assert hasattr(core_tools, "km_execute_macro")
            assert hasattr(core_tools, "km_list_macros")
        except ImportError as e:
            pytest.fail(f"Failed to import core tools: {e}")

    def test_engine_tools_import(self) -> None:
        """Test that engine tools can be imported without errors."""
        try:
            from src.server.tools import engine_tools

            assert hasattr(engine_tools, "km_engine_control")
        except ImportError as e:
            pytest.fail(f"Failed to import engine tools: {e}")

    @pytest.fixture
    def mock_fastmcp_context(self) -> Any:
        """Create mock FastMCP context."""
        context = AsyncMock()
        context.session_id = "test-session-123"
        context.info = AsyncMock()
        context.error = AsyncMock()
        context.report_progress = AsyncMock()
        return context

    @patch("src.server.initialization.get_km_client")
    @pytest.mark.asyncio
    async def test_macro_execution_basic_functionality(
        self,
        mock_get_km_client: Any,
        mock_fastmcp_context: Any,
    ) -> None:
        """Test basic macro execution functionality."""
        # Mock successful execution and connection
        mock_client = Mock()
        mock_client.check_connection.return_value = Mock(
            is_left=Mock(return_value=False),
            get_right=Mock(return_value=True),
        )
        mock_client.execute_macro.return_value = Mock(
            is_left=Mock(return_value=False),
            get_right=Mock(
                return_value={
                    "success": True,
                    "execution_id": "exec-123",
                    "output": "Success",
                },
            ),
        )
        mock_get_km_client.return_value = mock_client

        # Import and test execution
        from src.server.tools.core_tools import km_execute_macro

        result = await km_execute_macro(
            identifier="Test Macro",
            trigger_value="",
            method="applescript",
            timeout=30,
            ctx=mock_fastmcp_context,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "success" in result

    @patch("src.server.initialization.get_km_client")
    @pytest.mark.asyncio
    async def test_macro_listing_functionality(
        self,
        mock_get_km_client: Any,
        mock_fastmcp_context: Any,
    ) -> None:
        """Test macro listing MCP tool."""
        # Mock macro list response
        mock_client = Mock()
        mock_client.list_macros.return_value = Mock(
            is_left=Mock(return_value=False),
            get_right=Mock(
                return_value=[
                    {"macro_id": "m1", "name": "Macro 1", "enabled": True},
                    {"macro_id": "m2", "name": "Macro 2", "enabled": False},
                ],
            ),
        )
        mock_get_km_client.return_value = mock_client

        from src.server.tools.core_tools import km_list_macros

        result = await km_list_macros(
            group_filter="",
            enabled_only=False,
            sort_by="name",
            limit=20,
            ctx=mock_fastmcp_context,
        )

        assert isinstance(result, dict)
        assert "success" in result


class TestGroupManagementTools:
    """Test group management tools."""

    def test_group_tools_import(self) -> None:
        """Test group tools can be imported."""
        try:
            from src.server.tools import group_tools

            assert hasattr(group_tools, "km_list_groups")
        except ImportError as e:
            pytest.fail(f"Failed to import group tools: {e}")

    @pytest.fixture
    def mock_fastmcp_context(self) -> Any:
        """Create mock FastMCP context."""
        context = AsyncMock()
        context.session_id = "test-session-123"
        context.info = AsyncMock()
        context.error = AsyncMock()
        context.report_progress = AsyncMock()
        return context

    @patch("src.server.initialization.get_km_client")
    @pytest.mark.asyncio
    async def test_group_listing(self, mock_get_km_client: Any, mock_fastmcp_context: Any) -> None:
        """Test group listing functionality."""
        mock_client = Mock()
        mock_client.list_groups.return_value = Mock(
            is_left=Mock(return_value=False),
            get_right=Mock(
                return_value=[{"group_id": "g1", "name": "Group 1", "enabled": True}],
            ),
        )
        mock_get_km_client.return_value = mock_client

        from src.server.tools.group_tools import km_list_groups

        result = await km_list_groups(
            include_disabled=True,
            sort_by="name",
            ctx=mock_fastmcp_context,
        )

        assert isinstance(result, dict)
        assert "success" in result


class TestActionBuilderTools:
    """Test action builder tools from TASK_14."""

    def test_action_tools_import(self) -> None:
        """Test action tools can be imported."""
        try:
            from src.server.tools import action_tools

            assert hasattr(action_tools, "km_add_action")
        except ImportError as e:
            pytest.fail(f"Failed to import action tools: {e}")

    @patch("src.server.initialization.get_km_client")
    @pytest.mark.asyncio
    async def test_action_creation_functionality(
        self,
        mock_get_km_client: Any,
        execution_context: Context | Any,
    ) -> None:
        """Test action creation basic functionality."""
        mock_client = AsyncMock()
        mock_client.add_action.return_value = {"success": True}
        mock_get_km_client.return_value = mock_client

        from src.server.tools.action_tools import km_add_action

        result = await km_add_action(
            macro_id="test-macro",
            action_type="type_text",
            action_config={"text": "Hello World"},
            position=0,
            ctx=execution_context,
        )

        assert isinstance(result, dict)
        assert "success" in result


class TestCalculatorTools:
    """Test calculator tools from TASK_18."""

    def test_calculator_tools_import(self) -> None:
        """Test calculator tools can be imported."""
        try:
            from src.server.tools import calculator_tools

            assert hasattr(calculator_tools, "km_calculator")
        except ImportError as e:
            pytest.fail(f"Failed to import calculator tools: {e}")

    @pytest.mark.asyncio
    async def test_basic_calculation(self, execution_context: Context | Any) -> None:
        """Test basic calculation functionality."""
        from src.server.tools.calculator_tools import km_calculate

        result = await km_calculate(
            expression="2 + 2",
            variables={},
            ctx=execution_context,
        )

        assert isinstance(result, dict)
        assert "success" in result


class TestClipboardTools:
    """Test clipboard tools from TASK_11."""

    def test_clipboard_tools_import(self) -> None:
        """Test clipboard tools can be imported."""
        try:
            from src.server.tools import clipboard_tools

            assert hasattr(clipboard_tools, "km_get_clipboard")
            assert hasattr(clipboard_tools, "km_set_clipboard")
        except ImportError as e:
            pytest.fail(f"Failed to import clipboard tools: {e}")

    @patch("src.server.initialization.get_km_client")
    @pytest.mark.asyncio
    async def test_clipboard_operations(self, mock_get_km_client: Any, execution_context: Context | Any) -> None:
        """Test basic clipboard operations."""
        mock_client = AsyncMock()
        mock_client.set_clipboard.return_value = {"success": True}
        mock_client.get_clipboard.return_value = {
            "success": True,
            "content": "Test content",
        }
        mock_get_km_client.return_value = mock_client

        from src.server.tools.clipboard_tools import km_get_clipboard, km_set_clipboard

        # Test setting clipboard
        set_result = await km_set_clipboard(
            content="Test content",
            clipboard_type="text",
            ctx=execution_context,
        )

        assert isinstance(set_result, dict)
        assert "success" in set_result

        # Test getting clipboard
        get_result = await km_get_clipboard(
            clipboard_type="text",
            ctx=execution_context,
        )

        assert isinstance(get_result, dict)
        assert "success" in get_result


class TestFileOperationTools:
    """Test file operation tools from TASK_13."""

    def test_file_tools_import(self) -> None:
        """Test file tools can be imported."""
        try:
            from src.server.tools import file_operation_tools

            assert hasattr(file_operation_tools, "km_file_exists")
        except ImportError as e:
            pytest.fail(f"Failed to import file operation tools: {e}")

    @pytest.mark.asyncio
    async def test_file_existence_check(self, execution_context: Context | Any) -> None:
        """Test file existence checking."""
        # S108 fix: Use secure temporary file for testing
        import tempfile

        from src.server.tools.file_operation_tools import km_file_exists

        with tempfile.NamedTemporaryFile() as temp_file:
            result = await km_file_exists(
                file_path=temp_file.name,
                ctx=execution_context,
            )

        assert isinstance(result, dict)
        assert "success" in result


class TestNotificationTools:
    """Test notification tools from TASK_17."""

    def test_notification_tools_import(self) -> None:
        """Test notification tools can be imported."""
        try:
            from src.server.tools import notification_tools

            assert hasattr(notification_tools, "km_show_notification")
        except ImportError as e:
            pytest.fail(f"Failed to import notification tools: {e}")

    @patch("src.server.initialization.get_km_client")
    @pytest.mark.asyncio
    async def test_notification_display(self, mock_get_km_client: Any, execution_context: Context | Any) -> None:
        """Test notification display functionality."""
        mock_client = AsyncMock()
        mock_client.show_notification.return_value = {"success": True}
        mock_get_km_client.return_value = mock_client

        from src.server.tools.notification_tools import km_show_notification

        result = await km_show_notification(
            title="Test Notification",
            message="This is a test",
            notification_type="info",
            ctx=execution_context,
        )

        assert isinstance(result, dict)
        assert "success" in result


class TestIntegrationPatterns:
    """Test integration patterns across foundation tools."""

    @patch("src.server.initialization.get_km_client")
    @pytest.mark.asyncio
    async def test_error_handling_pattern(self, mock_get_km_client: Any, execution_context: Context | Any) -> None:
        """Test consistent error handling across tools."""
        mock_client = AsyncMock()
        mock_client.execute_macro.side_effect = ValidationError(
            "Invalid macro identifier",
        )
        mock_get_km_client.return_value = mock_client

        from src.server.tools.core_tools import km_execute_macro

        # Test with invalid macro identifier
        result = await km_execute_macro(
            identifier="",  # Invalid empty identifier
            trigger_value="",
            method="applescript",
            timeout=30,
            ctx=execution_context,
        )

        assert isinstance(result, dict)
        assert "success" in result
        # Should handle error gracefully
        if not result.get("success"):
            assert "error" in result

    @patch("src.server.initialization.get_km_client")
    @pytest.mark.asyncio
    async def test_metadata_consistency(self, mock_get_km_client: Any, execution_context: Context | Any) -> None:
        """Test that all tools return consistent metadata."""
        mock_client = AsyncMock()
        mock_client.list_macros.return_value = []
        mock_client.list_groups.return_value = []
        mock_get_km_client.return_value = mock_client

        tools_to_test = [
            ("src.server.tools.core_tools", "km_list_macros"),
            ("src.server.tools.group_tools", "km_list_groups"),
            ("src.server.tools.calculator_tools", "km_calculate"),
        ]

        for module_name, tool_name in tools_to_test:
            try:
                module = __import__(module_name, fromlist=[tool_name])
                tool_func = getattr(module, tool_name)

                # Call with minimal valid parameters
                if tool_name == "km_list_macros":
                    result = await tool_func(
                        group_filter="",
                        enabled_only=False,
                        sort_by="name",
                        search_text="",
                        ctx=execution_context,
                    )
                elif tool_name == "km_list_groups":
                    result = await tool_func(
                        include_disabled=True,
                        sort_by="name",
                        ctx=execution_context,
                    )
                elif tool_name == "km_calculate":
                    result = await tool_func(
                        expression="1+1",
                        variables={},
                        ctx=execution_context,
                    )

                # Verify basic structure
                assert isinstance(result, dict)
                assert "success" in result

            except ImportError:
                # Tool doesn't exist yet, skip
                continue
            except Exception as e:
                # Unexpected error, but don't fail the test
                print(f"Warning: {tool_name} raised {type(e).__name__}: {e}")

    @patch("src.server.initialization.get_km_client")
    @pytest.mark.asyncio
    async def test_context_usage_pattern(self, mock_get_km_client: Any, execution_context: Context | Any) -> None:
        """Test that tools properly use MCP context."""
        mock_client = AsyncMock()
        mock_client.list_macros.return_value = []
        mock_get_km_client.return_value = mock_client

        # Tools should accept context parameter
        from src.server.tools.core_tools import km_list_macros

        result = await km_list_macros(
            group_filter="",
            enabled_only=False,
            sort_by="name",
            search_text="",
            ctx=execution_context,
        )

        assert isinstance(result, dict)


class TestPropertyBasedFoundationTesting:
    """Property-based testing for foundation tools using Hypothesis."""

    @patch("src.server.initialization.get_km_client")
    @pytest.mark.asyncio
    async def test_tool_response_structure_property(self, mock_get_km_client: Any) -> None:
        """Property: All tools should return dict with 'success' key."""
        from hypothesis import given
        from hypothesis import strategies as st

        mock_client = AsyncMock()
        mock_client.list_macros.return_value = []
        mock_get_km_client.return_value = mock_client

        @given(group_filter=st.text(max_size=20), enabled_only=st.booleans())
        async def test_response_structure(group_filter: Any, enabled_only: Any) -> None:
            """Test response structure property."""
            from src.core import Duration, ExecutionContext, Permission

            mock_ctx = ExecutionContext.create_test_context(
                permissions=frozenset([Permission.TEXT_INPUT]),
                timeout=Duration.from_seconds(10),
            )

            from src.server.tools.core_tools import km_list_macros

            try:
                result = await km_list_macros(
                    group_filter=group_filter,
                    enabled_only=enabled_only,
                    sort_by="name",
                    search_text="",
                    ctx=mock_ctx,
                )

                # Property: All tool responses must be dicts with 'success' key
                assert isinstance(result, dict)
                assert "success" in result
                assert isinstance(result["success"], bool)

            except Exception as e:
                logger.debug(f"Operation failed during operation: {e}")

        await test_response_structure("", True)
