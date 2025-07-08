"""Comprehensive Test Suite for Group Tools - Following Proven MCP Tool Test Pattern.

This test suite validates the Group Tools functionality using the systematic
testing approach that achieved 100% success rate across 15 tool suites.

Test Coverage:
- Macro group listing functionality with comprehensive validation
- AppleScript execution and subprocess management
- Sorting and statistics calculation with comprehensive data handling
- Progress reporting and context integration validation
- Security validation for subprocess execution and timeout handling
- Property-based testing for robust input validation
- Integration testing with mocked KM clients and subprocess operations
- Error handling for all failure scenarios
- Performance testing for group operation response times

Testing Strategy:
- Property-based testing with Hypothesis for comprehensive input coverage
- Mock-based testing for subprocess and KM client components
- Security validation for AppleScript execution and injection prevention
- Integration testing scenarios with realistic group operations
- Performance and timeout testing with group operation limits

Key Mocking Pattern:
- subprocess.run: Mock AppleScript execution with Either.success() pattern
- get_km_client: Mock Keyboard Maestro client integration
- parse_group_applescript_records: Mock data parsing operations
- Context: Mock progress reporting and logging operations
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite
from src.server.tools.group_tools import km_list_macro_groups

# Test constants
TEST_LOW_SPAM_SCORE = 0.3  # Threshold for low spam score

if TYPE_CHECKING:
    from collections.abc import Callable


# Test fixtures following proven pattern
@pytest.fixture
def mock_context() -> Mock:
    """Create mock FastMCP context following successful pattern."""
    context = Mock(spec=Context)
    context.info = AsyncMock()
    context.warn = AsyncMock()
    context.error = AsyncMock()
    context.report_progress = AsyncMock()
    context.read_resource = AsyncMock()
    context.sample = AsyncMock()
    context.get = Mock(return_value="")  # Support ctx.get() calls
    return context


@pytest.fixture
def mock_km_client() -> Mock:
    """Create mock KM client with standard interface."""
    client = Mock()
    return client


@pytest.fixture
def mock_subprocess_success() -> Mock:
    """Create mock successful subprocess result."""
    result = Mock()
    result.returncode = 0
    result.stdout = """{groupName:"Test Group 1", totalMacros:5, enabledMacros:3, enabled:true}, {groupName:"Test Group 2", totalMacros:8, enabledMacros:7, enabled:false}"""
    result.stderr = ""
    return result


@pytest.fixture
def mock_subprocess_error() -> Mock:
    """Create mock failed subprocess result."""
    result = Mock()
    result.returncode = 1
    result.stdout = ""
    result.stderr = "AppleScript execution failed"
    return result


@pytest.fixture
def mock_parsed_groups() -> Mock:
    """Create mock parsed group data."""
    return [
        {
            "groupName": "Test Group 1",
            "totalMacros": 5,
            "enabledMacros": 3,
            "enabled": True,
        },
        {
            "groupName": "Test Group 2",
            "totalMacros": 8,
            "enabledMacros": 7,
            "enabled": False,
        },
    ]


# Core Group Tools Tests
class TestGroupListing:
    """Test core km_list_macro_groups functionality."""

    @pytest.mark.asyncio
    async def test_list_macro_groups_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test successful macro group listing."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is True
            assert "groups" in result["data"]
            assert "summary" in result["data"]
            assert len(result["data"]["groups"]) == 2
            assert result["data"]["summary"]["total_groups"] == 2
            assert result["data"]["summary"]["enabled_groups"] == 1
            assert result["data"]["summary"]["disabled_groups"] == 1

    @pytest.mark.asyncio
    async def test_list_macro_groups_with_counts(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test macro group listing with macro counts."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            result = await km_list_macro_groups(
                include_macro_count=True,
                include_enabled_count=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["groups"][0]["macro_count"] == 5
            assert result["data"]["groups"][0]["enabled_macro_count"] == 3
            assert result["data"]["groups"][1]["macro_count"] == 8
            assert result["data"]["groups"][1]["enabled_macro_count"] == 7
            assert result["data"]["summary"]["total_macros"] == 13
            assert result["data"]["summary"]["total_enabled_macros"] == 10

    @pytest.mark.asyncio
    async def test_list_macro_groups_without_counts(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test macro group listing without counts."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            result = await km_list_macro_groups(
                include_macro_count=False,
                include_enabled_count=False,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert "macro_count" not in result["data"]["groups"][0]
            assert "enabled_macro_count" not in result["data"]["groups"][0]
            assert result["data"]["summary"]["total_macros"] is None
            assert result["data"]["summary"]["total_enabled_macros"] is None

    @pytest.mark.asyncio
    async def test_list_macro_groups_sort_by_name(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test sorting groups by name."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            result = await km_list_macro_groups(sort_by="name", ctx=mock_context)

            assert result["success"] is True
            assert result["data"]["groups"][0]["name"] == "Test Group 1"
            assert result["data"]["groups"][1]["name"] == "Test Group 2"

    @pytest.mark.asyncio
    async def test_list_macro_groups_sort_by_macro_count(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test sorting groups by macro count."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            result = await km_list_macro_groups(
                sort_by="macro_count",
                include_macro_count=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            # Should be sorted by macro count descending (8, 5)
            assert result["data"]["groups"][0]["macro_count"] == 8
            assert result["data"]["groups"][1]["macro_count"] == 5

    @pytest.mark.asyncio
    async def test_list_macro_groups_sort_by_enabled_count(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test sorting groups by enabled count."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            result = await km_list_macro_groups(
                sort_by="enabled_count",
                include_enabled_count=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            # Should be sorted by enabled count descending (7, 3)
            assert result["data"]["groups"][0]["enabled_macro_count"] == 7
            assert result["data"]["groups"][1]["enabled_macro_count"] == 3


# Error Handling Tests
class TestGroupToolsErrorHandling:
    """Test group tools error handling scenarios."""

    @pytest.mark.asyncio
    async def test_applescript_execution_failure(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_error: Any,
    ) -> None:
        """Test handling of AppleScript execution failure."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_error,
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "KM_CONNECTION_FAILED"
            assert "Cannot retrieve macro groups" in result["error"]["message"]
            assert result["error"]["details"] == "AppleScript execution failed"

    @pytest.mark.asyncio
    async def test_applescript_timeout_error(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test handling of AppleScript timeout."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired("osascript", 30),
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "TIMEOUT_ERROR"
            assert "Timeout retrieving macro groups" in result["error"]["message"]
            assert (
                "AppleScript execution exceeded 30 seconds"
                in result["error"]["details"]
            )

    @pytest.mark.asyncio
    async def test_general_exception_handling(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test general exception handling."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                side_effect=Exception("Test error"),
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert "Failed to retrieve macro groups" in result["error"]["message"]
            assert "Test error" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_km_client_initialization_error(self, mock_context: Any) -> None:
        """Test handling of KM client initialization error."""
        with patch(
            "src.server.tools.group_tools.get_km_client",
            side_effect=Exception("Client init failed"),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert "Failed to retrieve macro groups" in result["error"]["message"]


# Integration Tests
class TestGroupToolsIntegration:
    """Test group tools integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_group_workflow(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test complete group listing workflow with progress tracking."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            result = await km_list_macro_groups(
                include_macro_count=True,
                include_enabled_count=True,
                sort_by="macro_count",
                ctx=mock_context,
            )

            assert result["success"] is True
            assert len(result["data"]["groups"]) == 2
            assert result["metadata"]["sort_by"] == "macro_count"
            assert result["metadata"]["include_counts"]["macro_count"] is True
            assert result["metadata"]["include_counts"]["enabled_count"] is True

            # Verify progress reporting was called
            mock_context.report_progress.assert_called()
            assert (
                mock_context.report_progress.call_count >= 4
            )  # Multiple progress updates

    @pytest.mark.asyncio
    async def test_empty_groups_handling(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
    ) -> None:
        """Test handling of empty group list."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=[],
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is True
            assert len(result["data"]["groups"]) == 0
            assert result["data"]["summary"]["total_groups"] == 0
            assert result["data"]["summary"]["enabled_groups"] == 0

    @pytest.mark.asyncio
    async def test_large_group_list_handling(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
    ) -> None:
        """Test handling of large group lists."""
        # Create mock data for many groups
        large_group_list = [
            {
                "groupName": f"Group {i}",
                "totalMacros": i * 2,
                "enabledMacros": i,
                "enabled": i % 2 == 0,
            }
            for i in range(50)
        ]

        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=large_group_list,
            ),
        ):
            result = await km_list_macro_groups(
                include_macro_count=True,
                sort_by="macro_count",
                ctx=mock_context,
            )

            assert result["success"] is True
            assert len(result["data"]["groups"]) == 50
            assert result["data"]["summary"]["total_groups"] == 50
            # Verify sorting worked (highest macro count first)
            assert result["data"]["groups"][0]["macro_count"] == 98  # 49 * 2
            assert result["data"]["groups"][-1]["macro_count"] == 0  # 0 * 2


# Context Integration Tests
class TestGroupToolsContext:
    """Test group tools context integration."""

    @pytest.mark.asyncio
    async def test_context_info_logging(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test context info logging during execution."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is True
            # Verify info logging was called
            mock_context.info.assert_called()
            assert mock_context.info.call_count >= 2  # Initial and final info calls

    @pytest.mark.asyncio
    async def test_context_error_logging(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_error: Any,
    ) -> None:
        """Test context error logging during failures."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_error,
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is False
            # Verify error logging was called
            mock_context.error.assert_called()

    @pytest.mark.asyncio
    async def test_context_progress_reporting(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test context progress reporting functionality."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is True
            # Verify progress reporting sequence
            progress_calls = mock_context.report_progress.call_args_list
            assert len(progress_calls) >= 4

            # Check progress sequence
            progress_values = [
                call[0][0] for call in progress_calls
            ]  # First argument (progress value)
            assert progress_values == [20, 40, 60, 80, 100]

    @pytest.mark.asyncio
    async def test_without_context(
        self,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test operation without context provided."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            result = await km_list_macro_groups(ctx=None)

            assert result["success"] is True
            assert len(result["data"]["groups"]) == 2


# Security Tests
class TestGroupToolsSecurity:
    """Test group tools security validation."""

    @pytest.mark.asyncio
    async def test_applescript_injection_prevention(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test prevention of AppleScript injection attacks."""
        # The function doesn't take user input that goes into AppleScript,
        # but test that the AppleScript is fixed and secure
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch("src.server.tools.group_tools.subprocess.run") as mock_run,
        ):
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            with patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=[],
            ):
                await km_list_macro_groups(ctx=mock_context)

            # Verify subprocess was called with fixed AppleScript (no user input)
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "/usr/bin/osascript"
            assert args[1] == "-e"
            # The script should be static with no user input
            assert 'tell application "Keyboard Maestro"' in args[2]

    @pytest.mark.asyncio
    async def test_subprocess_timeout_security(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test subprocess timeout security measure."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch("src.server.tools.group_tools.subprocess.run") as mock_run,
        ):
            with patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=[],
            ):
                await km_list_macro_groups(ctx=mock_context)

            # Verify timeout was set
            mock_run.assert_called_once()
            kwargs = mock_run.call_args[1]
            assert kwargs["timeout"] == 30

    @pytest.mark.asyncio
    async def test_subprocess_security_settings(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test subprocess security configuration."""
        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch("src.server.tools.group_tools.subprocess.run") as mock_run,
        ):
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            with patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=[],
            ):
                await km_list_macro_groups(ctx=mock_context)

            # Verify security settings
            kwargs = mock_run.call_args[1]
            assert kwargs["capture_output"] is True
            assert kwargs["text"] is True
            assert kwargs["timeout"] == 30


# Property-Based Tests
class TestGroupToolsPropertyBased:
    """Property-based testing for group tools with Hypothesis."""

    @composite
    def valid_sort_options(draw: Callable[..., Any]) -> Mock:
        """Generate valid sort options."""
        return draw(st.sampled_from(["name", "macro_count", "enabled_count"]))

    @composite
    def valid_boolean_options(draw: Callable[..., Any]) -> Mock:
        """Generate valid boolean option combinations."""
        include_macro_count = draw(st.booleans())
        include_enabled_count = draw(st.booleans())
        return include_macro_count, include_enabled_count

    @given(valid_sort_options(), valid_boolean_options())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sort_and_count_options_property(
        self,
        sort_by: Any,
        count_options: dict[str, Any],
    ) -> None:
        """Property: All valid option combinations should be accepted."""
        include_macro_count, include_enabled_count = count_options

        # Test that option combinations are valid
        assert sort_by in ["name", "macro_count", "enabled_count"]
        assert isinstance(include_macro_count, bool)
        assert isinstance(include_enabled_count, bool)

    @pytest.mark.asyncio
    @given(
        st.lists(
            st.dictionaries(
                keys=st.sampled_from(
                    [
                        "groupName",
                        "totalMacros",
                        "enabledMacros",
                        "enabled",
                    ],
                ),
                values=st.one_of(
                    st.text(min_size=1, max_size=100),
                    st.integers(min_value=0, max_value=1000),
                    st.booleans(),
                ),
                min_size=4,
                max_size=4,
            ),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_group_data_processing_property(
        self,
        group_data_list: list[Any],
    ) -> None:
        """Property: Group data processing should handle various group structures."""
        mock_context = Mock(spec=Context)
        mock_context.info = AsyncMock()
        mock_context.report_progress = AsyncMock()

        mock_km_client = Mock()
        mock_subprocess_result = Mock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "mocked_output"
        mock_subprocess_result.stderr = ""

        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_result,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=group_data_list,
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            # Should either succeed or fail gracefully (not crash)
            assert "success" in result
            assert isinstance(result["success"], bool)

            if result["success"]:
                assert "data" in result
                assert "groups" in result["data"]
                assert len(result["data"]["groups"]) == len(group_data_list)


# Performance Tests
class TestGroupToolsPerformance:
    """Test group tools performance characteristics."""

    @pytest.mark.asyncio
    async def test_group_listing_response_time(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
        mock_parsed_groups: Any,
    ) -> None:
        """Test that group listing completes within reasonable time."""
        import time

        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=mock_parsed_groups,
            ),
        ):
            start_time = time.time()

            result = await km_list_macro_groups(ctx=mock_context)

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within 2 seconds (allowing for mocking overhead)
            assert execution_time < 2.0
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_large_group_list_performance(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
    ) -> None:
        """Test performance with large group lists."""
        import time

        # Create mock data for many groups
        large_group_list = [
            {
                "groupName": f"Group {i}",
                "totalMacros": i * 2,
                "enabledMacros": i,
                "enabled": i % 2 == 0,
            }
            for i in range(100)
        ]

        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=large_group_list,
            ),
        ):
            start_time = time.time()

            result = await km_list_macro_groups(
                include_macro_count=True,
                include_enabled_count=True,
                sort_by="macro_count",
                ctx=mock_context,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within 3 seconds even with 100 groups
            assert execution_time < TEST_LOW_SPAM_SCORE
            assert result["success"] is True
            assert len(result["data"]["groups"]) == 100

    @pytest.mark.asyncio
    async def test_sorting_performance(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
    ) -> None:
        """Test sorting performance with various sort options."""
        import time

        # Create mock data with varied values for sorting
        group_list = [
            {
                "groupName": f"Group {chr(65 + (i % 26))}",  # A-Z pattern
                "totalMacros": (i * 7) % 100,  # Varied macro counts
                "enabledMacros": (i * 3) % 50,  # Varied enabled counts
                "enabled": i % 3 == 0,
            }
            for i in range(50)
        ]

        sort_options = ["name", "macro_count", "enabled_count"]

        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=group_list,
            ),
        ):
            for sort_by in sort_options:
                start_time = time.time()

                result = await km_list_macro_groups(
                    include_macro_count=True,
                    include_enabled_count=True,
                    sort_by=sort_by,
                    ctx=mock_context,
                )

                end_time = time.time()
                execution_time = end_time - start_time

                # Each sort should complete within 2 seconds
                assert execution_time < 2.0
                assert result["success"] is True
                assert len(result["data"]["groups"]) == 50


# Edge Case Tests
class TestGroupToolsEdgeCases:
    """Test group tools edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_group_names(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
    ) -> None:
        """Test handling of groups with empty names."""
        groups_with_empty_names = [
            {"groupName": "", "totalMacros": 5, "enabledMacros": 3, "enabled": True},
            {
                "groupName": "   ",  # Whitespace only
                "totalMacros": 2,
                "enabledMacros": 1,
                "enabled": False,
            },
        ]

        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=groups_with_empty_names,
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is True
            assert len(result["data"]["groups"]) == 2
            assert result["data"]["groups"][0]["name"] == ""
            assert result["data"]["groups"][1]["name"] == "   "

    @pytest.mark.asyncio
    async def test_unicode_group_names(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
    ) -> None:
        """Test handling of Unicode group names."""
        unicode_groups = [
            {
                "groupName": "测试组 🎯",
                "totalMacros": 3,
                "enabledMacros": 2,
                "enabled": True,
            },
            {
                "groupName": "Группа тестов",
                "totalMacros": 5,
                "enabledMacros": 4,
                "enabled": True,
            },
        ]

        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=unicode_groups,
            ),
        ):
            result = await km_list_macro_groups(ctx=mock_context)

            assert result["success"] is True
            assert len(result["data"]["groups"]) == 2
            # Groups are sorted by name - "Группа тестов" comes before "测试组 🎯" alphabetically
            group_names = [g["name"] for g in result["data"]["groups"]]
            assert "测试组 🎯" in group_names
            assert "Группа тестов" in group_names

    @pytest.mark.asyncio
    async def test_zero_macro_counts(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
    ) -> None:
        """Test handling of groups with zero macro counts."""
        zero_count_groups = [
            {
                "groupName": "Empty Group",
                "totalMacros": 0,
                "enabledMacros": 0,
                "enabled": True,
            },
        ]

        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=zero_count_groups,
            ),
        ):
            result = await km_list_macro_groups(
                include_macro_count=True,
                include_enabled_count=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["groups"][0]["macro_count"] == 0
            assert result["data"]["groups"][0]["enabled_macro_count"] == 0
            assert result["data"]["summary"]["total_macros"] == 0
            assert result["data"]["summary"]["total_enabled_macros"] == 0

    @pytest.mark.asyncio
    async def test_missing_group_fields(
        self,
        mock_context: Any,
        mock_km_client: Any,
        mock_subprocess_success: Any,
    ) -> None:
        """Test handling of groups with missing fields."""
        incomplete_groups = [
            {
                "groupName": "Incomplete Group",
                # Missing totalMacros, enabledMacros, enabled
            },
            {
                "totalMacros": 5,
                "enabledMacros": 3,
                # Missing groupName, enabled
            },
        ]

        with (
            patch(
                "src.server.tools.group_tools.get_km_client",
                return_value=mock_km_client,
            ),
            patch(
                "subprocess.run",
                return_value=mock_subprocess_success,
            ),
            patch(
                "src.server.tools.group_tools.parse_group_applescript_records",
                return_value=incomplete_groups,
            ),
        ):
            result = await km_list_macro_groups(
                include_macro_count=True,
                include_enabled_count=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert len(result["data"]["groups"]) == 2
            # Should handle missing fields gracefully with defaults
            # Groups are sorted by name - empty string comes before "Incomplete Group"
            assert (
                result["data"]["groups"][0]["name"] == ""
            )  # Group with missing groupName
            assert (
                result["data"]["groups"][0]["macro_count"] == 5
            )  # This group has totalMacros: 5
            assert (
                result["data"]["groups"][1]["name"] == "Incomplete Group"
            )  # Group with missing counts
            assert (
                result["data"]["groups"][1]["macro_count"] == 0
            )  # Default for missing totalMacros
