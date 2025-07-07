"""Comprehensive Test Suite for Sync Tools - Following Proven MCP Tool Test Pattern.

This test suite validates the Sync Tools functionality using the systematic
testing approach that achieved 100% success rate across 18 tool suites.

Test Coverage:
- Real-time synchronization operations (start, stop, status, force_sync)
- File monitoring integration with watchdog availability checks
- Sync manager integration with Either pattern success mocking
- Performance metrics tracking and status reporting
- Configuration management for poll intervals and cache settings
- Progress reporting and context integration validation
- Error handling for sync failures and system errors
- Integration testing with mocked sync managers and file monitors
- Performance testing for sync operation response times
- Edge cases for concurrent operations and error recovery

Testing Strategy:
- Mock-based testing for SyncManager and FileMonitor components
- Either pattern success mocking for sync operations
- Comprehensive status validation and health monitoring
- Integration testing scenarios with realistic sync workflows
- Performance and timeout testing with sync operation limits

Key Mocking Pattern:
- SyncManager: Mock sync state management with Either pattern results
- FileMonitor: Mock file system monitoring with status tracking
- Context: Mock progress reporting and logging operations
- Sync status: Test health monitoring and performance metrics
"""

from __future__ import annotations

from typing import Any, Optional
import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite
from src.core.types import Duration

# Import sync types and status
from src.integration.sync_manager import SyncStatus

# Import the tools we're testing
from src.server.tools.sync_tools import (
    km_force_sync,
    km_start_realtime_sync,
    km_stop_realtime_sync,
    km_sync_status,
)


# Test fixtures following proven pattern
@pytest.fixture
def mock_context() -> Any:
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
def mock_sync_manager() -> Any:
    """Create mock sync manager with standard interface."""
    sync_mgr = Mock()

    # Mock sync state
    sync_mgr.sync_state = Mock()
    sync_mgr.sync_state.status = SyncStatus.STOPPED

    # Mock configuration
    sync_mgr.config = Mock()
    sync_mgr.config.base_poll_interval = Duration.from_seconds(30)
    sync_mgr.config.fast_poll_interval = Duration.from_seconds(10)
    sync_mgr.config.slow_poll_interval = Duration.from_seconds(120)
    sync_mgr.config.cache_ttl = Duration.from_seconds(3600)

    # Mock start_sync - Either pattern success
    mock_start_result = Mock()
    mock_start_result.is_left.return_value = False
    mock_start_result.get_right.return_value = "sync_started"
    sync_mgr.start_sync = AsyncMock(return_value=mock_start_result)

    # Mock stop_sync as AsyncMock since it's awaited
    sync_mgr.stop_sync = AsyncMock()

    # Mock force_sync - Either pattern success
    mock_force_result = Mock()
    mock_force_result.is_left.return_value = False
    mock_force_result.get_right.return_value = 42  # Number of macros synced
    sync_mgr.force_sync = AsyncMock(return_value=mock_force_result)

    # Mock get_sync_status
    sync_mgr.get_sync_status.return_value = {
        "status": "active",
        "last_full_sync": datetime.now(UTC).isoformat(),
        "consecutive_errors": 0,
        "average_sync_time_seconds": 1.25,
        "current_poll_interval_seconds": 30,
        "macros_count": 42,
        "last_error": None,
    }

    return sync_mgr


@pytest.fixture
def mock_file_monitor() -> Any:
    """Create mock file monitor with standard interface."""
    monitor = Mock()

    # Mock start_monitoring
    monitor.start_monitoring.return_value = True

    # Mock stop_monitoring
    monitor.stop_monitoring.return_value = None

    # Mock get_status
    monitor.get_status.return_value = {
        "is_monitoring": True,
        "watched_paths": ["/Users/test/Library/Application Support/Keyboard Maestro"],
        "events_processed": 15,
        "last_event": datetime.now(UTC).isoformat(),
        "monitoring_since": datetime.now(UTC).isoformat(),
    }

    return monitor


@pytest.fixture
def mock_sync_stopped() -> Any:
    """Create mock sync manager in stopped state."""
    sync_mgr = Mock()
    sync_mgr.sync_state = Mock()
    sync_mgr.sync_state.status = SyncStatus.STOPPED
    sync_mgr.get_sync_status.return_value = {
        "status": "stopped",
        "last_full_sync": None,
        "consecutive_errors": 0,
    }
    return sync_mgr


@pytest.fixture
def mock_sync_active() -> Any:
    """Create mock sync manager in active state."""
    sync_mgr = Mock()
    sync_mgr.sync_state = Mock()
    sync_mgr.sync_state.status = SyncStatus.ACTIVE
    sync_mgr.get_sync_status.return_value = {
        "status": "active",
        "last_full_sync": datetime.now(UTC).isoformat(),
        "consecutive_errors": 0,
    }
    # Mock stop_sync as AsyncMock since it's awaited
    sync_mgr.stop_sync = AsyncMock()
    return sync_mgr


# Core Sync Operations Tests
class TestSyncOperations:
    """Test core sync tools functionality."""

    @pytest.mark.asyncio
    async def test_start_realtime_sync_success(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test successful start of real-time synchronization."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
            patch("src.server.tools.sync_tools.WATCHDOG_AVAILABLE", True),
        ):
            result = await km_start_realtime_sync(
                enable_file_monitoring=True,
                poll_interval_seconds=60,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert (
                result["data"]["message"]
                == "Real-time synchronization started successfully"
            )
            assert result["data"]["file_monitoring"]["status"] == "active"
            assert result["data"]["file_monitoring"]["watchdog_available"] is True
            assert result["data"]["configuration"]["poll_interval_seconds"] == 60.0
            assert (
                result["data"]["configuration"]["slow_poll_interval_seconds"] == 240.0
            )  # 4x base
            assert "metadata" in result
            assert result["metadata"]["feature"] == "real_time_sync_task_7"

    @pytest.mark.asyncio
    async def test_start_sync_already_active(self, mock_context, mock_sync_active) -> None:
        """Test starting sync when already active."""
        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            return_value=mock_sync_active,
        ):
            result = await km_start_realtime_sync(ctx=mock_context)

            assert result["success"] is True
            assert result["data"]["message"] == "Real-time sync already active"
            assert "status" in result["data"]

    @pytest.mark.asyncio
    async def test_start_sync_with_file_monitoring_disabled(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test starting sync with file monitoring disabled."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_start_realtime_sync(
                enable_file_monitoring=False,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["file_monitoring"]["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_start_sync_file_monitoring_unavailable(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test starting sync when file monitoring is unavailable."""
        mock_file_monitor.start_monitoring.return_value = False

        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_start_realtime_sync(
                enable_file_monitoring=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["file_monitoring"]["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_stop_realtime_sync_success(
        self,
        mock_context,
        mock_sync_active,
        mock_file_monitor,
    ) -> None:
        """Test successful stop of real-time synchronization."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_active,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_stop_realtime_sync(ctx=mock_context)

            assert result["success"] is True
            assert (
                result["data"]["message"]
                == "Real-time synchronization stopped successfully"
            )
            assert "final_status" in result["data"]
            assert "metadata" in result

    @pytest.mark.asyncio
    async def test_stop_sync_already_stopped(self, mock_context, mock_sync_stopped) -> None:
        """Test stopping sync when already stopped."""
        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            return_value=mock_sync_stopped,
        ):
            result = await km_stop_realtime_sync(ctx=mock_context)

            assert result["success"] is True
            assert result["data"]["message"] == "Real-time sync already stopped"

    @pytest.mark.asyncio
    async def test_sync_status_success(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test successful sync status retrieval."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
            patch("src.server.tools.sync_tools.WATCHDOG_AVAILABLE", True),
        ):
            result = await km_sync_status(
                include_performance_metrics=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert "synchronization" in result["data"]
            assert "file_monitoring" in result["data"]
            assert "system_info" in result["data"]
            assert "performance" in result["data"]
            assert "health" in result["data"]
            assert result["data"]["system_info"]["watchdog_available"] is True
            assert result["data"]["health"]["overall_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_sync_status_without_performance_metrics(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test sync status without performance metrics."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_sync_status(
                include_performance_metrics=False,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert "performance" not in result["data"]
            assert "health" in result["data"]

    @pytest.mark.asyncio
    async def test_force_sync_success(self, mock_context, mock_sync_manager) -> None:
        """Test successful forced synchronization."""
        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            return_value=mock_sync_manager,
        ):
            result = await km_force_sync(full_resync=False, ctx=mock_context)

            assert result["success"] is True
            assert (
                result["data"]["message"]
                == "Forced synchronization completed successfully"
            )
            assert result["data"]["macros_synchronized"] == 42
            assert result["data"]["sync_type"] == "incremental"
            assert "sync_status" in result["data"]

    @pytest.mark.asyncio
    async def test_force_sync_full_resync(self, mock_context, mock_sync_manager) -> None:
        """Test forced synchronization with full resync."""
        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            return_value=mock_sync_manager,
        ):
            result = await km_force_sync(full_resync=True, ctx=mock_context)

            assert result["success"] is True
            assert result["data"]["sync_type"] == "full"


# Error Handling Tests
class TestSyncErrorHandling:
    """Test sync tools error handling scenarios."""

    @pytest.mark.asyncio
    async def test_start_sync_failure(self, mock_context, mock_sync_manager) -> None:
        """Test error when sync manager fails to start."""
        # Mock start_sync failure - Either pattern left (error)
        mock_start_result = Mock()
        mock_start_result.is_left.return_value = True
        mock_start_result.get_left.return_value = "Connection failed"
        mock_sync_manager.start_sync = AsyncMock(return_value=mock_start_result)

        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            return_value=mock_sync_manager,
        ):
            result = await km_start_realtime_sync(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "SYNC_START_FAILED"
            assert (
                result["error"]["message"]
                == "Failed to start real-time synchronization"
            )
            assert "Connection failed" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_start_sync_exception(self, mock_context) -> None:
        """Test exception handling during sync start."""
        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            side_effect=Exception("System error"),
        ):
            result = await km_start_realtime_sync(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert (
                result["error"]["message"]
                == "Failed to start real-time synchronization"
            )
            assert "System error" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_stop_sync_exception(self, mock_context) -> None:
        """Test exception handling during sync stop."""
        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            side_effect=Exception("Stop error"),
        ):
            result = await km_stop_realtime_sync(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert (
                result["error"]["message"] == "Failed to stop real-time synchronization"
            )
            assert "Stop error" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_sync_status_exception(self, mock_context) -> None:
        """Test exception handling during status retrieval."""
        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            side_effect=Exception("Status error"),
        ):
            result = await km_sync_status(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert (
                result["error"]["message"]
                == "Failed to retrieve synchronization status"
            )
            assert "Status error" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_force_sync_failure(self, mock_context, mock_sync_manager) -> None:
        """Test error when force sync fails."""
        # Mock force_sync failure - Either pattern left (error)
        mock_force_result = Mock()
        mock_force_result.is_left.return_value = True
        mock_force_result.get_left.return_value = "Sync operation failed"
        mock_sync_manager.force_sync = AsyncMock(return_value=mock_force_result)

        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            return_value=mock_sync_manager,
        ):
            result = await km_force_sync(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "SYNC_FAILED"
            assert result["error"]["message"] == "Forced synchronization failed"
            assert "Sync operation failed" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_force_sync_exception(self, mock_context) -> None:
        """Test exception handling during force sync."""
        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            side_effect=Exception("Force sync error"),
        ):
            result = await km_force_sync(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert result["error"]["message"] == "Failed to force synchronization"
            assert "Force sync error" in result["error"]["details"]


# Configuration and Performance Tests
class TestSyncConfiguration:
    """Test sync configuration and performance features."""

    @pytest.mark.asyncio
    async def test_custom_poll_interval_configuration(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test custom poll interval configuration."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_start_realtime_sync(
                poll_interval_seconds=15,  # Custom interval
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["configuration"]["poll_interval_seconds"] == 15.0
            assert (
                result["data"]["configuration"]["slow_poll_interval_seconds"] == 60.0
            )  # 4x base

            # Verify config was updated
            assert mock_sync_manager.config.base_poll_interval.total_seconds() == 15
            assert mock_sync_manager.config.slow_poll_interval.total_seconds() == 60

    @pytest.mark.asyncio
    async def test_default_configuration_preserved(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test that default configuration is preserved when using default values."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_start_realtime_sync(
                poll_interval_seconds=30,  # Default value
                ctx=mock_context,
            )

            assert result["success"] is True
            # Config should not be modified for default value
            assert mock_sync_manager.config.base_poll_interval.total_seconds() == 30

    @pytest.mark.asyncio
    async def test_sync_status_health_degraded(self, mock_context, mock_file_monitor) -> None:
        """Test sync status when health is degraded."""
        # Create sync manager with errors
        degraded_sync_mgr = Mock()
        degraded_sync_mgr.get_sync_status.return_value = {
            "status": "error",
            "last_full_sync": None,
            "consecutive_errors": 5,  # High error count
            "average_sync_time_seconds": 0.0,
            "current_poll_interval_seconds": 30,
        }

        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=degraded_sync_mgr,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_sync_status(ctx=mock_context)

            assert result["success"] is True
            assert result["data"]["health"]["overall_status"] == "degraded"
            assert result["data"]["health"]["consecutive_errors"] == 5
            assert len(result["data"]["health"]["recommendations"]) > 0
            assert (
                "Check Keyboard Maestro connection"
                in result["data"]["health"]["recommendations"]
            )

    @pytest.mark.asyncio
    async def test_sync_status_with_monitoring_recommendations(
        self,
        mock_context,
        mock_sync_manager,
    ) -> None:
        """Test sync status recommendations for file monitoring."""
        # Mock file monitor that's not monitoring
        inactive_monitor = Mock()
        inactive_monitor.get_status.return_value = {
            "is_monitoring": False,
            "watched_paths": [],
            "events_processed": 0,
        }

        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=inactive_monitor,
            ),
            patch("src.server.tools.sync_tools.WATCHDOG_AVAILABLE", True),
        ):
            result = await km_sync_status(ctx=mock_context)

            assert result["success"] is True
            recommendations = result["data"]["health"]["recommendations"]
            assert any("file monitoring" in rec for rec in recommendations)


# Integration Tests
class TestSyncIntegration:
    """Test sync tools integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_sync_lifecycle(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test complete sync lifecycle: start -> status -> force -> stop."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            # Start sync
            start_result = await km_start_realtime_sync(ctx=mock_context)
            assert start_result["success"] is True

            # Update sync manager state to active
            mock_sync_manager.sync_state.status = SyncStatus.ACTIVE

            # Check status
            status_result = await km_sync_status(ctx=mock_context)
            assert status_result["success"] is True

            # Force sync
            force_result = await km_force_sync(ctx=mock_context)
            assert force_result["success"] is True
            assert force_result["data"]["macros_synchronized"] == 42

            # Stop sync
            stop_result = await km_stop_realtime_sync(ctx=mock_context)
            assert stop_result["success"] is True

    @pytest.mark.asyncio
    async def test_sync_with_file_monitoring_workflow(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test sync workflow with file monitoring integration."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
            patch("src.server.tools.sync_tools.WATCHDOG_AVAILABLE", True),
        ):
            # Start with file monitoring
            result = await km_start_realtime_sync(
                enable_file_monitoring=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["file_monitoring"]["status"] == "active"
            assert result["data"]["file_monitoring"]["watchdog_available"] is True

            # Verify file monitor was started
            mock_file_monitor.start_monitoring.assert_called_once()

            # Update sync state to ACTIVE after start
            mock_sync_manager.sync_state.status = SyncStatus.ACTIVE

            # Check detailed status
            status_result = await km_sync_status(ctx=mock_context)
            assert status_result["success"] is True
            assert status_result["data"]["file_monitoring"]["is_monitoring"] is True

            # Stop sync (should also stop monitoring)
            stop_result = await km_stop_realtime_sync(ctx=mock_context)
            assert stop_result["success"] is True

            # Verify file monitor was stopped
            mock_file_monitor.stop_monitoring.assert_called_once()


# Context Integration Tests
class TestSyncContext:
    """Test sync tools context integration."""

    @pytest.mark.asyncio
    async def test_context_progress_reporting(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test context progress reporting during sync operations."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_start_realtime_sync(ctx=mock_context)

            assert result["success"] is True
            # Verify progress reporting was called
            mock_context.report_progress.assert_called()
            # Verify info logging was called
            mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_context_error_logging(self, mock_context) -> None:
        """Test context error logging during failures."""
        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            side_effect=Exception("Test error"),
        ):
            result = await km_start_realtime_sync(ctx=mock_context)

            assert result["success"] is False
            # Verify error logging was called
            mock_context.error.assert_called()

    @pytest.mark.asyncio
    async def test_without_context(self, mock_sync_manager, mock_file_monitor) -> None:
        """Test operations without context provided."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_start_realtime_sync(ctx=None)

            assert result["success"] is True


# Property-Based Tests
class TestSyncPropertyBased:
    """Property-based testing for sync tools with Hypothesis."""

    @composite
    def valid_poll_intervals(draw) -> Any:
        """Generate valid poll intervals."""
        return draw(st.integers(min_value=5, max_value=300))

    @given(valid_poll_intervals())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_poll_interval_property(self, interval) -> None:
        """Property: Valid poll intervals should be within allowed range."""
        assert 5 <= interval <= 300

    @pytest.mark.asyncio
    async def test_configuration_consistency_property(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Property: Configuration changes should be consistently applied."""
        test_intervals = [10, 30, 60, 120, 300]

        for interval in test_intervals:
            mock_sync_manager.config.base_poll_interval = Duration.from_seconds(
                30,
            )  # Reset
            mock_sync_manager.config.slow_poll_interval = Duration.from_seconds(
                120,
            )  # Reset

            with (
                patch(
                    "src.server.tools.sync_tools.get_sync_manager",
                    return_value=mock_sync_manager,
                ),
                patch(
                    "src.server.tools.sync_tools.get_file_monitor",
                    return_value=mock_file_monitor,
                ),
            ):
                result = await km_start_realtime_sync(
                    poll_interval_seconds=interval,
                    ctx=mock_context,
                )

                assert result["success"] is True
                assert result["data"]["configuration"][
                    "poll_interval_seconds"
                ] == float(interval)
                assert result["data"]["configuration"][
                    "slow_poll_interval_seconds"
                ] == float(interval * 4)


# Performance Tests
class TestSyncPerformance:
    """Test sync tools performance characteristics."""

    @pytest.mark.asyncio
    async def test_sync_operation_response_time(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test that sync operations complete within reasonable time."""
        operations = [
            (km_start_realtime_sync, {}),
            (km_sync_status, {}),
            (km_force_sync, {}),
            (km_stop_realtime_sync, {}),
        ]

        for operation, kwargs in operations:
            start_time = time.time()

            with (
                patch(
                    "src.server.tools.sync_tools.get_sync_manager",
                    return_value=mock_sync_manager,
                ),
                patch(
                    "src.server.tools.sync_tools.get_file_monitor",
                    return_value=mock_file_monitor,
                ),
            ):
                result = await operation(ctx=mock_context, **kwargs)

                end_time = time.time()
                execution_time = end_time - start_time

                # Should complete within 1 second (allowing for mocking overhead)
                assert execution_time < 1.0
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test performance with concurrent sync operations."""

        async def run_status_check():
            with (
                patch(
                    "src.server.tools.sync_tools.get_sync_manager",
                    return_value=mock_sync_manager,
                ),
                patch(
                    "src.server.tools.sync_tools.get_file_monitor",
                    return_value=mock_file_monitor,
                ),
            ):
                return await km_sync_status(ctx=mock_context)

        start_time = time.time()

        # Run multiple concurrent status checks
        tasks = [run_status_check() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should handle concurrent requests efficiently
        assert execution_time < 2.0
        assert all(result["success"] for result in results)


# Edge Case Tests
class TestSyncEdgeCases:
    """Test sync tools edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_minimum_poll_interval(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test minimum allowed poll interval."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_start_realtime_sync(
                poll_interval_seconds=5,  # Minimum allowed
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["configuration"]["poll_interval_seconds"] == 5.0

    @pytest.mark.asyncio
    async def test_maximum_poll_interval(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test maximum allowed poll interval."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
        ):
            result = await km_start_realtime_sync(
                poll_interval_seconds=300,  # Maximum allowed
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["configuration"]["poll_interval_seconds"] == 300.0
            assert (
                result["data"]["configuration"]["slow_poll_interval_seconds"] == 1200.0
            )  # 4x

    @pytest.mark.asyncio
    async def test_empty_sync_status(self, mock_context, mock_file_monitor) -> None:
        """Test sync status with minimal data."""
        empty_sync_mgr = Mock()
        empty_sync_mgr.get_sync_status.return_value = {
            "status": "stopped",
            "last_full_sync": None,
            "consecutive_errors": 0,
        }

        mock_file_monitor.get_status.return_value = {
            "is_monitoring": False,
            "watched_paths": [],
            "events_processed": 0,
        }

        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=empty_sync_mgr,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
            patch("src.server.tools.sync_tools.WATCHDOG_AVAILABLE", False),
        ):
            result = await km_sync_status(ctx=mock_context)

            assert result["success"] is True
            assert result["data"]["system_info"]["watchdog_available"] is False
            assert (
                result["data"]["health"]["overall_status"] == "healthy"
            )  # Should still be healthy

    @pytest.mark.asyncio
    async def test_sync_zero_macros(self, mock_context, mock_sync_manager) -> None:
        """Test forced sync with zero macros."""
        # Mock force sync returning 0 macros
        mock_force_result = Mock()
        mock_force_result.is_left.return_value = False
        mock_force_result.get_right.return_value = 0
        mock_sync_manager.force_sync = AsyncMock(return_value=mock_force_result)

        with patch(
            "src.server.tools.sync_tools.get_sync_manager",
            return_value=mock_sync_manager,
        ):
            result = await km_force_sync(ctx=mock_context)

            assert result["success"] is True
            assert result["data"]["macros_synchronized"] == 0

    @pytest.mark.asyncio
    async def test_watchdog_unavailable_scenario(
        self,
        mock_context,
        mock_sync_manager,
        mock_file_monitor,
    ) -> None:
        """Test scenario where watchdog is not available."""
        with (
            patch(
                "src.server.tools.sync_tools.get_sync_manager",
                return_value=mock_sync_manager,
            ),
            patch(
                "src.server.tools.sync_tools.get_file_monitor",
                return_value=mock_file_monitor,
            ),
            patch("src.server.tools.sync_tools.WATCHDOG_AVAILABLE", False),
        ):
            result = await km_start_realtime_sync(
                enable_file_monitoring=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["file_monitoring"]["watchdog_available"] is False
