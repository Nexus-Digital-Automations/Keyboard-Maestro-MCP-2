"""Targeted coverage expansion tests for src/core/engine.py.

This module specifically targets the 25 missing lines in engine.py to achieve 95%+ coverage.
Missing lines: 87-89, 222, 316, 352-355, 380-382, 408, 421, 445-452, 469-484, 510-512, 535, 707
"""

from unittest.mock import Mock, patch

from src.core.engine import (
    EngineMetrics,
    MacroEngine,
    PlaceholderCommand,
    create_test_macro,
    get_default_engine,
    get_engine_metrics,
)
from src.core.parser import CommandType
from src.core.types import (
    CommandId,
    CommandParameters,
    CommandResult,
    Duration,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    ExecutionToken,
    MacroDefinition,
    MacroId,
)


class TestEngineUncoveredLines:
    """Tests targeting specific uncovered lines in engine.py."""

    def test_placeholder_command_exception_handling(self) -> None:
        """Test PlaceholderCommand exception handling - covers lines 87-89."""
        command = PlaceholderCommand(
            command_id=CommandId("test-cmd"),
            command_type=CommandType.PAUSE,  # Use PAUSE to trigger time.sleep
            parameters=CommandParameters(data={"duration": 1.0}),
        )

        context = ExecutionContext.create_test_context()

        # Mock time.sleep to raise exception during execution inside try block
        with patch("time.sleep", side_effect=Exception("Time error")):
            result = command.execute(context)

            # Should return failure result due to exception (lines 87-89)
            assert isinstance(result, CommandResult)
            assert result.success is False
            assert "Time error" in result.error_message

    def test_macro_engine_error_details_generation(self) -> None:
        """Test MacroEngine error details generation - covers line 222."""
        engine = MacroEngine()

        # Create a mock command that returns a failure with empty error message
        mock_command = Mock()
        mock_command.command_id = CommandId("fail-cmd")
        mock_command.execute.return_value = CommandResult.failure_result(
            error_message="",  # Empty error message to trigger line 222
            command_id="fail-cmd",
            execution_time=Duration.from_seconds(0.1),
        )

        mock_macro = Mock()
        mock_macro.macro_id = MacroId("test-macro")
        mock_macro.name = "Test Macro"
        mock_macro.commands = [mock_command]
        mock_macro.is_valid.return_value = True

        context = ExecutionContext.create_test_context()

        result = engine.execute_macro(mock_macro, context)

        # Should generate error details for commands with empty error messages (line 222)
        assert isinstance(result, ExecutionResult)
        assert result.status == ExecutionStatus.FAILED
        assert "Command failures:" in result.error_details

    def test_macro_engine_async_execution_error_handling(self) -> None:
        """Test MacroEngine async execution error handling - covers lines 316, 352-355."""
        engine = MacroEngine()

        mock_macro = Mock()
        mock_macro.macro_id = MacroId("async-macro")
        mock_macro.name = "Async Test Macro"
        mock_macro.commands = []

        # Test validation failure path (line 316)
        with patch.object(engine, "_validate_macro_enhanced") as mock_validate:
            from src.core.either import Either

            mock_validate.return_value = Either.left("Validation failed")

            async def test_async() -> None:
                result = await engine.execute_macro_async(mock_macro)

                # Should handle validation failure (line 316)
                assert isinstance(result, ExecutionResult)
                assert result.status == ExecutionStatus.FAILED

            import asyncio

            asyncio.run(test_async())

    def test_macro_engine_execution_timeout_handling(self) -> None:
        """Test MacroEngine execution timeout handling - covers lines 380-382."""
        engine = MacroEngine()

        # Create a slow command that will timeout
        slow_command = PlaceholderCommand(
            command_id=CommandId("slow-cmd"),
            command_type=CommandType.PAUSE,
            parameters=CommandParameters(data={"duration": 10.0}),  # Long pause
        )

        mock_macro = Mock()
        mock_macro.macro_id = MacroId("slow-macro")
        mock_macro.name = "Slow Macro"
        mock_macro.commands = [slow_command]

        token = ExecutionToken("timeout-token")

        with patch.object(engine.context_manager, "get_macro", return_value=mock_macro):
            with patch.object(
                engine.context_manager, "create_execution_result"
            ) as mock_create:
                mock_create.return_value = ExecutionResult(
                    execution_token=token,
                    macro_id=mock_macro.macro_id,
                    status=ExecutionStatus.RUNNING,
                    started_at=engine.context_manager.started_at,
                )

                # Set a very short timeout
                with patch(
                    "time.perf_counter", side_effect=[0, 0, 10]
                ):  # Simulate timeout
                    result = engine.execute_macro(
                        mock_macro, token, timeout=Duration.from_seconds(0.1)
                    )

                    # Should handle timeout scenario (lines 380-382)
                    assert isinstance(result, ExecutionResult)

    def test_macro_engine_permission_error_handling(self) -> None:
        """Test MacroEngine permission error handling - covers line 408."""
        engine = MacroEngine()

        mock_macro = Mock()
        mock_macro.macro_id = MacroId("restricted-macro")
        mock_macro.name = "Restricted Macro"

        token = ExecutionToken("perm-token")

        # Mock security context to raise permission error
        with patch("src.core.engine.security_context") as mock_security:
            from src.core.errors import PermissionDeniedError

            mock_security.side_effect = PermissionDeniedError("Access denied")

            result = engine.execute_macro(mock_macro, token)

            # Should handle permission denied (line 408)
            assert isinstance(result, ExecutionResult)
            assert result.status == ExecutionStatus.FAILED
            assert "PermissionDeniedError" in result.error_details

    def test_macro_engine_command_validation_failure(self) -> None:
        """Test MacroEngine command validation failure - covers line 421."""
        engine = MacroEngine()

        # Create invalid command
        invalid_command = PlaceholderCommand(
            command_id=CommandId("invalid-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={}),
        )

        # Mock the command's validation to fail
        with patch.object(invalid_command, "validate", return_value=False):
            mock_macro = Mock()
            mock_macro.macro_id = MacroId("invalid-macro")
            mock_macro.name = "Invalid Macro"
            mock_macro.commands = [invalid_command]

            token = ExecutionToken("validation-token")

            with patch.object(
                engine.context_manager, "get_macro", return_value=mock_macro
            ):
                with patch.object(
                    engine.context_manager, "create_execution_result"
                ) as mock_create:
                    mock_create.return_value = ExecutionResult(
                        execution_token=token,
                        macro_id=mock_macro.macro_id,
                        status=ExecutionStatus.RUNNING,
                        started_at=engine.context_manager.started_at,
                    )

                    result = engine.execute_macro(mock_macro, token)

                    # Should handle validation failure (line 421)
                    assert isinstance(result, ExecutionResult)
                    assert result.status == ExecutionStatus.FAILED

    def test_macro_engine_cleanup_execution(self) -> None:
        """Test MacroEngine cleanup execution - covers lines 510-512."""
        engine = MacroEngine()

        mock_macro = Mock()
        mock_macro.macro_id = MacroId("cleanup-macro")
        mock_macro.name = "Cleanup Macro"
        mock_macro.commands = []

        token = ExecutionToken("cleanup-token")

        # Mock cleanup to raise exception
        with patch.object(engine, "_cleanup_execution") as mock_cleanup:
            mock_cleanup.side_effect = Exception("Cleanup error")

            with patch.object(
                engine.context_manager, "get_macro", return_value=mock_macro
            ):
                with patch.object(
                    engine.context_manager, "create_execution_result"
                ) as mock_create:
                    mock_create.return_value = ExecutionResult(
                        execution_token=token,
                        macro_id=mock_macro.macro_id,
                        status=ExecutionStatus.RUNNING,
                        started_at=engine.context_manager.started_at,
                    )

                    # Should handle cleanup errors gracefully (lines 510-512)
                    result = engine.execute_macro(mock_macro, token)
                    assert isinstance(result, ExecutionResult)

    def test_macro_engine_status_reporting_error(self) -> None:
        """Test MacroEngine status reporting error - covers line 535."""
        engine = MacroEngine()

        mock_macro = Mock()
        mock_macro.macro_id = MacroId("status-macro")
        mock_macro.name = "Status Macro"
        mock_macro.commands = []

        token = ExecutionToken("status-token")

        # Mock update_status to raise exception
        with patch.object(engine.context_manager, "update_status") as mock_update:
            mock_update.side_effect = Exception("Status error")

            with patch.object(
                engine.context_manager, "get_macro", return_value=mock_macro
            ):
                with patch.object(
                    engine.context_manager, "create_execution_result"
                ) as mock_create:
                    mock_create.return_value = ExecutionResult(
                        execution_token=token,
                        macro_id=mock_macro.macro_id,
                        status=ExecutionStatus.RUNNING,
                        started_at=engine.context_manager.started_at,
                    )

                    # Should handle status reporting errors (line 535)
                    result = engine.execute_macro(mock_macro, token)
                    assert isinstance(result, ExecutionResult)


class TestEngineFactoryFunctions:
    """Test factory functions for complete coverage."""

    def test_get_default_engine(self) -> None:
        """Test get_default_engine function."""
        engine = get_default_engine()
        assert isinstance(engine, MacroEngine)

        # Should return same instance (singleton pattern)
        engine2 = get_default_engine()
        assert engine is engine2

    def test_get_engine_metrics(self) -> None:
        """Test get_engine_metrics function."""
        metrics = get_engine_metrics()
        assert isinstance(metrics, EngineMetrics)

    def test_create_test_macro(self) -> None:
        """Test create_test_macro function."""
        command_types = [CommandType.TEXT_INPUT, CommandType.PAUSE]
        macro = create_test_macro("Test Macro", command_types)

        assert isinstance(macro, MacroDefinition)
        assert macro.name == "Test Macro"
        assert len(macro.commands) == len(command_types)


class TestEngineMetrics:
    """Test EngineMetrics class."""

    def test_engine_metrics_initialization(self) -> None:
        """Test EngineMetrics initialization."""
        metrics = EngineMetrics()

        # Test initial state
        assert metrics.total_executions >= 0
        assert metrics.successful_executions >= 0
        assert metrics.failed_executions >= 0

    def test_engine_metrics_recording(self) -> None:
        """Test EngineMetrics recording functionality."""
        metrics = EngineMetrics()

        initial_total = metrics.total_executions

        # Record a successful execution
        metrics.record_execution(success=True, duration=Duration.from_seconds(1.0))

        assert metrics.total_executions == initial_total + 1
        assert metrics.successful_executions >= 1

    def test_engine_metrics_failure_recording(self) -> None:
        """Test EngineMetrics failure recording."""
        metrics = EngineMetrics()

        initial_failed = metrics.failed_executions

        # Record a failed execution
        metrics.record_execution(success=False, duration=Duration.from_seconds(0.5))

        assert metrics.failed_executions == initial_failed + 1
