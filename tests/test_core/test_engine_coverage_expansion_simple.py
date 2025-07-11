"""Simple coverage expansion tests for src/core/engine.py.

This module provides targeted tests to improve coverage of engine.py
with simplified test cases that avoid complex mocking scenarios.
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
    Duration,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    ExecutionToken,
    MacroDefinition,
)


class TestEngineSimpleCoverage:
    """Simple tests targeting uncovered functionality."""

    def test_placeholder_command_parameters_coverage(self) -> None:
        """Test PlaceholderCommand with various parameter scenarios."""
        context = ExecutionContext.create_test_context()

        # Test TEXT_INPUT with missing text parameter
        command = PlaceholderCommand(
            command_id=CommandId("test-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={}),  # Missing 'text' parameter
        )
        result = command.execute(context)
        assert result.success
        assert "Typed text:" in result.output

        # Test PAUSE with missing duration parameter
        pause_command = PlaceholderCommand(
            command_id=CommandId("pause-cmd"),
            command_type=CommandType.PAUSE,
            parameters=CommandParameters(data={}),  # Missing 'duration' parameter
        )
        result = pause_command.execute(context)
        assert result.success
        assert "Paused for 1.0" in result.output

        # Test PLAY_SOUND with missing sound_name parameter
        sound_command = PlaceholderCommand(
            command_id=CommandId("sound-cmd"),
            command_type=CommandType.PLAY_SOUND,
            parameters=CommandParameters(data={}),  # Missing 'sound_name' parameter
        )
        result = sound_command.execute(context)
        assert result.success
        assert "Played sound: beep" in result.output

    def test_placeholder_command_different_types(self) -> None:
        """Test PlaceholderCommand with different command types."""
        context = ExecutionContext.create_test_context()

        # Test PAUSE command
        pause_command = PlaceholderCommand(
            command_id=CommandId("pause-cmd"),
            command_type=CommandType.PAUSE,
            parameters=CommandParameters(data={"duration": 0.01}),
        )
        result = pause_command.execute(context)
        assert result.success
        assert "Paused for" in result.output

        # Test PLAY_SOUND command
        sound_command = PlaceholderCommand(
            command_id=CommandId("sound-cmd"),
            command_type=CommandType.PLAY_SOUND,
            parameters=CommandParameters(data={"sound_name": "beep"}),
        )
        result = sound_command.execute(context)
        assert result.success
        assert "Played sound" in result.output

        # Test unknown command type
        unknown_command = PlaceholderCommand(
            command_id=CommandId("unknown-cmd"),
            command_type=CommandType.VARIABLE_SET,  # Not handled in PlaceholderCommand
            parameters=CommandParameters(data={}),
        )
        result = unknown_command.execute(context)
        assert result.success
        assert "Executed" in result.output

    def test_placeholder_command_validation(self) -> None:
        """Test PlaceholderCommand validation logic."""
        # Valid command
        valid_command = PlaceholderCommand(
            command_id=CommandId("valid-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "test"}),
        )
        assert valid_command.validate() is True

        # Test validation failure by mocking CommandValidator
        with patch(
            "src.core.engine.CommandValidator.validate_command_parameters"
        ) as mock_validate:
            mock_validate.side_effect = Exception("Validation error")

            invalid_command = PlaceholderCommand(
                command_id=CommandId("invalid-cmd"),
                command_type=CommandType.TEXT_INPUT,
                parameters=CommandParameters(data={}),
            )
            assert invalid_command.validate() is False

    def test_macro_engine_permission_requirements(self) -> None:
        """Test MacroEngine checking command permission requirements."""
        engine = MacroEngine()

        # Create a macro with a command that has specific permission requirements
        macro = create_test_macro("Test Macro", [CommandType.TEXT_INPUT])

        # Test that get_required_permissions is called
        context = ExecutionContext.create_test_context()

        # This should exercise the permission checking code path
        result = engine.execute_macro(macro, context)

        # Should complete successfully
        assert isinstance(result, ExecutionResult)
        assert result.status == ExecutionStatus.COMPLETED

    def test_macro_engine_cancellation_check(self) -> None:
        """Test execution cancellation check in _execute_commands."""
        engine = MacroEngine()

        # Create a simple command
        command = PlaceholderCommand(
            command_id=CommandId("test-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "test"}),
        )

        context = ExecutionContext.create_test_context()

        # Mock the context manager to return CANCELLED status
        with patch.object(engine.context_manager, "get_status") as mock_get_status:
            mock_get_status.return_value = ExecutionStatus.CANCELLED

            # This should trigger early termination in _execute_commands
            results = engine._execute_commands([command], context)

            # Should return empty results due to cancellation
            assert len(results) == 0

    def test_macro_engine_cleanup_execution(self) -> None:
        """Test _cleanup_execution method."""
        engine = MacroEngine()
        token = ExecutionToken("test-token")

        # Mock the dependencies
        with patch.object(
            engine.context_manager, "cleanup_context"
        ) as mock_cleanup_context:
            with patch("src.core.engine.get_variable_manager") as mock_get_var_manager:
                mock_var_manager = Mock()
                mock_get_var_manager.return_value = mock_var_manager

                engine._cleanup_execution(token)

                # Verify cleanup methods were called
                mock_cleanup_context.assert_called_once_with(token)
                mock_var_manager.cleanup_context_variables.assert_called_once_with(
                    token
                )


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
        """Test create_test_macro function with different command types."""
        command_types = [CommandType.TEXT_INPUT, CommandType.PAUSE]
        macro = create_test_macro("Test Macro", command_types)

        assert isinstance(macro, MacroDefinition)
        assert macro.name == "Test Macro"
        assert len(macro.commands) == len(command_types)

        # Test with more complex command types
        complex_types = [
            CommandType.VARIABLE_SET,
            CommandType.VARIABLE_GET,
            CommandType.APPLICATION_CONTROL,
            CommandType.SYSTEM_CONTROL,
            CommandType.CONDITIONAL,
            CommandType.LOOP,
        ]
        complex_macro = create_test_macro("Complex Macro", complex_types)
        assert len(complex_macro.commands) == len(complex_types)

        # Test with unknown command type to trigger default case
        unknown_types = [
            CommandType.PLAY_SOUND
        ]  # Uses default case in create_test_macro
        unknown_macro = create_test_macro("Unknown Macro", unknown_types)
        assert len(unknown_macro.commands) == 1


class TestEngineMetrics:
    """Test EngineMetrics class."""

    def test_engine_metrics_initialization(self) -> None:
        """Test EngineMetrics initialization."""
        metrics = EngineMetrics()

        # Test initial state
        assert metrics.execution_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.average_execution_time == 0.0

    def test_engine_metrics_recording(self) -> None:
        """Test EngineMetrics recording functionality."""
        metrics = EngineMetrics()

        initial_count = metrics.execution_count

        # Record a successful execution
        metrics.record_execution(Duration.from_seconds(1.0), success=True)

        assert metrics.execution_count == initial_count + 1
        assert metrics.success_count == 1
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 1.0
        assert metrics.average_execution_time == 1.0

    def test_engine_metrics_failure_recording(self) -> None:
        """Test EngineMetrics failure recording."""
        metrics = EngineMetrics()

        # Record a failed execution
        metrics.record_execution(Duration.from_seconds(0.5), success=False)

        assert metrics.execution_count == 1
        assert metrics.success_count == 0
        assert metrics.failure_count == 1
        assert metrics.total_execution_time == 0.5

    def test_engine_metrics_get_metrics(self) -> None:
        """Test EngineMetrics get_metrics method."""
        metrics = EngineMetrics()

        # Record some executions
        metrics.record_execution(Duration.from_seconds(1.0), success=True)
        metrics.record_execution(Duration.from_seconds(2.0), success=False)

        metrics_dict = metrics.get_metrics()

        assert metrics_dict["execution_count"] == 2
        assert metrics_dict["success_count"] == 1
        assert metrics_dict["failure_count"] == 1
        assert metrics_dict["success_rate"] == 0.5
        assert metrics_dict["total_execution_time"] == 3.0
        assert metrics_dict["average_execution_time"] == 1.5

    def test_engine_metrics_reset(self) -> None:
        """Test EngineMetrics reset functionality."""
        metrics = EngineMetrics()

        # Record some data
        metrics.record_execution(Duration.from_seconds(1.0), success=True)
        metrics.record_execution(Duration.from_seconds(2.0), success=False)

        # Reset
        metrics.reset_metrics()

        assert metrics.execution_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.average_execution_time == 0.0

    def test_engine_metrics_empty_success_rate(self) -> None:
        """Test success rate calculation with no executions."""
        metrics = EngineMetrics()
        metrics_dict = metrics.get_metrics()

        # Should handle division by zero gracefully
        assert metrics_dict["success_rate"] == 0.0
        assert metrics_dict["average_execution_time"] == 0.0
