"""Simple targeted coverage tests for src/core/engine.py.

This module provides focused tests to improve engine.py coverage with simple, reliable test cases.
"""

from unittest.mock import patch

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


class TestPlaceholderCommandCoverage:
    """Tests for PlaceholderCommand to improve coverage."""

    def test_placeholder_command_exception_in_execute(self):
        """Test PlaceholderCommand exception handling in execute method - covers lines 87-89."""
        command = PlaceholderCommand(
            command_id=CommandId("test-cmd"),
            command_type=CommandType.PAUSE,
            parameters=CommandParameters(data={"duration": 1.0}),
        )

        context = ExecutionContext.create_test_context()

        # Mock time.sleep to raise exception
        with patch("time.sleep", side_effect=Exception("Mock error")):
            result = command.execute(context)

            # Should return failure result due to exception
            assert isinstance(result, CommandResult)
            assert result.success is False
            assert "Mock error" in result.error_message
            assert result.execution_time is not None

    def test_placeholder_command_validate_exception(self):
        """Test PlaceholderCommand validation exception handling - covers lines 105-106."""
        command = PlaceholderCommand(
            command_id=CommandId("test-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "test"}),
        )

        # Mock CommandValidator to raise exception
        with patch(
            "src.core.engine.CommandValidator.validate_command_parameters",
            side_effect=Exception("Validation error"),
        ):
            result = command.validate()

            # Should return False when validation raises exception
            assert result is False

    def test_placeholder_command_get_dependencies(self):
        """Test PlaceholderCommand get_dependencies method - covers line 110."""
        command = PlaceholderCommand(
            command_id=CommandId("test-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "test"}),
        )

        dependencies = command.get_dependencies()
        assert dependencies == []


class TestMacroEngineErrorHandling:
    """Tests for MacroEngine error handling paths."""

    def test_macro_engine_invalid_macro_handling(self):
        """Test MacroEngine handling of invalid macro - covers lines 185-194."""
        engine = MacroEngine()

        # Create invalid macro (no commands)
        invalid_macro = MacroDefinition(
            macro_id=MacroId("invalid-macro"),
            name="",  # Empty name makes it invalid
            commands=[],
        )

        context = ExecutionContext.create_test_context()
        result = engine.execute_macro(invalid_macro, context)

        # Should return failed result due to invalid macro
        assert isinstance(result, ExecutionResult)
        assert result.status == ExecutionStatus.FAILED
        assert "ValidationError" in result.error_details

    def test_macro_engine_command_failure_error_details(self):
        """Test MacroEngine error details generation for failed commands - covers lines 215-222."""
        engine = MacroEngine()

        # Create a command that will fail
        failing_command = PlaceholderCommand(
            command_id=CommandId("fail-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "test"}),
        )

        # Mock the command to return failure with empty error message
        with patch.object(failing_command, "execute") as mock_execute:
            mock_execute.return_value = CommandResult.failure_result(
                error_message="",  # Empty error message to trigger line 222
                execution_time=Duration.from_seconds(0.1),
            )

            macro = MacroDefinition(
                macro_id=MacroId("test-macro"),
                name="Test Macro",
                commands=[failing_command],
            )

            context = ExecutionContext.create_test_context()
            result = engine.execute_macro(macro, context)

            # Should generate error details for commands with empty error messages
            assert isinstance(result, ExecutionResult)
            assert result.status == ExecutionStatus.FAILED
            assert "command(s) failed" in result.error_details

    def test_macro_engine_general_exception_handling(self):
        """Test MacroEngine general exception handling - covers lines 239-264."""
        engine = MacroEngine()

        # Create a macro that will cause an exception during execution
        failing_command = PlaceholderCommand(
            command_id=CommandId("exception-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "test"}),
        )

        # Mock _execute_commands to raise exception
        with patch.object(
            engine, "_execute_commands", side_effect=Exception("Execution error")
        ):
            macro = MacroDefinition(
                macro_id=MacroId("exception-macro"),
                name="Exception Macro",
                commands=[failing_command],
            )

            context = ExecutionContext.create_test_context()
            result = engine.execute_macro(macro, context)

            # Should return error result instead of raising
            assert isinstance(result, ExecutionResult)
            assert result.status == ExecutionStatus.FAILED
            assert "Exception: Execution error" in result.error_details
            assert result.command_results == []


class TestEngineUtilityMethods:
    """Tests for MacroEngine utility methods."""

    def test_engine_cancel_execution_success(self):
        """Test successful execution cancellation - covers lines 507-514."""
        engine = MacroEngine()
        token = ExecutionToken("test-token")

        # Mock context manager to return RUNNING status
        with patch.object(
            engine.context_manager, "get_status", return_value=ExecutionStatus.RUNNING
        ):
            with patch.object(engine.context_manager, "update_status") as mock_update:
                with patch.object(engine, "_cleanup_execution") as mock_cleanup:
                    result = engine.cancel_execution(token)

                    assert result is True
                    mock_update.assert_called_once_with(
                        token, ExecutionStatus.CANCELLED
                    )
                    mock_cleanup.assert_called_once_with(token)

    def test_engine_cancel_execution_already_finished(self):
        """Test cancellation of already finished execution - covers remainder of cancel_execution."""
        engine = MacroEngine()
        token = ExecutionToken("finished-token")

        # Mock context manager to return COMPLETED status
        with patch.object(
            engine.context_manager, "get_status", return_value=ExecutionStatus.COMPLETED
        ):
            result = engine.cancel_execution(token)

            assert result is False

    def test_engine_get_active_executions(self):
        """Test getting active executions - covers line 518."""
        engine = MacroEngine()

        with patch.object(
            engine.context_manager,
            "get_active_contexts",
            return_value=["token1", "token2"],
        ):
            active = engine.get_active_executions()

            assert active == ["token1", "token2"]

    def test_engine_cleanup_expired_executions(self):
        """Test cleanup of expired executions - covers lines 588-590."""
        engine = MacroEngine()

        with patch.object(
            engine.context_manager, "cleanup_expired_contexts", return_value=3
        ):
            cleaned = engine.cleanup_expired_executions(max_age_seconds=1800)

            assert cleaned == 3


class TestEngineMetrics:
    """Tests for EngineMetrics class."""

    def test_engine_metrics_initialization(self):
        """Test EngineMetrics initialization - covers lines 596-602."""
        metrics = EngineMetrics()

        assert metrics.execution_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.average_execution_time == 0.0

    def test_engine_metrics_record_success(self):
        """Test recording successful execution - covers lines 604-619."""
        metrics = EngineMetrics()
        duration = Duration.from_seconds(2.5)

        metrics.record_execution(duration, success=True)

        assert metrics.execution_count == 1
        assert metrics.success_count == 1
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 2.5
        assert metrics.average_execution_time == 2.5

    def test_engine_metrics_record_failure(self):
        """Test recording failed execution - covers lines 604-619."""
        metrics = EngineMetrics()
        duration = Duration.from_seconds(1.5)

        metrics.record_execution(duration, success=False)

        assert metrics.execution_count == 1
        assert metrics.success_count == 0
        assert metrics.failure_count == 1
        assert metrics.total_execution_time == 1.5
        assert metrics.average_execution_time == 1.5

    def test_engine_metrics_get_metrics(self):
        """Test getting metrics dictionary - covers lines 621-635."""
        metrics = EngineMetrics()

        # Record some executions
        metrics.record_execution(Duration.from_seconds(2.0), success=True)
        metrics.record_execution(Duration.from_seconds(4.0), success=False)

        result = metrics.get_metrics()

        assert result["execution_count"] == 2
        assert result["success_count"] == 1
        assert result["failure_count"] == 1
        assert result["success_rate"] == 0.5
        assert result["average_execution_time"] == 3.0
        assert result["total_execution_time"] == 6.0

    def test_engine_metrics_reset(self):
        """Test resetting metrics - covers lines 637-644."""
        metrics = EngineMetrics()

        # Record some data first
        metrics.record_execution(Duration.from_seconds(2.0), success=True)

        # Reset
        metrics.reset_metrics()

        assert metrics.execution_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.average_execution_time == 0.0


class TestEngineGlobalFunctions:
    """Tests for global engine functions."""

    def test_get_default_engine(self):
        """Test get_default_engine function - covers lines 652-654."""
        engine = get_default_engine()

        assert isinstance(engine, MacroEngine)

        # Should return same instance (singleton)
        engine2 = get_default_engine()
        assert engine is engine2

    def test_get_engine_metrics(self):
        """Test get_engine_metrics function - covers lines 657-659."""
        metrics = get_engine_metrics()

        assert isinstance(metrics, EngineMetrics)

        # Should return same instance (singleton)
        metrics2 = get_engine_metrics()
        assert metrics is metrics2

    def test_create_test_macro_basic(self):
        """Test create_test_macro function with basic types - covers lines 662-716."""
        macro = create_test_macro(
            "Test Macro", [CommandType.TEXT_INPUT, CommandType.PAUSE]
        )

        assert isinstance(macro, MacroDefinition)
        assert macro.name == "Test Macro"
        assert len(macro.commands) == 2

        # Check first command (TEXT_INPUT)
        text_cmd = macro.commands[0]
        assert isinstance(text_cmd, PlaceholderCommand)
        assert text_cmd.command_type == CommandType.TEXT_INPUT
        assert text_cmd.parameters.get("text") == "Test text 0"
        assert text_cmd.parameters.get("speed") == "normal"

        # Check second command (PAUSE)
        pause_cmd = macro.commands[1]
        assert isinstance(pause_cmd, PlaceholderCommand)
        assert pause_cmd.command_type == CommandType.PAUSE
        assert pause_cmd.parameters.get("duration") == 1.0

    def test_create_test_macro_comprehensive_types(self):
        """Test create_test_macro with various command types - covers more of lines 662-716."""
        command_types = [
            CommandType.PLAY_SOUND,
            CommandType.VARIABLE_SET,
            CommandType.VARIABLE_GET,
            CommandType.APPLICATION_CONTROL,
            CommandType.SYSTEM_CONTROL,
            CommandType.CONDITIONAL,
            CommandType.LOOP,
        ]

        macro = create_test_macro("Comprehensive Test", command_types)

        assert len(macro.commands) == len(command_types)

        # Test each command type has appropriate parameters
        commands = macro.commands

        # PLAY_SOUND
        assert commands[0].parameters.get("sound_name") == "beep"
        assert commands[0].parameters.get("volume") == 50

        # VARIABLE_SET
        assert commands[1].parameters.get("name") == "test_var_1"
        assert commands[1].parameters.get("value") == "test_value_1"

        # VARIABLE_GET
        assert commands[2].parameters.get("name") == "test_var_2"
        assert commands[2].parameters.get("default") == "default_value"

        # APPLICATION_CONTROL
        assert commands[3].parameters.get("action") == "activate"
        assert commands[3].parameters.get("application") == "TextEdit"

        # SYSTEM_CONTROL
        assert commands[4].parameters.get("action") == "volume"
        assert commands[4].parameters.get("value") == 50

        # CONDITIONAL
        assert commands[5].parameters.get("condition") == "true"
        assert commands[5].parameters.get("then_commands") == []

        # LOOP
        assert commands[6].parameters.get("count") == 3
        assert commands[6].parameters.get("commands") == []
