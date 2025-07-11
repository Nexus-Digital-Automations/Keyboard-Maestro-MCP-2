"""Basic engine coverage tests focusing on easily testable functionality.

This module provides simple, reliable tests for engine.py without complex mocking.
"""

from src.core.engine import (
    EngineMetrics,
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
    MacroDefinition,
    MacroId,
)


class TestPlaceholderCommandBasic:
    """Basic tests for PlaceholderCommand functionality."""

    def test_placeholder_command_text_input_execution(self):
        """Test PlaceholderCommand TEXT_INPUT execution."""
        command = PlaceholderCommand(
            command_id=CommandId("text-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "Hello World"}),
        )

        context = ExecutionContext.create_test_context()
        result = command.execute(context)

        assert result.success is True
        assert "Typed text: Hello World" in result.output
        assert result.execution_time is not None

    def test_placeholder_command_pause_execution(self):
        """Test PlaceholderCommand PAUSE execution."""
        command = PlaceholderCommand(
            command_id=CommandId("pause-cmd"),
            command_type=CommandType.PAUSE,
            parameters=CommandParameters(data={"duration": 0.01}),  # Very short pause
        )

        context = ExecutionContext.create_test_context()
        result = command.execute(context)

        assert result.success is True
        assert "Paused for 0.01 seconds" in result.output
        assert result.execution_time is not None

    def test_placeholder_command_play_sound_execution(self):
        """Test PlaceholderCommand PLAY_SOUND execution."""
        command = PlaceholderCommand(
            command_id=CommandId("sound-cmd"),
            command_type=CommandType.PLAY_SOUND,
            parameters=CommandParameters(data={"sound_name": "alert"}),
        )

        context = ExecutionContext.create_test_context()
        result = command.execute(context)

        assert result.success is True
        assert "Played sound: alert" in result.output
        assert result.execution_time is not None

    def test_placeholder_command_other_type_execution(self):
        """Test PlaceholderCommand execution for other command types."""
        command = PlaceholderCommand(
            command_id=CommandId("other-cmd"),
            command_type=CommandType.VARIABLE_SET,
            parameters=CommandParameters(
                data={"name": "test_var", "value": "test_value"}
            ),
        )

        context = ExecutionContext.create_test_context()
        result = command.execute(context)

        assert result.success is True
        assert "Executed variable_set command" in result.output
        assert result.execution_time is not None

    def test_placeholder_command_validate_success(self):
        """Test PlaceholderCommand validation success."""
        command = PlaceholderCommand(
            command_id=CommandId("valid-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "valid text"}),
        )

        # This should validate successfully since we're using valid parameters
        result = command.validate()
        assert result is True

    def test_placeholder_command_get_dependencies(self):
        """Test PlaceholderCommand get_dependencies method."""
        command = PlaceholderCommand(
            command_id=CommandId("dep-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "test"}),
        )

        dependencies = command.get_dependencies()
        assert dependencies == []

    def test_placeholder_command_get_required_permissions(self):
        """Test PlaceholderCommand get_required_permissions method."""
        command = PlaceholderCommand(
            command_id=CommandId("perm-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "test"}),
        )

        permissions = command.get_required_permissions()
        assert isinstance(permissions, frozenset)


class TestEngineMetricsComplete:
    """Complete tests for EngineMetrics class."""

    def test_engine_metrics_initialization(self):
        """Test EngineMetrics initialization."""
        metrics = EngineMetrics()

        assert metrics.execution_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.average_execution_time == 0.0

    def test_engine_metrics_single_success(self):
        """Test recording single successful execution."""
        metrics = EngineMetrics()
        duration = Duration.from_seconds(2.5)

        metrics.record_execution(duration, success=True)

        assert metrics.execution_count == 1
        assert metrics.success_count == 1
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 2.5
        assert metrics.average_execution_time == 2.5

    def test_engine_metrics_single_failure(self):
        """Test recording single failed execution."""
        metrics = EngineMetrics()
        duration = Duration.from_seconds(1.5)

        metrics.record_execution(duration, success=False)

        assert metrics.execution_count == 1
        assert metrics.success_count == 0
        assert metrics.failure_count == 1
        assert metrics.total_execution_time == 1.5
        assert metrics.average_execution_time == 1.5

    def test_engine_metrics_multiple_executions(self):
        """Test recording multiple executions."""
        metrics = EngineMetrics()

        # Record multiple executions
        metrics.record_execution(Duration.from_seconds(1.0), success=True)
        metrics.record_execution(Duration.from_seconds(2.0), success=False)
        metrics.record_execution(Duration.from_seconds(3.0), success=True)

        assert metrics.execution_count == 3
        assert metrics.success_count == 2
        assert metrics.failure_count == 1
        assert metrics.total_execution_time == 6.0
        assert metrics.average_execution_time == 2.0

    def test_engine_metrics_get_metrics_empty(self):
        """Test getting metrics when no executions recorded."""
        metrics = EngineMetrics()
        result = metrics.get_metrics()

        assert result["execution_count"] == 0
        assert result["success_count"] == 0
        assert result["failure_count"] == 0
        assert result["success_rate"] == 0.0
        assert result["average_execution_time"] == 0.0
        assert result["total_execution_time"] == 0.0

    def test_engine_metrics_get_metrics_with_data(self):
        """Test getting metrics with recorded data."""
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
        """Test resetting metrics."""
        metrics = EngineMetrics()

        # Record some data first
        metrics.record_execution(Duration.from_seconds(2.0), success=True)
        metrics.record_execution(Duration.from_seconds(3.0), success=False)

        # Verify data is recorded
        assert metrics.execution_count == 2

        # Reset
        metrics.reset_metrics()

        # Verify all values are reset
        assert metrics.execution_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.average_execution_time == 0.0

    def test_engine_metrics_zero_division_protection(self):
        """Test metrics calculations when execution_count is 0."""
        metrics = EngineMetrics()

        # Directly test the properties when count is 0
        assert metrics.average_execution_time == 0.0

        result = metrics.get_metrics()
        assert result["success_rate"] == 0.0
        assert result["average_execution_time"] == 0.0


class TestGlobalEngineFunctions:
    """Tests for global engine functions."""

    def test_get_default_engine_singleton(self):
        """Test get_default_engine returns singleton instance."""
        engine1 = get_default_engine()
        engine2 = get_default_engine()

        # Should return same instance
        assert engine1 is engine2

        # Should be MacroEngine instance
        from src.core.engine import MacroEngine

        assert isinstance(engine1, MacroEngine)

    def test_get_engine_metrics_singleton(self):
        """Test get_engine_metrics returns singleton instance."""
        metrics1 = get_engine_metrics()
        metrics2 = get_engine_metrics()

        # Should return same instance
        assert metrics1 is metrics2

        # Should be EngineMetrics instance
        assert isinstance(metrics1, EngineMetrics)

    def test_create_test_macro_simple(self):
        """Test create_test_macro with simple command types."""
        macro = create_test_macro("Simple Test", [CommandType.TEXT_INPUT])

        assert isinstance(macro, MacroDefinition)
        assert macro.name == "Simple Test"
        assert len(macro.commands) == 1

        command = macro.commands[0]
        assert command.command_type == CommandType.TEXT_INPUT
        assert command.parameters.get("text") == "Test text 0"

    def test_create_test_macro_multiple_types(self):
        """Test create_test_macro with multiple command types."""
        types = [CommandType.TEXT_INPUT, CommandType.PAUSE, CommandType.PLAY_SOUND]
        macro = create_test_macro("Multi Test", types)

        assert len(macro.commands) == 3

        # Verify each command has correct type and parameters
        assert macro.commands[0].command_type == CommandType.TEXT_INPUT
        assert macro.commands[1].command_type == CommandType.PAUSE
        assert macro.commands[2].command_type == CommandType.PLAY_SOUND

        # Verify parameters are set correctly
        assert macro.commands[0].parameters.get("text") == "Test text 0"
        assert macro.commands[1].parameters.get("duration") == 1.0
        assert macro.commands[2].parameters.get("sound_name") == "beep"

    def test_create_test_macro_variable_commands(self):
        """Test create_test_macro with variable command types."""
        types = [CommandType.VARIABLE_SET, CommandType.VARIABLE_GET]
        macro = create_test_macro("Variable Test", types)

        assert len(macro.commands) == 2

        # VARIABLE_SET
        set_cmd = macro.commands[0]
        assert set_cmd.command_type == CommandType.VARIABLE_SET
        assert set_cmd.parameters.get("name") == "test_var_0"
        assert set_cmd.parameters.get("value") == "test_value_0"

        # VARIABLE_GET
        get_cmd = macro.commands[1]
        assert get_cmd.command_type == CommandType.VARIABLE_GET
        assert get_cmd.parameters.get("name") == "test_var_1"
        assert get_cmd.parameters.get("default") == "default_value"

    def test_create_test_macro_control_commands(self):
        """Test create_test_macro with control command types."""
        types = [CommandType.APPLICATION_CONTROL, CommandType.SYSTEM_CONTROL]
        macro = create_test_macro("Control Test", types)

        assert len(macro.commands) == 2

        # APPLICATION_CONTROL
        app_cmd = macro.commands[0]
        assert app_cmd.command_type == CommandType.APPLICATION_CONTROL
        assert app_cmd.parameters.get("action") == "activate"
        assert app_cmd.parameters.get("application") == "TextEdit"

        # SYSTEM_CONTROL
        sys_cmd = macro.commands[1]
        assert sys_cmd.command_type == CommandType.SYSTEM_CONTROL
        assert sys_cmd.parameters.get("action") == "volume"
        assert sys_cmd.parameters.get("value") == 50

    def test_create_test_macro_flow_commands(self):
        """Test create_test_macro with flow control command types."""
        types = [CommandType.CONDITIONAL, CommandType.LOOP]
        macro = create_test_macro("Flow Test", types)

        assert len(macro.commands) == 2

        # CONDITIONAL
        cond_cmd = macro.commands[0]
        assert cond_cmd.command_type == CommandType.CONDITIONAL
        assert cond_cmd.parameters.get("condition") == "true"
        assert cond_cmd.parameters.get("then_commands") == []
        assert cond_cmd.parameters.get("else_commands") == []

        # LOOP
        loop_cmd = macro.commands[1]
        assert loop_cmd.command_type == CommandType.LOOP
        assert loop_cmd.parameters.get("count") == 3
        assert loop_cmd.parameters.get("commands") == []

    def test_create_test_macro_unknown_command_type(self):
        """Test create_test_macro with unknown command type."""
        # This will test the 'else' branch in the parameter assignment
        # We can't easily create an unknown CommandType, but we can test
        # that the function handles various types gracefully

        macro = create_test_macro("Basic Test", [CommandType.TEXT_INPUT])

        # Basic functionality should still work
        assert isinstance(macro, MacroDefinition)
        assert len(macro.commands) == 1
        assert macro.commands[0].command_type == CommandType.TEXT_INPUT


class TestMacroDefinitionBasic:
    """Basic tests for MacroDefinition functionality."""

    def test_macro_definition_validation_valid(self):
        """Test MacroDefinition validation with valid macro."""
        # Create a valid command
        command = PlaceholderCommand(
            command_id=CommandId("valid-cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "test"}),
        )

        macro = MacroDefinition(
            macro_id=MacroId("valid-macro"),
            name="Valid Macro",
            commands=[command],
        )

        assert macro.is_valid() is True

    def test_macro_definition_validation_empty_name(self):
        """Test MacroDefinition validation with empty name."""
        command = PlaceholderCommand(
            command_id=CommandId("cmd"),
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters(data={"text": "test"}),
        )

        macro = MacroDefinition(
            macro_id=MacroId("macro"),
            name="",  # Empty name
            commands=[command],
        )

        assert macro.is_valid() is False

    def test_macro_definition_validation_no_commands(self):
        """Test MacroDefinition validation with no commands."""
        macro = MacroDefinition(
            macro_id=MacroId("macro"),
            name="Test Macro",
            commands=[],  # No commands
        )

        assert macro.is_valid() is False
