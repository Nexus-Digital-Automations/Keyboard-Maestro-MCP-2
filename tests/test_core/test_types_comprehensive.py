"""Comprehensive tests for core types and data structures.

import logging

logging.basicConfig(level=logging.DEBUG)
Tests cover branded types, enums, dataclasses, protocols, and type safety
with property-based testing and comprehensive validation.
"""

import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, cast
from unittest.mock import Mock

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from src.core.types import (
    CommandId,
    CommandParameters,
    CommandResult,
    Duration,
    ExecutionContext,
    ExecutionResult,
    # Enums
    ExecutionStatus,
    ExecutionToken,
    GroupId,
    # Protocol
    MacroCommand,
    MacroCreationStatus,
    MacroDefinition,
    # Branded Types
    MacroId,
    MacroMoveResult,
    MoveConflictType,
    Permission,
    Priority,
    Result,
    UserId,
    ValidationResult,
    VariableName,
    create_error_result,
    create_macro_id,
    create_success_result,
)


# Test data generators
@st.composite
def duration_strategy(draw: Callable[..., Any]) -> float:
    """Generate valid duration values."""
    return cast("float", draw(st.floats(min_value=0.0, max_value=3600.0)))


@st.composite
def command_parameters_strategy(draw: Callable[..., Any]) -> dict[str, Any]:
    """Generate valid command parameters."""
    params = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(st.text(max_size=100), st.integers(), st.floats(), st.booleans()),
            max_size=10,
        ),
    )
    return cast("dict[str, Any]", params)


@st.composite
def permission_set_strategy(draw: Callable[..., Any]) -> frozenset[Permission]:
    """Generate valid permission sets."""
    permissions = draw(
        st.lists(st.sampled_from(list(Permission)), min_size=0, max_size=5),
    )
    return frozenset(permissions)


@st.composite
def variable_dict_strategy(draw: Callable[..., Any]) -> dict[VariableName, str]:
    """Generate valid variable dictionaries."""
    return cast(
        "dict[VariableName, str]",
        draw(
            st.dictionaries(
                st.text(min_size=1, max_size=50).map(VariableName),
                st.text(max_size=100),
                max_size=10,
            ),
        ),
    )


class TestBrandedTypes:
    """Test branded type functionality and type safety."""

    def test_macro_id_creation(self) -> None:
        """Test MacroId creation and type safety."""
        macro_id = MacroId("test_macro_123")
        assert isinstance(macro_id, str)
        assert macro_id == "test_macro_123"

        # Test with UUID
        uuid_str = str(uuid.uuid4())
        macro_id_uuid = MacroId(uuid_str)
        assert macro_id_uuid == uuid_str

    def test_command_id_creation(self) -> None:
        """Test CommandId creation."""
        command_id = CommandId("cmd_123")
        assert isinstance(command_id, str)
        assert command_id == "cmd_123"

    def test_execution_token_creation(self) -> None:
        """Test ExecutionToken creation."""
        import secrets

        token_value = f"exec_token_{secrets.token_hex(8)}"
        token = ExecutionToken(token_value)
        assert isinstance(token, str)
        assert token == token_value  # S105 fix: Use secure random token

    def test_user_id_creation(self) -> None:
        """Test UserId creation."""
        user_id = UserId("user_789")
        assert isinstance(user_id, str)
        assert user_id == "user_789"

    def test_create_macro_id_function(self) -> None:
        """Test create_macro_id utility function."""
        macro_id = create_macro_id()

        assert isinstance(macro_id, str)  # MacroId is a NewType of str
        assert len(macro_id) > 0

        # Should be UUID format
        uuid.UUID(macro_id)  # Will raise if not valid UUID

        # Each call should produce unique ID
        macro_id2 = create_macro_id()
        assert macro_id != macro_id2

    @given(st.text(min_size=1, max_size=100))
    def test_branded_types_property_validation(self, text_value: str) -> None:
        """Property test for branded type creation."""
        # Test various branded types with same underlying string
        macro_id = MacroId(text_value)
        command_id = CommandId(text_value)
        token = ExecutionToken(text_value)

        # All should preserve the original value
        assert str(macro_id) == text_value
        assert str(command_id) == text_value
        assert str(token) == text_value

        # But they should be distinct types (at runtime they're all str)
        assert isinstance(macro_id, str)
        assert isinstance(command_id, str)
        assert isinstance(token, str)


class TestEnumerations:
    """Test enum definitions and values."""

    def test_execution_status_enum(self) -> None:
        """Test ExecutionStatus enum values."""
        expected_statuses = {
            ExecutionStatus.PENDING: "pending",
            ExecutionStatus.RUNNING: "running",
            ExecutionStatus.COMPLETED: "completed",
            ExecutionStatus.FAILED: "failed",
            ExecutionStatus.CANCELLED: "cancelled",
            ExecutionStatus.TIMEOUT: "timeout",
        }

        for status, expected_value in expected_statuses.items():
            assert status.value == expected_value

        # Test all values are present
        assert len(list(ExecutionStatus)) == 6

    def test_macro_creation_status_enum(self) -> None:
        """Test MacroCreationStatus enum values."""
        expected_statuses = {
            MacroCreationStatus.VALIDATING: "validating",
            MacroCreationStatus.CREATING: "creating",
            MacroCreationStatus.COMPLETED: "completed",
            MacroCreationStatus.FAILED: "failed",
            MacroCreationStatus.ROLLED_BACK: "rolled_back",
        }

        for status, expected_value in expected_statuses.items():
            assert status.value == expected_value

        assert len(list(MacroCreationStatus)) == 5

    def test_priority_enum(self) -> None:
        """Test Priority enum values."""
        expected_priorities = {
            Priority.LOW: "low",
            Priority.MEDIUM: "medium",
            Priority.HIGH: "high",
            Priority.CRITICAL: "critical",
        }

        for priority, expected_value in expected_priorities.items():
            assert priority.value == expected_value

        assert len(list(Priority)) == 4

    def test_permission_enum(self) -> None:
        """Test Permission enum comprehensiveness."""
        expected_permissions = [
            "text_input",
            "system_control",
            "file_access",
            "application_control",
            "network_access",
            "clipboard_access",
            "clipboard_history",
            "clipboard_named",
            "system_sound",
            "screen_capture",
            "audio_output",
            "window_management",
            "flow_control",
            "mouse_control",
            "automation_control",
            "read_access",
            "admin_access",
        ]

        actual_permissions = [p.value for p in Permission]

        for expected_perm in expected_permissions:
            assert expected_perm in actual_permissions

        # Test specific permissions
        assert Permission.TEXT_INPUT.value == "text_input"
        assert Permission.AUTOMATION_CONTROL.value == "automation_control"
        assert Permission.ADMIN_ACCESS.value == "admin_access"

    def test_move_conflict_type_enum(self) -> None:
        """Test MoveConflictType enum values."""
        expected_conflicts = {
            MoveConflictType.NAME_COLLISION: "name_collision",
            MoveConflictType.PERMISSION_DENIED: "permission_denied",
            MoveConflictType.GROUP_NOT_FOUND: "group_not_found",
            MoveConflictType.MACRO_NOT_FOUND: "macro_not_found",
            MoveConflictType.MACRO_ENABLED_IN_SOURCE: "macro_enabled_in_source",
            MoveConflictType.TARGET_GROUP_DISABLED: "target_group_disabled",
        }

        for conflict, expected_value in expected_conflicts.items():
            assert conflict.value == expected_value

        assert len(list(MoveConflictType)) == 6


class TestDuration:
    """Test Duration class functionality."""

    def test_duration_creation(self) -> None:
        """Test Duration creation with valid values."""
        duration = Duration(5.0)
        assert duration.seconds == 5.0
        assert duration.total_seconds() == 5.0

    def test_duration_negative_validation(self) -> None:
        """Test Duration validation for negative values."""
        with pytest.raises(ValueError, match="Duration cannot be negative"):
            Duration(-1.0)

    def test_duration_from_seconds(self) -> None:
        """Test Duration.from_seconds class method."""
        duration = Duration.from_seconds(10.5)
        assert duration.seconds == 10.5
        assert duration.total_seconds() == 10.5

    def test_duration_from_milliseconds(self) -> None:
        """Test Duration.from_milliseconds class method."""
        duration = Duration.from_milliseconds(1500)
        assert duration.seconds == 1.5
        assert duration.total_seconds() == 1.5

    def test_duration_arithmetic_operations(self) -> None:
        """Test Duration arithmetic operations."""
        d1 = Duration(5.0)
        d2 = Duration(3.0)

        # Addition
        d3 = d1 + d2
        assert d3.seconds == 8.0

        # Addition should return new instance
        assert d1.seconds == 5.0
        assert d2.seconds == 3.0

    def test_duration_comparison_operations(self) -> None:
        """Test Duration comparison operations."""
        d1 = Duration(5.0)
        d2 = Duration(3.0)
        d3 = Duration(5.0)

        # Less than
        assert d2 < d1
        assert not (d1 < d2)

        # Less than or equal
        assert d2 <= d1
        assert d1 <= d3
        assert not (d1 <= d2)

        # Greater than
        assert d1 > d2
        assert not (d2 > d1)

        # Greater than or equal
        assert d1 >= d2
        assert d1 >= d3
        assert not (d2 >= d1)

        # Equality
        assert d1 == d3
        assert not (d1 == d2)
        assert d1 != d2

        # Equality with non-Duration
        assert d1 != "5.0"
        assert d1 != 5.0

    def test_duration_zero_constant(self) -> None:
        """Test Duration.ZERO constant."""
        zero = cast("Duration", Duration.ZERO)
        assert zero.seconds == 0.0
        assert zero.total_seconds() == 0.0

        # Should be immutable
        assert zero == Duration(0.0)

    @given(duration_strategy())
    def test_duration_property_validation(self, seconds: float) -> None:
        """Property test for Duration behavior."""
        assume(seconds >= 0.0)

        duration = Duration(seconds)

        # Properties that should always hold
        assert duration.seconds == seconds
        assert duration.total_seconds() == seconds
        assert duration >= cast("Duration", Duration.ZERO)

        # Test with from_seconds
        duration2 = Duration.from_seconds(seconds)
        assert duration == duration2

        # Test with from_milliseconds
        milliseconds = int(seconds * 1000)
        duration3 = Duration.from_milliseconds(milliseconds)
        assert abs(duration3.seconds - seconds) < 0.001  # Allow for float precision


class TestCommandParameters:
    """Test CommandParameters functionality."""

    def test_command_parameters_creation(self) -> None:
        """Test CommandParameters creation."""
        params = CommandParameters({"key1": "value1", "key2": 42})

        assert params.get("key1") == "value1"
        assert params.get("key2") == 42
        assert params.get("nonexistent") is None
        assert params.get("nonexistent", "default") == "default"

    def test_command_parameters_empty(self) -> None:
        """Test CommandParameters.empty() class method."""
        params = CommandParameters.empty()

        assert isinstance(params.data, dict)
        assert len(params.data) == 0
        assert params.get("anything") is None

    def test_command_parameters_with_parameter(self) -> None:
        """Test immutable parameter addition."""
        original = CommandParameters({"existing": "value"})
        updated = original.with_parameter("new_key", "new_value")

        # Original should be unchanged
        assert original.get("new_key") is None
        assert original.get("existing") == "value"

        # Updated should have both
        assert updated.get("existing") == "value"
        assert updated.get("new_key") == "new_value"

    def test_command_parameters_with_parameter_override(self) -> None:
        """Test parameter override with with_parameter."""
        original = CommandParameters({"key": "old_value"})
        updated = original.with_parameter("key", "new_value")

        assert original.get("key") == "old_value"
        assert updated.get("key") == "new_value"

    @given(command_parameters_strategy())
    def test_command_parameters_property_validation(
        self,
        param_dict: dict[str, Any],
    ) -> None:
        """Property test for CommandParameters behavior."""
        import math

        params = CommandParameters(param_dict)

        # All original keys should be accessible
        for key, value in param_dict.items():
            retrieved = params.get(key)
            # Handle NaN comparison specially
            if isinstance(value, float) and math.isnan(value):
                assert isinstance(retrieved, float) and math.isnan(retrieved)
            else:
                assert retrieved == value

        # Non-existent keys should return None or default
        assert params.get("__nonexistent__") is None
        assert params.get("__nonexistent__", "default") == "default"

        # with_parameter should create new instance
        if param_dict:
            first_key = next(iter(param_dict.keys()))
            updated = params.with_parameter(first_key, "updated_value")

            # Handle NaN comparison for original value
            original_value = param_dict[first_key]
            retrieved_original = params.get(first_key)
            if isinstance(original_value, float) and math.isnan(original_value):
                assert isinstance(retrieved_original, float) and math.isnan(
                    retrieved_original,
                )
            else:
                assert retrieved_original == original_value  # Original unchanged

            assert updated.get(first_key) == "updated_value"  # Updated changed


class TestExecutionContext:
    """Test ExecutionContext functionality."""

    def test_execution_context_creation(self) -> None:
        """Test ExecutionContext creation."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_SOUND])
        timeout = Duration.from_seconds(30)
        variables = {VariableName("var1"): "value1"}

        context = ExecutionContext(
            permissions=permissions,
            timeout=timeout,
            variables=variables,
        )

        assert context.permissions == permissions
        assert context.timeout == timeout
        assert context.variables == variables
        assert isinstance(context.execution_id, str)
        assert isinstance(context.created_at, datetime)

    def test_execution_context_permission_checking(self) -> None:
        """Test ExecutionContext permission checking."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.CLIPBOARD_ACCESS])
        context = ExecutionContext(
            permissions=permissions,
            timeout=Duration.from_seconds(30),
        )

        # Has permission
        assert context.has_permission(Permission.TEXT_INPUT)
        assert context.has_permission(Permission.CLIPBOARD_ACCESS)

        # Doesn't have permission
        assert not context.has_permission(Permission.ADMIN_ACCESS)
        assert not context.has_permission(Permission.SYSTEM_CONTROL)

        # Has all permissions
        assert context.has_permissions(frozenset([Permission.TEXT_INPUT]))
        assert context.has_permissions(
            frozenset([Permission.TEXT_INPUT, Permission.CLIPBOARD_ACCESS]),
        )

        # Doesn't have all permissions
        assert not context.has_permissions(
            frozenset([Permission.TEXT_INPUT, Permission.ADMIN_ACCESS]),
        )

    def test_execution_context_variable_operations(self) -> None:
        """Test ExecutionContext variable operations."""
        variables = {VariableName("var1"): "value1", VariableName("var2"): "value2"}
        context = ExecutionContext(
            permissions=frozenset([Permission.TEXT_INPUT]),
            timeout=Duration.from_seconds(30),
            variables=variables,
        )

        # Get existing variable
        assert context.get_variable(VariableName("var1")) == "value1"
        assert context.get_variable(VariableName("var2")) == "value2"

        # Get non-existent variable
        assert context.get_variable(VariableName("nonexistent")) is None

    def test_execution_context_with_variable(self) -> None:
        """Test ExecutionContext immutable variable addition."""
        original_context = ExecutionContext(
            permissions=frozenset([Permission.TEXT_INPUT]),
            timeout=Duration.from_seconds(30),
            variables={VariableName("existing"): "value"},
        )

        # Add new variable
        updated_context = original_context.with_variable(
            VariableName("new_var"),
            "new_value",
        )

        # Original context unchanged
        assert original_context.get_variable(VariableName("new_var")) is None
        assert original_context.get_variable(VariableName("existing")) == "value"

        # Updated context has both
        assert updated_context.get_variable(VariableName("existing")) == "value"
        assert updated_context.get_variable(VariableName("new_var")) == "new_value"

        # Other fields should be preserved
        assert updated_context.permissions == original_context.permissions
        assert updated_context.timeout == original_context.timeout
        assert updated_context.execution_id == original_context.execution_id

    def test_execution_context_create_test_context(self) -> None:
        """Test ExecutionContext.create_test_context() class method."""
        # Default test context
        context = ExecutionContext.create_test_context()

        assert Permission.TEXT_INPUT in context.permissions
        assert Permission.SYSTEM_SOUND in context.permissions
        assert context.timeout.seconds == 30

        # Custom test context
        custom_permissions = frozenset([Permission.ADMIN_ACCESS])
        custom_timeout = Duration.from_seconds(60)

        custom_context = ExecutionContext.create_test_context(
            permissions=custom_permissions,
            timeout=custom_timeout,
        )

        assert custom_context.permissions == custom_permissions
        assert custom_context.timeout == custom_timeout

    def test_execution_context_default(self) -> None:
        """Test ExecutionContext.default() class method."""
        context = ExecutionContext.default()

        assert Permission.TEXT_INPUT in context.permissions
        assert Permission.SYSTEM_SOUND in context.permissions
        assert context.timeout.seconds == 30
        assert isinstance(context.execution_id, str)
        assert isinstance(context.created_at, datetime)

    @given(permission_set_strategy(), duration_strategy(), variable_dict_strategy())
    def test_execution_context_property_validation(
        self,
        permissions: frozenset,
        timeout_seconds: float,
        variables: dict,
    ) -> None:
        """Property test for ExecutionContext behavior."""
        timeout = Duration.from_seconds(timeout_seconds)

        context = ExecutionContext(
            permissions=permissions,
            timeout=timeout,
            variables=variables,
        )

        # Properties that should always hold
        assert context.permissions == permissions
        assert context.timeout == timeout
        assert context.variables == variables
        assert isinstance(context.execution_id, str)
        assert isinstance(context.created_at, datetime)

        # Permission checking should work
        for permission in permissions:
            assert context.has_permission(permission)

        # Variable access should work
        for var_name, var_value in variables.items():
            assert context.get_variable(var_name) == var_value


class TestCommandResult:
    """Test CommandResult functionality."""

    def test_command_result_success_creation(self) -> None:
        """Test successful CommandResult creation."""
        result = CommandResult(
            success=True,
            output="Operation completed",
            execution_time=Duration.from_seconds(1.5),
            metadata={"command_id": "cmd_123"},
        )

        assert result.success
        assert result.output == "Operation completed"
        assert result.error_message is None
        assert result.execution_time is not None
        assert result.execution_time.seconds == 1.5
        assert result.metadata["command_id"] == "cmd_123"

    def test_command_result_failure_creation(self) -> None:
        """Test failed CommandResult creation."""
        result = CommandResult(
            success=False,
            error_message="Operation failed",
            execution_time=Duration.from_seconds(0.5),
        )

        assert not result.success
        assert result.output is None
        assert result.error_message == "Operation failed"
        assert result.execution_time is not None
        assert result.execution_time.seconds == 0.5

    def test_command_result_success_result_factory(self) -> None:
        """Test CommandResult.success_result() factory method."""
        result = CommandResult.success_result(
            output="Success message",
            execution_time=Duration.from_seconds(2.0),
            command_id="test_cmd",
            extra_data="additional",
        )

        assert result.success
        assert result.output == "Success message"
        assert result.error_message is None
        assert result.execution_time is not None
        assert result.execution_time.seconds == 2.0
        assert result.metadata["command_id"] == "test_cmd"
        assert result.metadata["extra_data"] == "additional"

    def test_command_result_failure_result_factory(self) -> None:
        """Test CommandResult.failure_result() factory method."""
        result = CommandResult.failure_result(
            error_message="Command execution failed",
            execution_time=Duration.from_seconds(0.1),
            error_code=500,
            component="parser",
        )

        assert not result.success
        assert result.output is None
        assert result.error_message == "Command execution failed"
        assert result.execution_time is not None
        assert result.execution_time.seconds == 0.1
        assert result.metadata["error_code"] == 500
        assert result.metadata["component"] == "parser"

    def test_command_result_minimal_creation(self) -> None:
        """Test CommandResult with minimal parameters."""
        success_result = CommandResult(success=True)
        assert success_result.success
        assert success_result.output is None
        assert success_result.error_message is None
        assert success_result.execution_time is None
        assert success_result.metadata == {}

        failure_result = CommandResult(success=False)
        assert not failure_result.success


class TestMacroDefinition:
    """Test MacroDefinition functionality."""

    def test_macro_definition_creation(self) -> None:
        """Test MacroDefinition creation."""
        # Create mock commands
        mock_command1 = Mock(spec=MacroCommand)
        mock_command1.validate.return_value = True
        mock_command2 = Mock(spec=MacroCommand)
        mock_command2.validate.return_value = True

        macro_id = MacroId("test_macro")
        group_id = GroupId("test_group")

        macro = MacroDefinition(
            macro_id=macro_id,
            name="Test Macro",
            commands=[mock_command1, mock_command2],
            enabled=True,
            group_id=group_id,
            description="A test macro",
        )

        assert macro.macro_id == macro_id
        assert macro.name == "Test Macro"
        assert len(macro.commands) == 2
        assert macro.enabled
        assert macro.group_id == group_id
        assert macro.description == "A test macro"
        assert isinstance(macro.created_at, datetime)

    def test_macro_definition_is_valid_success(self) -> None:
        """Test MacroDefinition.is_valid() with valid macro."""
        mock_command = Mock(spec=MacroCommand)
        mock_command.validate.return_value = True

        macro = MacroDefinition(
            macro_id=MacroId("valid_macro"),
            name="Valid Macro",
            commands=[mock_command],
        )

        assert macro.is_valid()
        mock_command.validate.assert_called_once()

    def test_macro_definition_is_valid_empty_name(self) -> None:
        """Test MacroDefinition.is_valid() with empty name."""
        mock_command = Mock(spec=MacroCommand)

        macro = MacroDefinition(
            macro_id=MacroId("invalid_macro"),
            name="",
            commands=[mock_command],
        )

        assert not macro.is_valid()

    def test_macro_definition_is_valid_no_commands(self) -> None:
        """Test MacroDefinition.is_valid() with no commands."""
        macro = MacroDefinition(
            macro_id=MacroId("empty_macro"),
            name="Empty Macro",
            commands=[],
        )

        assert not macro.is_valid()

    def test_macro_definition_is_valid_invalid_command(self) -> None:
        """Test MacroDefinition.is_valid() with invalid command."""
        mock_command1 = Mock(spec=MacroCommand)
        mock_command1.validate.return_value = True
        mock_command2 = Mock(spec=MacroCommand)
        mock_command2.validate.return_value = False

        macro = MacroDefinition(
            macro_id=MacroId("mixed_macro"),
            name="Mixed Macro",
            commands=[mock_command1, mock_command2],
        )

        assert not macro.is_valid()

    def test_macro_definition_create_test_macro(self) -> None:
        """Test MacroDefinition.create_test_macro() factory method."""
        mock_command = Mock(spec=MacroCommand)

        macro = MacroDefinition.create_test_macro("Test Factory", [mock_command])

        assert macro.name == "Test Factory"
        assert len(macro.commands) == 1
        assert macro.commands[0] == mock_command
        assert isinstance(macro.macro_id, str)  # MacroId is NewType of str
        assert macro.enabled  # Default value

        # Should create unique IDs
        macro2 = MacroDefinition.create_test_macro("Test Factory 2", [mock_command])
        assert macro.macro_id != macro2.macro_id


class TestExecutionResult:
    """Test ExecutionResult functionality."""

    def test_execution_result_creation(self) -> None:
        """Test ExecutionResult creation."""
        token = ExecutionToken("exec_123")
        macro_id = MacroId("macro_456")
        started_at = datetime.now()
        completed_at = started_at + timedelta(seconds=5)

        command_result1 = CommandResult.success_result("Command 1 output")
        command_result2 = CommandResult.success_result("Command 2 output")

        result = ExecutionResult(
            execution_token=token,
            macro_id=macro_id,
            status=ExecutionStatus.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
            total_duration=Duration.from_seconds(5.0),
            command_results=[command_result1, command_result2],
        )

        assert result.execution_token == token
        assert result.macro_id == macro_id
        assert result.status == ExecutionStatus.COMPLETED
        assert result.started_at == started_at
        assert result.completed_at == completed_at
        assert result.total_duration is not None
        assert result.total_duration.seconds == 5.0
        assert len(result.command_results) == 2
        assert result.error_details is None

    def test_execution_result_is_successful_true(self) -> None:
        """Test ExecutionResult.is_successful() with successful execution."""
        command_result1 = CommandResult.success_result("Success 1")
        command_result2 = CommandResult.success_result("Success 2")

        result = ExecutionResult(
            execution_token=ExecutionToken("token"),
            macro_id=MacroId("macro"),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(),
            command_results=[command_result1, command_result2],
        )

        assert result.is_successful()

    def test_execution_result_is_successful_false_status(self) -> None:
        """Test ExecutionResult.is_successful() with failed status."""
        command_result = CommandResult.success_result("Success")

        result = ExecutionResult(
            execution_token=ExecutionToken("token"),
            macro_id=MacroId("macro"),
            status=ExecutionStatus.FAILED,
            started_at=datetime.now(),
            command_results=[command_result],
        )

        assert not result.is_successful()

    def test_execution_result_is_successful_false_command(self) -> None:
        """Test ExecutionResult.is_successful() with failed command."""
        command_result1 = CommandResult.success_result("Success")
        command_result2 = CommandResult.failure_result("Command failed")

        result = ExecutionResult(
            execution_token=ExecutionToken("token"),
            macro_id=MacroId("macro"),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(),
            command_results=[command_result1, command_result2],
        )

        assert not result.is_successful()

    def test_execution_result_has_error_info(self) -> None:
        """Test ExecutionResult.has_error_info() method."""
        # With error details
        result_with_error = ExecutionResult(
            execution_token=ExecutionToken("token"),
            macro_id=MacroId("macro"),
            status=ExecutionStatus.FAILED,
            started_at=datetime.now(),
            error_details="Execution failed due to timeout",
        )

        assert result_with_error.has_error_info()

        # With failed command
        failed_command = CommandResult.failure_result("Command error")
        result_with_failed_command = ExecutionResult(
            execution_token=ExecutionToken("token"),
            macro_id=MacroId("macro"),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(),
            command_results=[failed_command],
        )

        assert result_with_failed_command.has_error_info()

        # Without errors
        success_command = CommandResult.success_result("Success")
        result_without_error = ExecutionResult(
            execution_token=ExecutionToken("token"),
            macro_id=MacroId("macro"),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(),
            command_results=[success_command],
        )

        assert not result_without_error.has_error_info()


class TestResult:
    """Test Result class functionality."""

    def test_create_success_result(self) -> None:
        """Test create_success_result factory function."""
        result = create_success_result("test_value")

        assert result.success is True
        assert result.value == "test_value"
        assert result.error_message is None
        assert result.error_code is None
        assert result.is_success() is True
        assert result.is_error() is False
        assert result.message == ""

    def test_create_error_result(self) -> None:
        """Test create_error_result factory function."""
        result = create_error_result("Test error", "ERR_TEST")

        assert result.success is False
        assert result.value is None
        assert result.error_message == "Test error"
        assert result.error_code == "ERR_TEST"
        assert result.is_success() is False
        assert result.is_error() is True
        assert result.message == "Test error"

    def test_result_message_property(self) -> None:
        """Test Result.message property for backwards compatibility."""
        # With error message
        result_with_message = Result(
            success=False, error_message="Error occurred", error_code="ERR_001"
        )
        assert result_with_message.message == "Error occurred"

        # Without error message
        result_without_message = Result(
            success=False, error_message=None, error_code="ERR_002"
        )
        assert result_without_message.message == ""


class TestValidationResult:
    """Test ValidationResult class functionality."""

    def test_validation_result_success_factory(self) -> None:
        """Test ValidationResult.success() factory method."""
        result = ValidationResult.success(check_type="syntax", duration=0.5)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata["check_type"] == "syntax"
        assert result.metadata["duration"] == 0.5

    def test_validation_result_failure_factory(self) -> None:
        """Test ValidationResult.failure() factory method."""
        errors = ["Invalid syntax", "Missing parameter"]
        result = ValidationResult.failure(errors, severity="high")

        assert result.is_valid is False
        assert result.errors == errors
        assert result.warnings == []
        assert result.metadata["severity"] == "high"

    def test_validation_result_add_error(self) -> None:
        """Test ValidationResult.add_error() method."""
        # Start with successful result
        original = ValidationResult.success()

        # Add an error
        with_error = original.add_error("First error")

        # Original should be unchanged
        assert original.is_valid is True
        assert original.errors == []

        # New result should have error
        assert with_error.is_valid is False
        assert with_error.errors == ["First error"]
        assert with_error.warnings == original.warnings
        assert with_error.metadata == original.metadata

        # Add another error
        with_two_errors = with_error.add_error("Second error")
        assert with_two_errors.errors == ["First error", "Second error"]
        assert with_error.errors == ["First error"]  # Original unchanged

    def test_validation_result_with_warnings(self) -> None:
        """Test ValidationResult with warnings."""
        result = ValidationResult(
            is_valid=True, warnings=["Deprecated API usage", "Unused parameter"]
        )

        assert result.is_valid is True
        assert result.errors == []
        assert len(result.warnings) == 2
        assert "Deprecated API usage" in result.warnings


class TestDurationAdditional:
    """Additional tests for Duration class to achieve 100% coverage."""

    def test_duration_from_minutes(self) -> None:
        """Test Duration.from_minutes() class method."""
        # Integer minutes
        duration1 = Duration.from_minutes(5)
        assert duration1.seconds == 300.0

        # Float minutes
        duration2 = Duration.from_minutes(1.5)
        assert duration2.seconds == 90.0

        # Zero minutes
        duration3 = Duration.from_minutes(0)
        assert duration3.seconds == 0.0


class TestExecutionContextAdditional:
    """Additional tests for ExecutionContext to achieve 100% coverage."""

    async def test_execution_context_info_method(self) -> None:
        """Test ExecutionContext.info() mock method."""
        context = ExecutionContext.default()

        # Should not raise exception
        await context.info("Test info message")

        # Method is a no-op, so nothing to assert

    async def test_execution_context_error_method(self) -> None:
        """Test ExecutionContext.error() mock method."""
        context = ExecutionContext.default()

        # Should not raise exception
        await context.error("Test error message")

        # Method is a no-op, so nothing to assert


class TestMacroMoveResult:
    """Test MacroMoveResult functionality."""

    def test_macro_move_result_creation(self) -> None:
        """Test MacroMoveResult creation."""
        macro_id = MacroId("move_macro")
        source_group = GroupId("source_group")
        target_group = GroupId("target_group")
        execution_time = Duration.from_seconds(2.0)
        conflicts = ["name_collision_resolved"]

        result = MacroMoveResult(
            macro_id=macro_id,
            source_group=source_group,
            target_group=target_group,
            execution_time=execution_time,
            conflicts_resolved=conflicts,
        )

        assert result.macro_id == macro_id
        assert result.source_group == source_group
        assert result.target_group == target_group
        assert result.execution_time == execution_time
        assert result.conflicts_resolved == conflicts
        assert result.rollback_info is None

    def test_macro_move_result_was_successful_true(self) -> None:
        """Test MacroMoveResult.was_successful() with successful move."""
        result = MacroMoveResult(
            macro_id=MacroId("macro"),
            source_group=GroupId("source"),
            target_group=GroupId("target"),
            execution_time=Duration.from_seconds(1.0),
            rollback_info=None,
        )

        assert result.was_successful()

    def test_macro_move_result_was_successful_false(self) -> None:
        """Test MacroMoveResult.was_successful() with failed move."""
        result = MacroMoveResult(
            macro_id=MacroId("macro"),
            source_group=GroupId("source"),
            target_group=GroupId("target"),
            execution_time=Duration.from_seconds(1.0),
            rollback_info={"error": "target_group_not_found"},
        )

        assert not result.was_successful()


class TestIntegration:
    """Integration tests for type interactions."""

    def test_complete_execution_workflow(self) -> None:
        """Test complete execution workflow with all types."""
        # Create execution context
        permissions = frozenset([Permission.TEXT_INPUT, Permission.AUTOMATION_CONTROL])
        context = ExecutionContext(
            permissions=permissions,
            timeout=Duration.from_seconds(30),
            variables={VariableName("test_var"): "test_value"},
        )

        # Create mock commands
        mock_command1 = Mock(spec=MacroCommand)
        mock_command1.validate.return_value = True
        mock_command2 = Mock(spec=MacroCommand)
        mock_command2.validate.return_value = True

        # Create macro definition
        macro = MacroDefinition.create_test_macro(
            "Integration Test Macro",
            [mock_command1, mock_command2],
        )

        # Create command results
        result1 = CommandResult.success_result(
            output="Command 1 executed",
            execution_time=Duration.from_seconds(0.5),
        )
        result2 = CommandResult.success_result(
            output="Command 2 executed",
            execution_time=Duration.from_seconds(0.3),
        )

        # Create execution result
        started_at = datetime.now()
        completed_at = started_at + timedelta(seconds=1)

        execution_result = ExecutionResult(
            execution_token=context.execution_id,
            macro_id=macro.macro_id,
            status=ExecutionStatus.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
            total_duration=Duration.from_seconds(0.8),
            command_results=[result1, result2],
        )

        # Verify complete workflow
        assert macro.is_valid()
        assert context.has_permissions(permissions)
        assert execution_result.is_successful()
        assert not execution_result.has_error_info()
        assert len(execution_result.command_results) == 2

    def test_type_immutability_workflow(self) -> None:
        """Test type immutability throughout workflow."""
        # Create initial context
        original_context = ExecutionContext.default()
        original_var_count = len(original_context.variables)

        # Add variable (should create new instance)
        updated_context = original_context.with_variable(
            VariableName("new_var"),
            "new_value",
        )

        # Original should be unchanged
        assert len(original_context.variables) == original_var_count
        assert original_context.get_variable(VariableName("new_var")) is None

        # Updated should have new variable
        assert len(updated_context.variables) == original_var_count + 1
        assert updated_context.get_variable(VariableName("new_var")) == "new_value"

        # Test CommandParameters immutability
        original_params = CommandParameters({"original": "value"})
        updated_params = original_params.with_parameter("new", "param")

        assert original_params.get("new") is None
        assert updated_params.get("new") == "param"
        assert updated_params.get("original") == "value"

    def test_error_propagation_workflow(self) -> None:
        """Test error propagation through type system."""
        # Create context with limited permissions
        limited_context = ExecutionContext(
            permissions=frozenset([Permission.TEXT_INPUT]),
            timeout=Duration.from_seconds(10),
        )

        # Test permission checking
        assert limited_context.has_permission(Permission.TEXT_INPUT)
        assert not limited_context.has_permission(Permission.ADMIN_ACCESS)
        assert not limited_context.has_permissions(
            frozenset([Permission.TEXT_INPUT, Permission.ADMIN_ACCESS]),
        )

        # Create failed command result
        failed_result = CommandResult.failure_result(
            error_message="Permission denied",
            execution_time=Duration.from_seconds(0.1),
        )

        # Create execution result with failure
        execution_result = ExecutionResult(
            execution_token=limited_context.execution_id,
            macro_id=MacroId("failed_macro"),
            status=ExecutionStatus.FAILED,
            started_at=datetime.now(),
            command_results=[failed_result],
            error_details="Macro execution failed due to insufficient permissions",
        )

        # Verify error propagation
        assert not execution_result.is_successful()
        assert execution_result.has_error_info()
        assert execution_result.error_details is not None
        assert not failed_result.success
