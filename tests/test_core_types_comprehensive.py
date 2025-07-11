"""Comprehensive tests for core types module.

Covers all branded types, dataclasses, enums, protocols, and factory functions
with property-based testing and comprehensive validation.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from src.core.types import (
    CommandId,
    CommandParameters,
    CommandResult,
    Duration,
    EmailAddress,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    ExecutionToken,
    GroupId,
    # Protocol
    MacroCommand,
    MacroCreationStatus,
    MacroDefinition,
    # Branded types
    MacroId,
    MacroMoveResult,
    MoveConflictType,
    Permission,
    Priority,
    # Dataclasses
    Result,
    ValidationResult,
    VariableName,
    create_error_result,
    create_macro_id,
    # Factory functions
    create_success_result,
)


class TestBrandedTypes:
    """Test branded type definitions and type safety."""

    def test_macro_id_branded_type(self):
        """Test MacroId branded type creation and usage."""
        test_id = MacroId("test-macro-123")
        assert test_id == "test-macro-123"
        assert isinstance(test_id, str)

    def test_command_id_branded_type(self):
        """Test CommandId branded type creation."""
        cmd_id = CommandId("cmd-456")
        assert cmd_id == "cmd-456"

    def test_execution_token_branded_type(self):
        """Test ExecutionToken branded type creation."""
        token = ExecutionToken("exec-token-789")
        assert token == "exec-token-789"  # noqa: S105

    def test_variable_name_branded_type(self):
        """Test VariableName branded type creation."""
        var_name = VariableName("my_variable")
        assert var_name == "my_variable"

    def test_email_address_branded_type(self):
        """Test EmailAddress branded type creation."""
        email = EmailAddress("user@example.com")
        assert email == "user@example.com"

    def test_all_branded_types_distinct(self):
        """Test that branded types maintain type safety."""
        macro_id = MacroId("test")
        command_id = CommandId("test")

        # Both have same value and are strings at runtime (NewType behavior)
        assert macro_id == command_id  # String comparison
        assert isinstance(macro_id, str)
        assert isinstance(command_id, str)
        # NewType creates aliases, not new runtime types
        assert type(macro_id).__name__ == "str"
        assert type(command_id).__name__ == "str"


class TestResultFactoryFunctions:
    """Test result factory functions."""

    def test_create_success_result(self):
        """Test successful result creation."""
        result = create_success_result("test_value")

        assert result.success is True
        assert result.value == "test_value"
        assert result.error_message is None
        assert result.error_code is None
        assert result.is_success() is True
        assert result.is_error() is False

    def test_create_error_result(self):
        """Test error result creation."""
        result = create_error_result("Test error", "ERR_001")

        assert result.success is False
        assert result.value is None
        assert result.error_message == "Test error"
        assert result.error_code == "ERR_001"
        assert result.is_success() is False
        assert result.is_error() is True

    def test_create_macro_id_unique(self):
        """Test macro ID generation creates unique values."""
        id1 = create_macro_id()
        id2 = create_macro_id()

        assert id1 != id2
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert len(str(id1)) > 0
        assert len(str(id2)) > 0


class TestResult:
    """Test Result dataclass functionality."""

    def test_result_success_creation(self):
        """Test successful result creation."""
        result = Result(success=True, value="test_data")

        assert result.success is True
        assert result.value == "test_data"
        assert result.error_message is None
        assert result.error_code is None
        assert result.is_success() is True
        assert result.is_error() is False
        assert result.message == ""

    def test_result_error_creation(self):
        """Test error result creation."""
        result = Result(
            success=False, error_message="Test error", error_code="ERR_TEST"
        )

        assert result.success is False
        assert result.value is None
        assert result.error_message == "Test error"
        assert result.error_code == "ERR_TEST"
        assert result.is_success() is False
        assert result.is_error() is True
        assert result.message == "Test error"

    def test_result_message_property_none(self):
        """Test message property with None error_message."""
        result = Result(success=False, error_message=None)
        assert result.message == ""

    def test_result_immutability(self):
        """Test that Result is immutable (frozen)."""
        result = Result(success=True, value="test")

        with pytest.raises(FrozenInstanceError):
            result.success = False


class TestMoveConflictType:
    """Test MoveConflictType enumeration."""

    def test_move_conflict_type_values(self):
        """Test all move conflict type values."""
        assert MoveConflictType.NAME_COLLISION.value == "name_collision"
        assert MoveConflictType.PERMISSION_DENIED.value == "permission_denied"
        assert MoveConflictType.GROUP_NOT_FOUND.value == "group_not_found"
        assert MoveConflictType.MACRO_NOT_FOUND.value == "macro_not_found"
        assert (
            MoveConflictType.MACRO_ENABLED_IN_SOURCE.value == "macro_enabled_in_source"
        )
        assert MoveConflictType.TARGET_GROUP_DISABLED.value == "target_group_disabled"

    def test_move_conflict_type_enumeration_complete(self):
        """Test move conflict type enumeration completeness."""
        expected_types = {
            "name_collision",
            "permission_denied",
            "group_not_found",
            "macro_not_found",
            "macro_enabled_in_source",
            "target_group_disabled",
        }
        actual_types = {conflict.value for conflict in MoveConflictType}
        assert actual_types == expected_types


class TestExecutionStatus:
    """Test ExecutionStatus enumeration."""

    def test_execution_status_values(self):
        """Test all execution status values."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.CANCELLED.value == "cancelled"
        assert ExecutionStatus.TIMEOUT.value == "timeout"

    def test_execution_status_enumeration_complete(self):
        """Test execution status enumeration completeness."""
        expected_statuses = {
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
            "timeout",
        }
        actual_statuses = {status.value for status in ExecutionStatus}
        assert actual_statuses == expected_statuses


class TestMacroCreationStatus:
    """Test MacroCreationStatus enumeration."""

    def test_macro_creation_status_values(self):
        """Test all macro creation status values."""
        assert MacroCreationStatus.VALIDATING.value == "validating"
        assert MacroCreationStatus.CREATING.value == "creating"
        assert MacroCreationStatus.COMPLETED.value == "completed"
        assert MacroCreationStatus.FAILED.value == "failed"
        assert MacroCreationStatus.ROLLED_BACK.value == "rolled_back"


class TestPriority:
    """Test Priority enumeration."""

    def test_priority_values(self):
        """Test all priority level values."""
        assert Priority.LOW.value == "low"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.HIGH.value == "high"
        assert Priority.CRITICAL.value == "critical"


class TestPermission:
    """Test Permission enumeration."""

    def test_permission_values(self):
        """Test comprehensive permission type values."""
        assert Permission.TEXT_INPUT.value == "text_input"
        assert Permission.SYSTEM_CONTROL.value == "system_control"
        assert Permission.FILE_ACCESS.value == "file_access"
        assert Permission.APPLICATION_CONTROL.value == "application_control"
        assert Permission.NETWORK_ACCESS.value == "network_access"
        assert Permission.CLIPBOARD_ACCESS.value == "clipboard_access"
        assert Permission.CLIPBOARD_HISTORY.value == "clipboard_history"
        assert Permission.CLIPBOARD_NAMED.value == "clipboard_named"
        assert Permission.SYSTEM_SOUND.value == "system_sound"
        assert Permission.SCREEN_CAPTURE.value == "screen_capture"
        assert Permission.AUDIO_OUTPUT.value == "audio_output"
        assert Permission.WINDOW_MANAGEMENT.value == "window_management"
        assert Permission.FLOW_CONTROL.value == "flow_control"
        assert Permission.MOUSE_CONTROL.value == "mouse_control"
        assert Permission.AUTOMATION_CONTROL.value == "automation_control"
        assert Permission.READ_ACCESS.value == "read_access"
        assert Permission.ADMIN_ACCESS.value == "admin_access"

    def test_permission_enumeration_complete(self):
        """Test permission enumeration completeness."""
        expected_permissions = {
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
        }
        actual_permissions = {perm.value for perm in Permission}
        assert actual_permissions == expected_permissions


class TestDuration:
    """Test Duration dataclass functionality."""

    def test_duration_creation(self):
        """Test basic duration creation."""
        duration = Duration(seconds=5.5)
        assert duration.seconds == 5.5
        assert duration.total_seconds() == 5.5

    def test_duration_negative_validation(self):
        """Test duration validation rejects negative values."""
        with pytest.raises(ValueError, match="Duration cannot be negative"):
            Duration(seconds=-1.0)

    def test_duration_from_seconds(self):
        """Test duration creation from seconds."""
        duration = Duration.from_seconds(10.0)
        assert duration.seconds == 10.0

    def test_duration_from_milliseconds(self):
        """Test duration creation from milliseconds."""
        duration = Duration.from_milliseconds(2000)
        assert duration.seconds == 2.0

    def test_duration_from_minutes(self):
        """Test duration creation from minutes."""
        duration = Duration.from_minutes(2.0)
        assert duration.seconds == 120.0

    def test_duration_addition(self):
        """Test duration addition."""
        d1 = Duration(seconds=5.0)
        d2 = Duration(seconds=3.0)
        result = d1 + d2
        assert result.seconds == 8.0

    def test_duration_comparison_operations(self):
        """Test duration comparison operations."""
        d1 = Duration(seconds=5.0)
        d2 = Duration(seconds=10.0)
        d3 = Duration(seconds=5.0)

        assert d1 < d2
        assert d1 <= d2
        assert d1 <= d3
        assert d2 > d1
        assert d2 >= d1
        assert d1 >= d3
        assert d1 == d3
        assert d1 != d2

    def test_duration_equality_with_non_duration(self):
        """Test duration equality with non-Duration objects."""
        duration = Duration(seconds=5.0)
        assert duration != 5.0
        assert duration != "5 seconds"
        assert duration is not None

    def test_duration_zero_constant(self):
        """Test Duration.ZERO constant."""
        assert Duration.ZERO.seconds == 0.0
        assert Duration.ZERO == Duration(seconds=0.0)

    def test_duration_immutability(self):
        """Test that Duration is immutable (frozen)."""
        duration = Duration(seconds=5.0)

        with pytest.raises(FrozenInstanceError):
            duration.seconds = 10.0


class TestCommandParameters:
    """Test CommandParameters dataclass functionality."""

    def test_command_parameters_creation(self):
        """Test command parameters creation."""
        params = CommandParameters(data={"key1": "value1", "key2": 42})
        assert params.data == {"key1": "value1", "key2": 42}

    def test_command_parameters_empty(self):
        """Test empty command parameters creation."""
        params = CommandParameters.empty()
        assert params.data == {}

    def test_command_parameters_get(self):
        """Test parameter retrieval."""
        params = CommandParameters(data={"test_key": "test_value"})

        assert params.get("test_key") == "test_value"
        assert params.get("nonexistent") is None
        assert params.get("nonexistent", "default") == "default"

    def test_command_parameters_with_parameter(self):
        """Test immutable parameter addition."""
        original = CommandParameters(data={"existing": "value"})
        new_params = original.with_parameter("new_key", "new_value")

        # Original unchanged
        assert original.data == {"existing": "value"}

        # New parameters include both
        assert new_params.data == {"existing": "value", "new_key": "new_value"}

    def test_command_parameters_default_factory(self):
        """Test default factory for empty data."""
        params = CommandParameters()
        assert params.data == {}

    def test_command_parameters_immutability(self):
        """Test that CommandParameters is immutable (frozen)."""
        params = CommandParameters(data={"key": "value"})

        with pytest.raises(FrozenInstanceError):
            params.data = {"new": "data"}


class TestExecutionContext:
    """Test ExecutionContext dataclass functionality."""

    def test_execution_context_creation(self):
        """Test execution context creation."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_CONTROL])
        timeout = Duration.from_seconds(30)

        context = ExecutionContext(permissions=permissions, timeout=timeout)

        assert context.permissions == permissions
        assert context.timeout == timeout
        assert context.variables == {}
        assert len(str(context.execution_id)) > 0
        assert isinstance(context.created_at, datetime)

    def test_execution_context_has_permission(self):
        """Test permission checking."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_CONTROL])
        context = ExecutionContext(permissions=permissions, timeout=Duration.ZERO)

        assert context.has_permission(Permission.TEXT_INPUT) is True
        assert context.has_permission(Permission.FILE_ACCESS) is False

    def test_execution_context_has_permissions_multiple(self):
        """Test multiple permission checking."""
        permissions = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_CONTROL])
        context = ExecutionContext(permissions=permissions, timeout=Duration.ZERO)

        # Subset check
        required = frozenset([Permission.TEXT_INPUT])
        assert context.has_permissions(required) is True

        # Exact match
        assert context.has_permissions(permissions) is True

        # Superset check (should fail)
        too_many = frozenset([Permission.TEXT_INPUT, Permission.FILE_ACCESS])
        assert context.has_permissions(too_many) is False

    def test_execution_context_variables(self):
        """Test variable management."""
        context = ExecutionContext(
            permissions=frozenset(),
            timeout=Duration.ZERO,
            variables={VariableName("test_var"): "test_value"},
        )

        assert context.get_variable(VariableName("test_var")) == "test_value"
        assert context.get_variable(VariableName("nonexistent")) is None

    def test_execution_context_with_variable(self):
        """Test immutable variable addition."""
        original = ExecutionContext(
            permissions=frozenset([Permission.TEXT_INPUT]),
            timeout=Duration.from_seconds(10),
            variables={VariableName("existing"): "value"},
        )

        new_context = original.with_variable(VariableName("new_var"), "new_value")

        # Original unchanged
        assert original.get_variable(VariableName("new_var")) is None

        # New context has both variables
        assert new_context.get_variable(VariableName("existing")) == "value"
        assert new_context.get_variable(VariableName("new_var")) == "new_value"

        # Other fields preserved
        assert new_context.permissions == original.permissions
        assert new_context.timeout == original.timeout
        assert new_context.execution_id == original.execution_id

    def test_execution_context_create_test_context(self):
        """Test test context creation."""
        context = ExecutionContext.create_test_context()

        expected_permissions = frozenset(
            [Permission.TEXT_INPUT, Permission.SYSTEM_SOUND]
        )
        assert context.permissions == expected_permissions
        assert context.timeout == Duration.from_seconds(30)

    def test_execution_context_create_test_context_custom(self):
        """Test test context creation with custom values."""
        custom_permissions = frozenset([Permission.FILE_ACCESS])
        custom_timeout = Duration.from_seconds(60)

        context = ExecutionContext.create_test_context(
            permissions=custom_permissions, timeout=custom_timeout
        )

        assert context.permissions == custom_permissions
        assert context.timeout == custom_timeout

    def test_execution_context_default(self):
        """Test default context creation."""
        context = ExecutionContext.default()

        expected_permissions = frozenset(
            [Permission.TEXT_INPUT, Permission.SYSTEM_SOUND]
        )
        assert context.permissions == expected_permissions
        assert context.timeout == Duration.from_seconds(30)

    async def test_execution_context_async_methods(self):
        """Test async logging methods for test compatibility."""
        context = ExecutionContext.default()

        # These should not raise exceptions
        await context.info("Test info message")
        await context.error("Test error message")


class TestCommandResult:
    """Test CommandResult dataclass functionality."""

    def test_command_result_creation(self):
        """Test command result creation."""
        execution_time = Duration.from_seconds(1.5)
        result = CommandResult(
            success=True,
            output="Command output",
            execution_time=execution_time,
            metadata={"key": "value"},
        )

        assert result.success is True
        assert result.output == "Command output"
        assert result.error_message is None
        assert result.execution_time == execution_time
        assert result.metadata == {"key": "value"}

    def test_command_result_success_factory(self):
        """Test successful command result factory method."""
        execution_time = Duration.from_seconds(2.0)
        result = CommandResult.success_result(
            output="Success output",
            execution_time=execution_time,
            extra_data="metadata",
        )

        assert result.success is True
        assert result.output == "Success output"
        assert result.execution_time == execution_time
        assert result.metadata == {"extra_data": "metadata"}
        assert result.error_message is None

    def test_command_result_failure_factory(self):
        """Test failed command result factory method."""
        execution_time = Duration.from_seconds(0.5)
        result = CommandResult.failure_result(
            error_message="Command failed",
            execution_time=execution_time,
            error_code="ERR_001",
        )

        assert result.success is False
        assert result.error_message == "Command failed"
        assert result.execution_time == execution_time
        assert result.metadata == {"error_code": "ERR_001"}
        assert result.output is None


class TestMacroMoveResult:
    """Test MacroMoveResult dataclass functionality."""

    def test_macro_move_result_successful(self):
        """Test successful macro move result."""
        result = MacroMoveResult(
            macro_id=MacroId("test-macro"),
            source_group=GroupId("source-group"),
            target_group=GroupId("target-group"),
            execution_time=Duration.from_seconds(1.0),
            conflicts_resolved=["name_conflict"],
            rollback_info=None,
        )

        assert result.macro_id == MacroId("test-macro")
        assert result.source_group == GroupId("source-group")
        assert result.target_group == GroupId("target-group")
        assert result.execution_time.seconds == 1.0
        assert result.conflicts_resolved == ["name_conflict"]
        assert result.rollback_info is None
        assert result.was_successful() is True

    def test_macro_move_result_failed(self):
        """Test failed macro move result."""
        rollback_info = {"original_group": "source-group"}
        result = MacroMoveResult(
            macro_id=MacroId("test-macro"),
            source_group=GroupId("source-group"),
            target_group=GroupId("target-group"),
            execution_time=Duration.from_seconds(0.5),
            rollback_info=rollback_info,
        )

        assert result.rollback_info == rollback_info
        assert result.was_successful() is False

    def test_macro_move_result_default_conflicts(self):
        """Test default conflicts_resolved list."""
        result = MacroMoveResult(
            macro_id=MacroId("test-macro"),
            source_group=GroupId("source-group"),
            target_group=GroupId("target-group"),
            execution_time=Duration.from_seconds(1.0),
        )

        assert result.conflicts_resolved == []


class TestMacroDefinition:
    """Test MacroDefinition dataclass functionality."""

    def test_macro_definition_creation(self):
        """Test macro definition creation."""
        # Create mock commands
        mock_command1 = Mock(spec=MacroCommand)
        mock_command1.validate.return_value = True
        mock_command2 = Mock(spec=MacroCommand)
        mock_command2.validate.return_value = True

        macro = MacroDefinition(
            macro_id=MacroId("test-macro"),
            name="Test Macro",
            commands=[mock_command1, mock_command2],
            description="Test description",
            group_id=GroupId("test-group"),
        )

        assert macro.macro_id == MacroId("test-macro")
        assert macro.name == "Test Macro"
        assert len(macro.commands) == 2
        assert macro.enabled is True  # Default value
        assert macro.description == "Test description"
        assert macro.group_id == GroupId("test-group")
        assert isinstance(macro.created_at, datetime)

    def test_macro_definition_is_valid_success(self):
        """Test macro validation success."""
        mock_command = Mock(spec=MacroCommand)
        mock_command.validate.return_value = True

        macro = MacroDefinition(
            macro_id=MacroId("valid-macro"), name="Valid Macro", commands=[mock_command]
        )

        assert macro.is_valid() is True

    def test_macro_definition_is_valid_no_name(self):
        """Test macro validation fails with no name."""
        mock_command = Mock(spec=MacroCommand)
        mock_command.validate.return_value = True

        macro = MacroDefinition(
            macro_id=MacroId("invalid-macro"),
            name="",  # Empty name
            commands=[mock_command],
        )

        assert macro.is_valid() is False

    def test_macro_definition_is_valid_no_commands(self):
        """Test macro validation fails with no commands."""
        macro = MacroDefinition(
            macro_id=MacroId("invalid-macro"),
            name="Invalid Macro",
            commands=[],  # No commands
        )

        assert macro.is_valid() is False

    def test_macro_definition_is_valid_invalid_command(self):
        """Test macro validation fails with invalid command."""
        mock_valid_command = Mock(spec=MacroCommand)
        mock_valid_command.validate.return_value = True

        mock_invalid_command = Mock(spec=MacroCommand)
        mock_invalid_command.validate.return_value = False

        macro = MacroDefinition(
            macro_id=MacroId("invalid-macro"),
            name="Macro with Invalid Command",
            commands=[mock_valid_command, mock_invalid_command],
        )

        assert macro.is_valid() is False

    def test_macro_definition_create_test_macro(self):
        """Test test macro creation factory."""
        mock_command = Mock(spec=MacroCommand)
        mock_command.validate.return_value = True

        macro = MacroDefinition.create_test_macro(
            name="Test Macro", commands=[mock_command]
        )

        assert macro.name == "Test Macro"
        assert len(macro.commands) == 1
        assert len(str(macro.macro_id)) > 0  # UUID generated
        assert macro.enabled is True


class TestExecutionResult:
    """Test ExecutionResult dataclass functionality."""

    def test_execution_result_successful(self):
        """Test successful execution result."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=2)

        successful_command_result = CommandResult.success_result("Command executed")

        result = ExecutionResult(
            execution_token=ExecutionToken("token-123"),
            macro_id=MacroId("macro-456"),
            status=ExecutionStatus.COMPLETED,
            started_at=start_time,
            completed_at=end_time,
            total_duration=Duration.from_seconds(2.0),
            command_results=[successful_command_result],
        )

        assert result.execution_token == ExecutionToken("token-123")
        assert result.macro_id == MacroId("macro-456")
        assert result.status == ExecutionStatus.COMPLETED
        assert result.started_at == start_time
        assert result.completed_at == end_time
        assert result.total_duration.seconds == 2.0
        assert len(result.command_results) == 1
        assert result.error_details is None
        assert result.is_successful() is True
        assert result.has_error_info() is False

    def test_execution_result_failed_status(self):
        """Test execution result with failed status."""
        result = ExecutionResult(
            execution_token=ExecutionToken("token-123"),
            macro_id=MacroId("macro-456"),
            status=ExecutionStatus.FAILED,  # Failed status
            started_at=datetime.now(),
            error_details="Execution failed",
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.error_details == "Execution failed"
        assert result.is_successful() is False
        assert result.has_error_info() is True

    def test_execution_result_failed_command(self):
        """Test execution result with failed command."""
        successful_command = CommandResult.success_result("Success")
        failed_command = CommandResult.failure_result("Command failed")

        result = ExecutionResult(
            execution_token=ExecutionToken("token-123"),
            macro_id=MacroId("macro-456"),
            status=ExecutionStatus.COMPLETED,  # Status OK
            started_at=datetime.now(),
            command_results=[successful_command, failed_command],  # But command failed
        )

        assert result.is_successful() is False  # Failed due to command failure
        assert result.has_error_info() is True

    def test_execution_result_default_values(self):
        """Test execution result default values."""
        result = ExecutionResult(
            execution_token=ExecutionToken("token-123"),
            macro_id=MacroId("macro-456"),
            status=ExecutionStatus.PENDING,
            started_at=datetime.now(),
        )

        assert result.completed_at is None
        assert result.total_duration is None
        assert result.command_results == []
        assert result.error_details is None


class TestValidationResult:
    """Test ValidationResult dataclass functionality."""

    def test_validation_result_success_factory(self):
        """Test successful validation result factory."""
        result = ValidationResult.success(
            validation_type="macro", timestamp="2023-01-01"
        )

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata == {
            "validation_type": "macro",
            "timestamp": "2023-01-01",
        }

    def test_validation_result_failure_factory(self):
        """Test failed validation result factory."""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult.failure(errors, validation_type="command")

        assert result.is_valid is False
        assert result.errors == errors
        assert result.warnings == []
        assert result.metadata == {"validation_type": "command"}

    def test_validation_result_add_error(self):
        """Test adding errors to validation result."""
        original = ValidationResult.success()
        result_with_error = original.add_error("New error")

        # Original unchanged
        assert original.is_valid is True
        assert original.errors == []

        # New result has error
        assert result_with_error.is_valid is False
        assert result_with_error.errors == ["New error"]

    def test_validation_result_add_error_preserves_data(self):
        """Test that adding error preserves other data."""
        original = ValidationResult(
            is_valid=True,
            errors=["Existing error"],
            warnings=["Warning"],
            metadata={"key": "value"},
        )

        result_with_error = original.add_error("New error")

        assert result_with_error.is_valid is False
        assert result_with_error.errors == ["Existing error", "New error"]
        assert result_with_error.warnings == ["Warning"]
        assert result_with_error.metadata == {"key": "value"}


class TestPropertyBasedTypes:
    """Property-based tests for core types."""

    @given(st.floats(min_value=0.0, max_value=86400.0))  # 0 to 24 hours
    def test_duration_creation_property(self, seconds):
        """Property test: Duration creation with valid values."""
        duration = Duration(seconds=seconds)
        assert duration.seconds == seconds
        assert duration.total_seconds() == seconds

    @given(st.floats(max_value=-0.001))
    def test_duration_negative_validation_property(self, negative_seconds):
        """Property test: Duration rejects negative values."""
        with pytest.raises(ValueError):
            Duration(seconds=negative_seconds)

    @given(
        st.text(min_size=1, max_size=100),
        st.dictionaries(
            st.text(min_size=1), st.one_of(st.text(), st.integers(), st.booleans())
        ),
    )
    def test_command_parameters_property(self, key, param_dict):
        """Property test: CommandParameters operations."""
        assume(key not in param_dict)  # Avoid key conflicts

        params = CommandParameters(data=param_dict)
        new_params = params.with_parameter(key, "test_value")

        # Original unchanged
        assert params.data == param_dict

        # New parameters include the addition
        expected_data = param_dict.copy()
        expected_data[key] = "test_value"
        assert new_params.data == expected_data

    @given(st.booleans(), st.text())
    def test_result_creation_property(self, success, value):
        """Property test: Result creation with various inputs."""
        if success:
            result = Result(success=True, value=value)
            assert result.is_success() is True
            assert result.is_error() is False
            assert result.value == value
        else:
            result = Result(success=False, error_message=value)
            assert result.is_success() is False
            assert result.is_error() is True
            assert result.error_message == value
