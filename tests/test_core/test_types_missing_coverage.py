"""Comprehensive tests to achieve 100% coverage of src/core/types.py.

This module provides targeted tests for uncovered lines in types.py to reach 95%+ coverage.
"""

from datetime import datetime

import pytest
from src.core.types import (
    ActionId,
    AppId,
    BundleId,
    ClipboardId,
    CommandId,
    CommandParameters,
    CommandResult,
    ConditionId,
    CreationToken,
    Duration,
    EmailAddress,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    ExecutionToken,
    GroupId,
    MacroCreationStatus,
    MacroDefinition,
    # Branded types
    MacroId,
    MacroMoveResult,
    MenuItemId,
    MoveConflictType,
    Permission,
    Priority,
    # Classes and enums
    Result,
    TemplateId,
    ToolId,
    TriggerId,
    UserId,
    ValidationResult,
    VariableName,
    create_error_result,
    create_success_result,
)


class TestBrandedTypes:
    """Test branded type creation and behavior."""

    def test_macro_id_creation(self) -> None:
        """Test MacroId branded type creation."""
        macro_id = MacroId("macro-123")
        assert str(macro_id) == "macro-123"
        assert isinstance(macro_id, str)

    def test_command_id_creation(self) -> None:
        """Test CommandId branded type creation."""
        command_id = CommandId("cmd-456")
        assert str(command_id) == "cmd-456"

    def test_execution_token_creation(self) -> None:
        """Test ExecutionToken branded type creation."""
        token = ExecutionToken("token-789")
        assert str(token) == "token-789"

    def test_trigger_id_creation(self) -> None:
        """Test TriggerId branded type creation."""
        trigger_id = TriggerId("trigger-abc")
        assert str(trigger_id) == "trigger-abc"

    def test_group_id_creation(self) -> None:
        """Test GroupId branded type creation."""
        group_id = GroupId("group-def")
        assert str(group_id) == "group-def"

    def test_variable_name_creation(self) -> None:
        """Test VariableName branded type creation."""
        var_name = VariableName("my_variable")
        assert str(var_name) == "my_variable"

    def test_template_id_creation(self) -> None:
        """Test TemplateId branded type creation."""
        template_id = TemplateId("template-ghi")
        assert str(template_id) == "template-ghi"

    def test_creation_token_creation(self) -> None:
        """Test CreationToken branded type creation."""
        creation_token = CreationToken("create-jkl")
        assert str(creation_token) == "create-jkl"

    def test_clipboard_id_creation(self) -> None:
        """Test ClipboardId branded type creation."""
        clipboard_id = ClipboardId("clip-mno")
        assert str(clipboard_id) == "clip-mno"

    def test_app_id_creation(self) -> None:
        """Test AppId branded type creation."""
        app_id = AppId("app-pqr")
        assert str(app_id) == "app-pqr"

    def test_bundle_id_creation(self) -> None:
        """Test BundleId branded type creation."""
        bundle_id = BundleId("com.example.app")
        assert str(bundle_id) == "com.example.app"

    def test_menu_item_id_creation(self) -> None:
        """Test MenuItemId branded type creation."""
        menu_id = MenuItemId("menu-stu")
        assert str(menu_id) == "menu-stu"

    def test_tool_id_creation(self) -> None:
        """Test ToolId branded type creation."""
        tool_id = ToolId("tool-vwx")
        assert str(tool_id) == "tool-vwx"

    def test_user_id_creation(self) -> None:
        """Test UserId branded type creation."""
        user_id = UserId("user-yz1")
        assert str(user_id) == "user-yz1"

    def test_action_id_creation(self) -> None:
        """Test ActionId branded type creation."""
        action_id = ActionId("action-234")
        assert str(action_id) == "action-234"

    def test_email_address_creation(self) -> None:
        """Test EmailAddress branded type creation."""
        email = EmailAddress("test@example.com")
        assert str(email) == "test@example.com"

    def test_condition_id_creation(self) -> None:
        """Test ConditionId branded type creation."""
        condition_id = ConditionId("condition-567")
        assert str(condition_id) == "condition-567"


class TestResultFunctions:
    """Test result factory functions."""

    def test_create_success_result(self) -> None:
        """Test create_success_result function."""
        result = create_success_result("test_value")
        assert result.success is True
        assert result.value == "test_value"
        assert result.error_message is None
        assert result.error_code is None

    def test_create_success_result_with_none(self) -> None:
        """Test create_success_result with None value."""
        result = create_success_result(None)
        assert result.success is True
        assert result.value is None
        assert result.error_message is None

    def test_create_success_result_with_complex_object(self) -> None:
        """Test create_success_result with complex object."""
        complex_obj = {"key": "value", "list": [1, 2, 3]}
        result = create_success_result(complex_obj)
        assert result.success is True
        assert result.value == complex_obj

    def test_create_error_result(self) -> None:
        """Test create_error_result function."""
        result = create_error_result("Test error", "ERR_001")
        assert result.success is False
        assert result.value is None
        assert result.error_message == "Test error"
        assert result.error_code == "ERR_001"

    def test_create_error_result_empty_message(self) -> None:
        """Test create_error_result with empty message."""
        result = create_error_result("", "ERR_002")
        assert result.success is False
        assert result.error_message == ""
        assert result.error_code == "ERR_002"

    def test_create_macro_id(self) -> None:
        """Test create_macro_id function."""
        from src.core.types import create_macro_id

        macro_id = create_macro_id()
        assert isinstance(macro_id, str)  # MacroId is a NewType(str)
        assert len(macro_id) == 36  # UUID string length
        assert "-" in macro_id  # UUID format

        # Test that multiple calls generate different IDs
        macro_id2 = create_macro_id()
        assert macro_id != macro_id2


class TestResultClass:
    """Test Result dataclass behavior."""

    def test_result_creation_success(self) -> None:
        """Test Result creation for success case."""
        result = Result(success=True, value="test", error_message=None, error_code=None)
        assert result.success is True
        assert result.value == "test"
        assert result.error_message is None
        assert result.error_code is None

    def test_result_creation_failure(self) -> None:
        """Test Result creation for failure case."""
        result = Result(
            success=False, value=None, error_message="Error", error_code="E001"
        )
        assert result.success is False
        assert result.value is None
        assert result.error_message == "Error"
        assert result.error_code == "E001"

    def test_result_immutability(self) -> None:
        """Test that Result instances are immutable (frozen)."""
        result = Result(success=True, value="test", error_message=None, error_code=None)
        with pytest.raises(AttributeError):
            result.success = False

    def test_result_equality(self) -> None:
        """Test Result equality comparison."""
        result1 = Result(
            success=True, value="test", error_message=None, error_code=None
        )
        result2 = Result(
            success=True, value="test", error_message=None, error_code=None
        )
        result3 = Result(
            success=False, value=None, error_message="Error", error_code="E001"
        )

        assert result1 == result2
        assert result1 != result3

    def test_result_status_methods(self) -> None:
        """Test Result is_success and is_error methods."""
        success_result = Result(
            success=True, value="test", error_message=None, error_code=None
        )
        error_result = Result(
            success=False, value=None, error_message="Error", error_code="E001"
        )

        # Test is_success method
        assert success_result.is_success() is True
        assert error_result.is_success() is False

        # Test is_error method
        assert success_result.is_error() is False
        assert error_result.is_error() is True

    def test_result_message_property(self) -> None:
        """Test Result message property for backwards compatibility."""
        result_with_message = Result(
            success=False,
            value=None,
            error_message="Test error message",
            error_code="E001",
        )
        result_without_message = Result(
            success=False, value=None, error_message=None, error_code="E001"
        )

        # Test message property returns error_message
        assert result_with_message.message == "Test error message"
        # Test message property returns empty string when error_message is None
        assert result_without_message.message == ""


class TestDurationClass:
    """Test Duration class methods and validation."""

    def test_duration_creation(self) -> None:
        """Test Duration creation with valid seconds."""
        duration = Duration(seconds=5.0)
        assert duration.seconds == 5.0

    def test_duration_negative_validation(self) -> None:
        """Test Duration validation for negative values."""
        with pytest.raises(ValueError, match="Duration cannot be negative"):
            Duration(seconds=-1.0)

    def test_duration_from_seconds(self) -> None:
        """Test Duration.from_seconds class method."""
        duration = Duration.from_seconds(10.5)
        assert duration.seconds == 10.5

    def test_duration_from_milliseconds(self) -> None:
        """Test Duration.from_milliseconds class method."""
        duration = Duration.from_milliseconds(2500)
        assert duration.seconds == 2.5

    def test_duration_from_minutes(self) -> None:
        """Test Duration.from_minutes class method."""
        duration = Duration.from_minutes(2.0)
        assert duration.seconds == 120.0

    def test_duration_total_seconds(self) -> None:
        """Test Duration.total_seconds method."""
        duration = Duration(seconds=45.25)
        assert duration.total_seconds() == 45.25

    def test_duration_addition(self) -> None:
        """Test Duration addition."""
        duration1 = Duration(seconds=5.0)
        duration2 = Duration(seconds=3.0)
        result = duration1 + duration2
        assert result.seconds == 8.0

    def test_duration_comparison_lt(self) -> None:
        """Test Duration less than comparison."""
        duration1 = Duration(seconds=2.0)
        duration2 = Duration(seconds=5.0)
        assert duration1 < duration2
        assert not duration2 < duration1

    def test_duration_comparison_le(self) -> None:
        """Test Duration less than or equal comparison."""
        duration1 = Duration(seconds=2.0)
        duration2 = Duration(seconds=5.0)
        duration3 = Duration(seconds=2.0)
        assert duration1 <= duration2
        assert duration1 <= duration3
        assert not duration2 <= duration1

    def test_duration_comparison_gt(self) -> None:
        """Test Duration greater than comparison."""
        duration1 = Duration(seconds=5.0)
        duration2 = Duration(seconds=2.0)
        assert duration1 > duration2
        assert not duration2 > duration1

    def test_duration_comparison_ge(self) -> None:
        """Test Duration greater than or equal comparison."""
        duration1 = Duration(seconds=5.0)
        duration2 = Duration(seconds=2.0)
        duration3 = Duration(seconds=5.0)
        assert duration1 >= duration2
        assert duration1 >= duration3
        assert not duration2 >= duration1

    def test_duration_equality(self) -> None:
        """Test Duration equality comparison."""
        duration1 = Duration(seconds=3.5)
        duration2 = Duration(seconds=3.5)
        duration3 = Duration(seconds=4.0)
        assert duration1 == duration2
        assert duration1 != duration3

    def test_duration_equality_with_non_duration(self) -> None:
        """Test Duration equality with non-Duration object."""
        duration = Duration(seconds=3.5)
        assert duration != 3.5
        assert duration != "3.5"
        assert duration is not None

    def test_duration_zero_constant(self) -> None:
        """Test Duration.ZERO constant."""
        assert Duration.ZERO.seconds == 0.0
        assert Duration.ZERO == Duration(seconds=0.0)


class TestEnums:
    """Test enum classes and their values."""

    def test_execution_status_enum_values(self) -> None:
        """Test ExecutionStatus enum has expected values."""
        assert ExecutionStatus.PENDING
        assert ExecutionStatus.RUNNING
        assert ExecutionStatus.COMPLETED
        assert ExecutionStatus.FAILED
        assert ExecutionStatus.CANCELLED
        assert ExecutionStatus.TIMEOUT
        # Test enum value types
        assert isinstance(ExecutionStatus.PENDING.value, str)

    def test_priority_enum_values(self) -> None:
        """Test Priority enum has expected values."""
        assert Priority.LOW
        assert Priority.MEDIUM
        assert Priority.HIGH
        assert Priority.CRITICAL
        assert isinstance(Priority.HIGH.value, str)

    def test_permission_enum_values(self) -> None:
        """Test Permission enum has expected values."""
        assert Permission.TEXT_INPUT
        assert Permission.SYSTEM_CONTROL
        assert Permission.FILE_ACCESS
        assert Permission.APPLICATION_CONTROL
        assert Permission.NETWORK_ACCESS
        assert Permission.CLIPBOARD_ACCESS
        assert isinstance(Permission.TEXT_INPUT.value, str)

    def test_move_conflict_type_enum_values(self) -> None:
        """Test MoveConflictType enum has expected values."""
        assert MoveConflictType.NAME_COLLISION
        assert MoveConflictType.PERMISSION_DENIED
        assert MoveConflictType.GROUP_NOT_FOUND
        assert MoveConflictType.MACRO_NOT_FOUND
        assert isinstance(MoveConflictType.NAME_COLLISION.value, str)

    def test_macro_creation_status_enum_values(self) -> None:
        """Test MacroCreationStatus enum has expected values."""
        assert MacroCreationStatus.VALIDATING
        assert MacroCreationStatus.CREATING
        assert MacroCreationStatus.COMPLETED
        assert MacroCreationStatus.FAILED
        assert MacroCreationStatus.ROLLED_BACK
        assert isinstance(MacroCreationStatus.VALIDATING.value, str)


class TestDataClasses:
    """Test dataclass creation and behavior."""

    def test_command_parameters_creation(self) -> None:
        """Test CommandParameters dataclass creation."""
        params = CommandParameters(data={"cmd": "test"})
        assert params.data == {"cmd": "test"}

    def test_command_parameters_get_method(self) -> None:
        """Test CommandParameters.get method."""
        params = CommandParameters(data={"key1": "value1", "key2": 42})
        assert params.get("key1") == "value1"
        assert params.get("key2") == 42
        assert params.get("nonexistent") is None
        assert params.get("nonexistent", "default") == "default"

    def test_command_parameters_with_parameter(self) -> None:
        """Test CommandParameters.with_parameter method."""
        params = CommandParameters(data={"original": "value"})
        new_params = params.with_parameter("new_key", "new_value")

        # Original should be unchanged (immutable)
        assert params.data == {"original": "value"}
        # New instance should have both values
        assert new_params.data == {"original": "value", "new_key": "new_value"}

    def test_command_parameters_empty_factory(self) -> None:
        """Test CommandParameters.empty() factory method."""
        empty_params = CommandParameters.empty()
        assert empty_params.data == {}
        assert isinstance(empty_params, CommandParameters)

    def test_execution_context_creation(self) -> None:
        """Test ExecutionContext dataclass creation."""
        context = ExecutionContext(
            permissions=frozenset([Permission.TEXT_INPUT]),
            timeout=Duration.from_seconds(30),
        )
        assert Permission.TEXT_INPUT in context.permissions
        assert context.timeout.seconds == 30

    def test_execution_context_permission_checks(self) -> None:
        """Test ExecutionContext permission checking methods."""
        context = ExecutionContext(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_CONTROL]),
            timeout=Duration.from_seconds(30),
        )

        assert context.has_permission(Permission.TEXT_INPUT)
        assert context.has_permission(Permission.SYSTEM_CONTROL)
        assert not context.has_permission(Permission.FILE_ACCESS)

        # Test multiple permissions check
        required_perms = frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_CONTROL])
        assert context.has_permissions(required_perms)

        required_perms_with_missing = frozenset(
            [Permission.TEXT_INPUT, Permission.FILE_ACCESS]
        )
        assert not context.has_permissions(required_perms_with_missing)

    def test_execution_context_variable_operations(self) -> None:
        """Test ExecutionContext variable operations."""
        context = ExecutionContext(
            permissions=frozenset([Permission.TEXT_INPUT]),
            timeout=Duration.from_seconds(30),
            variables={VariableName("existing"): "value"},
        )

        # Test getting existing variable
        assert context.get_variable(VariableName("existing")) == "value"
        assert context.get_variable(VariableName("nonexistent")) is None

        # Test with_variable (immutable update)
        new_context = context.with_variable(VariableName("new_var"), "new_value")
        assert (
            context.get_variable(VariableName("new_var")) is None
        )  # Original unchanged
        assert new_context.get_variable(VariableName("new_var")) == "new_value"
        assert (
            new_context.get_variable(VariableName("existing")) == "value"
        )  # Preserved

    def test_execution_context_factory_methods(self) -> None:
        """Test ExecutionContext factory methods."""
        test_context = ExecutionContext.create_test_context()
        assert Permission.TEXT_INPUT in test_context.permissions
        assert Permission.SYSTEM_SOUND in test_context.permissions
        assert test_context.timeout.seconds == 30

        default_context = ExecutionContext.default()
        assert default_context.permissions == test_context.permissions

    async def test_execution_context_async_methods(self) -> None:
        """Test ExecutionContext async info and error methods."""
        context = ExecutionContext.create_test_context()

        # These methods are no-ops for test compatibility
        await context.info("Test info message")
        await context.error("Test error message")

        # Should complete without errors (no-op methods)

    def test_command_result_creation(self) -> None:
        """Test CommandResult dataclass creation."""
        result = CommandResult(
            success=True,
            output="Test output",
            execution_time=Duration.from_seconds(1.5),
        )
        assert result.success is True
        assert result.output == "Test output"
        assert result.execution_time.seconds == 1.5

    def test_command_result_factory_methods(self) -> None:
        """Test CommandResult factory methods."""
        success_result = CommandResult.success_result(
            output="Success",
            execution_time=Duration.from_seconds(1.0),
            extra_data="metadata",
        )
        assert success_result.success is True
        assert success_result.output == "Success"
        assert success_result.metadata["extra_data"] == "metadata"

        failure_result = CommandResult.failure_result(
            error_message="Failed",
            execution_time=Duration.from_seconds(0.5),
            error_code="E001",
        )
        assert failure_result.success is False
        assert failure_result.error_message == "Failed"
        assert failure_result.metadata["error_code"] == "E001"

    def test_macro_definition_creation(self) -> None:
        """Test MacroDefinition dataclass creation."""
        from unittest.mock import Mock

        mock_command = Mock()
        mock_command.validate.return_value = True

        macro = MacroDefinition(
            macro_id=MacroId("test-macro"),
            name="Test Macro",
            commands=[mock_command],
            enabled=True,
            group_id=GroupId("test-group"),
        )
        assert macro.macro_id == MacroId("test-macro")
        assert macro.name == "Test Macro"
        assert macro.enabled is True
        assert macro.group_id == GroupId("test-group")

    def test_macro_definition_validation(self) -> None:
        """Test MacroDefinition validation logic."""
        from unittest.mock import Mock

        # Test invalid macro (no name)
        invalid_macro = MacroDefinition(
            macro_id=MacroId("invalid"), name="", commands=[]
        )
        assert not invalid_macro.is_valid()

        # Test invalid macro (no commands)
        invalid_macro2 = MacroDefinition(
            macro_id=MacroId("invalid2"), name="Valid Name", commands=[]
        )
        assert not invalid_macro2.is_valid()

        # Test valid macro
        mock_command = Mock()
        mock_command.validate.return_value = True

        valid_macro = MacroDefinition(
            macro_id=MacroId("valid"), name="Valid Macro", commands=[mock_command]
        )
        assert valid_macro.is_valid()

    def test_macro_definition_create_test_macro(self) -> None:
        """Test MacroDefinition.create_test_macro factory method."""
        from unittest.mock import Mock

        mock_command1 = Mock()
        mock_command2 = Mock()
        commands = [mock_command1, mock_command2]

        test_macro = MacroDefinition.create_test_macro("Test Macro", commands)

        assert test_macro.name == "Test Macro"
        assert test_macro.commands == commands
        assert test_macro.enabled is True  # Default value
        assert test_macro.group_id is None  # Default value
        assert isinstance(test_macro.macro_id, str)  # Should be a UUID string

    def test_execution_result_creation(self) -> None:
        """Test ExecutionResult dataclass creation."""
        result = ExecutionResult(
            execution_token=ExecutionToken("token-123"),
            macro_id=MacroId("macro-456"),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(),
        )
        assert result.execution_token == ExecutionToken("token-123")
        assert result.macro_id == MacroId("macro-456")
        assert result.status == ExecutionStatus.COMPLETED

    def test_execution_result_success_check(self) -> None:
        """Test ExecutionResult success checking."""
        success_result = CommandResult.success_result("Success")
        failure_result = CommandResult.failure_result("Failed")

        # Test successful execution
        exec_result = ExecutionResult(
            execution_token=ExecutionToken("token-1"),
            macro_id=MacroId("macro-1"),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(),
            command_results=[success_result],
        )
        assert exec_result.is_successful()

        # Test failed execution (wrong status)
        exec_result_failed = ExecutionResult(
            execution_token=ExecutionToken("token-2"),
            macro_id=MacroId("macro-2"),
            status=ExecutionStatus.FAILED,
            started_at=datetime.now(),
            command_results=[success_result],
        )
        assert not exec_result_failed.is_successful()

        # Test failed execution (failed command)
        exec_result_cmd_failed = ExecutionResult(
            execution_token=ExecutionToken("token-3"),
            macro_id=MacroId("macro-3"),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(),
            command_results=[failure_result],
        )
        assert not exec_result_cmd_failed.is_successful()

    def test_execution_result_has_error_info(self) -> None:
        """Test ExecutionResult.has_error_info method."""
        success_result = CommandResult.success_result("Success")
        failure_result = CommandResult.failure_result("Failed")

        # Test no error info
        clean_result = ExecutionResult(
            execution_token=ExecutionToken("token-1"),
            macro_id=MacroId("macro-1"),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(),
            command_results=[success_result],
            error_details=None,
        )
        assert not clean_result.has_error_info()

        # Test has error details
        error_details_result = ExecutionResult(
            execution_token=ExecutionToken("token-2"),
            macro_id=MacroId("macro-2"),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(),
            command_results=[success_result],
            error_details="Some error occurred",
        )
        assert error_details_result.has_error_info()

        # Test has failed command results
        failed_cmd_result = ExecutionResult(
            execution_token=ExecutionToken("token-3"),
            macro_id=MacroId("macro-3"),
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now(),
            command_results=[failure_result],
            error_details=None,
        )
        assert failed_cmd_result.has_error_info()

    def test_validation_result_creation(self) -> None:
        """Test ValidationResult dataclass creation."""
        result = ValidationResult(is_valid=True, errors=[], warnings=["Minor warning"])
        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_validation_result_factory_methods(self) -> None:
        """Test ValidationResult factory methods."""
        success_result = ValidationResult.success(extra_info="metadata")
        assert success_result.is_valid is True
        assert success_result.metadata["extra_info"] == "metadata"

        failure_result = ValidationResult.failure(
            errors=["Error 1", "Error 2"], context="test"
        )
        assert failure_result.is_valid is False
        assert len(failure_result.errors) == 2
        assert failure_result.metadata["context"] == "test"

    def test_validation_result_add_error(self) -> None:
        """Test ValidationResult.add_error method."""
        result = ValidationResult(is_valid=True)
        new_result = result.add_error("New error")

        # Original should be unchanged
        assert result.is_valid is True
        assert len(result.errors) == 0

        # New result should have error and be invalid
        assert new_result.is_valid is False
        assert "New error" in new_result.errors

    def test_macro_move_result_creation(self) -> None:
        """Test MacroMoveResult dataclass creation."""
        result = MacroMoveResult(
            macro_id=MacroId("move-macro"),
            source_group=GroupId("source"),
            target_group=GroupId("target"),
            execution_time=Duration.from_seconds(0.5),
            conflicts_resolved=["conflict1"],
            rollback_info=None,
        )
        assert result.macro_id == MacroId("move-macro")
        assert result.was_successful()  # No rollback info means success

        # Test failed move
        failed_result = MacroMoveResult(
            macro_id=MacroId("failed-macro"),
            source_group=GroupId("source"),
            target_group=GroupId("target"),
            execution_time=Duration.from_seconds(0.1),
            rollback_info={"reason": "permission denied"},
        )
        assert not failed_result.was_successful()  # Has rollback info means failure
