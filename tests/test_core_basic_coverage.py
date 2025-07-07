"""Basic coverage tests for core modules to improve overall coverage.

These tests focus on increasing test coverage by testing basic functionality
that doesn't require complex setup or external dependencies.
"""

from __future__ import annotations

from typing import Any, Optional
import pytest

# Import core modules for basic testing
try:
    from src.core.types import (
        CommandId,
        ExecutionToken,
        GroupId,
        TemplateId,
        TriggerId,
        VariableName,
        create_macro_id,
    )

    TYPES_AVAILABLE = True
except ImportError:
    TYPES_AVAILABLE = False

try:
    from src.core.either import Left, Right

    EITHER_AVAILABLE = True
except ImportError:
    EITHER_AVAILABLE = False

try:
    from src.core.errors import ErrorCategory, ErrorContext, ErrorSeverity

    ERRORS_AVAILABLE = True
except ImportError:
    ERRORS_AVAILABLE = False


class TestCoreTypes:
    """Test core type creation and validation."""

    @pytest.mark.skipif(not TYPES_AVAILABLE, reason="Core types module not available")
    def test_macro_id_creation(self) -> None:
        """Test MacroId creation and validation."""
        macro_id = create_macro_id()
        # MacroId is a NewType, so check it's a string with UUID format
        assert isinstance(macro_id, str)
        assert len(str(macro_id)) > 0
        # Should be UUID format
        import uuid

        uuid.UUID(macro_id)  # This will raise ValueError if invalid UUID

    @pytest.mark.skipif(not TYPES_AVAILABLE, reason="Core types module not available")
    def test_type_creation(self) -> None:
        """Test basic type creation."""
        # Test branded type creation - NewTypes are just their underlying types at runtime
        command_id = CommandId("test-command")
        import secrets

        exec_token = ExecutionToken(f"test_token_{secrets.token_hex(8)}")
        trigger_id = TriggerId("test-trigger")
        variable_name = VariableName("test_var")
        template_id = TemplateId("test-template")

        # NewType creates aliases, so test the underlying type
        assert isinstance(command_id, str)
        assert isinstance(exec_token, str)
        assert isinstance(trigger_id, str)
        assert isinstance(variable_name, str)
        assert isinstance(template_id, str)

        # Test values are preserved
        assert command_id == "test-command"
        assert exec_token.startswith("test_token_")
        assert trigger_id == "test-trigger"

    @pytest.mark.skipif(not TYPES_AVAILABLE, reason="Core types module not available")
    def test_group_id_type(self) -> None:
        """Test GroupId type."""
        group_id = GroupId("test-group")
        # NewType creates aliases, so test the underlying type
        assert isinstance(group_id, str)
        assert str(group_id) == "test-group"
        assert group_id == "test-group"


class TestEitherMonad:
    """Test Either monad functionality."""

    @pytest.mark.skipif(not EITHER_AVAILABLE, reason="Either module not available")
    def test_right_creation(self) -> None:
        """Test Right value creation."""
        right = Right("success")
        assert right.is_right()
        assert not right.is_left()
        assert right.get_right() == "success"

    @pytest.mark.skipif(not EITHER_AVAILABLE, reason="Either module not available")
    def test_left_creation(self) -> None:
        """Test Left value creation."""
        left = Left("error")
        assert left.is_left()
        assert not left.is_right()
        assert left.get_left() == "error"

    @pytest.mark.skipif(not EITHER_AVAILABLE, reason="Either module not available")
    def test_either_map_right(self) -> None:
        """Test Either map function with Right value."""
        right = Right(5)
        mapped = right.map(lambda x: x * 2)
        assert mapped.is_right()
        assert mapped.get_right() == 10

    @pytest.mark.skipif(not EITHER_AVAILABLE, reason="Either module not available")
    def test_either_map_left(self) -> None:
        """Test Either map function with Left value."""
        left = Left("error")
        mapped = left.map(lambda x: x * 2)
        assert mapped.is_left()
        assert mapped.get_left() == "error"

    @pytest.mark.skipif(not EITHER_AVAILABLE, reason="Either module not available")
    def test_either_flat_map_right(self) -> None:
        """Test Either flat_map function with Right value."""
        right = Right(5)
        flat_mapped = right.flat_map(lambda x: Right(x * 2))
        assert flat_mapped.is_right()
        assert flat_mapped.get_right() == 10

    @pytest.mark.skipif(not EITHER_AVAILABLE, reason="Either module not available")
    def test_either_flat_map_left(self) -> None:
        """Test Either flat_map function with Left value."""
        left = Left("error")
        flat_mapped = left.flat_map(lambda x: Right(x * 2))
        assert flat_mapped.is_left()
        assert flat_mapped.get_left() == "error"

    @pytest.mark.skipif(not EITHER_AVAILABLE, reason="Either module not available")
    def test_either_get_or_else_right(self) -> None:
        """Test Either get_or_else with Right value."""
        right = Right("success")
        result = right.get_or_else("default")
        assert result == "success"

    @pytest.mark.skipif(not EITHER_AVAILABLE, reason="Either module not available")
    def test_either_get_or_else_left(self) -> None:
        """Test Either get_or_else with Left value."""
        left = Left("error")
        result = left.get_or_else("default")
        assert result == "default"


class TestErrorTypes:
    """Test error type creation and handling."""

    @pytest.mark.skipif(not ERRORS_AVAILABLE, reason="Errors module not available")
    def test_error_category_enum(self) -> None:
        """Test ErrorCategory enum."""
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.SECURITY.value == "security"
        assert ErrorCategory.EXECUTION.value == "execution"
        assert ErrorCategory.TIMEOUT.value == "timeout"

    @pytest.mark.skipif(not ERRORS_AVAILABLE, reason="Errors module not available")
    def test_error_severity_enum(self) -> None:
        """Test ErrorSeverity enum."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    @pytest.mark.skipif(not ERRORS_AVAILABLE, reason="Errors module not available")
    def test_error_context_creation(self) -> None:
        """Test ErrorContext creation."""
        context = ErrorContext(operation="test_operation", component="test_component")
        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.timestamp is not None
        assert isinstance(context.metadata, dict)

    @pytest.mark.skipif(not ERRORS_AVAILABLE, reason="Errors module not available")
    def test_error_context_with_metadata(self) -> None:
        """Test ErrorContext with metadata."""
        context = ErrorContext(operation="test_operation", component="test_component")
        new_context = context.with_metadata(param="value", user_id="user123")
        assert new_context.metadata["param"] == "value"
        assert new_context.metadata["user_id"] == "user123"


class TestCoreLogging:
    """Test core logging functionality."""

    def test_logging_import(self) -> None:
        """Test that logging module can be imported."""
        try:
            from src.core.logging import get_logger

            logger = get_logger("test")
            assert logger is not None
        except ImportError:
            pytest.skip("Logging module not available")


class TestCoreDataStructures:
    """Test core data structures."""

    def test_data_structures_import(self) -> None:
        """Test that data structures can be imported."""
        try:
            from src.core.data_structures import Cache, Queue, Stack

            # Test basic instantiation
            stack = Stack()
            queue = Queue()
            cache = Cache(max_size=10)
            assert stack is not None
            assert queue is not None
            assert cache is not None
        except ImportError:
            pytest.skip("Data structures module not available")


class TestCoreContracts:
    """Test core contract functionality."""

    def test_contracts_import(self) -> None:
        """Test that contracts module can be imported."""
        try:
            from src.core.contracts import ensure, invariant, require

            # Test basic contract decorators exist
            assert callable(require)
            assert callable(ensure)
            assert callable(invariant)
        except ImportError:
            pytest.skip("Contracts module not available")


if __name__ == "__main__":
    pytest.main([__file__])
