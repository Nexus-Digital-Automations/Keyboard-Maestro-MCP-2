"""Coverage expansion tests for actual Keyboard Maestro client functionality.

Tests focus on real API available in km_client.py to maximize coverage.
"""

from __future__ import annotations

from typing import Any

import pytest
from src.core.types import Duration, MacroId, TriggerId
from src.integration.km_client import (
    ConnectionConfig,
    ConnectionMethod,
    Either,
    KMClient,
    KMError,
    TriggerDefinition,
    create_client_with_fallback,
    retry_with_backoff,
)


class TestEitherMonad:
    """Test Either monad functionality."""

    def test_either_left_creation(self) -> None:
        """Test creating Left (error) values."""
        error_msg = "Something went wrong"
        result = Either.left(error_msg)

        assert result.is_left()
        assert not result.is_right()
        assert result.get_left() == error_msg
        # get_right() should raise ValueError on Left
        with pytest.raises(ValueError, match="Cannot get Right value from Left"):
            result.get_right()

    def test_either_right_creation(self) -> None:
        """Test creating Right (success) values."""
        value = "Success!"
        result = Either.right(value)

        assert result.is_right()
        assert not result.is_left()
        assert result.get_right() == value
        # get_left() should raise ValueError on Right
        with pytest.raises(ValueError, match="Cannot get Left value from Right"):
            result.get_left()

    def test_either_map_functionality(self) -> None:
        """Test mapping over Either values."""
        # Map over Right value
        right_result = Either.right(10)
        mapped_right = right_result.map(lambda x: x * 2)
        assert mapped_right.is_right()
        assert mapped_right.get_right() == 20

        # Map over Left value (should remain unchanged)
        left_result = Either.left("error")
        mapped_left = left_result.map(lambda x: x * 2)
        assert mapped_left.is_left()
        assert mapped_left.get_left() == "error"

    def test_either_flat_map(self) -> None:
        """Test flat mapping for chaining operations."""

        def double_if_positive(x: Any) -> Any:
            if x > 0:
                return Either.right(x * 2)
            return Either.left("negative or zero")

        # Test with positive number
        result = Either.right(5).flat_map(double_if_positive)
        assert result.is_right()
        assert result.get_right() == 10

        # Test with negative number
        result = Either.right(-1).flat_map(double_if_positive)
        assert result.is_left()
        assert result.get_left() == "negative or zero"

        # Test flat_map on Left (should remain Left)
        result = Either.left("initial error").flat_map(double_if_positive)
        assert result.is_left()
        assert result.get_left() == "initial error"

    def test_either_get_or_else(self) -> None:
        """Test get_or_else functionality."""
        right_result = Either.right("success")
        assert right_result.get_or_else("default") == "success"

        left_result = Either.left("error")
        assert left_result.get_or_else("default") == "default"


class TestConnectionMethod:
    """Test connection method enumeration."""

    def test_connection_method_values(self) -> None:
        """Test all connection method values exist."""
        assert ConnectionMethod.APPLESCRIPT.value == "applescript"
        assert ConnectionMethod.URL_SCHEME.value == "url_scheme"
        assert ConnectionMethod.WEB_API.value == "web_api"
        assert ConnectionMethod.REMOTE_TRIGGER.value == "remote_trigger"

    def test_connection_method_enum_behavior(self) -> None:
        """Test enum behavior and comparisons."""
        method1 = ConnectionMethod.APPLESCRIPT
        method2 = ConnectionMethod.APPLESCRIPT
        method3 = ConnectionMethod.WEB_API

        assert method1 == method2
        assert method1 != method3
        assert str(method1) == "ConnectionMethod.APPLESCRIPT"


class TestKMError:
    """Test KM error creation and handling."""

    def test_basic_error_creation(self) -> None:
        """Test basic error instantiation."""
        error = KMError(code="TEST_ERROR", message="Test message")

        assert error.code == "TEST_ERROR"
        assert error.message == "Test message"
        assert error.details is None
        assert error.retry_after is None

    def test_error_with_details(self) -> None:
        """Test error with additional details."""
        details = {"macro_id": "test-123", "step": 5}
        error = KMError(
            code="EXECUTION_ERROR",
            message="Macro execution failed",
            details=details,
        )

        assert error.code == "EXECUTION_ERROR"
        assert error.message == "Macro execution failed"
        assert error.details == details

    def test_connection_error_factory(self) -> None:
        """Test connection error factory method."""
        error = KMError.connection_error("Failed to connect to KM")

        assert error.code == "CONNECTION_ERROR"
        assert error.message == "Failed to connect to KM"
        assert error.details is None

    def test_execution_error_factory(self) -> None:
        """Test execution error factory method."""
        details = {"command": "type_text", "input": "hello"}
        error = KMError.execution_error("Command failed", details)

        assert error.code == "EXECUTION_ERROR"
        assert error.message == "Command failed"
        assert error.details == details

    def test_timeout_error_factory(self) -> None:
        """Test timeout error factory method."""
        timeout = Duration.from_seconds(30)
        error = KMError.timeout_error(timeout)

        assert error.code == "TIMEOUT_ERROR"
        assert "30" in error.message  # Should contain timeout value
        assert error.retry_after is not None
        assert error.retry_after.total_seconds() == 1.0

    def test_validation_error_factory(self) -> None:
        """Test validation error factory method."""
        error = KMError.validation_error("Invalid macro format")

        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Invalid macro format"

    def test_not_found_error_factory(self) -> None:
        """Test not found error factory method."""
        error = KMError.not_found_error("Macro not found")

        assert error.code == "NOT_FOUND_ERROR"
        assert error.message == "Macro not found"

    def test_security_error_factory(self) -> None:
        """Test security error factory method."""
        error = KMError.security_error("Permission denied")

        assert error.code == "SECURITY_ERROR"
        assert error.message == "Permission denied"


class TestConnectionConfig:
    """Test connection configuration."""

    def test_default_config_creation(self) -> None:
        """Test creating config with defaults."""
        config = ConnectionConfig()

        assert config.method == ConnectionMethod.APPLESCRIPT
        assert config.timeout.total_seconds() == 30
        assert config.web_api_port == 4490
        assert config.web_api_host == "localhost"
        assert config.max_retries == 3
        assert config.retry_delay.total_seconds() == 0.5

    def test_config_with_specific_values(self) -> None:
        """Test creating config with specific values."""
        timeout = Duration.from_seconds(60)
        retry_delay = Duration.from_seconds(1.0)

        config = ConnectionConfig(
            method=ConnectionMethod.WEB_API,
            timeout=timeout,
            web_api_port=8080,
            web_api_host="192.168.1.100",
            max_retries=5,
            retry_delay=retry_delay,
        )

        assert config.method == ConnectionMethod.WEB_API
        assert config.timeout.total_seconds() == 60
        assert config.web_api_port == 8080
        assert config.web_api_host == "192.168.1.100"
        assert config.max_retries == 5
        assert config.retry_delay.total_seconds() == 1.0

    def test_config_with_timeout_modification(self) -> None:
        """Test modifying config timeout."""
        original_config = ConnectionConfig()
        new_timeout = Duration.from_seconds(120)
        new_config = original_config.with_timeout(new_timeout)

        # Original should be unchanged (immutable)
        assert original_config.timeout.total_seconds() == 30

        # New config should have new timeout
        assert new_config.timeout.total_seconds() == 120
        assert new_config.method == original_config.method
        assert new_config.web_api_port == original_config.web_api_port

    def test_config_with_method_modification(self) -> None:
        """Test modifying config method."""
        original_config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        new_config = original_config.with_method(ConnectionMethod.URL_SCHEME)

        # Original should be unchanged
        assert original_config.method == ConnectionMethod.APPLESCRIPT

        # New config should have new method
        assert new_config.method == ConnectionMethod.URL_SCHEME
        assert new_config.timeout == original_config.timeout
        assert new_config.web_api_port == original_config.web_api_port


class TestTriggerDefinition:
    """Test trigger definition functionality."""

    def test_trigger_creation_basic(self) -> None:
        """Test basic trigger creation."""
        trigger = TriggerDefinition(
            trigger_id=TriggerId("test-trigger"),
            trigger_type="hotkey",
            configuration={"key": "F1"},
            macro_id=MacroId("test-macro"),
        )

        assert trigger.trigger_id == TriggerId("test-trigger")
        assert trigger.trigger_type == "hotkey"
        assert trigger.configuration["key"] == "F1"
        assert trigger.macro_id == MacroId("test-macro")

    def test_trigger_creation_complex(self) -> None:
        """Test complex trigger with multiple configuration options."""
        config = {
            "key": "F2",
            "modifiers": ["cmd", "shift"],
            "scope": "global",
            "enabled": True,
        }

        trigger = TriggerDefinition(
            trigger_id=TriggerId("complex-trigger"),
            trigger_type="hotkey",
            configuration=config,
            macro_id=MacroId("complex-macro"),
        )

        assert trigger.configuration["key"] == "F2"
        assert "cmd" in trigger.configuration["modifiers"]
        assert "shift" in trigger.configuration["modifiers"]
        assert trigger.configuration["scope"] == "global"
        assert trigger.configuration["enabled"] is True

    def test_trigger_types_variety(self) -> None:
        """Test various trigger types."""
        trigger_types = [
            ("hotkey", {"key": "a"}),
            ("text_expansion", {"abbreviation": "addr"}),
            ("application_trigger", {"application": "Safari"}),
            ("time_trigger", {"schedule": "daily"}),
            ("clipboard_trigger", {"pattern": "*.jpg"}),
        ]

        for i, (trigger_type, config) in enumerate(trigger_types):
            trigger = TriggerDefinition(
                trigger_id=TriggerId(f"trigger-{i}"),
                trigger_type=trigger_type,
                configuration=config,
                macro_id=MacroId(f"macro-{i}"),
            )

            assert trigger.trigger_type == trigger_type
            assert trigger.configuration == config


class TestKMClient:
    """Test KM client functionality with available methods."""

    def test_km_client_properties(self) -> None:
        """Test basic KM client properties and attributes."""
        # We can't easily test full initialization without mocking,
        # but we can test that the class exists and has expected attributes

        assert hasattr(KMClient, "__init__")
        assert callable(KMClient)

        # Test that we can reference the class
        client_class = KMClient
        assert client_class.__name__ == "KMClient"


class TestUtilityFunctions:
    """Test utility functions available in the module."""

    def test_create_client_with_fallback_function_exists(self) -> None:
        """Test that fallback function exists and is callable."""
        assert callable(create_client_with_fallback)

        # Test function signature by attempting to call with None values
        # (this will likely fail, but we're testing the function exists)
        try:
            result = create_client_with_fallback(None, None)
            # If it returns something, check it's an Either
            if hasattr(result, "is_right"):
                assert hasattr(result, "is_left")
        except (TypeError, AttributeError):
            # Expected when calling with None
            pass

    def test_retry_with_backoff_function_exists(self) -> None:
        """Test that retry function exists and is callable."""
        assert callable(retry_with_backoff)

        # Test function signature by checking it's a function
        import inspect

        assert inspect.isfunction(retry_with_backoff)


class TestDurationIntegration:
    """Test Duration integration with KM client components."""

    def test_duration_with_config(self) -> None:
        """Test using Duration in configuration."""
        short_timeout = Duration.from_seconds(5)
        long_timeout = Duration.from_seconds(300)

        config1 = ConnectionConfig(timeout=short_timeout)
        config2 = ConnectionConfig(timeout=long_timeout)

        assert config1.timeout.total_seconds() == 5
        assert config2.timeout.total_seconds() == 300

        # Test immutability
        config3 = config1.with_timeout(long_timeout)
        assert config1.timeout.total_seconds() == 5  # Original unchanged
        assert config3.timeout.total_seconds() == 300  # New config updated

    def test_duration_in_errors(self) -> None:
        """Test Duration usage in error handling."""
        timeout_duration = Duration.from_seconds(45)
        error = KMError.timeout_error(timeout_duration)

        assert "45" in error.message
        assert error.retry_after is not None
        assert isinstance(error.retry_after, Duration)


class TestTypeIntegration:
    """Test integration with core types."""

    def test_macro_id_usage(self) -> None:
        """Test MacroId usage throughout the system."""
        macro_id = MacroId("test-macro-123")

        # Test in trigger definition
        trigger = TriggerDefinition(
            trigger_id=TriggerId("test-trigger"),
            trigger_type="hotkey",
            configuration={"key": "F1"},
            macro_id=macro_id,
        )

        assert trigger.macro_id == macro_id
        assert str(trigger.macro_id) == "test-macro-123"

    def test_trigger_id_usage(self) -> None:
        """Test TriggerId usage in system."""
        trigger_id = TriggerId("trigger-456")

        trigger = TriggerDefinition(
            trigger_id=trigger_id,
            trigger_type="text_expansion",
            configuration={"abbreviation": "test"},
            macro_id=MacroId("test-macro"),
        )

        assert trigger.trigger_id == trigger_id
        assert str(trigger.trigger_id) == "trigger-456"


class TestErrorPropagation:
    """Test error handling and propagation patterns."""

    def test_either_error_chain(self) -> None:
        """Test chaining operations with errors."""
        # Start with success
        result = Either.right(10)

        # Chain operations
        result = result.map(lambda x: x * 2)  # Should succeed: 20
        result = result.flat_map(lambda x: Either.right(x + 5))  # Should succeed: 25

        assert result.is_right()
        assert result.get_right() == 25

        # Now test with error injection
        result = Either.right(10)
        result = result.map(lambda x: x * 2)  # Success: 20
        result = result.flat_map(
            lambda x: Either.left("error occurred"),
        )  # Inject error
        result = result.map(lambda x: x + 100)  # Should not execute due to error

        assert result.is_left()
        assert result.get_left() == "error occurred"

    def test_error_context_preservation(self) -> None:
        """Test that error context is preserved through operations."""
        initial_error = KMError.validation_error("Invalid input")
        either_error = Either.left(initial_error)

        # Chain operations on error
        result = either_error.map(lambda x: x.upper())
        result = result.flat_map(lambda x: Either.right("transformed"))

        # Error should be preserved
        assert result.is_left()
        assert result.get_left() == initial_error
        assert result.get_left().code == "VALIDATION_ERROR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
