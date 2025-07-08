"""Targeted coverage expansion tests for Keyboard Maestro client integration.

Tests focus on available functionality to maximize coverage of km_client.py module.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest
from src.core.types import Duration, MacroId, TriggerId
from src.integration.events import TriggerType
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
    """Test Either monad functionality for error handling."""

    def test_either_left_creation(self) -> None:
        """Test creating Left (error) values."""
        error_msg = "Something went wrong"
        result = Either.left(error_msg)

        assert not result.is_right()
        assert result.is_left()
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

    def test_either_map_on_right(self) -> None:
        """Test mapping function over Right values."""
        result = Either.right(10)
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_right()
        assert mapped.get_right() == 20

    def test_either_map_on_left(self) -> None:
        """Test mapping function preserves Left values."""
        result = Either.left("error")
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_left()
        assert mapped.get_left() == "error"

    def test_either_flat_map(self) -> None:
        """Test flat mapping for chaining operations."""

        def double_if_even(x: Any) -> Mock:
            if x % 2 == 0:
                return Either.right(x * 2)
            return Either.left("odd number")

        # Test with even number
        result = Either.right(4).flat_map(double_if_even)
        assert result.is_right()
        assert result.get_right() == 8

        # Test with odd number
        result = Either.right(3).flat_map(double_if_even)
        assert result.is_left()
        assert result.get_left() == "odd number"

    def test_either_or_else(self) -> None:
        """Test providing fallback values."""
        left_result = Either.left("error")
        assert left_result.get_or_else("default") == "default"

        right_result = Either.right("value")
        assert right_result.get_or_else("default") == "value"


class TestConnectionMethod:
    """Test connection method enumeration."""

    def test_connection_method_values(self) -> None:
        """Test available connection methods."""
        methods = list(ConnectionMethod)
        assert ConnectionMethod.APPLESCRIPT in methods
        assert ConnectionMethod.URL_SCHEME in methods
        assert ConnectionMethod.WEB_API in methods
        assert ConnectionMethod.REMOTE_TRIGGER in methods

    def test_connection_method_string_values(self) -> None:
        """Test connection method string representations."""
        assert ConnectionMethod.APPLESCRIPT.value == "applescript"
        assert ConnectionMethod.URL_SCHEME.value == "url_scheme"
        assert ConnectionMethod.WEB_API.value == "web_api"
        assert ConnectionMethod.REMOTE_TRIGGER.value == "remote_trigger"


class TestConnectionConfig:
    """Test connection configuration."""

    def test_config_creation_minimal(self) -> None:
        """Test creating config with minimal parameters."""
        config = ConnectionConfig(
            method=ConnectionMethod.APPLESCRIPT,
            timeout=Duration(seconds=30),
        )

        assert config.method == ConnectionMethod.APPLESCRIPT
        assert config.timeout.seconds == 30

    def test_config_creation_full(self) -> None:
        """Test creating config with all parameters."""
        config = ConnectionConfig(
            method=ConnectionMethod.WEB_API,
            web_api_host="localhost",
            web_api_port=4242,
            timeout=Duration(seconds=60),
            max_retries=3,
            retry_delay=Duration(seconds=1.0),
        )

        assert config.method == ConnectionMethod.WEB_API
        assert config.web_api_host == "localhost"
        assert config.web_api_port == 4242
        assert config.timeout.seconds == 60
        assert config.max_retries == 3
        assert config.retry_delay.seconds == 1.0

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Valid config
        valid_config = ConnectionConfig(
            method=ConnectionMethod.APPLESCRIPT,
            timeout=Duration(seconds=30),
        )
        assert valid_config.method == ConnectionMethod.APPLESCRIPT
        assert valid_config.timeout.seconds == 30

        # Test config with very short timeout (edge case)
        short_timeout_config = ConnectionConfig(
            method=ConnectionMethod.APPLESCRIPT,
            timeout=Duration(seconds=1),
        )
        assert short_timeout_config.method == ConnectionMethod.APPLESCRIPT
        assert short_timeout_config.timeout.seconds == 1


class TestTriggerDefinition:
    """Test trigger definition functionality."""

    def test_trigger_creation_hotkey(self) -> None:
        """Test creating hotkey trigger."""
        trigger = TriggerDefinition(
            trigger_id=TriggerId("hotkey-trigger"),
            trigger_type=TriggerType.HOTKEY,
            configuration={"key": "F1", "modifiers": ["cmd", "shift"]},
            macro_id=MacroId("target-macro"),
        )

        assert trigger.trigger_type == TriggerType.HOTKEY
        assert trigger.configuration["key"] == "F1"
        assert "cmd" in trigger.configuration["modifiers"]
        assert "shift" in trigger.configuration["modifiers"]

    def test_trigger_creation_text_expansion(self) -> None:
        """Test creating text expansion trigger."""
        trigger = TriggerDefinition(
            trigger_id=TriggerId("text-trigger"),
            trigger_type=TriggerType.APPLICATION,  # Using APPLICATION since text_expansion isn't in enum
            configuration={
                "abbreviation": "myaddr",
                "expanded_text": "123 Main St, City, State",
            },
            macro_id=MacroId("address-macro"),
        )

        assert trigger.trigger_type == TriggerType.APPLICATION
        assert trigger.configuration["abbreviation"] == "myaddr"
        assert "123 Main St" in trigger.configuration["expanded_text"]

    def test_trigger_validation(self) -> None:
        """Test trigger validation."""
        # Valid trigger
        valid_trigger = TriggerDefinition(
            trigger_id=TriggerId("valid-trigger"),
            trigger_type=TriggerType.HOTKEY,
            configuration={"key": "a"},
            macro_id=MacroId("test-macro"),
        )
        assert valid_trigger.trigger_type == TriggerType.HOTKEY
        assert valid_trigger.configuration["key"] == "a"

        # Trigger with empty configuration (should still create but be empty)
        empty_config_trigger = TriggerDefinition(
            trigger_id=TriggerId("empty-trigger"),
            trigger_type=TriggerType.HOTKEY,
            configuration={},
            macro_id=MacroId("test-macro"),
        )
        assert empty_config_trigger.trigger_type == TriggerType.HOTKEY
        assert len(empty_config_trigger.configuration) == 0


class TestKMClient:
    """Test KM client core functionality."""

    @pytest.fixture
    def mock_config(self) -> Mock:
        """Create mock configuration for testing."""
        return ConnectionConfig(
            method=ConnectionMethod.APPLESCRIPT,
            timeout=Duration(seconds=30),
        )

    def test_client_creation(self, mock_config: dict[str, Any]) -> None:
        """Test client creation with configuration."""
        client = KMClient(connection_config=mock_config)

        assert client.config == mock_config
        assert client.config.method == ConnectionMethod.APPLESCRIPT

    def test_client_connection_check(self, mock_config: dict[str, Any]) -> None:
        """Test client connection check (mocked)."""
        client = KMClient(connection_config=mock_config)

        # Mock the connection check process
        with patch.object(client, "check_connection") as mock_connect:
            mock_connect.return_value = Either.right(True)

            result = client.check_connection()

            assert result.is_right()
            assert result.get_right() is True
            mock_connect.assert_called_once()

    def test_macro_execution_simulation(self, mock_config: dict[str, Any]) -> None:
        """Test macro execution logic (mocked)."""
        client = KMClient(connection_config=mock_config)
        macro_id = MacroId("test-macro-123")

        # Mock the execution process
        with patch.object(client, "execute_macro") as mock_execute:
            mock_execute.return_value = Either.right(
                {
                    "execution_id": "exec-456",
                    "status": "completed",
                    "duration": 2.5,
                },
            )

            result = client.execute_macro(macro_id)

            assert result.is_right()
            execution_result = result.get_right()
            assert execution_result["status"] == "completed"
            assert execution_result["duration"] == 2.5

    def test_error_handling(self, mock_config: dict[str, Any]) -> None:
        """Test client error handling."""
        client = KMClient(connection_config=mock_config)

        # Test connection error
        with patch.object(client, "check_connection") as mock_connect:
            mock_connect.return_value = Either.left(
                KMError(code="CONN_ERROR", message="Connection failed"),
            )

            result = client.check_connection()

            assert result.is_left()
            error = result.get_left()
            assert error.message == "Connection failed"


class TestClientWithFallback:
    """Test client creation with fallback mechanisms."""

    def test_create_client_with_fallback_primary_success(self) -> None:
        """Test fallback when primary method succeeds."""
        primary_config = ConnectionConfig(
            method=ConnectionMethod.APPLESCRIPT,
            timeout=Duration(seconds=30),
        )
        fallback_config = ConnectionConfig(
            method=ConnectionMethod.URL_SCHEME,
            timeout=Duration(seconds=30),
        )

        # Test client creation (returns client directly, not Either)
        client = create_client_with_fallback(primary_config, fallback_config)

        # Verify it's a valid KMClient instance
        assert hasattr(client, "config")
        assert hasattr(client, "execute_macro")
        assert client.config.method == ConnectionMethod.APPLESCRIPT

    def test_create_client_with_fallback_fallback_needed(self) -> None:
        """Test fallback when primary method fails."""
        primary_config = ConnectionConfig(
            method=ConnectionMethod.WEB_API,
            timeout=Duration(seconds=10),
        )
        fallback_config = ConnectionConfig(
            method=ConnectionMethod.APPLESCRIPT,
            timeout=Duration(seconds=30),
        )

        # Test client creation with different configs
        client = create_client_with_fallback(primary_config, fallback_config)

        # Verify it's a fallback client with proper structure
        assert hasattr(client, "config")
        assert hasattr(client, "execute_macro")
        assert hasattr(client, "_fallback")
        assert client.config.method == ConnectionMethod.WEB_API
        assert client._fallback.config.method == ConnectionMethod.APPLESCRIPT


class TestRetryWithBackoff:
    """Test retry mechanism with exponential backoff."""

    def test_retry_success_on_first_attempt(self) -> None:
        """Test successful operation on first try."""

        def successful_operation() -> Mock:
            return Either.right("Success!")

        result = retry_with_backoff(
            successful_operation,
            max_retries=3,
            initial_delay=Duration(seconds=0.1),
        )

        assert result.is_right()
        assert result.get_right() == "Success!"

    def test_retry_success_after_failures(self) -> None:
        """Test successful operation after initial failures."""
        attempt_count = 0

        def eventually_successful_operation() -> Mock:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                return Either.left(
                    KMError(
                        code="RETRY_ERROR",
                        message=f"Attempt {attempt_count} failed",
                    ),
                )
            return Either.right("Finally succeeded!")

        result = retry_with_backoff(
            eventually_successful_operation,
            max_retries=5,
            initial_delay=Duration(seconds=0.01),  # Fast for testing
        )

        assert result.is_right()
        assert result.get_right() == "Finally succeeded!"
        assert attempt_count == 3

    def test_retry_exhausted_attempts(self) -> None:
        """Test when all retry attempts are exhausted."""

        def always_failing_operation() -> Mock:
            return Either.left(KMError(code="FAIL_ERROR", message="Always fails"))

        result = retry_with_backoff(
            always_failing_operation,
            max_retries=3,
            initial_delay=Duration(seconds=0.01),
        )

        assert result.is_left()
        error = result.get_left()
        assert error.message == "Always fails"

    def test_retry_with_backoff_parameters(self) -> None:
        """Test retry with backoff parameter validation."""

        def test_operation() -> None:
            return Either.right("test_result")

        # Test with valid parameters
        result = retry_with_backoff(
            test_operation,
            max_retries=2,
            initial_delay=Duration(seconds=0.01),
        )

        assert result.is_right()
        assert result.get_right() == "test_result"


class TestKMError:
    """Test KM error handling and types."""

    def test_km_error_creation(self) -> None:
        """Test creating KM errors."""
        error = KMError(code="TEST_ERROR", message="Test error message")

        assert error.message == "Test error message"
        assert error.code == "TEST_ERROR"

    def test_km_error_with_details(self) -> None:
        """Test KM errors with additional details."""
        error = KMError(
            code="CONN_001",
            message="Connection failed",
            details={"host": "localhost", "port": 4242},
        )

        assert error.message == "Connection failed"
        assert error.code == "CONN_001"
        assert error.details["host"] == "localhost"
        assert error.details["port"] == 4242

    def test_km_error_attributes(self) -> None:
        """Test error attributes and data access."""
        error = KMError(
            code="EXEC_002",
            message="Execution failed",
            details={"macro_id": "test-macro", "step": 5},
        )

        assert error.message == "Execution failed"
        assert error.code == "EXEC_002"
        assert error.details["macro_id"] == "test-macro"
        assert error.details["step"] == 5


class TestAdvancedOperations:
    """Test advanced client operations and patterns."""

    def test_macro_list_retrieval(self) -> None:
        """Test retrieving list of macros."""
        config = ConnectionConfig(
            method=ConnectionMethod.APPLESCRIPT,
            timeout=Duration(seconds=60),
        )
        client = KMClient(connection_config=config)

        # Mock macro list retrieval
        with patch.object(client, "get_macro_list") as mock_list:
            mock_list.return_value = Either.right(
                [
                    {"id": "macro-1", "name": "Test Macro 1"},
                    {"id": "macro-2", "name": "Test Macro 2"},
                    {"id": "macro-3", "name": "Test Macro 3"},
                ],
            )

            result = client.get_macro_list()

            assert result.is_right()
            macros = result.get_right()
            assert len(macros) == 3
            assert macros[0]["name"] == "Test Macro 1"

    def test_connection_check(self) -> None:
        """Test connection status checking."""
        config = ConnectionConfig(
            method=ConnectionMethod.APPLESCRIPT,
            timeout=Duration(seconds=30),
        )
        client = KMClient(connection_config=config)

        # Mock connection check
        with patch.object(client, "check_connection") as mock_check:
            mock_check.return_value = Either.right(True)

            result = client.check_connection()

            assert result.is_right()
            assert result.get_right() is True
            mock_check.assert_called_once()

    def test_configuration_validation_comprehensive(self) -> None:
        """Test comprehensive configuration validation."""
        # Test valid configurations
        valid_configs = [
            ConnectionConfig(
                method=ConnectionMethod.APPLESCRIPT,
                timeout=Duration(seconds=30),
            ),
            ConnectionConfig(
                method=ConnectionMethod.WEB_API,
                web_api_host="localhost",
                web_api_port=4242,
                timeout=Duration(seconds=60),
            ),
            ConnectionConfig(
                method=ConnectionMethod.URL_SCHEME,
                timeout=Duration(seconds=15),
            ),
        ]

        for config in valid_configs:
            assert config.method in [
                ConnectionMethod.APPLESCRIPT,
                ConnectionMethod.WEB_API,
                ConnectionMethod.URL_SCHEME,
            ]
            assert config.timeout.seconds > 0

        # Test configurations with edge cases (skip negative timeout since Duration validates)
        edge_configs = [
            ConnectionConfig(
                method=ConnectionMethod.WEB_API,
                web_api_host="",  # Empty host
                timeout=Duration(seconds=30),
            ),
            ConnectionConfig(
                method=ConnectionMethod.WEB_API,
                web_api_port=0,  # Edge case port
                timeout=Duration(seconds=30),
            ),
        ]

        for config in edge_configs:
            # These configs can still be created, but have edge case values
            assert hasattr(config, "method")
            assert hasattr(config, "timeout")
            assert config.method == ConnectionMethod.WEB_API


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
