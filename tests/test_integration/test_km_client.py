"""Tests for Keyboard Maestro Client functional interface.

Tests the functional KM client with Either monad error handling,
connection management, and integration with Keyboard Maestro APIs.
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.core.either import Either
from src.core.types import Duration, MacroId, TriggerId
from src.integration.events import TriggerType
from src.integration.km_client import (
    ConnectionConfig,
    ConnectionMethod,
    KMClient,
    KMError,
    TriggerDefinition,
    create_client_with_fallback,
    retry_with_backoff,
)


@pytest.fixture
def applescript_config() -> ConnectionConfig:
    """Create AppleScript connection configuration."""
    return ConnectionConfig(
        method=ConnectionMethod.APPLESCRIPT,
        timeout=Duration.from_seconds(5),
    )


@pytest.fixture
def web_api_config() -> ConnectionConfig:
    """Create Web API connection configuration."""
    return ConnectionConfig(
        method=ConnectionMethod.WEB_API,
        timeout=Duration.from_seconds(10),
        web_api_port=4490,
    )


@pytest.fixture
def sample_trigger_def() -> bool:
    """Create sample trigger definition."""
    return TriggerDefinition(
        trigger_id=TriggerId("test-trigger-123"),
        macro_id=MacroId("test-macro-456"),
        trigger_type=TriggerType.HOTKEY,
        configuration={"key": "cmd+space", "modifiers": ["cmd"]},
        enabled=True,
    )


@pytest.fixture
def km_client_applescript(applescript_config: ConnectionConfig) -> KMClient:
    """Create KM client with AppleScript configuration."""
    return KMClient(applescript_config)


@pytest.fixture
def km_client_web(web_api_config: ConnectionConfig) -> KMClient:
    """Create KM client with Web API configuration."""
    return KMClient(web_api_config)


class TestKMError:
    """Test KM error creation and handling."""

    def test_connection_error_creation(self) -> None:
        """Test creating connection errors."""
        error = KMError.connection_error("Connection failed")
        assert error.code == "CONNECTION_ERROR"
        assert error.message == "Connection failed"
        assert error.retry_after is None

    def test_execution_error_creation(self) -> None:
        """Test creating execution errors."""
        details = {"script_line": 5, "error_type": "syntax"}
        error = KMError.execution_error("Script failed", details)
        assert error.code == "EXECUTION_ERROR"
        assert error.message == "Script failed"
        assert error.details == details

    def test_timeout_error_creation(self) -> None:
        """Test creating timeout errors."""
        timeout = Duration.from_seconds(30)
        error = KMError.timeout_error(timeout)
        assert error.code == "TIMEOUT_ERROR"
        assert "30" in error.message
        assert error.retry_after is not None


class TestEither:
    """Test Either monad for functional error handling."""

    def test_right_value_creation(self) -> None:
        """Test creating successful Either values."""
        either = Either.right("success")
        assert either.is_right()
        assert not either.is_left()
        assert either.get_right() == "success"
        # Correctly test that get_left() raises exception on Right value
        with pytest.raises(ValueError, match="Cannot get Left value from Right"):
            either.get_left()
        assert either.get_or_else("default") == "success"

    def test_left_value_creation(self) -> None:
        """Test creating error Either values."""
        error = KMError.connection_error("Failed")
        either = Either.left(error)
        assert either.is_left()
        assert not either.is_right()
        assert either.get_left() == error
        # Correctly test that get_right() raises exception on Left value
        with pytest.raises(ValueError, match="Cannot get Right value from Left"):
            either.get_right()
        assert either.get_or_else("default") == "default"

    def test_map_operation(self) -> None:
        """Test mapping functions over Either values."""
        # Map over right value
        right_either = Either.right(5)
        mapped = right_either.map(lambda x: x * 2)
        assert mapped.is_right()
        assert mapped.get_right() == 10

        # Map over left value (should not apply function)
        left_either = Either.left("error")
        mapped_left = left_either.map(lambda x: x * 2)
        assert mapped_left.is_left()
        assert mapped_left.get_left() == "error"

    def test_flat_map_operation(self) -> None:
        """Test flat mapping for chaining Either operations."""

        def double_if_positive(x: Any) -> Mock:
            if x > 0:
                return Either.right(x * 2)
            return Either.left("negative number")

        # Positive number chain
        result1 = Either.right(5).flat_map(double_if_positive)
        assert result1.is_right()
        assert result1.get_right() == 10

        # Negative number chain
        result2 = Either.right(-3).flat_map(double_if_positive)
        assert result2.is_left()
        assert result2.get_left() == "negative number"

        # Error input chain
        result3 = Either.left("initial error").flat_map(double_if_positive)
        assert result3.is_left()
        assert result3.get_left() == "initial error"


class TestConnectionConfig:
    """Test connection configuration."""

    def test_config_immutability(self, applescript_config: dict[str, Any]) -> None:
        """Test that connection configs are immutable."""
        new_timeout = Duration.from_seconds(60)
        new_config = applescript_config.with_timeout(new_timeout)

        # Original config unchanged
        assert applescript_config.timeout.total_seconds() == 5

        # New config has updated timeout
        assert new_config.timeout.total_seconds() == 60
        assert new_config.method == applescript_config.method

    def test_method_change(self, applescript_config: dict[str, Any]) -> None:
        """Test changing connection method."""
        web_config = applescript_config.with_method(ConnectionMethod.WEB_API)

        assert applescript_config.method == ConnectionMethod.APPLESCRIPT
        assert web_config.method == ConnectionMethod.WEB_API
        assert web_config.timeout == applescript_config.timeout


class TestTriggerDefinition:
    """Test trigger definition handling."""

    def test_to_dict_conversion(self, sample_trigger_def: Any) -> None:
        """Test converting trigger definition to dictionary."""
        trigger_dict = sample_trigger_def.to_dict()

        expected_keys = {
            "trigger_id",
            "macro_id",
            "trigger_type",
            "configuration",
            "enabled",
        }
        assert set(trigger_dict.keys()) == expected_keys
        assert trigger_dict["trigger_type"] == "hotkey"
        assert trigger_dict["enabled"] is True


class TestKMClientAppleScript:
    """Test KM client with AppleScript method."""

    @patch("subprocess.run")
    def test_execute_macro_success(
        self,
        mock_run: Any,
        km_client_applescript: Any,
    ) -> None:
        """Test successful macro execution via AppleScript."""
        # Mock successful subprocess execution
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Macro executed successfully",
            stderr="",
        )

        result = km_client_applescript.execute_macro(MacroId("test-macro"))

        assert result.is_right()
        output = result.get_right()
        assert output["success"] is True
        assert "executed successfully" in output["output"]

        # Verify AppleScript was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert (
            "osascript" in call_args[0][0][0]
        )  # Check the executable path contains osascript

    @patch("subprocess.run")
    def test_execute_macro_with_trigger_value(
        self,
        mock_run: Any,
        km_client_applescript: Any,
    ) -> None:
        """Test macro execution with trigger value."""
        mock_run.return_value = Mock(returncode=0, stdout="Success", stderr="")

        result = km_client_applescript.execute_macro(
            MacroId("test-macro"),
            trigger_value="test_value",
        )

        assert result.is_right()

        # Check that trigger value was included in AppleScript command
        call_args = mock_run.call_args
        # The call structure is ['osascript', '-e', 'script_content']
        assert any("test_value" in str(arg) for arg in call_args[0]), (
            "trigger_value should be in the AppleScript command"
        )

    @patch("subprocess.run")
    def test_execute_macro_error(
        self,
        mock_run: Any,
        km_client_applescript: Any,
    ) -> None:
        """Test macro execution error handling."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="AppleScript Error: Macro not found",
        )

        result = km_client_applescript.execute_macro(MacroId("nonexistent"))

        assert result.is_left()
        error = result.get_left()
        assert error.code == "EXECUTION_ERROR"
        assert "Macro not found" in error.message

    @patch("subprocess.run")
    def test_execute_macro_timeout(
        self,
        mock_run: Any,
        km_client_applescript: Any,
    ) -> None:
        """Test macro execution timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("osascript", timeout=5)

        result = km_client_applescript.execute_macro(MacroId("slow-macro"))

        assert result.is_left()
        error = result.get_left()
        assert error.code == "TIMEOUT_ERROR"

    @patch("subprocess.run")
    def test_check_connection(self, mock_run: Any, km_client_applescript: Any) -> None:
        """Test connection check via AppleScript."""
        mock_run.return_value = Mock(returncode=0, stdout="true", stderr="")

        result = km_client_applescript.check_connection()

        assert result.is_right()
        assert result.get_right() is True


class TestKMClientWebAPI:
    """Test KM client with Web API method."""

    @patch("httpx.Client")
    def test_execute_macro_web_success(
        self,
        mock_client: Any,
        km_client_web: Any,
    ) -> None:
        """Test successful macro execution via Web API."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Macro executed"

        mock_client_instance = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance

        result = km_client_web.execute_macro(MacroId("test-macro"))

        assert result.is_right()
        output = result.get_right()
        assert output["success"] is True
        assert "Macro executed" in output["response"]

    @patch("httpx.Client")
    def test_execute_macro_web_error(
        self,
        mock_client: Any,
        km_client_web: Any,
    ) -> None:
        """Test Web API error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Macro not found"

        mock_client_instance = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance

        result = km_client_web.execute_macro(MacroId("nonexistent"))

        assert result.is_left()
        error = result.get_left()
        assert error.code == "EXECUTION_ERROR"
        assert "404" in error.message


class TestTriggerOperations:
    """Test trigger registration and management operations."""

    @patch("subprocess.run")
    def test_register_trigger(
        self,
        mock_run: Any,
        km_client_applescript: Any,
        sample_trigger_def: Any,
    ) -> None:
        """Test trigger registration."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="SUCCESS: Trigger registered",
            stderr="",
        )

        result = km_client_applescript.register_trigger(sample_trigger_def)

        assert result.is_right()
        trigger_id = result.get_right()
        assert trigger_id == sample_trigger_def.trigger_id

    @patch("subprocess.run")
    def test_unregister_trigger(
        self,
        mock_run: Any,
        km_client_applescript: Any,
    ) -> None:
        """Test trigger unregistration."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="SUCCESS: Trigger unregistered",
            stderr="",
        )

        result = km_client_applescript.unregister_trigger(TriggerId("test-trigger"))

        assert result.is_right()
        success = result.get_right()
        assert success is True


class TestAsyncOperations:
    """Test async client operations."""

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_register_trigger_async(
        self,
        mock_subprocess: Any,
        km_client_applescript: Any,
        sample_trigger_def: Any,
    ) -> None:
        """Test async trigger registration."""
        # Mock the subprocess
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"SUCCESS: Trigger registered", b""),
        )
        mock_subprocess.return_value = mock_process

        result = await km_client_applescript.register_trigger_async(sample_trigger_def)

        assert result.is_right()
        trigger_id = result.get_right()
        assert trigger_id == sample_trigger_def.trigger_id

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_activate_trigger_async(
        self,
        mock_run: Any,
        km_client_applescript: Any,
    ) -> None:
        """Test async trigger activation."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="SUCCESS: Trigger activated",
            stderr="",
        )

        result = await km_client_applescript.activate_trigger_async(
            TriggerId("test-trigger"),
        )

        assert result.is_right()
        success = result.get_right()
        assert success is True


class TestFunctionalUtilities:
    """Test functional utilities for KM client."""

    def test_retry_with_backoff_success(self) -> None:
        """Test retry utility with successful operation."""
        call_count = 0

        def flaky_operation() -> Mock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return Either.left(KMError.connection_error("Temporary failure"))
            return Either.right("success")

        result = retry_with_backoff(
            flaky_operation,
            max_retries=5,
            initial_delay=Duration.from_milliseconds(1),  # Fast for testing
        )

        assert result.is_right()
        assert result.get_right() == "success"
        assert call_count == 3

    def test_retry_with_backoff_max_retries(self) -> None:
        """Test retry utility reaching max retries."""

        def always_fail() -> Mock:
            return Either.left(KMError.connection_error("Persistent failure"))

        result = retry_with_backoff(
            always_fail,
            max_retries=2,
            initial_delay=Duration.from_milliseconds(1),
        )

        assert result.is_left()
        error = result.get_left()
        assert error.code == "CONNECTION_ERROR"

    def test_create_client_with_fallback(
        self,
        applescript_config: dict[str, Any],
        web_api_config: dict[str, Any],
    ) -> None:
        """Test client with fallback configuration."""
        fallback_client = create_client_with_fallback(
            applescript_config,
            web_api_config,
        )

        assert isinstance(fallback_client, KMClient)
        # The fallback logic would be tested with actual network calls


# Property-based testing for client robustness
@pytest.mark.parametrize(
    "macro_id,trigger_value",
    [
        ("simple-macro", None),
        ("macro-with-spaces", "simple value"),
        ("macro_with_underscores", "complex/value\\with|chars"),
        ("12345-uuid-style", ""),
    ],
)
def test_execute_macro_parameter_handling(
    macro_id: str,
    trigger_value: Any,
    km_client_applescript: Any,
) -> None:
    """Property test: Client should handle various macro ID and trigger value formats."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="success", stderr="")

        result = km_client_applescript.execute_macro(
            MacroId(macro_id),
            trigger_value=trigger_value,
        )

        # Should not raise exceptions and should return Either
        assert isinstance(result, Either)
        if result.is_right():
            assert "success" in result.get_right().get("output", "")


@pytest.mark.parametrize("timeout_seconds", [1, 5, 30, 60])
def test_connection_config_timeout_bounds(timeout_seconds: Any) -> None:
    """Property test: Connection configs should accept reasonable timeout values."""
    timeout = Duration.from_seconds(timeout_seconds)
    config = ConnectionConfig(timeout=timeout)

    assert config.timeout.total_seconds() == timeout_seconds
    assert config.timeout.total_seconds() > 0
