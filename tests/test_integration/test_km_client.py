"""
Tests for Keyboard Maestro Client functional interface.

Tests the functional KM client with Either monad error handling,
connection management, and integration with Keyboard Maestro APIs.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import subprocess
from dataclasses import replace

from src.integration.km_client import (
    KMClient, ConnectionConfig, ConnectionMethod, KMError, Either,
    TriggerDefinition, retry_with_backoff, create_client_with_fallback
)
from src.integration.events import TriggerType
from src.core.types import MacroId, TriggerId, Duration


@pytest.fixture
def applescript_config():
    """Create AppleScript connection configuration."""
    return ConnectionConfig(
        method=ConnectionMethod.APPLESCRIPT,
        timeout=Duration.from_seconds(5)
    )


@pytest.fixture
def web_api_config():
    """Create Web API connection configuration."""
    return ConnectionConfig(
        method=ConnectionMethod.WEB_API,
        timeout=Duration.from_seconds(10),
        web_api_port=4490
    )


@pytest.fixture
def sample_trigger_def():
    """Create sample trigger definition."""
    return TriggerDefinition(
        trigger_id=TriggerId("test-trigger-123"),
        macro_id=MacroId("test-macro-456"), 
        trigger_type=TriggerType.HOTKEY,
        configuration={"key": "cmd+space", "modifiers": ["cmd"]},
        enabled=True
    )


@pytest.fixture
def km_client_applescript(applescript_config):
    """Create KM client with AppleScript configuration."""
    return KMClient(applescript_config)


@pytest.fixture
def km_client_web(web_api_config):
    """Create KM client with Web API configuration."""
    return KMClient(web_api_config)


class TestKMError:
    """Test KM error creation and handling."""
    
    def test_connection_error_creation(self):
        """Test creating connection errors."""
        error = KMError.connection_error("Connection failed")
        assert error.code == "CONNECTION_ERROR"
        assert error.message == "Connection failed"
        assert error.retry_after is None
    
    def test_execution_error_creation(self):
        """Test creating execution errors."""
        details = {"script_line": 5, "error_type": "syntax"}
        error = KMError.execution_error("Script failed", details)
        assert error.code == "EXECUTION_ERROR"
        assert error.message == "Script failed"
        assert error.details == details
    
    def test_timeout_error_creation(self):
        """Test creating timeout errors."""
        timeout = Duration.from_seconds(30)
        error = KMError.timeout_error(timeout)
        assert error.code == "TIMEOUT_ERROR"
        assert "30" in error.message
        assert error.retry_after is not None


class TestEither:
    """Test Either monad for functional error handling."""
    
    def test_right_value_creation(self):
        """Test creating successful Either values."""
        either = Either.right("success")
        assert either.is_right()
        assert not either.is_left()
        assert either.get_right() == "success"
        assert either.get_left() is None
        assert either.get_or_else("default") == "success"
    
    def test_left_value_creation(self):
        """Test creating error Either values."""
        error = KMError.connection_error("Failed")
        either = Either.left(error)
        assert either.is_left()
        assert not either.is_right()
        assert either.get_left() == error
        assert either.get_right() is None
        assert either.get_or_else("default") == "default"
    
    def test_map_operation(self):
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
    
    def test_flat_map_operation(self):
        """Test flat mapping for chaining Either operations."""
        def double_if_positive(x):
            if x > 0:
                return Either.right(x * 2)
            else:
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
    
    def test_config_immutability(self, applescript_config):
        """Test that connection configs are immutable."""
        new_timeout = Duration.from_seconds(60)
        new_config = applescript_config.with_timeout(new_timeout)
        
        # Original config unchanged
        assert applescript_config.timeout.total_seconds() == 5
        
        # New config has updated timeout
        assert new_config.timeout.total_seconds() == 60
        assert new_config.method == applescript_config.method
    
    def test_method_change(self, applescript_config):
        """Test changing connection method."""
        web_config = applescript_config.with_method(ConnectionMethod.WEB_API)
        
        assert applescript_config.method == ConnectionMethod.APPLESCRIPT
        assert web_config.method == ConnectionMethod.WEB_API
        assert web_config.timeout == applescript_config.timeout


class TestTriggerDefinition:
    """Test trigger definition handling."""
    
    def test_to_dict_conversion(self, sample_trigger_def):
        """Test converting trigger definition to dictionary."""
        trigger_dict = sample_trigger_def.to_dict()
        
        expected_keys = {
            "trigger_id", "macro_id", "trigger_type", 
            "configuration", "enabled"
        }
        assert set(trigger_dict.keys()) == expected_keys
        assert trigger_dict["trigger_type"] == "hotkey"
        assert trigger_dict["enabled"] is True


class TestKMClientAppleScript:
    """Test KM client with AppleScript method."""
    
    @patch('subprocess.run')
    def test_execute_macro_success(self, mock_run, km_client_applescript):
        """Test successful macro execution via AppleScript."""
        # Mock successful subprocess execution
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Macro executed successfully",
            stderr=""
        )
        
        result = km_client_applescript.execute_macro(MacroId("test-macro"))
        
        assert result.is_right()
        output = result.get_right()
        assert output["success"] is True
        assert "executed successfully" in output["output"]
        
        # Verify AppleScript was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "osascript" in call_args[0][0]
    
    @patch('subprocess.run')
    def test_execute_macro_with_trigger_value(self, mock_run, km_client_applescript):
        """Test macro execution with trigger value."""
        mock_run.return_value = Mock(returncode=0, stdout="Success", stderr="")
        
        result = km_client_applescript.execute_macro(
            MacroId("test-macro"),
            trigger_value="test_value"
        )
        
        assert result.is_right()
        
        # Check that trigger value was included in AppleScript
        call_args = mock_run.call_args
        script_content = call_args[0][1][1]  # Second argument to osascript -e
        assert "test_value" in script_content
    
    @patch('subprocess.run')
    def test_execute_macro_error(self, mock_run, km_client_applescript):
        """Test macro execution error handling."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="AppleScript Error: Macro not found"
        )
        
        result = km_client_applescript.execute_macro(MacroId("nonexistent"))
        
        assert result.is_left()
        error = result.get_left()
        assert error.code == "EXECUTION_ERROR"
        assert "Macro not found" in error.message
    
    @patch('subprocess.run')
    def test_execute_macro_timeout(self, mock_run, km_client_applescript):
        """Test macro execution timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            "osascript", timeout=5
        )
        
        result = km_client_applescript.execute_macro(MacroId("slow-macro"))
        
        assert result.is_left()
        error = result.get_left()
        assert error.code == "TIMEOUT_ERROR"
    
    @patch('subprocess.run')
    def test_check_connection(self, mock_run, km_client_applescript):
        """Test connection check via AppleScript."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="true",
            stderr=""
        )
        
        result = km_client_applescript.check_connection()
        
        assert result.is_right()
        assert result.get_right() is True


class TestKMClientWebAPI:
    """Test KM client with Web API method."""
    
    @patch('httpx.Client')
    def test_execute_macro_web_success(self, mock_client, km_client_web):
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
    
    @patch('httpx.Client')
    def test_execute_macro_web_error(self, mock_client, km_client_web):
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
    
    @patch('subprocess.run')
    def test_register_trigger(self, mock_run, km_client_applescript, sample_trigger_def):
        """Test trigger registration."""
        mock_run.return_value = Mock(returncode=0, stdout="registered", stderr="")
        
        result = km_client_applescript.register_trigger(sample_trigger_def)
        
        assert result.is_right()
        trigger_id = result.get_right()
        assert trigger_id == sample_trigger_def.trigger_id
    
    @patch('subprocess.run') 
    def test_unregister_trigger(self, mock_run, km_client_applescript):
        """Test trigger unregistration."""
        mock_run.return_value = Mock(returncode=0, stdout="unregistered", stderr="")
        
        result = km_client_applescript.unregister_trigger(TriggerId("test-trigger"))
        
        assert result.is_right()
        assert result.get_right() is True


class TestAsyncOperations:
    """Test async client operations."""
    
    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_register_trigger_async(self, mock_run, km_client_applescript, sample_trigger_def):
        """Test async trigger registration."""
        mock_run.return_value = Mock(returncode=0, stdout="registered", stderr="")
        
        result = await km_client_applescript.register_trigger_async(sample_trigger_def)
        
        assert result.is_right()
        assert result.get_right() == sample_trigger_def.trigger_id
    
    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_activate_trigger_async(self, mock_run, km_client_applescript):
        """Test async trigger activation."""
        mock_run.return_value = Mock(returncode=0, stdout="activated", stderr="")
        
        result = await km_client_applescript.activate_trigger_async(TriggerId("test-trigger"))
        
        assert result.is_right()
        assert result.get_right() is True


class TestFunctionalUtilities:
    """Test functional utilities for KM client."""
    
    def test_retry_with_backoff_success(self):
        """Test retry utility with successful operation."""
        call_count = 0
        
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return Either.left(KMError.connection_error("Temporary failure"))
            return Either.right("success")
        
        result = retry_with_backoff(
            flaky_operation, 
            max_retries=5,
            initial_delay=Duration.from_milliseconds(1)  # Fast for testing
        )
        
        assert result.is_right()
        assert result.get_right() == "success"
        assert call_count == 3
    
    def test_retry_with_backoff_max_retries(self):
        """Test retry utility reaching max retries."""
        def always_fail():
            return Either.left(KMError.connection_error("Persistent failure"))
        
        result = retry_with_backoff(
            always_fail,
            max_retries=2,
            initial_delay=Duration.from_milliseconds(1)
        )
        
        assert result.is_left()
        error = result.get_left()
        assert error.code == "CONNECTION_ERROR"
    
    def test_create_client_with_fallback(self, applescript_config, web_api_config):
        """Test client with fallback configuration."""
        fallback_client = create_client_with_fallback(applescript_config, web_api_config)
        
        assert isinstance(fallback_client, KMClient)
        # The fallback logic would be tested with actual network calls


# Property-based testing for client robustness
@pytest.mark.parametrize("macro_id,trigger_value", [
    ("simple-macro", None),
    ("macro-with-spaces", "simple value"),
    ("macro_with_underscores", "complex/value\\with|chars"),
    ("12345-uuid-style", ""),
])
def test_execute_macro_parameter_handling(macro_id, trigger_value, km_client_applescript):
    """Property test: Client should handle various macro ID and trigger value formats."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="success", stderr="")
        
        result = km_client_applescript.execute_macro(
            MacroId(macro_id),
            trigger_value=trigger_value
        )
        
        # Should not raise exceptions and should return Either
        assert isinstance(result, Either)
        if result.is_right():
            assert "success" in result.get_right().get("output", "")


@pytest.mark.parametrize("timeout_seconds", [1, 5, 30, 60])
def test_connection_config_timeout_bounds(timeout_seconds):
    """Property test: Connection configs should accept reasonable timeout values."""
    timeout = Duration.from_seconds(timeout_seconds)
    config = ConnectionConfig(timeout=timeout)
    
    assert config.timeout.total_seconds() == timeout_seconds
    assert config.timeout.total_seconds() > 0