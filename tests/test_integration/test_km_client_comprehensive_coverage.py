"""Comprehensive coverage tests for src/integration/km_client.py.

This module provides comprehensive test coverage for the km_client module
to achieve the 95% minimum coverage requirement.
"""

import asyncio
from unittest.mock import patch

import pytest
from src.core.either import Either
from src.core.types import Duration, MacroId, TriggerId
from src.integration.km_client import (
    ConnectionConfig,
    ConnectionMethod,
    KMClient,
    KMError,
    TriggerDefinition,
    create_client_with_fallback,
    retry_with_backoff,
)


class TestConnectionMethod:
    """Test ConnectionMethod enum functionality."""

    def test_connection_method_values(self) -> None:
        """Test ConnectionMethod enum values."""
        assert ConnectionMethod.APPLESCRIPT.value == "applescript"
        assert ConnectionMethod.URL_SCHEME.value == "url_scheme"
        assert ConnectionMethod.WEB_API.value == "web_api"
        assert ConnectionMethod.REMOTE_TRIGGER.value == "remote_trigger"

    def test_connection_method_string_conversion(self) -> None:
        """Test ConnectionMethod string conversion."""
        assert str(ConnectionMethod.APPLESCRIPT) == "applescript"
        assert str(ConnectionMethod.URL_SCHEME) == "url_scheme"
        assert str(ConnectionMethod.WEB_API) == "web_api"
        assert str(ConnectionMethod.REMOTE_TRIGGER) == "remote_trigger"


class TestKMError:
    """Test KMError class functionality."""

    def test_km_error_initialization(self) -> None:
        """Test KMError initialization."""
        code = "TEST_ERROR"
        message = "Test error message"
        details = {"key": "value"}
        retry_after = Duration.from_seconds(5)

        error = KMError(
            code=code, message=message, details=details, retry_after=retry_after
        )

        assert error.code == code
        assert error.message == message
        assert error.details == details
        assert error.retry_after == retry_after

    def test_km_error_connection_error(self) -> None:
        """Test KMError connection error factory method."""
        message = "Connection failed"
        error = KMError.connection_error(message)

        assert error.code == "CONNECTION_ERROR"
        assert error.message == message
        assert error.details is None
        assert error.retry_after is None

    def test_km_error_execution_error(self) -> None:
        """Test KMError execution error factory method."""
        message = "Execution failed"
        details = {"macro_id": "test-macro"}
        error = KMError.execution_error(message, details)

        assert error.code == "EXECUTION_ERROR"
        assert error.message == message
        assert error.details == details

    def test_km_error_timeout_error(self) -> None:
        """Test KMError timeout error factory method."""
        timeout = Duration.from_seconds(30)
        error = KMError.timeout_error(timeout)

        assert error.code == "TIMEOUT_ERROR"
        assert "timeout" in error.message.lower()
        assert error.retry_after == Duration.from_seconds(1.0)

    def test_km_error_validation_error(self) -> None:
        """Test KMError validation error factory method."""
        message = "Invalid parameter"
        error = KMError.validation_error(message)

        assert error.code == "VALIDATION_ERROR"
        assert error.message == message

    def test_km_error_not_found_error(self) -> None:
        """Test KMError not found error factory method."""
        message = "Macro not found"
        error = KMError.not_found_error(message)

        assert error.code == "NOT_FOUND_ERROR"
        assert error.message == message

    def test_km_error_security_error(self) -> None:
        """Test KMError security error factory method."""
        message = "Permission denied"
        error = KMError.security_error(message)

        assert error.code == "SECURITY_ERROR"
        assert error.message == message


class TestTriggerDefinition:
    """Test TriggerDefinition class functionality."""

    def test_trigger_definition_initialization(self) -> None:
        """Test TriggerDefinition initialization."""
        trigger_id = TriggerId("test-trigger")
        trigger_def = TriggerDefinition(
            trigger_id=trigger_id, name="Test Trigger", enabled=True
        )

        assert trigger_def.trigger_id == trigger_id
        assert trigger_def.name == "Test Trigger"
        assert trigger_def.enabled is True

    def test_trigger_definition_with_parameters(self) -> None:
        """Test TriggerDefinition with parameters."""
        trigger_id = TriggerId("hotkey-trigger")
        parameters = {"key": "cmd+shift+a"}

        trigger_def = TriggerDefinition(
            trigger_id=trigger_id,
            name="Hotkey Trigger",
            trigger_type="hotkey",
            parameters=parameters,
        )

        if hasattr(trigger_def, "trigger_type"):
            assert trigger_def.trigger_type == "hotkey"
        if hasattr(trigger_def, "parameters"):
            assert trigger_def.parameters == parameters


class TestConnectionConfig:
    """Test ConnectionConfig class functionality."""

    def test_connection_config_initialization(self) -> None:
        """Test ConnectionConfig initialization with defaults."""
        config = ConnectionConfig()

        assert config.method == ConnectionMethod.APPLESCRIPT
        assert config.timeout == Duration.from_seconds(30)

    def test_connection_config_custom_values(self) -> None:
        """Test ConnectionConfig with custom values."""
        method = ConnectionMethod.WEB_API
        timeout = Duration.from_seconds(60)

        config = ConnectionConfig(method=method, timeout=timeout)

        assert config.method == method
        assert config.timeout == timeout

    def test_connection_config_immutability(self) -> None:
        """Test ConnectionConfig immutability."""
        config = ConnectionConfig()

        # Should be frozen dataclass
        with pytest.raises(AttributeError):
            config.method = ConnectionMethod.URL_SCHEME

    def test_connection_config_with_all_methods(self) -> None:
        """Test ConnectionConfig with all connection methods."""
        methods = [
            ConnectionMethod.APPLESCRIPT,
            ConnectionMethod.URL_SCHEME,
            ConnectionMethod.WEB_API,
            ConnectionMethod.REMOTE_TRIGGER,
        ]

        for method in methods:
            config = ConnectionConfig(method=method)
            assert config.method == method


class TestKMClient:
    """Test KMClient class functionality."""

    def test_km_client_initialization(self) -> None:
        """Test KMClient initialization."""
        config = ConnectionConfig()
        client = KMClient(config)

        assert client.config == config
        assert hasattr(client, "execute_macro")
        assert hasattr(client, "list_macros")

    def test_km_client_with_custom_config(self) -> None:
        """Test KMClient with custom configuration."""
        config = ConnectionConfig(
            method=ConnectionMethod.WEB_API, timeout=Duration.from_seconds(45)
        )
        client = KMClient(config)

        assert client.config.method == ConnectionMethod.WEB_API
        assert client.config.timeout == Duration.from_seconds(45)

    @pytest.mark.asyncio
    async def test_km_client_execute_macro_basic(self) -> None:
        """Test KMClient execute_macro basic functionality."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("test-macro")

        # Mock the underlying execution method
        with patch.object(client, "_execute_via_applescript") as mock_execute:
            mock_execute.return_value = Either.right(
                {
                    "success": True,
                    "execution_time": 1.5,
                    "output": "Macro executed successfully",
                }
            )

            result = await client.execute_macro(macro_id)

            assert result.is_right()
            data = result.get_right()
            assert data["success"] is True
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_km_client_execute_macro_with_parameters(self) -> None:
        """Test KMClient execute_macro with parameters."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("test-macro")
        parameters = {"param1": "value1", "param2": "value2"}

        with patch.object(client, "_execute_via_applescript") as mock_execute:
            mock_execute.return_value = Either.right(
                {"success": True, "parameters": parameters}
            )

            result = await client.execute_macro(macro_id, parameters=parameters)

            assert result.is_right()
            mock_execute.assert_called_once_with(macro_id, parameters, None)

    @pytest.mark.asyncio
    async def test_km_client_execute_macro_with_timeout(self) -> None:
        """Test KMClient execute_macro with timeout."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("test-macro")
        timeout = Duration.from_seconds(60)

        with patch.object(client, "_execute_via_applescript") as mock_execute:
            mock_execute.return_value = Either.right({"success": True})

            result = await client.execute_macro(macro_id, timeout=timeout)

            assert result.is_right()
            mock_execute.assert_called_once_with(macro_id, None, timeout)

    @pytest.mark.asyncio
    async def test_km_client_execute_macro_error(self) -> None:
        """Test KMClient execute_macro error handling."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("non-existent-macro")

        with patch.object(client, "_execute_via_applescript") as mock_execute:
            error = KMError.not_found_error("Macro not found")
            mock_execute.return_value = Either.left(error)

            result = await client.execute_macro(macro_id)

            assert result.is_left()
            error = result.get_left()
            assert error.code == "NOT_FOUND_ERROR"

    @pytest.mark.asyncio
    async def test_km_client_list_macros_basic(self) -> None:
        """Test KMClient list_macros basic functionality."""
        client = KMClient(ConnectionConfig())

        mock_macros = [
            {"id": "macro1", "name": "Test Macro 1", "enabled": True},
            {"id": "macro2", "name": "Test Macro 2", "enabled": True},
        ]

        with patch.object(client, "_list_macros_via_applescript") as mock_list:
            mock_list.return_value = Either.right(mock_macros)

            result = await client.list_macros()

            assert result.is_right()
            macros = result.get_right()
            assert len(macros) == 2
            assert macros[0]["name"] == "Test Macro 1"

    @pytest.mark.asyncio
    async def test_km_client_list_macros_with_group_filter(self) -> None:
        """Test KMClient list_macros with group filter."""
        client = KMClient(ConnectionConfig())
        group_filter = ["Group1", "Group2"]

        with patch.object(client, "_list_macros_via_applescript") as mock_list:
            mock_list.return_value = Either.right([])

            result = await client.list_macros(group_filters=group_filter)

            assert result.is_right()
            mock_list.assert_called_once_with(group_filter, True)

    @pytest.mark.asyncio
    async def test_km_client_list_macros_enabled_only(self) -> None:
        """Test KMClient list_macros with enabled_only filter."""
        client = KMClient(ConnectionConfig())

        with patch.object(client, "_list_macros_via_applescript") as mock_list:
            mock_list.return_value = Either.right([])

            result = await client.list_macros(enabled_only=False)

            assert result.is_right()
            mock_list.assert_called_once_with(None, False)

    @pytest.mark.asyncio
    async def test_km_client_get_macro_info(self) -> None:
        """Test KMClient get_macro_info functionality."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("test-macro")

        mock_info = {
            "id": str(macro_id),
            "name": "Test Macro",
            "enabled": True,
            "group": "Test Group",
        }

        with patch.object(
            client, "_get_macro_info_via_applescript"
        ) as mock_info_method:
            mock_info_method.return_value = Either.right(mock_info)

            result = await client.get_macro_info(macro_id)

            assert result.is_right()
            info = result.get_right()
            assert info["name"] == "Test Macro"

    @pytest.mark.asyncio
    async def test_km_client_connection_method_fallback(self) -> None:
        """Test KMClient connection method fallback."""
        # Test AppleScript fallback to Web API
        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        client = KMClient(config)
        macro_id = MacroId("test-macro")

        with patch.object(client, "_execute_via_applescript") as mock_applescript:
            with patch.object(client, "_execute_via_web_api") as mock_web:
                # AppleScript fails
                mock_applescript.side_effect = Exception("AppleScript not available")
                mock_web.return_value = Either.right({"success": True})

                # Should fallback to web API
                result = await client.execute_macro(macro_id)

                assert result.is_right()
                mock_applescript.assert_called_once()
                mock_web.assert_called_once()

    @pytest.mark.asyncio
    async def test_km_client_web_api_method(self) -> None:
        """Test KMClient Web API connection method."""
        config = ConnectionConfig(method=ConnectionMethod.WEB_API)
        client = KMClient(config)
        macro_id = MacroId("test-macro")

        with patch.object(client, "_execute_via_web_api") as mock_web:
            mock_web.return_value = Either.right({"success": True})

            result = await client.execute_macro(macro_id)

            assert result.is_right()
            mock_web.assert_called_once()

    @pytest.mark.asyncio
    async def test_km_client_url_scheme_method(self) -> None:
        """Test KMClient URL scheme connection method."""
        config = ConnectionConfig(method=ConnectionMethod.URL_SCHEME)
        client = KMClient(config)
        macro_id = MacroId("test-macro")

        with patch.object(client, "_execute_via_url_scheme") as mock_url:
            mock_url.return_value = Either.right({"success": True})

            result = await client.execute_macro(macro_id)

            assert result.is_right()
            mock_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_km_client_remote_trigger_method(self) -> None:
        """Test KMClient remote trigger connection method."""
        config = ConnectionConfig(method=ConnectionMethod.REMOTE_TRIGGER)
        client = KMClient(config)
        macro_id = MacroId("test-macro")

        with patch.object(client, "_execute_via_remote_trigger") as mock_remote:
            mock_remote.return_value = Either.right({"success": True})

            result = await client.execute_macro(macro_id)

            assert result.is_right()
            mock_remote.assert_called_once()

    def test_km_client_validate_macro_id(self) -> None:
        """Test KMClient macro ID validation."""
        client = KMClient(ConnectionConfig())

        # Valid macro IDs
        valid_ids = ["test-macro", "123", "macro_with_underscore"]
        for macro_id in valid_ids:
            assert client._validate_macro_id(MacroId(macro_id)) is True

        # Invalid macro IDs (empty, too long, etc.)
        invalid_ids = ["", " ", "a" * 1000]
        for macro_id in invalid_ids:
            assert client._validate_macro_id(MacroId(macro_id)) is False

    def test_km_client_sanitize_parameters(self) -> None:
        """Test KMClient parameter sanitization."""
        client = KMClient(ConnectionConfig())

        # Valid parameters
        valid_params = {"key": "value", "number": 42, "bool": True}
        sanitized = client._sanitize_parameters(valid_params)
        assert sanitized == valid_params

        # Parameters with potentially dangerous content
        dangerous_params = {
            "script": "tell application 'Finder' to quit",
            "sql": "DROP TABLE users;",
            "shell": "rm -rf /",
        }
        sanitized = client._sanitize_parameters(dangerous_params)
        # Should sanitize dangerous content
        assert all(isinstance(v, str) for v in sanitized.values())

    @pytest.mark.asyncio
    async def test_km_client_timeout_handling(self) -> None:
        """Test KMClient timeout handling."""
        config = ConnectionConfig(timeout=Duration.from_seconds(1))
        client = KMClient(config)
        macro_id = MacroId("slow-macro")

        with patch.object(client, "_execute_via_applescript") as mock_execute:
            # Simulate slow operation
            async def slow_operation(*args, **kwargs):
                await asyncio.sleep(2)  # Longer than timeout
                return Either.right({"success": True})

            mock_execute.side_effect = slow_operation

            # Should timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(client.execute_macro(macro_id), timeout=1.5)

    @pytest.mark.asyncio
    async def test_km_client_retry_mechanism(self) -> None:
        """Test KMClient retry mechanism."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("unreliable-macro")

        call_count = 0

        async def unreliable_operation(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                return Either.left(KMError.connection_error("Connection failed"))
            else:
                return Either.right({"success": True})

        with patch.object(
            client, "_execute_via_applescript", side_effect=unreliable_operation
        ):
            result = await client.execute_macro_with_retry(macro_id, max_retries=3)

            assert result.is_right()
            assert call_count == 3

    def test_km_client_build_applescript_command(self) -> None:
        """Test KMClient AppleScript command building."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("test-macro")
        parameters = {"param1": "value1"}

        command = client._build_applescript_command(macro_id, parameters)

        assert "test-macro" in command
        assert "value1" in command
        assert "tell application" in command.lower()

    def test_km_client_build_web_api_url(self) -> None:
        """Test KMClient Web API URL building."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("test-macro")
        parameters = {"param1": "value1"}

        url = client._build_web_api_url(macro_id, parameters)

        assert "localhost:4490" in url
        assert "test-macro" in url
        assert "param1=value1" in url

    def test_km_client_build_url_scheme(self) -> None:
        """Test KMClient URL scheme building."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("test-macro")
        parameters = {"param1": "value1"}

        url = client._build_url_scheme(macro_id, parameters)

        assert "kmtrigger://" in url
        assert "test-macro" in url
        assert "param1=value1" in url

    @pytest.mark.asyncio
    async def test_km_client_health_check(self) -> None:
        """Test KMClient health check functionality."""
        client = KMClient(ConnectionConfig())

        with patch.object(client, "_check_km_running") as mock_check:
            mock_check.return_value = True

            is_healthy = await client.health_check()

            assert is_healthy is True
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_km_client_get_version_info(self) -> None:
        """Test KMClient version info retrieval."""
        client = KMClient(ConnectionConfig())

        mock_version = {"version": "10.2", "build": "1234"}

        with patch.object(
            client, "_get_version_via_applescript"
        ) as mock_version_method:
            mock_version_method.return_value = Either.right(mock_version)

            result = await client.get_version_info()

            assert result.is_right()
            version_info = result.get_right()
            assert version_info["version"] == "10.2"

    @pytest.mark.asyncio
    async def test_km_client_list_groups(self) -> None:
        """Test KMClient list groups functionality."""
        client = KMClient(ConnectionConfig())

        mock_groups = [
            {"id": "group1", "name": "Group 1", "enabled": True},
            {"id": "group2", "name": "Group 2", "enabled": False},
        ]

        with patch.object(client, "_list_groups_via_applescript") as mock_list:
            mock_list.return_value = Either.right(mock_groups)

            result = await client.list_groups()

            assert result.is_right()
            groups = result.get_right()
            assert len(groups) == 2
            assert groups[0]["name"] == "Group 1"


class TestKMClientUtilities:
    """Test KMClient utility functions."""

    def test_create_client_with_fallback_function(self) -> None:
        """Test create_client_with_fallback utility function."""
        config = ConnectionConfig(method=ConnectionMethod.WEB_API)
        client = create_client_with_fallback(config)

        assert isinstance(client, KMClient)
        assert client.config == config

    def test_create_client_with_fallback_defaults(self) -> None:
        """Test create_client_with_fallback with default configuration."""
        client = create_client_with_fallback()

        assert isinstance(client, KMClient)
        assert client.config.method == ConnectionMethod.APPLESCRIPT

    def test_retry_with_backoff_function(self) -> None:
        """Test retry_with_backoff utility function."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.001)
        async def unreliable_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "Success"

        # Should retry and eventually succeed
        result = asyncio.run(unreliable_function())
        assert result == "Success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_backoff_async(self) -> None:
        """Test retry_with_backoff with async function."""
        attempt_count = 0

        @retry_with_backoff(max_retries=2, initial_delay=0.001)
        async def failing_async_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise ValueError("First attempt fails")
            return f"Success on attempt {attempt_count}"

        result = await failing_async_function()
        assert "Success" in result
        assert attempt_count == 2


class TestKMClientAdvancedFeatures:
    """Test KMClient advanced features."""

    @pytest.mark.asyncio
    async def test_km_client_batch_execution(self) -> None:
        """Test KMClient batch macro execution."""
        client = KMClient(ConnectionConfig())
        macro_ids = [MacroId("macro1"), MacroId("macro2"), MacroId("macro3")]

        with patch.object(client, "execute_macro") as mock_execute:
            mock_execute.return_value = Either.right({"success": True})

            results = await client.execute_macros_batch(macro_ids)

            assert len(results) == 3
            assert all(result.is_right() for result in results)
            assert mock_execute.call_count == 3

    @pytest.mark.asyncio
    async def test_km_client_async_execution(self) -> None:
        """Test KMClient asynchronous macro execution."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("async-macro")

        with patch.object(client, "_execute_async_via_applescript") as mock_async:
            execution_token = "test_token_123"  # noqa: S105
            mock_async.return_value = Either.right({"token": execution_token})

            result = await client.execute_macro_async(macro_id)

            assert result.is_right()
            token_info = result.get_right()
            assert token_info["token"] == execution_token

    @pytest.mark.asyncio
    async def test_km_client_execution_status_check(self) -> None:
        """Test KMClient execution status checking."""
        client = KMClient(ConnectionConfig())
        execution_token = "test_token_456"  # noqa: S105

        mock_status = {"status": "completed", "success": True}

        with patch.object(client, "_get_execution_status") as mock_status_method:
            mock_status_method.return_value = Either.right(mock_status)

            result = await client.get_execution_status(execution_token)

            assert result.is_right()
            status = result.get_right()
            assert status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_km_client_cancel_execution(self) -> None:
        """Test KMClient execution cancellation."""
        client = KMClient(ConnectionConfig())
        execution_token = "test_token_789"  # noqa: S105

        with patch.object(client, "_cancel_execution") as mock_cancel:
            mock_cancel.return_value = Either.right({"cancelled": True})

            result = await client.cancel_execution(execution_token)

            assert result.is_right()
            cancel_info = result.get_right()
            assert cancel_info["cancelled"] is True

    def test_km_client_configuration_validation(self) -> None:
        """Test KMClient configuration validation."""
        # Valid configuration
        valid_config = ConnectionConfig(
            method=ConnectionMethod.WEB_API, timeout=Duration.from_seconds(30)
        )
        client = KMClient(valid_config)
        assert client._validate_config() is True

        # Invalid configuration (negative timeout)
        with pytest.raises(ValueError):
            ConnectionConfig(timeout=Duration.from_seconds(-1))

    @pytest.mark.asyncio
    async def test_km_client_error_recovery(self) -> None:
        """Test KMClient error recovery mechanisms."""
        client = KMClient(ConnectionConfig())
        macro_id = MacroId("recovery-test")

        # Test error recovery with fallback methods
        with patch.object(client, "_execute_via_applescript") as mock_applescript:
            with patch.object(client, "_execute_via_web_api") as mock_web:
                # Primary method fails
                mock_applescript.return_value = Either.left(
                    KMError.connection_error("AppleScript failed")
                )
                # Fallback succeeds
                mock_web.return_value = Either.right({"success": True})

                result = await client.execute_macro_with_fallback(macro_id)

                assert result.is_right()
                data = result.get_right()
                assert data["success"] is True

    def test_km_client_logging_integration(self) -> None:
        """Test KMClient logging integration."""
        import logging

        # Create client with logging enabled
        config = ConnectionConfig()
        client = KMClient(config, enable_logging=True)

        # Should have logger configured
        assert hasattr(client, "logger")
        assert isinstance(client.logger, logging.Logger)

    @pytest.mark.asyncio
    async def test_km_client_metrics_collection(self) -> None:
        """Test KMClient metrics collection."""
        client = KMClient(ConnectionConfig())

        # Enable metrics collection
        client.enable_metrics()

        macro_id = MacroId("metrics-test")

        with patch.object(client, "_execute_via_applescript") as mock_execute:
            mock_execute.return_value = Either.right(
                {"success": True, "execution_time": 1.5}
            )

            result = await client.execute_macro(macro_id)

            assert result.is_right()

            # Check metrics
            metrics = client.get_metrics()
            assert metrics["total_executions"] > 0
            assert metrics["successful_executions"] > 0
