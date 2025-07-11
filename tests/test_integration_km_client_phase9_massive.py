"""Comprehensive tests for src/integration/km_client.py - MASSIVE 767 statements coverage.

🚨 CRITICAL COVERAGE ENFORCEMENT: Phase 9 targeting highest-impact zero-coverage modules.
This test covers src/integration/km_client.py (767 statements - HIGHEST REMAINING IMPACT) to achieve
maximum progress toward mandatory 95% coverage threshold.

Coverage Focus: Keyboard Maestro client interface, connection management, API integration,
error handling, functional programming patterns, AppleScript integration, and all KM client functionality.
"""

from unittest.mock import Mock, patch

import pytest
from src.core.either import Either
from src.core.types import Duration, GroupId, MacroId, TriggerId
from src.integration.km_client import (
    ConnectionConfig,
    ConnectionMethod,
    KMClient,
    KMError,
    TriggerDefinition,
)


class TestConnectionMethod:
    """Comprehensive tests for ConnectionMethod enumeration."""

    def test_connection_method_values(self):
        """Test ConnectionMethod enumeration values."""
        assert ConnectionMethod.APPLESCRIPT.value == "applescript"
        assert ConnectionMethod.URL_SCHEME.value == "url_scheme"
        assert ConnectionMethod.WEB_API.value == "web_api"
        assert ConnectionMethod.REMOTE_TRIGGER.value == "remote_trigger"

    def test_connection_method_all_values_present(self):
        """Test all expected connection methods are present."""
        expected_methods = {"applescript", "url_scheme", "web_api", "remote_trigger"}
        actual_methods = {method.value for method in ConnectionMethod}
        assert actual_methods == expected_methods

    def test_connection_method_membership(self):
        """Test ConnectionMethod membership testing."""
        assert ConnectionMethod.APPLESCRIPT in ConnectionMethod
        assert ConnectionMethod.WEB_API in ConnectionMethod
        assert ConnectionMethod.URL_SCHEME in ConnectionMethod
        assert ConnectionMethod.REMOTE_TRIGGER in ConnectionMethod

    def test_connection_method_string_representation(self):
        """Test string representation of ConnectionMethod values."""
        assert str(ConnectionMethod.APPLESCRIPT) == "ConnectionMethod.APPLESCRIPT"
        assert repr(ConnectionMethod.WEB_API) == "<ConnectionMethod.WEB_API: 'web_api'>"


class TestKMError:
    """Comprehensive tests for KMError class."""

    def test_km_error_creation_basic(self):
        """Test basic KMError creation."""
        error = KMError(code="TEST_ERROR", message="Test error message")

        assert error.code == "TEST_ERROR"
        assert error.message == "Test error message"
        assert error.details is None
        assert error.retry_after is None

    def test_km_error_creation_with_details(self):
        """Test KMError creation with details."""
        details = {"context": "test", "value": 42}
        retry_after = Duration.from_seconds(5)

        error = KMError(
            code="DETAILED_ERROR",
            message="Error with details",
            details=details,
            retry_after=retry_after
        )

        assert error.code == "DETAILED_ERROR"
        assert error.message == "Error with details"
        assert error.details == details
        assert error.retry_after == retry_after

    def test_connection_error_factory(self):
        """Test connection_error factory method."""
        message = "Connection failed to KM"
        error = KMError.connection_error(message)

        assert error.code == "CONNECTION_ERROR"
        assert error.message == message
        assert error.details is None
        assert error.retry_after is None

    def test_execution_error_factory_simple(self):
        """Test execution_error factory method without details."""
        message = "Macro execution failed"
        error = KMError.execution_error(message)

        assert error.code == "EXECUTION_ERROR"
        assert error.message == message
        assert error.details is None
        assert error.retry_after is None

    def test_execution_error_factory_with_details(self):
        """Test execution_error factory method with details."""
        message = "Macro execution failed"
        details = {"macro_id": "test_macro", "step": 3}
        error = KMError.execution_error(message, details)

        assert error.code == "EXECUTION_ERROR"
        assert error.message == message
        assert error.details == details
        assert error.retry_after is None

    def test_timeout_error_factory(self):
        """Test timeout_error factory method."""
        timeout = Duration.from_seconds(30)
        error = KMError.timeout_error(timeout)

        assert error.code == "TIMEOUT_ERROR"
        assert error.message == "Operation timed out after 30.0s"
        assert error.retry_after == Duration.from_seconds(1.0)

    def test_timeout_error_with_different_durations(self):
        """Test timeout_error with various durations."""
        timeout1 = Duration.from_seconds(60)
        error1 = KMError.timeout_error(timeout1)
        assert error1.message == "Operation timed out after 60.0s"

        timeout2 = Duration.from_seconds(0.5)
        error2 = KMError.timeout_error(timeout2)
        assert error2.message == "Operation timed out after 0.5s"

    def test_validation_error_factory(self):
        """Test validation_error factory method."""
        message = "Invalid macro configuration"
        error = KMError.validation_error(message)

        assert error.code == "VALIDATION_ERROR"
        assert error.message == message
        assert error.details is None
        assert error.retry_after is None

    def test_not_found_error_factory(self):
        """Test not_found_error factory method."""
        message = "Macro not found"
        error = KMError.not_found_error(message)

        assert error.code == "NOT_FOUND_ERROR"
        assert error.message == message
        assert error.details is None
        assert error.retry_after is None

    def test_security_error_factory(self):
        """Test security_error factory method."""
        message = "Unauthorized access"
        error = KMError.security_error(message)

        assert error.code == "SECURITY_ERROR"
        assert error.message == message
        assert error.details is None
        assert error.retry_after is None

    def test_km_error_immutability(self):
        """Test that KMError is immutable."""
        error = KMError(code="TEST", message="Test")

        with pytest.raises(AttributeError):
            error.code = "MODIFIED"

        with pytest.raises(AttributeError):
            error.message = "Modified message"

    def test_km_error_equality(self):
        """Test KMError equality comparison."""
        error1 = KMError(code="TEST", message="Test message")
        error2 = KMError(code="TEST", message="Test message")
        error3 = KMError(code="OTHER", message="Test message")

        assert error1 == error2
        assert error1 != error3

    def test_km_error_repr(self):
        """Test KMError string representation."""
        error = KMError(code="TEST", message="Test message")
        repr_str = repr(error)

        assert "KMError" in repr_str
        assert "TEST" in repr_str
        assert "Test message" in repr_str


class TestConnectionConfig:
    """Comprehensive tests for ConnectionConfig class."""

    def test_connection_config_defaults(self):
        """Test ConnectionConfig with default values."""
        config = ConnectionConfig()

        assert config.method == ConnectionMethod.APPLESCRIPT
        assert config.timeout == Duration.from_seconds(30)
        assert config.web_api_port == 4490
        assert config.web_api_host == "localhost"
        assert config.max_retries == 3
        assert config.retry_delay == Duration.from_seconds(0.5)

    def test_connection_config_custom_values(self):
        """Test ConnectionConfig with custom values."""
        config = ConnectionConfig(
            method=ConnectionMethod.WEB_API,
            timeout=Duration.from_seconds(60),
            web_api_port=8080,
            web_api_host="km.example.com",
            max_retries=5,
            retry_delay=Duration.from_seconds(2)
        )

        assert config.method == ConnectionMethod.WEB_API
        assert config.timeout == Duration.from_seconds(60)
        assert config.web_api_port == 8080
        assert config.web_api_host == "km.example.com"
        assert config.max_retries == 5
        assert config.retry_delay == Duration.from_seconds(2)

    def test_connection_config_with_timeout_method(self):
        """Test with_timeout method creates new config."""
        original_config = ConnectionConfig()
        new_timeout = Duration.from_seconds(120)
        new_config = original_config.with_timeout(new_timeout)

        # Original should be unchanged
        assert original_config.timeout == Duration.from_seconds(30)

        # New config should have new timeout
        assert new_config.timeout == new_timeout
        assert new_config.method == original_config.method
        assert new_config.web_api_port == original_config.web_api_port

    def test_connection_config_with_method_change(self):
        """Test with_method method creates new config."""
        original_config = ConnectionConfig()
        new_method = ConnectionMethod.WEB_API
        new_config = original_config.with_method(new_method)

        # Original should be unchanged
        assert original_config.method == ConnectionMethod.APPLESCRIPT

        # New config should have new method
        assert new_config.method == new_method
        assert new_config.timeout == original_config.timeout
        assert new_config.web_api_port == original_config.web_api_port

    def test_connection_config_immutability(self):
        """Test that ConnectionConfig is immutable."""
        config = ConnectionConfig()

        with pytest.raises(AttributeError):
            config.method = ConnectionMethod.WEB_API

        with pytest.raises(AttributeError):
            config.timeout = Duration.from_seconds(60)

    def test_connection_config_all_methods(self):
        """Test ConnectionConfig with all connection methods."""
        for method in ConnectionMethod:
            config = ConnectionConfig(method=method)
            assert config.method == method

    def test_connection_config_edge_case_values(self):
        """Test ConnectionConfig with edge case values."""
        # Very short timeout
        config1 = ConnectionConfig(timeout=Duration.from_seconds(0.1))
        assert config1.timeout == Duration.from_seconds(0.1)

        # High port number
        config2 = ConnectionConfig(web_api_port=65535)
        assert config2.web_api_port == 65535

        # High retry count
        config3 = ConnectionConfig(max_retries=100)
        assert config3.max_retries == 100

    def test_connection_config_equality(self):
        """Test ConnectionConfig equality comparison."""
        config1 = ConnectionConfig()
        config2 = ConnectionConfig()
        config3 = ConnectionConfig(method=ConnectionMethod.WEB_API)

        assert config1 == config2
        assert config1 != config3

    def test_connection_config_repr(self):
        """Test ConnectionConfig string representation."""
        config = ConnectionConfig()
        repr_str = repr(config)

        assert "ConnectionConfig" in repr_str


class TestTriggerDefinition:
    """Comprehensive tests for TriggerDefinition class."""

    @pytest.fixture
    def mock_trigger_type(self):
        """Create mock trigger type for testing."""
        mock_type = Mock()
        mock_type.value = "hotkey"
        return mock_type

    def test_trigger_definition_creation_basic(self, mock_trigger_type):
        """Test basic TriggerDefinition creation."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+shift+t"}
        )

        assert trigger_def.trigger_id == "test_trigger"
        assert trigger_def.macro_id == "test_macro"
        assert trigger_def.trigger_type == mock_trigger_type
        assert trigger_def.configuration == {"key": "cmd+shift+t"}
        assert trigger_def.enabled is True

    def test_trigger_definition_creation_disabled(self, mock_trigger_type):
        """Test TriggerDefinition creation with disabled trigger."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+shift+t"},
            enabled=False
        )

        assert trigger_def.enabled is False

    def test_trigger_definition_to_dict(self, mock_trigger_type):
        """Test to_dict method conversion."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+shift+t", "modifiers": ["cmd", "shift"]},
            enabled=True
        )

        result_dict = trigger_def.to_dict()

        expected = {
            "trigger_id": "test_trigger",
            "macro_id": "test_macro",
            "trigger_type": "hotkey",
            "configuration": {"key": "cmd+shift+t", "modifiers": ["cmd", "shift"]},
            "enabled": True
        }

        assert result_dict == expected

    def test_trigger_definition_to_dict_disabled(self, mock_trigger_type):
        """Test to_dict method with disabled trigger."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "F1"},
            enabled=False
        )

        result_dict = trigger_def.to_dict()

        assert result_dict["enabled"] is False

    def test_trigger_definition_complex_configuration(self, mock_trigger_type):
        """Test TriggerDefinition with complex configuration."""
        complex_config = {
            "key": "cmd+shift+t",
            "modifiers": ["cmd", "shift"],
            "repeat": False,
            "when_pressed": "activate",
            "context": {
                "application": "TextEdit",
                "window_title": "Document"
            }
        }

        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("complex_trigger"),
            macro_id=MacroId("complex_macro"),
            trigger_type=mock_trigger_type,
            configuration=complex_config
        )

        assert trigger_def.configuration == complex_config
        result_dict = trigger_def.to_dict()
        assert result_dict["configuration"] == complex_config

    def test_trigger_definition_immutability(self, mock_trigger_type):
        """Test that TriggerDefinition is immutable."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+t"}
        )

        with pytest.raises(AttributeError):
            trigger_def.trigger_id = "modified_trigger"

        with pytest.raises(AttributeError):
            trigger_def.enabled = False

    def test_trigger_definition_equality(self, mock_trigger_type):
        """Test TriggerDefinition equality comparison."""
        trigger_def1 = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+t"}
        )

        trigger_def2 = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+t"}
        )

        trigger_def3 = TriggerDefinition(
            trigger_id=TriggerId("other_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+t"}
        )

        assert trigger_def1 == trigger_def2
        assert trigger_def1 != trigger_def3


class TestKMClient:
    """Comprehensive tests for KMClient class."""

    def test_km_client_creation_default_config(self):
        """Test KMClient creation with default configuration."""
        client = KMClient()

        assert client.config.method == ConnectionMethod.APPLESCRIPT
        assert client.config.timeout == Duration.from_seconds(30)
        assert client.config.web_api_port == 4490
        assert client.config.web_api_host == "localhost"

    def test_km_client_creation_custom_config(self):
        """Test KMClient creation with custom configuration."""
        config = ConnectionConfig(
            method=ConnectionMethod.WEB_API,
            timeout=Duration.from_seconds(60),
            web_api_port=8080
        )
        client = KMClient(config)

        assert client.config == config
        assert client.config.method == ConnectionMethod.WEB_API
        assert client.config.timeout == Duration.from_seconds(60)
        assert client.config.web_api_port == 8080

    def test_km_client_config_property(self):
        """Test KMClient config property access."""
        config = ConnectionConfig(method=ConnectionMethod.URL_SCHEME)
        client = KMClient(config)

        retrieved_config = client.config
        assert retrieved_config == config
        assert retrieved_config.method == ConnectionMethod.URL_SCHEME

    def test_execute_macro_basic(self):
        """Test basic macro execution."""
        client = KMClient()
        macro_id = MacroId("test_macro")

        with patch.object(client, '_send_command') as mock_send:
            mock_send.return_value = Either.right({"status": "success"})

            result = client.execute_macro(macro_id)

            assert result.is_right()
            assert result.get_right() == {"status": "success"}
            mock_send.assert_called_once_with("execute_macro", {"macro_id": "test_macro"})

    def test_execute_macro_with_trigger_value(self):
        """Test macro execution with trigger value."""
        client = KMClient()
        macro_id = MacroId("test_macro")
        trigger_value = "test_trigger_value"

        with patch.object(client, '_send_command') as mock_send:
            mock_send.return_value = Either.right({"status": "success"})

            result = client.execute_macro(macro_id, trigger_value)

            assert result.is_right()
            expected_data = {
                "macro_id": "test_macro",
                "trigger_value": "test_trigger_value"
            }
            mock_send.assert_called_once_with("execute_macro", expected_data)

    def test_execute_macro_error_handling(self):
        """Test macro execution error handling."""
        client = KMClient()
        macro_id = MacroId("test_macro")

        with patch.object(client, '_send_command') as mock_send:
            error = KMError.execution_error("Macro failed")
            mock_send.return_value = Either.left(error)

            result = client.execute_macro(macro_id)

            assert result.is_left()
            assert result.get_left() == error

    def test_register_trigger_success(self):
        """Test successful trigger registration."""
        client = KMClient()
        mock_trigger_type = Mock()
        mock_trigger_type.value = "hotkey"

        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+t"}
        )

        with patch.object(client, '_send_command') as mock_send:
            mock_send.return_value = Either.right({"trigger_id": "new_trigger_id"})

            result = client.register_trigger(trigger_def)

            assert result.is_right()
            assert result.get_right() == "new_trigger_id"

    def test_register_trigger_fallback_id(self):
        """Test trigger registration with fallback to original ID."""
        client = KMClient()
        mock_trigger_type = Mock()
        mock_trigger_type.value = "hotkey"

        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+t"}
        )

        with patch.object(client, '_send_command') as mock_send:
            mock_send.return_value = Either.right({"status": "success"})  # No trigger_id returned

            result = client.register_trigger(trigger_def)

            assert result.is_right()
            assert result.get_right() == "test_trigger"  # Falls back to original ID

    def test_register_trigger_error(self):
        """Test trigger registration error handling."""
        client = KMClient()
        mock_trigger_type = Mock()
        mock_trigger_type.value = "hotkey"

        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+t"}
        )

        with patch.object(client, '_send_command') as mock_send:
            error = KMError.validation_error("Invalid trigger configuration")
            mock_send.return_value = Either.left(error)

            result = client.register_trigger(trigger_def)

            assert result.is_left()
            assert result.get_left() == error

    def test_unregister_trigger_success(self):
        """Test successful trigger unregistration."""
        client = KMClient()
        trigger_id = TriggerId("test_trigger")

        with patch.object(client, '_send_command') as mock_send:
            mock_send.return_value = Either.right({"success": True})

            result = client.unregister_trigger(trigger_id)

            assert result.is_right()
            assert result.get_right() is True
            mock_send.assert_called_once_with("unregister_trigger", {"trigger_id": "test_trigger"})

    def test_unregister_trigger_failure(self):
        """Test trigger unregistration failure."""
        client = KMClient()
        trigger_id = TriggerId("test_trigger")

        with patch.object(client, '_send_command') as mock_send:
            mock_send.return_value = Either.right({"success": False})

            result = client.unregister_trigger(trigger_id)

            assert result.is_right()
            assert result.get_right() is False

    def test_unregister_trigger_missing_success_key(self):
        """Test trigger unregistration with missing success key."""
        client = KMClient()
        trigger_id = TriggerId("test_trigger")

        with patch.object(client, '_send_command') as mock_send:
            mock_send.return_value = Either.right({"status": "completed"})  # No success key

            result = client.unregister_trigger(trigger_id)

            assert result.is_right()
            assert result.get_right() is False  # Defaults to False

    def test_get_macro_list_no_filter(self):
        """Test getting macro list without filter."""
        client = KMClient()

        with patch.object(client, '_send_command') as mock_send:
            macros_data = [
                {"id": "macro1", "name": "Macro 1"},
                {"id": "macro2", "name": "Macro 2"}
            ]
            mock_send.return_value = Either.right({"macros": macros_data})

            result = client.get_macro_list()

            assert result.is_right()
            assert result.get_right() == macros_data
            mock_send.assert_called_once_with("list_macros", {})

    def test_get_macro_list_with_filter(self):
        """Test getting macro list with group filter."""
        client = KMClient()
        group_filter = "MyGroup"

        with patch.object(client, '_send_command') as mock_send:
            macros_data = [{"id": "macro1", "name": "Macro 1", "group": "MyGroup"}]
            mock_send.return_value = Either.right({"macros": macros_data})

            result = client.get_macro_list(group_filter)

            assert result.is_right()
            assert result.get_right() == macros_data
            mock_send.assert_called_once_with("list_macros", {"group_filter": "MyGroup"})

    def test_get_macro_list_empty_result(self):
        """Test getting macro list with empty result."""
        client = KMClient()

        with patch.object(client, '_send_command') as mock_send:
            mock_send.return_value = Either.right({"status": "success"})  # No macros key

            result = client.get_macro_list()

            assert result.is_right()
            assert result.get_right() == []  # Defaults to empty list

    def test_list_macros_compatibility(self):
        """Test list_macros method for compatibility."""
        client = KMClient()

        with patch.object(client, 'get_macro_list') as mock_get_list:
            mock_get_list.return_value = Either.right([{"id": "macro1"}])

            result = client.list_macros("TestGroup")

            assert result.is_right()
            assert result.get_right() == [{"id": "macro1"}]
            mock_get_list.assert_called_once_with("TestGroup")

    def test_create_macro_success(self):
        """Test successful macro creation."""
        client = KMClient()
        macro_data = {
            "name": "Test Macro",
            "actions": [{"type": "type_text", "text": "Hello World"}]
        }

        with patch.object(client, '_send_command') as mock_send:
            mock_send.return_value = Either.right({"macro_id": "new_macro_id"})

            result = client.create_macro(macro_data)

            assert result.is_right()
            assert result.get_right() == {"macro_id": "new_macro_id"}

    def test_create_macro_validation_error(self):
        """Test macro creation validation error."""
        client = KMClient()
        macro_data = {"actions": []}  # Missing name

        result = client.create_macro(macro_data)

        assert result.is_left()
        error = result.get_left()
        assert error.code == "VALIDATION_ERROR"
        assert "name is required" in error.message

    def test_create_macro_empty_name(self):
        """Test macro creation with empty name."""
        client = KMClient()
        macro_data = {"name": "", "actions": []}

        result = client.create_macro(macro_data)

        assert result.is_left()
        error = result.get_left()
        assert error.code == "VALIDATION_ERROR"


class TestKMClientSafeOperations:
    """Test KMClient safe operations and error handling."""

    def test_safe_send_method_integration(self):
        """Test _safe_send method integration."""
        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        client = KMClient(config)

        # Test that _safe_send is properly configured
        assert hasattr(client, '_send_command')
        assert callable(client._send_command)

    def test_client_with_different_connection_methods(self):
        """Test client creation with different connection methods."""
        methods = [
            ConnectionMethod.APPLESCRIPT,
            ConnectionMethod.WEB_API,
            ConnectionMethod.URL_SCHEME,
            ConnectionMethod.REMOTE_TRIGGER
        ]

        for method in methods:
            config = ConnectionConfig(method=method)
            client = KMClient(config)
            assert client.config.method == method

    def test_client_timeout_configuration(self):
        """Test client with various timeout configurations."""
        timeouts = [
            Duration.from_seconds(5),
            Duration.from_seconds(30),
            Duration.from_seconds(120),
            Duration.from_seconds(0.5)
        ]

        for timeout in timeouts:
            config = ConnectionConfig(timeout=timeout)
            client = KMClient(config)
            assert client.config.timeout == timeout

    def test_client_retry_configuration(self):
        """Test client with various retry configurations."""
        retry_configs = [
            (1, Duration.from_seconds(0.1)),
            (3, Duration.from_seconds(0.5)),
            (5, Duration.from_seconds(1.0)),
            (10, Duration.from_seconds(2.0))
        ]

        for max_retries, retry_delay in retry_configs:
            config = ConnectionConfig(max_retries=max_retries, retry_delay=retry_delay)
            client = KMClient(config)
            assert client.config.max_retries == max_retries
            assert client.config.retry_delay == retry_delay

    def test_client_web_api_configuration(self):
        """Test client with Web API configuration variations."""
        web_configs = [
            ("localhost", 4490),
            ("127.0.0.1", 8080),
            ("km.example.com", 4242),
            ("remote-km.internal", 9999)
        ]

        for host, port in web_configs:
            config = ConnectionConfig(
                method=ConnectionMethod.WEB_API,
                web_api_host=host,
                web_api_port=port
            )
            client = KMClient(config)
            assert client.config.web_api_host == host
            assert client.config.web_api_port == port


class TestModuleIntegration:
    """Integration tests for KM client module functionality."""

    def test_complete_workflow_macro_management(self):
        """Test complete workflow of macro management."""
        client = KMClient()

        # Create a macro
        macro_data = {"name": "Test Workflow Macro", "actions": []}

        with patch.object(client, '_send_command') as mock_send:
            # Mock macro creation
            mock_send.return_value = Either.right({"macro_id": "workflow_macro"})

            create_result = client.create_macro(macro_data)
            assert create_result.is_right()

            # Mock macro execution
            mock_send.return_value = Either.right({"status": "executed"})

            execute_result = client.execute_macro(MacroId("workflow_macro"))
            assert execute_result.is_right()

    def test_trigger_management_workflow(self):
        """Test complete trigger management workflow."""
        client = KMClient()
        mock_trigger_type = Mock()
        mock_trigger_type.value = "hotkey"

        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("workflow_trigger"),
            macro_id=MacroId("workflow_macro"),
            trigger_type=mock_trigger_type,
            configuration={"key": "cmd+shift+w"}
        )

        with patch.object(client, '_send_command') as mock_send:
            # Mock trigger registration
            mock_send.return_value = Either.right({"trigger_id": "workflow_trigger"})

            register_result = client.register_trigger(trigger_def)
            assert register_result.is_right()

            # Mock trigger unregistration
            mock_send.return_value = Either.right({"success": True})

            unregister_result = client.unregister_trigger(TriggerId("workflow_trigger"))
            assert unregister_result.is_right()
            assert unregister_result.get_right() is True

    def test_error_propagation_through_operations(self):
        """Test error propagation through various operations."""
        client = KMClient()

        with patch.object(client, '_send_command') as mock_send:
            error = KMError.connection_error("Network error")
            mock_send.return_value = Either.left(error)

            # Test error propagation in all operations
            execute_result = client.execute_macro(MacroId("test"))
            assert execute_result.is_left()
            assert execute_result.get_left() == error

            list_result = client.get_macro_list()
            assert list_result.is_left()
            assert list_result.get_left() == error

            unregister_result = client.unregister_trigger(TriggerId("test"))
            assert unregister_result.is_left()
            assert unregister_result.get_left() == error

    def test_configuration_immutability_integration(self):
        """Test configuration immutability throughout client usage."""
        original_config = ConnectionConfig(
            method=ConnectionMethod.WEB_API,
            timeout=Duration.from_seconds(30),
            web_api_port=4490
        )

        client = KMClient(original_config)
        retrieved_config = client.config

        # Verify configs are equal
        assert retrieved_config == original_config

        # Verify immutability
        with pytest.raises(AttributeError):
            retrieved_config.method = ConnectionMethod.APPLESCRIPT

        # Original config should still be unchanged
        assert original_config.method == ConnectionMethod.WEB_API

    def test_functional_programming_patterns(self):
        """Test functional programming patterns in the client."""
        client = KMClient()

        with patch.object(client, '_send_command') as mock_send:
            # Test map operation with successful result
            mock_send.return_value = Either.right({"macros": [{"id": "1"}, {"id": "2"}]})

            result = client.get_macro_list()
            mapped_result = result.map(lambda macros: [m["id"] for m in macros])

            assert mapped_result.is_right()
            assert mapped_result.get_right() == ["1", "2"]

            # Test map operation with error
            error = KMError.not_found_error("No macros found")
            mock_send.return_value = Either.left(error)

            error_result = client.get_macro_list()
            mapped_error_result = error_result.map(lambda macros: len(macros))

            assert mapped_error_result.is_left()
            assert mapped_error_result.get_left() == error

    def test_type_safety_integration(self):
        """Test type safety across all operations."""
        client = KMClient()

        # Test with proper types
        macro_id = MacroId("test_macro")
        trigger_id = TriggerId("test_trigger")
        group_id = GroupId("test_group")

        with patch.object(client, '_send_command') as mock_send:
            mock_send.return_value = Either.right({"status": "success"})

            # These should all work with proper types
            client.execute_macro(macro_id)
            client.execute_macro(macro_id, "trigger_value")
            client.get_macro_list(str(group_id))
            client.unregister_trigger(trigger_id)

    def test_comprehensive_error_scenarios(self):
        """Test comprehensive error scenarios across all methods."""
        client = KMClient()
        error_types = [
            KMError.connection_error("Connection failed"),
            KMError.execution_error("Execution failed"),
            KMError.timeout_error(Duration.from_seconds(30)),
            KMError.validation_error("Invalid input"),
            KMError.not_found_error("Resource not found"),
            KMError.security_error("Access denied")
        ]

        for error in error_types:
            with patch.object(client, '_send_command') as mock_send:
                mock_send.return_value = Either.left(error)

                # Test all methods handle errors properly
                execute_result = client.execute_macro(MacroId("test"))
                assert execute_result.is_left()
                assert execute_result.get_left() == error

                list_result = client.get_macro_list()
                assert list_result.is_left()
                assert list_result.get_left() == error
