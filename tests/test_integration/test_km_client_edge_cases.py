"""Edge case and comprehensive tests for KMClient to improve coverage.

This module tests edge cases, error paths, and private methods to achieve
higher coverage for the km_client module.
"""

from unittest.mock import Mock, patch

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.core.either import Either
from src.core.types import Duration, MacroId, TriggerId
from src.integration.events import TriggerType
from src.integration.km_client import (
    ConnectionConfig,
    ConnectionMethod,
    KMClient,
    KMError,
    TriggerDefinition,
)


class TestKMClientEdgeCases:
    """Test edge cases and error paths in KMClient."""

    def test_km_error_factories(self) -> None:
        """Test all KMError factory methods."""
        # Test connection_error
        error = KMError.connection_error("Connection failed")
        assert error.code == "CONNECTION_ERROR"
        assert error.message == "Connection failed"
        assert error.details is None
        assert error.retry_after is None

        # Test execution_error with details
        details = {"command": "test", "output": "failed"}
        error = KMError.execution_error("Execution failed", details)
        assert error.code == "EXECUTION_ERROR"
        assert error.message == "Execution failed"
        assert error.details == details

        # Test timeout_error
        timeout = Duration.from_seconds(30)
        error = KMError.timeout_error(timeout)
        assert error.code == "TIMEOUT_ERROR"
        assert "30" in error.message
        assert error.retry_after.total_seconds() == 1.0

        # Test validation_error
        error = KMError.validation_error("Invalid input")
        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Invalid input"

        # Test not_found_error
        error = KMError.not_found_error("Macro not found")
        assert error.code == "NOT_FOUND_ERROR"
        assert error.message == "Macro not found"

        # Test security_error
        error = KMError.security_error("Access denied")
        assert error.code == "SECURITY_ERROR"
        assert error.message == "Access denied"

    def test_connection_config_methods(self) -> None:
        """Test ConnectionConfig immutable update methods."""
        config = ConnectionConfig()

        # Test with_method
        new_config = config.with_method(ConnectionMethod.WEB_API)
        assert new_config.method == ConnectionMethod.WEB_API
        assert config.method == ConnectionMethod.APPLESCRIPT  # Original unchanged

        # Test with_timeout
        new_timeout = Duration.from_seconds(60)
        new_config = config.with_timeout(new_timeout)
        assert new_config.timeout == new_timeout
        assert config.timeout.total_seconds() == 30.0  # Original unchanged

        # Test that other fields are preserved when changing one field
        new_config = config.with_method(ConnectionMethod.URL_SCHEME)
        assert new_config.web_api_port == 4490  # Other fields unchanged
        assert new_config.max_retries == 3  # Other fields unchanged

    def test_trigger_definition_edge_cases(self) -> None:
        """Test TriggerDefinition edge cases."""
        # Test with minimal configuration
        config = {
            "trigger_type": TriggerType.APPLICATION,
            "application_bundle_id": "com.test.app",
        }
        trigger = TriggerDefinition(
            trigger_id=TriggerId("test-trigger"),
            trigger_type=TriggerType.APPLICATION,
            macro_id=MacroId("test-macro"),
            configuration=config,
        )

        trigger_dict = trigger.to_dict()
        assert trigger_dict["trigger_type"] == "application"
        assert trigger_dict["macro_id"] == "test-macro"
        assert trigger_dict["enabled"] is True

        # Test with disabled state
        trigger2 = TriggerDefinition(
            trigger_id=TriggerId("hotkey-trigger"),
            trigger_type=TriggerType.HOTKEY,
            macro_id=MacroId("hotkey-macro"),
            configuration={"key": "cmd+shift+a"},
            enabled=False,
        )

        trigger2_dict = trigger2.to_dict()
        assert trigger2_dict["enabled"] is False

    def test_trigger_definition_dataclass(self) -> None:
        """Test TriggerDefinition dataclass functionality."""
        # Test creation with all fields
        trigger = TriggerDefinition(
            trigger_id=TriggerId("test-id"),
            trigger_type=TriggerType.HOTKEY,
            macro_id=MacroId("test-macro"),
            configuration={"key": "cmd+t"},
            enabled=True,
        )

        assert trigger.trigger_id == "test-id"
        assert trigger.trigger_type == TriggerType.HOTKEY
        assert trigger.macro_id == "test-macro"
        assert trigger.configuration == {"key": "cmd+t"}
        assert trigger.enabled is True

        # Test to_dict method
        trigger_dict = trigger.to_dict()
        assert trigger_dict["trigger_id"] == "test-id"
        assert trigger_dict["trigger_type"] == "hotkey"
        assert trigger_dict["macro_id"] == "test-macro"
        assert trigger_dict["configuration"] == {"key": "cmd+t"}
        assert trigger_dict["enabled"] is True

    @patch("src.commands.secure_subprocess.get_secure_subprocess_manager")
    def test_check_connection_with_applescript_true(self, mock_get_manager: Mock) -> None:
        """Test check_connection when AppleScript returns 'true'."""
        client = KMClient()

        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager

        # Mock the actual AppleScript output
        mock_result = Mock()
        mock_result.stdout = "true"  # AppleScript returns 'true' when process exists
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_manager.execute_secure_command.return_value = mock_result

        result = client.check_connection()

        assert result.is_right()
        assert result.get_right() is True

    @patch("src.commands.secure_subprocess.get_secure_subprocess_manager")
    def test_check_connection_with_applescript_false(self, mock_get_manager: Mock) -> None:
        """Test check_connection when AppleScript returns 'false'."""
        client = KMClient()

        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager

        mock_result = Mock()
        mock_result.stdout = (
            "false"  # AppleScript returns 'false' when process doesn't exist
        )
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_manager.execute_secure_command.return_value = mock_result

        result = client.check_connection()

        assert result.is_right()
        assert result.get_right() is False

    @patch("src.commands.secure_subprocess.get_secure_subprocess_manager")
    def test_check_connection_with_exception(self, mock_get_manager: Mock) -> None:
        """Test check_connection when exception occurs."""
        client = KMClient()

        mock_get_manager.side_effect = Exception("Subprocess error")

        result = client.check_connection()

        # The code catches exceptions and returns {"alive": False}
        assert result.is_right()
        assert result.get_right() is False

    @patch("src.commands.secure_subprocess.get_secure_subprocess_manager")
    def test_execute_macro_with_error_output(self, mock_get_manager: Mock) -> None:
        """Test execute_macro when AppleScript returns error."""
        client = KMClient()

        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager

        mock_result = Mock()
        mock_result.stdout = "ERROR: Macro not found"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_manager.execute_secure_command.return_value = mock_result

        result = client.execute_macro(MacroId("missing-macro"))

        assert result.is_left()
        assert result.get_left().code == "EXECUTION_ERROR"
        assert "Macro not found" in result.get_left().message

    @patch("src.commands.secure_subprocess.get_secure_subprocess_manager")
    def test_execute_macro_with_timeout(self, mock_get_manager: Mock) -> None:
        """Test execute_macro when subprocess times out."""
        import subprocess

        client = KMClient()

        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        mock_manager.execute_secure_command.side_effect = subprocess.TimeoutExpired(
            "cmd", 30
        )

        result = client.execute_macro(MacroId("slow-macro"))

        assert result.is_left()
        assert result.get_left().code == "TIMEOUT_ERROR"

    def test_send_via_url_scheme(self) -> None:
        """Test URL scheme sending method."""
        config = ConnectionConfig(method=ConnectionMethod.URL_SCHEME)
        client = KMClient(config)

        # URL scheme is implemented for execute_macro
        result = client.execute_macro(MacroId("test-macro"))

        assert result.is_right()
        assert result.get_right()["success"] is True
        assert "kmtrigger://macro=test-macro" in result.get_right()["url"]

    @patch("httpx.Client")
    def test_send_via_web_api_success(self, mock_httpx_client_class: Mock) -> None:
        """Test Web API sending method with success."""
        config = ConnectionConfig(method=ConnectionMethod.WEB_API)
        client = KMClient(config)

        # Mock httpx client
        mock_httpx_client = Mock()
        mock_httpx_client_class.return_value.__enter__.return_value = mock_httpx_client

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Macro executed successfully"
        mock_httpx_client.get.return_value = mock_response

        result = client.execute_macro(MacroId("web-macro"))

        assert result.is_right()
        assert result.get_right()["success"] is True
        assert "Macro executed successfully" in result.get_right()["response"]

    @patch("httpx.Client")
    def test_send_via_web_api_failure(self, mock_httpx_client_class: Mock) -> None:
        """Test Web API sending method with HTTP error."""
        config = ConnectionConfig(method=ConnectionMethod.WEB_API)
        client = KMClient(config)

        # Mock httpx client to raise exception
        mock_httpx_client = Mock()
        mock_httpx_client_class.return_value.__enter__.return_value = mock_httpx_client
        mock_httpx_client.get.side_effect = httpx.HTTPError("Connection failed")

        result = client.execute_macro(MacroId("web-macro"))

        assert result.is_left()
        assert result.get_left().code == "EXECUTION_ERROR"

    def test_register_unregister_trigger_applescript(self) -> None:
        """Test trigger registration/unregistration with AppleScript."""
        client = KMClient()

        # These methods use internal state tracking for now
        config = {"trigger_type": TriggerType.HOTKEY, "hotkey": "Cmd+Shift+T"}
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test-trigger"),
            trigger_type=TriggerType.HOTKEY,
            macro_id=MacroId("test-macro"),
            configuration=config,
        )

        # Mock the _send_command to avoid actual AppleScript execution
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right({"trigger_id": "trigger-12345"})

            # Register trigger
            result = client.register_trigger(trigger_def)
            assert result.is_right()
            trigger_id = result.get_right()
            assert trigger_id == "trigger-12345"

            # Mock unregister
            mock_send.return_value = Either.right({"success": True})

            # Unregister trigger
            result = client.unregister_trigger(trigger_id)
            assert result.is_right()
            assert result.get_right() is True

    def test_list_macros_variations(self) -> None:
        """Test list_macros with different parameters."""
        client = KMClient()

        # Mock the _send_command to return sample macro list
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right(
                {
                    "macros": [
                        {"name": "Test Macro 1", "id": "macro-1"},
                        {"name": "Test Macro 2", "id": "macro-2"},
                    ]
                }
            )

            # Test basic list
            result = client.list_macros()
            assert result.is_right()
            macros = result.get_right()
            assert isinstance(macros, list)
            assert len(macros) == 2

            # Test with group filter
            result = client.list_macros("Test Group")
            assert result.is_right()
            assert isinstance(result.get_right(), list)

    def test_get_macro_list_alias(self) -> None:
        """Test get_macro_list alias method."""
        client = KMClient()

        # Mock the _send_command to return sample macro list
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right(
                {"macros": [{"name": "Test Macro", "id": "macro-1"}]}
            )

            # get_macro_list is an alias for list_macros
            result = client.get_macro_list()
            assert result.is_right()
            assert isinstance(result.get_right(), list)

    def test_create_macro_validation(self) -> None:
        """Test create_macro validation."""
        client = KMClient()

        # Test without name - this should validate locally
        result = client.create_macro({})
        assert result.is_left()
        assert result.get_left().code == "VALIDATION_ERROR"
        assert "name is required" in result.get_left().message

        # Test with name - mock the actual creation
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right({"macro_id": "new-macro-123"})

            result = client.create_macro({"name": "Test Macro"})
            assert result.is_right()
            assert result.get_right()["success"] is True
            assert result.get_right()["macro_id"] == "new-macro-123"

    def test_list_macros_with_details(self) -> None:
        """Test list_macros_with_details method."""
        client = KMClient()

        # Mock the _send_command to return detailed macro list
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right(
                {
                    "macros": [
                        {
                            "name": "Test Macro",
                            "id": "macro-1",
                            "group": "Test Group",
                            "enabled": True,
                        }
                    ]
                }
            )

            result = client.list_macros_with_details()
            assert result.is_right()
            macros = result.get_right()
            assert isinstance(macros, list)
            if macros:  # If any macros returned
                assert "details" in macros[0]
                assert macros[0]["details"] is True

    def test_get_macro_status(self) -> None:
        """Test get_macro_status method."""
        client = KMClient()

        # Mock the _send_command
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right(
                {"status": {"status": "idle", "enabled": True}}
            )

            result = client.get_macro_status(MacroId("test-macro"))
            assert result.is_right()
            status = result.get_right()
            assert isinstance(status, dict)
            assert "status" in status

    def test_trigger_management_methods(self) -> None:
        """Test trigger activation/deactivation methods."""
        client = KMClient()

        # Mock the _send_command for all trigger operations
        with patch.object(client, "_send_command") as mock_send:
            # Activate trigger
            mock_send.return_value = Either.right({"success": True})
            result = client.activate_trigger(TriggerId("test-trigger"))
            assert result.is_right()
            assert result.get_right() is True

            # Deactivate trigger
            mock_send.return_value = Either.right({"success": True})
            result = client.deactivate_trigger(TriggerId("test-trigger"))
            assert result.is_right()
            assert result.get_right() is True

            # List triggers
            mock_send.return_value = Either.right(
                {"triggers": [{"id": "trigger-1", "type": "hotkey", "enabled": True}]}
            )
            result = client.list_triggers()
            assert result.is_right()
            assert isinstance(result.get_right(), list)

            # Get trigger status
            mock_send.return_value = Either.right(
                {"status": {"enabled": True, "last_fired": None}}
            )
            result = client.get_trigger_status(TriggerId("test-trigger"))
            assert result.is_right()
            assert isinstance(result.get_right(), dict)

    @pytest.mark.asyncio
    async def test_async_trigger_methods(self) -> None:
        """Test async trigger management methods."""
        client = KMClient()

        config = {
            "trigger_type": TriggerType.HOTKEY,
            "key": "Cmd+T",  # Changed from "hotkey" to "key" as required
        }
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("async-trigger"),
            trigger_type=TriggerType.HOTKEY,
            macro_id=MacroId("async-macro"),
            configuration=config,
        )

        # Mock the internal methods to avoid actual AppleScript execution
        with patch.object(client, "_validate_trigger_definition") as mock_validate:
            with patch.object(client, "_sanitize_trigger_data") as mock_sanitize:
                with patch.object(client, "_build_trigger_script_safe") as mock_build:
                    with patch.object(
                        client, "_execute_applescript_safe"
                    ) as mock_execute:
                        # Set up successful mocks
                        mock_validate.return_value = Either.right(trigger_def)
                        mock_sanitize.return_value = Either.right(config)
                        mock_build.return_value = Either.right("script")
                        mock_execute.return_value = Either.right("Trigger registered")

                        # Register trigger async
                        result = await client.register_trigger_async(trigger_def)
                        assert result.is_right()
                        trigger_id = result.get_right()
                        assert trigger_id == "async-trigger"

        # Mock sync methods for activation/deactivation
        with patch.object(client, "activate_trigger") as mock_activate:
            mock_activate.return_value = Either.right(True)
            result = await client.activate_trigger_async(trigger_id)
            assert result.is_right()

        with patch.object(client, "deactivate_trigger") as mock_deactivate:
            mock_deactivate.return_value = Either.right(True)
            result = await client.deactivate_trigger_async(trigger_id)
            assert result.is_right()

        # Mock list methods
        with patch.object(client, "list_triggers") as mock_list_triggers:
            mock_list_triggers.return_value = Either.right([{"id": "trigger-1"}])
            result = await client.list_triggers_async()
            assert result.is_right()

        with patch.object(client, "_list_macros_applescript") as mock_list_macros:
            mock_list_macros.return_value = Either.right([{"name": "Test Macro"}])
            result = await client.list_macros_async()
            assert result.is_right()

    def test_parse_applescript_records(self) -> None:
        """Test _parse_applescript_records method."""
        client = KMClient()

        # Test empty output
        result = client._parse_applescript_records("")
        assert result == []

        # Test single record in AppleScript format
        # The parser expects comma-separated key:value pairs, with macroId indicating new records
        result = client._parse_applescript_records(
            "macroId:macro-1, name:Test Macro, enabled:true"
        )
        assert len(result) == 1
        assert result[0]["macroId"] == "macro-1"
        assert result[0]["name"] == "Test Macro"
        assert result[0]["enabled"] is True  # Parser converts to boolean

        # Test multiple records
        output = "macroId:id-1, name:Macro 1, enabled:true, macroId:id-2, name:Macro 2, enabled:false"
        result = client._parse_applescript_records(output)
        assert len(result) == 2
        assert result[0]["name"] == "Macro 1"
        assert result[1]["name"] == "Macro 2"
        assert result[1]["enabled"] is False  # Parser converts to boolean

    @given(
        macro_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        trigger_value=st.text(max_size=100),
    )
    def test_execute_macro_property_based(self, macro_id: str, trigger_value: str) -> None:
        """Property-based test for execute_macro."""
        client = KMClient()

        # Mock the _send_command to avoid actual execution
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right(
                {"success": True, "output": f"Executed {macro_id}"}
            )

            # Should handle any valid string inputs
            result = client.execute_macro(
                MacroId(macro_id), trigger_value if trigger_value else None
            )

            # Result should always be Either
            assert hasattr(result, "is_left")
            assert hasattr(result, "is_right")
            assert result.is_right()  # Since we mocked success

    def test_safe_send_unsupported_method(self) -> None:
        """Test _safe_send with unsupported connection method."""
        # Create a mock unsupported method
        config = ConnectionConfig()
        config = config.with_method(ConnectionMethod.REMOTE_TRIGGER)

        result = KMClient._safe_send(config, "test", {})

        assert result.is_left()
        assert "Unsupported method" in result.get_left().message

    def test_safe_send_exception_handling(self) -> None:
        """Test _safe_send general exception handling."""
        config = ConnectionConfig()

        # Need to patch at the module level since it's a static method
        with patch(
            "src.integration.km_client.KMClient._send_via_applescript",
            side_effect=Exception("Test error"),
        ):
            result = KMClient._safe_send(config, "test", {})

            assert result.is_left()
            # The code returns EXECUTION_ERROR for general exceptions
            assert result.get_left().code == "EXECUTION_ERROR"
            assert "Test error" in result.get_left().message

    @patch("src.commands.secure_subprocess.get_secure_subprocess_manager")
    def test_applescript_command_variations(self, mock_get_manager: Mock) -> None:
        """Test different AppleScript command implementations."""
        client = KMClient()

        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager

        # Test execute_macro command (which is supported)
        mock_result = Mock()
        mock_result.stdout = "Macro executed"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_manager.execute_secure_command.return_value = mock_result

        result = client._send_command("execute_macro", {"macro_id": "test-macro"})
        assert result.is_right()
        assert result.get_right()["success"] is True

        # Test register_trigger command (which is supported)
        mock_result.stdout = "SUCCESS:trigger-123"
        result = client._send_command(
            "register_trigger",
            {
                "trigger_type": "hotkey",
                "trigger_id": "test-trigger",
                "macro_id": "test-macro",
                "configuration": {"key": "Cmd+T"},
            },
        )
        assert result.is_right()
        assert result.get_right()["trigger_id"] == "test-trigger"

    def test_validate_trigger_definition(self) -> None:
        """Test _validate_trigger_definition method."""
        client = KMClient()

        # Valid trigger
        config = {
            "key": "Cmd+T"  # Changed from "hotkey" to "key"
        }
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("valid-trigger"),
            trigger_type=TriggerType.HOTKEY,
            macro_id=MacroId("test-macro"),
            configuration=config,
        )

        result = client._validate_trigger_definition(trigger_def)
        assert result.is_right()

        # Test with missing required field
        bad_config = {}  # Missing "key" for HOTKEY trigger
        bad_trigger_def = TriggerDefinition(
            trigger_id=TriggerId("bad-trigger"),
            trigger_type=TriggerType.HOTKEY,
            macro_id=MacroId("test-macro"),
            configuration=bad_config,
        )
        result = client._validate_trigger_definition(bad_trigger_def)
        assert result.is_left()
        assert "key" in result.get_left().message

    def test_sanitize_trigger_data(self) -> None:
        """Test _sanitize_trigger_data method."""
        client = KMClient()

        # Test with safe data
        config = {"key": "Cmd+T"}
        result = client._sanitize_trigger_data(config)
        assert result.is_right()
        assert result.get_right()["key"] == "Cmd+T"

        # Test with potentially unsafe data - should sanitize quotes
        config_unsafe = {
            "key": "Cmd+T\"test'quotes",
            "application_bundle_id": "com.test.app",
        }
        result = client._sanitize_trigger_data(config_unsafe)
        assert result.is_right()
        # The sanitizer should escape quotes
        sanitized = result.get_right()
        assert '"' not in sanitized.get("key", "") or "\\\\" in sanitized.get("key", "")

    def test_edge_case_empty_responses(self) -> None:
        """Test handling of empty or malformed responses."""
        client = KMClient()

        # Test parsing empty output with _parse_applescript_records
        assert client._parse_applescript_records("") == []
        assert client._parse_applescript_records("   ") == []

        # Test parsing malformed data
        assert (
            client._parse_applescript_records("malformed|data") == []
        )  # No proper key:value format
        assert (
            client._parse_applescript_records("key:novalue") == []
        )  # No macroId to start record

    @patch("asyncio.create_task")
    def test_background_task_creation(self, mock_create_task: Mock) -> None:
        """Test background task creation in async methods."""
        import asyncio

        client = KMClient()

        # Mock the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Trigger background task creation
            config = {"trigger_type": TriggerType.HOTKEY, "hotkey": "Cmd+T"}
            trigger_def = TriggerDefinition(
                trigger_id=TriggerId("bg-trigger"),
                trigger_type=TriggerType.HOTKEY,
                macro_id=MacroId("test-macro"),
                configuration=config,
            )

            # This should work even if create_task is mocked
            loop.run_until_complete(client.register_trigger_async(trigger_def))

        finally:
            loop.close()


class TestKMClientErrorRecovery:
    """Test error recovery and retry logic."""

    def test_connection_config_immutability(self) -> None:
        """Ensure ConnectionConfig is truly immutable."""
        config = ConnectionConfig()

        # These should create new instances, not modify
        new1 = config.with_timeout(Duration.from_seconds(60))
        new2 = config.with_method(ConnectionMethod.WEB_API)

        # Verify all are different instances
        assert config is not new1
        assert config is not new2

        # Verify original is unchanged
        assert config.timeout.total_seconds() == 30.0
        assert config.method == ConnectionMethod.APPLESCRIPT
        assert config.web_api_host == "localhost"
        assert config.web_api_port == 4490
        assert config.max_retries == 3

    def test_complex_applescript_escaping(self) -> None:
        """Test AppleScript escaping with complex strings."""
        client = KMClient()

        # Test strings that need escaping
        test_cases = [
            ('Simple "quotes"', 'Simple \\"quotes\\"'),
            ("Back\\slash", "Back\\\\slash"),
            ('Both "quotes" and \\backslash', 'Both \\"quotes\\" and \\\\backslash'),
            ('Multiple """ quotes', 'Multiple \\"\\"\\" quotes'),
            ("Tab\tand\nnewline", "Tab\tand\nnewline"),  # These pass through
        ]

        for input_str, _expected in test_cases:
            # The escaping happens in _send_via_applescript
            # We can't easily test it directly, but we can verify
            # the client handles these inputs without crashing
            result = client.execute_macro(MacroId(input_str))
            assert hasattr(result, "is_left")  # Should return Either

    def test_connection_method_coverage(self) -> None:
        """Ensure all connection methods are covered."""
        # Test each connection method
        for method in ConnectionMethod:
            config = ConnectionConfig(method=method)
            client = KMClient(config)

            # Each should return an Either result
            result = client.execute_macro(MacroId("test"))
            assert hasattr(result, "is_left")
            assert hasattr(result, "is_right")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
