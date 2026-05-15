"""Simplified comprehensive tests for KMClient focusing on core coverage.

This module provides streamlined testing for the KMClient class to improve
coverage from 26% to 60%+ by focusing on the most important functionality.
"""

from unittest.mock import Mock, patch

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
)


class TestKMClientCore:
    """Test core KMClient functionality with simplified mocking."""

    @pytest.fixture
    def client(self):
        """Create KMClient instance."""
        return KMClient()

    def test_initialization(self) -> None:
        """Test client initialization."""
        # Default config
        client = KMClient()
        assert client.config.method == ConnectionMethod.APPLESCRIPT
        assert client.config.timeout.total_seconds() == 30.0

        # Custom config
        config = ConnectionConfig(
            method=ConnectionMethod.WEB_API,
            timeout=Duration.from_seconds(60),
            web_api_port=8080,
        )
        client = KMClient(config)
        assert client.config.method == ConnectionMethod.WEB_API
        assert client.config.timeout.total_seconds() == 60.0
        assert client.config.web_api_port == 8080

    def test_execute_macro(self, client: KMClient) -> None:
        """Test macro execution."""
        # Mock _send_command
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right(
                {
                    "output": "Executed",
                    "success": True,
                }
            )

            # Execute without trigger value
            result = client.execute_macro(MacroId("test-macro"))
            assert result.is_right()
            assert result.get_right()["success"] is True

            # Execute with trigger value
            result = client.execute_macro(
                MacroId("test-macro"), trigger_value="test-value"
            )
            assert result.is_right()
            mock_send.assert_called_with(
                "execute_macro",
                {"macro_id": "test-macro", "trigger_value": "test-value"},
            )

    def test_register_and_unregister_trigger(self, client: KMClient) -> None:
        """Test trigger registration and unregistration."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("trigger1"),
            macro_id=MacroId("macro1"),
            trigger_type=TriggerType.HOTKEY,
            configuration={"key": "cmd+shift+a"},
            enabled=True,
        )

        # Test registration
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right({"trigger_id": "trigger1"})

            result = client.register_trigger(trigger_def)
            assert result.is_right()
            assert result.get_right() == "trigger1"

        # Test unregistration
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right({"success": True})

            result = client.unregister_trigger(TriggerId("trigger1"))
            assert result.is_right()
            assert result.get_right() is True

    def test_macro_listing_methods(self, client: KMClient) -> None:
        """Test various macro listing methods."""
        test_macros = [
            {"name": "Macro 1", "id": "123", "group": "Test"},
            {"name": "Macro 2", "id": "456", "group": "Test"},
        ]

        # Test get_macro_list
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right({"macros": test_macros})

            result = client.get_macro_list()
            assert result.is_right()
            assert len(result.get_right()) == 2

            # With group filter
            result = client.get_macro_list(group_filter="Test")
            assert result.is_right()
            mock_send.assert_called_with("list_macros", {"group_filter": "Test"})

        # Test list_macros (alias)
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right({"macros": test_macros})

            result = client.list_macros()
            assert result.is_right()
            assert len(result.get_right()) == 2

    def test_create_macro(self, client: KMClient) -> None:
        """Test macro creation."""
        macro_data = {
            "name": "New Macro",
            "group": "Test Group",
            "actions": [],
        }

        # Success case
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right({"macro_id": "new-123"})

            result = client.create_macro(macro_data)
            assert result.is_right()
            assert result.get_right()["macro_id"] == "new-123"
            assert result.get_right()["success"] is True
            assert result.get_right()["created"] is True

        # Missing name
        result = client.create_macro({"group": "Test"})
        assert result.is_left()
        assert result.get_left().code == "VALIDATION_ERROR"

    def test_list_macros_with_details(self, client: KMClient) -> None:
        """Test detailed macro listing."""
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right(
                {
                    "macros": [
                        {
                            "name": "Macro 1",
                            "id": "123",
                            "action_count": 5,
                            "trigger_count": 2,
                            "created_date": "2024-01-01",
                            "enabled": True,
                        }
                    ]
                }
            )

            result = client.list_macros_with_details()
            assert result.is_right()
            macros = result.get_right()
            assert len(macros) == 1
            assert macros[0]["details"] is True
            assert macros[0]["actions"] == 5
            assert macros[0]["triggers"] == 2
            assert "metadata" in macros[0]

    def test_connection_check(self, client: KMClient) -> None:
        """Test connection checking."""
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right({"alive": True})

            result = client.check_connection()
            assert result.is_right()
            assert result.get_right() is True

    def test_macro_status(self, client: KMClient) -> None:
        """Test getting macro status."""
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right(
                {
                    "status": {
                        "running": True,
                        "progress": 50,
                    }
                }
            )

            result = client.get_macro_status(MacroId("123"))
            assert result.is_right()
            status = result.get_right()
            assert status["running"] is True
            assert status["progress"] == 50

    def test_trigger_management(self, client: KMClient) -> None:
        """Test trigger activation, deactivation, listing."""
        # Activate trigger
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right({"success": True})

            result = client.activate_trigger(TriggerId("trigger1"))
            assert result.is_right()
            assert result.get_right() is True

        # Deactivate trigger
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right({"success": True})

            result = client.deactivate_trigger(TriggerId("trigger1"))
            assert result.is_right()
            assert result.get_right() is True

        # List triggers
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right(
                {
                    "triggers": [
                        {"id": "trigger1", "type": "hotkey"},
                        {"id": "trigger2", "type": "typed_string"},
                    ]
                }
            )

            result = client.list_triggers()
            assert result.is_right()
            triggers = result.get_right()
            assert len(triggers) == 2

        # Get trigger status
        with patch.object(client, "_send_command") as mock_send:
            mock_send.return_value = Either.right(
                {"status": {"enabled": True, "last_fired": "2024-01-01"}}
            )

            result = client.get_trigger_status(TriggerId("trigger1"))
            assert result.is_right()
            assert result.get_right()["enabled"] is True


class TestKMClientAsync:
    """Test async methods."""

    @pytest.fixture
    def client(self):
        """Create KMClient instance."""
        return KMClient()

    @pytest.mark.asyncio
    async def test_register_trigger_async(self, client: KMClient) -> None:
        """Test async trigger registration."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("trigger1"),
            macro_id=MacroId("macro1"),
            trigger_type=TriggerType.HOTKEY,
            configuration={"key": "cmd+shift+a"},
        )

        # Mock the required methods
        client._validate_trigger_definition = Mock(return_value=Either.right(True))
        client._sanitize_trigger_data = Mock(
            return_value=Either.right({"key": "cmd+shift+a"})
        )
        client._build_trigger_script_safe = Mock(return_value=Either.right("script"))

        # _execute_applescript_safe is async, so we need to mock it as an async function
        async def mock_execute(script):
            return Either.right("Success")

        client._execute_applescript_safe = Mock(side_effect=mock_execute)

        result = await client.register_trigger_async(trigger_def)
        assert result.is_right()
        assert result.get_right() == TriggerId("trigger1")

    @pytest.mark.asyncio
    async def test_activate_trigger_async(self, client: KMClient) -> None:
        """Test async trigger activation."""
        with patch.object(client, "activate_trigger") as mock_activate:
            mock_activate.return_value = Either.right(True)

            result = await client.activate_trigger_async(TriggerId("trigger1"))
            assert result.is_right()
            assert result.get_right() is True

    @pytest.mark.asyncio
    async def test_deactivate_trigger_async(self, client: KMClient) -> None:
        """Test async trigger deactivation."""
        with patch.object(client, "deactivate_trigger") as mock_deactivate:
            mock_deactivate.return_value = Either.right(True)

            result = await client.deactivate_trigger_async(TriggerId("trigger1"))
            assert result.is_right()
            assert result.get_right() is True

    @pytest.mark.asyncio
    async def test_list_triggers_async(self, client: KMClient) -> None:
        """Test async trigger listing."""
        with patch.object(client, "list_triggers") as mock_list:
            mock_list.return_value = Either.right([{"id": "trigger1"}])

            result = await client.list_triggers_async()
            assert result.is_right()
            assert len(result.get_right()) == 1

    @pytest.mark.asyncio
    async def test_list_macros_async(self, client: KMClient) -> None:
        """Test async macro listing with fallback."""
        # Test AppleScript success
        with patch.object(client, "_list_macros_applescript") as mock_as:
            mock_as.return_value = Either.right([{"name": "Macro 1"}])

            result = await client.list_macros_async()
            assert result.is_right()
            assert len(result.get_right()) == 1

        # Test fallback to Web API
        with patch.object(client, "_list_macros_applescript") as mock_as:
            with patch.object(client, "_list_macros_web_api") as mock_web:
                mock_as.return_value = Either.left(KMError.connection_error("Failed"))
                mock_web.return_value = Either.right([{"name": "Web Macro"}])

                result = await client.list_macros_async()
                assert result.is_right()
                assert result.get_right()[0]["name"] == "Web Macro"

        # Test both fail
        with patch.object(client, "_list_macros_applescript") as mock_as:
            with patch.object(client, "_list_macros_web_api") as mock_web:
                mock_as.return_value = Either.left(
                    KMError.connection_error("AS Failed")
                )
                mock_web.return_value = Either.left(
                    KMError.connection_error("Web Failed")
                )

                result = await client.list_macros_async()
                assert result.is_left()
                assert "Cannot connect" in result.get_left().message


class TestKMClientPrivateMethods:
    """Test private methods for better coverage."""

    def test_safe_send_applescript(self) -> None:
        """Test _safe_send with AppleScript."""
        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)

        # Mock successful ping
        with patch(
            "src.commands.secure_subprocess.get_secure_subprocess_manager"
        ) as mock_get:
            mock_manager = Mock()
            mock_get.return_value = mock_manager

            mock_result = Mock()
            mock_result.stdout = "true"
            mock_result.stderr = ""
            mock_result.returncode = 0
            mock_manager.execute_secure_command.return_value = mock_result

            result = KMClient._safe_send(config, "ping", {})
            assert result.is_right()
            assert result.get_right()["alive"] is True

    def test_safe_send_web_api(self) -> None:
        """Test _safe_send with Web API."""
        config = ConnectionConfig(method=ConnectionMethod.WEB_API)

        # Web API only supports execute_macro command
        with patch("src.integration.km_client.httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "Success"
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client_class.return_value.__exit__.return_value = None

            result = KMClient._safe_send(
                config, "execute_macro", {"macro_id": "test-macro"}
            )
            assert result.is_right()
            assert result.get_right()["success"] is True

    def test_safe_send_unsupported(self) -> None:
        """Test _safe_send with unsupported method."""
        # Most commands don't support REMOTE_TRIGGER
        config = ConnectionConfig(method=ConnectionMethod.REMOTE_TRIGGER)

        result = KMClient._safe_send(config, "unsupported_command", {})
        assert result.is_left()
        assert result.get_left().code == "CONNECTION_ERROR"


class TestKMError:
    """Test KMError factory methods."""

    def test_error_types(self) -> None:
        """Test all error type creation."""
        # Connection error
        error = KMError.connection_error("Failed to connect")
        assert error.code == "CONNECTION_ERROR"
        assert error.message == "Failed to connect"

        # Execution error
        error = KMError.execution_error("Failed", {"detail": "value"})
        assert error.code == "EXECUTION_ERROR"
        assert error.details == {"detail": "value"}

        # Timeout error
        error = KMError.timeout_error(Duration.from_seconds(30))
        assert error.code == "TIMEOUT_ERROR"
        assert "30" in error.message
        assert error.retry_after is not None

        # Validation error
        error = KMError.validation_error("Invalid input")
        assert error.code == "VALIDATION_ERROR"

        # Not found error
        error = KMError.not_found_error("Macro not found")
        assert error.code == "NOT_FOUND_ERROR"

        # Security error
        error = KMError.security_error("Unauthorized")
        assert error.code == "SECURITY_ERROR"


class TestConnectionConfig:
    """Test ConnectionConfig functionality."""

    def test_immutability(self) -> None:
        """Test that ConnectionConfig is immutable."""
        config = ConnectionConfig()
        new_config = config.with_timeout(Duration.from_seconds(60))

        # Original unchanged
        assert config.timeout.total_seconds() == 30.0
        # New config has new value
        assert new_config.timeout.total_seconds() == 60.0

        # Test with_method
        new_config = config.with_method(ConnectionMethod.WEB_API)
        assert config.method == ConnectionMethod.APPLESCRIPT
        assert new_config.method == ConnectionMethod.WEB_API


class TestTriggerDefinition:
    """Test TriggerDefinition functionality."""

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("trigger1"),
            macro_id=MacroId("macro1"),
            trigger_type=TriggerType.HOTKEY,
            configuration={"key": "cmd+a"},
            enabled=False,
        )

        result = trigger_def.to_dict()
        assert result["trigger_id"] == "trigger1"
        assert result["macro_id"] == "macro1"
        assert result["trigger_type"] == "hotkey"  # Enum value lowercase
        assert result["configuration"] == {"key": "cmd+a"}
        assert result["enabled"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
