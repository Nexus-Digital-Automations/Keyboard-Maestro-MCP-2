"""Comprehensive coverage tests for src/integration/km_client.py.

This module provides comprehensive test coverage for the km_client module
to achieve the 95% minimum coverage requirement.
"""


import pytest
from src.core.types import Duration
from src.integration.km_client import (
    ConnectionConfig,
    ConnectionMethod,
    KMClient,
    KMError,
)


class TestConnectionMethod:
    """Test ConnectionMethod enum functionality."""

    def test_connection_method_values(self) -> None:
        """Test ConnectionMethod enum values."""
        assert ConnectionMethod.APPLESCRIPT.value == "applescript"
        assert ConnectionMethod.URL_SCHEME.value == "url_scheme"
        assert ConnectionMethod.WEB_API.value == "web_api"
        assert ConnectionMethod.REMOTE_TRIGGER.value == "remote_trigger"



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

        # setattr (vs `config.method = ...`) bypasses mypy's read-only
        # property check while still exercising the frozen-dataclass runtime.
        with pytest.raises(AttributeError):
            setattr(config, "method", ConnectionMethod.URL_SCHEME)  # noqa: B010

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






































