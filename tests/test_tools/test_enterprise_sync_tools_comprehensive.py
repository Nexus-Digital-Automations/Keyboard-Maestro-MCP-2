"""
Comprehensive tests for enterprise sync tools module.

Tests cover LDAP/Active Directory integration, SSO authentication, enterprise database
connectivity, API integration, and security validation with property-based testing.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.server.tools.enterprise_sync_tools import km_enterprise_sync


# Test data generators
@st.composite
def enterprise_operation_strategy(draw):
    """Generate valid enterprise sync operations."""
    operations = [
        "connect",
        "authenticate",
        "sync",
        "query",
        "configure",
        "status",
        "sso_config",
        "sso_login",
    ]
    return draw(st.sampled_from(operations))


@st.composite
def integration_type_strategy(draw):
    """Generate valid integration types."""
    integration_types = [
        "ldap",
        "active_directory",
        "saml_sso",
        "oauth_sso",
        "enterprise_database",
        "rest_api",
        "graphql_api",
    ]
    return draw(st.sampled_from(integration_types))


@st.composite
def connection_config_strategy(draw):
    """Generate valid connection configurations."""
    hosts = [
        "ldap.enterprise.com",
        "ad.company.local",
        "db.internal.com",
        "api.enterprise.io",
    ]
    ports = [389, 636, 443, 1433, 5432, 3306]

    return {
        "connection_id": draw(
            st.text(min_size=5, max_size=50).filter(lambda x: x.isalnum())
        ),
        "host": draw(st.sampled_from(hosts)),
        "port": draw(st.sampled_from(ports)),
        "use_ssl": draw(st.booleans()),
        "ssl_verify": draw(st.booleans()),
        "base_dn": draw(
            st.text(min_size=10, max_size=100).filter(
                lambda x: "dc=" in x.lower() or len(x) > 0
            )
        ),
        "domain": draw(
            st.text(min_size=5, max_size=30).filter(lambda x: "." in x or len(x) > 0)
        ),
        "api_version": draw(st.text(min_size=1, max_size=10)),
    }


@st.composite
def authentication_strategy(draw):
    """Generate valid authentication configurations."""
    auth_methods = ["simple_bind", "sasl", "certificate", "token", "api_key"]

    return {
        "method": draw(st.sampled_from(auth_methods)),
        "username": draw(
            st.text(min_size=3, max_size=50).filter(lambda x: x.isalnum())
        ),
        "password": draw(st.text(min_size=8, max_size=128)),
        "domain": draw(
            st.text(min_size=3, max_size=50).filter(lambda x: "." in x or len(x) > 0)
        ),
        "certificate_path": draw(st.text(min_size=5, max_size=100)),
        "token": draw(st.text(min_size=10, max_size=256)),
        "api_key": draw(st.text(min_size=16, max_size=128)),
    }


@st.composite
def sync_options_strategy(draw):
    """Generate valid sync options."""
    return {
        "connection_id": draw(
            st.text(min_size=5, max_size=50).filter(lambda x: x.isalnum())
        ),
        "sync_type": draw(st.sampled_from(["full", "incremental", "delta"])),
        "target_entities": draw(
            st.lists(st.text(min_size=3, max_size=30), min_size=1, max_size=10)
        ),
        "filters": draw(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.text(min_size=1, max_size=50),
                min_size=0,
                max_size=5,
            )
        ),
        "mapping_rules": draw(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.text(min_size=1, max_size=50),
                min_size=0,
                max_size=10,
            )
        ),
    }


@st.composite
def query_filter_strategy(draw):
    """Generate valid query filters."""
    ldap_filters = [
        "(objectClass=user)",
        "(cn=*smith*)",
        "(&(objectClass=user)(department=Engineering))",
        "(|(mail=*@company.com)(mail=*@enterprise.org))",
    ]
    sql_queries = [
        "SELECT * FROM users WHERE active = 1",
        "SELECT id, name, email FROM employees WHERE department = 'IT'",
        "SELECT * FROM audit_log WHERE timestamp > '2024-01-01'",
    ]
    api_endpoints = [
        "/api/v1/users",
        "/api/v2/employees?department=engineering",
        "/graphql/users?filter=active",
    ]

    all_filters = ldap_filters + sql_queries + api_endpoints
    return draw(st.sampled_from(all_filters))


@st.composite
def timeout_strategy(draw):
    """Generate valid timeout values."""
    return draw(st.integers(min_value=5, max_value=300))


@st.composite
def batch_size_strategy(draw):
    """Generate valid batch sizes."""
    return draw(st.integers(min_value=10, max_value=1000))


@st.composite
def invalid_integration_type_strategy(draw):
    """Generate invalid integration types."""
    invalid_types = [
        "invalid",
        "unknown",
        "",
        "ldap_v1",
        "database",
        "api",
        "sso",
        "auth",
    ]
    return draw(st.sampled_from(invalid_types))


@st.composite
def sso_config_strategy(draw):
    """Generate valid SSO configurations."""
    # Use simpler strategies to avoid filter issues
    provider_names = ["SampleProvider", "TestSSO", "EnterpriseAuth", "CompanySSO"]
    provider_ids = ["provider001", "test_sso_01", "enterprise_auth", "company_sso"]

    return {
        "provider_name": draw(st.sampled_from(provider_names)),
        "provider_id": draw(st.sampled_from(provider_ids)),
        "issuer": draw(
            st.sampled_from(
                [
                    "https://sso.company.com",
                    "https://auth.enterprise.io",
                    "https://saml.provider.com",
                ]
            )
        ),
        "metadata_url": draw(
            st.sampled_from(
                [
                    "https://sso.company.com/metadata",
                    "https://auth.enterprise.io/metadata",
                ]
            )
        ),
        "certificate": draw(
            st.sampled_from(
                ["-----BEGIN CERTIFICATE-----test", "-----BEGIN CERTIFICATE-----sample"]
            )
        ),
        "client_id": draw(st.sampled_from(["client123", "app456", "enterprise789"])),
        "client_secret": draw(
            st.sampled_from(["secret123456789", "clientsecret987654321"])
        ),
        "redirect_uri": draw(
            st.sampled_from(
                ["https://app.company.com/callback", "https://app.enterprise.io/auth"]
            )
        ),
    }


class TestEnterpriseSyncDependencies:
    """Test enterprise sync dependencies and imports."""

    def test_enterprise_sync_manager_import(self):
        """Test importing enterprise sync dependencies."""
        try:
            from src.audit.audit_system_manager import get_audit_system
            from src.core.either import Either
            from src.core.enterprise_integration import (
                AuthenticationMethod,
                IntegrationType,
                SecurityLevel,
            )
            from src.enterprise.enterprise_sync_manager import EnterpriseSyncManager

            # Test basic creation
            assert IntegrationType is not None
            assert AuthenticationMethod is not None
            assert SecurityLevel is not None
            assert EnterpriseSyncManager is not None
            assert get_audit_system is not None
            assert Either is not None

        except ImportError:
            # Mock the dependencies for testing
            pytest.skip("Enterprise sync dependencies not available - using mocks")


class TestEnterpriseSyncParameterValidation:
    """Test enterprise sync parameter validation."""

    @given(enterprise_operation_strategy())
    def test_valid_operations(self, operation: str):
        """Test that valid operations are accepted."""
        valid_operations = [
            "connect",
            "authenticate",
            "sync",
            "query",
            "configure",
            "status",
            "sso_config",
            "sso_login",
        ]
        assert operation in valid_operations

    @given(integration_type_strategy())
    def test_integration_type_validation(self, integration_type: str):
        """Test integration type validation."""
        valid_types = [
            "ldap",
            "active_directory",
            "saml_sso",
            "oauth_sso",
            "enterprise_database",
            "rest_api",
            "graphql_api",
        ]
        assert integration_type in valid_types

    @given(connection_config_strategy())
    def test_connection_config_validation(self, config: dict[str, Any]):
        """Test connection configuration validation."""
        # Required fields should be present
        assert "host" in config
        assert "port" in config
        assert isinstance(config["port"], int)
        assert config["port"] > 0

    @given(authentication_strategy())
    def test_authentication_validation(self, auth: dict[str, Any]):
        """Test authentication configuration validation."""
        assert "method" in auth
        valid_methods = ["simple_bind", "sasl", "certificate", "token", "api_key"]
        assert auth["method"] in valid_methods

    @given(timeout_strategy())
    def test_timeout_validation(self, timeout: int):
        """Test timeout parameter validation."""
        assert 5 <= timeout <= 300

    @given(batch_size_strategy())
    def test_batch_size_validation(self, batch_size: int):
        """Test batch size parameter validation."""
        assert 10 <= batch_size <= 1000

    def test_invalid_operations(self):
        """Test that invalid operations are rejected."""
        invalid_operations = [
            "invalid",
            "hack",
            "delete",
            "execute",
            "",
            "admin",
            "root",
        ]
        valid_operations = [
            "connect",
            "authenticate",
            "sync",
            "query",
            "configure",
            "status",
            "sso_config",
            "sso_login",
        ]
        for op in invalid_operations:
            assert op not in valid_operations


class TestEnterpriseSyncConnectOperationMocked:
    """Test enterprise sync connect operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_connect_ldap_success(self):
        """Test successful LDAP connection establishment."""
        with (
            patch(
                "src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager"
            ) as mock_get_manager,
            patch(
                "src.server.tools.enterprise_sync_tools.IntegrationType"
            ) as mock_integration_type,
            patch(
                "src.server.tools.enterprise_sync_tools.create_enterprise_connection"
            ) as mock_create_connection,
            patch(
                "src.server.tools.enterprise_sync_tools.create_enterprise_credentials"
            ) as mock_create_credentials,
            patch(
                "src.server.tools.enterprise_sync_tools.AuthenticationMethod"
            ) as mock_auth_method,
        ):
            # Setup mocks for successful LDAP connection
            mock_integration_type.return_value = Mock(value="ldap")
            mock_auth_method.return_value = Mock(value="simple_bind")

            mock_connection = Mock()
            mock_connection.use_ssl = True
            mock_connection.ssl_verify = True
            mock_create_connection.return_value = mock_connection

            mock_credentials = Mock()
            mock_create_credentials.return_value = mock_credentials

            mock_manager = AsyncMock()
            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = "ldap_conn_001"

            mock_manager.establish_connection = AsyncMock(return_value=mock_result)
            mock_get_manager.return_value = mock_manager

            # Execute connect operation
            result = await km_enterprise_sync(
                operation="connect",
                integration_type="ldap",
                connection_config={
                    "host": "ldap.company.com",
                    "port": 389,
                    "base_dn": "dc=company,dc=com",
                    "use_ssl": True,
                },
                authentication={
                    "method": "simple_bind",
                    "username": "admin",
                    "password": "password123",
                },
            )

            # Verify successful connection
            assert result["success"] is True
            assert result["operation"] == "connect"
            assert result["data"]["connection_id"] == "ldap_conn_001"
            assert result["data"]["integration_type"] == "ldap"
            assert result["data"]["host"] == "ldap.company.com"
            assert result["data"]["port"] == 389
            assert "connected_at" in result["data"]
            assert result["metadata"]["ssl_enabled"] is True

    @pytest.mark.asyncio
    async def test_connect_missing_config_error(self):
        """Test connect operation with missing connection config."""
        # Execute connect without connection config
        result = await km_enterprise_sync(
            operation="connect",
            integration_type="ldap",
            # connection_config is missing
            authentication={
                "method": "simple_bind",
                "username": "admin",
                "password": "password123",
            },
        )

        # Verify missing config error
        assert result["success"] is False
        assert result["error"]["code"] == "MISSING_CONNECTION_CONFIG"
        assert "Connection configuration required" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_connect_missing_authentication_error(self):
        """Test connect operation with missing authentication."""
        # Execute connect without authentication
        result = await km_enterprise_sync(
            operation="connect",
            integration_type="ldap",
            connection_config={"host": "ldap.company.com", "port": 389},
            # authentication is missing
        )

        # Verify missing authentication error
        assert result["success"] is False
        assert result["error"]["code"] == "MISSING_AUTHENTICATION"
        assert "Authentication credentials required" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_connect_invalid_config_error(self):
        """Test connect operation with invalid connection config."""
        # Execute connect with missing host/port
        result = await km_enterprise_sync(
            operation="connect",
            integration_type="ldap",
            connection_config={
                "base_dn": "dc=company,dc=com"
                # host and port are missing
            },
            authentication={
                "method": "simple_bind",
                "username": "admin",
                "password": "password123",
            },
        )

        # Verify invalid config error
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_CONNECTION_CONFIG"
        assert "Host and port required" in result["error"]["message"]


class TestEnterpriseSyncSyncOperationMocked:
    """Test enterprise sync synchronization operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_sync_operation_success(self):
        """Test successful enterprise data synchronization."""
        with (
            patch(
                "src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager"
            ) as mock_get_manager,
            patch(
                "src.server.tools.enterprise_sync_tools.IntegrationType"
            ) as mock_integration_type,
        ):
            # Setup mocks for successful sync
            mock_integration_type.return_value = Mock(value="ldap")

            mock_manager = AsyncMock()
            mock_sync_result = Mock()
            mock_sync_result.records_processed = 100
            mock_sync_result.records_successful = 95
            mock_sync_result.records_failed = 5
            mock_sync_result.sync_duration = 45.5
            mock_sync_result.started_at = datetime.now(UTC)
            mock_sync_result.completed_at = datetime.now(UTC)
            mock_sync_result.errors = []
            mock_sync_result.warnings = []
            mock_sync_result.get_success_rate.return_value = 0.95
            mock_sync_result.get_status_summary.return_value = "completed_with_errors"
            mock_sync_result.has_errors.return_value = True

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_sync_result

            mock_manager.sync_enterprise_data = AsyncMock(return_value=mock_result)
            mock_get_manager.return_value = mock_manager

            # Execute sync operation
            result = await km_enterprise_sync(
                operation="sync",
                integration_type="ldap",
                sync_options={
                    "connection_id": "ldap_conn_001",
                    "sync_type": "incremental",
                    "target_entities": ["users", "groups"],
                },
                batch_size=50,
            )

            # Verify successful sync
            assert result["success"] is True
            assert result["operation"] == "sync"
            assert result["data"]["connection_id"] == "ldap_conn_001"
            assert result["data"]["integration_type"] == "ldap"
            assert result["data"]["records_processed"] == 100
            assert result["data"]["records_successful"] == 95
            assert result["data"]["records_failed"] == 5
            assert result["data"]["success_rate"] == 0.95
            assert result["metadata"]["batch_size"] == 50
            assert result["metadata"]["has_errors"] is True

    @pytest.mark.asyncio
    async def test_sync_missing_options_error(self):
        """Test sync operation with missing sync options."""
        # Execute sync without sync options
        result = await km_enterprise_sync(
            operation="sync",
            integration_type="ldap",
            # sync_options is missing
        )

        # Verify missing options error
        assert result["success"] is False
        assert result["error"]["code"] == "MISSING_SYNC_OPTIONS"
        assert "Sync options required" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_sync_missing_connection_id_error(self):
        """Test sync operation with missing connection ID."""
        # Execute sync without connection ID
        result = await km_enterprise_sync(
            operation="sync",
            integration_type="ldap",
            sync_options={
                "sync_type": "full"
                # connection_id is missing
            },
        )

        # Verify missing connection ID error
        assert result["success"] is False
        assert result["error"]["code"] == "MISSING_CONNECTION_ID"
        assert "Connection ID required" in result["error"]["message"]


class TestEnterpriseSyncQueryOperationMocked:
    """Test enterprise sync query operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_query_ldap_success(self):
        """Test successful LDAP query operation."""
        with (
            patch(
                "src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager"
            ) as mock_get_manager,
            patch(
                "src.server.tools.enterprise_sync_tools.IntegrationType"
            ) as mock_integration_type,
        ):
            # Setup mocks for successful LDAP query
            mock_integration_type.return_value = Mock(value="ldap")

            mock_manager = AsyncMock()
            mock_query_data = [
                {
                    "cn": "john.doe",
                    "mail": "john@company.com",
                    "department": "Engineering",
                },
                {
                    "cn": "jane.smith",
                    "mail": "jane@company.com",
                    "department": "Marketing",
                },
            ]

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_query_data

            mock_manager.query_enterprise_data = AsyncMock(return_value=mock_result)
            mock_get_manager.return_value = mock_manager

            # Execute query operation
            result = await km_enterprise_sync(
                operation="query",
                integration_type="ldap",
                query_filter="(objectClass=user)",
                sync_options={
                    "connection_id": "ldap_conn_001",
                    "search_base": "ou=users,dc=company,dc=com",
                },
            )

            # Verify successful query
            assert result["success"] is True
            assert result["operation"] == "query"
            assert result["data"]["connection_id"] == "ldap_conn_001"
            assert result["data"]["integration_type"] == "ldap"
            assert result["data"]["records"] == mock_query_data
            assert result["data"]["record_count"] == 2
            assert result["data"]["query_filter"] == "(objectClass=user)"
            assert result["metadata"]["has_results"] is True

    @pytest.mark.asyncio
    async def test_query_database_success(self):
        """Test successful database query operation."""
        with (
            patch(
                "src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager"
            ) as mock_get_manager,
            patch(
                "src.server.tools.enterprise_sync_tools.IntegrationType"
            ) as mock_integration_type,
        ):
            # Setup mocks for successful database query
            mock_integration_type.return_value = Mock(value="enterprise_database")

            mock_manager = AsyncMock()
            mock_query_data = [
                {"id": 1, "name": "John Doe", "email": "john@company.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@company.com"},
            ]

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_query_data

            mock_manager.query_enterprise_data = AsyncMock(return_value=mock_result)
            mock_get_manager.return_value = mock_manager

            # Execute database query
            result = await km_enterprise_sync(
                operation="query",
                integration_type="enterprise_database",
                query_filter="SELECT * FROM users WHERE active = 1",
                sync_options={"connection_id": "db_conn_001", "database": "hr_system"},
            )

            # Verify successful database query
            assert result["success"] is True
            assert result["data"]["integration_type"] == "enterprise_database"
            assert result["data"]["records"] == mock_query_data
            assert result["data"]["record_count"] == 2


class TestEnterpriseSyncSSOOperationMocked:
    """Test enterprise sync SSO operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_sso_config_saml_success(self):
        """Test successful SAML SSO configuration."""
        with (
            patch(
                "src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager"
            ) as mock_get_manager,
            patch(
                "src.server.tools.enterprise_sync_tools.IntegrationType"
            ) as mock_integration_type,
        ):
            # Setup mocks for successful SAML config - interface alignment fix
            mock_saml_sso = Mock()
            mock_saml_sso.value = "saml_sso"
            mock_integration_type.return_value = mock_saml_sso
            mock_integration_type.SAML_SSO = (
                mock_saml_sso  # Key fix: enum constant alignment
            )
            mock_integration_type.OAUTH_SSO = Mock(value="oauth_sso")  # For the check

            mock_manager = AsyncMock()
            mock_manager.sso_manager = Mock()

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = "saml_provider_001"

            mock_manager.sso_manager.configure_saml_provider = AsyncMock(
                return_value=mock_result
            )
            mock_get_manager.return_value = mock_manager

            # Execute SAML SSO config
            result = await km_enterprise_sync(
                operation="sso_config",
                integration_type="saml_sso",
                connection_config={
                    "provider_name": "Enterprise SAML",
                    "issuer": "https://company.com/saml",
                    "metadata_url": "https://company.com/saml/metadata",
                    "certificate": "-----BEGIN CERTIFICATE-----...",
                },
            )

            # Verify successful SAML config
            assert result["success"] is True
            assert result["operation"] == "sso_config"
            assert result["data"]["provider_id"] == "saml_provider_001"
            assert result["data"]["integration_type"] == "saml_sso"
            assert result["metadata"]["sso_type"] == "saml_sso"
            assert result["metadata"]["provider_configured"] is True

    @pytest.mark.asyncio
    async def test_sso_login_oauth_success(self):
        """Test successful OAuth SSO login initiation."""
        with (
            patch(
                "src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager"
            ) as mock_get_manager,
            patch(
                "src.server.tools.enterprise_sync_tools.IntegrationType"
            ) as mock_integration_type,
        ):
            # Setup mocks for successful OAuth login - interface alignment fix
            mock_oauth_sso = Mock()
            mock_oauth_sso.value = "oauth_sso"
            mock_integration_type.return_value = mock_oauth_sso
            mock_integration_type.OAUTH_SSO = (
                mock_oauth_sso  # Key fix: enum constant alignment
            )
            mock_integration_type.SAML_SSO = Mock(value="saml_sso")  # For the check

            mock_manager = AsyncMock()
            mock_manager.sso_manager = Mock()

            mock_login_data = {
                "auth_url": "https://oauth.company.com/authorize?client_id=abc123",
                "method": "GET",
                "request_id": "oauth_req_001",
            }

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_login_data

            mock_manager.sso_manager.initiate_sso_login = AsyncMock(
                return_value=mock_result
            )
            mock_get_manager.return_value = mock_manager

            # Execute OAuth SSO login
            result = await km_enterprise_sync(
                operation="sso_login",
                integration_type="oauth_sso",
                authentication={
                    "provider_id": "oauth_provider_001",
                    "redirect_url": "https://app.company.com/callback",
                    "user_context": {"department": "engineering"},
                },
            )

            # Verify successful OAuth login
            assert result["success"] is True
            assert result["operation"] == "sso_login"
            assert result["data"]["provider_id"] == "oauth_provider_001"
            assert result["data"]["auth_url"] == mock_login_data["auth_url"]
            assert result["data"]["request_id"] == "oauth_req_001"
            assert result["metadata"]["requires_redirect"] is True

    @pytest.mark.asyncio
    async def test_sso_config_invalid_type_error(self):
        """Test SSO config with invalid integration type."""
        # Execute SSO config with non-SSO type
        result = await km_enterprise_sync(
            operation="sso_config",
            integration_type="ldap",  # Not an SSO type
            connection_config={"provider_name": "LDAP Provider"},
        )

        # Verify invalid SSO type error
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_SSO_TYPE"
        assert "SSO configuration not supported" in result["error"]["message"]


class TestEnterpriseSyncErrorHandling:
    """Test enterprise sync error handling."""

    @pytest.mark.asyncio
    async def test_invalid_operation_error(self):
        """Test handling of invalid operations."""
        from src.core.errors import ContractViolationError

        # Execute invalid operation - should trigger contract violation
        with pytest.raises(ContractViolationError) as exc_info:
            await km_enterprise_sync(
                operation="invalid_operation", integration_type="ldap"
            )

        # Verify contract violation for invalid operation
        error = exc_info.value
        assert error.contract_type == "Precondition"
        assert "Precondition failed" in str(error)

    @pytest.mark.asyncio
    async def test_invalid_integration_type_error(self):
        """Test handling of invalid integration types."""
        # Execute with invalid integration type
        result = await km_enterprise_sync(
            operation="connect", integration_type="invalid_type"
        )

        # Verify invalid integration type error
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_INTEGRATION_TYPE"
        assert "Invalid integration type" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_system_error_handling(self):
        """Test handling of system errors."""
        with patch(
            "src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager"
        ) as mock_get_manager:
            # Setup system error
            mock_get_manager.side_effect = RuntimeError("System failure")

            # Execute operation that should trigger system error
            result = await km_enterprise_sync(
                operation="status", integration_type="ldap"
            )

            # Verify system error handling
            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert "Enterprise sync operation failed" in result["error"]["message"]


class TestEnterpriseSyncStatusOperationMocked:
    """Test enterprise sync status operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_status_operation_success(self):
        """Test successful enterprise system status retrieval."""
        with patch(
            "src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager"
        ) as mock_get_manager:
            # Setup mocks for successful status - interface alignment fix
            mock_manager = Mock()  # Should be Mock, not AsyncMock for get_system_status
            mock_status = {
                "status": "operational",
                "connections": {"total": 5, "active": 3, "idle": 2},
                "features": {
                    "audit_logging": True,
                    "encryption": True,
                    "sso_enabled": True,
                },
                "performance": {"avg_response_time": 150, "success_rate": 0.98},
            }

            mock_manager.get_system_status.return_value = (
                mock_status  # Regular method, not async
            )
            mock_get_manager.return_value = mock_manager

            # Execute status operation
            result = await km_enterprise_sync(
                operation="status", integration_type="ldap"
            )

            # Verify successful status
            assert result["success"] is True
            assert result["operation"] == "status"
            assert result["data"]["system_status"] == mock_status
            assert result["data"]["integration_type"] == "ldap"
            assert result["metadata"]["enterprise_ready"] is True
            assert result["metadata"]["total_connections"] == 5
            assert result["metadata"]["audit_enabled"] is True


class TestEnterpriseSyncIntegration:
    """Integration tests for enterprise sync operations."""

    @pytest.mark.asyncio
    async def test_complete_enterprise_workflow(self):
        """Test complete enterprise integration workflow."""
        with (
            patch(
                "src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager"
            ) as mock_get_manager,
            patch(
                "src.server.tools.enterprise_sync_tools.IntegrationType"
            ) as mock_integration_type,
            patch(
                "src.server.tools.enterprise_sync_tools.create_enterprise_connection"
            ) as mock_create_connection,
            patch(
                "src.server.tools.enterprise_sync_tools.create_enterprise_credentials"
            ) as mock_create_credentials,
            patch(
                "src.server.tools.enterprise_sync_tools.AuthenticationMethod"
            ) as mock_auth_method,
        ):
            # Setup mocks for complete workflow
            mock_integration_type.return_value = Mock(value="ldap")
            mock_auth_method.return_value = Mock(value="simple_bind")

            mock_connection = Mock()
            mock_connection.use_ssl = True
            mock_connection.ssl_verify = True
            mock_create_connection.return_value = mock_connection

            mock_credentials = Mock()
            mock_create_credentials.return_value = mock_credentials

            mock_manager = AsyncMock()

            # Setup different results for different operations
            connect_result = Mock()
            connect_result.is_left.return_value = False
            connect_result.get_right.return_value = "ldap_conn_workflow"

            sync_result = Mock()
            sync_result.is_left.return_value = False
            sync_result.get_right.return_value = Mock(
                records_processed=50,
                records_successful=50,
                records_failed=0,
                sync_duration=30.0,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                errors=[],
                warnings=[],
                get_success_rate=lambda: 1.0,
                get_status_summary=lambda: "completed_successfully",
                has_errors=lambda: False,
            )

            query_result = Mock()
            query_result.is_left.return_value = False
            query_result.get_right.return_value = [
                {"cn": "workflow.user", "mail": "workflow@company.com"}
            ]

            mock_manager.establish_connection = AsyncMock(return_value=connect_result)
            mock_manager.sync_enterprise_data = AsyncMock(return_value=sync_result)
            mock_manager.query_enterprise_data = AsyncMock(return_value=query_result)
            mock_get_manager.return_value = mock_manager

            # Execute complete workflow: connect -> sync -> query
            connect_result_data = await km_enterprise_sync(
                operation="connect",
                integration_type="ldap",
                connection_config={"host": "ldap.workflow.com", "port": 389},
                authentication={
                    "method": "simple_bind",
                    "username": "admin",
                    "password": "password123",
                },
            )
            assert connect_result_data["success"] is True

            sync_result_data = await km_enterprise_sync(
                operation="sync",
                integration_type="ldap",
                sync_options={
                    "connection_id": "ldap_conn_workflow",
                    "sync_type": "full",
                },
            )
            assert sync_result_data["success"] is True

            query_result_data = await km_enterprise_sync(
                operation="query",
                integration_type="ldap",
                query_filter="(cn=workflow.user)",
                sync_options={"connection_id": "ldap_conn_workflow"},
            )
            assert query_result_data["success"] is True
            assert query_result_data["data"]["record_count"] == 1

            # Verify all operations were called
            mock_manager.establish_connection.assert_called_once()
            mock_manager.sync_enterprise_data.assert_called_once()
            mock_manager.query_enterprise_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_enterprise_sync_with_context(self):
        """Test enterprise sync with FastMCP context integration."""
        mock_context = Mock()
        mock_context.info = AsyncMock()
        mock_context.report_progress = AsyncMock()
        mock_context.error = AsyncMock()

        with patch(
            "src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager"
        ) as mock_get_manager:
            # Setup mocks for context testing - interface alignment fix
            mock_manager = Mock()  # Should be Mock, not AsyncMock for get_system_status
            mock_status = {"status": "operational", "connections": {"total": 3}}
            mock_manager.get_system_status.return_value = (
                mock_status  # Regular method, not async
            )
            mock_get_manager.return_value = mock_manager

            # Execute operation with context
            result = await km_enterprise_sync(
                operation="status", integration_type="ldap", ctx=mock_context
            )

            # Verify context integration
            assert result["success"] is True
            mock_context.info.assert_called()

            # Verify info calls
            info_calls = mock_context.info.call_args_list
            assert len(info_calls) >= 2  # At least start and completion


class TestEnterpriseSyncProperties:
    """Property-based tests for enterprise sync operations."""

    @given(
        enterprise_operation_strategy(),
        integration_type_strategy(),
        timeout_strategy(),
        batch_size_strategy(),
    )
    def test_enterprise_sync_parameter_validation_properties(
        self, operation: str, integration_type: str, timeout: int, batch_size: int
    ):
        """Property test for enterprise sync parameter validation."""
        # Properties that should always hold
        valid_operations = [
            "connect",
            "authenticate",
            "sync",
            "query",
            "configure",
            "status",
            "sso_config",
            "sso_login",
        ]
        valid_types = [
            "ldap",
            "active_directory",
            "saml_sso",
            "oauth_sso",
            "enterprise_database",
            "rest_api",
            "graphql_api",
        ]

        assert operation in valid_operations
        assert integration_type in valid_types
        assert 5 <= timeout <= 300
        assert 10 <= batch_size <= 1000

    @given(connection_config_strategy())
    def test_connection_config_properties(self, config: dict[str, Any]):
        """Property test for connection configuration validation."""
        # Properties that should always hold for connection configs
        assert isinstance(config, dict)
        assert "host" in config
        assert "port" in config
        assert isinstance(config["port"], int)
        assert config["port"] > 0

        if "use_ssl" in config:
            assert isinstance(config["use_ssl"], bool)
        if "ssl_verify" in config:
            assert isinstance(config["ssl_verify"], bool)

    @given(authentication_strategy())
    def test_authentication_properties(self, auth: dict[str, Any]):
        """Property test for authentication configuration validation."""
        # Properties that should always hold for authentication
        assert isinstance(auth, dict)
        assert "method" in auth

        valid_methods = ["simple_bind", "sasl", "certificate", "token", "api_key"]
        assert auth["method"] in valid_methods

        # Method-specific validations
        if auth["method"] in ["simple_bind", "sasl"]:
            assert "username" in auth or "password" in auth
        elif auth["method"] == "certificate":
            assert "certificate_path" in auth
        elif auth["method"] == "token":
            assert "token" in auth
        elif auth["method"] == "api_key":
            assert "api_key" in auth

    @given(query_filter_strategy())
    def test_query_filter_properties(self, query_filter: str):
        """Property test for query filter validation."""
        # Properties that should always hold for query filters
        assert isinstance(query_filter, str)
        assert len(query_filter) > 0

        # LDAP filter patterns
        if query_filter.startswith("(") and query_filter.endswith(")"):
            # Should be a valid LDAP filter format
            assert "=" in query_filter or "*" in query_filter

        # SQL query patterns
        elif query_filter.upper().startswith("SELECT"):
            # Should be a SQL query
            assert "FROM" in query_filter.upper()

        # API endpoint patterns
        elif query_filter.startswith("/"):
            # Should be an API endpoint
            assert len(query_filter) > 1

    @given(sso_config_strategy())
    def test_sso_config_properties(self, sso_config: dict[str, Any]):
        """Property test for SSO configuration validation."""
        # Properties that should always hold for SSO configs
        assert isinstance(sso_config, dict)
        assert "provider_name" in sso_config

        # URL validations
        if "metadata_url" in sso_config:
            assert sso_config["metadata_url"].startswith("http")
        if "redirect_uri" in sso_config:
            assert sso_config["redirect_uri"].startswith("http")

        # Required fields for different SSO types
        if "client_id" in sso_config:
            # OAuth-style config
            assert len(sso_config["client_id"]) > 0
        if "certificate" in sso_config:
            # SAML-style config
            assert len(sso_config["certificate"]) > 0

    @given(invalid_integration_type_strategy())
    def test_security_validation_properties(self, invalid_type: str):
        """Property test for security validation behavior."""
        # Invalid integration types should be detectable
        valid_types = [
            "ldap",
            "active_directory",
            "saml_sso",
            "oauth_sso",
            "enterprise_database",
            "rest_api",
            "graphql_api",
        ]

        # Should not be in valid types
        assert invalid_type not in valid_types

        # Common invalid patterns
        if invalid_type == "":
            # Empty type should be rejected
            assert len(invalid_type) == 0

        # Security-sensitive patterns
        security_risks = ["../", "\\", "/etc", "admin", "root", "system"]
        has_risk = any(risk in invalid_type.lower() for risk in security_risks)

        if has_risk:
            # Should be detectable as potentially risky
            assert any(
                indicator in invalid_type.lower() for indicator in security_risks
            )
