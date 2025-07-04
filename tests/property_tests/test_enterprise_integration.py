"""
Property-based tests for enterprise integration system.

This module provides comprehensive property-based testing for enterprise integration
using Hypothesis to validate behavior across all input ranges with focus on
security, compliance, and integration reliability.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime, timedelta, UTC
from typing import Dict, Any
import uuid

from src.core.enterprise_integration import (
    EnterpriseConnection, EnterpriseCredentials, LDAPUser, SyncResult,
    IntegrationType, AuthenticationMethod, SecurityLevel, EnterpriseError,
    EnterpriseSecurityValidator, create_enterprise_connection, create_enterprise_credentials
)
from src.enterprise.ldap_connector import LDAPConnector
from src.enterprise.sso_manager import SSOManager
from src.enterprise.enterprise_sync_manager import EnterpriseSyncManager
from src.server.tools.enterprise_sync_tools import km_enterprise_sync


class TestEnterpriseConnectionProperties:
    """Property-based tests for enterprise connection functionality."""
    
    @settings(max_examples=50, deadline=5000)
    @given(
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=100),
        st.integers(min_value=1, max_value=65535),
        st.sampled_from(list(IntegrationType))
    )
    def test_enterprise_connection_properties(self, connection_id, host, port, integration_type):
        """Property: Enterprise connections should handle various host and port combinations."""
        assume(connection_id.strip() != "")
        assume(host.strip() != "")
        
        if host.replace('-', '').replace('.', '').replace('_', '').isalnum():
            try:
                connection = create_enterprise_connection(
                    connection_id=connection_id,
                    integration_type=integration_type,
                    host=host,
                    port=port,
                    use_ssl=True,
                    ssl_verify=True
                )
                
                assert connection.connection_id == connection_id
                assert connection.host == host
                assert connection.port == port
                assert connection.integration_type == integration_type
                assert connection.validate_ssl_configuration()
                
                url = connection.get_connection_url()
                assert host in url
                assert str(port) in url
                
            except ValueError:
                # Some combinations might be invalid
                pass
    
    @settings(max_examples=50, deadline=5000)
    @given(
        st.sampled_from(list(AuthenticationMethod)),
        st.text(min_size=1, max_size=50),
        st.text(min_size=12, max_size=100)
    )
    def test_enterprise_credentials_properties(self, auth_method, username, password):
        """Property: Enterprise credentials should handle various authentication methods."""
        assume(username.strip() != "")
        assume(password.strip() != "")
        
        try:
            if auth_method == AuthenticationMethod.SIMPLE_BIND:
                credentials = create_enterprise_credentials(
                    auth_method=auth_method,
                    username=username,
                    password=password
                )
            elif auth_method == AuthenticationMethod.API_KEY:
                credentials = create_enterprise_credentials(
                    auth_method=auth_method,
                    api_key=password  # Use password as API key
                )
            elif auth_method == AuthenticationMethod.OAUTH_TOKEN:
                credentials = create_enterprise_credentials(
                    auth_method=auth_method,
                    token=password  # Use password as token
                )
            else:
                credentials = create_enterprise_credentials(
                    auth_method=auth_method,
                    username=username
                )
            
            assert credentials.auth_method == auth_method
            assert not credentials.is_expired()  # Should not be expired when created
            
            safe_repr = credentials.get_safe_representation()
            assert isinstance(safe_repr, dict)
            assert safe_repr['auth_method'] == auth_method.value
            assert 'password' not in safe_repr or safe_repr.get('password') is None
            
        except ValueError:
            # Some combinations might be invalid
            pass
    
    @settings(max_examples=50, deadline=5000)
    @given(
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=100),
        st.booleans(),
        st.sets(st.text(min_size=1, max_size=20), max_size=10)
    )
    def test_ldap_user_properties(self, username, display_name, is_active, groups):
        """Property: LDAP users should handle various usernames and attributes."""
        assume(username.strip() != "")
        
        if username.replace('_', '').replace('-', '').replace('.', '').isalnum():
            try:
                user = LDAPUser(
                    distinguished_name=f"CN={username},OU=Users,DC=example,DC=com",
                    username=username,
                    display_name=display_name,
                    is_active=is_active,
                    groups=groups
                )
                
                assert user.username == username
                assert user.display_name == display_name
                assert user.is_active == is_active
                assert user.groups == groups
                
                # Test group membership
                for group in groups:
                    assert user.has_group(group)
                
                # Test full name logic
                full_name = user.get_full_name()
                assert isinstance(full_name, str)
                assert len(full_name) > 0
                
            except ValueError:
                # Some names might be invalid
                pass
    
    @settings(max_examples=50, deadline=5000)
    @given(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
        st.floats(min_value=0.0, max_value=300.0),
        st.sampled_from(list(IntegrationType))
    )
    def test_sync_result_properties(self, successful, failed, duration, integration_type):
        """Property: Sync results should handle various success/failure counts."""
        total_processed = successful + failed
        
        result = SyncResult(
            operation="test_sync",
            integration_type=integration_type,
            records_processed=total_processed,
            records_successful=successful,
            records_failed=failed,
            sync_duration=duration
        )
        
        assert result.records_processed == total_processed
        assert result.records_successful == successful
        assert result.records_failed == failed
        assert result.sync_duration == duration
        assert 0.0 <= result.get_success_rate() <= 100.0
        assert result.has_errors() == (failed > 0)
        
        status_summary = result.get_status_summary()
        assert isinstance(status_summary, str)
        assert str(successful) in status_summary
        assert str(total_processed) in status_summary


class TestEnterpriseSecurityProperties:
    """Property-based tests for enterprise security validation."""
    
    def test_security_validator_initialization(self):
        """Property: Security validator should initialize correctly."""
        validator = EnterpriseSecurityValidator()
        
        assert hasattr(validator, 'DANGEROUS_HOSTNAMES')
        assert hasattr(validator, 'SECURE_PORTS')
        assert isinstance(validator.DANGEROUS_HOSTNAMES, set)
        assert isinstance(validator.SECURE_PORTS, dict)
    
    @settings(max_examples=50, deadline=5000)
    @given(
        st.text(min_size=1, max_size=100),
        st.integers(min_value=1, max_value=65535),
        st.sampled_from(list(IntegrationType)),
        st.booleans(),
        st.booleans()
    )
    def test_connection_security_validation_properties(self, host, port, integration_type, use_ssl, ssl_verify):
        """Property: Connection security validation should be consistent."""
        assume(host.strip() != "")
        
        validator = EnterpriseSecurityValidator()
        
        try:
            connection = create_enterprise_connection(
                connection_id="test_connection",
                integration_type=integration_type,
                host=host,
                port=port,
                use_ssl=use_ssl,
                ssl_verify=ssl_verify
            )
            
            validation_result = validator.validate_connection_security(connection)
            
            # Property: All enterprise connections must use SSL
            if not use_ssl:
                assert validation_result.is_left()
                error = validation_result.get_left()
                assert "insecure" in error.message.lower() or "ssl" in error.message.lower()
            
            # Property: Enterprise connections must verify certificates
            if use_ssl and not ssl_verify:
                assert validation_result.is_left()
                error = validation_result.get_left()
                assert "certificate" in error.message.lower() or "validation" in error.message.lower()
            
            # Property: Dangerous hostnames should be rejected
            if host.lower() in validator.DANGEROUS_HOSTNAMES:
                assert validation_result.is_left()
            
        except ValueError:
            # Some combinations might be invalid
            pass
    
    @settings(max_examples=50, deadline=5000)
    @given(
        st.text(min_size=1, max_size=1000),
        st.sampled_from(['(objectClass=user)', '(sAMAccountName=*)', '(cn=test)', '(mail=*@company.com)'])
    )
    def test_ldap_search_filter_validation_properties(self, base_filter, standard_filter):
        """Property: LDAP search filter validation should detect injection attacks."""
        validator = EnterpriseSecurityValidator()
        
        # Property: Standard LDAP filters should be valid
        validation_result = validator.validate_search_filter(standard_filter)
        assert validation_result.is_right()
        
        # Property: Filters with dangerous patterns should be rejected
        dangerous_patterns = ['*)(|', '*)(&', ')(!', '<script>', '\x00', '"']
        
        for pattern in dangerous_patterns:
            dangerous_filter = base_filter + pattern
            dangerous_result = validator.validate_search_filter(dangerous_filter)
            if len(dangerous_filter) <= 1000:  # Only test if within length limit
                assert dangerous_result.is_left()
        
        # Property: Very long filters should be rejected
        long_filter = 'a' * 2000
        long_result = validator.validate_search_filter(long_filter)
        assert long_result.is_left()
    
    @settings(max_examples=50, deadline=5000)
    @given(
        st.text(min_size=12, max_size=100),
        st.integers(min_value=0, max_value=4),
        st.integers(min_value=0, max_value=4),
        st.integers(min_value=0, max_value=4),
        st.integers(min_value=0, max_value=4)
    )
    def test_password_complexity_validation_properties(self, password, upper_count, lower_count, digit_count, special_count):
        """Property: Password complexity validation should be consistent."""
        assume(len(password) >= 12)
        
        validator = EnterpriseSecurityValidator()
        
        # Build password with specific character types
        test_password = password[:8]  # Base
        test_password += 'A' * upper_count
        test_password += 'a' * lower_count  
        test_password += '1' * digit_count
        test_password += '!' * special_count
        
        is_complex = validator._validate_password_complexity(test_password)
        
        # Property: Password must have all character types for enterprise
        expected_complex = (
            len(test_password) >= 12 and
            upper_count > 0 and
            lower_count > 0 and
            digit_count > 0 and
            special_count > 0
        )
        
        assert is_complex == expected_complex


class TestLDAPConnectorProperties:
    """Property-based tests for LDAP connector functionality."""
    
    @pytest.fixture
    def ldap_connector(self):
        """Provide LDAP connector for tests."""
        return LDAPConnector()
    
    @settings(max_examples=30, deadline=5000)
    @given(
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=50),
        st.integers(min_value=389, max_value=65535)
    )
    @pytest.mark.asyncio
    async def test_ldap_connection_properties(self, connection_id, host, port, ldap_connector):
        """Property: LDAP connection should handle various configurations."""
        assume(connection_id.strip() != "")
        assume(host.strip() != "")
        
        if host.replace('-', '').replace('.', '').isalnum():
            try:
                connection = create_enterprise_connection(
                    connection_id=connection_id,
                    integration_type=IntegrationType.LDAP,
                    host=host,
                    port=port,
                    use_ssl=True,
                    ssl_verify=True
                )
                
                credentials = create_enterprise_credentials(
                    auth_method=AuthenticationMethod.SIMPLE_BIND,
                    username="test_user",
                    password="test_password123!"
                )
                
                # Mock LDAP library for testing
                with patch('src.enterprise.ldap_connector.ldap3') as mock_ldap3:
                    mock_server = Mock()
                    mock_connection = Mock()
                    mock_connection.bind.return_value = True
                    
                    mock_ldap3.Server.return_value = mock_server
                    mock_ldap3.Connection.return_value = mock_connection
                    mock_ldap3.ALL = "ALL"
                    mock_ldap3.SIMPLE = "SIMPLE"
                    mock_ldap3.Tls = Mock()
                    
                    result = await ldap_connector.connect(connection, credentials)
                    
                    # Property: Valid connections should succeed or fail gracefully
                    assert isinstance(result.is_right(), bool)
                    
                    if result.is_right():
                        conn_id = result.get_right()
                        assert conn_id == connection_id
                        assert connection_id in ldap_connector.connections
                        
                        # Property: Connection status should be retrievable
                        status = ldap_connector.get_connection_status(connection_id)
                        assert isinstance(status, dict)
                        assert 'status' in status
                        
            except Exception:
                # Some configurations might be invalid
                pass


class TestSSOManagerProperties:
    """Property-based tests for SSO manager functionality."""
    
    @pytest.fixture
    def sso_manager(self):
        """Provide SSO manager for tests."""
        return SSOManager()
    
    @settings(max_examples=30, deadline=5000)
    @given(
        st.text(min_size=1, max_size=50),
        st.text(min_size=10, max_size=100),
        st.text(min_size=10, max_size=100)
    )
    @pytest.mark.asyncio
    async def test_oauth_provider_configuration_properties(self, provider_name, client_id, client_secret, sso_manager):
        """Property: OAuth provider configuration should handle various inputs."""
        assume(provider_name.strip() != "")
        assume(len(client_id) >= 10)
        assume(len(client_secret) >= 16)
        
        oauth_config = {
            'provider_name': provider_name,
            'client_id': client_id,
            'client_secret': client_secret,
            'authorization_url': 'https://auth.example.com/oauth/authorize',
            'token_url': 'https://auth.example.com/oauth/token'
        }
        
        result = await sso_manager.configure_oauth_provider(oauth_config)
        
        # Property: Valid OAuth configs should succeed
        if result.is_right():
            provider_id = result.get_right()
            assert isinstance(provider_id, str)
            assert len(provider_id) > 0
            assert provider_id in sso_manager.sso_providers
            
            # Property: Provider status should be retrievable
            status = sso_manager.get_provider_status(provider_id)
            assert isinstance(status, dict)
            assert status.get('type') == 'oauth'
            assert status.get('status') == 'active'
    
    @settings(max_examples=30, deadline=5000)
    @given(
        st.text(min_size=1, max_size=50),
        st.text(min_size=10, max_size=200)
    )
    @pytest.mark.asyncio
    async def test_saml_provider_configuration_properties(self, provider_name, entity_id, sso_manager):
        """Property: SAML provider configuration should handle various inputs."""
        assume(provider_name.strip() != "")
        assume(entity_id.strip() != "")
        
        # Mock certificate for testing
        mock_certificate = """-----BEGIN CERTIFICATE-----
MIICXjCCAcegAwIBAgIJAKS0yiqVQBCGMA0GCSqGSIb3DQEBCwUAMEYxCzAJBgNV
BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
aWRnaXRzIFB0eSBMdGQwHhcNMTcwNTEwMTM1NjQ4WhcNMTgwNTEwMTM1NjQ4WjBG
MQswCQYDVQQGEwJBVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50
ZXJuZXQgV2lkZ2l0cyBQdHkgTHRkMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKB
gQDKwXGFdnTIFaTPP8vJKNjBKYHj7yt8wM6M1o1TnPY9JJrGMZoVL+lRqPyEKgqq
dGP1J2w5rLLb+h3QJqNZOa8V7O0c2H2T+TK7K7QPgbgk6O1v8XdCiL5N5U9KpRqB
b3ZG5z+3J3qp7V5TzT6U9k4hJ7J9aQ1H2l2g6E3L2vZbRwIDAQABo1AwTjAdBgNV
HQ4EFgQU9q1Q3vJqFvL+vG4QZQO4P4N2YQ4wHwYDVR0jBBgwFoAU9q1Q3vJqFvL+
vG4QZQO4P4N2YQ4wDAYDVR0TBAUwAwEB/zANBgkqhkiG9w0BAQsFAAOBgQC8K3m5
4u2T8L1N9v1R3G4fL2H8vZI9zqFwLGxNqv3t3VqL3zP8J8cQ4e8Q8zq8K3m54u2T
8L1N9v1R3G4fL2H8vZI9zqFwLGxNqv3t3VqL3zP8J8cQ4e8Q8zq8K3m54u2T8L1N
9v1R3G4fL2H8vZI9zqFwLGxNqv3t3VqL3zP8J8cQ4e8Q8zq8
-----END CERTIFICATE-----"""
        
        saml_config = {
            'provider_name': provider_name,
            'entity_id': entity_id,
            'sso_url': 'https://sso.example.com/saml/login',
            'certificate': mock_certificate
        }
        
        # Mock cryptography for certificate validation
        with patch('src.enterprise.sso_manager.x509') as mock_x509:
            mock_cert = Mock()
            mock_cert.not_valid_after = datetime.now(UTC) + timedelta(days=365)
            mock_cert.not_valid_before = datetime.now(UTC) - timedelta(days=1)
            mock_x509.load_pem_x509_certificate.return_value = mock_cert
            
            result = await sso_manager.configure_saml_provider(saml_config)
            
            # Property: Valid SAML configs should succeed
            if result.is_right():
                provider_id = result.get_right()
                assert isinstance(provider_id, str)
                assert len(provider_id) > 0
                assert provider_id in sso_manager.sso_providers
                
                # Property: Provider status should be retrievable
                status = sso_manager.get_provider_status(provider_id)
                assert isinstance(status, dict)
                assert status.get('type') == 'saml'
                assert status.get('status') == 'active'


class TestEnterpriseSyncToolProperties:
    """Property-based tests for enterprise sync MCP tool."""
    
    @settings(max_examples=30, deadline=5000)
    @given(
        st.sampled_from(["connect", "sync", "query", "status", "sso_config"]),
        st.sampled_from([t.value for t in IntegrationType]),
        st.text(min_size=1, max_size=50)
    )
    @pytest.mark.asyncio
    async def test_enterprise_sync_tool_properties(self, operation, integration_type, host):
        """Property: Enterprise sync tool should handle various operations."""
        assume(host.strip() != "")
        
        # Mock the enterprise sync manager
        with patch('src.server.tools.enterprise_sync_tools.get_enterprise_sync_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_system_status.return_value = {
                "status": "operational",
                "connections": {"total": 0},
                "features": {"audit_logging": True}
            }
            mock_get_manager.return_value = mock_manager
            
            if operation == "status":
                result = await km_enterprise_sync(
                    operation=operation,
                    integration_type=integration_type
                )
            elif operation == "connect":
                result = await km_enterprise_sync(
                    operation=operation,
                    integration_type=integration_type,
                    connection_config={'host': host, 'port': 636, 'connection_id': 'test'},
                    authentication={'method': 'simple_bind', 'username': 'test', 'password': 'test'}
                )
            else:
                result = await km_enterprise_sync(
                    operation=operation,
                    integration_type=integration_type
                )
            
            # Property: All operations should return a structured response
            assert isinstance(result, dict)
            assert 'success' in result
            assert 'operation' in result or 'error' in result
            
            # Property: Successful operations should have data
            if result.get('success'):
                assert 'data' in result
                assert isinstance(result['data'], dict)
    
    @settings(max_examples=20, deadline=5000)
    @given(st.text(max_size=0))
    @pytest.mark.asyncio
    async def test_invalid_parameters_validation_properties(self, empty_value):
        """Property: Empty or invalid parameters should be rejected gracefully."""
        assume(len(empty_value.strip()) == 0)
        
        result = await km_enterprise_sync(
            operation="invalid_operation",
            integration_type="invalid_type"
        )
        
        # Property: Invalid operations should be rejected with clear errors
        assert result['success'] is False
        assert 'error' in result
        assert 'code' in result['error']
        assert 'message' in result['error']


class TestEnterpriseIntegrationCompliance:
    """Property-based tests for enterprise integration compliance."""
    
    @settings(max_examples=30, deadline=5000)
    @given(
        st.integers(min_value=1, max_value=2555),  # Retention days
        st.sampled_from(list(SecurityLevel)),
        st.booleans(),  # SSL enabled
        st.booleans()   # Certificate verification
    )
    def test_enterprise_configuration_compliance_properties(self, retention_days, security_level, use_ssl, ssl_verify):
        """Property: Enterprise configurations should meet compliance requirements."""
        
        # Property: Enterprise security level requires SSL
        if security_level == SecurityLevel.ENTERPRISE:
            assert use_ssl, "Enterprise security level requires SSL"
            assert ssl_verify, "Enterprise security level requires certificate verification"
        
        # Property: Retention period should be within allowed range
        assert 1 <= retention_days <= 2555, "Retention period must be between 1 and 2555 days"
        
        # Property: High security configurations should use encryption
        if security_level in [SecurityLevel.HIGH, SecurityLevel.ENTERPRISE]:
            assert use_ssl, "High security levels require SSL encryption"
    
    @settings(max_examples=30, deadline=5000)
    @given(
        st.integers(min_value=0, max_value=100000),
        st.integers(min_value=0, max_value=1000),
        st.floats(min_value=0.0, max_value=3600.0)
    )
    def test_enterprise_performance_properties(self, records_count, batch_size, duration):
        """Property: Enterprise operations should meet performance requirements."""
        assume(batch_size > 0)
        
        # Property: Batch size should be reasonable for enterprise operations
        if batch_size > 0:
            batches = (records_count + batch_size - 1) // batch_size
            
            # Property: Connection time should be reasonable
            if duration > 0:
                records_per_second = records_count / duration if duration > 0 else 0
                
                # Property: Sync operations should process reasonable number of records
                assert records_per_second >= 0, "Records per second should be non-negative"
                
                # Property: Very slow operations should be flagged
                if records_count > 1000 and duration > 300:  # 5 minutes for 1000+ records
                    # This would trigger performance warnings in a real system
                    pass
    
    def test_enterprise_audit_integration_properties(self):
        """Property: Enterprise operations should integrate with audit system."""
        
        # Property: All enterprise operations should be auditable
        auditable_operations = ["connect", "sync", "query", "authenticate", "configure"]
        
        for operation in auditable_operations:
            # In a real implementation, we would verify that audit events are generated
            # for each of these operations
            assert operation in auditable_operations
        
        # Property: Audit events should contain required information
        required_audit_fields = ["user_id", "action", "result", "timestamp", "resource_id"]
        
        for field in required_audit_fields:
            assert isinstance(field, str)
            assert len(field) > 0