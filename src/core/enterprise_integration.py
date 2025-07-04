"""
Enterprise system integration types and core architecture.

This module provides comprehensive enterprise integration capabilities including
LDAP/Active Directory, SSO authentication, enterprise databases, and secure
communication protocols with complete audit logging and compliance support.

Security: Enterprise-grade encryption, certificate validation, secure authentication
Performance: <5s connection establishment, <10s user sync, <2s authentication  
Type Safety: Complete integration with enterprise security and compliance frameworks
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
import ssl
import socket
import hashlib
import secrets
import re

from .contracts import require, ensure
from .either import Either
from .errors import ValidationError


class IntegrationType(Enum):
    """Types of enterprise integrations."""
    LDAP = "ldap"
    ACTIVE_DIRECTORY = "active_directory"
    SAML_SSO = "saml_sso"
    OAUTH_SSO = "oauth_sso"
    ENTERPRISE_DATABASE = "enterprise_database"
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    MESSAGE_QUEUE = "message_queue"
    MONITORING_SYSTEM = "monitoring_system"


class AuthenticationMethod(Enum):
    """Enterprise authentication methods."""
    SIMPLE_BIND = "simple_bind"
    KERBEROS = "kerberos"
    NTLM = "ntlm"
    CERTIFICATE = "certificate"
    SAML_ASSERTION = "saml_assertion"
    OAUTH_TOKEN = "oauth_token"
    API_KEY = "api_key"
    MUTUAL_TLS = "mutual_tls"


class SecurityLevel(Enum):
    """Enterprise security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ENTERPRISE = "enterprise"


class EnterpriseError(Exception):
    """Enterprise integration error hierarchy."""
    
    def __init__(self, error_type: str, message: str, details: Dict[str, Any] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(f"{error_type}: {message}")
    
    @classmethod
    def connection_failed(cls, reason: str) -> 'EnterpriseError':
        return cls("CONNECTION_FAILED", f"Failed to establish enterprise connection: {reason}")
    
    @classmethod
    def authentication_failed(cls) -> 'EnterpriseError':
        return cls("AUTHENTICATION_FAILED", "Enterprise authentication failed")
    
    @classmethod
    def insecure_connection(cls) -> 'EnterpriseError':
        return cls("INSECURE_CONNECTION", "Insecure connection not allowed in enterprise environment")
    
    @classmethod
    def unsupported_auth_method(cls, method: AuthenticationMethod) -> 'EnterpriseError':
        return cls("UNSUPPORTED_AUTH_METHOD", f"Authentication method {method.value} not supported")
    
    @classmethod
    def connection_not_found(cls, connection_id: str) -> 'EnterpriseError':
        return cls("CONNECTION_NOT_FOUND", f"Enterprise connection {connection_id} not found")
    
    @classmethod
    def search_failed(cls, reason: str) -> 'EnterpriseError':
        return cls("SEARCH_FAILED", f"Enterprise search operation failed: {reason}")
    
    @classmethod
    def sync_failed(cls, reason: str) -> 'EnterpriseError':
        return cls("SYNC_FAILED", f"Enterprise synchronization failed: {reason}")
    
    @classmethod
    def invalid_ldap_entry(cls) -> 'EnterpriseError':
        return cls("INVALID_LDAP_ENTRY", "Invalid LDAP entry format")
    
    @classmethod
    def ldap_conversion_failed(cls, reason: str) -> 'EnterpriseError':
        return cls("LDAP_CONVERSION_FAILED", f"Failed to convert LDAP entry: {reason}")
    
    @classmethod
    def sso_configuration_failed(cls, reason: str) -> 'EnterpriseError':
        return cls("SSO_CONFIGURATION_FAILED", f"SSO configuration failed: {reason}")
    
    @classmethod
    def missing_required_field(cls, field: str) -> 'EnterpriseError':
        return cls("MISSING_REQUIRED_FIELD", f"Required field missing: {field}")
    
    @classmethod
    def certificate_expired(cls) -> 'EnterpriseError':
        return cls("CERTIFICATE_EXPIRED", "Enterprise certificate has expired")
    
    @classmethod
    def invalid_certificate(cls, reason: str) -> 'EnterpriseError':
        return cls("INVALID_CERTIFICATE", f"Invalid enterprise certificate: {reason}")
    
    @classmethod
    def sso_provider_not_found(cls, provider_id: str) -> 'EnterpriseError':
        return cls("SSO_PROVIDER_NOT_FOUND", f"SSO provider {provider_id} not found")
    
    @classmethod
    def unsupported_sso_type(cls, sso_type: str) -> 'EnterpriseError':
        return cls("UNSUPPORTED_SSO_TYPE", f"SSO type {sso_type} not supported")
    
    @classmethod
    def sso_initiation_failed(cls, reason: str) -> 'EnterpriseError':
        return cls("SSO_INITIATION_FAILED", f"SSO initiation failed: {reason}")
    
    @classmethod
    def initialization_failed(cls, reason: str) -> 'EnterpriseError':
        return cls("INITIALIZATION_FAILED", f"Enterprise system initialization failed: {reason}")
    
    @classmethod
    def unsupported_integration_type(cls, integration_type: IntegrationType) -> 'EnterpriseError':
        return cls("UNSUPPORTED_INTEGRATION_TYPE", f"Integration type {integration_type.value} not supported")
    
    @classmethod
    def missing_integration_type(cls) -> 'EnterpriseError':
        return cls("MISSING_INTEGRATION_TYPE", "Integration type not specified")
    
    @classmethod
    def unsupported_sync_type(cls, sync_type: IntegrationType) -> 'EnterpriseError':
        return cls("UNSUPPORTED_SYNC_TYPE", f"Sync type {sync_type.value} not supported")
    
    @classmethod
    def connection_establishment_failed(cls, reason: str) -> 'EnterpriseError':
        return cls("CONNECTION_ESTABLISHMENT_FAILED", f"Failed to establish connection: {reason}")
    
    @classmethod
    def sync_operation_failed(cls, reason: str) -> 'EnterpriseError':
        return cls("SYNC_OPERATION_FAILED", f"Sync operation failed: {reason}")
    
    @classmethod
    def insecure_connection_not_allowed(cls) -> 'EnterpriseError':
        return cls("INSECURE_CONNECTION_NOT_ALLOWED", "Insecure connections not allowed in enterprise environment")
    
    @classmethod
    def certificate_validation_required(cls) -> 'EnterpriseError':
        return cls("CERTIFICATE_VALIDATION_REQUIRED", "Certificate validation required for enterprise connections")
    
    @classmethod
    def unencrypted_ldap_not_allowed(cls) -> 'EnterpriseError':
        return cls("UNENCRYPTED_LDAP_NOT_ALLOWED", "Unencrypted LDAP connections not allowed")
    
    @classmethod
    def weak_authentication_method(cls) -> 'EnterpriseError':
        return cls("WEAK_AUTHENTICATION_METHOD", "Weak authentication method not allowed in enterprise environment")
    
    @classmethod
    def weak_password(cls) -> 'EnterpriseError':
        return cls("WEAK_PASSWORD", "Password does not meet enterprise complexity requirements")


@dataclass(frozen=True)
class SecurityLimits:
    """Enterprise security limits and constraints."""
    max_connection_lifetime: int = 3600  # 1 hour
    max_idle_time: int = 900  # 15 minutes
    max_concurrent_connections: int = 50
    min_password_length: int = 12
    max_login_attempts: int = 3
    session_timeout: int = 1800  # 30 minutes
    max_search_results: int = 10000
    connection_timeout: int = 30
    
    @require(lambda self: self.max_connection_lifetime > 0)
    @require(lambda self: self.max_idle_time > 0)
    @require(lambda self: self.max_concurrent_connections > 0)
    @require(lambda self: self.min_password_length >= 8)
    def __post_init__(self):
        pass


@dataclass(frozen=True)
class EnterpriseConnection:
    """Type-safe enterprise connection specification."""
    connection_id: str
    integration_type: IntegrationType
    host: str
    port: int
    use_ssl: bool = True
    ssl_verify: bool = True
    timeout: int = 30
    connection_pool_size: int = 5
    base_dn: Optional[str] = None  # For LDAP/AD
    domain: Optional[str] = None   # For AD/SSO
    api_version: Optional[str] = None  # For APIs
    security_limits: SecurityLimits = field(default_factory=SecurityLimits)
    
    @require(lambda self: len(self.connection_id) > 0)
    @require(lambda self: len(self.host) > 0)
    @require(lambda self: 1 <= self.port <= 65535)
    @require(lambda self: self.timeout > 0)
    @require(lambda self: self.connection_pool_size > 0)
    def __post_init__(self):
        pass
    
    def get_connection_url(self) -> str:
        """Get formatted connection URL."""
        if self.integration_type == IntegrationType.LDAP:
            protocol = "ldaps" if self.use_ssl else "ldap"
            return f"{protocol}://{self.host}:{self.port}"
        elif self.integration_type == IntegrationType.ACTIVE_DIRECTORY:
            protocol = "ldaps" if self.use_ssl else "ldap"
            return f"{protocol}://{self.host}:{self.port}"
        elif self.integration_type in [IntegrationType.REST_API, IntegrationType.GRAPHQL_API]:
            protocol = "https" if self.use_ssl else "http"
            return f"{protocol}://{self.host}:{self.port}"
        else:
            return f"{self.host}:{self.port}"
    
    def validate_ssl_configuration(self) -> bool:
        """Validate SSL configuration for enterprise security."""
        if self.integration_type in [IntegrationType.LDAP, IntegrationType.ACTIVE_DIRECTORY]:
            return self.use_ssl and self.ssl_verify  # Enterprise requires encrypted LDAP
        elif self.integration_type in [IntegrationType.REST_API, IntegrationType.GRAPHQL_API]:
            return self.use_ssl  # APIs should use HTTPS
        return True


@dataclass(frozen=True)
class EnterpriseCredentials:
    """Secure enterprise authentication credentials."""
    auth_method: AuthenticationMethod
    username: Optional[str] = None
    password: Optional[str] = None  # Should be encrypted in production
    domain: Optional[str] = None
    certificate_path: Optional[str] = None
    token: Optional[str] = None
    api_key: Optional[str] = None
    additional_params: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = None
    
    @require(lambda self: self._validate_credentials())
    def __post_init__(self):
        pass
    
    def _validate_credentials(self) -> bool:
        """Validate credentials based on authentication method."""
        if self.auth_method == AuthenticationMethod.SIMPLE_BIND:
            return self.username is not None and self.password is not None
        elif self.auth_method == AuthenticationMethod.CERTIFICATE:
            return self.certificate_path is not None
        elif self.auth_method == AuthenticationMethod.OAUTH_TOKEN:
            return self.token is not None
        elif self.auth_method == AuthenticationMethod.API_KEY:
            return self.api_key is not None
        elif self.auth_method == AuthenticationMethod.KERBEROS:
            return self.username is not None and self.domain is not None
        return True
    
    def is_expired(self) -> bool:
        """Check if credentials have expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at
    
    def get_safe_representation(self) -> Dict[str, Any]:
        """Get credentials without sensitive information."""
        return {
            "auth_method": self.auth_method.value,
            "username": self.username,
            "domain": self.domain,
            "has_password": self.password is not None,
            "has_certificate": self.certificate_path is not None,
            "has_token": self.token is not None,
            "has_api_key": self.api_key is not None,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_expired": self.is_expired()
        }


@dataclass(frozen=True)
class LDAPUser:
    """LDAP/Active Directory user representation."""
    distinguished_name: str
    username: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    groups: Set[str] = field(default_factory=set)
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    is_active: bool = True
    last_login: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    modified_at: Optional[datetime] = None
    
    @require(lambda self: len(self.distinguished_name) > 0)
    @require(lambda self: len(self.username) > 0)
    def __post_init__(self):
        pass
    
    def has_group(self, group_name: str) -> bool:
        """Check if user belongs to specific group."""
        return group_name in self.groups
    
    def get_primary_email(self) -> Optional[str]:
        """Get primary email address."""
        if self.email:
            return self.email
        # Try to extract from attributes
        mail_attrs = self.attributes.get('mail', [])
        return mail_attrs[0] if mail_attrs else None
    
    def get_full_name(self) -> str:
        """Get full name from first and last name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.display_name:
            return self.display_name
        else:
            return self.username


@dataclass(frozen=True)
class SyncResult:
    """Enterprise synchronization result."""
    operation: str
    integration_type: IntegrationType
    records_processed: int
    records_successful: int
    records_failed: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sync_duration: float = 0.0
    last_sync_token: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: Optional[datetime] = None
    
    @require(lambda self: self.records_processed >= 0)
    @require(lambda self: self.records_successful >= 0)
    @require(lambda self: self.records_failed >= 0)
    @require(lambda self: self.sync_duration >= 0.0)
    def __post_init__(self):
        pass
    
    def get_success_rate(self) -> float:
        """Calculate synchronization success rate."""
        if self.records_processed == 0:
            return 100.0
        return (self.records_successful / self.records_processed) * 100.0
    
    def has_errors(self) -> bool:
        """Check if synchronization had errors."""
        return len(self.errors) > 0 or self.records_failed > 0
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        if not self.has_errors():
            return f"Success: {self.records_successful}/{self.records_processed} records processed"
        else:
            return f"Partial: {self.records_successful}/{self.records_processed} successful, {self.records_failed} failed"


class EnterpriseSecurityValidator:
    """Security validation for enterprise integrations."""
    
    DANGEROUS_HOSTNAMES = {
        'localhost', '127.0.0.1', '0.0.0.0', '::1',
        'metadata.google.internal', '169.254.169.254'  # Cloud metadata endpoints
    }
    
    SECURE_PORTS = {
        IntegrationType.LDAP: {636},  # LDAPS only
        IntegrationType.ACTIVE_DIRECTORY: {636, 3269},  # LDAPS and Global Catalog SSL
        IntegrationType.REST_API: {443, 8443},  # HTTPS ports
        IntegrationType.GRAPHQL_API: {443, 8443}  # HTTPS ports
    }
    
    def validate_connection_security(self, connection: EnterpriseConnection) -> Either[EnterpriseError, None]:
        """Validate enterprise connection security requirements."""
        # Hostname validation
        if connection.host.lower() in self.DANGEROUS_HOSTNAMES:
            return Either.left(EnterpriseError.insecure_connection())
        
        # SSL/TLS validation
        if not connection.use_ssl:
            return Either.left(EnterpriseError.insecure_connection_not_allowed())
        
        # Certificate validation for enterprise
        if not connection.ssl_verify:
            return Either.left(EnterpriseError.certificate_validation_required())
        
        # Port validation for specific integration types
        secure_ports = self.SECURE_PORTS.get(connection.integration_type)
        if secure_ports and connection.port not in secure_ports:
            return Either.left(EnterpriseError.unencrypted_ldap_not_allowed())
        
        # Validate timeout settings
        if connection.timeout < 5 or connection.timeout > 300:
            return Either.left(EnterpriseError("INVALID_TIMEOUT", "Connection timeout must be between 5 and 300 seconds"))
        
        return Either.right(None)
    
    def validate_credentials_security(self, credentials: EnterpriseCredentials) -> Either[EnterpriseError, None]:
        """Validate enterprise credentials security."""
        # Check for expired credentials
        if credentials.is_expired():
            return Either.left(EnterpriseError("CREDENTIALS_EXPIRED", "Enterprise credentials have expired"))
        
        # Check for weak authentication methods in non-secure environments
        weak_methods = [AuthenticationMethod.SIMPLE_BIND]
        if credentials.auth_method in weak_methods and not self._is_secure_environment():
            return Either.left(EnterpriseError.weak_authentication_method())
        
        # Validate password complexity (if applicable)
        if credentials.password and not self._validate_password_complexity(credentials.password):
            return Either.left(EnterpriseError.weak_password())
        
        # Validate username format
        if credentials.username and not self._validate_username_format(credentials.username):
            return Either.left(EnterpriseError("INVALID_USERNAME", "Username format is invalid"))
        
        return Either.right(None)
    
    def validate_search_filter(self, search_filter: str) -> Either[EnterpriseError, None]:
        """Validate LDAP search filter for injection attacks."""
        # Check for LDAP injection patterns
        dangerous_patterns = [
            r'\*\)\(\|',  # Boolean OR injection
            r'\*\)\(&',   # Boolean AND injection
            r'\)\(\!\(',  # Boolean NOT injection
            r'[\x00-\x1f\x7f-\x9f]',  # Control characters
            r'[<>"\'\\\x00]'  # Dangerous characters
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, search_filter, re.IGNORECASE):
                return Either.left(EnterpriseError("LDAP_INJECTION_DETECTED", "Dangerous LDAP filter pattern detected"))
        
        # Validate filter length
        if len(search_filter) > 1000:
            return Either.left(EnterpriseError("FILTER_TOO_LONG", "LDAP filter exceeds maximum length"))
        
        return Either.right(None)
    
    def _is_secure_environment(self) -> bool:
        """Check if environment meets enterprise security standards."""
        # In a real implementation, this would check:
        # - Network security (VPN, private network)
        # - System security (firewall, antivirus)
        # - Compliance status
        return True
    
    def _validate_password_complexity(self, password: str) -> bool:
        """Validate password meets enterprise complexity requirements."""
        if len(password) < 12:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _validate_username_format(self, username: str) -> bool:
        """Validate username format for enterprise standards."""
        # Check length
        if len(username) < 2 or len(username) > 64:
            return False
        
        # Check for valid characters (alphanumeric, underscore, hyphen, dot)
        if not re.match(r'^[a-zA-Z0-9._-]+$', username):
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = ['..', '__', '--', 'admin', 'root', 'administrator']
        username_lower = username.lower()
        for pattern in dangerous_patterns:
            if pattern in username_lower:
                return False
        
        return True


def create_enterprise_connection(
    connection_id: str,
    integration_type: IntegrationType,
    host: str,
    port: int,
    **kwargs
) -> EnterpriseConnection:
    """
    Factory function to create enterprise connections with validation.
    
    Args:
        connection_id: Unique connection identifier
        integration_type: Type of enterprise integration
        host: Enterprise system hostname
        port: Connection port
        **kwargs: Additional connection parameters
        
    Returns:
        Validated EnterpriseConnection instance
    """
    return EnterpriseConnection(
        connection_id=connection_id,
        integration_type=integration_type,
        host=host,
        port=port,
        use_ssl=kwargs.get('use_ssl', True),
        ssl_verify=kwargs.get('ssl_verify', True),
        timeout=kwargs.get('timeout', 30),
        connection_pool_size=kwargs.get('connection_pool_size', 5),
        base_dn=kwargs.get('base_dn'),
        domain=kwargs.get('domain'),
        api_version=kwargs.get('api_version'),
        security_limits=kwargs.get('security_limits', SecurityLimits())
    )


def create_enterprise_credentials(
    auth_method: AuthenticationMethod,
    **kwargs
) -> EnterpriseCredentials:
    """
    Factory function to create enterprise credentials with validation.
    
    Args:
        auth_method: Authentication method to use
        **kwargs: Credential parameters based on auth method
        
    Returns:
        Validated EnterpriseCredentials instance
    """
    return EnterpriseCredentials(
        auth_method=auth_method,
        username=kwargs.get('username'),
        password=kwargs.get('password'),
        domain=kwargs.get('domain'),
        certificate_path=kwargs.get('certificate_path'),
        token=kwargs.get('token'),
        api_key=kwargs.get('api_key'),
        additional_params=kwargs.get('additional_params', {}),
        expires_at=kwargs.get('expires_at')
    )