"""
Authentication framework for HTTP requests.

This module implements comprehensive authentication support for web requests,
including API key, Bearer token, Basic auth, OAuth2, and custom header methods.
Provides secure credential handling and authentication header generation.

Security: Credential validation, secure storage patterns, token sanitization
Type Safety: Branded authentication types with contract validation
Performance: Efficient credential processing and header generation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from enum import Enum
import base64
import re

from src.core.either import Either
from src.core.errors import ValidationError, SecurityError
from src.core.contracts import require, ensure


class AuthenticationType(Enum):
    """Supported authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom_header"


@dataclass(frozen=True)
class AuthenticationCredentials:
    """Type-safe authentication credentials container."""
    auth_type: AuthenticationType
    credentials: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate authentication credentials."""
        if self.auth_type == AuthenticationType.NONE:
            return
        
        if not self.credentials:
            raise ValueError(f"Credentials required for {self.auth_type.value}")
        
        # Validate based on auth type
        if self.auth_type == AuthenticationType.API_KEY:
            if 'api_key' not in self.credentials:
                raise ValueError("API key authentication requires 'api_key' field")
        
        elif self.auth_type == AuthenticationType.BEARER_TOKEN:
            if 'token' not in self.credentials:
                raise ValueError("Bearer token authentication requires 'token' field")
        
        elif self.auth_type == AuthenticationType.BASIC_AUTH:
            if 'username' not in self.credentials or 'password' not in self.credentials:
                raise ValueError("Basic auth requires 'username' and 'password' fields")
        
        elif self.auth_type == AuthenticationType.OAUTH2:
            if 'access_token' not in self.credentials:
                raise ValueError("OAuth2 authentication requires 'access_token' field")
        
        elif self.auth_type == AuthenticationType.CUSTOM_HEADER:
            if 'header_name' not in self.credentials or 'header_value' not in self.credentials:
                raise ValueError("Custom header auth requires 'header_name' and 'header_value' fields")


class AuthenticationManager:
    """Secure authentication handling for HTTP requests."""
    
    @staticmethod
    def create_authentication(
        auth_type: str,
        credentials: Dict[str, str]
    ) -> Either[ValidationError, AuthenticationCredentials]:
        """Create authentication credentials with validation."""
        try:
            # Parse authentication type
            try:
                auth_type_enum = AuthenticationType(auth_type.lower())
            except ValueError:
                valid_types = [t.value for t in AuthenticationType]
                return Either.left(ValidationError(
                    field_name="auth_type",
                    value=auth_type,
                    constraint=f"Must be one of: {', '.join(valid_types)}"
                ))
            
            # Validate and sanitize credentials
            sanitized_credentials = AuthenticationManager._sanitize_credentials(
                auth_type_enum, credentials
            )
            
            if sanitized_credentials.is_left():
                return Either.left(sanitized_credentials.get_left())
            
            # Create authentication object
            auth = AuthenticationCredentials(
                auth_type=auth_type_enum,
                credentials=sanitized_credentials.get_right()
            )
            
            return Either.right(auth)
            
        except Exception as e:
            return Either.left(ValidationError(
                field_name="authentication",
                value=str(credentials),
                constraint=f"Failed to create authentication: {str(e)}"
            ))
    
    @staticmethod
    def _sanitize_credentials(
        auth_type: AuthenticationType,
        credentials: Dict[str, str]
    ) -> Either[ValidationError, Dict[str, str]]:
        """Sanitize and validate credentials based on auth type."""
        sanitized = {}
        
        try:
            if auth_type == AuthenticationType.API_KEY:
                # Validate API key
                api_key = credentials.get('api_key', '').strip()
                if not api_key:
                    return Either.left(ValidationError(
                        field_name="api_key",
                        value="",
                        constraint="API key cannot be empty"
                    ))
                
                if len(api_key) > 512:
                    return Either.left(ValidationError(
                        field_name="api_key",
                        value=api_key[:50] + "...",
                        constraint="API key too long (max 512 characters)"
                    ))
                
                sanitized['api_key'] = api_key
                
                # Optional header name
                header_name = credentials.get('header_name', 'X-API-Key').strip()
                if not re.match(r'^[a-zA-Z0-9\-_]+$', header_name):
                    return Either.left(ValidationError(
                        field_name="header_name",
                        value=header_name,
                        constraint="Header name must contain only alphanumeric, dash, and underscore"
                    ))
                
                sanitized['header_name'] = header_name
            
            elif auth_type == AuthenticationType.BEARER_TOKEN:
                # Validate Bearer token
                token = credentials.get('token', '').strip()
                if not token:
                    return Either.left(ValidationError(
                        field_name="token",
                        value="",
                        constraint="Bearer token cannot be empty"
                    ))
                
                if len(token) > 2048:
                    return Either.left(ValidationError(
                        field_name="token",
                        value=token[:50] + "...",
                        constraint="Bearer token too long (max 2048 characters)"
                    ))
                
                sanitized['token'] = token
            
            elif auth_type == AuthenticationType.BASIC_AUTH:
                # Validate username and password
                username = credentials.get('username', '').strip()
                password = credentials.get('password', '')
                
                if not username:
                    return Either.left(ValidationError(
                        field_name="username",
                        value="",
                        constraint="Username cannot be empty"
                    ))
                
                if len(username) > 255:
                    return Either.left(ValidationError(
                        field_name="username",
                        value=username[:50] + "...",
                        constraint="Username too long (max 255 characters)"
                    ))
                
                if len(password) > 255:
                    return Either.left(ValidationError(
                        field_name="password",
                        value="[REDACTED]",
                        constraint="Password too long (max 255 characters)"
                    ))
                
                # Check for dangerous characters
                if '\n' in username or '\r' in username:
                    return Either.left(ValidationError(
                        field_name="username",
                        value=username,
                        constraint="Username cannot contain newline characters"
                    ))
                
                sanitized['username'] = username
                sanitized['password'] = password
            
            elif auth_type == AuthenticationType.OAUTH2:
                # Validate OAuth2 access token
                access_token = credentials.get('access_token', '').strip()
                if not access_token:
                    return Either.left(ValidationError(
                        field_name="access_token",
                        value="",
                        constraint="OAuth2 access token cannot be empty"
                    ))
                
                if len(access_token) > 2048:
                    return Either.left(ValidationError(
                        field_name="access_token",
                        value=access_token[:50] + "...",
                        constraint="Access token too long (max 2048 characters)"
                    ))
                
                sanitized['access_token'] = access_token
                
                # Optional token type
                token_type = credentials.get('token_type', 'Bearer').strip()
                if not re.match(r'^[a-zA-Z0-9\-_]+$', token_type):
                    return Either.left(ValidationError(
                        field_name="token_type",
                        value=token_type,
                        constraint="Token type must contain only alphanumeric, dash, and underscore"
                    ))
                
                sanitized['token_type'] = token_type
            
            elif auth_type == AuthenticationType.CUSTOM_HEADER:
                # Validate custom header
                header_name = credentials.get('header_name', '').strip()
                header_value = credentials.get('header_value', '').strip()
                
                if not header_name:
                    return Either.left(ValidationError(
                        field_name="header_name",
                        value="",
                        constraint="Header name cannot be empty"
                    ))
                
                if not header_value:
                    return Either.left(ValidationError(
                        field_name="header_value",
                        value="",
                        constraint="Header value cannot be empty"
                    ))
                
                # Validate header name format
                if not re.match(r'^[a-zA-Z0-9\-_]+$', header_name):
                    return Either.left(ValidationError(
                        field_name="header_name",
                        value=header_name,
                        constraint="Header name must contain only alphanumeric, dash, and underscore"
                    ))
                
                if len(header_name) > 100:
                    return Either.left(ValidationError(
                        field_name="header_name",
                        value=header_name,
                        constraint="Header name too long (max 100 characters)"
                    ))
                
                if len(header_value) > 8192:
                    return Either.left(ValidationError(
                        field_name="header_value",
                        value=header_value[:50] + "...",
                        constraint="Header value too long (max 8192 characters)"
                    ))
                
                # Remove control characters from header value
                clean_value = ''.join(
                    char for char in header_value 
                    if ord(char) >= 32 or char in '\t\n\r'
                )
                
                sanitized['header_name'] = header_name
                sanitized['header_value'] = clean_value
            
            return Either.right(sanitized)
            
        except Exception as e:
            return Either.left(ValidationError(
                field_name="credentials",
                value=str(credentials),
                constraint=f"Failed to sanitize credentials: {str(e)}"
            ))
    
    @staticmethod
    @require(lambda auth: isinstance(auth, AuthenticationCredentials))
    @ensure(lambda result: result.is_right() or isinstance(result.get_left(), ValidationError))
    def apply_authentication(
        auth: AuthenticationCredentials
    ) -> Either[ValidationError, Dict[str, Any]]:
        """Apply authentication to generate request headers/auth parameters."""
        try:
            if auth.auth_type == AuthenticationType.NONE:
                return Either.right({})
            
            elif auth.auth_type == AuthenticationType.API_KEY:
                api_key = auth.credentials['api_key']
                header_name = auth.credentials.get('header_name', 'X-API-Key')
                
                return Either.right({
                    "headers": {header_name: api_key}
                })
            
            elif auth.auth_type == AuthenticationType.BEARER_TOKEN:
                token = auth.credentials['token']
                
                return Either.right({
                    "headers": {"Authorization": f"Bearer {token}"}
                })
            
            elif auth.auth_type == AuthenticationType.BASIC_AUTH:
                username = auth.credentials['username']
                password = auth.credentials['password']
                
                # Encode credentials
                credentials_str = f"{username}:{password}"
                encoded_credentials = base64.b64encode(
                    credentials_str.encode('utf-8')
                ).decode('ascii')
                
                return Either.right({
                    "headers": {"Authorization": f"Basic {encoded_credentials}"}
                })
            
            elif auth.auth_type == AuthenticationType.OAUTH2:
                access_token = auth.credentials['access_token']
                token_type = auth.credentials.get('token_type', 'Bearer')
                
                return Either.right({
                    "headers": {"Authorization": f"{token_type} {access_token}"}
                })
            
            elif auth.auth_type == AuthenticationType.CUSTOM_HEADER:
                header_name = auth.credentials['header_name']
                header_value = auth.credentials['header_value']
                
                return Either.right({
                    "headers": {header_name: header_value}
                })
            
            else:
                return Either.left(ValidationError(
                    field_name="auth_type",
                    value=auth.auth_type.value,
                    constraint="Unsupported authentication type"
                ))
                
        except Exception as e:
            return Either.left(ValidationError(
                field_name="authentication",
                value=str(auth),
                constraint=f"Failed to apply authentication: {str(e)}"
            ))


class AuthenticationValidator:
    """Security validation for authentication credentials."""
    
    # Patterns that might indicate credential leakage
    CREDENTIAL_PATTERNS = [
        r'password\s*[:=]\s*["\']?([^"\'\\s]+)',
        r'secret\s*[:=]\s*["\']?([^"\'\\s]+)',
        r'token\s*[:=]\s*["\']?([^"\'\\s]+)',
        r'key\s*[:=]\s*["\']?([^"\'\\s]+)',
    ]
    
    @staticmethod
    def validate_credential_security(credential_value: str) -> Either[SecurityError, str]:
        """Validate credential for security issues."""
        if not credential_value or not credential_value.strip():
            return Either.left(SecurityError(
                "EMPTY_CREDENTIAL",
                "Credential cannot be empty"
            ))
        
        clean_value = credential_value.strip()
        
        # Check for minimum length
        if len(clean_value) < 8:
            return Either.left(SecurityError(
                "WEAK_CREDENTIAL",
                "Credential too short (minimum 8 characters)"
            ))
        
        # Check for obvious test/default values
        test_values = [
            'test', 'demo', 'example', 'sample', 'default',
            'password', 'secret', 'token', 'key', 'admin',
            '12345678', 'abcdefgh', 'password123'
        ]
        
        if clean_value.lower() in test_values:
            return Either.left(SecurityError(
                "TEST_CREDENTIAL",
                "Credential appears to be a test/default value"
            ))
        
        # Check for credential patterns that might indicate leakage
        for pattern in AuthenticationValidator.CREDENTIAL_PATTERNS:
            if re.search(pattern, clean_value, re.IGNORECASE):
                return Either.left(SecurityError(
                    "CREDENTIAL_PATTERN_DETECTED",
                    "Credential contains suspicious patterns"
                ))
        
        # Check for control characters
        if any(ord(char) < 32 and char not in '\t\n\r' for char in clean_value):
            return Either.left(SecurityError(
                "INVALID_CHARACTERS",
                "Credential contains invalid control characters"
            ))
        
        return Either.right(clean_value)
    
    @staticmethod
    def sanitize_credential_for_logging(credential: str) -> str:
        """Safely sanitize credential for logging purposes."""
        if not credential or len(credential) < 8:
            return "[REDACTED]"
        
        # Show first 2 and last 2 characters, mask the rest
        return f"{credential[:2]}{'*' * (len(credential) - 4)}{credential[-2:]}"


# Authentication factory functions for convenience
def create_api_key_auth(api_key: str, header_name: str = "X-API-Key") -> Either[ValidationError, AuthenticationCredentials]:
    """Create API key authentication."""
    return AuthenticationManager.create_authentication(
        "api_key",
        {"api_key": api_key, "header_name": header_name}
    )


def create_bearer_token_auth(token: str) -> Either[ValidationError, AuthenticationCredentials]:
    """Create Bearer token authentication."""
    return AuthenticationManager.create_authentication(
        "bearer_token",
        {"token": token}
    )


def create_basic_auth(username: str, password: str) -> Either[ValidationError, AuthenticationCredentials]:
    """Create HTTP Basic authentication."""
    return AuthenticationManager.create_authentication(
        "basic_auth",
        {"username": username, "password": password}
    )


def create_oauth2_auth(access_token: str, token_type: str = "Bearer") -> Either[ValidationError, AuthenticationCredentials]:
    """Create OAuth2 authentication."""
    return AuthenticationManager.create_authentication(
        "oauth2",
        {"access_token": access_token, "token_type": token_type}
    )


def create_custom_header_auth(header_name: str, header_value: str) -> Either[ValidationError, AuthenticationCredentials]:
    """Create custom header authentication."""
    return AuthenticationManager.create_authentication(
        "custom_header",
        {"header_name": header_name, "header_value": header_value}
    )