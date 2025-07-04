"""
Cloud integration type definitions for multi-platform cloud automation.

This module provides comprehensive type definitions for cloud providers, services,
authentication methods, and resource management with enterprise-grade security
and performance optimization for scalable cloud automation workflows.

Security: Enterprise-grade authentication, encryption, access control validation
Performance: <10s cloud connection, <30s resource creation, intelligent caching
Type Safety: Complete integration with enterprise security and audit frameworks
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
import json
import hashlib
import secrets

from .contracts import require, ensure
from .either import Either
from .errors import ValidationError


class CloudProvider(Enum):
    """Supported cloud providers with comprehensive coverage."""
    AWS = "aws"
    AZURE = "azure" 
    GOOGLE_CLOUD = "gcp"
    ALIBABA_CLOUD = "alibaba"
    DIGITAL_OCEAN = "digitalocean"
    LINODE = "linode"
    GENERIC = "generic"
    MULTI_CLOUD = "multi"


class CloudServiceType(Enum):
    """Types of cloud services for automation workflows."""
    STORAGE = "storage"
    COMPUTE = "compute"
    DATABASE = "database"
    MESSAGING = "messaging"
    AI_ML = "ai_ml"
    MONITORING = "monitoring"
    NETWORKING = "networking"
    SECURITY = "security"
    ANALYTICS = "analytics"
    SERVERLESS = "serverless"


class CloudAuthMethod(Enum):
    """Cloud authentication methods with security validation."""
    API_KEY = "api_key"
    SERVICE_ACCOUNT = "service_account"
    MANAGED_IDENTITY = "managed_identity"
    ROLE_BASED = "role_based"
    OAUTH2 = "oauth2"
    ACCESS_TOKEN = "access_token"
    CLIENT_CERTIFICATE = "client_certificate"


class CloudSecurityLevel(Enum):
    """Security levels for cloud operations."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    ENTERPRISE = "enterprise"
    GOVERNMENT = "government"


class CloudRegion(Enum):
    """Cloud regions for global deployment."""
    # AWS Regions
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    
    # Azure Regions
    AZURE_EAST_US = "eastus"
    AZURE_WEST_EUROPE = "westeurope"
    AZURE_SOUTHEAST_ASIA = "southeastasia"
    
    # GCP Regions
    GCP_US_CENTRAL1 = "us-central1"
    GCP_EUROPE_WEST1 = "europe-west1"
    GCP_ASIA_SOUTHEAST1 = "asia-southeast1"


class CloudError:
    """Cloud operation error types with detailed context."""
    
    def __init__(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now(UTC)
    
    @classmethod
    def authentication_failed(cls, details: str) -> 'CloudError':
        return cls("AUTHENTICATION_FAILED", f"Cloud authentication failed: {details}")
    
    @classmethod
    def connection_failed(cls, details: str) -> 'CloudError':
        return cls("CONNECTION_FAILED", f"Cloud connection failed: {details}")
    
    @classmethod
    def unsupported_provider(cls, provider: CloudProvider) -> 'CloudError':
        return cls("UNSUPPORTED_PROVIDER", f"Cloud provider {provider.value} not supported")
    
    @classmethod
    def resource_creation_failed(cls, details: str) -> 'CloudError':
        return cls("RESOURCE_CREATION_FAILED", f"Cloud resource creation failed: {details}")
    
    @classmethod
    def sync_operation_failed(cls, details: str) -> 'CloudError':
        return cls("SYNC_OPERATION_FAILED", f"Cloud sync operation failed: {details}")
    
    @classmethod
    def cost_analysis_failed(cls, details: str) -> 'CloudError':
        return cls("COST_ANALYSIS_FAILED", f"Cloud cost analysis failed: {details}")
    
    @classmethod
    def session_not_found(cls, session_id: str) -> 'CloudError':
        return cls("SESSION_NOT_FOUND", f"Cloud session {session_id} not found")
    
    @classmethod
    def insecure_auth_method(cls, method: CloudAuthMethod) -> 'CloudError':
        return cls("INSECURE_AUTH_METHOD", f"Authentication method {method.value} not allowed for enterprise")
    
    @classmethod
    def unsupported_auth_method(cls, method: CloudAuthMethod) -> 'CloudError':
        return cls("UNSUPPORTED_AUTH_METHOD", f"Authentication method {method.value} not supported")
    
    @classmethod
    def orchestration_failed(cls, details: str) -> 'CloudError':
        return cls("ORCHESTRATION_FAILED", f"Multi-cloud orchestration failed: {details}")
    
    @classmethod
    def monitoring_failed(cls, details: str) -> 'CloudError':
        return cls("MONITORING_FAILED", f"Resource monitoring failed: {details}")
    
    @classmethod
    def unsupported_operation_type(cls, operation_type: str) -> 'CloudError':
        return cls("UNSUPPORTED_OPERATION_TYPE", f"Operation type {operation_type} not supported")
    
    @classmethod
    def unsupported_provider_for_operation(cls, provider: CloudProvider) -> 'CloudError':
        return cls("UNSUPPORTED_PROVIDER_FOR_OPERATION", f"Provider {provider.value} does not support this operation")
    
    @classmethod
    def operation_not_implemented(cls, operation: str) -> 'CloudError':
        return cls("OPERATION_NOT_IMPLEMENTED", f"Operation {operation} not yet implemented")
    
    @classmethod
    def source_path_not_found(cls, path: str) -> 'CloudError':
        return cls("SOURCE_PATH_NOT_FOUND", f"Source path not found: {path}")
    
    @classmethod
    def no_subscriptions_found(cls) -> 'CloudError':
        return cls("NO_SUBSCRIPTIONS_FOUND", "No Azure subscriptions found")
    
    @classmethod
    def missing_subscription_id(cls) -> 'CloudError':
        return cls("MISSING_SUBSCRIPTION_ID", "Azure subscription ID required")
    
    @classmethod
    def missing_service_account_file(cls) -> 'CloudError':
        return cls("MISSING_SERVICE_ACCOUNT_FILE", "GCP service account file required")
    
    @classmethod
    def connection_establishment_failed(cls, details: str) -> 'CloudError':
        return cls("CONNECTION_ESTABLISHMENT_FAILED", f"Failed to establish connection: {details}")
    
    @classmethod
    def sync_not_supported_for_provider(cls, provider: CloudProvider) -> 'CloudError':
        return cls("SYNC_NOT_SUPPORTED_FOR_PROVIDER", f"Sync not supported for {provider.value}")
    
    @classmethod
    def incomplete_credentials(cls) -> 'CloudError':
        return cls("INCOMPLETE_CREDENTIALS", "Cloud credentials are incomplete")
    
    @classmethod
    def encryption_required_for_storage(cls) -> 'CloudError':
        return cls("ENCRYPTION_REQUIRED_FOR_STORAGE", "Encryption required for storage")
    
    @classmethod
    def public_access_not_allowed(cls) -> 'CloudError':
        return cls("PUBLIC_ACCESS_NOT_ALLOWED", "Public access not allowed")


@dataclass(frozen=True)
class CloudCredentials:
    """Cloud authentication credentials with security validation."""
    provider: CloudProvider
    auth_method: CloudAuthMethod
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    service_account_file: Optional[str] = None
    token: Optional[str] = None
    region: Optional[str] = None
    security_level: CloudSecurityLevel = CloudSecurityLevel.ENTERPRISE
    expires_at: Optional[datetime] = None
    additional_params: Dict[str, str] = field(default_factory=dict)
    
    @require(lambda self: self._validate_credentials())
    def __post_init__(self):
        """Validate credentials on creation."""
        if self.expires_at is None and self.token:
            # Default 1 hour expiry for tokens
            object.__setattr__(self, 'expires_at', datetime.now(UTC) + timedelta(hours=1))
    
    def _validate_credentials(self) -> bool:
        """Validate credentials based on provider and auth method."""
        if self.provider == CloudProvider.AWS:
            if self.auth_method == CloudAuthMethod.API_KEY:
                return bool(self.access_key and self.secret_key)
            elif self.auth_method == CloudAuthMethod.ROLE_BASED:
                return True  # Role-based auth uses IAM roles
        elif self.provider == CloudProvider.AZURE:
            if self.auth_method == CloudAuthMethod.SERVICE_ACCOUNT:
                return bool(self.tenant_id and self.client_id and self.client_secret)
            elif self.auth_method == CloudAuthMethod.MANAGED_IDENTITY:
                return True  # Managed identity doesn't need explicit credentials
        elif self.provider == CloudProvider.GOOGLE_CLOUD:
            if self.auth_method == CloudAuthMethod.SERVICE_ACCOUNT:
                return bool(self.service_account_file)
        
        return True
    
    def is_expired(self) -> bool:
        """Check if credentials are expired."""
        if self.expires_at:
            return datetime.now(UTC) > self.expires_at
        return False
    
    def get_safe_representation(self) -> Dict[str, Any]:
        """Get credentials without sensitive information."""
        return {
            "provider": self.provider.value,
            "auth_method": self.auth_method.value,
            "region": self.region,
            "security_level": self.security_level.value,
            "has_access_key": bool(self.access_key),
            "has_secret_key": bool(self.secret_key),
            "has_service_account": bool(self.service_account_file),
            "has_token": bool(self.token),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_expired": self.is_expired()
        }


@dataclass(frozen=True)
class CloudResource:
    """Cloud resource specification with comprehensive metadata."""
    resource_id: str
    provider: CloudProvider
    service_type: CloudServiceType
    resource_type: str
    region: str
    configuration: Dict[str, Any]
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: str = "unknown"
    cost_estimate: Optional[float] = None
    
    @require(lambda self: len(self.resource_id) > 0)
    @require(lambda self: len(self.resource_type) > 0)
    @require(lambda self: len(self.region) > 0)
    def __post_init__(self):
        """Initialize timestamps and cost estimates."""
        now = datetime.now(UTC)
        if self.created_at is None:
            object.__setattr__(self, 'created_at', now)
        if self.updated_at is None:
            object.__setattr__(self, 'updated_at', now)
        if self.cost_estimate is None:
            object.__setattr__(self, 'cost_estimate', self._estimate_monthly_cost())
    
    def get_arn(self) -> Optional[str]:
        """Get AWS ARN if applicable."""
        if self.provider == CloudProvider.AWS:
            service_name = self._get_aws_service_name()
            if service_name:
                return f"arn:aws:{service_name}:{self.region}:*:{self.resource_type}/{self.resource_id}"
        return None
    
    def _get_aws_service_name(self) -> Optional[str]:
        """Get AWS service name for ARN construction."""
        service_mapping = {
            CloudServiceType.STORAGE: "s3",
            CloudServiceType.COMPUTE: "ec2", 
            CloudServiceType.DATABASE: "rds",
            CloudServiceType.MESSAGING: "sqs",
            CloudServiceType.SERVERLESS: "lambda"
        }
        return service_mapping.get(self.service_type)
    
    def _estimate_monthly_cost(self) -> float:
        """Estimate monthly cost for resource."""
        # Base cost estimates (simplified for implementation)
        base_costs = {
            CloudServiceType.STORAGE: 0.023,  # per GB
            CloudServiceType.COMPUTE: 72.0,   # per month (t3.medium)
            CloudServiceType.DATABASE: 150.0, # per month (db.t3.micro)
            CloudServiceType.SERVERLESS: 0.20 # per 1M requests
        }
        return base_costs.get(self.service_type, 10.0)


@dataclass(frozen=True)
class CloudOperation:
    """Cloud operation specification with tracking."""
    operation_id: str
    operation_type: str
    provider: CloudProvider
    resource: CloudResource
    parameters: Dict[str, Any]
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    @require(lambda self: len(self.operation_id) > 0)
    @require(lambda self: len(self.operation_type) > 0)
    def __post_init__(self):
        """Initialize operation timestamps."""
        if self.started_at is None:
            object.__setattr__(self, 'started_at', datetime.now(UTC))
    
    def get_duration(self) -> Optional[timedelta]:
        """Get operation duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def is_completed(self) -> bool:
        """Check if operation is completed."""
        return self.status in ["completed", "failed", "cancelled"]


def create_cloud_credentials(
    provider: CloudProvider,
    auth_method: CloudAuthMethod,
    **kwargs
) -> CloudCredentials:
    """Factory function to create cloud credentials with validation."""
    return CloudCredentials(
        provider=provider,
        auth_method=auth_method,
        **kwargs
    )


def create_cloud_resource(
    resource_id: str,
    provider: CloudProvider,
    service_type: CloudServiceType,
    resource_type: str,
    region: str,
    configuration: Dict[str, Any],
    **kwargs
) -> CloudResource:
    """Factory function to create cloud resource with validation."""
    return CloudResource(
        resource_id=resource_id,
        provider=provider,
        service_type=service_type,
        resource_type=resource_type,
        region=region,
        configuration=configuration,
        **kwargs
    )