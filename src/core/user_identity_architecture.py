"""
User Identity Architecture - TASK_67 Phase 1 Architecture & Design

Type-safe user identity management with enterprise-grade security, privacy protection,
and personalized automation for username-based authentication systems.

Architecture: Type Safety + Design by Contract + Privacy Protection + Security Boundaries
Performance: <100ms authentication, <50ms profile lookup, <200ms personalization
Security: Username/password encryption, privacy protection, session management, audit logging
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, NewType

from ..core.contracts import require

# Branded types for user identity security
UserSessionId = NewType("UserSessionId", str)
UserProfileId = NewType("UserProfileId", str)
UsernameHash = NewType("UsernameHash", str)
SessionToken = NewType("SessionToken", str)


class AuthenticationMethod(Enum):
    """Supported authentication methods."""

    PASSWORD = "password"
    TOKEN = "token"
    SSO = "sso"
    SESSION = "session"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"


class SecurityLevel(Enum):
    """Security levels for user authentication."""

    LOW = "low"  # Basic username/password
    MEDIUM = "medium"  # Username/password + session timeout
    HIGH = "high"  # Multi-factor + comprehensive validation
    CRITICAL = "critical"  # Multi-factor + continuous validation + audit


class PrivacyLevel(Enum):
    """Privacy protection levels for user data."""

    MINIMAL = "minimal"  # Basic encryption
    STANDARD = "standard"  # Standard encryption + data anonymization
    ENHANCED = "enhanced"  # Advanced encryption + zero-knowledge patterns
    MAXIMUM = "maximum"  # Full privacy-preserving + local processing only


class IdentityError(Exception):
    """User identity system errors with detailed categorization."""

    def __init__(
        self,
        message: str,
        error_code: str = "IDENTITY_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(UTC)

    @classmethod
    def authentication_failed(cls, username: str) -> IdentityError:
        return cls(
            f"Authentication failed for user {username}",
            "AUTH_FAILED",
            {"username": username},
        )

    @classmethod
    def user_not_found(cls, user_id: str) -> IdentityError:
        return cls(f"User not found: {user_id}", "USER_NOT_FOUND", {"user_id": user_id})

    @classmethod
    def session_expired(cls, session_id: str) -> IdentityError:
        return cls(
            f"Session expired: {session_id}",
            "SESSION_EXPIRED",
            {"session_id": session_id},
        )

    @classmethod
    def insufficient_permissions(cls, operation: str) -> IdentityError:
        return cls(
            f"Insufficient permissions for operation: {operation}",
            "INSUFFICIENT_PERMISSIONS",
            {"operation": operation},
        )

    @classmethod
    def privacy_violation(cls, operation: str) -> IdentityError:
        return cls(
            f"Privacy violation detected in {operation}",
            "PRIVACY_VIOLATION",
            {"operation": operation},
        )


@dataclass(frozen=True)
class UserCredentials:
    """Encrypted user credentials with security metadata."""

    username_hash: UsernameHash
    password_hash: str
    salt: str
    created_at: datetime
    last_updated: datetime
    failed_attempts: int = 0
    locked_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @require(lambda self: len(self.password_hash) > 0)
    @require(lambda self: len(self.salt) > 0)
    @require(lambda self: self.failed_attempts >= 0)
    def __post_init__(self):
        if self.locked_until and self.locked_until <= self.created_at:
            raise ValueError("Lock expiration must be after creation time")

    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if not self.locked_until:
            return False
        return datetime.now(UTC) < self.locked_until

    def should_lock_account(self, max_attempts: int = 5) -> bool:
        """Check if account should be locked due to failed attempts."""
        return self.failed_attempts >= max_attempts


@dataclass(frozen=True)
class AuthenticationRequest:
    """User authentication request with security parameters."""

    username: str
    authentication_method: AuthenticationMethod
    security_level: SecurityLevel
    privacy_level: PrivacyLevel
    session_duration_hours: int = 8
    remember_session: bool = True
    multi_factor: bool = False
    timeout_seconds: int = 30
    client_info: dict[str, Any] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @require(lambda self: len(self.username) > 0)
    @require(lambda self: self.session_duration_hours > 0)
    @require(lambda self: self.timeout_seconds > 0)
    def __post_init__(self):
        if self.multi_factor and self.security_level == SecurityLevel.LOW:
            raise ValueError(
                "Multi-factor authentication requires security level medium or higher"
            )


@dataclass(frozen=True)
class AuthenticationResult:
    """User authentication result with session information."""

    session_id: UserSessionId
    success: bool
    user_profile_id: UserProfileId | None
    username: str | None
    authentication_method: AuthenticationMethod
    security_level: SecurityLevel
    session_token: SessionToken | None
    processing_time_ms: float
    authenticated_at: datetime
    expires_at: datetime | None = None
    permissions: set[str] = field(default_factory=set)
    security_warnings: list[str] = field(default_factory=list)

    @require(lambda self: self.processing_time_ms >= 0.0)
    def __post_init__(self):
        if self.success and not self.user_profile_id:
            raise ValueError("Successful authentication must include user profile ID")
        if self.success and not self.session_token:
            raise ValueError("Successful authentication must include session token")

    def is_session_valid(self) -> bool:
        """Check if authentication session is still valid."""
        if not self.expires_at:
            return self.success
        return self.success and datetime.now(UTC) < self.expires_at

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions


@dataclass(frozen=True)
class UserProfile:
    """User profile with personalization preferences and behavioral data."""

    profile_id: UserProfileId
    username: str
    display_name: str
    email: str | None
    authentication_methods: set[AuthenticationMethod]
    personalization_preferences: dict[str, Any]
    accessibility_settings: dict[str, Any]
    behavioral_patterns: dict[str, Any]
    privacy_settings: dict[str, Any]
    permissions: set[str]
    created_at: datetime
    last_updated: datetime
    last_authenticated: datetime | None = None
    is_active: bool = True

    @require(lambda self: len(self.username) > 0)
    @require(lambda self: len(self.authentication_methods) > 0)
    def __post_init__(self):
        if self.last_updated < self.created_at:
            raise ValueError("Last updated time cannot be before creation time")

    def has_authentication_method(self, method: AuthenticationMethod) -> bool:
        """Check if user has enrolled a specific authentication method."""
        return method in self.authentication_methods

    def is_recently_authenticated(self, hours: int = 24) -> bool:
        """Check if user was recently authenticated."""
        if not self.last_authenticated:
            return False
        return datetime.now(UTC) - self.last_authenticated <= timedelta(hours=hours)

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions


@dataclass(frozen=True)
class PersonalizationSettings:
    """User personalization settings for adaptive automation."""

    user_profile_id: UserProfileId
    automation_preferences: dict[str, Any]
    interface_preferences: dict[str, Any]
    accessibility_requirements: dict[str, Any]
    behavioral_adaptations: dict[str, Any]
    privacy_preferences: dict[str, Any]
    learning_enabled: bool
    adaptation_level: str  # light|moderate|comprehensive
    cross_session_sync: bool
    created_at: datetime
    last_updated: datetime

    @require(lambda self: len(self.automation_preferences) >= 0)
    @require(
        lambda self: self.adaptation_level in ["light", "moderate", "comprehensive"]
    )
    def __post_init__(self):
        pass

    def get_preference(self, category: str, key: str, default: Any = None) -> Any:
        """Get user preference value."""
        category_prefs = getattr(self, f"{category}_preferences", {})
        return category_prefs.get(key, default)

    def is_learning_enabled(self) -> bool:
        """Check if behavioral learning is enabled."""
        return self.learning_enabled and self.privacy_preferences.get(
            "allow_learning", True
        )


@dataclass(frozen=True)
class UserSession:
    """Active user session with security tracking."""

    session_id: UserSessionId
    user_profile_id: UserProfileId
    session_token: SessionToken
    authentication_result: AuthenticationResult
    security_level: SecurityLevel
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    activity_count: int = 0
    client_info: dict[str, Any] = field(default_factory=dict)
    security_events: list[str] = field(default_factory=list)

    @require(lambda self: self.expires_at > self.created_at)
    @require(lambda self: self.activity_count >= 0)
    def __post_init__(self):
        pass

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(UTC) >= self.expires_at

    def is_active(self) -> bool:
        """Check if session is active and valid."""
        return not self.is_expired() and self.authentication_result.success

    def update_activity(self) -> UserSession:
        """Update session activity timestamp."""
        return UserSession(
            session_id=self.session_id,
            user_profile_id=self.user_profile_id,
            session_token=self.session_token,
            authentication_result=self.authentication_result,
            security_level=self.security_level,
            created_at=self.created_at,
            expires_at=self.expires_at,
            last_activity=datetime.now(UTC),
            activity_count=self.activity_count + 1,
            client_info=self.client_info,
            security_events=self.security_events,
        )


@dataclass(frozen=True)
class IdentityConfiguration:
    """User identity system configuration parameters."""

    enabled_auth_methods: set[AuthenticationMethod]
    security_level: SecurityLevel
    privacy_level: PrivacyLevel
    session_timeout_hours: int
    max_failed_attempts: int
    account_lockout_duration_minutes: int
    password_complexity_requirements: dict[str, Any]
    encryption_enabled: bool
    audit_logging_enabled: bool
    compliance_mode: bool

    @require(lambda self: self.session_timeout_hours > 0)
    @require(lambda self: self.max_failed_attempts > 0)
    @require(lambda self: self.account_lockout_duration_minutes > 0)
    def __post_init__(self):
        if len(self.enabled_auth_methods) == 0:
            raise ValueError("At least one authentication method must be enabled")

    def is_auth_method_enabled(self, method: AuthenticationMethod) -> bool:
        """Check if authentication method is enabled."""
        return method in self.enabled_auth_methods

    def meets_security_requirements(self) -> bool:
        """Check if configuration meets security requirements."""
        return (
            self.encryption_enabled
            and self.audit_logging_enabled
            and self.max_failed_attempts <= 5
            and self.session_timeout_hours <= 24
        )


# Type aliases for complex types
UserCredentialsData = dict[str, str]
SecurityMetrics = dict[str, float]
PersonalizationData = dict[str, Any]
BehaviorAnalysisData = dict[str, Any]


# Utility functions for user identity operations
def generate_session_id() -> UserSessionId:
    """Generate unique user session ID."""
    return UserSessionId(f"us_{uuid.uuid4().hex}")


def generate_profile_id(username: str) -> UserProfileId:
    """Generate user profile ID from username."""
    username_hash = hashlib.sha256(username.lower().encode()).hexdigest()[:16]
    return UserProfileId(f"up_{username_hash}")


def generate_session_token() -> SessionToken:
    """Generate secure session token."""
    return SessionToken(f"st_{uuid.uuid4().hex}_{uuid.uuid4().hex}")


def calculate_username_hash(username: str, salt: str) -> UsernameHash:
    """Calculate privacy-preserving hash of username."""
    combined = f"{username.lower()}:{salt}"
    return UsernameHash(hashlib.sha3_256(combined.encode()).hexdigest())


def calculate_password_hash(password: str, salt: str) -> str:
    """Calculate secure password hash with salt."""
    combined = f"{password}:{salt}"
    return hashlib.pbkdf2_hmac("sha256", combined.encode(), salt.encode(), 100000).hex()


def generate_salt() -> str:
    """Generate cryptographically secure salt."""
    import secrets

    return secrets.token_hex(32)


def validate_password_complexity(
    password: str, requirements: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Validate password meets complexity requirements."""
    errors = []

    min_length = requirements.get("min_length", 8)
    if len(password) < min_length:
        errors.append(f"Password must be at least {min_length} characters")

    if requirements.get("require_uppercase", True):
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

    if requirements.get("require_lowercase", True):
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

    if requirements.get("require_digits", True):
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")

    if requirements.get("require_special", True):
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            errors.append("Password must contain at least one special character")

    return len(errors) == 0, errors


def calculate_session_expiry(duration_hours: int) -> datetime:
    """Calculate session expiry time."""
    return datetime.now(UTC) + timedelta(hours=duration_hours)


def validate_user_permissions(
    user_permissions: set[str], required_permission: str
) -> bool:
    """Validate user has required permissions."""
    return required_permission in user_permissions


def create_default_identity_config() -> IdentityConfiguration:
    """Create default identity configuration with security best practices."""
    return IdentityConfiguration(
        enabled_auth_methods={
            AuthenticationMethod.PASSWORD,
            AuthenticationMethod.SESSION,
        },
        security_level=SecurityLevel.MEDIUM,
        privacy_level=PrivacyLevel.STANDARD,
        session_timeout_hours=8,
        max_failed_attempts=3,
        account_lockout_duration_minutes=15,
        password_complexity_requirements={
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
            "require_special": False,
        },
        encryption_enabled=True,
        audit_logging_enabled=True,
        compliance_mode=True,
    )


def analyze_authentication_risk(
    auth_request: AuthenticationRequest, user_profile: UserProfile | None = None
) -> dict[str, Any]:
    """Analyze authentication request for security risks."""
    risk_factors = {"risk_level": "low", "risk_score": 0.0, "factors": []}

    # Time-based risk analysis
    current_hour = datetime.now(UTC).hour
    if current_hour < 6 or current_hour > 22:  # Outside normal hours
        risk_factors["risk_score"] += 0.2
        risk_factors["factors"].append("outside_normal_hours")

    # Security level vs method mismatch
    if (
        auth_request.security_level == SecurityLevel.HIGH
        and auth_request.authentication_method == AuthenticationMethod.PASSWORD
    ):
        risk_factors["risk_score"] += 0.3
        risk_factors["factors"].append("security_method_mismatch")

    # User pattern analysis
    if user_profile and user_profile.is_recently_authenticated(hours=1):
        risk_factors["risk_score"] -= 0.1  # Lower risk for recent auth
        risk_factors["factors"].append("recent_authentication")

    # Calculate final risk level
    if risk_factors["risk_score"] > 0.7:
        risk_factors["risk_level"] = "high"
    elif risk_factors["risk_score"] > 0.3:
        risk_factors["risk_level"] = "medium"

    return risk_factors
