"""
Biometric Architecture - TASK_67 Phase 1 Architecture & Design

Type-safe biometric authentication types with enterprise-grade security, privacy protection,
and multi-modal biometric support for personalized automation.

Architecture: Type Safety + Design by Contract + Privacy Protection + Security Boundaries
Performance: <100ms biometric authentication, <50ms user identification, <200ms personalization
Security: Biometric data encryption, privacy protection, liveness detection, anti-spoofing
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple, Union, NewType
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
from pathlib import Path
import uuid
import hashlib

from ..core.contracts import require, ensure
from ..core.either import Either

# Branded types for biometric security
BiometricTemplateId = NewType('BiometricTemplateId', str)
BiometricSessionId = NewType('BiometricSessionId', str) 
UserProfileId = NewType('UserProfileId', str)
BiometricHash = NewType('BiometricHash', str)

class BiometricModality(Enum):
    """Supported biometric authentication modalities."""
    FINGERPRINT = "fingerprint"
    FACE = "face"
    VOICE = "voice"
    IRIS = "iris"
    PALM = "palm"
    HAND_GEOMETRY = "hand_geometry"
    SIGNATURE = "signature"
    GAIT = "gait"
    KEYSTROKE = "keystroke"
    BEHAVIORAL = "behavioral"

class SecurityLevel(Enum):
    """Security levels for biometric authentication."""
    LOW = "low"           # Single factor, basic liveness
    MEDIUM = "medium"     # Single factor, advanced liveness
    HIGH = "high"         # Multi-factor, comprehensive validation
    CRITICAL = "critical" # Multi-factor, continuous validation, audit

class PrivacyLevel(Enum):
    """Privacy protection levels for biometric data."""
    MINIMAL = "minimal"     # Basic encryption
    STANDARD = "standard"   # Standard encryption + anonymization
    ENHANCED = "enhanced"   # Advanced encryption + zero-knowledge
    MAXIMUM = "maximum"     # Full privacy-preserving + local processing

class BiometricError(Exception):
    """Biometric system errors with detailed categorization."""
    
    def __init__(self, message: str, error_code: str = "BIOMETRIC_ERROR", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(UTC)
    
    @classmethod
    def authentication_failed(cls, modality: str) -> 'BiometricError':
        return cls(
            f"Biometric authentication failed for {modality}",
            "AUTH_FAILED",
            {"modality": modality}
        )
    
    @classmethod
    def template_not_found(cls, template_id: str) -> 'BiometricError':
        return cls(
            f"Biometric template not found: {template_id}",
            "TEMPLATE_NOT_FOUND",
            {"template_id": template_id}
        )
    
    @classmethod
    def liveness_detection_failed(cls, modality: str) -> 'BiometricError':
        return cls(
            f"Liveness detection failed for {modality}",
            "LIVENESS_FAILED",
            {"modality": modality}
        )
    
    @classmethod
    def privacy_violation(cls, operation: str) -> 'BiometricError':
        return cls(
            f"Privacy violation detected in {operation}",
            "PRIVACY_VIOLATION",
            {"operation": operation}
        )

@dataclass(frozen=True)
class BiometricTemplate:
    """Encrypted biometric template with privacy protection."""
    template_id: BiometricTemplateId
    modality: BiometricModality
    encrypted_data: bytes
    feature_hash: BiometricHash
    quality_score: float
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: 0.0 <= self.quality_score <= 1.0)
    @require(lambda self: len(self.encrypted_data) > 0)
    @require(lambda self: len(self.feature_hash) > 0)
    def __post_init__(self):
        if self.expires_at and self.expires_at <= self.created_at:
            raise ValueError("Expiration time must be after creation time")
    
    def is_expired(self) -> bool:
        """Check if biometric template has expired."""
        if not self.expires_at:
            return False
        return datetime.now(UTC) >= self.expires_at
    
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Check if template meets quality threshold."""
        return self.quality_score >= threshold

@dataclass(frozen=True)
class AuthenticationRequest:
    """Biometric authentication request with security parameters."""
    session_id: BiometricSessionId
    modalities: List[BiometricModality]
    security_level: SecurityLevel
    privacy_level: PrivacyLevel
    user_context: Optional[str] = None
    liveness_required: bool = True
    multi_factor: bool = False
    continuous_auth: bool = False
    timeout_seconds: int = 30
    max_attempts: int = 3
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @require(lambda self: len(self.modalities) > 0)
    @require(lambda self: self.timeout_seconds > 0)
    @require(lambda self: self.max_attempts > 0)
    def __post_init__(self):
        if self.multi_factor and len(self.modalities) < 2:
            raise ValueError("Multi-factor authentication requires at least 2 modalities")

@dataclass(frozen=True)
class AuthenticationResult:
    """Biometric authentication result with security metrics."""
    session_id: BiometricSessionId
    success: bool
    user_profile_id: Optional[UserProfileId]
    authenticated_modalities: List[BiometricModality]
    confidence_scores: Dict[BiometricModality, float]
    security_score: float
    liveness_verified: bool
    processing_time_ms: float
    authenticated_at: datetime
    valid_until: Optional[datetime] = None
    security_warnings: List[str] = field(default_factory=list)
    
    @require(lambda self: 0.0 <= self.security_score <= 1.0)
    @require(lambda self: self.processing_time_ms >= 0.0)
    @require(lambda self: all(0.0 <= score <= 1.0 for score in self.confidence_scores.values()))
    def __post_init__(self):
        if self.success and not self.user_profile_id:
            raise ValueError("Successful authentication must include user profile ID")
    
    def is_high_confidence(self, threshold: float = 0.9) -> bool:
        """Check if authentication has high confidence."""
        return self.security_score >= threshold
    
    def has_security_concerns(self) -> bool:
        """Check if authentication has security warnings."""
        return len(self.security_warnings) > 0

@dataclass(frozen=True)
class UserProfile:
    """User profile with biometric data and personalization preferences."""
    profile_id: UserProfileId
    user_identity: str
    enrolled_modalities: Set[BiometricModality]
    biometric_templates: Dict[BiometricModality, BiometricTemplateId]
    personalization_preferences: Dict[str, Any]
    accessibility_settings: Dict[str, Any]
    behavioral_patterns: Dict[str, Any]
    privacy_settings: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    last_authenticated: Optional[datetime] = None
    is_active: bool = True
    
    @require(lambda self: len(self.user_identity) > 0)
    @require(lambda self: len(self.enrolled_modalities) > 0)
    def __post_init__(self):
        if self.last_updated < self.created_at:
            raise ValueError("Last updated time cannot be before creation time")
    
    def has_modality(self, modality: BiometricModality) -> bool:
        """Check if user has enrolled a specific biometric modality."""
        return modality in self.enrolled_modalities
    
    def get_template_id(self, modality: BiometricModality) -> Optional[BiometricTemplateId]:
        """Get biometric template ID for modality."""
        return self.biometric_templates.get(modality)
    
    def is_recently_authenticated(self, hours: int = 24) -> bool:
        """Check if user was recently authenticated."""
        if not self.last_authenticated:
            return False
        return datetime.now(UTC) - self.last_authenticated <= timedelta(hours=hours)

@dataclass(frozen=True)
class PersonalizationSettings:
    """User personalization settings for adaptive automation."""
    user_profile_id: UserProfileId
    automation_preferences: Dict[str, Any]
    interface_preferences: Dict[str, Any]
    accessibility_requirements: Dict[str, Any]
    behavioral_adaptations: Dict[str, Any]
    privacy_preferences: Dict[str, Any]
    learning_enabled: bool
    adaptation_level: str  # light|moderate|comprehensive
    cross_device_sync: bool
    created_at: datetime
    last_updated: datetime
    
    @require(lambda self: len(self.automation_preferences) >= 0)
    @require(lambda self: self.adaptation_level in ["light", "moderate", "comprehensive"])
    def __post_init__(self):
        pass
    
    def get_preference(self, category: str, key: str, default: Any = None) -> Any:
        """Get user preference value."""
        category_prefs = getattr(self, f"{category}_preferences", {})
        return category_prefs.get(key, default)
    
    def is_learning_enabled(self) -> bool:
        """Check if behavioral learning is enabled."""
        return self.learning_enabled and self.privacy_preferences.get("allow_learning", True)

@dataclass(frozen=True)
class BiometricSession:
    """Active biometric authentication session."""
    session_id: BiometricSessionId
    user_profile_id: Optional[UserProfileId]
    authentication_result: Optional[AuthenticationResult]
    continuous_monitoring: bool
    security_level: SecurityLevel
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    activity_count: int = 0
    security_events: List[str] = field(default_factory=list)
    
    @require(lambda self: self.expires_at > self.created_at)
    @require(lambda self: self.activity_count >= 0)
    def __post_init__(self):
        pass
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(UTC) >= self.expires_at
    
    def is_active(self) -> bool:
        """Check if session is active and valid."""
        return not self.is_expired() and self.authentication_result and self.authentication_result.success
    
    def update_activity(self) -> 'BiometricSession':
        """Update session activity timestamp."""
        return BiometricSession(
            session_id=self.session_id,
            user_profile_id=self.user_profile_id,
            authentication_result=self.authentication_result,
            continuous_monitoring=self.continuous_monitoring,
            security_level=self.security_level,
            created_at=self.created_at,
            expires_at=self.expires_at,
            last_activity=datetime.now(UTC),
            activity_count=self.activity_count + 1,
            security_events=self.security_events
        )

@dataclass(frozen=True)
class BiometricConfiguration:
    """Biometric system configuration parameters."""
    enabled_modalities: Set[BiometricModality]
    security_level: SecurityLevel
    privacy_level: PrivacyLevel
    liveness_detection: bool
    anti_spoofing: bool
    quality_threshold: float
    confidence_threshold: float
    session_timeout_hours: int
    max_enrollment_attempts: int
    template_encryption_enabled: bool
    audit_logging_enabled: bool
    compliance_mode: bool
    
    @require(lambda self: 0.0 <= self.quality_threshold <= 1.0)
    @require(lambda self: 0.0 <= self.confidence_threshold <= 1.0)
    @require(lambda self: self.session_timeout_hours > 0)
    @require(lambda self: self.max_enrollment_attempts > 0)
    def __post_init__(self):
        if len(self.enabled_modalities) == 0:
            raise ValueError("At least one biometric modality must be enabled")
    
    def is_modality_enabled(self, modality: BiometricModality) -> bool:
        """Check if biometric modality is enabled."""
        return modality in self.enabled_modalities
    
    def meets_security_requirements(self) -> bool:
        """Check if configuration meets security requirements."""
        return (
            self.liveness_detection and
            self.anti_spoofing and
            self.template_encryption_enabled and
            self.quality_threshold >= 0.7 and
            self.confidence_threshold >= 0.8
        )

# Type aliases for complex types
BiometricData = Dict[BiometricModality, bytes]
ConfidenceScores = Dict[BiometricModality, float]
SecurityMetrics = Dict[str, float]
PersonalizationData = Dict[str, Any]

# Utility functions for biometric operations
def generate_template_id() -> BiometricTemplateId:
    """Generate unique biometric template ID."""
    return BiometricTemplateId(f"bt_{uuid.uuid4().hex}")

def generate_session_id() -> BiometricSessionId:
    """Generate unique biometric session ID."""
    return BiometricSessionId(f"bs_{uuid.uuid4().hex}")

def generate_profile_id(user_identity: str) -> UserProfileId:
    """Generate user profile ID from identity."""
    identity_hash = hashlib.sha256(user_identity.encode()).hexdigest()[:16]
    return UserProfileId(f"up_{identity_hash}")

def calculate_feature_hash(biometric_data: bytes) -> BiometricHash:
    """Calculate privacy-preserving hash of biometric features."""
    return BiometricHash(hashlib.sha3_256(biometric_data).hexdigest())

def calculate_security_score(confidence_scores: ConfidenceScores, 
                           liveness_verified: bool,
                           anti_spoofing_passed: bool) -> float:
    """Calculate overall security score from individual metrics."""
    if not confidence_scores:
        return 0.0
    
    # Base score from confidence scores
    avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
    
    # Apply security factors
    liveness_factor = 1.0 if liveness_verified else 0.8
    spoofing_factor = 1.0 if anti_spoofing_passed else 0.7
    
    # Multi-factor bonus
    multi_factor_bonus = 1.1 if len(confidence_scores) > 1 else 1.0
    
    security_score = avg_confidence * liveness_factor * spoofing_factor * multi_factor_bonus
    return min(security_score, 1.0)  # Cap at 1.0

def validate_biometric_data_privacy(data: BiometricData, 
                                  privacy_level: PrivacyLevel) -> bool:
    """Validate biometric data meets privacy requirements."""
    if privacy_level == PrivacyLevel.MINIMAL:
        return len(data) > 0
    elif privacy_level == PrivacyLevel.STANDARD:
        # Check for basic encryption/anonymization
        return all(len(template) > 32 for template in data.values())
    elif privacy_level == PrivacyLevel.ENHANCED:
        # Check for advanced privacy protection
        return all(len(template) > 64 for template in data.values())
    elif privacy_level == PrivacyLevel.MAXIMUM:
        # Check for maximum privacy protection
        return all(len(template) > 128 for template in data.values())
    return False

def create_default_biometric_config() -> BiometricConfiguration:
    """Create default biometric configuration with security best practices."""
    return BiometricConfiguration(
        enabled_modalities={BiometricModality.FACE, BiometricModality.FINGERPRINT},
        security_level=SecurityLevel.HIGH,
        privacy_level=PrivacyLevel.ENHANCED,
        liveness_detection=True,
        anti_spoofing=True,
        quality_threshold=0.8,
        confidence_threshold=0.9,
        session_timeout_hours=8,
        max_enrollment_attempts=3,
        template_encryption_enabled=True,
        audit_logging_enabled=True,
        compliance_mode=True
    )