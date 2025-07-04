"""
Zero Trust Architecture - TASK_62 Phase 1 Core Implementation

Zero trust security type definitions and architectural framework for continuous validation.
Provides comprehensive types, enums, and utilities for trust validation, policy enforcement, and security monitoring.

Architecture: Branded Types + Design by Contract + Zero Trust Principles + Continuous Validation + Policy Enforcement
Performance: <100ms trust validation, <200ms policy enforcement, <300ms security monitoring
Security: Never trust always verify, continuous validation, comprehensive threat detection
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, NewType
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import re
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure


# Branded Types for Zero Trust Security Type Safety
TrustScore = NewType('TrustScore', float)
PolicyId = NewType('PolicyId', str)
SecurityContextId = NewType('SecurityContextId', str)
ThreatId = NewType('ThreatId', str)
ValidationId = NewType('ValidationId', str)
AccessTokenId = NewType('AccessTokenId', str)
RiskScore = NewType('RiskScore', float)
ComplianceId = NewType('ComplianceId', str)


def create_trust_score(score: float) -> TrustScore:
    """Create validated trust score (0.0 to 1.0)."""
    if not (0.0 <= score <= 1.0):
        raise ValueError("Trust score must be between 0.0 and 1.0")
    return TrustScore(score)


def create_policy_id(policy_name: str) -> PolicyId:
    """Create validated policy identifier."""
    if not policy_name or not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', policy_name):
        raise ValueError("Policy ID must be a valid identifier")
    return PolicyId(policy_name.lower())


def create_security_context_id() -> SecurityContextId:
    """Create unique security context identifier."""
    import uuid
    return SecurityContextId(f"sec_ctx_{uuid.uuid4().hex[:12]}")


def create_threat_id() -> ThreatId:
    """Create unique threat identifier."""
    import uuid
    return ThreatId(f"threat_{uuid.uuid4().hex[:8]}")


def create_validation_id() -> ValidationId:
    """Create unique validation identifier."""
    import uuid
    return ValidationId(f"validation_{uuid.uuid4().hex[:8]}")


def create_access_token_id() -> AccessTokenId:
    """Create unique access token identifier."""
    import uuid
    return AccessTokenId(f"token_{uuid.uuid4().hex[:16]}")


def create_risk_score(score: float) -> RiskScore:
    """Create validated risk score (0.0 to 1.0)."""
    if not (0.0 <= score <= 1.0):
        raise ValueError("Risk score must be between 0.0 and 1.0")
    return RiskScore(score)


def create_compliance_id(compliance_framework: str) -> ComplianceId:
    """Create compliance framework identifier."""
    if not compliance_framework or not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', compliance_framework):
        raise ValueError("Compliance ID must be a valid identifier")
    return ComplianceId(compliance_framework.upper())


class TrustLevel(Enum):
    """Trust levels for zero trust validation."""
    UNKNOWN = "unknown"              # Trust level not determined
    UNTRUSTED = "untrusted"          # Explicitly untrusted
    LOW = "low"                      # Low trust (0.0-0.3)
    MEDIUM = "medium"                # Medium trust (0.3-0.7)
    HIGH = "high"                    # High trust (0.7-0.9)
    VERIFIED = "verified"            # Verified trust (0.9-1.0)


class ValidationScope(Enum):
    """Scopes for trust validation."""
    USER = "user"                    # User identity validation
    DEVICE = "device"                # Device trust validation
    APPLICATION = "application"      # Application validation
    NETWORK = "network"              # Network security validation
    DATA = "data"                    # Data access validation
    SESSION = "session"              # Session validation
    TRANSACTION = "transaction"      # Transaction validation
    SYSTEM = "system"                # System-level validation


class SecurityOperation(Enum):
    """Types of security operations."""
    AUTHENTICATE = "authenticate"    # Authentication operation
    AUTHORIZE = "authorize"          # Authorization operation
    VALIDATE = "validate"            # Trust validation
    MONITOR = "monitor"              # Security monitoring
    ENFORCE = "enforce"              # Policy enforcement
    AUDIT = "audit"                  # Security auditing
    DETECT = "detect"                # Threat detection
    RESPOND = "respond"              # Incident response
    REMEDIATE = "remediate"          # Security remediation


class PolicyType(Enum):
    """Types of security policies."""
    ACCESS_CONTROL = "access_control"        # Access control policies
    DATA_PROTECTION = "data_protection"     # Data protection policies
    NETWORK_SECURITY = "network_security"   # Network security policies
    DEVICE_COMPLIANCE = "device_compliance" # Device compliance policies
    USER_BEHAVIOR = "user_behavior"         # User behavior policies
    APPLICATION_SECURITY = "application_security"  # Application security
    THREAT_RESPONSE = "threat_response"     # Threat response policies
    COMPLIANCE = "compliance"               # Regulatory compliance
    RISK_MANAGEMENT = "risk_management"     # Risk management policies


class EnforcementMode(Enum):
    """Policy enforcement modes."""
    MONITOR = "monitor"              # Monitor only, no action
    WARN = "warn"                    # Warn users about violations
    BLOCK = "block"                  # Block violating actions
    REMEDIATE = "remediate"          # Automatically remediate
    ADAPTIVE = "adaptive"            # Adaptive enforcement based on context


class ThreatSeverity(Enum):
    """Threat severity levels."""
    INFO = "info"                    # Informational
    LOW = "low"                      # Low severity
    MEDIUM = "medium"                # Medium severity
    HIGH = "high"                    # High severity
    CRITICAL = "critical"            # Critical severity
    EMERGENCY = "emergency"          # Emergency response required


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"                    # SOC 2 Type II
    HIPAA = "hipaa"                  # Health Insurance Portability and Accountability Act
    GDPR = "gdpr"                    # General Data Protection Regulation
    PCI_DSS = "pci_dss"              # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"            # ISO 27001 Information Security
    NIST = "nist"                    # NIST Cybersecurity Framework
    FedRAMP = "fedramp"              # Federal Risk and Authorization Management Program
    FISMA = "fisma"                  # Federal Information Security Management Act


@dataclass(frozen=True)
class ZeroTrustError(Exception):
    """Base class for zero trust security errors."""
    message: str
    error_code: str
    operation: Optional[SecurityOperation] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrustValidationError(ZeroTrustError):
    """Error in trust validation processing."""
    pass


@dataclass(frozen=True)
class PolicyEnforcementError(ZeroTrustError):
    """Error in security policy enforcement."""
    pass


@dataclass(frozen=True)
class SecurityMonitoringError(ZeroTrustError):
    """Error in security monitoring operations."""
    pass


@dataclass(frozen=True)
class AccessControlError(ZeroTrustError):
    """Error in access control operations."""
    pass


@dataclass(frozen=True)
class ComplianceError(ZeroTrustError):
    """Error in compliance operations."""
    pass


@dataclass(frozen=True)
class ValidationCriteria:
    """Criteria for trust validation."""
    identity_verification: bool = True
    device_compliance: bool = True
    location_verification: bool = True
    behavior_analysis: bool = True
    network_security: bool = True
    temporal_validation: bool = True
    risk_assessment: bool = True
    compliance_check: bool = True
    additional_factors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrustValidationResult:
    """Result of trust validation."""
    validation_id: ValidationId
    scope: ValidationScope
    target_id: str
    trust_score: TrustScore
    trust_level: TrustLevel
    validation_timestamp: datetime
    criteria_results: Dict[str, bool]
    risk_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.trust_score <= 1.0):
            raise ValueError("Trust score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class SecurityContext:
    """Security context for operations."""
    context_id: SecurityContextId
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    location: Optional[str] = None
    user_agent: Optional[str] = None
    application_context: Optional[str] = None
    security_clearance: Optional[str] = None
    trust_level: TrustLevel = TrustLevel.UNKNOWN
    risk_score: RiskScore = RiskScore(0.5)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SecurityPolicy:
    """Security policy definition."""
    policy_id: PolicyId
    policy_name: str
    policy_type: PolicyType
    description: str
    enforcement_mode: EnforcementMode
    scope: List[ValidationScope]
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
    exceptions: List[str] = field(default_factory=list)
    priority: int = 50  # 1-100, higher = more important
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = None
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (1 <= self.priority <= 100):
            raise ValueError("Policy priority must be between 1 and 100")


@dataclass(frozen=True)
class PolicyViolation:
    """Security policy violation."""
    violation_id: str
    policy_id: PolicyId
    target_id: str
    violation_type: str
    severity: ThreatSeverity
    description: str
    detected_at: datetime
    context: SecurityContext
    evidence: Dict[str, Any] = field(default_factory=dict)
    remediation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SecurityThreat:
    """Security threat detection."""
    threat_id: ThreatId
    threat_type: str
    severity: ThreatSeverity
    description: str
    source: str
    target: str
    detected_at: datetime
    indicators: List[str]
    mitigation_actions: List[str] = field(default_factory=list)
    confidence: float = 0.5
    risk_score: RiskScore = RiskScore(0.5)
    context: Optional[SecurityContext] = None
    related_threats: List[ThreatId] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class AccessDecision:
    """Access control decision."""
    decision_id: str
    request_id: str
    decision: str  # allow, deny, conditional
    reason: str
    context: SecurityContext
    resource: str
    action: str
    conditions: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    decided_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComplianceAssessment:
    """Compliance framework assessment."""
    assessment_id: str
    compliance_id: ComplianceId
    framework: ComplianceFramework
    assessment_date: datetime
    compliance_score: float
    requirements_met: int
    total_requirements: int
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_assessment: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.compliance_score <= 1.0):
            raise ValueError("Compliance score must be between 0.0 and 1.0")
        if self.requirements_met < 0 or self.total_requirements < 0:
            raise ValueError("Requirements counts cannot be negative")
        if self.requirements_met > self.total_requirements:
            raise ValueError("Requirements met cannot exceed total requirements")


@dataclass(frozen=True)
class SecurityMetrics:
    """Security metrics and KPIs."""
    metrics_id: str
    period_start: datetime
    period_end: datetime
    trust_validations: int
    policy_violations: int
    threats_detected: int
    incidents_resolved: int
    average_trust_score: float
    average_risk_score: float
    compliance_scores: Dict[str, float] = field(default_factory=dict)
    response_times: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Utility Functions
def determine_trust_level(trust_score: TrustScore) -> TrustLevel:
    """Determine trust level from numeric trust score."""
    if trust_score >= 0.9:
        return TrustLevel.VERIFIED
    elif trust_score >= 0.7:
        return TrustLevel.HIGH
    elif trust_score >= 0.3:
        return TrustLevel.MEDIUM
    elif trust_score > 0.0:
        return TrustLevel.LOW
    else:
        return TrustLevel.UNTRUSTED


def calculate_composite_trust_score(
    identity_score: float,
    device_score: float,
    behavior_score: float,
    location_score: float,
    weights: Optional[Dict[str, float]] = None
) -> TrustScore:
    """Calculate composite trust score from multiple factors."""
    if weights is None:
        weights = {
            "identity": 0.3,
            "device": 0.25,
            "behavior": 0.25,
            "location": 0.2
        }
    
    composite_score = (
        identity_score * weights.get("identity", 0.3) +
        device_score * weights.get("device", 0.25) +
        behavior_score * weights.get("behavior", 0.25) +
        location_score * weights.get("location", 0.2)
    )
    
    return create_trust_score(min(1.0, max(0.0, composite_score)))


def validate_security_context(context: SecurityContext) -> Either[ZeroTrustError, SecurityContext]:
    """Validate security context integrity."""
    try:
        # Check required fields
        if not context.context_id:
            return Either.left(ZeroTrustError("Security context ID is required", "MISSING_CONTEXT_ID"))
        
        # Validate IP address format if provided
        if context.ip_address:
            import ipaddress
            try:
                ipaddress.ip_address(context.ip_address)
            except ValueError:
                return Either.left(ZeroTrustError("Invalid IP address format", "INVALID_IP_ADDRESS"))
        
        # Check expiration
        if context.expires_at and context.expires_at < datetime.now(UTC):
            return Either.left(ZeroTrustError("Security context has expired", "CONTEXT_EXPIRED"))
        
        # Validate trust level and risk score consistency
        if context.trust_level == TrustLevel.HIGH and context.risk_score > 0.3:
            return Either.left(ZeroTrustError("Inconsistent trust level and risk score", "INCONSISTENT_SCORES"))
        
        return Either.right(context)
        
    except Exception as e:
        return Either.left(ZeroTrustError(f"Security context validation failed: {str(e)}", "VALIDATION_ERROR"))


def is_policy_applicable(
    policy: SecurityPolicy,
    context: SecurityContext,
    resource: str,
    action: str
) -> bool:
    """Check if a security policy is applicable to the current context."""
    # Check if policy is enabled and not expired
    if not policy.enabled:
        return False
    
    if policy.expires_at and policy.expires_at < datetime.now(UTC):
        return False
    
    # Check scope applicability
    applicable_scopes = []
    if context.user_id:
        applicable_scopes.append(ValidationScope.USER)
    if context.device_id:
        applicable_scopes.append(ValidationScope.DEVICE)
    if context.session_id:
        applicable_scopes.append(ValidationScope.SESSION)
    
    if not any(scope in policy.scope for scope in applicable_scopes):
        return False
    
    # Check resource and action conditions
    if "resources" in policy.conditions:
        resource_patterns = policy.conditions["resources"]
        if not any(re.match(pattern, resource) for pattern in resource_patterns):
            return False
    
    if "actions" in policy.conditions:
        allowed_actions = policy.conditions["actions"]
        if action not in allowed_actions:
            return False
    
    # Check trust level requirements
    if "min_trust_level" in policy.conditions:
        required_level = policy.conditions["min_trust_level"]
        if context.trust_level.value < required_level:
            return False
    
    return True


def calculate_threat_risk_score(
    threat: SecurityThreat,
    context: Optional[SecurityContext] = None
) -> RiskScore:
    """Calculate risk score for a security threat."""
    # Base risk from severity
    severity_weights = {
        ThreatSeverity.INFO: 0.1,
        ThreatSeverity.LOW: 0.3,
        ThreatSeverity.MEDIUM: 0.5,
        ThreatSeverity.HIGH: 0.8,
        ThreatSeverity.CRITICAL: 0.95,
        ThreatSeverity.EMERGENCY: 1.0
    }
    
    base_risk = severity_weights.get(threat.severity, 0.5)
    
    # Adjust for confidence
    confidence_adjusted_risk = base_risk * threat.confidence
    
    # Adjust for context if available
    if context:
        # Higher risk if trust level is low
        if context.trust_level == TrustLevel.UNTRUSTED:
            confidence_adjusted_risk *= 1.2
        elif context.trust_level == TrustLevel.LOW:
            confidence_adjusted_risk *= 1.1
        elif context.trust_level == TrustLevel.VERIFIED:
            confidence_adjusted_risk *= 0.9
        
        # Factor in existing risk score
        final_risk = (confidence_adjusted_risk + context.risk_score) / 2
    else:
        final_risk = confidence_adjusted_risk
    
    return create_risk_score(min(1.0, final_risk))


def generate_compliance_recommendations(
    assessment: ComplianceAssessment
) -> List[str]:
    """Generate compliance improvement recommendations."""
    recommendations = []
    
    compliance_percentage = assessment.requirements_met / assessment.total_requirements
    
    if compliance_percentage < 0.5:
        recommendations.append("Immediate attention required: Compliance below 50%")
        recommendations.append("Conduct comprehensive security review")
        recommendations.append("Implement emergency compliance measures")
    elif compliance_percentage < 0.8:
        recommendations.append("Address medium-priority compliance gaps")
        recommendations.append("Enhance monitoring and documentation")
    elif compliance_percentage < 0.95:
        recommendations.append("Fine-tune existing compliance measures")
        recommendations.append("Address remaining minor gaps")
    else:
        recommendations.append("Maintain current compliance posture")
        recommendations.append("Monitor for regulatory changes")
    
    # Framework-specific recommendations
    if assessment.framework == ComplianceFramework.SOC2:
        recommendations.append("Review security controls quarterly")
        recommendations.append("Maintain audit trail documentation")
    elif assessment.framework == ComplianceFramework.GDPR:
        recommendations.append("Review data processing activities")
        recommendations.append("Update privacy notices if needed")
    elif assessment.framework == ComplianceFramework.HIPAA:
        recommendations.append("Conduct security risk assessment")
        recommendations.append("Review business associate agreements")
    
    return recommendations


@require(lambda validation_result: isinstance(validation_result, TrustValidationResult))
def validate_trust_result(validation_result: TrustValidationResult) -> bool:
    """Validate trust validation result integrity."""
    # Check trust score consistency
    if not (0.0 <= validation_result.trust_score <= 1.0):
        return False
    
    # Check trust level alignment
    expected_level = determine_trust_level(validation_result.trust_score)
    if validation_result.trust_level != expected_level:
        return False
    
    # Check expiration
    if validation_result.expires_at:
        if validation_result.expires_at <= validation_result.validation_timestamp:
            return False
    
    # Check criteria results completeness
    required_criteria = ["identity", "device", "location", "behavior"]
    if not all(criteria in validation_result.criteria_results for criteria in required_criteria):
        return False
    
    return True


def is_zero_trust_related(description: str) -> bool:
    """Check if description is related to zero trust security operations."""
    zero_trust_keywords = [
        "trust", "verify", "validation", "security", "policy", "compliance",
        "authentication", "authorization", "access", "control", "monitor",
        "threat", "risk", "audit", "incident", "response", "remediation",
        "identity", "device", "network", "data", "session", "zero trust",
        "continuous", "validation", "enforcement", "detection", "protection"
    ]
    
    description_lower = description.lower()
    return any(keyword in description_lower for keyword in zero_trust_keywords)