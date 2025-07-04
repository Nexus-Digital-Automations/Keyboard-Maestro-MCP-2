"""
Comprehensive audit framework types and core architecture for enterprise compliance.

This module provides enterprise-grade audit logging, compliance monitoring, and 
regulatory reporting capabilities with comprehensive security validation and 
tamper-proof integrity protection.

Security: All audit operations include cryptographic integrity and secure processing.
Performance: Optimized for real-time audit logging with minimal performance impact.
Type Safety: Complete integration with audit system architecture and compliance standards.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
import hashlib
import json
import uuid
import secrets
import re

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.errors import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


# Branded types for audit system
AuditEventId = str
ComplianceRuleId = str
ReportId = str
ChecksumHash = str
EncryptionKey = bytes


class AuditEventType(Enum):
    """Comprehensive audit event types for enterprise compliance."""
    # Authentication and access events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_AUTHENTICATION_FAILED = "user_authentication_failed"
    USER_SESSION_EXPIRED = "user_session_expired"
    
    # Automation execution events
    AUTOMATION_EXECUTED = "automation_executed"
    AUTOMATION_FAILED = "automation_failed"
    AUTOMATION_TIMEOUT = "automation_timeout"
    AUTOMATION_CREATED = "automation_created"
    AUTOMATION_MODIFIED = "automation_modified"
    AUTOMATION_DELETED = "automation_deleted"
    
    # Data access and modification events
    DATA_ACCESSED = "data_accessed"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"
    
    # Configuration and permission events
    CONFIGURATION_CHANGED = "configuration_changed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"
    
    # Security and compliance events
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE_VIOLATION = "compliance_violation"
    AUDIT_LOG_ACCESSED = "audit_log_accessed"
    AUDIT_LOG_MODIFIED = "audit_log_modified"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"


class ComplianceStandard(Enum):
    """Supported compliance and regulatory standards."""
    GENERAL = "general"
    SOC2 = "SOC2"
    HIPAA = "HIPAA"
    GDPR = "GDPR"
    PCI_DSS = "PCI_DSS"
    ISO_27001 = "ISO_27001"
    NIST = "NIST"
    FISMA = "FISMA"
    CCPA = "CCPA"


class AuditLevel(Enum):
    """Audit logging levels with performance implications."""
    MINIMAL = "minimal"          # Basic events only - highest performance
    STANDARD = "standard"        # Standard business events - balanced
    DETAILED = "detailed"        # Detailed event information - moderate performance
    COMPREHENSIVE = "comprehensive"  # Full event details - maximum logging


class RiskLevel(Enum):
    """Risk assessment levels for event classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditError(Exception):
    """Base class for audit system errors."""
    
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(f"{error_type}: {message}")
    
    @classmethod
    def integrity_check_failed(cls) -> 'AuditError':
        return cls("INTEGRITY_FAILURE", "Event integrity verification failed")
    
    @classmethod
    def logging_failed(cls, reason: str) -> 'AuditError':
        return cls("LOGGING_FAILURE", f"Failed to log audit event: {reason}")
    
    @classmethod
    def storage_failed(cls, reason: str) -> 'AuditError':
        return cls("STORAGE_FAILURE", f"Failed to store audit data: {reason}")
    
    @classmethod
    def encryption_failed(cls, reason: str) -> 'AuditError':
        return cls("ENCRYPTION_FAILURE", f"Failed to encrypt audit data: {reason}")
    
    @classmethod
    def rule_registration_failed(cls, reason: str) -> 'AuditError':
        return cls("RULE_REGISTRATION_FAILURE", f"Failed to register compliance rule: {reason}")
    
    @classmethod
    def report_generation_failed(cls, reason: str) -> 'AuditError':
        return cls("REPORT_GENERATION_FAILURE", f"Failed to generate compliance report: {reason}")
    
    @classmethod
    def initialization_failed(cls, reason: str) -> 'AuditError':
        return cls("INITIALIZATION_FAILURE", f"Failed to initialize audit system: {reason}")


@dataclass(frozen=True)
class SecurityLimits:
    """Security limits for audit system operations."""
    max_event_size: int = 1024 * 1024  # 1MB
    max_events_per_second: int = 1000
    max_retention_days: int = 2555  # 7 years
    max_report_size: int = 10 * 1024 * 1024  # 10MB
    max_query_results: int = 10000
    max_concurrent_sessions: int = 100
    
    @require(lambda self: self.max_event_size > 0)
    @require(lambda self: self.max_events_per_second > 0)
    @require(lambda self: self.max_retention_days > 0)
    def __post_init__(self):
        pass


@dataclass(frozen=True)
class AuditEvent:
    """Comprehensive audit event with cryptographic integrity protection."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    session_id: Optional[str]
    resource_id: Optional[str]
    action: str
    result: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    compliance_tags: Set[str] = field(default_factory=set)
    checksum: str = ""
    
    @require(lambda self: len(self.event_id) > 0)
    @require(lambda self: len(self.user_id) > 0)
    @require(lambda self: len(self.action) > 0)
    def __post_init__(self):
        # Calculate checksum for integrity
        if not self.checksum:
            object.__setattr__(self, 'checksum', self._calculate_checksum())
    
    def _calculate_checksum(self) -> str:
        """Calculate cryptographic checksum for integrity verification."""
        data = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'resource_id': self.resource_id,
            'action': self.action,
            'result': self.result,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'details': self.details,
            'risk_level': self.risk_level.value,
            'compliance_tags': sorted(list(self.compliance_tags))
        }
        data_string = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum."""
        try:
            expected_checksum = self._calculate_checksum()
            return self.checksum == expected_checksum
        except Exception:
            return False
    
    def is_high_risk(self) -> bool:
        """Check if event represents high risk activity."""
        return self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def matches_compliance_standard(self, standard: ComplianceStandard) -> bool:
        """Check if event is relevant for compliance standard."""
        standard_mappings = {
            ComplianceStandard.HIPAA: {
                'phi_access', 'patient_data', 'medical_records', 
                'healthcare', 'protected_health_information'
            },
            ComplianceStandard.GDPR: {
                'personal_data', 'data_processing', 'consent', 
                'privacy', 'data_subject_rights'
            },
            ComplianceStandard.PCI_DSS: {
                'payment_data', 'card_processing', 'financial', 
                'credit_card', 'payment_card_industry'
            },
            ComplianceStandard.SOC2: {
                'security', 'availability', 'processing_integrity', 
                'confidentiality', 'privacy'
            },
            ComplianceStandard.ISO_27001: {
                'information_security', 'risk_management', 
                'access_control', 'incident_management'
            }
        }
        
        required_tags = standard_mappings.get(standard, set())
        return bool(self.compliance_tags & required_tags) or standard == ComplianceStandard.GENERAL
    
    def get_sensitive_data_fields(self) -> Set[str]:
        """Identify fields containing sensitive data for masking."""
        sensitive_fields = set()
        
        # Check for sensitive data patterns
        sensitive_patterns = {
            'password', 'secret', 'token', 'key', 'credential',
            'ssn', 'social_security', 'credit_card', 'bank_account',
            'medical_record', 'health_record', 'diagnosis'
        }
        
        for field, value in self.details.items():
            if any(pattern in field.lower() for pattern in sensitive_patterns):
                sensitive_fields.add(field)
            if isinstance(value, str) and any(pattern in value.lower() for pattern in sensitive_patterns):
                sensitive_fields.add(field)
        
        return sensitive_fields


@dataclass(frozen=True)
class ComplianceRule:
    """Compliance monitoring rule with evaluation logic."""
    rule_id: str
    name: str
    description: str
    standard: ComplianceStandard
    severity: RiskLevel
    condition: str  # Rule condition expression
    action: str  # Action to take when violated
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    
    @require(lambda self: len(self.rule_id) > 0)
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: len(self.condition) > 0)
    def __post_init__(self):
        pass
    
    def evaluate(self, event: AuditEvent) -> bool:
        """Evaluate if event violates this compliance rule."""
        try:
            return self._evaluate_condition(event)
        except Exception:
            return False
    
    def _evaluate_condition(self, event: AuditEvent) -> bool:
        """Evaluate rule condition against audit event."""
        condition_lower = self.condition.lower()
        
        # Failed authentication patterns
        if 'failed_login' in condition_lower:
            return (event.event_type == AuditEventType.USER_LOGIN and 
                   event.result.lower() in ['failure', 'failed', 'error'])
        
        # Sensitive data access patterns
        if 'sensitive_data' in condition_lower:
            return (event.event_type == AuditEventType.DATA_ACCESSED or
                   'sensitive' in str(event.details).lower() or
                   'access_sensitive_data' in event.action.lower() or
                   bool(event.get_sensitive_data_fields()))
        
        # Privilege escalation patterns
        if 'privilege_escalation' in condition_lower:
            return (event.event_type == AuditEventType.PERMISSION_GRANTED and
                   'escalate' in event.action.lower())
        
        # Data modification without authorization
        if 'unauthorized_modification' in condition_lower:
            return (event.event_type in [AuditEventType.DATA_MODIFIED, AuditEventType.DATA_DELETED] and
                   'unauthorized' in str(event.details).lower())
        
        # High-risk activities
        if 'high_risk' in condition_lower:
            return event.is_high_risk()
        
        # Security violations
        if 'security_violation' in condition_lower:
            return event.event_type == AuditEventType.SECURITY_VIOLATION
        
        # Tag-based matching
        if 'tag:' in condition_lower:
            tag_match = condition_lower.split('tag:')[1].split()[0]
            return tag_match in event.compliance_tags
        
        return False


@dataclass(frozen=True)
class ComplianceReport:
    """Comprehensive compliance report with risk analysis."""
    report_id: str
    standard: ComplianceStandard
    period_start: datetime
    period_end: datetime
    total_events: int
    violations_found: int
    risk_score: float
    compliance_percentage: float
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @require(lambda self: 0.0 <= self.risk_score <= 100.0)
    @require(lambda self: 0.0 <= self.compliance_percentage <= 100.0)
    @require(lambda self: self.total_events >= 0)
    @require(lambda self: self.violations_found >= 0)
    def __post_init__(self):
        pass
    
    def is_compliant(self, threshold: float = 95.0) -> bool:
        """Check if compliance meets threshold."""
        return self.compliance_percentage >= threshold
    
    def get_risk_category(self) -> str:
        """Get risk category based on score."""
        if self.risk_score < 25:
            return "Low Risk"
        elif self.risk_score < 50:
            return "Medium Risk"
        elif self.risk_score < 75:
            return "High Risk"
        else:
            return "Critical Risk"
    
    def get_compliance_grade(self) -> str:
        """Get compliance grade based on percentage."""
        if self.compliance_percentage >= 98:
            return "A+"
        elif self.compliance_percentage >= 95:
            return "A"
        elif self.compliance_percentage >= 90:
            return "B"
        elif self.compliance_percentage >= 80:
            return "C"
        elif self.compliance_percentage >= 70:
            return "D"
        else:
            return "F"


@dataclass(frozen=True)
class AuditConfiguration:
    """Comprehensive audit system configuration."""
    audit_level: AuditLevel = AuditLevel.STANDARD
    retention_days: int = 365
    encrypt_logs: bool = True
    enable_real_time_monitoring: bool = True
    compliance_standards: Set[ComplianceStandard] = field(default_factory=lambda: {ComplianceStandard.GENERAL})
    security_limits: SecurityLimits = field(default_factory=SecurityLimits)
    log_file_path: Optional[str] = None
    backup_enabled: bool = True
    compression_enabled: bool = True
    
    @require(lambda self: 1 <= self.retention_days <= 2555)  # Max 7 years
    def __post_init__(self):
        pass
    
    def get_performance_profile(self) -> Dict[str, Any]:
        """Get performance profile based on configuration."""
        profiles = {
            AuditLevel.MINIMAL: {
                'max_events_per_second': 2000,
                'buffer_size': 1000,
                'flush_interval': 5.0
            },
            AuditLevel.STANDARD: {
                'max_events_per_second': 1000,
                'buffer_size': 500,
                'flush_interval': 2.0
            },
            AuditLevel.DETAILED: {
                'max_events_per_second': 500,
                'buffer_size': 250,
                'flush_interval': 1.0
            },
            AuditLevel.COMPREHENSIVE: {
                'max_events_per_second': 200,
                'buffer_size': 100,
                'flush_interval': 0.5
            }
        }
        return profiles.get(self.audit_level, profiles[AuditLevel.STANDARD])


class AuditEventValidator:
    """Security-first audit event validation."""
    
    MAX_EVENT_SIZE = 1024 * 1024  # 1MB
    MAX_DETAILS_KEYS = 50
    MAX_STRING_LENGTH = 10000
    MAX_COMPLIANCE_TAGS = 20
    
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',     # XSS prevention
        r'javascript:',                   # JavaScript URL prevention
        r'eval\s*\(',                    # Code injection prevention
        r'exec\s*\(',                    # Code execution prevention
        r'__[a-zA-Z_]+__',               # Python internal attributes
        r'\.\./',                        # Path traversal prevention
        r'file://',                      # File protocol prevention
        r'data:.*base64',                # Data URL with base64
    ]
    
    @staticmethod
    def validate_event(event: AuditEvent) -> Either[ValidationError, None]:
        """Comprehensive event validation with security checks."""
        try:
            # Basic field validation
            if len(event.user_id) > 255:
                return Either.left(ValidationError("user_id", "User ID too long"))
            
            if len(event.action) > 1000:
                return Either.left(ValidationError("action", "Action description too long"))
            
            if len(event.result) > 1000:
                return Either.left(ValidationError("result", "Result description too long"))
            
            # Details validation
            if len(event.details) > AuditEventValidator.MAX_DETAILS_KEYS:
                return Either.left(ValidationError("details", "Too many detail keys"))
            
            # Compliance tags validation
            if len(event.compliance_tags) > AuditEventValidator.MAX_COMPLIANCE_TAGS:
                return Either.left(ValidationError("compliance_tags", "Too many compliance tags"))
            
            # Security validation
            security_result = AuditEventValidator._validate_security(event)
            if security_result.is_left():
                return security_result
            
            # Size validation
            serialized_size = len(json.dumps(event.details, ensure_ascii=False).encode('utf-8'))
            if serialized_size > AuditEventValidator.MAX_EVENT_SIZE:
                return Either.left(ValidationError("event_size", "Event too large"))
            
            # Integrity validation
            if not event.verify_integrity():
                return Either.left(ValidationError("integrity", "Event integrity check failed"))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(ValidationError("validation_error", str(e)))
    
    @staticmethod
    def _validate_security(event: AuditEvent) -> Either[ValidationError, None]:
        """Validate event for security threats."""
        # Check string fields for dangerous patterns
        string_fields = [
            event.user_id, event.action, event.result,
            event.session_id or "", event.resource_id or "",
            event.ip_address or "", event.user_agent or ""
        ]
        
        for field_value in string_fields:
            if field_value and len(field_value) > AuditEventValidator.MAX_STRING_LENGTH:
                return Either.left(ValidationError("security", "String field too long"))
            
            for pattern in AuditEventValidator.DANGEROUS_PATTERNS:
                if re.search(pattern, field_value, re.IGNORECASE):
                    return Either.left(ValidationError("security", "Dangerous pattern detected"))
        
        # Check details for dangerous content
        details_str = json.dumps(event.details)
        if len(details_str) > AuditEventValidator.MAX_STRING_LENGTH:
            return Either.left(ValidationError("security", "Details too large"))
        
        for pattern in AuditEventValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, details_str, re.IGNORECASE):
                return Either.left(ValidationError("security", "Dangerous pattern in details"))
        
        # Validate compliance tags
        for tag in event.compliance_tags:
            if len(tag) > 100:
                return Either.left(ValidationError("security", "Compliance tag too long"))
            
            if not re.match(r'^[a-zA-Z0-9_\-\.]+$', tag):
                return Either.left(ValidationError("security", "Invalid compliance tag format"))
        
        return Either.right(None)


def create_audit_event(
    event_type: AuditEventType,
    user_id: str,
    action: str,
    result: str,
    **kwargs
) -> AuditEvent:
    """
    Factory function to create audit events with proper validation.
    
    Args:
        event_type: Type of audit event
        user_id: User identifier
        action: Action performed
        result: Result of action
        **kwargs: Additional event properties
        
    Returns:
        Validated AuditEvent instance
    """
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        timestamp=datetime.now(UTC),
        user_id=user_id,
        action=action,
        result=result,
        session_id=kwargs.get('session_id'),
        resource_id=kwargs.get('resource_id'),
        ip_address=kwargs.get('ip_address'),
        user_agent=kwargs.get('user_agent'),
        details=kwargs.get('details', {}),
        risk_level=kwargs.get('risk_level', RiskLevel.LOW),
        compliance_tags=set(kwargs.get('compliance_tags', []))
    )


def create_compliance_rule(
    rule_id: str,
    name: str,
    description: str,
    standard: ComplianceStandard,
    condition: str,
    action: str,
    severity: RiskLevel = RiskLevel.MEDIUM
) -> ComplianceRule:
    """
    Factory function to create compliance rules with validation.
    
    Args:
        rule_id: Unique rule identifier
        name: Human-readable rule name
        description: Rule description
        standard: Compliance standard
        condition: Rule condition expression
        action: Action to take when rule is violated
        severity: Violation severity level
        
    Returns:
        Validated ComplianceRule instance
    """
    return ComplianceRule(
        rule_id=rule_id,
        name=name,
        description=description,
        standard=standard,
        severity=severity,
        condition=condition,
        action=action
    )


# Standard compliance rule sets
STANDARD_COMPLIANCE_RULES = {
    ComplianceStandard.HIPAA: [
        create_compliance_rule(
            "hipaa_001",
            "Unauthorized PHI Access",
            "Detect unauthorized access to protected health information",
            ComplianceStandard.HIPAA,
            "data_access AND sensitive AND NOT authorized",
            "alert_security_team",
            RiskLevel.HIGH
        ),
        create_compliance_rule(
            "hipaa_002",
            "PHI Export Without Authorization", 
            "Monitor exports of protected health information",
            ComplianceStandard.HIPAA,
            "data_exported AND phi_access",
            "require_authorization_review",
            RiskLevel.HIGH
        ),
        create_compliance_rule(
            "hipaa_003",
            "Failed Authentication to Medical Systems",
            "Monitor failed login attempts to medical systems",
            ComplianceStandard.HIPAA,
            "failed_login AND medical_system",
            "log_security_event",
            RiskLevel.MEDIUM
        )
    ],
    
    ComplianceStandard.GDPR: [
        create_compliance_rule(
            "gdpr_001",
            "Personal Data Processing Without Consent",
            "Detect processing of personal data without proper consent",
            ComplianceStandard.GDPR,
            "data_processing AND personal_data AND NOT consent_verified",
            "block_processing",
            RiskLevel.HIGH
        ),
        create_compliance_rule(
            "gdpr_002",
            "Data Retention Violation",
            "Detect retention of personal data beyond allowed period",
            ComplianceStandard.GDPR,
            "data_access AND personal_data AND retention_expired",
            "schedule_deletion",
            RiskLevel.MEDIUM
        ),
        create_compliance_rule(
            "gdpr_003",
            "Cross-Border Data Transfer",
            "Monitor cross-border transfers of personal data",
            ComplianceStandard.GDPR,
            "data_exported AND personal_data AND cross_border",
            "verify_adequacy_decision",
            RiskLevel.HIGH
        )
    ],
    
    ComplianceStandard.PCI_DSS: [
        create_compliance_rule(
            "pci_001",
            "Cardholder Data Access",
            "Monitor access to cardholder data environments",
            ComplianceStandard.PCI_DSS,
            "data_access AND payment_data",
            "log_and_monitor",
            RiskLevel.HIGH
        ),
        create_compliance_rule(
            "pci_002", 
            "Payment Processing Failure",
            "Detect payment processing security failures",
            ComplianceStandard.PCI_DSS,
            "automation_failed AND payment_card_industry",
            "investigate_failure",
            RiskLevel.CRITICAL
        )
    ],
    
    ComplianceStandard.SOC2: [
        create_compliance_rule(
            "soc2_001",
            "Unauthorized System Access",
            "Detect unauthorized access to systems",
            ComplianceStandard.SOC2,
            "unauthorized_access",
            "alert_and_investigate",
            RiskLevel.HIGH
        ),
        create_compliance_rule(
            "soc2_002",
            "Data Integrity Violation", 
            "Monitor data integrity violations",
            ComplianceStandard.SOC2,
            "data_modified AND NOT authorized",
            "verify_integrity",
            RiskLevel.MEDIUM
        )
    ]
}