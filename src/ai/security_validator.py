"""
AI security validation system for enterprise-grade protection.

This module provides comprehensive security validation for AI processing
operations including content filtering, PII detection, threat analysis,
and privacy protection. Implements defense-in-depth security architecture.

Security: Multi-layer validation with threat detection and content filtering.
Performance: Optimized security scanning with intelligent pattern matching.
Type Safety: Complete integration with AI security architecture.
"""

import re
import hashlib
import json
from typing import Dict, List, Optional, Any, Set, Union, Tuple, Pattern
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum

from src.core.ai_integration import (
    AIRequest, AIResponse, AISecurityLevel, AISecurityConfig,
    AIOperation, ProcessingMode
)
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.errors import SecurityError, ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class SecurityThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityThreatType(Enum):
    """Types of security threats detected."""
    PII_DETECTED = "pii_detected"
    MALWARE_SIGNATURE = "malware_signature"
    INJECTION_ATTEMPT = "injection_attempt"
    EXCESSIVE_SIZE = "excessive_size"
    SPAM_CONTENT = "spam_content"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    CREDENTIAL_EXPOSURE = "credential_exposure"
    DANGEROUS_PATTERN = "dangerous_pattern"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"


@dataclass(frozen=True)
class SecurityThreat:
    """Detected security threat with details."""
    threat_type: SecurityThreatType
    severity: SecurityThreatLevel
    description: str
    detected_content: str
    confidence: float
    mitigation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: len(self.description) > 0)
    def __post_init__(self):
        """Validate security threat data."""
        pass
    
    def is_blocking(self) -> bool:
        """Check if threat should block processing."""
        return self.severity in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]
    
    def get_risk_score(self) -> float:
        """Calculate numeric risk score (0-100)."""
        severity_scores = {
            SecurityThreatLevel.LOW: 25,
            SecurityThreatLevel.MEDIUM: 50,
            SecurityThreatLevel.HIGH: 75,
            SecurityThreatLevel.CRITICAL: 100
        }
        base_score = severity_scores[self.severity]
        return base_score * self.confidence


@dataclass
class SecurityScanResult:
    """Result of security scanning operation."""
    is_safe: bool
    risk_score: float
    threats: List[SecurityThreat]
    scan_time: float
    recommendations: List[str]
    
    def get_blocking_threats(self) -> List[SecurityThreat]:
        """Get threats that should block processing."""
        return [threat for threat in self.threats if threat.is_blocking()]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of scan results."""
        return {
            "is_safe": self.is_safe,
            "risk_score": self.risk_score,
            "threat_count": len(self.threats),
            "blocking_threats": len(self.get_blocking_threats()),
            "highest_severity": max((t.severity.value for t in self.threats), default="none"),
            "scan_time": self.scan_time,
            "recommendations": self.recommendations
        }


class PIIDetector:
    """Personal Identifiable Information detection system."""
    
    def __init__(self):
        # Compiled regex patterns for performance
        self.patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            "api_key": re.compile(r'(?i)(?:api[_-]?key|token|secret)["\s:=]+[a-zA-Z0-9_-]{20,}'),
            "password": re.compile(r'(?i)(?:password|pwd)["\s:=]+[^\s"\']{8,}'),
            "address": re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr)\b', re.IGNORECASE)
        }
    
    def detect_pii(self, text: str) -> List[SecurityThreat]:
        """Detect PII in text content."""
        threats = []
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                # Mask the detected content
                masked_content = self._mask_sensitive_data(match, pii_type)
                
                threat = SecurityThreat(
                    threat_type=SecurityThreatType.PII_DETECTED,
                    severity=self._get_pii_severity(pii_type),
                    description=f"Detected {pii_type.upper()} in content",
                    detected_content=masked_content,
                    confidence=0.9,
                    mitigation=f"Remove or mask {pii_type} before processing"
                )
                threats.append(threat)
        
        return threats
    
    def _mask_sensitive_data(self, content: str, pii_type: str) -> str:
        """Mask sensitive data for logging."""
        if pii_type == "email":
            parts = content.split('@')
            if len(parts) == 2:
                return f"{parts[0][:2]}***@{parts[1]}"
        elif pii_type in ["phone", "ssn", "credit_card"]:
            return content[:4] + "*" * (len(content) - 4)
        elif pii_type in ["api_key", "password"]:
            return content[:4] + "*" * 8
        else:
            return content[:4] + "***" if len(content) > 4 else "***"
    
    def _get_pii_severity(self, pii_type: str) -> SecurityThreatLevel:
        """Get severity level for PII type."""
        high_risk = ["ssn", "credit_card", "api_key", "password"]
        medium_risk = ["email", "phone", "address"]
        
        if pii_type in high_risk:
            return SecurityThreatLevel.HIGH
        elif pii_type in medium_risk:
            return SecurityThreatLevel.MEDIUM
        else:
            return SecurityThreatLevel.LOW


class ContentFilter:
    """Content filtering and threat detection system."""
    
    def __init__(self):
        # Malicious patterns
        self.malware_patterns = [
            r'(?i)(?:eval|exec|system|shell_exec|passthru|proc_open)\s*\(',
            r'(?i)(?:script|javascript|vbscript)\s*:',
            r'(?i)<\s*script[^>]*>',
            r'(?i)document\.(?:write|cookie)',
            r'(?i)window\.(?:location|open)',
            r'(?i)(?:drop|delete|truncate|alter)\s+table',
            r'(?i)union\s+select',
            r'(?i)xp_cmdshell',
            r'(?i)cmd\.exe|powershell\.exe'
        ]
        
        # Injection patterns
        self.injection_patterns = [
            r'[\'";]\s*(?:or|and)\s+[\'"]?\d+[\'"]?\s*[=<>]',
            r'(?i)(?:union|select|insert|update|delete|drop|create|alter)\s+',
            r'(?i)(?:script|iframe|object|embed|form)\s*>',
            r'(?i)on(?:load|error|click|mouse)\s*=',
            r'(?i)(?:http|https|ftp)://[^\s]+\.(exe|bat|com|scr|vbs|js)',
            r'[<>"\'](?:.*[<>"\']){2,}'  # Potential tag injection
        ]
        
        # Spam indicators
        self.spam_patterns = [
            r'(?i)(?:free|win|winner|congratulations|urgent|act\s+now|limited\s+time)',
            r'(?i)(?:click\s+here|call\s+now|buy\s+now|order\s+now)',
            r'(?i)(?:\$\d+|money|cash|income|profit|earn)',
            r'(?i)(?:guarantee|risk\s*free|no\s*obligation)',
            r'[A-Z]{4,}.*[A-Z]{4,}.*[A-Z]{4,}',  # Excessive caps
            r'[!]{3,}|[?]{3,}',  # Excessive punctuation
            r'(?:https?://[^\s]+\s*){3,}'  # Multiple URLs
        ]
        
        # Compile patterns for performance
        self.compiled_malware = [re.compile(p) for p in self.malware_patterns]
        self.compiled_injection = [re.compile(p) for p in self.injection_patterns]
        self.compiled_spam = [re.compile(p) for p in self.spam_patterns]
    
    def scan_content(self, content: str) -> List[SecurityThreat]:
        """Scan content for various security threats."""
        threats = []
        
        # Check for malware signatures
        threats.extend(self._scan_malware(content))
        
        # Check for injection attempts
        threats.extend(self._scan_injection(content))
        
        # Check for spam content
        threats.extend(self._scan_spam(content))
        
        # Check content size
        threats.extend(self._check_size_limits(content))
        
        return threats
    
    def _scan_malware(self, content: str) -> List[SecurityThreat]:
        """Scan for malware signatures."""
        threats = []
        
        for pattern in self.compiled_malware:
            matches = pattern.findall(content)
            for match in matches:
                threat = SecurityThreat(
                    threat_type=SecurityThreatType.MALWARE_SIGNATURE,
                    severity=SecurityThreatLevel.CRITICAL,
                    description="Potential malware signature detected",
                    detected_content=match[:50] + "..." if len(match) > 50 else match,
                    confidence=0.85,
                    mitigation="Remove suspicious code patterns"
                )
                threats.append(threat)
        
        return threats
    
    def _scan_injection(self, content: str) -> List[SecurityThreat]:
        """Scan for injection attempts."""
        threats = []
        
        for pattern in self.compiled_injection:
            matches = pattern.findall(content)
            for match in matches:
                threat = SecurityThreat(
                    threat_type=SecurityThreatType.INJECTION_ATTEMPT,
                    severity=SecurityThreatLevel.HIGH,
                    description="Potential injection attempt detected",
                    detected_content=match[:50] + "..." if len(match) > 50 else match,
                    confidence=0.8,
                    mitigation="Sanitize input and validate content"
                )
                threats.append(threat)
        
        return threats
    
    def _scan_spam(self, content: str) -> List[SecurityThreat]:
        """Scan for spam indicators."""
        threats = []
        spam_score = 0
        detected_patterns = []
        
        for pattern in self.compiled_spam:
            matches = pattern.findall(content)
            if matches:
                spam_score += len(matches)
                detected_patterns.extend(matches[:3])  # Limit examples
        
        if spam_score >= 3:  # Threshold for spam detection
            threat = SecurityThreat(
                threat_type=SecurityThreatType.SPAM_CONTENT,
                severity=SecurityThreatLevel.MEDIUM,
                description=f"High spam score detected ({spam_score} indicators)",
                detected_content=str(detected_patterns[:3]),
                confidence=min(0.95, spam_score * 0.2),
                mitigation="Review content for spam characteristics"
            )
            threats.append(threat)
        
        return threats
    
    def _check_size_limits(self, content: str) -> List[SecurityThreat]:
        """Check content size limits."""
        threats = []
        max_size = 1_000_000  # 1MB
        
        if len(content) > max_size:
            threat = SecurityThreat(
                threat_type=SecurityThreatType.EXCESSIVE_SIZE,
                severity=SecurityThreatLevel.HIGH,
                description=f"Content size {len(content)} exceeds limit {max_size}",
                detected_content=f"Content length: {len(content)} bytes",
                confidence=1.0,
                mitigation="Reduce content size or split into smaller chunks"
            )
            threats.append(threat)
        
        return threats


class AISecurityValidator:
    """Comprehensive AI security validation system."""
    
    def __init__(self, config: Optional[AISecurityConfig] = None):
        self.config = config or AISecurityConfig()
        self.pii_detector = PIIDetector()
        self.content_filter = ContentFilter()
        self.threat_log: List[SecurityThreat] = []
        self.rate_limiter = RateLimiter()
    
    async def validate_request(
        self,
        request: AIRequest,
        user_id: Optional[str] = None
    ) -> Either[SecurityError, SecurityScanResult]:
        """Validate AI request for security compliance."""
        try:
            start_time = datetime.now(UTC)
            threats = []
            recommendations = []
            
            # Rate limiting check
            if user_id:
                rate_check = self.rate_limiter.check_rate_limit(user_id, request.operation)
                if rate_check.is_left():
                    return rate_check
            
            # Input validation
            input_threats = await self._validate_input_content(request)
            threats.extend(input_threats)
            
            # Privacy mode validation
            if request.privacy_mode:
                privacy_threats = self._validate_privacy_compliance(request)
                threats.extend(privacy_threats)
            
            # Model security validation
            model_threats = self._validate_model_security(request)
            threats.extend(model_threats)
            
            # Cost and resource validation
            resource_threats = self._validate_resource_limits(request)
            threats.extend(resource_threats)
            
            # Calculate overall risk
            risk_score = self._calculate_risk_score(threats)
            
            # Determine if processing should be blocked
            blocking_threats = [t for t in threats if t.is_blocking()]
            is_safe = len(blocking_threats) == 0 and risk_score < 70
            
            # Generate recommendations
            if threats:
                recommendations = self._generate_recommendations(threats)
            
            # Log threats
            self.threat_log.extend(threats)
            self._cleanup_old_threats()
            
            scan_time = (datetime.now(UTC) - start_time).total_seconds()
            
            result = SecurityScanResult(
                is_safe=is_safe,
                risk_score=risk_score,
                threats=threats,
                scan_time=scan_time,
                recommendations=recommendations
            )
            
            if not is_safe:
                logger.warning(f"Security validation failed: {len(blocking_threats)} blocking threats")
                return Either.left(SecurityError(
                    "security_validation_failed",
                    f"Request blocked due to security threats: {[t.threat_type.value for t in blocking_threats]}"
                ))
            
            logger.debug(f"Security validation passed: risk score {risk_score:.1f}")
            return Either.right(result)
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return Either.left(SecurityError("validation_error", str(e)))
    
    async def _validate_input_content(self, request: AIRequest) -> List[SecurityThreat]:
        """Validate input content for security threats."""
        threats = []
        content = request.prepare_input_for_model()
        
        # PII detection
        if self.config.enable_pii_detection:
            pii_threats = self.pii_detector.detect_pii(content)
            threats.extend(pii_threats)
        
        # Content filtering
        if self.config.enable_content_filtering:
            content_threats = self.content_filter.scan_content(content)
            threats.extend(content_threats)
        
        # System prompt validation
        if request.system_prompt:
            system_threats = self.content_filter.scan_content(request.system_prompt)
            threats.extend(system_threats)
        
        return threats
    
    def _validate_privacy_compliance(self, request: AIRequest) -> List[SecurityThreat]:
        """Validate privacy compliance for request."""
        threats = []
        
        if not request.privacy_mode:
            return threats
        
        # Enhanced PII detection in privacy mode
        content = request.prepare_input_for_model()
        
        # Stricter patterns in privacy mode
        strict_patterns = [
            r'\b\d{4}\s*-?\s*\d{4}\s*-?\s*\d{4}\s*-?\s*\d{4}\b',  # Credit card patterns
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'  # Phone
        ]
        
        for pattern_str in strict_patterns:
            pattern = re.compile(pattern_str)
            if pattern.search(content):
                threat = SecurityThreat(
                    threat_type=SecurityThreatType.PII_DETECTED,
                    severity=SecurityThreatLevel.HIGH,
                    description="PII detected in privacy mode",
                    detected_content="[REDACTED]",
                    confidence=0.9,
                    mitigation="Remove all PII before processing in privacy mode"
                )
                threats.append(threat)
                break  # Don't duplicate threats
        
        return threats
    
    def _validate_model_security(self, request: AIRequest) -> List[SecurityThreat]:
        """Validate model security settings."""
        threats = []
        
        # Check for dangerous temperature settings
        if request.temperature > 1.5:
            threat = SecurityThreat(
                threat_type=SecurityThreatType.SUSPICIOUS_BEHAVIOR,
                severity=SecurityThreatLevel.MEDIUM,
                description=f"High temperature setting: {request.temperature}",
                detected_content=f"temperature={request.temperature}",
                confidence=0.7,
                mitigation="Use temperature <= 1.5 for safer outputs"
            )
            threats.append(threat)
        
        # Check for excessive token requests
        max_tokens = request.get_effective_max_tokens()
        if max_tokens > 8192:
            threat = SecurityThreat(
                threat_type=SecurityThreatType.EXCESSIVE_SIZE,
                severity=SecurityThreatLevel.MEDIUM,
                description=f"Excessive max tokens: {max_tokens}",
                detected_content=f"max_tokens={max_tokens}",
                confidence=0.8,
                mitigation="Reduce max_tokens to reasonable limits"
            )
            threats.append(threat)
        
        return threats
    
    def _validate_resource_limits(self, request: AIRequest) -> List[SecurityThreat]:
        """Validate resource usage limits."""
        threats = []
        
        # Estimate cost
        estimated_tokens = request.estimate_input_tokens()
        estimated_cost = request.model.estimate_cost(estimated_tokens, TokenCount(estimated_tokens // 2))
        
        # Check cost limits
        if estimated_cost > 10.0:  # $10 threshold
            threat = SecurityThreat(
                threat_type=SecurityThreatType.EXCESSIVE_SIZE,
                severity=SecurityThreatLevel.HIGH,
                description=f"High estimated cost: ${estimated_cost:.2f}",
                detected_content=f"estimated_cost=${estimated_cost:.2f}",
                confidence=0.9,
                mitigation="Reduce input size or use cheaper model"
            )
            threats.append(threat)
        
        return threats
    
    def _calculate_risk_score(self, threats: List[SecurityThreat]) -> float:
        """Calculate overall risk score from threats."""
        if not threats:
            return 0.0
        
        # Weighted average based on severity and confidence
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for threat in threats:
            weight = threat.confidence
            score = threat.get_risk_score()
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, threats: List[SecurityThreat]) -> List[str]:
        """Generate security recommendations based on threats."""
        recommendations = []
        threat_types = set(threat.threat_type for threat in threats)
        
        if SecurityThreatType.PII_DETECTED in threat_types:
            recommendations.append("Remove or mask personal information before processing")
        
        if SecurityThreatType.MALWARE_SIGNATURE in threat_types:
            recommendations.append("Scan content for malicious code patterns")
        
        if SecurityThreatType.INJECTION_ATTEMPT in threat_types:
            recommendations.append("Validate and sanitize all input content")
        
        if SecurityThreatType.SPAM_CONTENT in threat_types:
            recommendations.append("Review content for spam characteristics")
        
        if SecurityThreatType.EXCESSIVE_SIZE in threat_types:
            recommendations.append("Reduce content size or split into smaller chunks")
        
        # Add general recommendations
        if len(threats) > 3:
            recommendations.append("Consider using stricter security settings")
        
        return recommendations
    
    def _cleanup_old_threats(self) -> None:
        """Remove old threats from log."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)
        self.threat_log = [
            threat for threat in self.threat_log
            if threat.timestamp > cutoff_time
        ]
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security validation statistics."""
        recent_threats = [
            threat for threat in self.threat_log
            if threat.timestamp > datetime.now(UTC) - timedelta(hours=1)
        ]
        
        threat_counts = {}
        for threat in recent_threats:
            threat_type = threat.threat_type.value
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        return {
            "total_threats_logged": len(self.threat_log),
            "recent_threats_1h": len(recent_threats),
            "threat_breakdown": threat_counts,
            "security_config": {
                "security_level": self.config.security_level.value,
                "content_filtering": self.config.enable_content_filtering,
                "pii_detection": self.config.enable_pii_detection,
                "malware_scanning": self.config.enable_malware_scanning
            },
            "rate_limit_status": self.rate_limiter.get_status()
        }


class RateLimiter:
    """Rate limiting for AI requests."""
    
    def __init__(self):
        self.request_history: Dict[str, List[datetime]] = {}
        self.limits = {
            AIOperation.GENERATE: 20,  # 20 per minute
            AIOperation.ANALYZE: 30,   # 30 per minute
            AIOperation.CLASSIFY: 40,  # 40 per minute
            # Default limit for other operations
        }
        self.default_limit = 25
    
    def check_rate_limit(self, user_id: str, operation: AIOperation) -> Either[SecurityError, None]:
        """Check if user is within rate limits."""
        try:
            now = datetime.now(UTC)
            cutoff = now - timedelta(minutes=1)
            
            # Clean old requests
            if user_id in self.request_history:
                self.request_history[user_id] = [
                    req_time for req_time in self.request_history[user_id]
                    if req_time > cutoff
                ]
            else:
                self.request_history[user_id] = []
            
            # Check limit
            limit = self.limits.get(operation, self.default_limit)
            current_count = len(self.request_history[user_id])
            
            if current_count >= limit:
                return Either.left(SecurityError(
                    "rate_limit_exceeded",
                    f"Rate limit exceeded: {current_count}/{limit} requests per minute"
                ))
            
            # Record this request
            self.request_history[user_id].append(now)
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(SecurityError("rate_limit_check_failed", str(e)))
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiting status."""
        active_users = len(self.request_history)
        total_requests = sum(len(history) for history in self.request_history.values())
        
        return {
            "active_users": active_users,
            "total_recent_requests": total_requests,
            "limits": {op.value: limit for op, limit in self.limits.items()},
            "default_limit": self.default_limit
        }