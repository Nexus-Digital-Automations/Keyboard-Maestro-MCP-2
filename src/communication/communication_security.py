"""
Comprehensive security validation and anti-spam protection for communication.

This module provides enterprise-grade security for all communication channels
with threat detection, rate limiting, and comprehensive validation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
import re
import time
import hashlib
from datetime import datetime, timedelta, UTC
from collections import defaultdict, deque

from ..core.communication import (
    CommunicationRequest, CommunicationType, EmailAddress, PhoneNumber
)
from ..core.either import Either
from ..core.errors import SecurityError, RateLimitError, ValidationError
from ..core.contracts import require, ensure


@dataclass(frozen=True)
class SecurityConfiguration:
    """Security configuration for communication validation."""
    # Rate limiting
    max_emails_per_hour: int = 50
    max_sms_per_hour: int = 20
    max_recipients_per_message: int = 100
    
    # Content security
    max_message_length: int = 10000
    max_subject_length: int = 998
    max_attachment_size_mb: int = 25
    max_attachments_per_email: int = 10
    
    # Spam detection
    spam_score_threshold: float = 7.0
    enable_content_analysis: bool = True
    enable_reputation_checking: bool = True
    
    # Security patterns
    blocked_domains: Set[str] = field(default_factory=set)
    allowed_domains: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.blocked_domains:
            # Default blocked domains
            blocked = {
                'tempmail.org', '10minutemail.com', 'guerrillamail.com',
                'mailinator.com', 'dispostable.com', 'throwaway.email'
            }
            object.__setattr__(self, 'blocked_domains', blocked)


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_type: str
    severity: str  # low, medium, high, critical
    description: str
    evidence: List[str]
    detection_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def get_risk_score(self) -> float:
        """Calculate numeric risk score based on severity."""
        severity_scores = {
            'low': 1.0,
            'medium': 3.0,
            'high': 7.0,
            'critical': 10.0
        }
        return severity_scores.get(self.severity, 5.0)


class RateLimitTracker:
    """Advanced rate limiting with per-user and global tracking."""
    
    def __init__(self, config: SecurityConfiguration):
        self.config = config
        # Track by user/sender
        self.user_history: Dict[str, deque] = defaultdict(lambda: deque())
        # Track by recipient
        self.recipient_history: Dict[str, deque] = defaultdict(lambda: deque())
        # Global tracking
        self.global_history: deque = deque()
    
    def can_send_communication(self, request: CommunicationRequest, sender_id: str) -> Either[RateLimitError, None]:
        """Check if communication can be sent based on rate limits."""
        current_time = time.time()
        
        # Clean old entries
        self._clean_old_entries(current_time)
        
        # Check sender rate limits
        sender_check = self._check_sender_limits(sender_id, request.communication_type, current_time)
        if sender_check.is_left():
            return sender_check
        
        # Check recipient rate limits
        recipient_check = self._check_recipient_limits(request.recipients, current_time)
        if recipient_check.is_left():
            return recipient_check
        
        # Check global rate limits
        global_check = self._check_global_limits(request.communication_type, current_time)
        if global_check.is_left():
            return global_check
        
        return Either.right(None)
    
    def record_communication_sent(self, request: CommunicationRequest, sender_id: str):
        """Record that a communication was sent for rate limiting."""
        current_time = time.time()
        
        # Record for sender
        self.user_history[sender_id].append((current_time, request.communication_type))
        
        # Record for recipients
        for recipient in request.recipients:
            recipient_key = self._get_recipient_key(recipient)
            self.recipient_history[recipient_key].append(current_time)
        
        # Record globally
        self.global_history.append((current_time, request.communication_type))
    
    def _clean_old_entries(self, current_time: float):
        """Remove entries older than 1 hour."""
        cutoff_time = current_time - 3600  # 1 hour
        
        # Clean user history
        for sender_id in list(self.user_history.keys()):
            history = self.user_history[sender_id]
            while history and history[0][0] < cutoff_time:
                history.popleft()
            if not history:
                del self.user_history[sender_id]
        
        # Clean recipient history
        for recipient_key in list(self.recipient_history.keys()):
            history = self.recipient_history[recipient_key]
            while history and history[0] < cutoff_time:
                history.popleft()
            if not history:
                del self.recipient_history[recipient_key]
        
        # Clean global history
        while self.global_history and self.global_history[0][0] < cutoff_time:
            self.global_history.popleft()
    
    def _check_sender_limits(self, sender_id: str, comm_type: CommunicationType, current_time: float) -> Either[RateLimitError, None]:
        """Check rate limits for specific sender."""
        if sender_id not in self.user_history:
            return Either.right(None)
        
        history = self.user_history[sender_id]
        
        # Count messages by type in last hour
        email_count = sum(1 for _, msg_type in history if msg_type == CommunicationType.EMAIL)
        sms_count = sum(1 for _, msg_type in history if msg_type in [CommunicationType.SMS, CommunicationType.IMESSAGE])
        
        # Check limits
        if comm_type == CommunicationType.EMAIL and email_count >= self.config.max_emails_per_hour:
            return Either.left(RateLimitError(
                f"Email rate limit exceeded for sender: {email_count}/{self.config.max_emails_per_hour}"
            ))
        
        if comm_type in [CommunicationType.SMS, CommunicationType.IMESSAGE] and sms_count >= self.config.max_sms_per_hour:
            return Either.left(RateLimitError(
                f"SMS rate limit exceeded for sender: {sms_count}/{self.config.max_sms_per_hour}"
            ))
        
        return Either.right(None)
    
    def _check_recipient_limits(self, recipients: List, current_time: float) -> Either[RateLimitError, None]:
        """Check rate limits for recipients."""
        # Limit messages per recipient (anti-harassment)
        max_per_recipient = 10
        
        for recipient in recipients:
            recipient_key = self._get_recipient_key(recipient)
            if recipient_key in self.recipient_history:
                count = len(self.recipient_history[recipient_key])
                if count >= max_per_recipient:
                    return Either.left(RateLimitError(
                        f"Recipient rate limit exceeded: {recipient_key}"
                    ))
        
        return Either.right(None)
    
    def _check_global_limits(self, comm_type: CommunicationType, current_time: float) -> Either[RateLimitError, None]:
        """Check global system rate limits."""
        max_global_per_hour = 1000  # System-wide limit
        
        total_count = len(self.global_history)
        if total_count >= max_global_per_hour:
            return Either.left(RateLimitError(
                f"Global rate limit exceeded: {total_count}/{max_global_per_hour}"
            ))
        
        return Either.right(None)
    
    def _get_recipient_key(self, recipient) -> str:
        """Get standardized key for recipient."""
        if isinstance(recipient, EmailAddress):
            return f"email:{recipient.address.lower()}"
        elif isinstance(recipient, PhoneNumber):
            return f"phone:{recipient.format_for_sms()}"
        else:
            return f"unknown:{str(recipient)}"


class SpamDetector:
    """Advanced spam detection with multiple analysis methods."""
    
    def __init__(self, config: SecurityConfiguration):
        self.config = config
        self._load_spam_patterns()
    
    def _load_spam_patterns(self):
        """Load spam detection patterns."""
        self.spam_keywords = {
            # Financial scams
            'make money fast', 'easy money', 'guaranteed income', 'work from home',
            'financial freedom', 'cash advance', 'credit repair', 'debt consolidation',
            
            # Pharmaceutical spam
            'viagra', 'cialis', 'weight loss', 'diet pills', 'prescription drugs',
            
            # Phishing
            'verify account', 'urgent action required', 'suspended account',
            'click here now', 'limited time offer', 'act now',
            
            # Lottery/Prize scams
            'congratulations winner', 'claim your prize', 'lottery winner',
            'inherited money', 'tax refund',
            
            # General spam indicators
            'no obligation', 'risk free', 'money back guarantee',
            'call now', 'order now', 'buy now'
        }
        
        self.spam_patterns = [
            r'\b(free|win|won|winner)\b.*\b(money|cash|prize|gift)\b',
            r'\b(urgent|immediate|limited)\b.*\b(action|offer|time)\b',
            r'\b(guarantee|guaranteed)\b.*\b(income|money|profit)\b',
            r'\b(click|call|order)\b.*\b(now|today|immediately)\b',
            r'\$\d+.*\b(per|each|every)\b.*\b(day|week|month)\b',
            r'\b(viagra|cialis|pharmacy|pills|medication)\b',
        ]
    
    def analyze_content(self, subject: Optional[str], body: str) -> Tuple[float, List[SecurityThreat]]:
        """Analyze content for spam indicators and return score and threats."""
        threats = []
        total_score = 0.0
        
        combined_text = f"{subject or ''} {body}".lower()
        
        # Keyword analysis
        keyword_score, keyword_threats = self._analyze_keywords(combined_text)
        total_score += keyword_score
        threats.extend(keyword_threats)
        
        # Pattern analysis
        pattern_score, pattern_threats = self._analyze_patterns(combined_text)
        total_score += pattern_score
        threats.extend(pattern_threats)
        
        # Structure analysis
        structure_score, structure_threats = self._analyze_structure(subject, body)
        total_score += structure_score
        threats.extend(structure_threats)
        
        # URL analysis
        url_score, url_threats = self._analyze_urls(combined_text)
        total_score += url_score
        threats.extend(url_threats)
        
        return total_score, threats
    
    def _analyze_keywords(self, text: str) -> Tuple[float, List[SecurityThreat]]:
        """Analyze for spam keywords."""
        threats = []
        score = 0.0
        
        found_keywords = []
        for keyword in self.spam_keywords:
            if keyword in text:
                found_keywords.append(keyword)
                score += 1.0
        
        if found_keywords:
            threats.append(SecurityThreat(
                threat_type="spam_keywords",
                severity="medium" if len(found_keywords) < 3 else "high",
                description=f"Contains {len(found_keywords)} spam keywords",
                evidence=found_keywords[:5]  # Limit evidence
            ))
        
        return score, threats
    
    def _analyze_patterns(self, text: str) -> Tuple[float, List[SecurityThreat]]:
        """Analyze for spam patterns."""
        threats = []
        score = 0.0
        
        found_patterns = []
        for pattern in self.spam_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found_patterns.extend(matches[:3])  # Limit matches
                score += 2.0
        
        if found_patterns:
            threats.append(SecurityThreat(
                threat_type="spam_patterns",
                severity="high",
                description=f"Contains {len(found_patterns)} spam patterns",
                evidence=[str(p) for p in found_patterns]
            ))
        
        return score, threats
    
    def _analyze_structure(self, subject: Optional[str], body: str) -> Tuple[float, List[SecurityThreat]]:
        """Analyze message structure for spam indicators."""
        threats = []
        score = 0.0
        
        # Excessive capitalization
        if subject:
            caps_ratio = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
            if caps_ratio > 0.7:
                score += 2.0
                threats.append(SecurityThreat(
                    threat_type="excessive_caps",
                    severity="medium",
                    description=f"Subject line {caps_ratio:.1%} uppercase",
                    evidence=[subject[:50]]
                ))
        
        # Excessive exclamation marks
        exclamation_count = body.count('!')
        if exclamation_count > 5:
            score += 1.0
            threats.append(SecurityThreat(
                threat_type="excessive_punctuation",
                severity="low",
                description=f"Contains {exclamation_count} exclamation marks",
                evidence=[f"Exclamation count: {exclamation_count}"]
            ))
        
        # Very short or very long messages (unusual for legitimate communication)
        if len(body) < 10:
            score += 1.0
            threats.append(SecurityThreat(
                threat_type="suspicious_length",
                severity="low",
                description="Message body too short",
                evidence=[f"Length: {len(body)} chars"]
            ))
        elif len(body) > 5000:
            score += 0.5
        
        return score, threats
    
    def _analyze_urls(self, text: str) -> Tuple[float, List[SecurityThreat]]:
        """Analyze URLs for suspicious characteristics."""
        threats = []
        score = 0.0
        
        # Find URLs
        url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        urls = re.findall(url_pattern, text.lower())
        
        if not urls:
            return score, threats
        
        suspicious_urls = []
        
        for url in urls:
            # URL shorteners (can hide destination)
            if any(shortener in url for shortener in ['bit.ly', 'tinyurl', 'goo.gl', 't.co']):
                suspicious_urls.append(url)
                score += 1.5
            
            # IP addresses instead of domains
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url):
                suspicious_urls.append(url)
                score += 2.0
            
            # Suspicious domain patterns
            if re.search(r'[a-z0-9]+-[a-z0-9]+-[a-z0-9]+\.', url):
                suspicious_urls.append(url)
                score += 1.0
            
            # Excessive subdomain levels
            if url.count('.') > 4:
                suspicious_urls.append(url)
                score += 0.5
        
        if suspicious_urls:
            threats.append(SecurityThreat(
                threat_type="suspicious_urls",
                severity="high" if len(suspicious_urls) > 2 else "medium",
                description=f"Contains {len(suspicious_urls)} suspicious URLs",
                evidence=suspicious_urls[:3]
            ))
        
        return score, threats


class CommunicationSecurityManager:
    """Comprehensive security management for all communication."""
    
    def __init__(self, config: Optional[SecurityConfiguration] = None):
        self.config = config or SecurityConfiguration()
        self.rate_limiter = RateLimitTracker(self.config)
        self.spam_detector = SpamDetector(self.config)
        self.threat_log: List[SecurityThreat] = []
    
    @require(lambda self, request: isinstance(request, CommunicationRequest))
    @ensure(lambda self, result: isinstance(result, Either))
    def validate_communication_security(self, request: CommunicationRequest, sender_id: str) -> Either[SecurityError, Dict[str, Any]]:
        """Comprehensive security validation for communication request."""
        try:
            validation_results = {
                "threats_detected": [],
                "spam_score": 0.0,
                "security_level": "safe",
                "validation_time": datetime.now(UTC).isoformat()
            }
            
            # Rate limiting check
            rate_check = self.rate_limiter.can_send_communication(request, sender_id)
            if rate_check.is_left():
                return Either.left(SecurityError(
                    f"Rate limit violation: {rate_check.get_left().message}"
                ))
            
            # Content security analysis
            if self.config.enable_content_analysis:
                content_result = self._analyze_content_security(request)
                if content_result.is_left():
                    return content_result
                
                content_analysis = content_result.get_right()
                validation_results.update(content_analysis)
            
            # Recipient validation
            recipient_result = self._validate_recipients(request)
            if recipient_result.is_left():
                return recipient_result
            
            # Attachment security (for emails)
            if request.communication_type == CommunicationType.EMAIL and request.attachments:
                attachment_result = self._validate_attachments(request.attachments)
                if attachment_result.is_left():
                    return attachment_result
            
            # Determine overall security level
            spam_score = validation_results.get("spam_score", 0.0)
            threat_count = len(validation_results.get("threats_detected", []))
            
            if spam_score >= self.config.spam_score_threshold:
                return Either.left(SecurityError(
                    "SPAM_DETECTED", f"Content appears to be spam (score: {spam_score:.1f})"
                ))
            elif spam_score >= 5.0 or threat_count >= 3:
                validation_results["security_level"] = "suspicious"
            elif spam_score >= 3.0 or threat_count >= 1:
                validation_results["security_level"] = "caution"
            
            # Record successful validation
            self.rate_limiter.record_communication_sent(request, sender_id)
            
            return Either.right(validation_results)
            
        except Exception as e:
            return Either.left(SecurityError("VALIDATION_FAILED", f"Security validation failed: {str(e)}"))
    
    def _analyze_content_security(self, request: CommunicationRequest) -> Either[SecurityError, Dict[str, Any]]:
        """Analyze content for security threats."""
        try:
            # Spam analysis
            spam_score, threats = self.spam_detector.analyze_content(
                request.subject, request.message_content
            )
            
            # Log threats
            self.threat_log.extend(threats)
            
            # Security checks
            security_threats = []
            
            # Check for injection attempts
            combined_content = f"{request.subject or ''} {request.message_content}"
            if self._contains_injection_attempts(combined_content):
                injection_threat = SecurityThreat(
                    threat_type="injection_attempt",
                    severity="critical",
                    description="Content contains potential injection attempts",
                    evidence=["Injection patterns detected"]
                )
                security_threats.append(injection_threat)
                threats.append(injection_threat)
            
            # Check for social engineering
            if self._contains_social_engineering(combined_content):
                social_threat = SecurityThreat(
                    threat_type="social_engineering",
                    severity="high",
                    description="Content contains social engineering patterns",
                    evidence=["Social engineering indicators found"]
                )
                security_threats.append(social_threat)
                threats.append(social_threat)
            
            return Either.right({
                "spam_score": spam_score,
                "threats_detected": [self._threat_to_dict(t) for t in threats],
                "security_threats": len(security_threats)
            })
            
        except Exception as e:
            return Either.left(SecurityError(f"Content analysis failed: {str(e)}"))
    
    def _validate_recipients(self, request: CommunicationRequest) -> Either[SecurityError, None]:
        """Validate recipients for security concerns."""
        if len(request.recipients) > self.config.max_recipients_per_message:
            return Either.left(SecurityError(
                f"Too many recipients: {len(request.recipients)} > {self.config.max_recipients_per_message}"
            ))
        
        # Email-specific validation
        if request.communication_type == CommunicationType.EMAIL:
            for recipient in request.recipients:
                if isinstance(recipient, EmailAddress):
                    domain = recipient.domain.lower()
                    
                    # Check blocked domains
                    if domain in self.config.blocked_domains:
                        return Either.left(SecurityError(
                            f"Recipient domain is blocked: {domain}"
                        ))
                    
                    # Check allowed domains (if configured)
                    if self.config.allowed_domains and domain not in self.config.allowed_domains:
                        return Either.left(SecurityError(
                            f"Recipient domain not in allowed list: {domain}"
                        ))
        
        return Either.right(None)
    
    def _validate_attachments(self, attachments: List[str]) -> Either[SecurityError, None]:
        """Validate email attachments for security."""
        if len(attachments) > self.config.max_attachments_per_email:
            return Either.left(SecurityError(
                f"Too many attachments: {len(attachments)} > {self.config.max_attachments_per_email}"
            ))
        
        dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
            '.jar', '.app', '.deb', '.rpm', '.dmg', '.pkg'
        }
        
        for attachment_path in attachments:
            # Check file extension
            extension = attachment_path.lower().split('.')[-1] if '.' in attachment_path else ''
            if f'.{extension}' in dangerous_extensions:
                return Either.left(SecurityError(
                    f"Dangerous attachment type: {extension}"
                ))
        
        return Either.right(None)
    
    def _contains_injection_attempts(self, content: str) -> bool:
        """Check for various injection attempts."""
        injection_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'data:text/html',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'shell_exec\s*\(',
            r'\{\{.*\}\}',  # Template injection
            r'\$\{.*\}',    # Expression injection
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower, re.IGNORECASE) 
                  for pattern in injection_patterns)
    
    def _contains_social_engineering(self, content: str) -> bool:
        """Check for social engineering tactics."""
        social_patterns = [
            r'verify.*account.*immediately',
            r'suspended.*account.*click',
            r'urgent.*security.*alert',
            r'unauthorized.*access.*detected',
            r'confirm.*identity.*link',
            r'update.*payment.*information',
            r'action.*required.*within.*\d+.*hours?',
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in social_patterns)
    
    def _threat_to_dict(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Convert threat object to dictionary."""
        return {
            "type": threat.threat_type,
            "severity": threat.severity,
            "description": threat.description,
            "evidence": threat.evidence,
            "risk_score": threat.get_risk_score(),
            "detected_at": threat.detection_time.isoformat()
        }
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security statistics and threat summary."""
        current_time = datetime.now(UTC)
        recent_threats = [
            t for t in self.threat_log 
            if (current_time - t.detection_time).total_seconds() < 3600
        ]
        
        threat_counts = defaultdict(int)
        for threat in recent_threats:
            threat_counts[threat.threat_type] += 1
        
        return {
            "total_threats_logged": len(self.threat_log),
            "recent_threats_1h": len(recent_threats),
            "threat_types": dict(threat_counts),
            "rate_limit_status": self._get_rate_limit_summary(),
            "security_config": {
                "spam_threshold": self.config.spam_score_threshold,
                "max_emails_per_hour": self.config.max_emails_per_hour,
                "max_sms_per_hour": self.config.max_sms_per_hour,
                "content_analysis_enabled": self.config.enable_content_analysis
            }
        }
    
    def _get_rate_limit_summary(self) -> Dict[str, int]:
        """Get rate limiting summary."""
        return {
            "active_senders": len(self.rate_limiter.user_history),
            "active_recipients": len(self.rate_limiter.recipient_history),
            "global_messages_1h": len(self.rate_limiter.global_history)
        }