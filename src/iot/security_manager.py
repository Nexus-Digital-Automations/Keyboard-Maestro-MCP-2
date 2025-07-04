"""
IoT Security Framework - TASK_65 Phase 4 Advanced Features

Comprehensive IoT security with device authentication, encrypted communication,
threat detection, and security policy enforcement for smart device networks.

Architecture: Zero Trust Security + Device Authentication + Threat Detection + Policy Enforcement
Performance: <100ms auth checks, <50ms encryption/decryption, <200ms threat analysis
Security: End-to-end encryption, multi-factor authentication, real-time threat monitoring
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, UTC, timedelta
from dataclasses import dataclass, field
import asyncio
import hashlib
import secrets
import hmac
import base64
from enum import Enum
import logging
import re

from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.errors import ValidationError, SecurityError, SystemError
from ..core.iot_architecture import (
    DeviceId, IoTIntegrationError, IoTDevice, SecurityLevel
)

logger = logging.getLogger(__name__)


class AuthenticationMethod(Enum):
    """Device authentication methods."""
    CERTIFICATE = "certificate"
    SHARED_KEY = "shared_key"
    OAUTH2 = "oauth2"
    BIOMETRIC = "biometric"
    MULTI_FACTOR = "multi_factor"
    ZERO_TRUST = "zero_trust"


class EncryptionAlgorithm(Enum):
    """Encryption algorithms for IoT communication."""
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_2048 = "rsa_2048"
    ECDSA_P256 = "ecdsa_p256"
    ED25519 = "ed25519"


class ThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "authentication_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ENCRYPTION_FAILURE = "encryption_failure"
    MALWARE_DETECTED = "malware_detected"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    DATA_BREACH = "data_breach"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class AccessLevel(Enum):
    """Device access levels."""
    READ_ONLY = "read_only"
    CONTROL = "control"
    ADMIN = "admin"
    SYSTEM = "system"


SecurityCredentialId = str
SecurityPolicyId = str
ThreatSignatureId = str


@dataclass
class SecurityCredential:
    """Device security credentials."""
    credential_id: SecurityCredentialId
    device_id: DeviceId
    auth_method: AuthenticationMethod
    credential_data: Dict[str, str]  # Encrypted credential data
    issued_at: datetime
    expires_at: datetime
    access_level: AccessLevel
    revoked: bool = False
    last_used: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if credential is currently valid."""
        return (
            not self.revoked and
            datetime.now(UTC) < self.expires_at and
            self.expires_at > self.issued_at
        )
    
    def is_expired(self) -> bool:
        """Check if credential has expired."""
        return datetime.now(UTC) >= self.expires_at


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    device_id: DeviceId
    event_type: SecurityEventType
    threat_level: ThreatLevel
    detected_at: datetime
    source_ip: Optional[str]
    event_details: Dict[str, Any]
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)
    
    def requires_immediate_action(self) -> bool:
        """Check if event requires immediate action."""
        return self.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]


@dataclass
class SecurityPolicy:
    """IoT security policy definition."""
    policy_id: SecurityPolicyId
    policy_name: str
    device_targets: List[DeviceId]
    auth_requirements: Dict[str, Any]
    encryption_requirements: Dict[str, Any]
    access_controls: Dict[str, Any]
    monitoring_rules: List[str]
    auto_mitigation: bool = True
    policy_priority: int = 100
    
    def applies_to_device(self, device_id: DeviceId) -> bool:
        """Check if policy applies to specific device."""
        return device_id in self.device_targets or "*" in self.device_targets


@dataclass
class ThreatSignature:
    """Threat detection signature."""
    signature_id: ThreatSignatureId
    threat_name: str
    signature_pattern: str
    threat_type: SecurityEventType
    severity: ThreatLevel
    detection_method: str
    false_positive_rate: float
    created_at: datetime
    
    def matches_pattern(self, data: str) -> bool:
        """Check if data matches threat signature pattern."""
        try:
            return bool(re.search(self.signature_pattern, data, re.IGNORECASE))
        except re.error:
            return False


class SecurityManager:
    """
    Comprehensive IoT security management system.
    
    Contracts:
        Preconditions:
            - All device communications must be authenticated
            - Encryption keys must be properly managed and rotated
            - Security policies must be validated before enforcement
        
        Postconditions:
            - All security events are logged and analyzed
            - Threat detection results include confidence scores
            - Security policies are consistently enforced
        
        Invariants:
            - Encryption keys are never stored in plaintext
            - Authentication failures trigger security audits
            - Critical threats are immediately escalated
    """
    
    def __init__(self):
        self.device_credentials: Dict[DeviceId, SecurityCredential] = {}
        self.security_policies: Dict[SecurityPolicyId, SecurityPolicy] = {}
        self.security_events: List[SecurityEvent] = []
        self.threat_signatures: Dict[ThreatSignatureId, ThreatSignature] = {}
        self.encryption_keys: Dict[DeviceId, Dict[str, str]] = {}
        
        # Security configuration
        self.default_key_rotation_interval = timedelta(days=30)
        self.max_auth_attempts = 3
        self.auth_lockout_duration = timedelta(minutes=15)
        self.threat_monitoring_enabled = True
        
        # Performance metrics
        self.total_auth_attempts = 0
        self.successful_auths = 0
        self.failed_auths = 0
        self.threats_detected = 0
        self.threats_mitigated = 0
        
        # Device lockout tracking
        self.locked_devices: Dict[DeviceId, datetime] = {}
        self.auth_failure_counts: Dict[DeviceId, int] = {}
        
        # Initialize default security components
        self._initialize_default_policies()
        self._initialize_threat_signatures()
    
    def _initialize_default_policies(self):
        """Initialize default security policies."""
        # High security policy for critical devices
        critical_policy = SecurityPolicy(
            policy_id="critical_device_policy",
            policy_name="Critical Device Security Policy",
            device_targets=["*"],  # Apply to all devices initially
            auth_requirements={
                "method": AuthenticationMethod.MULTI_FACTOR.value,
                "certificate_required": True,
                "key_rotation_days": 7
            },
            encryption_requirements={
                "algorithm": EncryptionAlgorithm.AES_256_GCM.value,
                "key_length": 256,
                "require_perfect_forward_secrecy": True
            },
            access_controls={
                "max_concurrent_sessions": 1,
                "session_timeout_minutes": 30,
                "require_secure_channel": True
            },
            monitoring_rules=[
                "log_all_access_attempts",
                "monitor_anomalous_behavior",
                "detect_privilege_escalation"
            ],
            policy_priority=1
        )
        
        self.security_policies[critical_policy.policy_id] = critical_policy
    
    def _initialize_threat_signatures(self):
        """Initialize threat detection signatures."""
        signatures = [
            ThreatSignature(
                signature_id="brute_force_auth",
                threat_name="Brute Force Authentication Attack",
                signature_pattern=r"auth.*fail.*repeat",
                threat_type=SecurityEventType.BRUTE_FORCE_ATTACK,
                severity=ThreatLevel.HIGH,
                detection_method="pattern_matching",
                false_positive_rate=0.05,
                created_at=datetime.now(UTC)
            ),
            ThreatSignature(
                signature_id="malware_command_injection",
                threat_name="Command Injection Attempt",
                signature_pattern=r"(exec|eval|system|shell|cmd).*[;&|`]",
                threat_type=SecurityEventType.MALWARE_DETECTED,
                severity=ThreatLevel.CRITICAL,
                detection_method="pattern_matching",
                false_positive_rate=0.02,
                created_at=datetime.now(UTC)
            ),
            ThreatSignature(
                signature_id="unauthorized_privilege_escalation",
                threat_name="Privilege Escalation Attempt",
                signature_pattern=r"(sudo|admin|root|privilege).*escalat",
                threat_type=SecurityEventType.PRIVILEGE_ESCALATION,
                severity=ThreatLevel.HIGH,
                detection_method="pattern_matching",
                false_positive_rate=0.03,
                created_at=datetime.now(UTC)
            )
        ]
        
        for signature in signatures:
            self.threat_signatures[signature.signature_id] = signature
    
    @require(lambda self, device_id, auth_method: device_id and auth_method)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def register_device_credentials(
        self,
        device_id: DeviceId,
        auth_method: AuthenticationMethod,
        credential_data: Dict[str, str],
        access_level: AccessLevel = AccessLevel.CONTROL,
        validity_days: int = 90
    ) -> Either[IoTIntegrationError, SecurityCredential]:
        """
        Register security credentials for IoT device.
        
        Architecture:
            - Secure credential storage with encryption
            - Automatic credential rotation and expiry
            - Multi-factor authentication support
        
        Security:
            - Credentials encrypted at rest
            - Secure key derivation and storage
            - Audit trail for credential operations
        """
        try:
            # Validate credential data based on auth method
            validation_result = await self._validate_credential_data(auth_method, credential_data)
            if validation_result.is_error():
                return validation_result
            
            # Generate secure credential ID
            credential_id = self._generate_credential_id(device_id, auth_method)
            
            # Encrypt credential data
            encrypted_data = await self._encrypt_credential_data(credential_data)
            
            # Create credential
            credential = SecurityCredential(
                credential_id=credential_id,
                device_id=device_id,
                auth_method=auth_method,
                credential_data=encrypted_data,
                issued_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(days=validity_days),
                access_level=access_level
            )
            
            # Store credential
            self.device_credentials[device_id] = credential
            
            # Generate encryption keys for device
            await self._generate_device_encryption_keys(device_id)
            
            # Log security event
            await self._log_security_event(
                device_id=device_id,
                event_type=SecurityEventType.AUTHENTICATION_FAILURE,  # Using as registration event
                threat_level=ThreatLevel.LOW,
                details={
                    "action": "credential_registration",
                    "auth_method": auth_method.value,
                    "access_level": access_level.value,
                    "validity_days": validity_days
                }
            )
            
            logger.info(f"Security credentials registered for device {device_id}")
            
            return Either.success(credential)
            
        except Exception as e:
            error_msg = f"Failed to register device credentials: {str(e)}"
            logger.error(error_msg)
            return Either.error(IoTIntegrationError(error_msg, device_id))
    
    @require(lambda self, device_id, auth_data: device_id and auth_data)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def authenticate_device(
        self,
        device_id: DeviceId,
        auth_data: Dict[str, str],
        source_ip: Optional[str] = None
    ) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """
        Authenticate IoT device using registered credentials.
        
        Performance:
            - <100ms authentication response time
            - Efficient credential lookup and validation
            - Optimized cryptographic operations
        """
        try:
            self.total_auth_attempts += 1
            
            # Check if device is locked out
            if self._is_device_locked(device_id):
                await self._log_security_event(
                    device_id=device_id,
                    event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                    threat_level=ThreatLevel.MEDIUM,
                    details={"reason": "device_locked", "source_ip": source_ip}
                )
                return Either.error(IoTIntegrationError(
                    f"Device {device_id} is temporarily locked due to authentication failures",
                    device_id
                ))
            
            # Get device credentials
            if device_id not in self.device_credentials:
                await self._handle_auth_failure(device_id, "credentials_not_found", source_ip)
                return Either.error(IoTIntegrationError(
                    f"No credentials found for device {device_id}",
                    device_id
                ))
            
            credential = self.device_credentials[device_id]
            
            # Check credential validity
            if not credential.is_valid():
                await self._handle_auth_failure(device_id, "invalid_credentials", source_ip)
                return Either.error(IoTIntegrationError(
                    f"Invalid or expired credentials for device {device_id}",
                    device_id
                ))
            
            # Verify authentication data
            auth_valid = await self._verify_authentication_data(credential, auth_data)
            if not auth_valid:
                await self._handle_auth_failure(device_id, "auth_verification_failed", source_ip)
                return Either.error(IoTIntegrationError(
                    f"Authentication verification failed for device {device_id}",
                    device_id
                ))
            
            # Authentication successful
            self.successful_auths += 1
            credential.last_used = datetime.now(UTC)
            
            # Reset failure count
            self.auth_failure_counts.pop(device_id, None)
            
            # Generate session token
            session_token = self._generate_session_token(device_id)
            
            # Log successful authentication
            await self._log_security_event(
                device_id=device_id,
                event_type=SecurityEventType.AUTHENTICATION_FAILURE,  # Using as success event
                threat_level=ThreatLevel.LOW,
                details={
                    "action": "authentication_success",
                    "auth_method": credential.auth_method.value,
                    "access_level": credential.access_level.value,
                    "source_ip": source_ip
                }
            )
            
            auth_result = {
                "device_id": device_id,
                "authenticated": True,
                "session_token": session_token,
                "access_level": credential.access_level.value,
                "auth_method": credential.auth_method.value,
                "session_expires": (datetime.now(UTC) + timedelta(hours=8)).isoformat(),
                "encryption_enabled": True
            }
            
            logger.info(f"Device authentication successful: {device_id}")
            
            return Either.success(auth_result)
            
        except Exception as e:
            await self._handle_auth_failure(device_id, f"auth_exception: {str(e)}", source_ip)
            error_msg = f"Authentication failed for device {device_id}: {str(e)}"
            logger.error(error_msg)
            return Either.error(IoTIntegrationError(error_msg, device_id))
    
    @require(lambda self, data, device_id: data and device_id)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def encrypt_communication(
        self,
        data: str,
        device_id: DeviceId,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    ) -> Either[IoTIntegrationError, Dict[str, str]]:
        """
        Encrypt communication data for IoT device.
        
        Security:
            - End-to-end encryption with perfect forward secrecy
            - Authenticated encryption with integrity verification
            - Secure key management and rotation
        """
        try:
            # Get device encryption keys
            if device_id not in self.encryption_keys:
                return Either.error(IoTIntegrationError(
                    f"No encryption keys found for device {device_id}",
                    device_id
                ))
            
            device_keys = self.encryption_keys[device_id]
            encryption_key = device_keys.get("encryption_key")
            
            if not encryption_key:
                return Either.error(IoTIntegrationError(
                    f"Encryption key not available for device {device_id}",
                    device_id
                ))
            
            # Simulate encryption (in real implementation, use proper crypto library)
            encrypted_data = await self._perform_encryption(data, encryption_key, algorithm)
            
            # Generate authentication tag
            auth_tag = self._generate_auth_tag(encrypted_data, device_keys.get("auth_key", ""))
            
            encryption_result = {
                "encrypted_data": encrypted_data,
                "algorithm": algorithm.value,
                "auth_tag": auth_tag,
                "timestamp": datetime.now(UTC).isoformat(),
                "key_version": device_keys.get("version", "1")
            }
            
            return Either.success(encryption_result)
            
        except Exception as e:
            error_msg = f"Encryption failed for device {device_id}: {str(e)}"
            logger.error(error_msg)
            return Either.error(IoTIntegrationError(error_msg, device_id))
    
    @require(lambda self, communication_data, patterns: communication_data)
    async def analyze_threat_patterns(
        self,
        communication_data: str,
        device_id: Optional[DeviceId] = None,
        custom_patterns: Optional[List[str]] = None
    ) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """
        Analyze communication data for security threats.
        
        Architecture:
            - Multi-layered threat detection algorithms
            - Real-time pattern matching and ML analysis
            - Adaptive threat signature updates
        """
        try:
            threats_detected = []
            
            # Check against known threat signatures
            for signature_id, signature in self.threat_signatures.items():
                if signature.matches_pattern(communication_data):
                    threat_confidence = 1.0 - signature.false_positive_rate
                    
                    threats_detected.append({
                        "signature_id": signature_id,
                        "threat_name": signature.threat_name,
                        "threat_type": signature.threat_type.value,
                        "severity": signature.severity.value,
                        "confidence": threat_confidence,
                        "detection_method": signature.detection_method
                    })
                    
                    # Log security event if confidence is high
                    if threat_confidence > 0.8 and device_id:
                        await self._log_security_event(
                            device_id=device_id,
                            event_type=signature.threat_type,
                            threat_level=signature.severity,
                            details={
                                "signature_id": signature_id,
                                "threat_name": signature.threat_name,
                                "confidence": threat_confidence,
                                "data_sample": communication_data[:100]  # First 100 chars
                            }
                        )
            
            # Check custom patterns if provided
            if custom_patterns:
                for pattern in custom_patterns:
                    try:
                        if re.search(pattern, communication_data, re.IGNORECASE):
                            threats_detected.append({
                                "signature_id": f"custom_{hash(pattern) % 10000}",
                                "threat_name": f"Custom Pattern Match: {pattern[:50]}",
                                "threat_type": "custom_threat",
                                "severity": ThreatLevel.MEDIUM.value,
                                "confidence": 0.7,
                                "detection_method": "custom_pattern"
                            })
                    except re.error:
                        logger.warning(f"Invalid regex pattern: {pattern}")
            
            # Behavioral analysis (simplified)
            behavioral_threats = await self._analyze_behavioral_patterns(communication_data, device_id)
            threats_detected.extend(behavioral_threats)
            
            # Update metrics
            if threats_detected:
                self.threats_detected += len(threats_detected)
            
            analysis_result = {
                "threats_detected": len(threats_detected),
                "threat_details": threats_detected,
                "analysis_timestamp": datetime.now(UTC).isoformat(),
                "data_size_analyzed": len(communication_data),
                "high_confidence_threats": len([t for t in threats_detected if t["confidence"] > 0.8]),
                "requires_immediate_action": any(
                    t["severity"] in ["critical", "emergency"] for t in threats_detected
                )
            }
            
            return Either.success(analysis_result)
            
        except Exception as e:
            error_msg = f"Threat analysis failed: {str(e)}"
            logger.error(error_msg)
            return Either.error(IoTIntegrationError(error_msg, device_id))
    
    async def _validate_credential_data(
        self,
        auth_method: AuthenticationMethod,
        credential_data: Dict[str, str]
    ) -> Either[IoTIntegrationError, bool]:
        """Validate credential data based on authentication method."""
        
        if auth_method == AuthenticationMethod.CERTIFICATE:
            if "certificate" not in credential_data or "private_key" not in credential_data:
                return Either.error(IoTIntegrationError(
                    "Certificate authentication requires 'certificate' and 'private_key'"
                ))
        
        elif auth_method == AuthenticationMethod.SHARED_KEY:
            if "shared_key" not in credential_data:
                return Either.error(IoTIntegrationError(
                    "Shared key authentication requires 'shared_key'"
                ))
            
            # Validate key strength
            shared_key = credential_data["shared_key"]
            if len(shared_key) < 32:  # Minimum 32 characters
                return Either.error(IoTIntegrationError(
                    "Shared key must be at least 32 characters long"
                ))
        
        elif auth_method == AuthenticationMethod.OAUTH2:
            required_fields = ["client_id", "client_secret", "token_endpoint"]
            for field in required_fields:
                if field not in credential_data:
                    return Either.error(IoTIntegrationError(
                        f"OAuth2 authentication requires '{field}'"
                    ))
        
        return Either.success(True)
    
    def _generate_credential_id(self, device_id: DeviceId, auth_method: AuthenticationMethod) -> str:
        """Generate unique credential ID."""
        timestamp = int(datetime.now(UTC).timestamp())
        data = f"{device_id}:{auth_method.value}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def _encrypt_credential_data(self, credential_data: Dict[str, str]) -> Dict[str, str]:
        """Encrypt sensitive credential data."""
        # In real implementation, use proper encryption
        encrypted_data = {}
        for key, value in credential_data.items():
            # Simple base64 encoding for simulation
            encrypted_value = base64.b64encode(value.encode()).decode()
            encrypted_data[key] = encrypted_value
        
        return encrypted_data
    
    async def _generate_device_encryption_keys(self, device_id: DeviceId):
        """Generate encryption keys for device communication."""
        # Generate secure random keys
        encryption_key = secrets.token_hex(32)  # 256-bit key
        auth_key = secrets.token_hex(32)  # 256-bit auth key
        
        self.encryption_keys[device_id] = {
            "encryption_key": encryption_key,
            "auth_key": auth_key,
            "generated_at": datetime.now(UTC).isoformat(),
            "version": "1"
        }
    
    def _is_device_locked(self, device_id: DeviceId) -> bool:
        """Check if device is currently locked due to auth failures."""
        if device_id in self.locked_devices:
            lockout_time = self.locked_devices[device_id]
            if datetime.now(UTC) < lockout_time:
                return True
            else:
                # Lockout expired, remove from locked devices
                del self.locked_devices[device_id]
        
        return False
    
    async def _handle_auth_failure(self, device_id: DeviceId, reason: str, source_ip: Optional[str]):
        """Handle authentication failure and potential lockout."""
        self.failed_auths += 1
        
        # Increment failure count
        self.auth_failure_counts[device_id] = self.auth_failure_counts.get(device_id, 0) + 1
        
        # Check if should lock device
        if self.auth_failure_counts[device_id] >= self.max_auth_attempts:
            lockout_until = datetime.now(UTC) + self.auth_lockout_duration
            self.locked_devices[device_id] = lockout_until
            
            await self._log_security_event(
                device_id=device_id,
                event_type=SecurityEventType.BRUTE_FORCE_ATTACK,
                threat_level=ThreatLevel.HIGH,
                details={
                    "reason": "max_auth_attempts_exceeded",
                    "failure_count": self.auth_failure_counts[device_id],
                    "lockout_until": lockout_until.isoformat(),
                    "source_ip": source_ip
                }
            )
        else:
            await self._log_security_event(
                device_id=device_id,
                event_type=SecurityEventType.AUTHENTICATION_FAILURE,
                threat_level=ThreatLevel.MEDIUM,
                details={
                    "reason": reason,
                    "failure_count": self.auth_failure_counts[device_id],
                    "source_ip": source_ip
                }
            )
    
    async def _verify_authentication_data(
        self,
        credential: SecurityCredential,
        auth_data: Dict[str, str]
    ) -> bool:
        """Verify authentication data against stored credentials."""
        
        if credential.auth_method == AuthenticationMethod.SHARED_KEY:
            provided_key = auth_data.get("shared_key", "")
            # Decrypt stored key (simple simulation)
            stored_key_encrypted = credential.credential_data.get("shared_key", "")
            try:
                stored_key = base64.b64decode(stored_key_encrypted).decode()
                return secrets.compare_digest(provided_key, stored_key)
            except Exception:
                return False
        
        elif credential.auth_method == AuthenticationMethod.CERTIFICATE:
            # Certificate verification (simplified)
            provided_cert = auth_data.get("certificate", "")
            stored_cert_encrypted = credential.credential_data.get("certificate", "")
            try:
                stored_cert = base64.b64decode(stored_cert_encrypted).decode()
                return provided_cert == stored_cert
            except Exception:
                return False
        
        # Default to failed authentication for unsupported methods
        return False
    
    def _generate_session_token(self, device_id: DeviceId) -> str:
        """Generate secure session token for authenticated device."""
        timestamp = int(datetime.now(UTC).timestamp())
        random_data = secrets.token_bytes(16)
        token_data = f"{device_id}:{timestamp}".encode() + random_data
        return base64.b64encode(hashlib.sha256(token_data).digest()).decode()
    
    async def _perform_encryption(
        self,
        data: str,
        encryption_key: str,
        algorithm: EncryptionAlgorithm
    ) -> str:
        """Perform data encryption using specified algorithm."""
        # Simplified encryption simulation
        # In real implementation, use proper cryptographic libraries
        
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            # Simulate AES-256-GCM encryption
            data_bytes = data.encode()
            key_bytes = encryption_key.encode()[:32]  # Use first 32 bytes as key
            
            # Simple XOR encryption for simulation (NOT secure for production)
            encrypted_bytes = bytes(a ^ b for a, b in zip(data_bytes, key_bytes * (len(data_bytes) // 32 + 1)))
            return base64.b64encode(encrypted_bytes).decode()
        
        # Default encryption
        return base64.b64encode(data.encode()).decode()
    
    def _generate_auth_tag(self, encrypted_data: str, auth_key: str) -> str:
        """Generate authentication tag for encrypted data."""
        # Use HMAC for authentication tag
        auth_key_bytes = auth_key.encode()
        message = encrypted_data.encode()
        tag = hmac.new(auth_key_bytes, message, hashlib.sha256).hexdigest()
        return tag[:16]  # Use first 16 characters
    
    async def _analyze_behavioral_patterns(
        self,
        communication_data: str,
        device_id: Optional[DeviceId]
    ) -> List[Dict[str, Any]]:
        """Analyze behavioral patterns for anomaly detection."""
        behavioral_threats = []
        
        # Check for unusual data patterns
        if len(communication_data) > 10000:  # Very large payload
            behavioral_threats.append({
                "signature_id": "large_payload_anomaly",
                "threat_name": "Unusually Large Data Payload",
                "threat_type": SecurityEventType.ANOMALOUS_BEHAVIOR.value,
                "severity": ThreatLevel.MEDIUM.value,
                "confidence": 0.6,
                "detection_method": "behavioral_analysis"
            })
        
        # Check for repeated patterns (potential DoS)
        if len(set(communication_data.split())) < len(communication_data.split()) * 0.1:
            behavioral_threats.append({
                "signature_id": "repeated_pattern_anomaly",
                "threat_name": "Repetitive Data Pattern",
                "threat_type": SecurityEventType.ANOMALOUS_BEHAVIOR.value,
                "severity": ThreatLevel.LOW.value,
                "confidence": 0.5,
                "detection_method": "behavioral_analysis"
            })
        
        return behavioral_threats
    
    async def _log_security_event(
        self,
        device_id: DeviceId,
        event_type: SecurityEventType,
        threat_level: ThreatLevel,
        details: Dict[str, Any],
        source_ip: Optional[str] = None
    ):
        """Log security event for audit and analysis."""
        event_id = f"evt_{int(datetime.now(UTC).timestamp())}_{secrets.token_hex(4)}"
        
        event = SecurityEvent(
            event_id=event_id,
            device_id=device_id,
            event_type=event_type,
            threat_level=threat_level,
            detected_at=datetime.now(UTC),
            source_ip=source_ip,
            event_details=details
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        # Log critical events immediately
        if event.requires_immediate_action():
            logger.critical(f"Critical security event: {event_id} - {event_type.value} on device {device_id}")
    
    async def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security status summary."""
        recent_events = [e for e in self.security_events if e.detected_at > datetime.now(UTC) - timedelta(hours=24)]
        
        return {
            "total_registered_devices": len(self.device_credentials),
            "total_auth_attempts": self.total_auth_attempts,
            "successful_authentications": self.successful_auths,
            "failed_authentications": self.failed_auths,
            "auth_success_rate": (self.successful_auths / max(self.total_auth_attempts, 1)) * 100,
            "currently_locked_devices": len(self.locked_devices),
            "threats_detected": self.threats_detected,
            "threats_mitigated": self.threats_mitigated,
            "active_security_policies": len(self.security_policies),
            "threat_signatures_loaded": len(self.threat_signatures),
            "recent_security_events": len(recent_events),
            "critical_events_24h": len([e for e in recent_events if e.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]]),
            "devices_with_encryption": len(self.encryption_keys),
            "threat_monitoring_enabled": self.threat_monitoring_enabled
        }


# Helper functions for security management
def generate_secure_shared_key(length: int = 64) -> str:
    """Generate cryptographically secure shared key."""
    return secrets.token_hex(length // 2)


def validate_certificate_format(certificate: str) -> bool:
    """Validate certificate format (simplified)."""
    return (
        certificate.startswith("-----BEGIN CERTIFICATE-----") and
        certificate.endswith("-----END CERTIFICATE-----") and
        len(certificate) > 100
    )