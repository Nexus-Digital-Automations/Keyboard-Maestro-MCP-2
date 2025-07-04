"""
Security Validation for Command Library

Provides comprehensive security validation utilities for all command types
with threat detection, input sanitization, and risk assessment.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Pattern
from dataclasses import dataclass, field
from enum import Enum
import re
import os
import os.path

from ..core.types import Permission
from ..core.errors import SecurityViolationError


class SecurityRiskLevel(Enum):
    """Security risk levels for commands."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats to detect."""
    SCRIPT_INJECTION = "script_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass(frozen=True)
class SecurityThreat:
    """Detected security threat."""
    threat_type: ThreatType
    severity: SecurityRiskLevel
    description: str
    mitigation: str
    field_name: Optional[str] = None


class CommandSecurityError(SecurityViolationError):
    """Security error specific to command validation."""
    
    def __init__(self, message: str, threats: List[SecurityThreat]):
        super().__init__(message)
        self.threats = threats


class SecurityValidator:
    """
    Comprehensive security validator for command parameters.
    
    Detects and prevents various types of security threats
    including injection attacks, path traversal, and privilege escalation.
    """
    
    # Security patterns for threat detection
    SCRIPT_INJECTION_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'vbscript:', re.IGNORECASE),
        re.compile(r'data:text/html', re.IGNORECASE),
        re.compile(r'eval\s*\(', re.IGNORECASE),
        re.compile(r'exec\s*\(', re.IGNORECASE),
        re.compile(r'setTimeout\s*\(', re.IGNORECASE),
        re.compile(r'setInterval\s*\(', re.IGNORECASE),
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        re.compile(r';\s*(rm|del|format|shutdown)', re.IGNORECASE),
        re.compile(r'\|\s*(nc|netcat|curl|wget)', re.IGNORECASE),
        re.compile(r'`[^`]*`'),
        re.compile(r'\$\([^)]*\)'),
        re.compile(r'&&\s*(rm|del|format)', re.IGNORECASE),
        re.compile(r'\|\|\s*(rm|del|format)', re.IGNORECASE),
        re.compile(r'system\s*\(', re.IGNORECASE),
        re.compile(r'os\.system', re.IGNORECASE),
        re.compile(r'subprocess\.', re.IGNORECASE),
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        re.compile(r'\.\./\.\.'),
        re.compile(r'\.\.\\\.\.'),
        re.compile(r'/etc/passwd'),
        re.compile(r'/etc/shadow'),
        re.compile(r'\\windows\\system32', re.IGNORECASE),
        re.compile(r'%SystemRoot%', re.IGNORECASE),
        re.compile(r'%USERPROFILE%', re.IGNORECASE),
    ]
    
    # Allowed base paths for file operations
    ALLOWED_BASE_PATHS = [
        os.path.expanduser("~/Documents"),
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Downloads"),
        "/tmp",
        "/var/tmp",
    ]
    
    # Maximum safe values
    MAX_TEXT_LENGTH = 10000
    MAX_LOOP_ITERATIONS = 1000
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_DURATION_SECONDS = 300  # 5 minutes
    
    def __init__(self):
        self.threats: List[SecurityThreat] = []
    
    def validate_text_input(self, text: str, field_name: str = "text") -> bool:
        """
        Validate text input for security threats.
        
        Args:
            text: Text to validate
            field_name: Name of the field for error reporting
            
        Returns:
            True if text is safe, False otherwise
        """
        if not isinstance(text, str):
            self._add_threat(
                ThreatType.SCRIPT_INJECTION,
                SecurityRiskLevel.HIGH,
                f"Invalid text type in {field_name}",
                "Ensure text is a string",
                field_name
            )
            return False
        
        # Check length
        if len(text) > self.MAX_TEXT_LENGTH:
            self._add_threat(
                ThreatType.RESOURCE_EXHAUSTION,
                SecurityRiskLevel.MEDIUM,
                f"Text too long in {field_name}: {len(text)} chars",
                f"Limit text to {self.MAX_TEXT_LENGTH} characters",
                field_name
            )
            return False
        
        # Check for script injection
        for pattern in self.SCRIPT_INJECTION_PATTERNS:
            if pattern.search(text):
                self._add_threat(
                    ThreatType.SCRIPT_INJECTION,
                    SecurityRiskLevel.CRITICAL,
                    f"Script injection detected in {field_name}",
                    "Remove script tags and JavaScript code",
                    field_name
                )
                return False
        
        # Check for command injection
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if pattern.search(text):
                self._add_threat(
                    ThreatType.COMMAND_INJECTION,
                    SecurityRiskLevel.CRITICAL,
                    f"Command injection detected in {field_name}",
                    "Remove shell commands and dangerous operators",
                    field_name
                )
                return False
        
        return True
    
    def validate_file_path(self, path: str, field_name: str = "path") -> bool:
        """
        Validate file path for directory traversal and access control.
        
        Args:
            path: File path to validate
            field_name: Name of the field for error reporting
            
        Returns:
            True if path is safe, False otherwise
        """
        if not isinstance(path, str):
            self._add_threat(
                ThreatType.PATH_TRAVERSAL,
                SecurityRiskLevel.HIGH,
                f"Invalid path type in {field_name}",
                "Ensure path is a string",
                field_name
            )
            return False
        
        # Normalize path
        try:
            normalized = os.path.normpath(os.path.abspath(path))
        except (ValueError, OSError):
            self._add_threat(
                ThreatType.PATH_TRAVERSAL,
                SecurityRiskLevel.HIGH,
                f"Invalid path format in {field_name}",
                "Use valid file path format",
                field_name
            )
            return False
        
        # Check for path traversal
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if pattern.search(path):
                self._add_threat(
                    ThreatType.PATH_TRAVERSAL,
                    SecurityRiskLevel.HIGH,
                    f"Path traversal detected in {field_name}",
                    "Use relative paths within allowed directories",
                    field_name
                )
                return False
        
        # Check if path is within allowed base paths
        allowed = False
        for base_path in self.ALLOWED_BASE_PATHS:
            try:
                base_norm = os.path.normpath(os.path.abspath(base_path))
                if normalized.startswith(base_norm):
                    allowed = True
                    break
            except (ValueError, OSError):
                continue
        
        if not allowed:
            self._add_threat(
                ThreatType.PATH_TRAVERSAL,
                SecurityRiskLevel.HIGH,
                f"Path outside allowed directories in {field_name}",
                f"Use paths within: {', '.join(self.ALLOWED_BASE_PATHS)}",
                field_name
            )
            return False
        
        return True
    
    def validate_numeric_range(
        self, 
        value: Any, 
        min_val: float, 
        max_val: float, 
        field_name: str = "value"
    ) -> bool:
        """
        Validate numeric value is within safe range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            field_name: Name of the field for error reporting
            
        Returns:
            True if value is within range, False otherwise
        """
        try:
            num_val = float(value)
        except (ValueError, TypeError):
            self._add_threat(
                ThreatType.RESOURCE_EXHAUSTION,
                SecurityRiskLevel.MEDIUM,
                f"Invalid numeric value in {field_name}",
                "Provide a valid number",
                field_name
            )
            return False
        
        if not (min_val <= num_val <= max_val):
            self._add_threat(
                ThreatType.RESOURCE_EXHAUSTION,
                SecurityRiskLevel.MEDIUM,
                f"Value out of range in {field_name}: {num_val}",
                f"Use value between {min_val} and {max_val}",
                field_name
            )
            return False
        
        return True
    
    def validate_permissions(self, required: Set[Permission], available: Set[Permission]) -> bool:
        """
        Validate that all required permissions are available.
        
        Args:
            required: Set of required permissions
            available: Set of available permissions
            
        Returns:
            True if all required permissions are available
        """
        missing = required - available
        if missing:
            self._add_threat(
                ThreatType.PRIVILEGE_ESCALATION,
                SecurityRiskLevel.HIGH,
                f"Missing required permissions: {missing}",
                "Grant required permissions or reduce command scope",
                "permissions"
            )
            return False
        
        return True
    
    def _add_threat(
        self, 
        threat_type: ThreatType, 
        severity: SecurityRiskLevel,
        description: str, 
        mitigation: str, 
        field_name: Optional[str] = None
    ) -> None:
        """Add a detected threat to the threat list."""
        threat = SecurityThreat(
            threat_type=threat_type,
            severity=severity,
            description=description,
            mitigation=mitigation,
            field_name=field_name
        )
        self.threats.append(threat)
    
    def get_threats(self) -> List[SecurityThreat]:
        """Get all detected threats."""
        return self.threats.copy()
    
    def has_critical_threats(self) -> bool:
        """Check if any critical threats were detected."""
        return any(threat.severity == SecurityRiskLevel.CRITICAL for threat in self.threats)
    
    def clear_threats(self) -> None:
        """Clear all detected threats."""
        self.threats.clear()


# Convenience functions for common validation tasks

def validate_text_input(text: str, field_name: str = "text") -> bool:
    """Validate text input for security threats."""
    validator = SecurityValidator()
    return validator.validate_text_input(text, field_name)


def validate_file_path(path: str, field_name: str = "path") -> bool:
    """Validate file path for directory traversal attacks."""
    validator = SecurityValidator()
    return validator.validate_file_path(path, field_name)


def validate_command_parameters(command_type: str, parameters: Dict[str, Any]) -> bool:
    """
    Validate command parameters against security policies.
    
    Args:
        command_type: Type of command being validated
        parameters: Dictionary of command parameters
        
    Returns:
        True if all parameters are safe, False otherwise
        
    Raises:
        CommandSecurityError: If critical security threats are detected
    """
    validator = SecurityValidator()
    
    # Validate each parameter based on its type
    for key, value in parameters.items():
        if isinstance(value, str):
            if "path" in key.lower() or "file" in key.lower():
                validator.validate_file_path(value, key)
            else:
                validator.validate_text_input(value, key)
        elif isinstance(value, (int, float)):
            # Basic numeric validation
            validator.validate_numeric_range(value, -1e6, 1e6, key)
    
    # Check for critical threats
    if validator.has_critical_threats():
        raise CommandSecurityError(
            f"Critical security threats detected in {command_type} command",
            validator.get_threats()
        )
    
    return len(validator.get_threats()) == 0