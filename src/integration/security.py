"""Security Validation and Input Sanitization.

Provides comprehensive security boundaries for Keyboard Maestro integration
with input validation, sanitization, and threat prevention.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeGuard

from ..core.contracts import ensure, require
from ..core.types import MacroId, Permission, TriggerId


class SecurityLevel(Enum):
    """Security validation levels."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ThreatType(Enum):
    """Types of security threats to detect."""

    SCRIPT_INJECTION = "script_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    MACRO_ABUSE = "macro_abuse"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass(frozen=True)
class SecurityViolation:
    """Detected security violation."""

    threat_type: ThreatType
    field_name: str
    violation_text: str
    severity: str
    recommendation: str

    @classmethod
    def create(
        cls,
        threat_type: ThreatType,
        field_name: str,
        violation_text: str,
        severity: str = "medium",
        recommendation: str = "Remove or sanitize the suspicious content",
    ) -> SecurityViolation:
        """Create security violation instance."""
        return cls(threat_type, field_name, violation_text, severity, recommendation)


@dataclass(frozen=True)
class ValidationResult:
    """Result of security validation."""

    is_safe: bool
    violations: list[SecurityViolation] = field(default_factory=list)
    sanitized_data: dict[str, Any] | None = None

    @classmethod
    def safe(cls, sanitized_data: dict[str, Any]) -> ValidationResult:
        """Create safe validation result."""
        return cls(is_safe=True, sanitized_data=sanitized_data)

    @classmethod
    def unsafe(
        cls,
        violations: list[SecurityViolation],
        sanitized_data: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Create unsafe validation result."""
        return cls(is_safe=False, violations=violations, sanitized_data=sanitized_data)

    def has_critical_violations(self) -> bool:
        """Check if any violations are critical."""
        return any(v.severity == "critical" for v in self.violations)


@dataclass(frozen=True)
class ValidatedKMInput:
    """Type-safe validated input from Keyboard Maestro."""

    macro_id: MacroId | None
    trigger_id: TriggerId | None
    parameters: dict[str, Any]
    metadata: dict[str, Any]
    validation_level: SecurityLevel

    @classmethod
    def create(
        cls,
        raw_input: dict[str, Any],
        validation_level: SecurityLevel = SecurityLevel.STANDARD,
    ) -> ValidatedKMInput:
        """Create validated input from raw data."""
        return cls(
            macro_id=MacroId(raw_input.get("macro_id", "")),
            trigger_id=TriggerId(raw_input.get("trigger_id", "")),
            parameters=raw_input.get("parameters", {}),
            metadata=raw_input.get("metadata", {}),
            validation_level=validation_level,
        )


@dataclass(frozen=True)
class SanitizedTriggerData:
    """Sanitized trigger data safe for processing."""

    trigger_type: str
    configuration: dict[str, Any]
    permissions_required: set[Permission]
    safety_level: SecurityLevel

    def requires_permission(self, permission: Permission) -> bool:
        """Check if trigger requires specific permission."""
        return permission in self.permissions_required


# Security patterns and blacklists

SCRIPT_INJECTION_PATTERNS = [
    r"<script\b[^>]*>.*?</script>",  # Enhanced script tag detection
    r"<script\b[^>]*>",  # Script opening tag without closing
    r"javascript:",  # JavaScript URLs
    r"vbscript:",  # VBScript URLs
    r"data:text/html",  # Data URLs
    r"on\w+\s*=",  # Any event handler (onload, onerror, onclick, etc.)
    r"eval\s*\(",  # eval() calls
    r"exec\s*\(",  # exec() calls
    r"system\s*\(",  # system() calls
    r"os\.system",  # os.system calls
    r"subprocess\.",  # subprocess calls
    r"__import__",  # import statements
    r"setTimeout\s*\(",  # setTimeout calls
    r"setInterval\s*\(",  # setInterval calls
    r"expression\s*\(",  # CSS expression
    r"alert\s*\(",  # JavaScript alert
    r"document\.",  # DOM manipulation
    r"window\.",  # Window object access
]

COMMAND_INJECTION_PATTERNS = [
    r";\s*\w+",  # Command chaining with semicolon
    r"\|\s*\w+",  # Piping commands
    r"&&\s*\w+",  # AND chaining
    r"\|\|\s*\w+",  # OR chaining
    r"`[^`]*`",  # Backtick execution
    r"\$\([^)]*\)",  # Command substitution
    r"rm\s+-rf",  # Dangerous remove command
    r"sudo\s+",  # Privilege escalation
    r"curl\s+",  # Network access
    r"wget\s+",  # Network access
    r"nc\s+",  # Netcat
    r"netcat\s+",  # Netcat
    r"ssh\s+",  # SSH access
    r"exec\s+",  # Execute command
]

PATH_TRAVERSAL_PATTERNS = [
    r"\.\.[/\\]",  # Basic path traversal
    r"[/\\]\.\.[/\\]",  # Path traversal in middle
    r"^\.\./",  # Path traversal at start
    r"\.\.\\",  # Windows path traversal
    r"/etc/passwd",  # Unix sensitive files
    r"/etc/shadow",
    r"\\windows\\system32",  # Windows sensitive directories
    r"%SystemRoot%",  # Windows environment variables
    r"%USERPROFILE%",
    r"~/",  # Home directory access
]

SQL_INJECTION_PATTERNS = [
    r"'\s*;\s*drop\s+table",  # Drop table
    r"'\s*;\s*delete\s+from",  # Delete from
    r"'\s*;\s*insert\s+into",  # Insert into
    r"'\s*union\s+select",  # Union select
    r"'\s*or\s+'1'\s*=\s*'1",  # Classic or 1=1
    r"'\s*and\s+'1'\s*=\s*'1",  # Classic and 1=1
    r"admin'\s*--",  # Comment out rest
    r"'\s*or\s+1\s*=\s*1",  # Numeric or 1=1
]

DANGEROUS_APPLESCRIPT_PATTERNS = [
    r"do\s+shell\s+script",  # Execute shell commands
    r'tell\s+application\s+"System Events"',  # System control
    r'tell\s+application\s+"Terminal"',  # Terminal access
    r"keystroke\s+",  # Keystroke simulation
    r"system\s+info",  # System information
    r"file\s+delete",  # File deletion
    r"folder\s+delete",  # Folder deletion
    r"activate\s+application",  # Application control
    r"set\s+volume",  # Volume control
    r"restart\s+computer",  # System restart
    r"shutdown\s+computer",  # System shutdown
]


def validate_km_input(
    raw_input: dict[str, Any],
    level: SecurityLevel = SecurityLevel.STANDARD,
) -> ValidationResult:
    """Comprehensive validation of KM input data."""
    violations = []
    sanitized = {}

    # Validate each field based on security level
    for field_name, value in raw_input.items():
        field_violations, sanitized_value = _validate_field(field_name, value, level)
        violations.extend(field_violations)
        sanitized[field_name] = sanitized_value

    # Additional validation for specific field combinations
    combination_violations = _validate_field_combinations(sanitized, level)
    violations.extend(combination_violations)

    # Enhanced security check: ANY critical violation should trigger unsafe result
    if violations:
        # Check for critical or high severity violations - these must be blocked
        critical_violations = [
            v for v in violations if v.severity in ("critical", "high")
        ]
        if critical_violations:
            return ValidationResult.unsafe(violations, sanitized)

        # Even medium violations should be treated carefully in strict modes
        medium_violations = [v for v in violations if v.severity == "medium"]
        if medium_violations and level in (
            SecurityLevel.STRICT,
            SecurityLevel.PARANOID,
        ):
            return ValidationResult.unsafe(violations, sanitized)

    # If we reach here and have sanitized the content properly, return safe result
    return ValidationResult.safe(sanitized)


def _validate_field(
    field_name: str,
    value: Any,
    level: SecurityLevel,
) -> tuple[list[SecurityViolation], Any]:
    """Validate individual field value."""
    violations = []

    # Handle nested dictionary validation (like configuration field)
    if isinstance(value, dict):
        sanitized_dict = {}
        for nested_field, nested_value in value.items():
            nested_violations, nested_sanitized = _validate_field(
                f"{field_name}.{nested_field}",
                nested_value,
                level,
            )
            violations.extend(nested_violations)
            sanitized_dict[nested_field] = nested_sanitized
        return violations, sanitized_dict

    if not isinstance(value, str):
        return violations, value

    # Always sanitize first to get clean value
    sanitized_value = _sanitize_value(value, level)

    # Check for script injection in original value
    for pattern in SCRIPT_INJECTION_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            violations.append(
                SecurityViolation.create(
                    ThreatType.SCRIPT_INJECTION,
                    field_name,
                    f"Potential script injection detected: {pattern}",
                    "critical",
                    "Remove script tags and JavaScript code",
                ),
            )

    # Check for command injection in original value
    for pattern in COMMAND_INJECTION_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            violations.append(
                SecurityViolation.create(
                    ThreatType.COMMAND_INJECTION,
                    field_name,
                    f"Potential command injection detected: {pattern}",
                    "critical",
                    "Remove shell commands and dangerous operators",
                ),
            )

    # Check for path traversal in original value
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            violations.append(
                SecurityViolation.create(
                    ThreatType.PATH_TRAVERSAL,
                    field_name,
                    f"Path traversal attempt detected: {pattern}",
                    "high",
                    "Use relative paths within allowed directories",
                ),
            )

    # Check for SQL injection in original value
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            violations.append(
                SecurityViolation.create(
                    ThreatType.SQL_INJECTION,
                    field_name,
                    f"SQL injection attempt detected: {pattern}",
                    "high",
                    "Use parameterized queries and escape SQL characters",
                ),
            )

    # AppleScript-specific validation in original value
    if field_name in ["script_content", "applescript", "command"]:
        for pattern in DANGEROUS_APPLESCRIPT_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                violations.append(
                    SecurityViolation.create(
                        ThreatType.MACRO_ABUSE,
                        field_name,
                        f"Dangerous AppleScript command detected: {pattern}",
                        "critical",
                        "Use safer alternatives or remove system-level commands",
                    ),
                )

    # Return violations and SANITIZED value (not original)
    return violations, sanitized_value


def _validate_field_combinations(
    data: dict[str, Any],
    _level: SecurityLevel,
) -> list[SecurityViolation]:
    """Validate combinations of fields for security issues."""
    violations = []

    # Check for privilege escalation attempts
    if data.get("permissions") and data.get("script_content"):
        high_perms = {"SYSTEM_CONTROL", "FILE_ACCESS", "APPLICATION_CONTROL"}
        requested_perms = set(data.get("permissions", []))

        if high_perms.intersection(requested_perms) and "sudo" in str(
            data.get("script_content", ""),
        ):
            violations.append(
                SecurityViolation.create(
                    ThreatType.PRIVILEGE_ESCALATION,
                    "permissions + script_content",
                    "High permissions requested with sudo in script",
                    "critical",
                    "Remove sudo commands or reduce permissions",
                ),
            )

    return violations


def _sanitize_value(value: str, level: SecurityLevel) -> str:
    """Sanitize string value based on security level."""
    original_value = value

    if level == SecurityLevel.MINIMAL:
        return value[:1000]  # Just truncate

    if level == SecurityLevel.STANDARD:
        # Enhanced sanitization for standard level
        sanitized = value  # Don't html.escape initially to check patterns on original

        # Remove script injection patterns first
        for pattern in SCRIPT_INJECTION_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = re.sub(
                    pattern,
                    "[BLOCKED_SCRIPT]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

        # Remove command injection patterns
        for pattern in COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = re.sub(
                    pattern,
                    "[BLOCKED_COMMAND]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

        # Remove path traversal patterns
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = re.sub(
                    pattern,
                    "[BLOCKED_PATH]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

        # Remove SQL injection patterns
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = re.sub(
                    pattern,
                    "[BLOCKED_SQL]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

        # Remove AppleScript danger patterns
        for pattern in DANGEROUS_APPLESCRIPT_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = re.sub(
                    pattern,
                    "[BLOCKED_APPLESCRIPT]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

        # Then apply HTML escaping for additional safety
        sanitized = html.escape(sanitized)

        # Debug: Add some logging to understand what's happening
        import sys

        if original_value != sanitized:
            print(
                f"SANITIZATION: {original_value!r} -> {sanitized!r}",
                file=sys.stderr,
            )

        return sanitized[:1000]

    if level == SecurityLevel.STRICT:
        # More aggressive sanitization
        sanitized = html.escape(value)

        # Remove all detected patterns completely
        for pattern_list in [
            SCRIPT_INJECTION_PATTERNS,
            COMMAND_INJECTION_PATTERNS,
            PATH_TRAVERSAL_PATTERNS,
            SQL_INJECTION_PATTERNS,
            DANGEROUS_APPLESCRIPT_PATTERNS,
        ]:
            for pattern in pattern_list:
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Remove all HTML tags
        sanitized = re.sub(r"<[^>]*>", "", sanitized)

        return sanitized[:500]

    # Very strict whitelist approach - only allow safe characters
    sanitized = re.sub(r"[^a-zA-Z0-9\s\-_.,!?]", "", value)
    return sanitized[:200]


def sanitize_trigger_data(
    trigger_data: dict[str, Any],
    level: SecurityLevel = SecurityLevel.STANDARD,
) -> SanitizedTriggerData:
    """Sanitize trigger data to prevent injection attacks."""
    trigger_type = trigger_data.get("trigger_type", "unknown")

    # Sanitize configuration
    sanitized_config = {}
    for key, value in trigger_data.get("configuration", {}).items():
        if isinstance(value, str):
            sanitized_config[key] = _sanitize_value(value, level)
        else:
            sanitized_config[key] = value

    # Determine required permissions based on trigger type and config
    permissions_required = _determine_permissions(trigger_type, sanitized_config)

    return SanitizedTriggerData(
        trigger_type=trigger_type,
        configuration=sanitized_config,
        permissions_required=permissions_required,
        safety_level=level,
    )


def _determine_permissions(
    trigger_type: str,
    config: dict[str, Any],
) -> set[Permission]:
    """Determine required permissions for trigger configuration."""
    permissions = set()

    if trigger_type == "hotkey":
        permissions.add(Permission.TEXT_INPUT)

    elif trigger_type == "application":
        permissions.add(Permission.APPLICATION_CONTROL)
        if config.get("launch_app"):
            permissions.add(Permission.SYSTEM_CONTROL)

    elif trigger_type == "file":
        permissions.add(Permission.FILE_ACCESS)
        if config.get("watch_system_directories"):
            permissions.add(Permission.SYSTEM_CONTROL)

    elif trigger_type == "system":
        permissions.add(Permission.SYSTEM_CONTROL)

    # Check for clipboard access in any config
    if any("clipboard" in str(v).lower() for v in config.values()):
        permissions.add(Permission.CLIPBOARD_ACCESS)

    # Check for network access
    if any(
        "http" in str(v).lower() or "url" in str(v).lower() for v in config.values()
    ):
        permissions.add(Permission.NETWORK_ACCESS)

    # Check for screen capture
    if any(
        "screen" in str(v).lower() or "capture" in str(v).lower()
        for v in config.values()
    ):
        permissions.add(Permission.SCREEN_CAPTURE)

    return permissions


@require(lambda data: isinstance(data, dict))
@ensure(lambda result: isinstance(result, bool))
def is_valid_km_format(data: dict[str, Any]) -> bool:
    """Check if data follows valid KM format."""
    required_fields = ["trigger_type", "configuration"]

    if not all(field in data for field in required_fields):
        return False

    if not isinstance(data["configuration"], dict):
        return False

    # Check trigger_type is valid
    valid_types = {
        "hotkey",
        "application",
        "time",
        "system",
        "file",
        "device",
        "periodic",
        "remote",
    }
    return data["trigger_type"] in valid_types


def is_sanitized(result: SanitizedTriggerData) -> bool:
    """Check if trigger data has been properly sanitized."""
    return (
        result.safety_level != SecurityLevel.MINIMAL
        and len(result.permissions_required) > 0
        and isinstance(result.configuration, dict)
    )


def process_km_event(event_data: dict[str, Any]) -> SanitizedTriggerData:
    """Process KM event with security boundaries."""
    # Validate input format
    if not is_valid_km_format(event_data):
        raise ValueError("Invalid KM event format")

    # Sanitize the data
    sanitized = sanitize_trigger_data(event_data, SecurityLevel.STANDARD)

    # Verify sanitization
    if not is_sanitized(sanitized):
        raise ValueError("Sanitization failed")

    return sanitized


def validate_km_input_safe(raw_input: dict) -> TypeGuard[ValidatedKMInput]:
    """Type guard for validated KM input - renamed to avoid conflict."""
    try:
        validation_result = validate_km_input(raw_input)
        return (
            validation_result.is_safe
            and not validation_result.has_critical_violations()
        )
    except Exception:
        return False


# TASK_2 Phase 2: Additional Security Functions for Trigger Management


def validate_trigger_input(trigger_config: dict[str, Any]) -> bool:
    """Validate trigger configuration input for security."""
    validation_result = validate_km_input(trigger_config, SecurityLevel.STANDARD)
    return validation_result.is_safe and not validation_result.has_critical_violations()


def sanitize_trigger_configuration(trigger_config: dict[str, Any]) -> dict[str, Any]:
    """Sanitize trigger configuration to remove dangerous content."""
    # Use existing sanitization infrastructure
    sanitized_data = sanitize_trigger_data(
        {"configuration": trigger_config, "trigger_type": "unknown"},
        SecurityLevel.STANDARD,
    )
    return sanitized_data.configuration


# Security utility functions


def create_security_report(violations: list[SecurityViolation]) -> dict[str, Any]:
    """Create comprehensive security report."""
    return {
        "total_violations": len(violations),
        "critical_count": sum(1 for v in violations if v.severity == "critical"),
        "high_count": sum(1 for v in violations if v.severity == "high"),
        "medium_count": sum(1 for v in violations if v.severity == "medium"),
        "threat_types": list({v.threat_type.value for v in violations}),
        "affected_fields": list({v.field_name for v in violations}),
        "recommendations": [v.recommendation for v in violations[:5]],  # Top 5
    }


def get_minimum_security_level(permissions: set[Permission]) -> SecurityLevel:
    """Get minimum required security level for permissions."""
    dangerous_perms = {
        Permission.SYSTEM_CONTROL,
        Permission.FILE_ACCESS,
        Permission.APPLICATION_CONTROL,
    }

    if dangerous_perms.intersection(permissions):
        return SecurityLevel.STRICT

    return SecurityLevel.STANDARD
