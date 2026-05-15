"""Security input validation module for comprehensive threat detection.

Provides validation for SQL injection, XSS, command injection, path traversal,
and other security threats with detailed threat analysis.
"""

import re
from dataclasses import dataclass
from enum import Enum


class ThreatType(Enum):
    """Types of security threats that can be detected."""

    SQL_INJECTION = "sql_injection"
    XSS_INJECTION = "xss_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    SCRIPT_INJECTION = "script_injection"
    LDAP_INJECTION = "ldap_injection"
    XPATH_INJECTION = "xpath_injection"


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_safe: bool
    threat_description: str = ""
    detected_threats: list[ThreatType] = None
    sanitized_input: str = ""
    confidence_score: float = 0.0

    def __post_init__(self) -> None:
        if self.detected_threats is None:
            self.detected_threats = []  # type: ignore[unreachable]  # runtime guard for callers passing None


class InputValidator:
    """Comprehensive input validator for security threats."""

    def __init__(self) -> None:
        """Initialize validator with threat patterns."""
        self.sql_patterns = [
            r"';.*--",  # SQL comment injection
            r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE)\b.*\b(TABLE|DATABASE|SCHEMA)\b",
            r"\bUNION\b.*\bSELECT\b",
            r"1\s*=\s*1",
            r"'\s*OR\s*'",  # Basic OR injection
            r"\d+'\s*OR\s*'",  # Number followed by OR injection
            r"\bSELECT\b.*\bFROM\b.*\bWHERE\b",
        ]

        self.xss_patterns = [
            r"<script.*?>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe.*?>",
            r"<object.*?>",
            r"<embed.*?>",
        ]

        self.command_patterns = [
            r";\s*(rm|del|format|mkfs)",
            r"\$\(.*\)",
            r"`.*`",
            r"&&|\|\|",
            r";\s*cat\s+/etc/passwd",
            r"nc\s+\w+\.\w+\s+\d+",
        ]

        self.path_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"/etc/passwd",
            r"/etc/shadow",
            r"windows/system32",
            r"\.\..*\.\..*\.\.",
        ]

    def validate_sql_input(self, user_input: str) -> ValidationResult:
        """Validate input for SQL injection threats."""
        if not user_input:
            return ValidationResult(is_safe=True, sanitized_input=user_input)

        detected_threats = []
        threat_descriptions = []

        # Check for SQL injection patterns
        for pattern in self.sql_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                detected_threats.append(ThreatType.SQL_INJECTION)
                threat_descriptions.append(f"SQL injection pattern detected: {pattern}")
                break

        is_safe = len(detected_threats) == 0
        threat_desc = "; ".join(threat_descriptions) if threat_descriptions else ""

        return ValidationResult(
            is_safe=is_safe,
            threat_description=threat_desc,
            detected_threats=detected_threats,
            sanitized_input=self._sanitize_sql_input(user_input)
            if not is_safe
            else user_input,
            confidence_score=0.9 if detected_threats else 1.0,
        )

    def validate_html_input(self, user_input: str) -> ValidationResult:
        """Validate input for XSS and HTML injection threats."""
        if not user_input:
            return ValidationResult(is_safe=True, sanitized_input=user_input)

        detected_threats = []
        threat_descriptions = []

        # Check for XSS patterns
        for pattern in self.xss_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                detected_threats.append(ThreatType.XSS_INJECTION)
                threat_descriptions.append(f"XSS pattern detected: {pattern}")
                break

        is_safe = len(detected_threats) == 0
        threat_desc = "; ".join(threat_descriptions) if threat_descriptions else ""

        return ValidationResult(
            is_safe=is_safe,
            threat_description=threat_desc,
            detected_threats=detected_threats,
            sanitized_input=self._sanitize_html_input(user_input)
            if not is_safe
            else user_input,
            confidence_score=0.9 if detected_threats else 1.0,
        )

    def validate_command_input(self, user_input: str) -> ValidationResult:
        """Validate input for command injection threats."""
        if not user_input:
            return ValidationResult(is_safe=True, sanitized_input=user_input)

        detected_threats = []
        threat_descriptions = []

        # Check for command injection patterns
        for pattern in self.command_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                detected_threats.append(ThreatType.COMMAND_INJECTION)
                threat_descriptions.append("Command injection pattern detected")
                break

        is_safe = len(detected_threats) == 0
        threat_desc = "; ".join(threat_descriptions) if threat_descriptions else ""

        return ValidationResult(
            is_safe=is_safe,
            threat_description=threat_desc,
            detected_threats=detected_threats,
            sanitized_input=self._sanitize_command_input(user_input)
            if not is_safe
            else user_input,
            confidence_score=0.9 if detected_threats else 1.0,
        )

    def validate_file_path(self, file_path: str) -> ValidationResult:
        """Validate file path for path traversal threats."""
        if not file_path:
            return ValidationResult(is_safe=True, sanitized_input=file_path)

        detected_threats = []
        threat_descriptions = []

        # Check for path traversal patterns
        for pattern in self.path_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                detected_threats.append(ThreatType.PATH_TRAVERSAL)
                threat_descriptions.append("Path traversal pattern detected")
                break

        is_safe = len(detected_threats) == 0
        threat_desc = "; ".join(threat_descriptions) if threat_descriptions else ""

        return ValidationResult(
            is_safe=is_safe,
            threat_description=threat_desc,
            detected_threats=detected_threats,
            sanitized_input=self._sanitize_file_path(file_path)
            if not is_safe
            else file_path,
            confidence_score=0.9 if detected_threats else 1.0,
        )

    def _sanitize_sql_input(self, user_input: str) -> str:
        """Sanitize SQL input by removing dangerous patterns."""
        sanitized = user_input

        # Remove SQL comments
        sanitized = re.sub(r"--.*$", "", sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r"/\*.*?\*/", "", sanitized, flags=re.DOTALL)

        # Remove dangerous SQL keywords
        dangerous_keywords = [
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "ALTER",
            "CREATE",
            "UNION",
            "SELECT",
            "EXEC",
            "EXECUTE",
        ]

        for keyword in dangerous_keywords:
            sanitized = re.sub(
                rf"\b{keyword}\b",
                f"[BLOCKED_SQL_{keyword}]",
                sanitized,
                flags=re.IGNORECASE,
            )

        return sanitized

    def _sanitize_html_input(self, user_input: str) -> str:
        """Sanitize HTML input by removing dangerous patterns."""
        sanitized = user_input

        # Remove script tags
        sanitized = re.sub(
            r"<script.*?>.*?</script>",
            "[BLOCKED_SCRIPT]",
            sanitized,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove dangerous attributes
        sanitized = re.sub(
            r"on\w+\s*=\s*[\"'].*?[\"']",
            "[BLOCKED_EVENT]",
            sanitized,
            flags=re.IGNORECASE,
        )

        # Remove javascript: URLs
        sanitized = re.sub(
            r"javascript:",
            "[BLOCKED_JS]",
            sanitized,
            flags=re.IGNORECASE,
        )

        return sanitized

    def _sanitize_command_input(self, user_input: str) -> str:
        """Sanitize command input by removing dangerous patterns."""
        sanitized = user_input

        # Remove command separators
        sanitized = re.sub(r"[;&|]", "[BLOCKED_SEPARATOR]", sanitized)

        # Remove command substitution
        sanitized = re.sub(r"\$\(.*?\)", "[BLOCKED_SUBSTITUTION]", sanitized)
        sanitized = re.sub(r"`.*?`", "[BLOCKED_BACKTICK]", sanitized)

        # Remove dangerous commands
        dangerous_commands = [
            "rm",
            "del",
            "format",
            "mkfs",
            "cat",
            "nc",
            "wget",
            "curl",
        ]
        for cmd in dangerous_commands:
            sanitized = re.sub(
                rf"\b{cmd}\b",
                "[BLOCKED_COMMAND]",
                sanitized,
                flags=re.IGNORECASE,
            )

        return sanitized

    def _sanitize_file_path(self, file_path: str) -> str:
        """Sanitize file path by removing dangerous patterns."""
        sanitized = file_path

        # Remove path traversal sequences
        sanitized = re.sub(r"\.\./", "[BLOCKED_TRAVERSAL]/", sanitized)
        sanitized = re.sub(r"\.\.[\\\\]", r"[BLOCKED_TRAVERSAL]\\", sanitized)

        # Block access to sensitive system files
        sensitive_paths = ["/etc/passwd", "/etc/shadow", "windows/system32"]
        for path in sensitive_paths:
            sanitized = re.sub(
                re.escape(path),
                "[BLOCKED_SYSTEM_PATH]",
                sanitized,
                flags=re.IGNORECASE,
            )

        return sanitized
