"""Comprehensive tests for the security input validator module.

import logging

logging.basicConfig(level=logging.DEBUG)
Tests all validation methods, threat detection, sanitization, and edge cases.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given
from hypothesis import strategies as st
from src.security.input_validator import InputValidator, ThreatType, ValidationResult


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation_safe(self) -> None:
        """Test creating a safe validation result."""
        result = ValidationResult(is_safe=True, sanitized_input="safe_input")

        assert result.is_safe is True
        assert result.threat_description == ""
        assert result.detected_threats == []
        assert result.sanitized_input == "safe_input"
        assert result.confidence_score == 0.0

    def test_validation_result_creation_unsafe(self) -> None:
        """Test creating an unsafe validation result."""
        threats = [ThreatType.SQL_INJECTION]
        result = ValidationResult(
            is_safe=False,
            threat_description="SQL injection detected",
            detected_threats=threats,
            sanitized_input="sanitized",
            confidence_score=0.9,
        )

        assert result.is_safe is False
        assert result.threat_description == "SQL injection detected"
        assert result.detected_threats == threats
        assert result.sanitized_input == "sanitized"
        assert result.confidence_score == 0.9

    def test_validation_result_post_init(self) -> None:
        """Test that post_init properly initializes detected_threats."""
        result = ValidationResult(is_safe=True)
        assert result.detected_threats == []

    def test_validation_result_with_multiple_threats(self) -> None:
        """Test validation result with multiple threat types."""
        threats = [ThreatType.SQL_INJECTION, ThreatType.XSS_INJECTION]
        result = ValidationResult(
            is_safe=False,
            detected_threats=threats,
            threat_description="Multiple threats detected",
        )

        assert len(result.detected_threats) == 2
        assert ThreatType.SQL_INJECTION in result.detected_threats
        assert ThreatType.XSS_INJECTION in result.detected_threats


class TestThreatType:
    """Test ThreatType enum."""

    def test_threat_type_values(self) -> None:
        """Test all threat type values."""
        assert ThreatType.SQL_INJECTION.value == "sql_injection"
        assert ThreatType.XSS_INJECTION.value == "xss_injection"
        assert ThreatType.COMMAND_INJECTION.value == "command_injection"
        assert ThreatType.PATH_TRAVERSAL.value == "path_traversal"
        assert ThreatType.SCRIPT_INJECTION.value == "script_injection"
        assert ThreatType.LDAP_INJECTION.value == "ldap_injection"
        assert ThreatType.XPATH_INJECTION.value == "xpath_injection"

    def test_threat_type_enum_complete(self) -> None:
        """Test that all expected threat types are defined."""
        expected_threats = {
            "sql_injection",
            "xss_injection",
            "command_injection",
            "path_traversal",
            "script_injection",
            "ldap_injection",
            "xpath_injection",
        }
        actual_threats = {threat.value for threat in ThreatType}
        assert actual_threats == expected_threats


class TestInputValidator:
    """Test InputValidator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_validator_initialization(self) -> None:
        """Test validator initialization with patterns."""
        assert len(self.validator.sql_patterns) > 0
        assert len(self.validator.xss_patterns) > 0
        assert len(self.validator.command_patterns) > 0
        assert len(self.validator.path_patterns) > 0

    def test_validator_pattern_types(self) -> None:
        """Test that all patterns are strings."""
        for pattern in self.validator.sql_patterns:
            assert isinstance(pattern, str)
        for pattern in self.validator.xss_patterns:
            assert isinstance(pattern, str)
        for pattern in self.validator.command_patterns:
            assert isinstance(pattern, str)
        for pattern in self.validator.path_patterns:
            assert isinstance(pattern, str)


class TestSQLValidation:
    """Test SQL injection validation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_safe_sql_input(self) -> None:
        """Test safe SQL input passes validation."""
        safe_inputs = [
            "normal text input",
            "user@example.com",
            "123456",
            "simple query",
            "",
        ]

        for safe_input in safe_inputs:
            result = self.validator.validate_sql_input(safe_input)
            assert result.is_safe is True
            assert result.detected_threats == []
            assert result.sanitized_input == safe_input
            if safe_input:  # Non-empty inputs should have confidence 1.0
                assert result.confidence_score == 1.0
            else:  # Empty input returns confidence 0.0
                assert result.confidence_score == 0.0

    def test_sql_injection_detection(self) -> None:
        """Test SQL injection pattern detection."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "1 OR 1=1",
            "UNION SELECT password FROM admin_users",
            "DELETE FROM users WHERE 1=1",
        ]

        for malicious_input in malicious_inputs:
            result = self.validator.validate_sql_input(malicious_input)
            assert result.is_safe is False
            assert ThreatType.SQL_INJECTION in result.detected_threats
            assert len(result.threat_description) > 0
            assert result.confidence_score == 0.9

    def test_sql_injection_specific_patterns(self) -> None:
        """Test specific SQL injection patterns that don't match all cases."""
        # This input doesn't match the current patterns but could be malicious in other contexts
        borderline_input = "INSERT INTO admin VALUES ('hacker', 'password')"
        result = self.validator.validate_sql_input(borderline_input)
        # Current implementation doesn't detect this as malicious
        assert result.is_safe is True

    def test_empty_sql_input(self) -> None:
        """Test empty SQL input handling."""
        result = self.validator.validate_sql_input("")
        assert result.is_safe is True
        assert result.sanitized_input == ""

    def test_none_sql_input(self) -> None:
        """Test None SQL input handling."""
        result = self.validator.validate_sql_input(None)
        assert result.is_safe is True

    def test_sql_sanitization(self) -> None:
        """Test SQL input sanitization."""
        malicious_input = "'; DROP TABLE users; --"
        result = self.validator.validate_sql_input(malicious_input)

        assert result.is_safe is False
        assert (
            "DROP" not in result.sanitized_input
            or "[BLOCKED_SQL_DROP]" in result.sanitized_input
        )
        assert "--" not in result.sanitized_input or result.sanitized_input.count(
            "--",
        ) < malicious_input.count("--")

    @given(st.text(min_size=1, max_size=100))
    def test_sql_validation_properties(self, text_input: Any) -> None:
        """Property-based test for SQL validation."""
        result = self.validator.validate_sql_input(text_input)

        # Result should always be a ValidationResult
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_safe, bool)
        assert isinstance(result.detected_threats, list)
        assert isinstance(result.sanitized_input, str)
        assert 0.0 <= result.confidence_score <= 1.0

        # If unsafe, should have threats and description
        if not result.is_safe:
            assert len(result.detected_threats) > 0
            assert len(result.threat_description) > 0


class TestHTMLValidation:
    """Test HTML/XSS validation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_safe_html_input(self) -> None:
        """Test safe HTML input passes validation."""
        safe_inputs = [
            "<p>Hello world</p>",
            "Plain text content",
            "<div>Safe content</div>",
            "<img src='image.jpg' alt='test'>",
            "",
        ]

        for safe_input in safe_inputs:
            result = self.validator.validate_html_input(safe_input)
            assert result.is_safe is True
            assert result.detected_threats == []
            assert result.sanitized_input == safe_input

    def test_xss_injection_detection(self) -> None:
        """Test XSS injection pattern detection."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<div onclick='alert(1)'>Click me</div>",
            "<iframe src='malicious.com'></iframe>",
            "<object data='malicious.swf'></object>",
            "<embed src='malicious.swf'></embed>",
        ]

        for malicious_input in malicious_inputs:
            result = self.validator.validate_html_input(malicious_input)
            assert result.is_safe is False
            assert ThreatType.XSS_INJECTION in result.detected_threats
            assert len(result.threat_description) > 0
            assert result.confidence_score == 0.9

    def test_html_sanitization(self) -> None:
        """Test HTML input sanitization."""
        malicious_input = "<script>alert('xss')</script>"
        result = self.validator.validate_html_input(malicious_input)

        assert result.is_safe is False
        assert (
            "script" not in result.sanitized_input.lower()
            or "[BLOCKED_SCRIPT]" in result.sanitized_input
        )

    def test_event_handler_sanitization(self) -> None:
        """Test event handler attribute sanitization."""
        malicious_input = "<div onclick='malicious()'>Test</div>"
        result = self.validator.validate_html_input(malicious_input)

        assert result.is_safe is False
        assert (
            "[BLOCKED_EVENT]" in result.sanitized_input
            or "onclick" not in result.sanitized_input
        )

    @given(st.text(min_size=1, max_size=100))
    def test_html_validation_properties(self, text_input: Any) -> None:
        """Property-based test for HTML validation."""
        result = self.validator.validate_html_input(text_input)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_safe, bool)
        assert isinstance(result.detected_threats, list)
        assert isinstance(result.sanitized_input, str)
        assert 0.0 <= result.confidence_score <= 1.0


class TestCommandValidation:
    """Test command injection validation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_safe_command_input(self) -> None:
        """Test safe command input passes validation."""
        safe_inputs = ["ls -la", "echo hello", "simple command", "filename.txt", ""]

        for safe_input in safe_inputs:
            result = self.validator.validate_command_input(safe_input)
            assert result.is_safe is True
            assert result.detected_threats == []
            assert result.sanitized_input == safe_input

    def test_command_injection_detection(self) -> None:
        """Test command injection pattern detection."""
        malicious_inputs = [
            "; rm -rf /",
            "$(cat /etc/passwd)",
            "`cat /etc/passwd`",
            "ls && rm file.txt",
            "echo test || format c:",
            "; cat /etc/passwd",
            "nc malicious.com 4444",
        ]

        for malicious_input in malicious_inputs:
            result = self.validator.validate_command_input(malicious_input)
            assert result.is_safe is False
            assert ThreatType.COMMAND_INJECTION in result.detected_threats
            assert len(result.threat_description) > 0
            assert result.confidence_score == 0.9

    def test_command_sanitization(self) -> None:
        """Test command input sanitization."""
        malicious_input = "; rm -rf /"
        result = self.validator.validate_command_input(malicious_input)

        assert result.is_safe is False
        assert (
            "[BLOCKED_SEPARATOR]" in result.sanitized_input
            or "[BLOCKED_COMMAND]" in result.sanitized_input
        )

    def test_command_substitution_sanitization(self) -> None:
        """Test command substitution sanitization."""
        inputs_and_blocks = [
            ("$(malicious)", "[BLOCKED_SUBSTITUTION]"),
            ("`malicious`", "[BLOCKED_BACKTICK]"),
            ("; rm file", "[BLOCKED_SEPARATOR]"),
        ]

        for malicious_input, expected_block in inputs_and_blocks:
            result = self.validator.validate_command_input(malicious_input)
            assert expected_block in result.sanitized_input

    @given(st.text(min_size=1, max_size=100))
    def test_command_validation_properties(self, text_input: Any) -> None:
        """Property-based test for command validation."""
        result = self.validator.validate_command_input(text_input)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_safe, bool)
        assert isinstance(result.detected_threats, list)
        assert isinstance(result.sanitized_input, str)
        assert 0.0 <= result.confidence_score <= 1.0


class TestPathValidation:
    """Test file path validation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_safe_file_paths(self) -> None:
        """Test safe file paths pass validation."""
        safe_paths = [
            "/home/user/file.txt",
            "documents/readme.md",
            "C:\\Users\\User\\Documents\\file.docx",
            "relative/path/file.txt",
            "",
        ]

        for safe_path in safe_paths:
            result = self.validator.validate_file_path(safe_path)
            assert result.is_safe is True
            assert result.detected_threats == []
            assert result.sanitized_input == safe_path

    def test_path_traversal_detection(self) -> None:
        """Test path traversal pattern detection."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\cmd.exe",
            "/etc/passwd",
            "/etc/shadow",
            "windows/system32/important.dll",
            "../../../../../etc/passwd",
        ]

        for malicious_path in malicious_paths:
            result = self.validator.validate_file_path(malicious_path)
            assert result.is_safe is False
            assert ThreatType.PATH_TRAVERSAL in result.detected_threats
            assert len(result.threat_description) > 0
            assert result.confidence_score == 0.9

    def test_path_sanitization(self) -> None:
        """Test file path sanitization."""
        malicious_path = "../../../etc/passwd"
        result = self.validator.validate_file_path(malicious_path)

        assert result.is_safe is False
        assert (
            "[BLOCKED_TRAVERSAL]" in result.sanitized_input
            or "[BLOCKED_SYSTEM_PATH]" in result.sanitized_input
        )

    def test_system_path_sanitization(self) -> None:
        """Test system path sanitization."""
        system_paths = ["/etc/passwd", "/etc/shadow", "windows/system32/config"]

        for system_path in system_paths:
            result = self.validator.validate_file_path(system_path)
            assert "[BLOCKED_SYSTEM_PATH]" in result.sanitized_input

    @given(st.text(min_size=1, max_size=100))
    def test_path_validation_properties(self, text_input: Any) -> None:
        """Property-based test for path validation."""
        result = self.validator.validate_file_path(text_input)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_safe, bool)
        assert isinstance(result.detected_threats, list)
        assert isinstance(result.sanitized_input, str)
        assert 0.0 <= result.confidence_score <= 1.0


class TestSanitizationMethods:
    """Test private sanitization methods."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_sql_sanitization_comprehensive(self) -> None:
        """Test comprehensive SQL sanitization."""
        test_cases = [
            ("/* comment */", ""),
            ("-- line comment", ""),
            ("DROP TABLE", "[BLOCKED_SQL_DROP] TABLE"),
            ("select * from users", "[BLOCKED_SQL_SELECT] * from users"),
            ("UNION SELECT", "[BLOCKED_SQL_UNION] [BLOCKED_SQL_SELECT]"),
        ]

        for input_text, expected_pattern in test_cases:
            sanitized = self.validator._sanitize_sql_input(input_text)
            if expected_pattern:
                assert expected_pattern.lower() in sanitized.lower()

    def test_html_sanitization_comprehensive(self) -> None:
        """Test comprehensive HTML sanitization."""
        test_cases = [
            ("<script>alert(1)</script>", "[BLOCKED_SCRIPT]"),
            ("javascript:alert(1)", "[BLOCKED_JS]"),
            ('onclick="alert(1)"', "[BLOCKED_EVENT]"),
        ]

        for input_text, expected_block in test_cases:
            sanitized = self.validator._sanitize_html_input(input_text)
            assert expected_block in sanitized

    def test_command_sanitization_comprehensive(self) -> None:
        """Test comprehensive command sanitization."""
        test_cases = [
            ("; rm file", "[BLOCKED_SEPARATOR]"),
            ("$(cmd)", "[BLOCKED_SUBSTITUTION]"),
            ("`cmd`", "[BLOCKED_BACKTICK]"),
            ("rm file", "[BLOCKED_COMMAND]"),
        ]

        for input_text, expected_block in test_cases:
            sanitized = self.validator._sanitize_command_input(input_text)
            assert expected_block in sanitized

    def test_path_sanitization_comprehensive(self) -> None:
        """Test comprehensive path sanitization."""
        test_cases = [
            ("../file", "[BLOCKED_TRAVERSAL]/file"),
            ("..\\file", "[BLOCKED_TRAVERSAL]\\file"),
            ("/etc/passwd", "[BLOCKED_SYSTEM_PATH]"),
        ]

        for input_text, expected_pattern in test_cases:
            sanitized = self.validator._sanitize_file_path(input_text)
            assert expected_pattern in sanitized


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_multiple_validation_types(self) -> None:
        """Test input that could trigger multiple validation types."""
        # This would be dangerous in multiple contexts
        dangerous_input = "<script>'; DROP TABLE users; --</script>"

        sql_result = self.validator.validate_sql_input(dangerous_input)
        html_result = self.validator.validate_html_input(dangerous_input)

        assert sql_result.is_safe is False
        assert html_result.is_safe is False
        assert ThreatType.SQL_INJECTION in sql_result.detected_threats
        assert ThreatType.XSS_INJECTION in html_result.detected_threats

    def test_case_insensitive_detection(self) -> None:
        """Test that detection is case insensitive."""
        uppercase_inputs = [
            "'; DROP TABLE USERS; --",
            "<SCRIPT>ALERT('XSS')</SCRIPT>",
            "; RM -RF /",
            "../ETC/PASSWD",
        ]

        sql_result = self.validator.validate_sql_input(uppercase_inputs[0])
        html_result = self.validator.validate_html_input(uppercase_inputs[1])
        cmd_result = self.validator.validate_command_input(uppercase_inputs[2])
        path_result = self.validator.validate_file_path(uppercase_inputs[3])

        assert sql_result.is_safe is False
        assert html_result.is_safe is False
        assert cmd_result.is_safe is False
        assert path_result.is_safe is False

    def test_complex_nested_attacks(self) -> None:
        """Test complex nested attack patterns."""
        complex_attacks = [
            "'; DROP TABLE users; SELECT '<script>alert(1)</script>' --",
            "$(echo ../../../etc/passwd | cat)",
            "<iframe src='javascript:$(rm -rf /)'></iframe>",
        ]

        for attack in complex_attacks:
            sql_result = self.validator.validate_sql_input(attack)
            html_result = self.validator.validate_html_input(attack)
            cmd_result = self.validator.validate_command_input(attack)

            # At least one should detect it as unsafe
            results = [sql_result, html_result, cmd_result]
            assert any(not result.is_safe for result in results)

    def test_edge_case_empty_strings(self) -> None:
        """Test edge cases with empty and whitespace strings."""
        edge_cases = ["", "   ", "\n", "\t", None]

        for edge_case in edge_cases:
            sql_result = self.validator.validate_sql_input(edge_case)
            html_result = self.validator.validate_html_input(edge_case)
            cmd_result = self.validator.validate_command_input(edge_case)
            path_result = self.validator.validate_file_path(edge_case)

            # All should be safe for empty/whitespace inputs
            assert sql_result.is_safe is True
            assert html_result.is_safe is True
            assert cmd_result.is_safe is True
            assert path_result.is_safe is True

    def test_performance_with_large_inputs(self) -> None:
        """Test performance with large inputs."""
        large_safe_input = "a" * 10000
        large_malicious_input = "a" * 5000 + "'; DROP TABLE users; --" + "a" * 5000

        # Should handle large inputs without errors
        sql_safe = self.validator.validate_sql_input(large_safe_input)
        sql_malicious = self.validator.validate_sql_input(large_malicious_input)

        assert sql_safe.is_safe is True
        assert sql_malicious.is_safe is False
        assert len(sql_safe.sanitized_input) == len(large_safe_input)
        assert len(sql_malicious.sanitized_input) > 0
