"""
Tests for Security Validation and Input Sanitization.

Tests comprehensive security boundaries, input validation,
sanitization, and threat prevention for KM integration.
"""

import pytest
from typing import Dict, Any

from src.integration.security import (
    validate_km_input, sanitize_trigger_data, SecurityLevel, ThreatType,
    SecurityViolation, ValidationResult, SanitizedTriggerData,
    validate_trigger_input, sanitize_trigger_configuration,
    is_valid_km_format, is_sanitized, process_km_event,
    create_security_report, get_minimum_security_level
)
from src.core.types import Permission


@pytest.fixture
def clean_input():
    """Clean input data for testing."""
    return {
        "trigger_type": "hotkey",
        "configuration": {
            "key": "space",
            "modifiers": ["cmd"],
            "description": "Simple hotkey trigger"
        },
        "macro_id": "safe-macro-123",
        "enabled": True
    }


@pytest.fixture
def malicious_input():
    """Malicious input data for security testing."""
    return {
        "trigger_type": "application",
        "configuration": {
            "script": "<script>alert('xss')</script>",
            "command": "rm -rf /",
            "path": "../../../etc/passwd",
            "sql": "'; DROP TABLE users; --",
            "js": "javascript:void(0)"
        },
        "macro_id": "malicious-macro",
        "applescript": "do shell script \"sudo shutdown now\""
    }


@pytest.fixture
def path_traversal_input():
    """Input with path traversal attempts."""
    return {
        "trigger_type": "file",
        "configuration": {
            "watch_path": "../../../../../../etc/shadow",
            "backup_path": "\\windows\\system32\\config",
            "target": "%SystemRoot%\\critical.dll"
        }
    }


class TestSecurityViolation:
    """Test security violation detection and reporting."""
    
    def test_violation_creation(self):
        """Test creating security violations."""
        violation = SecurityViolation.create(
            ThreatType.SCRIPT_INJECTION,
            "script_field",
            "<script>alert('test')</script>",
            "critical",
            "Remove script tags"
        )
        
        assert violation.threat_type == ThreatType.SCRIPT_INJECTION
        assert violation.field_name == "script_field"
        assert violation.severity == "critical"
        assert "Remove script tags" in violation.recommendation


class TestValidationResult:
    """Test validation result creation and handling."""
    
    def test_safe_result_creation(self, clean_input):
        """Test creating safe validation results."""
        result = ValidationResult.safe(clean_input)
        
        assert result.is_safe
        assert len(result.violations) == 0
        assert result.sanitized_data == clean_input
        assert not result.has_critical_violations()
    
    def test_unsafe_result_creation(self):
        """Test creating unsafe validation results."""
        violations = [
            SecurityViolation.create(
                ThreatType.SCRIPT_INJECTION,
                "test_field",
                "dangerous content",
                "critical"
            )
        ]
        result = ValidationResult.unsafe(violations)
        
        assert not result.is_safe
        assert len(result.violations) == 1
        assert result.sanitized_data is None
        assert result.has_critical_violations()


class TestInputValidation:
    """Test comprehensive input validation."""
    
    def test_clean_input_validation(self, clean_input):
        """Test validation of clean input."""
        result = validate_km_input(clean_input, SecurityLevel.STANDARD)
        
        assert result.is_safe
        assert len(result.violations) == 0
        assert result.sanitized_data is not None
    
    def test_script_injection_detection(self, malicious_input):
        """Test detection of script injection attempts."""
        result = validate_km_input(malicious_input, SecurityLevel.STANDARD)
        
        assert not result.is_safe
        script_violations = [
            v for v in result.violations 
            if v.threat_type == ThreatType.SCRIPT_INJECTION
        ]
        assert len(script_violations) > 0
        assert any("script" in v.violation_text.lower() for v in script_violations)
    
    def test_command_injection_detection(self, malicious_input):
        """Test detection of command injection attempts."""
        result = validate_km_input(malicious_input, SecurityLevel.STANDARD)
        
        command_violations = [
            v for v in result.violations
            if v.threat_type == ThreatType.COMMAND_INJECTION
        ]
        assert len(command_violations) > 0
        assert any("rm" in v.violation_text for v in command_violations)
    
    def test_path_traversal_detection(self, path_traversal_input):
        """Test detection of path traversal attempts."""
        result = validate_km_input(path_traversal_input, SecurityLevel.STANDARD)
        
        path_violations = [
            v for v in result.violations
            if v.threat_type == ThreatType.PATH_TRAVERSAL
        ]
        assert len(path_violations) > 0
        assert any(".." in v.violation_text for v in path_violations)
    
    def test_sql_injection_detection(self, malicious_input):
        """Test detection of SQL injection attempts."""
        result = validate_km_input(malicious_input, SecurityLevel.STANDARD)
        
        sql_violations = [
            v for v in result.violations
            if v.threat_type == ThreatType.SQL_INJECTION
        ]
        assert len(sql_violations) > 0
        assert any("DROP TABLE" in v.violation_text for v in sql_violations)
    
    def test_applescript_danger_detection(self, malicious_input):
        """Test detection of dangerous AppleScript commands."""
        result = validate_km_input(malicious_input, SecurityLevel.STANDARD)
        
        macro_violations = [
            v for v in result.violations
            if v.threat_type == ThreatType.MACRO_ABUSE
        ]
        assert len(macro_violations) > 0
        assert any("sudo" in v.violation_text for v in macro_violations)


class TestSecurityLevels:
    """Test different security validation levels."""
    
    def test_minimal_security_level(self, malicious_input):
        """Test minimal security level (most permissive)."""
        result = validate_km_input(malicious_input, SecurityLevel.MINIMAL)
        
        # Should still detect critical violations but be more lenient
        assert result.sanitized_data is not None
        # Check that string values are truncated
        for value in result.sanitized_data.values():
            if isinstance(value, str):
                assert len(value) <= 1000
    
    def test_standard_security_level(self, malicious_input):
        """Test standard security level."""
        result = validate_km_input(malicious_input, SecurityLevel.STANDARD)
        
        assert not result.is_safe
        assert len(result.violations) > 0
        
        # Check sanitization applied
        sanitized = result.sanitized_data
        for key, value in sanitized.items():
            if isinstance(value, str):
                assert "<script" not in value  # Should be escaped
                assert len(value) <= 1000
    
    def test_strict_security_level(self, malicious_input):
        """Test strict security level."""
        result = validate_km_input(malicious_input, SecurityLevel.STRICT)
        
        assert not result.is_safe
        assert len(result.violations) > 0
        
        # More aggressive sanitization
        sanitized = result.sanitized_data
        for key, value in sanitized.items():
            if isinstance(value, str):
                assert "<" not in value  # All HTML tags removed
                assert "javascript:" not in value
                assert len(value) <= 500
    
    def test_paranoid_security_level(self, malicious_input):
        """Test paranoid security level (most restrictive)."""
        result = validate_km_input(malicious_input, SecurityLevel.PARANOID)
        
        # Very restrictive whitelist approach
        sanitized = result.sanitized_data
        for key, value in sanitized.items():
            if isinstance(value, str):
                # Should only contain alphanumeric and basic punctuation
                allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.,!?")
                assert all(c in allowed_chars for c in value)
                assert len(value) <= 200


class TestTriggerDataSanitization:
    """Test trigger-specific data sanitization."""
    
    def test_sanitize_clean_trigger_data(self, clean_input):
        """Test sanitizing clean trigger data."""
        result = sanitize_trigger_data(clean_input, SecurityLevel.STANDARD)
        
        assert isinstance(result, SanitizedTriggerData)
        assert result.trigger_type == "hotkey"
        assert result.safety_level == SecurityLevel.STANDARD
        assert len(result.permissions_required) > 0
    
    def test_sanitize_malicious_trigger_data(self, malicious_input):
        """Test sanitizing malicious trigger data."""
        result = sanitize_trigger_data(malicious_input, SecurityLevel.STANDARD)
        
        # Configuration should be sanitized
        config = result.configuration
        assert "&lt;script" in config.get("script", "")  # HTML escaped
        assert "javascript_" in config.get("js", "")  # Neutralized
    
    def test_permission_determination(self):
        """Test determination of required permissions."""
        hotkey_data = {
            "trigger_type": "hotkey",
            "configuration": {"key": "space"}
        }
        result = sanitize_trigger_data(hotkey_data)
        assert Permission.TEXT_INPUT in result.permissions_required
        
        app_data = {
            "trigger_type": "application",
            "configuration": {"launch_app": True}
        }
        result = sanitize_trigger_data(app_data)
        assert Permission.APPLICATION_CONTROL in result.permissions_required
        assert Permission.SYSTEM_CONTROL in result.permissions_required
        
        file_data = {
            "trigger_type": "file",
            "configuration": {"watch_system_directories": True}
        }
        result = sanitize_trigger_data(file_data)
        assert Permission.FILE_ACCESS in result.permissions_required
        assert Permission.SYSTEM_CONTROL in result.permissions_required
    
    def test_clipboard_permission_detection(self):
        """Test detection of clipboard access requirements."""
        clipboard_data = {
            "trigger_type": "system",
            "configuration": {
                "action": "copy to clipboard",
                "target": "clipboard content"
            }
        }
        result = sanitize_trigger_data(clipboard_data)
        assert Permission.CLIPBOARD_ACCESS in result.permissions_required
    
    def test_network_permission_detection(self):
        """Test detection of network access requirements."""
        network_data = {
            "trigger_type": "remote",
            "configuration": {
                "url": "https://example.com/trigger",
                "http_method": "POST"
            }
        }
        result = sanitize_trigger_data(network_data)
        assert Permission.NETWORK_ACCESS in result.permissions_required
    
    def test_screen_capture_permission_detection(self):
        """Test detection of screen capture requirements."""
        screen_data = {
            "trigger_type": "system",
            "configuration": {
                "action": "screen capture",
                "area": "full screen"
            }
        }
        result = sanitize_trigger_data(screen_data)
        assert Permission.SCREEN_CAPTURE in result.permissions_required


class TestKMFormatValidation:
    """Test KM format validation functions."""
    
    def test_valid_km_format(self, clean_input):
        """Test validation of correct KM format."""
        assert is_valid_km_format(clean_input)
    
    def test_invalid_km_format_missing_fields(self):
        """Test validation with missing required fields."""
        invalid_data = {"trigger_type": "hotkey"}  # Missing configuration
        assert not is_valid_km_format(invalid_data)
        
        invalid_data2 = {"configuration": {}}  # Missing trigger_type
        assert not is_valid_km_format(invalid_data2)
    
    def test_invalid_km_format_wrong_types(self):
        """Test validation with wrong field types."""
        invalid_data = {
            "trigger_type": "hotkey",
            "configuration": "not a dict"  # Should be dict
        }
        assert not is_valid_km_format(invalid_data)
    
    def test_invalid_trigger_type(self):
        """Test validation with invalid trigger type."""
        invalid_data = {
            "trigger_type": "invalid_type",
            "configuration": {}
        }
        assert not is_valid_km_format(invalid_data)


class TestSanitizationValidation:
    """Test sanitization validation functions."""
    
    def test_is_sanitized_check(self):
        """Test sanitization validation."""
        sanitized_data = SanitizedTriggerData(
            trigger_type="hotkey",
            configuration={"key": "space"},
            permissions_required={Permission.TEXT_INPUT},
            safety_level=SecurityLevel.STANDARD
        )
        assert is_sanitized(sanitized_data)
        
        # Minimal safety level should fail
        minimal_data = SanitizedTriggerData(
            trigger_type="hotkey",
            configuration={"key": "space"},
            permissions_required={Permission.TEXT_INPUT},
            safety_level=SecurityLevel.MINIMAL
        )
        assert not is_sanitized(minimal_data)
        
        # No permissions should fail
        no_perms_data = SanitizedTriggerData(
            trigger_type="hotkey",
            configuration={"key": "space"},
            permissions_required=set(),
            safety_level=SecurityLevel.STANDARD
        )
        assert not is_sanitized(no_perms_data)


class TestContractValidation:
    """Test contract-based validation functions."""
    
    def test_process_km_event_success(self, clean_input):
        """Test successful KM event processing with contracts."""
        result = process_km_event(clean_input)
        
        assert isinstance(result, SanitizedTriggerData)
        assert is_sanitized(result)
    
    def test_process_km_event_invalid_format(self):
        """Test KM event processing with invalid format."""
        invalid_input = {"invalid": "data"}
        
        with pytest.raises(ValueError, match="Invalid KM event format"):
            process_km_event(invalid_input)


class TestTriggerManagementSecurity:
    """Test security functions for trigger management."""
    
    def test_validate_trigger_input(self, clean_input):
        """Test trigger input validation."""
        config = clean_input["configuration"]
        assert validate_trigger_input(config)
        
        # Invalid input should fail
        invalid_config = {"script": "<script>alert('xss')</script>"}
        assert not validate_trigger_input(invalid_config)
    
    def test_sanitize_trigger_configuration(self, malicious_input):
        """Test trigger configuration sanitization."""
        config = malicious_input["configuration"]
        sanitized = sanitize_trigger_configuration(config)
        
        assert isinstance(sanitized, dict)
        # Should be sanitized
        assert "&lt;script" in sanitized.get("script", "")


class TestSecurityUtilities:
    """Test security utility functions."""
    
    def test_create_security_report(self):
        """Test security report creation."""
        violations = [
            SecurityViolation.create(ThreatType.SCRIPT_INJECTION, "field1", "content", "critical"),
            SecurityViolation.create(ThreatType.COMMAND_INJECTION, "field2", "content", "high"),
            SecurityViolation.create(ThreatType.PATH_TRAVERSAL, "field3", "content", "medium"),
        ]
        
        report = create_security_report(violations)
        
        assert report["total_violations"] == 3
        assert report["critical_count"] == 1
        assert report["high_count"] == 1
        assert report["medium_count"] == 1
        assert ThreatType.SCRIPT_INJECTION.value in report["threat_types"]
        assert len(report["recommendations"]) <= 5
    
    def test_get_minimum_security_level(self):
        """Test minimum security level determination."""
        # Safe permissions
        safe_perms = {Permission.TEXT_INPUT, Permission.CLIPBOARD_ACCESS}
        assert get_minimum_security_level(safe_perms) == SecurityLevel.STANDARD
        
        # Dangerous permissions
        dangerous_perms = {Permission.SYSTEM_CONTROL, Permission.FILE_ACCESS}
        assert get_minimum_security_level(dangerous_perms) == SecurityLevel.STRICT


# Property-based testing for security robustness
@pytest.mark.parametrize("security_level", list(SecurityLevel))
def test_validation_robustness_across_levels(security_level, clean_input):
    """Property test: Clean input should always validate across security levels."""
    result = validate_km_input(clean_input, security_level)
    
    # Clean input should always be safe or have only minor violations
    assert result.is_safe or not result.has_critical_violations()
    assert result.sanitized_data is not None


@pytest.mark.parametrize("malicious_content", [
    "<script>alert('test')</script>",
    "javascript:void(0)",
    "'; DROP TABLE users; --",
    "../../../etc/passwd",
    "rm -rf /",
    "do shell script \"dangerous command\""
])
def test_malicious_content_detection(malicious_content):
    """Property test: Various malicious content should be detected."""
    test_input = {
        "trigger_type": "hotkey",
        "configuration": {"dangerous_field": malicious_content}
    }
    
    result = validate_km_input(test_input, SecurityLevel.STANDARD)
    
    # Should either be marked unsafe or have the content sanitized
    if result.is_safe:
        # If marked safe, content should be sanitized
        sanitized_value = result.sanitized_data["configuration"]["dangerous_field"]
        assert sanitized_value != malicious_content
    else:
        # If marked unsafe, should have violations
        assert len(result.violations) > 0


@pytest.mark.parametrize("trigger_type,expected_permission", [
    ("hotkey", Permission.TEXT_INPUT),
    ("application", Permission.APPLICATION_CONTROL),
    ("file", Permission.FILE_ACCESS),
    ("system", Permission.SYSTEM_CONTROL),
])
def test_permission_mapping_consistency(trigger_type, expected_permission):
    """Property test: Trigger types should consistently map to permissions."""
    test_data = {
        "trigger_type": trigger_type,
        "configuration": {}
    }
    
    result = sanitize_trigger_data(test_data)
    assert expected_permission in result.permissions_required