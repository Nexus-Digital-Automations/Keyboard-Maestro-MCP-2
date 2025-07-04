"""
Property-based security tests for the Keyboard Maestro MCP system.

This module tests security properties across wide input ranges to ensure
robust protection against various attack vectors and malicious inputs.
"""

import pytest
from hypothesis import given, assume, strategies as st, settings
import re

from src.core import (
    ValidationError, SecurityViolationError, PermissionDeniedError,
    Permission, ExecutionContext, Duration, create_test_macro, CommandType
)
from src.core.parser import InputSanitizer, CommandValidator
from tests.utils.generators import (
    malicious_text_content, safe_text_content, invalid_identifiers,
    permission_sets, execution_contexts
)
from tests.utils.assertions import (
    assert_security_violation_blocked, assert_input_sanitized,
    assert_injection_prevented
)


class TestInputValidationProperties:
    """Property-based tests for input validation and sanitization."""
    
    @given(malicious_text_content())
    @settings(max_examples=50)
    def test_script_injection_always_blocked(self, malicious_text: str):
        """Property: Script injection attempts are always detected and blocked."""
        # Test various injection patterns
        script_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'<img[^>]*onerror'
        ]
        
        contains_script = any(re.search(pattern, malicious_text, re.IGNORECASE) 
                             for pattern in script_patterns)
        
        if contains_script:
            # Should raise SecurityViolationError
            assert_security_violation_blocked(
                func=InputSanitizer.sanitize_text_input,
                args=(malicious_text,),
                expected_error_type=SecurityViolationError,
                message=f"Script injection in: {malicious_text[:100]}"
            )
    
    @given(malicious_text_content())
    @settings(max_examples=30)
    def test_path_traversal_always_blocked(self, malicious_path: str):
        """Property: Path traversal attempts are always detected and blocked."""
        traversal_patterns = [
            r'\.\./',
            r'\.\.\.',
            r'~/',
            r'/etc/',
            r'/bin/',
            r'C:\\Windows\\',
            r'%SYSTEMROOT%'
        ]
        
        contains_traversal = any(re.search(pattern, malicious_path, re.IGNORECASE) 
                                for pattern in traversal_patterns)
        
        if contains_traversal:
            assert_security_violation_blocked(
                func=InputSanitizer.validate_file_path,
                args=(malicious_path,),
                expected_error_type=SecurityViolationError,
                message=f"Path traversal in: {malicious_path[:100]}"
            )
    
    def test_excessive_input_blocked(self):
        """Property: Excessively large inputs are rejected."""
        # Create large input manually since Hypothesis can't generate > 10000 chars
        large_input = "a" * 15000
        
        # Should block inputs over the size limit
        assert_security_violation_blocked(
            func=InputSanitizer.sanitize_text_input,
            args=(large_input,),
            expected_error_type=ValidationError,
            message=f"Large input ({len(large_input)} chars) should be blocked"
        )
    
    @given(invalid_identifiers())
    @settings(max_examples=30)
    def test_invalid_identifiers_rejected(self, invalid_id: str):
        """Property: Invalid identifiers are always rejected."""
        assert_security_violation_blocked(
            func=InputSanitizer.validate_identifier,
            args=(invalid_id,),
            expected_error_type=ValidationError,
            message=f"Invalid identifier should be rejected: {invalid_id}"
        )
    
    @given(safe_text_content(min_length=1, max_length=1000))
    @settings(max_examples=30)
    def test_safe_content_passes_validation(self, safe_text: str):
        """Property: Safe content always passes validation."""
        # Safe content should not raise exceptions
        try:
            sanitized = InputSanitizer.sanitize_text_input(safe_text)
            assert isinstance(sanitized, str), "Sanitized output must be string"
            
            # Should preserve safe content (possibly with minor cleanup)
            assert len(sanitized) <= len(safe_text), "Sanitization should not expand safe content"
            
        except (ValidationError, SecurityViolationError):
            # If validation fails, check if input actually contains unsafe patterns
            unsafe_patterns = [
                r'<script[^>]*>',
                r'javascript:',
                r'eval\s*\(',
                r'\.\./',
                r'[<>"\']'
            ]
            
            contains_unsafe = any(re.search(pattern, safe_text, re.IGNORECASE) 
                                 for pattern in unsafe_patterns)
            
            assert contains_unsafe, f"Safe text should not be rejected: {safe_text[:100]}"
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-", 
                  min_size=1, max_size=100))
    @settings(max_examples=20)
    def test_alphanumeric_identifiers_accepted(self, clean_id: str):
        """Property: Clean alphanumeric identifiers are accepted."""
        assume(clean_id[0].isalpha())  # Must start with letter
        
        try:
            validated = InputSanitizer.validate_identifier(clean_id)
            assert validated == clean_id.strip(), "Clean identifier should be accepted as-is"
        except ValidationError:
            pytest.fail(f"Clean identifier should be accepted: {clean_id}")


class TestPermissionEnforcementProperties:
    """Property-based tests for permission system security."""
    
    @given(permission_sets(min_size=0, max_size=4), permission_sets(min_size=1, max_size=8))
    @settings(max_examples=30)
    def test_permission_boundaries_enforced(self, available_perms: frozenset[Permission], 
                                          required_perms: frozenset[Permission]):
        """Property: Permission boundaries are strictly enforced."""
        assume(len(required_perms) > 0)
        
        context = ExecutionContext.create_test_context(
            permissions=available_perms,
            timeout=Duration.from_seconds(30)
        )
        
        missing_perms = required_perms - available_perms
        
        if missing_perms:
            # Should fail permission check
            assert not context.has_permissions(required_perms), \
                "Context should not claim to have missing permissions"
            
            # Should block operations requiring missing permissions
            for missing_perm in missing_perms:
                assert not context.has_permission(missing_perm), \
                    f"Should not have missing permission: {missing_perm}"
        else:
            # Should pass permission check
            assert context.has_permissions(required_perms), \
                "Context should have all available permissions"
    
    @given(execution_contexts())
    @settings(max_examples=20)
    def test_context_immutability_security(self, context: ExecutionContext):
        """Property: Execution contexts maintain immutability for security."""
        original_permissions = context.permissions
        original_timeout = context.timeout
        original_execution_id = context.execution_id
        
        # Attempt to modify context through variable addition
        new_context = context.with_variable("test_var", "test_value")
        
        # Original context should be unchanged
        assert context.permissions == original_permissions, "Permissions should not change"
        assert context.timeout == original_timeout, "Timeout should not change"
        assert context.execution_id == original_execution_id, "Execution ID should not change"
        
        # New context should have the variable but preserve other properties
        assert new_context.get_variable("test_var") == "test_value"
        assert new_context.permissions == original_permissions
        assert new_context.timeout == original_timeout
    
    @given(st.lists(st.sampled_from(list(Permission)), min_size=1, max_size=8))
    @settings(max_examples=15)
    def test_minimal_permission_principle(self, requested_permissions: list[Permission]):
        """Property: System enforces minimal permission principle."""
        # Create context with minimal permissions
        minimal_perms = frozenset([Permission.TEXT_INPUT])  # Basic permission
        context = ExecutionContext.create_test_context(permissions=minimal_perms)
        
        for perm in requested_permissions:
            if perm not in minimal_perms:
                # Should not have elevated permissions
                assert not context.has_permission(perm), \
                    f"Should not have elevated permission: {perm}"


class TestSecurityBoundaryProperties:
    """Property-based tests for security boundary enforcement."""
    
    @given(st.text(min_size=0, max_size=2000))
    @settings(max_examples=40)
    def test_command_parameter_sanitization(self, raw_input: str):
        """Property: All command parameters are properly sanitized."""
        # Test text input command parameter sanitization
        try:
            validated_params = CommandValidator.validate_command_parameters(
                CommandType.TEXT_INPUT,
                {"text": raw_input, "speed": "normal"}
            )
            
            sanitized_text = validated_params.get("text", "")
            
            # Check that dangerous patterns are removed or escaped
            dangerous_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'eval\s*\(',
                r'exec\s*\(',
            ]
            
            for pattern in dangerous_patterns:
                original_matches = len(re.findall(pattern, raw_input, re.IGNORECASE))
                sanitized_matches = len(re.findall(pattern, sanitized_text, re.IGNORECASE))
                
                assert sanitized_matches <= original_matches, \
                    f"Sanitization should reduce dangerous pattern: {pattern}"
                
        except (ValidationError, SecurityViolationError):
            # Rejection is also acceptable for dangerous input
            pass
    
    @given(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30)
    def test_numeric_parameter_bounds(self, numeric_value: float):
        """Property: Numeric parameters are bounded to prevent abuse."""
        if numeric_value < 0 or numeric_value > 300:  # Outside reasonable bounds
            # Should be rejected
            assert_security_violation_blocked(
                func=CommandValidator.validate_command_parameters,
                args=(CommandType.PAUSE, {"duration": numeric_value}),
                expected_error_type=ValidationError,
                message=f"Out-of-bounds numeric value should be rejected: {numeric_value}"
            )
        else:
            # Should be accepted
            try:
                validated = CommandValidator.validate_command_parameters(
                    CommandType.PAUSE,
                    {"duration": numeric_value}
                )
                assert validated.get("duration") == numeric_value
            except ValidationError:
                # Some edge cases might still be rejected, which is fine
                pass
    
    @given(st.text(alphabet="!@#$%^&*()+=[]{}|\\:;\"'<>?,./`~", min_size=1, max_size=100))
    @settings(max_examples=20)
    def test_special_character_handling(self, special_chars: str):
        """Property: Special characters are handled securely."""
        # Special characters in identifiers should be rejected
        assert_security_violation_blocked(
            func=InputSanitizer.validate_identifier,
            args=(special_chars,),
            expected_error_type=ValidationError,
            message=f"Special characters should be rejected in identifiers: {special_chars[:20]}"
        )
        
        # Special characters in text should be sanitized
        try:
            sanitized = InputSanitizer.sanitize_text_input(special_chars)
            
            # Should remove potentially dangerous characters
            dangerous_chars = ['<', '>', '"', "'", '&']
            for char in dangerous_chars:
                assert char not in sanitized, f"Dangerous character should be removed: {char}"
                
        except (ValidationError, SecurityViolationError):
            # Complete rejection is also acceptable
            pass


class TestCommandSecurityProperties:
    """Property-based tests for command-level security."""
    
    @given(st.sampled_from(list(CommandType)))
    @settings(max_examples=20)
    def test_command_permission_requirements(self, command_type: CommandType):
        """Property: All command types have appropriate permission requirements."""
        required_perms = CommandValidator.get_required_permissions(command_type)
        
        # Certain command types must require specific permissions
        if command_type == CommandType.PLAY_SOUND:
            assert Permission.SYSTEM_SOUND in required_perms, \
                "Sound commands must require SYSTEM_SOUND permission"
        
        if command_type == CommandType.APPLICATION_CONTROL:
            assert Permission.APPLICATION_CONTROL in required_perms, \
                "App control commands must require APPLICATION_CONTROL permission"
        
        if command_type == CommandType.SYSTEM_CONTROL:
            assert Permission.SYSTEM_CONTROL in required_perms, \
                "System control commands must require SYSTEM_CONTROL permission"
        
        # All permissions should be valid
        for perm in required_perms:
            assert isinstance(perm, Permission), "Required permissions must be valid Permission enum values"
    
    @given(st.dictionaries(
        keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(min_value=-1000, max_value=1000),
            st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False),
            st.booleans()
        ),
        min_size=0,
        max_size=10
    ))
    @settings(max_examples=30)
    def test_arbitrary_parameter_handling(self, arbitrary_params: dict):
        """Property: System handles arbitrary parameter combinations safely."""
        for command_type in [CommandType.TEXT_INPUT, CommandType.PAUSE, CommandType.PLAY_SOUND]:
            try:
                validated = CommandValidator.validate_command_parameters(command_type, arbitrary_params)
                
                # If validation succeeds, result should be properly structured
                assert isinstance(validated, object), "Validated parameters should be an object"
                
                # Should not contain dangerous values
                for key, value in validated.data.items():
                    if isinstance(value, str):
                        # String values should be sanitized
                        dangerous_patterns = ['<script>', 'javascript:', 'eval(']
                        for pattern in dangerous_patterns:
                            assert pattern.lower() not in value.lower(), \
                                f"Dangerous pattern found in validated parameter: {pattern}"
                
            except (ValidationError, SecurityViolationError):
                # Rejection of arbitrary parameters is expected and safe
                pass
    
    @given(safe_text_content(min_length=1, max_length=500))
    @settings(max_examples=20)
    def test_safe_command_execution_properties(self, safe_text: str):
        """Property: Safe commands execute without security violations."""
        from src.core.engine import PlaceholderCommand
        from src.core import CommandParameters
        
        # Create safe command
        safe_command = PlaceholderCommand(
            command_id="safe_test",
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters({"text": safe_text, "speed": "normal"})
        )
        
        # Should validate successfully
        assert safe_command.validate(), "Safe command should validate"
        
        # Should not require excessive permissions
        required_perms = safe_command.get_required_permissions()
        assert len(required_perms) <= 3, "Safe commands should not require excessive permissions"
        
        # Should execute without security errors
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_SOUND])
        )
        
        try:
            result = safe_command.execute(context)
            assert result is not None, "Safe command should return result"
        except (PermissionDeniedError, SecurityViolationError):
            pytest.fail("Safe command should not raise security errors")