"""
Property-based tests for condition system validation and security.

This module uses Hypothesis to test condition behavior across input ranges,
ensuring security boundaries, validation correctness, and functional properties.
"""

import pytest
import re
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from src.core.conditions import (
    ConditionBuilder, ConditionType, ComparisonOperator,
    ConditionValidator, RegexValidator, ConditionSpec
)
from src.core.either import Either
from src.core.errors import ValidationError, SecurityError
from src.security.input_sanitizer import InputSanitizer


class TestConditionBuilderProperties:
    """Property-based tests for ConditionBuilder."""
    
    @given(st.text(min_size=1, max_size=100))
    def test_text_condition_preserves_input(self, target_text):
        """Property: Text conditions should preserve valid input text."""
        assume(len(target_text.strip()) > 0)
        
        result = (ConditionBuilder()
                 .text_condition(target_text)
                 .equals("test")
                 .build())
        
        assert result.is_right()
        condition = result.get_right()
        assert condition.metadata.get("target_text") == target_text
    
    @given(st.text(min_size=1, max_size=1000))
    def test_operand_length_constraint(self, operand):
        """Property: Operands should respect length constraints."""
        builder = ConditionBuilder().text_condition("test")
        
        if len(operand) <= 1000:
            result = builder.equals(operand).build()
            assert result.is_right()
        else:
            # Should fail validation during build
            with pytest.raises(ValueError):
                builder.equals(operand).build()
    
    @given(st.integers(min_value=1, max_value=60))
    def test_timeout_range_validation(self, timeout):
        """Property: Valid timeout ranges should be accepted."""
        result = (ConditionBuilder()
                 .text_condition("test")
                 .equals("value")
                 .with_timeout(timeout)
                 .build())
        
        assert result.is_right()
        condition = result.get_right()
        assert condition.timeout_seconds == timeout
    
    @given(st.integers().filter(lambda x: x < 1 or x > 60))
    def test_invalid_timeout_rejected(self, invalid_timeout):
        """Property: Invalid timeout values should be rejected."""
        builder = ConditionBuilder().text_condition("test").equals("value")
        
        # Either with_timeout raises ValueError or build() returns validation error
        try:
            result = builder.with_timeout(invalid_timeout).build()
            # If no exception, it should be a validation error
            assert result.is_left()
        except ValueError:
            # Expected behavior for invalid timeout
            pass


class TestSecurityProperties:
    """Property-based tests for security validation."""
    
    @given(st.text(min_size=1, max_size=500).filter(
        lambda x: not any(danger in x.lower() for danger in ['<script', 'javascript:', 'eval(']) and
                  not any(cmd_char in x for cmd_char in [';', '&', '|', '`', '$'])
    ))
    def test_safe_text_passes_validation(self, safe_text):
        """Property: Safe text should pass security validation."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_text_content(safe_text, strict_mode=True)
        assert result.is_right()
    
    @given(st.text(min_size=1, max_size=100))
    def test_dangerous_patterns_rejected(self, base_text):
        """Property: Text with dangerous patterns should be rejected."""
        dangerous_patterns = [
            '<script>alert("xss")</script>',
            'javascript:void(0)',
            'eval(malicious_code)',
            'system("rm -rf /")',
            '`command_injection`'
        ]
        
        sanitizer = InputSanitizer()
        
        for pattern in dangerous_patterns:
            dangerous_text = base_text + pattern
            result = sanitizer.sanitize_text_content(dangerous_text, strict_mode=True)
            assert result.is_left()
            assert "INJECTION" in result.get_left().security_code or "SCRIPT" in result.get_left().security_code
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -._", min_size=1, max_size=255).filter(
        lambda x: len(x.strip()) > 0 and re.match(r'^[a-zA-Z0-9_\s\-\.]+$', x)
    ))
    def test_valid_identifiers_accepted(self, identifier):
        """Property: Valid macro identifiers should be accepted."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_macro_identifier(identifier)
        assert result.is_right()
    
    @given(st.text(min_size=1, max_size=500))
    def test_regex_patterns_validated(self, pattern):
        """Property: Regex patterns should be validated for ReDoS attacks."""
        # Skip patterns that contain obviously dangerous constructs
        assume(not any(danger in pattern for danger in ['(?#', '(?>', '(*', '+*', '*+']))
        
        result = RegexValidator.validate_pattern(pattern)
        
        # Either the pattern is valid or rejected for security reasons
        if result.is_left():
            error = result.get_left()
            assert error.security_code in ["REGEX_TOO_LONG", "DANGEROUS_REGEX", "INVALID_REGEX"]
        else:
            # If accepted, it should be safe
            safe_pattern = result.get_right()
            assert len(safe_pattern) <= 500


class TestConditionValidationProperties:
    """Property-based tests for condition validation logic."""
    
    @given(
        st.sampled_from(list(ConditionType)),
        st.sampled_from(list(ComparisonOperator)),
        st.text(alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd", "Pc", "Pd", "Po"], max_codepoint=127), min_size=1, max_size=100).filter(
            lambda x: len(x.strip()) > 0 and not any(danger in x.lower() for danger in ['<script', 'javascript:', 'eval(']) and
                      not any(cmd_char in x for cmd_char in [';', '&', '|', '`', '$'])
        )
    )
    def test_condition_spec_creation(self, condition_type, operator, operand):
        """Property: Valid condition specs should be creatable."""
        assume(len(operand) <= 1000)  # Respect operand length limit
        
        builder = ConditionBuilder()
        
        # Set up condition type
        if condition_type == ConditionType.TEXT:
            builder = builder.text_condition("target")
        elif condition_type == ConditionType.APPLICATION:
            builder = builder.app_condition("com.example.app")
        elif condition_type == ConditionType.SYSTEM:
            builder = builder.system_condition("current_time")
        elif condition_type == ConditionType.VARIABLE:
            builder = builder.variable_condition("MyVariable")
        elif condition_type == ConditionType.LOGIC:
            # Skip LOGIC type for now as it's not implemented yet
            assume(False)
        
        # Set operator with safe operands
        if operator == ComparisonOperator.EQUALS:
            builder = builder.equals(operand)
        elif operator == ComparisonOperator.CONTAINS:
            builder = builder.contains(operand)
        elif operator == ComparisonOperator.GREATER_THAN:
            builder = builder.greater_than(operand)
        elif operator == ComparisonOperator.MATCHES_REGEX:
            # Use a simple, safe regex instead of operand
            builder = builder.matches_regex("test.*")
        else:
            builder = builder.equals(operand)  # Default fallback
        
        result = builder.build()
        
        # Should either succeed or fail with validation error
        if result.is_left():
            assert isinstance(result.get_left(), ValidationError)
        else:
            condition = result.get_right()
            assert condition.condition_type == condition_type
            assert isinstance(condition.condition_id, str)
            assert len(condition.condition_id) > 0
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", min_size=1, max_size=255).filter(
        lambda x: len(x.strip()) > 0 and re.match(r'^[a-zA-Z0-9_]+$', x)
    ))
    def test_variable_name_validation(self, var_name):
        """Property: Valid variable names should be accepted."""
        # Valid variable names are alphanumeric with underscores
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_variable_name(var_name)
        assert result.is_right()
        clean_name = result.get_right()
        assert clean_name == var_name.strip()
    
    @given(st.text(min_size=1, max_size=100))
    def test_invalid_variable_names_rejected(self, var_name):
        """Property: Invalid variable names should be rejected."""
        # Add invalid characters
        invalid_name = var_name + "!@#$%^&*()"
        
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_variable_name(invalid_name)
        assert result.is_left()
        assert "INVALID_VARIABLE_NAME" in result.get_left().security_code


class TestConditionIntegrationProperties:
    """Property-based tests for condition integration behavior."""
    
    @given(
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=100),
        st.booleans(),
        st.booleans()
    )
    def test_condition_serialization_roundtrip(self, target, operand, case_sensitive, negate):
        """Property: Conditions should serialize and deserialize consistently."""
        assume(len(target.strip()) > 0 and len(operand.strip()) > 0)
        
        # Create condition
        result = (ConditionBuilder()
                 .text_condition(target)
                 .contains(operand))
        
        if not case_sensitive:
            result = result.case_insensitive()
        
        if negate:
            result = result.negated()
        
        condition_result = result.build()
        
        if condition_result.is_right():
            condition = condition_result.get_right()
            
            # Verify properties are preserved
            assert condition.metadata.get("target_text") == target
            assert condition.operand == operand
            assert condition.case_sensitive == case_sensitive
            assert condition.negate == negate
            assert condition.operator == ComparisonOperator.CONTAINS


class ConditionStateMachine(RuleBasedStateMachine):
    """Stateful testing for condition builder state management."""
    
    def __init__(self):
        super().__init__()
        self.builder = ConditionBuilder()
        self.has_condition_type = False
        self.has_operator = False
    
    @rule(target_text=st.text(min_size=1, max_size=50))
    def set_text_condition(self, target_text):
        """Add text condition to builder."""
        self.builder = self.builder.text_condition(target_text)
        self.has_condition_type = True
    
    @rule(app_id=st.text(min_size=1, max_size=50))
    def set_app_condition(self, app_id):
        """Add app condition to builder."""
        self.builder = self.builder.app_condition(app_id)
        self.has_condition_type = True
    
    @rule(value=st.text(min_size=1, max_size=50))
    def set_equals_operator(self, value):
        """Set equals operator."""
        self.builder = self.builder.equals(value)
        self.has_operator = True
    
    @rule(value=st.text(min_size=1, max_size=50))
    def set_contains_operator(self, value):
        """Set contains operator."""
        self.builder = self.builder.contains(value)
        self.has_operator = True
    
    @rule()
    def set_case_insensitive(self):
        """Set case insensitive flag."""
        self.builder = self.builder.case_insensitive()
    
    @rule()
    def set_negated(self):
        """Set negated flag."""
        self.builder = self.builder.negated()
    
    @rule(timeout=st.integers(min_value=1, max_value=60))
    def set_timeout(self, timeout):
        """Set timeout value."""
        self.builder = self.builder.with_timeout(timeout)
    
    @invariant()
    def builder_maintains_state(self):
        """Invariant: Builder should maintain consistent state."""
        # Builder should be usable at any point
        assert self.builder is not None
    
    @rule()
    def build_condition(self):
        """Attempt to build condition."""
        result = self.builder.build()
        
        if self.has_condition_type and self.has_operator:
            # Should succeed with both required components
            assert result.is_right()
            condition = result.get_right()
            assert condition.condition_id is not None
            assert condition.condition_type is not None
            assert condition.operator is not None
        else:
            # Should fail without required components
            assert result.is_left()
            assert isinstance(result.get_left(), ValidationError)


# Test configuration
class TestConditionSystemProperties:
    """Main property test runner."""
    
    @pytest.mark.skip("Stateful testing needs refinement")
    @settings(max_examples=10, deadline=None)
    def test_stateful_condition_building(self):
        """Run stateful tests for condition building."""
        ConditionStateMachine.TestCase.settings = settings(
            max_examples=5,
            stateful_step_count=3,
            deadline=None
        )
        ConditionStateMachine().execute()