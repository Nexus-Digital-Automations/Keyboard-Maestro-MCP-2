"""
Core condition types and validation for intelligent macro automation.

This module implements the fundamental condition system that enables conditional logic
in Keyboard Maestro macros, supporting text, application, system, and variable conditions
with comprehensive security validation and functional programming patterns.
"""

from __future__ import annotations
from typing import NewType, Protocol, TypeVar, Generic, Union, Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re
import uuid
from datetime import datetime

from src.core.types import Permission, Duration
from src.core.either import Either
from src.core.errors import ValidationError, SecurityError

# Branded Types for Condition System
ConditionId = NewType('ConditionId', str)
PatternId = NewType('PatternId', str)

class ConditionType(Enum):
    """Supported condition types for macro logic."""
    TEXT = "text"
    APPLICATION = "application"
    SYSTEM = "system"
    VARIABLE = "variable"
    LOGIC = "logic"
    FILE = "file"
    TIME = "time"
    NETWORK = "network"

class ComparisonOperator(Enum):
    """Comparison operators for condition evaluation."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES_REGEX = "matches_regex"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    IN_RANGE = "in_range"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"

class LogicOperator(Enum):
    """Logic operators for combining conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"

@dataclass(frozen=True)
class ConditionSpec:
    """Type-safe condition specification with comprehensive validation."""
    condition_id: ConditionId
    condition_type: ConditionType
    operator: ComparisonOperator
    operand: str
    case_sensitive: bool = True
    negate: bool = False
    timeout_seconds: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Contract validation
        if len(self.operand) > 1000:
            raise ValueError("Operand too long (max 1000 characters)")
        if self.timeout_seconds < 1 or self.timeout_seconds > 60:
            raise ValueError("Timeout must be between 1 and 60 seconds")

@dataclass(frozen=True)
class TextCondition:
    """Text-based condition with security validation."""
    target_text: str
    operator: ComparisonOperator
    comparison_value: str
    case_sensitive: bool = True
    
    def __post_init__(self):
        if len(self.comparison_value) > 1000:
            raise ValueError("Comparison value too long")
        if self.operator == ComparisonOperator.MATCHES_REGEX:
            # Validate regex pattern for security
            RegexValidator.validate_pattern(self.comparison_value)

@dataclass(frozen=True)
class ApplicationCondition:
    """Application state condition."""
    app_identifier: str  # Bundle ID or app name
    property_name: str   # frontmost, running, window_count, etc.
    operator: ComparisonOperator
    expected_value: str
    
    def __post_init__(self):
        if len(self.app_identifier) == 0:
            raise ValueError("App identifier cannot be empty")
        if ".." in self.app_identifier or "/" in self.app_identifier:
            raise ValueError("Invalid app identifier format")

@dataclass(frozen=True)
class SystemCondition:
    """System state condition."""
    property_name: str   # time, date, battery, network, etc.
    operator: ComparisonOperator
    expected_value: str
    
    ALLOWED_PROPERTIES = {
        "current_time", "current_date", "battery_level", "network_connected",
        "wifi_connected", "display_count", "volume_level", "brightness_level",
        "idle_time", "uptime", "free_memory", "cpu_usage"
    }
    
    def __post_init__(self):
        if self.property_name not in self.ALLOWED_PROPERTIES:
            raise ValueError(f"Invalid system property: {self.property_name}")

@dataclass(frozen=True)
class VariableCondition:
    """Keyboard Maestro variable condition."""
    variable_name: str
    operator: ComparisonOperator
    comparison_value: str
    convert_to_number: bool = False
    
    def __post_init__(self):
        if len(self.variable_name) == 0:
            raise ValueError("Variable name cannot be empty")
        if len(self.variable_name) > 255:
            raise ValueError("Variable name too long")
        # Prevent injection via variable names
        if not re.match(r'^[a-zA-Z0-9_]+$', self.variable_name):
            raise ValueError("Invalid variable name format")

@dataclass(frozen=True)
class LogicCondition:
    """Composite condition with logic operators."""
    operator: LogicOperator
    conditions: List[ConditionSpec]
    
    def __post_init__(self):
        if len(self.conditions) == 0:
            raise ValueError("Logic condition requires at least one condition")
        if len(self.conditions) > 10:
            raise ValueError("Too many conditions (max 10)")
        if self.operator == LogicOperator.NOT and len(self.conditions) != 1:
            raise ValueError("NOT operator requires exactly one condition")

class RegexValidator:
    """Security-focused regex pattern validation."""
    
    DANGEROUS_PATTERNS = [
        r'\(\?\#',      # Comment group
        r'\(\?\>',      # Atomic group
        r'\(\?\<',      # Lookbehind
        r'\(\?\=',      # Lookahead
        r'\*\+',        # Nested quantifiers
        r'\+\*',        # Nested quantifiers
        r'\{\d{5,}',    # Large repetition count
    ]
    
    @staticmethod
    def validate_pattern(pattern: str) -> Either[SecurityError, str]:
        """Validate regex pattern for security vulnerabilities."""
        if len(pattern) > 500:
            return Either.left(SecurityError("REGEX_TOO_LONG", "Regex pattern too long"))
        
        # Check for dangerous patterns that could cause ReDoS
        for dangerous in RegexValidator.DANGEROUS_PATTERNS:
            if re.search(dangerous, pattern):
                return Either.left(SecurityError("DANGEROUS_REGEX", f"Dangerous regex pattern detected: {dangerous}"))
        
        # Test compile to ensure validity
        try:
            re.compile(pattern)
        except re.error as e:
            return Either.left(SecurityError("INVALID_REGEX", f"Invalid regex pattern: {str(e)}"))
        
        return Either.right(pattern)

class ConditionValidator:
    """Comprehensive condition validation with security checks."""
    
    @staticmethod
    def validate_text_condition(condition: TextCondition) -> Either[ValidationError, None]:
        """Validate text condition for security and correctness."""
        # Check for script injection patterns
        dangerous_text_patterns = [
            r'<script', r'javascript:', r'eval\s*\(', r'exec\s*\(',
            r'system\s*\(', r'shell_exec', r'passthru', r'file_get_contents'
        ]
        
        text_to_check = condition.target_text.lower() + " " + condition.comparison_value.lower()
        for pattern in dangerous_text_patterns:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                return Either.left(ValidationError("SCRIPT_INJECTION", f"Potential script injection detected: {pattern}"))
        
        # Validate regex if using regex operator
        if condition.operator == ComparisonOperator.MATCHES_REGEX:
            regex_result = RegexValidator.validate_pattern(condition.comparison_value)
            if regex_result.is_left():
                return Either.left(ValidationError("INVALID_REGEX", regex_result.get_left().message))
        
        return Either.right(None)
    
    @staticmethod
    def validate_file_path(path: str) -> Either[SecurityError, str]:
        """Validate file path to prevent directory traversal."""
        # Resolve to absolute path
        import os
        try:
            abs_path = os.path.abspath(path)
        except Exception:
            return Either.left(SecurityError("INVALID_PATH", "Invalid file path format"))
        
        # Check for forbidden paths
        forbidden_paths = [
            "/System", "/usr/bin", "/usr/sbin", "/bin", "/sbin",
            "/private/etc", "/Library/Keychains", "/var/root"
        ]
        
        for forbidden in forbidden_paths:
            if abs_path.startswith(forbidden):
                return Either.left(SecurityError("FORBIDDEN_PATH", f"Access denied to protected path: {forbidden}"))
        
        # Prevent access to parent directories
        if ".." in path:
            return Either.left(SecurityError("PATH_TRAVERSAL", "Path traversal detected"))
        
        return Either.right(abs_path)

class ConditionBuilder:
    """Fluent API for building type-safe conditions."""
    
    def __init__(self):
        self._condition_id = ConditionId(str(uuid.uuid4()))
        self._condition_type: Optional[ConditionType] = None
        self._operator: Optional[ComparisonOperator] = None
        self._operand: str = ""
        self._case_sensitive: bool = True
        self._negate: bool = False
        self._timeout_seconds: int = 10
        self._metadata: Dict[str, Any] = {}
    
    def text_condition(self, target: str) -> ConditionBuilder:
        """Create a text-based condition."""
        self._condition_type = ConditionType.TEXT
        self._metadata["target_text"] = target
        return self
    
    def app_condition(self, app_identifier: str) -> ConditionBuilder:
        """Create an application-based condition."""
        self._condition_type = ConditionType.APPLICATION
        self._metadata["app_identifier"] = app_identifier
        return self
    
    def system_condition(self, property_name: str) -> ConditionBuilder:
        """Create a system property condition."""
        self._condition_type = ConditionType.SYSTEM
        self._metadata["property_name"] = property_name
        return self
    
    def variable_condition(self, variable_name: str) -> ConditionBuilder:
        """Create a variable comparison condition."""
        self._condition_type = ConditionType.VARIABLE
        self._metadata["variable_name"] = variable_name
        return self
    
    def equals(self, value: str) -> ConditionBuilder:
        """Set equals comparison."""
        self._operator = ComparisonOperator.EQUALS
        self._operand = value
        return self
    
    def contains(self, value: str) -> ConditionBuilder:
        """Set contains comparison."""
        self._operator = ComparisonOperator.CONTAINS
        self._operand = value
        return self
    
    def matches_regex(self, pattern: str) -> ConditionBuilder:
        """Set regex matching comparison."""
        self._operator = ComparisonOperator.MATCHES_REGEX
        self._operand = pattern
        return self
    
    def greater_than(self, value: str) -> ConditionBuilder:
        """Set greater than comparison."""
        self._operator = ComparisonOperator.GREATER_THAN
        self._operand = value
        return self
    
    def case_insensitive(self) -> ConditionBuilder:
        """Make comparison case insensitive."""
        self._case_sensitive = False
        return self
    
    def negated(self) -> ConditionBuilder:
        """Negate the condition result."""
        self._negate = True
        return self
    
    def with_timeout(self, seconds: int) -> ConditionBuilder:
        """Set condition evaluation timeout."""
        self._timeout_seconds = seconds
        return self
    
    def build(self) -> Either[ValidationError, ConditionSpec]:
        """Build and validate the condition specification."""
        if self._condition_type is None:
            return Either.left(ValidationError(
                field_name="condition_type",
                value="None",
                constraint="Condition type must be specified"
            ))
        
        if self._operator is None:
            return Either.left(ValidationError(
                field_name="operator",
                value="None", 
                constraint="Comparison operator must be specified"
            ))
        
        try:
            condition = ConditionSpec(
                condition_id=self._condition_id,
                condition_type=self._condition_type,
                operator=self._operator,
                operand=self._operand,
                case_sensitive=self._case_sensitive,
                negate=self._negate,
                timeout_seconds=self._timeout_seconds,
                metadata=self._metadata.copy()
            )
            
            # Validate the built condition
            validation_result = self._validate_condition(condition)
            if validation_result.is_left():
                return Either.left(validation_result.get_left())
            
            return Either.right(condition)
            
        except ValueError as e:
            return Either.left(ValidationError(
                field_name="condition_spec",
                value="invalid",
                constraint=str(e)
            ))
    
    def _validate_condition(self, condition: ConditionSpec) -> Either[ValidationError, None]:
        """Validate the constructed condition."""
        # Type-specific validation
        if condition.condition_type == ConditionType.TEXT:
            target_text = condition.metadata.get("target_text", "")
            text_condition = TextCondition(
                target_text=target_text,
                operator=condition.operator,
                comparison_value=condition.operand,
                case_sensitive=condition.case_sensitive
            )
            return ConditionValidator.validate_text_condition(text_condition)
        
        elif condition.condition_type == ConditionType.APPLICATION:
            app_id = condition.metadata.get("app_identifier", "")
            if not app_id:
                return Either.left(ValidationError("MISSING_APP_ID", "Application identifier required"))
        
        elif condition.condition_type == ConditionType.SYSTEM:
            prop_name = condition.metadata.get("property_name", "")
            if prop_name not in SystemCondition.ALLOWED_PROPERTIES:
                return Either.left(ValidationError("INVALID_PROPERTY", f"Invalid system property: {prop_name}"))
        
        elif condition.condition_type == ConditionType.VARIABLE:
            var_name = condition.metadata.get("variable_name", "")
            if not re.match(r'^[a-zA-Z0-9_]+$', var_name):
                return Either.left(ValidationError("INVALID_VARIABLE", "Invalid variable name format"))
        
        return Either.right(None)

# Example usage functions for documentation
def create_text_contains_condition(target: str, search_value: str) -> Either[ValidationError, ConditionSpec]:
    """Example: Create a text contains condition."""
    return (ConditionBuilder()
            .text_condition(target)
            .contains(search_value)
            .build())

def create_app_running_condition(app_name: str) -> Either[ValidationError, ConditionSpec]:
    """Example: Create an application running condition."""
    return (ConditionBuilder()
            .app_condition(app_name)
            .equals("running")
            .build())

def create_variable_comparison_condition(var_name: str, value: str) -> Either[ValidationError, ConditionSpec]:
    """Example: Create a variable comparison condition."""
    return (ConditionBuilder()
            .variable_condition(var_name)
            .equals(value)
            .build())