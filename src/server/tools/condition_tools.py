"""
Condition tools for adding conditional logic to Keyboard Maestro macros.

This module implements the km_add_condition MCP tool that enables intelligent automation
through comprehensive conditional logic, supporting text, application, system, and variable
conditions with advanced security validation and functional programming patterns.
"""

from typing import Dict, Any, Optional, List
import re

from src.core.conditions import (
    ConditionBuilder, ConditionType, ComparisonOperator,
    ConditionValidator, RegexValidator
)
from src.core.types import MacroId, ConditionId
from src.core.either import Either
from src.core.errors import ValidationError, SecurityError, PermissionDeniedError
from src.integration.km_conditions import KMConditionIntegrator
from src.security.input_sanitizer import InputSanitizer
from src.core.logging import get_logger

logger = get_logger(__name__)

async def km_add_condition(
    macro_identifier: str,                    # Target macro (name or UUID)
    condition_type: str,                      # text|app|system|variable|logic
    operator: str,                           # contains|equals|greater|less|regex|exists
    operand: str,                            # Comparison value with validation
    case_sensitive: bool = True,             # Text comparison sensitivity
    negate: bool = False,                    # Invert condition result
    action_on_true: Optional[str] = None,    # Action for true condition
    action_on_false: Optional[str] = None,   # Action for false condition
    timeout_seconds: int = 10,               # Condition evaluation timeout
    ctx = None
) -> Dict[str, Any]:
    """
    Add conditional logic to a Keyboard Maestro macro for intelligent automation.
    
    This tool creates sophisticated conditional statements that enable macros to make
    intelligent decisions based on text content, application state, system properties,
    or variable values. Supports comprehensive security validation and type safety.
    
    Args:
        macro_identifier: Target macro name or UUID
        condition_type: Type of condition (text, app, system, variable, logic)
        operator: Comparison operator (contains, equals, greater, less, regex, exists)
        operand: Value to compare against
        case_sensitive: Whether text comparisons are case sensitive
        negate: Whether to invert the condition result
        action_on_true: Optional action to execute when condition is true
        action_on_false: Optional action to execute when condition is false
        timeout_seconds: Maximum time to evaluate condition
        
    Returns:
        Dict containing condition ID, validation status, and integration details
        
    Raises:
        ValidationError: If condition parameters are invalid
        SecurityError: If condition contains security risks
        PermissionDeniedError: If insufficient permissions for condition type
    """
    try:
        logger.info(f"Adding condition to macro: {macro_identifier}")
        
        # Input sanitization and validation
        sanitizer = InputSanitizer()
        
        # Sanitize string inputs
        macro_id_result = sanitizer.sanitize_macro_identifier(macro_identifier)
        if macro_id_result.is_left():
            return {
                "success": False,
                "error": "INVALID_MACRO_ID",
                "message": macro_id_result.get_left().message
            }
        
        operand_result = sanitizer.sanitize_text_content(operand, strict_mode=True)
        if operand_result.is_left():
            return {
                "success": False,
                "error": "INVALID_OPERAND",
                "message": operand_result.get_left().message
            }
        
        macro_id = MacroId(macro_id_result.get_right())
        clean_operand = operand_result.get_right()
        
        # Validate condition type and operator
        condition_type_result = _validate_condition_type(condition_type)
        if condition_type_result.is_left():
            return {
                "success": False,
                "error": "INVALID_CONDITION_TYPE",
                "message": condition_type_result.get_left().message
            }
        
        operator_result = _validate_operator(operator)
        if operator_result.is_left():
            return {
                "success": False,
                "error": "INVALID_OPERATOR", 
                "message": operator_result.get_left().message
            }
        
        condition_type_enum = condition_type_result.get_right()
        operator_enum = operator_result.get_right()
        
        # Build condition using fluent API
        builder = ConditionBuilder()
        
        # Set condition type with appropriate metadata
        if condition_type_enum == ConditionType.TEXT:
            target_text = ctx.get("clipboard_content", "") if ctx else ""
            builder = builder.text_condition(target_text)
        elif condition_type_enum == ConditionType.APPLICATION:
            app_identifier = clean_operand  # For app conditions, operand is the app
            builder = builder.app_condition(app_identifier)
        elif condition_type_enum == ConditionType.SYSTEM:
            property_name = clean_operand  # For system conditions, operand is the property
            builder = builder.system_condition(property_name)
        elif condition_type_enum == ConditionType.VARIABLE:
            variable_name = clean_operand  # For variable conditions, operand is the variable name
            builder = builder.variable_condition(variable_name)
        
        # Set operator and comparison value
        builder = _apply_operator(builder, operator_enum, clean_operand)
        
        # Set additional options
        if not case_sensitive:
            builder = builder.case_insensitive()
        
        if negate:
            builder = builder.negated()
        
        if 1 <= timeout_seconds <= 60:
            builder = builder.with_timeout(timeout_seconds)
        
        # Build and validate condition
        condition_result = builder.build()
        if condition_result.is_left():
            return {
                "success": False,
                "error": "CONDITION_BUILD_FAILED",
                "message": condition_result.get_left().message
            }
        
        condition_spec = condition_result.get_right()
        
        # Additional security validation
        security_result = _perform_security_validation(condition_spec, clean_operand)
        if security_result.is_left():
            return {
                "success": False,
                "error": "SECURITY_VIOLATION",
                "message": security_result.get_left().message
            }
        
        # Integrate with Keyboard Maestro
        integrator = KMConditionIntegrator()
        integration_result = await integrator.add_condition_to_macro(
            macro_id=macro_id,
            condition=condition_spec,
            action_on_true=action_on_true,
            action_on_false=action_on_false
        )
        
        if integration_result.is_left():
            return {
                "success": False,
                "error": "INTEGRATION_FAILED",
                "message": integration_result.get_left().message
            }
        
        integration_details = integration_result.get_right()
        
        logger.info(f"Successfully added condition {condition_spec.condition_id} to macro {macro_id}")
        
        return {
            "success": True,
            "condition_id": condition_spec.condition_id,
            "macro_id": macro_id,
            "condition_type": condition_type,
            "operator": operator,
            "operand": clean_operand,
            "case_sensitive": case_sensitive,
            "negate": negate,
            "timeout_seconds": timeout_seconds,
            "km_integration": integration_details,
            "security_validated": True,
            "created_at": condition_spec.metadata.get("created_at", ""),
            "performance_metrics": {
                "validation_time_ms": integration_details.get("validation_time_ms", 0),
                "integration_time_ms": integration_details.get("integration_time_ms", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error adding condition to macro {macro_identifier}: {str(e)}")
        return {
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": f"Failed to add condition: {str(e)}"
        }

def _validate_condition_type(condition_type: str) -> Either[ValidationError, ConditionType]:
    """Validate and convert condition type string."""
    try:
        return Either.right(ConditionType(condition_type.lower()))
    except ValueError:
        valid_types = [t.value for t in ConditionType]
        return Either.left(ValidationError(
            "INVALID_CONDITION_TYPE",
            f"Invalid condition type '{condition_type}'. Valid types: {valid_types}"
        ))

def _validate_operator(operator: str) -> Either[ValidationError, ComparisonOperator]:
    """Validate and convert operator string."""
    try:
        return Either.right(ComparisonOperator(operator.lower()))
    except ValueError:
        valid_operators = [op.value for op in ComparisonOperator]
        return Either.left(ValidationError(
            "INVALID_OPERATOR",
            f"Invalid operator '{operator}'. Valid operators: {valid_operators}"
        ))

def _apply_operator(builder: ConditionBuilder, operator: ComparisonOperator, operand: str) -> ConditionBuilder:
    """Apply the specified operator to the condition builder."""
    if operator == ComparisonOperator.EQUALS:
        return builder.equals(operand)
    elif operator == ComparisonOperator.CONTAINS:
        return builder.contains(operand)
    elif operator == ComparisonOperator.MATCHES_REGEX:
        return builder.matches_regex(operand)
    elif operator == ComparisonOperator.GREATER_THAN:
        return builder.greater_than(operand)
    else:
        # Default to equals for other operators
        return builder.equals(operand)

def _perform_security_validation(condition_spec, operand: str) -> Either[SecurityError, None]:
    """Perform additional security validation on the condition."""
    # Validate regex patterns for ReDoS attacks
    if condition_spec.operator == ComparisonOperator.MATCHES_REGEX:
        regex_result = RegexValidator.validate_pattern(operand)
        if regex_result.is_left():
            return Either.left(SecurityError(
                "DANGEROUS_REGEX",
                f"Regex pattern validation failed: {regex_result.get_left().message}"
            ))
    
    # Check for injection patterns in operand
    dangerous_patterns = [
        r'<script', r'javascript:', r'eval\s*\(', r'exec\s*\(',
        r'system\s*\(', r'shell_exec', r'`[^`]*`'  # Command injection
    ]
    
    operand_lower = operand.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, operand_lower):
            return Either.left(SecurityError(
                "INJECTION_DETECTED",
                f"Potential code injection detected in operand: {pattern}"
            ))
    
    # Validate file paths for system conditions
    if condition_spec.condition_type == ConditionType.SYSTEM:
        property_name = condition_spec.metadata.get("property_name", "")
        if "file" in property_name.lower():
            path_result = ConditionValidator.validate_file_path(operand)
            if path_result.is_left():
                return Either.left(path_result.get_left())
    
    return Either.right(None)

# Register the tool with the MCP server
def register_condition_tools(server: Server):
    """Register condition-related tools with the MCP server."""
    server.add_tool(km_add_condition)