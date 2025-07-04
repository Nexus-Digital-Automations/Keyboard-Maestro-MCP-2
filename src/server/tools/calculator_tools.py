"""
Calculator Tools - MCP Tool Implementation

Provides mathematical calculation capabilities with comprehensive security validation,
KM token support, and multiple result formatting options.
"""

from __future__ import annotations
import uuid
from datetime import datetime, UTC
from typing import Dict, Any

from ...calculations.calculator import (
    Calculator, 
    CalculationExpression, 
    NumberFormat
)
from ...calculations.km_math_integration import KMTokenCalculator
from ...integration.km_client import KMError


async def km_calculator(
    expression: str,
    variables: Dict[str, float],
    format_result: str,
    precision: int,
    use_km_engine: bool,
    validate_only: bool,
    ctx = None
) -> Dict[str, Any]:
    """
    Evaluate mathematical expressions with comprehensive security and token support.
    
    Features:
    - Secure expression evaluation using AST parsing (no eval())
    - Support for variables in expressions
    - Multiple result formatting options (decimal, scientific, percentage, currency)
    - Integration with Keyboard Maestro's calculation engine
    - Token processing for dynamic expressions
    - Validation mode for expression checking without evaluation
    
    Security:
    - Expression validation against code injection
    - Whitelist-based function calls
    - Result bounds checking
    - Safe variable substitution
    
    Returns calculation results with metadata and security validation status.
    """
    if ctx:
        await ctx.info(f"Evaluating mathematical expression: {expression[:50]}...")
    
    try:
        # Input validation
        if not expression.strip():
            return {
                "success": False,
                "error": {
                    "code": "INVALID_EXPRESSION",
                    "message": "Expression cannot be empty",
                    "details": {"expression": expression}
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
        
        # Create calculation expression with validation
        try:
            calc_expression = CalculationExpression(
                expression=expression.strip(),
                variables=variables
            )
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "EXPRESSION_VALIDATION_ERROR",
                    "message": f"Expression validation failed: {str(e)}",
                    "details": {"expression": expression}
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
        
        # Validation-only mode
        if validate_only:
            return {
                "success": True,
                "validation": {
                    "expression": expression,
                    "is_valid": True,
                    "variables_required": list(variables.keys()),
                    "contains_functions": calc_expression.contains_functions(),
                    "security_status": "safe"
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "validation_only": True
                }
            }
        
        # Determine calculation method
        if use_km_engine and calc_expression.contains_km_tokens():
            # Use KM token calculator for expressions with tokens
            if ctx:
                await ctx.info("Using Keyboard Maestro token calculator")
            
            km_calculator = KMTokenCalculator()
            km_result = await km_calculator.calculate_with_tokens(expression, "calculation")
            
            if km_result.is_left():
                # Fallback to local calculator
                calculator = Calculator()
                calc_result = await calculator.evaluate(calc_expression)
            else:
                # Use KM result
                km_value = km_result.get_right()
                try:
                    result_value = float(km_value)
                    from ...calculations.calculator import CalculationResult
                    calc_result = Calculator()._create_either_right(
                        CalculationResult(
                            value=result_value,
                            formatted_value=km_value,
                            format_used=ResultFormat.from_string(format_result),
                            expression=expression,
                            variables_used=variables,
                            execution_time=0.0
                        )
                    )
                except ValueError:
                    # KM returned non-numeric result, treat as error
                    calc_result = Calculator()._create_either_left(
                        KMError.execution_error(f"KM returned non-numeric result: {km_value}")
                    )
        else:
            # Use local calculator
            if ctx:
                await ctx.info("Using local calculation engine")
            
            calculator = Calculator()
            calc_result = await calculator.evaluate(calc_expression)
        
        # Process calculation result
        if calc_result.is_left():
            error = calc_result.get_left()
            return {
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": {
                        "expression": expression,
                        "variables": variables
                    }
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "calculation_id": str(uuid.uuid4())
                }
            }
        
        result = calc_result.get_right()
        
        # Apply result formatting if different from what was calculated
        if format_result != "auto" and result.format_used.value != format_result:
            calculator = Calculator()
            formatted_result = calculator.format_result(
                result.value, 
                ResultFormat.from_string(format_result),
                precision
            )
            result = result._replace(
                formatted_value=formatted_result,
                format_used=ResultFormat.from_string(format_result)
            )
        
        if ctx:
            await ctx.info(f"Calculation complete: {result.formatted_value}")
        
        return {
            "success": True,
            "calculation": {
                "expression": result.expression,
                "result": result.value,
                "formatted_result": result.formatted_value,
                "format": result.format_used.value,
                "precision": precision,
                "variables_used": result.variables_used,
                "execution_time": result.execution_time,
                "contains_tokens": calc_expression.contains_km_tokens(),
                "engine_used": "keyboard_maestro" if use_km_engine and calc_expression.contains_km_tokens() else "local"
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "calculation_id": str(uuid.uuid4()),
                "security_validated": True
            }
        }
        
    except Exception as e:
        # Comprehensive error handling
        return {
            "success": False,
            "error": {
                "code": "CALCULATION_ERROR",
                "message": str(e),
                "details": {
                    "expression": expression,
                    "variables": variables,
                    "format_result": format_result
                }
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat()
            }
        }