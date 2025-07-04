"""
Secure Mathematical Calculator with Expression Evaluation

Provides comprehensive mathematical operations with security validation,
format conversion, and integration with Keyboard Maestro's calculation engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from enum import Enum
import re
import math
import ast
import operator
import time

from ..core.types import Duration
from ..core.contracts import require, ensure
from ..integration.km_client import Either, KMError


class NumberFormat(Enum):
    """Supported number formats for input and output."""
    DECIMAL = "decimal"
    HEXADECIMAL = "hex"
    BINARY = "binary"
    SCIENTIFIC = "scientific"
    PERCENTAGE = "percentage"


@dataclass(frozen=True)
class CalculationExpression:
    """Type-safe mathematical expression with security validation."""
    expression: str
    variables: Dict[str, float] = field(default_factory=dict)
    
    @require(lambda self: len(self.expression) > 0 and len(self.expression) <= 1000)
    @require(lambda self: self._is_safe_expression(self.expression))
    def __post_init__(self):
        """Post-initialization validation for expression safety."""
        pass
    
    def _is_safe_expression(self, expr: str) -> bool:
        """Validate expression contains only safe mathematical operations."""
        # Allow only numbers, operators, parentheses, and whitelisted functions
        safe_pattern = r'^[0-9\+\-\*\/\(\)\.\s\,a-zA-Z_]+$'
        if not re.match(safe_pattern, expr):
            return False
        
        # Whitelist of allowed functions
        allowed_functions = {
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
            'log', 'log10', 'exp', 'sqrt', 'pow', 'abs', 'round',
            'floor', 'ceil', 'min', 'max', 'sum', 'pi', 'e'
        }
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'__\w+__',        # Python magic methods
            r'import\s+',      # Import statements
            r'exec\s*\(',      # Exec function
            r'eval\s*\(',      # Eval function
            r'open\s*\(',      # File operations
            r'input\s*\(',     # Input functions
            r'subprocess',     # System calls
            r'os\.',          # OS module access
            r'sys\.',         # System module access
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expr, re.IGNORECASE):
                return False
        
        return True


@dataclass(frozen=True)
class CalculationResult:
    """Type-safe calculation result with formatting options."""
    result: float
    formatted_result: str
    expression: str
    format: NumberFormat
    execution_time: float
    variables_used: Dict[str, float] = field(default_factory=dict)
    
    def to_format(self, target_format: NumberFormat) -> str:
        """Convert result to specified format."""
        if target_format == NumberFormat.DECIMAL:
            return str(self.result)
        elif target_format == NumberFormat.HEXADECIMAL:
            try:
                return hex(int(self.result))
            except (ValueError, OverflowError):
                return f"0x{int(self.result):x}" if abs(self.result) < 2**63 else "overflow"
        elif target_format == NumberFormat.BINARY:
            try:
                return bin(int(self.result))
            except (ValueError, OverflowError):
                return f"0b{int(self.result):b}" if abs(self.result) < 2**63 else "overflow"
        elif target_format == NumberFormat.SCIENTIFIC:
            return f"{self.result:.6e}"
        elif target_format == NumberFormat.PERCENTAGE:
            return f"{self.result * 100:.2f}%"
        else:
            return str(self.result)


class SafeExpressionEvaluator:
    """Safe mathematical expression evaluator using AST parsing."""
    
    # Supported operators
    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
    }
    
    # Supported functions
    _functions = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'sqrt': math.sqrt,
        'abs': abs,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
        'min': min,
        'max': max,
        'sum': sum,
    }
    
    # Mathematical constants
    _constants = {
        'pi': math.pi,
        'e': math.e,
    }
    
    def evaluate(self, expression: str, variables: Dict[str, float] = None) -> float:
        """Safely evaluate mathematical expression using AST parsing."""
        if variables is None:
            variables = {}
        
        try:
            # Parse expression into AST
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body, variables)
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {str(e)}")
    
    def _eval_node(self, node: ast.AST, variables: Dict[str, float]) -> float:
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return float(node.value)
        elif isinstance(node, ast.Num):  # Python < 3.8
            return float(node.n)
        elif isinstance(node, ast.Name):
            # Variable or constant lookup
            name = node.id
            if name in variables:
                return float(variables[name])
            elif name in self._constants:
                return self._constants[name]
            else:
                raise ValueError(f"Unknown variable or constant: {name}")
        elif isinstance(node, ast.BinOp):
            # Binary operation
            left = self._eval_node(node.left, variables)
            right = self._eval_node(node.right, variables)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
            try:
                return op(left, right)
            except ZeroDivisionError:
                raise ValueError("Division by zero")
            except (OverflowError, ValueError) as e:
                raise ValueError(f"Mathematical error: {str(e)}")
        elif isinstance(node, ast.UnaryOp):
            # Unary operation
            operand = self._eval_node(node.operand, variables)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operation: {type(node.op)}")
            return op(operand)
        elif isinstance(node, ast.Call):
            # Function call
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self._functions:
                raise ValueError(f"Unsupported function: {func_name}")
            
            args = [self._eval_node(arg, variables) for arg in node.args]
            try:
                return self._functions[func_name](*args)
            except Exception as e:
                raise ValueError(f"Function {func_name} error: {str(e)}")
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")


class Calculator:
    """Secure mathematical calculator with expression evaluation."""
    
    def __init__(self):
        self.evaluator = SafeExpressionEvaluator()
    
    @require(lambda expression: expression.expression != "")
    @ensure(lambda result: result.is_right() or result.get_left().code in ["CALCULATION_ERROR", "SECURITY_ERROR"])
    async def calculate(self, expression: CalculationExpression) -> Either[KMError, CalculationResult]:
        """Evaluate mathematical expression with security validation."""
        start_time = time.time()
        
        try:
            # Additional security validation
            if not self._validate_expression_security(expression.expression):
                return Either.left(KMError.security_error("Expression contains dangerous patterns"))
            
            # Preprocess expression for common mathematical notation
            processed_expr = self._preprocess_expression(expression.expression)
            
            # Evaluate using safe AST evaluator
            result = self.evaluator.evaluate(processed_expr, expression.variables)
            
            # Validate result is within safe bounds
            if not self._validate_result(result):
                return Either.left(KMError.validation_error("Result exceeds safe numerical bounds"))
            
            execution_time = time.time() - start_time
            
            # Format result
            formatted_result = self._format_result(result, NumberFormat.DECIMAL)
            
            calculation_result = CalculationResult(
                result=result,
                formatted_result=formatted_result,
                expression=expression.expression,
                format=NumberFormat.DECIMAL,
                execution_time=execution_time,
                variables_used=expression.variables
            )
            
            return Either.right(calculation_result)
            
        except ValueError as e:
            return Either.left(KMError.validation_error(f"Calculation error: {str(e)}"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Unexpected calculation error: {str(e)}"))
    
    def _validate_expression_security(self, expr: str) -> bool:
        """Additional comprehensive security validation."""
        # Check expression length
        if len(expr) > 1000:
            return False
        
        # Check for nested function calls (potential DoS)
        paren_depth = 0
        max_depth = 10
        for char in expr:
            if char == '(':
                paren_depth += 1
                if paren_depth > max_depth:
                    return False
            elif char == ')':
                paren_depth -= 1
        
        # Check for dangerous character sequences
        dangerous_sequences = ['\\', '`', '$', '&', '|', ';', '>', '<']
        if any(seq in expr for seq in dangerous_sequences):
            return False
        
        return True
    
    def _preprocess_expression(self, expr: str) -> str:
        """Preprocess expression for common mathematical notation."""
        # Replace common mathematical constants and functions
        replacements = {
            'π': 'pi',
            '²': '**2',
            '³': '**3',
        }
        
        processed = expr
        for old, new in replacements.items():
            processed = processed.replace(old, new)
        
        # Handle implicit multiplication (e.g., "2x" -> "2*x")
        processed = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', processed)
        processed = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', processed)
        
        return processed
    
    def _validate_result(self, result: float) -> bool:
        """Validate calculation result is within safe bounds."""
        # Check for infinity or NaN
        if math.isinf(result) or math.isnan(result):
            return False
        
        # Check for extremely large numbers that could cause issues
        if abs(result) > 1e308:  # Near float64 limit
            return False
        
        return True
    
    def _format_result(self, result: float, format_type: NumberFormat) -> str:
        """Format calculation result in specified format."""
        if format_type == NumberFormat.DECIMAL:
            # Use appropriate precision for display
            if abs(result) < 1e-10:
                return "0"
            elif abs(result) > 1e10 or abs(result) < 1e-3:
                return f"{result:.6e}"
            else:
                return f"{result:.10g}"
        elif format_type == NumberFormat.SCIENTIFIC:
            return f"{result:.6e}"
        else:
            return str(result)