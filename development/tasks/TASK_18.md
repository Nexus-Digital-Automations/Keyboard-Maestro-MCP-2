# TASK_18: km_calculator - Mathematical Operations

**Created By**: Agent_ADDER+ (High-Impact Tool Implementation) | **Priority**: MEDIUM | **Duration**: 2 hours
**Technique Focus**: Expression Security + Mathematical Validation + Type Safety + Parser Design
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED ✅ (MCP Tool Registered Successfully)
**Assigned**: Agent_12
**Dependencies**: TASK_10 (macro creation for calculation workflows)
**Blocking**: None (standalone calculation functionality)
**Completion**: All calculation functionality implemented and registered in main.py:450-484

## 📖 Required Reading (Complete before starting)
- [x] **development/protocols/KM_MCP.md**: km_calculator specification (lines 994-1005) ✅
- [x] **src/creation/**: Macro creation patterns from TASK_10 ✅
- [x] **Keyboard Maestro Calculate Function**: Understanding KM's built-in calculation engine ✅
- [x] **Mathematical Security**: Expression parsing security and injection prevention ✅
- [x] **tests/TESTING.md**: Mathematical validation and expression testing ✅

## 🎯 Implementation Overview
Create a secure mathematical operations engine that enables AI assistants to perform calculations with expression parsing, format conversion, and integration with Keyboard Maestro's calculation system while maintaining security boundaries against code injection.

<thinking>
Mathematical operations need security and validation:
1. Expression Security: Prevent code injection through mathematical expressions
2. Parser Safety: Secure parsing of mathematical expressions and functions
3. Format Support: Multiple number formats (decimal, hex, binary, scientific)
4. Function Library: Standard mathematical functions with validation
5. Variable Support: Integration with KM variables in calculations
6. Error Handling: Graceful handling of division by zero, overflow, etc.
</thinking>

## ✅ Implementation Subtasks (Sequential completion) - ALL COMPLETED ✅

### Phase 1: Core Calculation Infrastructure - COMPLETED ✅
- [x] **Calculation types**: Define Expression, CalculationResult, NumberFormat, MathFunction types ✅
- [x] **Expression security**: Safe expression parsing with whitelist validation ✅
- [x] **Parser implementation**: Mathematical expression parser with security boundaries ✅
- [x] **Format handling**: Support for decimal, hexadecimal, binary, scientific notation ✅

### Phase 2: Mathematical Functions & Operations - COMPLETED ✅
- [x] **Basic operations**: Addition, subtraction, multiplication, division with overflow checking ✅
- [x] **Advanced functions**: Trigonometric, logarithmic, exponential functions ✅
- [x] **Utility functions**: Rounding, absolute value, modulo, power operations ✅
- [x] **KM integration**: Interface with Keyboard Maestro's calculation engine ✅

### Phase 3: Security & Validation - COMPLETED ✅
- [x] **Expression validation**: Prevent code injection and dangerous function calls ✅
- [x] **Input sanitization**: Clean mathematical expressions for safe evaluation ✅
- [x] **Result validation**: Verify calculation results are within safe bounds ✅
- [x] **Error handling**: Comprehensive error handling for mathematical edge cases ✅

### Phase 4: MCP Tool Integration - COMPLETED ✅
- [x] **Tool implementation**: km_calculator MCP tool with expression evaluation ✅
- [x] **Format support**: Multiple input/output number formats ✅
- [x] **Response formatting**: Calculation results with metadata and validation status ✅
- [x] **Testing integration**: Mathematical property testing and edge case validation ✅

## 🔧 Implementation Files & Specifications

### New Files to Create:

#### src/calculations/calculator.py - Core Calculation Engine
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import re
import math

class NumberFormat(Enum):
    """Supported number formats."""
    DECIMAL = "decimal"
    HEXADECIMAL = "hex"
    BINARY = "binary"
    SCIENTIFIC = "scientific"
    PERCENTAGE = "percentage"

@dataclass(frozen=True)
class CalculationExpression:
    """Type-safe mathematical expression."""
    expression: str
    variables: Dict[str, float] = field(default_factory=dict)
    
    @require(lambda self: len(self.expression) > 0 and len(self.expression) <= 1000)
    @require(lambda self: self._is_safe_expression(self.expression))
    def __post_init__(self):
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
            'floor', 'ceil', 'min', 'max', 'sum'
        }
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'__\w+__',  # Python magic methods
            r'import\s+',  # Import statements
            r'exec\s*\(',  # Exec function
            r'eval\s*\(',  # Eval function
            r'open\s*\(',  # File operations
            r'input\s*\(',  # Input functions
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expr, re.IGNORECASE):
                return False
        
        return True

@dataclass(frozen=True)
class CalculationResult:
    """Type-safe calculation result."""
    result: float
    formatted_result: str
    expression: str
    format: NumberFormat
    execution_time: float
    
    def to_format(self, target_format: NumberFormat) -> str:
        """Convert result to specified format."""
        if target_format == NumberFormat.DECIMAL:
            return str(self.result)
        elif target_format == NumberFormat.HEXADECIMAL:
            return hex(int(self.result))
        elif target_format == NumberFormat.BINARY:
            return bin(int(self.result))
        elif target_format == NumberFormat.SCIENTIFIC:
            return f"{self.result:.6e}"
        elif target_format == NumberFormat.PERCENTAGE:
            return f"{self.result * 100:.2f}%"
        else:
            return str(self.result)

class Calculator:
    """Secure mathematical calculator with expression evaluation."""
    
    @require(lambda expression: expression.expression != "")
    @ensure(lambda result: result.is_right() or result.get_left().code in ["CALCULATION_ERROR", "SECURITY_ERROR"])
    async def calculate(self, expression: CalculationExpression) -> Either[KMError, CalculationResult]:
        """Evaluate mathematical expression with security validation."""
        pass
    
    def _validate_expression_security(self, expr: str) -> bool:
        """Comprehensive security validation for mathematical expressions."""
        pass
    
    def _parse_and_evaluate(self, expr: str, variables: Dict[str, float]) -> float:
        """Parse and evaluate mathematical expression safely."""
        pass
    
    def _format_result(self, result: float, format: NumberFormat) -> str:
        """Format calculation result in specified format."""
        pass
```

#### src/calculations/km_math_integration.py - Keyboard Maestro Integration
```python
class KMCalculationEngine:
    """Integration with Keyboard Maestro's calculation system."""
    
    async def evaluate_with_km(self, expression: str, variables: Dict[str, str] = None) -> Either[KMError, str]:
        """Evaluate expression using KM's calculation engine."""
        # Build AppleScript to use KM's calculate function
        script = f'''
        tell application "Keyboard Maestro Engine"
            try
                set result to calculate "{expression}"
                return result as string
            on error errorMessage
                return "ERROR: " & errorMessage
            end try
        end tell
        '''
        
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return Either.left(KMError.execution_error(f"KM calculation failed: {result.stderr}"))
            
            output = result.stdout.strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:].strip()))
            
            return Either.right(output)
            
        except subprocess.TimeoutExpired:
            return Either.left(KMError.timeout_error("KM calculation timeout"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"KM calculation error: {str(e)}"))
```

#### src/server/tools/calculator_tools.py - MCP Tool Implementation
```python
async def km_calculator(
    expression: Annotated[str, Field(
        description="Mathematical expression to evaluate",
        min_length=1,
        max_length=1000,
        pattern=r"^[0-9\+\-\*\/\(\)\.\s\,a-zA-Z_]+$"
    )],
    variables: Annotated[Dict[str, float], Field(
        default_factory=dict,
        description="Variable values for substitution"
    )] = {},
    output_format: Annotated[str, Field(
        default="decimal",
        description="Output number format",
        pattern=r"^(decimal|hex|binary|scientific|percentage)$"
    )] = "decimal",
    precision: Annotated[int, Field(
        default=6,
        description="Decimal precision for results",
        ge=0,
        le=15
    )] = 6,
    use_km_engine: Annotated[bool, Field(
        default=True,
        description="Use Keyboard Maestro's calculation engine"
    )] = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Perform mathematical calculations with comprehensive security and format support.
    
    Features:
    - Secure expression parsing with injection prevention
    - Support for basic arithmetic and advanced mathematical functions
    - Multiple output formats (decimal, hex, binary, scientific, percentage)
    - Variable substitution in expressions
    - Integration with Keyboard Maestro's calculation engine
    - Comprehensive error handling for mathematical edge cases
    
    Security:
    - Expression whitelist validation
    - Prevention of code injection attacks
    - Safe evaluation with bounded execution
    - Input sanitization and validation
    
    Returns calculation results with formatted output and execution metadata.
    """
    if ctx:
        await ctx.info(f"Evaluating mathematical expression: {expression[:50]}...")
    
    try:
        import time
        start_time = time.time()
        
        # Create calculation expression with validation
        calc_expr = CalculationExpression(
            expression=expression,
            variables=variables
        )
        
        # Choose calculation method
        if use_km_engine:
            # Use Keyboard Maestro's calculation engine
            km_calc = KMCalculationEngine()
            km_result = await km_calc.evaluate_with_km(expression, {k: str(v) for k, v in variables.items()})
            
            if km_result.is_left():
                # Fallback to local calculator
                calculator = Calculator()
                calc_result = await calculator.calculate(calc_expr)
            else:
                # Parse KM result
                km_value = float(km_result.get_right())
                execution_time = time.time() - start_time
                
                calc_result = Either.right(CalculationResult(
                    result=km_value,
                    formatted_result=str(km_value),
                    expression=expression,
                    format=NumberFormat(output_format),
                    execution_time=execution_time
                ))
        else:
            # Use local calculator
            calculator = Calculator()
            calc_result = await calculator.calculate(calc_expr)
        
        if calc_result.is_left():
            error = calc_result.get_left()
            return {
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": {"expression": expression}
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        result = calc_result.get_right()
        
        # Format result in requested format
        formatted_output = result.to_format(NumberFormat(output_format))
        
        if ctx:
            await ctx.info(f"Calculation complete: {formatted_output}")
        
        return {
            "success": True,
            "calculation": {
                "expression": expression,
                "result": result.result,
                "formatted_result": formatted_output,
                "format": output_format,
                "precision": precision,
                "variables_used": variables,
                "execution_time": result.execution_time
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "calculation_id": str(uuid.uuid4()),
                "engine": "keyboard_maestro" if use_km_engine else "local"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "CALCULATION_ERROR",
                "message": str(e),
                "details": {"expression": expression}
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat()
            }
        }
```

## ✅ Success Criteria
- [x] Complete mathematical calculator with secure expression parsing ✅
- [x] Support for basic arithmetic and advanced mathematical functions ✅
- [x] Multiple number format support (decimal, hex, binary, scientific, percentage) ✅
- [x] Real Keyboard Maestro calculation engine integration ✅
- [x] Comprehensive security validation against code injection ✅
- [x] Property-based testing for mathematical operations and edge cases ✅
- [x] Performance meets sub-200ms calculation targets for most expressions ✅
- [x] Integration with macro creation for calculation-based workflows ✅
- [x] TESTING.md updated with mathematical validation and security tests ✅
- [x] Documentation with mathematical function reference and security guidelines ✅

## 🎨 Usage Examples

### Basic Calculations
```python
# Simple arithmetic
result = await client.call_tool("km_calculator", {
    "expression": "15 + 25 * 2",
    "output_format": "decimal"
})

# With variables
result = await client.call_tool("km_calculator", {
    "expression": "price * (1 + tax_rate)",
    "variables": {"price": 100.0, "tax_rate": 0.08},
    "output_format": "decimal",
    "precision": 2
})
```

### Advanced Mathematical Operations
```python
# Trigonometric functions
result = await client.call_tool("km_calculator", {
    "expression": "sin(pi/2) + cos(0)",
    "output_format": "scientific"
})

# Format conversion
result = await client.call_tool("km_calculator", {
    "expression": "255",
    "output_format": "hex"
})
```