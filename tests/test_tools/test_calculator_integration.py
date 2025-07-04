"""
Integration tests for calculator tools with proper MCP integration.

Tests the calculator MCP tools that provide mathematical operations
and expression evaluation with security validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from typing import Dict, Any

from src.server.tools.calculator_tools import (
    km_calculate_expression,
    km_calculate_math_function,
    km_convert_number_format,
    km_evaluate_formula,
    km_calculator
)
from src.calculations.calculator import Calculator, SafeExpressionEvaluator
from src.core.errors import ValidationError, SecurityError


class TestCalculatorToolsIntegration:
    """Integration tests for calculator MCP tools."""
    
    @pytest.fixture
    def mock_calculator(self):
        """Mock calculator for testing."""
        mock = Mock(spec=Calculator)
        # Mock the async calculate method to return an Either.right result
        from src.integration.km_client import Either
        from src.calculations.calculator import CalculationResult, NumberFormat
        
        mock_result = CalculationResult(
            result=42.0,
            formatted_result="42",
            expression="2 + 2 * 10",
            format=NumberFormat.DECIMAL,
            execution_time=0.001,
            variables_used={}
        )
        mock.calculate = AsyncMock(return_value=Either.right(mock_result))
        return mock
    
    @pytest.mark.asyncio
    async def test_km_calculate_expression_success(self, mock_calculator):
        """Test successful expression calculation."""
        with patch('src.server.tools.calculator_tools.Calculator', return_value=mock_calculator):
            result = await km_calculate_expression(expression="2 + 2 * 10")
        
        assert result["success"] is True
        assert "result" in result
        assert "expression" in result
        assert result["expression"] == "2 + 2 * 10"
        
    @pytest.mark.asyncio
    async def test_km_calculate_expression_validation_error(self):
        """Test expression validation."""
        # Test empty expression
        with pytest.raises(ValidationError) as exc_info:
            await km_calculate_expression(expression="")
        assert "expression" in str(exc_info.value)
        
        # Test None expression  
        with pytest.raises(ValidationError):
            await km_calculate_expression(expression=None)
    
    @pytest.mark.asyncio
    async def test_km_calculate_expression_security_error(self, mock_calculator):
        """Test expression security validation."""
        mock_calculator.evaluate_expression.side_effect = SecurityError(
            "DANGEROUS_EXPRESSION", "Expression contains dangerous functions"
        )
        
        with patch('src.server.tools.calculator_tools.Calculator', return_value=mock_calculator):
            result = await km_calculate_expression(expression="__import__('os').system('rm -rf /')")
        
        assert result["success"] is False
        assert "error" in result
        assert "security" in result["error"]["message"].lower()
        
    @pytest.mark.asyncio
    async def test_km_calculate_math_function_success(self, mock_calculator):
        """Test successful math function calculation."""
        mock_calculator.calculate_function.return_value = Decimal('1.0')
        
        with patch('src.server.tools.calculator_tools.Calculator', return_value=mock_calculator):
            result = await km_calculate_math_function(
                function="sin",
                value=90,
                angle_unit="degrees"
            )
        
        assert result["success"] is True
        assert "result" in result
        assert "function" in result
        assert result["function"] == "sin"
        
    @pytest.mark.asyncio
    async def test_km_calculate_math_function_validation(self):
        """Test math function parameter validation."""
        # Test empty function
        with pytest.raises(ValidationError):
            await km_calculate_math_function(function="", value=1)
        
        # Test invalid angle unit
        with pytest.raises(ValidationError):
            await km_calculate_math_function(
                function="sin", 
                value=90, 
                angle_unit="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_km_convert_number_format_success(self, mock_calculator):
        """Test successful number format conversion."""
        mock_calculator.convert_format.return_value = "0x2A"
        
        with patch('src.server.tools.calculator_tools.Calculator', return_value=mock_calculator):
            result = await km_convert_number_format(
                value=42,
                from_format="decimal",
                to_format="hexadecimal"
            )
        
        assert result["success"] is True
        assert "result" in result
        assert result["result"] == "0x2A"
        assert result["from_format"] == "decimal"
        assert result["to_format"] == "hexadecimal"
        
    @pytest.mark.asyncio
    async def test_km_convert_number_format_validation(self):
        """Test number format conversion validation."""
        # Test invalid format
        with pytest.raises(ValidationError):
            await km_convert_number_format(
                value=42,
                from_format="invalid",
                to_format="hexadecimal"
            )
        
        # Test same format conversion
        with pytest.raises(ValidationError):
            await km_convert_number_format(
                value=42,
                from_format="decimal", 
                to_format="decimal"
            )
    
    @pytest.mark.asyncio
    async def test_km_evaluate_formula_success(self, mock_calculator):
        """Test successful formula evaluation with variables."""
        mock_calculator.evaluate_expression.return_value = Decimal('25')
        
        with patch('src.server.tools.calculator_tools.Calculator', return_value=mock_calculator):
            result = await km_evaluate_formula(
                formula="x^2 + y",
                variables={"x": 4, "y": 9}
            )
        
        assert result["success"] is True
        assert "result" in result
        assert "formula" in result
        assert "variables" in result
        
    @pytest.mark.asyncio
    async def test_km_evaluate_formula_validation(self):
        """Test formula evaluation validation."""
        # Test empty formula
        with pytest.raises(ValidationError):
            await km_evaluate_formula(formula="", variables={})
        
        # Test invalid variables type
        with pytest.raises(ValidationError):
            await km_evaluate_formula(
                formula="x + y",
                variables="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_concurrent_calculations(self, mock_calculator):
        """Test concurrent calculator operations."""
        mock_calculator.evaluate_expression.return_value = Decimal('10')
        mock_calculator.calculate_function.return_value = Decimal('1')
        mock_calculator.convert_format.return_value = "0xA"
        
        with patch('src.server.tools.calculator_tools.Calculator', return_value=mock_calculator):
            tasks = [
                km_calculate_expression(expression="5 + 5"),
                km_calculate_math_function(function="cos", value=0),
                km_convert_number_format(value=10, from_format="decimal", to_format="hexadecimal")
            ]
            
            results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(result["success"] for result in results)
    
    @pytest.mark.asyncio
    async def test_calculator_error_handling(self, mock_calculator):
        """Test comprehensive error handling."""
        # Test division by zero
        mock_calculator.evaluate_expression.side_effect = ZeroDivisionError("Division by zero")
        
        with patch('src.server.tools.calculator_tools.Calculator', return_value=mock_calculator):
            result = await km_calculate_expression(expression="1/0")
        
        assert result["success"] is False
        assert "error" in result
        assert "division" in result["error"]["message"].lower()
        
    @pytest.mark.asyncio
    async def test_calculator_precision_handling(self, mock_calculator):
        """Test precision handling in calculations."""
        # Test high precision decimal
        mock_calculator.evaluate_expression.return_value = Decimal('3.141592653589793')
        
        with patch('src.server.tools.calculator_tools.Calculator', return_value=mock_calculator):
            result = await km_calculate_expression(expression="pi")
        
        assert result["success"] is True
        # Result should preserve precision
        assert len(str(result["result"]).split('.')[-1]) > 10
    
    @pytest.mark.asyncio
    async def test_calculator_variable_substitution(self, mock_calculator):
        """Test variable substitution in formulas."""
        mock_calculator.evaluate_expression.return_value = Decimal('15')
        
        with patch('src.server.tools.calculator_tools.Calculator', return_value=mock_calculator):
            result = await km_evaluate_formula(
                formula="a * b + c",
                variables={"a": 2, "b": 5, "c": 5}
            )
        
        assert result["success"] is True
        assert result["variables"]["a"] == 2
        assert result["variables"]["b"] == 5
        assert result["variables"]["c"] == 5


class TestCalculatorSecurityValidation:
    """Test security validation in calculator tools."""
    
    @pytest.mark.asyncio
    async def test_dangerous_expression_rejection(self):
        """Test rejection of dangerous expressions."""
        dangerous_expressions = [
            "__import__('os').system('ls')",
            "eval('print(1)')",
            "exec('import sys')",
            "open('/etc/passwd')",
            "subprocess.call(['ls'])"
        ]
        
        for expr in dangerous_expressions:
            with patch('src.server.tools.calculator_tools.Calculator') as mock_calc_class:
                mock_calc = Mock()
                mock_calc.evaluate_expression.side_effect = SecurityError(
                    "DANGEROUS_EXPRESSION", f"Expression contains dangerous content: {expr}"
                )
                mock_calc_class.return_value = mock_calc
                
                result = await km_calculate_expression(expression=expr)
                
                assert result["success"] is False
                assert "error" in result
                assert "security" in result["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_expression_length_limits(self):
        """Test expression length validation."""
        # Very long expression should be rejected
        long_expression = "1 + " * 10000 + "1"
        
        with pytest.raises(ValidationError) as exc_info:
            await km_calculate_expression(expression=long_expression)
        
        assert "too long" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_variable_name_validation(self):
        """Test variable name security validation."""
        dangerous_variables = {
            "__import__": 1,
            "eval": 2, 
            "exec": 3,
            "open": 4
        }
        
        with pytest.raises(ValidationError):
            await km_evaluate_formula(
                formula="x + y",
                variables=dangerous_variables
            )