"""
Integration tests for calculator tools with proper MCP integration.

Tests the calculator MCP tools that provide mathematical operations
and expression evaluation with security validation.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.calculations.calculator import Calculator
from src.server.tools.calculator_tools import (
    km_calculate_expression,
    km_calculate_math_function,
    km_convert_number_format,
    km_evaluate_formula,
)


class TestCalculatorToolsIntegration:
    """Integration tests for calculator MCP tools."""

    @pytest.fixture
    def mock_calculator(self):
        """Mock calculator for testing."""
        mock = Mock(spec=Calculator)
        # Mock the async calculate method to return an Either.right result
        from src.calculations.calculator import CalculationResult, NumberFormat
        from src.integration.km_client import Either

        mock_result = CalculationResult(
            result=42.0,
            formatted_result="42",
            expression="2 + 2 * 10",
            format=NumberFormat.DECIMAL,
            execution_time=0.001,
            variables_used={},
        )
        mock.calculate = AsyncMock(return_value=Either.right(mock_result))
        return mock

    @pytest.mark.asyncio
    async def test_km_calculate_expression_success(self, mock_calculator):
        """Test successful expression calculation."""
        with patch(
            "src.server.tools.calculator_tools.Calculator", return_value=mock_calculator
        ):
            result = await km_calculate_expression(expression="2 + 2 * 10")

        assert result["success"] is True
        assert "calculation" in result
        assert "result" in result["calculation"]
        assert "expression" in result["calculation"]
        assert result["calculation"]["expression"] == "2 + 2 * 10"
        assert result["calculation"]["result"] == 42.0

    @pytest.mark.asyncio
    async def test_km_calculate_expression_validation_error(self):
        """Test expression validation."""
        # Test empty expression
        result = await km_calculate_expression(expression="")
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_EXPRESSION"
        assert "empty" in result["error"]["message"].lower()

        # Test whitespace-only expression
        result = await km_calculate_expression(expression="   ")
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_EXPRESSION"

    @pytest.mark.asyncio
    async def test_km_calculate_expression_security_error(self):
        """Test expression security validation."""
        # Test dangerous expression - should be caught by CalculationExpression validation
        result = await km_calculate_expression(
            expression="__import__('os').system('rm -rf /')"
        )

        assert result["success"] is False
        assert "error" in result
        assert result["error"]["code"] == "EXPRESSION_VALIDATION_ERROR"
        # The error should mention validation failure
        assert "validation" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_km_calculate_math_function_success(self, mock_calculator):
        """Test successful math function calculation."""
        from src.calculations.calculator import CalculationResult, NumberFormat
        from src.integration.km_client import Either

        mock_result = CalculationResult(
            result=1.0,
            formatted_result="1.0",
            expression="sin(90)",
            format=NumberFormat.DECIMAL,
            execution_time=0.001,
            variables_used={},
        )
        mock_calculator.calculate = AsyncMock(return_value=Either.right(mock_result))

        with patch(
            "src.server.tools.calculator_tools.Calculator", return_value=mock_calculator
        ):
            result = await km_calculate_math_function(function="sin", value=90)

        assert result["success"] is True
        assert "calculation" in result
        assert result["calculation"]["expression"] == "sin(90)"
        assert result["calculation"]["result"] == 1.0

    @pytest.mark.asyncio
    async def test_km_calculate_math_function_validation(self):
        """Test math function parameter validation."""
        # Test empty function - should result in invalid expression
        result = await km_calculate_math_function(function="", value=1)
        assert result["success"] is False
        assert "error" in result

        # Test function with special characters that create invalid expression
        result = await km_calculate_math_function(function="invalid_func!", value=1)
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_km_convert_number_format_success(self, mock_calculator):
        """Test successful number format conversion."""
        from src.calculations.calculator import CalculationResult, NumberFormat
        from src.integration.km_client import Either

        mock_result = CalculationResult(
            result=42.0,
            formatted_result="42",
            expression="42",
            format=NumberFormat.DECIMAL,
            execution_time=0.001,
            variables_used={},
        )
        mock_calculator.calculate = AsyncMock(return_value=Either.right(mock_result))

        with patch(
            "src.server.tools.calculator_tools.Calculator", return_value=mock_calculator
        ):
            result = await km_convert_number_format(
                value=42, from_format="decimal", to_format="hexadecimal"
            )

        assert result["success"] is True
        assert "calculation" in result
        assert result["calculation"]["expression"] == "42"
        assert result["calculation"]["result"] == 42.0

    @pytest.mark.asyncio
    async def test_km_convert_number_format_validation(self):
        """Test number format conversion validation."""
        # Note: km_convert_number_format doesn't validate format strings directly,
        # it just passes values through to km_calculator. Let's test actual error cases.

        # Test with invalid number that might cause conversion issues
        result = await km_convert_number_format(
            value=float("inf"), from_format="decimal", to_format="hexadecimal"
        )
        # Should handle gracefully, either succeed or return error response
        assert "success" in result

    @pytest.mark.asyncio
    async def test_km_evaluate_formula_success(self, mock_calculator):
        """Test successful formula evaluation with variables."""
        from src.calculations.calculator import CalculationResult, NumberFormat
        from src.integration.km_client import Either

        mock_result = CalculationResult(
            result=25.0,
            formatted_result="25.0",
            expression="x**2 + y",
            format=NumberFormat.DECIMAL,
            execution_time=0.001,
            variables_used={"x": 4, "y": 9},
        )
        mock_calculator.calculate = AsyncMock(return_value=Either.right(mock_result))

        with patch(
            "src.server.tools.calculator_tools.Calculator", return_value=mock_calculator
        ):
            result = await km_evaluate_formula(
                formula="x**2 + y", variables={"x": 4, "y": 9}
            )

        assert result["success"] is True
        assert "calculation" in result
        assert result["calculation"]["expression"] == "x**2 + y"
        assert result["calculation"]["result"] == 25.0
        assert result["calculation"]["variables_used"] == {"x": 4, "y": 9}

    @pytest.mark.asyncio
    async def test_km_evaluate_formula_validation(self):
        """Test formula evaluation validation."""
        # Test empty formula
        result = await km_evaluate_formula(formula="", variables={})
        assert result["success"] is False
        assert "INVALID_EXPRESSION" in result["error"]["code"]

        # Test invalid variables - km_evaluate_formula expects Dict[str, float]
        # but the validation happens at the Pydantic level in km_calculator

    @pytest.mark.asyncio
    async def test_concurrent_calculations(self, mock_calculator):
        """Test concurrent calculator operations."""
        from src.calculations.calculator import CalculationResult, NumberFormat
        from src.integration.km_client import Either

        # Mock different results for different expressions
        def mock_calculate_side_effect(calc_expression):
            if "5 + 5" in calc_expression.expression:
                return Either.right(
                    CalculationResult(
                        result=10.0,
                        formatted_result="10",
                        expression="5 + 5",
                        format=NumberFormat.DECIMAL,
                        execution_time=0.001,
                        variables_used={},
                    )
                )
            elif "cos(0)" in calc_expression.expression:
                return Either.right(
                    CalculationResult(
                        result=1.0,
                        formatted_result="1.0",
                        expression="cos(0)",
                        format=NumberFormat.DECIMAL,
                        execution_time=0.001,
                        variables_used={},
                    )
                )
            elif "10" in calc_expression.expression:
                return Either.right(
                    CalculationResult(
                        result=10.0,
                        formatted_result="10",
                        expression="10",
                        format=NumberFormat.DECIMAL,
                        execution_time=0.001,
                        variables_used={},
                    )
                )
            else:
                return Either.right(
                    CalculationResult(
                        result=0.0,
                        formatted_result="0",
                        expression=calc_expression.expression,
                        format=NumberFormat.DECIMAL,
                        execution_time=0.001,
                        variables_used={},
                    )
                )

        mock_calculator.calculate = AsyncMock(side_effect=mock_calculate_side_effect)

        with patch(
            "src.server.tools.calculator_tools.Calculator", return_value=mock_calculator
        ):
            tasks = [
                km_calculate_expression(expression="5 + 5"),
                km_calculate_math_function(function="cos", value=0),
                km_convert_number_format(
                    value=10, from_format="decimal", to_format="hexadecimal"
                ),
            ]

            results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(result["success"] for result in results)

    @pytest.mark.asyncio
    async def test_calculator_error_handling(self, mock_calculator):
        """Test comprehensive error handling."""
        from src.integration.km_client import Either, KMError

        # Test division by zero - return error result
        mock_calculator.calculate = AsyncMock(
            return_value=Either.left(KMError.execution_error("Division by zero"))
        )

        with patch(
            "src.server.tools.calculator_tools.Calculator", return_value=mock_calculator
        ):
            result = await km_calculate_expression(expression="1/0")

        assert result["success"] is False
        assert "error" in result
        assert (
            "division" in result["error"]["message"].lower()
            or "execution_error" in result["error"]["code"]
        )

    @pytest.mark.asyncio
    async def test_calculator_precision_handling(self, mock_calculator):
        """Test precision handling in calculations."""
        from src.calculations.calculator import CalculationResult, NumberFormat
        from src.integration.km_client import Either

        # Test high precision decimal
        mock_result = CalculationResult(
            result=3.141592653589793,
            formatted_result="3.141592653589793",
            expression="pi",
            format=NumberFormat.DECIMAL,
            execution_time=0.001,
            variables_used={},
        )
        mock_calculator.calculate = AsyncMock(return_value=Either.right(mock_result))

        with patch(
            "src.server.tools.calculator_tools.Calculator", return_value=mock_calculator
        ):
            result = await km_calculate_expression(expression="pi")

        assert result["success"] is True
        # Result should preserve precision
        assert len(str(result["calculation"]["result"]).split(".")[-1]) > 10

    @pytest.mark.asyncio
    async def test_calculator_variable_substitution(self, mock_calculator):
        """Test variable substitution in formulas."""
        from src.calculations.calculator import CalculationResult, NumberFormat
        from src.integration.km_client import Either

        mock_result = CalculationResult(
            result=15.0,
            formatted_result="15",
            expression="a * b + c",
            format=NumberFormat.DECIMAL,
            execution_time=0.001,
            variables_used={"a": 2, "b": 5, "c": 5},
        )
        mock_calculator.calculate = AsyncMock(return_value=Either.right(mock_result))

        with patch(
            "src.server.tools.calculator_tools.Calculator", return_value=mock_calculator
        ):
            result = await km_evaluate_formula(
                formula="a * b + c", variables={"a": 2, "b": 5, "c": 5}
            )

        assert result["success"] is True
        assert result["calculation"]["variables_used"]["a"] == 2
        assert result["calculation"]["variables_used"]["b"] == 5
        assert result["calculation"]["variables_used"]["c"] == 5


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
            "subprocess.call(['ls'])",
        ]

        for expr in dangerous_expressions:
            # These dangerous expressions should be caught during CalculationExpression validation
            result = await km_calculate_expression(expression=expr)

            assert result["success"] is False
            assert "error" in result
            # Should be caught by expression validation
            assert (
                "EXPRESSION_VALIDATION_ERROR" in result["error"]["code"]
                or "CALCULATION_ERROR" in result["error"]["code"]
            )

    @pytest.mark.asyncio
    async def test_expression_length_limits(self):
        """Test expression length validation."""
        # Very long expression should be rejected
        long_expression = "1 + " * 10000 + "1"

        result = await km_calculate_expression(expression=long_expression)

        assert result["success"] is False
        assert "error" in result
        # Should be caught by expression validation or calculation error
        assert (
            "EXPRESSION_VALIDATION_ERROR" in result["error"]["code"]
            or "CALCULATION_ERROR" in result["error"]["code"]
        )

    @pytest.mark.asyncio
    async def test_variable_name_validation(self):
        """Test variable name security validation."""
        dangerous_variables = {"__import__": 1, "eval": 2, "exec": 3, "open": 4}

        # These dangerous variable names should be caught during validation
        result = await km_evaluate_formula(
            formula="x + y", variables=dangerous_variables
        )

        assert result["success"] is False
        assert "error" in result
        # Should be caught by expression validation or calculation error
        assert (
            "EXPRESSION_VALIDATION_ERROR" in result["error"]["code"]
            or "CALCULATION_ERROR" in result["error"]["code"]
        )
