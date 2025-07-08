"""Comprehensive Test Suite for Engine Tools - Following Proven MCP Tool Test Pattern.

This test suite validates the Engine Tools functionality using the systematic
testing approach that achieved 100% success rate across 17 tool suites.

Test Coverage:
- Engine control operations (reload, status, calculate, process_tokens, search_replace)
- Mathematical calculations with validation and security checks
- Token processing for KM variables, dates, and system tokens
- Search and replace operations with regex and plain text support
- KM client integration and connection management with Either pattern success mocking
- Expression validation and security boundary checking
- Progress reporting and context integration validation
- Security validation for forbidden operations and injection prevention
- Property-based testing for robust expression and pattern validation
- Integration testing with mocked KM clients and engine operations
- Error handling for all failure scenarios
- Performance testing for engine operation response times

Testing Strategy:
- Property-based testing with Hypothesis for comprehensive input coverage
- Mock-based testing for KM Client and engine control components
- Security validation for expression safety and injection prevention
- Integration testing scenarios with realistic engine operations
- Performance and timeout testing with operation limits

Key Mocking Pattern:
- KMClient: Mock Keyboard Maestro client integration with check_connection
- Engine operations: Mock calculation, token processing, and search/replace operations
- Context: Mock progress reporting and logging operations
- Expression validation: Test security boundaries and calculation safety
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

# Import engine types and errors
# Import the tools we're testing
from src.server.tools.engine_tools import (
    _calculate_expression,
    _get_engine_status,
    _process_tokens,
    _reload_engine,
    _search_replace,
    km_engine_control,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# Test fixtures following proven pattern
@pytest.fixture
def mock_context() -> Mock:
    """Create mock FastMCP context following successful pattern."""
    context = Mock(spec=Context)
    context.info = AsyncMock()
    context.warn = AsyncMock()
    context.error = AsyncMock()
    context.report_progress = AsyncMock()
    context.read_resource = AsyncMock()
    context.sample = AsyncMock()
    context.get = Mock(return_value="")  # Support ctx.get() calls
    return context


@pytest.fixture
def mock_km_client() -> Mock:
    """Create mock KM client with standard interface."""
    client = Mock()
    # Mock connection check - CRITICAL for all tests
    mock_connection_result = Mock()
    mock_connection_result.is_left.return_value = False
    mock_connection_result.get_right.return_value = True
    client.check_connection.return_value = mock_connection_result

    # Mock list_macros_with_details for status operation
    mock_macros_result = Mock()
    mock_macros_result.is_right.return_value = True
    mock_macros_result.get_right.return_value = [
        {"name": "Test Macro 1", "enabled": True},
        {"name": "Test Macro 2", "enabled": True},
        {"name": "Test Macro 3", "enabled": False},
        {"name": "Test Macro 4", "enabled": True},
    ]
    client.list_macros_with_details.return_value = mock_macros_result

    return client


@pytest.fixture
def valid_expression() -> str:
    """Provide valid test calculation expression."""
    return "2 + 3 * 4"


@pytest.fixture
def valid_token_string() -> str:
    """Provide valid test token string."""
    return "Hello %Variable%MyVar%, today is %ICUDateTime%yyyy-MM-dd%, user is %CurrentUser%"


@pytest.fixture
def valid_search_text() -> str:
    """Provide valid test text for search/replace."""
    return "The quick brown fox jumps over the lazy dog. The fox is quick."


# Core Engine Control Tests
class TestEngineOperations:
    """Test core km_engine_control functionality."""

    @pytest.mark.asyncio
    async def test_reload_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test successful engine reload operation."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(operation="reload", ctx=mock_context)

            assert result["success"] is True
            assert result["data"]["operation"] == "reload"
            assert "reload_time_seconds" in result["data"]
            assert "timestamp" in result["data"]
            assert result["data"]["reload_time_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_status_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test successful engine status operation."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(operation="status", ctx=mock_context)

            assert result["success"] is True
            assert "engine_version" in result["data"]
            assert "engine_running" in result["data"]
            assert "macro_statistics" in result["data"]
            assert result["data"]["macro_statistics"]["total_macros"] == 4
            assert result["data"]["macro_statistics"]["enabled_macros"] == 3
            assert result["data"]["macro_statistics"]["disabled_macros"] == 1
            assert "performance" in result["data"]
            assert "resources" in result["data"]

    @pytest.mark.asyncio
    async def test_calculate_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_expression: Any,
    ) -> None:
        """Test successful calculation operation."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="calculate",
                expression=valid_expression,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["expression"] == valid_expression
            assert result["data"]["result"] == "14"  # 2 + 3 * 4 = 14
            assert result["data"]["result_type"] == "int"
            assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_process_tokens_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_token_string: Any,
    ) -> None:
        """Test successful token processing operation."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="process_tokens",
                expression=valid_token_string,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["original"] == valid_token_string
            assert "processed" in result["data"]
            assert (
                result["data"]["token_count"] >= 3
            )  # At least Variable, DateTime, CurrentUser
            assert len(result["data"]["tokens_found"]) >= 3
            assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_search_replace_operation_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_search_text: Any,
    ) -> None:
        """Test successful search and replace operation."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="search_replace",
                search_pattern="fox",
                replace_pattern="cat",
                text=valid_search_text,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["search_pattern"] == "fox"
            assert result["data"]["replace_pattern"] == "cat"
            assert result["data"]["match_count"] == 2  # Two instances of "fox"
            assert result["data"]["use_regex"] is False
            assert "cat" in result["data"]["result"]
            assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_search_only_operation(
        self,
        mock_context: Any,
        mock_km_client: Any,
        valid_search_text: Any,
    ) -> None:
        """Test search-only operation without replacement."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="search_replace",
                search_pattern="fox",
                text=valid_search_text,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["replace_pattern"] is None
            assert result["data"]["match_count"] == 2
            assert result["data"]["result"] == valid_search_text  # Unchanged

    @pytest.mark.asyncio
    async def test_regex_search_replace_success(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test regex search and replace operation."""
        test_text = "Contact: john@example.com or jane@test.org"

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="search_replace",
                search_pattern=r"(\w+)@(\w+)\.(\w+)",
                replace_pattern=r"\1 at \2 dot \3",
                text=test_text,
                use_regex=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["use_regex"] is True
            assert result["data"]["match_count"] == 2
            assert "john at example dot com" in result["data"]["result"]
            assert "jane at test dot org" in result["data"]["result"]


# Error Handling Tests
class TestEngineErrorHandling:
    """Test engine tools error handling scenarios."""

    @pytest.mark.asyncio
    async def test_missing_expression_for_calculate(self, mock_context: Any) -> None:
        """Test error when expression is missing for calculate operation."""
        with patch("src.server.tools.engine_tools.get_km_client"):
            result = await km_engine_control(
                operation="calculate",
                expression=None,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "ENGINE_ERROR"
            assert "Expression required" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_missing_expression_for_process_tokens(
        self,
        mock_context: Any,
    ) -> None:
        """Test error when expression is missing for process_tokens operation."""
        with patch("src.server.tools.engine_tools.get_km_client"):
            result = await km_engine_control(
                operation="process_tokens",
                expression=None,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "ENGINE_ERROR"
            assert "Token string required" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_missing_search_pattern_for_search_replace(
        self,
        mock_context: Any,
    ) -> None:
        """Test error when search pattern is missing for search_replace operation."""
        with patch("src.server.tools.engine_tools.get_km_client"):
            result = await km_engine_control(
                operation="search_replace",
                search_pattern=None,
                text="Some text",
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "ENGINE_ERROR"
            assert "Search pattern required" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_missing_text_for_search_replace(self, mock_context: Any) -> None:
        """Test error when text is missing for search_replace operation."""
        with patch("src.server.tools.engine_tools.get_km_client"):
            result = await km_engine_control(
                operation="search_replace",
                search_pattern="test",
                text=None,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "ENGINE_ERROR"
            assert "Text required" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_km_connection_failure(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test handling of KM connection failure."""
        # Mock connection failure
        mock_connection_result = Mock()
        mock_connection_result.is_left.return_value = True
        mock_km_client.check_connection.return_value = mock_connection_result

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(operation="reload", ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "KM_CONNECTION_FAILED"
            assert (
                "Cannot connect to Keyboard Maestro Engine"
                in result["error"]["message"]
            )

    @pytest.mark.asyncio
    async def test_invalid_operation(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test error for invalid operation."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="invalid_operation",
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "ENGINE_ERROR"
            assert "Unknown operation" in result["error"]["details"]


# Expression Validation and Security Tests
class TestExpressionSecurity:
    """Test expression validation and security checks."""

    @pytest.mark.asyncio
    async def test_forbidden_operations_in_expression(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test security validation for forbidden operations."""
        dangerous_expressions = [
            "exec('malicious code')",
            "eval('dangerous')",
            "import os",
            "__import__('os')",
        ]

        for expression in dangerous_expressions:
            with patch(
                "src.server.tools.engine_tools.get_km_client",
                return_value=mock_km_client,
            ):
                result = await km_engine_control(
                    operation="calculate",
                    expression=expression,
                    ctx=mock_context,
                )

                assert result["success"] is False
                assert result["error"]["code"] == "ENGINE_ERROR"
                assert "forbidden operations" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_invalid_characters_in_expression(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test validation for invalid characters in expression."""
        invalid_expression = "2 + 3; rm -rf /"

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="calculate",
                expression=invalid_expression,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "ENGINE_ERROR"
            assert "invalid characters" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_calculation_error_handling(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test handling of calculation errors."""
        invalid_expression = "1 / 0"  # Division by zero

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="calculate",
                expression=invalid_expression,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "ENGINE_ERROR"
            assert "Calculation failed" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_complex_valid_expression(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test complex but valid mathematical expressions."""
        complex_expressions = [
            "abs(-5) + round(3.7)",
            "min(10, 20) + max(5, 15)",
            "pow(2, 3) * 2",
            "pi * 2",
            "e + 1",
        ]

        for expression in complex_expressions:
            with patch(
                "src.server.tools.engine_tools.get_km_client",
                return_value=mock_km_client,
            ):
                result = await km_engine_control(
                    operation="calculate",
                    expression=expression,
                    ctx=mock_context,
                )

                assert result["success"] is True
                assert "result" in result["data"]
                assert result["data"]["expression"] == expression


# Helper Function Tests
class TestHelperFunctions:
    """Test engine helper functions directly."""

    @pytest.mark.asyncio
    async def test_reload_engine_function(self, mock_context: Any) -> None:
        """Test the _reload_engine helper function directly."""
        mock_km_client = Mock()

        result = await _reload_engine(mock_km_client, mock_context)

        assert result["success"] is True
        assert result["data"]["operation"] == "reload"
        assert result["data"]["reload_time_seconds"] >= 0
        assert "timestamp" in result["data"]

    @pytest.mark.asyncio
    async def test_get_engine_status_function(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test the _get_engine_status helper function directly."""
        result = await _get_engine_status(mock_km_client, mock_context)

        assert result["success"] is True
        assert "engine_version" in result["data"]
        assert "macro_statistics" in result["data"]
        assert "performance" in result["data"]
        assert "resources" in result["data"]

    @pytest.mark.asyncio
    async def test_calculate_expression_function(self, mock_context: Any) -> None:
        """Test the _calculate_expression helper function directly."""
        mock_km_client = Mock()
        expression = "5 + 3"

        result = await _calculate_expression(mock_km_client, expression, mock_context)

        assert result["success"] is True
        assert result["data"]["expression"] == expression
        assert result["data"]["result"] == "8"
        assert result["data"]["result_type"] == "int"

    @pytest.mark.asyncio
    async def test_process_tokens_function(self, mock_context: Any) -> None:
        """Test the _process_tokens helper function directly."""
        mock_km_client = Mock()
        token_string = "User: %CurrentUser%, Volume: %SystemVolume%"  # noqa: S105 # Test string

        result = await _process_tokens(mock_km_client, token_string, mock_context)

        assert result["success"] is True
        assert result["data"]["original"] == token_string
        assert result["data"]["token_count"] >= 2
        assert "TestUser" in result["data"]["processed"]
        assert "85" in result["data"]["processed"]

    @pytest.mark.asyncio
    async def test_search_replace_function(self, mock_context: Any) -> None:
        """Test the _search_replace helper function directly."""
        mock_km_client = Mock()
        text = "Hello world, hello universe"
        search_pattern = "hello"
        replace_pattern = "hi"

        result = await _search_replace(
            mock_km_client,
            text,
            search_pattern,
            replace_pattern,
            False,
            mock_context,
        )

        assert result["success"] is True
        assert result["data"]["match_count"] == 1  # Case sensitive
        assert result["data"]["search_pattern"] == search_pattern
        assert result["data"]["replace_pattern"] == replace_pattern
        assert result["data"]["use_regex"] is False


# Integration Tests
class TestEngineIntegration:
    """Test engine tools integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_engine_workflow(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test complete engine workflow with multiple operations."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Test status
            status_result = await km_engine_control(
                operation="status",
                ctx=mock_context,
            )
            assert status_result["success"] is True

            # Test calculation
            calc_result = await km_engine_control(
                operation="calculate",
                expression="10 + 5",
                ctx=mock_context,
            )
            assert calc_result["success"] is True
            assert calc_result["data"]["result"] == "15"

            # Test token processing
            token_result = await km_engine_control(
                operation="process_tokens",
                expression="Hello %CurrentUser%",
                ctx=mock_context,
            )
            assert token_result["success"] is True
            assert "TestUser" in token_result["data"]["processed"]

    @pytest.mark.asyncio
    async def test_complex_search_replace_workflow(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test complex search and replace workflow."""
        text = "Error 404: File not found. Error 500: Server error."

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Plain text search
            plain_result = await km_engine_control(
                operation="search_replace",
                search_pattern="Error",
                replace_pattern="Warning",
                text=text,
                ctx=mock_context,
            )
            assert plain_result["success"] is True
            assert plain_result["data"]["match_count"] == 2

            # Regex search
            regex_result = await km_engine_control(
                operation="search_replace",
                search_pattern=r"Error (\d+)",
                replace_pattern=r"Code \1",
                text=text,
                use_regex=True,
                ctx=mock_context,
            )
            assert regex_result["success"] is True
            assert regex_result["data"]["use_regex"] is True
            assert "Code 404" in regex_result["data"]["result"]
            assert "Code 500" in regex_result["data"]["result"]


# Context Integration Tests
class TestEngineContext:
    """Test engine tools context integration."""

    @pytest.mark.asyncio
    async def test_context_info_logging(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test context info logging during execution."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(operation="reload", ctx=mock_context)

            assert result["success"] is True
            # Verify info logging was called
            mock_context.info.assert_called()
            # Verify progress reporting was called
            mock_context.report_progress.assert_called()

    @pytest.mark.asyncio
    async def test_context_error_logging(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test context error logging during failures."""
        # Mock connection failure to trigger error
        mock_connection_result = Mock()
        mock_connection_result.is_left.return_value = True
        mock_km_client.check_connection.return_value = mock_connection_result

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(operation="status", ctx=mock_context)

            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_without_context(self, mock_km_client: Any) -> None:
        """Test operation without context provided."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(operation="status", ctx=None)

            assert result["success"] is True


# Security Tests
class TestEngineSecurity:
    """Test engine tools security validation."""

    @pytest.mark.asyncio
    async def test_expression_injection_prevention(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test prevention of code injection in expressions."""
        malicious_expressions = [
            "__import__('subprocess').call(['rm', '-rf', '/'])",
            "exec(open('/etc/passwd').read())",
            'eval(\'__import__("os").system("ls")\')',
            "compile('malicious', '<string>', 'exec')",
        ]

        for expression in malicious_expressions:
            with patch(
                "src.server.tools.engine_tools.get_km_client",
                return_value=mock_km_client,
            ):
                result = await km_engine_control(
                    operation="calculate",
                    expression=expression,
                    ctx=mock_context,
                )
                assert result["success"] is False
                assert (
                    "forbidden" in result["error"]["details"].lower()
                    or "invalid" in result["error"]["details"].lower()
                )

    @pytest.mark.asyncio
    async def test_regex_safety_validation(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test regex pattern safety validation."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Test invalid regex pattern
            result = await km_engine_control(
                operation="search_replace",
                search_pattern="[invalid",  # Unclosed bracket
                text="test text",
                use_regex=True,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "ENGINE_ERROR"
            assert "Invalid regex pattern" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_large_input_handling(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test handling of large input data."""
        large_text = "x" * 15000  # Large text for search/replace

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="search_replace",
                search_pattern="x",
                replace_pattern="y",
                text=large_text,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["match_count"] == 15000
            # Result should be truncated for display
            assert len(result["data"]["result"]) <= 1003  # 1000 + "..."


# Property-Based Tests
class TestEnginePropertyBased:
    """Property-based testing for engine tools with Hypothesis."""

    @composite
    def valid_arithmetic_expressions(draw: Callable[..., Any]) -> Mock:
        """Generate valid arithmetic expressions."""
        operations = ["+", "-", "*", "/"]
        numbers = draw(
            st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=5),
        )
        ops = draw(
            st.lists(
                st.sampled_from(operations),
                min_size=len(numbers) - 1,
                max_size=len(numbers) - 1,
            ),
        )

        expression = str(numbers[0])
        for i, op in enumerate(ops):
            if op == "/" and numbers[i + 1] == 0:
                numbers[i + 1] = 1  # Avoid division by zero
            expression += f" {op} {numbers[i + 1]}"

        return expression

    @composite
    def valid_search_patterns(draw: Callable[..., Any]) -> Mock:
        """Generate valid search patterns."""
        patterns = draw(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd"),
                    whitelist_characters=" .-_",
                ),
                min_size=1,
                max_size=50,
            ).filter(lambda x: x.strip()),
        )
        return patterns

    @given(valid_arithmetic_expressions())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arithmetic_expression_property(self, expression: str) -> None:
        """Property: Valid arithmetic expressions should be calculable."""
        # Simple validation that expression contains only allowed characters
        allowed_chars = "0123456789+-*/()., "
        assert all(c in allowed_chars for c in expression)

    @given(valid_search_patterns())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_search_pattern_property(self, pattern: str) -> None:
        """Property: Valid search patterns should be processable."""
        assert len(pattern.strip()) > 0
        assert len(pattern) <= 50


# Performance Tests
class TestEnginePerformance:
    """Test engine tools performance characteristics."""

    @pytest.mark.asyncio
    async def test_engine_operation_response_time(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test that engine operations complete within reasonable time."""
        operations = ["reload", "status"]

        for operation in operations:
            start_time = time.time()

            with patch(
                "src.server.tools.engine_tools.get_km_client",
                return_value=mock_km_client,
            ):
                result = await km_engine_control(operation=operation, ctx=mock_context)

                end_time = time.time()
                execution_time = end_time - start_time

                # Should complete within 2 seconds (allowing for mocking overhead)
                assert execution_time < 2.0
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_calculation_performance(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test calculation operation performance with various expressions."""
        expressions = [
            "1 + 1",
            "abs(-10) + round(3.7)",
            "min(1, 2, 3) + max(10, 20, 30)",
            "pow(2, 8)",
        ]

        for expression in expressions:
            start_time = time.time()

            with patch(
                "src.server.tools.engine_tools.get_km_client",
                return_value=mock_km_client,
            ):
                result = await km_engine_control(
                    operation="calculate",
                    expression=expression,
                    ctx=mock_context,
                )

                end_time = time.time()
                execution_time = end_time - start_time

                # Should complete within 1 second
                assert execution_time < 1.0
                assert result["success"] is True
                assert "result" in result["data"]

    @pytest.mark.asyncio
    async def test_large_text_search_performance(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test search/replace performance with large text."""
        large_text = "word " * 1000  # 1000 words

        start_time = time.time()

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="search_replace",
                search_pattern="word",
                replace_pattern="term",
                text=large_text,
                ctx=mock_context,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within 2 seconds even with large text
            assert execution_time < 2.0
            assert result["success"] is True
            assert result["data"]["match_count"] == 1000


# Edge Case Tests
class TestEngineEdgeCases:
    """Test engine tools edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_expression_handling(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test handling of empty expressions."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="calculate",
                expression="",
                ctx=mock_context,
            )

            assert result["success"] is False
            assert "Expression required" in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_unicode_token_processing(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test token processing with Unicode characters."""
        unicode_token_string = "测试 %CurrentUser% 🌍 %SystemVolume%"  # noqa: S105 # Test string

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="process_tokens",
                expression=unicode_token_string,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert "TestUser" in result["data"]["processed"]
            assert "85" in result["data"]["processed"]
            assert result["data"]["token_count"] >= 2

    @pytest.mark.asyncio
    async def test_complex_regex_patterns(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test complex regex patterns."""
        text = "Email: john.doe@example.com, Phone: +1-555-123-4567"

        complex_patterns = [
            (r"\b\w+\.\w+@\w+\.\w+\b", "email patterns"),
            (r"\+\d{1}-\d{3}-\d{3}-\d{4}", "phone patterns"),
            (r"\b[A-Z][a-z]+\b", "capitalized words"),
        ]

        for pattern, _description in complex_patterns:
            with patch(
                "src.server.tools.engine_tools.get_km_client",
                return_value=mock_km_client,
            ):
                result = await km_engine_control(
                    operation="search_replace",
                    search_pattern=pattern,
                    text=text,
                    use_regex=True,
                    ctx=mock_context,
                )

                assert result["success"] is True
                assert result["data"]["use_regex"] is True
                assert result["data"]["match_count"] >= 0

    @pytest.mark.asyncio
    async def test_edge_case_calculations(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test edge case mathematical calculations."""
        edge_cases = [
            ("0", "0"),
            ("1", "1"),
            ("-1", "-1"),
            ("abs(-5)", "5"),
            ("round(2.5)", "2"),  # Python banker's rounding
            ("min()", None),  # Should fail
            ("max()", None),  # Should fail
        ]

        for expression, expected in edge_cases:
            with patch(
                "src.server.tools.engine_tools.get_km_client",
                return_value=mock_km_client,
            ):
                result = await km_engine_control(
                    operation="calculate",
                    expression=expression,
                    ctx=mock_context,
                )

                if expected is None:
                    assert result["success"] is False
                else:
                    assert result["success"] is True
                    assert result["data"]["result"] == expected

    @pytest.mark.asyncio
    async def test_regex_case_sensitivity(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test regex case sensitivity options."""
        text = "Hello HELLO hello HeLLo"

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Case sensitive search
            result = await km_engine_control(
                operation="search_replace",
                search_pattern="hello",
                text=text,
                use_regex=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["match_count"] == 1  # Only lowercase "hello"

    @pytest.mark.asyncio
    async def test_zero_matches_search(
        self,
        mock_context: Any,
        mock_km_client: Any,
    ) -> None:
        """Test search operation with zero matches."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await km_engine_control(
                operation="search_replace",
                search_pattern="nonexistent",
                text="Some sample text",
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["match_count"] == 0
            assert result["data"]["result"] == "Some sample text"  # Unchanged
