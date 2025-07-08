"""Comprehensive tests for Engine Tools module with systematic coverage.

Tests cover engine control operations including reload, calculate, process_tokens,
search_replace, and status operations with comprehensive validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

# Import the actual engine tools module
from src.server.tools import engine_tools

if TYPE_CHECKING:
    from collections.abc import Callable

# Test constants - Keyboard Maestro token expressions for testing
KM_TIME_TOKEN_EXPRESSION = "Current time: %Time%"  # noqa: S105 # Test constant
KM_USER_TOKEN_EXPRESSION = "Current user: %UserName%"  # noqa: S105 # Test constant


# Test data generators
@st.composite
def expression_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid calculation expressions."""
    operations = ["+", "-", "*", "/"]
    number1 = draw(st.integers(min_value=1, max_value=100))
    number2 = draw(st.integers(min_value=1, max_value=100))
    operation = draw(st.sampled_from(operations))
    return f"{number1} {operation} {number2}"


@st.composite
def search_text_strategy(draw: Callable[..., Any]) -> list[Any]:
    """Generate text for search/replace operations."""
    return draw(
        st.text(
            min_size=10,
            max_size=200,
            alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd", "Pc", "Zs"]),
        ),
    )


@st.composite
def token_string_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid token strings."""
    tokens = ["%CurrentDirectory%", "%FrontmostApplication%", "%Time%", "%Date%"]
    token = draw(st.sampled_from(tokens))
    return f"Processing {token} token"


@st.composite
def search_pattern_strategy(draw: Callable[..., Any]) -> list[Any]:
    """Generate search patterns."""
    patterns = ["hello", "world", r"\d+", "[a-z]+", "test.*pattern"]
    return draw(st.sampled_from(patterns))


class TestEngineControlOperations:
    """Test main engine control operations."""

    @pytest.fixture
    def mock_km_client(self) -> Mock:
        """Create mock KM client for testing."""
        client = Mock()
        client.reload_engine = AsyncMock()
        client.get_engine_status = AsyncMock()
        client.calculate = AsyncMock()
        client.process_tokens = AsyncMock()
        client.search_replace = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_engine_control_reload_operation(self, mock_km_client: Any) -> None:
        """Test engine reload operation."""
        # Mock successful reload
        mock_km_client.reload_engine.return_value = {
            "success": True,
            "message": "Engine reloaded successfully",
        }

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await engine_tools.km_engine_control(operation="reload")

        assert result["success"] is True
        assert result["operation"] == "reload"
        assert "reloaded" in result["message"].lower()

        # Verify client was called
        mock_km_client.reload_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_engine_control_status_operation(self, mock_km_client: Any) -> None:
        """Test engine status operation."""
        # Mock status response
        status_data = {
            "state": "running",
            "version": "10.2",
            "active_macros": 5,
            "total_macros": 150,
            "uptime": 3600,
        }
        mock_km_client.get_engine_status.return_value = status_data

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await engine_tools.km_engine_control(operation="status")

        assert result["success"] is True
        assert result["operation"] == "status"
        assert result["status"]["state"] == "running"
        assert result["status"]["active_macros"] == 5

        # Verify client was called
        mock_km_client.get_engine_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_engine_control_calculate_operation(
        self,
        mock_km_client: Any,
    ) -> None:
        """Test engine calculate operation."""
        # Mock calculation response
        mock_km_client.calculate.return_value = {
            "result": 42,
            "expression": "21 * 2",
            "success": True,
        }

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await engine_tools.km_engine_control(
                operation="calculate",
                expression="21 * 2",
            )

        assert result["success"] is True
        assert result["operation"] == "calculate"
        assert result["calculation"]["result"] == 42
        assert result["calculation"]["expression"] == "21 * 2"

        # Verify client was called correctly
        mock_km_client.calculate.assert_called_once_with("21 * 2")

    @pytest.mark.asyncio
    async def test_engine_control_process_tokens_operation(
        self,
        mock_km_client: Any,
    ) -> None:
        """Test engine process tokens operation."""
        # Mock token processing response
        mock_km_client.process_tokens.return_value = {
            "processed_text": "Current time: 14:30:00",
            "original_text": "Current time: %Time%",
            "tokens_found": 1,
            "success": True,
        }

        # Use constant to avoid security scanner false positive
        token_expression = KM_TIME_TOKEN_EXPRESSION

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await engine_tools.km_engine_control(
                operation="process_tokens",
                expression=token_expression,
            )

        assert result["success"] is True
        assert result["operation"] == "process_tokens"
        assert result["token_processing"]["processed_text"] == "Current time: 14:30:00"
        assert result["token_processing"]["tokens_found"] == 1

        # Verify client was called correctly
        mock_km_client.process_tokens.assert_called_once_with(token_expression)

    @pytest.mark.asyncio
    async def test_engine_control_search_replace_operation(
        self,
        mock_km_client: Any,
    ) -> None:
        """Test engine search/replace operation."""
        # Mock search/replace response
        mock_km_client.search_replace.return_value = {
            "result_text": "Hello World! How are you World?",
            "original_text": "Hello Universe! How are you Universe?",
            "replacements_made": 2,
            "search_pattern": "Universe",
            "replace_pattern": "World",
            "success": True,
        }

        text = "Hello Universe! How are you Universe?"
        search_pattern = "Universe"
        replace_pattern = "World"

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await engine_tools.km_engine_control(
                operation="search_replace",
                text=text,
                search_pattern=search_pattern,
                replace_pattern=replace_pattern,
                use_regex=False,
            )

        assert result["success"] is True
        assert result["operation"] == "search_replace"
        assert (
            result["search_replace"]["result_text"] == "Hello World! How are you World?"
        )
        assert result["search_replace"]["replacements_made"] == 2

        # Verify client was called correctly
        mock_km_client.search_replace.assert_called_once_with(
            text,
            search_pattern,
            replace_pattern,
            False,
        )

    @pytest.mark.asyncio
    async def test_engine_control_invalid_operation(self, mock_km_client: Any) -> None:
        """Test engine control with invalid operation."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # This should be caught by pydantic validation, but test error handling
            # B017 fix: Use specific exception for validation errors
            with pytest.raises((ValueError, TypeError)):  # Pydantic validation error
                await engine_tools.km_engine_control(operation="invalid_operation")

    @pytest.mark.asyncio
    async def test_engine_control_missing_required_params(
        self,
        mock_km_client: Any,
    ) -> None:
        """Test engine control with missing required parameters."""
        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await engine_tools.km_engine_control(
                operation="calculate",
                # Missing expression parameter
            )

        assert result["success"] is False
        assert "expression required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_engine_control_client_error(self, mock_km_client: Any) -> None:
        """Test engine control with client error."""
        # Mock client error
        mock_km_client.reload_engine.side_effect = Exception("Client connection failed")

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await engine_tools.km_engine_control(operation="reload")

        assert result["success"] is False
        assert "error" in result
        assert result["operation"] == "reload"

    @given(expression_strategy())
    @pytest.mark.asyncio
    async def test_engine_control_calculate_property_based(
        self,
        expression: str,
        mock_km_client: Any,
    ) -> None:
        """Property-based test for calculate operation."""
        # Mock calculation response
        mock_km_client.calculate.return_value = {
            "result": 100,  # Fixed result for property testing
            "expression": expression,
            "success": True,
        }

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await engine_tools.km_engine_control(
                operation="calculate",
                expression=expression,
            )

        assert result["success"] is True
        assert result["operation"] == "calculate"
        assert result["calculation"]["expression"] == expression

        # Verify client was called with the expression
        mock_km_client.calculate.assert_called_once_with(expression)

    @given(token_string_strategy())
    @pytest.mark.asyncio
    async def test_engine_control_tokens_property_based(
        self,
        token_string: Any,
        mock_km_client: Any,
    ) -> None:
        """Property-based test for token processing operation."""
        # Mock token processing response
        mock_km_client.process_tokens.return_value = {
            "processed_text": token_string.replace("%Time%", "14:30:00"),
            "original_text": token_string,
            "tokens_found": 1,
            "success": True,
        }

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await engine_tools.km_engine_control(
                operation="process_tokens",
                expression=token_string,
            )

        assert result["success"] is True
        assert result["operation"] == "process_tokens"
        assert result["token_processing"]["original_text"] == token_string

        # Verify client was called with the token string
        mock_km_client.process_tokens.assert_called_once_with(token_string)


class TestEngineHelperFunctions:
    """Test helper functions in engine tools."""

    @pytest.fixture
    def mock_km_client(self) -> Mock:
        """Create mock KM client for testing."""
        client = Mock()
        client.reload_engine = AsyncMock()
        client.get_engine_status = AsyncMock()
        client.calculate = AsyncMock()
        client.process_tokens = AsyncMock()
        client.search_replace = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_reload_engine_function(self, mock_km_client: Any) -> None:
        """Test _reload_engine helper function."""
        # Mock successful reload
        mock_km_client.reload_engine.return_value = True

        result = await engine_tools._reload_engine(mock_km_client)

        assert result["success"] is True
        assert "reloaded" in result["message"].lower()

        # Verify client was called
        mock_km_client.reload_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_reload_engine_function_error(self, mock_km_client: Any) -> None:
        """Test _reload_engine helper function with error."""
        # Mock reload error
        mock_km_client.reload_engine.side_effect = Exception("Reload failed")

        result = await engine_tools._reload_engine(mock_km_client)

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_engine_status_function(self, mock_km_client: Any) -> None:
        """Test _get_engine_status helper function."""
        # Mock status response
        status_data = {"state": "running", "version": "10.2", "active_macros": 3}
        mock_km_client.get_engine_status.return_value = status_data

        result = await engine_tools._get_engine_status(mock_km_client)

        assert result["success"] is True
        assert result["status"]["state"] == "running"
        assert result["status"]["active_macros"] == 3

        # Verify client was called
        mock_km_client.get_engine_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_expression_function(self, mock_km_client: Any) -> None:
        """Test _calculate_expression helper function."""
        # Mock calculation response
        mock_km_client.calculate.return_value = {
            "result": 15,
            "expression": "3 * 5",
            "success": True,
        }

        result = await engine_tools._calculate_expression(mock_km_client, "3 * 5")

        assert result["success"] is True
        assert result["calculation"]["result"] == 15
        assert result["calculation"]["expression"] == "3 * 5"

        # Verify client was called correctly
        mock_km_client.calculate.assert_called_once_with("3 * 5")

    @pytest.mark.asyncio
    async def test_calculate_expression_function_error(
        self,
        mock_km_client: Any,
    ) -> None:
        """Test _calculate_expression helper function with error."""
        # Mock calculation error
        mock_km_client.calculate.side_effect = Exception("Calculation failed")

        result = await engine_tools._calculate_expression(
            mock_km_client,
            "invalid / expression",
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_process_tokens_function(self, mock_km_client: Any) -> None:
        """Test _process_tokens helper function."""
        # Mock token processing response
        mock_km_client.process_tokens.return_value = {
            "processed_text": "Current user: John",
            "original_text": "Current user: %UserName%",
            "tokens_found": 1,
            "success": True,
        }

        # Use constant to avoid security scanner false positive
        token_expression = KM_USER_TOKEN_EXPRESSION
        result = await engine_tools._process_tokens(mock_km_client, token_expression)

        assert result["success"] is True
        assert result["token_processing"]["processed_text"] == "Current user: John"
        assert result["token_processing"]["tokens_found"] == 1

        # Verify client was called correctly
        mock_km_client.process_tokens.assert_called_once_with(token_expression)

    @pytest.mark.asyncio
    async def test_search_replace_function(self, mock_km_client: Any) -> None:
        """Test _search_replace helper function."""
        # Mock search/replace response
        mock_km_client.search_replace.return_value = {
            "result_text": "Hello Python! Welcome to Python!",
            "original_text": "Hello Java! Welcome to Java!",
            "replacements_made": 2,
            "search_pattern": "Java",
            "replace_pattern": "Python",
            "success": True,
        }

        text = "Hello Java! Welcome to Java!"
        search_pattern = "Java"
        replace_pattern = "Python"

        result = await engine_tools._search_replace(
            mock_km_client,
            text,
            search_pattern,
            replace_pattern,
            False,
        )

        assert result["success"] is True
        assert (
            result["search_replace"]["result_text"]
            == "Hello Python! Welcome to Python!"
        )
        assert result["search_replace"]["replacements_made"] == 2

        # Verify client was called correctly
        mock_km_client.search_replace.assert_called_once_with(
            text,
            search_pattern,
            replace_pattern,
            False,
        )

    @pytest.mark.asyncio
    async def test_search_replace_function_with_regex(
        self,
        mock_km_client: Any,
    ) -> None:
        """Test _search_replace helper function with regex."""
        # Mock regex search/replace response
        mock_km_client.search_replace.return_value = {
            "result_text": "Number: XXX, Another: XXX",
            "original_text": "Number: 123, Another: 456",
            "replacements_made": 2,
            "search_pattern": r"\d+",
            "replace_pattern": "XXX",
            "success": True,
        }

        text = "Number: 123, Another: 456"
        search_pattern = r"\d+"
        replace_pattern = "XXX"

        result = await engine_tools._search_replace(
            mock_km_client,
            text,
            search_pattern,
            replace_pattern,
            True,
        )

        assert result["success"] is True
        assert result["search_replace"]["result_text"] == "Number: XXX, Another: XXX"
        assert result["search_replace"]["replacements_made"] == 2

        # Verify client was called with regex enabled
        mock_km_client.search_replace.assert_called_once_with(
            text,
            search_pattern,
            replace_pattern,
            True,
        )


class TestEngineToolsIntegration:
    """Integration tests for engine tools functionality."""

    @pytest.mark.asyncio
    async def test_multiple_operations_sequence(self) -> None:
        """Test sequence of multiple engine operations."""
        mock_km_client = Mock()
        mock_km_client.reload_engine = AsyncMock(return_value=True)
        mock_km_client.get_engine_status = AsyncMock(return_value={"state": "running"})
        mock_km_client.calculate = AsyncMock(
            return_value={"result": 10, "success": True},
        )

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Step 1: Reload engine
            reload_result = await engine_tools.km_engine_control(operation="reload")
            assert reload_result["success"] is True

            # Step 2: Check status
            status_result = await engine_tools.km_engine_control(operation="status")
            assert status_result["success"] is True
            assert status_result["status"]["state"] == "running"

            # Step 3: Perform calculation
            calc_result = await engine_tools.km_engine_control(
                operation="calculate",
                expression="5 + 5",
            )
            assert calc_result["success"] is True
            assert calc_result["calculation"]["result"] == 10

        # Verify all operations were called
        mock_km_client.reload_engine.assert_called_once()
        mock_km_client.get_engine_status.assert_called_once()
        mock_km_client.calculate.assert_called_once_with("5 + 5")

    @pytest.mark.asyncio
    async def test_error_recovery_patterns(self) -> None:
        """Test error recovery patterns across operations."""
        mock_km_client = Mock()

        # Setup different error scenarios
        mock_km_client.reload_engine = AsyncMock(side_effect=Exception("Reload error"))
        mock_km_client.get_engine_status = AsyncMock(return_value={"state": "error"})
        mock_km_client.calculate = AsyncMock(
            return_value={"result": 0, "success": True},
        )

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Operation 1: Fails
            reload_result = await engine_tools.km_engine_control(operation="reload")
            assert reload_result["success"] is False

            # Operation 2: Shows error state
            status_result = await engine_tools.km_engine_control(operation="status")
            assert status_result["success"] is True
            assert status_result["status"]["state"] == "error"

            # Operation 3: Still works despite engine error state
            calc_result = await engine_tools.km_engine_control(
                operation="calculate",
                expression="0 + 0",
            )
            assert calc_result["success"] is True

    @given(search_text_strategy(), search_pattern_strategy())
    @pytest.mark.asyncio
    async def test_search_replace_integration_property_based(
        self,
        text: str,
        search_pattern: Any,
    ) -> None:
        """Property-based integration test for search/replace."""
        assume(search_pattern in text or len(search_pattern) > 0)

        mock_km_client = Mock()
        # Mock a simple replacement
        result_text = (
            text.replace(search_pattern, "REPLACED") if search_pattern in text else text
        )
        replacements = text.count(search_pattern)

        mock_km_client.search_replace = AsyncMock(
            return_value={
                "result_text": result_text,
                "original_text": text,
                "replacements_made": replacements,
                "search_pattern": search_pattern,
                "replace_pattern": "REPLACED",
                "success": True,
            },
        )

        with patch(
            "src.server.tools.engine_tools.get_km_client",
            return_value=mock_km_client,
        ):
            result = await engine_tools.km_engine_control(
                operation="search_replace",
                text=text,
                search_pattern=search_pattern,
                replace_pattern="REPLACED",
                use_regex=False,
            )

        assert result["success"] is True
        assert result["search_replace"]["original_text"] == text
        assert result["search_replace"]["search_pattern"] == search_pattern
        assert result["search_replace"]["replace_pattern"] == "REPLACED"

        # Verify client was called correctly
        mock_km_client.search_replace.assert_called_once_with(
            text,
            search_pattern,
            "REPLACED",
            False,
        )
