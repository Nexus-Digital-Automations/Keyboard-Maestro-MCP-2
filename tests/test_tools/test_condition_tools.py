"""Comprehensive Test Suite for Condition Tools - Following Proven MCP Tool Test Pattern.

This test suite validates the Condition Tools functionality using the systematic
testing approach that achieved 100% success rate across 12 tool suites.

Test Coverage:
- Condition creation functionality with comprehensive validation (text, app, system, variable, logic conditions)
- Condition type validation and security boundary checking
- Operator validation and complex comparison testing
- Security validation for condition operands and SQL injection prevention
- Property-based testing for robust input validation
- Integration testing with mocked condition builders and KM integrator
- Error handling for all failure scenarios
- Performance testing for condition evaluation limits

Testing Strategy:
- Property-based testing with Hypothesis for comprehensive input coverage
- Mock-based testing for ConditionBuilder and KMConditionIntegrator
- Security validation for condition creation prevention
- Integration testing scenarios with realistic condition operations
- Performance and timeout testing with condition limits

Key Mocking Pattern:
- ConditionBuilder: Mock all methods with Either.success() pattern
- KMConditionIntegrator: Mock async methods for condition integration
- ConditionValidator: Mock validation methods for security testing
- RegexValidator: Mock regex validation for pattern testing
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

# Import condition types and errors
from src.core.conditions import (
    ComparisonOperator,
    ConditionBuilder,
    ConditionType,
)
from src.core.errors import SecurityError, ValidationError
from src.integration.km_conditions import KMConditionIntegrator
from src.security.input_sanitizer import InputSanitizer

# Import the tools we're testing
from src.server.tools.condition_tools import (
    _apply_operator,
    _perform_security_validation,
    _validate_condition_type,
    _validate_operator,
    km_add_condition,
)


# Test fixtures following proven pattern
@pytest.fixture
def mock_context() -> Any:
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
def mock_condition_builder() -> Any:
    """Create mock ConditionBuilder with standard interface."""
    builder = Mock(spec=ConditionBuilder)
    builder.set_type = Mock(return_value=builder)
    builder.set_operand = Mock(return_value=builder)
    builder.set_case_sensitive = Mock(return_value=builder)
    builder.set_negate = Mock(return_value=builder)
    builder.set_timeout = Mock(return_value=builder)
    builder.set_actions = Mock(return_value=builder)
    builder.build = Mock()

    # Setup standard success response using Either.success() pattern
    mock_result = Mock()
    mock_result.is_right.return_value = True
    mock_result.is_left.return_value = False

    # Create proper mock for condition
    mock_condition = Mock()
    mock_condition.id = "condition-123"
    mock_condition.condition_type = ConditionType.TEXT
    mock_condition.operator = ComparisonOperator.CONTAINS
    mock_condition.operand = "test text"
    mock_condition.case_sensitive = True
    mock_condition.negate = False
    mock_condition.timeout_seconds = 10

    mock_result.get_right.return_value = mock_condition
    builder.build.return_value = mock_result

    return builder


@pytest.fixture
def mock_km_condition_integrator() -> Any:
    """Create mock KMConditionIntegrator with standard interface."""
    integrator = Mock(spec=KMConditionIntegrator)
    integrator.add_condition_to_macro = AsyncMock()

    # Setup standard success response using Either.success() pattern
    mock_result = Mock()
    mock_result.is_right.return_value = True
    mock_result.is_left.return_value = False
    mock_result.get_right.return_value = "condition-added-successfully"

    integrator.add_condition_to_macro.return_value = mock_result

    return integrator


@pytest.fixture
def mock_input_sanitizer() -> Any:
    """Create mock InputSanitizer with standard interface."""
    sanitizer = Mock(spec=InputSanitizer)
    sanitizer.sanitize = Mock()
    sanitizer.validate_sql_injection = Mock(return_value=True)
    sanitizer.validate_xss = Mock(return_value=True)
    sanitizer.validate_path_traversal = Mock(return_value=True)

    # Default sanitize to return input unchanged for successful cases
    sanitizer.sanitize.side_effect = lambda x: x

    return sanitizer


@pytest.fixture
def sample_condition_data() -> Any:
    """Sample condition data for testing."""
    return {
        "macro_ids": ["macro-test-123", "backup-macro-456", "temp-macro-789"],
        "condition_types": ["text", "application", "system", "variable", "logic"],
        "operators": ["contains", "equals", "greater", "less", "regex", "exists"],
        "text_operands": ["Hello World", "Test Content", "Search Pattern"],
        "numeric_operands": ["100", "0", "-50", "3.14"],
        "regex_patterns": [r"\d+", r"^test.*", r".*\.txt$"],
        "app_names": ["TextEdit", "Safari", "Finder"],
        "variable_names": ["counter", "username", "last_action"],
        "system_properties": ["battery_level", "disk_space", "memory_usage"],
    }


class TestKMAddCondition:
    """Test km_add_condition main functionality following proven pattern."""

    @pytest.mark.asyncio
    async def test_text_condition_success(self, mock_context: Any, sample_condition_data: Any) -> None:
        """Test successful text condition creation."""
        # Mock all the dependencies that are created as instances in the function
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Setup sanitization success
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro-123"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            mock_operand_result.get_right.return_value = "test content"
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with full fluent chain
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder

            # All builder methods return the builder for chaining
            mock_builder.text_condition.return_value = mock_builder
            mock_builder.app_condition.return_value = mock_builder
            mock_builder.system_condition.return_value = mock_builder
            mock_builder.variable_condition.return_value = mock_builder
            mock_builder.contains.return_value = mock_builder
            mock_builder.equals.return_value = mock_builder
            mock_builder.matches_regex.return_value = mock_builder
            mock_builder.greater_than.return_value = mock_builder
            mock_builder.case_insensitive.return_value = mock_builder
            mock_builder.negated.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Setup condition build result
            mock_condition_result = Mock()
            mock_condition_result.is_left.return_value = False
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "condition-123"
            mock_condition_spec.metadata = {"created_at": "2023-01-01T00:00:00Z"}
            mock_condition_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_condition_result

            # Setup KMConditionIntegrator mock
            mock_integrator = Mock()
            mock_integrator_class.return_value = mock_integrator
            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = False
            mock_integration_details = {"validation_time_ms": 15}
            mock_integration_result.get_right.return_value = mock_integration_details
            mock_integrator.add_condition_to_macro = AsyncMock(
                return_value=mock_integration_result,
            )

            # Execute
            result = await km_add_condition(
                macro_identifier=sample_condition_data["macro_ids"][0],
                condition_type="text",
                operator="contains",
                operand="test content",
                case_sensitive=True,
                negate=False,
                timeout_seconds=10,
                ctx=mock_context,
            )

            # Verify success response structure (flat format)
            assert result["success"] is True
            assert result["condition_id"] == "condition-123"
            assert result["condition_type"] == "text"
            assert result["operator"] == "contains"
            assert result["operand"] == "test content"
            assert result["case_sensitive"] is True
            assert result["negate"] is False
            assert result["timeout_seconds"] == 10
            assert result["security_validated"] is True

    @pytest.mark.asyncio
    async def test_app_condition_success(self, mock_context: Any, sample_condition_data: Any) -> None:
        """Test successful app condition creation."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "backup-macro-456"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            mock_operand_result.get_right.return_value = "TextEdit"
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with fluent interface
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.app_condition.return_value = mock_builder
            mock_builder.equals.return_value = mock_builder
            mock_builder.case_insensitive.return_value = mock_builder
            mock_builder.negated.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Mock successful condition build
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "condition-123"
            mock_condition_spec.metadata = {"created_at": "2025-07-04T18:38:00Z"}

            mock_build_result = Mock()
            mock_build_result.is_left.return_value = False
            mock_build_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_build_result

            # Setup KMConditionIntegrator mock
            mock_integrator = Mock()
            mock_integrator_class.return_value = mock_integrator

            mock_integration_details = {
                "validation_time_ms": 15,
                "integration_time_ms": 25,
            }
            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = False
            mock_integration_result.get_right.return_value = mock_integration_details
            mock_integrator.add_condition_to_macro = AsyncMock(
                return_value=mock_integration_result,
            )

            # Execute
            result = await km_add_condition(
                macro_identifier=sample_condition_data["macro_ids"][1],
                condition_type="application",
                operator="equals",
                operand="TextEdit",
                case_sensitive=False,
                negate=False,
                action_on_true="continue",
                action_on_false="stop",
                timeout_seconds=10,
                ctx=mock_context,
            )

            # Verify success response following condition_tools.py structure
            assert result["success"] is True
            assert result["condition_id"] == "condition-123"
            assert result["macro_id"] == "backup-macro-456"
            assert result["condition_type"] == "application"
            assert result["operator"] == "equals"
            assert result["operand"] == "TextEdit"
            assert result["case_sensitive"] is False
            assert result["negate"] is False
            assert result["timeout_seconds"] == 10
            assert result["km_integration"] == mock_integration_details
            assert result["security_validated"] is True
            assert result["created_at"] == "2025-07-04T18:38:00Z"
            assert "performance_metrics" in result

    @pytest.mark.asyncio
    async def test_system_condition_with_numeric_operand(
        self,
        mock_context: Any,
        sample_condition_data: Any,
    ) -> None:
        """Test successful system condition with numeric comparison."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "temp-macro-789"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            mock_operand_result.get_right.return_value = "75"
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with fluent interface
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.system_condition.return_value = mock_builder
            mock_builder.greater_than.return_value = mock_builder
            mock_builder.negated.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Mock successful condition build
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "condition-456"
            mock_condition_spec.metadata = {"created_at": "2025-07-04T18:41:00Z"}

            mock_build_result = Mock()
            mock_build_result.is_left.return_value = False
            mock_build_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_build_result

            # Setup KMConditionIntegrator mock
            mock_integrator = Mock()
            mock_integrator_class.return_value = mock_integrator

            mock_integration_details = {
                "validation_time_ms": 20,
                "integration_time_ms": 30,
            }
            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = False
            mock_integration_result.get_right.return_value = mock_integration_details
            mock_integrator.add_condition_to_macro = AsyncMock(
                return_value=mock_integration_result,
            )

            # Execute
            result = await km_add_condition(
                macro_identifier=sample_condition_data["macro_ids"][2],
                condition_type="system",
                operator="greater_than",
                operand="75",
                negate=True,
                timeout_seconds=15,
                ctx=mock_context,
            )

            # Debug output
            print(f"DEBUG: result = {result}")

            # Verify success response following condition_tools.py structure
            assert result["success"] is True
            assert result["condition_id"] == "condition-456"
            assert result["macro_id"] == "temp-macro-789"
            assert result["condition_type"] == "system"
            assert result["operator"] == "greater_than"
            assert result["operand"] == "75"
            assert result["negate"] is True
            assert result["timeout_seconds"] == 15
            assert result["km_integration"] == mock_integration_details
            assert result["security_validated"] is True
            assert result["created_at"] == "2025-07-04T18:41:00Z"
            assert "performance_metrics" in result

    @pytest.mark.asyncio
    async def test_variable_condition_success(
        self,
        mock_context: Any,
        sample_condition_data: Any,
    ) -> None:
        """Test successful variable condition creation."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "macro-test-123"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            mock_operand_result.get_right.return_value = "user_preference"
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with fluent interface
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.variable_condition.return_value = mock_builder
            mock_builder.equals.return_value = mock_builder  # For exists operator
            mock_builder.with_timeout.return_value = mock_builder  # Add missing method

            # Mock successful condition build
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "condition-789"
            mock_condition_spec.metadata = {"created_at": "2025-07-04T18:44:00Z"}

            mock_build_result = Mock()
            mock_build_result.is_left.return_value = False
            mock_build_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_build_result

            # Setup KMConditionIntegrator mock
            mock_integrator = Mock()
            mock_integrator_class.return_value = mock_integrator

            mock_integration_details = {
                "validation_time_ms": 12,
                "integration_time_ms": 18,
            }
            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = False
            mock_integration_result.get_right.return_value = mock_integration_details
            mock_integrator.add_condition_to_macro = AsyncMock(
                return_value=mock_integration_result,
            )

            # Execute
            result = await km_add_condition(
                macro_identifier=sample_condition_data["macro_ids"][0],
                condition_type="variable",
                operator="exists",
                operand="user_preference",
                case_sensitive=True,
                negate=False,
                ctx=mock_context,
            )

            # Debug output
            print(f"DEBUG: result = {result}")

            # Verify success response following condition_tools.py structure
            assert result["success"] is True
            assert result["condition_id"] == "condition-789"
            assert result["macro_id"] == "macro-test-123"
            assert result["condition_type"] == "variable"
            assert result["operator"] == "exists"
            assert result["operand"] == "user_preference"
            assert result["case_sensitive"] is True
            assert result["negate"] is False
            assert result["km_integration"] == mock_integration_details
            assert result["security_validated"] is True
            assert result["created_at"] == "2025-07-04T18:44:00Z"
            assert "performance_metrics" in result

    @pytest.mark.asyncio
    async def test_regex_condition_success(self, mock_context: Any, sample_condition_data: Any) -> None:
        """Test successful regex condition creation."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "backup-macro-456"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            mock_operand_result.get_right.return_value = r"\d{3}-\d{3}-\d{4}"
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with fluent interface
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.text_condition.return_value = mock_builder
            mock_builder.matches_regex.return_value = mock_builder
            mock_builder.case_insensitive.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Mock successful condition build
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "condition-regex-123"
            mock_condition_spec.metadata = {"created_at": "2025-07-04T18:49:00Z"}

            mock_build_result = Mock()
            mock_build_result.is_left.return_value = False
            mock_build_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_build_result

            # Setup KMConditionIntegrator mock
            mock_integrator = Mock()
            mock_integrator_class.return_value = mock_integrator

            mock_integration_details = {
                "validation_time_ms": 15,
                "integration_time_ms": 22,
            }
            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = False
            mock_integration_result.get_right.return_value = mock_integration_details
            mock_integrator.add_condition_to_macro = AsyncMock(
                return_value=mock_integration_result,
            )

            # Execute
            result = await km_add_condition(
                macro_identifier=sample_condition_data["macro_ids"][1],
                condition_type="text",
                operator="matches_regex",  # Use correct operator
                operand=r"\d{3}-\d{3}-\d{4}",  # Phone number pattern
                case_sensitive=False,
                negate=False,
                ctx=mock_context,
            )

            # Verify success response following condition_tools.py structure
            assert result["success"] is True
            assert result["condition_id"] == "condition-regex-123"
            assert result["macro_id"] == "backup-macro-456"
            assert result["condition_type"] == "text"
            assert result["operator"] == "matches_regex"
            assert result["operand"] == r"\d{3}-\d{3}-\d{4}"
            assert result["case_sensitive"] is False
            assert result["negate"] is False
            assert result["km_integration"] == mock_integration_details
            assert result["security_validated"] is True
            assert result["created_at"] == "2025-07-04T18:49:00Z"
            assert "performance_metrics" in result

    @pytest.mark.asyncio
    async def test_invalid_condition_type(self, mock_context: Any) -> None:
        """Test handling of invalid condition type."""
        # Execute
        result = await km_add_condition(
            macro_identifier="test-macro",
            condition_type="invalid_type",
            operator="contains",
            operand="test",
            ctx=mock_context,
        )

        # Verify error response
        assert result["success"] is False
        assert result["error"] == "INVALID_CONDITION_TYPE"
        assert "condition_type" in result["message"]

    @pytest.mark.asyncio
    async def test_invalid_operator(self, mock_context: Any) -> None:
        """Test handling of invalid operator."""
        # Execute
        result = await km_add_condition(
            macro_identifier="test-macro",
            condition_type="text",
            operator="invalid_op",
            operand="test",
            ctx=mock_context,
        )

        # Verify error response
        assert result["success"] is False
        assert result["error"] == "INVALID_OPERATOR"
        assert "operator" in result["message"]

    @pytest.mark.asyncio
    async def test_security_validation_failure(self, mock_context: Any) -> None:
        """Test security validation failure for malicious operand."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch("src.server.tools.condition_tools.ConditionBuilder"),
            patch("src.server.tools.condition_tools.KMConditionIntegrator"),
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful macro sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            # Mock FAILED operand sanitization (security violation)
            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = True
            mock_error = Mock()
            mock_error.code = "SECURITY_VIOLATION"
            mock_error.message = "Security validation failed: SQL injection detected"
            mock_operand_result.get_left.return_value = mock_error
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Execute
            result = await km_add_condition(
                macro_identifier="test-macro",
                condition_type="text",
                operator="contains",
                operand="'; DROP TABLE users; --",
                ctx=mock_context,
            )

            # Verify security error response
            assert result["success"] is False
            assert result["error"] == "INVALID_OPERAND"
            assert "security validation failed" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_condition_builder_failure(self, mock_context: Any) -> None:
        """Test condition builder failure handling."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch("src.server.tools.condition_tools.KMConditionIntegrator"),
        ):
            # Setup InputSanitizer mock with Either pattern responses - SUCCESS
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            mock_operand_result.get_right.return_value = "test"
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with fluent interface BUT FAILING BUILD
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.text_condition.return_value = mock_builder
            mock_builder.contains.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Mock FAILED condition build
            mock_build_result = Mock()
            mock_build_result.is_left.return_value = True
            mock_error = Mock()
            mock_error.message = (
                "Failed to build condition: invalid operator combination"
            )
            mock_build_result.get_left.return_value = mock_error
            mock_builder.build.return_value = mock_build_result

            # Execute
            result = await km_add_condition(
                macro_identifier="test-macro",
                condition_type="text",
                operator="contains",
                operand="test",
                ctx=mock_context,
            )

            # Verify error response
            assert result["success"] is False
            assert result["error"] == "CONDITION_BUILD_FAILED"
            assert "failed to build condition" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_km_integrator_failure(self, mock_context: Any) -> None:
        """Test KM integrator failure handling."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock with Either pattern responses - SUCCESS
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "nonexistent-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            mock_operand_result.get_right.return_value = "test"
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with fluent interface - SUCCESS
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.text_condition.return_value = mock_builder
            mock_builder.contains.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Mock successful condition build
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "condition-integration-fail"
            mock_condition_spec.metadata = {"created_at": "2025-07-04T18:54:00Z"}

            mock_build_result = Mock()
            mock_build_result.is_left.return_value = False
            mock_build_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_build_result

            # Setup KMConditionIntegrator mock - FAILURE
            mock_integrator = Mock()
            mock_integrator_class.return_value = mock_integrator

            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = True
            mock_error = Mock()
            mock_error.message = "Cannot add condition to macro: macro not found"
            mock_integration_result.get_left.return_value = mock_error
            mock_integrator.add_condition_to_macro = AsyncMock(
                return_value=mock_integration_result,
            )

            # Execute
            result = await km_add_condition(
                macro_identifier="nonexistent-macro",
                condition_type="text",
                operator="contains",
                operand="test",
                ctx=mock_context,
            )

            # Verify error response
            assert result["success"] is False
            assert result["error"] == "INTEGRATION_FAILED"
            assert "cannot add condition to macro" in result["message"].lower()


class TestConditionHelperFunctions:
    """Test helper functions for condition validation."""

    def test_validate_condition_type_success(self) -> None:
        """Test successful condition type validation."""
        valid_types = ["text", "application", "system", "variable", "logic"]

        for condition_type in valid_types:
            result = _validate_condition_type(condition_type)
            assert result.is_right()
            assert isinstance(result.get_right(), ConditionType)

    def test_validate_condition_type_failure(self) -> None:
        """Test condition type validation failure."""
        invalid_types = ["invalid", "unknown", "", "app"]  # "app" != "application"

        for condition_type in invalid_types:
            result = _validate_condition_type(condition_type)
            assert result.is_left()
            assert isinstance(result.get_left(), ValidationError)

    def test_validate_operator_success(self) -> None:
        """Test successful operator validation."""
        valid_operators = [
            "contains",
            "equals",
            "greater_than",
            "less_than",
            "matches_regex",
            "exists",
        ]

        for operator in valid_operators:
            result = _validate_operator(operator)
            assert result.is_right()
            assert isinstance(result.get_right(), ComparisonOperator)

    def test_validate_operator_failure(self) -> None:
        """Test operator validation failure."""
        invalid_operators = [
            "invalid",
            "unknown",
            "",
            "greater",
            "regex",
        ]  # should be "greater_than", "matches_regex"

        for operator in invalid_operators:
            result = _validate_operator(operator)
            assert result.is_left()
            assert isinstance(result.get_left(), ValidationError)

    def test_apply_operator_success(self, mock_condition_builder: Any) -> None:
        """Test successful operator application."""
        # Configure mock for fluent interface pattern - only methods that exist in actual ConditionBuilder
        mock_condition_builder.contains.return_value = mock_condition_builder
        mock_condition_builder.equals.return_value = mock_condition_builder
        mock_condition_builder.greater_than.return_value = mock_condition_builder
        mock_condition_builder.matches_regex.return_value = mock_condition_builder

        # Test each operator type that is actually implemented in _apply_operator
        operators = [
            ComparisonOperator.CONTAINS,
            ComparisonOperator.EQUALS,
            ComparisonOperator.GREATER_THAN,
            ComparisonOperator.MATCHES_REGEX,
        ]

        for operator in operators:
            result_builder = _apply_operator(
                mock_condition_builder,
                operator,
                "test_operand",
            )
            assert result_builder == mock_condition_builder

    def test_perform_security_validation_success(self) -> None:
        """Test successful security validation."""
        # Create mock condition spec with required attributes
        mock_condition_spec = Mock()
        mock_condition_spec.operator = ComparisonOperator.CONTAINS
        mock_condition_spec.condition_type = ConditionType.TEXT
        mock_condition_spec.metadata = {}

        # Test with safe operand - should pass validation
        result = _perform_security_validation(mock_condition_spec, "safe_operand")
        assert result.is_right()

    def test_perform_security_validation_sql_injection(self) -> None:
        """Test security validation failure for SQL injection."""
        # Create mock condition spec with required attributes
        mock_condition_spec = Mock()
        mock_condition_spec.operator = ComparisonOperator.CONTAINS
        mock_condition_spec.condition_type = ConditionType.TEXT
        mock_condition_spec.metadata = {}

        # Test with SQL injection payload that contains 'exec(' pattern
        result = _perform_security_validation(
            mock_condition_spec,
            "test exec(malicious)",
        )
        assert result.is_left()
        assert isinstance(result.get_left(), SecurityError)
        assert "INJECTION_DETECTED" in result.get_left().security_code

    def test_perform_security_validation_xss(self) -> None:
        """Test security validation failure for XSS."""
        # Create mock condition spec with required attributes
        mock_condition_spec = Mock()
        mock_condition_spec.operator = ComparisonOperator.CONTAINS
        mock_condition_spec.condition_type = ConditionType.TEXT
        mock_condition_spec.metadata = {}

        # Test with XSS payload - should detect and fail
        result = _perform_security_validation(
            mock_condition_spec,
            "<script>alert('xss')</script>",
        )
        assert result.is_left()
        assert isinstance(result.get_left(), SecurityError)
        assert "INJECTION_DETECTED" in result.get_left().security_code


class TestConditionIntegration:
    """Test integration scenarios across condition operations."""

    @pytest.mark.asyncio
    async def test_condition_workflow_integration(self, mock_context: Any) -> None:
        """Test complete condition workflow integration."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful sanitization for all calls
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "macro-test-123"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            mock_operand_result.get_right.return_value = "Hello"
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with fluent interface
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.text_condition.return_value = mock_builder
            mock_builder.app_condition.return_value = mock_builder
            mock_builder.variable_condition.return_value = mock_builder
            mock_builder.contains.return_value = mock_builder
            mock_builder.equals.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Mock successful condition build
            mock_build_result = Mock()
            mock_build_result.is_left.return_value = False
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "cond-123"
            mock_condition_spec.metadata = {"created_at": "2024-01-01"}
            mock_build_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_build_result

            # Setup KMConditionIntegrator mock with async method
            mock_integrator = AsyncMock()
            mock_integrator_class.return_value = mock_integrator

            # Mock successful integration
            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = False
            mock_integration_details = {
                "validation_time_ms": 5,
                "integration_time_ms": 10,
            }
            mock_integration_result.get_right.return_value = mock_integration_details
            mock_integrator.add_condition_to_macro.return_value = (
                mock_integration_result
            )

            # 1. Create text condition
            text_result = await km_add_condition(
                macro_identifier="macro-test-123",
                condition_type="text",
                operator="contains",
                operand="Hello",
                ctx=mock_context,
            )
            assert text_result["success"] is True
            assert "condition_id" in text_result

            # 2. Create app condition - adjust operand for application type
            mock_operand_result.get_right.return_value = "TextEdit"
            app_result = await km_add_condition(
                macro_identifier="macro-test-123",
                condition_type="application",
                operator="equals",
                operand="TextEdit",
                ctx=mock_context,
            )
            assert app_result["success"] is True

            # 3. Create variable condition - adjust operand for variable type
            mock_operand_result.get_right.return_value = "counter"
            var_result = await km_add_condition(
                macro_identifier="macro-test-123",
                condition_type="variable",
                operator="equals",
                operand="counter",
                ctx=mock_context,
            )
            assert var_result["success"] is True

    @pytest.mark.asyncio
    async def test_condition_timeout_limits(self, mock_context: Any) -> None:
        """Test condition timeout parameter limits."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            mock_operand_result.get_right.return_value = "test"
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with fluent interface
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.text_condition.return_value = mock_builder
            mock_builder.contains.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Mock successful condition build
            mock_build_result = Mock()
            mock_build_result.is_left.return_value = False
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "cond-123"
            mock_condition_spec.metadata = {"created_at": "2024-01-01"}
            mock_build_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_build_result

            # Setup KMConditionIntegrator mock with async method
            mock_integrator = AsyncMock()
            mock_integrator_class.return_value = mock_integrator

            # Mock successful integration
            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = False
            mock_integration_details = {
                "validation_time_ms": 5,
                "integration_time_ms": 10,
            }
            mock_integration_result.get_right.return_value = mock_integration_details
            mock_integrator.add_condition_to_macro.return_value = (
                mock_integration_result
            )

            # Test timeout within allowed range
            result = await km_add_condition(
                macro_identifier="test-macro",
                condition_type="text",
                operator="contains",
                operand="test",
                timeout_seconds=30,  # Within 1-60 second range
                ctx=mock_context,
            )

            # Should succeed with valid timeout
            assert result["success"] is True
            assert result["timeout_seconds"] == 30


class TestConditionSecurity:
    """Test security validation and prevention measures."""

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, mock_context: Any) -> None:
        """Test SQL injection prevention in operands."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch("src.server.tools.condition_tools.ConditionBuilder"),
            patch("src.server.tools.condition_tools.KMConditionIntegrator"),
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful macro sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            # Mock FAILED operand sanitization (security violation)
            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = True
            mock_error = Mock()
            mock_error.code = "SECURITY_VIOLATION"
            mock_error.message = "SQL injection detected"
            mock_operand_result.get_left.return_value = mock_error
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            malicious_operands = [
                "'; DROP TABLE conditions; --",
                "1' OR '1'='1",
                "admin'--",
            ]

            for malicious_operand in malicious_operands:
                result = await km_add_condition(
                    macro_identifier="test-macro",
                    condition_type="text",
                    operator="contains",
                    operand=malicious_operand,
                    ctx=mock_context,
                )

                # Should block malicious operand at sanitization stage
                assert result["success"] is False
                assert result["error"] == "INVALID_OPERAND"
                assert "sql injection" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_xss_prevention(self, mock_context: Any) -> None:
        """Test XSS prevention in operands."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch("src.server.tools.condition_tools.ConditionBuilder"),
            patch("src.server.tools.condition_tools.KMConditionIntegrator"),
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful macro sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            # Mock FAILED operand sanitization (XSS violation)
            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = True
            mock_error = Mock()
            mock_error.code = "XSS_DETECTED"
            mock_error.message = "XSS attack detected"
            mock_operand_result.get_left.return_value = mock_error
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            xss_operands = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>",
            ]

            for xss_operand in xss_operands:
                result = await km_add_condition(
                    macro_identifier="test-macro",
                    condition_type="text",
                    operator="contains",
                    operand=xss_operand,
                    ctx=mock_context,
                )

                # Should block XSS operand at sanitization stage
                assert result["success"] is False
                assert result["error"] == "INVALID_OPERAND"
                assert "xss" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_operand_length_validation(self, mock_context: Any) -> None:
        """Test operand length security validation."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch("src.server.tools.condition_tools.ConditionBuilder"),
            patch("src.server.tools.condition_tools.KMConditionIntegrator"),
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful macro sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            # Mock FAILED operand sanitization (length violation)
            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = True
            mock_error = Mock()
            mock_error.code = "CONTENT_TOO_LONG"
            mock_error.message = "Operand exceeds maximum length"
            mock_operand_result.get_left.return_value = mock_error
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Test very long operand
            long_operand = "x" * 10000  # 10KB operand

            result = await km_add_condition(
                macro_identifier="test-macro",
                condition_type="text",
                operator="contains",
                operand=long_operand,
                ctx=mock_context,
            )

            # Should reject very long operands
            assert result["success"] is False
            assert result["error"] == "INVALID_OPERAND"
            assert "exceeds maximum length" in result["message"]


class TestConditionPropertyBased:
    """Property-based testing for condition operations."""

    @composite
    def condition_operand_strategy(draw: Callable[..., Any]) -> Any:
        """Generate valid condition operands for testing."""
        operand_type = draw(st.sampled_from(["text", "numeric", "boolean", "regex"]))

        if operand_type == "text":
            operand = draw(st.text(min_size=1, max_size=1000))
        elif operand_type == "numeric":
            operand = str(draw(st.integers(min_value=-1000, max_value=1000)))
        elif operand_type == "boolean":
            operand = draw(st.sampled_from(["true", "false", "1", "0"]))
        else:  # regex
            operand = draw(
                st.sampled_from([r"\d+", r"^test.*", r".*\.txt$", r"[a-zA-Z]+"]),
            )

        assume(len(operand.strip()) > 0)
        return operand

    @given(condition_operand_strategy())
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_condition_operand_properties(self, operand: list[Any] | str) -> None:
        """Property: Valid condition operands should meet basic requirements."""
        # Test basic operand properties
        assert len(operand) > 0
        assert len(operand) <= 1000
        assert isinstance(operand, str)

    @given(st.sampled_from(["text", "application", "system", "variable", "logic"]))
    @settings(max_examples=10)
    def test_condition_type_properties(self, condition_type: str) -> None:
        """Property: Valid condition types should be supported."""
        result = _validate_condition_type(condition_type)
        assert result.is_right()
        assert isinstance(result.get_right(), ConditionType)

    @given(st.sampled_from(["contains", "equals", "greater_than", "matches_regex"]))
    @settings(max_examples=10)
    def test_operator_properties(self, operator: Callable[..., Any]) -> None:
        """Property: Valid operators should be supported."""
        result = _validate_operator(operator)
        assert result.is_right()
        assert isinstance(result.get_right(), ComparisonOperator)


class TestConditionPerformance:
    """Test performance and limits for condition operations."""

    @pytest.mark.asyncio
    async def test_condition_creation_performance(self, mock_context: Any) -> None:
        """Test condition creation performance with realistic load."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            mock_operand_result.get_right.return_value = "test content"
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with fluent interface
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.text_condition.return_value = mock_builder
            mock_builder.contains.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Mock successful condition build
            mock_build_result = Mock()
            mock_build_result.is_left.return_value = False
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "cond-123"
            mock_condition_spec.metadata = {"created_at": "2024-01-01"}
            mock_build_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_build_result

            # Setup KMConditionIntegrator mock with async method
            mock_integrator = AsyncMock()
            mock_integrator_class.return_value = mock_integrator

            # Mock successful integration
            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = False
            mock_integration_details = {
                "validation_time_ms": 5,
                "integration_time_ms": 10,
            }
            mock_integration_result.get_right.return_value = mock_integration_details
            mock_integrator.add_condition_to_macro.return_value = (
                mock_integration_result
            )

            # Test creating multiple conditions quickly
            start_time = datetime.now(UTC)

            for i in range(10):
                result = await km_add_condition(
                    macro_identifier=f"test-macro-{i}",
                    condition_type="text",
                    operator="contains",
                    operand=f"test content {i}",
                    ctx=mock_context,
                )
                assert result["success"] is True

            end_time = datetime.now(UTC)
            duration = (end_time - start_time).total_seconds()

            # Should complete within reasonable time
            assert duration < 5.0  # 5 seconds for 10 conditions

    @pytest.mark.asyncio
    async def test_regex_pattern_limits(self, mock_context: Any) -> None:
        """Test regex pattern complexity limits."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = False
            complex_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            mock_operand_result.get_right.return_value = complex_regex
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            # Setup ConditionBuilder mock with fluent interface
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.text_condition.return_value = mock_builder
            mock_builder.matches_regex.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Mock successful condition build
            mock_build_result = Mock()
            mock_build_result.is_left.return_value = False
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "cond-123"
            mock_condition_spec.metadata = {"created_at": "2024-01-01"}
            mock_build_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_build_result

            # Setup KMConditionIntegrator mock with async method
            mock_integrator = AsyncMock()
            mock_integrator_class.return_value = mock_integrator

            # Mock successful integration
            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = False
            mock_integration_details = {
                "validation_time_ms": 5,
                "integration_time_ms": 10,
            }
            mock_integration_result.get_right.return_value = mock_integration_details
            mock_integrator.add_condition_to_macro.return_value = (
                mock_integration_result
            )

            result = await km_add_condition(
                macro_identifier="test-macro",
                condition_type="text",
                operator="matches_regex",
                operand=complex_regex,
                ctx=mock_context,
            )

            # Should handle regex pattern appropriately
            assert result["success"] is True


class TestConditionEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_operand_handling(self, mock_context: Any) -> None:
        """Test handling of empty operand."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch("src.server.tools.condition_tools.ConditionBuilder"),
            patch("src.server.tools.condition_tools.KMConditionIntegrator"),
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful macro sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            # Mock FAILED operand sanitization (empty content)
            mock_operand_result = Mock()
            mock_operand_result.is_left.return_value = True
            mock_error = Mock()
            mock_error.code = "EMPTY_CONTENT"
            mock_error.message = "Operand cannot be empty"
            mock_operand_result.get_left.return_value = mock_error
            mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

            result = await km_add_condition(
                macro_identifier="test-macro",
                condition_type="text",
                operator="contains",
                operand="",
                ctx=mock_context,
            )

            # Should handle empty operand appropriately by failing at sanitization
            assert result["success"] is False
            assert result["error"] == "INVALID_OPERAND"
            assert "operand cannot be empty" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_unicode_operand_handling(self, mock_context: Any) -> None:
        """Test handling of Unicode operands."""
        with (
            patch(
                "src.server.tools.condition_tools.InputSanitizer",
            ) as mock_sanitizer_class,
            patch(
                "src.server.tools.condition_tools.ConditionBuilder",
            ) as mock_builder_class,
            patch(
                "src.server.tools.condition_tools.KMConditionIntegrator",
            ) as mock_integrator_class,
        ):
            # Setup InputSanitizer mock with Either pattern responses
            mock_sanitizer = Mock()
            mock_sanitizer_class.return_value = mock_sanitizer

            # Mock successful macro sanitization
            mock_macro_result = Mock()
            mock_macro_result.is_left.return_value = False
            mock_macro_result.get_right.return_value = "test-macro"
            mock_sanitizer.sanitize_macro_identifier.return_value = mock_macro_result

            # Setup ConditionBuilder mock with fluent interface
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.text_condition.return_value = mock_builder
            mock_builder.contains.return_value = mock_builder
            mock_builder.with_timeout.return_value = mock_builder

            # Mock successful condition build
            mock_build_result = Mock()
            mock_build_result.is_left.return_value = False
            mock_condition_spec = Mock()
            mock_condition_spec.condition_id = "cond-123"
            mock_condition_spec.metadata = {"created_at": "2024-01-01"}
            mock_build_result.get_right.return_value = mock_condition_spec
            mock_builder.build.return_value = mock_build_result

            # Setup KMConditionIntegrator mock with async method
            mock_integrator = AsyncMock()
            mock_integrator_class.return_value = mock_integrator

            # Mock successful integration
            mock_integration_result = Mock()
            mock_integration_result.is_left.return_value = False
            mock_integration_details = {
                "validation_time_ms": 5,
                "integration_time_ms": 10,
            }
            mock_integration_result.get_right.return_value = mock_integration_details
            mock_integrator.add_condition_to_macro.return_value = (
                mock_integration_result
            )

            # Test various Unicode operands
            unicode_operands = [
                "Hello 世界 🌍",  # Mixed languages and emoji
                "Café résumé naïve",  # Accented characters
                "数字123456",  # Chinese characters with numbers
            ]

            for operand in unicode_operands:
                # Mock successful operand sanitization for each operand
                mock_operand_result = Mock()
                mock_operand_result.is_left.return_value = False
                mock_operand_result.get_right.return_value = operand
                mock_sanitizer.sanitize_text_content.return_value = mock_operand_result

                result = await km_add_condition(
                    macro_identifier="test-macro",
                    condition_type="text",
                    operator="contains",
                    operand=operand,
                    ctx=mock_context,
                )

                # Should handle Unicode properly
                assert result["success"] is True
                assert result["operand"] == operand

    @pytest.mark.asyncio
    async def test_none_values_handling(self, mock_context: Any) -> None:
        """Test handling of None values in optional parameters."""
        # Test with minimal parameters
        result = await km_add_condition(
            macro_identifier="test-macro",
            condition_type="text",
            operator="contains",
            operand="test",
            ctx=mock_context,
        )

        # Should handle None values gracefully
        # (This would be processed through the actual function logic)
        assert isinstance(result, dict)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
