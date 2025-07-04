"""
Unit tests for control flow tools.

Tests the MCP tool interface for control flow functionality including parameter
validation, security checks, and Keyboard Maestro integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List, Optional

from src.server.tools.control_flow_tools import (
    km_control_flow, _validate_control_flow_inputs, _build_control_flow_structure,
    _generate_km_control_flow, _apply_control_flow_to_macro, _get_structure_info
)
from src.core.control_flow import (
    ControlFlowValidator, SecurityLimits, ComparisonOperator,
    IfThenElseNode, ForLoopNode, WhileLoopNode, SwitchCaseNode
)
from src.core.errors import ValidationError, SecurityError, ExecutionError


class TestKMControlFlowTool:
    """Test the main km_control_flow MCP tool."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        context = Mock()
        context.info = AsyncMock()
        context.error = AsyncMock()
        return context
    
    @pytest.mark.asyncio
    async def test_if_then_else_success(self, mock_context):
        """Test successful if/then/else creation."""
        with patch('src.server.tools.control_flow_tools._apply_control_flow_to_macro') as mock_apply:
            mock_apply.return_value = {"applied": True, "macro_id": "test_macro"}
            
            result = await km_control_flow(
                macro_identifier="test_macro",
                control_type="if_then_else",
                condition="clipboard_content",
                operator="contains",
                operand="password",
                actions_true=[{"type": "show_notification", "title": "Security Alert"}],
                actions_false=[{"type": "continue_processing"}],
                ctx=mock_context
            )
        
        assert result["success"] is True
        assert result["data"]["control_type"] == "if_then_else"
        assert result["data"]["macro_id"] == "test_macro"
        assert "control_flow_id" in result["data"]
        assert result["data"]["security_validation"] == "passed"
        
        # Verify context calls
        mock_context.info.assert_called()
    
    @pytest.mark.asyncio
    async def test_for_loop_success(self, mock_context):
        """Test successful for loop creation."""
        with patch('src.server.tools.control_flow_tools._apply_control_flow_to_macro') as mock_apply:
            mock_apply.return_value = {"applied": True, "macro_id": "test_macro"}
            
            result = await km_control_flow(
                macro_identifier="test_macro",
                control_type="for_loop",
                iterator="file",
                collection="selected_files_in_finder",
                loop_actions=[
                    {"type": "open_file", "file": "%Variable%file%"},
                    {"type": "process_document"}
                ],
                max_iterations=50,
                ctx=mock_context
            )
        
        assert result["success"] is True
        assert result["data"]["control_type"] == "for_loop"
        assert result["data"]["structure_info"]["iterator_variable"] == "file"
        assert result["data"]["structure_info"]["max_iterations"] == 50
    
    @pytest.mark.asyncio
    async def test_while_loop_success(self, mock_context):
        """Test successful while loop creation."""
        with patch('src.server.tools.control_flow_tools._apply_control_flow_to_macro') as mock_apply:
            mock_apply.return_value = {"applied": True, "macro_id": "test_macro"}
            
            result = await km_control_flow(
                macro_identifier="test_macro",
                control_type="while_loop",
                condition="counter",
                operator="less_than",
                operand="10",
                loop_actions=[{"type": "increment_counter"}],
                max_iterations=20,
                ctx=mock_context
            )
        
        assert result["success"] is True
        assert result["data"]["control_type"] == "while_loop"
        assert result["data"]["structure_info"]["max_iterations"] == 20
    
    @pytest.mark.asyncio
    async def test_switch_case_success(self, mock_context):
        """Test successful switch/case creation."""
        with patch('src.server.tools.control_flow_tools._apply_control_flow_to_macro') as mock_apply:
            mock_apply.return_value = {"applied": True, "macro_id": "test_macro"}
            
            cases = [
                {"value": "Safari", "actions": [{"type": "screenshot"}]},
                {"value": "Chrome", "actions": [{"type": "export_bookmarks"}]}
            ]
            
            result = await km_control_flow(
                macro_identifier="test_macro",
                control_type="switch_case",
                condition="frontmost_application",
                cases=cases,
                default_actions=[{"type": "show_notification", "text": "Unsupported app"}],
                ctx=mock_context
            )
        
        assert result["success"] is True
        assert result["data"]["control_type"] == "switch_case"
        assert result["data"]["structure_info"]["case_count"] == 2
        assert result["data"]["structure_info"]["has_default"] is True
    
    @pytest.mark.asyncio
    async def test_validation_error(self, mock_context):
        """Test validation error handling."""
        result = await km_control_flow(
            macro_identifier="",  # Invalid empty identifier
            control_type="if_then_else",
            condition="test",
            ctx=mock_context
        )
        
        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert "identifier cannot be empty" in result["error"]["message"]
        mock_context.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_security_error(self, mock_context):
        """Test security error handling."""
        result = await km_control_flow(
            macro_identifier="test_macro",
            control_type="if_then_else",
            condition="exec('rm -rf /')",  # Dangerous condition
            operator="equals",
            operand="test",
            actions_true=[{"type": "test"}],
            ctx=mock_context
        )
        
        assert result["success"] is False
        assert result["error"]["code"] == "SECURITY_ERROR"
        assert "dangerous pattern" in result["error"]["message"].lower()
        mock_context.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_missing_required_parameters(self, mock_context):
        """Test missing required parameters."""
        # If/then/else without condition
        result1 = await km_control_flow(
            macro_identifier="test_macro",
            control_type="if_then_else",
            # Missing condition
            ctx=mock_context
        )
        
        assert result1["success"] is False
        assert result1["error"]["code"] == "VALIDATION_ERROR"
        
        # For loop without iterator
        result2 = await km_control_flow(
            macro_identifier="test_macro",
            control_type="for_loop",
            # Missing iterator and collection
            ctx=mock_context
        )
        
        assert result2["success"] is False
        assert result2["error"]["code"] == "VALIDATION_ERROR"
    
    @pytest.mark.asyncio
    async def test_security_bounds_validation(self, mock_context):
        """Test security bounds validation."""
        # Test max iterations limit
        result = await km_control_flow(
            macro_identifier="test_macro",
            control_type="for_loop",
            iterator="i",
            collection="items",
            loop_actions=[{"type": "test"}],
            max_iterations=15000,  # Over limit
            ctx=mock_context
        )
        
        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert "max_iterations must be between 1 and 10000" in result["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_timeout_validation(self, mock_context):
        """Test timeout validation."""
        result = await km_control_flow(
            macro_identifier="test_macro",
            control_type="if_then_else",
            condition="test",
            operator="equals",
            operand="value",
            actions_true=[{"type": "test"}],
            timeout_seconds=500,  # Over limit
            ctx=mock_context
        )
        
        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert "timeout_seconds must be between 1 and 300" in result["error"]["message"]


class TestInputValidation:
    """Test input validation functions."""
    
    @pytest.mark.asyncio
    async def test_valid_inputs(self):
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
            ctx=None
        )
        # Should not raise any exception
    
    @pytest.mark.asyncio
    async def test_invalid_macro_identifier(self):
        """Test invalid macro identifier validation."""
        with pytest.raises(ValidationError, match="identifier cannot be empty"):
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
                ctx=None
            )
        
        with pytest.raises(ValidationError, match="identifier too long"):
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
                ctx=None
            )
    
    @pytest.mark.asyncio
    async def test_invalid_control_type(self):
        """Test invalid control type validation."""
        with pytest.raises(ValidationError, match="Invalid control type"):
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
                ctx=None
            )
    
    @pytest.mark.asyncio
    async def test_invalid_operator(self):
        """Test invalid operator validation."""
        with pytest.raises(ValidationError, match="Invalid operator"):
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
                ctx=None
            )
    
    @pytest.mark.asyncio
    async def test_dangerous_condition_patterns(self):
        """Test dangerous pattern detection in conditions."""
        dangerous_conditions = [
            "exec('malicious code')",
            "import os; os.system('rm -rf /')",
            "subprocess.call(['rm', '-rf', '/'])",
            "eval('dangerous')",
            "format(malicious)",
            "curl http://evil.com"
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
                    ctx=None
                )
    
    @pytest.mark.asyncio
    async def test_for_loop_validation(self):
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
                ctx=None
            )
        
        with pytest.raises(ValidationError, match="For loop requires a collection"):
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
                ctx=None
            )


class TestStructureBuilding:
    """Test control flow structure building."""
    
    @pytest.mark.asyncio
    async def test_build_if_then_else(self):
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
            ctx=None
        )
        
        assert isinstance(node, IfThenElseNode)
        assert node.condition.expression == "clipboard_content"
        assert node.condition.operator == ComparisonOperator.CONTAINS
        assert node.condition.operand == "password"
        assert len(node.then_actions.actions) == 1
        assert len(node.else_actions.actions) == 1
    
    @pytest.mark.asyncio
    async def test_build_for_loop(self):
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
            ctx=None
        )
        
        assert isinstance(node, ForLoopNode)
        assert node.loop_config.iterator_variable == "file"
        assert node.loop_config.collection_expression == "selected_files"
        assert node.loop_config.max_iterations == 100
        assert len(node.loop_actions.actions) == 1
    
    @pytest.mark.asyncio
    async def test_build_while_loop(self):
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
            ctx=None
        )
        
        assert isinstance(node, WhileLoopNode)
        assert node.condition.expression == "counter"
        assert node.condition.operator == ComparisonOperator.LESS_THAN
        assert node.condition.operand == "10"
        assert node.max_iterations == 50
        assert len(node.loop_actions.actions) == 1
    
    @pytest.mark.asyncio
    async def test_build_switch_case(self):
        """Test switch/case structure building."""
        validator = ControlFlowValidator()
        
        cases = [
            {"value": "Safari", "actions": [{"type": "screenshot"}]},
            {"value": "Chrome", "actions": [{"type": "export_bookmarks"}]},
            {"value": "Firefox", "actions": [{"type": "clear_cache"}]}
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
            ctx=None
        )
        
        assert isinstance(node, SwitchCaseNode)
        assert node.switch_variable == "frontmost_application"
        assert len(node.cases) == 3
        assert node.has_default_case() is True
        assert node.cases[0].case_value == "Safari"
        assert node.cases[1].case_value == "Chrome"
        assert node.cases[2].case_value == "Firefox"
    
    @pytest.mark.asyncio
    async def test_missing_required_data(self):
        """Test building with missing required data."""
        validator = ControlFlowValidator()
        
        with pytest.raises(ValidationError, match="requires condition and true actions"):
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
                ctx=None
            )


class TestStructureInfo:
    """Test structure information extraction."""
    
    def test_if_then_else_info(self):
        """Test if/then/else structure info."""
        from src.core.control_flow import create_simple_if, ComparisonOperator
        
        node = create_simple_if(
            "test_condition",
            ComparisonOperator.EQUALS,
            "test_value",
            [{"type": "action1"}, {"type": "action2"}],
            [{"type": "action3"}]
        )
        
        info = _get_structure_info(node)
        
        assert info["node_type"] == "IfThenElseNode"
        assert info["has_else_branch"] is True
        assert info["condition_operator"] == "equals"
        assert info["then_action_count"] == 2
        assert info["else_action_count"] == 1
    
    def test_for_loop_info(self):
        """Test for loop structure info."""
        from src.core.control_flow import create_for_loop
        
        node = create_for_loop(
            "item",
            "collection",
            [{"type": "action1"}, {"type": "action2"}, {"type": "action3"}],
            max_iterations=100
        )
        
        info = _get_structure_info(node)
        
        assert info["node_type"] == "ForLoopNode"
        assert info["iterator_variable"] == "item"
        assert info["collection_expression"] == "collection"
        assert info["max_iterations"] == 100
        assert info["action_count"] == 3
    
    def test_while_loop_info(self):
        """Test while loop structure info."""
        from src.core.control_flow import create_while_loop, ComparisonOperator
        
        node = create_while_loop(
            "counter",
            ComparisonOperator.LESS_THAN,
            "10",
            [{"type": "increment"}],
            max_iterations=50
        )
        
        info = _get_structure_info(node)
        
        assert info["node_type"] == "WhileLoopNode"
        assert info["condition_operator"] == "less_than"
        assert info["max_iterations"] == 50
        assert info["action_count"] == 1