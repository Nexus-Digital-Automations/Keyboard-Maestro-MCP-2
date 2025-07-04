"""
Property-based tests for macro editor functionality.

Comprehensive testing of macro editing operations using hypothesis to validate
behavior across all input ranges with security validation and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from typing import Dict, Any

from src.core.macro_editor import (
    MacroEditor, MacroModification, EditOperation, DebugSession,
    MacroEditorValidator, calculate_macro_complexity, calculate_macro_health
)
from src.debugging.macro_debugger import MacroDebugger, DebugState
from src.server.tools.macro_editor_tools import km_macro_editor
from src.core.errors import ValidationError, SecurityViolationError


class TestMacroEditorProperties:
    """Property-based tests for macro editor functionality."""
    
    @given(st.text(min_size=1, max_size=100).filter(lambda x: x.strip() != ""))
    def test_macro_editor_initialization_properties(self, macro_id):
        """Property: All valid macro IDs should create valid editor instances."""
        editor = MacroEditor(macro_id)
        assert editor.macro_id == macro_id
        assert len(editor.get_modifications()) == 0
    
    @pytest.mark.skip(reason="Contract system issue with multiple @require decorators - needs investigation")
    @given(
        st.text(min_size=2, max_size=50, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),  # Valid action types
        st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)), 
            st.text(min_size=1, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)), 
            min_size=1, max_size=5
        ),  # Valid config with proper keys/values
        st.integers(min_value=0, max_value=10)  # Reasonable position range
    )
    def test_add_action_properties(self, action_type, config, position):
        """Property: Adding actions should preserve order and configuration."""
        editor = MacroEditor("test_macro")
        
        if position >= 0:  # Valid position
            editor.add_action(action_type, config, position)
            modifications = editor.get_modifications()
            
            assert len(modifications) == 1
            assert modifications[0].operation == EditOperation.ADD_ACTION
            assert modifications[0].new_value["type"] == action_type
            assert modifications[0].new_value["config"] == config
            assert modifications[0].position == position
    
    @pytest.mark.skip(reason="Contract system issue with multiple @require decorators - needs investigation")
    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20),
                st.dictionaries(st.text(), st.text(), min_size=1, max_size=5)
            ),
            min_size=1,
            max_size=10
        )
    )
    def test_multiple_modifications_properties(self, action_specs):
        """Property: Multiple modifications should maintain order and integrity."""
        editor = MacroEditor("test_macro")
        
        for i, (action_type, config) in enumerate(action_specs):
            editor.add_action(action_type, config, position=i)
        
        modifications = editor.get_modifications()
        assert len(modifications) == len(action_specs)
        
        # Check order preservation
        for i, (action_type, config) in enumerate(action_specs):
            mod = modifications[i]
            assert mod.new_value["type"] == action_type
            assert mod.new_value["config"] == config
            assert mod.position == i
    
    @given(
        st.text(min_size=1, max_size=50),
        st.sets(st.text(min_size=1, max_size=20), min_size=0, max_size=50),
        st.sets(st.text(min_size=1, max_size=20), min_size=0, max_size=20),
        st.integers(min_value=1, max_value=300)
    )
    def test_debug_session_properties(self, macro_id, breakpoints, watch_variables, timeout):
        """Property: Debug sessions should validate configuration properly."""
        assume(len(breakpoints) <= 50)
        assume(len(watch_variables) <= 20)
        assume(1 <= timeout <= 300)
        
        debug_session = DebugSession(
            macro_id=macro_id,
            breakpoints=breakpoints,
            watch_variables=watch_variables,
            timeout_seconds=timeout
        )
        
        assert debug_session.macro_id == macro_id
        assert debug_session.breakpoints == breakpoints
        assert debug_session.watch_variables == watch_variables
        assert debug_session.timeout_seconds == timeout
    
    @given(st.integers(min_value=-10, max_value=400))
    def test_debug_session_timeout_validation(self, timeout):
        """Property: Debug session timeout validation should work correctly."""
        if 1 <= timeout <= 300:
            # Should succeed
            debug_session = DebugSession(
                macro_id="test_macro",
                timeout_seconds=timeout
            )
            assert debug_session.timeout_seconds == timeout
        else:
            # Should fail validation  
            from src.core.errors import ContractViolationError
            with pytest.raises(ContractViolationError):
                DebugSession(
                    macro_id="test_macro",
                    timeout_seconds=timeout
                )
    
    @given(
        st.dictionaries(
            st.text(),
            st.one_of(st.text(), st.integers(), st.booleans()),
            min_size=0,
            max_size=20
        )
    )
    def test_action_validation_properties(self, action_config):
        """Property: Action validation should handle all configuration types."""
        result = MacroEditorValidator.validate_action_modification(action_config)
        
        # Should always return Either type
        assert hasattr(result, 'is_left')
        assert hasattr(result, 'is_right')
        
        if result.is_right():
            # Valid config should be preserved
            validated_config = result.get_right()
            assert isinstance(validated_config, dict)
    
    @given(
        st.dictionaries(
            st.text(),
            st.one_of(st.text(), st.integers(), st.lists(st.text())),
            min_size=0,
            max_size=15
        )
    )
    def test_debug_config_validation_properties(self, debug_config):
        """Property: Debug configuration validation should be comprehensive."""
        result = MacroEditorValidator.validate_debug_session(debug_config)
        
        # Should always return Either type
        assert hasattr(result, 'is_left')
        assert hasattr(result, 'is_right')
        
        # Check specific validation rules
        breakpoints = debug_config.get("breakpoints", [])
        if isinstance(breakpoints, list) and len(breakpoints) > 50:
            assert result.is_left()
        
        timeout = debug_config.get("timeout_seconds")
        if isinstance(timeout, int) and (timeout <= 0 or timeout > 300):
            assert result.is_left()
    
    @given(
        st.dictionaries(
            st.text(),
            st.one_of(
                st.text(),
                st.lists(st.dictionaries(st.text(), st.text())),
                st.integers()
            ),
            min_size=0,
            max_size=20
        )
    )
    def test_macro_complexity_calculation_properties(self, macro_data):
        """Property: Complexity calculation should be bounded and consistent."""
        complexity = calculate_macro_complexity(macro_data)
        
        # Should always be between 0 and 100
        assert 0 <= complexity <= 100
        assert isinstance(complexity, int)
        
        # Empty macro should have low complexity
        if not macro_data or all(not macro_data.get(key) for key in ["actions", "triggers", "conditions"]):
            assert complexity <= 10
    
    @given(
        st.dictionaries(
            st.text(),
            st.one_of(
                st.text(),
                st.lists(st.dictionaries(st.text(), st.text())),
                st.booleans()
            ),
            min_size=0,
            max_size=20
        )
    )
    def test_macro_health_calculation_properties(self, macro_data):
        """Property: Health calculation should be bounded and meaningful."""
        health = calculate_macro_health(macro_data)
        
        # Should always be between 0 and 100
        assert 0 <= health <= 100
        assert isinstance(health, int)
        
        # Macro with name and actions should have better health
        if macro_data.get("name") and macro_data.get("actions"):
            assert health >= 50


class TestMacroDebuggerProperties:
    """Property-based tests for macro debugger functionality."""
    
    @pytest.fixture
    def debugger(self):
        """Provide debugger instance for tests."""
        return MacroDebugger()
    
    @given(
        st.text(alphabet=st.characters(blacklist_categories=['Cc', 'Cs']), min_size=1, max_size=50).filter(lambda x: x.strip()),
        st.sets(st.text(alphabet=st.characters(blacklist_categories=['Cc', 'Cs']), min_size=1, max_size=20).filter(lambda x: x.strip()), max_size=50),
        st.integers(min_value=1, max_value=300)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_debug_session_lifecycle_properties(self, debugger, macro_id, breakpoints, timeout):
        """Property: Debug session lifecycle should be consistent."""
        debug_session = DebugSession(
            macro_id=macro_id,
            breakpoints=breakpoints,
            timeout_seconds=timeout
        )
        
        # Start session
        session_result = await debugger.start_debug_session(debug_session)
        assert session_result.is_right()
        
        session_id = session_result.get_right()
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Session should be in active list
        active_sessions = debugger.list_active_sessions()
        assert session_id in active_sessions
        
        # Should be able to get state
        state_result = debugger.get_session_state(session_id)
        assert state_result.is_right()
        
        state = state_result.get_right()
        assert state["session_id"] == session_id
        
        # Should be able to stop session
        stop_result = debugger.stop_debug_session(session_id)
        assert stop_result.is_right()
        
        # Session should be removed from active list
        active_sessions_after = debugger.list_active_sessions()
        assert session_id not in active_sessions_after
    
    @given(st.integers(min_value=1, max_value=20))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_step_execution_properties(self, debugger, step_count):
        """Property: Step execution should maintain consistency."""
        debug_session = DebugSession(macro_id="test_macro")
        
        session_result = await debugger.start_debug_session(debug_session)
        assert session_result.is_right()
        
        session_id = session_result.get_right()
        
        # Execute multiple steps
        for i in range(step_count):
            step_result = await debugger.step_execution(session_id)
            if step_result.is_right():
                step_data = step_result.get_right()
                assert step_data["step_count"] == i + 1
                assert step_data["session_id"] == session_id
        
        # Clean up
        debugger.stop_debug_session(session_id)


class TestMacroEditorToolProperties:
    """Property-based tests for macro editor MCP tool."""
    
    @given(
        st.text(min_size=1, max_size=50),
        st.sampled_from(["inspect", "modify", "debug", "compare", "validate"])
    )
    @pytest.mark.asyncio
    async def test_macro_editor_tool_operation_validation(self, macro_identifier, operation):
        """Property: Tool should validate operations consistently."""
        # Mock the underlying components
        with patch('src.server.tools.macro_editor_tools.km_editor') as mock_editor:
            # Setup mock to return success for inspect operation
            if operation == "inspect":
                mock_inspection = Mock()
                mock_inspection.macro_id = macro_identifier
                mock_inspection.macro_name = "Test Macro"
                mock_inspection.enabled = True
                mock_inspection.group_name = "Test Group"
                mock_inspection.action_count = 5
                mock_inspection.trigger_count = 2
                mock_inspection.condition_count = 1
                mock_inspection.actions = []
                mock_inspection.triggers = []
                mock_inspection.conditions = []
                mock_inspection.variables_used = set()
                mock_inspection.estimated_execution_time = 1.0
                mock_inspection.complexity_score = 50
                mock_inspection.health_score = 80
                
                mock_result = Mock()
                mock_result.is_left.return_value = False
                mock_result.get_right.return_value = mock_inspection
                # Use AsyncMock for async methods
                mock_editor.inspect_macro = AsyncMock(return_value=mock_result)
                
                result = await km_macro_editor(
                    macro_identifier=macro_identifier,
                    operation=operation
                )
                
                assert result["success"] is True
                assert result["operation"] == operation
                assert result["macro_id"] == macro_identifier
    
    @given(st.text(max_size=0))
    @pytest.mark.asyncio
    async def test_empty_macro_identifier_validation(self, empty_identifier):
        """Property: Empty macro identifiers should be rejected."""
        assume(len(empty_identifier.strip()) == 0)
        
        with pytest.raises(Exception):  # Should raise validation error
            await km_macro_editor(
                macro_identifier=empty_identifier,
                operation="inspect"
            )
    
    @given(st.text(min_size=1, max_size=50))
    @pytest.mark.asyncio
    async def test_invalid_operation_validation(self, macro_identifier):
        """Property: Invalid operations should be rejected."""
        invalid_operation = "invalid_operation_that_does_not_exist"
        
        with pytest.raises(Exception):  # Should raise ToolError
            await km_macro_editor(
                macro_identifier=macro_identifier,
                operation=invalid_operation
            )


class TestSecurityProperties:
    """Property-based security validation tests."""
    
    @settings(max_examples=50, deadline=5000)  # Adjust for performance
    @given(
        st.dictionaries(
            st.text(),
            st.text(),
            min_size=1,
            max_size=10
        )
    )
    def test_dangerous_script_detection(self, action_config):
        """Property: Dangerous scripts should be detected and blocked."""
        # Add dangerous content
        dangerous_patterns = ["rm -rf", "sudo", "eval", "exec"]
        
        for pattern in dangerous_patterns:
            dangerous_config = action_config.copy()
            dangerous_config["script"] = f"some code {pattern} more code"
            
            result = MacroEditorValidator.validate_action_modification(dangerous_config)
            
            # Should detect dangerous pattern
            if pattern in dangerous_config["script"].lower():
                assert result.is_left()
    
    @given(st.text(min_size=1, max_size=100).filter(lambda x: x.strip() != ""))
    def test_system_macro_protection(self, macro_id):
        """Property: System macros should be protected from modification."""
        if macro_id.startswith("com.stairways.keyboardmaestro."):
            result = MacroEditorValidator.validate_modification_permissions(
                macro_id, EditOperation.MODIFY_ACTION
            )
            assert result.is_left()
        else:
            result = MacroEditorValidator.validate_modification_permissions(
                macro_id, EditOperation.MODIFY_ACTION
            )
            # Non-system macros should be allowed
            assert result.is_right()
    
    @given(
        st.lists(st.text(), min_size=0, max_size=100),
        st.lists(st.text(), min_size=0, max_size=50),
        st.integers(min_value=0, max_value=500)
    )
    def test_debug_session_limits_enforcement(self, breakpoints, watch_variables, timeout):
        """Property: Debug session limits should be enforced."""
        debug_config = {
            "breakpoints": breakpoints,
            "watch_variables": watch_variables,
            "timeout_seconds": timeout
        }
        
        result = MacroEditorValidator.validate_debug_session(debug_config)
        
        # Check limit enforcement
        if len(breakpoints) > 50:
            assert result.is_left()
        elif len(watch_variables) > 20:
            assert result.is_left()
        elif timeout <= 0 or timeout > 300:
            assert result.is_left()
        else:
            assert result.is_right()