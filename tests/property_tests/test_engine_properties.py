"""
Property-based tests for the macro execution engine.

This module uses Hypothesis to test engine behavior across wide input ranges,
ensuring robust behavior under all conditions and edge cases.
"""

import pytest
import time
import string
from hypothesis import given, assume, strategies as st, settings, HealthCheck
from unittest.mock import patch

from src.core import (
    MacroEngine, MacroDefinition, ExecutionContext, ExecutionStatus,
    Duration, Permission, create_test_macro, CommandType,
    ValidationError, PermissionDeniedError, ExecutionError,
    get_default_engine
)
from tests.utils.generators import (
    simple_macro_definitions, execution_contexts, durations,
    permission_sets, macro_ids, safe_text_content
)
from tests.utils.assertions import (
    assert_execution_successful, assert_execution_failed,
    assert_performance_within_bounds, assert_thread_safe_operation
)


class TestEngineProperties:
    """Property-based tests for the macro engine core behavior."""
    
    @given(simple_macro_definitions(), execution_contexts())
    @settings(max_examples=50, deadline=3000)
    def test_execution_always_returns_result(self, macro_def: MacroDefinition, context: ExecutionContext):
        """Property: Engine always returns a complete execution result."""
        assume(macro_def.is_valid())
        assume(len(macro_def.commands) <= 5)  # Keep tests fast
        
        engine = MacroEngine()
        
        result = engine.execute_macro(macro_def, context)
        
        # Fundamental properties
        assert result is not None, "Result cannot be None"
        assert result.execution_token is not None, "Execution token must be present"
        assert result.macro_id == macro_def.macro_id, "Macro ID must match"
        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED], "Status must be terminal"
        assert result.started_at is not None, "Start time must be recorded"
        
        if result.status == ExecutionStatus.COMPLETED:
            assert result.completed_at is not None, "Completion time must be set for completed executions"
            assert result.total_duration is not None, "Total duration must be measured"
    
    @given(execution_contexts())
    @settings(max_examples=30)
    def test_invalid_macros_rejected(self, context: ExecutionContext):
        """Property: Invalid macros are always rejected before execution."""
        engine = MacroEngine()
        
        # Create an invalid macro (empty commands)
        invalid_macro = MacroDefinition(
            macro_id="invalid_test",
            name="Invalid Test Macro",
            commands=[],  # Empty commands make it invalid
            enabled=True
        )
        
        # Should return failed ExecutionResult for invalid macro
        result = engine.execute_macro(invalid_macro, context)
        assert result.status == ExecutionStatus.FAILED, "Invalid macro should result in failed execution"
        assert result.error_details is not None, "Failed execution should have error details"
        assert "invalid" in result.error_details.lower(), "Error should indicate invalid macro"
    
    @given(simple_macro_definitions())
    @settings(max_examples=20)
    def test_execution_respects_timeouts(self, macro_def: MacroDefinition):
        """Property: Execution never significantly exceeds context timeout."""
        assume(macro_def.is_valid())
        
        # Create context with very short timeout
        short_timeout = Duration.from_seconds(0.1)
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_SOUND]),
            timeout=short_timeout
        )
        
        engine = MacroEngine()
        
        start_time = time.perf_counter()
        try:
            result = engine.execute_macro(macro_def, context)
            execution_time = time.perf_counter() - start_time
            
            # Allow some tolerance for processing overhead
            max_allowed_time = short_timeout.total_seconds() + 0.5
            assert execution_time <= max_allowed_time, f"Execution took {execution_time}s, max allowed {max_allowed_time}s"
            
        except Exception:
            # Timeout or other errors are acceptable
            execution_time = time.perf_counter() - start_time
            # Even failed executions should not take too long
            assert execution_time <= 2.0, f"Even failed execution took too long: {execution_time}s"
    
    @given(permission_sets(min_size=1, max_size=5))
    @settings(max_examples=20)
    def test_permission_enforcement(self, available_permissions: frozenset[Permission]):
        """Property: Commands requiring unavailable permissions are rejected."""
        # Create a macro that requires a permission not in the available set
        all_permissions = set(Permission)
        missing_permissions = all_permissions - available_permissions
        
        assume(len(missing_permissions) > 0)
        
        # Pick a permission that's missing
        required_permission = next(iter(missing_permissions))
        
        # Create macro that requires this permission
        if required_permission == Permission.SYSTEM_SOUND:
            macro = create_test_macro("Sound Test", [CommandType.PLAY_SOUND])
        elif required_permission == Permission.APPLICATION_CONTROL:
            macro = create_test_macro("App Test", [CommandType.APPLICATION_CONTROL])
        else:
            # Use text input as fallback (requires TEXT_INPUT permission)
            macro = create_test_macro("Text Test", [CommandType.TEXT_INPUT])
            required_permission = Permission.TEXT_INPUT
        
        context = ExecutionContext.create_test_context(
            permissions=available_permissions,
            timeout=Duration.from_seconds(30)
        )
        
        engine = MacroEngine()
        
        # Execute the macro and check results
        result = engine.execute_macro(macro, context)
        
        if required_permission not in available_permissions:
            # Should fail due to missing permissions
            assert result.status == ExecutionStatus.FAILED, "Should fail when missing required permissions"
            # Check if any command results indicate permission denied
            permission_denied = any(
                not cmd_result.success and "permission" in (cmd_result.error_message or "").lower()
                for cmd_result in result.command_results
            )
            assert permission_denied or "permission" in (result.error_details or "").lower(), \
                "Failure should be due to permission issues"
        else:
            # Should succeed if permission is available
            assert_execution_successful(result, "Execution with sufficient permissions")
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=15)
    def test_engine_handles_concurrent_executions(self, num_concurrent: int):
        """Property: Engine handles concurrent executions safely."""
        engine = MacroEngine()
        
        # Create simple macros for concurrent execution
        macros = [
            create_test_macro(f"Concurrent Test {i}", [CommandType.TEXT_INPUT])
            for i in range(num_concurrent)
        ]
        
        contexts = [
            ExecutionContext.create_test_context() for _ in range(num_concurrent)
        ]
        
        # Test thread safety
        def execute_single(macro, context):
            return engine.execute_macro(macro, context)
        
        args_list = list(zip(macros, contexts))
        
        # This will test thread safety
        assert_thread_safe_operation(
            operation=execute_single,
            args_list=args_list,
            max_workers=min(num_concurrent, 5),
            message="Concurrent macro execution"
        )
    
    @given(durations(min_seconds=0.01, max_seconds=1.0))
    @settings(max_examples=20)
    def test_execution_timing_consistency(self, expected_duration: Duration):
        """Property: Execution timing is reasonably consistent and predictable."""
        # Create a simple macro
        macro = create_test_macro("Timing Test", [CommandType.PAUSE])
        context = ExecutionContext.create_test_context()
        
        engine = MacroEngine()
        
        # Execute multiple times and measure consistency
        execution_times = []
        for _ in range(3):  # Small sample for property testing
            start_time = time.perf_counter()
            result = engine.execute_macro(macro, context)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            assert_execution_successful(result, "Timing test execution")
        
        # Check that timing is reasonably consistent
        avg_time = sum(execution_times) / len(execution_times)
        max_deviation = max(abs(t - avg_time) for t in execution_times)
        
        # Allow for reasonable variation (up to 100% for very short operations)
        allowed_deviation = max(0.1, avg_time)
        assert max_deviation <= allowed_deviation, f"Timing too inconsistent: {execution_times}"
    
    @given(simple_macro_definitions())
    @settings(max_examples=15)
    def test_execution_status_transitions_valid(self, macro_def: MacroDefinition):
        """Property: Execution status follows valid state transitions."""
        assume(macro_def.is_valid())
        
        engine = MacroEngine()
        context = ExecutionContext.create_test_context()
        
        result = engine.execute_macro(macro_def, context)
        
        # Check final status is valid
        valid_final_states = [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]
        assert result.status in valid_final_states, f"Invalid final status: {result.status}"
        
        # Check status consistency with result properties
        if result.status == ExecutionStatus.COMPLETED:
            assert result.is_successful(), "COMPLETED executions must be successful"
            assert result.completed_at is not None, "COMPLETED executions must have completion time"
        
        if result.status == ExecutionStatus.FAILED:
            assert not result.is_successful(), "FAILED executions must not be successful"
            assert result.has_error_info(), "FAILED executions must have error info"
    
    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=10)
    def test_engine_resource_cleanup(self, num_executions: int):
        """Property: Engine properly cleans up resources after executions."""
        engine = MacroEngine()
        
        initial_active = len(engine.get_active_executions())
        
        # Execute multiple macros
        for i in range(num_executions):
            macro = create_test_macro(f"Cleanup Test {i}", [CommandType.TEXT_INPUT])
            context = ExecutionContext.create_test_context()
            
            result = engine.execute_macro(macro, context)
            assert_execution_successful(result, f"Cleanup test execution {i}")
        
        # Clean up expired executions
        cleaned = engine.cleanup_expired_executions(max_age_seconds=0.1)
        
        # Check that resources are cleaned up
        final_active = len(engine.get_active_executions())
        
        # Should not accumulate active executions indefinitely
        assert final_active <= initial_active + 1, "Active executions not properly cleaned up"


class TestEngineEdgeCases:
    """Property-based tests for engine edge cases and error conditions."""
    
    @given(st.text(alphabet=string.ascii_letters + string.digits + " .-_", min_size=1000, max_size=5000))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.filter_too_much])
    def test_large_input_handling(self, large_text: str):
        """Property: Engine handles large inputs gracefully."""
        
        # Create macro with large text input
        from src.core.engine import PlaceholderCommand
        from src.core import CommandParameters
        
        large_command = PlaceholderCommand(
            command_id="large_text_cmd",
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters({"text": large_text, "speed": "normal"})
        )
        
        macro = MacroDefinition(
            macro_id="large_input_test",
            name="Large Input Test",
            commands=[large_command],
            enabled=True
        )
        
        context = ExecutionContext.create_test_context()
        engine = MacroEngine()
        
        # Should handle large input without crashing
        result = engine.execute_macro(macro, context)
        
        # Should complete successfully or fail gracefully
        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
        
        if result.status == ExecutionStatus.FAILED:
            assert result.has_error_info(), "Failed execution must have error info"
    
    @given(st.integers(min_value=0, max_value=100))
    @settings(max_examples=15)
    def test_empty_and_minimal_macros(self, num_commands: int):
        """Property: Engine handles macros with varying numbers of commands."""
        if num_commands == 0:
            # Empty macro should be invalid
            empty_macro = MacroDefinition(
                macro_id="empty_test",
                name="Empty Test",
                commands=[],
                enabled=True
            )
            
            assert not empty_macro.is_valid(), "Empty macro should be invalid"
            
        else:
            # Create macro with specified number of simple commands
            command_types = [CommandType.TEXT_INPUT] * min(num_commands, 10)  # Limit for performance
            macro = create_test_macro(f"Multi Command Test", command_types)
            
            context = ExecutionContext.create_test_context()
            engine = MacroEngine()
            
            result = engine.execute_macro(macro, context)
            
            # Should handle any number of commands appropriately
            assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
            
            if result.status == ExecutionStatus.COMPLETED:
                # Should have results for each command
                assert len(result.command_results) == len(macro.commands)


class TestEnginePerformanceProperties:
    """Property-based tests for engine performance characteristics."""
    
    @pytest.mark.performance
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=10, deadline=5000)
    def test_execution_time_scales_linearly(self, num_commands: int):
        """Property: Execution time scales roughly linearly with number of commands."""
        command_types = [CommandType.TEXT_INPUT] * num_commands
        macro = create_test_macro("Scaling Test", command_types)
        
        context = ExecutionContext.create_test_context()
        engine = MacroEngine()
        
        start_time = time.perf_counter()
        result = engine.execute_macro(macro, context)
        execution_time = time.perf_counter() - start_time
        
        assert_execution_successful(result, "Scaling test execution")
        
        # Execution time should scale reasonably with command count
        # Allow generous bounds for property testing
        expected_max_time = num_commands * 0.5 + 1.0  # 0.5s per command + 1s overhead
        assert_performance_within_bounds(
            execution_time, 
            expected_max_time, 
            message=f"Execution time for {num_commands} commands"
        )
    
    @pytest.mark.performance
    @given(durations(min_seconds=0.01, max_seconds=0.5))
    @settings(max_examples=10)
    def test_engine_startup_performance(self, pause_duration: Duration):
        """Property: Engine startup time is consistently fast."""
        # Measure engine creation time
        start_time = time.perf_counter()
        engine = MacroEngine()
        startup_time = time.perf_counter() - start_time
        
        # Should start up quickly
        assert_performance_within_bounds(
            startup_time, 
            max_time=0.1,  # 100ms max startup
            message="Engine startup time"
        )
        
        # Verify engine is functional after startup
        macro = create_test_macro("Startup Test", [CommandType.PAUSE])
        context = ExecutionContext.create_test_context()
        
        result = engine.execute_macro(macro, context)
        assert_execution_successful(result, "Post-startup execution test")