"""
End-to-end integration tests for the complete macro system.

This module tests the entire macro execution pipeline from parsing
through execution to result validation with realistic scenarios.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List

from src.core import (
    MacroEngine, MacroDefinition, ExecutionContext, ExecutionStatus,
    CommandType, Permission, Duration, create_test_macro,
    parse_macro_from_json, get_default_engine
)
from src.core.types import VariableName
from tests.utils.mocks import (
    MockKeyboardMaestroClient, create_reliable_km_client,
    create_failing_km_client, MockFileSystem
)
from tests.utils.assertions import (
    assert_execution_successful, assert_execution_failed,
    assert_macro_valid, assert_context_valid
)


class TestCompleteWorkflows:
    """End-to-end tests for complete macro workflows."""
    
    def test_simple_text_macro_workflow(self):
        """Test complete workflow for a simple text input macro."""
        # 1. Create macro definition
        macro = create_test_macro("Simple Text Test", [CommandType.TEXT_INPUT])
        assert_macro_valid(macro, "Simple text macro should be valid")
        
        # 2. Create execution context
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT]),
            timeout=Duration.from_seconds(30)
        )
        assert_context_valid(context, "Execution context should be valid")
        
        # 3. Execute macro
        engine = MacroEngine()
        result = engine.execute_macro(macro, context)
        
        # 4. Validate results
        assert_execution_successful(result, "Simple text macro execution")
        assert len(result.command_results) == 1, "Should have one command result"
        assert result.command_results[0].success, "Text command should succeed"
    
    def test_complex_multi_command_workflow(self):
        """Test workflow with multiple different command types."""
        # Create complex macro with multiple command types
        command_types = [
            CommandType.TEXT_INPUT,
            CommandType.PAUSE,
            CommandType.PLAY_SOUND,
            CommandType.VARIABLE_SET,
            CommandType.VARIABLE_GET
        ]
        
        macro = create_test_macro("Complex Workflow", command_types)
        assert_macro_valid(macro, "Complex macro should be valid")
        
        # Create context with all necessary permissions
        context = ExecutionContext.create_test_context(
            permissions=frozenset([
                Permission.TEXT_INPUT,
                Permission.SYSTEM_SOUND,
                Permission.APPLICATION_CONTROL
            ]),
            timeout=Duration.from_seconds(60)
        )
        
        engine = MacroEngine()
        result = engine.execute_macro(macro, context)
        
        assert_execution_successful(result, "Complex macro execution")
        assert len(result.command_results) == len(command_types), "Should have result for each command"
        
        # Validate individual command results
        for i, cmd_result in enumerate(result.command_results):
            assert cmd_result.success, f"Command {i} should succeed"
            assert cmd_result.execution_time is not None, f"Command {i} should have execution time"
    
    def test_macro_parsing_to_execution_workflow(self):
        """Test complete workflow from JSON parsing to execution."""
        # JSON macro definition
        macro_json = """
        {
            "name": "Parsed Macro Test",
            "id": "parsed_test",
            "enabled": true,
            "description": "Test macro from JSON parsing",
            "commands": [
                {
                    "type": "text_input",
                    "parameters": {
                        "text": "Hello from parsed macro",
                        "speed": "normal"
                    }
                },
                {
                    "type": "pause",
                    "parameters": {
                        "duration": 0.5
                    }
                }
            ]
        }
        """
        
        # 1. Parse macro from JSON
        parse_result = parse_macro_from_json(macro_json)
        assert parse_result.success, f"Parsing should succeed: {parse_result.errors}"
        
        macro = parse_result.macro_definition
        assert_macro_valid(macro, "Parsed macro should be valid")
        
        # 2. Execute parsed macro
        context = ExecutionContext.create_test_context()
        engine = MacroEngine()
        
        result = engine.execute_macro(macro, context)
        
        # 3. Validate execution
        assert_execution_successful(result, "Parsed macro execution")
        assert result.macro_id == "parsed_test", "Should preserve parsed macro ID"
    
    def test_error_handling_workflow(self):
        """Test complete error handling workflow."""
        # Create macro that will fail due to missing permissions
        macro = create_test_macro("Permission Test", [CommandType.PLAY_SOUND])
        
        # Create context without required permission
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT]),  # Missing SYSTEM_SOUND
            timeout=Duration.from_seconds(30)
        )
        
        engine = MacroEngine()
        
        # Should fail due to permission error
        result = engine.execute_macro(macro, context)
        
        # Verify the result indicates failure due to permission error
        assert_execution_failed(result, "Permission denied")
        assert "SYSTEM_SOUND" in result.error_details or "system_sound" in result.error_details
    
    def test_concurrent_macro_execution_workflow(self):
        """Test concurrent execution of multiple macros."""
        import threading
        import concurrent.futures
        
        engine = MacroEngine()
        num_concurrent = 5
        
        # Create different macros for concurrent execution
        macros = [
            create_test_macro(f"Concurrent {i}", [CommandType.TEXT_INPUT])
            for i in range(num_concurrent)
        ]
        
        contexts = [
            ExecutionContext.create_test_context() for _ in range(num_concurrent)
        ]
        
        results = []
        errors = []
        
        def execute_macro(macro, context):
            try:
                result = engine.execute_macro(macro, context)
                return result
            except Exception as e:
                return e
        
        # Execute concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(execute_macro, macro, context)
                for macro, context in zip(macros, contexts)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if isinstance(result, Exception):
                    errors.append(result)
                else:
                    results.append(result)
        
        # Validate results
        assert len(errors) == 0, f"No errors should occur in concurrent execution: {errors}"
        assert len(results) == num_concurrent, "All executions should complete"
        
        for i, result in enumerate(results):
            assert_execution_successful(result, f"Concurrent execution {i}")


class TestRealisticScenarios:
    """Tests with realistic usage scenarios."""
    
    def test_text_automation_scenario(self):
        """Test realistic text automation scenario."""
        # Simulate a text automation workflow
        steps = [
            CommandType.TEXT_INPUT,  # Type greeting
            CommandType.PAUSE,       # Wait
            CommandType.TEXT_INPUT,  # Type main content
            CommandType.PAUSE,       # Wait
            CommandType.TEXT_INPUT   # Type closing
        ]
        
        macro = create_test_macro("Text Automation Scenario", steps)
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT]),
            timeout=Duration.from_seconds(45)
        )
        
        engine = MacroEngine()
        start_time = time.perf_counter()
        
        result = engine.execute_macro(macro, context)
        execution_time = time.perf_counter() - start_time
        
        assert_execution_successful(result, "Text automation scenario")
        assert execution_time < 5.0, "Should complete within reasonable time"
        assert len(result.command_results) == len(steps), "Should execute all steps"
    
    def test_system_integration_scenario(self):
        """Test scenario involving system interactions."""
        # Simulate system automation workflow
        steps = [
            CommandType.VARIABLE_SET,     # Set configuration
            CommandType.APPLICATION_CONTROL, # Control app
            CommandType.PAUSE,            # Wait for app
            CommandType.PLAY_SOUND,       # Audio feedback
            CommandType.VARIABLE_GET      # Check result
        ]
        
        macro = create_test_macro("System Integration Scenario", steps)
        context = ExecutionContext.create_test_context(
            permissions=frozenset([
                Permission.TEXT_INPUT,
                Permission.SYSTEM_SOUND,
                Permission.APPLICATION_CONTROL
            ]),
            timeout=Duration.from_seconds(60)
        )
        
        engine = MacroEngine()
        result = engine.execute_macro(macro, context)
        
        assert_execution_successful(result, "System integration scenario")
        
        # Validate that system interactions completed
        for cmd_result in result.command_results:
            assert cmd_result.success, "All system interactions should succeed"
    
    def test_error_recovery_scenario(self):
        """Test error recovery in realistic scenarios."""
        # Create macro that might encounter errors
        macro = create_test_macro("Error Recovery Test", [
            CommandType.TEXT_INPUT,
            CommandType.PLAY_SOUND,  # This might fail
            CommandType.TEXT_INPUT   # Should continue after error
        ])
        
        # Test with insufficient permissions (will cause error)
        limited_context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT]),  # Missing SYSTEM_SOUND
            timeout=Duration.from_seconds(30)
        )
        
        engine = MacroEngine()
        
        # Should handle permission error gracefully
        try:
            result = engine.execute_macro(macro, limited_context)
            # If it doesn't raise exception, check that it failed gracefully
            if result.status == ExecutionStatus.FAILED:
                assert result.has_error_info(), "Failed execution should have error info"
        except Exception as e:
            # Exception is also acceptable for permission errors
            assert "permission" in str(e).lower(), "Should be permission-related error"
    
    def test_performance_sensitive_scenario(self):
        """Test scenario with performance requirements."""
        # Create macro with timing constraints
        macro = create_test_macro("Performance Sensitive", [
            CommandType.TEXT_INPUT,
            CommandType.PAUSE,
            CommandType.TEXT_INPUT
        ])
        
        context = ExecutionContext.create_test_context(
            timeout=Duration.from_seconds(10)  # Tight timeout
        )
        
        engine = MacroEngine()
        
        # Measure execution time
        start_time = time.perf_counter()
        result = engine.execute_macro(macro, context)
        execution_time = time.perf_counter() - start_time
        
        assert_execution_successful(result, "Performance sensitive scenario")
        assert execution_time < 2.0, "Should complete quickly"
        assert result.total_duration.total_seconds() < 2.0, "Recorded duration should be reasonable"


class TestKMIntegrationScenarios:
    """Tests for Keyboard Maestro integration scenarios."""
    
    def test_reliable_km_integration(self):
        """Test integration with reliable KM client."""
        km_client = create_reliable_km_client()
        
        # Test trigger registration
        trigger_config = {
            "trigger_type": "hotkey",
            "key": "F1",
            "modifiers": ["Command"]
        }
        
        response = km_client.register_trigger(trigger_config)
        assert response.status == "success", "Trigger registration should succeed"
        
        # Test macro execution
        macro_response = km_client.execute_macro("test_macro", {"param": "value"})
        assert macro_response.status == "completed", "Macro execution should complete"
        
        # Check statistics
        stats = km_client.get_statistics()
        assert stats["success_rate"] > 0.9, "Should have high success rate"
        assert stats["call_count"] >= 2, "Should record all calls"
    
    def test_unreliable_km_integration(self):
        """Test integration with unreliable KM client."""
        km_client = create_failing_km_client()
        
        # Attempt multiple operations to test error handling
        successes = 0
        failures = 0
        
        for i in range(10):
            try:
                response = km_client.execute_macro(f"test_macro_{i}")
                if response.status == "completed":
                    successes += 1
                else:
                    failures += 1
            except Exception:
                failures += 1
        
        # Should have some failures due to unreliable client
        assert failures > 0, "Should encounter some failures with unreliable client"
        
        # Should handle failures gracefully
        stats = km_client.get_statistics()
        assert stats["error_count"] > 0, "Should record errors"
    
    @pytest.mark.asyncio
    async def test_async_km_integration(self):
        """Test asynchronous KM integration."""
        km_client = create_reliable_km_client()
        
        # Test multiple async operations
        tasks = []
        for i in range(5):
            tasks.append(km_client.register_trigger_async({"trigger": f"test_{i}"}))
            tasks.append(km_client.execute_macro_async(f"async_macro_{i}"))
        
        # Wait for all operations to complete
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        for result in results:
            assert result.status in ["success", "completed"], "Async operation should succeed"


class TestSystemIntegration:
    """Tests for system-level integration."""
    
    def test_file_system_integration(self):
        """Test integration with file system operations."""
        mock_fs = MockFileSystem()
        
        # Set up test files
        mock_fs.write_file("/test/input.txt", "Test input content")
        mock_fs.write_file("/test/config.json", '{"setting": "value"}')
        
        # Test file operations
        assert mock_fs.exists("/test/input.txt"), "Test file should exist"
        
        content = mock_fs.read_file("/test/input.txt")
        assert content == "Test input content", "Should read correct content"
        
        # Test file listing
        files = mock_fs.list_directory("/test")
        assert len(files) >= 2, "Should list test files"
        
        # Check access logging
        assert mock_fs.read_count > 0, "Should track read operations"
        assert mock_fs.write_count > 0, "Should track write operations"
    
    def test_resource_management_integration(self):
        """Test integration of resource management across components."""
        from src.core.context import get_context_manager, get_variable_manager
        
        context_manager = get_context_manager()
        variable_manager = get_variable_manager()
        
        # Create test context
        context = ExecutionContext.create_test_context()
        token = context_manager.register_context(context)
        
        # Set up variables
        global_var = VariableName("test_global")
        local_var = VariableName("test_local")
        variable_manager.set_global_variable(global_var, "global_value")
        variable_manager.set_context_variable(token, local_var, "local_value")
        
        # Test resource access
        assert context_manager.get_context(token) is not None, "Should retrieve context"
        assert variable_manager.get_global_variable(global_var) == "global_value"
        assert variable_manager.get_context_variable(token, local_var) == "local_value"
        
        # Test cleanup
        context_manager.cleanup_context(token)
        variable_manager.cleanup_context_variables(token)
        
        # Should be cleaned up
        assert context_manager.get_context(token) is None, "Context should be cleaned up"
        assert variable_manager.get_context_variable(token, "test_local") is None
    
    def test_metrics_integration(self):
        """Test integration of metrics and monitoring."""
        from src.core.engine import get_engine_metrics
        
        metrics = get_engine_metrics()
        metrics.reset_metrics()
        
        # Execute some operations to generate metrics
        engine = MacroEngine()
        
        for i in range(5):
            macro = create_test_macro(f"Metrics Test {i}", [CommandType.TEXT_INPUT])
            context = ExecutionContext.create_test_context()
            
            start_time = time.perf_counter()
            result = engine.execute_macro(macro, context)
            execution_time = time.perf_counter() - start_time
            
            # Record metrics
            metrics.record_execution(
                Duration.from_seconds(execution_time),
                result.status == ExecutionStatus.COMPLETED
            )
        
        # Validate metrics
        final_metrics = metrics.get_metrics()
        assert final_metrics["execution_count"] == 5, "Should record all executions"
        assert final_metrics["average_execution_time"] > 0, "Should calculate average time"
        assert 0 <= final_metrics["success_rate"] <= 1, "Success rate should be valid percentage"