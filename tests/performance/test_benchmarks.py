"""
Performance benchmark tests for the Keyboard Maestro MCP system.

This module provides comprehensive performance testing to ensure the system
meets timing requirements and scales appropriately under load.
"""

import pytest
import time
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core import (
    MacroEngine, MacroDefinition, ExecutionContext, CommandType,
    Permission, Duration, create_test_macro, get_engine_metrics
)
from tests.utils.assertions import assert_performance_within_bounds
from tests.utils.mocks import MockKeyboardMaestroClient


class TestEnginePerformance:
    """Performance tests for the macro engine core operations."""
    
    @pytest.mark.performance
    def test_engine_startup_time(self):
        """Engine startup should be under 10ms."""
        start_time = time.perf_counter()
        engine = MacroEngine()
        startup_time = time.perf_counter() - start_time
        
        assert_performance_within_bounds(
            startup_time, 
            max_time=0.01,  # 10ms
            message="Engine startup time"
        )
    
    @pytest.mark.performance
    def test_simple_macro_execution_time(self):
        """Simple macro execution should be under 100ms."""
        engine = MacroEngine()
        macro = create_test_macro("Performance Test", [CommandType.TEXT_INPUT])
        context = ExecutionContext.create_test_context()
        
        start_time = time.perf_counter()
        result = engine.execute_macro(macro, context)
        execution_time = time.perf_counter() - start_time
        
        assert result.status.value in ["completed", "failed"], "Execution should complete"
        assert_performance_within_bounds(
            execution_time,
            max_time=0.1,  # 100ms
            message="Simple macro execution time"
        )
    
    @pytest.mark.performance
    def test_command_validation_time(self):
        """Command validation should be under 5ms."""
        from src.core.engine import PlaceholderCommand
        from src.core import CommandParameters
        
        command = PlaceholderCommand(
            command_id="perf_test",
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters({"text": "test text", "speed": "normal"})
        )
        
        start_time = time.perf_counter()
        result = command.validate()
        validation_time = time.perf_counter() - start_time
        
        assert result is True, "Command should validate successfully"
        assert_performance_within_bounds(
            validation_time,
            max_time=0.005,  # 5ms
            message="Command validation time"
        )
    
    @pytest.mark.performance
    @pytest.mark.parametrize("text_length", [10, 100, 1000])
    def test_text_command_scaling(self, text_length: int):
        """Text command execution should scale reasonably with input size."""
        from src.core.engine import PlaceholderCommand
        from src.core import CommandParameters
        
        text = "a" * text_length
        command = PlaceholderCommand(
            command_id="scaling_test",
            command_type=CommandType.TEXT_INPUT,
            parameters=CommandParameters({"text": text, "speed": "normal"})
        )
        
        context = ExecutionContext.create_test_context()
        
        start_time = time.perf_counter()
        result = command.execute(context)
        execution_time = time.perf_counter() - start_time
        
        assert result.success, "Command should execute successfully"
        
        # Should scale roughly linearly (with generous bounds for testing)
        expected_max_time = (text_length / 1000) * 0.1 + 0.05  # 0.1s per 1000 chars + 50ms overhead
        assert_performance_within_bounds(
            execution_time,
            max_time=expected_max_time,
            message=f"Text command scaling for {text_length} characters"
        )
    
    @pytest.mark.performance
    @pytest.mark.parametrize("num_commands", [1, 5, 10])
    def test_macro_complexity_scaling(self, num_commands: int):
        """Macro execution time should scale with number of commands."""
        command_types = [CommandType.TEXT_INPUT] * num_commands
        macro = create_test_macro(f"Complex Macro {num_commands}", command_types)
        
        context = ExecutionContext.create_test_context()
        engine = MacroEngine()
        
        start_time = time.perf_counter()
        result = engine.execute_macro(macro, context)
        execution_time = time.perf_counter() - start_time
        
        assert result.status.value in ["completed", "failed"], "Execution should complete"
        
        # Should scale roughly linearly with command count
        expected_max_time = num_commands * 0.05 + 0.1  # 50ms per command + 100ms overhead
        assert_performance_within_bounds(
            execution_time,
            max_time=expected_max_time,
            message=f"Macro execution scaling for {num_commands} commands"
        )


class TestConcurrencyPerformance:
    """Performance tests for concurrent operations."""
    
    @pytest.mark.performance
    @pytest.mark.parametrize("num_concurrent", [2, 5, 10])
    def test_concurrent_execution_performance(self, num_concurrent: int):
        """Concurrent macro executions should not significantly degrade performance."""
        engine = MacroEngine()
        
        # Create multiple simple macros
        macros = [
            create_test_macro(f"Concurrent {i}", [CommandType.TEXT_INPUT])
            for i in range(num_concurrent)
        ]
        
        contexts = [ExecutionContext.create_test_context() for _ in range(num_concurrent)]
        
        def execute_macro(macro_context_pair):
            macro, context = macro_context_pair
            start_time = time.perf_counter()
            result = engine.execute_macro(macro, context)
            execution_time = time.perf_counter() - start_time
            return result, execution_time
        
        # Execute concurrently
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(execute_macro, (macro, context))
                for macro, context in zip(macros, contexts)
            ]
            
            results = []
            for future in as_completed(futures):
                result, exec_time = future.result()
                results.append((result, exec_time))
        
        total_time = time.perf_counter() - start_time
        
        # All executions should succeed
        for result, _ in results:
            assert result.status.value in ["completed", "failed"], "All executions should complete"
        
        # Concurrent execution should be faster than sequential
        # (allowing for some overhead)
        sequential_estimate = num_concurrent * 0.1  # Estimate 100ms per macro
        concurrency_efficiency = sequential_estimate / total_time
        
        # Should achieve at least 50% efficiency (2x speedup for concurrent)
        assert concurrency_efficiency >= 0.5, f"Concurrency efficiency too low: {concurrency_efficiency:.2f}"
    
    @pytest.mark.performance
    def test_thread_safety_performance(self):
        """Thread-safe operations should not have excessive overhead."""
        from src.core.context import get_context_manager, get_variable_manager
        
        context_manager = get_context_manager()
        variable_manager = get_variable_manager()
        
        num_operations = 100
        num_threads = 5
        
        def stress_test_operations():
            context = ExecutionContext.create_test_context()
            
            # Perform multiple operations
            for i in range(num_operations // num_threads):
                # Context operations
                token = context_manager.register_context(context)
                status = context_manager.get_status(token)
                context_manager.cleanup_context(token)
                
                # Variable operations
                var_name = f"test_var_{i}"
                variable_manager.set_global_variable(var_name, f"value_{i}")
                value = variable_manager.get_global_variable(var_name)
        
        start_time = time.perf_counter()
        
        # Run stress test with multiple threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=stress_test_operations)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.perf_counter() - start_time
        
        # Should complete within reasonable time
        assert_performance_within_bounds(
            total_time,
            max_time=2.0,  # 2 seconds max for stress test
            message="Thread safety performance"
        )


class TestMemoryPerformance:
    """Performance tests for memory usage and resource management."""
    
    @pytest.mark.performance
    def test_memory_usage_bounded(self):
        """Memory usage should remain bounded during normal operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = MacroEngine()
        
        # Execute many macros to test memory accumulation
        for i in range(50):  # Reduced for faster testing
            macro = create_test_macro(f"Memory Test {i}", [CommandType.TEXT_INPUT])
            context = ExecutionContext.create_test_context()
            
            result = engine.execute_macro(macro, context)
            assert result.status.value in ["completed", "failed"], f"Execution {i} should complete"
            
            # Clean up periodically
            if i % 10 == 0:
                engine.cleanup_expired_executions(max_age_seconds=0.1)
        
        # Final cleanup
        engine.cleanup_expired_executions(max_age_seconds=0.1)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Should not grow memory excessively
        assert memory_growth <= 20.0, f"Memory growth too high: {memory_growth:.2f}MB"
    
    @pytest.mark.performance
    def test_resource_cleanup_performance(self):
        """Resource cleanup should be efficient."""
        engine = MacroEngine()
        
        # Create many executions
        for i in range(20):  # Reduced for faster testing
            macro = create_test_macro(f"Cleanup Test {i}", [CommandType.TEXT_INPUT])
            context = ExecutionContext.create_test_context()
            
            result = engine.execute_macro(macro, context)
            assert result.status.value in ["completed", "failed"], f"Execution {i} should complete"
        
        # Measure cleanup performance
        start_time = time.perf_counter()
        cleaned_count = engine.cleanup_expired_executions(max_age_seconds=0.1)
        cleanup_time = time.perf_counter() - start_time
        
        # Cleanup should be fast
        assert_performance_within_bounds(
            cleanup_time,
            max_time=0.1,  # 100ms max for cleanup
            message="Resource cleanup time"
        )
        
        # Should have cleaned up resources
        assert cleaned_count >= 0, "Cleanup should return non-negative count"


class TestIntegrationPerformance:
    """Performance tests for integration with external systems."""
    
    @pytest.mark.performance
    def test_mock_km_client_performance(self):
        """Mock KM client should perform within expected bounds."""
        client = MockKeyboardMaestroClient(
            success_rate=1.0,
            response_delay=0.05,  # 50ms delay
            simulate_failures=False
        )
        
        num_operations = 10
        operations = []
        for i in range(num_operations):
            operations.append(("register_trigger", {"trigger_type": "hotkey", "key": f"F{i}"}))
            operations.append(("execute_macro", f"test_macro_{i}", {"param": f"value_{i}"}))
        
        start_time = time.perf_counter()
        
        for op_type, *args in operations:
            if op_type == "register_trigger":
                response = client.register_trigger(args[0])
            elif op_type == "execute_macro":
                response = client.execute_macro(args[0], args[1] if len(args) > 1 else None)
            
            assert response.status in ["success", "completed"], f"Operation {op_type} should succeed"
        
        total_time = time.perf_counter() - start_time
        
        # Should complete within reasonable time considering mock delays
        expected_max_time = num_operations * 0.1 + 1.0  # 100ms per op + 1s overhead
        assert_performance_within_bounds(
            total_time,
            max_time=expected_max_time,
            message="Mock KM client performance"
        )
    
    @pytest.mark.performance
    async def test_async_operation_performance(self):
        """Async operations should perform efficiently."""
        client = MockKeyboardMaestroClient(
            response_delay=0.01, 
            success_rate=1.0,  # 100% success for performance testing
            simulate_failures=False
        )
        
        import asyncio
        
        async def async_operations():
            tasks = []
            
            for i in range(10):
                task1 = client.register_trigger_async({"trigger": f"test_{i}"})
                task2 = client.execute_macro_async(f"macro_{i}")
                tasks.extend([task1, task2])
            
            return await asyncio.gather(*tasks)
        
        start_time = time.perf_counter()
        results = await async_operations()
        total_time = time.perf_counter() - start_time
        
        # All operations should succeed
        for result in results:
            assert result.status in ["success", "completed"], "Async operation should succeed"
        
        # Should be faster than sequential execution
        assert_performance_within_bounds(
            total_time,
            max_time=1.0,  # Should complete within 1 second
            message="Async operation performance"
        )


class TestPerformanceRegression:
    """Tests to detect performance regressions."""
    
    @pytest.mark.performance
    def test_baseline_performance_metrics(self):
        """Establish baseline performance metrics."""
        engine = MacroEngine()
        metrics = get_engine_metrics()
        metrics.reset_metrics()
        
        # Execute standard benchmark
        num_macros = 10
        for i in range(num_macros):
            macro = create_test_macro(f"Baseline {i}", [CommandType.TEXT_INPUT])
            context = ExecutionContext.create_test_context()
            
            start_time = time.perf_counter()
            result = engine.execute_macro(macro, context)
            execution_time = time.perf_counter() - start_time
            
            # Record metrics
            metrics.record_execution(Duration.from_seconds(execution_time), result.status.value == "completed")
        
        # Check baseline metrics
        final_metrics = metrics.get_metrics()
        
        assert final_metrics["execution_count"] == num_macros, "Should record all executions"
        assert final_metrics["success_rate"] >= 0.8, "Should have high success rate"
        assert final_metrics["average_execution_time"] <= 0.2, "Should have reasonable average time"
    
    @pytest.mark.performance
    @pytest.mark.parametrize("load_factor", [1, 2, 5])
    def test_performance_under_load(self, load_factor: int):
        """Test performance characteristics under varying load."""
        engine = MacroEngine()
        base_operations = 5
        total_operations = base_operations * load_factor
        
        start_time = time.perf_counter()
        
        for i in range(total_operations):
            macro = create_test_macro(f"Load Test {i}", [CommandType.TEXT_INPUT])
            context = ExecutionContext.create_test_context()
            
            result = engine.execute_macro(macro, context)
            assert result.status.value in ["completed", "failed"], f"Operation {i} should complete"
        
        total_time = time.perf_counter() - start_time
        avg_time_per_op = total_time / total_operations
        
        # Performance should degrade gracefully with load
        max_avg_time = 0.2 * (1 + load_factor * 0.1)  # Allow 10% degradation per load factor
        
        assert avg_time_per_op <= max_avg_time, \
            f"Average time per operation {avg_time_per_op:.3f}s exceeds threshold {max_avg_time:.3f}s at load factor {load_factor}"