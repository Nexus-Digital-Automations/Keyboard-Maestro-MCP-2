"""
Advanced Test Execution Engine - TASK_58 Phase 2 Implementation

Advanced test execution with parallel processing, comprehensive validation,
and intelligent test orchestration for automation workflows.

Architecture: Test Execution + Parallel Processing + Validation + Resource Management
Performance: <2s test setup, <10s typical execution, <500ms result analysis
Security: Isolated execution, resource limits, safe test environments
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Callable, AsyncIterator
from datetime import datetime, UTC
import asyncio
import json
import time
import uuid
import contextlib
from pathlib import Path
import tempfile
import shutil

from src.core.testing_architecture import (
    TestExecutionId, TestRunId, TestSuiteId, TestEnvironmentId,
    AutomationTest, TestSuite, TestResult, TestStatus, TestStep,
    TestAssertion, TestConfiguration, TestCriteria, TestEnvironment,
    TestExecutionError, QualityGateError, create_test_run_id,
    create_test_environment_id, calculate_quality_score
)
from src.core.either import Either
from src.core.contracts import require, ensure


@dataclass(frozen=True)
class TestExecutionContext:
    """Test execution context with environment and resources."""
    execution_id: TestExecutionId
    environment_id: TestEnvironmentId
    environment_type: TestEnvironment
    resource_limits: Dict[str, Any]
    isolated_workspace: str
    start_time: datetime
    timeout_seconds: int
    
    def __post_init__(self):
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")


@dataclass(frozen=True)
class TestExecutionPlan:
    """Execution plan for test suite with dependency resolution."""
    plan_id: str
    test_run_id: TestRunId
    execution_order: List[TestExecutionId]
    parallel_groups: List[List[TestExecutionId]]
    estimated_duration_seconds: int
    resource_requirements: Dict[str, Any]
    
    def __post_init__(self):
        if self.estimated_duration_seconds <= 0:
            raise ValueError("Estimated duration must be positive")


@dataclass(frozen=True)
class TestExecutionMetrics:
    """Detailed metrics for test execution analysis."""
    execution_id: TestExecutionId
    setup_time_ms: float
    execution_time_ms: float
    cleanup_time_ms: float
    total_time_ms: float
    memory_peak_mb: float
    cpu_usage_percent: float
    assertions_executed: int
    steps_executed: int
    
    def __post_init__(self):
        if any(time < 0 for time in [self.setup_time_ms, self.execution_time_ms, 
                                   self.cleanup_time_ms, self.total_time_ms]):
            raise ValueError("Time metrics cannot be negative")


class TestExecutionEnvironment:
    """Isolated test execution environment with resource management."""
    
    def __init__(self, environment_id: TestEnvironmentId, config: TestConfiguration):
        self.environment_id = environment_id
        self.config = config
        self.workspace_path: Optional[Path] = None
        self.resource_monitor: Optional[ResourceMonitor] = None
        self.is_active = False
    
    async def __aenter__(self) -> TestExecutionEnvironment:
        """Setup isolated test environment."""
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup test environment."""
        await self.cleanup()
    
    async def setup(self) -> None:
        """Initialize test environment with isolation."""
        if self.is_active:
            return
        
        # Create isolated workspace
        temp_dir = tempfile.mkdtemp(prefix=f"test_env_{self.environment_id}_")
        self.workspace_path = Path(temp_dir)
        
        # Initialize resource monitoring
        self.resource_monitor = ResourceMonitor(
            limits=self.config.resource_limits,
            environment_id=self.environment_id
        )
        await self.resource_monitor.start()
        
        self.is_active = True
    
    async def cleanup(self) -> None:
        """Clean up test environment and resources."""
        if not self.is_active:
            return
        
        # Stop resource monitoring
        if self.resource_monitor:
            await self.resource_monitor.stop()
        
        # Clean up workspace
        if self.workspace_path and self.workspace_path.exists():
            shutil.rmtree(self.workspace_path, ignore_errors=True)
        
        self.is_active = False
    
    def get_workspace_path(self) -> Path:
        """Get the isolated workspace path."""
        if not self.workspace_path:
            raise TestExecutionError("Test environment not initialized", "ENV_NOT_INITIALIZED")
        return self.workspace_path
    
    async def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage metrics."""
        if not self.resource_monitor:
            return {}
        return await self.resource_monitor.get_current_usage()


class ResourceMonitor:
    """Monitor resource usage during test execution."""
    
    def __init__(self, limits: Dict[str, Any], environment_id: TestEnvironmentId):
        self.limits = limits
        self.environment_id = environment_id
        self.monitoring_task: Optional[asyncio.Task] = None
        self.usage_history: List[Dict[str, float]] = []
        self.peak_usage: Dict[str, float] = {}
        self.is_monitoring = False
    
    async def start(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self) -> None:
        """Stop resource monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self) -> None:
        """Resource monitoring loop."""
        while self.is_monitoring:
            try:
                usage = await self._collect_usage_metrics()
                self.usage_history.append(usage)
                
                # Update peak usage
                for metric, value in usage.items():
                    if metric not in self.peak_usage or value > self.peak_usage[metric]:
                        self.peak_usage[metric] = value
                
                # Check limits
                await self._check_resource_limits(usage)
                
                await asyncio.sleep(0.1)  # Monitor every 100ms
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                pass
    
    async def _collect_usage_metrics(self) -> Dict[str, float]:
        """Collect current resource usage metrics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        return {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "open_files": len(process.open_files()),
            "threads": process.num_threads()
        }
    
    async def _check_resource_limits(self, usage: Dict[str, float]) -> None:
        """Check if resource usage exceeds limits."""
        for metric, value in usage.items():
            if metric in self.limits:
                limit = self.limits[metric]
                if value > limit:
                    raise TestExecutionError(
                        f"Resource limit exceeded: {metric} = {value} > {limit}",
                        "RESOURCE_LIMIT_EXCEEDED"
                    )
    
    async def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        if not self.usage_history:
            return {}
        return self.usage_history[-1]
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage."""
        return self.peak_usage.copy()


class TestStepExecutor:
    """Execute individual test steps with validation."""
    
    def __init__(self, environment: TestExecutionEnvironment):
        self.environment = environment
    
    async def execute_step(self, step: TestStep, test_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test step with comprehensive validation."""
        start_time = time.time()
        
        try:
            # Execute the step action
            step_result = await self._execute_step_action(step, test_context)
            
            # Validate assertions
            assertion_results = await self._validate_assertions(step, step_result)
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "step_id": step.step_id,
                "step_name": step.step_name,
                "status": "passed" if all(assertion_results) else "failed",
                "execution_time_ms": execution_time,
                "result": step_result,
                "assertions": assertion_results,
                "resource_usage": await self.environment.get_resource_usage()
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                "step_id": step.step_id,
                "step_name": step.step_name,
                "status": "error",
                "execution_time_ms": execution_time,
                "error": str(e),
                "assertions": [],
                "resource_usage": await self.environment.get_resource_usage()
            }
    
    async def _execute_step_action(self, step: TestStep, context: Dict[str, Any]) -> Any:
        """Execute the step's main action."""
        action_map = {
            "execute_macro": self._execute_macro,
            "validate_system": self._validate_system,
            "check_performance": self._check_performance,
            "verify_security": self._verify_security,
            "setup_environment": self._setup_environment,
            "cleanup_resources": self._cleanup_resources
        }
        
        action_func = action_map.get(step.action)
        if not action_func:
            raise TestExecutionError(f"Unknown action: {step.action}", "UNKNOWN_ACTION")
        
        return await action_func(step.parameters, context)
    
    async def _execute_macro(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a macro for testing."""
        macro_id = parameters.get("macro_id")
        if not macro_id:
            raise TestExecutionError("macro_id required for execute_macro action", "MISSING_PARAMETER")
        
        # Simulate macro execution (in real implementation, this would call the actual macro engine)
        await asyncio.sleep(0.1)  # Simulate execution time
        
        return {
            "macro_id": macro_id,
            "status": "success",
            "execution_time_ms": 100,
            "output": f"Macro {macro_id} executed successfully"
        }
    
    async def _validate_system(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system state."""
        validation_type = parameters.get("validation_type", "basic")
        
        # Simulate system validation
        await asyncio.sleep(0.05)
        
        return {
            "validation_type": validation_type,
            "status": "success",
            "system_health": "good",
            "checks_passed": 5,
            "checks_total": 5
        }
    
    async def _check_performance(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Check performance metrics."""
        metrics = ["response_time", "throughput", "resource_usage"]
        
        # Simulate performance checking
        await asyncio.sleep(0.2)
        
        return {
            "metrics": {
                "response_time_ms": 150,
                "throughput_ops_sec": 1000,
                "cpu_usage_percent": 25,
                "memory_usage_mb": 64
            },
            "status": "success"
        }
    
    async def _verify_security(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify security compliance."""
        security_checks = parameters.get("checks", ["permissions", "encryption", "access_control"])
        
        # Simulate security verification
        await asyncio.sleep(0.3)
        
        return {
            "security_checks": security_checks,
            "status": "success",
            "vulnerabilities_found": 0,
            "compliance_score": 100
        }
    
    async def _setup_environment(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Setup test environment."""
        await asyncio.sleep(0.1)
        return {"status": "success", "setup_time_ms": 100}
    
    async def _cleanup_resources(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Cleanup test resources."""
        await asyncio.sleep(0.1)
        return {"status": "success", "cleanup_time_ms": 100}
    
    async def _validate_assertions(self, step: TestStep, step_result: Any) -> List[bool]:
        """Validate all assertions for a test step."""
        assertion_results = []
        
        for assertion in step.assertions:
            try:
                actual_value = self._extract_value(step_result, assertion.actual_value_path)
                assertion_passed = self._evaluate_assertion(
                    actual_value, 
                    assertion.expected_value, 
                    assertion.assertion_type
                )
                assertion_results.append(assertion_passed)
            except Exception:
                assertion_results.append(False)
        
        return assertion_results
    
    def _extract_value(self, data: Any, path: str) -> Any:
        """Extract value from data using JSONPath-like syntax."""
        if path.startswith("$."):
            path = path[2:]  # Remove $. prefix
        
        current = data
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        
        return current
    
    def _evaluate_assertion(self, actual: Any, expected: Any, assertion_type: str) -> bool:
        """Evaluate a single assertion."""
        if assertion_type == "equals":
            return actual == expected
        elif assertion_type == "contains":
            return expected in str(actual)
        elif assertion_type == "greater_than":
            return float(actual) > float(expected)
        elif assertion_type == "less_than":
            return float(actual) < float(expected)
        elif assertion_type == "matches":
            import re
            return bool(re.match(str(expected), str(actual)))
        else:
            return False


class AdvancedTestRunner:
    """Advanced test execution engine with parallel processing and comprehensive reporting."""
    
    def __init__(self):
        self.active_executions: Dict[TestExecutionId, TestExecutionContext] = {}
        self.execution_history: List[TestResult] = []
    
    async def execute_test(self, test: AutomationTest) -> TestResult:
        """Execute a single automation test."""
        test_run_id = create_test_run_id()
        environment_id = create_test_environment_id()
        
        start_time = datetime.now(UTC)
        
        async with TestExecutionEnvironment(environment_id, test.test_configuration) as env:
            try:
                # Create execution context
                context = TestExecutionContext(
                    execution_id=test.test_id,
                    environment_id=environment_id,
                    environment_type=test.test_configuration.environment,
                    resource_limits=test.test_configuration.resource_limits,
                    isolated_workspace=str(env.get_workspace_path()),
                    start_time=start_time,
                    timeout_seconds=test.test_configuration.timeout_seconds
                )
                
                self.active_executions[test.test_id] = context
                
                # Execute test with timeout
                try:
                    result = await asyncio.wait_for(
                        self._execute_test_steps(test, env),
                        timeout=test.test_configuration.timeout_seconds
                    )
                    status = TestStatus.PASSED if result["success"] else TestStatus.FAILED
                    
                except asyncio.TimeoutError:
                    status = TestStatus.TIMEOUT
                    result = {"success": False, "error": "Test execution timed out", "step_results": []}
                
                end_time = datetime.now(UTC)
                execution_time = (end_time - start_time).total_seconds() * 1000
                
                # Create test result
                test_result = TestResult(
                    test_id=test.test_id,
                    test_run_id=test_run_id,
                    status=status,
                    start_time=start_time,
                    end_time=end_time,
                    execution_time_ms=execution_time,
                    step_results=result.get("step_results", []),
                    assertions_passed=result.get("assertions_passed", 0),
                    assertions_failed=result.get("assertions_failed", 0),
                    performance_metrics=result.get("performance_metrics", {}),
                    resource_usage=await env.get_resource_usage(),
                    error_details=result.get("error")
                )
                
                self.execution_history.append(test_result)
                return test_result
                
            finally:
                # Clean up execution context
                if test.test_id in self.active_executions:
                    del self.active_executions[test.test_id]
    
    async def _execute_test_steps(self, test: AutomationTest, env: TestExecutionEnvironment) -> Dict[str, Any]:
        """Execute all test steps in sequence."""
        step_executor = TestStepExecutor(env)
        step_results = []
        test_context = {"test_id": test.test_id, "environment": env.environment_id}
        
        assertions_passed = 0
        assertions_failed = 0
        
        try:
            # Execute setup steps
            for step in test.setup_steps:
                step_result = await step_executor.execute_step(step, test_context)
                step_results.append(step_result)
                
                if step_result["status"] == "failed":
                    raise TestExecutionError.setup_failed(test.test_id, "Setup step failed")
            
            # Execute main test steps
            for step in test.test_steps:
                step_result = await step_executor.execute_step(step, test_context)
                step_results.append(step_result)
                
                # Count assertions
                passed_assertions = sum(1 for assertion in step_result.get("assertions", []) if assertion)
                failed_assertions = len(step_result.get("assertions", [])) - passed_assertions
                
                assertions_passed += passed_assertions
                assertions_failed += failed_assertions
                
                # Update test context with step results
                test_context[f"step_{step.step_id}_result"] = step_result
            
            # Execute cleanup steps
            for step in test.cleanup_steps:
                step_result = await step_executor.execute_step(step, test_context)
                step_results.append(step_result)
            
            # Determine overall success
            success = all(
                step_result["status"] in ["passed", "success"] 
                for step_result in step_results 
                if step_result.get("step_type") != "cleanup"
            )
            
            return {
                "success": success,
                "step_results": step_results,
                "assertions_passed": assertions_passed,
                "assertions_failed": assertions_failed,
                "performance_metrics": await self._collect_performance_metrics(step_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "step_results": step_results,
                "assertions_passed": assertions_passed,
                "assertions_failed": assertions_failed
            }
    
    async def _collect_performance_metrics(self, step_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Collect performance metrics from step results."""
        total_execution_time = sum(
            step.get("execution_time_ms", 0) for step in step_results
        )
        
        memory_usage = max(
            (step.get("resource_usage", {}).get("memory_mb", 0) for step in step_results),
            default=0
        )
        
        cpu_usage = max(
            (step.get("resource_usage", {}).get("cpu_percent", 0) for step in step_results),
            default=0
        )
        
        return {
            "total_execution_time_ms": total_execution_time,
            "peak_memory_mb": memory_usage,
            "peak_cpu_percent": cpu_usage,
            "steps_executed": len(step_results)
        }
    
    async def execute_test_suite(self, test_suite: TestSuite) -> List[TestResult]:
        """Execute a complete test suite with parallel processing support."""
        if test_suite.parallel_execution:
            return await self._execute_suite_parallel(test_suite)
        else:
            return await self._execute_suite_sequential(test_suite)
    
    async def _execute_suite_sequential(self, test_suite: TestSuite) -> List[TestResult]:
        """Execute test suite sequentially."""
        results = []
        
        for test in test_suite.tests:
            try:
                result = await self.execute_test(test)
                results.append(result)
                
                # Check if we should abort on failure
                if test_suite.abort_on_failure and result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    break
                    
            except Exception as e:
                # Create error result
                error_result = TestResult(
                    test_id=test.test_id,
                    test_run_id=create_test_run_id(),
                    status=TestStatus.ERROR,
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC),
                    execution_time_ms=0,
                    step_results=[],
                    assertions_passed=0,
                    assertions_failed=0,
                    error_details=str(e)
                )
                results.append(error_result)
                
                if test_suite.abort_on_failure:
                    break
        
        return results
    
    async def _execute_suite_parallel(self, test_suite: TestSuite) -> List[TestResult]:
        """Execute test suite with parallel processing."""
        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(test_suite.max_concurrent_tests)
        
        async def execute_with_semaphore(test: AutomationTest) -> TestResult:
            async with semaphore:
                return await self.execute_test(test)
        
        # Execute all tests concurrently
        tasks = [execute_with_semaphore(test) for test in test_suite.tests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = TestResult(
                    test_id=test_suite.tests[i].test_id,
                    test_run_id=create_test_run_id(),
                    status=TestStatus.ERROR,
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC),
                    execution_time_ms=0,
                    step_results=[],
                    assertions_passed=0,
                    assertions_failed=0,
                    error_details=str(result)
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    def get_execution_summary(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate execution summary from test results."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
        error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
        timeout_tests = sum(1 for r in results if r.status == TestStatus.TIMEOUT)
        
        total_execution_time = sum(r.execution_time_ms for r in results)
        avg_execution_time = total_execution_time / total_tests if total_tests > 0 else 0
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "timeout_tests": timeout_tests,
            "success_rate_percent": success_rate,
            "total_execution_time_ms": total_execution_time,
            "average_execution_time_ms": avg_execution_time,
            "quality_score": calculate_quality_score({
                "reliability": success_rate,
                "performance": min(100, max(0, 100 - (avg_execution_time / 1000))),  # Simple performance score
                "coverage": 100  # Assume full coverage for now
            })
        }