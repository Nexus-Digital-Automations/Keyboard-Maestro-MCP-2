"""
Mock objects and test doubles for comprehensive testing of the macro system.

This module provides sophisticated mocks for external dependencies and
complex system components to enable isolated testing.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from unittest.mock import MagicMock, AsyncMock, Mock
from datetime import datetime
import asyncio
import threading
import time
from dataclasses import dataclass, field

from src.core import (
    ExecutionToken, MacroId, CommandId, ExecutionStatus, ExecutionResult,
    ExecutionContext, CommandResult, Duration, Permission
)


@dataclass
class MockKMResponse:
    """Mock response from Keyboard Maestro operations."""
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.1
    timestamp: datetime = field(default_factory=datetime.now)


class MockKeyboardMaestroClient:
    """
    Sophisticated mock for Keyboard Maestro client integration.
    
    Simulates realistic behavior including delays, errors, and state management.
    """
    
    def __init__(self, 
                 success_rate: float = 0.95,
                 response_delay: float = 0.1,
                 simulate_failures: bool = True):
        self.success_rate = success_rate
        self.response_delay = response_delay
        self.simulate_failures = simulate_failures
        
        # Internal state
        self.registered_triggers: Dict[str, Dict[str, Any]] = {}
        self.macro_executions: Dict[str, MockKMResponse] = {}
        self.connection_status = "connected"
        self.last_error = None
        
        # Statistics
        self.call_count = 0
        self.error_count = 0
        
        # Thread safety
        self._lock = threading.Lock()
    
    def register_trigger(self, trigger_config: Dict[str, Any]) -> MockKMResponse:
        """Mock trigger registration with realistic behavior."""
        self.call_count += 1
        
        # Simulate network delay
        time.sleep(self.response_delay)
        
        # Simulate failures
        if self.simulate_failures and self._should_fail():
            self.error_count += 1
            self.last_error = "Failed to register trigger with Keyboard Maestro"
            return MockKMResponse(
                status="error",
                error=self.last_error
            )
        
        # Success case
        trigger_id = f"trigger_{len(self.registered_triggers)}"
        with self._lock:
            self.registered_triggers[trigger_id] = trigger_config.copy()
        
        return MockKMResponse(
            status="success",
            data={"trigger_id": trigger_id, "registered": True}
        )
    
    def execute_macro(self, macro_id: str, parameters: Optional[Dict[str, Any]] = None) -> MockKMResponse:
        """Mock macro execution with realistic timing."""
        self.call_count += 1
        
        # Simulate execution time based on complexity
        execution_time = self.response_delay + (len(parameters or {}) * 0.01)
        time.sleep(execution_time)
        
        if self.simulate_failures and self._should_fail():
            self.error_count += 1
            self.last_error = f"Macro {macro_id} execution failed"
            return MockKMResponse(
                status="error",
                error=self.last_error,
                execution_time=execution_time
            )
        
        # Success case
        response = MockKMResponse(
            status="completed",
            data={
                "macro_id": macro_id,
                "parameters": parameters or {},
                "output": f"Executed macro {macro_id}"
            },
            execution_time=execution_time
        )
        
        with self._lock:
            self.macro_executions[macro_id] = response
        
        return response
    
    def get_macro_status(self, macro_id: str) -> MockKMResponse:
        """Mock macro status retrieval."""
        self.call_count += 1
        
        if macro_id in self.macro_executions:
            last_execution = self.macro_executions[macro_id]
            return MockKMResponse(
                status="success",
                data={
                    "macro_id": macro_id,
                    "enabled": True,
                    "last_execution": last_execution.timestamp.isoformat(),
                    "execution_count": 1
                }
            )
        
        return MockKMResponse(
            status="success",
            data={
                "macro_id": macro_id,
                "enabled": True,
                "last_execution": None,
                "execution_count": 0
            }
        )
    
    async def register_trigger_async(self, trigger_config: Dict[str, Any]) -> MockKMResponse:
        """Async version of trigger registration."""
        await asyncio.sleep(self.response_delay)
        return self.register_trigger(trigger_config)
    
    async def execute_macro_async(self, macro_id: str, parameters: Optional[Dict[str, Any]] = None) -> MockKMResponse:
        """Async version of macro execution."""
        await asyncio.sleep(self.response_delay)
        return self.execute_macro(macro_id, parameters)
    
    def _should_fail(self) -> bool:
        """Determine if operation should fail based on success rate."""
        import random
        return random.random() > self.success_rate
    
    def reset_stats(self):
        """Reset statistics and state."""
        with self._lock:
            self.call_count = 0
            self.error_count = 0
            self.registered_triggers.clear()
            self.macro_executions.clear()
            self.last_error = None
    
    def set_connection_status(self, status: str):
        """Simulate connection status changes."""
        self.connection_status = status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mock client statistics."""
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "success_rate": (self.call_count - self.error_count) / max(self.call_count, 1),
            "registered_triggers": len(self.registered_triggers),
            "executed_macros": len(self.macro_executions),
            "connection_status": self.connection_status
        }


class MockExecutionContext:
    """Mock execution context for testing command execution."""
    
    def __init__(self, 
                 permissions: Optional[frozenset[Permission]] = None,
                 timeout: Optional[Duration] = None,
                 variables: Optional[Dict[str, str]] = None):
        self.permissions = permissions or frozenset([Permission.TEXT_INPUT])
        self.timeout = timeout or Duration.from_seconds(30)
        self.variables = variables or {}
        self.execution_id = ExecutionToken(f"mock_token_{id(self)}")
        self.created_at = datetime.now()
        
        # Mock methods
        self.has_permission = Mock(side_effect=lambda p: p in self.permissions)
        self.has_permissions = Mock(side_effect=lambda ps: ps.issubset(self.permissions))
        self.get_variable = Mock(side_effect=lambda name: self.variables.get(name))
    
    def with_variable(self, name: str, value: str):
        """Add variable to mock context."""
        new_vars = self.variables.copy()
        new_vars[name] = value
        return MockExecutionContext(
            permissions=self.permissions,
            timeout=self.timeout,
            variables=new_vars
        )


class MockCommand:
    """Mock command for testing command execution."""
    
    def __init__(self, 
                 command_id: str,
                 execution_time: float = 0.1,
                 should_succeed: bool = True,
                 required_permissions: Optional[List[Permission]] = None):
        self.command_id = CommandId(command_id)
        self.execution_time = execution_time
        self.should_succeed = should_succeed
        self.required_permissions = frozenset(required_permissions or [])
        
        # Execution tracking
        self.execution_count = 0
        self.last_context = None
    
    def execute(self, context: ExecutionContext) -> CommandResult:
        """Mock command execution."""
        self.execution_count += 1
        self.last_context = context
        
        # Simulate execution time
        time.sleep(self.execution_time)
        
        if self.should_succeed:
            return CommandResult.success_result(
                output=f"Mock command {self.command_id} executed",
                execution_time=Duration.from_seconds(self.execution_time)
            )
        else:
            return CommandResult.failure_result(
                error_message=f"Mock command {self.command_id} failed",
                execution_time=Duration.from_seconds(self.execution_time)
            )
    
    def validate(self) -> bool:
        """Mock validation always succeeds."""
        return True
    
    def get_dependencies(self) -> List[CommandId]:
        """Mock dependencies."""
        return []
    
    def get_required_permissions(self) -> frozenset[Permission]:
        """Get required permissions."""
        return self.required_permissions


class MockMacroEngine:
    """Mock macro engine for testing integration scenarios."""
    
    def __init__(self):
        self.executions: Dict[ExecutionToken, ExecutionResult] = {}
        self.execution_count = 0
        self.active_executions: Dict[ExecutionToken, ExecutionStatus] = {}
        
        # Mock behavior configuration
        self.default_execution_time = 0.1
        self.should_succeed = True
        self.max_concurrent = 10
    
    def execute_macro(self, macro, context=None) -> ExecutionResult:
        """Mock macro execution."""
        self.execution_count += 1
        
        # Generate execution token
        token = ExecutionToken(f"mock_exec_{self.execution_count}")
        
        # Simulate execution
        start_time = datetime.now()
        time.sleep(self.default_execution_time)
        
        # Create result
        status = ExecutionStatus.COMPLETED if self.should_succeed else ExecutionStatus.FAILED
        
        result = ExecutionResult(
            execution_token=token,
            macro_id=macro.macro_id,
            status=status,
            started_at=start_time,
            completed_at=datetime.now(),
            total_duration=Duration.from_seconds(self.default_execution_time)
        )
        
        self.executions[token] = result
        return result
    
    def get_execution_status(self, token: ExecutionToken) -> Optional[ExecutionStatus]:
        """Get execution status."""
        if token in self.executions:
            return self.executions[token].status
        return None
    
    def cancel_execution(self, token: ExecutionToken) -> bool:
        """Mock cancellation."""
        if token in self.executions:
            result = self.executions[token]
            if result.status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
                # Update status to cancelled
                self.active_executions[token] = ExecutionStatus.CANCELLED
                return True
        return False
    
    def get_active_executions(self) -> List[ExecutionToken]:
        """Get active executions."""
        return list(self.active_executions.keys())


class MockFileSystem:
    """Mock file system for testing file operations."""
    
    def __init__(self):
        self.files: Dict[str, str] = {}
        self.directories: set[str] = {"/", "/tmp", "/test"}
        self.permissions: Dict[str, str] = {}  # path -> permission level
        
        # Access tracking
        self.read_count = 0
        self.write_count = 0
        self.access_log: List[Dict[str, Any]] = []
    
    def exists(self, path: str) -> bool:
        """Check if file exists."""
        self._log_access("exists", path)
        return path in self.files or path in self.directories
    
    def read_file(self, path: str) -> str:
        """Read file content."""
        self._log_access("read", path)
        self.read_count += 1
        
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        
        return self.files[path]
    
    def write_file(self, path: str, content: str) -> None:
        """Write file content."""
        self._log_access("write", path)
        self.write_count += 1
        self.files[path] = content
    
    def delete_file(self, path: str) -> None:
        """Delete file."""
        self._log_access("delete", path)
        if path in self.files:
            del self.files[path]
    
    def list_directory(self, path: str) -> List[str]:
        """List directory contents."""
        self._log_access("list", path)
        
        if path not in self.directories:
            raise NotADirectoryError(f"Not a directory: {path}")
        
        # Return files in the directory
        prefix = path.rstrip("/") + "/"
        return [f for f in self.files.keys() if f.startswith(prefix)]
    
    def _log_access(self, operation: str, path: str):
        """Log file system access."""
        self.access_log.append({
            "operation": operation,
            "path": path,
            "timestamp": datetime.now()
        })
    
    def reset(self):
        """Reset file system state."""
        self.files.clear()
        self.directories = {"/", "/tmp", "/test"}
        self.read_count = 0
        self.write_count = 0
        self.access_log.clear()


# Factory functions for creating configured mocks
def create_failing_km_client() -> MockKeyboardMaestroClient:
    """Create a KM client that frequently fails."""
    return MockKeyboardMaestroClient(
        success_rate=0.3,
        response_delay=0.2,
        simulate_failures=True
    )


def create_slow_km_client() -> MockKeyboardMaestroClient:
    """Create a KM client with slow responses."""
    return MockKeyboardMaestroClient(
        success_rate=0.95,
        response_delay=1.0,
        simulate_failures=False
    )


def create_reliable_km_client() -> MockKeyboardMaestroClient:
    """Create a highly reliable KM client."""
    return MockKeyboardMaestroClient(
        success_rate=0.99,
        response_delay=0.05,
        simulate_failures=False
    )


def create_mock_command_sequence(count: int, 
                                execution_time: float = 0.1,
                                failure_rate: float = 0.0) -> List[MockCommand]:
    """Create a sequence of mock commands."""
    commands = []
    for i in range(count):
        should_succeed = (i / count) >= failure_rate
        command = MockCommand(
            command_id=f"mock_cmd_{i}",
            execution_time=execution_time,
            should_succeed=should_succeed
        )
        commands.append(command)
    return commands


def create_privileged_mock_context() -> MockExecutionContext:
    """Create mock context with all permissions."""
    return MockExecutionContext(
        permissions=frozenset(Permission),
        timeout=Duration.from_seconds(300),
        variables={"test_var": "test_value"}
    )