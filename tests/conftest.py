"""
Pytest configuration and shared fixtures for the Keyboard Maestro MCP test suite.

This module provides comprehensive test configuration including Hypothesis settings,
mock frameworks, and reusable fixtures for property-based and integration testing.
"""

import pytest
import asyncio
import threading
import time
from typing import Dict, Any, List, Generator
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

# Hypothesis configuration
from hypothesis import settings, HealthCheck, Verbosity

# Core imports
from src.core import (
    MacroId, CommandId, ExecutionToken, ExecutionContext, MacroDefinition,
    Permission, Duration, MacroEngine, CommandType, create_test_macro
)

# Configure Hypothesis profiles for different testing scenarios
settings.register_profile(
    "default",
    max_examples=50,
    deadline=2000,  # 2 seconds
    suppress_health_check=[HealthCheck.too_slow],
    verbosity=Verbosity.normal
)

settings.register_profile(
    "ci",
    max_examples=200,
    deadline=5000,  # 5 seconds
    suppress_health_check=[HealthCheck.too_slow],
    verbosity=Verbosity.verbose
)

settings.register_profile(
    "fast",
    max_examples=10,
    deadline=500,  # 0.5 seconds
    suppress_health_check=[HealthCheck.too_slow]
)

# Load default profile
settings.load_profile("default")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_km_client() -> MagicMock:
    """Mock Keyboard Maestro client for integration testing."""
    mock = MagicMock()
    
    # Configure mock responses
    mock.register_trigger.return_value = {"status": "success", "trigger_id": "test_trigger"}
    mock.execute_macro.return_value = {"status": "completed", "execution_time": 0.1}
    mock.get_macro_status.return_value = {"status": "enabled", "last_run": datetime.now()}
    
    # Add async variants
    mock.register_trigger_async = AsyncMock(return_value={"status": "success"})
    mock.execute_macro_async = AsyncMock(return_value={"status": "completed"})
    
    return mock


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Standard execution context for testing."""
    return ExecutionContext.create_test_context(
        permissions=frozenset([
            Permission.TEXT_INPUT,
            Permission.SYSTEM_SOUND,
            Permission.APPLICATION_CONTROL
        ]),
        timeout=Duration.from_seconds(30)
    )


@pytest.fixture
def minimal_context() -> ExecutionContext:
    """Minimal execution context with basic permissions."""
    return ExecutionContext.create_test_context(
        permissions=frozenset([Permission.TEXT_INPUT]),
        timeout=Duration.from_seconds(10)
    )


@pytest.fixture
def privileged_context() -> ExecutionContext:
    """High-privilege execution context for testing system operations."""
    return ExecutionContext.create_test_context(
        permissions=frozenset([
            Permission.TEXT_INPUT,
            Permission.SYSTEM_SOUND,
            Permission.APPLICATION_CONTROL,
            Permission.SYSTEM_CONTROL,
            Permission.FILE_ACCESS,
            Permission.NETWORK_ACCESS
        ]),
        timeout=Duration.from_seconds(60)
    )


@pytest.fixture
def macro_engine() -> MacroEngine:
    """Clean macro engine instance for testing."""
    return MacroEngine()


@pytest.fixture
def sample_macro() -> MacroDefinition:
    """Sample macro definition for testing."""
    return create_test_macro("Test Macro", [CommandType.TEXT_INPUT, CommandType.PAUSE])


@pytest.fixture
def complex_macro() -> MacroDefinition:
    """Complex macro with multiple command types."""
    return create_test_macro("Complex Macro", [
        CommandType.TEXT_INPUT,
        CommandType.PAUSE,
        CommandType.PLAY_SOUND,
        CommandType.CONDITIONAL,
        CommandType.VARIABLE_SET
    ])


@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                raise ValueError("Timer not properly started and stopped")
            return self.end_time - self.start_time
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()
    
    return PerformanceTimer()


@pytest.fixture
def security_test_data() -> Dict[str, List[str]]:
    """Security test data for injection and validation testing."""
    return {
        "script_injection": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "eval(malicious_code)",
            "exec(dangerous_command)",
            "__import__('os').system('rm -rf /')"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\",
            "~/../../secret",
            "/bin/bash",
            "C:\\Windows\\System32\\cmd.exe",
            "%SYSTEMROOT%\\system32\\",
            "$HOME/../.ssh/id_rsa"
        ],
        "sql_injection": [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM passwords--"
        ],
        "command_injection": [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& format c:",
            "`cat /etc/shadow`",
            "$(whoami)",
            "${PATH}"
        ]
    }


@pytest.fixture
def mock_file_system():
    """Mock file system for testing file operations."""
    mock_fs = {
        "/test/file.txt": "Test content",
        "/test/config.json": '{"setting": "value"}',
        "/secure/secret.txt": "SECRET_DATA",
        "/temp/output.log": "Log entries..."
    }
    
    def mock_file_exists(path: str) -> bool:
        return path in mock_fs
    
    def mock_file_read(path: str) -> str:
        return mock_fs.get(path, "")
    
    def mock_file_write(path: str, content: str) -> None:
        mock_fs[path] = content
    
    class MockFileSystem:
        exists = mock_file_exists
        read = mock_file_read
        write = mock_file_write
        files = mock_fs
    
    return MockFileSystem()


@pytest.fixture
def thread_safety_helper():
    """Helper for testing thread safety."""
    class ThreadSafetyHelper:
        def __init__(self):
            self.results = []
            self.lock = threading.Lock()
            self.errors = []
        
        def run_concurrent(self, func, args_list, max_workers=5):
            """Run function concurrently with different arguments."""
            threads = []
            
            def worker(args):
                try:
                    result = func(*args) if isinstance(args, (list, tuple)) else func(args)
                    with self.lock:
                        self.results.append(result)
                except Exception as e:
                    with self.lock:
                        self.errors.append(e)
            
            for args in args_list[:max_workers]:
                thread = threading.Thread(target=worker, args=(args,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join(timeout=5.0)
            
            return self.results, self.errors
    
    return ThreadSafetyHelper()


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security-focused"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "property: marks tests as property-based tests"
    )


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to tests that take longer than expected
        if "test_performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Add integration marker to integration tests
        if "test_integration" in item.nodeid or "/integration/" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add security marker to security tests
        if "security" in item.nodeid or "injection" in item.nodeid:
            item.add_marker(pytest.mark.security)
        
        # Add property marker to property-based tests
        if "property" in item.nodeid or "hypothesis" in str(item.function):
            item.add_marker(pytest.mark.property)


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_engine_state():
    """Clean up engine state between tests."""
    yield
    
    # Clean up any global state
    from src.core.context import get_context_manager, get_variable_manager
    
    context_manager = get_context_manager()
    variable_manager = get_variable_manager()
    
    # Clean up expired contexts
    context_manager.cleanup_expired_contexts(max_age_seconds=0)
    
    # Reset any global engine metrics
    from src.core.engine import get_engine_metrics
    metrics = get_engine_metrics()
    metrics.reset_metrics()