"""Pytest configuration and shared fixtures for the Keyboard Maestro MCP test suite.

This module provides comprehensive test configuration including Hypothesis settings,
mock frameworks, and reusable fixtures for property-based and integration testing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from hypothesis import HealthCheck, Verbosity, settings
from src.core import (
    CommandType,
    Duration,
    ExecutionContext,
    MacroDefinition,
    MacroEngine,
    Permission,
    create_test_macro,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

# Configure Hypothesis profiles for different testing scenarios
settings.register_profile(
    "default",
    max_examples=50,
    deadline=2000,  # 2 seconds
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    verbosity=Verbosity.normal,
)

settings.register_profile(
    "ci",
    max_examples=200,
    deadline=5000,  # 5 seconds
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    verbosity=Verbosity.verbose,
)

settings.register_profile(
    "fast",
    max_examples=10,
    deadline=500,  # 0.5 seconds
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

# Load default profile
settings.load_profile("default")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    yield loop

    # Clean up pending tasks
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception as cleanup_error:
        logger.debug(f"Event loop cleanup failed: {cleanup_error}")
    finally:
        try:
            loop.close()
        except Exception as close_error:
            logger.debug(f"Event loop close failed: {close_error}")


@pytest.fixture
def mock_km_client() -> MagicMock:
    """Mock Keyboard Maestro client for integration testing."""
    mock = MagicMock()

    # Configure mock responses
    mock.register_trigger.return_value = {
        "status": "success",
        "trigger_id": "test_trigger",
    }
    mock.execute_macro.return_value = {"status": "completed", "execution_time": 0.1}
    mock.get_macro_status.return_value = {
        "status": "enabled",
        "last_run": datetime.now(),
    }

    # Add async variants
    mock.register_trigger_async = AsyncMock(return_value={"status": "success"})
    mock.execute_macro_async = AsyncMock(return_value={"status": "completed"})

    return mock


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Standard execution context for testing."""
    return ExecutionContext.create_test_context(
        permissions=frozenset(
            [
                Permission.TEXT_INPUT,
                Permission.SYSTEM_SOUND,
                Permission.APPLICATION_CONTROL,
            ],
        ),
        timeout=Duration.from_seconds(30),
    )


@pytest.fixture
def minimal_context() -> ExecutionContext:
    """Minimal execution context with basic permissions."""
    return ExecutionContext.create_test_context(
        permissions=frozenset([Permission.TEXT_INPUT]),
        timeout=Duration.from_seconds(10),
    )


@pytest.fixture
def privileged_context() -> ExecutionContext:
    """High-privilege execution context for testing system operations."""
    return ExecutionContext.create_test_context(
        permissions=frozenset(
            [
                Permission.TEXT_INPUT,
                Permission.SYSTEM_SOUND,
                Permission.APPLICATION_CONTROL,
                Permission.SYSTEM_CONTROL,
                Permission.FILE_ACCESS,
                Permission.NETWORK_ACCESS,
            ],
        ),
        timeout=Duration.from_seconds(60),
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
    return create_test_macro(
        "Complex Macro",
        [
            CommandType.TEXT_INPUT,
            CommandType.PAUSE,
            CommandType.PLAY_SOUND,
            CommandType.CONDITIONAL,
            CommandType.VARIABLE_SET,
        ],
    )


@pytest.fixture
def performance_timer() -> bool:
    """Timer utility for performance testing."""

    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self) -> bool:
            self.start_time = time.perf_counter()

        def stop(self) -> bool:
            self.end_time = time.perf_counter()

        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                raise ValueError("Timer not properly started and stopped")
            return self.end_time - self.start_time

        def __enter__(self):
            self.start()
            return self

        def __exit__(
            self,
            exc_type: str,
            exc_val: Exception | str,
            exc_tb: Exception | str,
        ):
            self.stop()

    return PerformanceTimer()


@pytest.fixture
def security_test_data() -> dict[str, list[str]]:
    """Security test data for injection and validation testing."""
    return {
        "script_injection": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "eval(malicious_code)",
            "exec(dangerous_command)",
            "__import__('os').system('rm -rf /')",
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\",
            "~/../../secret",
            "/bin/bash",
            "C:\\Windows\\System32\\cmd.exe",
            "%SYSTEMROOT%\\system32\\",
            "$HOME/../.ssh/id_rsa",
        ],
        "sql_injection": [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM passwords--",
        ],
        "command_injection": [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& format c:",
            "`cat /etc/shadow`",
            "$(whoami)",
            "${PATH}",
        ],
    }


@pytest.fixture
def mock_file_system() -> bool:
    """Mock file system for testing file operations."""
    mock_fs = {
        "/test/file.txt": "Test content",
        "/test/config.json": '{"setting": "value"}',
        "/secure/secret.txt": "SECRET_DATA",
        "/temp/output.log": "Log entries...",
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
def thread_safety_helper() -> Mock:
    """Helper for testing thread safety."""

    class ThreadSafetyHelper:
        def __init__(self):
            self.results = []
            self.lock = threading.Lock()
            self.errors = []

        def run_concurrent(
            self,
            func: Callable[..., Any],
            args_list: list[Any],
            max_workers: Any = 5,
        ) -> None:
            """Run function concurrently with different arguments."""
            threads = []

            def worker(args: list[Any]) -> Mock:
                try:
                    result = (
                        func(*args) if isinstance(args, list | tuple) else func(args)
                    )
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
def pytest_configure(config: dict[str, Any]) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "security: marks tests as security-focused")
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance benchmarks",
    )
    config.addinivalue_line("markers", "property: marks tests as property-based tests")


# Pytest collection hooks
def pytest_collection_modifyitems(config: dict[str, Any], items: list[Any]) -> None:
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
def cleanup_engine_state() -> None:
    """Clean up engine state between tests."""
    yield

    # Clean up any global state
    from src.core.context import get_context_manager, get_variable_manager

    context_manager = get_context_manager()
    get_variable_manager()

    # Clean up expired contexts
    context_manager.cleanup_expired_contexts(max_age_seconds=0)

    # Reset any global engine metrics
    from src.core.engine import get_engine_metrics

    metrics = get_engine_metrics()
    metrics.reset_metrics()


@pytest.fixture(autouse=True)
async def async_cleanup():
    """Clean up async resources between tests."""
    yield

    # Clean up any pending tasks with proper recursion protection
    try:
        current_task = asyncio.current_task()
        tasks = [
            task
            for task in asyncio.all_tasks()
            if not task.done() and task is not current_task
        ]
        if tasks:
            # Cancel tasks without recursive dependency
            for task in tasks:
                if not task.cancelled():
                    try:
                        task.cancel()
                    except Exception as cancel_error:
                        logger.debug(
                            f"Task cancellation failed (already cancelled): {cancel_error}"
                        )

            # Wait briefly for clean shutdown, but don't wait indefinitely
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=0.1
                    )
                except (asyncio.TimeoutError, Exception) as cleanup_error:
                    logger.debug(f"Task cleanup timeout/error: {cleanup_error}")
    except RuntimeError:
        pass  # No event loop running


@pytest.fixture
def async_mock_helper():
    """Helper for creating properly configured AsyncMock objects."""

    def create_async_mock(return_value=None, side_effect=None, **kwargs):
        mock = AsyncMock(**kwargs)
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock

    return create_async_mock
