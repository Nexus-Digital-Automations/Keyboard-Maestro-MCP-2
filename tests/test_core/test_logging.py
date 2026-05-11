"""Comprehensive tests for logging infrastructure.

This module provides extensive test coverage for the logging utilities,
including configuration, handler setup, and edge case scenarios.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from src.core.logging import get_logger


class TestGetLogger:
    """Test the get_logger function comprehensively."""

    def test_get_logger_returns_logger_instance(self) -> None:
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_get_logger_sets_info_level(self) -> None:
        """Test that logger level is set to INFO."""
        logger = get_logger("test_logger_level")
        assert logger.level == logging.INFO

    def test_get_logger_adds_console_handler(self) -> None:
        """Test that console handler is added to stderr."""
        logger = get_logger("test_console_handler")

        # Find console handler
        console_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream == sys.stderr
        ]
        assert len(console_handlers) == 1

        console_handler = console_handlers[0]
        assert console_handler.level == logging.INFO
        assert isinstance(console_handler.formatter, logging.Formatter)

    def test_get_logger_formatter_format(self) -> None:
        """Test the formatter format string."""
        logger = get_logger("test_formatter")

        console_handler = next(
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        )
        formatter = console_handler.formatter

        # Check format string
        assert formatter is not None
        assert formatter._fmt == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def test_get_logger_no_propagation(self) -> None:
        """Test that logger doesn't propagate to root logger."""
        logger = get_logger("test_no_propagate")
        assert logger.propagate is False

    def test_get_logger_idempotent(self) -> None:
        """Test that calling get_logger multiple times doesn't add duplicate handlers."""
        logger_name = "test_idempotent"

        # Clear any existing handlers
        existing_logger = logging.getLogger(logger_name)
        existing_logger.handlers.clear()

        # Get logger twice
        logger1 = get_logger(logger_name)
        initial_handler_count = len(logger1.handlers)

        logger2 = get_logger(logger_name)

        assert logger1 is logger2  # Same instance
        assert len(logger2.handlers) == initial_handler_count  # No new handlers added

    @patch("pathlib.Path.exists")
    def test_get_logger_with_file_handler(self, mock_exists: MagicMock) -> None:
        """Test that file handler is added when logs directory exists."""
        mock_exists.return_value = True

        # Create a unique logger name to avoid conflicts
        logger_name = "test_file_handler"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()  # Clear any existing handlers

        with patch("logging.FileHandler") as mock_file_handler_class:
            mock_file_handler = MagicMock()
            mock_file_handler_class.return_value = mock_file_handler

            logger = get_logger(logger_name)

            # Verify FileHandler was created with correct path
            mock_file_handler_class.assert_called_once_with(
                Path("logs") / "km-mcp-server.log"
            )

            # Verify file handler was configured
            mock_file_handler.setLevel.assert_called_once_with(logging.DEBUG)
            mock_file_handler.setFormatter.assert_called_once()

            # Verify it was added to logger
            assert mock_file_handler in logger.handlers

    @patch("pathlib.Path.exists")
    def test_get_logger_without_logs_directory(self, mock_exists: MagicMock) -> None:
        """Test that no file handler is added when logs directory doesn't exist."""
        mock_exists.return_value = False

        logger_name = "test_no_file_handler"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()

        logger = get_logger(logger_name)

        # Should only have console handler
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 0

    def test_get_logger_with_empty_name(self) -> None:
        """Test get_logger with empty string name."""
        logger = get_logger("")
        assert isinstance(logger, logging.Logger)
        # Empty string logger name becomes 'root'
        assert logger.name == "root"

    def test_get_logger_with_special_characters(self) -> None:
        """Test get_logger with special characters in name."""
        special_names = [
            "test.logger",
            "test-logger",
            "test_logger",
            "test:logger",
            "test/logger",
            "test\\logger",
            "test logger with spaces",
            "тест_logger",  # Unicode
            "🔧logger",  # Emoji
        ]

        for name in special_names:
            logger = get_logger(name)
            assert isinstance(logger, logging.Logger)
            assert logger.name == name

    def test_get_logger_handler_inheritance(self) -> None:
        """Test that child loggers don't inherit handlers due to propagate=False."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        # Both should have their own handlers
        assert len(parent_logger.handlers) > 0
        assert len(child_logger.handlers) > 0

        # Child should not propagate
        assert child_logger.propagate is False

    def test_logger_actually_logs_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that logger actually outputs to stderr."""
        logger = get_logger("test_stderr_output")

        # Log a message
        test_message = "Test message for stderr"
        logger.info(test_message)

        # Check stderr
        captured = capsys.readouterr()
        assert test_message in captured.err
        assert captured.out == ""  # Nothing in stdout

    def test_logger_respects_log_level(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that logger respects log level settings."""
        logger = get_logger("test_log_level")

        # DEBUG should not appear (logger level is INFO)
        logger.debug("This debug message should not appear")
        captured = capsys.readouterr()
        assert "This debug message should not appear" not in captured.err

        # INFO should appear
        logger.info("This info message should appear")
        captured = capsys.readouterr()
        assert "This info message should appear" in captured.err

        # WARNING should appear
        logger.warning("This warning should appear")
        captured = capsys.readouterr()
        assert "This warning should appear" in captured.err

    @patch("pathlib.Path.exists")
    @patch("logging.FileHandler")
    def test_file_handler_level_is_debug(
        self, mock_file_handler_class: MagicMock, mock_exists: MagicMock,
    ) -> None:
        """Test that file handler level is set to DEBUG while console is INFO."""
        mock_exists.return_value = True

        # Create mock file handler
        mock_file_handler = MagicMock()
        mock_file_handler_class.return_value = mock_file_handler

        logger_name = "test_file_debug_level"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()

        logger = get_logger(logger_name)

        # Verify file handler has DEBUG level
        mock_file_handler.setLevel.assert_called_once_with(logging.DEBUG)

        # Console handler should still be INFO
        console_handler = next(
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream == sys.stderr
        )
        assert console_handler.level == logging.INFO

    def test_multiple_loggers_independent(self) -> None:
        """Test that multiple loggers are independent."""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")

        assert logger1 is not logger2
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"

        # Each should have its own handlers
        assert logger1.handlers != logger2.handlers

    def test_logger_thread_safety(self) -> None:
        """Test that get_logger is thread-safe."""
        from concurrent.futures import ThreadPoolExecutor

        logger_name = "test_thread_safety"
        results = []

        def get_logger_in_thread() -> None:
            logger = get_logger(logger_name)
            results.append(logger)

        # Get logger from multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_logger_in_thread) for _ in range(10)]
            for future in futures:
                future.result()

        # All should be the same logger instance
        assert all(logger is results[0] for logger in results)

        # Should not have duplicate handlers
        # (might have more than 1 if logger was created before, but should be reasonable)
        assert len(results[0].handlers) <= 2  # Console + possibly file

    @patch("src.core.logging.Path")
    def test_file_handler_creation_failure(self, mock_path_class: MagicMock) -> None:
        """Test graceful handling when file handler creation fails."""
        # Make Path("logs").exists() return True
        mock_logs_path = Mock()
        mock_logs_path.exists.return_value = True
        mock_logs_path.__truediv__ = Mock(return_value="logs/km-mcp-server.log")
        mock_path_class.return_value = mock_logs_path

        # Make file handler creation fail
        with patch("logging.FileHandler", side_effect=OSError("Permission denied")):
            logger_name = "test_file_handler_fail"
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()

            # Should not raise exception (the current implementation doesn't handle this gracefully)
            # This test documents the current behavior, which crashes on file handler failure
            with pytest.raises(OSError):
                logger = get_logger(logger_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
