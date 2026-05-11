"""Keyboard Maestro Token Engine Integration.

Provides integration with Keyboard Maestro's native token processing system
for enhanced token resolution and context-aware processing.
"""

from __future__ import annotations

import subprocess
import time

from ..core.contracts import ensure, require
from ..core.types import Duration
from ..integration.km_client import Either, KMError
from .token_processor import ProcessingContext


class KMTokenEngine:
    """Integration with Keyboard Maestro's token processing system."""

    def __init__(self, timeout: Duration | None = None):
        # B008 fix: Move function call from default argument to function body
        if timeout is None:
            timeout = Duration.from_seconds(30)
        self.timeout = timeout
        self._processing_stats = {"km_calls": 0, "km_errors": 0, "fallbacks": 0}

    @require(
        lambda _self, text: len(text) > 0 and len(text) <= 50000,
        "Text must be 1-50000 characters",
    )
    @ensure(
        lambda _self, _text, _context, result: result.is_right()
        or result.get_left().code
        in ["EXECUTION_ERROR", "TIMEOUT_ERROR", "VALIDATION_ERROR"],
        "Must return valid Either with expected error codes",
    )
    async def process_with_km(
        self,
        text: str,
        context: ProcessingContext = ProcessingContext.TEXT,
    ) -> Either[KMError, str]:
        """Process tokens using KM's token processing engine with security validation."""
        start_time = time.time()

        try:
            # Validate input for safety
            if not self._is_safe_for_km_processing(text):
                return Either.left(
                    KMError.validation_error(
                        "Text contains patterns unsafe for KM processing",
                    ),
                )

            # Build AppleScript for KM token processing
            context_param = self._context_to_km_parameter(context)
            escaped_text = self._escape_for_applescript(text)

            script = f"""
            tell application "Keyboard Maestro Engine"
                try
                    set result to process tokens "{escaped_text}" {context_param}
                    return result as string
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            """

            # S607 SECURITY FIX: Use secure subprocess execution
            from ..commands.secure_subprocess import (
                CommandType,
                SecureCommand,
                get_secure_subprocess_manager,
            )

            secure_manager = get_secure_subprocess_manager()
            command = SecureCommand(
                command_type=CommandType.SYSTEM_INFO,
                executable="osascript",
                args=["-e", script],
                timeout=self.timeout.total_seconds(),
                allowed_return_codes={0, 1},
            )
            result = secure_manager.execute_secure_command(command)

            self._processing_stats["km_calls"] += 1

            if result.returncode != 0:
                self._processing_stats["km_errors"] += 1
                return Either.left(
                    KMError.execution_error(
                        f"KM token processing failed: {result.stderr}",
                        details={"returncode": result.returncode},
                    ),
                )

            output = result.stdout.strip()
            if output.startswith("ERROR:"):
                self._processing_stats["km_errors"] += 1
                return Either.left(KMError.execution_error(output[6:].strip()))

            return Either.right(output)

        except subprocess.TimeoutExpired:
            self._processing_stats["km_errors"] += 1
            return Either.left(KMError.timeout_error(self.timeout))
        except Exception as e:
            self._processing_stats["km_errors"] += 1
            return Either.left(
                KMError.execution_error(
                    f"KM token processing error: {e!s}",
                    details={"processing_time": time.time() - start_time},
                ),
            )

    def _is_safe_for_km_processing(self, text: str) -> bool:
        """Validate text is safe for KM processing to prevent injection."""
        # Check for dangerous AppleScript patterns
        dangerous_patterns = [
            "do shell script",
            "system events",
            "mount volume",
            "unmount",
            "restart",
            "shutdown",
            'tell application "Terminal"',
            'tell application "Script Editor"',
            "-- dangerous comment patterns that could hide code",
        ]

        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern in text_lower:
                return False

        # Check for excessive nesting or complex patterns
        if text.count("%") > 50:  # Too many tokens
            return False

        # SIM103 fix: Return the negated condition directly
        return not (
            text.count('"') > text.count('\\"') + 10
        )  # Check for unescaped quotes

    def _context_to_km_parameter(self, context: ProcessingContext) -> str:
        """Convert processing context to KM parameter safely."""
        context_map = {
            ProcessingContext.TEXT: "",
            ProcessingContext.CALCULATION: "for calculation",
            ProcessingContext.REGEX: "for regex",
            ProcessingContext.FILENAME: "for filename",
            ProcessingContext.URL: "for url",
        }
        return context_map.get(context, "")

    def _escape_for_applescript(self, text: str) -> str:
        """Escape text for safe AppleScript usage with comprehensive protection."""
        # Replace backslashes first to avoid double escaping
        escaped = text.replace("\\", "\\\\")

        # Escape quotes
        escaped = escaped.replace('"', '\\"')

        # Escape other special characters
        escaped = escaped.replace("\n", "\\n")
        escaped = escaped.replace("\r", "\\r")
        escaped = escaped.replace("\t", "\\t")

        # Truncate if too long for AppleScript safety
        if len(escaped) > 10000:
            escaped = escaped[:10000] + "..."

        return escaped

    async def test_km_connection(self) -> Either[KMError, bool]:
        """Test connection to Keyboard Maestro Engine."""
        test_result = await self.process_with_km(
            "Test Connection",
            ProcessingContext.TEXT,
        )

        if test_result.is_right():
            return Either.right(True)
        return Either.left(test_result.get_left())

    def get_processing_stats(self) -> dict[str, int]:
        """Get processing statistics for monitoring."""
        return self._processing_stats.copy()

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._processing_stats = {"km_calls": 0, "km_errors": 0, "fallbacks": 0}
