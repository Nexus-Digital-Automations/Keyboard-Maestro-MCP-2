"""Secure subprocess execution framework for enterprise command processing.

This module provides secure subprocess execution with comprehensive validation,
path resolution, and command injection prevention for enterprise applications.

Security: Command validation, path resolution, input sanitization, audit logging
Performance: <100ms execution, efficient path caching, optimized validation
Type Safety: Complete type validation and secure command execution patterns
"""

from __future__ import annotations

import asyncio
import logging
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.types import Duration

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Allowed command types for secure execution."""

    PROCESS_DISCOVERY = "process_discovery"
    PROCESS_TERMINATION = "process_termination"
    PROCESS_SIGNAL = "process_signal"
    SYSTEM_INFO = "system_info"


@dataclass(frozen=True)
class SecureCommand:
    """Secure command specification with validation."""

    command_type: CommandType
    executable: str
    args: list[str]
    timeout: float = 5.0
    allowed_return_codes: set[int] = frozenset({0})

    def __post_init__(self) -> None:
        """Validate command specification."""
        if self.timeout <= 0 or self.timeout > 30:
            raise ValueError(f"Invalid timeout: {self.timeout}")
        if len(self.args) > 10:
            raise ValueError(f"Too many arguments: {len(self.args)}")


class SecureSubprocessManager:
    """Enterprise-grade secure subprocess execution manager."""

    def __init__(self) -> None:
        self._executable_cache: dict[str, str | None] = {}
        self._platform = platform.system().lower()

        # Define allowed commands per platform
        self._allowed_commands = {
            "darwin": {
                CommandType.PROCESS_DISCOVERY: ["pgrep", "ps"],
                CommandType.PROCESS_TERMINATION: ["kill"],
                CommandType.PROCESS_SIGNAL: ["kill"],
                CommandType.SYSTEM_INFO: ["uname", "sw_vers", "osascript", "open"],
            },
            "linux": {
                CommandType.PROCESS_DISCOVERY: ["pgrep", "ps", "pidof"],
                CommandType.PROCESS_TERMINATION: ["kill"],
                CommandType.PROCESS_SIGNAL: ["kill"],
                CommandType.SYSTEM_INFO: ["uname", "lsb_release"],
            },
            "windows": {
                CommandType.PROCESS_DISCOVERY: ["tasklist"],
                CommandType.PROCESS_TERMINATION: ["taskkill"],
                CommandType.PROCESS_SIGNAL: ["taskkill"],
                CommandType.SYSTEM_INFO: ["systeminfo", "ver"],
            },
        }

    def _resolve_executable_path(self, command_name: str) -> str | None:
        """Resolve command to full executable path with caching."""
        if command_name in self._executable_cache:
            return self._executable_cache[command_name]

        try:
            # Use shutil.which for secure path resolution
            executable_path = shutil.which(command_name)

            if executable_path and self._validate_executable_path(executable_path):
                self._executable_cache[command_name] = executable_path
                logger.debug(f"Resolved {command_name} to {executable_path}")
                return executable_path
            logger.warning(f"Command not found or invalid: {command_name}")
            self._executable_cache[command_name] = None
            return None

        except Exception as e:
            logger.error(f"Error resolving executable {command_name}: {e}")
            self._executable_cache[command_name] = None
            return None

    def _validate_executable_path(self, path: str) -> bool:
        """Validate executable path for security."""
        try:
            path_obj = Path(path)

            # Must be an absolute path
            if not path_obj.is_absolute():
                return False

            # Must exist and be executable
            if not (path_obj.exists() and path_obj.is_file()):
                return False

            # Platform-specific validation
            if self._platform == "windows":
                # Windows executables must be in system directories or have .exe extension
                safe_dirs = [
                    "system32",
                    "windows\\system32",
                    "program files",
                ]
                path_lower = str(path_obj).lower()
                return any(
                    safe_dir in path_lower for safe_dir in safe_dirs
                ) or path_lower.endswith(".exe")
            # Unix-like: must be in standard system directories
            safe_dirs = [
                "/bin/",
                "/usr/bin/",
                "/usr/local/bin/",
                "/sbin/",
                "/usr/sbin/",
            ]
            return any(str(path_obj).startswith(safe_dir) for safe_dir in safe_dirs)

        except Exception:
            return False

    # FIXME: Contract disabled - @require(lambda app_name: len(app_name) > 0 and len(app_name) <= 100)
    def _validate_app_name(self, app_name: str) -> bool:
        """Validate application name to prevent command injection."""
        # Allow only alphanumeric, dots, hyphens, underscores, spaces, and forward slashes
        if not re.match(r"^[a-zA-Z0-9._\-/\s]+$", app_name):
            return False

        # Prevent path traversal
        if ".." in app_name:
            return False

        # Prevent shell metacharacters
        dangerous_chars = ["&", "|", ";", "`", "$", "(", ")", "<", ">", "\\"]
        return not any(char in app_name for char in dangerous_chars)

    def execute_secure_command(
        self,
        command: SecureCommand,
    ) -> subprocess.CompletedProcess:
        """Execute command with comprehensive security validation."""
        # Validate command type is allowed on this platform
        allowed_commands = self._allowed_commands.get(self._platform, {})
        if command.command_type not in allowed_commands:
            raise ValueError(
                f"Command type {command.command_type} not allowed on {self._platform}",
            )

        # Validate executable is in allowed list
        if command.executable not in allowed_commands[command.command_type]:
            raise ValueError(
                f"Executable {command.executable} not allowed for {command.command_type}",
            )

        # Resolve to full path
        executable_path = self._resolve_executable_path(command.executable)
        if not executable_path:
            raise ValueError(f"Cannot resolve executable: {command.executable}")

        # Build secure command list
        cmd_list = [executable_path] + command.args

        try:
            logger.info(f"Executing secure command: {command.command_type.value}")
            logger.debug(f"Command: {cmd_list}")

            # Execute with security constraints
            result = subprocess.run(  # noqa: S603 # Secured subprocess with validation
                cmd_list,
                capture_output=True,
                text=True,
                timeout=command.timeout,
                check=False,  # Don't raise on non-zero exit
            )

            # Validate return code if specified
            if (
                command.allowed_return_codes
                and result.returncode not in command.allowed_return_codes
            ):
                logger.warning(
                    f"Command returned unexpected code {result.returncode}, "
                    f"expected one of {command.allowed_return_codes}",
                )

            return result

        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timeout after {command.timeout}s: {e}")
            raise
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise

    async def execute_secure_command_async(
        self, command: SecureCommand
    ) -> subprocess.CompletedProcess[str]:
        """Execute secure command asynchronously with proper resource cleanup."""
        # Validate command first (same as sync version)
        allowed_commands = self._allowed_commands.get(self._platform)
        if not allowed_commands:
            raise ValueError(
                f"Command type {command.command_type} not allowed on {self._platform}",
            )

        if command.executable not in allowed_commands[command.command_type]:
            raise ValueError(
                f"Executable {command.executable} not allowed for {command.command_type}",
            )

        executable_path = self._resolve_executable_path(command.executable)
        if not executable_path:
            raise ValueError(f"Cannot resolve executable: {command.executable}")

        cmd_list = [executable_path] + command.args

        try:
            logger.info(f"Executing async secure command: {command.command_type.value}")
            logger.debug(f"Command: {cmd_list}")

            # Create subprocess with proper cleanup
            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                # Wait for completion with timeout
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(), timeout=command.timeout
                )

                # Create result compatible with subprocess.CompletedProcess
                result = subprocess.CompletedProcess(
                    args=cmd_list,
                    returncode=process.returncode or 0,
                    stdout=stdout_data.decode() if stdout_data else "",
                    stderr=stderr_data.decode() if stderr_data else "",
                )

                # Validate return code if specified
                if (
                    command.allowed_return_codes
                    and result.returncode not in command.allowed_return_codes
                ):
                    logger.warning(
                        f"Command returned unexpected code {result.returncode}, "
                        f"expected one of {command.allowed_return_codes}",
                    )

                return result

            except asyncio.TimeoutError:
                # Kill the process if it times out
                try:
                    process.kill()
                    await process.wait()
                except Exception as cleanup_error:
                    logger.debug(
                        f"Process cleanup failed during timeout: {cleanup_error}"
                    )

                logger.error(f"Async command timeout after {command.timeout}s")
                raise subprocess.TimeoutExpired(cmd_list, command.timeout) from None

        except Exception as e:
            logger.error(f"Async command execution failed: {e}")
            raise

    # FIXME: Contract disabled - @require(lambda app_name: len(app_name) > 0)
    def find_application_pids(self, app_name: str) -> list[int]:
        """Find PIDs for application using secure process discovery."""
        if not self._validate_app_name(app_name):
            logger.warning(f"Invalid application name: {app_name}")
            return []

        try:
            if self._platform in ["darwin", "linux"]:
                # Use pgrep for Unix-like systems
                command = SecureCommand(
                    command_type=CommandType.PROCESS_DISCOVERY,
                    executable="pgrep",
                    args=["-f", app_name],
                    allowed_return_codes={0, 1},  # 1 means no matches found
                )

                result = self.execute_secure_command(command)

                if result.returncode == 0:
                    pids = []
                    for line in result.stdout.strip().split("\n"):
                        if line.strip():
                            try:
                                pids.append(int(line.strip()))
                            except ValueError:
                                continue
                    return pids

            elif self._platform == "windows":
                # Use tasklist for Windows
                # Escape app_name for Windows command line
                safe_app_name = app_name.replace('"', '""')

                command = SecureCommand(
                    command_type=CommandType.PROCESS_DISCOVERY,
                    executable="tasklist",
                    args=["/FI", f"IMAGENAME eq {safe_app_name}*"],
                    allowed_return_codes={0, 1},  # 1 means no matches found
                )

                result = self.execute_secure_command(command)

                if result.returncode == 0:
                    pids = []
                    lines = result.stdout.split("\n")
                    for line in lines[3:]:  # Skip header lines
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    pids.append(int(parts[1]))
                                except ValueError:
                                    continue
                    return pids

            return []

        except Exception as e:
            logger.error(f"Error finding application PIDs: {e}")
            return []

    # FIXME: Contract disabled - @require(lambda pid: isinstance(pid, int) and pid > 0)
    def terminate_process(
        self,
        pid: int,
        force: bool = False,
        timeout: Duration | None = None,
    ) -> dict[str, Any]:
        """Terminate process using secure process termination."""
        if timeout is None:
            timeout = Duration.from_seconds(10)
        try:
            if self._platform == "windows":
                if force:
                    command = SecureCommand(
                        command_type=CommandType.PROCESS_TERMINATION,
                        executable="taskkill",
                        args=["/F", "/PID", str(pid)],
                        allowed_return_codes={0, 128, 1},  # Process might not exist
                    )
                else:
                    command = SecureCommand(
                        command_type=CommandType.PROCESS_TERMINATION,
                        executable="taskkill",
                        args=["/PID", str(pid)],
                        allowed_return_codes={0, 128, 1},  # Process might not exist
                    )
            else:
                # Unix-like systems
                signal_arg = "-9" if force else "-TERM"
                command = SecureCommand(
                    command_type=CommandType.PROCESS_TERMINATION,
                    executable="kill",
                    args=[signal_arg, str(pid)],
                    allowed_return_codes={0, 1},  # 1 means process not found
                )

            result = self.execute_secure_command(command)

            return {
                "pid": pid,
                "success": result.returncode == 0,
                "method": "force_kill" if force else "graceful_term",
                "error": result.stderr.strip() if result.stderr else None,
            }

        except Exception as e:
            logger.error(f"Error terminating process {pid}: {e}")
            return {
                "pid": pid,
                "success": False,
                "method": "exception",
                "error": str(e),
            }

    # FIXME: Contract disabled - @require(lambda pid: isinstance(pid, int) and pid > 0)
    def is_process_running(self, pid: int) -> bool:
        """Check if process is running using secure process discovery."""
        try:
            if self._platform == "windows":
                command = SecureCommand(
                    command_type=CommandType.PROCESS_DISCOVERY,
                    executable="tasklist",
                    args=["/FI", f"PID eq {pid}"],
                    allowed_return_codes={0, 1},
                )

                result = self.execute_secure_command(command)
                return (
                    f"PID {pid}" in result.stdout if result.returncode == 0 else False
                )
            # Unix-like: use kill -0 to check if process exists
            command = SecureCommand(
                command_type=CommandType.PROCESS_SIGNAL,
                executable="kill",
                args=["-0", str(pid)],
                allowed_return_codes={0, 1},  # 1 means process not found
            )

            result = self.execute_secure_command(command)
            return result.returncode == 0

        except Exception:
            return False


# Global secure subprocess manager instance
_secure_manager: SecureSubprocessManager | None = None


def get_secure_subprocess_manager() -> SecureSubprocessManager:
    """Get global secure subprocess manager instance."""
    global _secure_manager
    if _secure_manager is None:
        _secure_manager = SecureSubprocessManager()
    return _secure_manager
