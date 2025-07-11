"""Core types and protocols for the Keyboard Maestro MCP macro engine.

This module defines branded types, protocols, and data structures for type-safe
macro execution with comprehensive validation and security boundaries.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, NewType, Protocol, TypeVar

# Branded Types for Type Safety
MacroId = NewType("MacroId", str)
CommandId = NewType("CommandId", str)
ExecutionToken = NewType("ExecutionToken", str)
TriggerId = NewType("TriggerId", str)
GroupId = NewType("GroupId", str)
VariableName = NewType("VariableName", str)
TemplateId = NewType("TemplateId", str)
CreationToken = NewType("CreationToken", str)
ClipboardId = NewType("ClipboardId", str)
AppId = NewType("AppId", str)
BundleId = NewType("BundleId", str)
MenuItemId = NewType("MenuItemId", str)
ToolId = NewType("ToolId", str)
UserId = NewType("UserId", str)
ActionId = NewType("ActionId", str)
EmailAddress = NewType("EmailAddress", str)
ConditionId = NewType("ConditionId", str)


# Result type factory functions
def create_success_result(value: Any) -> Result:
    """Create a successful result with the given value."""
    return Result(success=True, value=value, error_message=None, error_code=None)


def create_error_result(message: str, error_code: str) -> Result:
    """Create an error result with the given message and error code."""
    return Result(
        success=False, value=None, error_message=message, error_code=error_code
    )


@dataclass(frozen=True)
class Result:
    """Generic result type for operations that may succeed or fail."""

    success: bool
    value: Any = None
    error_message: str | None = None
    error_code: str | None = None

    def is_success(self) -> bool:
        """Check if the result represents success."""
        return self.success

    def is_error(self) -> bool:
        """Check if the result represents an error."""
        return not self.success

    @property
    def message(self) -> str:
        """Get the error message (for backwards compatibility)."""
        return self.error_message or ""


def create_macro_id() -> MacroId:
    """Create a new unique MacroId."""
    return MacroId(str(uuid.uuid4()))


@dataclass(frozen=True)
class MacroMoveResult:
    """Result of macro movement operation with comprehensive tracking."""

    macro_id: MacroId
    source_group: GroupId
    target_group: GroupId
    execution_time: Duration
    conflicts_resolved: list[str] = field(default_factory=list)
    rollback_info: dict[str, Any] | None = None

    def was_successful(self) -> bool:
        """Check if movement was successful."""
        return self.rollback_info is None


class MoveConflictType(Enum):
    """Types of conflicts that can occur during macro movement."""

    NAME_COLLISION = "name_collision"
    PERMISSION_DENIED = "permission_denied"
    GROUP_NOT_FOUND = "group_not_found"
    MACRO_NOT_FOUND = "macro_not_found"
    MACRO_ENABLED_IN_SOURCE = "macro_enabled_in_source"
    TARGET_GROUP_DISABLED = "target_group_disabled"


class ExecutionStatus(Enum):
    """Macro execution status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class MacroCreationStatus(Enum):
    """Macro creation status tracking."""

    VALIDATING = "validating"
    CREATING = "creating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class Priority(Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Permission(Enum):
    """Security permission types for macro operations."""

    TEXT_INPUT = "text_input"
    SYSTEM_CONTROL = "system_control"
    FILE_ACCESS = "file_access"
    APPLICATION_CONTROL = "application_control"
    NETWORK_ACCESS = "network_access"
    CLIPBOARD_ACCESS = "clipboard_access"
    CLIPBOARD_HISTORY = "clipboard_history"
    CLIPBOARD_NAMED = "clipboard_named"
    SYSTEM_SOUND = "system_sound"
    SCREEN_CAPTURE = "screen_capture"
    AUDIO_OUTPUT = "audio_output"
    WINDOW_MANAGEMENT = "window_management"
    FLOW_CONTROL = "flow_control"
    MOUSE_CONTROL = "mouse_control"
    AUTOMATION_CONTROL = "automation_control"
    READ_ACCESS = "read_access"
    ADMIN_ACCESS = "admin_access"


@dataclass(frozen=True)
class Duration:
    """Immutable duration representation with validation."""

    seconds: float

    def __post_init__(self):
        if self.seconds < 0:
            raise ValueError("Duration cannot be negative")

    @classmethod
    def from_seconds(cls, seconds: float) -> Duration:
        return cls(seconds=seconds)

    @classmethod
    def from_milliseconds(cls, milliseconds: int) -> Duration:
        return cls(seconds=milliseconds / 1000.0)

    @classmethod
    def from_minutes(cls, minutes: float) -> Duration:
        return cls(seconds=minutes * 60.0)

    def total_seconds(self) -> float:
        return self.seconds

    def __add__(self, other: Duration) -> Duration:
        return Duration(self.seconds + other.seconds)

    def __lt__(self, other: Duration) -> bool:
        return self.seconds < other.seconds

    def __le__(self, other: Duration) -> bool:
        return self.seconds <= other.seconds

    def __gt__(self, other: Duration) -> bool:
        return self.seconds > other.seconds

    def __ge__(self, other: Duration) -> bool:
        return self.seconds >= other.seconds

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Duration):
            return self.seconds == other.seconds
        return False

    # Class constants
    ZERO = None  # Will be set after class definition


@dataclass(frozen=True)
class CommandParameters:
    """Type-safe command parameter container."""

    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> CommandParameters:
        return cls()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def with_parameter(self, key: str, value: Any) -> CommandParameters:
        new_data = self.data.copy()
        new_data[key] = value
        return CommandParameters(new_data)


# Initialize Duration constants after class definition
Duration.ZERO = Duration(0.0)


class MacroCommand(Protocol):
    """Protocol defining the interface for all macro commands."""

    def execute(self, context: ExecutionContext) -> CommandResult:
        """Execute the command in the given context."""
        ...

    def validate(self) -> bool:
        """Validate command parameters and state."""
        ...

    def get_dependencies(self) -> list[CommandId]:
        """Get list of command dependencies."""
        ...

    def get_required_permissions(self) -> frozenset[Permission]:
        """Get required permissions for this command."""
        ...


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable execution context for macro commands."""

    permissions: frozenset[Permission]
    timeout: Duration
    variables: dict[VariableName, str] = field(default_factory=dict)
    execution_id: ExecutionToken = field(
        default_factory=lambda: ExecutionToken(str(uuid.uuid4())),
    )
    created_at: datetime = field(default_factory=datetime.now)

    def has_permission(self, permission: Permission) -> bool:
        """Check if context has the required permission."""
        return permission in self.permissions

    def has_permissions(self, permissions: frozenset[Permission]) -> bool:
        """Check if context has all required permissions."""
        return permissions.issubset(self.permissions)

    def get_variable(self, name: VariableName) -> str | None:
        """Get variable value from context."""
        return self.variables.get(name)

    def with_variable(self, name: VariableName, value: str) -> ExecutionContext:
        """Create new context with added variable."""
        new_vars = self.variables.copy()
        new_vars[name] = value
        return ExecutionContext(
            permissions=self.permissions,
            timeout=self.timeout,
            variables=new_vars,
            execution_id=self.execution_id,
            created_at=self.created_at,
        )

    @classmethod
    def create_test_context(
        cls,
        permissions: frozenset[Permission] | None = None,
        timeout: Duration | None = None,
    ) -> ExecutionContext:
        """Create a test execution context with default values."""
        return cls(
            permissions=permissions
            if permissions is not None
            else frozenset([Permission.TEXT_INPUT, Permission.SYSTEM_SOUND]),
            timeout=timeout or Duration.from_seconds(30),
        )

    @classmethod
    def default(cls) -> ExecutionContext:
        """Create default execution context."""
        return cls.create_test_context()

    async def info(self, message: str) -> None:
        """Mock method for test compatibility - logs informational message."""
        pass  # No-op for test compatibility

    async def error(self, message: str) -> None:
        """Mock method for test compatibility - logs error message."""
        pass  # No-op for test compatibility


@dataclass(frozen=True)
class CommandResult:
    """Immutable result from command execution."""

    success: bool
    output: str | None = None
    error_message: str | None = None
    execution_time: Duration | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        output: str | None = None,
        execution_time: Duration | None = None,
        **metadata: Any,
    ) -> CommandResult:
        """Create a successful command result."""
        return cls(
            success=True,
            output=output,
            execution_time=execution_time,
            metadata=metadata,
        )

    @classmethod
    def failure_result(
        cls,
        error_message: str,
        execution_time: Duration | None = None,
        **metadata: Any,
    ) -> CommandResult:
        """Create a failed command result."""
        return cls(
            success=False,
            error_message=error_message,
            execution_time=execution_time,
            metadata=metadata,
        )


@dataclass(frozen=True)
class MacroDefinition:
    """Complete macro definition with metadata."""

    macro_id: MacroId
    name: str
    commands: list[MacroCommand]
    enabled: bool = True
    group_id: GroupId | None = None
    description: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def is_valid(self) -> bool:
        """Validate macro definition."""
        if not self.name or not self.commands:
            return False
        return all(cmd.validate() for cmd in self.commands)

    @classmethod
    def create_test_macro(
        cls,
        name: str,
        commands: list[MacroCommand],
    ) -> MacroDefinition:
        """Create a test macro definition."""
        return cls(macro_id=MacroId(str(uuid.uuid4())), name=name, commands=commands)


@dataclass(frozen=True)
class ExecutionResult:
    """Complete macro execution result."""

    execution_token: ExecutionToken
    macro_id: MacroId
    status: ExecutionStatus
    started_at: datetime
    completed_at: datetime | None = None
    total_duration: Duration | None = None
    command_results: list[CommandResult] = field(default_factory=list)
    error_details: str | None = None

    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.COMPLETED and all(
            r.success for r in self.command_results
        )

    def has_error_info(self) -> bool:
        """Check if error information is available."""
        return self.error_details is not None or any(
            not r.success for r in self.command_results
        )


@dataclass(frozen=True)
class ValidationResult:
    """Result of validation operations."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, **metadata: Any) -> ValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True, metadata=metadata)

    @classmethod
    def failure(cls, errors: list[str], **metadata: Any) -> ValidationResult:
        """Create a failed validation result."""
        return cls(is_valid=False, errors=errors, metadata=metadata)

    def add_error(self, error: str) -> ValidationResult:
        """Add an error to the validation result."""
        new_errors = self.errors + [error]
        return ValidationResult(
            is_valid=False,
            errors=new_errors,
            warnings=self.warnings,
            metadata=self.metadata,
        )


# Type variables for generic types
T = TypeVar("T")
CommandT = TypeVar("CommandT", bound=MacroCommand)
