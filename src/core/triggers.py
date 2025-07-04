"""
Advanced trigger system for event-driven automation.

This module implements sophisticated trigger types that enable Keyboard Maestro macros
to respond automatically to environmental changes including time events, file system
modifications, application lifecycle, and system state changes.

Security: All trigger configurations include comprehensive validation and resource limits.
Performance: Efficient event monitoring with debouncing and throttling capabilities.
Type Safety: Complete branded type system with contract-driven development.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Set
from enum import Enum
from datetime import datetime, timedelta
import uuid
import os
import re

from src.core.either import Either
from src.core.errors import ValidationError, SecurityError
from src.core.contracts import require, ensure


class TriggerType(Enum):
    """Comprehensive trigger types for sophisticated automation."""
    # Time-based triggers
    TIME_SCHEDULED = "time_scheduled"
    TIME_RECURRING = "time_recurring"
    TIME_CRON = "time_cron"
    TIME_COUNTDOWN = "time_countdown"
    
    # File system triggers
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    FOLDER_CHANGED = "folder_changed"
    
    # System event triggers
    APP_LAUNCHED = "app_launched"
    APP_QUIT = "app_quit"
    APP_ACTIVATED = "app_activated"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_WAKE = "system_wake"
    SYSTEM_SLEEP = "system_sleep"
    
    # User activity triggers
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_IDLE = "user_idle"
    USER_ACTIVE = "user_active"
    USER_LOCKED = "user_locked"
    USER_UNLOCKED = "user_unlocked"
    
    # Network triggers
    NETWORK_CONNECTED = "network_connected"
    NETWORK_DISCONNECTED = "network_disconnected"
    NETWORK_CHANGED = "network_changed"
    VPN_CONNECTED = "vpn_connected"
    VPN_DISCONNECTED = "vpn_disconnected"
    
    # Hardware triggers
    BATTERY_LOW = "battery_low"
    BATTERY_CHARGED = "battery_charged"
    BATTERY_CHARGING = "battery_charging"
    DEVICE_MOUNTED = "device_mounted"
    DEVICE_UNMOUNTED = "device_unmounted"
    
    # Composite triggers
    COMPOSITE_AND = "composite_and"
    COMPOSITE_OR = "composite_or"
    SEQUENCE = "sequence"


@dataclass(frozen=True)
class TriggerSpec:
    """Type-safe trigger specification."""
    trigger_id: str
    trigger_type: TriggerType
    config: Dict[str, Any]
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    timeout_seconds: int = 30
    max_executions: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Contract validation for trigger specification."""
        if not (0 < self.timeout_seconds <= 300):
            raise ValueError("Timeout must be between 1 and 300 seconds")
        if not (-10 <= self.priority <= 10):
            raise ValueError("Priority must be between -10 and 10")
        if self.max_executions is not None and self.max_executions <= 0:
            raise ValueError("Max executions must be positive")


@dataclass(frozen=True)
class TimeTriggerConfig:
    """Configuration for time-based triggers."""
    schedule_time: Optional[datetime] = None
    recurring_interval: Optional[timedelta] = None
    recurring_pattern: Optional[str] = None  # cron-style pattern
    timezone: str = "local"
    
    def __post_init__(self):
        """Validate time trigger configuration."""
        has_schedule = self.schedule_time is not None
        has_interval = self.recurring_interval is not None
        has_pattern = self.recurring_pattern is not None
        
        if not (has_schedule or has_interval or has_pattern):
            raise ValueError("At least one time specification must be provided")
        
        if self.recurring_interval and self.recurring_interval.total_seconds() < 1:
            raise ValueError("Recurring interval must be at least 1 second")


@dataclass(frozen=True)
class FileTriggerConfig:
    """Configuration for file-based triggers."""
    watch_path: str
    recursive: bool = False
    file_pattern: Optional[str] = None
    ignore_hidden: bool = True
    debounce_seconds: float = 1.0
    
    def __post_init__(self):
        """Validate file trigger configuration."""
        if not self.watch_path or len(self.watch_path.strip()) == 0:
            raise ValueError("Watch path cannot be empty")
        if not (0.1 <= self.debounce_seconds <= 10.0):
            raise ValueError("Debounce must be between 0.1 and 10.0 seconds")


@dataclass(frozen=True)
class SystemTriggerConfig:
    """Configuration for system-based triggers."""
    app_bundle_id: Optional[str] = None
    app_name: Optional[str] = None
    network_interface: Optional[str] = None
    battery_threshold: Optional[int] = None
    idle_threshold_seconds: Optional[int] = None
    
    def __post_init__(self):
        """Validate system trigger configuration."""
        if self.battery_threshold is not None:
            if not (0 <= self.battery_threshold <= 100):
                raise ValueError("Battery threshold must be between 0 and 100")
        
        if self.idle_threshold_seconds is not None:
            if self.idle_threshold_seconds <= 0:
                raise ValueError("Idle threshold must be positive")


class TriggerValidator:
    """Security-first trigger validation."""
    
    FORBIDDEN_PATHS = [
        "/System", "/usr/bin", "/usr/sbin", "/bin", "/sbin",
        "/private/etc", "/Library/Keychains", "/Users/*/Library/Keychains",
        "/private/var/db", "/private/var/root"
    ]
    
    @staticmethod
    def validate_file_path(path: str) -> Either[SecurityError, str]:
        """Prevent unauthorized file system access."""
        try:
            # Resolve path to absolute
            abs_path = os.path.abspath(os.path.expanduser(path))
            
            # Check against forbidden paths
            for forbidden in TriggerValidator.FORBIDDEN_PATHS:
                if "*" in forbidden:
                    # Handle wildcard patterns
                    pattern = forbidden.replace("*", ".*")
                    if re.match(pattern, abs_path):
                        return Either.left(SecurityError(
                            "PATH_ACCESS_DENIED",
                            f"Access denied to protected path matching pattern: {forbidden}"
                        ))
                else:
                    if abs_path.startswith(forbidden):
                        return Either.left(SecurityError(
                            "PATH_ACCESS_DENIED",
                            f"Access denied to protected path: {forbidden}"
                        ))
            
            # Check for directory traversal attempts
            if ".." in path or path.startswith("~/../"):
                return Either.left(SecurityError(
                    "DIRECTORY_TRAVERSAL",
                    "Directory traversal attempts are not allowed"
                ))
            
            return Either.right(abs_path)
            
        except Exception as e:
            return Either.left(SecurityError(
                "PATH_VALIDATION_ERROR",
                f"Failed to validate path: {str(e)}"
            ))
    
    @staticmethod
    def validate_app_identifier(app_id: str) -> Either[SecurityError, str]:
        """Validate application bundle identifier."""
        if not app_id or len(app_id.strip()) == 0:
            return Either.left(SecurityError(
                "EMPTY_APP_ID",
                "Application identifier cannot be empty"
            ))
        
        clean_id = app_id.strip()
        
        # Check for malicious patterns
        dangerous_patterns = ["..", "/", "\\", "<", ">", "|", "&", ";", "`", "$"]
        for pattern in dangerous_patterns:
            if pattern in clean_id:
                return Either.left(SecurityError(
                    "DANGEROUS_APP_ID",
                    f"Application identifier contains dangerous pattern: {pattern}"
                ))
        
        # Validate format (standard bundle identifier pattern)
        if not re.match(r'^[a-zA-Z0-9._-]+$', clean_id):
            return Either.left(SecurityError(
                "INVALID_APP_ID_FORMAT",
                "Application identifier contains invalid characters"
            ))
        
        # Check length limits
        if len(clean_id) > 255:
            return Either.left(SecurityError(
                "APP_ID_TOO_LONG",
                "Application identifier too long (max 255 characters)"
            ))
        
        return Either.right(clean_id)
    
    @staticmethod
    def validate_cron_pattern(pattern: str) -> Either[SecurityError, str]:
        """Validate cron pattern for security and correctness."""
        if not pattern or len(pattern.strip()) == 0:
            return Either.left(SecurityError(
                "EMPTY_CRON_PATTERN",
                "Cron pattern cannot be empty"
            ))
        
        clean_pattern = pattern.strip()
        
        # Basic cron pattern validation (simplified)
        # Format: minute hour day month weekday
        parts = clean_pattern.split()
        if len(parts) != 5:
            return Either.left(SecurityError(
                "INVALID_CRON_FORMAT",
                "Cron pattern must have exactly 5 fields"
            ))
        
        # Check for dangerous characters
        allowed_chars = set("0123456789-,/*")
        for part in parts:
            if not all(c in allowed_chars for c in part):
                return Either.left(SecurityError(
                    "INVALID_CRON_CHARACTERS",
                    "Cron pattern contains invalid characters"
                ))
        
        return Either.right(clean_pattern)
    
    @staticmethod
    def validate_resource_limits(trigger_spec: TriggerSpec) -> Either[SecurityError, None]:
        """Prevent resource exhaustion from trigger monitoring."""
        # Check timeout limits
        if trigger_spec.timeout_seconds > 300:
            return Either.left(SecurityError(
                "TIMEOUT_TOO_LONG",
                "Timeout exceeds maximum allowed (300 seconds)"
            ))
        
        # Check execution limits
        if trigger_spec.max_executions is not None and trigger_spec.max_executions > 10000:
            return Either.left(SecurityError(
                "MAX_EXECUTIONS_TOO_HIGH",
                "Maximum executions exceeds limit (10000)"
            ))
        
        # Check file monitoring limits
        file_trigger_types = {
            TriggerType.FILE_CREATED, TriggerType.FILE_MODIFIED,
            TriggerType.FILE_DELETED, TriggerType.FOLDER_CHANGED
        }
        
        if trigger_spec.trigger_type in file_trigger_types:
            config = trigger_spec.config
            watch_path = config.get("watch_path", "")
            
            if config.get("recursive", False):
                # Recursive monitoring can be resource intensive
                sensitive_paths = ["/", "/Users", "/Applications", "/System", "/Library"]
                for sensitive in sensitive_paths:
                    # Exact match or starts with path + "/"
                    if watch_path == sensitive or watch_path.startswith(sensitive + "/"):
                        return Either.left(SecurityError(
                            "RECURSIVE_MONITORING_DENIED",
                            f"Recursive monitoring of {sensitive} is not allowed"
                        ))
        
        return Either.right(None)


class TriggerBuilder:
    """Fluent API for building advanced trigger specifications."""
    
    def __init__(self):
        self._trigger_id: str = str(uuid.uuid4())
        self._trigger_type: Optional[TriggerType] = None
        self._config: Dict[str, Any] = {}
        self._conditions: List[Dict[str, Any]] = []
        self._enabled: bool = True
        self._priority: int = 0
        self._timeout: int = 30
        self._max_executions: Optional[int] = None
        self._metadata: Dict[str, Any] = {}
    
    def scheduled_at(self, when: datetime, timezone: str = "local") -> 'TriggerBuilder':
        """Create a scheduled time trigger."""
        self._trigger_type = TriggerType.TIME_SCHEDULED
        self._config = {"schedule_time": when, "timezone": timezone}
        return self
    
    def recurring_every(self, interval: timedelta) -> 'TriggerBuilder':
        """Create a recurring time trigger."""
        self._trigger_type = TriggerType.TIME_RECURRING
        self._config = {"recurring_interval": interval}
        return self
    
    def cron_pattern(self, pattern: str) -> 'TriggerBuilder':
        """Create a cron-style recurring trigger."""
        validation_result = TriggerValidator.validate_cron_pattern(pattern)
        if validation_result.is_left():
            raise ValueError(f"Invalid cron pattern: {validation_result.get_left().message}")
        
        self._trigger_type = TriggerType.TIME_RECURRING
        self._config = {"recurring_pattern": validation_result.get_right()}
        return self
    
    def when_file_created(self, path: str, recursive: bool = False) -> 'TriggerBuilder':
        """Create a file creation trigger."""
        path_validation = TriggerValidator.validate_file_path(path)
        if path_validation.is_left():
            raise ValueError(f"Invalid file path: {path_validation.get_left().message}")
        
        self._trigger_type = TriggerType.FILE_CREATED
        self._config = {"watch_path": path_validation.get_right(), "recursive": recursive}
        return self
    
    def when_file_modified(self, path: str, pattern: Optional[str] = None) -> 'TriggerBuilder':
        """Create a file modification trigger."""
        path_validation = TriggerValidator.validate_file_path(path)
        if path_validation.is_left():
            raise ValueError(f"Invalid file path: {path_validation.get_left().message}")
        
        self._trigger_type = TriggerType.FILE_MODIFIED
        config = {"watch_path": path_validation.get_right()}
        if pattern:
            config["file_pattern"] = pattern
        self._config = config
        return self
    
    def when_app_launches(self, app_identifier: str) -> 'TriggerBuilder':
        """Create an application launch trigger."""
        app_validation = TriggerValidator.validate_app_identifier(app_identifier)
        if app_validation.is_left():
            raise ValueError(f"Invalid app identifier: {app_validation.get_left().message}")
        
        self._trigger_type = TriggerType.APP_LAUNCHED
        self._config = {"app_bundle_id": app_validation.get_right()}
        return self
    
    def when_app_quits(self, app_identifier: str) -> 'TriggerBuilder':
        """Create an application quit trigger."""
        app_validation = TriggerValidator.validate_app_identifier(app_identifier)
        if app_validation.is_left():
            raise ValueError(f"Invalid app identifier: {app_validation.get_left().message}")
        
        self._trigger_type = TriggerType.APP_QUIT
        self._config = {"app_bundle_id": app_validation.get_right()}
        return self
    
    def when_user_idle(self, threshold_seconds: int) -> 'TriggerBuilder':
        """Create a user idle trigger."""
        if threshold_seconds <= 0:
            raise ValueError("Idle threshold must be positive")
        
        self._trigger_type = TriggerType.USER_IDLE
        self._config = {"idle_threshold_seconds": threshold_seconds}
        return self
    
    def when_battery_low(self, threshold_percent: int) -> 'TriggerBuilder':
        """Create a battery low trigger."""
        if not (0 <= threshold_percent <= 100):
            raise ValueError("Battery threshold must be between 0 and 100")
        
        self._trigger_type = TriggerType.BATTERY_LOW
        self._config = {"battery_threshold": threshold_percent}
        return self
    
    def when_network_connected(self, interface: Optional[str] = None) -> 'TriggerBuilder':
        """Create a network connection trigger."""
        self._trigger_type = TriggerType.NETWORK_CONNECTED
        config = {}
        if interface:
            config["network_interface"] = interface
        self._config = config
        return self
    
    def when_system_wakes(self) -> 'TriggerBuilder':
        """Create a system wake trigger."""
        self._trigger_type = TriggerType.SYSTEM_WAKE
        self._config = {}
        return self
    
    def when_system_sleeps(self) -> 'TriggerBuilder':
        """Create a system sleep trigger."""
        self._trigger_type = TriggerType.SYSTEM_SLEEP
        self._config = {}
        return self
    
    def when_app_activated(self, app_identifier: str) -> 'TriggerBuilder':
        """Create an application activation trigger."""
        app_validation = TriggerValidator.validate_app_identifier(app_identifier)
        if app_validation.is_left():
            raise ValueError(f"Invalid app identifier: {app_validation.get_left().message}")
        
        self._trigger_type = TriggerType.APP_ACTIVATED
        self._config = {"app_bundle_id": app_validation.get_right()}
        return self
    
    def when_user_locked(self) -> 'TriggerBuilder':
        """Create a user screen lock trigger."""
        self._trigger_type = TriggerType.USER_LOCKED
        self._config = {}
        return self
    
    def when_user_unlocked(self) -> 'TriggerBuilder':
        """Create a user screen unlock trigger."""
        self._trigger_type = TriggerType.USER_UNLOCKED
        self._config = {}
        return self
    
    def when_vpn_connected(self) -> 'TriggerBuilder':
        """Create a VPN connection trigger."""
        self._trigger_type = TriggerType.VPN_CONNECTED
        self._config = {}
        return self
    
    def when_device_mounted(self, device_pattern: Optional[str] = None) -> 'TriggerBuilder':
        """Create a device mount trigger."""
        self._trigger_type = TriggerType.DEVICE_MOUNTED
        config = {}
        if device_pattern:
            config["device_pattern"] = device_pattern
        self._config = config
        return self
    
    def countdown_timer(self, duration_seconds: int) -> 'TriggerBuilder':
        """Create a countdown timer trigger."""
        if duration_seconds <= 0:
            raise ValueError("Countdown duration must be positive")
        
        self._trigger_type = TriggerType.TIME_COUNTDOWN
        self._config = {"duration_seconds": duration_seconds}
        return self
    
    def when_file_moved(self, path: str) -> 'TriggerBuilder':
        """Create a file move/rename trigger."""
        path_validation = TriggerValidator.validate_file_path(path)
        if path_validation.is_left():
            raise ValueError(f"Invalid file path: {path_validation.get_left().message}")
        
        self._trigger_type = TriggerType.FILE_MOVED
        self._config = {"watch_path": path_validation.get_right()}
        return self
    
    def with_condition(self, condition: Dict[str, Any]) -> 'TriggerBuilder':
        """Add conditional logic to trigger."""
        self._conditions.append(condition)
        return self
    
    def with_priority(self, priority: int) -> 'TriggerBuilder':
        """Set execution priority."""
        if not (-10 <= priority <= 10):
            raise ValueError("Priority must be between -10 and 10")
        self._priority = priority
        return self
    
    def with_timeout(self, timeout_seconds: int) -> 'TriggerBuilder':
        """Set execution timeout."""
        if not (1 <= timeout_seconds <= 300):
            raise ValueError("Timeout must be between 1 and 300 seconds")
        self._timeout = timeout_seconds
        return self
    
    def limit_executions(self, max_count: int) -> 'TriggerBuilder':
        """Limit maximum executions."""
        if max_count <= 0:
            raise ValueError("Max executions must be positive")
        self._max_executions = max_count
        return self
    
    def enabled(self, is_enabled: bool) -> 'TriggerBuilder':
        """Set trigger enabled state."""
        self._enabled = is_enabled
        return self
    
    def with_metadata(self, **metadata) -> 'TriggerBuilder':
        """Add metadata to trigger."""
        self._metadata.update(metadata)
        return self
    
    def build(self) -> Either[ValidationError, TriggerSpec]:
        """Build and validate the trigger specification."""
        if self._trigger_type is None:
            return Either.left(ValidationError(
                field_name="trigger_type",
                value="None",
                constraint="Trigger type must be specified"
            ))
        
        try:
            # Add creation metadata
            self._metadata.update({
                "created_at": datetime.now().isoformat(),
                "builder_version": "1.0"
            })
            
            trigger_spec = TriggerSpec(
                trigger_id=self._trigger_id,
                trigger_type=self._trigger_type,
                config=self._config,
                conditions=self._conditions,
                enabled=self._enabled,
                priority=self._priority,
                timeout_seconds=self._timeout,
                max_executions=self._max_executions,
                metadata=self._metadata
            )
            
            # Security validation
            security_result = TriggerValidator.validate_resource_limits(trigger_spec)
            if security_result.is_left():
                return Either.left(ValidationError(
                    field_name="resource_limits",
                    value=str(trigger_spec),
                    constraint=security_result.get_left().message
                ))
            
            return Either.right(trigger_spec)
            
        except ValueError as e:
            return Either.left(ValidationError(
                field_name="trigger_spec",
                value="invalid",
                constraint=str(e)
            ))


# Convenience functions for common trigger patterns
def create_daily_trigger(hour: int, minute: int = 0) -> TriggerBuilder:
    """Create a daily recurring trigger."""
    return TriggerBuilder().cron_pattern(f"{minute} {hour} * * *")


def create_file_watcher(directory: str, file_pattern: Optional[str] = None) -> TriggerBuilder:
    """Create a file monitoring trigger."""
    builder = TriggerBuilder().when_file_modified(directory, file_pattern)
    return builder


def create_app_lifecycle_trigger(app_id: str, on_launch: bool = True) -> TriggerBuilder:
    """Create an application lifecycle trigger."""
    if on_launch:
        return TriggerBuilder().when_app_launches(app_id)
    else:
        return TriggerBuilder().when_app_quits(app_id)