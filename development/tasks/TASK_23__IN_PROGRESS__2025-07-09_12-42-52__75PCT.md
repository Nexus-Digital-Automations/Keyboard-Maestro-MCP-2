# TASK_23: km_create_trigger_advanced - Time-based, File, System Triggers

**Created By**: Agent_ADDER+ (Protocol Gap Analysis) | **Priority**: HIGH | **Duration**: 4 hours
**Technique Focus**: Event-Driven Architecture + Design by Contract + Functional Programming
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: IN_PROGRESS
**Assigned**: Agent_ADDER+
**Dependencies**: TASK_21 (km_add_condition), TASK_22 (km_control_flow)
**Blocking**: Advanced automation workflows requiring event-driven execution

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Advanced trigger specification
- [ ] **KM Documentation**: development/protocols/KM_MCP.md - Trigger types and configuration
- [ ] **Foundation**: development/tasks/TASK_21.md - Conditional logic for trigger conditions
- [ ] **Event System**: src/integration/events.py - Event-driven architecture patterns
- [ ] **Testing Framework**: tests/TESTING.md - Property-based testing for event systems

## 🎯 Problem Analysis
**Classification**: Missing Critical Functionality
**Gap Identified**: Only basic macro execution - no event-driven automation triggers
**Impact**: AI limited to manual macro execution - cannot create automated, responsive workflows

<thinking>
Root Cause Analysis:
1. Current implementation only supports manual macro execution
2. Missing event-driven triggers that make automation truly automated
3. Keyboard Maestro has sophisticated trigger system but no MCP exposure
4. Without advanced triggers, automation remains reactive rather than proactive
5. Time-based, file-based, and system-based triggers are essential for intelligent automation
6. This enables macros to respond to environmental changes automatically
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **Trigger type system**: Define branded types for all trigger categories
- [ ] **Event monitoring**: File system watchers, time schedulers, system monitors
- [ ] **Validation framework**: Security boundaries and resource usage limits

### Phase 2: Core Trigger Types
- [ ] **Time triggers**: Scheduled execution, recurring events, countdown timers
- [ ] **File triggers**: File creation, modification, deletion, folder monitoring
- [ ] **System triggers**: Application launch/quit, network changes, hardware events
- [ ] **User triggers**: Login/logout, idle detection, hotkey combinations

### Phase 3: Advanced Trigger Logic
- [ ] **Conditional triggers**: Triggers with conditional logic integration
- [ ] **Composite triggers**: Multiple trigger conditions with AND/OR logic
- [ ] **Trigger chains**: Sequential trigger dependencies and workflows
- [ ] **Trigger throttling**: Rate limiting and debouncing for high-frequency events

### Phase 4: Integration & Security
- [ ] **AppleScript generation**: Safe trigger XML with proper event handling
- [ ] **Resource management**: Prevent resource exhaustion from trigger monitoring
- [ ] **Property-based tests**: Hypothesis validation for all trigger scenarios
- [ ] **TESTING.md update**: Advanced trigger test coverage and monitoring

## 🔧 Implementation Files & Specifications
```
src/server/tools/advanced_trigger_tools.py   # Main advanced trigger tool implementation
src/core/triggers.py                         # Trigger type definitions and builders
src/integration/km_triggers.py               # KM-specific trigger integration
src/monitoring/trigger_monitor.py            # Event monitoring and detection
tests/tools/test_advanced_trigger_tools.py   # Unit and integration tests
tests/property_tests/test_triggers.py        # Property-based trigger validation
```

### km_create_trigger_advanced Tool Specification
```python
@mcp.tool()
async def km_create_trigger_advanced(
    macro_identifier: str,                    # Target macro (name or UUID)
    trigger_type: str,                       # time|file|system|user|composite
    trigger_config: Dict[str, Any],          # Trigger-specific configuration
    conditions: Optional[List[Dict]] = None, # Additional conditions (TASK_21 integration)
    enabled: bool = True,                    # Initial enabled state
    priority: int = 0,                       # Execution priority
    timeout_seconds: int = 30,               # Trigger timeout
    max_executions: Optional[int] = None,    # Execution limit
    ctx = None
) -> Dict[str, Any]:
```

### Advanced Trigger Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Set
from enum import Enum
from datetime import datetime, timedelta

class TriggerType(Enum):
    """Supported advanced trigger types."""
    TIME_SCHEDULED = "time_scheduled"
    TIME_RECURRING = "time_recurring"
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FOLDER_CHANGED = "folder_changed"
    APP_LAUNCHED = "app_launched"
    APP_QUIT = "app_quit"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_IDLE = "user_idle"
    USER_ACTIVE = "user_active"
    NETWORK_CONNECTED = "network_connected"
    NETWORK_DISCONNECTED = "network_disconnected"
    BATTERY_LOW = "battery_low"
    BATTERY_CHARGED = "battery_charged"
    COMPOSITE = "composite"

@dataclass(frozen=True)
class TriggerSpec:
    """Type-safe trigger specification."""
    trigger_type: TriggerType
    config: Dict[str, Any]
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    timeout_seconds: int = 30
    max_executions: Optional[int] = None
    
    @require(lambda self: self.timeout_seconds > 0 and self.timeout_seconds <= 300)
    @require(lambda self: self.priority >= -10 and self.priority <= 10)
    @require(lambda self: self.max_executions is None or self.max_executions > 0)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class TimeTriggerConfig:
    """Configuration for time-based triggers."""
    schedule_time: Optional[datetime] = None
    recurring_interval: Optional[timedelta] = None
    recurring_pattern: Optional[str] = None  # cron-style pattern
    timezone: str = "local"
    
    @require(lambda self: self.schedule_time is not None or self.recurring_interval is not None or self.recurring_pattern is not None)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class FileTriggerConfig:
    """Configuration for file-based triggers."""
    watch_path: str
    recursive: bool = False
    file_pattern: Optional[str] = None
    ignore_hidden: bool = True
    debounce_seconds: float = 1.0
    
    @require(lambda self: len(self.watch_path) > 0)
    @require(lambda self: self.debounce_seconds >= 0.1 and self.debounce_seconds <= 10.0)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class SystemTriggerConfig:
    """Configuration for system-based triggers."""
    app_bundle_id: Optional[str] = None
    app_name: Optional[str] = None
    network_interface: Optional[str] = None
    battery_threshold: Optional[int] = None
    idle_threshold_seconds: Optional[int] = None
    
    @require(lambda self: self.battery_threshold is None or (0 <= self.battery_threshold <= 100))
    @require(lambda self: self.idle_threshold_seconds is None or self.idle_threshold_seconds > 0)
    def __post_init__(self):
        pass

class TriggerBuilder:
    """Fluent API for building advanced trigger specifications."""
    
    def __init__(self):
        self._trigger_type: Optional[TriggerType] = None
        self._config: Dict[str, Any] = {}
        self._conditions: List[Dict[str, Any]] = []
        self._enabled: bool = True
        self._priority: int = 0
        self._timeout: int = 30
        self._max_executions: Optional[int] = None
    
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
        self._trigger_type = TriggerType.TIME_RECURRING
        self._config = {"recurring_pattern": pattern}
        return self
    
    def when_file_created(self, path: str, recursive: bool = False) -> 'TriggerBuilder':
        """Create a file creation trigger."""
        self._trigger_type = TriggerType.FILE_CREATED
        self._config = {"watch_path": path, "recursive": recursive}
        return self
    
    def when_file_modified(self, path: str, pattern: Optional[str] = None) -> 'TriggerBuilder':
        """Create a file modification trigger."""
        self._trigger_type = TriggerType.FILE_MODIFIED
        self._config = {"watch_path": path, "file_pattern": pattern}
        return self
    
    def when_app_launches(self, app_identifier: str) -> 'TriggerBuilder':
        """Create an application launch trigger."""
        self._trigger_type = TriggerType.APP_LAUNCHED
        self._config = {"app_bundle_id": app_identifier}
        return self
    
    def when_user_idle(self, threshold_seconds: int) -> 'TriggerBuilder':
        """Create a user idle trigger."""
        self._trigger_type = TriggerType.USER_IDLE
        self._config = {"idle_threshold_seconds": threshold_seconds}
        return self
    
    def when_battery_low(self, threshold_percent: int) -> 'TriggerBuilder':
        """Create a battery low trigger."""
        self._trigger_type = TriggerType.BATTERY_LOW
        self._config = {"battery_threshold": threshold_percent}
        return self
    
    def with_condition(self, condition: Dict[str, Any]) -> 'TriggerBuilder':
        """Add conditional logic to trigger."""
        self._conditions.append(condition)
        return self
    
    def with_priority(self, priority: int) -> 'TriggerBuilder':
        """Set execution priority."""
        self._priority = priority
        return self
    
    def with_timeout(self, timeout_seconds: int) -> 'TriggerBuilder':
        """Set execution timeout."""
        self._timeout = timeout_seconds
        return self
    
    def limit_executions(self, max_count: int) -> 'TriggerBuilder':
        """Limit maximum executions."""
        self._max_executions = max_count
        return self
    
    def build(self) -> TriggerSpec:
        """Build the trigger specification."""
        if self._trigger_type is None:
            raise ValueError("Trigger type must be specified")
        
        return TriggerSpec(
            trigger_type=self._trigger_type,
            config=self._config,
            conditions=self._conditions,
            enabled=self._enabled,
            priority=self._priority,
            timeout_seconds=self._timeout,
            max_executions=self._max_executions
        )
```

## 🔒 Security Implementation
```python
class TriggerValidator:
    """Security-first trigger validation."""
    
    @staticmethod
    def validate_file_path(path: str) -> Either[SecurityError, str]:
        """Prevent unauthorized file system access."""
        # Resolve path to absolute
        abs_path = os.path.abspath(path)
        
        # Check against forbidden paths
        forbidden_paths = [
            "/System", "/usr/bin", "/usr/sbin", "/bin", "/sbin",
            "/private/etc", "/Library/Keychains", "/Users/*/Library/Keychains"
        ]
        
        for forbidden in forbidden_paths:
            if abs_path.startswith(forbidden):
                return Either.left(SecurityError(f"Access denied to protected path: {forbidden}"))
        
        return Either.right(abs_path)
    
    @staticmethod
    def validate_app_identifier(app_id: str) -> Either[SecurityError, str]:
        """Validate application bundle identifier."""
        # Check for malicious patterns
        if ".." in app_id or "/" in app_id or app_id.startswith("."):
            return Either.left(SecurityError("Invalid application identifier"))
        
        # Validate format
        if not re.match(r'^[a-zA-Z0-9._-]+$', app_id):
            return Either.left(SecurityError("Invalid application identifier format"))
        
        return Either.right(app_id)
    
    @staticmethod
    def validate_resource_limits(trigger_spec: TriggerSpec) -> Either[SecurityError, None]:
        """Prevent resource exhaustion from trigger monitoring."""
        # Check timeout limits
        if trigger_spec.timeout_seconds > 300:
            return Either.left(SecurityError("Timeout too long"))
        
        # Check execution limits
        if trigger_spec.max_executions is not None and trigger_spec.max_executions > 10000:
            return Either.left(SecurityError("Max executions too high"))
        
        # Check file monitoring limits
        if trigger_spec.trigger_type in [TriggerType.FILE_CREATED, TriggerType.FILE_MODIFIED]:
            config = trigger_spec.config
            if config.get("recursive", False):
                # Recursive monitoring can be resource intensive
                watch_path = config.get("watch_path", "")
                if watch_path in ["/", "/Users", "/Applications"]:
                    return Either.left(SecurityError("Recursive monitoring of system directories not allowed"))
        
        return Either.right(None)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st
from datetime import datetime, timedelta

@given(st.integers(min_value=1, max_value=86400))
def test_time_trigger_properties(seconds):
    """Property: Time triggers should handle all valid time intervals."""
    interval = timedelta(seconds=seconds)
    trigger = TriggerBuilder().recurring_every(interval).build()
    assert trigger.trigger_type == TriggerType.TIME_RECURRING
    assert trigger.config["recurring_interval"] == interval

@given(st.text(min_size=1, max_size=100))
def test_file_path_security_properties(path):
    """Property: File paths should be validated for security."""
    validation_result = TriggerValidator.validate_file_path(path)
    if validation_result.is_left():
        error = validation_result.get_left()
        assert error.code == "SECURITY_ERROR"
        assert "protected path" in error.message or "Access denied" in error.message

@given(st.integers(min_value=1, max_value=3600))
def test_idle_threshold_properties(threshold):
    """Property: Idle thresholds should be within reasonable bounds."""
    trigger = TriggerBuilder().when_user_idle(threshold).build()
    assert trigger.config["idle_threshold_seconds"] == threshold
    assert trigger.trigger_type == TriggerType.USER_IDLE
```

## 🏗️ Modularity Strategy
- **advanced_trigger_tools.py**: Main MCP tool interface (<250 lines)
- **triggers.py**: Type definitions, builders, and validation (<300 lines)
- **km_triggers.py**: KM integration and XML generation (<250 lines)
- **trigger_monitor.py**: Event monitoring and detection (<200 lines)

## 📋 Advanced Trigger Examples

### Scheduled Time Trigger
```python
# Example: Daily backup at 2 AM
trigger = TriggerBuilder() \
    .scheduled_at(datetime(2024, 1, 1, 2, 0)) \
    .with_condition({
        "type": "system",
        "operator": "equals",
        "operand": "weekday"
    }) \
    .build()
```

### File System Monitoring
```python
# Example: Process new downloads
trigger = TriggerBuilder() \
    .when_file_created("~/Downloads") \
    .with_condition({
        "type": "text",
        "operator": "regex",
        "operand": r"\.(pdf|doc|docx)$"
    }) \
    .build()
```

### Application Lifecycle
```python
# Example: Clean up when app quits
trigger = TriggerBuilder() \
    .when_app_launches("com.adobe.Photoshop") \
    .with_priority(5) \
    .with_timeout(60) \
    .build()
```

### Composite Trigger
```python
# Example: Evening automation
trigger = TriggerBuilder() \
    .cron_pattern("0 18 * * *")  # 6 PM daily \
    .with_condition({
        "type": "system",
        "operator": "equals",
        "operand": "network_connected"
    }) \
    .limit_executions(1) \
    .build()
```

## ✅ Success Criteria
- Complete advanced trigger implementation with all major trigger types (time, file, system, user)
- Comprehensive security validation prevents unauthorized access and resource exhaustion
- Property-based tests validate behavior across all trigger scenarios and edge cases
- Integration with condition system (TASK_21) for intelligent trigger logic
- Performance: <100ms trigger setup, <1s for complex composite triggers
- Documentation: Complete API documentation with security considerations and examples
- TESTING.md shows 95%+ test coverage with all security and performance tests passing
- Tool enables AI to create fully automated, event-driven workflows that respond to environmental changes

## 🔄 Integration Points
- **TASK_21 (km_add_condition)**: Conditions enhance trigger logic with complex decision making
- **TASK_22 (km_control_flow)**: Control flow enables sophisticated trigger responses
- **TASK_10 (km_create_macro)**: Add advanced triggers to newly created macros
- **All Future Tasks**: Advanced triggers enable automated execution of any tool
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- This transforms macros from manual tools to intelligent, automated systems
- Essential for creating responsive automation that reacts to environmental changes
- Security is critical - triggers can monitor sensitive system events and file system
- Must maintain functional programming patterns for testability and composability
- Success here enables truly proactive automation that doesn't require human intervention
- Combined with conditions (TASK_21) and control flow (TASK_22), creates complete intelligent automation platform