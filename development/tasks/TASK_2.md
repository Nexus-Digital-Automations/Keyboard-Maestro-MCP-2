# TASK_2: Keyboard Maestro Integration Layer

**Created By**: Agent_1 | **Priority**: HIGH | **Duration**: 3 hours  
**Technique Focus**: Functional Programming + Property-Based Testing + Security Boundaries
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: TASK_1 (Core engine must be complete)
**Blocking**: TASK_3, TASK_4

## 📖 Required Reading (Complete before starting)
- [x] **development/protocols/KM_MCP.md**: Keyboard Maestro MCP specifications
- [x] **src/core/**: Review completed core engine implementation from TASK_1
- [x] **CLAUDE.md**: Functional programming patterns and security requirements
- [x] **tests/TESTING.md**: Current test status and framework setup

## 🎯 Implementation Overview
Build the integration layer connecting the core macro engine to Keyboard Maestro automation system with event-driven architecture and functional programming patterns.

<thinking>
Integration architecture considerations:
1. Event-Driven Design: React to KM triggers and system events
2. Functional Composition: Pure functions for data transformation
3. Immutable State: Event sourcing for trigger management
4. Error Recovery: Graceful handling of KM connection failures
5. Security: Validate all external inputs from KM system
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: KM Protocol Integration
- [x] **KM API wrapper**: Create functional interface to Keyboard Maestro APIs
- [x] **Event system**: Implement event-driven trigger handling with immutable events
- [x] **Message protocol**: Handle MCP communication with KM using pure functions
- [x] **Connection management**: Robust connection handling with automatic recovery

### Phase 2: Trigger Management System
- [x] **Trigger registration**: Register macro triggers with KM system
- [x] **Event routing**: Route KM events to appropriate macro handlers
- [x] **State synchronization**: Sync trigger states between engine and KM
- [x] **Security validation**: Validate all incoming trigger data

### Phase 3: Integration Testing & Validation
- [x] **Property-based tests**: Test event handling across input ranges
- [x] **Integration tests**: Validate KM communication protocols
- [x] **TESTING.md update**: Document integration test results
- [x] **Performance testing**: Ensure trigger response < 50ms

## 🔧 Implementation Files & Specifications

### Integration Files to Create:
```
src/
├── integration/
│   ├── __init__.py           # Public integration API
│   ├── km_client.py          # KM API client (150-200 lines)
│   ├── events.py             # Event types and handlers (100-150 lines)
│   ├── triggers.py           # Trigger management (100-150 lines)
│   ├── protocol.py           # MCP protocol implementation (75-100 lines)
│   └── security.py           # Input validation and sanitization (75-100 lines)
└── tests/
    └── test_integration/
        ├── test_km_client.py     # KM client tests
        ├── test_events.py        # Event system tests
        ├── test_triggers.py      # Trigger management tests
        └── test_security.py      # Security validation tests
```

### Key Implementation Requirements:

#### events.py - Functional Event System
```python
from dataclasses import dataclass
from typing import Callable, TypeVar, Generic
from functools import reduce

@dataclass(frozen=True)
class KMEvent:
    """Immutable event from Keyboard Maestro."""
    event_id: EventId
    trigger_type: TriggerType
    payload: EventPayload
    timestamp: datetime
    
    def transform(self, transformer: Callable[['KMEvent'], 'KMEvent']) -> 'KMEvent':
        """Pure transformation of event data."""
        return transformer(self)

def compose_event_handlers(*handlers: EventHandler) -> EventHandler:
    """Compose multiple event handlers into single function."""
    return lambda event: reduce(lambda e, handler: handler(e), handlers, event)
```

#### km_client.py - Functional KM Interface  
```python
from typing import Optional, Callable
from functools import partial

class KMClient:
    """Functional interface to Keyboard Maestro APIs."""
    
    def __init__(self, connection_config: ConnectionConfig):
        self._send_command = partial(self._safe_send, connection_config)
    
    def register_trigger(self, trigger_def: TriggerDefinition) -> Either[Error, TriggerId]:
        """Register trigger with functional error handling."""
        return self._send_command("register_trigger", trigger_def.to_dict())
    
    def _safe_send(self, config: ConnectionConfig, command: str, payload: dict) -> Either[Error, dict]:
        """Pure function for safe command sending with error handling."""
```

#### triggers.py - Immutable Trigger Management
```python
from typing import NamedTuple, FrozenSet
from dataclasses import dataclass

@dataclass(frozen=True)
class TriggerState:
    """Immutable trigger state."""
    registered_triggers: FrozenSet[TriggerId]
    active_triggers: FrozenSet[TriggerId] 
    failed_triggers: FrozenSet[TriggerId]
    
    def with_registered(self, trigger_id: TriggerId) -> 'TriggerState':
        """Pure function to add registered trigger."""
        return TriggerState(
            self.registered_triggers | {trigger_id},
            self.active_triggers,
            self.failed_triggers
        )

def update_trigger_state(current: TriggerState, event: TriggerEvent) -> TriggerState:
    """Pure state transition function."""
```

#### security.py - Input Validation  
```python
from typing import TypeGuard
from .errors import SecurityValidationError

def validate_km_input(raw_input: dict) -> TypeGuard[ValidatedKMInput]:
    """Comprehensive validation of KM input data."""
    
def sanitize_trigger_data(trigger_data: dict) -> SanitizedTriggerData:
    """Sanitize trigger data to prevent injection attacks."""
    
@require(lambda data: is_valid_km_format(data))
@ensure(lambda result: is_sanitized(result))
def process_km_event(event_data: dict) -> ProcessedEvent:
    """Process KM event with security boundaries."""
```

## 🏗️ Modularity Strategy
- **km_client.py**: KM API communication layer (target: 175 lines)
- **events.py**: Event system with functional patterns (target: 125 lines)
- **triggers.py**: Trigger management with immutable state (target: 125 lines)  
- **protocol.py**: MCP protocol implementation (target: 90 lines)
- **security.py**: Input validation and sanitization (target: 90 lines)

## ✅ Success Criteria
- Functional programming patterns implemented throughout integration layer
- Property-based testing for event handling and trigger management
- Security boundaries prevent malicious input from KM system
- Integration tests validate KM communication protocols
- Performance: Trigger response time < 50ms
- TESTING.md updated with integration test status
- All advanced techniques properly implemented
- Error recovery handles KM connection failures gracefully
- Immutable state management with pure functions for transformations
- Zero regressions: All existing tests continue passing