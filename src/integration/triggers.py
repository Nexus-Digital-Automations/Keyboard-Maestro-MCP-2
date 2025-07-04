"""
Immutable Trigger Management System

Manages Keyboard Maestro triggers with immutable state transitions
and functional state management patterns.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import FrozenSet, Dict, Optional, List, Callable, Any, NamedTuple
from datetime import datetime
from enum import Enum

from ..core.types import TriggerId, MacroId
from ..core.contracts import require, ensure
from .events import TriggerType, KMEvent


class TriggerStatus(Enum):
    """Status of individual triggers."""
    INACTIVE = "inactive"
    REGISTERED = "registered"
    ACTIVE = "active"
    FAILED = "failed"
    SUSPENDED = "suspended"


class TriggerEventType(Enum):
    """Types of trigger state change events."""
    REGISTER = "register"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"
    FAIL = "fail"
    SUSPEND = "suspend"
    RESUME = "resume"
    UNREGISTER = "unregister"


@dataclass(frozen=True)
class TriggerMetadata:
    """Immutable metadata for triggers."""
    name: str
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    failure_count: int = 0
    
    def with_trigger_event(self) -> TriggerMetadata:
        """Create new metadata with incremented trigger count."""
        return dataclass_replace(
            self,
            last_triggered=datetime.now(),
            trigger_count=self.trigger_count + 1
        )
    
    def with_failure(self) -> TriggerMetadata:
        """Create new metadata with incremented failure count."""
        return dataclass_replace(
            self,
            failure_count=self.failure_count + 1
        )


@dataclass(frozen=True)
class TriggerInfo:
    """Complete information about a registered trigger."""
    trigger_id: TriggerId
    macro_id: MacroId
    trigger_type: TriggerType
    status: TriggerStatus
    configuration: Dict[str, Any]
    metadata: TriggerMetadata
    
    def is_active(self) -> bool:
        """Check if trigger is currently active."""
        return self.status == TriggerStatus.ACTIVE
    
    def can_trigger(self) -> bool:
        """Check if trigger can fire."""
        return self.status in (TriggerStatus.ACTIVE, TriggerStatus.REGISTERED)
    
    def with_status(self, status: TriggerStatus) -> TriggerInfo:
        """Create new trigger info with different status."""
        return dataclass_replace(self, status=status)
    
    def with_trigger_event(self) -> TriggerInfo:
        """Create new trigger info with updated metadata after trigger."""
        return dataclass_replace(self, metadata=self.metadata.with_trigger_event())
    
    def with_failure(self) -> TriggerInfo:
        """Create new trigger info with failure recorded."""
        new_metadata = self.metadata.with_failure()
        new_status = TriggerStatus.FAILED if new_metadata.failure_count >= 3 else self.status
        return dataclass_replace(self, metadata=new_metadata, status=new_status)


@dataclass(frozen=True)
class TriggerState:
    """Immutable state of all registered triggers."""
    triggers: Dict[TriggerId, TriggerInfo] = field(default_factory=dict)
    active_triggers: FrozenSet[TriggerId] = field(default_factory=frozenset)
    failed_triggers: FrozenSet[TriggerId] = field(default_factory=frozenset)
    suspended_triggers: FrozenSet[TriggerId] = field(default_factory=frozenset)
    state_version: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def empty(cls) -> TriggerState:
        """Create empty trigger state."""
        return cls()
    
    def get_trigger(self, trigger_id: TriggerId) -> Optional[TriggerInfo]:
        """Get trigger info by ID."""
        return self.triggers.get(trigger_id)
    
    def has_trigger(self, trigger_id: TriggerId) -> bool:
        """Check if trigger is registered."""
        return trigger_id in self.triggers
    
    def get_active_triggers(self) -> List[TriggerInfo]:
        """Get list of all active triggers."""
        return [info for info in self.triggers.values() if info.is_active()]
    
    def get_triggers_by_type(self, trigger_type: TriggerType) -> List[TriggerInfo]:
        """Get triggers of specific type."""
        return [info for info in self.triggers.values() if info.trigger_type == trigger_type]
    
    def get_triggers_for_macro(self, macro_id: MacroId) -> List[TriggerInfo]:
        """Get all triggers for specific macro."""
        return [info for info in self.triggers.values() if info.macro_id == macro_id]
    
    # @require(lambda self, trigger_id: trigger_id)
    # @ensure(lambda result: result.state_version > self.state_version)
    def with_registered(self, trigger_info: TriggerInfo) -> TriggerState:
        """Pure function to add registered trigger."""
        new_triggers = self.triggers.copy()
        new_triggers[trigger_info.trigger_id] = trigger_info
        
        return TriggerState(
            triggers=new_triggers,
            active_triggers=self.active_triggers,
            failed_triggers=self.failed_triggers,
            suspended_triggers=self.suspended_triggers,
            state_version=self.state_version + 1,
            last_updated=datetime.now()
        )
    
    # @require(lambda self, trigger_id: trigger_id in self.triggers)
    # @ensure(lambda result: result.state_version > self.state_version)
    def with_activated(self, trigger_id: TriggerId) -> TriggerState:
        """Pure function to activate trigger."""
        trigger_info = self.triggers[trigger_id].with_status(TriggerStatus.ACTIVE)
        new_triggers = self.triggers.copy()
        new_triggers[trigger_id] = trigger_info
        
        return TriggerState(
            triggers=new_triggers,
            active_triggers=self.active_triggers | {trigger_id},
            failed_triggers=self.failed_triggers - {trigger_id},
            suspended_triggers=self.suspended_triggers - {trigger_id},
            state_version=self.state_version + 1,
            last_updated=datetime.now()
        )
    
    @require(lambda self, trigger_id: trigger_id in self.triggers)
    @ensure(lambda result: result.state_version > self.state_version)
    def with_deactivated(self, trigger_id: TriggerId) -> TriggerState:
        """Pure function to deactivate trigger."""
        trigger_info = self.triggers[trigger_id].with_status(TriggerStatus.INACTIVE)
        new_triggers = self.triggers.copy()
        new_triggers[trigger_id] = trigger_info
        
        return TriggerState(
            triggers=new_triggers,
            active_triggers=self.active_triggers - {trigger_id},
            failed_triggers=self.failed_triggers,
            suspended_triggers=self.suspended_triggers,
            state_version=self.state_version + 1,
            last_updated=datetime.now()
        )
    
    @require(lambda self, trigger_id: trigger_id in self.triggers)
    @ensure(lambda result: result.state_version > self.state_version)
    def with_failed(self, trigger_id: TriggerId) -> TriggerState:
        """Pure function to mark trigger as failed."""
        trigger_info = self.triggers[trigger_id].with_failure()
        new_triggers = self.triggers.copy()
        new_triggers[trigger_id] = trigger_info
        
        return TriggerState(
            triggers=new_triggers,
            active_triggers=self.active_triggers - {trigger_id},
            failed_triggers=self.failed_triggers | {trigger_id},
            suspended_triggers=self.suspended_triggers,
            state_version=self.state_version + 1,
            last_updated=datetime.now()
        )
    
    @require(lambda self, trigger_id: trigger_id in self.triggers)
    @ensure(lambda result: result.state_version > self.state_version)
    def with_triggered(self, trigger_id: TriggerId) -> TriggerState:
        """Pure function to record trigger activation."""
        trigger_info = self.triggers[trigger_id].with_trigger_event()
        new_triggers = self.triggers.copy()
        new_triggers[trigger_id] = trigger_info
        
        return TriggerState(
            triggers=new_triggers,
            active_triggers=self.active_triggers,
            failed_triggers=self.failed_triggers,
            suspended_triggers=self.suspended_triggers,
            state_version=self.state_version + 1,
            last_updated=datetime.now()
        )
    
    @require(lambda self, trigger_id: trigger_id in self.triggers)
    @ensure(lambda result: result.state_version > self.state_version)
    def without_trigger(self, trigger_id: TriggerId) -> TriggerState:
        """Pure function to remove trigger."""
        new_triggers = self.triggers.copy()
        del new_triggers[trigger_id]
        
        return TriggerState(
            triggers=new_triggers,
            active_triggers=self.active_triggers - {trigger_id},
            failed_triggers=self.failed_triggers - {trigger_id},
            suspended_triggers=self.suspended_triggers - {trigger_id},
            state_version=self.state_version + 1,
            last_updated=datetime.now()
        )


@dataclass(frozen=True)
class TriggerEvent:
    """Event representing a trigger state change."""
    trigger_id: TriggerId
    event_type: TriggerEventType
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def register_event(cls, trigger_id: TriggerId, **details) -> TriggerEvent:
        """Create trigger registration event."""
        return cls(trigger_id, TriggerEventType.REGISTER, details=details)
    
    @classmethod
    def activate_event(cls, trigger_id: TriggerId, **details) -> TriggerEvent:
        """Create trigger activation event."""
        return cls(trigger_id, TriggerEventType.ACTIVATE, details=details)
    
    @classmethod
    def fail_event(cls, trigger_id: TriggerId, error_message: str, **details) -> TriggerEvent:
        """Create trigger failure event."""
        failure_details = {"error_message": error_message, **details}
        return cls(trigger_id, TriggerEventType.FAIL, details=failure_details)


# Pure state transition functions

# @require(lambda current, event: isinstance(current, TriggerState) and isinstance(event, TriggerEvent))
# @ensure(lambda result: isinstance(result, TriggerState))
def update_trigger_state(current: TriggerState, event: TriggerEvent) -> TriggerState:
    """Pure state transition function for trigger events."""
    if event.event_type == TriggerEventType.REGISTER:
        # Extract trigger info from event details
        trigger_info = TriggerInfo(
            trigger_id=event.trigger_id,
            macro_id=event.details.get("macro_id"),
            trigger_type=event.details.get("trigger_type"),
            status=TriggerStatus.REGISTERED,
            configuration=event.details.get("configuration", {}),
            metadata=TriggerMetadata(
                name=event.details.get("name", f"Trigger {event.trigger_id}"),
                description=event.details.get("description")
            )
        )
        return current.with_registered(trigger_info)
    
    elif event.event_type == TriggerEventType.ACTIVATE:
        if current.has_trigger(event.trigger_id):
            return current.with_activated(event.trigger_id)
    
    elif event.event_type == TriggerEventType.DEACTIVATE:
        if current.has_trigger(event.trigger_id):
            return current.with_deactivated(event.trigger_id)
    
    elif event.event_type == TriggerEventType.FAIL:
        if current.has_trigger(event.trigger_id):
            return current.with_failed(event.trigger_id)
    
    elif event.event_type == TriggerEventType.UNREGISTER:
        if current.has_trigger(event.trigger_id):
            return current.without_trigger(event.trigger_id)
    
    # Return unchanged state if event doesn't apply
    return current


def apply_trigger_events(initial_state: TriggerState, events: List[TriggerEvent]) -> TriggerState:
    """Apply sequence of trigger events to state."""
    return _reduce_state(initial_state, events, update_trigger_state)


def _reduce_state(
    initial: TriggerState, 
    events: List[TriggerEvent], 
    reducer: Callable[[TriggerState, TriggerEvent], TriggerState]
) -> TriggerState:
    """Reduce events into final state."""
    current = initial
    for event in events:
        current = reducer(current, event)
    return current


# Trigger matching and filtering utilities

def create_trigger_matcher(
    trigger_type: Optional[TriggerType] = None,
    macro_id: Optional[MacroId] = None,
    status: Optional[TriggerStatus] = None
) -> Callable[[TriggerInfo], bool]:
    """Create predicate function for matching triggers."""
    def matches(trigger_info: TriggerInfo) -> bool:
        if trigger_type and trigger_info.trigger_type != trigger_type:
            return False
        if macro_id and trigger_info.macro_id != macro_id:
            return False
        if status and trigger_info.status != status:
            return False
        return True
    
    return matches


def find_triggers_matching(
    state: TriggerState, 
    predicate: Callable[[TriggerInfo], bool]
) -> List[TriggerInfo]:
    """Find all triggers matching predicate."""
    return [info for info in state.triggers.values() if predicate(info)]


def get_trigger_statistics(state: TriggerState) -> Dict[str, int]:
    """Get statistics about trigger state."""
    return {
        "total_triggers": len(state.triggers),
        "active_triggers": len(state.active_triggers),
        "failed_triggers": len(state.failed_triggers),
        "suspended_triggers": len(state.suspended_triggers),
        "total_trigger_count": sum(info.metadata.trigger_count for info in state.triggers.values()),
        "total_failure_count": sum(info.metadata.failure_count for info in state.triggers.values())
    }


# TASK_2 Phase 2 Implementation: Trigger Registration & Event Routing System

from typing import AsyncIterator
from asyncio import Queue
import asyncio
# Avoid circular import - import MacroEngine when needed
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.engine import MacroEngine
from .km_client import KMClient, Either, KMError, TriggerDefinition as KMTriggerDefinition
from .security import validate_trigger_input, sanitize_trigger_configuration


@dataclass
class TriggerDefinition:
    """Definition for registering a new trigger with Keyboard Maestro."""
    trigger_id: TriggerId
    macro_id: MacroId
    trigger_type: TriggerType
    configuration: Dict[str, Any]
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    
    def to_km_format(self) -> Dict[str, Any]:
        """Convert to Keyboard Maestro registration format."""
        return {
            "triggerId": self.trigger_id,
            "macroId": self.macro_id,
            "type": self.trigger_type.value,
            "config": self.configuration,
            "name": self.name or f"Trigger {self.trigger_id}",
            "description": self.description,
            "enabled": self.enabled
        }
    
    def to_km_trigger_definition(self) -> KMTriggerDefinition:
        """Convert to KMClient TriggerDefinition format."""
        return KMTriggerDefinition(
            trigger_id=self.trigger_id,
            macro_id=self.macro_id,
            trigger_type=self.trigger_type,
            configuration=self.configuration,
            enabled=self.enabled
        )


@dataclass
class EventRoutingRule:
    """Rule for routing KM events to macro handlers."""
    trigger_matcher: Callable[[TriggerInfo], bool]
    event_filter: Callable[[KMEvent], bool]
    priority: int = 0
    
    def matches(self, trigger_info: TriggerInfo, event: KMEvent) -> bool:
        """Check if this rule matches the trigger and event."""
        return self.trigger_matcher(trigger_info) and self.event_filter(event)


class TriggerRegistrationManager:
    """Manages trigger registration with Keyboard Maestro system."""
    
    def __init__(self, km_client: KMClient):
        self._km_client = km_client
        self._state = TriggerState.empty()
        self._state_lock = asyncio.Lock()
    
    # @require(lambda self, trigger_def: isinstance(trigger_def, TriggerDefinition))
    async def register_trigger(self, trigger_def: TriggerDefinition) -> Either[KMError, TriggerId]:
        """Register a new trigger with Keyboard Maestro."""
        try:
            # Validate and sanitize trigger configuration
            is_valid = validate_trigger_input(trigger_def.configuration)
            if not is_valid:
                return Either.left(KMError.validation_error("Invalid trigger configuration"))
            
            sanitized_config = sanitize_trigger_configuration(trigger_def.configuration)
            
            # Update trigger definition with sanitized config
            safe_trigger_def = TriggerDefinition(
                trigger_id=trigger_def.trigger_id,
                macro_id=trigger_def.macro_id,
                trigger_type=trigger_def.trigger_type,
                configuration=sanitized_config,
                name=trigger_def.name,
                description=trigger_def.description,
                enabled=trigger_def.enabled
            )
            
            # Register with Keyboard Maestro
            km_trigger_def = safe_trigger_def.to_km_trigger_definition()
            km_result = await self._km_client.register_trigger_async(km_trigger_def)
            
            if km_result.is_left():
                return km_result
            
            # Update local state
            async with self._state_lock:
                trigger_info = TriggerInfo(
                    trigger_id=trigger_def.trigger_id,
                    macro_id=trigger_def.macro_id,
                    trigger_type=trigger_def.trigger_type,
                    status=TriggerStatus.REGISTERED,
                    configuration=sanitized_config,
                    metadata=TriggerMetadata(
                        name=trigger_def.name or f"Trigger {trigger_def.trigger_id}",
                        description=trigger_def.description
                    )
                )
                self._state = self._state.with_registered(trigger_info)
            
            return Either.right(trigger_def.trigger_id)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Trigger registration failed: {str(e)}"))
    
    # @require(lambda self, trigger_id: trigger_id)
    async def activate_trigger(self, trigger_id: TriggerId) -> Either[KMError, bool]:
        """Activate a registered trigger."""
        async with self._state_lock:
            if not self._state.has_trigger(trigger_id):
                return Either.left(KMError.not_found_error(f"Trigger {trigger_id} not found"))
            
            # Activate in Keyboard Maestro
            km_result = await self._km_client.activate_trigger_async(trigger_id)
            if km_result.is_left():
                return km_result
            
            # Update local state
            self._state = self._state.with_activated(trigger_id)
            return Either.right(True)
    
    # @require(lambda self, trigger_id: trigger_id)
    async def deactivate_trigger(self, trigger_id: TriggerId) -> Either[KMError, bool]:
        """Deactivate a trigger."""
        async with self._state_lock:
            if not self._state.has_trigger(trigger_id):
                return Either.left(KMError.not_found_error(f"Trigger {trigger_id} not found"))
            
            # Deactivate in Keyboard Maestro
            km_result = await self._km_client.deactivate_trigger_async(trigger_id)
            if km_result.is_left():
                return km_result
            
            # Update local state
            self._state = self._state.with_deactivated(trigger_id)
            return Either.right(True)
    
    async def sync_state_with_km(self) -> Either[KMError, TriggerState]:
        """Synchronize trigger state with Keyboard Maestro."""
        try:
            # Get current state from KM
            km_result = await self._km_client.list_triggers_async()
            if km_result.is_left():
                return km_result
            
            km_triggers = km_result.get_right()
            
            # Update local state based on KM state
            async with self._state_lock:
                # Apply state updates based on KM response
                events = []
                for km_trigger in km_triggers:
                    trigger_id = TriggerId(km_trigger["triggerId"])
                    if km_trigger["status"] == "active":
                        events.append(TriggerEvent.activate_event(trigger_id))
                    elif km_trigger["status"] == "inactive":
                        events.append(TriggerEvent(trigger_id, TriggerEventType.DEACTIVATE, datetime.now()))
                
                self._state = apply_trigger_events(self._state, events)
            
            return Either.right(self._state)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"State sync failed: {str(e)}"))
    
    def get_current_state(self) -> TriggerState:
        """Get current trigger state (thread-safe read)."""
        return self._state


class EventRouter:
    """Routes Keyboard Maestro events to appropriate macro handlers."""
    
    def __init__(self, macro_engine: MacroEngine, trigger_manager: TriggerRegistrationManager):
        self._macro_engine = macro_engine
        self._trigger_manager = trigger_manager
        self._routing_rules: List[EventRoutingRule] = []
        self._event_queue: Queue = Queue()
        self._running = False
    
    def add_routing_rule(self, rule: EventRoutingRule) -> None:
        """Add an event routing rule."""
        self._routing_rules.append(rule)
        # Sort by priority (higher priority first)
        self._routing_rules.sort(key=lambda r: r.priority, reverse=True)
    
    def remove_routing_rule(self, rule: EventRoutingRule) -> None:
        """Remove an event routing rule."""
        if rule in self._routing_rules:
            self._routing_rules.remove(rule)
    
    async def route_event(self, event: KMEvent) -> Either[KMError, bool]:
        """Route a single KM event to appropriate macro handler."""
        try:
            trigger_state = self._trigger_manager.get_current_state()
            
            # Find matching trigger
            if not event.trigger_id:
                return Either.left(KMError.validation_error("Event missing trigger ID"))
            
            trigger_info = trigger_state.get_trigger(event.trigger_id)
            if not trigger_info:
                return Either.left(KMError.not_found_error(f"Trigger {event.trigger_id} not found"))
            
            if not trigger_info.can_trigger():
                return Either.left(KMError.execution_error(f"Trigger {event.trigger_id} cannot fire"))
            
            # Apply routing rules to find handler
            matching_rules = [
                rule for rule in self._routing_rules 
                if rule.matches(trigger_info, event)
            ]
            
            if not matching_rules:
                # Default routing: execute macro directly
                return await self._execute_macro_for_event(trigger_info, event)
            
            # Use highest priority matching rule
            # For now, default to macro execution
            return await self._execute_macro_for_event(trigger_info, event)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Event routing failed: {str(e)}"))
    
    async def _execute_macro_for_event(self, trigger_info: TriggerInfo, event: KMEvent) -> Either[KMError, bool]:
        """Execute macro associated with trigger."""
        try:
            # Execute macro through engine
            execution_result = await self._macro_engine.execute_macro_async(
                macro_id=trigger_info.macro_id,
                trigger_value=event.get_payload_value("trigger_value"),
                context_data=event.payload
            )
            
            if execution_result.status.value in ["completed"]:
                # Update trigger state to record successful trigger
                async with self._trigger_manager._state_lock:
                    self._trigger_manager._state = self._trigger_manager._state.with_triggered(trigger_info.trigger_id)
                return Either.right(True)
            else:
                # Update trigger state to record failure
                async with self._trigger_manager._state_lock:
                    self._trigger_manager._state = self._trigger_manager._state.with_failed(trigger_info.trigger_id)
                return Either.left(KMError.execution_error(f"Macro execution failed: {execution_result.error_message}"))
                
        except Exception as e:
            return Either.left(KMError.execution_error(f"Macro execution error: {str(e)}"))
    
    async def start_event_processing(self) -> None:
        """Start the event processing loop."""
        self._running = True
        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                
                # Route the event
                result = await self.route_event(event)
                
                # Log result (could be enhanced with proper logging)
                if result.is_left():
                    print(f"Event routing failed: {result.get_left().message}")
                
                # Mark task done
                self._event_queue.task_done()
                
            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                continue
            except Exception as e:
                print(f"Event processing error: {str(e)}")
    
    async def stop_event_processing(self) -> None:
        """Stop the event processing loop."""
        self._running = False
        # Wait for queue to empty
        await self._event_queue.join()
    
    async def enqueue_event(self, event: KMEvent) -> None:
        """Add event to processing queue."""
        await self._event_queue.put(event)


# Built-in routing rules for common patterns

def create_hotkey_routing_rule() -> EventRoutingRule:
    """Create routing rule for hotkey triggers."""
    return EventRoutingRule(
        trigger_matcher=lambda trigger: trigger.trigger_type == TriggerType.HOTKEY,
        event_filter=lambda event: event.trigger_type == TriggerType.HOTKEY,
        priority=100
    )

def create_application_routing_rule() -> EventRoutingRule:
    """Create routing rule for application triggers."""
    return EventRoutingRule(
        trigger_matcher=lambda trigger: trigger.trigger_type == TriggerType.APPLICATION,
        event_filter=lambda event: event.trigger_type == TriggerType.APPLICATION,
        priority=90
    )

def create_high_priority_routing_rule() -> EventRoutingRule:
    """Create routing rule for high-priority events."""
    return EventRoutingRule(
        trigger_matcher=lambda trigger: True,  # Match all triggers
        event_filter=lambda event: event.is_high_priority(),
        priority=200
    )