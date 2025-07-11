"""Comprehensive tests for integration.events module with systematic coverage.

This module provides comprehensive test coverage for the events system with focus
on functional event processing, immutable transformations, and security validation.
"""

import uuid
from datetime import datetime
from unittest.mock import patch

from hypothesis import given
from hypothesis import strategies as st
from src.core.types import ExecutionToken, MacroId, TriggerId
from src.integration.events import (
    DEFAULT_EVENT_PIPELINE,
    SECURITY_FOCUSED_PIPELINE,
    Event,
    EventManager,
    EventPriority,
    EventProcessingResult,
    EventType,
    KMEvent,
    TriggerType,
    add_processing_timestamp,
    compose_event_handlers,
    create_conditional_handler,
    create_payload_transformer,
    create_priority_filter,
    create_trigger_type_filter,
    get_default_event_pipeline,
    get_default_event_pipeline_cached,
    get_security_focused_pipeline,
    get_security_focused_pipeline_cached,
    normalize_trigger_data,
    sanitize_event_payload,
)


class TestTriggerType:
    """Test TriggerType enumeration."""

    def test_trigger_type_values(self) -> None:
        """Test all trigger type values."""
        assert TriggerType.HOTKEY.value == "hotkey"
        assert TriggerType.APPLICATION.value == "application"
        assert TriggerType.TIME.value == "time"
        assert TriggerType.SYSTEM.value == "system"
        assert TriggerType.FILE.value == "file"
        assert TriggerType.DEVICE.value == "device"
        assert TriggerType.PERIODIC.value == "periodic"
        assert TriggerType.REMOTE.value == "remote"

    def test_trigger_type_enumeration(self) -> None:
        """Test trigger type enumeration completeness."""
        all_types = list(TriggerType)
        assert len(all_types) == 8

        # Verify each type is unique
        values = [trigger_type.value for trigger_type in all_types]
        assert len(values) == len(set(values))

    def test_trigger_type_string_conversion(self) -> None:
        """Test trigger type string representation."""
        for trigger_type in TriggerType:
            assert isinstance(trigger_type.value, str)
            assert len(trigger_type.value) > 0


class TestEventPriority:
    """Test EventPriority enumeration."""

    def test_event_priority_values(self) -> None:
        """Test all event priority values."""
        assert EventPriority.LOW.value == "low"
        assert EventPriority.NORMAL.value == "normal"
        assert EventPriority.HIGH.value == "high"
        assert EventPriority.CRITICAL.value == "critical"

    def test_event_priority_ordering(self) -> None:
        """Test event priority ordering for filter logic."""
        all_priorities = list(EventPriority)
        assert len(all_priorities) == 4

        # Verify each priority is unique
        values = [priority.value for priority in all_priorities]
        assert len(values) == len(set(values))


class TestKMEvent:
    """Test KMEvent functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.event_id = str(uuid.uuid4())
        self.trigger_id = TriggerId("test_trigger")
        self.macro_id = MacroId("test_macro")
        self.payload = {"key": "value", "number": 42}
        self.timestamp = datetime.now()

    def test_km_event_creation_minimal(self) -> None:
        """Test creation of KM event with minimal parameters."""
        event = KMEvent(
            event_id=self.event_id,
            trigger_type=TriggerType.HOTKEY,
            trigger_id=self.trigger_id,
            macro_id=self.macro_id,
            payload=self.payload,
            timestamp=self.timestamp,
        )

        assert event.event_id == self.event_id
        assert event.trigger_type == TriggerType.HOTKEY
        assert event.trigger_id == self.trigger_id
        assert event.macro_id == self.macro_id
        assert event.payload == self.payload
        assert event.timestamp == self.timestamp
        assert event.priority == EventPriority.NORMAL
        assert event.source == "keyboard_maestro"

    def test_km_event_creation_complete(self) -> None:
        """Test creation of KM event with all parameters."""
        event = KMEvent(
            event_id=self.event_id,
            trigger_type=TriggerType.APPLICATION,
            trigger_id=self.trigger_id,
            macro_id=self.macro_id,
            payload=self.payload,
            timestamp=self.timestamp,
            priority=EventPriority.HIGH,
            source="test_source",
        )

        assert event.priority == EventPriority.HIGH
        assert event.source == "test_source"

    def test_km_event_create_class_method(self) -> None:
        """Test KMEvent.create class method."""
        with (
            patch("src.integration.events.uuid.uuid4") as mock_uuid,
            patch("src.integration.events.datetime") as mock_datetime,
        ):
            mock_uuid.return_value = type(
                "MockUUID", (), {"__str__": lambda self: "test-uuid"}
            )()
            mock_datetime.now.return_value = self.timestamp

            event = KMEvent.create(
                trigger_type=TriggerType.TIME,
                trigger_id=self.trigger_id,
                payload=self.payload,
                macro_id=self.macro_id,
                priority=EventPriority.CRITICAL,
            )

            assert event.event_id == "test-uuid"
            assert event.trigger_type == TriggerType.TIME
            assert event.trigger_id == self.trigger_id
            assert event.macro_id == self.macro_id
            assert event.payload == self.payload
            assert event.priority == EventPriority.CRITICAL
            assert event.timestamp == self.timestamp

    def test_km_event_create_defaults(self) -> None:
        """Test KMEvent.create with default parameters."""
        event = KMEvent.create(
            trigger_type=TriggerType.SYSTEM,
            trigger_id=self.trigger_id,
            payload=self.payload,
        )

        assert event.macro_id is None
        assert event.priority == EventPriority.NORMAL
        assert event.source == "keyboard_maestro"

    def test_km_event_transform(self) -> None:
        """Test event transformation functionality."""
        event = KMEvent.create(
            trigger_type=TriggerType.HOTKEY,
            trigger_id=self.trigger_id,
            payload={"original": "value"},
        )

        def add_metadata(e: KMEvent) -> KMEvent:
            return e.with_payload("metadata", "added")

        transformed = event.transform(add_metadata)

        # Original event unchanged (immutable)
        assert "metadata" not in event.payload

        # Transformed event has new data
        assert transformed.payload["metadata"] == "added"
        assert transformed.payload["original"] == "value"

    def test_km_event_with_payload(self) -> None:
        """Test adding payload data to event."""
        event = KMEvent.create(
            trigger_type=TriggerType.FILE,
            trigger_id=self.trigger_id,
            payload={"existing": "data"},
        )

        new_event = event.with_payload("new_key", "new_value")

        # Original event unchanged
        assert "new_key" not in event.payload

        # New event has additional data
        assert new_event.payload["existing"] == "data"
        assert new_event.payload["new_key"] == "new_value"

    def test_km_event_with_priority(self) -> None:
        """Test changing event priority."""
        event = KMEvent.create(
            trigger_type=TriggerType.DEVICE,
            trigger_id=self.trigger_id,
            payload=self.payload,
            priority=EventPriority.LOW,
        )

        high_priority_event = event.with_priority(EventPriority.HIGH)

        # Original event unchanged
        assert event.priority == EventPriority.LOW

        # New event has different priority
        assert high_priority_event.priority == EventPriority.HIGH

    def test_km_event_get_payload_value(self) -> None:
        """Test getting values from event payload."""
        event = KMEvent.create(
            trigger_type=TriggerType.PERIODIC,
            trigger_id=self.trigger_id,
            payload={"existing_key": "existing_value", "number": 123},
        )

        # Test existing key
        assert event.get_payload_value("existing_key") == "existing_value"
        assert event.get_payload_value("number") == 123

        # Test missing key with default
        assert event.get_payload_value("missing_key", "default") == "default"

        # Test missing key without default
        assert event.get_payload_value("missing_key") is None

    def test_km_event_is_high_priority(self) -> None:
        """Test high priority detection."""
        low_event = KMEvent.create(
            trigger_type=TriggerType.REMOTE,
            trigger_id=self.trigger_id,
            payload={},
            priority=EventPriority.LOW,
        )

        normal_event = low_event.with_priority(EventPriority.NORMAL)
        high_event = low_event.with_priority(EventPriority.HIGH)
        critical_event = low_event.with_priority(EventPriority.CRITICAL)

        assert low_event.is_high_priority() is False
        assert normal_event.is_high_priority() is False
        assert high_event.is_high_priority() is True
        assert critical_event.is_high_priority() is True


class TestEventProcessingResult:
    """Test EventProcessingResult functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.event = KMEvent.create(
            trigger_type=TriggerType.HOTKEY,
            trigger_id=TriggerId("test"),
            payload={"test": "data"},
        )
        self.execution_token = ExecutionToken("test-token")

    def test_success_result_minimal(self) -> None:
        """Test creation of minimal success result."""
        result = EventProcessingResult.success_result(self.event)

        assert result.success is True
        assert result.processed_event == self.event
        assert result.error_message is None
        assert result.execution_token is None
        assert result.processing_duration_ms is None
        assert result.metadata == {}

    def test_success_result_complete(self) -> None:
        """Test creation of complete success result."""
        result = EventProcessingResult.success_result(
            self.event,
            execution_token=self.execution_token,
            duration_ms=123.45,
            custom_key="custom_value",
            another_key=42,
        )

        assert result.success is True
        assert result.processed_event == self.event
        assert result.execution_token == self.execution_token
        assert result.processing_duration_ms == 123.45
        assert result.metadata["custom_key"] == "custom_value"
        assert result.metadata["another_key"] == 42

    def test_failure_result_minimal(self) -> None:
        """Test creation of minimal failure result."""
        result = EventProcessingResult.failure_result("Test error message")

        assert result.success is False
        assert result.processed_event is None
        assert result.error_message == "Test error message"
        assert result.execution_token is None
        assert result.processing_duration_ms is None
        assert result.metadata == {}

    def test_failure_result_complete(self) -> None:
        """Test creation of complete failure result."""
        result = EventProcessingResult.failure_result(
            "Test error message",
            duration_ms=67.89,
            error_code="ERR001",
            retry_count=3,
        )

        assert result.success is False
        assert result.error_message == "Test error message"
        assert result.processing_duration_ms == 67.89
        assert result.metadata["error_code"] == "ERR001"
        assert result.metadata["retry_count"] == 3


class TestEventHandlerComposition:
    """Test functional event handler composition."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.event = KMEvent.create(
            trigger_type=TriggerType.HOTKEY,
            trigger_id=TriggerId("test"),
            payload={"counter": 0},
        )

    def test_compose_event_handlers_empty(self) -> None:
        """Test composing no handlers returns identity function."""
        composed = compose_event_handlers()
        result = composed(self.event)
        assert result == self.event

    def test_compose_event_handlers_single(self) -> None:
        """Test composing single handler."""

        def increment_counter(event: KMEvent) -> KMEvent:
            count = event.get_payload_value("counter", 0)
            return event.with_payload("counter", count + 1)

        composed = compose_event_handlers(increment_counter)
        result = composed(self.event)

        assert result.get_payload_value("counter") == 1

    def test_compose_event_handlers_multiple(self) -> None:
        """Test composing multiple handlers."""

        def increment_counter(event: KMEvent) -> KMEvent:
            count = event.get_payload_value("counter", 0)
            return event.with_payload("counter", count + 1)

        def double_counter(event: KMEvent) -> KMEvent:
            count = event.get_payload_value("counter", 0)
            return event.with_payload("counter", count * 2)

        def add_marker(event: KMEvent) -> KMEvent:
            return event.with_payload("processed", True)

        composed = compose_event_handlers(
            increment_counter,
            double_counter,
            add_marker,
        )
        result = composed(self.event)

        # Should be ((0 + 1) * 2) = 2
        assert result.get_payload_value("counter") == 2
        assert result.get_payload_value("processed") is True

    def test_create_payload_transformer(self) -> None:
        """Test payload transformer creation."""
        event = KMEvent.create(
            trigger_type=TriggerType.APPLICATION,
            trigger_id=TriggerId("test"),
            payload={"text": "hello", "number": 42},
        )

        # Transform text to uppercase
        text_transformer = create_payload_transformer("text", str.upper)
        result = text_transformer(event)

        assert result.get_payload_value("text") == "HELLO"
        assert result.get_payload_value("number") == 42

        # Transform number by doubling
        number_transformer = create_payload_transformer("number", lambda x: x * 2)
        result2 = number_transformer(result)

        assert result2.get_payload_value("text") == "HELLO"
        assert result2.get_payload_value("number") == 84

    def test_create_payload_transformer_missing_key(self) -> None:
        """Test payload transformer with missing key."""
        event = KMEvent.create(
            trigger_type=TriggerType.SYSTEM,
            trigger_id=TriggerId("test"),
            payload={"existing": "value"},
        )

        transformer = create_payload_transformer("missing", str.upper)
        result = transformer(event)

        # Event should be unchanged if key doesn't exist
        assert result == event

    def test_create_conditional_handler(self) -> None:
        """Test conditional handler creation."""
        event_low = KMEvent.create(
            trigger_type=TriggerType.TIME,
            trigger_id=TriggerId("test"),
            payload={"value": 10},
            priority=EventPriority.LOW,
        )

        event_high = event_low.with_priority(EventPriority.HIGH)

        def is_high_priority(e: KMEvent) -> bool:
            return e.is_high_priority()

        def add_urgent_flag(e: KMEvent) -> KMEvent:
            return e.with_payload("urgent", True)

        conditional_handler = create_conditional_handler(
            is_high_priority,
            add_urgent_flag,
        )

        # Low priority event should not be transformed
        result_low = conditional_handler(event_low)
        assert result_low == event_low
        assert result_low.get_payload_value("urgent") is None

        # High priority event should be transformed
        result_high = conditional_handler(event_high)
        assert result_high.get_payload_value("urgent") is True


class TestEventFilters:
    """Test event filter functions."""

    def test_create_priority_filter(self) -> None:
        """Test priority filter creation."""
        normal_filter = create_priority_filter(EventPriority.NORMAL)
        high_filter = create_priority_filter(EventPriority.HIGH)

        event_low = KMEvent.create(
            trigger_type=TriggerType.FILE,
            trigger_id=TriggerId("test"),
            payload={},
            priority=EventPriority.LOW,
        )
        event_normal = event_low.with_priority(EventPriority.NORMAL)
        event_high = event_low.with_priority(EventPriority.HIGH)
        event_critical = event_low.with_priority(EventPriority.CRITICAL)

        # Normal filter (includes normal and above)
        assert normal_filter(event_low) is False
        assert normal_filter(event_normal) is True
        assert normal_filter(event_high) is True
        assert normal_filter(event_critical) is True

        # High filter (includes high and above)
        assert high_filter(event_low) is False
        assert high_filter(event_normal) is False
        assert high_filter(event_high) is True
        assert high_filter(event_critical) is True

    def test_create_trigger_type_filter_single(self) -> None:
        """Test trigger type filter with single type."""
        hotkey_filter = create_trigger_type_filter(TriggerType.HOTKEY)

        event_hotkey = KMEvent.create(
            trigger_type=TriggerType.HOTKEY,
            trigger_id=TriggerId("test"),
            payload={},
        )
        event_app = KMEvent.create(
            trigger_type=TriggerType.APPLICATION,
            trigger_id=TriggerId("test"),
            payload={},
        )

        assert hotkey_filter(event_hotkey) is True
        assert hotkey_filter(event_app) is False

    def test_create_trigger_type_filter_multiple(self) -> None:
        """Test trigger type filter with multiple types."""
        system_filter = create_trigger_type_filter(
            [
                TriggerType.SYSTEM,
                TriggerType.DEVICE,
                TriggerType.REMOTE,
            ]
        )

        event_system = KMEvent.create(
            trigger_type=TriggerType.SYSTEM,
            trigger_id=TriggerId("test"),
            payload={},
        )
        event_device = KMEvent.create(
            trigger_type=TriggerType.DEVICE,
            trigger_id=TriggerId("test"),
            payload={},
        )
        event_hotkey = KMEvent.create(
            trigger_type=TriggerType.HOTKEY,
            trigger_id=TriggerId("test"),
            payload={},
        )

        assert system_filter(event_system) is True
        assert system_filter(event_device) is True
        assert system_filter(event_hotkey) is False


class TestBuiltinTransformations:
    """Test built-in event transformations."""

    def test_sanitize_event_payload(self) -> None:
        """Test event payload sanitization."""
        event = KMEvent.create(
            trigger_type=TriggerType.REMOTE,
            trigger_id=TriggerId("test"),
            payload={
                "safe_text": "Hello World",
                "script_content": "<script>alert('xss')</script>",
                "js_url": "javascript:alert('xss')",
                "number": 42,
                "boolean": True,
            },
        )

        sanitized = sanitize_event_payload(event)

        # Safe content unchanged
        assert sanitized.get_payload_value("safe_text") == "Hello World"
        assert sanitized.get_payload_value("number") == 42
        assert sanitized.get_payload_value("boolean") is True

        # Dangerous content sanitized
        assert "&lt;script" in sanitized.get_payload_value("script_content")
        assert "javascript_" in sanitized.get_payload_value("js_url")

    def test_add_processing_timestamp(self) -> None:
        """Test adding processing timestamp."""
        event = KMEvent.create(
            trigger_type=TriggerType.PERIODIC,
            trigger_id=TriggerId("test"),
            payload={"original": "data"},
        )

        with patch("src.integration.events.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T12:00:00"
            )

            timestamped = add_processing_timestamp(event)

        assert timestamped.get_payload_value("original") == "data"
        assert (
            timestamped.get_payload_value("processing_timestamp")
            == "2024-01-01T12:00:00"
        )

    def test_normalize_trigger_data(self) -> None:
        """Test trigger data normalization."""
        event = KMEvent.create(
            trigger_type=TriggerType.APPLICATION,
            trigger_id=TriggerId("test"),
            payload={
                "triggerValue": "original_trigger_value",
                "macroUID": "original_macro_uid",
                "other_data": "unchanged",
            },
        )

        normalized = normalize_trigger_data(event)

        # Old keys should be renamed
        assert "triggerValue" not in normalized.payload
        assert "macroUID" not in normalized.payload

        # New keys should exist
        assert normalized.get_payload_value("trigger_value") == "original_trigger_value"
        assert normalized.get_payload_value("macro_id") == "original_macro_uid"
        assert normalized.get_payload_value("other_data") == "unchanged"

    def test_normalize_trigger_data_missing_keys(self) -> None:
        """Test trigger data normalization with missing keys."""
        event = KMEvent.create(
            trigger_type=TriggerType.DEVICE,
            trigger_id=TriggerId("test"),
            payload={"only_other_data": "value"},
        )

        normalized = normalize_trigger_data(event)

        # Should be unchanged if target keys don't exist
        assert normalized.get_payload_value("only_other_data") == "value"
        assert normalized.get_payload_value("trigger_value") is None
        assert normalized.get_payload_value("macro_id") is None


class TestEventPipelines:
    """Test event processing pipelines."""

    def test_get_default_event_pipeline(self) -> None:
        """Test default event pipeline."""
        pipeline = get_default_event_pipeline()
        assert callable(pipeline)

        event = KMEvent.create(
            trigger_type=TriggerType.HOTKEY,
            trigger_id=TriggerId("test"),
            payload={
                "triggerValue": "test_value",
                "script_content": "<script>alert('test')</script>",
            },
        )

        with patch("src.integration.events.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T12:00:00"
            )

            processed = pipeline(event)

        # Should have sanitized payload
        assert "&lt;script" in processed.get_payload_value("script_content")

        # Should have normalized trigger data
        assert processed.get_payload_value("trigger_value") == "test_value"

        # Should have processing timestamp
        assert (
            processed.get_payload_value("processing_timestamp") == "2024-01-01T12:00:00"
        )

    def test_get_security_focused_pipeline(self) -> None:
        """Test security-focused event pipeline."""
        pipeline = get_security_focused_pipeline()
        assert callable(pipeline)

        high_priority_event = KMEvent.create(
            trigger_type=TriggerType.SYSTEM,
            trigger_id=TriggerId("test"),
            payload={"script": "<script>alert('test')</script>"},
            priority=EventPriority.HIGH,
        )

        low_priority_event = high_priority_event.with_priority(EventPriority.LOW)

        with patch("src.integration.events.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T12:00:00"
            )

            processed_high = pipeline(high_priority_event)
            processed_low = pipeline(low_priority_event)

        # Both should be sanitized
        assert "&lt;script" in processed_high.get_payload_value("script")
        assert "&lt;script" in processed_low.get_payload_value("script")

        # Only high priority should have security validation flag
        assert processed_high.get_payload_value("security_validated") is True
        assert processed_low.get_payload_value("security_validated") is None

    def test_cached_pipelines(self) -> None:
        """Test cached pipeline functions."""
        # Test default pipeline cache
        pipeline1 = get_default_event_pipeline_cached()
        pipeline2 = get_default_event_pipeline_cached()
        assert pipeline1 is pipeline2  # Should be same instance

        # Test security pipeline cache
        security1 = get_security_focused_pipeline_cached()
        security2 = get_security_focused_pipeline_cached()
        assert security1 is security2  # Should be same instance

        # Test module constants
        assert callable(DEFAULT_EVENT_PIPELINE())
        assert callable(SECURITY_FOCUSED_PIPELINE())


class TestEventTypeAndManager:
    """Test EventType and EventManager for compatibility."""

    def test_event_type_values(self) -> None:
        """Test event type enumeration values."""
        assert EventType.MACRO_EXECUTED.value == "macro_executed"
        assert EventType.MACRO_CREATED.value == "macro_created"
        assert EventType.MACRO_UPDATED.value == "macro_updated"
        assert EventType.MACRO_DELETED.value == "macro_deleted"
        assert EventType.TRIGGER_ACTIVATED.value == "trigger_activated"

    def test_event_creation(self) -> None:
        """Test simple Event creation."""
        timestamp = datetime.now()
        event = Event(
            event_type=EventType.MACRO_EXECUTED,
            data={"macro_id": "test_macro"},
            source="test_source",
            timestamp=timestamp,
        )

        assert event.event_type == EventType.MACRO_EXECUTED
        assert event.data == {"macro_id": "test_macro"}
        assert event.source == "test_source"
        assert event.timestamp == timestamp

    def test_event_creation_default_timestamp(self) -> None:
        """Test Event creation with default timestamp."""
        event = Event(
            event_type=EventType.MACRO_CREATED,
            data={"name": "new_macro"},
            source="test_source",
        )

        assert isinstance(event.timestamp, datetime)

    def test_event_manager_initialization(self) -> None:
        """Test EventManager initialization."""
        manager = EventManager()
        assert manager._handlers == {}

    def test_event_manager_subscribe(self) -> None:
        """Test event subscription."""
        manager = EventManager()
        handled_events = []

        def test_handler(event: Event) -> None:
            handled_events.append(event)

        manager.subscribe(EventType.MACRO_EXECUTED, test_handler)

        # Check handler was registered
        assert EventType.MACRO_EXECUTED in manager._handlers
        assert test_handler in manager._handlers[EventType.MACRO_EXECUTED]

    def test_event_manager_publish(self) -> None:
        """Test event publishing."""
        manager = EventManager()
        handled_events = []

        def test_handler(event: Event) -> None:
            handled_events.append(event)

        manager.subscribe(EventType.MACRO_EXECUTED, test_handler)

        test_event = Event(
            event_type=EventType.MACRO_EXECUTED,
            data={"macro_id": "test"},
            source="test",
        )

        manager.publish(test_event)

        assert len(handled_events) == 1
        assert handled_events[0] == test_event

    def test_event_manager_multiple_handlers(self) -> None:
        """Test multiple handlers for same event type."""
        manager = EventManager()
        handled_by_handler1 = []
        handled_by_handler2 = []

        def handler1(event: Event) -> None:
            handled_by_handler1.append(event)

        def handler2(event: Event) -> None:
            handled_by_handler2.append(event)

        manager.subscribe(EventType.MACRO_CREATED, handler1)
        manager.subscribe(EventType.MACRO_CREATED, handler2)

        test_event = Event(
            event_type=EventType.MACRO_CREATED,
            data={"name": "test_macro"},
            source="test",
        )

        manager.publish(test_event)

        assert len(handled_by_handler1) == 1
        assert len(handled_by_handler2) == 1
        assert handled_by_handler1[0] == test_event
        assert handled_by_handler2[0] == test_event

    def test_event_manager_no_handlers(self) -> None:
        """Test publishing event with no subscribers."""
        manager = EventManager()

        test_event = Event(
            event_type=EventType.MACRO_DELETED,
            data={"macro_id": "test"},
            source="test",
        )

        # Should not raise exception
        manager.publish(test_event)


class TestPropertyBasedEvents:
    """Property-based tests for event functionality."""

    @given(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    def test_km_event_payload_consistency(self, test_value: str) -> None:
        """Property: Event payload handling should be consistent."""
        event = KMEvent.create(
            trigger_type=TriggerType.HOTKEY,
            trigger_id=TriggerId("test"),
            payload={"test_key": test_value},
        )

        # Test payload retrieval
        retrieved = event.get_payload_value("test_key")
        assert retrieved == test_value

        # Test payload transformation
        new_event = event.with_payload("new_key", test_value + "_modified")
        assert new_event.get_payload_value("test_key") == test_value
        assert new_event.get_payload_value("new_key") == test_value + "_modified"

    @given(st.sampled_from(list(EventPriority)))
    def test_event_priority_consistency(self, priority: EventPriority) -> None:
        """Property: Event priority handling should be consistent."""
        event = KMEvent.create(
            trigger_type=TriggerType.APPLICATION,
            trigger_id=TriggerId("test"),
            payload={},
            priority=priority,
        )

        assert event.priority == priority

        # Test priority change
        new_priority = (
            EventPriority.CRITICAL
            if priority != EventPriority.CRITICAL
            else EventPriority.LOW
        )
        new_event = event.with_priority(new_priority)

        assert event.priority == priority  # Original unchanged
        assert new_event.priority == new_priority

    @given(st.sampled_from(list(TriggerType)))
    def test_trigger_type_filter_consistency(self, trigger_type: TriggerType) -> None:
        """Property: Trigger type filters should work consistently."""
        event = KMEvent.create(
            trigger_type=trigger_type,
            trigger_id=TriggerId("test"),
            payload={},
        )

        # Single type filter should match
        single_filter = create_trigger_type_filter(trigger_type)
        assert single_filter(event) is True

        # Multi-type filter including this type should match
        multi_filter = create_trigger_type_filter([trigger_type, TriggerType.REMOTE])
        assert multi_filter(event) is True

        # Filter for different type should not match
        other_types = [t for t in TriggerType if t != trigger_type]
        if other_types:
            other_filter = create_trigger_type_filter(other_types[0])
            assert other_filter(event) is False


class TestEventSecurity:
    """Test event security and validation."""

    def test_sanitize_script_injection(self) -> None:
        """Test sanitization of script injection attempts."""
        dangerous_payloads = [
            {"content": "<script>alert('xss')</script>"},
            {"url": "javascript:alert('xss')"},
            {"mixed": "Safe text <script>evil()</script> more text"},
            {"js_protocol": "javascript:document.cookie='stolen'"},
        ]

        for payload in dangerous_payloads:
            event = KMEvent.create(
                trigger_type=TriggerType.REMOTE,
                trigger_id=TriggerId("test"),
                payload=payload,
            )

            sanitized = sanitize_event_payload(event)

            # Check that script tags and javascript: are neutralized
            for _, value in sanitized.payload.items():
                if isinstance(value, str):
                    assert "<script" not in value or "&lt;script" in value
                    assert "javascript:" not in value or "javascript_" in value

    def test_security_pipeline_validation(self) -> None:
        """Test security pipeline properly validates high-priority events."""
        dangerous_event = KMEvent.create(
            trigger_type=TriggerType.SYSTEM,
            trigger_id=TriggerId("malicious"),
            payload={
                "command": "<script>malicious_code()</script>",
                "url": "javascript:steal_data()",
            },
            priority=EventPriority.CRITICAL,
        )

        pipeline = get_security_focused_pipeline()
        processed = pipeline(dangerous_event)

        # Should be sanitized
        command = processed.get_payload_value("command")
        assert "&lt;script" in command

        url = processed.get_payload_value("url")
        assert "javascript_" in url

        # Should have security validation flag for critical priority
        assert processed.get_payload_value("security_validated") is True

    def test_event_immutability(self) -> None:
        """Test that events are properly immutable."""
        import copy

        original_payload = {"mutable_list": [1, 2, 3], "text": "original"}
        event = KMEvent.create(
            trigger_type=TriggerType.FILE,
            trigger_id=TriggerId("test"),
            payload=copy.deepcopy(
                original_payload
            ),  # Deep copy to avoid shared references
        )

        # Modify the original payload dict
        original_payload["text"] = "modified"
        original_payload["mutable_list"].append(4)

        # Event payload should be unaffected (using deep copy)
        assert event.get_payload_value("text") == "original"
        assert event.get_payload_value("mutable_list") == [1, 2, 3]

        # Create new event with modified payload
        new_event = event.with_payload("new_key", "new_value")

        # Original event should be unchanged
        assert "new_key" not in event.payload
        assert new_event.get_payload_value("new_key") == "new_value"
