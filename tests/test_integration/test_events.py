"""
Tests for functional event system.

Tests immutable event handling, functional composition patterns,
and event processing for Keyboard Maestro integration.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.integration.events import (
    KMEvent, TriggerType, EventPriority, EventProcessingResult,
    compose_event_handlers, create_payload_transformer,
    create_conditional_handler, create_priority_filter,
    sanitize_event_payload, normalize_trigger_data,
    DEFAULT_EVENT_PIPELINE, SECURITY_FOCUSED_PIPELINE
)
from src.core.types import TriggerId, MacroId


@pytest.fixture
def sample_event():
    """Create sample KM event for testing."""
    return KMEvent.create(
        trigger_type=TriggerType.HOTKEY,
        trigger_id=TriggerId("test-trigger-123"),
        payload={"key": "cmd+space", "value": "test"},
        macro_id=MacroId("test-macro-456"),
        priority=EventPriority.NORMAL
    )


@pytest.fixture
def malicious_event():
    """Create event with malicious payload for security testing."""
    return KMEvent.create(
        trigger_type=TriggerType.APPLICATION,
        trigger_id=TriggerId("malicious-trigger"),
        payload={
            "script": "<script>alert('xss')</script>",
            "command": "javascript:void(0)",
            "triggerValue": "normal_value"
        }
    )


class TestKMEvent:
    """Test KMEvent immutable event structure."""
    
    def test_event_creation(self, sample_event):
        """Test basic event creation and properties."""
        assert sample_event.trigger_type == TriggerType.HOTKEY
        assert sample_event.trigger_id == TriggerId("test-trigger-123")
        assert sample_event.macro_id == MacroId("test-macro-456")
        assert sample_event.priority == EventPriority.NORMAL
        assert sample_event.get_payload_value("key") == "cmd+space"
        assert sample_event.get_payload_value("nonexistent", "default") == "default"
    
    def test_event_immutability(self, sample_event):
        """Test that events are immutable."""
        original_payload = sample_event.payload
        new_event = sample_event.with_payload("new_key", "new_value")
        
        # Original event unchanged
        assert sample_event.payload == original_payload
        assert "new_key" not in sample_event.payload
        
        # New event has additional data
        assert new_event.get_payload_value("new_key") == "new_value"
        assert new_event.get_payload_value("key") == "cmd+space"  # Original data preserved
    
    def test_priority_checking(self, sample_event):
        """Test priority level checking."""
        assert not sample_event.is_high_priority()
        
        high_priority_event = sample_event.with_priority(EventPriority.HIGH)
        assert high_priority_event.is_high_priority()
        
        critical_event = sample_event.with_priority(EventPriority.CRITICAL)
        assert critical_event.is_high_priority()
    
    def test_event_transformation(self, sample_event):
        """Test functional event transformation."""
        def add_timestamp(event):
            return event.with_payload("processed_at", "2025-06-28T10:00:00")
        
        transformed = sample_event.transform(add_timestamp)
        assert transformed.get_payload_value("processed_at") == "2025-06-28T10:00:00"
        assert transformed.get_payload_value("key") == "cmd+space"  # Original preserved


class TestEventProcessingResult:
    """Test event processing results."""
    
    def test_success_result_creation(self, sample_event):
        """Test creating successful processing results."""
        result = EventProcessingResult.success_result(
            sample_event,
            duration_ms=25.5,
            actions_count=2
        )
        
        assert result.success
        assert result.processed_event == sample_event
        assert result.processing_duration_ms == 25.5
        assert result.metadata["actions_count"] == 2
    
    def test_failure_result_creation(self):
        """Test creating failed processing results."""
        result = EventProcessingResult.failure_result(
            "Permission denied",
            duration_ms=5.0,
            error_code="PERMISSION_ERROR"
        )
        
        assert not result.success
        assert result.error_message == "Permission denied"
        assert result.processing_duration_ms == 5.0
        assert result.metadata["error_code"] == "PERMISSION_ERROR"


class TestFunctionalComposition:
    """Test functional event handler composition."""
    
    def test_compose_event_handlers(self, sample_event):
        """Test composing multiple event handlers."""
        def add_field1(event):
            return event.with_payload("field1", "value1")
        
        def add_field2(event):
            return event.with_payload("field2", "value2")
        
        def multiply_value(event):
            value = event.get_payload_value("value", "")
            return event.with_payload("value", f"{value}_processed")
        
        composed = compose_event_handlers(add_field1, add_field2, multiply_value)
        result = composed(sample_event)
        
        assert result.get_payload_value("field1") == "value1"
        assert result.get_payload_value("field2") == "value2"
        assert result.get_payload_value("value") == "test_processed"
    
    def test_empty_composition(self, sample_event):
        """Test composition with no handlers."""
        composed = compose_event_handlers()
        result = composed(sample_event)
        
        assert result == sample_event  # Identity function
    
    def test_payload_transformer(self, sample_event):
        """Test payload field transformation."""
        uppercase_key = create_payload_transformer("key", str.upper)
        result = uppercase_key(sample_event)
        
        assert result.get_payload_value("key") == "CMD+SPACE"
        assert result.get_payload_value("value") == "test"  # Other fields unchanged
    
    def test_conditional_handler(self, sample_event):
        """Test conditional event handling."""
        def is_hotkey_event(event):
            return event.trigger_type == TriggerType.HOTKEY
        
        def add_hotkey_metadata(event):
            return event.with_payload("is_hotkey", True)
        
        conditional = create_conditional_handler(is_hotkey_event, add_hotkey_metadata)
        
        # Should apply to hotkey event
        result1 = conditional(sample_event)
        assert result1.get_payload_value("is_hotkey") is True
        
        # Should not apply to non-hotkey event
        app_event = sample_event.with_payload("temp", "temp")
        app_event = KMEvent.create(
            TriggerType.APPLICATION,
            TriggerId("app-trigger"),
            {"app": "TestApp"}
        )
        result2 = conditional(app_event)
        assert result2.get_payload_value("is_hotkey") is None


class TestEventFilters:
    """Test event filtering functions."""
    
    def test_priority_filter(self, sample_event):
        """Test priority-based filtering."""
        high_filter = create_priority_filter(EventPriority.HIGH)
        
        # Normal priority should not pass high filter
        assert not high_filter(sample_event)
        
        # High priority should pass
        high_event = sample_event.with_priority(EventPriority.HIGH)
        assert high_filter(high_event)
        
        # Critical priority should pass high filter
        critical_event = sample_event.with_priority(EventPriority.CRITICAL)
        assert high_filter(critical_event)
    
    def test_trigger_type_filter(self, sample_event):
        """Test trigger type filtering."""
        from src.integration.events import create_trigger_type_filter
        
        # Single type filter
        hotkey_filter = create_trigger_type_filter(TriggerType.HOTKEY)
        assert hotkey_filter(sample_event)
        
        app_event = KMEvent.create(
            TriggerType.APPLICATION,
            TriggerId("app-trigger"),
            {"app": "TestApp"}
        )
        assert not hotkey_filter(app_event)
        
        # Multiple type filter
        multi_filter = create_trigger_type_filter([TriggerType.HOTKEY, TriggerType.APPLICATION])
        assert multi_filter(sample_event)
        assert multi_filter(app_event)


class TestBuiltInTransformations:
    """Test built-in event transformation functions."""
    
    def test_sanitize_event_payload(self, malicious_event):
        """Test payload sanitization."""
        sanitized = sanitize_event_payload(malicious_event)
        
        script_value = sanitized.get_payload_value("script")
        assert "<script" not in script_value
        assert "&lt;script" in script_value
        
        command_value = sanitized.get_payload_value("command")
        assert "javascript:" not in command_value
        assert "javascript_" in command_value
        
        # Normal values should be preserved
        assert sanitized.get_payload_value("triggerValue") == "normal_value"
    
    def test_normalize_trigger_data(self):
        """Test trigger data normalization."""
        event = KMEvent.create(
            TriggerType.REMOTE,
            TriggerId("remote-trigger"),
            {
                "triggerValue": "old_format",
                "macroUID": "12345-abcde",
                "normalField": "stays_same"
            }
        )
        
        normalized = normalize_trigger_data(event)
        
        # Should convert camelCase to snake_case
        assert normalized.get_payload_value("trigger_value") == "old_format"
        assert normalized.get_payload_value("macro_id") == "12345-abcde"
        assert normalized.get_payload_value("normalField") == "stays_same"
        
        # Old fields should be removed
        assert normalized.get_payload_value("triggerValue") is None
        assert normalized.get_payload_value("macroUID") is None
    
    def test_add_processing_timestamp(self, sample_event):
        """Test adding processing timestamp."""
        from src.integration.events import add_processing_timestamp
        
        before_time = datetime.now()
        result = add_processing_timestamp(sample_event)
        after_time = datetime.now()
        
        timestamp_str = result.get_payload_value("processing_timestamp")
        assert timestamp_str is not None
        
        # Parse timestamp and verify it's recent
        timestamp = datetime.fromisoformat(timestamp_str)
        assert before_time <= timestamp <= after_time


class TestEventPipelines:
    """Test pre-built event processing pipelines."""
    
    def test_default_pipeline(self, malicious_event):
        """Test default event processing pipeline."""
        # Get the actual pipeline function
        pipeline = DEFAULT_EVENT_PIPELINE()
        result = pipeline(malicious_event)
        
        # Should be sanitized
        assert "&lt;script" in result.get_payload_value("script")
        
        # Should be normalized (if it had camelCase fields)
        # Should have processing timestamp
        assert result.get_payload_value("processing_timestamp") is not None
    
    def test_security_focused_pipeline(self, sample_event):
        """Test security-focused pipeline."""
        # Get the actual pipeline function
        pipeline = SECURITY_FOCUSED_PIPELINE()
        
        high_priority_event = sample_event.with_priority(EventPriority.HIGH)
        result = pipeline(high_priority_event)
        
        # High priority events should get security validation
        assert result.get_payload_value("security_validated") is True
        assert result.get_payload_value("processing_timestamp") is not None
        
        # Normal priority events should not get extra security marking
        normal_result = pipeline(sample_event)
        assert normal_result.get_payload_value("security_validated") is None


# Property-based testing for event immutability
@pytest.mark.parametrize("field_name,field_value", [
    ("test_field", "test_value"),
    ("number_field", 42),
    ("bool_field", True),
    ("dict_field", {"nested": "value"}),
])
def test_event_immutability_property(sample_event, field_name, field_value):
    """Property test: Events should always remain immutable."""
    original_payload = sample_event.payload.copy()
    
    # Add new field
    new_event = sample_event.with_payload(field_name, field_value)
    
    # Original should be unchanged
    assert sample_event.payload == original_payload
    
    # New event should have additional field
    assert new_event.get_payload_value(field_name) == field_value
    
    # All original fields should be preserved
    for key, value in original_payload.items():
        assert new_event.get_payload_value(key) == value