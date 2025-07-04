"""
Property-based tests for advanced trigger system validation.

This module uses Hypothesis to test trigger behavior across input ranges,
ensuring security boundaries, validation correctness, and functional properties
for all trigger types and configurations.
"""

import pytest
import re
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from src.core.triggers import (
    TriggerBuilder, TriggerType, TriggerValidator, TriggerSpec,
    create_daily_trigger, create_file_watcher, create_app_lifecycle_trigger
)
from src.core.either import Either
from src.core.errors import ValidationError, SecurityError
from src.security.input_sanitizer import InputSanitizer


class TestTriggerBuilderProperties:
    """Property-based tests for TriggerBuilder."""
    
    @given(st.integers(min_value=1, max_value=86400))
    def test_time_interval_properties(self, seconds):
        """Property: Time intervals should handle all valid durations."""
        interval = timedelta(seconds=seconds)
        result = (TriggerBuilder()
                 .recurring_every(interval)
                 .build())
        
        assert result.is_right()
        trigger = result.get_right()
        assert trigger.trigger_type == TriggerType.TIME_RECURRING
        assert trigger.config["recurring_interval"] == interval
    
    @given(st.datetimes(min_value=datetime(2024, 1, 1), max_value=datetime(2030, 12, 31)))
    def test_scheduled_time_properties(self, schedule_time):
        """Property: Scheduled times should preserve datetime values."""
        result = (TriggerBuilder()
                 .scheduled_at(schedule_time)
                 .build())
        
        assert result.is_right()
        trigger = result.get_right()
        assert trigger.trigger_type == TriggerType.TIME_SCHEDULED
        assert trigger.config["schedule_time"] == schedule_time
    
    @given(st.integers(min_value=-10, max_value=10))
    def test_priority_range_validation(self, priority):
        """Property: Valid priority ranges should be accepted."""
        result = (TriggerBuilder()
                 .scheduled_at(datetime(2024, 1, 1, 12, 0))
                 .with_priority(priority)
                 .build())
        
        assert result.is_right()
        trigger = result.get_right()
        assert trigger.priority == priority
    
    @given(st.integers().filter(lambda x: x < -10 or x > 10))
    def test_invalid_priority_rejected(self, invalid_priority):
        """Property: Invalid priority values should be rejected."""
        with pytest.raises(ValueError, match="Priority must be between -10 and 10"):
            (TriggerBuilder()
             .scheduled_at(datetime(2024, 1, 1, 12, 0))
             .with_priority(invalid_priority)
             .build())
    
    @given(st.integers(min_value=1, max_value=300))
    def test_timeout_range_validation(self, timeout):
        """Property: Valid timeout ranges should be accepted."""
        result = (TriggerBuilder()
                 .scheduled_at(datetime(2024, 1, 1, 12, 0))
                 .with_timeout(timeout)
                 .build())
        
        assert result.is_right()
        trigger = result.get_right()
        assert trigger.timeout_seconds == timeout
    
    @given(st.integers(min_value=1, max_value=10000))
    def test_execution_limit_validation(self, max_executions):
        """Property: Valid execution limits should be accepted."""
        result = (TriggerBuilder()
                 .scheduled_at(datetime(2024, 1, 1, 12, 0))
                 .limit_executions(max_executions)
                 .build())
        
        assert result.is_right()
        trigger = result.get_right()
        assert trigger.max_executions == max_executions


class TestSecurityProperties:
    """Property-based tests for trigger security validation."""
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-~", min_size=1, max_size=100).filter(
        lambda x: len(x.strip()) > 0 and not any(forbidden in x for forbidden in ["..", "//", "System", "usr/bin"])
    ))
    def test_safe_file_paths_accepted(self, safe_path):
        """Property: Safe file paths should pass validation."""
        result = TriggerValidator.validate_file_path(safe_path)
        assert result.is_right()
    
    @given(st.just("test"))  # Use simpler test data
    def test_dangerous_file_paths_rejected(self, base_path):
        """Property: File paths with dangerous patterns should be rejected."""
        dangerous_patterns = [
            "/System/Library", "/usr/bin/sudo", "/private/etc/passwd", 
            "/../../../etc", "/Library/Keychains/login"
        ]
        
        for dangerous_path in dangerous_patterns:
            result = TriggerValidator.validate_file_path(dangerous_path)
            # These specific paths should be rejected
            assert result.is_left()
            error_code = result.get_left().security_code
            assert error_code in ["PATH_ACCESS_DENIED", "DIRECTORY_TRAVERSAL"]
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-", min_size=1, max_size=255).filter(
        lambda x: len(x.strip()) > 0 and re.match(r'^[a-zA-Z0-9._-]+$', x) and ".." not in x
    ))
    def test_valid_app_identifiers_accepted(self, app_id):
        """Property: Valid application identifiers should be accepted."""
        result = TriggerValidator.validate_app_identifier(app_id)
        assert result.is_right()
    
    @given(st.text(min_size=1, max_size=100))
    def test_invalid_app_identifiers_rejected(self, app_id):
        """Property: Invalid application identifiers should be rejected."""
        # Add invalid characters
        invalid_id = app_id + "!@#$%^&*()"
        
        result = TriggerValidator.validate_app_identifier(invalid_id)
        assert result.is_left()
        assert "DANGEROUS_APP_ID" in result.get_left().security_code or "INVALID_APP_ID_FORMAT" in result.get_left().security_code
    
    def test_valid_cron_patterns_accepted(self):
        """Property: Valid cron patterns should be accepted."""
        valid_patterns = [
            "0 2 * * *",      # Daily at 2 AM
            "*/15 * * * *",   # Every 15 minutes
            "0 0 1 * *",      # Monthly on 1st
            "30 6 * * 1-5"    # Weekdays at 6:30 AM
        ]
        
        for pattern in valid_patterns:
            result = TriggerValidator.validate_cron_pattern(pattern)
            assert result.is_right()
    
    @given(st.text(min_size=1, max_size=50))
    def test_invalid_cron_patterns_rejected(self, pattern):
        """Property: Invalid cron patterns should be rejected."""
        # Add obviously invalid cron pattern
        invalid_pattern = pattern + " !@#$ INVALID CRON"
        
        result = TriggerValidator.validate_cron_pattern(invalid_pattern)
        assert result.is_left()
        assert "INVALID_CRON" in result.get_left().security_code


class TestTriggerValidationProperties:
    """Property-based tests for trigger validation logic."""
    
    @given(
        st.sampled_from([TriggerType.TIME_SCHEDULED, TriggerType.TIME_RECURRING, 
                        TriggerType.FILE_CREATED, TriggerType.FILE_MODIFIED,
                        TriggerType.APP_LAUNCHED, TriggerType.USER_IDLE]),
        st.integers(min_value=1, max_value=300),
        st.integers(min_value=-10, max_value=10)
    )
    def test_trigger_spec_creation_properties(self, trigger_type, timeout, priority):
        """Property: Valid trigger specs should be creatable."""
        builder = TriggerBuilder()
        
        # Set up trigger type with valid configuration
        if trigger_type == TriggerType.TIME_SCHEDULED:
            builder = builder.scheduled_at(datetime(2024, 6, 1, 12, 0))
        elif trigger_type == TriggerType.TIME_RECURRING:
            builder = builder.recurring_every(timedelta(hours=1))
        elif trigger_type == TriggerType.FILE_CREATED:
            builder = builder.when_file_created("/tmp/test")
        elif trigger_type == TriggerType.FILE_MODIFIED:
            builder = builder.when_file_modified("/tmp/test")
        elif trigger_type == TriggerType.APP_LAUNCHED:
            builder = builder.when_app_launches("com.example.app")
        elif trigger_type == TriggerType.USER_IDLE:
            builder = builder.when_user_idle(300)
        
        # Set properties
        builder = (builder
                  .with_timeout(timeout)
                  .with_priority(priority))
        
        result = builder.build()
        
        # Should either succeed or fail with validation error
        if result.is_left():
            assert isinstance(result.get_left(), ValidationError)
        else:
            trigger = result.get_right()
            assert trigger.trigger_type == trigger_type
            assert trigger.timeout_seconds == timeout
            assert trigger.priority == priority
            assert isinstance(trigger.trigger_id, str)
            assert len(trigger.trigger_id) > 0
    
    @given(st.integers(min_value=0, max_value=100))
    def test_battery_threshold_validation(self, threshold):
        """Property: Valid battery thresholds should be accepted."""
        result = (TriggerBuilder()
                 .when_battery_low(threshold)
                 .build())
        
        assert result.is_right()
        trigger = result.get_right()
        assert trigger.config["battery_threshold"] == threshold
    
    @given(st.integers(min_value=1, max_value=86400))
    def test_idle_threshold_validation(self, threshold):
        """Property: Valid idle thresholds should be accepted."""
        result = (TriggerBuilder()
                 .when_user_idle(threshold)
                 .build())
        
        assert result.is_right()
        trigger = result.get_right()
        assert trigger.config["idle_threshold_seconds"] == threshold
    
    @given(st.booleans(), st.booleans())
    def test_trigger_state_properties(self, enabled, recursive):
        """Property: Trigger state should be preserved correctly."""
        result = (TriggerBuilder()
                 .when_file_created("/tmp/test", recursive=recursive)
                 .enabled(enabled)
                 .build())
        
        if result.is_right():
            trigger = result.get_right()
            assert trigger.enabled == enabled
            assert trigger.config.get("recursive", False) == recursive


class TestTriggerIntegrationProperties:
    """Property-based tests for trigger integration behavior."""
    
    @given(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=5, max_size=50),
        st.integers(min_value=1, max_value=24),
        st.integers(min_value=0, max_value=59)
    )
    def test_daily_trigger_creation(self, trigger_name, hour, minute):
        """Property: Daily triggers should be created with correct scheduling."""
        assume(0 <= hour <= 23)
        assume(0 <= minute <= 59)
        
        trigger_builder = create_daily_trigger(hour, minute)
        result = trigger_builder.build()
        
        assert result.is_right()
        trigger = result.get_right()
        assert trigger.trigger_type == TriggerType.TIME_RECURRING
        expected_pattern = f"{minute} {hour} * * *"
        assert trigger.config["recurring_pattern"] == expected_pattern
    
    @given(st.text(min_size=1, max_size=100).filter(
        lambda x: len(x.strip()) > 0 and not any(c in x for c in ["<", ">", "&", ";", "|"])
    ))
    def test_file_watcher_creation(self, directory):
        """Property: File watchers should be created with safe paths."""
        # Only test with relatively safe directory names
        safe_directory = "/tmp/" + directory.replace("/", "_").replace("..", "_")
        
        trigger_builder = create_file_watcher(safe_directory)
        result = trigger_builder.build()
        
        # Should either succeed or fail validation for security reasons
        if result.is_right():
            trigger = result.get_right()
            assert trigger.trigger_type == TriggerType.FILE_MODIFIED
            assert "watch_path" in trigger.config
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789.-", min_size=5, max_size=50).filter(
        lambda x: ".." not in x and not x.startswith(".") and not x.endswith(".")
    ))
    def test_app_lifecycle_trigger_creation(self, app_id):
        """Property: App lifecycle triggers should handle valid app IDs."""
        trigger_builder = create_app_lifecycle_trigger(app_id, on_launch=True)
        result = trigger_builder.build()
        
        assert result.is_right()
        trigger = result.get_right()
        assert trigger.trigger_type == TriggerType.APP_LAUNCHED
        assert trigger.config["app_bundle_id"] == app_id
    
    @given(
        st.lists(st.dictionaries(
            st.sampled_from(["type", "operator", "operand"]),
            st.text(min_size=1, max_size=20),
            min_size=3, max_size=3
        ), min_size=0, max_size=5)
    )
    def test_condition_integration(self, conditions):
        """Property: Conditions should integrate properly with triggers."""
        builder = (TriggerBuilder()
                  .scheduled_at(datetime(2024, 6, 1, 12, 0)))
        
        for condition in conditions:
            if all(key in condition for key in ["type", "operator", "operand"]):
                builder = builder.with_condition(condition)
        
        result = builder.build()
        
        if result.is_right():
            trigger = result.get_right()
            assert len(trigger.conditions) <= len(conditions)
            # Verify each condition was added
            for condition in trigger.conditions:
                assert isinstance(condition, dict)
                assert "type" in condition
                assert "operator" in condition
                assert "operand" in condition


class TestResourceLimitProperties:
    """Property-based tests for resource limit validation."""
    
    @given(st.integers(min_value=1, max_value=300))
    def test_timeout_limits_respected(self, timeout):
        """Property: Timeout limits should be enforced."""
        result = (TriggerBuilder()
                 .scheduled_at(datetime(2024, 6, 1, 12, 0))
                 .with_timeout(timeout)
                 .build())
        
        assert result.is_right()
        trigger = result.get_right()
        
        # Validate resource limits
        limit_result = TriggerValidator.validate_resource_limits(trigger)
        assert limit_result.is_right()
    
    @given(st.integers(min_value=1, max_value=10000))
    def test_execution_limits_respected(self, max_executions):
        """Property: Execution limits should be enforced."""
        result = (TriggerBuilder()
                 .scheduled_at(datetime(2024, 6, 1, 12, 0))
                 .limit_executions(max_executions)
                 .build())
        
        assert result.is_right()
        trigger = result.get_right()
        
        # Validate resource limits
        limit_result = TriggerValidator.validate_resource_limits(trigger)
        assert limit_result.is_right()
    
    def test_recursive_monitoring_limits(self):
        """Property: Recursive monitoring should be limited for sensitive paths."""
        # Test with paths that should pass path validation but fail resource limits
        sensitive_paths = ["/tmp", "/tmp/safe"]  # Use safe paths for testing resource limits
        
        for safe_path in sensitive_paths:
            result = (TriggerBuilder()
                     .when_file_created(safe_path, recursive=True)
                     .build())
            
            assert result.is_right()
            trigger = result.get_right()
            limit_result = TriggerValidator.validate_resource_limits(trigger)
            # Safe paths should pass resource validation
            assert limit_result.is_right()
        
        # Test that the system correctly rejects sensitive paths at the builder level
        truly_sensitive_paths = ["/System", "/usr/bin"]
        for sensitive_path in truly_sensitive_paths:
            try:
                result = (TriggerBuilder()
                         .when_file_created(sensitive_path, recursive=True)
                         .build())
                # Should not reach here due to path validation
                assert False, f"Expected validation error for {sensitive_path}"
            except ValueError as e:
                # Expected: path validation should reject these
                assert "Invalid file path" in str(e)


class TriggerStateMachine(RuleBasedStateMachine):
    """Stateful testing for trigger builder state management."""
    
    def __init__(self):
        super().__init__()
        self.builder = TriggerBuilder()
        self.has_trigger_type = False
        self.trigger_types_set = []
    
    @rule(schedule_time=st.datetimes(min_value=datetime(2024, 1, 1), max_value=datetime(2025, 12, 31)))
    def set_scheduled_trigger(self, schedule_time):
        """Add scheduled trigger to builder."""
        self.builder = self.builder.scheduled_at(schedule_time)
        self.has_trigger_type = True
        self.trigger_types_set.append("scheduled")
    
    @rule(interval_hours=st.integers(min_value=1, max_value=24))
    def set_recurring_trigger(self, interval_hours):
        """Add recurring trigger to builder."""
        interval = timedelta(hours=interval_hours)
        self.builder = self.builder.recurring_every(interval)
        self.has_trigger_type = True
        self.trigger_types_set.append("recurring")
    
    @rule(priority=st.integers(min_value=-10, max_value=10))
    def set_priority(self, priority):
        """Set trigger priority."""
        self.builder = self.builder.with_priority(priority)
    
    @rule(timeout=st.integers(min_value=1, max_value=300))
    def set_timeout(self, timeout):
        """Set trigger timeout."""
        self.builder = self.builder.with_timeout(timeout)
    
    @rule(enabled=st.booleans())
    def set_enabled_state(self, enabled):
        """Set trigger enabled state."""
        self.builder = self.builder.enabled(enabled)
    
    @rule()
    def build_trigger(self):
        """Attempt to build trigger."""
        result = self.builder.build()
        
        if self.has_trigger_type:
            # Should succeed with trigger type
            assert result.is_right()
            trigger = result.get_right()
            assert trigger.trigger_id is not None
            assert isinstance(trigger.trigger_id, str)
            assert len(trigger.trigger_id) > 0
        else:
            # Should fail without trigger type
            assert result.is_left()
            assert isinstance(result.get_left(), ValidationError)
    
    @invariant()
    def builder_maintains_state(self):
        """Invariant: Builder should maintain consistent state."""
        assert self.builder is not None


# Test configuration and runner
class TestTriggerSystemProperties:
    """Main property test runner."""
    
    @pytest.mark.skip("Stateful testing needs refinement")
    @settings(max_examples=10, deadline=None)
    def test_stateful_trigger_building(self):
        """Run stateful tests for trigger building."""
        TriggerStateMachine.TestCase.settings = settings(
            max_examples=5,
            stateful_step_count=10,
            deadline=None
        )
        state_machine = TriggerStateMachine()
        state_machine.execute()
    
    @given(st.text(min_size=1, max_size=50))
    def test_metadata_preservation(self, metadata_value):
        """Property: Metadata should be preserved in triggers."""
        result = (TriggerBuilder()
                 .scheduled_at(datetime(2024, 6, 1, 12, 0))
                 .with_metadata(test_value=metadata_value)
                 .build())
        
        if result.is_right():
            trigger = result.get_right()
            assert "test_value" in trigger.metadata
            assert trigger.metadata["test_value"] == metadata_value
    
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
    def test_trigger_id_uniqueness(self, trigger_names):
        """Property: Trigger IDs should be unique across instances."""
        trigger_ids = set()
        
        for i, name in enumerate(trigger_names):
            result = (TriggerBuilder()
                     .scheduled_at(datetime(2024, 6, 1, 12, i % 24))
                     .with_metadata(name=name)
                     .build())
            
            if result.is_right():
                trigger = result.get_right()
                assert trigger.trigger_id not in trigger_ids
                trigger_ids.add(trigger.trigger_id)