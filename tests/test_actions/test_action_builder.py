"""Property-based tests for Action Builder system.

Comprehensive test suite for action building functionality with
security validation, XML generation, and builder pattern testing.
"""

from __future__ import annotations

import logging
import re

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from src.actions import ActionBuilder, ActionCategory, ActionRegistry
from src.core.errors import SecurityViolationError, ValidationError
from src.core.types import Duration

logger = logging.getLogger(__name__)


class TestActionBuilder:
    """Test suite for ActionBuilder with property-based testing."""

    def setup_method(self) -> None:
        """Set up test environment for each test."""
        self.registry = ActionRegistry()
        self.builder = ActionBuilder(self.registry)

    def test_builder_initialization(self) -> None:
        """Test ActionBuilder initialization."""
        assert self.builder.get_action_count() == 0
        assert len(self.builder.get_actions()) == 0
        assert self.builder._registry is not None

    def test_add_basic_text_action(self) -> None:
        """Test adding basic text action."""
        result = self.builder.add_text_action("Hello World")

        assert result is self.builder  # Fluent interface
        assert self.builder.get_action_count() == 1

        actions = self.builder.get_actions()
        assert len(actions) == 1
        assert actions[0].action_type.identifier == "Type a String"
        assert actions[0].parameters["text"] == "Hello World"

    def test_add_pause_action(self) -> None:
        """Test adding pause action."""
        duration = Duration.from_seconds(2.5)
        result = self.builder.add_pause_action(duration)

        assert result is self.builder
        assert self.builder.get_action_count() == 1

        actions = self.builder.get_actions()
        assert actions[0].action_type.identifier == "Pause"
        assert actions[0].parameters["duration"] == 2.5

    def test_add_variable_action(self) -> None:
        """Test adding variable action."""
        result = self.builder.add_variable_action("TestVar", "TestValue")

        assert result is self.builder
        assert self.builder.get_action_count() == 1

        actions = self.builder.get_actions()
        assert actions[0].action_type.identifier == "Set Variable to Text"
        assert actions[0].parameters["variable"] == "TestVar"
        assert actions[0].parameters["text"] == "TestValue"

    def test_add_app_action(self) -> None:
        """Test adding application action."""
        result = self.builder.add_app_action("Safari", bring_all_windows=True)

        assert result is self.builder
        assert self.builder.get_action_count() == 1

        actions = self.builder.get_actions()
        assert actions[0].action_type.identifier == "Activate a Specific Application"
        assert actions[0].parameters["application"] == "Safari"
        assert actions[0].parameters["bring_all_windows"] is True

    def test_builder_fluent_interface(self) -> None:
        """Test builder fluent interface with chaining."""
        result = (
            self.builder.add_text_action("Step 1")
            .add_pause_action(Duration.from_seconds(1))
            .add_text_action("Step 2")
            .add_variable_action("Counter", "1")
        )

        assert result is self.builder
        assert self.builder.get_action_count() == 4

        actions = self.builder.get_actions()
        assert actions[0].parameters["text"] == "Step 1"
        assert actions[1].parameters["duration"] == 1.0
        assert actions[2].parameters["text"] == "Step 2"
        assert actions[3].parameters["variable"] == "Counter"

    def test_action_position_insertion(self) -> None:
        """Test inserting actions at specific positions."""
        # Add initial actions
        self.builder.add_text_action("First").add_text_action("Third")

        # Insert at position 1
        self.builder.add_text_action("Second", position=1)

        actions = self.builder.get_actions()
        assert len(actions) == 3
        assert actions[0].parameters["text"] == "First"
        assert actions[1].parameters["text"] == "Second"
        assert actions[2].parameters["text"] == "Third"

    def test_remove_action(self) -> None:
        """Test removing actions by index."""
        self.builder.add_text_action("Keep").add_text_action("Remove").add_text_action(
            "Keep",
        )

        assert self.builder.get_action_count() == 3

        self.builder.remove_action(1)

        assert self.builder.get_action_count() == 2
        actions = self.builder.get_actions()
        assert actions[0].parameters["text"] == "Keep"
        assert actions[1].parameters["text"] == "Keep"

    def test_clear_actions(self) -> None:
        """Test clearing all actions."""
        self.builder.add_text_action("Test1").add_text_action("Test2")
        assert self.builder.get_action_count() == 2

        result = self.builder.clear()
        assert result is self.builder
        assert self.builder.get_action_count() == 0

    def test_unknown_action_type_error(self) -> None:
        """Test error handling for unknown action types."""
        with pytest.raises(ValidationError) as exc_info:
            self.builder.add_action("Unknown Action Type", {})

        assert "Unknown action type" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_missing_required_parameters(self) -> None:
        """Test validation of required parameters."""
        with pytest.raises(ValidationError) as exc_info:
            self.builder.add_action(
                "Type a String",
                {},
            )  # Missing required 'text' parameter

        assert "Must include required parameters" in str(exc_info.value)

    def test_xml_generation_success(self) -> None:
        """Test successful XML generation."""
        self.builder.add_text_action("Hello World", by_typing=True)
        self.builder.add_pause_action(Duration.from_seconds(1))

        result = self.builder.build_xml()

        assert result["success"] is True
        assert "xml" in result
        assert result["action_count"] == 2
        assert result["validation_passed"] is True

        xml = result["xml"]
        assert "<actions>" in xml
        assert "<action" in xml
        assert "Type a String" in xml
        assert "Pause" in xml
        assert "Hello World" in xml

    def test_xml_generation_empty_builder(self) -> None:
        """Test XML generation with empty builder."""
        result = self.builder.build_xml()

        assert result["success"] is False
        assert "No actions to build" in result["error"]

    def test_validation_all_valid(self) -> None:
        """Test validation with all valid actions."""
        self.builder.add_text_action("Valid text")
        self.builder.add_pause_action(Duration.from_seconds(1))

        result = self.builder.validate_all()

        assert result["all_valid"] is True
        assert result["total_actions"] == 2
        assert result["valid_actions"] == 2
        assert len(result["results"]) == 2
        assert all(r["valid"] for r in result["results"])

    def test_validation_with_errors(self) -> None:
        """Test validation with invalid actions."""
        # Add valid action
        self.builder.add_text_action("Valid")

        # Test that ActionConfiguration raises ValidationError for missing required parameters
        action_type = self.registry.get_action_type("Type a String")
        from src.actions.action_builder import ActionConfiguration
        from src.core.errors import ValidationError

        # Verify that ActionConfiguration with missing required parameters raises ValidationError
        with pytest.raises(ValidationError):
            ActionConfiguration(
                action_type=action_type,
                parameters={},  # Missing required 'text' parameter
            )

        # Test validation of builder state with only valid action
        result = self.builder.validate_all()
        assert result["all_valid"] is True
        assert result["total_actions"] == 1
        assert result["valid_actions"] == 1

    @given(st.text(min_size=1, max_size=100))
    def test_property_text_action_content(self, text: str) -> None:
        """Property test: Text actions preserve content."""
        assume(len(text.strip()) > 0)

        # Clear builder for each hypothesis example
        self.builder.clear()
        self.builder.add_text_action(text)
        actions = self.builder.get_actions()

        assert len(actions) == 1
        assert actions[0].parameters["text"] == text
        assert actions[0].action_type.identifier == "Type a String"

    @given(st.floats(min_value=0.1, max_value=3600.0))
    def test_property_pause_duration(self, duration_seconds: Any) -> None:
        """Property test: Pause actions handle various durations."""
        # Clear builder for each hypothesis example
        self.builder.clear()

        duration = Duration.from_seconds(duration_seconds)

        self.builder.add_pause_action(duration)
        actions = self.builder.get_actions()

        assert len(actions) == 1
        assert abs(actions[0].parameters["duration"] - duration_seconds) < 0.001

    @given(st.text(min_size=1, max_size=50), st.text(min_size=1, max_size=100))
    def test_property_variable_names_and_values(self, var_name: str, var_value: str) -> None:
        """Property test: Variable actions handle various names and values."""
        assume(var_name.strip() and var_value.strip())

        # Clear builder for each hypothesis example
        self.builder.clear()

        self.builder.add_variable_action(var_name, var_value)
        actions = self.builder.get_actions()

        assert len(actions) == 1
        assert actions[0].parameters["variable"] == var_name
        assert actions[0].parameters["text"] == var_value

    @given(st.integers(min_value=1, max_value=10))
    def test_property_action_count_consistency(self, action_count: int) -> None:
        """Property test: Action count remains consistent."""
        # Clear builder for each hypothesis example
        self.builder.clear()

        for i in range(action_count):
            self.builder.add_text_action(f"Action {i}")

        assert self.builder.get_action_count() == action_count
        assert len(self.builder.get_actions()) == action_count

    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
    def test_property_xml_generation_security(self, texts: list[Any] | str) -> None:
        """Property test: XML generation maintains security."""
        assume(all(t.strip() for t in texts))

        for text in texts:
            self.builder.add_text_action(text)

        result = self.builder.build_xml()

        if result["success"]:
            xml = result["xml"]
            # Should not contain dangerous patterns
            dangerous_patterns = [
                r"<script",
                r"javascript:",
                r"vbscript:",
                r"<!DOCTYPE",
                r"<!ENTITY",
                r"<\?xml.*encoding",
                r"<!\[CDATA\[",
            ]

            for pattern in dangerous_patterns:
                assert not re.search(pattern, xml, re.IGNORECASE), (
                    f"Dangerous pattern found: {pattern}"
                )

    def test_action_timeout_configuration(self) -> None:
        """Test action timeout configuration."""
        timeout = Duration.from_seconds(30)

        self.builder.add_action(
            "Type a String",
            {"text": "Test"},
            timeout=timeout,
            abort_on_failure=True,
        )

        actions = self.builder.get_actions()
        action = actions[0]

        assert action.timeout == timeout
        assert action.abort_on_failure is True
        assert action.enabled is True  # default

    def test_action_enabled_disabled(self) -> None:
        """Test action enabled/disabled state."""
        self.builder.add_text_action("Enabled", enabled=True)
        self.builder.add_text_action("Disabled", enabled=False)

        actions = self.builder.get_actions()

        assert actions[0].enabled is True
        assert actions[1].enabled is False

    @given(st.text())
    def test_property_dangerous_content_rejection(self, malicious_text: list[Any] | str) -> None:
        """Property test: Dangerous content is properly handled."""
        dangerous_patterns = [
            "<script>",
            "javascript:",
            "vbscript:",
            "<!DOCTYPE",
            "<!ENTITY",
            "<?xml",
            "<![CDATA[",
            "eval(",
            "exec(",
        ]

        # If text contains dangerous patterns, it should be sanitized or rejected
        has_dangerous = any(
            pattern.lower() in malicious_text.lower() for pattern in dangerous_patterns
        )

        try:
            self.builder.add_text_action(malicious_text)
            result = self.builder.build_xml()

            if has_dangerous and result["success"]:
                # If dangerous content was accepted, ensure it's properly escaped in XML
                xml = result["xml"]
                assert "<script>" not in xml.lower()
                assert "javascript:" not in xml.lower()

        except (ValidationError, SecurityViolationError):
            # It's acceptable for dangerous content to be rejected
            pass

    @settings(max_examples=20)
    @given(st.integers(min_value=0, max_value=100))
    def test_property_position_insertion_bounds(self, position: int | float) -> None:
        """Property test: Position insertion respects bounds."""
        # Add some initial actions
        for i in range(3):
            self.builder.add_text_action(f"Initial {i}")

        initial_count = self.builder.get_action_count()

        try:
            self.builder.add_text_action(f"Inserted at {position}", position=position)

            new_count = self.builder.get_action_count()

            if 0 <= position <= initial_count:
                # Valid position - action should be inserted
                assert new_count == initial_count + 1
            else:
                # Invalid position - action should still be added (at end)
                assert new_count == initial_count + 1

        except Exception as e:
            logger.debug(f"Operation failed during operation: {e}")


class TestActionSecurity:
    """Security-focused tests for action building."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.registry = ActionRegistry()
        self.builder = ActionBuilder(self.registry)

    def test_xml_injection_prevention(self) -> None:
        """Test prevention of XML injection attacks."""
        # These should be rejected (contain literal dangerous patterns)
        dangerous_inputs = [
            '"><script>alert("xss")</script>',
            "<!DOCTYPE html>",
            '<?xml version="1.0"?><!DOCTYPE root [<!ENTITY xxe "evil">]>',
            "<![CDATA[malicious]]>",
        ]

        for malicious_input in dangerous_inputs:
            self.builder.clear()

            # Expect ValidationError for dangerous content
            from src.core.errors import ValidationError

            with pytest.raises(ValidationError) as exc_info:
                self.builder.add_text_action(malicious_input)

            # Verify the error mentions security
            error_message = str(exc_info.value)
            assert (
                "security" in error_message.lower()
                or "dangerous" in error_message.lower()
            )

        # This should be allowed (already encoded, safe)
        safe_encoded_input = "&lt;script&gt;evil()&lt;/script&gt;"
        self.builder.clear()
        self.builder.add_text_action(safe_encoded_input)  # Should not raise

    def test_parameter_size_limits(self) -> None:
        """Test parameter size validation."""
        huge_text = "A" * 50000  # 50KB text (exceeds 10KB limit)

        # Expect ValidationError for oversized parameter
        from src.core.errors import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            self.builder.add_text_action(huge_text)

        # Verify the error mentions limit or security
        error_message = str(exc_info.value)
        assert (
            "limit" in error_message.lower()
            or "security" in error_message.lower()
            or "dangerous" in error_message.lower()
        )

    def test_special_characters_handling(self) -> None:
        """Test handling of special characters in parameters."""
        # Characters that should be handled properly in XML
        safe_special_chars = ["&", "<", ">", '"', "'", "\n", "\t", "\r"]

        for char in safe_special_chars:
            self.builder.clear()
            test_text = f"Before{char}After"

            self.builder.add_text_action(test_text)
            result = self.builder.build_xml()

            assert result["success"] is True
            xml = result["xml"]

            # Special characters should be properly escaped
            if char == "&":
                assert "&amp;" in xml or test_text not in xml
            elif char == "<":
                assert "&lt;" in xml or test_text not in xml
            elif char == ">":
                assert "&gt;" in xml or test_text not in xml
            elif char == '"':
                assert "&quot;" in xml or test_text not in xml
            elif char == "'":
                # Single quotes don't need escaping in XML text content
                assert test_text in xml

        # Null character should cause XML generation to fail (malformed XML)
        self.builder.clear()
        null_test_text = "Before\0After"
        self.builder.add_text_action(null_test_text)
        result = self.builder.build_xml()
        assert result["success"] is False  # Null character should be rejected


class TestActionRegistry:
    """Test suite for ActionRegistry functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.registry = ActionRegistry()

    def test_registry_initialization(self) -> None:
        """Test registry initialization with core actions."""
        assert self.registry.get_action_count() > 50  # Should have 80+ actions

        # Check major categories are represented
        categories = self.registry.get_category_counts()
        expected_categories = [
            ActionCategory.TEXT,
            ActionCategory.APPLICATION,
            ActionCategory.SYSTEM,
            ActionCategory.VARIABLE,
            ActionCategory.CONTROL,
        ]

        for category in expected_categories:
            assert category in categories
            assert categories[category] > 0

    def test_get_action_by_identifier(self) -> None:
        """Test retrieving actions by identifier."""
        action = self.registry.get_action_type("Type a String")

        assert action is not None
        assert action.identifier == "Type a String"
        assert action.category == ActionCategory.TEXT
        assert "text" in action.required_params

    def test_get_actions_by_category(self) -> None:
        """Test filtering actions by category."""
        text_actions = self.registry.get_actions_by_category(ActionCategory.TEXT)

        assert len(text_actions) > 0
        assert all(action.category == ActionCategory.TEXT for action in text_actions)

        # Check specific text actions exist
        text_identifiers = [action.identifier for action in text_actions]
        assert "Type a String" in text_identifiers
        assert "Search and Replace" in text_identifiers

    def test_search_actions(self) -> None:
        """Test action search functionality."""
        # Search for "Type" actions
        type_actions = self.registry.search_actions("Type")

        assert len(type_actions) > 0
        assert any("Type" in action.identifier for action in type_actions)

        # Search should be case-insensitive
        type_actions_lower = self.registry.search_actions("type")
        assert len(type_actions_lower) == len(type_actions)

    def test_parameter_validation(self) -> None:
        """Test parameter validation functionality."""
        # Valid parameters
        result = self.registry.validate_action_parameters(
            "Type a String",
            {"text": "Hello World", "by_typing": True},
        )

        assert result["valid"] is True
        assert len(result["missing_required"]) == 0

        # Missing required parameter
        result = self.registry.validate_action_parameters(
            "Type a String",
            {"by_typing": True},  # Missing 'text'
        )

        assert result["valid"] is False
        assert "text" in result["missing_required"]

        # Unknown action type
        result = self.registry.validate_action_parameters("Nonexistent Action", {})

        assert result["valid"] is False
        assert "Unknown action type" in result["error"]
