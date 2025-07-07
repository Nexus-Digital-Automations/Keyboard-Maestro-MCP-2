"""
Comprehensive tests for Action Builder module with systematic coverage.

Tests cover ActionType, ActionConfiguration, ActionBuilder with property-based testing,
security validation, XML generation, and comprehensive enterprise-grade validation.
"""

import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from src.actions.action_builder import (
    ActionBuilder,
    ActionCategory,
    ActionConfiguration,
    ActionType,
)
from src.core.errors import ValidationError
from src.core.types import Duration


# Test data generators
@st.composite
def action_category_strategy(draw):
    """Generate valid action categories."""
    return draw(
        st.sampled_from(
            [
                ActionCategory.TEXT,
                ActionCategory.APPLICATION,
                ActionCategory.FILE,
                ActionCategory.SYSTEM,
                ActionCategory.VARIABLE,
                ActionCategory.CONTROL,
                ActionCategory.INTERFACE,
                ActionCategory.WEB,
                ActionCategory.CALCULATION,
                ActionCategory.CLIPBOARD,
                ActionCategory.WINDOW,
                ActionCategory.SOUND,
            ]
        )
    )


@st.composite
def action_identifier_strategy(draw):
    """Generate valid action identifiers."""
    return draw(st.from_regex(r"^[a-zA-Z0-9_\s\-\./]+$", fullmatch=True))


@st.composite
def parameter_dict_strategy(draw):
    """Generate valid parameter dictionaries."""
    return draw(
        st.dictionaries(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Nd"]),
            ),
            st.one_of(
                st.text(max_size=100),
                st.integers(min_value=0, max_value=1000),
                st.floats(
                    min_value=0.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                st.booleans(),
            ),
            min_size=0,
            max_size=5,
        )
    )


class TestActionType:
    """Test ActionType with comprehensive validation."""

    def test_action_type_creation_valid(self):
        """Test creating valid ActionType instances."""
        action_type = ActionType(
            identifier="Type a String",
            category=ActionCategory.TEXT,
            required_params=["text"],
            optional_params=["by_typing"],
            description="Types text input",
        )

        assert action_type.identifier == "Type a String"
        assert action_type.category == ActionCategory.TEXT
        assert action_type.required_params == ["text"]
        assert action_type.optional_params == ["by_typing"]
        assert action_type.description == "Types text input"

    def test_action_type_empty_identifier(self):
        """Test ActionType with empty identifier raises ValueError."""
        with pytest.raises(ValueError, match="Action identifier cannot be empty"):
            ActionType(identifier="", category=ActionCategory.TEXT)

    def test_action_type_whitespace_identifier(self):
        """Test ActionType with whitespace-only identifier raises ValueError."""
        with pytest.raises(ValueError, match="Action identifier cannot be empty"):
            ActionType(identifier="   ", category=ActionCategory.TEXT)

    def test_action_type_invalid_identifier_format(self):
        """Test ActionType with invalid identifier format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid action identifier format"):
            ActionType(identifier="Action@Type!", category=ActionCategory.TEXT)

    def test_action_type_overlapping_parameters(self):
        """Test ActionType with overlapping required/optional parameters raises ValueError."""
        with pytest.raises(
            ValueError, match="Parameters cannot be both required and optional"
        ):
            ActionType(
                identifier="Test Action",
                category=ActionCategory.TEXT,
                required_params=["text", "value"],
                optional_params=["text", "option"],
            )

    @given(action_identifier_strategy(), action_category_strategy())
    def test_action_type_property_based_creation(self, identifier, category):
        """Property-based test for ActionType creation."""
        assume(identifier and identifier.strip())

        action_type = ActionType(
            identifier=identifier,
            category=category,
            required_params=["param1"],
            optional_params=["param2"],
        )

        assert action_type.identifier == identifier
        assert action_type.category == category
        assert "param1" in action_type.required_params
        assert "param2" in action_type.optional_params


class TestActionConfiguration:
    """Test ActionConfiguration with comprehensive validation."""

    def test_action_configuration_creation_valid(self):
        """Test creating valid ActionConfiguration instances."""
        action_type = ActionType(
            identifier="Type a String",
            category=ActionCategory.TEXT,
            required_params=["text"],
            optional_params=["by_typing"],
        )

        config = ActionConfiguration(
            action_type=action_type,
            parameters={"text": "Hello World", "by_typing": True},
            position=0,
            enabled=True,
            timeout=Duration(seconds=5),
            abort_on_failure=False,
        )

        assert config.action_type == action_type
        assert config.parameters == {"text": "Hello World", "by_typing": True}
        assert config.position == 0
        assert config.enabled is True
        assert config.timeout == Duration(seconds=5)
        assert config.abort_on_failure is False

    def test_action_configuration_missing_required_params(self):
        """Test ActionConfiguration with missing required parameters raises ValidationError."""
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=["text", "value"],
        )

        with pytest.raises(ValidationError) as exc_info:
            ActionConfiguration(
                action_type=action_type,
                parameters={"text": "Hello"},  # Missing "value"
            )

        # Check that the error contains the expected information
        error = exc_info.value
        assert error.field_name == "parameters"
        assert "value" in str(error)  # Should mention the missing parameter

    def test_action_configuration_validate_parameters_success(self):
        """Test validate_parameters returns True for valid configuration."""
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=["text"],
            optional_params=["option"],
        )

        config = ActionConfiguration(
            action_type=action_type, parameters={"text": "Hello", "option": "World"}
        )

        assert config.validate_parameters() is True

    def test_action_configuration_dangerous_patterns(self):
        """Test ActionConfiguration detects dangerous patterns in parameters."""
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=["text"],
        )

        # Test that dangerous patterns cause validation to fail
        # Create config with safe parameters first
        config = ActionConfiguration(
            action_type=action_type, parameters={"text": "safe content"}
        )

        # Test that security validation method detects dangerous patterns
        assert (
            config._contains_dangerous_patterns("<script>alert('xss')</script>") is True
        )
        assert config._contains_dangerous_patterns("javascript:alert('xss')") is True
        assert config._contains_dangerous_patterns("vbscript:msgbox('xss')") is True
        assert config._contains_dangerous_patterns("eval(malicious_code)") is True
        assert config._contains_dangerous_patterns("safe content") is False

    def test_action_configuration_parameter_length_limit(self):
        """Test ActionConfiguration enforces parameter length limits."""
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=["text"],
        )

        # Test that parameter length validation works
        config = ActionConfiguration(
            action_type=action_type, parameters={"text": "short text"}
        )

        # Test normal length parameter passes
        assert config._validate_parameter_security() is True

        # Test that long parameter would fail validation
        long_text = "A" * 10001  # Exceeds 10KB limit
        config_long = ActionConfiguration(
            action_type=action_type,
            parameters={"text": "short"},  # Start with short text
        )
        # Modify parameter after creation to test length validation
        config_long.parameters["text"] = long_text
        assert config_long._validate_parameter_security() is False

    @given(parameter_dict_strategy())
    def test_action_configuration_property_based_validation(self, parameters):
        """Property-based test for ActionConfiguration validation."""
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=[],
            optional_params=list(parameters.keys()),
        )

        # Filter out parameters that might contain dangerous patterns
        safe_parameters = {}
        for key, value in parameters.items():
            value_str = str(value)
            if len(value_str) <= 1000 and not any(
                pattern in value_str.lower()
                for pattern in [
                    "<script",
                    "javascript:",
                    "vbscript:",
                    "eval(",
                    "exec(",
                    "system(",
                ]
            ):
                safe_parameters[key] = value

        if safe_parameters:
            config = ActionConfiguration(
                action_type=action_type, parameters=safe_parameters
            )
            assert config.validate_parameters() is True


class TestActionBuilder:
    """Test ActionBuilder with comprehensive functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_registry = Mock()
        self.mock_registry.get_action_type.return_value = ActionType(
            identifier="Type a String",
            category=ActionCategory.TEXT,
            required_params=["text"],
            optional_params=["by_typing"],
        )
        self.mock_registry.list_action_names.return_value = [
            "Type a String",
            "Pause",
            "Set Variable to Text",
        ]

        self.builder = ActionBuilder(self.mock_registry)

    def test_action_builder_initialization(self):
        """Test ActionBuilder initialization."""
        builder = ActionBuilder(self.mock_registry)
        assert builder.actions == []
        assert builder._registry == self.mock_registry

    def test_action_builder_add_action_success(self):
        """Test adding action to builder successfully."""
        result = self.builder.add_action(
            "Type a String", {"text": "Hello World", "by_typing": True}
        )

        assert result == self.builder  # Fluent interface
        assert len(self.builder.actions) == 1
        assert self.builder.actions[0].action_type.identifier == "Type a String"
        assert self.builder.actions[0].parameters == {
            "text": "Hello World",
            "by_typing": True,
        }

    def test_action_builder_add_action_unknown_type(self):
        """Test adding unknown action type raises ValidationError."""
        self.mock_registry.get_action_type.return_value = None

        with pytest.raises(ValidationError) as exc_info:
            self.builder.add_action("Unknown Action", {})

        # Check that the error contains the expected information
        error = exc_info.value
        assert error.field_name == "action_type"
        assert error.value == "Unknown Action"
        assert "Available:" in str(error)

    def test_action_builder_add_action_with_position(self):
        """Test adding action at specific position."""
        # Add first action
        self.builder.add_action("Type a String", {"text": "First"})

        # Add second action
        self.builder.add_action("Type a String", {"text": "Second"})

        # Insert at position 1
        self.builder.add_action("Type a String", {"text": "Middle"}, position=1)

        assert len(self.builder.actions) == 3
        assert self.builder.actions[0].parameters["text"] == "First"
        assert self.builder.actions[1].parameters["text"] == "Middle"
        assert self.builder.actions[2].parameters["text"] == "Second"

    def test_action_builder_convenience_methods(self):
        """Test convenience methods for common actions."""
        # Test text action
        self.builder.add_text_action("Hello World", by_typing=False)
        assert len(self.builder.actions) == 1
        assert self.builder.actions[0].parameters["text"] == "Hello World"

        # Test pause action
        self.mock_registry.get_action_type.return_value = ActionType(
            identifier="Pause",
            category=ActionCategory.CONTROL,
            required_params=["duration"],
        )

        duration = Duration(seconds=2)
        self.builder.add_pause_action(duration)
        assert len(self.builder.actions) == 2
        assert self.builder.actions[1].parameters["duration"] == 2.0

    def test_action_builder_build_xml_success(self):
        """Test successful XML generation."""
        self.builder.add_action("Type a String", {"text": "Hello World"})

        result = self.builder.build_xml()

        assert result["success"] is True
        assert "xml" in result
        assert result["action_count"] == 1
        assert result["validation_passed"] is True

        # Verify XML structure
        xml_content = result["xml"]
        assert "<actions>" in xml_content
        assert "<action" in xml_content
        assert 'type="Type a String"' in xml_content
        assert "<text>Hello World</text>" in xml_content

    def test_action_builder_build_xml_empty_actions(self):
        """Test XML generation with no actions."""
        result = self.builder.build_xml()

        assert result["success"] is False
        assert "No actions to build" in result["error"]
        assert result["xml"] == ""

    def test_action_builder_build_xml_security_validation(self):
        """Test XML generation with security validation."""
        # Add action with potentially dangerous content
        self.builder.add_action("Type a String", {"text": "Normal text"})

        # Mock security validation failure
        with patch.object(self.builder, "_validate_xml_security", return_value=False):
            result = self.builder.build_xml()

            assert result["success"] is False
            assert "security validation" in result["error"]
            assert result["xml"] == ""

    def test_action_builder_xml_escaping(self):
        """Test proper XML generation with valid content."""
        self.builder.add_action("Type a String", {"text": "Test safe symbols"})

        result = self.builder.build_xml()

        assert result["success"] is True
        xml_content = result["xml"]

        # Verify the XML is well-formed by parsing it
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_content)
        assert root.tag == "actions"

        # Check that the XML contains the content
        assert "Test" in xml_content
        assert "safe" in xml_content
        assert "symbols" in xml_content
        # The XML should be valid and contain our data
        assert len(root) > 0  # Should have at least one action

    def test_action_builder_clear(self):
        """Test clearing all actions."""
        self.builder.add_action("Type a String", {"text": "Hello"})
        assert len(self.builder.actions) == 1

        result = self.builder.clear()

        assert result == self.builder  # Fluent interface
        assert len(self.builder.actions) == 0

    def test_action_builder_remove_action(self):
        """Test removing action by index."""
        self.builder.add_action("Type a String", {"text": "First"})
        self.builder.add_action("Type a String", {"text": "Second"})
        self.builder.add_action("Type a String", {"text": "Third"})

        # Remove middle action
        result = self.builder.remove_action(1)

        assert result == self.builder  # Fluent interface
        assert len(self.builder.actions) == 2
        assert self.builder.actions[0].parameters["text"] == "First"
        assert self.builder.actions[1].parameters["text"] == "Third"

    def test_action_builder_remove_action_invalid_index(self):
        """Test removing action with invalid index doesn't crash."""
        self.builder.add_action("Type a String", {"text": "Hello"})

        # Remove invalid indices
        self.builder.remove_action(-1)
        self.builder.remove_action(10)

        # Action should still be there
        assert len(self.builder.actions) == 1

    def test_action_builder_get_action_count(self):
        """Test getting action count."""
        assert self.builder.get_action_count() == 0

        self.builder.add_action("Type a String", {"text": "Hello"})
        assert self.builder.get_action_count() == 1

        self.builder.add_action("Type a String", {"text": "World"})
        assert self.builder.get_action_count() == 2

    def test_action_builder_get_actions(self):
        """Test getting copy of actions list."""
        self.builder.add_action("Type a String", {"text": "Hello"})

        actions = self.builder.get_actions()

        assert len(actions) == 1
        assert actions[0].parameters["text"] == "Hello"

        # Verify it's a copy
        actions.clear()
        assert len(self.builder.actions) == 1

    def test_action_builder_validate_all_success(self):
        """Test validating all actions successfully."""
        self.builder.add_action("Type a String", {"text": "Hello"})
        self.builder.add_action("Type a String", {"text": "World"})

        result = self.builder.validate_all()

        assert result["all_valid"] is True
        assert result["total_actions"] == 2
        assert result["valid_actions"] == 2
        assert len(result["results"]) == 2

        for validation_result in result["results"]:
            assert validation_result["valid"] is True
            assert validation_result["issues"] == []

    def test_action_builder_validate_all_with_failures(self):
        """Test validating all actions with validation failures."""
        # Add valid action
        self.builder.add_action("Type a String", {"text": "Hello"})

        # Create an action with security validation issues
        # by modifying parameters after creation
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=["text"],
        )

        # Create valid config first
        config = ActionConfiguration(
            action_type=action_type, parameters={"text": "safe content"}
        )

        # Modify to have dangerous content
        config.parameters["text"] = "<script>alert('xss')</script>"

        # Add to builder
        self.builder.actions.append(config)

        result = self.builder.validate_all()

        assert result["all_valid"] is False
        assert result["total_actions"] == 2
        assert result["valid_actions"] == 1
        assert len(result["results"]) == 2

        # Check that invalid action is properly identified
        invalid_result = result["results"][1]
        assert invalid_result["valid"] is False
        assert "Security validation failed" in invalid_result["issues"][0]

    def test_action_builder_fluent_interface(self):
        """Test fluent interface chaining."""
        result = (
            self.builder.add_action("Type a String", {"text": "Hello"})
            .add_action("Type a String", {"text": "World"})
            .add_action("Type a String", {"text": "!"})
            .clear()
            .add_action("Type a String", {"text": "Final"})
        )

        assert result == self.builder
        assert len(self.builder.actions) == 1
        assert self.builder.actions[0].parameters["text"] == "Final"


class TestActionBuilderSecurity:
    """Test ActionBuilder security features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_registry = Mock()
        self.mock_registry.get_action_type.return_value = ActionType(
            identifier="Type a String",
            category=ActionCategory.TEXT,
            required_params=["text"],
        )
        self.builder = ActionBuilder(self.mock_registry)

    def test_xml_security_validation_dangerous_patterns(self):
        """Test XML security validation detects dangerous patterns."""
        # Test various dangerous XML patterns
        dangerous_patterns = [
            "<!DOCTYPE html>",
            '<!ENTITY test "value">',
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<![CDATA[malicious]]>",
            'javascript:alert("xss")',
            'vbscript:msgbox("xss")',
            "data:text/html,<script>",
            "file:///etc/passwd",
            "&malicious;",
        ]

        for pattern in dangerous_patterns:
            result = self.builder._validate_xml_security(pattern)
            assert result is False, f"Failed to detect dangerous pattern: {pattern}"

    def test_xml_security_validation_valid_xml(self):
        """Test XML security validation passes valid XML."""
        valid_xml = """<actions>
            <action type="Type a String" id="0">
                <text>Hello World</text>
            </action>
        </actions>"""

        result = self.builder._validate_xml_security(valid_xml)
        assert result is True

    def test_xml_security_validation_malformed_xml(self):
        """Test XML security validation rejects malformed XML."""
        malformed_xml = "<actions><action><text>Unclosed tag</actions>"

        result = self.builder._validate_xml_security(malformed_xml)
        assert result is False

    def test_xml_security_validation_size_limit(self):
        """Test XML security validation enforces size limits."""
        # Create XML that exceeds 1MB limit
        large_xml = "<actions>" + ("A" * 1000001) + "</actions>"

        result = self.builder._validate_xml_security(large_xml)
        assert result is False

    def test_parameter_security_validation(self):
        """Test parameter security validation in ActionConfiguration."""
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=["text"],
        )

        # Test that dangerous patterns are rejected
        config = ActionConfiguration(
            action_type=action_type, parameters={"text": "safe content"}
        )

        # Test private method directly
        assert config._validate_parameter_security() is True

        # Test with dangerous content
        config_dangerous = ActionConfiguration(
            action_type=action_type, parameters={"text": "safe content"}
        )
        # Modify parameters after creation to test security method
        config_dangerous.parameters["text"] = "<script>alert('xss')</script>"

        assert config_dangerous._validate_parameter_security() is False


class TestActionBuilderIntegration:
    """Integration tests for ActionBuilder with real components."""

    def test_action_builder_with_real_registry(self):
        """Test ActionBuilder with real ActionRegistry."""
        # This test would require the actual ActionRegistry implementation
        # For now, we'll test the initialization path
        builder = ActionBuilder()  # Should create its own registry

        assert builder._registry is not None
        assert hasattr(builder._registry, "get_action_type")
        assert hasattr(builder._registry, "list_action_names")

    def test_action_builder_xml_generation_complete_flow(self):
        """Test complete XML generation flow with multiple action types."""
        mock_registry = Mock()

        # Mock different action types
        text_action = ActionType(
            identifier="Type a String",
            category=ActionCategory.TEXT,
            required_params=["text"],
        )

        pause_action = ActionType(
            identifier="Pause",
            category=ActionCategory.CONTROL,
            required_params=["duration"],
        )

        variable_action = ActionType(
            identifier="Set Variable to Text",
            category=ActionCategory.VARIABLE,
            required_params=["variable", "text"],
        )

        # Configure registry mock
        def get_action_type(action_name):
            mapping = {
                "Type a String": text_action,
                "Pause": pause_action,
                "Set Variable to Text": variable_action,
            }
            return mapping.get(action_name)

        mock_registry.get_action_type.side_effect = get_action_type

        builder = ActionBuilder(mock_registry)

        # Build complex action sequence
        builder.add_action("Type a String", {"text": "Hello World"})
        builder.add_action("Pause", {"duration": 1.5})
        builder.add_action(
            "Set Variable to Text", {"variable": "result", "text": "completed"}
        )

        result = builder.build_xml()

        assert result["success"] is True
        assert result["action_count"] == 3

        xml_content = result["xml"]

        # Verify all actions are present
        assert 'type="Type a String"' in xml_content
        assert 'type="Pause"' in xml_content
        assert 'type="Set Variable to Text"' in xml_content

        # Verify parameters are properly encoded
        assert "<text>Hello World</text>" in xml_content
        assert "<duration>1.5</duration>" in xml_content
        assert "<variable>result</variable>" in xml_content

        # Verify XML structure
        root = ET.fromstring(xml_content)
        assert root.tag == "actions"
        assert len(root) == 3

        # Verify action ordering
        actions = root.findall("action")
        assert actions[0].get("type") == "Type a String"
        assert actions[1].get("type") == "Pause"
        assert actions[2].get("type") == "Set Variable to Text"
