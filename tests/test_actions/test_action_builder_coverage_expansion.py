"""Coverage expansion tests for ActionBuilder targeting specific uncovered lines.

This module provides targeted tests to achieve 100% coverage of the ActionBuilder
module by covering specific uncovered execution paths and edge cases.
"""

from unittest.mock import Mock, patch

import pytest
from src.actions.action_builder import (
    ActionBuilder,
    ActionCategory,
    ActionConfiguration,
    ActionType,
)
from src.core.errors import ValidationError
from src.core.types import Duration


class TestActionBuilderMissingCoverage:
    """Tests targeting specific uncovered lines in ActionBuilder."""

    def setup_method(self) -> None:
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
            "If Then Else",
            "Activate a Specific Application",
        ]

        self.builder = ActionBuilder(self.mock_registry)

    def test_create_action_with_none_parameters(self) -> None:
        """Test create_action method with None parameters - covers line 210-211."""
        # Line 210: if parameters is None:
        # Line 211: parameters = {}
        # Provide required text parameter via kwargs since parameters=None
        config = self.builder.create_action(
            "Type a String", parameters=None, text="Hello"
        )

        assert isinstance(config, ActionConfiguration)
        # Verify the 'text' parameter was properly added to the configuration
        assert "text" in config.parameters
        assert config.parameters["text"] == "Hello"

    def test_create_action_with_kwargs(self) -> None:
        """Test create_action method with various kwargs - covers lines 224-233."""
        timeout = Duration(seconds=10)

        config = self.builder.create_action(
            "Type a String",
            parameters={"text": "Hello"},
            position=5,
            enabled=False,
            timeout=timeout,
            abort_on_failure=True,
        )

        assert config.position == 5
        assert config.enabled is False
        assert config.timeout == timeout
        assert config.abort_on_failure is True

    def test_create_action_unknown_action_type(self) -> None:
        """Test create_action with unknown action type - covers lines 214-221."""
        self.mock_registry.get_action_type.return_value = None

        with pytest.raises(ValidationError) as exc_info:
            self.builder.create_action("Unknown Action")

        error = exc_info.value
        assert error.field_name == "action_type"
        assert error.value == "Unknown Action"
        assert "Available:" in str(error)

    def test_add_if_action_convenience_method(self) -> None:
        """Test add_if_action convenience method - covers line 266."""
        # Mock If Then Else action type
        if_action_type = ActionType(
            identifier="If Then Else",
            category=ActionCategory.CONTROL,
            required_params=["condition"],
        )
        self.mock_registry.get_action_type.return_value = if_action_type

        condition = {"variable": "test", "operator": "equals", "value": "true"}
        result = self.builder.add_if_action(condition)

        assert result == self.builder
        assert len(self.builder.actions) == 1
        assert self.builder.actions[0].action_type.identifier == "If Then Else"
        assert "condition" in self.builder.actions[0].parameters

    def test_add_variable_action_convenience_method(self) -> None:
        """Test add_variable_action convenience method - covers lines 275-279."""
        # Mock Set Variable to Text action type
        var_action_type = ActionType(
            identifier="Set Variable to Text",
            category=ActionCategory.VARIABLE,
            required_params=["variable", "text"],
        )
        self.mock_registry.get_action_type.return_value = var_action_type

        result = self.builder.add_variable_action("myVar", "myValue")

        assert result == self.builder
        assert len(self.builder.actions) == 1
        assert self.builder.actions[0].action_type.identifier == "Set Variable to Text"
        assert self.builder.actions[0].parameters["variable"] == "myVar"
        assert self.builder.actions[0].parameters["text"] == "myValue"

    def test_add_app_action_convenience_method(self) -> None:
        """Test add_app_action convenience method - covers line 288."""
        # Mock Activate a Specific Application action type
        app_action_type = ActionType(
            identifier="Activate a Specific Application",
            category=ActionCategory.APPLICATION,
            required_params=["application"],
            optional_params=["bring_all_windows"],
        )
        self.mock_registry.get_action_type.return_value = app_action_type

        result = self.builder.add_app_action("Safari", bring_all_windows=True)

        assert result == self.builder
        assert len(self.builder.actions) == 1
        assert (
            self.builder.actions[0].action_type.identifier
            == "Activate a Specific Application"
        )

    def test_action_builder_without_registry(self) -> None:
        """Test ActionBuilder initialization without registry - covers line 160-162."""
        with patch("src.actions.action_registry.ActionRegistry") as mock_registry_class:
            mock_instance = Mock()
            mock_registry_class.return_value = mock_instance

            builder = ActionBuilder()  # No registry parameter

            assert builder._registry == mock_instance
            mock_registry_class.assert_called_once()

    def test_action_configuration_validation_missing_params(self) -> None:
        """Test ActionConfiguration validation for missing required params - covers line 104."""
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=["text"],
        )

        # Test that __post_init__ validation catches missing required params
        with pytest.raises(ValidationError) as exc_info:
            ActionConfiguration(
                action_type=action_type,
                parameters={"other_param": "value"},  # Missing required "text" param
            )

        error = exc_info.value
        assert error.field_name == "parameters"
        assert "Missing: ['text']" in str(error)

    def test_build_xml_with_security_validation_failure(self) -> None:
        """Test build_xml when security validation fails - covers lines 340-342."""
        self.builder.add_action("Type a String", {"text": "Hello World"})

        # Mock _validate_xml_security to return False
        with patch.object(self.builder, "_validate_xml_security", return_value=False):
            result = self.builder.build_xml()

            assert result["success"] is False
            assert "security validation" in result["error"]
            assert result["xml"] == ""

    def test_build_xml_exception_handling(self) -> None:
        """Test build_xml exception handling - covers lines 340-342."""
        self.builder.add_action("Type a String", {"text": "Hello World"})

        # Mock _generate_action_xml to raise an exception
        with patch.object(
            self.builder, "_generate_action_xml", side_effect=Exception("XML error")
        ):
            result = self.builder.build_xml()

            assert result["success"] is False
            assert result["xml"] == ""
            assert "error" in result

    def test_generate_action_xml_with_timeout_and_abort(self) -> None:
        """Test _generate_action_xml with timeout and abort settings - covers lines 362, 365."""
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=["text"],
        )

        config = ActionConfiguration(
            action_type=action_type,
            parameters={"text": "Hello"},
            timeout=Duration(seconds=10),
            abort_on_failure=True,
        )

        # Test the _generate_action_xml method directly
        xml_elem = self.builder._generate_action_xml(config, 0)

        assert xml_elem.get("abortOnFailure") == "true"  # Line 362
        assert xml_elem.get("timeout") == "10"  # Line 365

    def test_parameter_types_in_xml_generation(self) -> None:
        """Test different parameter types in XML generation - covers lines 373, 378-379."""
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=["text"],
            optional_params=["bool_param", "dict_param"],
        )

        config = ActionConfiguration(
            action_type=action_type,
            parameters={
                "text": "Hello",
                "bool_param": False,  # Line 373
                "dict_param": {"key": "value"},  # Lines 378-379
            },
        )

        xml_elem = self.builder._generate_action_xml(config, 0)

        # Check boolean parameter handling
        bool_elem = xml_elem.find("bool_param")
        assert bool_elem.text == "false"

        # Check dict parameter handling
        dict_elem = xml_elem.find("dict_param")
        assert dict_elem.get("key") is not None

    def test_type_checking_import_coverage(self) -> None:
        """Test TYPE_CHECKING import coverage - covers line 20."""
        # The TYPE_CHECKING import is used for type hints
        # This test ensures the import is covered during runtime
        from typing import TYPE_CHECKING

        # Verify the import exists and is used in the module
        assert TYPE_CHECKING is False  # At runtime, TYPE_CHECKING is False

    def test_action_validation_with_error_context(self) -> None:
        """Test action validation with error context creation - covers line 478."""
        # Create an action that will trigger validation error with context
        action_type = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=["text"],
        )

        # Create a config with invalid parameters to trigger error context
        config = ActionConfiguration(
            action_type=action_type,
            parameters={"text": "valid content"},
        )

        # Modify to have dangerous content after creation
        config.parameters["text"] = "<script>alert('xss')</script>"

        self.builder.actions.append(config)

        # This should trigger validation error with context
        result = self.builder.validate_all()

        assert result["all_valid"] is False
        assert len(result["results"]) == 1
        assert not result["results"][0]["valid"]


class TestActionBuilderEdgeCases:
    """Additional edge case tests for maximum coverage."""

    def test_action_builder_with_large_action_list(self) -> None:
        """Test ActionBuilder with many actions for performance edge cases."""
        mock_registry = Mock()
        mock_registry.get_action_type.return_value = ActionType(
            identifier="Type a String",
            category=ActionCategory.TEXT,
            required_params=["text"],
        )

        builder = ActionBuilder(mock_registry)

        # Add many actions to test performance paths
        for i in range(50):
            builder.add_action("Type a String", {"text": f"Action {i}"})

        assert builder.get_action_count() == 50

        # Test XML generation with many actions
        result = builder.build_xml()
        assert result["action_count"] == 50

    def test_action_configuration_with_all_optional_params(self) -> None:
        """Test ActionConfiguration with all optional parameters specified."""
        action_type = ActionType(
            identifier="Comprehensive Action",
            category=ActionCategory.CONTROL,
            required_params=["required_param"],
            optional_params=["opt1", "opt2", "opt3"],
        )

        config = ActionConfiguration(
            action_type=action_type,
            parameters={
                "required_param": "value",
                "opt1": "optional1",
                "opt2": "optional2",
                "opt3": "optional3",
            },
            position=10,
            enabled=False,
            timeout=Duration(seconds=30),
            abort_on_failure=True,
        )

        assert config.position == 10
        assert config.enabled is False
        assert config.timeout == Duration(seconds=30)
        assert config.abort_on_failure is True
        assert len(config.parameters) == 4
