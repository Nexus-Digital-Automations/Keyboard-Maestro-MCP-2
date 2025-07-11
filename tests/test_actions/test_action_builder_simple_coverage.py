"""Simple coverage tests for ActionBuilder targeting specific uncovered lines."""

from unittest.mock import Mock

from src.actions.action_builder import ActionBuilder, ActionCategory, ActionType


class TestActionBuilderSimpleCoverage:
    """Simple tests targeting specific uncovered lines."""

    def test_create_action_none_parameters_handling(self) -> None:
        """Test create_action with None parameters - covers lines 210-211."""
        mock_registry = Mock()
        mock_registry.get_action_type.return_value = ActionType(
            identifier="Type a String",
            category=ActionCategory.TEXT,
            required_params=[],  # No required params to avoid validation error
        )

        builder = ActionBuilder(mock_registry)

        # This should trigger lines 210-211: if parameters is None: parameters = {}
        config = builder.create_action("Type a String", parameters=None)

        assert config.parameters == {}

    def test_convenience_methods_coverage(self) -> None:
        """Test convenience methods to cover specific lines."""
        mock_registry = Mock()

        # Mock different action types for convenience methods
        def get_action_type_side_effect(action_name):
            action_types = {
                "If Then Else": ActionType(
                    identifier="If Then Else",
                    category=ActionCategory.CONTROL,
                    required_params=["condition"],
                ),
                "Set Variable to Text": ActionType(
                    identifier="Set Variable to Text",
                    category=ActionCategory.VARIABLE,
                    required_params=["variable", "text"],
                ),
                "Activate a Specific Application": ActionType(
                    identifier="Activate a Specific Application",
                    category=ActionCategory.APPLICATION,
                    required_params=["application"],
                ),
            }
            return action_types.get(action_name)

        mock_registry.get_action_type.side_effect = get_action_type_side_effect
        builder = ActionBuilder(mock_registry)

        # Test add_if_action - covers line 266
        builder.add_if_action({"variable": "test", "operator": "equals"})
        assert len(builder.actions) == 1

        # Test add_variable_action - covers lines 275-279
        builder.add_variable_action("myVar", "myValue")
        assert len(builder.actions) == 2

        # Test add_app_action - covers line 288
        builder.add_app_action("Safari")
        assert len(builder.actions) == 3

    def test_action_builder_default_registry(self) -> None:
        """Test ActionBuilder with default registry creation."""
        # This will create a real ActionRegistry, which covers the import and creation
        builder = ActionBuilder()
        assert builder._registry is not None

    def test_xml_parameter_type_handling(self) -> None:
        """Test XML generation with different parameter types."""
        mock_registry = Mock()
        mock_registry.get_action_type.return_value = ActionType(
            identifier="Test Action",
            category=ActionCategory.TEXT,
            required_params=[],
            optional_params=["bool_param", "dict_param", "number_param"],
        )

        builder = ActionBuilder(mock_registry)

        # Add action with various parameter types to cover XML generation lines
        builder.add_action(
            "Test Action",
            {
                "bool_param": False,  # Covers line 373
                "dict_param": {"key": "value"},  # Covers lines 378-379
                "number_param": 42,  # Covers lines 374-375
            },
        )

        # Generate XML to trigger the parameter type handling
        result = builder.build_xml()
        assert result["success"] is True

        # Verify different parameter types are handled
        xml_content = result["xml"]
        assert "false" in xml_content  # Boolean handling
        assert "42" in xml_content  # Number handling
