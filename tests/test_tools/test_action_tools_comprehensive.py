"""Comprehensive tests for action tools module.

import logging

logging.basicConfig(level=logging.DEBUG)
Tests cover action building functionality, XML generation, security validation,
parameter validation, and integration with Keyboard Maestro with property-based testing.
"""

import asyncio
import uuid
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from src.actions import ActionCategory
from src.core.errors import PermissionDeniedError
from src.server.tools.action_tools import (
    _add_action_to_km_macro,
    _truncate_xml_for_preview,
    km_add_action,
    km_list_action_types,
)


# Test data generators
@st.composite
def action_config_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid action configurations."""
    config_types = [
        # Text actions
        {
            "text": draw(st.text(min_size=1, max_size=100)),
            "by_typing": draw(st.booleans()),
        },
        # System actions
        {"duration": draw(st.floats(min_value=0.1, max_value=10.0))},
        # Application actions
        {
            "application": draw(st.text(min_size=1, max_size=50)),
            "bring_all_windows": draw(st.booleans()),
        },
        # Variable actions
        {
            "name": draw(st.text(min_size=1, max_size=50)),
            "value": draw(st.text(max_size=100)),
        },
        # Basic parameter set
        draw(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
                max_size=5,
            ),
        ),
    ]
    return draw(st.sampled_from(config_types))


@st.composite
def macro_id_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid macro IDs."""
    id_types = [
        # UUID format
        str(uuid.uuid4()),
        # Name format
        draw(st.text(min_size=1, max_size=100).filter(lambda x: len(x.strip()) > 0)),
        # Mixed format
        f"Macro_{draw(st.integers(min_value=1, max_value=1000))}",
    ]
    return draw(st.sampled_from(id_types))


@st.composite
def action_type_strategy(draw: Callable[..., Any]) -> Mock:
    """Generate valid action type names."""
    action_types = [
        "Type a String",
        "Pause",
        "If Then Else",
        "Activate a Specific Application",
        "Play Sound",
        "Set Variable to Text",
        "Get Variable",
        "Execute Shell Script",
        "Display Text",
        "Copy to Clipboard",
        "Paste from Clipboard",
        "Click at Found Image",
        "Type the Keystroke",
    ]
    return draw(st.sampled_from(action_types))


@st.composite
def search_term_strategy(draw: Callable[..., Any]) -> list[Any]:
    """Generate valid search terms."""
    search_terms = [
        draw(st.text(min_size=1, max_size=50)),
        "type",
        "pause",
        "application",
        "text",
        "variable",
        "script",
    ]
    return draw(st.sampled_from(search_terms))


class TestActionBuilderDependencies:
    """Test action builder and registry dependencies."""

    def test_action_registry_creation(self) -> None:
        """Test action registry creation."""
        with patch("src.server.tools.action_tools.ActionRegistry") as mock_registry:
            mock_instance = Mock()
            mock_registry.return_value = mock_instance

            # Create registry
            from src.server.tools.action_tools import ActionRegistry

            registry = ActionRegistry()

            assert registry is not None

    def test_action_builder_creation(self) -> None:
        """Test action builder creation."""
        with (
            patch("src.server.tools.action_tools.ActionBuilder") as mock_builder,
            patch("src.server.tools.action_tools.ActionRegistry") as mock_registry,
        ):
            mock_registry_instance = Mock()
            mock_builder_instance = Mock()
            mock_registry.return_value = mock_registry_instance
            mock_builder.return_value = mock_builder_instance

            # Create builder
            from src.server.tools.action_tools import ActionBuilder, ActionRegistry

            registry = ActionRegistry()
            builder = ActionBuilder(registry)

            assert builder is not None


class TestKMAddAction:
    """Test km_add_action functionality."""

    @pytest.mark.asyncio
    async def test_km_add_action_success(self) -> None:
        """Test successful action addition."""
        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch("src.server.tools.action_tools.ActionBuilder") as mock_builder_class,
            patch("src.server.tools.action_tools.get_km_client") as mock_get_client,
            patch(
                "src.server.tools.action_tools._add_action_to_km_macro",
            ) as mock_add_action,
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks
            mock_registry = Mock()
            mock_builder = Mock()
            mock_action_def = Mock()
            mock_action_def.category.value = "text"
            mock_action_def.required_params = ["text"]
            mock_action_def.optional_params = ["by_typing"]

            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {"valid": True}
            mock_registry.get_action_count.return_value = 85

            mock_builder.actions = []
            mock_builder.get_action_count.return_value = 1
            mock_builder.build_xml.return_value = {
                "success": True,
                "xml": "<action>test</action>",
            }

            mock_registry_class.return_value = mock_registry
            mock_builder_class.return_value = mock_builder
            mock_get_client.return_value = Mock()
            mock_add_action.return_value = True

            # Mock asyncio loop
            mock_loop = Mock()
            mock_executor = Mock()
            mock_executor.return_value = True
            mock_loop.run_in_executor = AsyncMock(return_value=True)
            mock_get_loop.return_value = mock_loop

            # Test action addition
            result = await km_add_action(
                macro_id="test_macro",
                action_type="Type a String",
                action_config={"text": "Hello World", "by_typing": True},
            )

            assert result["success"]
            assert "action_added" in result["data"]
            assert result["data"]["action_added"]["action_type"] == "Type a String"
            assert result["data"]["action_added"]["macro_id"] == "test_macro"
            assert "xml_preview" in result["data"]
            assert "validation" in result["data"]
            assert result["data"]["validation"]["xml_validated"]

    @pytest.mark.asyncio
    async def test_km_add_action_invalid_action_type(self) -> None:
        """Test action addition with invalid action type."""
        with patch(
            "src.server.tools.action_tools.ActionRegistry",
        ) as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_action_type.return_value = None
            mock_registry.list_action_names.return_value = [
                "Type a String",
                "Pause",
                "Play Sound",
            ]
            mock_registry.get_action_count.return_value = 85
            mock_registry_class.return_value = mock_registry

            result = await km_add_action(
                macro_id="test_macro",
                action_type="Invalid Action Type",
                action_config={"text": "test"},
            )

            assert not result["success"]
            assert result["error"]["code"] == "VALIDATION_ERROR"
            assert (
                "Validation failed for field 'action_type'"
                in result["error"]["message"]
            )
            assert "Invalid Action Type" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_km_add_action_parameter_validation_failure(self) -> None:
        """Test action addition with parameter validation failure."""
        with patch(
            "src.server.tools.action_tools.ActionRegistry",
        ) as mock_registry_class:
            mock_registry = Mock()
            mock_action_def = Mock()
            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {
                "valid": False,
                "missing_required": ["text"],
                "unknown_params": ["invalid_param"],
            }
            mock_registry_class.return_value = mock_registry

            result = await km_add_action(
                macro_id="test_macro",
                action_type="Type a String",
                action_config={"invalid_param": "value"},
            )

            assert not result["success"]
            assert result["error"]["code"] == "VALIDATION_ERROR"
            assert (
                "Validation failed for field 'action_config'"
                in result["error"]["message"]
            )
            assert "Missing required parameters" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_km_add_action_xml_generation_failure(self) -> None:
        """Test action addition with XML generation failure."""
        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch("src.server.tools.action_tools.ActionBuilder") as mock_builder_class,
        ):
            mock_registry = Mock()
            mock_builder = Mock()
            mock_action_def = Mock()

            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {"valid": True}
            mock_builder.build_xml.return_value = {
                "success": False,
                "error": "XML generation failed",
            }

            mock_registry_class.return_value = mock_registry
            mock_builder_class.return_value = mock_builder

            result = await km_add_action(
                macro_id="test_macro",
                action_type="Type a String",
                action_config={"text": "test"},
            )

            assert not result["success"]
            assert result["error"]["code"] == "VALIDATION_ERROR"
            assert "XML generation failed" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_km_add_action_with_timeout_and_position(self) -> None:
        """Test action addition with timeout and position parameters."""
        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch("src.server.tools.action_tools.ActionBuilder") as mock_builder_class,
            patch("src.server.tools.action_tools.get_km_client") as mock_get_client,
            patch(
                "src.server.tools.action_tools._add_action_to_km_macro",
            ) as mock_add_action,
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks
            mock_registry = Mock()
            mock_builder = Mock()
            mock_action_def = Mock()
            mock_action_def.category.value = "system"
            mock_action_def.required_params = ["duration"]
            mock_action_def.optional_params = []

            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {"valid": True}
            mock_registry.get_action_count.return_value = 85

            mock_builder.actions = [Mock(), Mock()]  # Existing actions
            mock_builder.get_action_count.return_value = 3
            mock_builder.build_xml.return_value = {
                "success": True,
                "xml": "<action>pause</action>",
            }

            mock_registry_class.return_value = mock_registry
            mock_builder_class.return_value = mock_builder
            mock_get_client.return_value = Mock()
            mock_add_action.return_value = True

            # Mock asyncio loop
            mock_loop = Mock()
            mock_loop.run_in_executor = AsyncMock(return_value=True)
            mock_get_loop.return_value = mock_loop

            result = await km_add_action(
                macro_id="test_macro",
                action_type="Pause",
                action_config={"duration": 2.5},
                position=1,
                timeout=30,
                enabled=False,
                abort_on_failure=True,
            )

            assert result["success"]
            assert result["data"]["action_added"]["position"] == 1
            assert result["data"]["action_added"]["timeout"] == 30
            assert not result["data"]["action_added"]["enabled"]
            assert result["data"]["action_added"]["abort_on_failure"]

    @pytest.mark.asyncio
    async def test_km_add_action_permission_denied(self) -> None:
        """Test action addition with permission denied error."""
        with patch(
            "src.server.tools.action_tools.ActionRegistry",
        ) as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_action_type.side_effect = PermissionDeniedError(
                required_permissions=["MACRO_EDIT"],
                available_permissions=[],
            )
            mock_registry_class.return_value = mock_registry

            result = await km_add_action(
                macro_id="test_macro",
                action_type="Type a String",
                action_config={"text": "test"},
            )

            assert not result["success"]
            assert result["error"]["code"] == "PERMISSION_DENIED"
            assert "Insufficient permissions" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_km_add_action_with_context(self) -> None:
        """Test action addition with context for progress reporting."""
        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.report_progress = AsyncMock()

        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch("src.server.tools.action_tools.ActionBuilder") as mock_builder_class,
            patch("src.server.tools.action_tools.get_km_client") as mock_get_client,
            patch(
                "src.server.tools.action_tools._add_action_to_km_macro",
            ) as mock_add_action,
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks for success case
            mock_registry = Mock()
            mock_builder = Mock()
            mock_action_def = Mock()
            mock_action_def.category.value = "text"
            mock_action_def.required_params = ["text"]
            mock_action_def.optional_params = []

            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {"valid": True}
            mock_registry.get_action_count.return_value = 85
            mock_builder.actions = []
            mock_builder.get_action_count.return_value = 1
            mock_builder.build_xml.return_value = {
                "success": True,
                "xml": "<action>test</action>",
            }

            mock_registry_class.return_value = mock_registry
            mock_builder_class.return_value = mock_builder
            mock_get_client.return_value = Mock()
            mock_add_action.return_value = True

            # Mock asyncio loop
            mock_loop = Mock()
            mock_loop.run_in_executor = AsyncMock(return_value=True)
            mock_get_loop.return_value = mock_loop

            result = await km_add_action(
                macro_id="test_macro",
                action_type="Type a String",
                action_config={"text": "test"},
                ctx=mock_ctx,
            )

            assert result["success"]

            # Verify context calls
            mock_ctx.info.assert_called()
            assert mock_ctx.report_progress.call_count >= 3  # Multiple progress reports

    @pytest.mark.asyncio
    @given(macro_id_strategy(), action_type_strategy(), action_config_strategy())
    async def test_km_add_action_property_validation(
        self,
        macro_id: str,
        action_type: str,
        action_config: dict[str, Any],
    ) -> None:
        """Property test for km_add_action behavior."""
        assume(len(macro_id.strip()) > 0)
        assume(len(action_type.strip()) > 0)

        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch("src.server.tools.action_tools.ActionBuilder") as mock_builder_class,
            patch("src.server.tools.action_tools.get_km_client") as mock_get_client,
            patch(
                "src.server.tools.action_tools._add_action_to_km_macro",
            ) as mock_add_action,
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks for success case
            mock_registry = Mock()
            mock_builder = Mock()
            mock_action_def = Mock()
            mock_action_def.category.value = "test"
            mock_action_def.required_params = []
            mock_action_def.optional_params = list(action_config.keys())

            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {"valid": True}
            mock_registry.get_action_count.return_value = 85
            mock_builder.actions = []
            mock_builder.get_action_count.return_value = 1
            mock_builder.build_xml.return_value = {
                "success": True,
                "xml": "<action>test</action>",
            }

            mock_registry_class.return_value = mock_registry
            mock_builder_class.return_value = mock_builder
            mock_get_client.return_value = Mock()
            mock_add_action.return_value = True

            # Mock asyncio loop
            mock_loop = Mock()
            mock_loop.run_in_executor = AsyncMock(return_value=True)
            mock_get_loop.return_value = mock_loop

            result = await km_add_action(
                macro_id=macro_id.strip(),
                action_type=action_type.strip(),
                action_config=action_config,
            )

            # Properties that should always hold
            assert isinstance(result, dict)
            assert "success" in result
            assert isinstance(result["success"], bool)

            if result["success"]:
                assert "data" in result
                assert "metadata" in result
                assert "timestamp" in result["metadata"]
                assert "correlation_id" in result["metadata"]
            else:
                assert "error" in result
                assert "code" in result["error"]
                assert "message" in result["error"]


class TestKMListActionTypes:
    """Test km_list_action_types functionality."""

    @pytest.mark.asyncio
    async def test_km_list_action_types_all(self) -> None:
        """Test listing all action types without filters."""
        with patch(
            "src.server.tools.action_tools.ActionRegistry",
        ) as mock_registry_class:
            mock_registry = Mock()

            # Create mock actions
            mock_action1 = Mock()
            mock_action1.identifier = "Type a String"
            mock_action1.category.value = "text"
            mock_action1.description = "Type text into the current application"
            mock_action1.required_params = ["text"]
            mock_action1.optional_params = ["by_typing"]

            mock_action2 = Mock()
            mock_action2.identifier = "Pause"
            mock_action2.category.value = "system"
            mock_action2.description = "Pause execution for specified duration"
            mock_action2.required_params = ["duration"]
            mock_action2.optional_params = []

            mock_registry.list_all_actions.return_value = [mock_action1, mock_action2]
            mock_registry.get_action_count.return_value = 85
            mock_registry.get_category_counts.return_value = {
                ActionCategory.TEXT: 15,
                ActionCategory.SYSTEM: 20,
            }
            mock_registry_class.return_value = mock_registry

            result = await km_list_action_types()

            assert result["success"]
            assert len(result["data"]["actions"]) == 2
            assert result["data"]["summary"]["total_available"] == 85
            assert result["data"]["summary"]["total_found"] == 2
            assert result["data"]["summary"]["returned"] == 2

            # Check action details
            action_names = [
                action["identifier"] for action in result["data"]["actions"]
            ]
            assert "Type a String" in action_names
            assert "Pause" in action_names

    @pytest.mark.asyncio
    async def test_km_list_action_types_with_category_filter(self) -> None:
        """Test listing action types with category filter."""
        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch(
                "src.server.tools.action_tools.ActionCategory",
            ) as mock_category_class,
        ):
            mock_registry = Mock()
            mock_category_enum = Mock()
            mock_category_class.return_value = mock_category_enum

            # Mock text actions only
            mock_action = Mock()
            mock_action.identifier = "Type a String"
            mock_action.category.value = "text"
            mock_action.description = "Type text"
            mock_action.required_params = ["text"]
            mock_action.optional_params = []

            mock_registry.get_actions_by_category.return_value = [mock_action]
            mock_registry.get_action_count.return_value = 85
            mock_registry.get_category_counts.return_value = {mock_category_enum: 15}
            mock_registry_class.return_value = mock_registry

            result = await km_list_action_types(category="text")

            assert result["success"]
            assert len(result["data"]["actions"]) == 1
            assert result["data"]["actions"][0]["identifier"] == "Type a String"
            assert result["data"]["summary"]["filtered_by_category"] == "text"

    @pytest.mark.asyncio
    async def test_km_list_action_types_with_search_filter(self) -> None:
        """Test listing action types with search filter."""
        with patch(
            "src.server.tools.action_tools.ActionRegistry",
        ) as mock_registry_class:
            mock_registry = Mock()

            # Create mock actions where only one matches search
            mock_action1 = Mock()
            mock_action1.identifier = "Type a String"
            mock_action1.category.value = "text"
            mock_action1.description = "Type text into application"
            mock_action1.required_params = ["text"]
            mock_action1.optional_params = []

            mock_action2 = Mock()
            mock_action2.identifier = "Pause"
            mock_action2.category.value = "system"
            mock_action2.description = "Pause execution for duration"
            mock_action2.required_params = ["duration"]
            mock_action2.optional_params = []

            mock_registry.list_all_actions.return_value = [mock_action1, mock_action2]
            mock_registry.get_action_count.return_value = 85
            mock_registry.get_category_counts.return_value = {}
            mock_registry_class.return_value = mock_registry

            result = await km_list_action_types(search="type")

            assert result["success"]
            assert len(result["data"]["actions"]) == 1
            assert result["data"]["actions"][0]["identifier"] == "Type a String"
            assert result["data"]["summary"]["filtered_by_search"] == "type"

    @pytest.mark.asyncio
    async def test_km_list_action_types_with_limit(self) -> None:
        """Test listing action types with limit."""
        with patch(
            "src.server.tools.action_tools.ActionRegistry",
        ) as mock_registry_class:
            mock_registry = Mock()

            # Create 5 mock actions
            mock_actions = []
            for i in range(5):
                mock_action = Mock()
                mock_action.identifier = f"Action {i}"
                mock_action.category.value = "test"
                mock_action.description = f"Test action {i}"
                mock_action.required_params = []
                mock_action.optional_params = []
                mock_actions.append(mock_action)

            mock_registry.list_all_actions.return_value = mock_actions
            mock_registry.get_action_count.return_value = 85
            mock_registry.get_category_counts.return_value = {}
            mock_registry_class.return_value = mock_registry

            result = await km_list_action_types(limit=3)

            assert result["success"]
            assert len(result["data"]["actions"]) == 3
            assert result["data"]["summary"]["total_found"] == 5
            assert result["data"]["summary"]["returned"] == 3
            assert result["data"]["summary"]["limit_applied"] == 3

    @pytest.mark.asyncio
    async def test_km_list_action_types_invalid_category(self) -> None:
        """Test listing action types with invalid category."""
        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch(
                "src.server.tools.action_tools.ActionCategory",
            ) as mock_category_class,
        ):
            mock_registry = Mock()
            mock_category_class.side_effect = ValueError("Invalid category")
            mock_registry_class.return_value = mock_registry

            result = await km_list_action_types(category="invalid_category")

            assert not result["success"]
            assert result["error"]["code"] == "ACTION_LIST_ERROR"
            assert "Failed to list action types" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_km_list_action_types_with_context(self) -> None:
        """Test listing action types with context."""
        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()

        with patch(
            "src.server.tools.action_tools.ActionRegistry",
        ) as mock_registry_class:
            mock_registry = Mock()
            mock_registry.list_all_actions.return_value = []
            mock_registry.get_action_count.return_value = 85
            mock_registry.get_category_counts.return_value = {}
            mock_registry_class.return_value = mock_registry

            result = await km_list_action_types(ctx=mock_ctx)

            assert result["success"]
            mock_ctx.info.assert_called_once()

    @pytest.mark.asyncio
    @given(
        st.one_of(
            st.none(),
            st.sampled_from(["text", "system", "application", "control"]),
        ),
        st.one_of(st.none(), search_term_strategy()),
        st.integers(min_value=1, max_value=200),
    )
    async def test_km_list_action_types_property_validation(
        self,
        category: str | None,
        search: str | None,
        limit: int,
    ) -> None:
        """Property test for km_list_action_types behavior."""
        if search is not None:
            assume(len(search.strip()) > 0)

        with patch(
            "src.server.tools.action_tools.ActionRegistry",
        ) as mock_registry_class:
            mock_registry = Mock()

            # Create mock actions
            mock_actions = []
            for i in range(
                min(limit + 10, 50),
            ):  # Create more than limit to test limiting
                mock_action = Mock()
                mock_action.identifier = f"Action {i}"
                mock_action.category.value = "test"
                mock_action.description = f"Test action {i}"
                mock_action.required_params = []
                mock_action.optional_params = []
                mock_actions.append(mock_action)

            if category:
                mock_registry.get_actions_by_category.return_value = mock_actions
            else:
                mock_registry.list_all_actions.return_value = mock_actions

            mock_registry.get_action_count.return_value = 85
            mock_registry.get_category_counts.return_value = {}
            mock_registry_class.return_value = mock_registry

            result = await km_list_action_types(
                category=category,
                search=search,
                limit=limit,
            )

            # Properties that should always hold
            assert isinstance(result, dict)
            assert "success" in result

            if result["success"]:
                assert "data" in result
                assert "actions" in result["data"]
                assert "summary" in result["data"]
                assert len(result["data"]["actions"]) <= limit

                # Verify metadata
                assert "metadata" in result
                assert "timestamp" in result["metadata"]
                assert "correlation_id" in result["metadata"]

                # Verify summary fields
                summary = result["data"]["summary"]
                assert "total_available" in summary
                assert "total_found" in summary
                assert "returned" in summary
                assert summary["returned"] == len(result["data"]["actions"])


class TestHelperFunctions:
    """Test helper functions."""

    def test_add_action_to_km_macro(self) -> None:
        """Test _add_action_to_km_macro helper function."""
        mock_client = Mock()
        macro_id = "test_macro"
        action_xml = "<action><string>Type a String</string><text>Hello</text></action>"
        position = 5

        result = _add_action_to_km_macro(mock_client, macro_id, action_xml, position)

        assert isinstance(result, bool)
        assert result  # Currently returns True as placeholder

    def test_truncate_xml_for_preview_short(self) -> None:
        """Test XML truncation with short XML."""
        xml = "<action><string>Type a String</string></action>"
        result = _truncate_xml_for_preview(xml, max_length=500)

        assert result == xml  # Should not be truncated

    def test_truncate_xml_for_preview_long(self) -> None:
        """Test XML truncation with long XML."""
        xml = "<action>" + "a" * 600 + "</action>"
        result = _truncate_xml_for_preview(xml, max_length=500)

        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")
        assert result.startswith("<action>")

    def test_truncate_xml_for_preview_exact_length(self) -> None:
        """Test XML truncation with exact max length."""
        xml = "a" * 500
        result = _truncate_xml_for_preview(xml, max_length=500)

        assert result == xml  # Should not be truncated
        assert len(result) == 500


class TestActionToolsIntegration:
    """Integration tests for action tools workflows."""

    @pytest.mark.asyncio
    async def test_complete_action_workflow(self) -> None:
        """Test complete action addition workflow."""
        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch("src.server.tools.action_tools.ActionBuilder") as mock_builder_class,
            patch("src.server.tools.action_tools.get_km_client") as mock_get_client,
            patch(
                "src.server.tools.action_tools._add_action_to_km_macro",
            ) as mock_add_action,
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup comprehensive mocks
            mock_registry = Mock()
            mock_builder = Mock()
            mock_action_def = Mock()
            mock_action_def.category.value = "text"
            mock_action_def.required_params = ["text"]
            mock_action_def.optional_params = ["by_typing", "speed"]

            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {"valid": True}
            mock_registry.get_action_count.return_value = 85

            mock_builder.actions = []
            mock_builder.get_action_count.return_value = 1
            mock_builder.build_xml.return_value = {
                "success": True,
                "xml": "<action><string>Type a String</string><text>Integration Test</text></action>",
            }

            mock_registry_class.return_value = mock_registry
            mock_builder_class.return_value = mock_builder
            mock_get_client.return_value = Mock()
            mock_add_action.return_value = True

            # Mock asyncio loop
            mock_loop = Mock()
            mock_loop.run_in_executor = AsyncMock(return_value=True)
            mock_get_loop.return_value = mock_loop

            # Test complete workflow
            result = await km_add_action(
                macro_id="integration_test_macro",
                action_type="Type a String",
                action_config={
                    "text": "Integration Test Message",
                    "by_typing": True,
                    "speed": "normal",
                },
                position=0,
                timeout=60,
                enabled=True,
                abort_on_failure=False,
            )

            # Verify complete workflow
            assert result["success"]

            # Verify action was configured correctly
            action_data = result["data"]["action_added"]
            assert action_data["action_type"] == "Type a String"
            assert action_data["category"] == "text"
            assert action_data["macro_id"] == "integration_test_macro"
            assert action_data["position"] == 0
            assert action_data["timeout"] == 60
            assert action_data["enabled"]
            assert not action_data["abort_on_failure"]
            assert action_data["parameter_count"] == 3

            # Verify XML generation
            assert "xml_preview" in result["data"]
            assert "Type a String" in result["data"]["xml_preview"]

            # Verify validation
            validation = result["data"]["validation"]
            assert validation["xml_validated"]
            assert validation["parameters_validated"]
            assert validation["security_passed"]

            # Verify integration
            integration = result["data"]["integration"]
            assert integration["km_client_status"] == "connected"
            assert integration["macro_exists"]
            assert integration["action_inserted"]

            # Verify metadata
            metadata = result["metadata"]
            assert "timestamp" in metadata
            assert "correlation_id" in metadata
            assert "execution_time_seconds" in metadata
            assert metadata["action_registry_size"] == 85

    @pytest.mark.asyncio
    async def test_error_propagation_workflow(self) -> None:
        """Test error propagation through action tools workflow."""
        with patch(
            "src.server.tools.action_tools.ActionRegistry",
        ) as mock_registry_class:
            # Test case 1: Registry creation failure
            mock_registry_class.side_effect = Exception(
                "Registry initialization failed",
            )

            result = await km_add_action(
                macro_id="test_macro",
                action_type="Type a String",
                action_config={"text": "test"},
            )

            assert not result["success"]
            assert result["error"]["code"] == "ACTION_ADDITION_ERROR"
            assert "Registry initialization failed" in result["error"]["message"]

            # Test case 2: Reset and test builder failure
            mock_registry_class.side_effect = None
            mock_registry = Mock()
            mock_action_def = Mock()
            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {"valid": True}
            mock_registry_class.return_value = mock_registry

            with patch(
                "src.server.tools.action_tools.ActionBuilder",
            ) as mock_builder_class:
                mock_builder_class.side_effect = Exception(
                    "Builder initialization failed",
                )

                result = await km_add_action(
                    macro_id="test_macro",
                    action_type="Type a String",
                    action_config={"text": "test"},
                )

                assert not result["success"]
                assert result["error"]["code"] == "ACTION_ADDITION_ERROR"
                assert "Builder initialization failed" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_action_type_discovery_workflow(self) -> None:
        """Test action type discovery and usage workflow."""
        with patch(
            "src.server.tools.action_tools.ActionRegistry",
        ) as mock_registry_class:
            mock_registry = Mock()

            # Create comprehensive action list for discovery
            mock_actions = [
                self._create_mock_action(
                    "Type a String",
                    "text",
                    "Type text",
                    ["text"],
                    ["by_typing"],
                ),
                self._create_mock_action(
                    "Pause",
                    "system",
                    "Pause execution",
                    ["duration"],
                    [],
                ),
                self._create_mock_action(
                    "Play Sound",
                    "system",
                    "Play system sound",
                    ["sound"],
                    ["volume"],
                ),
                self._create_mock_action(
                    "If Then Else",
                    "control",
                    "Conditional execution",
                    ["condition"],
                    ["else_actions"],
                ),
            ]

            mock_registry.list_all_actions.return_value = mock_actions
            mock_registry.get_action_count.return_value = len(mock_actions)
            mock_registry.get_category_counts.return_value = {
                ActionCategory.TEXT: 1,
                ActionCategory.SYSTEM: 2,
                ActionCategory.CONTROL: 1,
            }
            mock_registry_class.return_value = mock_registry

            # Test discovery
            discovery_result = await km_list_action_types(search="system")

            assert discovery_result["success"]
            assert (
                len(discovery_result["data"]["actions"]) == 1
            )  # Only "Play Sound" matches "system" search

            # Find specific action for usage
            play_sound_action = next(
                action
                for action in discovery_result["data"]["actions"]
                if action["identifier"] == "Play Sound"
            )

            assert play_sound_action["category"] == "system"
            assert "sound" in play_sound_action["required_parameters"]
            assert "volume" in play_sound_action["optional_parameters"]

    def _create_mock_action(
        self,
        identifier: str,
        category: str,
        description: str,
        required_params: list[str],
        optional_params: list[str],
    ) -> Mock:
        """Create mock action for testing."""
        mock_action = Mock()
        mock_action.identifier = identifier
        mock_action.category.value = category
        mock_action.description = description
        mock_action.required_params = required_params
        mock_action.optional_params = optional_params
        return mock_action


class TestActionToolsErrorHandling:
    """Test comprehensive error handling scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_action_addition(self) -> None:
        """Test concurrent action addition scenarios."""
        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch("src.server.tools.action_tools.ActionBuilder") as mock_builder_class,
            patch("src.server.tools.action_tools.get_km_client") as mock_get_client,
            patch(
                "src.server.tools.action_tools._add_action_to_km_macro",
            ) as mock_add_action,
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks for concurrent execution
            mock_registry = Mock()
            mock_builder = Mock()
            mock_action_def = Mock()
            mock_action_def.category.value = "text"
            mock_action_def.required_params = ["text"]
            mock_action_def.optional_params = []

            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {"valid": True}
            mock_registry.get_action_count.return_value = 85

            mock_builder.actions = []
            mock_builder.get_action_count.return_value = 1
            mock_builder.build_xml.return_value = {
                "success": True,
                "xml": "<action>test</action>",
            }

            mock_registry_class.return_value = mock_registry
            mock_builder_class.return_value = mock_builder
            mock_get_client.return_value = Mock()
            mock_add_action.return_value = True

            # Mock asyncio loop
            mock_loop = Mock()
            mock_loop.run_in_executor = AsyncMock(return_value=True)
            mock_get_loop.return_value = mock_loop

            # Run concurrent operations
            tasks = [
                km_add_action(
                    macro_id=f"test_macro_{i}",
                    action_type="Type a String",
                    action_config={"text": f"Message {i}"},
                )
                for i in range(3)
            ]

            results = await asyncio.gather(*tasks)

            # Verify all operations succeeded
            for i, result in enumerate(results):
                assert result["success"]
                assert result["data"]["action_added"]["macro_id"] == f"test_macro_{i}"

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        """Test timeout handling in action addition."""
        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch("src.server.tools.action_tools.ActionBuilder") as mock_builder_class,
            patch("src.server.tools.action_tools.get_km_client") as mock_get_client,
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks
            mock_registry = Mock()
            mock_builder = Mock()
            mock_action_def = Mock()
            mock_action_def.category.value = "text"
            mock_action_def.required_params = ["text"]
            mock_action_def.optional_params = []

            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {"valid": True}
            mock_registry.get_action_count.return_value = 85

            mock_builder.actions = []
            mock_builder.get_action_count.return_value = 1
            mock_builder.build_xml.return_value = {
                "success": True,
                "xml": "<action>test</action>",
            }

            mock_registry_class.return_value = mock_registry
            mock_builder_class.return_value = mock_builder
            mock_get_client.return_value = Mock()

            # Mock timeout in executor
            mock_loop = Mock()
            mock_loop.run_in_executor = AsyncMock(
                side_effect=asyncio.TimeoutError("Operation timed out"),
            )
            mock_get_loop.return_value = mock_loop

            result = await km_add_action(
                macro_id="test_macro",
                action_type="Type a String",
                action_config={"text": "test"},
                timeout=1,  # Very short timeout
            )

            assert not result["success"]
            assert result["error"]["code"] == "ACTION_ADDITION_ERROR"

    @pytest.mark.asyncio
    async def test_malformed_xml_handling(self) -> None:
        """Test handling of malformed XML generation."""
        with (
            patch(
                "src.server.tools.action_tools.ActionRegistry",
            ) as mock_registry_class,
            patch("src.server.tools.action_tools.ActionBuilder") as mock_builder_class,
        ):
            mock_registry = Mock()
            mock_builder = Mock()
            mock_action_def = Mock()

            mock_registry.get_action_type.return_value = mock_action_def
            mock_registry.validate_action_parameters.return_value = {"valid": True}

            # Mock XML generation returning malformed XML
            mock_builder.build_xml.return_value = {
                "success": False,
                "error": "XML parsing failed: mismatched tags",
            }

            mock_registry_class.return_value = mock_registry
            mock_builder_class.return_value = mock_builder

            result = await km_add_action(
                macro_id="test_macro",
                action_type="Type a String",
                action_config={"text": "test"},
            )

            assert not result["success"]
            assert result["error"]["code"] == "VALIDATION_ERROR"
            assert (
                "Validation failed for field 'xml_generation'"
                in result["error"]["message"]
            )
            assert "mismatched tags" in result["error"]["message"]
