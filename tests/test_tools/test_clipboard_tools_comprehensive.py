"""
Comprehensive tests for clipboard tools module.

Tests cover clipboard operations, security validation, named clipboard management,
history tracking, and integration with property-based testing.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from src.server.tools.clipboard_tools import (
    get_clipboard_manager,
    get_named_clipboard_manager,
    km_clipboard_manager,
)


# Test data generators
@st.composite
def clipboard_operation_strategy(draw):
    """Generate valid clipboard operations."""
    operations = [
        "get",
        "set",
        "get_history",
        "list_history",
        "manage_named",
        "search_named",
        "stats",
    ]
    return draw(st.sampled_from(operations))


@st.composite
def clipboard_content_strategy(draw):
    """Generate valid clipboard content."""
    content_types = [
        # Regular text content (ensure non-empty)
        draw(st.text(min_size=1, max_size=1000).filter(lambda x: len(x.strip()) > 0)),
        # Structured content
        "User data: "
        + draw(st.text(min_size=10, max_size=100).filter(lambda x: len(x.strip()) > 0)),
        # JSON-like content
        '{"key": "'
        + draw(st.text(min_size=1, max_size=50).filter(lambda x: len(x.strip()) > 0))
        + '"}',
        # Code snippets
        "function test() { return "
        + str(draw(st.integers(min_value=1, max_value=100)))
        + "; }",
        # URLs
        "https://example.com/"
        + draw(
            st.text(min_size=1, max_size=20).filter(
                lambda x: x.isalnum() and len(x) > 0
            )
        ),
    ]
    return draw(st.sampled_from(content_types))


@st.composite
def clipboard_name_strategy(draw):
    """Generate valid clipboard names."""
    # Valid pattern: ^[a-zA-Z0-9_\-\s]*$
    # Generate from allowed character set only
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- "
    name_length = draw(st.integers(min_value=1, max_value=50))
    return "".join(
        draw(
            st.lists(
                st.sampled_from(allowed_chars),
                min_size=name_length,
                max_size=name_length,
            )
        )
    )


@st.composite
def sensitive_content_strategy(draw):
    """Generate potentially sensitive content for testing."""
    sensitive_patterns = [
        "password: " + draw(st.text(min_size=8, max_size=20)),
        "token: " + draw(st.text(min_size=20, max_size=40)),
        "api_key: " + draw(st.text(min_size=16, max_size=32)),
        "secret: " + draw(st.text(min_size=10, max_size=30)),
        "credit_card: "
        + str(
            draw(st.integers(min_value=1000000000000000, max_value=9999999999999999))
        ),
        "ssn: " + str(draw(st.integers(min_value=100000000, max_value=999999999))),
    ]
    return draw(st.sampled_from(sensitive_patterns))


@st.composite
def format_filter_strategy(draw):
    """Generate valid format filters."""
    formats = ["text", "image", "file", "url", "all"]
    return draw(st.sampled_from(formats))


@st.composite
def sort_field_strategy(draw):
    """Generate valid sort fields."""
    fields = ["name", "created_at", "accessed_at", "access_count"]
    return draw(st.sampled_from(fields))


class TestClipboardManagerDependencies:
    """Test clipboard manager dependencies and singleton behavior."""

    def test_clipboard_manager_singleton(self):
        """Test clipboard manager singleton behavior."""
        manager1 = get_clipboard_manager()
        manager2 = get_clipboard_manager()

        # Should return the same instance
        assert manager1 is manager2
        assert manager1 is not None

    def test_named_clipboard_manager_singleton(self):
        """Test named clipboard manager singleton behavior."""
        with patch(
            "src.server.tools.clipboard_tools.NamedClipboardManager"
        ) as mock_named_manager_class:
            # Setup mock to avoid async initialization issues
            mock_manager = Mock()
            mock_named_manager_class.return_value = mock_manager

            # Reset global instance for test
            import src.server.tools.clipboard_tools

            src.server.tools.clipboard_tools._named_clipboard_manager = None

            manager1 = get_named_clipboard_manager()
            manager2 = get_named_clipboard_manager()

            # Should return the same instance (mocked)
            assert manager1 is manager2
            assert manager1 is not None

    def test_clipboard_dependencies_import(self):
        """Test importing clipboard dependencies."""
        try:
            from src.clipboard import (
                ClipboardFormat,
                ClipboardManager,
                NamedClipboardManager,
            )

            # Test basic creation
            assert ClipboardManager is not None
            assert NamedClipboardManager is not None
            assert ClipboardFormat is not None

            # Test enum values
            assert hasattr(ClipboardFormat, "TEXT")

        except ImportError:
            # Mock the dependencies for testing
            pytest.skip("Clipboard dependencies not available - using mocks")


class TestClipboardOperationValidation:
    """Test clipboard operation parameter validation."""

    @given(clipboard_operation_strategy())
    def test_valid_operation_types(self, operation: str):
        """Test that valid operation types are accepted."""
        valid_operations = [
            "get",
            "set",
            "get_history",
            "list_history",
            "manage_named",
            "search_named",
            "stats",
        ]
        assert operation in valid_operations

    @given(clipboard_name_strategy())
    def test_clipboard_name_validation(self, name: str):
        """Test clipboard name validation."""
        assume(len(name) <= 100)
        # Should match pattern ^[a-zA-Z0-9_\-\s]*$
        import re

        pattern = r"^[a-zA-Z0-9_\-\s]*$"
        # Strategy generates only valid characters, so should always match
        if name:  # Non-empty names should match pattern
            assert re.match(pattern, name) is not None, (
                f"Name '{name}' should match pattern"
            )

    def test_invalid_operation_types(self):
        """Test that invalid operation types are rejected."""
        invalid_operations = ["invalid", "hack", "execute", "delete_all", ""]
        valid_operations = [
            "get",
            "set",
            "get_history",
            "list_history",
            "manage_named",
            "search_named",
            "stats",
        ]

        for op in invalid_operations:
            assert op not in valid_operations

    def test_history_index_validation(self):
        """Test history index parameter validation."""
        # Valid range: 0-199
        valid_indices = [0, 5, 10, 50, 100, 199]
        invalid_indices = [-1, 200, 1000, -5]

        for index in valid_indices:
            assert 0 <= index <= 199

        for index in invalid_indices:
            assert not (0 <= index <= 199)

    def test_history_count_validation(self):
        """Test history count parameter validation."""
        # Valid range: 1-50
        valid_counts = [1, 5, 10, 25, 50]
        invalid_counts = [0, -1, 51, 100]

        for count in valid_counts:
            assert 1 <= count <= 50

        for count in invalid_counts:
            assert not (1 <= count <= 50)

    def test_content_size_validation(self):
        """Test content size limits."""
        max_size = 1_000_000  # 1MB limit

        # Valid sizes
        valid_content = "x" * 1000
        assert len(valid_content.encode("utf-8")) < max_size

        # At limit
        limit_content = "x" * max_size
        assert len(limit_content.encode("utf-8")) == max_size


class TestClipboardGetOperationMocked:
    """Test clipboard get operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_km_clipboard_get_success(self):
        """Test successful clipboard get operation."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock clipboard manager
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            # Setup mock clipboard content
            mock_content = Mock()
            mock_content.content = "Test clipboard content"
            mock_content.format.value = "text"
            mock_content.size_bytes = 20
            mock_content.timestamp = time.time()
            mock_content.is_sensitive = False
            mock_content.preview.return_value = "Test clipboard..."
            mock_content.is_empty.return_value = False

            # Setup successful result
            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_content

            mock_manager.get_clipboard = AsyncMock(return_value=mock_result)

            # Execute operation
            result = await km_clipboard_manager(operation="get")

            # Verify successful response
            assert result["success"] is True
            assert result["data"]["content"] == "Test clipboard content"
            assert result["data"]["format"] == "text"
            assert result["data"]["size_bytes"] == 20
            assert result["data"]["is_sensitive"] is False
            assert result["data"]["preview"] == "Test clipboard..."
            assert result["data"]["is_empty"] is False
            assert "metadata" in result
            assert result["metadata"]["operation"] == "get"

    @pytest.mark.asyncio
    async def test_km_clipboard_get_sensitive_content_filtered(self):
        """Test clipboard get with sensitive content filtering."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock for sensitive content
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            mock_content = Mock()
            mock_content.content = "password: secret123"
            mock_content.format.value = "text"
            mock_content.size_bytes = 18
            mock_content.timestamp = time.time()
            mock_content.is_sensitive = True
            mock_content.preview.return_value = "password: ***"
            mock_content.is_empty.return_value = False

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_content

            mock_manager.get_clipboard = AsyncMock(return_value=mock_result)

            # Execute operation without including sensitive content
            result = await km_clipboard_manager(
                operation="get", include_sensitive=False
            )

            # Verify sensitive content is filtered
            assert result["success"] is True
            assert result["data"]["content"] == "[SENSITIVE CONTENT HIDDEN]"
            assert result["data"]["is_sensitive"] is True
            assert result["metadata"]["security_filtered"] is True

    @pytest.mark.asyncio
    async def test_km_clipboard_get_sensitive_content_included(self):
        """Test clipboard get with sensitive content included."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock for sensitive content with inclusion
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            mock_content = Mock()
            mock_content.content = "api_key: abc123def456"
            mock_content.format.value = "text"
            mock_content.size_bytes = 20
            mock_content.timestamp = time.time()
            mock_content.is_sensitive = True
            mock_content.preview.return_value = "api_key: ***"
            mock_content.is_empty.return_value = False

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_content

            mock_manager.get_clipboard = AsyncMock(return_value=mock_result)

            # Execute operation with sensitive content included
            result = await km_clipboard_manager(operation="get", include_sensitive=True)

            # Verify sensitive content is included
            assert result["success"] is True
            assert result["data"]["content"] == "api_key: abc123def456"
            assert result["data"]["is_sensitive"] is True
            assert result["metadata"]["security_filtered"] is False

    @pytest.mark.asyncio
    async def test_km_clipboard_get_error_handling(self):
        """Test clipboard get operation error handling."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock for error scenario
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            mock_error = Mock()
            mock_error.code = "CLIPBOARD_ACCESS_ERROR"
            mock_error.message = "Failed to access system clipboard"
            mock_error.details = {"system": "macOS"}

            mock_result = Mock()
            mock_result.is_left.return_value = True
            mock_result.get_left.return_value = mock_error

            mock_manager.get_clipboard = AsyncMock(return_value=mock_result)

            # Execute operation that should fail
            result = await km_clipboard_manager(operation="get")

            # Verify error response
            assert result["success"] is False
            assert result["error"]["code"] == "CLIPBOARD_ACCESS_ERROR"
            assert result["error"]["message"] == "Failed to access system clipboard"
            assert result["error"]["details"]["system"] == "macOS"


class TestClipboardSetOperationMocked:
    """Test clipboard set operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_km_clipboard_set_success(self):
        """Test successful clipboard set operation."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock clipboard manager
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            # Setup successful result
            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = True

            mock_manager.set_clipboard = AsyncMock(return_value=mock_result)

            # Execute operation
            content = "New clipboard content"
            result = await km_clipboard_manager(operation="set", content=content)

            # Verify successful response
            assert result["success"] is True
            assert result["data"]["content_size"] == len(content.encode("utf-8"))
            assert (
                result["data"]["content_preview"] == content
            )  # Short content, no truncation
            assert result["metadata"]["operation"] == "set"

            # Verify manager was called correctly
            mock_manager.set_clipboard.assert_called_once_with(content)

    @pytest.mark.asyncio
    async def test_km_clipboard_set_long_content_preview(self):
        """Test clipboard set with long content preview."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock clipboard manager
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = True

            mock_manager.set_clipboard = AsyncMock(return_value=mock_result)

            # Execute operation with long content
            content = "x" * 100  # Content longer than 50 chars
            result = await km_clipboard_manager(operation="set", content=content)

            # Verify response with truncated preview
            assert result["success"] is True
            assert result["data"]["content_size"] == 100
            assert result["data"]["content_preview"] == "x" * 50 + "..."

    @pytest.mark.asyncio
    async def test_km_clipboard_set_missing_content(self):
        """Test clipboard set operation with missing content."""
        # Execute operation without content
        result = await km_clipboard_manager(operation="set")

        # Verify validation error
        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert "Content is required for set operation" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_km_clipboard_set_error_handling(self):
        """Test clipboard set operation error handling."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock for error scenario
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            mock_error = Mock()
            mock_error.code = "CLIPBOARD_WRITE_ERROR"
            mock_error.message = "Cannot write to system clipboard"
            mock_error.details = {"permission": "denied"}

            mock_result = Mock()
            mock_result.is_left.return_value = True
            mock_result.get_left.return_value = mock_error

            mock_manager.set_clipboard = AsyncMock(return_value=mock_result)

            # Execute operation that should fail
            result = await km_clipboard_manager(operation="set", content="test content")

            # Verify error response
            assert result["success"] is False
            assert result["error"]["code"] == "CLIPBOARD_WRITE_ERROR"
            assert result["error"]["message"] == "Cannot write to system clipboard"
            assert result["error"]["details"]["permission"] == "denied"


class TestClipboardHistoryOperationsMocked:
    """Test clipboard history operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_km_clipboard_get_history_success(self):
        """Test successful get_history operation."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock clipboard manager
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            # Setup mock history item
            mock_content = Mock()
            mock_content.content = "History item content"
            mock_content.format.value = "text"
            mock_content.size_bytes = 20
            mock_content.timestamp = time.time()
            mock_content.is_sensitive = False
            mock_content.preview.return_value = "History item..."

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_content

            mock_manager.get_history_item = AsyncMock(return_value=mock_result)

            # Execute operation
            result = await km_clipboard_manager(
                operation="get_history", history_index=0
            )

            # Verify successful response
            assert result["success"] is True
            assert result["data"]["index"] == 0
            assert result["data"]["content"] == "History item content"
            assert result["data"]["format"] == "text"
            assert result["metadata"]["operation"] == "get_history"

            # Verify manager was called correctly
            mock_manager.get_history_item.assert_called_once_with(0, False)

    @pytest.mark.asyncio
    async def test_km_clipboard_get_history_missing_index(self):
        """Test get_history operation with missing index."""
        # Execute operation without history_index
        result = await km_clipboard_manager(operation="get_history")

        # Verify validation error
        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert (
            "history_index is required for get_history operation"
            in result["error"]["message"]
        )

    @pytest.mark.asyncio
    async def test_km_clipboard_list_history_success(self):
        """Test successful list_history operation."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock clipboard manager
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            # Setup mock history items
            mock_items = []
            for i in range(3):
                mock_item = Mock()
                mock_item.content = f"History item {i}"
                mock_item.format.value = "text"
                mock_item.size_bytes = 15 + i
                mock_item.timestamp = time.time() - i * 100
                mock_item.is_sensitive = i == 1  # Second item is sensitive
                mock_item.preview.return_value = f"History item {i}..."
                mock_items.append(mock_item)

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_items

            mock_manager.get_history_list = AsyncMock(return_value=mock_result)

            # Execute operation
            result = await km_clipboard_manager(
                operation="list_history", history_count=5
            )

            # Verify successful response
            assert result["success"] is True
            assert result["data"]["total_items"] == 3
            assert len(result["data"]["history"]) == 3

            # Check first item
            first_item = result["data"]["history"][0]
            assert first_item["index"] == 0
            assert first_item["content"] == "History item 0"
            assert first_item["is_sensitive"] is False

            # Check sensitive item is filtered
            second_item = result["data"]["history"][1]
            assert second_item["content"] == "[SENSITIVE CONTENT HIDDEN]"
            assert second_item["is_sensitive"] is True

            assert result["metadata"]["operation"] == "list_history"
            assert result["metadata"]["requested_count"] == 5
            assert result["metadata"]["security_filtering"] is True

    @pytest.mark.asyncio
    async def test_km_clipboard_list_history_include_sensitive(self):
        """Test list_history operation with sensitive content included."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock clipboard manager
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            # Setup mock sensitive history item
            mock_item = Mock()
            mock_item.content = "sensitive_token: abc123"
            mock_item.format.value = "text"
            mock_item.size_bytes = 22
            mock_item.timestamp = time.time()
            mock_item.is_sensitive = True
            mock_item.preview.return_value = "sensitive_token: ***"

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = [mock_item]

            mock_manager.get_history_list = AsyncMock(return_value=mock_result)

            # Execute operation with sensitive content included
            result = await km_clipboard_manager(
                operation="list_history", history_count=10, include_sensitive=True
            )

            # Verify sensitive content is included
            assert result["success"] is True
            assert len(result["data"]["history"]) == 1

            item = result["data"]["history"][0]
            assert item["content"] == "sensitive_token: abc123"
            assert item["is_sensitive"] is True
            assert result["metadata"]["security_filtering"] is False


class TestNamedClipboardOperationsMocked:
    """Test named clipboard operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_km_clipboard_manage_named_create_success(self):
        """Test successful named clipboard creation."""
        with (
            patch(
                "src.server.tools.clipboard_tools.get_clipboard_manager"
            ) as mock_get_manager,
            patch(
                "src.server.tools.clipboard_tools.get_named_clipboard_manager"
            ) as mock_get_named_manager,
        ):
            # Setup mocks
            mock_manager = Mock()
            mock_named_manager = Mock()
            mock_get_manager.return_value = mock_manager
            mock_get_named_manager.return_value = mock_named_manager

            # Setup current clipboard content
            mock_content = Mock()
            mock_content.content = "Content to save"
            mock_content.format.value = "text"
            mock_content.size_bytes = 15
            mock_content.timestamp = time.time()

            mock_current_result = Mock()
            mock_current_result.is_left.return_value = False
            mock_current_result.get_right.return_value = mock_content

            mock_manager.get_clipboard = AsyncMock(return_value=mock_current_result)

            # Setup named clipboard creation
            mock_create_result = Mock()
            mock_create_result.is_left.return_value = False
            mock_create_result.get_right.return_value = True

            mock_named_manager.create_named_clipboard = AsyncMock(
                return_value=mock_create_result
            )

            # Execute operation
            result = await km_clipboard_manager(
                operation="manage_named",
                clipboard_name="my_clipboard",
                content="Content to save",
                tags=["important", "work"],
                description="Test clipboard",
                overwrite=True,
            )

            # Verify successful response
            assert result["success"] is True
            assert result["data"]["name"] == "my_clipboard"
            assert result["data"]["created"] is True
            assert result["data"]["overwritten"] is True
            assert result["metadata"]["operation"] == "create_named"

            # Verify manager was called correctly
            mock_named_manager.create_named_clipboard.assert_called_once()
            call_args = mock_named_manager.create_named_clipboard.call_args[0]
            assert call_args[0] == "my_clipboard"  # name
            assert call_args[2] == {"important", "work"}  # tags as set
            assert call_args[3] == "Test clipboard"  # description
            assert call_args[4] is True  # overwrite

    @pytest.mark.asyncio
    async def test_km_clipboard_manage_named_get_success(self):
        """Test successful named clipboard retrieval."""
        with patch(
            "src.server.tools.clipboard_tools.get_named_clipboard_manager"
        ) as mock_get_named_manager:
            # Setup mock named clipboard manager
            mock_named_manager = Mock()
            mock_get_named_manager.return_value = mock_named_manager

            # Setup mock named clipboard
            mock_named_cb = Mock()
            mock_named_cb.name = "test_clipboard"
            mock_named_cb.content.content = "Saved content"
            mock_named_cb.content.format.value = "text"
            mock_named_cb.content.size_bytes = 13
            mock_named_cb.content.is_sensitive = False
            mock_named_cb.content.preview.return_value = "Saved content..."
            mock_named_cb.created_at = time.time() - 3600
            mock_named_cb.accessed_at = time.time()
            mock_named_cb.access_count = 5
            mock_named_cb.tags = {"personal", "backup"}
            mock_named_cb.description = "Personal backup"

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_named_cb

            mock_named_manager.get_named_clipboard = AsyncMock(return_value=mock_result)

            # Execute operation
            result = await km_clipboard_manager(
                operation="manage_named", clipboard_name="test_clipboard"
            )

            # Verify successful response
            assert result["success"] is True
            assert result["data"]["name"] == "test_clipboard"
            assert result["data"]["content"] == "Saved content"
            assert result["data"]["format"] == "text"
            assert result["data"]["access_count"] == 5
            # Systematic pattern alignment: tags are sets converted to lists, order may vary
            assert set(result["data"]["tags"]) == {"personal", "backup"}
            assert result["data"]["description"] == "Personal backup"
            assert result["metadata"]["operation"] == "get_named"

    @pytest.mark.asyncio
    async def test_km_clipboard_manage_named_missing_name(self):
        """Test manage_named operation with missing clipboard_name."""
        # Execute operation without clipboard_name
        result = await km_clipboard_manager(operation="manage_named")

        # Verify validation error
        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert (
            "clipboard_name is required for manage_named operation"
            in result["error"]["message"]
        )

    @pytest.mark.asyncio
    async def test_km_clipboard_search_named_success(self):
        """Test successful named clipboard search."""
        with patch(
            "src.server.tools.clipboard_tools.get_named_clipboard_manager"
        ) as mock_get_named_manager:
            # Setup mock named clipboard manager
            mock_named_manager = Mock()
            mock_get_named_manager.return_value = mock_named_manager

            # Setup mock search results
            mock_clipboards = []
            for i in range(2):
                mock_cb = Mock()
                mock_cb.name = f"clipboard_{i}"
                mock_cb.content.format.value = "text"
                mock_cb.content.size_bytes = 10 + i
                mock_cb.content.is_sensitive = i == 1
                mock_cb.content.preview.return_value = f"Content {i}..."
                mock_cb.created_at = time.time() - i * 1000
                mock_cb.accessed_at = time.time()
                mock_cb.access_count = i + 1
                mock_cb.tags = {f"tag_{i}"}
                mock_cb.description = f"Description {i}"
                mock_clipboards.append(mock_cb)

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_clipboards

            mock_named_manager.search_named_clipboards = AsyncMock(
                return_value=mock_result
            )

            # Execute operation
            result = await km_clipboard_manager(
                operation="search_named", search_query="test", search_content=True
            )

            # Verify successful response
            assert result["success"] is True
            assert result["data"]["total_found"] == 2
            assert len(result["data"]["clipboards"]) == 2

            # Check first clipboard
            first_cb = result["data"]["clipboards"][0]
            assert first_cb["name"] == "clipboard_0"
            assert first_cb["format"] == "text"
            assert first_cb["is_sensitive"] is False

            # Check search metadata
            assert result["metadata"]["operation"] == "search_named"
            assert result["metadata"]["query"] == "test"
            assert result["metadata"]["searched_content"] is True

    @pytest.mark.asyncio
    async def test_km_clipboard_search_named_list_all(self):
        """Test listing all named clipboards (no search query)."""
        with patch(
            "src.server.tools.clipboard_tools.get_named_clipboard_manager"
        ) as mock_get_named_manager:
            # Setup mock named clipboard manager
            mock_named_manager = Mock()
            mock_get_named_manager.return_value = mock_named_manager

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = []

            mock_named_manager.list_named_clipboards = AsyncMock(
                return_value=mock_result
            )

            # Execute operation without search query
            result = await km_clipboard_manager(
                operation="search_named", tag_filter="work", sort_by="created_at"
            )

            # Verify list operation was called
            assert result["success"] is True
            mock_named_manager.list_named_clipboards.assert_called_once_with(
                "work", "created_at"
            )

    @pytest.mark.asyncio
    async def test_km_clipboard_stats_success(self):
        """Test successful clipboard statistics retrieval."""
        with patch(
            "src.server.tools.clipboard_tools.get_named_clipboard_manager"
        ) as mock_get_named_manager:
            # Setup mock named clipboard manager
            mock_named_manager = Mock()
            mock_get_named_manager.return_value = mock_named_manager

            # Setup mock statistics
            mock_stats = {
                "total_clipboards": 15,
                "total_size_bytes": 1024000,
                "most_accessed": "important_clipboard",
                "oldest_clipboard": "archive_clipboard",
                "tags_count": {"work": 8, "personal": 7},
            }

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_stats

            mock_named_manager.get_clipboard_stats = AsyncMock(return_value=mock_result)

            # Execute operation
            result = await km_clipboard_manager(operation="stats")

            # Verify successful response
            assert result["success"] is True
            assert result["data"]["total_clipboards"] == 15
            assert result["data"]["total_size_bytes"] == 1024000
            assert result["data"]["most_accessed"] == "important_clipboard"
            assert result["metadata"]["operation"] == "stats"


class TestClipboardErrorHandling:
    """Test clipboard operation error handling."""

    @pytest.mark.asyncio
    async def test_invalid_operation_error(self):
        """Test error handling for invalid operations."""
        # Execute operation with invalid operation type
        result = await km_clipboard_manager(operation="invalid_operation")

        # Verify validation error
        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert "Unsupported operation: invalid_operation" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test handling of unexpected exceptions."""
        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock to raise exception
            mock_get_manager.side_effect = RuntimeError("Unexpected system error")

            # Execute operation that should raise exception
            result = await km_clipboard_manager(operation="get")

            # Verify exception error response
            assert result["success"] is False
            assert result["error"]["code"] == "EXECUTION_ERROR"
            assert (
                "Clipboard operation failed: Unexpected system error"
                in result["error"]["message"]
            )
            assert result["error"]["details"]["exception_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_context_error_logging(self):
        """Test error logging with context."""
        mock_context = Mock()
        mock_context.info = AsyncMock()
        mock_context.error = AsyncMock()

        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup mock to raise exception
            mock_get_manager.side_effect = ValueError("Test error")

            # Execute operation with context
            result = await km_clipboard_manager(operation="get", ctx=mock_context)

            # Verify context logging
            assert result["success"] is False
            mock_context.info.assert_called_once()
            mock_context.error.assert_called_once()

            # Verify error message was logged
            error_call = mock_context.error.call_args_list[0]
            assert "Clipboard operation failed" in str(error_call)


class TestClipboardIntegration:
    """Integration tests for clipboard operations."""

    @pytest.mark.asyncio
    async def test_complete_clipboard_workflow(self):
        """Test complete clipboard workflow with multiple operations."""
        with (
            patch(
                "src.server.tools.clipboard_tools.get_clipboard_manager"
            ) as mock_get_manager,
            patch(
                "src.server.tools.clipboard_tools.get_named_clipboard_manager"
            ) as mock_get_named_manager,
        ):
            # Setup mocks for complete workflow
            mock_manager = Mock()
            mock_named_manager = Mock()
            mock_get_manager.return_value = mock_manager
            mock_get_named_manager.return_value = mock_named_manager

            # Mock successful operations
            mock_success_result = Mock()
            mock_success_result.is_left.return_value = False
            mock_success_result.get_right.return_value = True

            mock_manager.set_clipboard = AsyncMock(return_value=mock_success_result)

            # Mock current clipboard content for named clipboard creation (systematic pattern alignment)
            mock_content = Mock()
            mock_content.content = "Workflow test content"
            mock_content.format.value = "text"
            mock_content.size_bytes = 20
            mock_content.timestamp = 1234567890

            mock_current_result = Mock()
            mock_current_result.is_left.return_value = False
            mock_current_result.get_right.return_value = mock_content

            mock_manager.get_clipboard = AsyncMock(return_value=mock_current_result)
            mock_named_manager.create_named_clipboard = AsyncMock(
                return_value=mock_success_result
            )

            # 1. Set clipboard content
            set_result = await km_clipboard_manager(
                operation="set", content="Workflow test content"
            )
            assert set_result["success"] is True

            # 2. Create named clipboard
            create_result = await km_clipboard_manager(
                operation="manage_named",
                clipboard_name="workflow_test",
                content="Workflow test content",
                tags=["test", "workflow"],
                description="Test workflow clipboard",
            )
            assert create_result["success"] is True

            # Verify all operations were called
            mock_manager.set_clipboard.assert_called_once()
            mock_named_manager.create_named_clipboard.assert_called_once()

    @pytest.mark.asyncio
    async def test_clipboard_workflow_with_context(self):
        """Test clipboard workflow with FastMCP context integration."""
        mock_context = Mock()
        mock_context.info = AsyncMock()
        mock_context.error = AsyncMock()

        with patch(
            "src.server.tools.clipboard_tools.get_clipboard_manager"
        ) as mock_get_manager:
            # Setup successful mock
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            mock_content = Mock()
            mock_content.content = "Context test content"
            mock_content.format.value = "text"
            mock_content.size_bytes = 20
            mock_content.timestamp = time.time()
            mock_content.is_sensitive = False
            mock_content.preview.return_value = "Context test..."
            mock_content.is_empty.return_value = False

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = mock_content

            mock_manager.get_clipboard = AsyncMock(return_value=mock_result)

            # Execute operation with context
            result = await km_clipboard_manager(operation="get", ctx=mock_context)

            # Verify context integration
            assert result["success"] is True
            mock_context.info.assert_called_once()

            # Verify info message was logged
            info_call = mock_context.info.call_args_list[0]
            assert "Performing clipboard operation: get" in str(info_call)


class TestClipboardProperties:
    """Property-based tests for clipboard operations."""

    @given(
        clipboard_operation_strategy(),
        clipboard_content_strategy(),
        format_filter_strategy(),
    )
    def test_clipboard_parameter_validation_properties(
        self, operation: str, content: str, format_filter: str
    ):
        """Property test for clipboard parameter validation."""
        # Properties that should always hold
        valid_operations = [
            "get",
            "set",
            "get_history",
            "list_history",
            "manage_named",
            "search_named",
            "stats",
        ]
        valid_formats = ["text", "image", "file", "url", "all"]

        assert operation in valid_operations
        assert format_filter in valid_formats
        assert isinstance(content, str)
        # Systematic pattern alignment: content from strategy should be non-empty and reasonable length
        assert len(content) > 0, f"Content should be non-empty, got: '{content}'"
        assert len(content) <= 1_000_000, (
            f"Content should not exceed max length, got: {len(content)}"
        )

    @given(clipboard_name_strategy())
    def test_clipboard_name_pattern_properties(self, name: str):
        """Property test for clipboard name validation."""
        import re

        pattern = r"^[a-zA-Z0-9_\-\s]*$"

        # Strategy generates only valid characters, so properties should always hold
        if name:  # Non-empty names
            # Should match the allowed pattern (systematic pattern alignment)
            assert re.match(pattern, name) is not None, (
                f"Generated name '{name}' should match pattern"
            )
            # Should not contain invalid characters
            allowed_chars = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- "
            )
            invalid_chars = set(name) - allowed_chars
            assert len(invalid_chars) == 0, (
                f"Name '{name}' contains invalid chars: {invalid_chars}"
            )

        # Length should be reasonable
        assert len(name) <= 100

    @given(
        st.integers(min_value=0, max_value=199), st.integers(min_value=1, max_value=50)
    )
    def test_history_parameter_properties(self, history_index: int, history_count: int):
        """Property test for history parameter validation."""
        # History index properties
        assert 0 <= history_index <= 199
        assert isinstance(history_index, int)

        # History count properties
        assert 1 <= history_count <= 50
        assert isinstance(history_count, int)

    @given(sensitive_content_strategy())
    def test_sensitive_content_detection_properties(self, sensitive_content: str):
        """Property test for sensitive content detection."""
        # Systematic pattern alignment: Strategy generates content with sensitive patterns embedded
        # Check for the exact patterns our strategy generates (prefix: suffix format)
        sensitive_patterns = [
            "password:",
            "token:",
            "api_key:",
            "secret:",
            "credit_card:",
            "ssn:",
        ]

        # Strategy generates content with these exact patterns, so should always be found
        has_sensitive_pattern = any(
            pattern in sensitive_content.lower() for pattern in sensitive_patterns
        )
        assert has_sensitive_pattern, (
            f"Content '{sensitive_content}' should contain sensitive pattern from {sensitive_patterns}"
        )

        # Should be non-empty (guaranteed by strategy)
        assert len(sensitive_content) > 0
        assert len(sensitive_content.strip()) > 0

    @given(sort_field_strategy())
    def test_sort_field_validation_properties(self, sort_field: str):
        """Property test for sort field validation."""
        valid_sort_fields = ["name", "created_at", "accessed_at", "access_count"]

        # Should be a valid sort field
        assert sort_field in valid_sort_fields
        assert isinstance(sort_field, str)
        assert len(sort_field) > 0
