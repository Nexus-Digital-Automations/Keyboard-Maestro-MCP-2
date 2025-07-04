"""
Property-Based Tests for Clipboard Manager

This module implements comprehensive property-based testing for the clipboard
management system using Hypothesis for security validation and edge case detection.
"""

import pytest
import asyncio
import time
from hypothesis import given, strategies as st, assume, settings
from typing import List, Set

from src.clipboard import ClipboardManager, ClipboardContent, ClipboardFormat
from src.clipboard.named_clipboards import NamedClipboardManager, NamedClipboard
from src.integration.km_client import KMError


class TestClipboardManager:
    """Test suite for ClipboardManager with property-based validation."""
    
    @pytest.fixture
    def clipboard_manager(self):
        """Create clipboard manager instance for testing."""
        return ClipboardManager()
    
    @pytest.fixture
    def named_manager(self):
        """Create named clipboard manager for testing."""
        return NamedClipboardManager()
    
    @pytest.mark.asyncio
    @given(st.text(min_size=1, max_size=1000))
    async def test_clipboard_content_detection(self, clipboard_manager, content: str):
        """Property: Sensitive content detection should be consistent."""
        assume(len(content.encode('utf-8')) <= 1_000_000)
        
        # Test sensitive content detection
        is_sensitive = clipboard_manager._detect_sensitive_content(content)
        
        # Should be deterministic
        assert clipboard_manager._detect_sensitive_content(content) == is_sensitive
        
        # Empty content should not be sensitive
        if not content.strip():
            assert not is_sensitive
    
    @pytest.mark.asyncio
    @given(st.text(max_size=50))
    async def test_format_detection_consistency(self, clipboard_manager, content: str):
        """Property: Format detection should be consistent and valid."""
        format_type = clipboard_manager._detect_format(content)
        
        # Should be deterministic
        assert clipboard_manager._detect_format(content) == format_type
        
        # Should be a valid format
        assert isinstance(format_type, ClipboardFormat)
        
        # URL detection should be accurate
        if content.startswith(('http://', 'https://')):
            if ' ' not in content and '\n' not in content:
                assert format_type == ClipboardFormat.URL
    
    @pytest.mark.asyncio
    @given(st.integers(min_value=0, max_value=199))
    async def test_history_bounds_checking(self, clipboard_manager, index: int):
        """Property: History access should respect bounds."""
        # This will typically fail as we don't have real clipboard history
        # but validates the bounds checking logic
        result = await clipboard_manager.get_history_item(index, include_sensitive=True)
        
        # Should either succeed or fail with appropriate error
        if result.is_left():
            error = result.get_left()
            assert error.code in ["NOT_FOUND_ERROR", "EXECUTION_ERROR", "ACCESS_ERROR"]
    
    @pytest.mark.asyncio
    @given(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    async def test_named_clipboard_validation(self, named_manager, name: str):
        """Property: Named clipboard names should be validated consistently."""
        # Clean the name to test validation
        cleaned_name = ''.join(c for c in name if c.isalnum() or c in ' _-').strip()
        assume(len(cleaned_name) > 0 and len(cleaned_name) <= 100)
        
        # Create test content
        test_content = ClipboardContent(
            content="test content",
            format=ClipboardFormat.TEXT,
            size_bytes=12,
            timestamp=time.time()
        )
        
        # Test creation
        result = await named_manager.create_named_clipboard(cleaned_name, test_content)
        
        # Should either succeed or fail with validation error
        if result.is_left():
            error = result.get_left()
            assert error.code in ["VALIDATION_ERROR", "NAME_CONFLICT", "STORAGE_FULL", "EXECUTION_ERROR"]
        else:
            # If successful, should be retrievable
            get_result = await named_manager.get_named_clipboard(cleaned_name)
            assert get_result.is_right()
            
            retrieved = get_result.get_right()
            assert retrieved.name == cleaned_name
            assert retrieved.content.content == "test content"
            
            # Clean up
            await named_manager.delete_named_clipboard(cleaned_name)
    
    @pytest.mark.asyncio
    @given(st.sets(st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum()), max_size=10))
    async def test_named_clipboard_tags(self, named_manager, tags: Set[str]):
        """Property: Tags should be properly validated and stored."""
        name = f"test_tags_{int(time.time())}"
        
        test_content = ClipboardContent(
            content="test content with tags",
            format=ClipboardFormat.TEXT,
            size_bytes=22,
            timestamp=time.time()
        )
        
        result = await named_manager.create_named_clipboard(name, test_content, tags=tags)
        
        if result.is_right():
            # Retrieve and verify tags
            get_result = await named_manager.get_named_clipboard(name)
            assert get_result.is_right()
            
            retrieved = get_result.get_right()
            # Tags should be normalized (lowercased)
            expected_tags = {tag.lower() for tag in tags if tag.isalnum()}
            assert retrieved.tags == expected_tags
            
            # Clean up
            await named_manager.delete_named_clipboard(name)
    
    @pytest.mark.asyncio
    async def test_sensitive_content_patterns(self, clipboard_manager):
        """Test specific sensitive content patterns."""
        sensitive_patterns = [
            "password=secret123",
            "api_key=abc123def456",
            "token:Bearer xyz789",
            "user@example.com",
            "1234-5678-9012-3456",  # Credit card pattern
            "123-45-6789",  # SSN pattern
            "ABCDEF1234567890ABCDEF1234567890",  # Hash pattern
        ]
        
        for pattern in sensitive_patterns:
            is_sensitive = clipboard_manager._detect_sensitive_content(pattern)
            assert is_sensitive, f"Pattern '{pattern}' should be detected as sensitive"
        
        # Non-sensitive content
        safe_patterns = [
            "Hello, world!",
            "This is regular text",
            "No sensitive data here",
            "123",
            "test",
        ]
        
        for pattern in safe_patterns:
            is_sensitive = clipboard_manager._detect_sensitive_content(pattern)
            assert not is_sensitive, f"Pattern '{pattern}' should not be detected as sensitive"
    
    @pytest.mark.asyncio
    async def test_clipboard_content_immutability(self):
        """Test that ClipboardContent is properly immutable."""
        content = ClipboardContent(
            content="test",
            format=ClipboardFormat.TEXT,
            size_bytes=4,
            timestamp=time.time()
        )
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            content.content = "modified"
        
        with pytest.raises(AttributeError):
            content.format = ClipboardFormat.URL
    
    @pytest.mark.asyncio
    async def test_named_clipboard_access_tracking(self, named_manager):
        """Test that access tracking works correctly."""
        name = f"test_access_{int(time.time())}"
        
        test_content = ClipboardContent(
            content="access tracking test",
            format=ClipboardFormat.TEXT,
            size_bytes=19,
            timestamp=time.time()
        )
        
        # Create clipboard
        result = await named_manager.create_named_clipboard(name, test_content)
        assert result.is_right()
        
        # Get it multiple times and verify access tracking
        for i in range(3):
            get_result = await named_manager.get_named_clipboard(name)
            assert get_result.is_right()
            
            retrieved = get_result.get_right()
            assert retrieved.access_count == i + 1
        
        # Clean up
        await named_manager.delete_named_clipboard(name)
    
    @pytest.mark.asyncio
    async def test_clipboard_preview_safety(self):
        """Test that clipboard preview respects sensitivity settings."""
        # Sensitive content
        sensitive_content = ClipboardContent(
            content="password=secret123",
            format=ClipboardFormat.TEXT,
            size_bytes=17,
            timestamp=time.time(),
            is_sensitive=True
        )
        
        preview = sensitive_content.preview()
        assert preview == "[SENSITIVE CONTENT HIDDEN]"
        
        # Non-sensitive content
        safe_content = ClipboardContent(
            content="Hello, world!",
            format=ClipboardFormat.TEXT,
            size_bytes=13,
            timestamp=time.time(),
            is_sensitive=False
        )
        
        preview = safe_content.preview()
        assert preview == "Hello, world!"
        
        # Long content truncation
        long_content = ClipboardContent(
            content="A" * 100,
            format=ClipboardFormat.TEXT,
            size_bytes=100,
            timestamp=time.time()
        )
        
        preview = long_content.preview(max_length=50)
        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")
    
    @pytest.mark.asyncio
    @given(st.text(min_size=1, max_size=20))
    async def test_search_functionality(self, named_manager, query: str):
        """Property: Search should be case-insensitive and consistent."""
        # Create some test clipboards
        test_data = [
            ("search_test_1", "This contains the search term"),
            ("search_test_2", "This does not contain it"),
            ("search_test_3", f"This contains {query.upper()}"),
        ]
        
        created_names = []
        for name, content in test_data:
            unique_name = f"{name}_{int(time.time())}"
            test_content = ClipboardContent(
                content=content,
                format=ClipboardFormat.TEXT,
                size_bytes=len(content),
                timestamp=time.time()
            )
            
            result = await named_manager.create_named_clipboard(unique_name, test_content)
            if result.is_right():
                created_names.append(unique_name)
        
        # Perform search
        if created_names and query.strip():
            search_result = await named_manager.search_named_clipboards(query, search_content=True)
            
            if search_result.is_right():
                found_clipboards = search_result.get_right()
                # Search should be case-insensitive
                for clipboard in found_clipboards:
                    content_lower = clipboard.content.content.lower() if isinstance(clipboard.content.content, str) else ""
                    name_lower = clipboard.name.lower()
                    query_lower = query.lower()
                    
                    # Should match either name or content
                    assert query_lower in content_lower or query_lower in name_lower
        
        # Clean up
        for name in created_names:
            await named_manager.delete_named_clipboard(name)


@pytest.mark.asyncio
class TestClipboardSecurity:
    """Security-focused property-based tests for clipboard operations."""
    
    @given(st.text(min_size=1, max_size=10000))
    @settings(max_examples=50, deadline=None)
    async def test_content_size_limits(self, content: str):
        """Property: Content size limits should be enforced."""
        clipboard_manager = ClipboardManager()
        
        content_bytes = content.encode('utf-8')
        
        # Test size validation in ClipboardContent
        if len(content_bytes) <= 100_000_000:  # 100MB limit
            clipboard_content = ClipboardContent(
                content=content,
                format=ClipboardFormat.TEXT,
                size_bytes=len(content_bytes),
                timestamp=time.time()
            )
            assert clipboard_content.size_bytes == len(content_bytes)
        else:
            # Should fail validation
            with pytest.raises(Exception):
                ClipboardContent(
                    content=content,
                    format=ClipboardFormat.TEXT,
                    size_bytes=len(content_bytes),
                    timestamp=time.time()
                )
    
    @given(st.text(min_size=1, max_size=1000))
    async def test_applescript_injection_prevention(self, content: str):
        """Property: AppleScript strings should be properly escaped."""
        clipboard_manager = ClipboardManager()
        
        # Test escaping function
        escaped = clipboard_manager._escape_applescript_string(content)
        
        # Should not contain unescaped dangerous characters
        assert '\\"' not in escaped or escaped.count('\\"') == escaped.count('"')
        assert '\\n' in escaped or '\n' not in content
        assert '\\r' in escaped or '\r' not in content
        assert '\\t' in escaped or '\t' not in content
    
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=100))
    async def test_storage_limits(self, clipboard_names: List[str]):
        """Property: Named clipboard storage should respect limits."""
        named_manager = NamedClipboardManager()
        
        created_count = 0
        max_allowed = named_manager._max_clipboards
        
        for i, name in enumerate(clipboard_names):
            # Create valid name
            clean_name = f"test_{i}_{int(time.time())}"
            
            test_content = ClipboardContent(
                content=f"test content {i}",
                format=ClipboardFormat.TEXT,
                size_bytes=len(f"test content {i}"),
                timestamp=time.time()
            )
            
            result = await named_manager.create_named_clipboard(clean_name, test_content)
            
            if result.is_right():
                created_count += 1
                
                # Should not exceed maximum
                assert created_count <= max_allowed
            else:
                error = result.get_left()
                if error.code == "STORAGE_FULL":
                    # Once full, no more should be created
                    assert created_count >= max_allowed
                    break
        
        # Clean up what we can
        for i in range(min(created_count, 50)):  # Limit cleanup for test performance
            clean_name = f"test_{i}_{int(time.time())}"
            await named_manager.delete_named_clipboard(clean_name)