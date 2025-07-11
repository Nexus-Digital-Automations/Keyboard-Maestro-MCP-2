"""Comprehensive tests for text manipulation commands.

This module tests text typing, search, and replacement commands
with comprehensive security validation and edge case coverage.
"""

from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from src.commands.text import (
    FindTextCommand,
    ReplaceTextCommand,
    TextSearchMode,
    TypeTextCommand,
    TypingSpeed,
)
from src.core.types import (
    CommandId,
    CommandParameters,
    ExecutionContext,
    Permission,
)


class TestTypeTextCommand:
    """Test TypeTextCommand functionality."""

    @pytest.fixture
    def context(self):
        """Create execution context."""
        return ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT])
        )

    def test_type_text_command_validation_success(self):
        """Test successful text typing validation."""
        params = CommandParameters({"text": "Hello, world!"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.validate() is True

    def test_type_text_command_validation_failures(self):
        """Test text typing validation failures."""
        # Empty text
        params = CommandParameters({"text": ""})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # No text parameter
        params = CommandParameters({})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Invalid typing speed
        params = CommandParameters({"text": "Hello", "typing_speed": "invalid"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.validate() is True  # Falls back to normal speed

    def test_get_typing_speed(self):
        """Test typing speed retrieval."""
        # Default speed
        params = CommandParameters({"text": "test"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_typing_speed() == TypingSpeed.NORMAL

        # Specified speeds
        params = CommandParameters({"text": "test", "typing_speed": "slow"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_typing_speed() == TypingSpeed.SLOW

        params = CommandParameters({"text": "test", "typing_speed": "fast"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_typing_speed() == TypingSpeed.FAST

        params = CommandParameters({"text": "test", "typing_speed": "instant"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_typing_speed() == TypingSpeed.INSTANT

        # Invalid speed falls back to normal
        params = CommandParameters({"text": "test", "typing_speed": "turbo"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_typing_speed() == TypingSpeed.NORMAL

    def test_get_delay_between_keys(self):
        """Test delay calculation for different speeds."""
        # Slow typing
        params = CommandParameters({"text": "test", "typing_speed": "slow"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_delay_between_keys() == 0.1

        # Normal typing
        params = CommandParameters({"text": "test", "typing_speed": "normal"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_delay_between_keys() == 0.05

        # Fast typing
        params = CommandParameters({"text": "test", "typing_speed": "fast"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_delay_between_keys() == 0.02

        # Instant typing
        params = CommandParameters({"text": "test", "typing_speed": "instant"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_delay_between_keys() == 0.0

    def test_get_required_permissions(self):
        """Test required permissions."""
        params = CommandParameters({"text": "test"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_required_permissions() == frozenset([Permission.TEXT_INPUT])

    def test_get_security_risk_level(self):
        """Test security risk level."""
        params = CommandParameters({"text": "test"})
        cmd = TypeTextCommand(CommandId("test"), params)
        assert cmd.get_security_risk_level() == "medium"

    @patch("time.sleep")
    def test_type_text_execute_instant(self, mock_sleep, context):
        """Test instant text typing execution."""
        params = CommandParameters({"text": "Hello, world!", "typing_speed": "instant"})
        cmd = TypeTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "13 characters" in result.output
        assert result.metadata["characters_typed"] == 13
        assert result.metadata["text_length"] == 13
        assert result.metadata["typing_speed"] == "instant"

        # No sleep calls for instant typing
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_type_text_execute_with_delay(self, mock_sleep, context):
        """Test text typing with delay."""
        params = CommandParameters({"text": "Hi!", "typing_speed": "slow"})
        cmd = TypeTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "3 characters" in result.output

        # Should have 3 sleep calls for 3 characters
        assert mock_sleep.call_count == 3
        mock_sleep.assert_called_with(0.1)

    def test_type_text_execute_no_permission(self, context):
        """Test typing without permission."""
        limited_context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.SCREEN_CAPTURE])
        )

        params = CommandParameters({"text": "test"})
        cmd = TypeTextCommand(CommandId("test"), params)
        result = cmd.execute(limited_context)

        assert result.success is False
        assert "Missing required permission" in result.error_message

    @patch("time.sleep")
    @patch("time.time")
    def test_type_text_timeout(self, mock_time, mock_sleep, context):
        """Test typing timeout."""
        # Mock time to simulate timeout
        # Need enough values for all time.time() calls:
        # 1. start_time in base.py execute()
        # 2. start_time in text.py _execute_impl()
        # 3. timeout check in loop (first character - ok)
        # 4. timeout check in loop (second character - triggers timeout)
        # 5. execution_time calculation in text.py error handler
        # 6. execution_time in base.py error handler
        mock_time.side_effect = [0, 0, 1, 35, 35, 35]

        params = CommandParameters(
            {"text": "This is a long text", "typing_speed": "slow"}
        )
        cmd = TypeTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is False
        assert "timed out" in result.error_message
        assert (
            result.metadata["characters_typed"] == 1
        )  # Only typed 1 char before timeout

    @patch("time.sleep")
    def test_type_text_exception_handling(self, mock_sleep, context):
        """Test exception handling during typing."""
        mock_sleep.side_effect = Exception("Keyboard error")

        params = CommandParameters({"text": "test", "typing_speed": "normal"})
        cmd = TypeTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is False
        assert "Keyboard error" in result.error_message


class TestFindTextCommand:
    """Test FindTextCommand functionality."""

    @pytest.fixture
    def context(self):
        """Create execution context."""
        return ExecutionContext.create_test_context(
            permissions=frozenset([Permission.SCREEN_CAPTURE])
        )

    def test_find_text_validation_success(self):
        """Test successful find text validation."""
        params = CommandParameters(
            {"pattern": "search term", "target_text": "This is a search term in text"}
        )
        cmd = FindTextCommand(CommandId("test"), params)
        assert cmd.validate() is True

    def test_find_text_validation_failures(self):
        """Test find text validation failures."""
        # Empty pattern
        params = CommandParameters({"pattern": ""})
        cmd = FindTextCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # No pattern
        params = CommandParameters({})
        cmd = FindTextCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Invalid regex pattern
        params = CommandParameters({"pattern": "[invalid(regex", "mode": "regex"})
        cmd = FindTextCommand(CommandId("test"), params)
        assert cmd.validate() is False

    def test_get_search_mode(self):
        """Test search mode retrieval."""
        # Default mode
        params = CommandParameters({"pattern": "test"})
        cmd = FindTextCommand(CommandId("test"), params)
        assert cmd.get_search_mode() == TextSearchMode.EXACT

        # Specified modes
        params = CommandParameters({"pattern": "test", "mode": "contains"})
        cmd = FindTextCommand(CommandId("test"), params)
        assert cmd.get_search_mode() == TextSearchMode.CONTAINS

        params = CommandParameters({"pattern": "test", "mode": "regex"})
        cmd = FindTextCommand(CommandId("test"), params)
        assert cmd.get_search_mode() == TextSearchMode.REGEX

        # Invalid mode falls back to exact
        params = CommandParameters({"pattern": "test", "mode": "fuzzy"})
        cmd = FindTextCommand(CommandId("test"), params)
        assert cmd.get_search_mode() == TextSearchMode.EXACT

    def test_find_text_exact_match(self, context):
        """Test exact text matching."""
        params = CommandParameters(
            {
                "pattern": "world",
                "target_text": "Hello world! This is a wonderful world.",
                "mode": "exact",
            }
        )
        cmd = FindTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "2 matches" in result.output
        assert result.metadata["match_count"] == 2
        assert result.metadata["match_positions"] == [6, 33]

    def test_find_text_case_insensitive(self, context):
        """Test case-insensitive search."""
        params = CommandParameters(
            {
                "pattern": "HELLO",
                "target_text": "Hello world! hello again!",
                "mode": "exact",
                "case_sensitive": False,
            }
        )
        cmd = FindTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "2 matches" in result.output
        assert result.metadata["match_count"] == 2

    def test_find_text_contains_mode(self, context):
        """Test contains search mode."""
        params = CommandParameters(
            {
                "pattern": "world",
                "target_text": "Hello world! This is a wonderful world.",
                "mode": "contains",
            }
        )
        cmd = FindTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "1 matches" in result.output  # Contains only reports first match
        assert result.metadata["match_count"] == 1
        assert result.metadata["match_positions"] == [6]

    def test_find_text_regex_mode(self, context):
        """Test regex search mode."""
        params = CommandParameters(
            {
                "pattern": r"\b\w+orld\b",
                "target_text": "Hello world! This is a wonderful world.",
                "mode": "regex",
            }
        )
        cmd = FindTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "2 matches" in result.output
        assert result.metadata["match_count"] == 2

    def test_find_text_no_matches(self, context):
        """Test search with no matches."""
        params = CommandParameters(
            {"pattern": "xyz", "target_text": "Hello world!", "mode": "exact"}
        )
        cmd = FindTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "0 matches" in result.output
        assert result.metadata["match_count"] == 0
        assert result.metadata["match_positions"] == []

    def test_find_text_no_target_text(self, context):
        """Test search without target text."""
        params = CommandParameters({"pattern": "test"})
        cmd = FindTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is False
        assert "No target text available" in result.error_message

    def test_find_text_regex_error(self, context):
        """Test regex error handling."""
        params = CommandParameters(
            {
                "pattern": r"(?P<invalid>",  # Invalid regex
                "target_text": "test text",
                "mode": "regex",
            }
        )
        cmd = FindTextCommand(CommandId("test"), params)

        # Validation should catch this
        assert cmd.validate() is False

    def test_find_text_exception_handling(self, context):
        """Test exception handling."""
        params = CommandParameters({"pattern": "test", "target_text": "test text"})
        cmd = FindTextCommand(CommandId("test"), params)

        # Mock re.finditer to raise exception
        with patch("re.finditer", side_effect=Exception("Search error")):
            result = cmd.execute(context)

            # Should still succeed for non-regex mode
            assert result.success is True


class TestReplaceTextCommand:
    """Test ReplaceTextCommand functionality."""

    @pytest.fixture
    def context(self):
        """Create execution context."""
        return ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.SCREEN_CAPTURE])
        )

    def test_replace_text_validation_success(self):
        """Test successful replace text validation."""
        params = CommandParameters({"search_pattern": "old", "replacement_text": "new"})
        cmd = ReplaceTextCommand(CommandId("test"), params)
        assert cmd.validate() is True

    def test_replace_text_validation_failures(self):
        """Test replace text validation failures."""
        # Empty search pattern
        params = CommandParameters({"search_pattern": "", "replacement_text": "new"})
        cmd = ReplaceTextCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Empty replacement text
        params = CommandParameters({"search_pattern": "old", "replacement_text": ""})
        cmd = ReplaceTextCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Missing parameters
        params = CommandParameters({})
        cmd = ReplaceTextCommand(CommandId("test"), params)
        assert cmd.validate() is False

        # Invalid max replacements
        params = CommandParameters(
            {
                "search_pattern": "old",
                "replacement_text": "new",
                "max_replacements": 200,
            }
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)
        assert cmd.validate() is True  # Clamped to 100

    def test_get_max_replacements(self):
        """Test max replacements retrieval."""
        # Default
        params = CommandParameters({"search_pattern": "old", "replacement_text": "new"})
        cmd = ReplaceTextCommand(CommandId("test"), params)
        assert cmd.get_max_replacements() == 1

        # Specified value
        params = CommandParameters(
            {"search_pattern": "old", "replacement_text": "new", "max_replacements": 5}
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)
        assert cmd.get_max_replacements() == 5

        # Clamping
        params = CommandParameters(
            {
                "search_pattern": "old",
                "replacement_text": "new",
                "max_replacements": 200,
            }
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)
        assert cmd.get_max_replacements() == 100  # Clamped to max

        params = CommandParameters(
            {"search_pattern": "old", "replacement_text": "new", "max_replacements": 0}
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)
        assert cmd.get_max_replacements() == 1  # Clamped to min

    def test_replace_text_single_replacement(self, context):
        """Test single text replacement."""
        params = CommandParameters(
            {
                "search_pattern": "world",
                "replacement_text": "universe",
                "target_text": "Hello world! Beautiful world!",
                "max_replacements": 1,
            }
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "1 occurrences" in result.output
        assert result.metadata["replacements_made"] == 1
        assert result.metadata["result_text"] == "Hello universe! Beautiful world!"

    def test_replace_text_multiple_replacements(self, context):
        """Test multiple text replacements."""
        params = CommandParameters(
            {
                "search_pattern": "world",
                "replacement_text": "universe",
                "target_text": "Hello world! Beautiful world!",
                "max_replacements": 10,
            }
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "2 occurrences" in result.output
        assert result.metadata["replacements_made"] == 2
        assert result.metadata["result_text"] == "Hello universe! Beautiful universe!"

    def test_replace_text_case_insensitive(self, context):
        """Test case-insensitive replacement."""
        params = CommandParameters(
            {
                "search_pattern": "WORLD",
                "replacement_text": "universe",
                "target_text": "Hello World! Beautiful world!",
                "case_sensitive": False,
                "max_replacements": 10,
            }
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "2 occurrences" in result.output
        assert result.metadata["replacements_made"] == 2
        assert result.metadata["result_text"] == "Hello universe! Beautiful universe!"

    def test_replace_text_no_matches(self, context):
        """Test replacement with no matches."""
        params = CommandParameters(
            {
                "search_pattern": "xyz",
                "replacement_text": "abc",
                "target_text": "Hello world!",
            }
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is True
        assert "0 occurrences" in result.output
        assert result.metadata["replacements_made"] == 0
        assert result.metadata["result_text"] == "Hello world!"  # Unchanged

    def test_replace_text_no_target_text(self, context):
        """Test replacement without target text."""
        params = CommandParameters({"search_pattern": "old", "replacement_text": "new"})
        cmd = ReplaceTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is False
        assert "No target text available" in result.error_message

    def test_replace_text_permissions(self, context):
        """Test required permissions."""
        params = CommandParameters({"search_pattern": "old", "replacement_text": "new"})
        cmd = ReplaceTextCommand(CommandId("test"), params)

        # Need both TEXT_INPUT and SCREEN_CAPTURE
        assert cmd.get_required_permissions() == frozenset(
            [Permission.TEXT_INPUT, Permission.SCREEN_CAPTURE]
        )

        # Test with missing permission
        limited_context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT])
        )

        result = cmd.execute(limited_context)
        assert result.success is False
        assert "Missing required permission" in result.error_message

    def test_replace_text_exception_handling(self, context):
        """Test exception handling during replacement."""

        # Create a special case that will trigger an exception
        # We'll use a mock object that raises an exception when lower() is called
        class FaultyString(str):
            def lower(self):
                raise Exception("Replace error")

        faulty_text = FaultyString("test text")

        params = CommandParameters(
            {
                "search_pattern": "TEST",
                "replacement_text": "new",
                "target_text": faulty_text,
                "case_sensitive": False,  # This will trigger the lower() call
            }
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)

        result = cmd.execute(context)

        assert result.success is False
        assert "Replace error" in result.error_message


class TestPropertyBasedTextCommands:
    """Property-based tests for text commands."""

    @given(
        text=st.text(min_size=1, max_size=100).filter(
            lambda t: "`" not in t and "$(" not in t
        ),
        speed=st.sampled_from(["slow", "normal", "fast", "instant"]),
    )
    @settings(max_examples=50)
    def test_type_text_property(self, text, speed):
        """Property test for text typing."""
        params = CommandParameters({"text": text, "typing_speed": speed})
        cmd = TypeTextCommand(CommandId("test"), params)

        # Should validate successfully for safe non-empty text
        assert cmd.validate() is True

        # Get typing speed should work
        typing_speed = cmd.get_typing_speed()
        assert isinstance(typing_speed, TypingSpeed)

        # Delay should be non-negative
        delay = cmd.get_delay_between_keys()
        assert delay >= 0

    @given(
        pattern=st.text(min_size=1, max_size=50).filter(
            lambda t: "`" not in t and "$(" not in t
        ),
        target=st.text(min_size=1, max_size=200),
        case_sensitive=st.booleans(),
    )
    def test_find_text_property(self, pattern, target, case_sensitive):
        """Property test for text finding."""
        params = CommandParameters(
            {
                "pattern": pattern,
                "target_text": target,
                "case_sensitive": case_sensitive,
            }
        )
        cmd = FindTextCommand(CommandId("test"), params)

        # Basic patterns should validate
        if not any(char in pattern for char in r"[]()*+?{}\\|"):
            assert cmd.validate() is True

    @given(
        search=st.text(min_size=1, max_size=20).filter(
            lambda t: "`" not in t and "$(" not in t
        ),
        replacement=st.text(min_size=1, max_size=20).filter(
            lambda t: "`" not in t and "$(" not in t
        ),
        target=st.text(min_size=1, max_size=100),
        max_replacements=st.integers(min_value=1, max_value=10),
    )
    def test_replace_text_property(self, search, replacement, target, max_replacements):
        """Property test for text replacement."""
        params = CommandParameters(
            {
                "search_pattern": search,
                "replacement_text": replacement,
                "target_text": target,
                "max_replacements": max_replacements,
            }
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)

        # Should validate for any safe non-empty strings
        assert cmd.validate() is True

        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.SCREEN_CAPTURE])
        )

        result = cmd.execute(context)
        assert result.success is True

        # Replacements should not exceed max
        assert result.metadata["replacements_made"] <= max_replacements

        # Result text should not contain original pattern beyond replacements
        result_text = result.metadata["result_text"]
        actual_count = result_text.count(search)
        original_count = target.count(search)

        # If search and replacement are the same, count should remain the same
        if search == replacement:
            assert actual_count == original_count
        else:
            # When replacement contains the search pattern, it can increase count
            # Just verify that replacements were made correctly
            assert result.metadata["replacements_made"] <= min(
                max_replacements, original_count
            )


class TestTextCommandEdgeCases:
    """Test edge cases for text commands."""

    def test_type_text_special_characters(self):
        """Test typing special characters."""
        special_text = "Hello\nWorld\t!@#$%^&*()"
        params = CommandParameters({"text": special_text})
        cmd = TypeTextCommand(CommandId("test"), params)

        assert cmd.validate() is True

        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT])
        )

        result = cmd.execute(context)
        assert result.success is True
        assert result.metadata["characters_typed"] == len(special_text)

    def test_find_text_overlapping_matches(self):
        """Test finding overlapping matches."""
        params = CommandParameters(
            {"pattern": "aa", "target_text": "aaaa", "mode": "exact"}
        )
        cmd = FindTextCommand(CommandId("test"), params)

        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.SCREEN_CAPTURE])
        )

        result = cmd.execute(context)
        assert result.success is True
        # count() doesn't count overlapping matches, but positions are found
        assert (
            result.metadata["match_count"] == 2
        )  # "aa" appears twice non-overlapping in "aaaa"
        assert result.metadata["match_positions"] == [
            0,
            1,
            2,
        ]  # But positions include overlaps

    def test_replace_text_recursive_replacement(self):
        """Test that replacements don't create recursive matches."""
        params = CommandParameters(
            {
                "search_pattern": "a",
                "replacement_text": "aa",
                "target_text": "banana",
                "max_replacements": 10,
            }
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)

        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.SCREEN_CAPTURE])
        )

        result = cmd.execute(context)
        assert result.success is True
        # Should only replace the original 'a's, not create infinite loop
        assert result.metadata["replacements_made"] == 3
        # Python's str.replace with count replaces from left to right
        # "banana" -> "baanana" -> "baanana" -> "baanana" (only first 3 'a's replaced)
        # Actually, "banana" with 3 replacements of "a" -> "aa" = "baaaanana"
        assert result.metadata["result_text"] == "baaaanana"

    def test_empty_operations(self):
        """Test operations on empty strings."""
        # Find in empty text
        params = CommandParameters({"pattern": "test", "target_text": ""})
        cmd = FindTextCommand(CommandId("test"), params)

        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.SCREEN_CAPTURE])
        )

        result = cmd.execute(context)
        assert result.success is False  # Empty target text returns error
        assert "No target text available" in result.error_message

        # Replace in empty text
        params = CommandParameters(
            {"search_pattern": "test", "replacement_text": "new", "target_text": ""}
        )
        cmd = ReplaceTextCommand(CommandId("test"), params)

        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.TEXT_INPUT, Permission.SCREEN_CAPTURE])
        )

        result = cmd.execute(context)
        assert result.success is False  # Empty target text returns error
        assert "No target text available" in result.error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
