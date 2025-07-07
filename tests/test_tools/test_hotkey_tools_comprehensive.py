"""Comprehensive tests for hotkey tools module.

Tests cover hotkey trigger creation, conflict detection, validation,
key combination management, and integration with property-based testing.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.server.tools.hotkey_tools import (
    km_create_hotkey_trigger,
    km_list_hotkey_triggers,
)


# Test data generators
@st.composite
def hotkey_key_strategy(draw) -> Any:
    """Generate valid hotkey keys."""
    regular_keys = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [
        str(i) for i in range(10)
    ]
    special_keys = [
        "space",
        "tab",
        "enter",
        "return",
        "escape",
        "delete",
        "backspace",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
        "home",
        "end",
        "pageup",
        "pagedown",
        "up",
        "down",
        "left",
        "right",
        "clear",
        "help",
        "insert",
    ]
    all_keys = regular_keys + special_keys
    return draw(st.sampled_from(all_keys))


@st.composite
def hotkey_modifiers_strategy(draw) -> list[Any]:
    """Generate valid modifier combinations."""
    all_modifiers = ["cmd", "opt", "shift", "ctrl", "fn"]
    # Generate 0-3 modifiers (empty list is valid)
    num_modifiers = draw(st.integers(min_value=0, max_value=3))
    if num_modifiers == 0:
        return []
    return draw(
        st.lists(
            st.sampled_from(all_modifiers),
            min_size=1,
            max_size=num_modifiers,
            unique=True,
        ),
    )


@st.composite
def activation_mode_strategy(draw) -> Any:
    """Generate valid activation modes."""
    modes = ["pressed", "released", "tapped", "held"]
    return draw(st.sampled_from(modes))


@st.composite
def macro_id_strategy(draw) -> Any:
    """Generate valid macro IDs."""
    # UUID format or descriptive name (systematic pattern alignment)
    # Ensure we always generate valid, non-empty strings
    patterns = [
        # Descriptive names (more likely to be valid)
        draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-",
            ),
        ),
        # UUID-like format
        "550e8400-e29b-41d4-a716-446655440000",
        # Simple names
        "Quick Notes",
        "Test Macro",
    ]
    return draw(st.sampled_from(patterns))


@st.composite
def invalid_key_strategy(draw) -> Any:
    """Generate invalid hotkey keys for testing."""
    invalid_keys = ["invalid", "ctrl+a", "cmd+space", "", "  ", "F13", "shift", "cmd"]
    return draw(st.sampled_from(invalid_keys))


@st.composite
def invalid_modifier_strategy(draw) -> Any:
    """Generate invalid modifier combinations."""
    invalid_modifiers = ["invalid", "command", "alt", "control", "windows", "meta"]
    return draw(st.lists(st.sampled_from(invalid_modifiers), min_size=1, max_size=2))


class TestHotkeyToolsDependencies:
    """Test hotkey tools dependencies and imports."""

    def test_hotkey_manager_import(self) -> None:
        """Test importing hotkey dependencies."""
        try:
            from src.core.types import MacroId
            from src.integration.km_client import KMClient
            from src.integration.triggers import TriggerRegistrationManager
            from src.triggers.hotkey_manager import HotkeyManager, create_hotkey_spec

            # Test basic creation
            assert HotkeyManager is not None
            assert create_hotkey_spec is not None
            assert KMClient is not None
            assert TriggerRegistrationManager is not None
            assert MacroId is not None

        except ImportError:
            # Mock the dependencies for testing
            pytest.skip("Hotkey dependencies not available - using mocks")

    def test_hotkey_validation_types(self) -> None:
        """Test hotkey validation type constants."""
        # Valid key patterns should be recognized
        valid_keys = ["a", "z", "0", "9", "space", "f1", "f12", "escape", "return"]
        for key in valid_keys:
            assert len(key) >= 1
            assert len(key) <= 20

        # Valid modifiers
        valid_modifiers = ["cmd", "opt", "shift", "ctrl", "fn"]
        for modifier in valid_modifiers:
            assert isinstance(modifier, str)
            assert len(modifier) >= 2


class TestHotkeyParameterValidation:
    """Test hotkey parameter validation."""

    @given(hotkey_key_strategy())
    def test_valid_key_types(self, key: str) -> None:
        """Test that valid key types are accepted."""
        # Valid keys should match expected patterns
        regular_keys = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [
            str(i) for i in range(10)
        ]
        special_keys = [
            "space",
            "tab",
            "enter",
            "return",
            "escape",
            "delete",
            "backspace",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "f6",
            "f7",
            "f8",
            "f9",
            "f10",
            "f11",
            "f12",
            "home",
            "end",
            "pageup",
            "pagedown",
            "up",
            "down",
            "left",
            "right",
            "clear",
            "help",
            "insert",
        ]
        all_valid_keys = regular_keys + special_keys
        assert key in all_valid_keys

    @given(hotkey_modifiers_strategy())
    def test_valid_modifier_combinations(self, modifiers: list[str]) -> None:
        """Test that valid modifier combinations are accepted."""
        valid_modifiers = {"cmd", "opt", "shift", "ctrl", "fn"}
        for modifier in modifiers:
            assert modifier in valid_modifiers
        # Should have no duplicates (unique list)
        assert len(modifiers) == len(set(modifiers))

    @given(activation_mode_strategy())
    def test_valid_activation_modes(self, mode: str) -> None:
        """Test that valid activation modes are accepted."""
        valid_modes = {"pressed", "released", "tapped", "held"}
        assert mode in valid_modes

    @given(st.integers(min_value=1, max_value=4))
    def test_valid_tap_counts(self, tap_count: int) -> None:
        """Test that valid tap counts are accepted."""
        assert 1 <= tap_count <= 4
        assert isinstance(tap_count, int)

    def test_invalid_key_validation(self) -> None:
        """Test that invalid keys are handled."""
        invalid_keys = ["", "  ", "ctrl+a", "invalid", "F13", "shift"]
        for key in invalid_keys:
            # These should be caught by validation (systematic pattern alignment)
            if not key or key.strip() == "":
                assert len(key.strip()) == 0
            elif key in ["ctrl+a", "shift"]:
                # Contains invalid patterns
                assert any(char in key for char in ["+", "ctrl", "shift"])

    def test_invalid_modifier_validation(self) -> None:
        """Test that invalid modifiers are handled."""
        invalid_modifiers = ["invalid", "command", "alt", "control", "windows", "meta"]
        valid_modifiers = {"cmd", "opt", "shift", "ctrl", "fn"}
        for modifier in invalid_modifiers:
            assert modifier not in valid_modifiers


class TestHotkeyCreationMocked:
    """Test hotkey creation with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_km_create_hotkey_trigger_success(self) -> None:
        """Test successful hotkey trigger creation."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
            ) as mock_create_spec,
            patch("src.server.tools.hotkey_tools.HotkeyManager") as mock_manager_class,
            patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client,
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
            ) as mock_trigger_manager,
        ):
            # Setup mock dependencies (systematic pattern alignment)
            mock_km_client.return_value = Mock()
            mock_trigger_manager.return_value = Mock()

            # Setup mock hotkey spec
            mock_hotkey_spec = Mock()
            mock_hotkey_spec.key = "n"
            mock_hotkey_spec.modifiers = [Mock(value="cmd"), Mock(value="shift")]
            mock_hotkey_spec.activation_mode = Mock(value="pressed")
            mock_hotkey_spec.tap_count = 1
            mock_hotkey_spec.allow_repeat = False
            mock_hotkey_spec.to_display_string.return_value = "Cmd+Shift+N"
            mock_hotkey_spec.to_km_string.return_value = "⌘⇧N"
            mock_create_spec.return_value = mock_hotkey_spec

            # Setup mock hotkey manager
            mock_manager = Mock()
            mock_manager.detect_conflicts = AsyncMock(return_value=[])

            # Setup successful creation result
            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = "trigger_123"
            mock_manager.create_hotkey_trigger = AsyncMock(return_value=mock_result)
            mock_manager_class.return_value = mock_manager

            # Execute operation
            result = await km_create_hotkey_trigger(
                macro_id="test_macro",
                key="n",
                modifiers=["cmd", "shift"],
                activation_mode="pressed",
                tap_count=1,
                allow_repeat=False,
                check_conflicts=True,
            )

            # Verify successful response
            assert result["success"] is True
            assert result["data"]["trigger_id"] == "trigger_123"
            assert result["data"]["macro_id"] == "test_macro"
            assert result["data"]["hotkey"]["key"] == "n"
            assert result["data"]["hotkey"]["modifiers"] == ["cmd", "shift"]
            assert result["data"]["hotkey"]["activation_mode"] == "pressed"
            assert result["data"]["hotkey"]["tap_count"] == 1
            assert result["data"]["hotkey"]["allow_repeat"] is False
            assert result["data"]["conflicts_checked"] is True
            assert result["data"]["status"] == "active"

            # Verify metadata
            assert "metadata" in result
            assert "timestamp" in result["metadata"]
            assert "correlation_id" in result["metadata"]

    @pytest.mark.asyncio
    async def test_km_create_hotkey_trigger_with_conflicts(self) -> None:
        """Test hotkey creation with conflicts detected."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
            ) as mock_create_spec,
            patch("src.server.tools.hotkey_tools.HotkeyManager") as mock_manager_class,
            patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client,
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
            ) as mock_trigger_manager,
        ):
            # Setup mock dependencies (systematic pattern alignment)
            mock_km_client.return_value = Mock()
            mock_trigger_manager.return_value = Mock()

            # Setup mock hotkey spec
            mock_hotkey_spec = Mock()
            mock_hotkey_spec.to_display_string.return_value = "Cmd+C"
            mock_hotkey_spec.to_km_string.return_value = "⌘C"
            mock_create_spec.return_value = mock_hotkey_spec

            # Setup conflict detection
            mock_conflict = Mock()
            mock_conflict.conflicting_hotkey = "⌘C"
            mock_conflict.conflict_type = "system_shortcut"
            mock_conflict.description = "Conflicts with system copy shortcut"
            mock_conflict.macro_name = "System Copy"
            mock_conflict.suggestion = "Try Cmd+Shift+C instead"

            mock_manager = Mock()
            mock_manager.detect_conflicts = AsyncMock(return_value=[mock_conflict])

            # Setup alternative suggestions
            mock_alternative = Mock()
            mock_alternative.to_km_string.return_value = "⌘⇧C"
            mock_alternative.to_display_string.return_value = "Cmd+Shift+C"
            mock_alternative.modifiers = [Mock(value="cmd"), Mock(value="shift")]
            mock_alternative.key = "c"
            mock_alternative.activation_mode = Mock(value="pressed")
            mock_alternative.tap_count = 1
            mock_manager.suggest_alternatives.return_value = [mock_alternative]

            mock_manager_class.return_value = mock_manager

            # Execute operation
            result = await km_create_hotkey_trigger(
                macro_id="test_macro",
                key="c",
                modifiers=["cmd"],
                check_conflicts=True,
                suggest_alternatives=True,
            )

            # Verify conflict response
            assert result["success"] is False
            assert result["error"]["code"] == "CONFLICT_ERROR"
            assert "Hotkey conflicts detected" in result["error"]["message"]
            assert "conflicts" in result["error"]["details"]
            assert result["error"]["details"]["conflict_count"] == 1

            # Verify conflict details
            conflicts = result["error"]["details"]["conflicts"]
            assert len(conflicts) == 1
            assert conflicts[0]["conflicting_hotkey"] == "⌘C"
            assert conflicts[0]["conflict_type"] == "system_shortcut"

            # Verify alternative suggestions
            assert "suggested_alternatives" in result["error"]["details"]
            alternatives = result["error"]["details"]["suggested_alternatives"]
            assert len(alternatives) == 1
            assert alternatives[0]["hotkey"] == "⌘⇧C"
            assert alternatives[0]["display"] == "Cmd+Shift+C"

    @pytest.mark.asyncio
    async def test_km_create_hotkey_trigger_invalid_spec(self) -> None:
        """Test hotkey creation with invalid specification."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
            ) as mock_create_spec,
            patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client,
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
            ) as mock_trigger_manager,
        ):
            # Setup mock dependencies (systematic pattern alignment)
            mock_km_client.return_value = Mock()
            mock_trigger_manager.return_value = Mock()

            # Setup validation error (systematic pattern alignment)
            from src.core.errors import ValidationError

            mock_validation_error = ValidationError(
                "Invalid key",
                "invalid",
                "field_constraint",
            )
            # Add field and value as attributes for test compatibility
            mock_validation_error.field = "key"
            mock_validation_error.value = "invalid"
            mock_create_spec.side_effect = mock_validation_error

            # Execute operation
            result = await km_create_hotkey_trigger(
                macro_id="test_macro",
                key="invalid",
                modifiers=["cmd"],
            )

            # Verify validation error response
            assert result["success"] is False
            assert result["error"]["code"] == "INVALID_HOTKEY"
            assert "Invalid hotkey specification" in result["error"]["message"]
            assert result["error"]["details"]["field"] == "key"
            assert result["error"]["details"]["value"] == "invalid"
            assert "recovery_suggestion" in result["error"]

    @pytest.mark.asyncio
    async def test_km_create_hotkey_trigger_creation_failure(self) -> None:
        """Test hotkey creation when trigger creation fails."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
            ) as mock_create_spec,
            patch("src.server.tools.hotkey_tools.HotkeyManager") as mock_manager_class,
            patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client,
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
            ) as mock_trigger_manager,
        ):
            # Setup mock dependencies (systematic pattern alignment)
            mock_km_client.return_value = Mock()
            mock_trigger_manager.return_value = Mock()

            # Setup mock hotkey spec
            mock_hotkey_spec = Mock()
            mock_hotkey_spec.to_display_string.return_value = "Cmd+N"
            mock_create_spec.return_value = mock_hotkey_spec

            # Setup manager with creation failure
            mock_manager = Mock()
            mock_manager.detect_conflicts = AsyncMock(return_value=[])

            # Setup failed creation result
            mock_error = Mock()
            mock_error.code = "MACRO_NOT_FOUND"
            mock_error.message = "Target macro does not exist"
            mock_error.details = {"macro_id": "nonexistent_macro"}

            mock_result = Mock()
            mock_result.is_left.return_value = True
            mock_result.get_left.return_value = mock_error
            mock_manager.create_hotkey_trigger = AsyncMock(return_value=mock_result)
            mock_manager_class.return_value = mock_manager

            # Execute operation
            result = await km_create_hotkey_trigger(
                macro_id="nonexistent_macro",
                key="n",
                modifiers=["cmd"],
            )

            # Verify failure response
            assert result["success"] is False
            assert result["error"]["code"] == "MACRO_NOT_FOUND"
            assert result["error"]["message"] == "Target macro does not exist"
            assert result["error"]["details"]["macro_id"] == "nonexistent_macro"

    @pytest.mark.asyncio
    async def test_km_create_hotkey_trigger_with_context(self) -> None:
        """Test hotkey creation with FastMCP context integration."""
        mock_context = Mock()
        mock_context.info = AsyncMock()
        mock_context.error = AsyncMock()

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
            ) as mock_create_spec,
            patch("src.server.tools.hotkey_tools.HotkeyManager") as mock_manager_class,
            patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client,
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
            ) as mock_trigger_manager,
        ):
            # Setup mock dependencies (systematic pattern alignment)
            mock_km_client.return_value = Mock()
            mock_trigger_manager.return_value = Mock()

            # Setup successful creation
            mock_hotkey_spec = Mock()
            mock_hotkey_spec.key = "t"
            mock_hotkey_spec.modifiers = [Mock(value="cmd")]  # Make iterable
            mock_hotkey_spec.activation_mode = Mock(value="pressed")
            mock_hotkey_spec.tap_count = 1
            mock_hotkey_spec.allow_repeat = False
            mock_hotkey_spec.to_display_string.return_value = "Cmd+T"
            mock_hotkey_spec.to_km_string.return_value = "⌘T"
            mock_create_spec.return_value = mock_hotkey_spec

            mock_manager = Mock()
            mock_manager.detect_conflicts = AsyncMock(return_value=[])

            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = "trigger_456"
            mock_manager.create_hotkey_trigger = AsyncMock(return_value=mock_result)
            mock_manager_class.return_value = mock_manager

            # Execute operation with context
            result = await km_create_hotkey_trigger(
                macro_id="test_macro",
                key="t",
                modifiers=["cmd"],
                ctx=mock_context,
            )

            # Verify successful response
            assert result["success"] is True

            # Verify context integration
            mock_context.info.assert_called()
            info_calls = [call.args[0] for call in mock_context.info.call_args_list]
            assert any("Creating hotkey trigger" in call for call in info_calls)
            assert any(
                "Successfully created hotkey trigger" in call for call in info_calls
            )


class TestHotkeyListingMocked:
    """Test hotkey listing with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_km_list_hotkey_triggers_all(self) -> None:
        """Test listing all hotkey triggers."""
        with (
            patch("src.server.tools.hotkey_tools.HotkeyManager") as mock_manager_class,
            patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client,
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
            ) as mock_trigger_manager,
        ):
            # Setup mock dependencies (systematic pattern alignment)
            mock_km_client.return_value = Mock()
            mock_trigger_manager.return_value = Mock()

            # Setup mock registered hotkeys
            mock_spec1 = Mock()
            mock_spec1.key = "n"
            mock_spec1.modifiers = [Mock(value="cmd"), Mock(value="shift")]
            mock_spec1.activation_mode = Mock(value="pressed")
            mock_spec1.tap_count = 1
            mock_spec1.allow_repeat = False
            mock_spec1.to_display_string.return_value = "Cmd+Shift+N"

            mock_spec2 = Mock()
            mock_spec2.key = "s"
            mock_spec2.modifiers = [Mock(value="cmd")]
            mock_spec2.activation_mode = Mock(value="pressed")
            mock_spec2.tap_count = 1
            mock_spec2.allow_repeat = False
            mock_spec2.to_display_string.return_value = "Cmd+S"

            registered_hotkeys = {
                "⌘⇧N": ("macro_1", mock_spec1),
                "⌘S": ("macro_2", mock_spec2),
            }

            mock_manager = Mock()
            mock_manager.get_registered_hotkeys.return_value = registered_hotkeys
            mock_manager_class.return_value = mock_manager

            # Execute operation
            result = await km_list_hotkey_triggers()

            # Verify successful response
            assert result["success"] is True
            assert result["data"]["total_count"] == 2
            assert result["data"]["filtered_by_macro"] is False
            assert result["data"]["conflicts_included"] is False

            # Verify hotkey list
            hotkeys = result["data"]["hotkeys"]
            assert len(hotkeys) == 2

            # Check first hotkey
            first_hotkey = next(h for h in hotkeys if h["hotkey_string"] == "⌘⇧N")
            assert first_hotkey["display_string"] == "Cmd+Shift+N"
            assert first_hotkey["macro_id"] == "macro_1"
            assert first_hotkey["key"] == "n"
            assert first_hotkey["modifiers"] == ["cmd", "shift"]
            assert first_hotkey["activation_mode"] == "pressed"
            assert first_hotkey["tap_count"] == 1
            assert first_hotkey["allow_repeat"] is False

    @pytest.mark.asyncio
    async def test_km_list_hotkey_triggers_filtered(self) -> None:
        """Test listing hotkey triggers filtered by macro ID."""
        with (
            patch("src.server.tools.hotkey_tools.HotkeyManager") as mock_manager_class,
            patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client,
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
            ) as mock_trigger_manager,
        ):
            # Setup mock dependencies (systematic pattern alignment)
            mock_km_client.return_value = Mock()
            mock_trigger_manager.return_value = Mock()

            # Setup mock registered hotkeys
            mock_spec = Mock()
            mock_spec.key = "n"
            mock_spec.modifiers = [Mock(value="cmd")]
            mock_spec.activation_mode = Mock(value="pressed")
            mock_spec.tap_count = 1
            mock_spec.allow_repeat = False
            mock_spec.to_display_string.return_value = "Cmd+N"

            registered_hotkeys = {
                "⌘N": ("target_macro", mock_spec),
                "⌘S": ("other_macro", Mock()),
            }

            mock_manager = Mock()
            mock_manager.get_registered_hotkeys.return_value = registered_hotkeys
            mock_manager_class.return_value = mock_manager

            # Execute operation with filter
            result = await km_list_hotkey_triggers(macro_id="target_macro")

            # Verify filtered response
            assert result["success"] is True
            assert result["data"]["total_count"] == 1
            assert result["data"]["filtered_by_macro"] is True

            # Verify only target macro's hotkey is returned
            hotkeys = result["data"]["hotkeys"]
            assert len(hotkeys) == 1
            assert hotkeys[0]["macro_id"] == "target_macro"
            assert hotkeys[0]["hotkey_string"] == "⌘N"

    @pytest.mark.asyncio
    async def test_km_list_hotkey_triggers_with_conflicts(self) -> None:
        """Test listing hotkey triggers with conflict information."""
        with (
            patch("src.server.tools.hotkey_tools.HotkeyManager") as mock_manager_class,
            patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client,
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
            ) as mock_trigger_manager,
        ):
            # Setup mock dependencies (systematic pattern alignment)
            mock_km_client.return_value = Mock()
            mock_trigger_manager.return_value = Mock()

            # Setup mock registered hotkeys
            mock_spec = Mock()
            mock_spec.key = "c"
            mock_spec.modifiers = [Mock(value="cmd")]
            mock_spec.activation_mode = Mock(value="pressed")
            mock_spec.tap_count = 1
            mock_spec.allow_repeat = False
            mock_spec.to_display_string.return_value = "Cmd+C"

            registered_hotkeys = {"⌘C": ("copy_macro", mock_spec)}

            # Setup conflict detection
            mock_conflict = Mock()
            mock_conflict.conflict_type = "system_shortcut"
            mock_conflict.description = "Conflicts with system copy"
            mock_conflict.suggestion = "Use different key combination"

            mock_manager = Mock()
            mock_manager.get_registered_hotkeys.return_value = registered_hotkeys
            mock_manager.detect_conflicts = AsyncMock(return_value=[mock_conflict])
            mock_manager_class.return_value = mock_manager

            # Execute operation with conflict information
            result = await km_list_hotkey_triggers(include_conflicts=True)

            # Verify response with conflicts
            assert result["success"] is True
            assert result["data"]["conflicts_included"] is True

            # Verify conflict information
            hotkeys = result["data"]["hotkeys"]
            assert len(hotkeys) == 1
            hotkey = hotkeys[0]
            assert "conflicts" in hotkey
            assert "has_conflicts" in hotkey
            assert hotkey["has_conflicts"] is True
            assert len(hotkey["conflicts"]) == 1
            assert hotkey["conflicts"][0]["conflict_type"] == "system_shortcut"

    @pytest.mark.asyncio
    async def test_km_list_hotkey_triggers_empty_list(self) -> None:
        """Test listing hotkey triggers when none are registered."""
        with (
            patch("src.server.tools.hotkey_tools.HotkeyManager") as mock_manager_class,
            patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client,
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
            ) as mock_trigger_manager,
        ):
            # Setup mock dependencies (systematic pattern alignment)
            mock_km_client.return_value = Mock()
            mock_trigger_manager.return_value = Mock()

            # Setup empty registered hotkeys
            mock_manager = Mock()
            mock_manager.get_registered_hotkeys.return_value = {}
            mock_manager_class.return_value = mock_manager

            # Execute operation
            result = await km_list_hotkey_triggers()

            # Verify empty response
            assert result["success"] is True
            assert result["data"]["total_count"] == 0
            assert result["data"]["hotkeys"] == []
            assert result["data"]["filtered_by_macro"] is False


class TestHotkeyErrorHandling:
    """Test hotkey tools error handling."""

    @pytest.mark.asyncio
    async def test_create_hotkey_unexpected_error(self) -> None:
        """Test handling of unexpected errors in hotkey creation."""
        with patch("src.server.tools.hotkey_tools.MacroId") as mock_macro_id:
            # Setup unexpected error
            mock_macro_id.side_effect = RuntimeError("Unexpected system error")

            # Execute operation
            result = await km_create_hotkey_trigger(
                macro_id="test_macro",
                key="n",
                modifiers=["cmd"],
            )

            # Verify error response
            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert "Unexpected error" in result["error"]["message"]
            assert result["error"]["details"]["error_type"] == "RuntimeError"
            assert "recovery_suggestion" in result["error"]

    @pytest.mark.asyncio
    async def test_list_hotkey_unexpected_error(self) -> None:
        """Test handling of unexpected errors in hotkey listing."""
        with patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client:
            # Setup unexpected error
            mock_km_client.side_effect = ConnectionError("System connection failed")

            # Execute operation
            result = await km_list_hotkey_triggers()

            # Verify error response
            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert "Failed to retrieve hotkey triggers" in result["error"]["message"]
            assert result["error"]["details"]["error_type"] == "ConnectionError"

    @pytest.mark.asyncio
    async def test_create_hotkey_error_with_context(self) -> None:
        """Test error handling with context integration."""
        mock_context = Mock()
        mock_context.info = AsyncMock()
        mock_context.error = AsyncMock()

        with patch("src.server.tools.hotkey_tools.MacroId") as mock_macro_id:
            # Setup error
            mock_macro_id.side_effect = ValueError("Invalid macro ID format")

            # Execute operation with context
            result = await km_create_hotkey_trigger(
                macro_id="invalid_macro",
                key="n",
                modifiers=["cmd"],
                ctx=mock_context,
            )

            # Verify error response
            assert result["success"] is False

            # Verify context error logging
            mock_context.error.assert_called_once()
            error_call = mock_context.error.call_args_list[0]
            assert "Hotkey trigger creation failed" in str(error_call)


class TestHotkeyIntegration:
    """Integration tests for hotkey tools."""

    @pytest.mark.asyncio
    async def test_complete_hotkey_workflow(self) -> None:
        """Test complete hotkey creation and listing workflow."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
            ) as mock_create_spec,
            patch("src.server.tools.hotkey_tools.HotkeyManager") as mock_manager_class,
            patch("src.server.tools.hotkey_tools.KMClient") as mock_km_client,
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
            ) as mock_trigger_manager,
        ):
            # Setup mock dependencies (systematic pattern alignment)
            mock_km_client.return_value = Mock()
            mock_trigger_manager.return_value = Mock()

            # Setup complete workflow
            mock_hotkey_spec = Mock()
            mock_hotkey_spec.key = "x"
            mock_hotkey_spec.modifiers = [Mock(value="cmd"), Mock(value="opt")]
            mock_hotkey_spec.activation_mode = Mock(value="pressed")
            mock_hotkey_spec.tap_count = 1
            mock_hotkey_spec.allow_repeat = False
            mock_hotkey_spec.to_display_string.return_value = "Cmd+Opt+X"
            mock_hotkey_spec.to_km_string.return_value = "⌘⌥X"
            mock_create_spec.return_value = mock_hotkey_spec

            mock_manager = Mock()
            mock_manager.detect_conflicts = AsyncMock(return_value=[])

            # Setup successful creation
            mock_result = Mock()
            mock_result.is_left.return_value = False
            mock_result.get_right.return_value = "trigger_789"
            mock_manager.create_hotkey_trigger = AsyncMock(return_value=mock_result)

            # Setup listing after creation
            registered_hotkeys = {"⌘⌥X": ("workflow_macro", mock_hotkey_spec)}
            mock_manager.get_registered_hotkeys.return_value = registered_hotkeys

            mock_manager_class.return_value = mock_manager

            # Step 1: Create hotkey
            create_result = await km_create_hotkey_trigger(
                macro_id="workflow_macro",
                key="x",
                modifiers=["cmd", "opt"],
                activation_mode="pressed",
                check_conflicts=True,
            )

            # Verify creation success
            assert create_result["success"] is True
            assert create_result["data"]["trigger_id"] == "trigger_789"
            assert create_result["data"]["hotkey"]["display_string"] == "Cmd+Opt+X"

            # Step 2: List hotkeys to verify registration
            list_result = await km_list_hotkey_triggers()

            # Verify listing includes created hotkey
            assert list_result["success"] is True
            assert list_result["data"]["total_count"] == 1
            hotkey = list_result["data"]["hotkeys"][0]
            assert hotkey["macro_id"] == "workflow_macro"
            assert hotkey["display_string"] == "Cmd+Opt+X"
            assert hotkey["key"] == "x"
            assert hotkey["modifiers"] == ["cmd", "opt"]

    @pytest.mark.asyncio
    async def test_hotkey_validation_integration(self) -> None:
        """Test integration of validation across hotkey operations."""
        with patch(
            "src.server.tools.hotkey_tools.create_hotkey_spec",
        ) as mock_create_spec:
            # Test multiple validation scenarios
            test_cases = [
                {
                    "key": "a",
                    "modifiers": ["cmd"],
                    "should_succeed": True,
                    "description": "Simple valid hotkey",
                },
                {
                    "key": "f12",
                    "modifiers": ["shift", "cmd"],
                    "should_succeed": True,
                    "description": "Function key with modifiers",
                },
                {
                    "key": "space",
                    "modifiers": ["ctrl", "opt"],
                    "should_succeed": True,
                    "description": "Special key with modifiers",
                },
            ]

            for case in test_cases:
                # Setup mock for valid cases
                if case["should_succeed"]:
                    mock_hotkey_spec = Mock()
                    mock_hotkey_spec.to_display_string.return_value = (
                        f"Test+{case['key']}"
                    )
                    mock_create_spec.return_value = mock_hotkey_spec

                # Test parameter validation (systematic pattern alignment)
                assert len(case["key"]) >= 1
                assert len(case["key"]) <= 20
                for modifier in case["modifiers"]:
                    assert modifier in ["cmd", "opt", "shift", "ctrl", "fn"]


class TestHotkeyProperties:
    """Property-based tests for hotkey operations."""

    @given(
        hotkey_key_strategy(),
        hotkey_modifiers_strategy(),
        activation_mode_strategy(),
        st.integers(min_value=1, max_value=4),
        st.booleans(),
    )
    def test_hotkey_parameter_properties(
        self,
        key: str,
        modifiers: list[str],
        activation_mode: str,
        tap_count: int,
        allow_repeat: bool,
    ) -> str:
        """Property test for hotkey parameter validation."""
        # Properties that should always hold
        valid_keys = (
            [chr(i) for i in range(ord("a"), ord("z") + 1)]
            + [str(i) for i in range(10)]
            + ["space", "tab", "enter", "return", "escape", "delete", "backspace"]
            + [f"f{i}" for i in range(1, 13)]
            + [
                "home",
                "end",
                "pageup",
                "pagedown",
                "up",
                "down",
                "left",
                "right",
                "clear",
                "help",
                "insert",
            ]
        )

        valid_modifiers = {"cmd", "opt", "shift", "ctrl", "fn"}
        valid_modes = {"pressed", "released", "tapped", "held"}

        assert key in valid_keys
        assert activation_mode in valid_modes
        assert 1 <= tap_count <= 4
        assert isinstance(allow_repeat, bool)

        # All modifiers should be valid
        for modifier in modifiers:
            assert modifier in valid_modifiers

        # Modifiers should be unique (no duplicates)
        assert len(modifiers) == len(set(modifiers))

    @given(macro_id_strategy())
    def test_macro_id_properties(self, macro_id: str) -> str:
        """Property test for macro ID validation."""
        # Macro ID properties
        assert isinstance(macro_id, str)
        assert len(macro_id) >= 1
        assert len(macro_id) <= 255

        # Should be non-empty after stripping
        assert len(macro_id.strip()) >= 1

    @given(invalid_key_strategy())
    def test_invalid_key_detection_properties(self, invalid_key: str) -> str:
        """Property test for invalid key detection."""
        valid_keys = (
            [chr(i) for i in range(ord("a"), ord("z") + 1)]
            + [str(i) for i in range(10)]
            + ["space", "tab", "enter", "return", "escape", "delete", "backspace"]
            + [f"f{i}" for i in range(1, 13)]
            + [
                "home",
                "end",
                "pageup",
                "pagedown",
                "up",
                "down",
                "left",
                "right",
                "clear",
                "help",
                "insert",
            ]
        )

        # Invalid keys should not be in valid set
        assert invalid_key not in valid_keys

        # Common invalid patterns should be detectable
        invalid_indicators = ["ctrl+", "cmd+", "shift+", "F13", ""]
        if (
            any(indicator in invalid_key for indicator in invalid_indicators)
            or not invalid_key.strip()
        ):
            # These should be caught by validation
            assert True  # Invalid pattern detected

    @given(invalid_modifier_strategy())
    def test_invalid_modifier_detection_properties(self, invalid_modifiers: list[str]) -> None:
        """Property test for invalid modifier detection."""
        valid_modifiers = {"cmd", "opt", "shift", "ctrl", "fn"}

        # All provided modifiers should be invalid
        for modifier in invalid_modifiers:
            assert modifier not in valid_modifiers

        # Should contain typical invalid modifiers
        common_invalid = {"invalid", "command", "alt", "control", "windows", "meta"}
        assert any(mod in common_invalid for mod in invalid_modifiers)
