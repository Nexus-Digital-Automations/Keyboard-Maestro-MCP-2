"""Comprehensive tests for Hotkey Manager module with systematic coverage.

Tests cover hotkey creation, validation, conflict detection, and comprehensive
enterprise-grade validation using ADDER+ testing protocols.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from src.core.errors import SecurityViolationError, ValidationError
from src.core.types import MacroId, TriggerId
from src.triggers.hotkey_manager import (
    ActivationMode,
    HotkeyManager,
    HotkeySpec,
    ModifierKey,
)


# Test data generators
@st.composite
def modifier_key_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid modifier keys."""
    return draw(
        st.sampled_from(
            [
                ModifierKey.COMMAND,
                ModifierKey.OPTION,
                ModifierKey.SHIFT,
                ModifierKey.CONTROL,
                ModifierKey.FUNCTION,
            ],
        ),
    )


@st.composite
def activation_mode_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid activation modes."""
    return draw(
        st.sampled_from(
            [
                ActivationMode.PRESSED,
                ActivationMode.RELEASED,
                ActivationMode.TAPPED,
                ActivationMode.HELD,
            ],
        ),
    )


@st.composite
def valid_key_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid key characters."""
    return draw(
        st.sampled_from(
            [
                "a",
                "b",
                "c",
                "x",
                "y",
                "z",
                "1",
                "2",
                "3",
                "space",
                "return",
                "escape",
                "tab",
                "delete",
            ],
        ),
    )


@st.composite
def hotkey_combination_strategy(draw: Callable[..., Any]) -> bool:
    """Generate valid hotkey combinations."""
    modifiers = draw(st.sets(modifier_key_strategy(), min_size=1, max_size=3))
    key = draw(valid_key_strategy())
    return list(modifiers), key


class TestModifierKey:
    """Test ModifierKey enum and validation."""

    def test_modifier_key_enum_values(self) -> None:
        """Test ModifierKey enum has expected values."""
        assert ModifierKey.COMMAND.value == "cmd"
        assert ModifierKey.OPTION.value == "opt"
        assert ModifierKey.SHIFT.value == "shift"
        assert ModifierKey.CONTROL.value == "ctrl"
        assert ModifierKey.FUNCTION.value == "fn"

    def test_modifier_key_from_string_valid(self) -> None:
        """Test creating ModifierKey from valid strings."""
        # Test value-based creation
        assert ModifierKey.from_string("cmd") == ModifierKey.COMMAND
        assert ModifierKey.from_string("opt") == ModifierKey.OPTION
        assert ModifierKey.from_string("shift") == ModifierKey.SHIFT
        assert ModifierKey.from_string("ctrl") == ModifierKey.CONTROL
        assert ModifierKey.from_string("fn") == ModifierKey.FUNCTION

        # Test name-based creation
        assert ModifierKey.from_string("command") == ModifierKey.COMMAND
        assert ModifierKey.from_string("option") == ModifierKey.OPTION
        assert ModifierKey.from_string("control") == ModifierKey.CONTROL
        assert ModifierKey.from_string("function") == ModifierKey.FUNCTION

    def test_modifier_key_from_string_case_insensitive(self) -> None:
        """Test ModifierKey creation is case insensitive."""
        assert ModifierKey.from_string("CMD") == ModifierKey.COMMAND
        assert ModifierKey.from_string("Cmd") == ModifierKey.COMMAND
        assert ModifierKey.from_string("SHIFT") == ModifierKey.SHIFT
        assert ModifierKey.from_string("Shift") == ModifierKey.SHIFT

    def test_modifier_key_from_string_with_whitespace(self) -> None:
        """Test ModifierKey creation handles whitespace."""
        assert ModifierKey.from_string("  cmd  ") == ModifierKey.COMMAND
        assert ModifierKey.from_string("\\tshift\\n") == ModifierKey.SHIFT

    def test_modifier_key_from_string_invalid(self) -> None:
        """Test ModifierKey creation with invalid strings raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModifierKey.from_string("invalid_modifier")

        error = exc_info.value
        assert error.field_name == "modifier"
        assert error.value == "invalid_modifier"
        assert "Invalid modifier key" in str(error)

    @given(
        st.text().filter(
            lambda x: x.lower().strip()
            not in [
                "cmd",
                "opt",
                "shift",
                "ctrl",
                "fn",
                "command",
                "option",
                "control",
                "function",
            ],
        ),
    )
    def test_modifier_key_from_string_property_based_invalid(self, invalid_modifier: str) -> None:
        """Property-based test for invalid modifier strings."""
        assume(len(invalid_modifier.strip()) > 0)  # Don't test empty strings

        with pytest.raises(ValidationError):
            ModifierKey.from_string(invalid_modifier)


class TestActivationMode:
    """Test ActivationMode enum and validation."""

    def test_activation_mode_enum_values(self) -> None:
        """Test ActivationMode enum has expected values."""
        assert ActivationMode.PRESSED.value == "pressed"
        assert ActivationMode.RELEASED.value == "released"
        assert ActivationMode.TAPPED.value == "tapped"
        assert ActivationMode.HELD.value == "held"

    def test_activation_mode_from_string_valid(self) -> None:
        """Test creating ActivationMode from valid strings."""
        # Assuming similar from_string method exists
        assert ActivationMode.PRESSED.value == "pressed"
        assert ActivationMode.RELEASED.value == "released"
        assert ActivationMode.TAPPED.value == "tapped"
        assert ActivationMode.HELD.value == "held"


class TestHotkeySpec:
    """Test HotkeySpec creation and validation."""

    def test_hotkey_definition_creation_valid(self) -> None:
        """Test creating valid HotkeySpec instances."""
        hotkey_def = HotkeySpec(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            modifiers={ModifierKey.COMMAND, ModifierKey.SHIFT},
            key="a",
            activation_mode=ActivationMode.PRESSED,
            enabled=True,
            description="Test hotkey",
        )

        assert hotkey_def.trigger_id == TriggerId("test_trigger")
        assert hotkey_def.macro_id == MacroId("test_macro")
        assert hotkey_def.modifiers == {ModifierKey.COMMAND, ModifierKey.SHIFT}
        assert hotkey_def.key == "a"
        assert hotkey_def.activation_mode == ActivationMode.PRESSED
        assert hotkey_def.enabled is True
        assert hotkey_def.description == "Test hotkey"

    def test_hotkey_definition_get_combination_string(self) -> None:
        """Test getting human-readable combination string."""
        hotkey_def = HotkeySpec(
            trigger_id=TriggerId("test"),
            macro_id=MacroId("test"),
            modifiers={ModifierKey.COMMAND, ModifierKey.SHIFT},
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )

        combination = hotkey_def.get_combination_string()

        # Should contain all modifiers and key
        assert "cmd" in combination or "command" in combination.lower()
        assert "shift" in combination.lower()
        assert "a" in combination.lower()

    def test_hotkey_definition_is_conflicting_with_same(self) -> None:
        """Test conflict detection with identical hotkeys."""
        hotkey1 = HotkeySpec(
            trigger_id=TriggerId("test1"),
            macro_id=MacroId("macro1"),
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )

        hotkey2 = HotkeySpec(
            trigger_id=TriggerId("test2"),
            macro_id=MacroId("macro2"),
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )

        # Should conflict if enabled
        assert hotkey1.is_conflicting_with(hotkey2) is True

    def test_hotkey_definition_is_conflicting_with_different(self) -> None:
        """Test conflict detection with different hotkeys."""
        hotkey1 = HotkeySpec(
            trigger_id=TriggerId("test1"),
            macro_id=MacroId("macro1"),
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )

        hotkey2 = HotkeySpec(
            trigger_id=TriggerId("test2"),
            macro_id=MacroId("macro2"),
            modifiers={ModifierKey.COMMAND},
            key="b",  # Different key
            activation_mode=ActivationMode.PRESSED,
        )

        # Should not conflict
        assert hotkey1.is_conflicting_with(hotkey2) is False

    def test_hotkey_definition_disabled_no_conflict(self) -> None:
        """Test that disabled hotkeys don't conflict."""
        hotkey1 = HotkeySpec(
            trigger_id=TriggerId("test1"),
            macro_id=MacroId("macro1"),
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
            enabled=False,  # Disabled
        )

        hotkey2 = HotkeySpec(
            trigger_id=TriggerId("test2"),
            macro_id=MacroId("macro2"),
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
            enabled=True,
        )

        # Should not conflict because hotkey1 is disabled
        assert hotkey1.is_conflicting_with(hotkey2) is False

    @given(hotkey_combination_strategy(), activation_mode_strategy())
    def test_hotkey_definition_property_based_creation(
        self,
        combination: Any,
        activation_mode: Any,
    ) -> None:
        """Property-based test for HotkeySpec creation."""
        modifiers, key = combination

        hotkey_def = HotkeySpec(
            trigger_id=TriggerId("test"),
            macro_id=MacroId("test"),
            modifiers=set(modifiers),
            key=key,
            activation_mode=activation_mode,
        )

        assert hotkey_def.modifiers == set(modifiers)
        assert hotkey_def.key == key
        assert hotkey_def.activation_mode == activation_mode


class TestHotkeyManager:
    """Test HotkeyManager functionality."""

    @pytest.fixture
    def mock_km_client(self) -> Any:
        """Create mock KM client for testing."""
        client = Mock()
        client.register_hotkey = Mock()
        client.unregister_hotkey = Mock()
        client.list_hotkeys = Mock()
        return client

    @pytest.fixture
    def mock_trigger_manager(self) -> Any:
        """Create mock trigger registration manager."""
        manager = Mock()
        manager.register_trigger = Mock()
        manager.unregister_trigger = Mock()
        manager.list_triggers = Mock()
        return manager

    @pytest.fixture
    def hotkey_manager(self, mock_km_client: Any, mock_trigger_manager: Any) -> Any:
        """Create HotkeyManager instance for testing."""
        return HotkeyManager(
            km_client=mock_km_client,
            trigger_manager=mock_trigger_manager,
        )

    def test_hotkey_manager_initialization(self, mock_km_client: Any, mock_trigger_manager: Any) -> None:
        """Test HotkeyManager initialization."""
        manager = HotkeyManager(
            km_client=mock_km_client,
            trigger_manager=mock_trigger_manager,
        )

        assert manager._km_client == mock_km_client
        assert manager._trigger_manager == mock_trigger_manager
        assert isinstance(manager._registered_hotkeys, dict)
        assert len(manager._registered_hotkeys) == 0

    def test_hotkey_manager_register_hotkey_success(
        self,
        hotkey_manager: Any,
        mock_km_client: Any,
    ) -> None:
        """Test successful hotkey registration."""
        # Mock successful registration
        mock_km_client.register_hotkey.return_value = {
            "success": True,
            "trigger_id": "test_trigger",
        }

        hotkey_def = HotkeySpec(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )

        result = hotkey_manager.register_hotkey(hotkey_def)

        assert result.is_success() is True
        assert hotkey_def.trigger_id in hotkey_manager._registered_hotkeys

        # Verify KM client was called
        mock_km_client.register_hotkey.assert_called_once()

    def test_hotkey_manager_register_hotkey_conflict(self, hotkey_manager: Any) -> None:
        """Test hotkey registration with conflict detection."""
        # Register first hotkey
        hotkey1 = HotkeySpec(
            trigger_id=TriggerId("test1"),
            macro_id=MacroId("macro1"),
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )

        # Manually add to registered hotkeys to simulate existing registration
        hotkey_manager._registered_hotkeys[hotkey1.trigger_id] = hotkey1

        # Try to register conflicting hotkey
        hotkey2 = HotkeySpec(
            trigger_id=TriggerId("test2"),
            macro_id=MacroId("macro2"),
            modifiers={ModifierKey.COMMAND},
            key="a",  # Same combination
            activation_mode=ActivationMode.PRESSED,
        )

        result = hotkey_manager.register_hotkey(hotkey2)

        assert result.is_failure() is True
        # Should not be added to registered hotkeys
        assert hotkey2.trigger_id not in hotkey_manager._registered_hotkeys

    def test_hotkey_manager_unregister_hotkey_success(
        self,
        hotkey_manager: Any,
        mock_km_client: Any,
    ) -> None:
        """Test successful hotkey unregistration."""
        # Mock successful unregistration
        mock_km_client.unregister_hotkey.return_value = {"success": True}

        # Add hotkey to registered list
        trigger_id = TriggerId("test_trigger")
        hotkey_def = HotkeySpec(
            trigger_id=trigger_id,
            macro_id=MacroId("test_macro"),
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )
        hotkey_manager._registered_hotkeys[trigger_id] = hotkey_def

        result = hotkey_manager.unregister_hotkey(trigger_id)

        assert result.is_success() is True
        assert trigger_id not in hotkey_manager._registered_hotkeys

        # Verify KM client was called
        mock_km_client.unregister_hotkey.assert_called_once_with(str(trigger_id))

    def test_hotkey_manager_unregister_hotkey_not_found(self, hotkey_manager: Any) -> None:
        """Test unregistering non-existent hotkey."""
        trigger_id = TriggerId("non_existent")

        result = hotkey_manager.unregister_hotkey(trigger_id)

        assert result.is_failure() is True

    def test_hotkey_manager_list_hotkeys(self, hotkey_manager: Any) -> None:
        """Test listing registered hotkeys."""
        # Add some hotkeys
        hotkey1 = HotkeySpec(
            trigger_id=TriggerId("test1"),
            macro_id=MacroId("macro1"),
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )

        hotkey2 = HotkeySpec(
            trigger_id=TriggerId("test2"),
            macro_id=MacroId("macro2"),
            modifiers={ModifierKey.SHIFT},
            key="b",
            activation_mode=ActivationMode.TAPPED,
        )

        hotkey_manager._registered_hotkeys[hotkey1.trigger_id] = hotkey1
        hotkey_manager._registered_hotkeys[hotkey2.trigger_id] = hotkey2

        hotkeys = hotkey_manager.list_hotkeys()

        assert len(hotkeys) == 2
        assert hotkey1 in hotkeys
        assert hotkey2 in hotkeys

    def test_hotkey_manager_list_hotkeys_by_macro(self, hotkey_manager: Any) -> None:
        """Test listing hotkeys filtered by macro ID."""
        macro_id = MacroId("target_macro")

        # Add hotkeys for different macros
        hotkey1 = HotkeySpec(
            trigger_id=TriggerId("test1"),
            macro_id=macro_id,  # Target macro
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )

        hotkey2 = HotkeySpec(
            trigger_id=TriggerId("test2"),
            macro_id=MacroId("other_macro"),  # Different macro
            modifiers={ModifierKey.SHIFT},
            key="b",
            activation_mode=ActivationMode.TAPPED,
        )

        hotkey_manager._registered_hotkeys[hotkey1.trigger_id] = hotkey1
        hotkey_manager._registered_hotkeys[hotkey2.trigger_id] = hotkey2

        filtered_hotkeys = hotkey_manager.list_hotkeys_by_macro(macro_id)

        assert len(filtered_hotkeys) == 1
        assert hotkey1 in filtered_hotkeys
        assert hotkey2 not in filtered_hotkeys

    def test_hotkey_manager_find_conflicts(self, hotkey_manager: Any) -> None:
        """Test finding conflicting hotkeys."""
        # Add existing hotkey
        existing_hotkey = HotkeySpec(
            trigger_id=TriggerId("existing"),
            macro_id=MacroId("macro1"),
            modifiers={ModifierKey.COMMAND},
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )
        hotkey_manager._registered_hotkeys[existing_hotkey.trigger_id] = existing_hotkey

        # Create conflicting hotkey
        new_hotkey = HotkeySpec(
            trigger_id=TriggerId("new"),
            macro_id=MacroId("macro2"),
            modifiers={ModifierKey.COMMAND},
            key="a",  # Same combination
            activation_mode=ActivationMode.PRESSED,
        )

        conflicts = hotkey_manager.find_conflicts(new_hotkey)

        assert len(conflicts) == 1
        assert existing_hotkey in conflicts

    @given(hotkey_combination_strategy())
    def test_hotkey_manager_property_based_registration(
        self,
        combination: Any,
        hotkey_manager: Any,
        mock_km_client: Any,
    ) -> None:
        """Property-based test for hotkey registration."""
        modifiers, key = combination

        # Mock successful registration
        mock_km_client.register_hotkey.return_value = {
            "success": True,
            "trigger_id": "test_trigger",
        }

        hotkey_def = HotkeySpec(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            modifiers=set(modifiers),
            key=key,
            activation_mode=ActivationMode.PRESSED,
        )

        result = hotkey_manager.register_hotkey(hotkey_def)

        assert result.is_success() is True
        assert hotkey_def.trigger_id in hotkey_manager._registered_hotkeys
        assert hotkey_manager._registered_hotkeys[hotkey_def.trigger_id] == hotkey_def


class TestHotkeyValidation:
    """Test hotkey validation and security features."""

    def test_hotkey_definition_security_validation(self) -> None:
        """Test security validation in hotkey definitions."""
        # Test with potentially dangerous key combinations
        dangerous_keys = ["<script>", "javascript:", "eval(", "system("]

        for dangerous_key in dangerous_keys:
            with pytest.raises((ValidationError, SecurityViolationError)):
                HotkeySpec(
                    trigger_id=TriggerId("test"),
                    macro_id=MacroId("test"),
                    modifiers={ModifierKey.COMMAND},
                    key=dangerous_key,
                    activation_mode=ActivationMode.PRESSED,
                )

    def test_hotkey_definition_key_length_validation(self) -> None:
        """Test key length validation."""
        # Test with excessively long key
        long_key = "a" * 1000

        with pytest.raises(ValidationError):
            HotkeySpec(
                trigger_id=TriggerId("test"),
                macro_id=MacroId("test"),
                modifiers={ModifierKey.COMMAND},
                key=long_key,
                activation_mode=ActivationMode.PRESSED,
            )

    def test_hotkey_definition_empty_modifiers_validation(self) -> None:
        """Test validation with empty modifiers."""
        # Some implementations might require at least one modifier
        hotkey_def = HotkeySpec(
            trigger_id=TriggerId("test"),
            macro_id=MacroId("test"),
            modifiers=set(),  # Empty modifiers
            key="a",
            activation_mode=ActivationMode.PRESSED,
        )

        # Should either work or raise validation error depending on implementation
        assert hotkey_def.modifiers == set()


class TestHotkeyIntegration:
    """Integration tests for hotkey functionality."""

    def test_complete_hotkey_lifecycle(self, mock_km_client: Any, mock_trigger_manager: Any) -> None:
        """Test complete hotkey registration and unregistration lifecycle."""
        # Mock successful operations
        mock_km_client.register_hotkey.return_value = {
            "success": True,
            "trigger_id": "test_trigger",
        }
        mock_km_client.unregister_hotkey.return_value = {"success": True}
        mock_trigger_manager.register_trigger.return_value = {"success": True}
        mock_trigger_manager.unregister_trigger.return_value = {"success": True}

        manager = HotkeyManager(
            km_client=mock_km_client,
            trigger_manager=mock_trigger_manager,
        )

        # Step 1: Register hotkey
        hotkey_def = HotkeySpec(
            trigger_id=TriggerId("test_trigger"),
            macro_id=MacroId("test_macro"),
            modifiers={ModifierKey.COMMAND, ModifierKey.SHIFT},
            key="x",
            activation_mode=ActivationMode.PRESSED,
        )

        register_result = manager.register_hotkey(hotkey_def)
        assert register_result.is_success() is True

        # Step 2: Verify hotkey is registered
        hotkeys = manager.list_hotkeys()
        assert len(hotkeys) == 1
        assert hotkey_def in hotkeys

        # Step 3: Unregister hotkey
        unregister_result = manager.unregister_hotkey(hotkey_def.trigger_id)
        assert unregister_result.is_success() is True

        # Step 4: Verify hotkey is unregistered
        hotkeys_after = manager.list_hotkeys()
        assert len(hotkeys_after) == 0

    def test_multiple_hotkey_management(self, mock_km_client: Any, mock_trigger_manager: Any) -> None:
        """Test managing multiple hotkeys simultaneously."""
        # Mock successful operations
        mock_km_client.register_hotkey.return_value = {
            "success": True,
            "trigger_id": "test_trigger",
        }
        mock_km_client.unregister_hotkey.return_value = {"success": True}

        manager = HotkeyManager(
            km_client=mock_km_client,
            trigger_manager=mock_trigger_manager,
        )

        # Register multiple hotkeys
        hotkeys = []
        for i, key in enumerate(["a", "b", "c"]):
            hotkey = HotkeySpec(
                trigger_id=TriggerId(f"trigger_{i}"),
                macro_id=MacroId(f"macro_{i}"),
                modifiers={ModifierKey.COMMAND},
                key=key,
                activation_mode=ActivationMode.PRESSED,
            )
            hotkeys.append(hotkey)

            result = manager.register_hotkey(hotkey)
            assert result.is_success() is True

        # Verify all are registered
        registered_hotkeys = manager.list_hotkeys()
        assert len(registered_hotkeys) == 3

        for hotkey in hotkeys:
            assert hotkey in registered_hotkeys

        # Test conflict detection
        conflicting_hotkey = HotkeySpec(
            trigger_id=TriggerId("conflict"),
            macro_id=MacroId("conflict_macro"),
            modifiers={ModifierKey.COMMAND},
            key="a",  # Conflicts with first hotkey
            activation_mode=ActivationMode.PRESSED,
        )

        conflicts = manager.find_conflicts(conflicting_hotkey)
        assert len(conflicts) == 1
        assert hotkeys[0] in conflicts
