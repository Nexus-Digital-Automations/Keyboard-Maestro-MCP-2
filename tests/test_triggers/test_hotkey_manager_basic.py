"""
Basic tests for Hotkey Manager module focusing on existing functionality.

Tests cover ModifierKey, ActivationMode, and basic validation patterns.
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from src.core.errors import ValidationError
from src.triggers.hotkey_manager import ActivationMode, ModifierKey


class TestModifierKey:
    """Test ModifierKey enum and validation."""

    def test_modifier_key_enum_values(self):
        """Test ModifierKey enum has expected values."""
        assert ModifierKey.COMMAND.value == "cmd"
        assert ModifierKey.OPTION.value == "opt"
        assert ModifierKey.SHIFT.value == "shift"
        assert ModifierKey.CONTROL.value == "ctrl"
        assert ModifierKey.FUNCTION.value == "fn"

    def test_modifier_key_from_string_valid(self):
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

    def test_modifier_key_from_string_case_insensitive(self):
        """Test ModifierKey creation is case insensitive."""
        assert ModifierKey.from_string("CMD") == ModifierKey.COMMAND
        assert ModifierKey.from_string("Cmd") == ModifierKey.COMMAND
        assert ModifierKey.from_string("SHIFT") == ModifierKey.SHIFT
        assert ModifierKey.from_string("Shift") == ModifierKey.SHIFT

    def test_modifier_key_from_string_with_whitespace(self):
        """Test ModifierKey creation handles whitespace."""
        assert ModifierKey.from_string("  cmd  ") == ModifierKey.COMMAND
        assert ModifierKey.from_string("\tshift\n") == ModifierKey.SHIFT

    def test_modifier_key_from_string_invalid(self):
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
            ]
        )
    )
    def test_modifier_key_from_string_property_based_invalid(self, invalid_modifier):
        """Property-based test for invalid modifier strings."""
        assume(len(invalid_modifier.strip()) > 0)  # Don't test empty strings

        with pytest.raises(ValidationError):
            ModifierKey.from_string(invalid_modifier)


class TestActivationMode:
    """Test ActivationMode enum and validation."""

    def test_activation_mode_enum_values(self):
        """Test ActivationMode enum has expected values."""
        assert ActivationMode.PRESSED.value == "pressed"
        assert ActivationMode.RELEASED.value == "released"
        assert ActivationMode.TAPPED.value == "tapped"
        assert ActivationMode.HELD.value == "held"

    def test_activation_mode_from_string_valid(self):
        """Test creating ActivationMode from valid strings."""
        assert ActivationMode.from_string("pressed") == ActivationMode.PRESSED
        assert ActivationMode.from_string("released") == ActivationMode.RELEASED
        assert ActivationMode.from_string("tapped") == ActivationMode.TAPPED
        assert ActivationMode.from_string("held") == ActivationMode.HELD

    def test_activation_mode_from_string_case_insensitive(self):
        """Test ActivationMode creation is case insensitive."""
        assert ActivationMode.from_string("PRESSED") == ActivationMode.PRESSED
        assert ActivationMode.from_string("Pressed") == ActivationMode.PRESSED
        assert ActivationMode.from_string("TAPPED") == ActivationMode.TAPPED
        assert ActivationMode.from_string("Tapped") == ActivationMode.TAPPED

    def test_activation_mode_from_string_with_whitespace(self):
        """Test ActivationMode creation handles whitespace."""
        assert ActivationMode.from_string("  pressed  ") == ActivationMode.PRESSED
        assert ActivationMode.from_string("\ttapped\n") == ActivationMode.TAPPED

    def test_activation_mode_from_string_invalid(self):
        """Test ActivationMode creation with invalid strings raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ActivationMode.from_string("invalid_mode")

        error = exc_info.value
        assert error.field_name == "activation_mode"
        assert error.value == "invalid_mode"
        assert "Invalid activation mode" in str(error)

    @given(
        st.text().filter(
            lambda x: x.lower().strip() not in ["pressed", "released", "tapped", "held"]
        )
    )
    def test_activation_mode_from_string_property_based_invalid(self, invalid_mode):
        """Property-based test for invalid activation mode strings."""
        assume(len(invalid_mode.strip()) > 0)  # Don't test empty strings

        with pytest.raises(ValidationError):
            ActivationMode.from_string(invalid_mode)


class TestHotkeyValidation:
    """Test hotkey validation patterns."""

    @given(
        st.sampled_from(
            [
                ModifierKey.COMMAND,
                ModifierKey.OPTION,
                ModifierKey.SHIFT,
                ModifierKey.CONTROL,
            ]
        )
    )
    def test_modifier_key_property_based_valid(self, modifier_key):
        """Property-based test with valid modifier keys."""
        # Test that all enum values are valid
        assert isinstance(modifier_key, ModifierKey)
        assert modifier_key.value in ["cmd", "opt", "shift", "ctrl", "fn"]

    @given(
        st.sampled_from(
            [
                ActivationMode.PRESSED,
                ActivationMode.RELEASED,
                ActivationMode.TAPPED,
                ActivationMode.HELD,
            ]
        )
    )
    def test_activation_mode_property_based_valid(self, activation_mode):
        """Property-based test with valid activation modes."""
        # Test that all enum values are valid
        assert isinstance(activation_mode, ActivationMode)
        assert activation_mode.value in ["pressed", "released", "tapped", "held"]

    def test_modifier_key_combinations(self):
        """Test combining multiple modifier keys."""
        modifiers = {ModifierKey.COMMAND, ModifierKey.SHIFT, ModifierKey.OPTION}

        assert len(modifiers) == 3
        assert ModifierKey.COMMAND in modifiers
        assert ModifierKey.SHIFT in modifiers
        assert ModifierKey.OPTION in modifiers
        assert ModifierKey.CONTROL not in modifiers

    def test_modifier_key_string_representations(self):
        """Test string representations of modifier keys."""
        assert str(ModifierKey.COMMAND.value) == "cmd"
        assert str(ModifierKey.OPTION.value) == "opt"
        assert str(ModifierKey.SHIFT.value) == "shift"
        assert str(ModifierKey.CONTROL.value) == "ctrl"
        assert str(ModifierKey.FUNCTION.value) == "fn"

    def test_activation_mode_string_representations(self):
        """Test string representations of activation modes."""
        assert str(ActivationMode.PRESSED.value) == "pressed"
        assert str(ActivationMode.RELEASED.value) == "released"
        assert str(ActivationMode.TAPPED.value) == "tapped"
        assert str(ActivationMode.HELD.value) == "held"


class TestHotkeyImports:
    """Test that hotkey module imports work correctly."""

    def test_imports_available(self):
        """Test that expected classes can be imported."""
        from src.triggers.hotkey_manager import ActivationMode, ModifierKey

        # Verify classes are available
        assert ModifierKey is not None
        assert ActivationMode is not None

        # Verify enums work
        assert hasattr(ModifierKey, "COMMAND")
        assert hasattr(ActivationMode, "PRESSED")

    def test_enum_iteration(self):
        """Test that enums can be iterated."""
        modifier_keys = list(ModifierKey)
        activation_modes = list(ActivationMode)

        assert len(modifier_keys) == 5  # cmd, opt, shift, ctrl, fn
        assert len(activation_modes) == 4  # pressed, released, tapped, held

        # Check all expected values are present
        modifier_values = [mk.value for mk in modifier_keys]
        assert "cmd" in modifier_values
        assert "opt" in modifier_values
        assert "shift" in modifier_values
        assert "ctrl" in modifier_values
        assert "fn" in modifier_values

        activation_values = [am.value for am in activation_modes]
        assert "pressed" in activation_values
        assert "released" in activation_values
        assert "tapped" in activation_values
        assert "held" in activation_values
