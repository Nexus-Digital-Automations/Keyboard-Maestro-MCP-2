"""
Property-based tests for hotkey trigger management.

Tests hotkey validation, conflict detection, and security boundaries using
hypothesis-driven property-based testing for comprehensive coverage.
"""

import pytest
from hypothesis import given, strategies as st, assume, example
from unittest.mock import Mock, AsyncMock, patch

from src.triggers.hotkey_manager import (
    HotkeySpec, ModifierKey, ActivationMode, HotkeyManager, 
    create_hotkey_spec, VALID_SPECIAL_KEYS, SYSTEM_RESERVED_HOTKEYS
)
from src.core.types import MacroId, TriggerId
from src.core.errors import ValidationError, SecurityViolationError
from src.integration.km_client import KMClient, Either, KMError
from src.integration.triggers import TriggerRegistrationManager


class TestHotkeySpecValidation:
    """Test HotkeySpec validation with property-based testing."""
    
    @given(st.text(min_size=1, max_size=1).filter(lambda x: x.isalnum() and x.isascii()))
    def test_valid_single_character_keys(self, key):
        """Property: Valid single character keys should create valid HotkeySpec."""
        spec = HotkeySpec(
            key=key.lower(),
            modifiers=set(),
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        assert spec.key == key.lower()
        assert spec.to_km_string() == key.lower()
    
    @given(st.sampled_from(list(VALID_SPECIAL_KEYS)))
    def test_valid_special_keys(self, special_key):
        """Property: All valid special keys should create valid HotkeySpec."""
        spec = HotkeySpec(
            key=special_key,
            modifiers=set(),
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        assert spec.key == special_key
        assert spec.to_km_string() == special_key
    
    @given(st.text().filter(lambda x: len(x) != 1 and x.lower() not in VALID_SPECIAL_KEYS))
    def test_invalid_keys_raise_validation_error(self, invalid_key):
        """Property: Invalid keys should raise ValidationError."""
        assume(invalid_key)  # Ensure not empty
        
        with pytest.raises(ValidationError):
            HotkeySpec(
                key=invalid_key,
                modifiers=set(),
                activation_mode=ActivationMode.PRESSED,
                tap_count=1,
                allow_repeat=False
            )
    
    @given(st.integers().filter(lambda x: x < 1 or x > 4))
    def test_invalid_tap_count_raises_error(self, invalid_tap_count):
        """Property: Tap counts outside 1-4 range should raise ValidationError."""
        with pytest.raises(ValidationError):
            HotkeySpec(
                key="a",
                modifiers=set(),
                activation_mode=ActivationMode.PRESSED,
                tap_count=invalid_tap_count,
                allow_repeat=False
            )
    
    @given(st.sets(st.sampled_from(ModifierKey), min_size=0, max_size=5))
    def test_modifier_combinations(self, modifiers):
        """Property: Any combination of valid modifiers should be accepted."""
        spec = HotkeySpec(
            key="a",
            modifiers=modifiers,
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        assert spec.modifiers == modifiers
    
    def test_system_conflict_detection(self):
        """Test detection of system-reserved hotkey conflicts."""
        # Test known system shortcuts
        with pytest.raises(SecurityViolationError):
            HotkeySpec(
                key="space",
                modifiers={ModifierKey.COMMAND},
                activation_mode=ActivationMode.PRESSED,
                tap_count=1,
                allow_repeat=False
            )
    
    @given(st.text(min_size=1, max_size=5))
    def test_display_string_formatting(self, key):
        """Property: Display strings should be consistently formatted."""
        assume(len(key) == 1 and key.isalnum() and key.isascii())
        
        spec = HotkeySpec(
            key=key.lower(),
            modifiers={ModifierKey.COMMAND, ModifierKey.SHIFT},
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        
        display = spec.to_display_string()
        assert "⌘" in display  # Command symbol
        assert "⇧" in display  # Shift symbol
        assert key.upper() in display


class TestCreateHotkeySpec:
    """Test the create_hotkey_spec factory function."""
    
    @given(
        st.text(min_size=1, max_size=1).filter(lambda x: x.isalnum() and x.isascii()),
        st.lists(st.sampled_from(["cmd", "opt", "shift", "ctrl", "fn"]), unique=True, max_size=5),
        st.sampled_from(["pressed", "released", "tapped", "held"]),
        st.integers(min_value=1, max_value=4),
        st.booleans()
    )
    def test_factory_function_creates_valid_specs(self, key, modifiers, activation_mode, tap_count, allow_repeat):
        """Property: Factory function should create valid HotkeySpec for valid inputs."""
        spec = create_hotkey_spec(
            key=key,
            modifiers=modifiers,
            activation_mode=activation_mode,
            tap_count=tap_count,
            allow_repeat=allow_repeat
        )
        
        assert isinstance(spec, HotkeySpec)
        assert spec.key == key.lower()
        assert len(spec.modifiers) == len(set(modifiers))  # Duplicates removed
        assert spec.activation_mode.value == activation_mode
        assert spec.tap_count == tap_count
        assert spec.allow_repeat == allow_repeat
    
    def test_invalid_modifier_strings_raise_error(self):
        """Test that invalid modifier strings raise ValidationError."""
        with pytest.raises(ValidationError):
            create_hotkey_spec(
                key="a",
                modifiers=["invalid_modifier"],
                activation_mode="pressed",
                tap_count=1,
                allow_repeat=False
            )
    
    def test_invalid_activation_mode_raises_error(self):
        """Test that invalid activation modes raise ValidationError."""
        with pytest.raises(ValidationError):
            create_hotkey_spec(
                key="a",
                modifiers=["cmd"],
                activation_mode="invalid_mode",
                tap_count=1,
                allow_repeat=False
            )


class TestHotkeyManager:
    """Test HotkeyManager functionality with mocked dependencies."""
    
    @pytest.fixture
    def mock_km_client(self):
        """Mock KMClient for testing."""
        return Mock(spec=KMClient)
    
    @pytest.fixture
    def mock_trigger_manager(self, mock_km_client):
        """Mock TriggerRegistrationManager for testing."""
        manager = Mock(spec=TriggerRegistrationManager)
        manager.register_trigger = AsyncMock(return_value=Either.right(TriggerId("test-trigger-id")))
        return manager
    
    @pytest.fixture
    def hotkey_manager(self, mock_km_client, mock_trigger_manager):
        """Create HotkeyManager instance with mocked dependencies."""
        return HotkeyManager(mock_km_client, mock_trigger_manager)
    
    @pytest.mark.asyncio
    async def test_create_hotkey_trigger_success(self, hotkey_manager, mock_trigger_manager):
        """Test successful hotkey trigger creation."""
        macro_id = MacroId("test-macro")
        hotkey_spec = HotkeySpec(
            key="n",
            modifiers={ModifierKey.COMMAND},
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        
        result = await hotkey_manager.create_hotkey_trigger(
            macro_id=macro_id,
            hotkey=hotkey_spec,
            check_conflicts=False  # Skip conflict checking for this test
        )
        
        assert result.is_right()
        trigger_id = result.get_right()
        assert isinstance(trigger_id, str)
        
        # Verify trigger was registered
        mock_trigger_manager.register_trigger.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_conflict_detection(self, hotkey_manager):
        """Test hotkey conflict detection."""
        # Create a hotkey that conflicts with system shortcuts
        hotkey_spec = HotkeySpec(
            key="space",
            modifiers={ModifierKey.COMMAND},
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        
        conflicts = await hotkey_manager.detect_conflicts(hotkey_spec)
        
        # Should detect system conflict
        assert len(conflicts) > 0
        assert any(conflict.conflict_type == "system" for conflict in conflicts)
    
    @pytest.mark.asyncio
    async def test_suggest_alternatives(self, hotkey_manager):
        """Test alternative hotkey suggestions."""
        hotkey_spec = HotkeySpec(
            key="a",
            modifiers={ModifierKey.SHIFT},
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        
        alternatives = hotkey_manager.suggest_alternatives(hotkey_spec, max_suggestions=3)
        
        assert len(alternatives) <= 3
        for alt in alternatives:
            assert isinstance(alt, HotkeySpec)
            assert alt != hotkey_spec  # Should be different from original
    
    def test_hotkey_availability_check(self, hotkey_manager):
        """Test hotkey availability checking."""
        # Available hotkey
        available_hotkey = HotkeySpec(
            key="z",
            modifiers={ModifierKey.COMMAND, ModifierKey.OPTION},
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        
        assert hotkey_manager.is_hotkey_available(available_hotkey)
        
        # System reserved hotkey
        system_hotkey = HotkeySpec(
            key="space",
            modifiers={ModifierKey.COMMAND},
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        
        # This should raise SecurityViolationError during creation
        with pytest.raises(SecurityViolationError):
            HotkeySpec(
                key="space",
                modifiers={ModifierKey.COMMAND},
                activation_mode=ActivationMode.PRESSED,
                tap_count=1,
                allow_repeat=False
            )


class TestHotkeySpecToKMString:
    """Test hotkey string conversion functionality."""
    
    def test_modifier_ordering(self):
        """Test that modifiers are consistently ordered."""
        spec = HotkeySpec(
            key="a",
            modifiers={ModifierKey.SHIFT, ModifierKey.COMMAND, ModifierKey.OPTION},
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        
        km_string = spec.to_km_string()
        # Should be in consistent order: cmd+ctrl+opt+shift+fn
        assert km_string == "cmd+opt+shift+a"
    
    def test_single_key_no_modifiers(self):
        """Test single key without modifiers."""
        spec = HotkeySpec(
            key="f",
            modifiers=set(),
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        
        assert spec.to_km_string() == "f"
    
    def test_special_key_with_modifiers(self):
        """Test special key with modifiers."""
        spec = HotkeySpec(
            key="f1",
            modifiers={ModifierKey.COMMAND, ModifierKey.CONTROL},
            activation_mode=ActivationMode.PRESSED,
            tap_count=1,
            allow_repeat=False
        )
        
        assert spec.to_km_string() == "cmd+ctrl+f1"


class TestHotkeySecurityValidation:
    """Test security aspects of hotkey validation."""
    
    @given(st.text().filter(lambda x: not x.isascii() if x else False))
    def test_non_ascii_keys_rejected(self, non_ascii_key):
        """Property: Non-ASCII characters should be rejected for security."""
        assume(len(non_ascii_key) == 1)  # Single character
        
        with pytest.raises((ValidationError, SecurityViolationError)):
            HotkeySpec(
                key=non_ascii_key,
                modifiers=set(),
                activation_mode=ActivationMode.PRESSED,
                tap_count=1,
                allow_repeat=False
            )
    
    def test_system_shortcut_protection(self):
        """Test that critical system shortcuts are protected."""
        critical_shortcuts = [
            ("space", {ModifierKey.COMMAND}),  # Spotlight
            ("tab", {ModifierKey.COMMAND}),    # App Switcher
        ]
        
        for key, modifiers in critical_shortcuts:
            with pytest.raises(SecurityViolationError):
                HotkeySpec(
                    key=key,
                    modifiers=modifiers,
                    activation_mode=ActivationMode.PRESSED,
                    tap_count=1,
                    allow_repeat=False
                )
    
    @given(st.text(min_size=2, max_size=10).filter(lambda x: x.isascii()))
    def test_multi_character_keys_validation(self, multi_char_key):
        """Property: Multi-character keys must be in valid special keys list."""
        assume(multi_char_key.lower() not in VALID_SPECIAL_KEYS)
        
        with pytest.raises(ValidationError):
            HotkeySpec(
                key=multi_char_key,
                modifiers=set(),
                activation_mode=ActivationMode.PRESSED,
                tap_count=1,
                allow_repeat=False
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])