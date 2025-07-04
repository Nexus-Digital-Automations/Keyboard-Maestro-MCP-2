"""
Hotkey Trigger Management System

Provides type-safe hotkey creation with conflict detection, validation,
and security boundaries for Keyboard Maestro automation.
"""

import re
import logging
import uuid
from dataclasses import dataclass, field
from typing import Set, List, Optional, Dict, Any, Tuple
from enum import Enum

from ..core.types import MacroId, TriggerId
from ..core.contracts import require, ensure
from ..core.errors import ValidationError, SecurityViolationError
from ..integration.km_client import KMError, KMClient, Either
from ..integration.triggers import TriggerDefinition, TriggerRegistrationManager
from ..integration.events import TriggerType

logger = logging.getLogger(__name__)


class ModifierKey(Enum):
    """Supported modifier keys with validation."""
    COMMAND = "cmd"
    OPTION = "opt" 
    SHIFT = "shift"
    CONTROL = "ctrl"
    FUNCTION = "fn"
    
    @classmethod
    def from_string(cls, modifier: str) -> "ModifierKey":
        """Create ModifierKey from string with validation."""
        normalized = modifier.lower().strip()
        for mod in cls:
            if mod.value == normalized or mod.name.lower() == normalized:
                return mod
        raise ValidationError("modifier", modifier, f"Invalid modifier key: {modifier}")


class ActivationMode(Enum):
    """Hotkey activation modes."""
    PRESSED = "pressed"
    RELEASED = "released"
    TAPPED = "tapped"
    HELD = "held"
    
    @classmethod
    def from_string(cls, mode: str) -> "ActivationMode":
        """Create ActivationMode from string with validation."""
        normalized = mode.lower().strip()
        for activation in cls:
            if activation.value == normalized:
                return activation
        raise ValidationError("activation_mode", mode, f"Invalid activation mode: {mode}")


# Valid special keys for keyboard shortcuts
VALID_SPECIAL_KEYS = frozenset([
    "space", "tab", "enter", "return", "escape", "delete", "backspace",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    "home", "end", "pageup", "pagedown", "up", "down", "left", "right",
    "clear", "help", "insert"
])

# System-reserved hotkey combinations that should not be overridden
SYSTEM_RESERVED_HOTKEYS = frozenset([
    "cmd+space",  # Spotlight
    "cmd+tab",    # App Switcher
    "cmd+shift+tab",  # Reverse App Switcher
    "cmd+opt+esc",    # Force Quit
    "cmd+ctrl+space", # Character Viewer
    "cmd+ctrl+f",     # Full Screen
])


@dataclass(frozen=True)
class HotkeySpec:
    """Type-safe hotkey specification with comprehensive validation."""
    key: str
    modifiers: Set[ModifierKey]
    activation_mode: ActivationMode = ActivationMode.PRESSED
    tap_count: int = 1
    allow_repeat: bool = False
    
    def __post_init__(self):
        """Validate hotkey specification."""
        self._validate_key()
        self._validate_tap_count()
        self._validate_modifiers()
        self._check_system_conflicts()
    
    @require(lambda self: len(self.key) >= 1)
    def _validate_key(self) -> None:
        """Validate key specification."""
        if not self.key:
            raise ValidationError("key", self.key, "Key cannot be empty")
        
        # Single character keys (a-z, 0-9)
        if len(self.key) == 1:
            if not (self.key.isalnum() and self.key.isascii()):
                raise ValidationError("key", self.key, "Single character keys must be alphanumeric ASCII")
            return
        
        # Special keys
        if self.key.lower() not in VALID_SPECIAL_KEYS:
            raise ValidationError("key", self.key, f"Invalid special key. Valid keys: {', '.join(sorted(VALID_SPECIAL_KEYS))}")
    
    @require(lambda self: 1 <= self.tap_count <= 4)
    def _validate_tap_count(self) -> None:
        """Validate tap count."""
        if not (1 <= self.tap_count <= 4):
            raise ValidationError("tap_count", self.tap_count, "Tap count must be between 1 and 4")
    
    def _validate_modifiers(self) -> None:
        """Validate modifier key combinations."""
        if not isinstance(self.modifiers, set):
            raise ValidationError("modifiers", self.modifiers, "Modifiers must be a set")
        
        # Ensure all modifiers are valid ModifierKey instances
        for mod in self.modifiers:
            if not isinstance(mod, ModifierKey):
                raise ValidationError("modifiers", mod, f"Invalid modifier type: {type(mod)}")
    
    def _check_system_conflicts(self) -> None:
        """Check for conflicts with system shortcuts."""
        hotkey_string = self.to_km_string()
        if hotkey_string.lower() in SYSTEM_RESERVED_HOTKEYS:
            raise SecurityViolationError("hotkey", hotkey_string, f"Conflicts with system shortcut: {hotkey_string}")
    
    def to_km_string(self) -> str:
        """Convert to Keyboard Maestro hotkey string format."""
        # Sort modifiers for consistent representation
        modifier_order = [ModifierKey.COMMAND, ModifierKey.CONTROL, ModifierKey.OPTION, ModifierKey.SHIFT, ModifierKey.FUNCTION]
        sorted_modifiers = [mod for mod in modifier_order if mod in self.modifiers]
        
        modifier_str = "+".join(mod.value for mod in sorted_modifiers)
        key_str = self.key.lower()
        
        if modifier_str:
            return f"{modifier_str}+{key_str}"
        return key_str
    
    def to_display_string(self) -> str:
        """Convert to human-readable display format."""
        # Use Unicode symbols for better display
        symbol_map = {
            ModifierKey.COMMAND: "⌘",
            ModifierKey.OPTION: "⌥", 
            ModifierKey.SHIFT: "⇧",
            ModifierKey.CONTROL: "⌃",
            ModifierKey.FUNCTION: "fn"
        }
        
        modifier_order = [ModifierKey.COMMAND, ModifierKey.CONTROL, ModifierKey.OPTION, ModifierKey.SHIFT, ModifierKey.FUNCTION]
        sorted_modifiers = [mod for mod in modifier_order if mod in self.modifiers]
        
        modifier_str = "".join(symbol_map.get(mod, mod.value) for mod in sorted_modifiers)
        key_str = self.key.upper() if len(self.key) == 1 else self.key.title()
        
        result = f"{modifier_str}{key_str}"
        
        if self.tap_count > 1:
            result += f" (×{self.tap_count})"
        
        if self.activation_mode != ActivationMode.PRESSED:
            result += f" ({self.activation_mode.value})"
            
        return result
    
    def to_km_trigger_config(self) -> Dict[str, Any]:
        """Convert to Keyboard Maestro trigger configuration."""
        config = {
            "key": self.key,
            "modifiers": [mod.value for mod in self.modifiers],
            "activation_mode": self.activation_mode.value,
            "allow_repeat": self.allow_repeat
        }
        
        if self.tap_count > 1:
            config["tap_count"] = self.tap_count
            
        return config


@dataclass(frozen=True)
class HotkeyConflict:
    """Information about a hotkey conflict."""
    conflicting_hotkey: str
    conflict_type: str  # "system", "existing_macro", "application"
    description: str
    macro_name: Optional[str] = None
    suggestion: Optional[str] = None


def create_hotkey_spec(
    key: str,
    modifiers: List[str],
    activation_mode: str = "pressed",
    tap_count: int = 1,
    allow_repeat: bool = False
) -> HotkeySpec:
    """Factory function to create HotkeySpec with validation."""
    try:
        # Convert string modifiers to ModifierKey enum
        modifier_set = set()
        for mod_str in modifiers:
            modifier_set.add(ModifierKey.from_string(mod_str))
        
        # Convert activation mode
        activation = ActivationMode.from_string(activation_mode)
        
        return HotkeySpec(
            key=key.lower().strip(),
            modifiers=modifier_set,
            activation_mode=activation,
            tap_count=tap_count,
            allow_repeat=allow_repeat
        )
    except Exception as e:
        if isinstance(e, (ValidationError, SecurityViolationError)):
            raise
        raise ValidationError("hotkey_spec", {"key": key, "modifiers": modifiers}, f"Failed to create hotkey spec: {str(e)}")


class HotkeyManager:
    """Manages hotkey creation and conflict detection."""
    
    def __init__(self, km_client: KMClient, trigger_manager: TriggerRegistrationManager):
        self._km_client = km_client
        self._trigger_manager = trigger_manager
        self._registered_hotkeys: Dict[str, Tuple[MacroId, HotkeySpec]] = {}
    
    @require(lambda macro_id: macro_id)
    @require(lambda hotkey: isinstance(hotkey, HotkeySpec))
    @ensure(lambda result: result.is_right() or result.get_left().code in ["CONFLICT_ERROR", "INVALID_HOTKEY", "VALIDATION_ERROR"])
    async def create_hotkey_trigger(
        self,
        macro_id: MacroId,
        hotkey: HotkeySpec,
        check_conflicts: bool = True
    ) -> Either[KMError, TriggerId]:
        """Create hotkey trigger with comprehensive validation and conflict detection."""
        try:
            # Conflict detection
            if check_conflicts:
                conflicts = await self.detect_conflicts(hotkey)
                if conflicts:
                    conflict_desc = "; ".join(c.description for c in conflicts)
                    return Either.left(KMError.validation_error(
                        f"Hotkey conflicts detected: {conflict_desc}",
                        details={"conflicts": [self._conflict_to_dict(c) for c in conflicts]}
                    ))
            
            # Generate unique trigger ID
            trigger_id = TriggerId(str(uuid.uuid4()))
            
            # Create trigger definition
            trigger_def = TriggerDefinition(
                trigger_id=trigger_id,
                macro_id=macro_id,
                trigger_type=TriggerType.HOTKEY,
                configuration=hotkey.to_km_trigger_config(),
                name=f"Hotkey: {hotkey.to_display_string()}",
                description=f"Hotkey trigger for macro {macro_id}",
                enabled=True
            )
            
            # Register with trigger management system
            registration_result = await self._trigger_manager.register_trigger(trigger_def)
            
            if registration_result.is_left():
                return registration_result
            
            # Track registered hotkey
            hotkey_string = hotkey.to_km_string()
            self._registered_hotkeys[hotkey_string] = (macro_id, hotkey)
            
            logger.info(f"Successfully created hotkey trigger {trigger_id} for macro {macro_id}: {hotkey.to_display_string()}")
            
            return Either.right(trigger_id)
            
        except Exception as e:
            logger.error(f"Failed to create hotkey trigger for macro {macro_id}: {str(e)}")
            return Either.left(KMError.execution_error(f"Hotkey trigger creation failed: {str(e)}"))
    
    async def detect_conflicts(self, hotkey: HotkeySpec) -> List[HotkeyConflict]:
        """Detect conflicts with existing hotkeys and system shortcuts."""
        conflicts = []
        hotkey_string = hotkey.to_km_string()
        
        # Check system conflicts
        if hotkey_string.lower() in SYSTEM_RESERVED_HOTKEYS:
            conflicts.append(HotkeyConflict(
                conflicting_hotkey=hotkey_string,
                conflict_type="system",
                description=f"Conflicts with system shortcut {hotkey_string}",
                suggestion=self._suggest_alternative_modifier(hotkey)
            ))
        
        # Check existing macro conflicts
        if hotkey_string in self._registered_hotkeys:
            existing_macro_id, existing_hotkey = self._registered_hotkeys[hotkey_string]
            conflicts.append(HotkeyConflict(
                conflicting_hotkey=hotkey_string,
                conflict_type="existing_macro",
                description=f"Hotkey already assigned to macro {existing_macro_id}",
                macro_name=str(existing_macro_id),
                suggestion=self._suggest_alternative_key(hotkey)
            ))
        
        # Check application-specific conflicts (simplified for now)
        app_conflicts = await self._check_application_conflicts(hotkey)
        conflicts.extend(app_conflicts)
        
        return conflicts
    
    async def _check_application_conflicts(self, hotkey: HotkeySpec) -> List[HotkeyConflict]:
        """Check for conflicts with application-specific shortcuts."""
        # This would integrate with system APIs to check app shortcuts
        # For now, implementing basic known conflicts
        conflicts = []
        hotkey_string = hotkey.to_km_string()
        
        known_app_shortcuts = {
            "cmd+c": "Copy (Universal)",
            "cmd+v": "Paste (Universal)",
            "cmd+x": "Cut (Universal)",
            "cmd+z": "Undo (Universal)",
            "cmd+shift+z": "Redo (Universal)",
            "cmd+a": "Select All (Universal)",
            "cmd+s": "Save (Universal)",
            "cmd+o": "Open (Universal)",
            "cmd+n": "New (Universal)",
            "cmd+w": "Close Window (Universal)",
            "cmd+q": "Quit Application (Universal)"
        }
        
        if hotkey_string.lower() in known_app_shortcuts:
            app_name = known_app_shortcuts[hotkey_string.lower()]
            conflicts.append(HotkeyConflict(
                conflicting_hotkey=hotkey_string,
                conflict_type="application",
                description=f"Conflicts with common application shortcut: {app_name}",
                suggestion=self._suggest_alternative_modifier(hotkey)
            ))
        
        return conflicts
    
    def suggest_alternatives(self, hotkey: HotkeySpec, max_suggestions: int = 3) -> List[HotkeySpec]:
        """Suggest alternative hotkey combinations."""
        suggestions = []
        
        # Try different modifier combinations
        alt_suggestions = [
            self._suggest_alternative_modifier(hotkey),
            self._suggest_alternative_key(hotkey),
            self._suggest_different_modifier_combo(hotkey)
        ]
        
        for suggestion_str in alt_suggestions:
            if suggestion_str and len(suggestions) < max_suggestions:
                try:
                    # Parse suggestion back to HotkeySpec
                    suggested_spec = self._parse_hotkey_string(suggestion_str, hotkey)
                    if suggested_spec and suggested_spec != hotkey:
                        suggestions.append(suggested_spec)
                except:
                    continue  # Skip invalid suggestions
        
        return suggestions[:max_suggestions]
    
    def _suggest_alternative_modifier(self, hotkey: HotkeySpec) -> Optional[str]:
        """Suggest alternative modifier combination."""
        current_mods = hotkey.modifiers
        
        # Try adding Command if not present
        if ModifierKey.COMMAND not in current_mods:
            new_mods = current_mods | {ModifierKey.COMMAND}
            return self._build_hotkey_string(hotkey.key, new_mods)
        
        # Try adding Option if not present
        if ModifierKey.OPTION not in current_mods:
            new_mods = current_mods | {ModifierKey.OPTION}
            return self._build_hotkey_string(hotkey.key, new_mods)
        
        # Try adding Control if not present
        if ModifierKey.CONTROL not in current_mods:
            new_mods = current_mods | {ModifierKey.CONTROL}
            return self._build_hotkey_string(hotkey.key, new_mods)
        
        return None
    
    def _suggest_alternative_key(self, hotkey: HotkeySpec) -> Optional[str]:
        """Suggest alternative key with same modifiers."""
        if len(hotkey.key) == 1 and hotkey.key.isalpha():
            # Try adjacent keys on QWERTY layout
            qwerty_adjacents = {
                'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'],
                'r': ['e', 't', 'f'], 't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'],
                'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'],
                'p': ['o', 'l'], 'a': ['q', 's', 'z'], 's': ['a', 'w', 'd', 'x'],
                'd': ['s', 'e', 'f', 'c'], 'f': ['d', 'r', 'g', 'v'],
                'g': ['f', 't', 'h', 'b'], 'h': ['g', 'y', 'j', 'n'],
                'j': ['h', 'u', 'k', 'm'], 'k': ['j', 'i', 'l'],
                'l': ['k', 'o', 'p'], 'z': ['a', 'x'], 'x': ['z', 's', 'c'],
                'c': ['x', 'd', 'v'], 'v': ['c', 'f', 'b'], 'b': ['v', 'g', 'n'],
                'n': ['b', 'h', 'm'], 'm': ['n', 'j']
            }
            
            adjacents = qwerty_adjacents.get(hotkey.key.lower(), [])
            for alt_key in adjacents:
                alt_hotkey_string = self._build_hotkey_string(alt_key, hotkey.modifiers)
                if alt_hotkey_string not in self._registered_hotkeys:
                    return alt_hotkey_string
        
        return None
    
    def _suggest_different_modifier_combo(self, hotkey: HotkeySpec) -> Optional[str]:
        """Suggest completely different modifier combination."""
        alternative_combos = [
            {ModifierKey.COMMAND, ModifierKey.SHIFT},
            {ModifierKey.COMMAND, ModifierKey.OPTION},
            {ModifierKey.COMMAND, ModifierKey.CONTROL},
            {ModifierKey.OPTION, ModifierKey.SHIFT},
            {ModifierKey.CONTROL, ModifierKey.SHIFT},
            {ModifierKey.COMMAND, ModifierKey.OPTION, ModifierKey.SHIFT}
        ]
        
        for combo in alternative_combos:
            if combo != hotkey.modifiers:
                alt_hotkey_string = self._build_hotkey_string(hotkey.key, combo)
                if alt_hotkey_string not in self._registered_hotkeys:
                    return alt_hotkey_string
        
        return None
    
    def _build_hotkey_string(self, key: str, modifiers: Set[ModifierKey]) -> str:
        """Build hotkey string from key and modifiers."""
        modifier_order = [ModifierKey.COMMAND, ModifierKey.CONTROL, ModifierKey.OPTION, ModifierKey.SHIFT, ModifierKey.FUNCTION]
        sorted_modifiers = [mod for mod in modifier_order if mod in modifiers]
        
        modifier_str = "+".join(mod.value for mod in sorted_modifiers)
        
        if modifier_str:
            return f"{modifier_str}+{key.lower()}"
        return key.lower()
    
    def _parse_hotkey_string(self, hotkey_string: str, reference: HotkeySpec) -> Optional[HotkeySpec]:
        """Parse hotkey string back to HotkeySpec."""
        try:
            parts = hotkey_string.split('+')
            key = parts[-1]
            modifier_strings = parts[:-1]
            
            return create_hotkey_spec(
                key=key,
                modifiers=modifier_strings,
                activation_mode=reference.activation_mode.value,
                tap_count=reference.tap_count,
                allow_repeat=reference.allow_repeat
            )
        except:
            return None
    
    def _conflict_to_dict(self, conflict: HotkeyConflict) -> Dict[str, Any]:
        """Convert HotkeyConflict to dictionary."""
        return {
            "conflicting_hotkey": conflict.conflicting_hotkey,
            "conflict_type": conflict.conflict_type,
            "description": conflict.description,
            "macro_name": conflict.macro_name,
            "suggestion": conflict.suggestion
        }
    
    def get_registered_hotkeys(self) -> Dict[str, Tuple[MacroId, HotkeySpec]]:
        """Get all registered hotkeys."""
        return self._registered_hotkeys.copy()
    
    def is_hotkey_available(self, hotkey: HotkeySpec) -> bool:
        """Check if hotkey is available for use."""
        hotkey_string = hotkey.to_km_string()
        return hotkey_string not in self._registered_hotkeys and hotkey_string.lower() not in SYSTEM_RESERVED_HOTKEYS