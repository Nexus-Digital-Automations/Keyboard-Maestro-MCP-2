"""
Trigger management system for Keyboard Maestro automation.

Provides specialized trigger implementations with comprehensive validation,
conflict detection, and security boundaries.
"""

from .hotkey_manager import (
    HotkeySpec,
    ModifierKey,
    ActivationMode,
    HotkeyManager,
    HotkeyConflict,
    create_hotkey_spec
)

__all__ = [
    "HotkeySpec",
    "ModifierKey", 
    "ActivationMode",
    "HotkeyManager",
    "HotkeyConflict",
    "create_hotkey_spec"
]