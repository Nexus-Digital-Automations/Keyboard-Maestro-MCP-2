"""Trigger management system for Keyboard Maestro automation.

Provides specialized trigger implementations with comprehensive validation,
conflict detection, and security boundaries.
"""

from .hotkey_manager import (
    ActivationMode,
    HotkeyConflict,
    HotkeyManager,
    HotkeySpec,
    ModifierKey,
    create_hotkey_spec,
)

__all__ = [
    "ActivationMode",
    "HotkeyConflict",
    "HotkeyManager",
    "HotkeySpec",
    "ModifierKey",
    "create_hotkey_spec",
]
