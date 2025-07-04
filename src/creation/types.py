"""
Creation module types and enums.

Type definitions for macro creation system to avoid circular imports.
"""

from enum import Enum


class MacroTemplate(Enum):
    """Pre-built macro templates for common automation patterns."""
    HOTKEY_ACTION = "hotkey_action"
    APP_LAUNCHER = "app_launcher" 
    TEXT_EXPANSION = "text_expansion"
    FILE_PROCESSOR = "file_processor"
    WINDOW_MANAGER = "window_manager"
    CUSTOM = "custom"