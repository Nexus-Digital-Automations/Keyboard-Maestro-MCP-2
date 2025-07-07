"""Macro Creation Engine.

This module provides comprehensive macro creation capabilities for Keyboard Maestro,
including template-based creation, security validation, and builder patterns.
"""

from .macro_builder import MacroBuilder, MacroCreationRequest, MacroTemplate
from .templates import (
    AppLauncherTemplate,
    FileProcessorTemplate,
    HotkeyActionTemplate,
    MacroTemplateGenerator,
    TextExpansionTemplate,
    WindowManagerTemplate,
)

__all__ = [
    "AppLauncherTemplate",
    "FileProcessorTemplate",
    "HotkeyActionTemplate",
    "MacroBuilder",
    "MacroCreationRequest",
    "MacroTemplate",
    "MacroTemplateGenerator",
    "TextExpansionTemplate",
    "WindowManagerTemplate",
]
