"""
Macro Creation Engine

This module provides comprehensive macro creation capabilities for Keyboard Maestro,
including template-based creation, security validation, and builder patterns.
"""

from .macro_builder import MacroBuilder, MacroCreationRequest, MacroTemplate
from .templates import (
    MacroTemplateGenerator, 
    HotkeyActionTemplate, 
    AppLauncherTemplate,
    TextExpansionTemplate,
    FileProcessorTemplate,
    WindowManagerTemplate
)

__all__ = [
    "MacroBuilder",
    "MacroCreationRequest", 
    "MacroTemplate",
    "MacroTemplateGenerator",
    "HotkeyActionTemplate",
    "AppLauncherTemplate", 
    "TextExpansionTemplate",
    "FileProcessorTemplate",
    "WindowManagerTemplate"
]