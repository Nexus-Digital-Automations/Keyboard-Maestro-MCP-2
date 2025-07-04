"""
Macro Command Library - Public API

Provides a comprehensive library of macro commands with strong type safety,
contract-based validation, and security boundaries.
"""

from .base import BaseCommand, CommandContract
from .registry import CommandRegistry, get_default_registry
from .text import TypeTextCommand, FindTextCommand, ReplaceTextCommand
from .system import PauseCommand, PlaySoundCommand, SetVolumeCommand
from .application import LaunchApplicationCommand, QuitApplicationCommand, ActivateApplicationCommand
from .flow import ConditionalCommand, LoopCommand, BreakCommand
from .validation import (
    validate_text_input, validate_file_path, validate_command_parameters,
    SecurityValidator, CommandSecurityError
)

# Command registry instance
_registry = None

def get_command_registry() -> CommandRegistry:
    """Get the global command registry."""
    global _registry
    if _registry is None:
        _registry = get_default_registry()
    return _registry

# Expose key types for external use
__all__ = [
    # Base classes
    'BaseCommand', 'CommandContract',
    
    # Registry
    'CommandRegistry', 'get_command_registry',
    
    # Text commands
    'TypeTextCommand', 'FindTextCommand', 'ReplaceTextCommand',
    
    # System commands
    'PauseCommand', 'PlaySoundCommand', 'SetVolumeCommand',
    
    # Application commands
    'LaunchApplicationCommand', 'QuitApplicationCommand', 'ActivateApplicationCommand',
    
    # Flow control commands
    'ConditionalCommand', 'LoopCommand', 'BreakCommand',
    
    # Validation
    'validate_text_input', 'validate_file_path', 'validate_command_parameters',
    'SecurityValidator', 'CommandSecurityError'
]