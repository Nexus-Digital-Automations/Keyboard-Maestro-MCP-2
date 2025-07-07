"""Macro Command Library - Public API.

Provides a comprehensive library of macro commands with strong type safety,
contract-based validation, and security boundaries.
"""

from .application import (
    ActivateApplicationCommand,
    LaunchApplicationCommand,
    QuitApplicationCommand,
)
from .base import BaseCommand, CommandContract
from .flow import BreakCommand, ConditionalCommand, LoopCommand
from .registry import CommandRegistry, get_default_registry
from .system import PauseCommand, PlaySoundCommand, SetVolumeCommand
from .text import FindTextCommand, ReplaceTextCommand, TypeTextCommand
from .validation import (
    CommandSecurityError,
    SecurityValidator,
    validate_command_parameters,
    validate_file_path,
    validate_text_input,
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
    "ActivateApplicationCommand",
    # Base classes
    "BaseCommand",
    "BreakCommand",
    "CommandContract",
    # Registry
    "CommandRegistry",
    "CommandSecurityError",
    # Flow control commands
    "ConditionalCommand",
    "FindTextCommand",
    # Application commands
    "LaunchApplicationCommand",
    "LoopCommand",
    # System commands
    "PauseCommand",
    "PlaySoundCommand",
    "QuitApplicationCommand",
    "ReplaceTextCommand",
    "SecurityValidator",
    "SetVolumeCommand",
    # Text commands
    "TypeTextCommand",
    "get_command_registry",
    "validate_command_parameters",
    "validate_file_path",
    # Validation
    "validate_text_input",
]
