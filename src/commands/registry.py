"""
Command Registry System

Provides type-safe command registration, discovery, and instantiation
with security validation and permission management.
"""

from __future__ import annotations
from typing import Dict, Type, Optional, FrozenSet, Any, List
from dataclasses import dataclass, field
from enum import Enum

from ..core.types import CommandId, CommandParameters, Permission
from ..core.parser import CommandType
from ..core.contracts import require, ensure
from .base import BaseCommand, CommandContract
from .validation import SecurityValidator, CommandSecurityError


@dataclass(frozen=True)
class CommandMetadata:
    """Metadata about a registered command."""
    command_type: CommandType
    command_class: Type[BaseCommand]
    required_permissions: FrozenSet[Permission]
    security_risk_level: str
    description: str
    parameter_schema: Dict[str, Any] = field(default_factory=dict)


class CommandRegistry:
    """
    Type-safe command registration and discovery system.
    
    Manages the registration of command types and provides
    safe instantiation with validation and security checks.
    """
    
    def __init__(self):
        self._commands: Dict[CommandType, CommandMetadata] = {}
        self._security_validator = SecurityValidator()
    
    # @require(lambda self, command_type, command_class: issubclass(command_class, BaseCommand))
    def register_command(
        self,
        command_type: CommandType,
        command_class: Type[BaseCommand],
        description: str = "",
        parameter_schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new command type with validation.
        
        Args:
            command_type: Type identifier for the command
            command_class: Command implementation class
            description: Human-readable description
            parameter_schema: JSON schema for parameter validation
            
        Raises:
            ValueError: If command_class is not a BaseCommand subclass
        """
        if not issubclass(command_class, BaseCommand):
            raise ValueError(f"Command class must inherit from BaseCommand: {command_class}")
        
        # Create a temporary instance to get metadata
        temp_instance = self._create_temp_instance(command_class)
        
        metadata = CommandMetadata(
            command_type=command_type,
            command_class=command_class,
            required_permissions=temp_instance.get_required_permissions(),
            security_risk_level=temp_instance.get_security_risk_level(),
            description=description,
            parameter_schema=parameter_schema or {}
        )
        
        self._commands[command_type] = metadata
    
    def is_registered(self, command_type: CommandType) -> bool:
        """Check if a command type is registered."""
        return command_type in self._commands
    
    def get_command_metadata(self, command_type: CommandType) -> Optional[CommandMetadata]:
        """Get metadata for a registered command type."""
        return self._commands.get(command_type)
    
    def get_registered_commands(self) -> List[CommandType]:
        """Get list of all registered command types."""
        return list(self._commands.keys())
    
    def get_commands_by_permission(self, permission: Permission) -> List[CommandType]:
        """Get commands that require a specific permission."""
        return [
            cmd_type for cmd_type, metadata in self._commands.items()
            if permission in metadata.required_permissions
        ]
    
    def get_commands_by_risk_level(self, risk_level: str) -> List[CommandType]:
        """Get commands with a specific security risk level."""
        return [
            cmd_type for cmd_type, metadata in self._commands.items()
            if metadata.security_risk_level == risk_level
        ]
    
    # @require(lambda self, command_type: command_type in self._commands)
    def create_command(
        self,
        command_type: CommandType,
        command_id: CommandId,
        parameters: Dict[str, Any]
    ) -> BaseCommand:
        """
        Create and validate a command instance.
        
        Args:
            command_type: Type of command to create
            command_id: Unique identifier for the command instance
            parameters: Command parameters
            
        Returns:
            Validated command instance ready for execution
            
        Raises:
            ValueError: If command type is not registered
            CommandSecurityError: If parameters fail security validation
        """
        if command_type not in self._commands:
            raise ValueError(f"Command type not registered: {command_type}")
        
        metadata = self._commands[command_type]
        
        # Validate parameters for security threats
        self._security_validator.clear_threats()
        is_safe = self._validate_parameters(command_type.value, parameters)
        
        if not is_safe or self._security_validator.has_critical_threats():
            raise CommandSecurityError(
                f"Security validation failed for {command_type.value} command",
                self._security_validator.get_threats()
            )
        
        # Create command parameters object
        command_params = CommandParameters(parameters)
        
        # Instantiate command
        command_instance = metadata.command_class(
            command_id=command_id,
            parameters=command_params
        )
        
        # Validate the complete command
        if not command_instance.validate():
            raise ValueError(f"Command validation failed for {command_type}")
        
        return command_instance
    
    def _create_temp_instance(self, command_class: Type[BaseCommand]) -> BaseCommand:
        """Create a temporary instance for metadata extraction."""
        try:
            # Try to create with minimal parameters
            return command_class(
                command_id=CommandId("temp"),
                parameters=CommandParameters.empty()
            )
        except Exception:
            # If that fails, we can't get metadata safely
            raise ValueError(f"Cannot create temporary instance of {command_class}")
    
    def _validate_parameters(self, command_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate command parameters using security validator."""
        try:
            # Use the validation function from validation module
            from .validation import validate_command_parameters
            return validate_command_parameters(command_type, parameters)
        except CommandSecurityError:
            # The validator already recorded the threats
            return False
        except Exception:
            # Any other validation error is considered unsafe
            return False


def get_default_registry() -> CommandRegistry:
    """
    Get a command registry with all default commands registered.
    
    Returns:
        Registry with standard command implementations
    """
    registry = CommandRegistry()
    
    # Import and register command implementations
    try:
        from .text import TypeTextCommand, FindTextCommand, ReplaceTextCommand
        from .system import PauseCommand, PlaySoundCommand, SetVolumeCommand
        from .application import LaunchApplicationCommand, QuitApplicationCommand, ActivateApplicationCommand
        from .flow import ConditionalCommand, LoopCommand, BreakCommand
        
        # Register text commands
        registry.register_command(
            CommandType.TEXT_INPUT,
            TypeTextCommand,
            "Type text with configurable speed and validation"
        )
        
        # Register system commands
        registry.register_command(
            CommandType.PAUSE,
            PauseCommand,
            "Pause execution for specified duration"
        )
        
        registry.register_command(
            CommandType.PLAY_SOUND,
            PlaySoundCommand,
            "Play system sound with volume control"
        )
        
        # Register application commands
        registry.register_command(
            CommandType.APPLICATION_CONTROL,
            LaunchApplicationCommand,
            "Launch, quit, or activate applications"
        )
        
        # Register flow control commands
        registry.register_command(
            CommandType.CONDITIONAL,
            ConditionalCommand,
            "Execute commands based on conditions"
        )
        
        registry.register_command(
            CommandType.LOOP,
            LoopCommand,
            "Execute commands in a loop with safety limits"
        )
        
    except ImportError as e:
        # Some command implementations may not be available yet
        # This is OK during development
        pass
    
    return registry