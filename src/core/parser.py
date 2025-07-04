"""
Command parsing and validation for the Keyboard Maestro MCP macro engine.

This module provides robust parsing of macro definitions with comprehensive
type validation, security checks, and contract-based validation.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Type, cast
from dataclasses import dataclass, field
import json
import re
from enum import Enum

from .types import (
    MacroId, CommandId, MacroDefinition, MacroCommand, CommandParameters,
    Permission, Duration
)
from .errors import (
    ValidationError, SecurityViolationError, create_error_context,
    MacroEngineError
)
from .contracts import require, ensure, is_not_none, is_valid_string


class CommandType(Enum):
    """Supported command types for macro parsing."""
    TEXT_INPUT = "text_input"
    PAUSE = "pause"
    PLAY_SOUND = "play_sound"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    VARIABLE_SET = "variable_set"
    VARIABLE_GET = "variable_get"
    APPLICATION_CONTROL = "application_control"
    SYSTEM_CONTROL = "system_control"


@dataclass(frozen=True)
class ParseResult:
    """Result of macro parsing operation."""
    success: bool
    macro_definition: Optional[MacroDefinition] = None
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def success_result(cls, macro_def: MacroDefinition) -> ParseResult:
        """Create successful parse result."""
        return cls(success=True, macro_definition=macro_def)
    
    @classmethod
    def failure_result(cls, errors: List[ValidationError]) -> ParseResult:
        """Create failed parse result."""
        return cls(success=False, errors=errors)


class InputSanitizer:
    """Handles input sanitization and security validation."""
    
    # Security patterns to detect potential threats
    SCRIPT_INJECTION_PATTERNS = [
        r'<script\b[^>]*>.*?</script>',  # Script tags
        r'<script\b[^>]*>',  # Script opening tags
        r'javascript:',  # JavaScript URLs
        r'vbscript:',  # VBScript URLs  
        r'data:text/html',  # Data URLs
        r'on\w+\s*=',  # Event handlers (onload, onerror, etc.)
        r'eval\s*\(',  # eval() calls
        r'exec\s*\(',  # exec() calls
        r'system\s*\(',  # system() calls
        r'os\.system',  # os.system calls
        r'subprocess\.',  # subprocess calls
        r'__import__',  # import statements
        r'alert\s*\(',  # JavaScript alerts
        r'document\.',  # DOM access
        r'window\.',  # Window object access
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\.',
        r'~/',
        r'/etc/',
        r'/bin/',
        r'/usr/bin/',
        r'C:\\Windows\\',
        r'C:\\System32\\',
        r'%SYSTEMROOT%',
        r'%USERPROFILE%',
        r'%TEMP%',
        r'%TMP%',
        r'\\system32\\',
        r'/system32/',
        r'\\windows\\',
        r'/windows/',
    ]
    
    @classmethod
    def sanitize_text_input(cls, text: str, max_length: int = 10000, strict_mode: bool = True) -> str:
        """
        Sanitize text input for security.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            strict_mode: If True, raises SecurityViolationError for dangerous content.
                        If False, removes dangerous content and returns cleaned text.
        """
        if not isinstance(text, str):
            raise ValidationError(
                field_name="text_input",
                value=text,
                constraint="must be string type"
            )
        
        if len(text) > max_length:
            raise ValidationError(
                field_name="text_input",
                value=f"length {len(text)}",
                constraint=f"must be <= {max_length} characters"
            )
        
        sanitized = text.strip()
        
        if strict_mode:
            # Strict mode: Detect and raise errors for dangerous content
            for pattern in cls.SCRIPT_INJECTION_PATTERNS:
                if re.search(pattern, sanitized, flags=re.IGNORECASE):
                    raise SecurityViolationError(
                        violation_type="script_injection",
                        details=f"Detected script injection pattern: {pattern}"
                    )
            
            for pattern in cls.PATH_TRAVERSAL_PATTERNS:
                if re.search(pattern, sanitized, flags=re.IGNORECASE):
                    raise SecurityViolationError(
                        violation_type="path_traversal",
                        details=f"Detected path traversal pattern: {pattern}"
                    )
            
            # Check for dangerous characters in strict mode
            dangerous_chars = r'[<>"\'\&;]'
            if re.search(dangerous_chars, sanitized):
                raise SecurityViolationError(
                    violation_type="dangerous_characters",
                    details=f"Detected dangerous characters in input"
                )
            
            return sanitized
        else:
            # Sanitization mode: Remove dangerous content
            # Remove script injection patterns
            for pattern in cls.SCRIPT_INJECTION_PATTERNS:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            
            # Remove path traversal patterns
            for pattern in cls.PATH_TRAVERSAL_PATTERNS:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            
            # Remove dangerous characters
            sanitized = re.sub(r'[<>"\'\&;]', '', sanitized)
            
            return sanitized
    
    @classmethod
    def validate_file_path(cls, path: str) -> str:
        """Validate and sanitize file paths."""
        if not isinstance(path, str):
            raise ValidationError(
                field_name="file_path",
                value=path,
                constraint="must be string type"
            )
        
        # Check for path traversal
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                raise SecurityViolationError(
                    violation_type="path_traversal",
                    details=f"Detected potential path traversal: {pattern}"
                )
        
        return path.strip()
    
    @classmethod
    def validate_identifier(cls, identifier: str, max_length: int = 255) -> str:
        """Validate macro/command identifiers."""
        if not is_valid_string(identifier, min_length=1, max_length=max_length):
            raise ValidationError(
                field_name="identifier",
                value=identifier,
                constraint=f"must be 1-{max_length} character string"
            )
        
        # Only allow safe characters for identifiers (must start with letter, no special injection patterns)
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_\-\s\.]*$', identifier):
            raise ValidationError(
                field_name="identifier",
                value=identifier,
                constraint="must start with a letter and contain only alphanumeric characters, spaces, dots, dashes, underscores"
            )
        
        # Additional security checks for injection patterns
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'eval\s*\(',
            r'system\s*\(',
            r'\.\.',
            r'DROP\s+TABLE',
            r'[\n\r\t]'  # No newlines, carriage returns, or tabs
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, identifier, re.IGNORECASE):
                raise ValidationError(
                    field_name="identifier",
                    value=identifier,
                    constraint=f"contains dangerous pattern: {pattern}"
                )
        
        return identifier.strip()


class CommandValidator:
    """Validates individual command specifications."""
    
    REQUIRED_PERMISSIONS = {
        CommandType.TEXT_INPUT: {Permission.TEXT_INPUT},
        CommandType.PAUSE: set(),
        CommandType.PLAY_SOUND: {Permission.SYSTEM_SOUND},
        CommandType.CONDITIONAL: set(),
        CommandType.LOOP: set(),
        CommandType.VARIABLE_SET: set(),
        CommandType.VARIABLE_GET: set(),
        CommandType.APPLICATION_CONTROL: {Permission.APPLICATION_CONTROL},
        CommandType.SYSTEM_CONTROL: {Permission.SYSTEM_CONTROL},
    }
    
    @classmethod
    def validate_command_type(cls, command_type: str) -> CommandType:
        """Validate and convert command type string."""
        try:
            return CommandType(command_type)
        except ValueError:
            valid_types = [ct.value for ct in CommandType]
            raise ValidationError(
                field_name="command_type",
                value=command_type,
                constraint=f"must be one of: {valid_types}"
            )
    
    @classmethod
    def validate_command_parameters(
        cls,
        command_type: CommandType,
        parameters: Dict[str, Any]
    ) -> CommandParameters:
        """Validate command parameters based on type."""
        validated_params = {}
        
        if command_type == CommandType.TEXT_INPUT:
            validated_params.update(cls._validate_text_input_params(parameters))
        elif command_type == CommandType.PAUSE:
            validated_params.update(cls._validate_pause_params(parameters))
        elif command_type == CommandType.PLAY_SOUND:
            validated_params.update(cls._validate_sound_params(parameters))
        elif command_type == CommandType.VARIABLE_SET:
            validated_params.update(cls._validate_variable_set_params(parameters))
        elif command_type == CommandType.VARIABLE_GET:
            validated_params.update(cls._validate_variable_get_params(parameters))
        else:
            # For other command types, perform basic validation
            validated_params = parameters.copy()
        
        return CommandParameters(validated_params)
    
    @classmethod
    def _validate_text_input_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate text input command parameters."""
        if 'text' not in params:
            raise ValidationError(
                field_name="text",
                value="missing",
                constraint="text parameter is required"
            )
        
        text = InputSanitizer.sanitize_text_input(params['text'], strict_mode=True)
        speed = params.get('speed', 'normal')
        
        if speed not in ['slow', 'normal', 'fast']:
            raise ValidationError(
                field_name="speed",
                value=speed,
                constraint="must be one of: slow, normal, fast"
            )
        
        return {'text': text, 'speed': speed}
    
    @classmethod
    def _validate_pause_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pause command parameters."""
        if 'duration' not in params:
            raise ValidationError(
                field_name="duration",
                value="missing",
                constraint="duration parameter is required"
            )
        
        try:
            duration_value = float(params['duration'])
            if duration_value <= 0 or duration_value > 300:  # Max 5 minutes
                raise ValidationError(
                    field_name="duration",
                    value=duration_value,
                    constraint="must be between 0.1 and 300 seconds"
                )
        except (ValueError, TypeError):
            raise ValidationError(
                field_name="duration",
                value=params['duration'],
                constraint="must be a numeric value"
            )
        
        return {'duration': duration_value}
    
    @classmethod
    def _validate_sound_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sound playback command parameters."""
        if 'sound_name' not in params:
            raise ValidationError(
                field_name="sound_name",
                value="missing",
                constraint="sound_name parameter is required"
            )
        
        sound_name = params['sound_name']
        allowed_sounds = ['beep', 'basso', 'blow', 'bottle', 'frog', 'funk', 'glass', 'hero',
                         'morse', 'ping', 'pop', 'purr', 'sosumi', 'submarine', 'tink']
        
        if sound_name not in allowed_sounds:
            raise ValidationError(
                field_name="sound_name",
                value=sound_name,
                constraint=f"must be one of: {allowed_sounds}"
            )
        
        volume = params.get('volume', 50)
        try:
            volume = int(volume)
            if volume < 0 or volume > 100:
                raise ValidationError(
                    field_name="volume",
                    value=volume,
                    constraint="must be between 0 and 100"
                )
        except (ValueError, TypeError):
            raise ValidationError(
                field_name="volume",
                value=params.get('volume'),
                constraint="must be an integer"
            )
        
        return {'sound_name': sound_name, 'volume': volume}
    
    @classmethod
    def _validate_variable_set_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate variable set command parameters."""
        if 'name' not in params or 'value' not in params:
            raise ValidationError(
                field_name="variable_params",
                value="missing name or value",
                constraint="both name and value parameters are required"
            )
        
        name = InputSanitizer.validate_identifier(params['name'])
        value = str(params['value'])  # Convert to string
        
        return {'name': name, 'value': value}
    
    @classmethod
    def _validate_variable_get_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate variable get command parameters."""
        if 'name' not in params:
            raise ValidationError(
                field_name="name",
                value="missing",
                constraint="name parameter is required"
            )
        
        name = InputSanitizer.validate_identifier(params['name'])
        default_value = params.get('default', '')
        
        return {'name': name, 'default': default_value}
    
    @classmethod
    def get_required_permissions(cls, command_type: CommandType) -> frozenset[Permission]:
        """Get required permissions for command type."""
        return frozenset(cls.REQUIRED_PERMISSIONS.get(command_type, set()))


class MacroParser:
    """Main parser for macro definitions with comprehensive validation."""
    
    @require(lambda self, macro_data: macro_data is not None, "macro_data cannot be None")
    @ensure(lambda self, macro_data, result: isinstance(result, ParseResult), "must return ParseResult")
    def parse_macro(self, macro_data: Dict[str, Any]) -> ParseResult:
        """Parse a macro definition from structured data."""
        errors = []
        warnings = []
        
        try:
            # Validate basic macro structure
            macro_id, name = self._validate_macro_header(macro_data)
            
            # Parse and validate commands
            commands = self._parse_commands(macro_data.get('commands', []))
            
            # Create macro definition
            macro_def = MacroDefinition(
                macro_id=macro_id,
                name=name,
                commands=commands,
                enabled=macro_data.get('enabled', True),
                description=macro_data.get('description')
            )
            
            # Final validation
            if not macro_def.is_valid():
                errors.append(ValidationError(
                    field_name="macro_definition",
                    value="invalid",
                    constraint="macro definition failed validation"
                ))
            
            if errors:
                return ParseResult.failure_result(errors)
            
            return ParseResult.success_result(macro_def)
            
        except ValidationError as e:
            errors.append(e)
            return ParseResult.failure_result(errors)
        except Exception as e:
            error = ValidationError(
                field_name="parsing",
                value=str(e),
                constraint="unexpected error during parsing"
            )
            return ParseResult.failure_result([error])
    
    def _validate_macro_header(self, macro_data: Dict[str, Any]) -> tuple[MacroId, str]:
        """Validate macro header information."""
        # Validate name
        if 'name' not in macro_data:
            raise ValidationError(
                field_name="name",
                value="missing",
                constraint="macro name is required"
            )
        
        name = InputSanitizer.validate_identifier(macro_data['name'])
        
        # Generate or validate macro ID
        macro_id_str = macro_data.get('id', name.lower().replace(' ', '_'))
        macro_id = MacroId(InputSanitizer.validate_identifier(macro_id_str))
        
        return macro_id, name
    
    def _parse_commands(self, commands_data: List[Dict[str, Any]]) -> List[MacroCommand]:
        """Parse and validate command list."""
        if not commands_data:
            raise ValidationError(
                field_name="commands",
                value="empty",
                constraint="at least one command is required"
            )
        
        commands = []
        for i, cmd_data in enumerate(commands_data):
            try:
                command = self._parse_single_command(cmd_data, i)
                commands.append(command)
            except ValidationError as e:
                # Add command index to error context
                e.context = create_error_context(
                    operation="command_parsing",
                    component="macro_parser",
                    command_index=i
                )
                raise e
        
        return commands
    
    def _parse_single_command(self, cmd_data: Dict[str, Any], index: int) -> MacroCommand:
        """Parse a single command definition."""
        # This would return an actual command instance
        # For now, we'll create a placeholder that matches the protocol
        from .engine import PlaceholderCommand  # We'll create this in engine.py
        
        if 'type' not in cmd_data:
            raise ValidationError(
                field_name="type",
                value="missing",
                constraint="command type is required"
            )
        
        command_type = CommandValidator.validate_command_type(cmd_data['type'])
        parameters = CommandValidator.validate_command_parameters(
            command_type,
            cmd_data.get('parameters', {})
        )
        
        return PlaceholderCommand(
            command_id=CommandId(f"cmd_{index}"),
            command_type=command_type,
            parameters=parameters
        )


def parse_macro_from_json(json_string: str) -> ParseResult:
    """Parse macro definition from JSON string."""
    try:
        data = json.loads(json_string)
        parser = MacroParser()
        return parser.parse_macro(data)
    except json.JSONDecodeError as e:
        error = ValidationError(
            field_name="json",
            value=str(e),
            constraint="must be valid JSON"
        )
        return ParseResult.failure_result([error])


def validate_macro_definition(macro_def: MacroDefinition) -> List[ValidationError]:
    """Validate a complete macro definition."""
    errors = []
    
    if not macro_def.is_valid():
        errors.append(ValidationError(
            field_name="macro_definition",
            value="invalid",
            constraint="failed basic validation"
        ))
    
    # Additional validation logic would go here
    
    return errors