"""
Macro Template System

Provides template generators for common automation patterns with security
validation and parameter processing for Keyboard Maestro macro creation.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum

from ..core.errors import ValidationError, SecurityViolationError
from .types import MacroTemplate

logger = logging.getLogger(__name__)


class MacroTemplateGenerator(ABC):
    """Abstract base for macro template generators with security validation."""
    
    @abstractmethod
    async def generate_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate action configurations for this template."""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate template-specific parameters."""
        pass
    
    async def validate_security(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters for security compliance."""
        # Base security validation for all templates
        if not isinstance(parameters, dict):
            raise ValidationError("parameters", parameters, "Parameters must be a dictionary")
        
        # Check for script injection in string parameters
        for key, value in parameters.items():
            if isinstance(value, str):
                self._validate_string_security(key, value)
    
    def _validate_string_security(self, key: str, value: str) -> None:
        """Validate string parameter for security threats."""
        # Security: Detect potential script injection patterns
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'\$\([^)]*\)',
            r'`[^`]*`',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'shell\s*\(',
            r'\/bin\/',
            r'\.\.\/\.\.\/',  # Path traversal
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise SecurityViolationError(
                    key, value, f"Security threat detected in parameter {key}: {pattern}"
                )


class HotkeyActionTemplate(MacroTemplateGenerator):
    """Template for hotkey-triggered actions with comprehensive validation."""
    
    async def generate_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hotkey action configuration with security validation."""
        if not self.validate_parameters(parameters):
            raise ValidationError("parameters", parameters, "Invalid hotkey action parameters")
        
        action_type = parameters.get('action', 'type_text')
        
        if action_type == 'open_app':
            return await self._generate_app_launch_action(parameters)
        elif action_type == 'type_text':
            return await self._generate_text_action(parameters)
        elif action_type == 'run_script':
            return await self._generate_script_action(parameters)
        else:
            raise ValidationError("action", action_type, "Unsupported hotkey action type")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate hotkey action parameters."""
        required_fields = ['action']
        
        for field in required_fields:
            if field not in parameters:
                raise ValidationError(field, None, f"Required field {field} missing")
        
        # Validate hotkey format if provided
        if 'hotkey' in parameters:
            self._validate_hotkey_format(parameters['hotkey'])
        
        return True
    
    def _validate_hotkey_format(self, hotkey: str) -> None:
        """Validate hotkey format for security and correctness."""
        # Security: Restrict to safe hotkey patterns
        valid_modifiers = {'Cmd', 'Command', 'Ctrl', 'Control', 'Opt', 'Option', 'Shift'}
        valid_keys = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        valid_special = {'Space', 'Return', 'Tab', 'Delete', 'Escape', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12'}
        
        # Parse hotkey components
        parts = [part.strip() for part in hotkey.split('+')]
        if len(parts) < 2:
            raise ValidationError("hotkey", hotkey, "Hotkey must include modifier + key")
        
        # Validate modifiers and key
        modifiers = parts[:-1]
        key = parts[-1]
        
        for modifier in modifiers:
            if modifier not in valid_modifiers:
                raise ValidationError("hotkey", hotkey, f"Invalid modifier: {modifier}")
        
        if key not in valid_keys and key not in valid_special:
            raise ValidationError("hotkey", hotkey, f"Invalid key: {key}")
    
    async def _generate_app_launch_action(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate application launch action."""
        app_name = parameters.get('app_name', '')
        if not app_name:
            raise ValidationError("app_name", app_name, "Application name required")
        
        # Security: Validate app name format
        if not re.match(r'^[a-zA-Z0-9\s\-\.]+$', app_name):
            raise SecurityViolationError("app_name", app_name, "Invalid application name format")
        
        return [{
            'type': 'Launch Application',
            'app_name': app_name,
            'parameters': {
                'bundleID': parameters.get('bundle_id', ''),
                'ignoreIfRunning': parameters.get('ignore_if_running', True)
            }
        }]
    
    async def _generate_text_action(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate text typing action."""
        text = parameters.get('text', '')
        if not text:
            raise ValidationError("text", text, "Text content required")
        
        # Security: Limit text length
        if len(text) > 10000:
            raise SecurityViolationError("text", text, "Text too long (max 10000 characters)")
        
        return [{
            'type': 'Type a String',
            'text': text,
            'parameters': {
                'typingSpeed': parameters.get('typing_speed', 'Normal'),
                'simulateTyping': parameters.get('simulate_typing', True)
            }
        }]
    
    async def _generate_script_action(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate script execution action with strict security."""
        script_type = parameters.get('script_type', 'applescript')
        script_content = parameters.get('script_content', '')
        
        # Security: Validate script content
        if not script_content:
            raise ValidationError("script_content", script_content, "Script content required")
        
        # Security: Strict validation for script execution
        await self._validate_script_security(script_type, script_content)
        
        return [{
            'type': 'Execute Script',
            'script_type': script_type,
            'script_content': script_content,
            'parameters': {
                'timeout': parameters.get('timeout', 30),
                'outputHandling': parameters.get('output_handling', 'None')
            }
        }]
    
    async def _validate_script_security(self, script_type: str, script_content: str) -> None:
        """Strict security validation for script content."""
        # Security: Only allow specific script types
        allowed_types = {'applescript', 'shell'}
        if script_type not in allowed_types:
            raise SecurityViolationError("script_type", script_type, "Script type not allowed")
        
        # Security: Block dangerous script patterns
        dangerous_patterns = [
            r'rm\s+-rf',
            r'sudo\s+',
            r'curl\s+.*\|\s*sh',
            r'wget\s+.*\|\s*sh',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'\/bin\/sh',
            r'\/bin\/bash',
            r'password',
            r'keychain',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, script_content, re.IGNORECASE):
                raise SecurityViolationError(
                    "script_content", script_content, f"Dangerous script pattern detected: {pattern}"
                )


class AppLauncherTemplate(MacroTemplateGenerator):
    """Template for application launcher macros with bundle ID validation."""
    
    async def generate_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate app launcher configuration with bundle ID validation."""
        if not self.validate_parameters(parameters):
            raise ValidationError("parameters", parameters, "Invalid app launcher parameters")
        
        app_identifier = parameters.get('app_name') or parameters.get('bundle_id')
        if not app_identifier:
            raise ValidationError("app_identifier", app_identifier, "App name or bundle ID required")
        
        # Determine if identifier is bundle ID or app name
        is_bundle_id = '.' in app_identifier and not app_identifier.endswith('.app')
        
        if is_bundle_id:
            self._validate_bundle_id(app_identifier)
        else:
            self._validate_app_name(app_identifier)
        
        return [{
            'type': 'Launch Application',
            'app_identifier': app_identifier,
            'parameters': {
                'bundleID': app_identifier if is_bundle_id else '',
                'applicationName': app_identifier if not is_bundle_id else '',
                'ignoreIfRunning': parameters.get('ignore_if_running', True),
                'bringToFront': parameters.get('bring_to_front', True)
            }
        }]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate app launcher parameters."""
        if not parameters.get('app_name') and not parameters.get('bundle_id'):
            raise ValidationError("app_identifier", None, "Either app_name or bundle_id required")
        
        return True
    
    def _validate_bundle_id(self, bundle_id: str) -> None:
        """Validate bundle ID format and security."""
        # Security: Validate bundle ID format
        bundle_pattern = re.compile(r'^[a-zA-Z0-9\-]+(\.[a-zA-Z0-9\-]+)+$')
        if not bundle_pattern.match(bundle_id):
            raise SecurityViolationError("bundle_id", bundle_id, "Invalid bundle ID format")
        
        # Security: Block suspicious bundle IDs
        suspicious_patterns = [
            r'system\.',
            r'kernel\.',
            r'root\.',
            r'admin\.',
            r'security\.',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, bundle_id, re.IGNORECASE):
                raise SecurityViolationError("bundle_id", bundle_id, "Suspicious bundle ID detected")
    
    def _validate_app_name(self, app_name: str) -> None:
        """Validate application name format and security."""
        # Security: Validate app name format
        if not re.match(r'^[a-zA-Z0-9\s\-\.]+$', app_name):
            raise SecurityViolationError("app_name", app_name, "Invalid application name format")
        
        # Security: Block system application names
        system_apps = {
            'system preferences', 'terminal', 'activity monitor', 'keychain access',
            'console', 'directory utility', 'disk utility', 'migration assistant'
        }
        
        if app_name.lower() in system_apps:
            raise SecurityViolationError("app_name", app_name, "System application access restricted")


class TextExpansionTemplate(MacroTemplateGenerator):
    """Template for text expansion macros with content validation."""
    
    async def generate_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate text expansion configuration."""
        if not self.validate_parameters(parameters):
            raise ValidationError("parameters", parameters, "Invalid text expansion parameters")
        
        expansion_text = parameters.get('expansion_text', '')
        abbreviation = parameters.get('abbreviation', '')
        
        return [{
            'type': 'Type a String',
            'text': expansion_text,
            'parameters': {
                'typingSpeed': parameters.get('typing_speed', 'Fast'),
                'simulateTyping': False,
                'restoreClipboard': True
            }
        }]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate text expansion parameters."""
        required_fields = ['expansion_text', 'abbreviation']
        
        for field in required_fields:
            if field not in parameters or not parameters[field]:
                raise ValidationError(field, parameters.get(field), f"Required field {field} missing")
        
        # Validate abbreviation format
        abbreviation = parameters['abbreviation']
        if not re.match(r'^[a-zA-Z0-9]+$', abbreviation):
            raise ValidationError("abbreviation", abbreviation, "Abbreviation must be alphanumeric")
        
        # Validate expansion text length
        expansion_text = parameters['expansion_text']
        if len(expansion_text) > 5000:
            raise ValidationError("expansion_text", expansion_text, "Expansion text too long (max 5000)")
        
        return True


class FileProcessorTemplate(MacroTemplateGenerator):
    """Template for file processing workflows with path validation."""
    
    async def generate_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate file processing workflow."""
        if not self.validate_parameters(parameters):
            raise ValidationError("parameters", parameters, "Invalid file processor parameters")
        
        actions = []
        
        # Add file trigger if watch_folder specified
        if parameters.get('watch_folder'):
            actions.append({
                'type': 'File Trigger',
                'watch_folder': parameters['watch_folder'],
                'parameters': {
                    'filePattern': parameters.get('file_pattern', '*'),
                    'recursive': parameters.get('recursive', False)
                }
            })
        
        # Add processing actions
        action_chain = parameters.get('action_chain', ['copy'])
        for action in action_chain:
            actions.append(await self._generate_file_action(action, parameters))
        
        return actions
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate file processor parameters with security checks."""
        # Validate paths for security
        if 'watch_folder' in parameters:
            self._validate_path_security(parameters['watch_folder'])
        
        if 'destination' in parameters:
            self._validate_path_security(parameters['destination'])
        
        # Validate action chain
        if 'action_chain' in parameters:
            allowed_actions = {'copy', 'move', 'rename', 'resize', 'optimize'}
            for action in parameters['action_chain']:
                if action not in allowed_actions:
                    raise ValidationError("action_chain", action, f"Action {action} not allowed")
        
        return True
    
    def _validate_path_security(self, path: str) -> None:
        """Validate file path for security compliance."""
        # Security: Block dangerous paths
        dangerous_paths = [
            r'\/System\/',
            r'\/usr\/bin\/',
            r'\/bin\/',
            r'\/etc\/',
            r'\/var\/log\/',
            r'\.\.\/\.\.\/',  # Path traversal
            r'~\/Library\/Keychains\/',
            r'~\/\.ssh\/',
        ]
        
        for pattern in dangerous_paths:
            if re.search(pattern, path, re.IGNORECASE):
                raise SecurityViolationError("path", path, f"Dangerous path detected: {pattern}")
    
    async def _generate_file_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific file action configuration."""
        if action == 'copy':
            return {
                'type': 'Copy File',
                'destination': parameters.get('destination', ''),
                'parameters': {
                    'overwrite': parameters.get('overwrite', False)
                }
            }
        elif action == 'move':
            return {
                'type': 'Move File',
                'destination': parameters.get('destination', ''),
                'parameters': {
                    'overwrite': parameters.get('overwrite', False)
                }
            }
        elif action == 'rename':
            return {
                'type': 'Rename File',
                'name_pattern': parameters.get('name_pattern', 'File %Index%'),
                'parameters': {}
            }
        else:
            raise ValidationError("action", action, f"Unsupported file action: {action}")


class WindowManagerTemplate(MacroTemplateGenerator):
    """Template for window management macros with coordinate validation."""
    
    async def generate_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate window management actions."""
        if not self.validate_parameters(parameters):
            raise ValidationError("parameters", parameters, "Invalid window manager parameters")
        
        operation = parameters.get('operation', 'move')
        
        if operation == 'move':
            return await self._generate_move_action(parameters)
        elif operation == 'resize':
            return await self._generate_resize_action(parameters)
        elif operation == 'arrange':
            return await self._generate_arrange_action(parameters)
        else:
            raise ValidationError("operation", operation, "Unsupported window operation")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate window manager parameters."""
        operation = parameters.get('operation', 'move')
        
        if operation in ['move', 'resize']:
            if 'position' in parameters:
                self._validate_coordinates(parameters['position'])
            if 'size' in parameters:
                self._validate_size(parameters['size'])
        
        return True
    
    def _validate_coordinates(self, position: Dict[str, int]) -> None:
        """Validate window coordinates for security."""
        if not isinstance(position, dict):
            raise ValidationError("position", position, "Position must be a dictionary")
        
        x = position.get('x', 0)
        y = position.get('y', 0)
        
        # Security: Limit coordinates to reasonable ranges
        if not (-5000 <= x <= 10000) or not (-5000 <= y <= 10000):
            raise SecurityViolationError("coordinates", position, "Coordinates outside safe range")
    
    def _validate_size(self, size: Dict[str, int]) -> None:
        """Validate window size for security."""
        if not isinstance(size, dict):
            raise ValidationError("size", size, "Size must be a dictionary")
        
        width = size.get('width', 800)
        height = size.get('height', 600)
        
        # Security: Limit size to reasonable ranges
        if not (50 <= width <= 10000) or not (50 <= height <= 10000):
            raise SecurityViolationError("size", size, "Size outside safe range")
    
    async def _generate_move_action(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate window move action."""
        position = parameters.get('position', {'x': 100, 'y': 100})
        
        return [{
            'type': 'Move Window',
            'position': position,
            'parameters': {
                'screen': parameters.get('screen', 'Main'),
                'animate': parameters.get('animate', True)
            }
        }]
    
    async def _generate_resize_action(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate window resize action."""
        size = parameters.get('size', {'width': 800, 'height': 600})
        
        return [{
            'type': 'Resize Window',
            'size': size,
            'parameters': {
                'animate': parameters.get('animate', True)
            }
        }]
    
    async def _generate_arrange_action(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate window arrangement action."""
        arrangement = parameters.get('arrangement', 'left_half')
        
        return [{
            'type': 'Arrange Window',
            'arrangement': arrangement,
            'parameters': {
                'screen': parameters.get('screen', 'Main')
            }
        }]


def get_template_generator(template: MacroTemplate) -> MacroTemplateGenerator:
    """Factory function to get appropriate template generator."""
    generators = {
        MacroTemplate.HOTKEY_ACTION: HotkeyActionTemplate(),
        MacroTemplate.APP_LAUNCHER: AppLauncherTemplate(),
        MacroTemplate.TEXT_EXPANSION: TextExpansionTemplate(),
        MacroTemplate.FILE_PROCESSOR: FileProcessorTemplate(),
        MacroTemplate.WINDOW_MANAGER: WindowManagerTemplate(),
    }
    
    if template not in generators:
        raise ValidationError("template", template, f"Unsupported template: {template}")
    
    return generators[template]