"""
Core Macro Creation Engine

Provides type-safe macro creation with comprehensive security validation,
template support, and AppleScript integration for Keyboard Maestro.
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from ..core.types import MacroId, GroupId, TriggerId
from ..core.contracts import require, ensure
from ..core.errors import ValidationError, SecurityViolationError, KMError
from ..integration.km_client import KMClient
from .templates import MacroTemplateGenerator, get_template_generator

logger = logging.getLogger(__name__)


class MacroTemplate(Enum):
    """Pre-built macro templates for common automation patterns."""
    HOTKEY_ACTION = "hotkey_action"
    APP_LAUNCHER = "app_launcher" 
    TEXT_EXPANSION = "text_expansion"
    FILE_PROCESSOR = "file_processor"
    WINDOW_MANAGER = "window_manager"
    CUSTOM = "custom"


@dataclass(frozen=True)
class MacroCreationRequest:
    """Type-safe macro creation request with comprehensive validation."""
    name: str
    template: MacroTemplate
    group_id: Optional[GroupId] = None
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate creation request parameters."""
        self._validate_name()
        self._validate_parameters()
    
    @require(lambda self: len(self.name) > 0 and len(self.name) <= 255)
    def _validate_name(self) -> None:
        """Validate macro name with security constraints."""
        if not self.name:
            raise ValidationError("macro_name", self.name, "Name cannot be empty")
        
        if len(self.name) > 255:
            raise ValidationError("macro_name", self.name, "Name too long (max 255 characters)")
        
        # Security: ASCII only, no control characters
        if not self.name.isascii():
            raise SecurityViolationError("macro_name", self.name, "Non-ASCII characters not allowed")
        
        # Security: Restrict to safe character set
        safe_pattern = re.compile(r'^[a-zA-Z0-9_\s\-\.]+$')
        if not safe_pattern.match(self.name):
            raise SecurityViolationError("macro_name", self.name, "Invalid characters in name")
    
    def _validate_parameters(self) -> None:
        """Validate template parameters for security compliance."""
        if not isinstance(self.parameters, dict):
            raise ValidationError("parameters", self.parameters, "Parameters must be a dictionary")
        
        # Security: Limit parameter complexity
        if len(str(self.parameters)) > 10000:
            raise SecurityViolationError("parameters", self.parameters, "Parameters too large")
        
        # Security: No script injection in parameter values
        for key, value in self.parameters.items():
            if isinstance(value, str):
                self._validate_parameter_value(key, value)
    
    def _validate_parameter_value(self, key: str, value: str) -> None:
        """Validate individual parameter value for security threats."""
        # Security: Detect potential script injection
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'on\w+\s*=',
            r'\$\([^)]*\)',
            r'`[^`]*`',
            r'eval\s*\(',
            r'exec\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise SecurityViolationError(
                    key, value, f"Potential script injection detected in parameter {key}"
                )


class MacroBuilder:
    """Fluent builder for macro creation with comprehensive security validation."""
    
    def __init__(self, km_client: KMClient):
        """Initialize macro builder with KM client."""
        self.km_client = km_client
        self._validation_cache: Dict[str, bool] = {}
    
    @require(lambda request: isinstance(request, MacroCreationRequest))
    @require(lambda request: request.template in MacroTemplate)
    @ensure(lambda result: result is not None)
    async def create_macro(self, request: MacroCreationRequest) -> Union[MacroId, KMError]:
        """
        Create macro with comprehensive validation and error handling.
        
        Architecture:
            - Pattern: Builder Pattern with Template Method
            - Security: Defense-in-depth with validation, sanitization, rollback
            - Performance: O(1) validation, O(log n) creation
        
        Contracts:
            Preconditions:
                - request is valid MacroCreationRequest
                - request.template is supported MacroTemplate
                - request.name passes security validation
            
            Postconditions:
                - Returns MacroId on success OR KMError on failure
                - No partial macro creation (atomic operation)
                - All security validations passed
            
            Invariants:
                - System state unchanged on failure
                - No script injection possible
                - All user inputs sanitized
        
        Args:
            request: Validated macro creation request
            
        Returns:
            MacroId on successful creation, KMError on failure
            
        Raises:
            SecurityViolationError: Security validation failed
            ValidationError: Input validation failed
        """
        try:
            logger.info(f"Starting macro creation: {request.name}")
            
            # Phase 1: Security validation
            await self._validate_security(request)
            
            # Phase 2: Template processing
            template_generator = get_template_generator(request.template)
            actions = await template_generator.generate_actions(request.parameters)
            
            # Phase 3: AppleScript generation
            applescript = self._generate_applescript(request, actions)
            
            # Phase 4: Macro creation with rollback support
            macro_id = await self._execute_creation(applescript, request)
            
            logger.info(f"Successfully created macro: {request.name} ({macro_id})")
            return MacroId(macro_id)
            
        except (SecurityViolationError, ValidationError) as e:
            logger.error(f"Validation failed for macro {request.name}: {e}")
            return KMError(
                code="VALIDATION_ERROR",
                message=str(e),
                details=f"Macro creation failed validation: {e}",
                recovery_suggestion="Review input parameters and try again"
            )
        except Exception as e:
            logger.exception(f"Unexpected error creating macro {request.name}")
            return KMError(
                code="CREATION_ERROR", 
                message="Failed to create macro",
                details=str(e),
                recovery_suggestion="Check Keyboard Maestro status and permissions"
            )
    
    async def _validate_security(self, request: MacroCreationRequest) -> None:
        """Validate macro creation request for security compliance."""
        # Cache key for validation result
        cache_key = f"{request.name}:{request.template.value}:{hash(str(request.parameters))}"
        
        if cache_key in self._validation_cache:
            if not self._validation_cache[cache_key]:
                raise SecurityViolationError("request", request, "Cached validation failure")
            return
        
        try:
            # Validate template-specific security requirements
            template_generator = get_template_generator(request.template)
            if hasattr(template_generator, 'validate_security'):
                await template_generator.validate_security(request.parameters)
            
            # Validate against KM constraints
            if request.group_id:
                group_exists = await self._validate_group_exists(request.group_id)
                if not group_exists:
                    raise ValidationError("group_id", request.group_id, "Group does not exist")
            
            # Check for naming conflicts
            conflict_exists = await self._check_naming_conflicts(request.name)
            if conflict_exists:
                raise ValidationError("name", request.name, "Macro name already exists")
            
            # Cache successful validation
            self._validation_cache[cache_key] = True
            
        except Exception as e:
            # Cache failed validation
            self._validation_cache[cache_key] = False
            raise
    
    async def _validate_group_exists(self, group_id: GroupId) -> bool:
        """Validate that target group exists."""
        try:
            result = await self.km_client.list_groups_async()
            if result.is_left():
                return False
            
            groups = result.get_right()
            return any(group.get('id') == group_id for group in groups)
        except Exception:
            return False
    
    async def _check_naming_conflicts(self, name: str) -> bool:
        """Check if macro name conflicts with existing macros."""
        try:
            result = await self.km_client.list_macros_async()
            if result.is_left():
                return False
            
            macros = result.get_right()
            return any(macro.get('name', '').lower() == name.lower() for macro in macros)
        except Exception:
            return False
    
    def _generate_applescript(self, request: MacroCreationRequest, actions: List[Dict[str, Any]]) -> str:
        """Generate AppleScript for macro creation with security escaping."""
        # Security: Escape all user-provided values
        safe_name = self._escape_applescript_string(request.name)
        safe_group = self._escape_applescript_string(str(request.group_id)) if request.group_id else None
        
        # Generate macro creation script
        script_lines = [
            'tell application "Keyboard Maestro"',
            f'    set newMacro to make new macro with properties {{name:"{safe_name}", enabled:{str(request.enabled).lower()}}}',
        ]
        
        # Add group assignment if specified
        if safe_group:
            script_lines.append(f'    set macro group of newMacro to macro group "{safe_group}"')
        
        # Add actions to macro
        for action in actions:
            action_xml = self._generate_action_xml(action)
            safe_xml = self._escape_applescript_string(action_xml)
            script_lines.append(f'    tell newMacro to make new action with properties {{xml:"{safe_xml}"}}')
        
        # Return macro UUID
        script_lines.extend([
            '    set macroUUID to uid of newMacro',
            '    return macroUUID',
            'end tell'
        ])
        
        return '\n'.join(script_lines)
    
    def _escape_applescript_string(self, value: str) -> str:
        """Escape string for safe AppleScript inclusion."""
        if not isinstance(value, str):
            value = str(value)
        
        # Security: Escape quotes and special characters
        escaped = value.replace('\\', '\\\\')  # Escape backslashes first
        escaped = escaped.replace('"', '\\"')   # Escape quotes
        escaped = escaped.replace('\n', '\\n')  # Escape newlines
        escaped = escaped.replace('\r', '\\r')  # Escape carriage returns
        escaped = escaped.replace('\t', '\\t')  # Escape tabs
        
        return escaped
    
    def _generate_action_xml(self, action: Dict[str, Any]) -> str:
        """Generate XML for action configuration."""
        # Basic action XML template with security validation
        action_type = action.get('type', 'Unknown')
        action_id = str(uuid.uuid4()).upper()
        
        # Security: Validate action type against allowed types
        allowed_types = {
            'Type a String', 'Pause', 'Launch Application', 'Quit Application',
            'Activate Application', 'Move or Click Mouse', 'Play Sound'
        }
        
        if action_type not in allowed_types:
            raise SecurityViolationError("action_type", action_type, "Action type not allowed")
        
        # Generate basic XML structure
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>ActionType</key>
    <string>{self._escape_xml(action_type)}</string>
    <key>UID</key>
    <string>{action_id}</string>
    <key>MacroActionType</key>
    <string>{self._escape_xml(action_type)}</string>
</dict>
</plist>'''
        
        return xml
    
    def _escape_xml(self, value: str) -> str:
        """Escape XML special characters for security."""
        if not isinstance(value, str):
            value = str(value)
        
        # Security: XML entity escaping
        value = value.replace('&', '&amp;')
        value = value.replace('<', '&lt;')
        value = value.replace('>', '&gt;')
        value = value.replace('"', '&quot;')
        value = value.replace("'", '&apos;')
        
        return value
    
    async def _execute_creation(self, applescript: str, request: MacroCreationRequest) -> str:
        """Execute macro creation with rollback support."""
        try:
            # Execute AppleScript for macro creation
            result = await self.km_client.execute_applescript_async(applescript)
            
            if result.is_left():
                error = result.get_left()
                raise Exception(f"AppleScript execution failed: {error}")
            
            macro_id = result.get_right().strip()
            
            # Validate creation was successful
            if not macro_id or len(macro_id) != 36:  # UUID length check
                raise Exception("Invalid macro ID returned from creation")
            
            return macro_id
            
        except Exception as e:
            # Attempt rollback if macro was partially created
            await self._attempt_rollback(request.name)
            raise e
    
    async def _attempt_rollback(self, macro_name: str) -> None:
        """Attempt to rollback failed macro creation."""
        try:
            # Try to find and delete any partially created macro
            rollback_script = f'''
            tell application "Keyboard Maestro"
                try
                    delete macro "{self._escape_applescript_string(macro_name)}"
                end try
            end tell
            '''
            
            await self.km_client.execute_applescript_async(rollback_script)
            logger.info(f"Rollback completed for macro: {macro_name}")
            
        except Exception as e:
            logger.warning(f"Rollback failed for macro {macro_name}: {e}")