"""
Core Action Building System with Builder Pattern and XML Security

Provides comprehensive action building functionality with type safety,
security validation, and XML generation for Keyboard Maestro automation.
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from enum import Enum
from xml.sax.saxutils import escape
import re

from ..core.types import MacroId, Duration
from ..core.contracts import require, ensure
from ..core.errors import ValidationError, SecurityViolationError, MacroEngineError

if TYPE_CHECKING:
    from .action_registry import ActionRegistry

logger = logging.getLogger(__name__)


class ActionCategory(Enum):
    """Action categories for organization and validation."""
    TEXT = "text"
    APPLICATION = "application"
    FILE = "file"
    SYSTEM = "system"
    VARIABLE = "variable"
    CONTROL = "control"
    INTERFACE = "interface"
    WEB = "web"
    CALCULATION = "calculation"
    CLIPBOARD = "clipboard"
    WINDOW = "window"
    SOUND = "sound"


@dataclass(frozen=True)
class ActionType:
    """Type-safe action type definition with validation."""
    identifier: str
    category: ActionCategory
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    description: str = ""
    
    def __post_init__(self):
        """Validate action type definition."""
        if not self.identifier or len(self.identifier.strip()) == 0:
            raise ValueError("Action identifier cannot be empty")
        
        # Validate identifier format (allow slashes for action names like "Encode/Decode Text")
        if not re.match(r'^[a-zA-Z0-9_\s\-\.\/]+$', self.identifier):
            raise ValueError(f"Invalid action identifier format: {self.identifier}")
        
        # Validate parameters don't overlap
        if set(self.required_params) & set(self.optional_params):
            raise ValueError("Parameters cannot be both required and optional")


@dataclass(frozen=True)
class ActionConfiguration:
    """Type-safe action configuration with comprehensive validation."""
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    position: Optional[int] = None
    enabled: bool = True
    timeout: Optional[Duration] = None
    abort_on_failure: bool = False
    
    def __post_init__(self):
        """Validate action configuration on creation."""
        if not self.validate_parameters():
            missing = [p for p in self.action_type.required_params if p not in self.parameters]
            raise ValidationError(f"Missing required parameters for {self.action_type.identifier}: {missing}")
    
    @require(lambda self: self.action_type is not None)
    def validate_parameters(self) -> bool:
        """Validate action parameters against type requirements."""
        # Check required parameters are present
        for param in self.action_type.required_params:
            if param not in self.parameters:
                return False
        
        # Validate parameter types and security
        return self._validate_parameter_security()
    
    def _validate_parameter_security(self) -> bool:
        """Validate parameters for security issues."""
        for param_name, param_value in self.parameters.items():
            # Convert to string for validation
            value_str = str(param_value) if param_value is not None else ""
            
            # Check for dangerous patterns
            if self._contains_dangerous_patterns(value_str):
                logger.warning(f"Dangerous pattern detected in parameter {param_name}")
                return False
            
            # Validate string length limits
            if len(value_str) > 10000:  # 10KB limit per parameter
                logger.warning(f"Parameter {param_name} exceeds length limit")
                return False
        
        return True
    
    def _contains_dangerous_patterns(self, value: str) -> bool:
        """Check for dangerous patterns in parameter values."""
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'vbscript:',
            r'data:text/html',
            r'<!DOCTYPE',
            r'<!ENTITY',
            r'<\?xml',
            r'<!\[CDATA\[',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'shell_exec\s*\(',
            r'passthru\s*\(',
            r'`[^`]*`',  # Backtick execution
        ]
        
        value_lower = value.lower()
        return any(re.search(pattern, value_lower) for pattern in dangerous_patterns)


class ActionBuilder:
    """Fluent builder for constructing action sequences with validation."""
    
    def __init__(self, action_registry: Optional['ActionRegistry'] = None):
        """Initialize builder with action registry."""
        self.actions: List[ActionConfiguration] = []
        self._registry = action_registry
        
        # Initialize registry if not provided
        if self._registry is None:
            from .action_registry import ActionRegistry
            self._registry = ActionRegistry()
    
    @require(lambda self, action_type, parameters, **kwargs: action_type and action_type.strip())
    def add_action(
        self, 
        action_type: str, 
        parameters: Dict[str, Any],
        position: Optional[int] = None,
        enabled: bool = True,
        timeout: Optional[Duration] = None,
        abort_on_failure: bool = False
    ) -> 'ActionBuilder':
        """Add action to sequence with parameter validation."""
        # Get action definition from registry
        action_def = self._registry.get_action_type(action_type)
        if not action_def:
            available = ", ".join(self._registry.list_action_names()[:10])
            raise ValidationError(f"Unknown action type: {action_type}. Available: {available}...")
        
        # Create configuration with validation
        config = ActionConfiguration(
            action_type=action_def,
            parameters=parameters,
            position=position,
            enabled=enabled,
            timeout=timeout,
            abort_on_failure=abort_on_failure
        )
        
        # Insert at specific position or append
        if position is not None and 0 <= position <= len(self.actions):
            self.actions.insert(position, config)
        else:
            self.actions.append(config)
        
        return self
    
    def add_text_action(self, text: str, by_typing: bool = True, **kwargs) -> 'ActionBuilder':
        """Convenience method for adding text input actions."""
        return self.add_action(
            "Type a String",
            {"text": text, "by_typing": by_typing},
            **kwargs
        )
    
    def add_pause_action(self, duration: Duration, **kwargs) -> 'ActionBuilder':
        """Convenience method for adding pause actions."""
        return self.add_action(
            "Pause",
            {"duration": duration.total_seconds()},
            **kwargs
        )
    
    def add_if_action(self, condition: Dict[str, Any], **kwargs) -> 'ActionBuilder':
        """Convenience method for adding conditional actions."""
        return self.add_action(
            "If Then Else",
            {"condition": condition},
            **kwargs
        )
    
    def add_variable_action(self, variable_name: str, value: str, **kwargs) -> 'ActionBuilder':
        """Convenience method for setting variables."""
        return self.add_action(
            "Set Variable to Text",
            {"variable": variable_name, "text": value},
            **kwargs
        )
    
    def add_app_action(self, application: str, bring_all_windows: bool = False, **kwargs) -> 'ActionBuilder':
        """Convenience method for activating applications."""
        return self.add_action(
            "Activate a Specific Application",
            {"application": application, "bring_all_windows": bring_all_windows},
            **kwargs
        )
    
    @ensure(lambda self, result: result and isinstance(result, dict) and 'success' in result)
    def build_xml(self) -> Dict[str, Any]:
        """Generate XML for all actions with security validation."""
        try:
            if not self.actions:
                return {
                    "success": False,
                    "error": "No actions to build",
                    "xml": ""
                }
            
            # Create root element
            root = ET.Element("actions")
            
            # Generate XML for each action
            for i, action in enumerate(self.actions):
                action_elem = self._generate_action_xml(action, i)
                root.append(action_elem)
            
            # Generate XML string
            xml_string = ET.tostring(root, encoding='unicode', method='xml')
            
            # Validate XML security
            if not self._validate_xml_security(xml_string):
                return {
                    "success": False,
                    "error": "Generated XML failed security validation",
                    "xml": ""
                }
            
            # Pretty format XML
            formatted_xml = self._format_xml(xml_string)
            
            return {
                "success": True,
                "xml": formatted_xml,
                "action_count": len(self.actions),
                "validation_passed": True
            }
            
        except Exception as e:
            logger.error(f"XML generation failed: {str(e)}")
            return {
                "success": False,
                "error": f"XML generation failed: {str(e)}",
                "xml": ""
            }
    
    def _generate_action_xml(self, action: ActionConfiguration, index: int) -> ET.Element:
        """Generate XML element for single action with proper escaping."""
        action_elem = ET.Element("action")
        action_elem.set("type", action.action_type.identifier)
        action_elem.set("id", str(index))
        action_elem.set("enabled", str(action.enabled).lower())
        
        if action.abort_on_failure:
            action_elem.set("abortOnFailure", "true")
        
        if action.timeout:
            action_elem.set("timeout", str(action.timeout.total_seconds()))
        
        # Add parameters with proper escaping
        for param_name, param_value in action.parameters.items():
            param_elem = ET.SubElement(action_elem, param_name)
            
            # Handle different parameter types
            if isinstance(param_value, bool):
                param_elem.text = str(param_value).lower()
            elif isinstance(param_value, (int, float)):
                param_elem.text = str(param_value)
            elif isinstance(param_value, dict):
                # For complex parameters, serialize as attributes
                for key, value in param_value.items():
                    param_elem.set(key, escape(str(value)))
            else:
                # String values - escape properly
                param_elem.text = escape(str(param_value))
        
        return action_elem
    
    def _validate_xml_security(self, xml_string: str) -> bool:
        """Validate XML for security issues."""
        # Check for XML injection patterns
        dangerous_patterns = [
            r'<!DOCTYPE',
            r'<!ENTITY',
            r'<\?xml[^>]*encoding[^>]*>',  # Only allow default encoding
            r'<!\[CDATA\[',
            r'javascript:',
            r'vbscript:',
            r'data:',
            r'file:',
            r'&[^;]*;',  # Entity references (except standard ones)
        ]
        
        xml_lower = xml_string.lower()
        
        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if re.search(pattern, xml_lower):
                logger.warning(f"Dangerous XML pattern detected: {pattern}")
                return False
        
        # Validate XML structure
        try:
            ET.fromstring(xml_string)
        except ET.ParseError:
            logger.warning("Generated XML is malformed")
            return False
        
        # Check size limits
        if len(xml_string) > 1000000:  # 1MB limit
            logger.warning("Generated XML exceeds size limit")
            return False
        
        return True
    
    def _format_xml(self, xml_string: str) -> str:
        """Format XML string for readability."""
        try:
            import xml.dom.minidom
            dom = xml.dom.minidom.parseString(xml_string)
            return dom.toprettyxml(indent="  ")
        except Exception:
            # Fallback to unformatted XML if pretty printing fails
            return xml_string
    
    def clear(self) -> 'ActionBuilder':
        """Clear all actions from builder."""
        self.actions.clear()
        return self
    
    def get_action_count(self) -> int:
        """Get number of actions in builder."""
        return len(self.actions)
    
    def get_actions(self) -> List[ActionConfiguration]:
        """Get copy of actions list."""
        return self.actions.copy()
    
    def remove_action(self, index: int) -> 'ActionBuilder':
        """Remove action at specific index."""
        if 0 <= index < len(self.actions):
            del self.actions[index]
        return self
    
    def validate_all(self) -> Dict[str, Any]:
        """Validate all actions in the builder."""
        validation_results = []
        
        for i, action in enumerate(self.actions):
            result = {
                "index": i,
                "action_type": action.action_type.identifier,
                "valid": action.validate_parameters(),
                "issues": []
            }
            
            if not result["valid"]:
                # Check specific validation issues
                missing_params = [
                    p for p in action.action_type.required_params 
                    if p not in action.parameters
                ]
                if missing_params:
                    result["issues"].append(f"Missing required parameters: {missing_params}")
                
                if not action._validate_parameter_security():
                    result["issues"].append("Security validation failed")
            
            validation_results.append(result)
        
        all_valid = all(r["valid"] for r in validation_results)
        
        return {
            "all_valid": all_valid,
            "total_actions": len(self.actions),
            "valid_actions": sum(1 for r in validation_results if r["valid"]),
            "results": validation_results
        }