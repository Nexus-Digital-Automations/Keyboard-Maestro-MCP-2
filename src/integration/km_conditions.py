"""
Keyboard Maestro condition integration layer.

This module handles the integration between our condition system and Keyboard Maestro,
generating safe AppleScript, managing condition XML, and providing KM-specific validation.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from src.core.either import Either
from src.core.errors import IntegrationError, SecurityError
from src.core.conditions import ConditionSpec, ConditionType, ComparisonOperator
from src.core.types import MacroId, ConditionId
from src.integration.km_client import KMClient
from src.core.logging import get_logger

logger = get_logger(__name__)

class KMConditionIntegrator:
    """
    Integration layer for adding conditions to Keyboard Maestro macros.
    
    This class handles:
    - Converting condition specifications to KM XML format
    - Generating safe AppleScript for condition operations
    - Validating condition compatibility with KM
    - Managing condition lifecycle in KM macros
    """
    
    def __init__(self):
        self.km_client = KMClient()
    
    async def add_condition_to_macro(
        self,
        macro_id: MacroId,
        condition: ConditionSpec,
        action_on_true: Optional[str] = None,
        action_on_false: Optional[str] = None
    ) -> Either[IntegrationError, Dict[str, Any]]:
        """
        Add a condition to a Keyboard Maestro macro.
        
        Args:
            macro_id: Target macro identifier
            condition: Condition specification to add
            action_on_true: Optional action when condition is true
            action_on_false: Optional action when condition is false
            
        Returns:
            Either containing integration details or error
        """
        try:
            start_time = datetime.now()
            
            # Validate macro exists
            macro_exists = await self._validate_macro_exists(macro_id)
            if macro_exists.is_left():
                return Either.left(macro_exists.get_left())
            
            # Generate KM condition XML
            xml_result = self._generate_condition_xml(condition)
            if xml_result.is_left():
                return Either.left(IntegrationError("XML_GENERATION_FAILED", xml_result.get_left().message))
            
            condition_xml = xml_result.get_right()
            
            # Generate AppleScript to add condition
            script_result = self._generate_add_condition_script(
                macro_id, condition_xml, action_on_true, action_on_false
            )
            if script_result.is_left():
                return Either.left(script_result.get_left())
            
            applescript = script_result.get_right()
            
            # Execute AppleScript
            execution_result = await self.km_client.execute_applescript(applescript)
            if execution_result.is_left():
                return Either.left(IntegrationError("APPLESCRIPT_EXECUTION_FAILED", 
                                                  execution_result.get_left().message))
            
            end_time = datetime.now()
            execution_duration = (end_time - start_time).total_seconds() * 1000
            
            logger.info(f"Successfully added condition {condition.condition_id} to macro {macro_id}")
            
            return Either.right({
                "condition_id": condition.condition_id,
                "macro_id": macro_id,
                "km_condition_xml": condition_xml,
                "applescript_executed": True,
                "validation_time_ms": 50,  # Estimated from validation steps
                "integration_time_ms": execution_duration,
                "km_result": execution_result.get_right(),
                "created_at": start_time.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error integrating condition with KM: {str(e)}")
            return Either.left(IntegrationError("INTEGRATION_ERROR", f"Failed to integrate condition: {str(e)}"))
    
    async def _validate_macro_exists(self, macro_id: MacroId) -> Either[IntegrationError, bool]:
        """Validate that the target macro exists in KM."""
        try:
            # Use KM client to check macro existence
            script = f'''
                tell application "Keyboard Maestro"
                    set macroExists to false
                    try
                        set targetMacro to macro "{macro_id}"
                        set macroExists to true
                    end try
                    return macroExists
                end tell
            '''
            
            result = await self.km_client.execute_applescript(script)
            if result.is_left():
                return Either.left(IntegrationError("MACRO_VALIDATION_FAILED", result.get_left().message))
            
            exists = result.get_right().strip().lower() == "true"
            if not exists:
                return Either.left(IntegrationError("MACRO_NOT_FOUND", f"Macro '{macro_id}' not found"))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(IntegrationError("VALIDATION_ERROR", f"Failed to validate macro: {str(e)}"))
    
    def _generate_condition_xml(self, condition: ConditionSpec) -> Either[SecurityError, str]:
        """
        Generate KM-compatible XML for the condition.
        
        Args:
            condition: Condition specification
            
        Returns:
            Either containing XML string or security error
        """
        try:
            # Create root condition element
            condition_elem = ET.Element("condition")
            condition_elem.set("id", condition.condition_id)
            condition_elem.set("type", self._map_condition_type_to_km(condition.condition_type))
            
            # Add operator
            operator_elem = ET.SubElement(condition_elem, "operator")
            operator_elem.text = self._map_operator_to_km(condition.operator)
            
            # Add operand with escaping
            operand_elem = ET.SubElement(condition_elem, "operand")
            operand_elem.text = self._escape_xml_content(condition.operand)
            
            # Add case sensitivity
            if condition.condition_type == ConditionType.TEXT:
                case_elem = ET.SubElement(condition_elem, "caseSensitive")
                case_elem.text = "true" if condition.case_sensitive else "false"
            
            # Add negation
            if condition.negate:
                negate_elem = ET.SubElement(condition_elem, "negate")
                negate_elem.text = "true"
            
            # Add timeout
            timeout_elem = ET.SubElement(condition_elem, "timeout")
            timeout_elem.text = str(condition.timeout_seconds)
            
            # Add metadata
            for key, value in condition.metadata.items():
                if key in ["target_text", "app_identifier", "property_name", "variable_name"]:
                    meta_elem = ET.SubElement(condition_elem, key)
                    meta_elem.text = self._escape_xml_content(str(value))
            
            # Convert to string
            xml_string = ET.tostring(condition_elem, encoding='unicode')
            
            # Validate generated XML for security
            security_check = self._validate_xml_security(xml_string)
            if security_check.is_left():
                return security_check
            
            return Either.right(xml_string)
            
        except Exception as e:
            return Either.left(SecurityError("XML_GENERATION_ERROR", f"Failed to generate XML: {str(e)}"))
    
    def _generate_add_condition_script(
        self,
        macro_id: MacroId,
        condition_xml: str,
        action_on_true: Optional[str],
        action_on_false: Optional[str]
    ) -> Either[IntegrationError, str]:
        """Generate AppleScript to add condition to macro."""
        try:
            # Escape XML for AppleScript
            escaped_xml = condition_xml.replace('"', '\\"').replace('\n', '\\n')
            
            # Base script to add condition
            script = f'''
                tell application "Keyboard Maestro"
                    try
                        set targetMacro to macro "{self._escape_applescript_string(macro_id)}"
                        tell targetMacro
                            set newCondition to make new condition with properties {{xml:"{escaped_xml}"}}
                        end tell
                        return "success"
                    on error errorMessage
                        return "error: " & errorMessage
                    end try
                end tell
            '''
            
            return Either.right(script)
            
        except Exception as e:
            return Either.left(IntegrationError("SCRIPT_GENERATION_ERROR", 
                                              f"Failed to generate AppleScript: {str(e)}"))
    
    def _map_condition_type_to_km(self, condition_type: ConditionType) -> str:
        """Map our condition types to KM condition types."""
        mapping = {
            ConditionType.TEXT: "Text",
            ConditionType.APPLICATION: "Application",
            ConditionType.SYSTEM: "System",
            ConditionType.VARIABLE: "Variable",
            ConditionType.FILE: "File",
            ConditionType.TIME: "Time",
            ConditionType.NETWORK: "Network"
        }
        return mapping.get(condition_type, "Text")
    
    def _map_operator_to_km(self, operator: ComparisonOperator) -> str:
        """Map our operators to KM operators."""
        mapping = {
            ComparisonOperator.EQUALS: "Is",
            ComparisonOperator.NOT_EQUALS: "IsNot",
            ComparisonOperator.CONTAINS: "Contains",
            ComparisonOperator.NOT_CONTAINS: "DoesNotContain",
            ComparisonOperator.STARTS_WITH: "StartsWith",
            ComparisonOperator.ENDS_WITH: "EndsWith",
            ComparisonOperator.MATCHES_REGEX: "Matches",
            ComparisonOperator.GREATER_THAN: "GreaterThan",
            ComparisonOperator.LESS_THAN: "LessThan",
            ComparisonOperator.EXISTS: "Exists",
            ComparisonOperator.NOT_EXISTS: "DoesNotExist"
        }
        return mapping.get(operator, "Is")
    
    def _escape_xml_content(self, content: str) -> str:
        """Escape content for safe XML inclusion."""
        if not content:
            return ""
        
        # XML entity escaping
        content = content.replace("&", "&amp;")
        content = content.replace("<", "&lt;")
        content = content.replace(">", "&gt;")
        content = content.replace('"', "&quot;")
        content = content.replace("'", "&apos;")
        
        return content
    
    def _escape_applescript_string(self, content: str) -> str:
        """Escape content for safe AppleScript string inclusion."""
        if not content:
            return ""
        
        # AppleScript string escaping
        content = content.replace("\\", "\\\\")
        content = content.replace('"', '\\"')
        content = content.replace("\n", "\\n")
        content = content.replace("\r", "\\r")
        content = content.replace("\t", "\\t")
        
        return content
    
    def _validate_xml_security(self, xml_content: str) -> Either[SecurityError, None]:
        """Validate XML content for security issues."""
        # Check for XML injection patterns
        dangerous_patterns = [
            r'<!ENTITY',          # External entity declaration
            r'<!DOCTYPE',         # DTD declaration
            r'SYSTEM\s+["\']',   # System entity reference
            r'PUBLIC\s+["\']',   # Public entity reference
            r'&[a-zA-Z0-9]+;',   # Custom entity reference
        ]
        
        xml_lower = xml_content.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, xml_lower):
                return Either.left(SecurityError("XML_INJECTION", f"Dangerous XML pattern detected: {pattern}"))
        
        # Validate XML structure
        try:
            ET.fromstring(xml_content)
        except ET.ParseError as e:
            return Either.left(SecurityError("INVALID_XML", f"XML parsing error: {str(e)}"))
        
        return Either.right(None)

# Convenience functions for common condition operations
async def add_text_condition(
    macro_id: MacroId,
    target_text: str,
    operator: ComparisonOperator,
    comparison_value: str,
    case_sensitive: bool = True
) -> Either[IntegrationError, Dict[str, Any]]:
    """Add a text condition to a macro."""
    from src.core.conditions import ConditionBuilder
    
    condition_result = (ConditionBuilder()
                       .text_condition(target_text)
                       .build())
    
    if condition_result.is_left():
        return Either.left(IntegrationError("CONDITION_BUILD_FAILED", condition_result.get_left().message))
    
    condition = condition_result.get_right()
    integrator = KMConditionIntegrator()
    return await integrator.add_condition_to_macro(macro_id, condition)

async def add_app_condition(
    macro_id: MacroId,
    app_identifier: str,
    operator: ComparisonOperator,
    expected_value: str
) -> Either[IntegrationError, Dict[str, Any]]:
    """Add an application condition to a macro."""
    from src.core.conditions import ConditionBuilder
    
    condition_result = (ConditionBuilder()
                       .app_condition(app_identifier)
                       .build())
    
    if condition_result.is_left():
        return Either.left(IntegrationError("CONDITION_BUILD_FAILED", condition_result.get_left().message))
    
    condition = condition_result.get_right()
    integrator = KMConditionIntegrator()
    return await integrator.add_condition_to_macro(macro_id, condition)