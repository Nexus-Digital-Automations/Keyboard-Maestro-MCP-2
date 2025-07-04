"""
Keyboard Maestro control flow integration.

Handles the generation of safe AppleScript and XML for Keyboard Maestro
control flow structures with comprehensive security validation.
"""

from typing import Dict, Any, List, Optional, Union
import xml.etree.ElementTree as ET
import re
import uuid
from datetime import datetime

from ..core.control_flow import (
    ControlFlowNodeType, IfThenElseNode, ForLoopNode, WhileLoopNode,
    SwitchCaseNode, TryCatchNode, ComparisonOperator, ActionBlock
)
from ..core.types import MacroId
from ..core.contracts import require, ensure
from ..core.errors import SecurityError, ValidationError


class KMControlFlowGenerator:
    """Generate Keyboard Maestro compatible control flow structures."""
    
    def __init__(self):
        """Initialize generator with security settings."""
        self.dangerous_patterns = [
            'tell application "Terminal"',
            'do shell script',
            'exec',
            'eval',
            'import',
            'subprocess',
            'rm ',
            'del ',
            'format',
            '`',
            'curl',
            'wget'
        ]
    
    def generate_control_flow_xml(
        self,
        node: ControlFlowNodeType,
        macro_id: str
    ) -> str:
        """
        Generate secure XML for Keyboard Maestro control flow structures.
        
        Creates properly formatted XML that Keyboard Maestro can import and execute,
        with comprehensive security validation and injection prevention.
        """
        try:
            if isinstance(node, IfThenElseNode):
                return self._generate_if_then_else_xml(node, macro_id)
            elif isinstance(node, ForLoopNode):
                return self._generate_for_loop_xml(node, macro_id)
            elif isinstance(node, WhileLoopNode):
                return self._generate_while_loop_xml(node, macro_id)
            elif isinstance(node, SwitchCaseNode):
                return self._generate_switch_case_xml(node, macro_id)
            elif isinstance(node, TryCatchNode):
                return self._generate_try_catch_xml(node, macro_id)
            else:
                raise ValidationError(f"Unsupported control flow node type: {type(node)}")
                
        except Exception as e:
            raise ValidationError(f"Failed to generate XML: {e}")
    
    def _generate_if_then_else_xml(self, node: IfThenElseNode, macro_id: str) -> str:
        """Generate XML for If/Then/Else structure."""
        action_id = str(uuid.uuid4()).upper()
        
        # Validate and escape condition
        condition_xml = self._generate_condition_xml(node.condition)
        
        # Generate actions XML
        then_actions_xml = self._generate_actions_xml(node.then_actions)
        else_actions_xml = ""
        
        if node.else_actions:
            else_actions_xml = self._generate_actions_xml(node.else_actions)
        
        xml_template = f"""
        <action id="{action_id}" name="If Then Else">
            <condition>
                {condition_xml}
            </condition>
            <then>
                {then_actions_xml}
            </then>
            <else>
                {else_actions_xml}
            </else>
        </action>
        """
        
        return self._validate_and_clean_xml(xml_template)
    
    def _generate_for_loop_xml(self, node: ForLoopNode, macro_id: str) -> str:
        """Generate XML for For Loop structure."""
        action_id = str(uuid.uuid4()).upper()
        
        # Validate iterator and collection
        iterator = self._escape_xml(node.loop_config.iterator_variable)
        collection = self._escape_xml(node.loop_config.collection_expression)
        
        # Validate max iterations
        max_iter = min(node.loop_config.max_iterations, 10000)
        
        # Generate loop actions
        loop_actions_xml = self._generate_actions_xml(node.loop_actions)
        
        xml_template = f"""
        <action id="{action_id}" name="For Each">
            <iterator>{iterator}</iterator>
            <collection>{collection}</collection>
            <maxIterations>{max_iter}</maxIterations>
            <timeout>{node.loop_config.timeout_seconds}</timeout>
            <breakOnError>{str(node.loop_config.break_on_error).lower()}</breakOnError>
            <actions>
                {loop_actions_xml}
            </actions>
        </action>
        """
        
        return self._validate_and_clean_xml(xml_template)
    
    def _generate_while_loop_xml(self, node: WhileLoopNode, macro_id: str) -> str:
        """Generate XML for While Loop structure."""
        action_id = str(uuid.uuid4()).upper()
        
        # Generate condition XML
        condition_xml = self._generate_condition_xml(node.condition)
        
        # Validate max iterations
        max_iter = min(node.max_iterations, 10000)
        
        # Generate loop actions
        loop_actions_xml = self._generate_actions_xml(node.loop_actions)
        
        xml_template = f"""
        <action id="{action_id}" name="While">
            <condition>
                {condition_xml}
            </condition>
            <maxIterations>{max_iter}</maxIterations>
            <actions>
                {loop_actions_xml}
            </actions>
        </action>
        """
        
        return self._validate_and_clean_xml(xml_template)
    
    def _generate_switch_case_xml(self, node: SwitchCaseNode, macro_id: str) -> str:
        """Generate XML for Switch/Case structure."""
        action_id = str(uuid.uuid4()).upper()
        
        # Validate switch variable
        switch_var = self._escape_xml(node.switch_variable)
        
        # Generate cases XML
        cases_xml = ""
        for case in node.cases:
            case_value = self._escape_xml(case.case_value)
            case_actions = self._generate_actions_xml(case.actions)
            
            cases_xml += f"""
            <case value="{case_value}" id="{case.case_id}">
                {case_actions}
            </case>
            """
        
        # Generate default case if present
        default_xml = ""
        if node.default_case:
            default_actions = self._generate_actions_xml(node.default_case)
            default_xml = f"""
            <default>
                {default_actions}
            </default>
            """
        
        xml_template = f"""
        <action id="{action_id}" name="Switch">
            <variable>{switch_var}</variable>
            <cases>
                {cases_xml}
            </cases>
            {default_xml}
        </action>
        """
        
        return self._validate_and_clean_xml(xml_template)
    
    def _generate_try_catch_xml(self, node: TryCatchNode, macro_id: str) -> str:
        """Generate XML for Try/Catch structure."""
        action_id = str(uuid.uuid4()).upper()
        
        # Generate actions XML
        try_actions_xml = self._generate_actions_xml(node.try_actions)
        catch_actions_xml = self._generate_actions_xml(node.catch_actions)
        
        finally_xml = ""
        if node.finally_actions:
            finally_actions_xml = self._generate_actions_xml(node.finally_actions)
            finally_xml = f"""
            <finally>
                {finally_actions_xml}
            </finally>
            """
        
        xml_template = f"""
        <action id="{action_id}" name="Try Catch">
            <try>
                {try_actions_xml}
            </try>
            <catch>
                {catch_actions_xml}
            </catch>
            {finally_xml}
        </action>
        """
        
        return self._validate_and_clean_xml(xml_template)
    
    def _generate_condition_xml(self, condition) -> str:
        """Generate secure XML for condition expressions."""
        # Escape and validate condition components
        expression = self._escape_xml(condition.expression)
        operand = self._escape_xml(condition.operand)
        
        # Validate for dangerous patterns
        self._validate_condition_security(condition.expression)
        self._validate_condition_security(condition.operand)
        
        # Map operator to KM format
        km_operator = self._map_operator_to_km(condition.operator)
        
        condition_xml = f"""
        <expression>{expression}</expression>
        <operator>{km_operator}</operator>
        <operand>{operand}</operand>
        <caseSensitive>{str(condition.case_sensitive).lower()}</caseSensitive>
        <negate>{str(condition.negate).lower()}</negate>
        <timeout>{condition.timeout_seconds}</timeout>
        """
        
        return condition_xml
    
    def _generate_actions_xml(self, action_block: ActionBlock) -> str:
        """Generate secure XML for action blocks."""
        actions_xml = ""
        
        for action in action_block.actions:
            # Validate action structure
            if not isinstance(action, dict) or 'type' not in action:
                continue
            
            action_type = self._escape_xml(action['type'])
            action_id = str(uuid.uuid4()).upper()
            
            # Generate parameters XML
            params_xml = ""
            for key, value in action.items():
                if key != 'type':
                    safe_key = self._escape_xml(str(key))
                    safe_value = self._escape_xml(str(value))
                    
                    # Validate parameter content
                    self._validate_action_parameter(safe_value)
                    
                    params_xml += f'<{safe_key}>{safe_value}</{safe_key}>'
            
            actions_xml += f"""
            <action id="{action_id}" type="{action_type}">
                {params_xml}
            </action>
            """
        
        return actions_xml
    
    def _map_operator_to_km(self, operator: ComparisonOperator) -> str:
        """Map internal operators to Keyboard Maestro format."""
        operator_map = {
            ComparisonOperator.EQUALS: "Is",
            ComparisonOperator.NOT_EQUALS: "IsNot",
            ComparisonOperator.GREATER_THAN: "IsGreaterThan",
            ComparisonOperator.LESS_THAN: "IsLessThan",
            ComparisonOperator.GREATER_EQUAL: "IsGreaterThanOrEqualTo",
            ComparisonOperator.LESS_EQUAL: "IsLessThanOrEqualTo",
            ComparisonOperator.CONTAINS: "Contains",
            ComparisonOperator.NOT_CONTAINS: "DoesNotContain",
            ComparisonOperator.MATCHES_REGEX: "MatchesRegularExpression",
            ComparisonOperator.EXISTS: "Exists"
        }
        
        return operator_map.get(operator, "Is")
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters and validate content."""
        if not isinstance(text, str):
            text = str(text)
        
        # Basic XML escaping
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&apos;')
        
        return text
    
    def _validate_condition_security(self, text: str) -> None:
        """Validate condition text for security threats."""
        if not isinstance(text, str):
            return
        
        text_lower = text.lower()
        
        for pattern in self.dangerous_patterns:
            if pattern in text_lower:
                raise SecurityError(f"Dangerous pattern detected: {pattern}")
        
        # Check for excessive length
        if len(text) > 1000:
            raise SecurityError("Condition text too long")
        
        # Check for suspicious characters
        suspicious_chars = ['`', '$', '\\', '|', ';', '&&', '||']
        for char in suspicious_chars:
            if char in text:
                raise SecurityError(f"Suspicious character detected: {char}")
    
    def _validate_action_parameter(self, value: str) -> None:
        """Validate action parameter for security."""
        if not isinstance(value, str):
            return
        
        value_lower = value.lower()
        
        for pattern in self.dangerous_patterns:
            if pattern in value_lower:
                raise SecurityError(f"Dangerous pattern in action parameter: {pattern}")
        
        # Limit parameter size
        if len(value) > 5000:
            raise SecurityError("Action parameter too long")
    
    def _validate_and_clean_xml(self, xml_string: str) -> str:
        """Validate and clean generated XML."""
        try:
            # Remove extra whitespace and newlines
            xml_string = re.sub(r'\s+', ' ', xml_string.strip())
            
            # Basic XML validation by parsing
            ET.fromstring(f"<root>{xml_string}</root>")
            
            return xml_string
            
        except ET.ParseError as e:
            raise ValidationError(f"Generated XML is invalid: {e}")


class KMAppleScriptGenerator:
    """Generate AppleScript for Keyboard Maestro control flow operations."""
    
    def __init__(self):
        """Initialize AppleScript generator."""
        self.dangerous_patterns = [
            'do shell script',
            'exec',
            'system',
            'rm ',
            'del ',
            'format',
            '`'
        ]
    
    def generate_control_flow_applescript(
        self,
        node: ControlFlowNodeType,
        macro_id: str
    ) -> str:
        """Generate AppleScript to add control flow to macro."""
        
        # Generate the XML representation
        generator = KMControlFlowGenerator()
        xml_content = generator.generate_control_flow_xml(node, macro_id)
        
        # Escape XML for AppleScript
        escaped_xml = self._escape_applescript_string(xml_content)
        
        applescript = f'''
        tell application "Keyboard Maestro"
            tell macro id "{macro_id}"
                make new action with properties {{xml:"{escaped_xml}"}}
            end tell
        end tell
        '''
        
        # Validate AppleScript for security
        self._validate_applescript_security(applescript)
        
        return applescript.strip()
    
    def _escape_applescript_string(self, text: str) -> str:
        """Escape string for safe AppleScript inclusion."""
        # Escape quotes and backslashes
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        
        # Remove potentially dangerous characters
        text = re.sub(r'[`$]', '', text)
        
        return text
    
    def _validate_applescript_security(self, script: str) -> None:
        """Validate AppleScript for security threats."""
        script_lower = script.lower()
        
        for pattern in self.dangerous_patterns:
            if pattern in script_lower:
                raise SecurityError(f"Dangerous AppleScript pattern: {pattern}")
        
        # Check script length
        if len(script) > 10000:
            raise SecurityError("AppleScript too long")


# Public interface functions
def generate_km_control_flow_xml(node: ControlFlowNodeType, macro_id: str) -> str:
    """Generate Keyboard Maestro XML for control flow node."""
    generator = KMControlFlowGenerator()
    return generator.generate_control_flow_xml(node, macro_id)


def generate_km_control_flow_applescript(node: ControlFlowNodeType, macro_id: str) -> str:
    """Generate AppleScript to add control flow to macro."""
    generator = KMAppleScriptGenerator()
    return generator.generate_control_flow_applescript(node, macro_id)


def validate_control_flow_security(node: ControlFlowNodeType) -> bool:
    """Validate control flow node for security compliance."""
    try:
        generator = KMControlFlowGenerator()
        xml_content = generator.generate_control_flow_xml(node, "test_macro")
        return True
    except (SecurityError, ValidationError):
        return False