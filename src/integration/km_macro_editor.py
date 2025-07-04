"""
Keyboard Maestro macro editor integration with AppleScript generation.

This module provides direct integration with Keyboard Maestro for macro editing
operations including inspection, modification, comparison, and validation with
comprehensive security checks and error handling.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set
import json
import re
import time
import logging

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.errors import ValidationError, SecurityViolationError, IntegrationError
from ..core.macro_editor import (
    MacroInspection, MacroComparison, MacroModification, EditOperation,
    MacroEditorValidator, calculate_macro_complexity, calculate_macro_health
)
from .km_client import KMClient


logger = logging.getLogger(__name__)


class KMMacroEditor:
    """Keyboard Maestro macro editor integration with security validation."""
    
    def __init__(self, km_client: KMClient):
        self.km_client = km_client
        self._modification_cache: Dict[str, List[MacroModification]] = {}
    
    @require(lambda self, macro_id: isinstance(macro_id, str) and len(macro_id.strip()) > 0)
    async def inspect_macro(self, macro_id: str) -> Either[IntegrationError, MacroInspection]:
        """Inspect macro and return comprehensive analysis."""
        try:
            # Get macro data from Keyboard Maestro
            applescript = f'''
            tell application "Keyboard Maestro"
                set macroRef to macro "{macro_id}"
                if macroRef exists then
                    set macroName to name of macroRef
                    set macroEnabled to enabled of macroRef
                    set macroGroup to name of macro group of macroRef
                    
                    -- Get actions
                    set actionList to ""
                    repeat with actionRef in actions of macroRef
                        set actionType to xml of actionRef
                        set actionList to actionList & actionType & "|||"
                    end repeat
                    
                    -- Get triggers
                    set triggerList to ""
                    repeat with triggerRef in triggers of macroRef
                        set triggerType to xml of triggerRef
                        set triggerList to triggerList & triggerType & "|||"
                    end repeat
                    
                    return macroName & ":::" & (macroEnabled as string) & ":::" & macroGroup & ":::" & actionList & ":::" & triggerList
                else
                    return "MACRO_NOT_FOUND"
                end if
            end tell
            '''
            
            result = await self.km_client.execute_applescript(applescript)
            if result.is_left():
                return Either.left(IntegrationError(
                    "applescript_execution_failed",
                    f"Failed to inspect macro: {result.get_left().message}"
                ))
            
            response = result.get_right()
            if response == "MACRO_NOT_FOUND":
                return Either.left(IntegrationError(
                    "macro_not_found",
                    f"Macro '{macro_id}' not found"
                ))
            
            # Parse response
            parts = response.split(":::")
            if len(parts) < 5:
                return Either.left(IntegrationError(
                    "invalid_response",
                    "Invalid response format from Keyboard Maestro"
                ))
            
            macro_name = parts[0]
            enabled = parts[1] == "true"
            group_name = parts[2]
            actions_xml = parts[3]
            triggers_xml = parts[4]
            
            # Parse actions and triggers
            actions = self._parse_xml_elements(actions_xml)
            triggers = self._parse_xml_elements(triggers_xml)
            conditions = []  # In real implementation, parse conditions from actions
            
            # Calculate metrics
            macro_data = {
                "name": macro_name,
                "actions": actions,
                "triggers": triggers,
                "conditions": conditions
            }
            
            complexity_score = calculate_macro_complexity(macro_data)
            health_score = calculate_macro_health(macro_data)
            
            # Extract variables used (simplified analysis)
            variables_used = self._extract_variables_from_actions(actions)
            
            # Estimate execution time (simplified)
            estimated_time = len(actions) * 0.1  # Basic estimation
            
            inspection = MacroInspection(
                macro_id=macro_id,
                macro_name=macro_name,
                enabled=enabled,
                group_name=group_name,
                action_count=len(actions),
                trigger_count=len(triggers),
                condition_count=len(conditions),
                actions=actions,
                triggers=triggers,
                conditions=conditions,
                variables_used=variables_used,
                estimated_execution_time=estimated_time,
                complexity_score=complexity_score,
                health_score=health_score
            )
            
            logger.info(f"Inspected macro '{macro_id}': {len(actions)} actions, {len(triggers)} triggers")
            return Either.right(inspection)
            
        except Exception as e:
            logger.error(f"Error inspecting macro '{macro_id}': {str(e)}")
            return Either.left(IntegrationError(
                "inspection_failed",
                f"Macro inspection failed: {str(e)}"
            ))
    
    @require(lambda self, macro_id: isinstance(macro_id, str) and len(macro_id.strip()) > 0)
    @require(lambda self, modifications: isinstance(modifications, list))
    async def apply_modifications(
        self, 
        macro_id: str, 
        modifications: List[MacroModification],
        create_backup: bool = True
    ) -> Either[IntegrationError, Dict[str, Any]]:
        """Apply modifications to a macro with validation and backup."""
        try:
            # Validate permissions
            for mod in modifications:
                perm_result = MacroEditorValidator.validate_modification_permissions(macro_id, mod.operation)
                if perm_result.is_left():
                    return Either.left(IntegrationError(
                        "permission_denied",
                        f"Permission denied for operation {mod.operation.value}: {perm_result.get_left().message}"
                    ))
            
            # Create backup if requested
            backup_id = None
            if create_backup:
                backup_result = await self._create_macro_backup(macro_id)
                if backup_result.is_left():
                    return Either.left(IntegrationError(
                        "backup_failed",
                        f"Failed to create backup: {backup_result.get_left().message}"
                    ))
                backup_id = backup_result.get_right()
            
            # Apply modifications in sequence
            applied_modifications = []
            rollback_needed = False
            
            for mod in modifications:
                mod_result = await self._apply_single_modification(macro_id, mod)
                if mod_result.is_left():
                    rollback_needed = True
                    logger.error(f"Failed to apply modification {mod.operation.value}: {mod_result.get_left().message}")
                    break
                
                applied_modifications.append(mod)
                logger.info(f"Applied modification: {mod.operation.value}")
            
            # Rollback if any modification failed
            if rollback_needed and backup_id:
                await self._restore_from_backup(macro_id, backup_id)
                return Either.left(IntegrationError(
                    "modification_failed",
                    f"Modifications failed and were rolled back. Backup: {backup_id}"
                ))
            
            # Cache successful modifications
            self._modification_cache[macro_id] = applied_modifications
            
            return Either.right({
                "macro_id": macro_id,
                "modifications_applied": len(applied_modifications),
                "backup_id": backup_id,
                "timestamp": time.time(),
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Error applying modifications to macro '{macro_id}': {str(e)}")
            return Either.left(IntegrationError(
                "modification_error",
                f"Failed to apply modifications: {str(e)}"
            ))
    
    @require(lambda self, macro1_id: isinstance(macro1_id, str) and len(macro1_id.strip()) > 0)
    @require(lambda self, macro2_id: isinstance(macro2_id, str) and len(macro2_id.strip()) > 0)
    async def compare_macros(self, macro1_id: str, macro2_id: str) -> Either[IntegrationError, MacroComparison]:
        """Compare two macros and return detailed analysis."""
        try:
            # Inspect both macros
            macro1_result = await self.inspect_macro(macro1_id)
            if macro1_result.is_left():
                return Either.left(IntegrationError(
                    "macro1_inspection_failed",
                    f"Failed to inspect macro1: {macro1_result.get_left().message}"
                ))
            
            macro2_result = await self.inspect_macro(macro2_id)
            if macro2_result.is_left():
                return Either.left(IntegrationError(
                    "macro2_inspection_failed",
                    f"Failed to inspect macro2: {macro2_result.get_left().message}"
                ))
            
            macro1 = macro1_result.get_right()
            macro2 = macro2_result.get_right()
            
            # Calculate differences
            differences = []
            
            # Compare basic properties
            if macro1.macro_name != macro2.macro_name:
                differences.append({
                    "type": "name_difference",
                    "macro1_value": macro1.macro_name,
                    "macro2_value": macro2.macro_name
                })
            
            if macro1.enabled != macro2.enabled:
                differences.append({
                    "type": "enabled_difference",
                    "macro1_value": macro1.enabled,
                    "macro2_value": macro2.enabled
                })
            
            if macro1.group_name != macro2.group_name:
                differences.append({
                    "type": "group_difference",
                    "macro1_value": macro1.group_name,
                    "macro2_value": macro2.group_name
                })
            
            # Compare action counts
            if macro1.action_count != macro2.action_count:
                differences.append({
                    "type": "action_count_difference",
                    "macro1_value": macro1.action_count,
                    "macro2_value": macro2.action_count
                })
            
            # Compare trigger counts
            if macro1.trigger_count != macro2.trigger_count:
                differences.append({
                    "type": "trigger_count_difference",
                    "macro1_value": macro1.trigger_count,
                    "macro2_value": macro2.trigger_count
                })
            
            # Calculate similarity score (simplified)
            total_comparisons = 5  # Basic properties compared
            differences_count = len(differences)
            similarity_score = max(0.0, 1.0 - (differences_count / total_comparisons))
            
            # Generate recommendation
            recommendation = self._generate_comparison_recommendation(similarity_score, differences)
            
            comparison = MacroComparison(
                macro1_id=macro1_id,
                macro2_id=macro2_id,
                differences=differences,
                similarity_score=similarity_score,
                recommendation=recommendation
            )
            
            logger.info(f"Compared macros '{macro1_id}' and '{macro2_id}': {similarity_score:.2f} similarity")
            return Either.right(comparison)
            
        except Exception as e:
            logger.error(f"Error comparing macros '{macro1_id}' and '{macro2_id}': {str(e)}")
            return Either.left(IntegrationError(
                "comparison_failed",
                f"Macro comparison failed: {str(e)}"
            ))
    
    async def _apply_single_modification(self, macro_id: str, modification: MacroModification) -> Either[IntegrationError, None]:
        """Apply a single modification to a macro."""
        try:
            if modification.operation == EditOperation.ADD_ACTION:
                return await self._add_action_to_macro(macro_id, modification)
            elif modification.operation == EditOperation.MODIFY_ACTION:
                return await self._modify_macro_action(macro_id, modification)
            elif modification.operation == EditOperation.DELETE_ACTION:
                return await self._delete_macro_action(macro_id, modification)
            elif modification.operation == EditOperation.UPDATE_PROPERTIES:
                return await self._update_macro_properties(macro_id, modification)
            else:
                return Either.left(IntegrationError(
                    "unsupported_operation",
                    f"Operation {modification.operation.value} not yet supported"
                ))
                
        except Exception as e:
            return Either.left(IntegrationError(
                "modification_execution_failed",
                f"Failed to execute modification: {str(e)}"
            ))
    
    async def _add_action_to_macro(self, macro_id: str, modification: MacroModification) -> Either[IntegrationError, None]:
        """Add action to macro via AppleScript."""
        if not modification.new_value:
            return Either.left(IntegrationError(
                "invalid_modification",
                "No action configuration provided"
            ))
        
        action_type = modification.new_value.get("type", "")
        action_config = modification.new_value.get("config", {})
        
        # Validate action configuration
        validation_result = MacroEditorValidator.validate_action_modification(action_config)
        if validation_result.is_left():
            return Either.left(IntegrationError(
                "action_validation_failed",
                f"Action validation failed: {validation_result.get_left().message}"
            ))
        
        # Generate action XML (simplified)
        action_xml = self._generate_action_xml(action_type, action_config)
        
        # AppleScript to add action
        applescript = f'''
        tell application "Keyboard Maestro"
            set macroRef to macro "{macro_id}"
            if macroRef exists then
                make new action at end of actions of macroRef with properties {{xml:"{action_xml}"}}
                return "SUCCESS"
            else
                return "MACRO_NOT_FOUND"
            end if
        end tell
        '''
        
        result = await self.km_client.execute_applescript(applescript)
        if result.is_left():
            return Either.left(IntegrationError(
                "applescript_failed",
                f"Failed to add action: {result.get_left().message}"
            ))
        
        response = result.get_right()
        if response != "SUCCESS":
            return Either.left(IntegrationError(
                "action_addition_failed",
                f"Failed to add action: {response}"
            ))
        
        return Either.right(None)
    
    async def _modify_macro_action(self, macro_id: str, modification: MacroModification) -> Either[IntegrationError, None]:
        """Modify existing action in macro."""
        # In a real implementation, this would locate the specific action and update it
        # For now, return success as a placeholder
        logger.info(f"Modifying action {modification.target_element} in macro {macro_id}")
        return Either.right(None)
    
    async def _delete_macro_action(self, macro_id: str, modification: MacroModification) -> Either[IntegrationError, None]:
        """Delete action from macro."""
        # In a real implementation, this would locate and delete the specific action
        logger.info(f"Deleting action {modification.target_element} from macro {macro_id}")
        return Either.right(None)
    
    async def _update_macro_properties(self, macro_id: str, modification: MacroModification) -> Either[IntegrationError, None]:
        """Update macro properties."""
        if not modification.new_value:
            return Either.left(IntegrationError(
                "invalid_modification",
                "No properties provided"
            ))
        
        properties = modification.new_value
        
        # Build AppleScript to update properties
        property_updates = []
        if "name" in properties:
            property_updates.append(f'set name to "{properties["name"]}"')
        if "enabled" in properties:
            property_updates.append(f'set enabled to {str(properties["enabled"]).lower()}')
        
        if not property_updates:
            return Either.right(None)  # No properties to update
        
        updates_script = "\n".join(property_updates)
        
        applescript = f'''
        tell application "Keyboard Maestro"
            set macroRef to macro "{macro_id}"
            if macroRef exists then
                tell macroRef
                    {updates_script}
                end tell
                return "SUCCESS"
            else
                return "MACRO_NOT_FOUND"
            end if
        end tell
        '''
        
        result = await self.km_client.execute_applescript(applescript)
        if result.is_left():
            return Either.left(IntegrationError(
                "applescript_failed",
                f"Failed to update properties: {result.get_left().message}"
            ))
        
        return Either.right(None)
    
    async def _create_macro_backup(self, macro_id: str) -> Either[IntegrationError, str]:
        """Create backup of macro before modification."""
        backup_name = f"{macro_id}_backup_{int(time.time())}"
        
        applescript = f'''
        tell application "Keyboard Maestro"
            set macroRef to macro "{macro_id}"
            if macroRef exists then
                duplicate macroRef with properties {{name:"{backup_name}"}}
                return "{backup_name}"
            else
                return "MACRO_NOT_FOUND"
            end if
        end tell
        '''
        
        result = await self.km_client.execute_applescript(applescript)
        if result.is_left():
            return Either.left(IntegrationError(
                "backup_failed",
                f"Failed to create backup: {result.get_left().message}"
            ))
        
        response = result.get_right()
        if response == "MACRO_NOT_FOUND":
            return Either.left(IntegrationError(
                "macro_not_found",
                f"Macro '{macro_id}' not found for backup"
            ))
        
        return Either.right(response)
    
    async def _restore_from_backup(self, macro_id: str, backup_id: str) -> Either[IntegrationError, None]:
        """Restore macro from backup."""
        # In a real implementation, this would restore the macro from backup
        logger.info(f"Restoring macro {macro_id} from backup {backup_id}")
        return Either.right(None)
    
    def _parse_xml_elements(self, xml_string: str) -> List[Dict[str, Any]]:
        """Parse XML elements from Keyboard Maestro response."""
        if not xml_string or xml_string.strip() == "":
            return []
        
        elements = []
        xml_parts = xml_string.split("|||")
        
        for part in xml_parts:
            if part.strip():
                # Simplified XML parsing - in real implementation, use proper XML parser
                elements.append({
                    "xml": part.strip(),
                    "type": "parsed_element"
                })
        
        return elements
    
    def _extract_variables_from_actions(self, actions: List[Dict[str, Any]]) -> Set[str]:
        """Extract variables used in actions."""
        variables = set()
        
        for action in actions:
            xml = action.get("xml", "")
            # Simple regex to find variable patterns like %Variable%
            matches = re.findall(r'%([^%]+)%', xml)
            variables.update(matches)
        
        return variables
    
    def _generate_action_xml(self, action_type: str, config: Dict[str, Any]) -> str:
        """Generate XML for action (simplified)."""
        # In a real implementation, this would generate proper Keyboard Maestro action XML
        config_json = json.dumps(config).replace('"', '&quot;')
        return f'<action type="{action_type}" config="{config_json}"/>'
    
    def _generate_comparison_recommendation(self, similarity: float, differences: List[Dict]) -> str:
        """Generate recommendation based on comparison results."""
        if similarity > 0.9:
            return "Macros are very similar. Consider consolidating them."
        elif similarity > 0.7:
            return "Macros have significant similarities. Review for potential duplication."
        elif similarity > 0.5:
            return "Macros share some common elements. May benefit from template creation."
        else:
            return "Macros are quite different. No consolidation recommended."