"""
Action Registry for Keyboard Maestro Action Types

Comprehensive registry of supported Keyboard Maestro action types with
validation, categorization, and parameter definitions.
"""

import logging
from typing import Dict, Optional, List, Set
from dataclasses import dataclass

from .action_builder import ActionType, ActionCategory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionParameterDef:
    """Parameter definition for action types."""
    name: str
    type: str  # text, number, boolean, file, application, etc.
    description: str
    default_value: Optional[str] = None
    validation_pattern: Optional[str] = None


class ActionRegistry:
    """Registry of supported Keyboard Maestro action types with validation."""
    
    def __init__(self):
        """Initialize registry with core action types."""
        self._actions: Dict[str, ActionType] = {}
        self._parameter_defs: Dict[str, List[ActionParameterDef]] = {}
        self._initialize_core_actions()
    
    def _initialize_core_actions(self):
        """Initialize registry with comprehensive Keyboard Maestro actions."""
        
        # TEXT ACTIONS
        self._register_text_actions()
        
        # APPLICATION ACTIONS  
        self._register_application_actions()
        
        # SYSTEM ACTIONS
        self._register_system_actions()
        
        # VARIABLE ACTIONS
        self._register_variable_actions()
        
        # CONTROL FLOW ACTIONS
        self._register_control_actions()
        
        # INTERFACE ACTIONS
        self._register_interface_actions()
        
        # FILE ACTIONS
        self._register_file_actions()
        
        # WEB ACTIONS
        self._register_web_actions()
        
        # CLIPBOARD ACTIONS
        self._register_clipboard_actions()
        
        # WINDOW ACTIONS
        self._register_window_actions()
        
        # SOUND ACTIONS
        self._register_sound_actions()
        
        # CALCULATION ACTIONS
        self._register_calculation_actions()
    
    def _register_text_actions(self):
        """Register text manipulation actions."""
        actions = [
            ("Type a String", ["text"], ["by_typing", "by_pasting"]),
            ("Insert Text by Typing", ["text"], ["restore_clipboard"]),
            ("Insert Text by Pasting", ["text"], ["restore_clipboard"]),
            ("Search and Replace", ["search_text", "replace_text"], ["source", "case_sensitive", "regular_expression"]),
            ("Filter Variable", ["variable", "filter"], ["destination_variable"]),
            ("Sort Lines", ["text"], ["sort_order", "case_sensitive", "numeric"]),
            ("Get Line", ["source", "line_number"], ["destination_variable"]),
            ("Count Lines", ["source"], ["destination_variable"]),
            ("Trim Whitespace", ["text"], ["leading", "trailing", "internal"]),
            ("Change Case", ["text", "case_type"], ["destination_variable"]),
            ("Encode/Decode Text", ["text", "encoding_type"], ["destination_variable"]),
            ("Extract URL Components", ["url"], ["component", "destination_variable"]),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.TEXT,
                required_params=required,
                optional_params=optional,
                description=f"Text action: {action_name}"
            ))
    
    def _register_application_actions(self):
        """Register application control actions."""
        actions = [
            ("Activate a Specific Application", ["application"], ["bring_all_windows", "launch_if_not_running"]),
            ("Launch Application", ["application"], ["parameters", "activate"]),
            ("Quit Application", ["application"], ["confirm", "save_documents"]),
            ("Quit All Applications", [], ["exclude_applications", "confirm"]),
            ("Hide Application", ["application"], []),
            ("Hide All Applications", [], ["exclude_applications"]),
            ("Select Menu Item", ["application", "menu_path"], ["via_applescript"]),
            ("Press Button", ["application", "button_name"], ["window_name"]),
            ("Manipulate Window", ["application", "action"], ["window_name", "parameters"]),
            ("Wait for Application Launch", ["application"], ["timeout"]),
            ("Get Application Version", ["application"], ["destination_variable"]),
            ("Get Application Path", ["application"], ["destination_variable"]),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.APPLICATION,
                required_params=required,
                optional_params=optional,
                description=f"Application action: {action_name}"
            ))
    
    def _register_system_actions(self):
        """Register system control actions."""
        actions = [
            ("Pause", ["duration"], ["unit"]),
            ("Beep", [], ["count", "wait_for_completion"]),
            ("Play Sound", ["sound_file"], ["wait_for_completion", "volume"]),
            ("Set System Volume", ["volume"], ["output_device"]),
            ("Mute System Audio", [], ["restore_after"]),
            ("Sleep", [], ["delay"]),
            ("Restart", [], ["delay", "confirm"]),
            ("Shut Down", [], ["delay", "confirm"]),
            ("Log Out", [], ["delay", "confirm"]),
            ("Lock Screen", [], []),
            ("Empty Trash", [], ["confirm", "secure_delete"]),
            ("Eject Disk", ["disk_name"], []),
            ("Execute Shell Script", ["script"], ["timeout", "as_user"]),
            ("Execute AppleScript", ["script"], ["timeout", "compile"]),
            ("Display Text", ["text"], ["title", "buttons", "timeout"]),
            ("Display Dialog", ["text"], ["title", "buttons", "default_button", "timeout"]),
            ("Prompt for User Input", ["prompt"], ["title", "default_answer", "timeout"]),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.SYSTEM,
                required_params=required,
                optional_params=optional,
                description=f"System action: {action_name}"
            ))
    
    def _register_variable_actions(self):
        """Register variable manipulation actions."""
        actions = [
            ("Set Variable to Text", ["variable", "text"], ["append", "trim"]),
            ("Set Variable to Calculation", ["variable", "calculation"], []),
            ("Increment Variable", ["variable"], ["by_amount"]),
            ("Decrement Variable", ["variable"], ["by_amount"]),
            ("Delete Variable", ["variable"], []),
            ("Get Variable", ["variable"], ["destination_variable"]),
            ("Set Dictionary Key", ["dictionary", "key", "value"], ["create_if_missing"]),
            ("Get Dictionary Key", ["dictionary", "key"], ["destination_variable", "default_value"]),
            ("Delete Dictionary Key", ["dictionary", "key"], []),
            ("Get Dictionary Keys", ["dictionary"], ["destination_variable"]),
            ("Set Variable to UTC Date/Time", ["variable"], ["format"]),
            ("Set Variable to Calculation Result", ["variable", "expression"], []),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.VARIABLE,
                required_params=required,
                optional_params=optional,
                description=f"Variable action: {action_name}"
            ))
    
    def _register_control_actions(self):
        """Register control flow actions."""
        actions = [
            ("If Then Else", ["condition"], ["else_condition"]),
            ("For Each", ["collection"], ["item_variable", "index_variable"]),
            ("While", ["condition"], ["timeout"]),
            ("Repeat", ["count"], ["counter_variable"]),
            ("Switch/Case", ["variable"], ["cases"]),
            ("Break from Loop", [], []),
            ("Continue Loop", [], []),
            ("Exit Macro", [], ["result"]),
            ("Cancel Macro", [], ["message"]),
            ("Execute Macro", ["macro_name"], ["parameters", "asynchronously"]),
            ("Call Subroutine", ["subroutine_name"], ["parameters"]),
            ("Return from Subroutine", [], ["result"]),
            ("Try/Catch", ["try_actions"], ["catch_actions", "error_variable"]),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.CONTROL,
                required_params=required,
                optional_params=optional,
                description=f"Control flow action: {action_name}"
            ))
    
    def _register_interface_actions(self):
        """Register user interface actions."""
        actions = [
            ("Click at Found Image", ["image"], ["fuzziness", "timeout"]),
            ("Click at Coordinates", ["x", "y"], ["click_type", "modifiers"]),
            ("Drag from Image to Image", ["source_image", "destination_image"], ["fuzziness"]),
            ("Type a Keystroke", ["keystroke"], ["modifiers"]),
            ("Press Key", ["key"], ["modifiers", "repeat_count"]),
            ("Scroll Wheel", ["direction", "amount"], ["x", "y"]),
            ("Move Mouse", ["x", "y"], ["relative", "speed"]),
            ("Move Mouse to Found Image", ["image"], ["fuzziness", "timeout"]),
            ("Get Mouse Location", [], ["destination_variable"]),
            ("Restore Mouse Location", [], []),
            ("Get Screen Resolution", [], ["destination_variable"]),
            ("Get Pixel Color", ["x", "y"], ["destination_variable"]),
            ("OCR Screen Area", ["x", "y", "width", "height"], ["language", "destination_variable"]),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.INTERFACE,
                required_params=required,
                optional_params=optional,
                description=f"Interface action: {action_name}"
            ))
    
    def _register_file_actions(self):
        """Register file system actions."""
        actions = [
            ("Copy File", ["source", "destination"], ["overwrite", "create_folders"]),
            ("Move File", ["source", "destination"], ["overwrite", "create_folders"]),
            ("Delete File", ["file_path"], ["confirm", "move_to_trash"]),
            ("Rename File", ["file_path", "new_name"], ["overwrite"]),
            ("Create Folder", ["folder_path"], ["create_intermediate"]),
            ("Open File", ["file_path"], ["application"]),
            ("Read File", ["file_path"], ["destination_variable", "encoding"]),
            ("Write File", ["file_path", "content"], ["encoding", "append"]),
            ("Get File Attributes", ["file_path"], ["attribute", "destination_variable"]),
            ("Set File Attributes", ["file_path", "attributes"], []),
            ("Get File Size", ["file_path"], ["destination_variable", "unit"]),
            ("Get Folder Contents", ["folder_path"], ["destination_variable", "recursive"]),
            ("Find Files", ["search_path", "pattern"], ["destination_variable", "recursive"]),
            ("Compress Files", ["files", "archive_path"], ["compression_type"]),
            ("Extract Archive", ["archive_path", "destination"], ["overwrite"]),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.FILE,
                required_params=required,
                optional_params=optional,
                description=f"File action: {action_name}"
            ))
    
    def _register_web_actions(self):
        """Register web request actions."""
        actions = [
            ("Get URL", ["url"], ["destination_variable", "headers", "timeout"]),
            ("Post to URL", ["url", "data"], ["destination_variable", "headers", "timeout"]),
            ("Download File", ["url", "destination"], ["overwrite", "timeout"]),
            ("Upload File", ["url", "file_path"], ["field_name", "additional_fields"]),
            ("Submit Web Form", ["url", "form_data"], ["destination_variable", "method"]),
            ("Get Web Page Title", ["url"], ["destination_variable"]),
            ("Extract Web Page Text", ["url"], ["destination_variable", "selector"]),
            ("Take Website Screenshot", ["url", "file_path"], ["width", "height"]),
            ("Open URL in Browser", ["url"], ["browser", "new_tab"]),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.WEB,
                required_params=required,
                optional_params=optional,
                description=f"Web action: {action_name}"
            ))
    
    def _register_clipboard_actions(self):
        """Register clipboard manipulation actions."""
        actions = [
            ("Copy to Clipboard", ["text"], ["append"]),
            ("Cut to Clipboard", [], []),
            ("Paste from Clipboard", [], ["destination"]),
            ("Get Clipboard Contents", [], ["destination_variable", "format"]),
            ("Set Clipboard to File", ["file_path"], []),
            ("Save Clipboard to File", ["file_path"], ["format", "overwrite"]),
            ("Set Named Clipboard", ["name", "content"], []),
            ("Get Named Clipboard", ["name"], ["destination_variable"]),
            ("Delete Named Clipboard", ["name"], []),
            ("Get Clipboard History", ["index"], ["destination_variable"]),
            ("Clear Clipboard History", [], []),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.CLIPBOARD,
                required_params=required,
                optional_params=optional,
                description=f"Clipboard action: {action_name}"
            ))
    
    def _register_window_actions(self):
        """Register window management actions."""
        actions = [
            ("Move Window", ["window", "x", "y"], ["application"]),
            ("Resize Window", ["window", "width", "height"], ["application"]),
            ("Minimize Window", ["window"], ["application"]),
            ("Maximize Window", ["window"], ["application"]),
            ("Close Window", ["window"], ["application", "save"]),
            ("Bring Window to Front", ["window"], ["application"]),
            ("Get Window Position", ["window"], ["application", "destination_variable"]),
            ("Get Window Size", ["window"], ["application", "destination_variable"]),
            ("Get Window Title", ["window"], ["application", "destination_variable"]),
            ("Set Window Title", ["window", "title"], ["application"]),
            ("Get Front Window", [], ["application", "destination_variable"]),
            ("Arrange Windows", ["arrangement"], ["applications"]),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.WINDOW,
                required_params=required,
                optional_params=optional,
                description=f"Window action: {action_name}"
            ))
    
    def _register_sound_actions(self):
        """Register sound and audio actions."""
        actions = [
            ("Speak Text", ["text"], ["voice", "rate", "volume"]),
            ("Set Volume", ["volume"], ["device"]),
            ("Get Volume", [], ["device", "destination_variable"]),
            ("Mute Audio", [], ["device"]),
            ("Unmute Audio", [], ["device"]),
            ("Play iTunes Track", ["track"], ["playlist"]),
            ("Pause iTunes", [], []),
            ("Stop iTunes", [], []),
            ("Get iTunes Info", [], ["property", "destination_variable"]),
            ("Record Audio", ["file_path", "duration"], ["quality", "input_device"]),
            ("Play Audio File", ["file_path"], ["wait_for_completion", "volume"]),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.SOUND,
                required_params=required,
                optional_params=optional,
                description=f"Sound action: {action_name}"
            ))
    
    def _register_calculation_actions(self):
        """Register calculation and math actions."""
        actions = [
            ("Calculate", ["expression"], ["destination_variable"]),
            ("Calculate Expression", ["expression"], ["destination_variable", "precision"]),
            ("Random Number", ["min", "max"], ["destination_variable", "integer"]),
            ("Round Number", ["number"], ["destination_variable", "decimals"]),
            ("Format Number", ["number", "format"], ["destination_variable"]),
            ("Convert Units", ["value", "from_unit", "to_unit"], ["destination_variable"]),
            ("Get Date/Time", [], ["format", "destination_variable"]),
            ("Format Date/Time", ["date", "format"], ["destination_variable"]),
            ("Date Math", ["date", "operation", "amount"], ["unit", "destination_variable"]),
        ]
        
        for action_name, required, optional in actions:
            self.register_action(ActionType(
                identifier=action_name,
                category=ActionCategory.CALCULATION,
                required_params=required,
                optional_params=optional,
                description=f"Calculation action: {action_name}"
            ))
    
    def register_action(self, action_type: ActionType):
        """Register new action type in registry."""
        if action_type.identifier in self._actions:
            logger.warning(f"Action type {action_type.identifier} already registered, replacing")
        
        self._actions[action_type.identifier] = action_type
        logger.debug(f"Registered action type: {action_type.identifier}")
    
    def get_action_type(self, identifier: str) -> Optional[ActionType]:
        """Get action type by identifier."""
        return self._actions.get(identifier)
    
    def get_actions_by_category(self, category: ActionCategory) -> List[ActionType]:
        """Get all actions in specific category."""
        return [
            action for action in self._actions.values() 
            if action.category == category
        ]
    
    def list_all_actions(self) -> List[ActionType]:
        """Get all registered action types."""
        return list(self._actions.values())
    
    def list_action_names(self) -> List[str]:
        """Get list of all action identifiers."""
        return list(self._actions.keys())
    
    def get_action_count(self) -> int:
        """Get total number of registered actions."""
        return len(self._actions)
    
    def get_category_counts(self) -> Dict[ActionCategory, int]:
        """Get count of actions by category."""
        counts = {}
        for action in self._actions.values():
            counts[action.category] = counts.get(action.category, 0) + 1
        return counts
    
    def search_actions(self, query: str) -> List[ActionType]:
        """Search for actions by name or description."""
        query_lower = query.lower()
        results = []
        
        for action in self._actions.values():
            if (query_lower in action.identifier.lower() or 
                query_lower in action.description.lower()):
                results.append(action)
        
        return results
    
    def validate_action_parameters(self, action_type: str, parameters: Dict[str, any]) -> Dict[str, any]:
        """Validate parameters for specific action type."""
        action = self.get_action_type(action_type)
        if not action:
            return {
                "valid": False,
                "error": f"Unknown action type: {action_type}",
                "missing_required": [],
                "unknown_params": []
            }
        
        # Check required parameters
        missing_required = [
            param for param in action.required_params 
            if param not in parameters
        ]
        
        # Check for unknown parameters
        all_valid_params = set(action.required_params + action.optional_params)
        unknown_params = [
            param for param in parameters.keys()
            if param not in all_valid_params
        ]
        
        return {
            "valid": len(missing_required) == 0,
            "missing_required": missing_required,
            "unknown_params": unknown_params,
            "action": action
        }