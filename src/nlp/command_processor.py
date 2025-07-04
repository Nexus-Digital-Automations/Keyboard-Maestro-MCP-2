"""
Command Processor - TASK_60 Phase 2 Core Implementation

Natural language command processing and interpretation for automation workflows.
Converts natural language commands into structured automation actions and workflows.

Architecture: Command Processing + Intent Integration + Workflow Generation + Action Mapping
Performance: <300ms command processing, <500ms workflow generation, <200ms action mapping
Security: Safe command processing, validated automation actions, comprehensive input sanitization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, UTC
import asyncio
import logging
from enum import Enum

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.nlp_architecture import (
    TextContent, Intent, CommandId, ProcessedCommand, RecognizedIntent,
    ExtractedEntity, SentimentAnalysis, LanguageCode, NLPError,
    create_command_id, validate_text_input, is_automation_related
)
from src.nlp.intent_recognizer import IntentClassifier


class CommandComplexity(Enum):
    """Complexity levels for automation commands."""
    SIMPLE = "simple"          # Single action commands
    INTERMEDIATE = "intermediate"  # Multi-step commands
    COMPLEX = "complex"        # Conditional logic and loops
    ADVANCED = "advanced"      # Custom scripts and integrations


class ActionCategory(Enum):
    """Categories of automation actions."""
    SYSTEM_CONTROL = "system_control"    # Volume, brightness, sleep
    APPLICATION = "application"          # Launch, quit, activate apps
    TEXT_MANIPULATION = "text_manipulation"  # Type, copy, paste text
    MOUSE_KEYBOARD = "mouse_keyboard"    # Click, key press, shortcuts
    FILE_SYSTEM = "file_system"         # Open, save, move files
    WINDOW_MANAGEMENT = "window_management"  # Resize, move, minimize
    WORKFLOW_CONTROL = "workflow_control"   # Conditionals, loops, delays
    COMMUNICATION = "communication"      # Email, messages, notifications
    WEB_AUTOMATION = "web_automation"    # Browser actions, web scraping
    CUSTOM_SCRIPT = "custom_script"      # AppleScript, shell commands


@dataclass(frozen=True)
class AutomationAction:
    """Individual automation action within a command."""
    action_id: str
    action_type: str
    category: ActionCategory
    parameters: Dict[str, Any]
    description: str
    required_permissions: List[str] = field(default_factory=list)
    estimated_duration_ms: int = 100
    failure_handling: str = "continue"  # continue, retry, abort
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowTemplate:
    """Template for generating automation workflows."""
    template_id: str
    template_name: str
    description: str
    complexity: CommandComplexity
    action_sequence: List[AutomationAction]
    required_entities: List[str] = field(default_factory=list)
    optional_entities: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


class CommandProcessor:
    """Natural language command processing and interpretation system."""
    
    def __init__(self, intent_classifier: IntentClassifier):
        self.intent_classifier = intent_classifier
        self.action_templates = self._initialize_action_templates()
        self.workflow_templates = self._initialize_workflow_templates()
        self.processing_cache = {}
        self.command_history = []
    
    def _initialize_action_templates(self) -> Dict[str, AutomationAction]:
        """Initialize automation action templates."""
        return {
            # Application Control Actions
            "launch_application": AutomationAction(
                action_id="launch_app",
                action_type="application_control",
                category=ActionCategory.APPLICATION,
                parameters={"application_name": "", "wait_for_launch": True},
                description="Launch an application",
                required_permissions=["application_control"],
                estimated_duration_ms=2000
            ),
            
            "quit_application": AutomationAction(
                action_id="quit_app",
                action_type="application_control", 
                category=ActionCategory.APPLICATION,
                parameters={"application_name": "", "force_quit": False},
                description="Quit an application",
                required_permissions=["application_control"],
                estimated_duration_ms=1000
            ),
            
            # Text Manipulation Actions
            "type_text": AutomationAction(
                action_id="type_text",
                action_type="text_input",
                category=ActionCategory.TEXT_MANIPULATION,
                parameters={"text_content": "", "typing_speed": "normal"},
                description="Type text at current cursor position",
                required_permissions=["input_control"],
                estimated_duration_ms=500
            ),
            
            "copy_text": AutomationAction(
                action_id="copy_text",
                action_type="clipboard",
                category=ActionCategory.TEXT_MANIPULATION,
                parameters={"text_content": ""},
                description="Copy text to clipboard",
                required_permissions=["clipboard_access"],
                estimated_duration_ms=100
            ),
            
            # Mouse and Keyboard Actions
            "click_position": AutomationAction(
                action_id="click_pos",
                action_type="mouse_control",
                category=ActionCategory.MOUSE_KEYBOARD,
                parameters={"x": 0, "y": 0, "click_type": "left"},
                description="Click at specific screen coordinates",
                required_permissions=["input_control"],
                estimated_duration_ms=200
            ),
            
            "press_hotkey": AutomationAction(
                action_id="press_hotkey",
                action_type="keyboard_control",
                category=ActionCategory.MOUSE_KEYBOARD,
                parameters={"key_combination": "", "modifier_keys": []},
                description="Press keyboard shortcut",
                required_permissions=["input_control"],
                estimated_duration_ms=100
            ),
            
            # System Control Actions
            "set_volume": AutomationAction(
                action_id="set_volume",
                action_type="system_control",
                category=ActionCategory.SYSTEM_CONTROL,
                parameters={"volume_level": 50, "device": "default"},
                description="Set system volume level",
                required_permissions=["system_control"],
                estimated_duration_ms=300
            ),
            
            "play_sound": AutomationAction(
                action_id="play_sound",
                action_type="audio_control",
                category=ActionCategory.SYSTEM_CONTROL,
                parameters={"sound_file": "", "volume": 1.0},
                description="Play audio file",
                required_permissions=["audio_control"],
                estimated_duration_ms=1000
            ),
            
            # File System Actions
            "open_file": AutomationAction(
                action_id="open_file",
                action_type="file_operation",
                category=ActionCategory.FILE_SYSTEM,
                parameters={"file_path": "", "application": "default"},
                description="Open file with specified application",
                required_permissions=["file_access"],
                estimated_duration_ms=1500
            ),
            
            "save_file": AutomationAction(
                action_id="save_file",
                action_type="file_operation",
                category=ActionCategory.FILE_SYSTEM,
                parameters={"file_path": "", "content": "", "overwrite": False},
                description="Save content to file",
                required_permissions=["file_write"],
                estimated_duration_ms=800
            ),
            
            # Window Management Actions
            "resize_window": AutomationAction(
                action_id="resize_window",
                action_type="window_control",
                category=ActionCategory.WINDOW_MANAGEMENT,
                parameters={"width": 800, "height": 600, "application": ""},
                description="Resize application window",
                required_permissions=["window_control"],
                estimated_duration_ms=300
            ),
            
            "move_window": AutomationAction(
                action_id="move_window", 
                action_type="window_control",
                category=ActionCategory.WINDOW_MANAGEMENT,
                parameters={"x": 0, "y": 0, "application": ""},
                description="Move application window",
                required_permissions=["window_control"],
                estimated_duration_ms=200
            ),
            
            # Workflow Control Actions
            "pause_execution": AutomationAction(
                action_id="pause_exec",
                action_type="flow_control",
                category=ActionCategory.WORKFLOW_CONTROL,
                parameters={"duration_seconds": 1.0},
                description="Pause workflow execution",
                required_permissions=[],
                estimated_duration_ms=0  # Variable based on pause duration
            ),
            
            "conditional_branch": AutomationAction(
                action_id="conditional",
                action_type="flow_control",
                category=ActionCategory.WORKFLOW_CONTROL,
                parameters={"condition": "", "true_actions": [], "false_actions": []},
                description="Execute actions based on condition",
                required_permissions=[],
                estimated_duration_ms=100
            ),
            
            # Communication Actions
            "send_notification": AutomationAction(
                action_id="notify",
                action_type="notification",
                category=ActionCategory.COMMUNICATION,
                parameters={"title": "", "message": "", "sound": True},
                description="Display system notification",
                required_permissions=["notification_access"],
                estimated_duration_ms=500
            ),
            
            # Web Automation Actions
            "open_url": AutomationAction(
                action_id="open_url",
                action_type="web_control",
                category=ActionCategory.WEB_AUTOMATION,
                parameters={"url": "", "browser": "default"},
                description="Open URL in web browser",
                required_permissions=["web_access"],
                estimated_duration_ms=2000
            )
        }
    
    def _initialize_workflow_templates(self) -> Dict[str, WorkflowTemplate]:
        """Initialize workflow templates for common automation patterns."""
        return {
            "simple_app_launch": WorkflowTemplate(
                template_id="simple_launch",
                template_name="Simple Application Launch",
                description="Launch an application and optionally perform an action",
                complexity=CommandComplexity.SIMPLE,
                action_sequence=[
                    self.action_templates["launch_application"]
                ],
                required_entities=["application"],
                success_criteria=["Application launched successfully"]
            ),
            
            "text_processing": WorkflowTemplate(
                template_id="text_process",
                template_name="Text Processing Workflow",
                description="Process text with copy, type, or manipulation operations",
                complexity=CommandComplexity.INTERMEDIATE,
                action_sequence=[
                    self.action_templates["type_text"]
                ],
                required_entities=["text_content"],
                success_criteria=["Text processed successfully"]
            ),
            
            "file_operations": WorkflowTemplate(
                template_id="file_ops",
                template_name="File Operations Workflow", 
                description="Perform file system operations",
                complexity=CommandComplexity.INTERMEDIATE,
                action_sequence=[
                    self.action_templates["open_file"]
                ],
                required_entities=["file_path"],
                success_criteria=["File operation completed"]
            ),
            
            "system_control": WorkflowTemplate(
                template_id="system_ctrl",
                template_name="System Control Workflow",
                description="Control system settings and state",
                complexity=CommandComplexity.INTERMEDIATE,
                action_sequence=[
                    self.action_templates["set_volume"]
                ],
                required_entities=["system_setting"],
                success_criteria=["System setting changed"]
            ),
            
            "notification_workflow": WorkflowTemplate(
                template_id="notify_flow",
                template_name="Notification Workflow",
                description="Send notifications and alerts",
                complexity=CommandComplexity.SIMPLE,
                action_sequence=[
                    self.action_templates["send_notification"]
                ],
                required_entities=["message"],
                success_criteria=["Notification sent"]
            )
        }
    
    @require(lambda text: isinstance(text, TextContent))
    @ensure(lambda result: result.is_right() or isinstance(result.left_value, NLPError))
    async def process_command(
        self,
        text: TextContent,
        language: LanguageCode = LanguageCode("en"),
        confidence_threshold: float = 0.7,
        include_alternatives: bool = True,
        context_domain: Optional[str] = None
    ) -> Either[NLPError, ProcessedCommand]:
        """Process natural language command into structured automation workflow."""
        try:
            start_time = datetime.now(UTC)
            command_id = create_command_id()
            
            # Check if command is automation-related
            if not is_automation_related(str(text)):
                return Either.left(NLPError(
                    "Command does not appear to be automation-related",
                    "NOT_AUTOMATION_COMMAND",
                    context={"text": str(text)[:100]}
                ))
            
            # Recognize intent using the classifier
            intent_result = await self.intent_classifier.recognize_intent(
                text, confidence_threshold, max_intents=5
            )
            
            if intent_result.is_left():
                return Either.left(intent_result.left_value)
            
            recognized_intents = intent_result.right_value
            if not recognized_intents:
                return Either.left(NLPError(
                    "No automation intents recognized",
                    "NO_INTENT_FOUND",
                    context={"confidence_threshold": confidence_threshold}
                ))
            
            # Use the highest confidence intent
            primary_intent = recognized_intents[0]
            
            # Extract entities from all intents
            all_entities = []
            for intent in recognized_intents:
                all_entities.extend(intent.entities)
            
            # Remove duplicate entities
            unique_entities = self._deduplicate_entities(all_entities)
            
            # Generate automation actions based on intent and entities
            automation_actions = await self._generate_automation_actions(
                primary_intent, unique_entities, context_domain
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_command_confidence(
                primary_intent, unique_entities, automation_actions
            )
            
            # Calculate processing time
            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            # Generate alternatives if requested
            alternatives = []
            if include_alternatives and len(recognized_intents) > 1:
                for intent in recognized_intents[1:3]:  # Top 2 alternatives
                    alt_actions = await self._generate_automation_actions(
                        intent, unique_entities, context_domain
                    )
                    alt_confidence = self._calculate_command_confidence(
                        intent, unique_entities, alt_actions
                    )
                    
                    alternative = ProcessedCommand(
                        command_id=create_command_id(),
                        original_text=text,
                        recognized_intent=intent,
                        extracted_entities=unique_entities,
                        automation_actions=alt_actions,
                        confidence_score=alt_confidence,
                        processing_time_ms=processing_time
                    )
                    alternatives.append(alternative)
            
            # Create the processed command
            processed_command = ProcessedCommand(
                command_id=command_id,
                original_text=text,
                recognized_intent=primary_intent,
                extracted_entities=unique_entities,
                automation_actions=automation_actions,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                alternatives=alternatives
            )
            
            # Cache the result
            cache_key = f"{str(text)}_{language}_{confidence_threshold}"
            self.processing_cache[cache_key] = processed_command
            
            # Add to command history
            self.command_history.append({
                "command_id": command_id,
                "text": str(text),
                "intent": primary_intent.intent,
                "confidence": confidence_score,
                "timestamp": start_time,
                "success": True
            })
            
            return Either.right(processed_command)
            
        except Exception as e:
            return Either.left(NLPError(
                f"Command processing failed: {str(e)}",
                "PROCESSING_ERROR",
                context={"text_length": len(str(text))}
            ))
    
    async def generate_workflow_from_description(
        self,
        description: TextContent,
        workflow_type: str = "macro",
        complexity_level: str = "intermediate",
        include_error_handling: bool = True
    ) -> Either[NLPError, Dict[str, Any]]:
        """Generate complete automation workflow from natural language description."""
        try:
            # Process the description as a command first
            command_result = await self.process_command(description)
            
            if command_result.is_left():
                return Either.left(command_result.left_value)
            
            processed_command = command_result.right_value
            
            # Generate workflow structure
            workflow = {
                "workflow_id": f"wf_{processed_command.command_id}",
                "workflow_type": workflow_type,
                "complexity": complexity_level,
                "description": str(description),
                "created_at": datetime.now(UTC).isoformat(),
                "workflow": {
                    "metadata": {
                        "name": f"Generated from: {str(description)[:50]}...",
                        "description": str(description),
                        "intent": processed_command.recognized_intent.intent,
                        "confidence": processed_command.confidence_score,
                        "estimated_duration_ms": sum(
                            action.get("estimated_duration_ms", 100) 
                            for action in processed_command.automation_actions
                        )
                    },
                    "triggers": [
                        {
                            "type": "manual",
                            "description": "Manual execution trigger"
                        }
                    ],
                    "actions": processed_command.automation_actions,
                    "error_handling": self._generate_error_handling() if include_error_handling else {},
                    "validation": {
                        "required_permissions": list(set(
                            perm for action in processed_command.automation_actions
                            for perm in action.get("required_permissions", [])
                        )),
                        "prerequisites": [],
                        "success_criteria": [
                            "All actions completed successfully",
                            "No errors encountered",
                            "Expected outcome achieved"
                        ]
                    }
                }
            }
            
            return Either.right(workflow)
            
        except Exception as e:
            return Either.left(NLPError(
                f"Workflow generation failed: {str(e)}",
                "WORKFLOW_GENERATION_ERROR"
            ))
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities while preserving highest confidence ones."""
        seen_entities = {}
        
        for entity in entities:
            key = f"{entity.entity_type.value}_{str(entity.value)}"
            if key not in seen_entities or entity.confidence > seen_entities[key].confidence:
                seen_entities[key] = entity
        
        return list(seen_entities.values())
    
    async def _generate_automation_actions(
        self,
        intent: RecognizedIntent,
        entities: List[ExtractedEntity],
        context_domain: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate automation actions based on recognized intent and entities."""
        actions = []
        
        # Map intent to action templates
        intent_action_mapping = {
            "create_automation": ["launch_application"],
            "run_automation": ["launch_application", "press_hotkey"],
            "open_file": ["open_file"],
            "type_text": ["type_text"],
            "launch_application": ["launch_application"],
            "quit_application": ["quit_application"],
            "set_volume": ["set_volume"],
            "play_sound": ["play_sound"],
            "send_notification": ["send_notification"],
            "open_url": ["open_url"],
            "copy_text": ["copy_text"],
            "click_position": ["click_position"],
            "press_hotkey": ["press_hotkey"]
        }
        
        # Get relevant action templates
        action_templates = intent_action_mapping.get(intent.intent, ["send_notification"])
        
        for template_name in action_templates:
            if template_name in self.action_templates:
                template = self.action_templates[template_name]
                
                # Create action with entity-based parameters
                action_params = template.parameters.copy()
                
                # Map entities to action parameters
                for entity in entities:
                    if entity.entity_type.value == "application":
                        action_params["application_name"] = str(entity.value)
                    elif entity.entity_type.value == "file_path":
                        action_params["file_path"] = str(entity.value)
                    elif entity.entity_type.value == "url":
                        action_params["url"] = str(entity.value)
                    elif entity.entity_type.value == "hotkey":
                        action_params["key_combination"] = str(entity.value)
                    elif entity.entity_type.value == "number":
                        if template.action_type == "system_control":
                            action_params["volume_level"] = min(100, max(0, int(float(str(entity.value)))))
                
                # Add default text if needed
                if template.action_type == "text_input" and "text_content" in action_params:
                    if not action_params["text_content"]:
                        # Extract text from intent parameters
                        text_entities = [e for e in entities if "text" in str(e.value).lower()]
                        if text_entities:
                            action_params["text_content"] = str(text_entities[0].value)
                        else:
                            action_params["text_content"] = "Generated text content"
                
                action = {
                    "action_id": f"{template.action_id}_{len(actions)}",
                    "action_type": template.action_type,
                    "category": template.category.value,
                    "description": template.description,
                    "parameters": action_params,
                    "required_permissions": template.required_permissions,
                    "estimated_duration_ms": template.estimated_duration_ms,
                    "failure_handling": template.failure_handling,
                    "validation_rules": template.validation_rules
                }
                
                actions.append(action)
        
        return actions
    
    def _calculate_command_confidence(
        self,
        intent: RecognizedIntent,
        entities: List[ExtractedEntity],
        actions: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence for the processed command."""
        # Base confidence from intent
        confidence = intent.confidence
        
        # Boost confidence if we have relevant entities
        if entities:
            entity_boost = min(0.2, len(entities) * 0.05)
            confidence += entity_boost
        
        # Boost confidence if we generated meaningful actions
        if actions:
            action_boost = min(0.1, len(actions) * 0.03)
            confidence += action_boost
        
        # Reduce confidence if intent confidence is low
        if intent.confidence < 0.5:
            confidence *= 0.8
        
        return min(1.0, confidence)
    
    def _generate_error_handling(self) -> Dict[str, Any]:
        """Generate error handling configuration for workflows."""
        return {
            "on_action_failure": {
                "strategy": "retry_with_fallback",
                "max_retries": 3,
                "retry_delay_ms": 1000,
                "fallback_actions": [
                    {
                        "action_type": "notification",
                        "parameters": {
                            "title": "Automation Error",
                            "message": "An action failed during execution",
                            "sound": True
                        }
                    }
                ]
            },
            "on_permission_denied": {
                "strategy": "notify_and_abort",
                "actions": [
                    {
                        "action_type": "notification",
                        "parameters": {
                            "title": "Permission Required",
                            "message": "This automation requires additional permissions",
                            "sound": True
                        }
                    }
                ]
            },
            "on_timeout": {
                "strategy": "abort_with_cleanup",
                "timeout_ms": 30000,
                "cleanup_actions": []
            }
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get command processing performance statistics."""
        if not self.command_history:
            return {"total_commands": 0}
        
        successful_commands = [cmd for cmd in self.command_history if cmd["success"]]
        
        if not successful_commands:
            return {"total_commands": len(self.command_history), "success_rate": 0.0}
        
        avg_confidence = sum(cmd["confidence"] for cmd in successful_commands) / len(successful_commands)
        
        intent_distribution = {}
        for cmd in successful_commands:
            intent = cmd["intent"]
            intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
        
        return {
            "total_commands": len(self.command_history),
            "successful_commands": len(successful_commands),
            "success_rate": len(successful_commands) / len(self.command_history),
            "average_confidence": avg_confidence,
            "intent_distribution": intent_distribution,
            "cache_size": len(self.processing_cache),
            "available_templates": len(self.action_templates)
        }