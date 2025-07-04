"""
Voice Command Intent Processor - TASK_66 Phase 2 Core Voice Engine

Voice command intent recognition, action mapping, and parameter extraction
with natural language understanding and automation workflow integration.

Architecture: Intent Recognition + Entity Extraction + Action Mapping + Context Awareness
Performance: <100ms intent recognition, <50ms parameter extraction, <200ms action mapping
Security: Command validation, intent verification, safe parameter processing
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, Pattern
from datetime import datetime, UTC
from dataclasses import dataclass, field
import re
import logging
import json
from enum import Enum

from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.voice_architecture import (
    VoiceCommand, VoiceCommandType, CommandPriority, VoiceCommandId,
    VoiceCommandError, VoiceControlError, SpeakerId,
    create_voice_command_id, validate_voice_command_security
)

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    """Categories of voice command intents."""
    AUTOMATION = "automation"  # Trigger automation workflows
    CONTROL = "control"  # Control system functions
    NAVIGATION = "navigation"  # Navigate applications/interfaces
    INPUT = "input"  # Text input and dictation
    QUERY = "query"  # Information queries
    CREATION = "creation"  # Create new items/workflows
    MODIFICATION = "modification"  # Modify existing items
    SYSTEM = "system"  # System-level commands


@dataclass(frozen=True)
class IntentPattern:
    """Voice command intent pattern for recognition."""
    intent_name: str
    category: IntentCategory
    command_type: VoiceCommandType
    patterns: List[str]  # Regex patterns for matching
    required_entities: List[str] = field(default_factory=list)
    optional_entities: List[str] = field(default_factory=list)
    priority: CommandPriority = CommandPriority.MEDIUM
    requires_confirmation: bool = False
    examples: List[str] = field(default_factory=list)
    
    @require(lambda self: len(self.intent_name) > 0)
    @require(lambda self: len(self.patterns) > 0)
    def __post_init__(self):
        pass
    
    def matches_text(self, text: str) -> bool:
        """Check if text matches any pattern."""
        text_lower = text.lower().strip()
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text using patterns."""
        entities = {}
        text_lower = text.lower().strip()
        
        for pattern in self.patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                entities.update(match.groupdict())
                break
        
        return entities


@dataclass
class EntityExtractor:
    """Extract named entities from voice commands."""
    
    # Common entity patterns
    ENTITY_PATTERNS = {
        "number": r'\b(\d+(?:\.\d+)?)\b',
        "application": r'\b(safari|chrome|firefox|mail|finder|calendar|notes|messages|music|photos|maps)\b',
        "file_name": r'(?:file|document)?\s*["\']([^"\']+)["\']',
        "url": r'(https?://[^\s]+)',
        "email": r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
        "time": r'\b(\d{1,2}:\d{2}(?:\s*[ap]m)?)\b',
        "date": r'\b(today|tomorrow|yesterday|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b',
        "direction": r'\b(up|down|left|right|forward|back|next|previous)\b',
        "action": r'\b(open|close|start|stop|pause|play|run|execute|create|delete|save|copy|paste)\b',
        "location": r'\b(desktop|documents|downloads|home|trash|applications)\b',
        "text_content": r'["\']([^"\']+)["\']|(?:say|type|write|input)\s+(.+?)(?:\s+(?:in|to|on)|$)',
        "macro_name": r'(?:macro|automation|workflow)\s+["\']?([^"\']+)["\']?',
    }
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract all entities from text."""
        entities = {}
        text_lower = text.lower().strip()
        
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            values = []
            
            for match in matches:
                if match.groups():
                    # Use first non-None group
                    value = next((group for group in match.groups() if group), None)
                    if value:
                        values.append(value.strip())
                else:
                    values.append(match.group().strip())
            
            if values:
                entities[entity_type] = values[0] if len(values) == 1 else values
        
        return entities
    
    def extract_specific_entity(self, text: str, entity_type: str) -> Optional[str]:
        """Extract specific entity type from text."""
        if entity_type not in self.ENTITY_PATTERNS:
            return None
        
        pattern = self.ENTITY_PATTERNS[entity_type]
        match = re.search(pattern, text.lower(), re.IGNORECASE)
        
        if match:
            if match.groups():
                return next((group for group in match.groups() if group), None)
            return match.group()
        
        return None


class IntentProcessor:
    """
    Voice command intent recognition and processing system.
    
    Contracts:
        Preconditions:
            - Voice recognition text must be non-empty and properly formatted
            - Intent patterns must be loaded and validated
            - Security validation required for all commands
        
        Postconditions:
            - Intent recognition returns confidence score and extracted entities
            - Command parameters are validated and sanitized
            - Action mapping provides executable automation instructions
        
        Invariants:
            - Intent confidence is always between 0.0 and 1.0
            - Command parameters are type-safe and validated
            - Security boundaries are maintained for all command types
    """
    
    def __init__(self):
        self.intent_patterns: List[IntentPattern] = []
        self.entity_extractor = EntityExtractor()
        self.custom_patterns: Dict[str, IntentPattern] = {}
        self.processing_stats = {
            "total_processed": 0,
            "successful_intents": 0,
            "failed_intents": 0,
            "average_confidence": 0.0,
            "most_common_intents": {}
        }
        
        # Initialize default intent patterns
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Initialize default voice command intent patterns."""
        
        # Automation trigger patterns
        self.intent_patterns.extend([
            IntentPattern(
                intent_name="trigger_automation",
                category=IntentCategory.AUTOMATION,
                command_type=VoiceCommandType.AUTOMATION_TRIGGER,
                patterns=[
                    r"(?:run|execute|trigger|start)\s+(?:automation|macro|workflow)\s+(?P<automation_name>[\w\s]+)",
                    r"(?:activate|launch)\s+(?P<automation_name>[\w\s]+)\s+(?:automation|macro)",
                    r"(?P<automation_name>[\w\s]+)\s+(?:now|please|go)"
                ],
                required_entities=["automation_name"],
                examples=[
                    "run automation morning routine",
                    "execute macro email backup", 
                    "trigger workflow data sync"
                ]
            ),
            
            IntentPattern(
                intent_name="create_automation",
                category=IntentCategory.CREATION,
                command_type=VoiceCommandType.CUSTOM_WORKFLOW,
                patterns=[
                    r"create\s+(?:automation|macro|workflow)\s+(?:called|named)\s+(?P<name>[\w\s]+)",
                    r"new\s+(?:automation|macro)\s+for\s+(?P<purpose>[\w\s]+)",
                    r"make\s+(?:a\s+)?(?:automation|macro)\s+(?:to|that)\s+(?P<action>[\w\s]+)"
                ],
                required_entities=["name", "purpose", "action"],
                priority=CommandPriority.LOW,
                examples=[
                    "create automation called morning setup",
                    "new macro for email processing",
                    "make automation to backup files"
                ]
            ),
        ])
        
        # Application control patterns
        self.intent_patterns.extend([
            IntentPattern(
                intent_name="open_application",
                category=IntentCategory.CONTROL,
                command_type=VoiceCommandType.APPLICATION_CONTROL,
                patterns=[
                    r"(?:open|launch|start)\s+(?P<application>\w+)",
                    r"(?:go\s+to|switch\s+to)\s+(?P<application>\w+)",
                    r"(?P<application>\w+)\s+(?:please|now)"
                ],
                required_entities=["application"],
                examples=[
                    "open safari",
                    "launch mail",
                    "switch to finder"
                ]
            ),
            
            IntentPattern(
                intent_name="close_application",
                category=IntentCategory.CONTROL,
                command_type=VoiceCommandType.APPLICATION_CONTROL,
                patterns=[
                    r"(?:close|quit|exit)\s+(?P<application>\w+)",
                    r"(?:shut\s+down|terminate)\s+(?P<application>\w+)"
                ],
                required_entities=["application"],
                examples=[
                    "close safari",
                    "quit mail",
                    "exit finder"
                ]
            ),
        ])
        
        # System control patterns
        self.intent_patterns.extend([
            IntentPattern(
                intent_name="system_volume",
                category=IntentCategory.CONTROL,
                command_type=VoiceCommandType.SYSTEM_CONTROL,
                patterns=[
                    r"(?:set|change)\s+volume\s+(?:to\s+)?(?P<volume>\d+)",
                    r"(?:turn\s+)?volume\s+(?P<direction>up|down)",
                    r"(?P<action>mute|unmute)\s+(?:volume|sound|audio)?"
                ],
                optional_entities=["volume", "direction", "action"],
                examples=[
                    "set volume to 50",
                    "turn volume up",
                    "mute sound"
                ]
            ),
            
            IntentPattern(
                intent_name="system_display",
                category=IntentCategory.CONTROL,
                command_type=VoiceCommandType.SYSTEM_CONTROL,
                patterns=[
                    r"(?:set|change)\s+brightness\s+(?:to\s+)?(?P<brightness>\d+)",
                    r"(?:turn\s+)?brightness\s+(?P<direction>up|down)",
                    r"(?P<action>lock|unlock)\s+screen"
                ],
                optional_entities=["brightness", "direction", "action"],
                examples=[
                    "set brightness to 75",
                    "turn brightness down",
                    "lock screen"
                ]
            ),
        ])
        
        # Text input patterns
        self.intent_patterns.extend([
            IntentPattern(
                intent_name="type_text",
                category=IntentCategory.INPUT,
                command_type=VoiceCommandType.TEXT_INPUT,
                patterns=[
                    r"(?:type|write|input)\s+(?P<text>.+)",
                    r"(?:say|dictate)\s+(?P<text>.+)",
                    r"insert\s+(?:text\s+)?(?P<text>.+)"
                ],
                required_entities=["text"],
                examples=[
                    "type hello world",
                    "write meeting notes",
                    "dictate this message"
                ]
            ),
        ])
        
        # Navigation patterns
        self.intent_patterns.extend([
            IntentPattern(
                intent_name="navigate_direction",
                category=IntentCategory.NAVIGATION,
                command_type=VoiceCommandType.NAVIGATION,
                patterns=[
                    r"(?:go|move|scroll)\s+(?P<direction>up|down|left|right)",
                    r"(?:page|screen)\s+(?P<direction>up|down)",
                    r"(?P<direction>next|previous)\s+(?:page|tab|window)?"
                ],
                required_entities=["direction"],
                examples=[
                    "go up",
                    "scroll down", 
                    "next page"
                ]
            ),
        ])
        
        # File operations patterns
        self.intent_patterns.extend([
            IntentPattern(
                intent_name="file_operation",
                category=IntentCategory.CONTROL,
                command_type=VoiceCommandType.SYSTEM_CONTROL,
                patterns=[
                    r"(?P<action>open|save|copy|delete|move)\s+(?:file\s+)?(?P<file_name>[\w\s\.]+)",
                    r"(?P<action>create|new)\s+(?:file|document)\s+(?:called\s+)?(?P<file_name>[\w\s\.]+)",
                    r"(?P<action>backup|archive)\s+(?P<file_name>[\w\s\.]+)"
                ],
                required_entities=["action"],
                optional_entities=["file_name"],
                examples=[
                    "open file report.pdf",
                    "save document notes.txt",
                    "create new file meeting.doc"
                ]
            ),
        ])
        
        logger.info(f"Initialized {len(self.intent_patterns)} default intent patterns")
    
    @require(lambda self, recognized_text: len(recognized_text.strip()) > 0)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def process_intent(
        self,
        recognized_text: str,
        speaker_id: Optional[SpeakerId] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Either[VoiceCommandError, VoiceCommand]:
        """
        Process recognized speech text to extract intent and create voice command.
        
        Performance:
            - <100ms intent pattern matching
            - <50ms entity extraction and validation
            - <50ms command object creation and security validation
        """
        try:
            start_time = datetime.now(UTC)
            
            # Clean and normalize input text
            cleaned_text = self._clean_text(recognized_text)
            if not cleaned_text:
                return Either.error(VoiceCommandError.intent_not_recognized(recognized_text))
            
            # Find matching intent pattern
            intent_result = self._match_intent_pattern(cleaned_text)
            if intent_result.is_error():
                return intent_result
            
            intent_pattern, confidence = intent_result.value
            
            # Extract entities from text
            entities = self.entity_extractor.extract_entities(cleaned_text)
            pattern_entities = intent_pattern.extract_entities(cleaned_text)
            entities.update(pattern_entities)
            
            # Validate required entities
            missing_entities = self._validate_required_entities(intent_pattern, entities)
            if missing_entities:
                return Either.error(VoiceCommandError.intent_not_recognized(
                    f"Missing required entities: {missing_entities}"
                ))
            
            # Create voice command
            command_id = create_voice_command_id()
            voice_command = VoiceCommand(
                command_id=command_id,
                command_type=intent_pattern.command_type,
                intent=intent_pattern.intent_name,
                parameters=entities,
                original_text=recognized_text,
                confidence=confidence,
                priority=intent_pattern.priority,
                speaker_id=speaker_id,
                requires_confirmation=intent_pattern.requires_confirmation
            )
            
            # Security validation
            security_result = validate_voice_command_security(voice_command)
            if security_result.is_error():
                return Either.error(VoiceCommandError.unsafe_command_detected(
                    voice_command.intent
                ))
            
            # Update processing statistics
            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            self._update_processing_stats(intent_pattern.intent_name, confidence, True)
            
            logger.info(f"Intent processed successfully: '{intent_pattern.intent_name}' "
                       f"(confidence: {confidence:.2f}, time: {processing_time:.0f}ms)")
            
            return Either.success(voice_command)
            
        except Exception as e:
            self._update_processing_stats("unknown", 0.0, False)
            error_msg = f"Intent processing failed: {str(e)}"
            logger.error(error_msg)
            return Either.error(VoiceCommandError.intent_not_recognized(str(e)))
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove extra whitespace and normalize
        cleaned = ' '.join(text.strip().split())
        
        # Remove common filler words
        filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know']
        words = cleaned.split()
        filtered_words = [word for word in words if word.lower() not in filler_words]
        
        return ' '.join(filtered_words)
    
    def _match_intent_pattern(self, text: str) -> Either[VoiceCommandError, Tuple[IntentPattern, float]]:
        """Match text against intent patterns and return best match with confidence."""
        matches = []
        
        # Check all patterns
        for pattern in self.intent_patterns:
            if pattern.matches_text(text):
                # Calculate confidence based on pattern specificity and match quality
                confidence = self._calculate_pattern_confidence(pattern, text)
                matches.append((pattern, confidence))
        
        # Check custom patterns
        for pattern in self.custom_patterns.values():
            if pattern.matches_text(text):
                confidence = self._calculate_pattern_confidence(pattern, text)
                matches.append((pattern, confidence))
        
        if not matches:
            return Either.error(VoiceCommandError.intent_not_recognized(text))
        
        # Return highest confidence match
        best_match = max(matches, key=lambda x: x[1])
        return Either.success(best_match)
    
    def _calculate_pattern_confidence(self, pattern: IntentPattern, text: str) -> float:
        """Calculate confidence score for pattern match."""
        base_confidence = 0.7
        
        # Bonus for exact keyword matches
        text_lower = text.lower()
        keyword_bonus = 0.0
        
        for pattern_text in pattern.patterns:
            # Extract keywords from pattern (simple approach)
            keywords = re.findall(r'\b[a-z]{3,}\b', pattern_text.lower())
            for keyword in keywords:
                if keyword in text_lower:
                    keyword_bonus += 0.05
        
        # Bonus for entity extraction success
        entities = pattern.extract_entities(text)
        entity_bonus = len(entities) * 0.02
        
        # Penalty for missing required entities
        missing_penalty = 0.0
        for required_entity in pattern.required_entities:
            if required_entity not in entities:
                missing_penalty += 0.1
        
        # Calculate final confidence
        confidence = base_confidence + keyword_bonus + entity_bonus - missing_penalty
        return max(0.0, min(1.0, confidence))
    
    def _validate_required_entities(self, pattern: IntentPattern, entities: Dict[str, Any]) -> List[str]:
        """Validate that all required entities are present."""
        missing = []
        for required_entity in pattern.required_entities:
            if required_entity not in entities or not entities[required_entity]:
                missing.append(required_entity)
        return missing
    
    def _update_processing_stats(self, intent_name: str, confidence: float, success: bool):
        """Update intent processing statistics."""
        self.processing_stats["total_processed"] += 1
        
        if success:
            self.processing_stats["successful_intents"] += 1
            
            # Update average confidence
            total_successful = self.processing_stats["successful_intents"]
            current_avg = self.processing_stats["average_confidence"]
            new_avg = ((current_avg * (total_successful - 1)) + confidence) / total_successful
            self.processing_stats["average_confidence"] = new_avg
            
            # Track most common intents
            if intent_name not in self.processing_stats["most_common_intents"]:
                self.processing_stats["most_common_intents"][intent_name] = 0
            self.processing_stats["most_common_intents"][intent_name] += 1
        else:
            self.processing_stats["failed_intents"] += 1
    
    async def add_custom_pattern(
        self,
        pattern_id: str,
        intent_pattern: IntentPattern
    ) -> Either[VoiceCommandError, None]:
        """Add custom intent pattern for personalized commands."""
        try:
            # Validate pattern
            if not intent_pattern.patterns:
                return Either.error(VoiceCommandError.intent_not_recognized(
                    "Intent pattern must have at least one pattern"
                ))
            
            # Test pattern compilation
            for pattern_text in intent_pattern.patterns:
                try:
                    re.compile(pattern_text, re.IGNORECASE)
                except re.error as e:
                    return Either.error(VoiceCommandError.intent_not_recognized(
                        f"Invalid regex pattern: {str(e)}"
                    ))
            
            self.custom_patterns[pattern_id] = intent_pattern
            logger.info(f"Custom intent pattern added: {pattern_id}")
            
            return Either.success(None)
            
        except Exception as e:
            return Either.error(VoiceCommandError.intent_not_recognized(f"Failed to add custom pattern: {str(e)}"))
    
    async def remove_custom_pattern(self, pattern_id: str) -> Either[VoiceCommandError, None]:
        """Remove custom intent pattern."""
        try:
            if pattern_id not in self.custom_patterns:
                return Either.error(VoiceCommandError.intent_not_recognized(
                    f"Custom pattern not found: {pattern_id}"
                ))
            
            del self.custom_patterns[pattern_id]
            logger.info(f"Custom intent pattern removed: {pattern_id}")
            
            return Either.success(None)
            
        except Exception as e:
            return Either.error(VoiceCommandError.intent_not_recognized(f"Failed to remove custom pattern: {str(e)}"))
    
    async def get_available_intents(self) -> List[Dict[str, Any]]:
        """Get list of available intent patterns with examples."""
        intents = []
        
        for pattern in self.intent_patterns:
            intents.append({
                "intent_name": pattern.intent_name,
                "category": pattern.category.value,
                "command_type": pattern.command_type.value,
                "required_entities": pattern.required_entities,
                "optional_entities": pattern.optional_entities,
                "priority": pattern.priority.value,
                "requires_confirmation": pattern.requires_confirmation,
                "examples": pattern.examples
            })
        
        for pattern_id, pattern in self.custom_patterns.items():
            intents.append({
                "intent_name": pattern.intent_name,
                "category": pattern.category.value,
                "command_type": pattern.command_type.value,
                "required_entities": pattern.required_entities,
                "optional_entities": pattern.optional_entities,
                "priority": pattern.priority.value,
                "requires_confirmation": pattern.requires_confirmation,
                "examples": pattern.examples,
                "custom_pattern_id": pattern_id
            })
        
        return intents
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get intent processing statistics."""
        stats = self.processing_stats.copy()
        stats["total_patterns"] = len(self.intent_patterns) + len(self.custom_patterns)
        stats["custom_patterns"] = len(self.custom_patterns)
        
        return stats
    
    async def configure_commands(self, configuration) -> Either[VoiceCommandError, Dict[str, Any]]:
        """Configure command mappings."""
        try:
            # Add custom command mappings
            for command_phrase, automation_target in configuration.command_mappings.items():
                pattern_id = f"custom_{len(self.custom_patterns)}"
                intent_pattern = create_simple_intent_pattern(
                    intent_name=f"custom_automation_{pattern_id}",
                    trigger_phrases=[command_phrase],
                    command_type=VoiceCommandType.AUTOMATION_TRIGGER
                )
                await self.add_custom_pattern(pattern_id, intent_pattern)
            
            result = {
                "commands_configured": len(configuration.command_mappings),
                "total_patterns": len(self.intent_patterns) + len(self.custom_patterns)
            }
            
            return Either.success(result)
        except Exception as e:
            return Either.error(VoiceCommandError.intent_not_recognized(f"Command configuration failed: {str(e)}"))


# Alias for compatibility
VoiceIntentProcessor = IntentProcessor


# Helper functions for intent processing
def create_simple_intent_pattern(
    intent_name: str,
    trigger_phrases: List[str],
    command_type: VoiceCommandType = VoiceCommandType.AUTOMATION_TRIGGER,
    category: IntentCategory = IntentCategory.AUTOMATION
) -> IntentPattern:
    """Create simple intent pattern from trigger phrases."""
    # Convert phrases to regex patterns
    patterns = []
    for phrase in trigger_phrases:
        # Escape special regex characters and make case insensitive
        escaped = re.escape(phrase.lower())
        # Allow for slight variations
        pattern = escaped.replace(r'\ ', r'\s+')
        patterns.append(f"\\b{pattern}\\b")
    
    return IntentPattern(
        intent_name=intent_name,
        category=category,
        command_type=command_type,
        patterns=patterns,
        examples=trigger_phrases
    )


def create_parameterized_intent_pattern(
    intent_name: str,
    pattern_template: str,
    required_params: List[str],
    command_type: VoiceCommandType = VoiceCommandType.AUTOMATION_TRIGGER,
    category: IntentCategory = IntentCategory.AUTOMATION
) -> IntentPattern:
    """Create intent pattern with named parameters."""
    return IntentPattern(
        intent_name=intent_name,
        category=category,
        command_type=command_type,
        patterns=[pattern_template],
        required_entities=required_params,
        examples=[pattern_template.replace(r'(?P<\w+>[\w\s]+)', '[parameter]')]
    )