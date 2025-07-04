"""
NLP Architecture - TASK_60 Phase 1 Core Implementation

Natural language processing type definitions and architectural framework.
Provides comprehensive types, enums, and utilities for NLP operations and command interpretation.

Architecture: Branded Types + Design by Contract + AI Integration + Language Processing
Performance: <100ms type operations, <200ms validation, <500ms complex analysis
Security: Safe text processing, validated inputs, comprehensive sanitization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, NewType
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import re
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure


# Branded Types for Type Safety
TextContent = NewType('TextContent', str)
Intent = NewType('Intent', str)
Entity = NewType('Entity', str)
ConversationId = NewType('ConversationId', str)
LanguageCode = NewType('LanguageCode', str)
CommandId = NewType('CommandId', str)
ContextId = NewType('ContextId', str)


def create_text_content(text: str) -> TextContent:
    """Create validated text content with sanitization."""
    if not text or len(text.strip()) == 0:
        raise ValueError("Text content cannot be empty")
    if len(text) > 100000:  # 100K character limit
        raise ValueError("Text content exceeds maximum length")
    
    # Basic sanitization - remove control characters except whitespace
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return TextContent(sanitized.strip())


def create_intent(intent_name: str) -> Intent:
    """Create validated intent identifier."""
    if not intent_name or not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', intent_name):
        raise ValueError("Intent must be a valid identifier")
    return Intent(intent_name.lower())


def create_entity(entity_value: str) -> Entity:
    """Create validated entity value."""
    if not entity_value or len(entity_value.strip()) == 0:
        raise ValueError("Entity value cannot be empty")
    return Entity(entity_value.strip())


def create_conversation_id() -> ConversationId:
    """Create unique conversation identifier."""
    import uuid
    return ConversationId(f"conv_{uuid.uuid4().hex[:12]}")


def create_command_id() -> CommandId:
    """Create unique command identifier."""
    import uuid
    return CommandId(f"cmd_{uuid.uuid4().hex[:8]}")


def create_context_id() -> ContextId:
    """Create unique context identifier."""
    import uuid
    return ContextId(f"ctx_{uuid.uuid4().hex[:8]}")


class NLPOperation(Enum):
    """Types of natural language processing operations."""
    INTENT_RECOGNITION = "intent_recognition"
    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_CLASSIFICATION = "text_classification"
    COMMAND_PARSING = "command_parsing"
    CONVERSATION_PROCESSING = "conversation_processing"
    LANGUAGE_DETECTION = "language_detection"
    TEXT_SUMMARIZATION = "text_summarization"
    TRANSLATION = "translation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"


class IntentCategory(Enum):
    """Categories of user intents."""
    AUTOMATION_COMMAND = "automation_command"
    WORKFLOW_CREATION = "workflow_creation"
    WORKFLOW_MODIFICATION = "workflow_modification"
    INFORMATION_REQUEST = "information_request"
    TROUBLESHOOTING = "troubleshooting"
    CONFIGURATION = "configuration"
    HELP_REQUEST = "help_request"
    FEEDBACK = "feedback"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """Types of entities that can be extracted."""
    APPLICATION = "application"
    FILE_PATH = "file_path"
    URL = "url"
    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"
    NUMBER = "number"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    AUTOMATION_ACTION = "automation_action"
    TRIGGER_EVENT = "trigger_event"
    CONDITION = "condition"
    VARIABLE = "variable"
    HOTKEY = "hotkey"
    CUSTOM = "custom"


class SentimentType(Enum):
    """Types of sentiment analysis results."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for NLP results."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


class ProcessingMode(Enum):
    """Modes for NLP processing."""
    FAST = "fast"              # Quick processing, lower accuracy
    BALANCED = "balanced"      # Balanced speed and accuracy
    ACCURATE = "accurate"      # Slower processing, higher accuracy
    COMPREHENSIVE = "comprehensive"  # Full analysis with all features


class ConversationMode(Enum):
    """Modes for conversational interfaces."""
    CREATION = "creation"      # Creating new automations
    MODIFICATION = "modification"  # Modifying existing automations
    TROUBLESHOOTING = "troubleshooting"  # Solving problems
    GUIDANCE = "guidance"      # Learning and help
    EXPLORATION = "exploration"  # Exploring capabilities
    FEEDBACK = "feedback"      # Providing feedback


@dataclass(frozen=True)
class NLPError(Exception):
    """Base class for NLP processing errors."""
    message: str
    error_code: str
    operation: Optional[NLPOperation] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntentRecognitionError(NLPError):
    """Error in intent recognition processing."""
    pass


@dataclass(frozen=True)
class EntityExtractionError(NLPError):
    """Error in entity extraction processing."""
    pass


@dataclass(frozen=True)
class ConversationError(NLPError):
    """Error in conversation processing."""
    pass


@dataclass(frozen=True)
class LanguageModelError(NLPError):
    """Error in language model operations."""
    pass


@dataclass(frozen=True)
class ExtractedEntity:
    """Entity extracted from text."""
    entity_id: str
    entity_type: EntityType
    value: Entity
    confidence: float
    start_position: int
    end_position: int
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.start_position < 0 or self.end_position <= self.start_position:
            raise ValueError("Invalid entity position")


@dataclass(frozen=True)
class RecognizedIntent:
    """Intent recognized from user input."""
    intent: Intent
    category: IntentCategory
    confidence: float
    entities: List[ExtractedEntity] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    context_requirements: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class SentimentAnalysis:
    """Sentiment analysis result."""
    sentiment: SentimentType
    confidence: float
    polarity: float  # -1.0 (negative) to 1.0 (positive)
    subjectivity: float  # 0.0 (objective) to 1.0 (subjective)
    emotions: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not (-1.0 <= self.polarity <= 1.0):
            raise ValueError("Polarity must be between -1.0 and 1.0")
        if not (0.0 <= self.subjectivity <= 1.0):
            raise ValueError("Subjectivity must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ProcessedCommand:
    """Natural language command processed into structured format."""
    command_id: CommandId
    original_text: TextContent
    recognized_intent: RecognizedIntent
    extracted_entities: List[ExtractedEntity]
    sentiment: Optional[SentimentAnalysis] = None
    automation_actions: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    alternatives: List['ProcessedCommand'] = field(default_factory=list)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ConversationContext:
    """Context for conversational interactions."""
    conversation_id: ConversationId
    context_id: ContextId
    mode: ConversationMode
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_topic: Optional[str] = None
    skill_level: str = "intermediate"  # beginner, intermediate, expert
    language: LanguageCode = LanguageCode("en")
    session_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_interaction: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if self.skill_level not in ["beginner", "intermediate", "expert"]:
            raise ValueError("Invalid skill level")


@dataclass(frozen=True)
class ConversationResponse:
    """Response in a conversational interaction."""
    response_id: str
    conversation_id: ConversationId
    response_text: str
    response_type: str  # answer, clarification, suggestion, confirmation, error
    suggestions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    automation_context: Optional[str] = None
    requires_action: bool = False
    confidence: float = 1.0
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.response_type not in ["answer", "clarification", "suggestion", "confirmation", "error"]:
            raise ValueError("Invalid response type")


@dataclass(frozen=True)
class LanguageModel:
    """Language model configuration and metadata."""
    model_id: str
    model_name: str
    model_type: str  # transformer, statistical, rule_based, hybrid
    supported_languages: List[LanguageCode]
    capabilities: List[NLPOperation]
    max_input_length: int
    processing_speed: str  # fast, medium, slow
    accuracy_level: str  # basic, standard, high, premium
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.max_input_length <= 0:
            raise ValueError("Max input length must be positive")
        if self.processing_speed not in ["fast", "medium", "slow"]:
            raise ValueError("Invalid processing speed")
        if self.accuracy_level not in ["basic", "standard", "high", "premium"]:
            raise ValueError("Invalid accuracy level")


@dataclass(frozen=True)
class NLPProcessingRequest:
    """Request for NLP processing."""
    request_id: str
    operation: NLPOperation
    text_input: TextContent
    language: LanguageCode = LanguageCode("en")
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    context: Optional[ConversationContext] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    
    def __post_init__(self):
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")


@dataclass(frozen=True)
class NLPProcessingResult:
    """Result from NLP processing."""
    result_id: str
    request_id: str
    operation: NLPOperation
    success: bool
    processing_time_ms: float
    recognized_intents: List[RecognizedIntent] = field(default_factory=list)
    extracted_entities: List[ExtractedEntity] = field(default_factory=list)
    sentiment_analysis: Optional[SentimentAnalysis] = None
    processed_commands: List[ProcessedCommand] = field(default_factory=list)
    conversation_response: Optional[ConversationResponse] = None
    confidence_score: float = 0.0
    errors: List[NLPError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        if self.processing_time_ms < 0:
            raise ValueError("Processing time cannot be negative")


# Utility Functions
def determine_confidence_level(confidence: float) -> ConfidenceLevel:
    """Determine confidence level from numeric confidence."""
    if confidence >= 0.8:
        return ConfidenceLevel.VERY_HIGH
    elif confidence >= 0.6:
        return ConfidenceLevel.HIGH
    elif confidence >= 0.4:
        return ConfidenceLevel.MEDIUM
    elif confidence >= 0.2:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.VERY_LOW


def validate_text_input(text: str, max_length: int = 10000) -> Either[NLPError, TextContent]:
    """Validate and sanitize text input for NLP processing."""
    try:
        if not text:
            return Either.left(NLPError("Empty text input", "EMPTY_INPUT"))
        
        if len(text) > max_length:
            return Either.left(NLPError(f"Text exceeds maximum length of {max_length}", "TEXT_TOO_LONG"))
        
        # Check for potential malicious content
        malicious_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'data:text/html',  # Data URLs
            r'vbscript:',  # VBScript
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return Either.left(NLPError("Potentially malicious content detected", "MALICIOUS_CONTENT"))
        
        return Either.right(create_text_content(text))
        
    except Exception as e:
        return Either.left(NLPError(f"Text validation failed: {str(e)}", "VALIDATION_ERROR"))


def extract_language_code(language_input: str) -> Either[NLPError, LanguageCode]:
    """Extract and validate language code."""
    try:
        # Convert to lowercase and extract first 2 characters
        lang_code = language_input.lower()[:2]
        
        # Basic validation for common language codes
        valid_codes = [
            "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
            "ar", "hi", "th", "vi", "pl", "nl", "sv", "da", "no", "fi"
        ]
        
        if lang_code not in valid_codes:
            return Either.left(NLPError(f"Unsupported language code: {lang_code}", "INVALID_LANGUAGE"))
        
        return Either.right(LanguageCode(lang_code))
        
    except Exception as e:
        return Either.left(NLPError(f"Language code extraction failed: {str(e)}", "LANGUAGE_ERROR"))


def merge_entities(entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
    """Merge overlapping entities, keeping the one with higher confidence."""
    if not entities:
        return entities
    
    # Sort by start position
    sorted_entities = sorted(entities, key=lambda e: e.start_position)
    merged = []
    
    for entity in sorted_entities:
        if not merged:
            merged.append(entity)
            continue
        
        last_entity = merged[-1]
        
        # Check for overlap
        if entity.start_position < last_entity.end_position:
            # Overlap detected, keep the one with higher confidence
            if entity.confidence > last_entity.confidence:
                merged[-1] = entity
        else:
            merged.append(entity)
    
    return merged


def calculate_intent_similarity(intent1: RecognizedIntent, intent2: RecognizedIntent) -> float:
    """Calculate similarity between two recognized intents."""
    # Compare intent names
    name_similarity = 1.0 if intent1.intent == intent2.intent else 0.0
    
    # Compare categories
    category_similarity = 1.0 if intent1.category == intent2.category else 0.0
    
    # Compare confidence scores
    confidence_diff = abs(intent1.confidence - intent2.confidence)
    confidence_similarity = 1.0 - confidence_diff
    
    # Compare number of entities
    entity_count_diff = abs(len(intent1.entities) - len(intent2.entities))
    entity_similarity = 1.0 / (1.0 + entity_count_diff)
    
    # Weighted average
    similarity = (
        name_similarity * 0.4 +
        category_similarity * 0.3 +
        confidence_similarity * 0.2 +
        entity_similarity * 0.1
    )
    
    return similarity


def is_automation_related(text: str) -> bool:
    """Check if text is related to automation commands."""
    automation_keywords = [
        "automate", "automation", "macro", "script", "workflow", "trigger",
        "action", "execute", "run", "launch", "open", "close", "click",
        "type", "press", "shortcut", "hotkey", "keyboard", "mouse",
        "window", "application", "app", "file", "folder", "system",
        "schedule", "timer", "condition", "if", "then", "else", "loop",
        "repeat", "pause", "wait", "delay", "notification", "alert"
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in automation_keywords)


@require(lambda processing_result: isinstance(processing_result, NLPProcessingResult))
def validate_processing_result(processing_result: NLPProcessingResult) -> bool:
    """Validate NLP processing result integrity."""
    # Check confidence scores
    for intent in processing_result.recognized_intents:
        if not (0.0 <= intent.confidence <= 1.0):
            return False
    
    for entity in processing_result.extracted_entities:
        if not (0.0 <= entity.confidence <= 1.0):
            return False
    
    # Check sentiment analysis if present
    if processing_result.sentiment_analysis:
        sentiment = processing_result.sentiment_analysis
        if not (0.0 <= sentiment.confidence <= 1.0):
            return False
        if not (-1.0 <= sentiment.polarity <= 1.0):
            return False
        if not (0.0 <= sentiment.subjectivity <= 1.0):
            return False
    
    # Check processing time
    if processing_result.processing_time_ms < 0:
        return False
    
    return True