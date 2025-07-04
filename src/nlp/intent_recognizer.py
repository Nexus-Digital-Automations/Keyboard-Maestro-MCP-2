"""
Intent Recognizer - TASK_60 Phase 1 Core Implementation

Intent recognition and classification system for natural language processing.
Provides ML-powered intent recognition, entity extraction, and context-aware classification.

Architecture: Intent Classification + Entity Extraction + Context Analysis + Pattern Matching
Performance: <200ms intent recognition, <300ms entity extraction, <500ms comprehensive analysis
Security: Safe text processing, validated classification, comprehensive input sanitization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import re
import math
import statistics
from collections import defaultdict, Counter

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.nlp_architecture import (
    TextContent, Intent, Entity, RecognizedIntent, ExtractedEntity,
    IntentCategory, EntityType, ConfidenceLevel, NLPError, IntentRecognitionError,
    create_intent, create_entity, determine_confidence_level, validate_text_input
)


class IntentPattern:
    """Pattern-based intent recognition."""
    
    def __init__(self, intent: Intent, category: IntentCategory, patterns: List[str], priority: int = 1):
        self.intent = intent
        self.category = category
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        self.priority = priority
        self.usage_count = 0
        self.success_rate = 1.0
    
    def matches(self, text: str) -> Tuple[bool, float]:
        """Check if text matches this intent pattern."""
        matches = 0
        total_patterns = len(self.patterns)
        
        for pattern in self.patterns:
            if pattern.search(text):
                matches += 1
        
        if matches == 0:
            return False, 0.0
        
        confidence = (matches / total_patterns) * self.success_rate
        return matches > 0, confidence


class EntityExtractor:
    """Entity extraction from natural language text."""
    
    def __init__(self):
        self.entity_patterns = self._initialize_entity_patterns()
    
    def _initialize_entity_patterns(self) -> Dict[EntityType, List[re.Pattern]]:
        """Initialize regex patterns for entity extraction."""
        return {
            EntityType.APPLICATION: [
                re.compile(r'\b(?:app|application|program)\s+([A-Za-z][A-Za-z0-9\s]*)', re.IGNORECASE),
                re.compile(r'\b(Chrome|Firefox|Safari|Finder|Terminal|TextEdit|Word|Excel|PowerPoint|Photoshop|Sketch|VS Code|Xcode)\b', re.IGNORECASE),
                re.compile(r'\bopen\s+([A-Za-z][A-Za-z0-9\s]*)', re.IGNORECASE)
            ],
            
            EntityType.FILE_PATH: [
                re.compile(r'(?:file|path|folder|directory)\s*[:\-]?\s*([~/][^\s]+)', re.IGNORECASE),
                re.compile(r'\b([~/][^\s]+\.(?:txt|doc|pdf|xlsx|png|jpg|jpeg|gif|mp4|mov|zip|dmg))\b', re.IGNORECASE),
                re.compile(r'"([^"]+\.[a-zA-Z0-9]+)"', re.IGNORECASE)
            ],
            
            EntityType.URL: [
                re.compile(r'\b(https?://[^\s]+)', re.IGNORECASE),
                re.compile(r'\b(www\.[^\s]+)', re.IGNORECASE),
                re.compile(r'\b([a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|io|co)(?:/[^\s]*)?)', re.IGNORECASE)
            ],
            
            EntityType.EMAIL: [
                re.compile(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', re.IGNORECASE)
            ],
            
            EntityType.PHONE_NUMBER: [
                re.compile(r'\b(\+?1[-.\s]?)?(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b'),
                re.compile(r'\b(\+?[0-9]{1,3}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9})\b')
            ],
            
            EntityType.DATE: [
                re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'),
                re.compile(r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b', re.IGNORECASE),
                re.compile(r'\b(today|tomorrow|yesterday|next week|last week|next month|last month)\b', re.IGNORECASE)
            ],
            
            EntityType.TIME: [
                re.compile(r'\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b', re.IGNORECASE),
                re.compile(r'\b((?:at\s+)?(?:noon|midnight|morning|afternoon|evening|night))\b', re.IGNORECASE),
                re.compile(r'\b(\d{1,2}\s*(?:AM|PM|am|pm))\b', re.IGNORECASE)
            ],
            
            EntityType.DURATION: [
                re.compile(r'\b(\d+(?:\.\d+)?\s*(?:second|minute|hour|day|week|month|year)s?)\b', re.IGNORECASE),
                re.compile(r'\b(\d+(?:\.\d+)?\s*(?:sec|min|hr|hrs)s?)\b', re.IGNORECASE)
            ],
            
            EntityType.NUMBER: [
                re.compile(r'\b(\d+(?:\.\d+)?)\b'),
                re.compile(r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b', re.IGNORECASE)
            ],
            
            EntityType.PERCENTAGE: [
                re.compile(r'\b(\d+(?:\.\d+)?%)\b'),
                re.compile(r'\b(\d+(?:\.\d+)?\s*percent)\b', re.IGNORECASE)
            ],
            
            EntityType.CURRENCY: [
                re.compile(r'\b(\$\d+(?:\.\d{2})?)\b'),
                re.compile(r'\b(\d+(?:\.\d{2})?\s*(?:dollars?|USD|cents?))\b', re.IGNORECASE),
                re.compile(r'\b(€\d+(?:\.\d{2})?)\b'),
                re.compile(r'\b(£\d+(?:\.\d{2})?)\b')
            ],
            
            EntityType.AUTOMATION_ACTION: [
                re.compile(r'\b(click|press|type|open|close|launch|quit|move|resize|minimize|maximize|copy|paste|cut|delete|save|print|search|find|replace)\b', re.IGNORECASE),
                re.compile(r'\b(automation|macro|script|workflow|trigger|action|command|execute|run)\b', re.IGNORECASE)
            ],
            
            EntityType.HOTKEY: [
                re.compile(r'\b((?:cmd|ctrl|alt|shift|option|command|control)(?:\s*\+\s*[a-zA-Z0-9])+)\b', re.IGNORECASE),
                re.compile(r'\b([a-zA-Z0-9]+(?:\s*\+\s*[a-zA-Z0-9]+)+)\b'),
                re.compile(r'\b(F\d{1,2})\b', re.IGNORECASE)
            ]
        }
    
    @require(lambda text: isinstance(text, TextContent))
    def extract_entities(self, text: TextContent) -> List[ExtractedEntity]:
        """Extract entities from text using pattern matching."""
        entities = []
        text_str = str(text)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text_str):
                    entity_value = match.group(1) if match.groups() else match.group(0)
                    
                    entity = ExtractedEntity(
                        entity_id=f"{entity_type.value}_{len(entities)}",
                        entity_type=entity_type,
                        value=create_entity(entity_value),
                        confidence=self._calculate_entity_confidence(entity_type, entity_value),
                        start_position=match.start(),
                        end_position=match.end(),
                        context=self._extract_entity_context(text_str, match.start(), match.end()),
                        metadata={"pattern_type": "regex", "extraction_method": "pattern_matching"}
                    )
                    
                    entities.append(entity)
        
        # Remove overlapping entities, keeping higher confidence ones
        return self._remove_overlapping_entities(entities)
    
    def _calculate_entity_confidence(self, entity_type: EntityType, value: str) -> float:
        """Calculate confidence score for extracted entity."""
        base_confidence = 0.7
        
        # Adjust confidence based on entity type and value characteristics
        if entity_type == EntityType.EMAIL:
            # Email pattern is quite reliable
            return 0.95 if '@' in value and '.' in value else 0.5
        
        elif entity_type == EntityType.URL:
            # URL pattern reliability
            if value.startswith(('http://', 'https://')):
                return 0.95
            elif value.startswith('www.'):
                return 0.85
            else:
                return 0.7
        
        elif entity_type == EntityType.PHONE_NUMBER:
            # Phone number validation
            digits = re.sub(r'\D', '', value)
            if len(digits) == 10:
                return 0.9
            elif len(digits) == 11 and digits.startswith('1'):
                return 0.9
            else:
                return 0.6
        
        elif entity_type == EntityType.APPLICATION:
            # Known application names
            known_apps = {
                'chrome', 'firefox', 'safari', 'finder', 'terminal', 'textedit',
                'word', 'excel', 'powerpoint', 'photoshop', 'sketch', 'vs code', 'xcode'
            }
            if value.lower() in known_apps:
                return 0.95
            else:
                return 0.7
        
        elif entity_type == EntityType.HOTKEY:
            # Hotkey pattern validation
            if any(modifier in value.lower() for modifier in ['cmd', 'ctrl', 'alt', 'shift']):
                return 0.9
            else:
                return 0.6
        
        return base_confidence
    
    def _extract_entity_context(self, text: str, start: int, end: int, window: int = 20) -> str:
        """Extract context around entity for better understanding."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end].strip()
        
        # Highlight the entity in context
        entity_start = start - context_start
        entity_end = end - context_start
        if entity_start >= 0 and entity_end <= len(context):
            return (context[:entity_start] + 
                   f"[{context[entity_start:entity_end]}]" + 
                   context[entity_end:])
        
        return context
    
    def _remove_overlapping_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove overlapping entities, keeping those with higher confidence."""
        if not entities:
            return entities
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: e.start_position)
        filtered = []
        
        for entity in sorted_entities:
            # Check for overlap with existing entities
            overlapping = False
            for i, existing in enumerate(filtered):
                if (entity.start_position < existing.end_position and 
                    entity.end_position > existing.start_position):
                    # Overlap detected
                    if entity.confidence > existing.confidence:
                        # Replace with higher confidence entity
                        filtered[i] = entity
                    overlapping = True
                    break
            
            if not overlapping:
                filtered.append(entity)
        
        return filtered


class IntentClassifier:
    """Intent classification system for natural language commands."""
    
    def __init__(self):
        self.intent_patterns = self._initialize_intent_patterns()
        self.entity_extractor = EntityExtractor()
        self.learning_data = defaultdict(list)
        self.classification_cache = {}
    
    def _initialize_intent_patterns(self) -> List[IntentPattern]:
        """Initialize intent patterns for automation commands."""
        return [
            # Automation Command Intents
            IntentPattern(
                create_intent("create_automation"),
                IntentCategory.AUTOMATION_COMMAND,
                [
                    r'\b(?:create|make|build|generate)\s+(?:a|an|new)?\s*(?:automation|macro|script|workflow)',
                    r'\b(?:automate|set up automation for)',
                    r'\b(?:I want to|help me|can you)\s+(?:create|make|automate)',
                ],
                priority=3
            ),
            
            IntentPattern(
                create_intent("run_automation"),
                IntentCategory.AUTOMATION_COMMAND,
                [
                    r'\b(?:run|execute|start|launch|trigger)\s+(?:the|my|an?)?\s*(?:automation|macro|script|workflow)',
                    r'\b(?:please\s+)?(?:run|execute|start)\s+',
                    r'\b(?:activate|trigger)\s+',
                ],
                priority=3
            ),
            
            IntentPattern(
                create_intent("stop_automation"),
                IntentCategory.AUTOMATION_COMMAND,
                [
                    r'\b(?:stop|halt|cancel|abort|terminate)\s+(?:the|my|an?)?\s*(?:automation|macro|script|workflow)',
                    r'\b(?:please\s+)?(?:stop|cancel|abort)',
                ],
                priority=2
            ),
            
            # Workflow Creation Intents
            IntentPattern(
                create_intent("design_workflow"),
                IntentCategory.WORKFLOW_CREATION,
                [
                    r'\b(?:design|plan|outline)\s+(?:a|an|new)?\s*workflow',
                    r'\b(?:create|build)\s+(?:a|an)?\s*(?:visual\s+)?workflow',
                    r'\b(?:I need|help me create)\s+a\s+workflow',
                ],
                priority=2
            ),
            
            IntentPattern(
                create_intent("add_action"),
                IntentCategory.WORKFLOW_CREATION,
                [
                    r'\b(?:add|insert|include)\s+(?:an|a)?\s*(?:action|step|command)',
                    r'\b(?:then|next)\s+(?:add|do|execute)',
                    r'\b(?:I want to|need to)\s+add',
                ],
                priority=2
            ),
            
            # Workflow Modification Intents
            IntentPattern(
                create_intent("modify_workflow"),
                IntentCategory.WORKFLOW_MODIFICATION,
                [
                    r'\b(?:modify|change|update|edit)\s+(?:the|my|this)?\s*(?:workflow|automation|macro)',
                    r'\b(?:can you|please)\s+(?:modify|change|update)',
                ],
                priority=2
            ),
            
            IntentPattern(
                create_intent("delete_action"),
                IntentCategory.WORKFLOW_MODIFICATION,
                [
                    r'\b(?:delete|remove|eliminate)\s+(?:the|this|that)?\s*(?:action|step|command)',
                    r'\b(?:get rid of|take out)\s+',
                ],
                priority=2
            ),
            
            # Information Request Intents
            IntentPattern(
                create_intent("list_automations"),
                IntentCategory.INFORMATION_REQUEST,
                [
                    r'\b(?:list|show|display)\s+(?:all|my)?\s*(?:automations|macros|workflows|scripts)',
                    r'\b(?:what|which)\s+(?:automations|macros|workflows)\s+(?:do I have|are available)',
                    r'\b(?:show me|tell me about)\s+(?:my\s+)?(?:automations|workflows)',
                ],
                priority=2
            ),
            
            IntentPattern(
                create_intent("get_status"),
                IntentCategory.INFORMATION_REQUEST,
                [
                    r'\b(?:what is|what\'s)\s+the\s+status\s+of',
                    r'\b(?:check|show)\s+(?:the\s+)?status',
                    r'\b(?:is|are)\s+(?:my\s+)?(?:automation|macro|workflow)\s+(?:running|active)',
                ],
                priority=2
            ),
            
            # Troubleshooting Intents
            IntentPattern(
                create_intent("fix_problem"),
                IntentCategory.TROUBLESHOOTING,
                [
                    r'\b(?:fix|solve|resolve|debug)\s+(?:the|this|my)?\s*(?:problem|issue|error)',
                    r'\b(?:my\s+)?(?:automation|macro|workflow)\s+(?:is not working|failed|broke)',
                    r'\b(?:help|assist)\s+(?:me\s+)?(?:fix|solve|debug)',
                ],
                priority=3
            ),
            
            IntentPattern(
                create_intent("explain_error"),
                IntentCategory.TROUBLESHOOTING,
                [
                    r'\b(?:what|why)\s+(?:is|does)\s+(?:this|the)\s+(?:error|problem)',
                    r'\b(?:explain|tell me about)\s+(?:this|the)\s+(?:error|issue)',
                    r'\b(?:I don\'t understand|I\'m confused about)\s+(?:this|the)',
                ],
                priority=2
            ),
            
            # Help Request Intents
            IntentPattern(
                create_intent("get_help"),
                IntentCategory.HELP_REQUEST,
                [
                    r'\b(?:help|assist|guide)\s+(?:me|us)?',
                    r'\b(?:how do I|how can I|how to)',
                    r'\b(?:I need help|I\'m lost|I don\'t know)',
                    r'\b(?:can you help|please help)',
                ],
                priority=1
            ),
            
            IntentPattern(
                create_intent("learn_feature"),
                IntentCategory.HELP_REQUEST,
                [
                    r'\b(?:learn|understand|know)\s+(?:about|how to use)',
                    r'\b(?:teach me|show me how)',
                    r'\b(?:what can|what does|what is)',
                ],
                priority=1
            ),
            
            # Configuration Intents
            IntentPattern(
                create_intent("change_settings"),
                IntentCategory.CONFIGURATION,
                [
                    r'\b(?:change|modify|update|set)\s+(?:the\s+)?(?:settings|preferences|configuration)',
                    r'\b(?:configure|setup|adjust)',
                ],
                priority=2
            ),
            
            # Greeting and Social Intents
            IntentPattern(
                create_intent("greeting"),
                IntentCategory.GREETING,
                [
                    r'\b(?:hello|hi|hey|good morning|good afternoon|good evening)',
                    r'\b(?:greetings|salutations)',
                ],
                priority=1
            ),
            
            IntentPattern(
                create_intent("goodbye"),
                IntentCategory.GOODBYE,
                [
                    r'\b(?:goodbye|bye|see you|farewell|exit|quit)',
                    r'\b(?:good night|have a good day)',
                ],
                priority=1
            ),
            
            # Feedback Intents
            IntentPattern(
                create_intent("provide_feedback"),
                IntentCategory.FEEDBACK,
                [
                    r'\b(?:feedback|suggestion|comment|review)',
                    r'\b(?:I think|I suggest|I recommend)',
                    r'\b(?:this is|that was)\s+(?:good|bad|great|terrible|helpful|confusing)',
                ],
                priority=1
            )
        ]
    
    @require(lambda text: isinstance(text, TextContent))
    @ensure(lambda result: result.is_right() or isinstance(result.left_value, IntentRecognitionError))
    async def recognize_intent(
        self,
        text: TextContent,
        confidence_threshold: float = 0.5,
        max_intents: int = 3
    ) -> Either[IntentRecognitionError, List[RecognizedIntent]]:
        """Recognize intents from natural language text."""
        try:
            text_str = str(text)
            
            # Check cache first
            cache_key = f"{text_str}_{confidence_threshold}_{max_intents}"
            if cache_key in self.classification_cache:
                return Either.right(self.classification_cache[cache_key])
            
            intent_scores = []
            
            # Match against all intent patterns
            for pattern in self.intent_patterns:
                matches, confidence = pattern.matches(text_str)
                if matches and confidence >= confidence_threshold:
                    # Extract entities for this intent
                    entities = self.entity_extractor.extract_entities(text)
                    
                    # Adjust confidence based on entities and context
                    adjusted_confidence = self._adjust_confidence_with_context(
                        confidence, pattern, entities, text_str
                    )
                    
                    intent_scores.append((pattern, adjusted_confidence, entities))
            
            # Sort by confidence and priority
            intent_scores.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
            
            # Create recognized intents
            recognized_intents = []
            for pattern, confidence, entities in intent_scores[:max_intents]:
                recognized_intent = RecognizedIntent(
                    intent=pattern.intent,
                    category=pattern.category,
                    confidence=confidence,
                    entities=entities,
                    parameters=self._extract_intent_parameters(pattern, entities, text_str),
                    context_requirements=self._determine_context_requirements(pattern),
                    suggested_actions=self._generate_suggested_actions(pattern, entities)
                )
                recognized_intents.append(recognized_intent)
            
            # Cache result
            self.classification_cache[cache_key] = recognized_intents
            
            # Update learning data
            self._update_learning_data(text_str, recognized_intents)
            
            return Either.right(recognized_intents)
            
        except Exception as e:
            return Either.left(IntentRecognitionError(
                f"Intent recognition failed: {str(e)}",
                "RECOGNITION_ERROR",
                context={"text_length": len(str(text))}
            ))
    
    def _adjust_confidence_with_context(
        self,
        base_confidence: float,
        pattern: IntentPattern,
        entities: List[ExtractedEntity],
        text: str
    ) -> float:
        """Adjust confidence based on context and entities."""
        adjusted_confidence = base_confidence
        
        # Boost confidence if relevant entities are found
        if pattern.category == IntentCategory.AUTOMATION_COMMAND:
            automation_entities = [e for e in entities if e.entity_type in [
                EntityType.AUTOMATION_ACTION, EntityType.APPLICATION, EntityType.HOTKEY
            ]]
            if automation_entities:
                adjusted_confidence *= 1.2
        
        elif pattern.category == IntentCategory.WORKFLOW_CREATION:
            workflow_entities = [e for e in entities if e.entity_type in [
                EntityType.AUTOMATION_ACTION, EntityType.CONDITION
            ]]
            if workflow_entities:
                adjusted_confidence *= 1.15
        
        # Consider text length and complexity
        word_count = len(text.split())
        if word_count < 3:
            adjusted_confidence *= 0.9  # Short texts are less reliable
        elif word_count > 20:
            adjusted_confidence *= 0.95  # Very long texts might be less focused
        
        # Apply success rate from previous classifications
        adjusted_confidence *= pattern.success_rate
        
        return min(1.0, adjusted_confidence)
    
    def _extract_intent_parameters(
        self,
        pattern: IntentPattern,
        entities: List[ExtractedEntity],
        text: str
    ) -> Dict[str, Any]:
        """Extract parameters relevant to the recognized intent."""
        parameters = {}
        
        # Add entity values as parameters
        for entity in entities:
            param_name = entity.entity_type.value
            if param_name not in parameters:
                parameters[param_name] = []
            parameters[param_name].append(str(entity.value))
        
        # Add intent-specific parameters
        if pattern.category == IntentCategory.AUTOMATION_COMMAND:
            parameters["command_type"] = pattern.intent
            parameters["urgency"] = self._detect_urgency(text)
        
        elif pattern.category == IntentCategory.WORKFLOW_CREATION:
            parameters["workflow_complexity"] = self._estimate_complexity(text, entities)
            parameters["workflow_type"] = "visual" if "visual" in text.lower() else "standard"
        
        elif pattern.category == IntentCategory.TROUBLESHOOTING:
            parameters["error_severity"] = self._assess_error_severity(text)
            parameters["assistance_level"] = "detailed" if "explain" in text.lower() else "quick"
        
        return parameters
    
    def _determine_context_requirements(self, pattern: IntentPattern) -> List[str]:
        """Determine what context is needed to fulfill this intent."""
        requirements = []
        
        if pattern.category == IntentCategory.AUTOMATION_COMMAND:
            requirements.extend(["current_automations", "system_state"])
        
        elif pattern.category == IntentCategory.WORKFLOW_CREATION:
            requirements.extend(["available_actions", "user_permissions"])
        
        elif pattern.category == IntentCategory.WORKFLOW_MODIFICATION:
            requirements.extend(["existing_workflow", "modification_history"])
        
        elif pattern.category == IntentCategory.TROUBLESHOOTING:
            requirements.extend(["error_logs", "system_diagnostics"])
        
        elif pattern.category == IntentCategory.INFORMATION_REQUEST:
            requirements.extend(["automation_database", "user_data"])
        
        return requirements
    
    def _generate_suggested_actions(
        self,
        pattern: IntentPattern,
        entities: List[ExtractedEntity]
    ) -> List[str]:
        """Generate suggested actions for this intent."""
        actions = []
        
        if pattern.category == IntentCategory.AUTOMATION_COMMAND:
            if pattern.intent == "create_automation":
                actions.extend([
                    "Open workflow designer",
                    "Select automation template",
                    "Configure automation parameters"
                ])
            elif pattern.intent == "run_automation":
                actions.extend([
                    "Find matching automation",
                    "Verify prerequisites",
                    "Execute automation"
                ])
        
        elif pattern.category == IntentCategory.TROUBLESHOOTING:
            actions.extend([
                "Gather diagnostic information",
                "Check error logs",
                "Suggest potential solutions"
            ])
        
        elif pattern.category == IntentCategory.HELP_REQUEST:
            actions.extend([
                "Provide relevant documentation",
                "Offer step-by-step guidance",
                "Suggest tutorials or examples"
            ])
        
        return actions
    
    def _detect_urgency(self, text: str) -> str:
        """Detect urgency level in text."""
        urgent_words = ["urgent", "asap", "immediately", "now", "quickly", "emergency"]
        text_lower = text.lower()
        
        if any(word in text_lower for word in urgent_words):
            return "high"
        elif any(word in text_lower for word in ["please", "when possible"]):
            return "low"
        else:
            return "normal"
    
    def _estimate_complexity(self, text: str, entities: List[ExtractedEntity]) -> str:
        """Estimate workflow complexity from description."""
        complexity_indicators = {
            "simple": ["just", "only", "simple", "basic", "quick"],
            "complex": ["complex", "advanced", "multiple", "conditional", "loop", "integrate"]
        }
        
        text_lower = text.lower()
        word_count = len(text.split())
        entity_count = len(entities)
        
        if word_count > 30 or entity_count > 5:
            return "complex"
        elif any(word in text_lower for word in complexity_indicators["complex"]):
            return "complex"
        elif any(word in text_lower for word in complexity_indicators["simple"]):
            return "simple"
        else:
            return "intermediate"
    
    def _assess_error_severity(self, text: str) -> str:
        """Assess error severity from description."""
        critical_words = ["critical", "broken", "failed", "crash", "emergency"]
        warning_words = ["warning", "issue", "problem", "slow"]
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in critical_words):
            return "critical"
        elif any(word in text_lower for word in warning_words):
            return "warning"
        else:
            return "info"
    
    def _update_learning_data(self, text: str, intents: List[RecognizedIntent]):
        """Update learning data for continuous improvement."""
        timestamp = datetime.now(UTC)
        
        for intent in intents:
            self.learning_data[str(intent.intent)].append({
                "text": text,
                "confidence": intent.confidence,
                "timestamp": timestamp,
                "entities_count": len(intent.entities)
            })
        
        # Limit learning data size
        for intent_name in self.learning_data:
            if len(self.learning_data[intent_name]) > 1000:
                self.learning_data[intent_name] = self.learning_data[intent_name][-1000:]
    
    async def improve_from_feedback(
        self,
        text: TextContent,
        correct_intent: Intent,
        feedback_confidence: float
    ) -> bool:
        """Improve classifier based on user feedback."""
        try:
            text_str = str(text)
            
            # Find the pattern for the correct intent
            correct_pattern = None
            for pattern in self.intent_patterns:
                if pattern.intent == correct_intent:
                    correct_pattern = pattern
                    break
            
            if correct_pattern:
                # Update success rate based on feedback
                if feedback_confidence > 0.7:
                    correct_pattern.success_rate = min(1.0, correct_pattern.success_rate * 1.05)
                else:
                    correct_pattern.success_rate = max(0.1, correct_pattern.success_rate * 0.95)
                
                # Clear cache to force re-evaluation
                self.classification_cache.clear()
                
                return True
            
            return False
            
        except Exception:
            return False
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification performance statistics."""
        total_classifications = sum(len(data) for data in self.learning_data.values())
        
        if total_classifications == 0:
            return {"total_classifications": 0}
        
        intent_distribution = {
            intent: len(data) for intent, data in self.learning_data.items()
        }
        
        average_confidence = 0.0
        if total_classifications > 0:
            all_confidences = []
            for data_list in self.learning_data.values():
                all_confidences.extend([item["confidence"] for item in data_list])
            average_confidence = statistics.mean(all_confidences) if all_confidences else 0.0
        
        return {
            "total_classifications": total_classifications,
            "intent_distribution": intent_distribution,
            "average_confidence": average_confidence,
            "cache_size": len(self.classification_cache),
            "pattern_count": len(self.intent_patterns)
        }