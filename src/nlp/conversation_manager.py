"""
Conversation Manager - TASK_60 Phase 2 Core Implementation

Conversational automation interface management for natural language interactions.
Provides context-aware conversation handling, guidance, and automation assistance.

Architecture: Conversation Management + Context Tracking + Response Generation + Learning
Performance: <400ms conversation processing, <300ms context analysis, <200ms response generation
Security: Safe conversation handling, validated responses, comprehensive context sanitization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import logging
from collections import defaultdict, deque

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.nlp_architecture import (
    TextContent, ConversationId, ContextId, ConversationContext, ConversationResponse,
    ConversationMode, LanguageCode, NLPError, ConversationError,
    create_conversation_id, create_context_id, validate_text_input
)
from src.nlp.intent_recognizer import IntentClassifier
from src.nlp.command_processor import CommandProcessor


class ConversationState(Enum):
    """States of conversational interactions."""
    INITIAL = "initial"              # Just started
    GATHERING_INFO = "gathering_info"  # Collecting requirements
    CLARIFYING = "clarifying"        # Asking for clarification
    PROCESSING = "processing"        # Processing request
    CONFIRMING = "confirming"        # Confirming actions
    EXECUTING = "executing"          # Executing automation
    COMPLETED = "completed"          # Task completed
    ERROR = "error"                  # Error state
    IDLE = "idle"                    # Waiting for input


class ResponseType(Enum):
    """Types of conversation responses."""
    GREETING = "greeting"
    INFORMATION = "information"
    CLARIFICATION = "clarification"
    SUGGESTION = "suggestion"
    CONFIRMATION = "confirmation"
    INSTRUCTION = "instruction"
    ERROR = "error"
    SUCCESS = "success"
    GUIDANCE = "guidance"
    EXAMPLE = "example"


@dataclass
class ConversationSession:
    """Active conversation session with state management."""
    session_id: ConversationId
    context: ConversationContext
    state: ConversationState
    current_topic: Optional[str] = None
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    conversation_log: deque = field(default_factory=lambda: deque(maxlen=50))
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    skill_assessment: Dict[str, float] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    

@dataclass
class ResponseTemplate:
    """Template for generating conversation responses."""
    template_id: str
    response_type: ResponseType
    context_mode: ConversationMode
    templates: Dict[str, str]  # skill_level -> template
    follow_up_suggestions: List[str] = field(default_factory=list)
    required_entities: List[str] = field(default_factory=list)


class ConversationManager:
    """Conversational automation interface management system."""
    
    def __init__(self, intent_classifier: IntentClassifier, command_processor: CommandProcessor):
        self.intent_classifier = intent_classifier
        self.command_processor = command_processor
        self.active_sessions: Dict[ConversationId, ConversationSession] = {}
        self.response_templates = self._initialize_response_templates()
        self.conversation_history = defaultdict(list)
        self.learning_data = defaultdict(list)
    
    def _initialize_response_templates(self) -> Dict[str, ResponseTemplate]:
        """Initialize response templates for different conversation scenarios."""
        return {
            # Greeting Templates
            "greeting_creation": ResponseTemplate(
                template_id="greeting_creation",
                response_type=ResponseType.GREETING,
                context_mode=ConversationMode.CREATION,
                templates={
                    "beginner": "Hi! I'm here to help you create automations. What would you like to automate today? I can guide you through the process step by step.",
                    "intermediate": "Hello! I can help you create powerful automations. What task would you like to automate?",
                    "expert": "Welcome! Ready to build some automation workflows? What's your automation goal?"
                },
                follow_up_suggestions=[
                    "Tell me what you want to automate",
                    "Describe a repetitive task you'd like to simplify",
                    "Ask about automation capabilities"
                ]
            ),
            
            "greeting_modification": ResponseTemplate(
                template_id="greeting_modification",
                response_type=ResponseType.GREETING,
                context_mode=ConversationMode.MODIFICATION,
                templates={
                    "beginner": "Hi! I can help you modify or improve your existing automations. Which automation would you like to change?",
                    "intermediate": "Hello! Let's modify your automation. Which one needs updating?",
                    "expert": "Ready to enhance your automations? Which workflow are we optimizing?"
                },
                follow_up_suggestions=[
                    "List my existing automations",
                    "Show automation details",
                    "Explain modification options"
                ]
            ),
            
            # Information Templates
            "automation_list": ResponseTemplate(
                template_id="automation_list",
                response_type=ResponseType.INFORMATION,
                context_mode=ConversationMode.GUIDANCE,
                templates={
                    "beginner": "Here are your automations. Each one performs a specific task automatically when triggered.",
                    "intermediate": "Your current automation library:",
                    "expert": "Automation inventory:"
                }
            ),
            
            # Clarification Templates
            "unclear_intent": ResponseTemplate(
                template_id="unclear_intent",
                response_type=ResponseType.CLARIFICATION,
                context_mode=ConversationMode.CREATION,
                templates={
                    "beginner": "I'm not sure I understand what you'd like to automate. Could you describe it differently? For example, 'I want to automatically organize my files' or 'I want to launch my work apps every morning'.",
                    "intermediate": "Could you clarify what you want to automate? Please provide more details about the task or goal.",
                    "expert": "Need more specifics on the automation requirements. What's the target workflow?"
                },
                follow_up_suggestions=[
                    "Describe the task step by step",
                    "Tell me what triggers the automation",
                    "Explain the desired outcome"
                ]
            ),
            
            # Suggestion Templates
            "automation_suggestions": ResponseTemplate(
                template_id="automation_suggestions",
                response_type=ResponseType.SUGGESTION,
                context_mode=ConversationMode.CREATION,
                templates={
                    "beginner": "Based on what you described, I suggest creating an automation that {suggestion}. This would save you time and effort. Would you like me to help set this up?",
                    "intermediate": "I recommend automating {suggestion}. This approach would be efficient and reliable. Shall we build this?",
                    "expert": "Optimal automation strategy: {suggestion}. Implementation approach: {approach}. Proceed?"
                }
            ),
            
            # Instruction Templates
            "next_steps": ResponseTemplate(
                template_id="next_steps",
                response_type=ResponseType.INSTRUCTION,
                context_mode=ConversationMode.CREATION,
                templates={
                    "beginner": "Great! Here's what we'll do next: {steps}. Don't worry, I'll guide you through each step.",
                    "intermediate": "Next steps: {steps}. Let me know when you're ready to proceed.",
                    "expert": "Implementation plan: {steps}. Ready to execute?"
                }
            ),
            
            # Confirmation Templates
            "action_confirmation": ResponseTemplate(
                template_id="action_confirmation",
                response_type=ResponseType.CONFIRMATION,
                context_mode=ConversationMode.CREATION,
                templates={
                    "beginner": "I'm about to create an automation that will {description}. This automation will {benefit}. Should I proceed?",
                    "intermediate": "Ready to create: {description}. Confirm to proceed.",
                    "expert": "Automation spec: {description}. Execute? (y/n)"
                }
            ),
            
            # Error Templates
            "processing_error": ResponseTemplate(
                template_id="processing_error",
                response_type=ResponseType.ERROR,
                context_mode=ConversationMode.TROUBLESHOOTING,
                templates={
                    "beginner": "I encountered a problem while processing your request. Let me try a different approach. Can you tell me more about what you're trying to accomplish?",
                    "intermediate": "Processing error occurred. Let's troubleshoot this. Please provide additional details.",
                    "expert": "Error in processing pipeline. Debug info: {error_details}. Alternative approach?"
                }
            ),
            
            # Success Templates
            "automation_created": ResponseTemplate(
                template_id="automation_created",
                response_type=ResponseType.SUCCESS,
                context_mode=ConversationMode.CREATION,
                templates={
                    "beginner": "Excellent! I've successfully created your automation. It will {description}. You can test it now or I can show you how to modify it later.",
                    "intermediate": "Automation created successfully: {description}. Ready for testing.",
                    "expert": "Deployment complete: {description}. Status: Active. Monitoring enabled."
                }
            )
        }
    
    @require(lambda conversation_id: isinstance(conversation_id, ConversationId))
    @require(lambda mode: isinstance(mode, ConversationMode))
    @require(lambda user_input: isinstance(user_input, TextContent))
    @ensure(lambda result: result.is_right() or isinstance(result.left_value, ConversationError))
    async def process_conversation(
        self,
        conversation_id: ConversationId,
        mode: ConversationMode,
        user_input: TextContent,
        automation_context: Optional[str] = None,
        include_suggestions: bool = True,
        provide_examples: bool = True
    ) -> Either[ConversationError, ConversationResponse]:
        """Process conversational interaction with context-aware response generation."""
        try:
            # Get or create conversation session
            session = self._get_or_create_session(conversation_id, mode)
            
            # Update last activity
            session.last_activity = datetime.now(UTC)
            
            # Log the user input
            session.conversation_log.append({
                "timestamp": datetime.now(UTC),
                "type": "user_input",
                "content": str(user_input),
                "mode": mode.value
            })
            
            # Analyze user intent
            intent_result = await self.intent_classifier.recognize_intent(user_input)
            
            if intent_result.is_left():
                return await self._handle_unclear_input(session, user_input)
            
            recognized_intents = intent_result.right_value
            
            # Update conversation state based on intent
            await self._update_conversation_state(session, recognized_intents, user_input)
            
            # Generate appropriate response
            response = await self._generate_response(
                session, recognized_intents, automation_context,
                include_suggestions, provide_examples
            )
            
            # Log the response
            session.conversation_log.append({
                "timestamp": datetime.now(UTC),
                "type": "assistant_response",
                "content": response.response_text,
                "response_type": response.response_type
            })
            
            # Update learning data
            self._update_learning_data(session, user_input, response)
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(ConversationError(
                f"Conversation processing failed: {str(e)}",
                "CONVERSATION_ERROR",
                context={"conversation_id": conversation_id, "mode": mode.value}
            ))
    
    def _get_or_create_session(
        self,
        conversation_id: ConversationId,
        mode: ConversationMode
    ) -> ConversationSession:
        """Get existing session or create new one."""
        if conversation_id not in self.active_sessions:
            context = ConversationContext(
                conversation_id=conversation_id,
                context_id=create_context_id(),
                mode=mode,
                session_start=datetime.now(UTC),
                last_interaction=datetime.now(UTC)
            )
            
            session = ConversationSession(
                session_id=conversation_id,
                context=context,
                state=ConversationState.INITIAL
            )
            
            self.active_sessions[conversation_id] = session
        
        return self.active_sessions[conversation_id]
    
    async def _update_conversation_state(
        self,
        session: ConversationSession,
        intents: List,
        user_input: TextContent
    ) -> None:
        """Update conversation state based on recognized intents."""
        if not intents:
            session.state = ConversationState.CLARIFYING
            return
        
        primary_intent = intents[0]
        
        # State transitions based on intent
        if primary_intent.category.value == "greeting":
            session.state = ConversationState.INITIAL
        elif primary_intent.category.value in ["automation_command", "workflow_creation"]:
            if session.state == ConversationState.INITIAL:
                session.state = ConversationState.GATHERING_INFO
            elif session.state == ConversationState.GATHERING_INFO:
                # Check if we have enough information
                if self._has_sufficient_info(primary_intent):
                    session.state = ConversationState.PROCESSING
                else:
                    session.state = ConversationState.CLARIFYING
        elif primary_intent.category.value == "information_request":
            session.state = ConversationState.PROCESSING
        elif primary_intent.category.value == "troubleshooting":
            session.state = ConversationState.ERROR
        
        # Update topic
        if primary_intent.intent != "greeting":
            session.current_topic = primary_intent.intent
    
    def _has_sufficient_info(self, intent) -> bool:
        """Check if we have sufficient information to process the intent."""
        # Check if intent has required entities or parameters
        if intent.category.value == "automation_command":
            return len(intent.entities) >= 1
        elif intent.category.value == "workflow_creation":
            return len(intent.entities) >= 1 or len(intent.parameters) >= 2
        return True
    
    async def _generate_response(
        self,
        session: ConversationSession,
        intents: List,
        automation_context: Optional[str],
        include_suggestions: bool,
        provide_examples: bool
    ) -> ConversationResponse:
        """Generate appropriate response based on conversation state and intents."""
        response_id = f"resp_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        
        if not intents:
            return await self._generate_clarification_response(session, response_id)
        
        primary_intent = intents[0]
        skill_level = session.context.skill_level
        
        # Select response template based on state and intent
        template_key = self._select_response_template(session.state, primary_intent, session.context.mode)
        template = self.response_templates.get(template_key)
        
        if not template:
            template = self.response_templates["unclear_intent"]
        
        # Generate response text
        response_text = self._format_response_template(template, skill_level, {
            "intent": primary_intent.intent,
            "entities": [str(e.value) for e in primary_intent.entities],
            "automation_context": automation_context
        })
        
        # Generate suggestions
        suggestions = []
        if include_suggestions:
            suggestions = await self._generate_suggestions(session, primary_intent)
        
        # Generate examples
        examples = []
        if provide_examples:
            examples = self._generate_examples(primary_intent, skill_level)
        
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(session, primary_intent)
        
        return ConversationResponse(
            response_id=response_id,
            conversation_id=session.session_id,
            response_text=response_text,
            response_type=template.response_type.value,
            suggestions=suggestions,
            examples=examples,
            follow_up_questions=follow_up_questions,
            automation_context=automation_context,
            requires_action=self._requires_action(session.state),
            confidence=primary_intent.confidence
        )
    
    def _select_response_template(
        self,
        state: ConversationState,
        intent,
        mode: ConversationMode
    ) -> str:
        """Select appropriate response template."""
        if state == ConversationState.INITIAL:
            if intent.category.value == "greeting":
                return f"greeting_{mode.value}"
            else:
                return f"greeting_{mode.value}"
        elif state == ConversationState.CLARIFYING:
            return "unclear_intent"
        elif state == ConversationState.PROCESSING:
            if intent.category.value == "information_request":
                return "automation_list"
            else:
                return "automation_suggestions"
        elif state == ConversationState.ERROR:
            return "processing_error"
        else:
            return "unclear_intent"
    
    def _format_response_template(
        self,
        template: ResponseTemplate,
        skill_level: str,
        context_vars: Dict[str, Any]
    ) -> str:
        """Format response template with context variables."""
        base_response = template.templates.get(skill_level, template.templates.get("intermediate", ""))
        
        # Replace placeholders
        for key, value in context_vars.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            placeholder = f"{{{key}}}"
            if placeholder in base_response:
                base_response = base_response.replace(placeholder, str(value))
        
        return base_response
    
    async def _generate_suggestions(
        self,
        session: ConversationSession,
        intent
    ) -> List[str]:
        """Generate contextual suggestions for the user."""
        suggestions = []
        
        if intent.category.value == "automation_command":
            suggestions.extend([
                "Add error handling to your automation",
                "Set up a keyboard shortcut trigger",
                "Test the automation before saving"
            ])
        elif intent.category.value == "workflow_creation":
            suggestions.extend([
                "Consider adding conditions for different scenarios",
                "Break complex workflows into smaller steps",
                "Add notifications for completion status"
            ])
        elif intent.category.value == "information_request":
            suggestions.extend([
                "Filter automations by category",
                "Sort by most recently used",
                "Search for specific functionality"
            ])
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _generate_examples(self, intent, skill_level: str) -> List[str]:
        """Generate relevant examples based on intent and skill level."""
        examples = []
        
        if intent.category.value == "automation_command":
            if skill_level == "beginner":
                examples.extend([
                    "Launch Safari and open my email",
                    "Set volume to 50% and play my morning playlist",
                    "Create a new document and type my signature"
                ])
            else:
                examples.extend([
                    "If battery is low, close non-essential apps and dim screen",
                    "When I connect headphones, pause current audio and switch output",
                    "Schedule weekly folder cleanup and file organization"
                ])
        
        return examples[:2]  # Limit to 2 examples
    
    def _generate_follow_up_questions(
        self,
        session: ConversationSession,
        intent
    ) -> List[str]:
        """Generate follow-up questions to gather more information."""
        questions = []
        
        if session.state == ConversationState.GATHERING_INFO:
            if intent.category.value == "automation_command":
                questions.extend([
                    "What should trigger this automation?",
                    "Should this run automatically or manually?",
                    "Are there any conditions when this shouldn't run?"
                ])
            elif intent.category.value == "workflow_creation":
                questions.extend([
                    "What's the first step in this workflow?",
                    "How often will you use this?",
                    "Should this work with specific applications?"
                ])
        
        return questions[:2]  # Limit to 2 questions
    
    def _requires_action(self, state: ConversationState) -> bool:
        """Determine if the current state requires user action."""
        return state in [
            ConversationState.CONFIRMING,
            ConversationState.CLARIFYING,
            ConversationState.GATHERING_INFO
        ]
    
    async def _handle_unclear_input(
        self,
        session: ConversationSession,
        user_input: TextContent
    ) -> Either[ConversationError, ConversationResponse]:
        """Handle unclear or unrecognized user input."""
        response_id = f"resp_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        
        template = self.response_templates["unclear_intent"]
        response_text = self._format_response_template(
            template, session.context.skill_level, {}
        )
        
        return Either.right(ConversationResponse(
            response_id=response_id,
            conversation_id=session.session_id,
            response_text=response_text,
            response_type="clarification",
            suggestions=template.follow_up_suggestions,
            requires_action=True,
            confidence=0.5
        ))
    
    async def _generate_clarification_response(
        self,
        session: ConversationSession,
        response_id: str
    ) -> ConversationResponse:
        """Generate a clarification response when input is unclear."""
        template = self.response_templates["unclear_intent"]
        response_text = self._format_response_template(
            template, session.context.skill_level, {}
        )
        
        return ConversationResponse(
            response_id=response_id,
            conversation_id=session.session_id,
            response_text=response_text,
            response_type="clarification",
            suggestions=template.follow_up_suggestions,
            requires_action=True,
            confidence=0.3
        )
    
    def _update_learning_data(
        self,
        session: ConversationSession,
        user_input: TextContent,
        response: ConversationResponse
    ) -> None:
        """Update learning data for conversation improvement."""
        learning_entry = {
            "timestamp": datetime.now(UTC),
            "conversation_id": session.session_id,
            "mode": session.context.mode.value,
            "state": session.state.value,
            "user_input": str(user_input),
            "response_type": response.response_type,
            "confidence": response.confidence,
            "skill_level": session.context.skill_level
        }
        
        self.learning_data[session.context.mode.value].append(learning_entry)
        
        # Limit learning data size
        if len(self.learning_data[session.context.mode.value]) > 1000:
            self.learning_data[session.context.mode.value] = \
                self.learning_data[session.context.mode.value][-1000:]
    
    def get_session_info(self, conversation_id: ConversationId) -> Optional[Dict[str, Any]]:
        """Get information about a conversation session."""
        if conversation_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[conversation_id]
        
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "mode": session.context.mode.value,
            "skill_level": session.context.skill_level,
            "current_topic": session.current_topic,
            "conversation_length": len(session.conversation_log),
            "last_activity": session.last_activity.isoformat(),
            "pending_actions": len(session.pending_actions)
        }
    
    def cleanup_inactive_sessions(self, max_idle_hours: int = 24) -> int:
        """Clean up inactive conversation sessions."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=max_idle_hours)
        inactive_sessions = [
            conv_id for conv_id, session in self.active_sessions.items()
            if session.last_activity < cutoff_time
        ]
        
        for conv_id in inactive_sessions:
            del self.active_sessions[conv_id]
        
        return len(inactive_sessions)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation management statistics."""
        total_sessions = len(self.active_sessions)
        
        if total_sessions == 0:
            return {"active_sessions": 0}
        
        # Calculate state distribution
        state_distribution = {}
        mode_distribution = {}
        
        for session in self.active_sessions.values():
            state = session.state.value
            mode = session.context.mode.value
            
            state_distribution[state] = state_distribution.get(state, 0) + 1
            mode_distribution[mode] = mode_distribution.get(mode, 0) + 1
        
        return {
            "active_sessions": total_sessions,
            "state_distribution": state_distribution,
            "mode_distribution": mode_distribution,
            "total_learning_entries": sum(len(data) for data in self.learning_data.values()),
            "available_templates": len(self.response_templates)
        }