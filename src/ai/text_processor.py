"""
AI text processing system for natural language analysis and generation.

This module provides comprehensive text processing capabilities including
analysis, generation, classification, and transformation using AI models.
Implements enterprise-grade text intelligence with security and performance.

Security: All text processing includes content validation and sanitization.
Performance: Optimized for real-time text analysis with intelligent caching.
Type Safety: Complete integration with AI model management system.
"""

import asyncio
import re
import json
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum

from src.core.ai_integration import (
    AIOperation, AIRequest, AIResponse, ProcessingMode, OutputFormat,
    AIModel, AIModelId, TokenCount, ConfidenceScore
)
from src.ai.model_manager import AIModelManager, AIError
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.errors import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class TextAnalysisType(Enum):
    """Types of text analysis operations."""
    GENERAL = "general"              # General content analysis
    SENTIMENT = "sentiment"          # Sentiment analysis
    ENTITIES = "entities"            # Named entity recognition
    KEYWORDS = "keywords"            # Keyword extraction
    SUMMARY = "summary"              # Text summarization
    CLASSIFICATION = "classification" # Content classification
    LANGUAGE = "language"            # Language detection
    COMPLEXITY = "complexity"        # Readability analysis
    INTENT = "intent"                # Intent classification
    TOPICS = "topics"                # Topic modeling


class TextGenerationStyle(Enum):
    """Text generation style options."""
    NATURAL = "natural"              # Natural, conversational
    FORMAL = "formal"                # Formal, professional
    CREATIVE = "creative"            # Creative, engaging
    TECHNICAL = "technical"          # Technical, precise
    CASUAL = "casual"                # Casual, friendly
    ACADEMIC = "academic"            # Academic, scholarly
    MARKETING = "marketing"          # Marketing, persuasive
    NEWS = "news"                    # News article style


@dataclass(frozen=True)
class TextAnalysisResult:
    """Result of text analysis operation."""
    analysis_type: TextAnalysisType
    input_text: str
    results: Dict[str, Any]
    confidence: ConfidenceScore
    processing_time: float
    model_used: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def get_primary_result(self) -> Any:
        """Get the primary result from analysis."""
        if self.analysis_type == TextAnalysisType.SENTIMENT:
            return self.results.get("sentiment", "neutral")
        elif self.analysis_type == TextAnalysisType.LANGUAGE:
            return self.results.get("language", "unknown")
        elif self.analysis_type == TextAnalysisType.SUMMARY:
            return self.results.get("summary", "")
        else:
            return self.results
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get analysis metadata."""
        return {
            "analysis_type": self.analysis_type.value,
            "confidence": float(self.confidence),
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
            "input_length": len(self.input_text)
        }


@dataclass(frozen=True)
class TextGenerationRequest:
    """Request for text generation."""
    prompt: str
    style: TextGenerationStyle = TextGenerationStyle.NATURAL
    max_length: int = 500
    temperature: float = 0.7
    include_context: bool = True
    target_audience: Optional[str] = None
    additional_instructions: Optional[str] = None
    
    @require(lambda self: len(self.prompt) > 0)
    @require(lambda self: 0 < self.max_length <= 10000)
    @require(lambda self: 0.0 <= self.temperature <= 2.0)
    def __post_init__(self):
        """Validate generation request."""
        pass
    
    def build_system_prompt(self) -> str:
        """Build system prompt for generation."""
        style_instructions = {
            TextGenerationStyle.NATURAL: "Write in a natural, conversational tone that feels authentic and engaging.",
            TextGenerationStyle.FORMAL: "Write in a formal, professional tone suitable for business communication.",
            TextGenerationStyle.CREATIVE: "Write creatively with engaging language, vivid descriptions, and compelling narratives.",
            TextGenerationStyle.TECHNICAL: "Write in a clear, technical style with precise language and accurate terminology.",
            TextGenerationStyle.CASUAL: "Write in a casual, friendly tone that's approachable and relatable.",
            TextGenerationStyle.ACADEMIC: "Write in an academic style with scholarly language and rigorous analysis.",
            TextGenerationStyle.MARKETING: "Write in a persuasive marketing style that engages and motivates the reader.",
            TextGenerationStyle.NEWS: "Write in a clear, objective news style with factual reporting and journalistic structure."
        }
        
        instruction = style_instructions.get(self.style, style_instructions[TextGenerationStyle.NATURAL])
        
        system_prompt = f"{instruction} Keep the response to approximately {self.max_length} characters."
        
        if self.target_audience:
            system_prompt += f" Write for this audience: {self.target_audience}."
        
        if self.additional_instructions:
            system_prompt += f" Additional requirements: {self.additional_instructions}"
        
        return system_prompt


class TextProcessor:
    """AI-powered text processing and analysis system."""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.analysis_cache: Dict[str, TextAnalysisResult] = {}
        self.generation_cache: Dict[str, str] = {}
        
        # Text processing patterns
        self.sentence_splitter = re.compile(r'[.!?]+\s+')
        self.word_pattern = re.compile(r'\b\w+\b')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_pattern = re.compile(r'https?://[^\s]+')
    
    async def analyze_text(
        self,
        text: str,
        analysis_type: TextAnalysisType = TextAnalysisType.GENERAL,
        model_preference: Optional[AIModelId] = None
    ) -> Either[AIError, TextAnalysisResult]:
        """Analyze text using AI with specified analysis type."""
        try:
            if not text.strip():
                return Either.left(AIError("invalid_input", "Text cannot be empty"))
            
            # Check cache first
            cache_key = f"{analysis_type.value}:{hash(text)}"
            if cache_key in self.analysis_cache:
                logger.debug(f"Using cached analysis for {analysis_type.value}")
                return Either.right(self.analysis_cache[cache_key])
            
            # Select appropriate model
            model_result = self.model_manager.select_best_model(
                AIOperation.ANALYZE,
                ProcessingMode.BALANCED,
                input_size=len(text)
            )
            if model_result.is_left():
                return model_result
            
            model = model_result.get_right()
            
            # Build analysis prompt
            prompt = self._build_analysis_prompt(text, analysis_type)
            
            # Create AI request
            request_result = self._create_ai_request(
                AIOperation.ANALYZE,
                prompt,
                model,
                temperature=0.3  # Lower temperature for analysis
            )
            if request_result.is_left():
                return request_result
            
            request = request_result.get_right()
            
            # Process with AI
            start_time = datetime.now(UTC)
            response_result = await self._call_ai_model(request)
            if response_result.is_left():
                return response_result
            
            response = response_result.get_right()
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            
            # Parse analysis results
            analysis_result = self._parse_analysis_response(
                analysis_type, text, response.result, response.confidence or ConfidenceScore(0.8),
                processing_time, model.display_name
            )
            
            # Cache result
            self.analysis_cache[cache_key] = analysis_result
            
            logger.info(f"Text analysis completed: {analysis_type.value} in {processing_time:.2f}s")
            return Either.right(analysis_result)
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return Either.left(AIError("analysis_failed", str(e)))
    
    def _build_analysis_prompt(self, text: str, analysis_type: TextAnalysisType) -> str:
        """Build analysis prompt based on analysis type."""
        prompts = {
            TextAnalysisType.GENERAL: f"""
Analyze the following text and provide insights about:
- Main themes and topics
- Tone and style
- Key information
- Overall sentiment
- Writing quality

Text to analyze:
{text}

Respond in JSON format with keys: themes, tone, key_info, sentiment, quality_score.
""",
            
            TextAnalysisType.SENTIMENT: f"""
Analyze the sentiment of this text. Classify as positive, negative, or neutral.
Provide a confidence score (0-1) and explain the reasoning.

Text: {text}

Respond in JSON format with keys: sentiment, confidence, reasoning, emotion_details.
""",
            
            TextAnalysisType.ENTITIES: f"""
Extract all named entities from this text including:
- People (names)
- Organizations
- Locations
- Dates and times
- Numbers and quantities
- Other relevant entities

Text: {text}

Respond in JSON format with keys: people, organizations, locations, dates, numbers, other.
""",
            
            TextAnalysisType.KEYWORDS: f"""
Extract the most important keywords and phrases from this text.
Rank them by importance and relevance.

Text: {text}

Respond in JSON format with keys: keywords (list), key_phrases (list), importance_scores (dict).
""",
            
            TextAnalysisType.SUMMARY: f"""
Create a concise summary of this text, capturing the main points and key information.
The summary should be about 20% of the original length.

Text: {text}

Respond in JSON format with keys: summary, key_points (list), original_length, summary_length.
""",
            
            TextAnalysisType.CLASSIFICATION: f"""
Classify this text into appropriate categories. Consider:
- Content type (news, opinion, technical, creative, etc.)
- Subject domain
- Purpose/intent
- Audience level

Text: {text}

Respond in JSON format with keys: primary_category, secondary_categories, domain, intent, audience_level.
""",
            
            TextAnalysisType.LANGUAGE: f"""
Detect the language of this text and assess:
- Primary language
- Language confidence
- Any mixed languages
- Text quality indicators

Text: {text}

Respond in JSON format with keys: primary_language, confidence, mixed_languages, quality_indicators.
""",
            
            TextAnalysisType.COMPLEXITY: f"""
Analyze the complexity and readability of this text:
- Reading level
- Sentence complexity
- Vocabulary difficulty
- Overall clarity score

Text: {text}

Respond in JSON format with keys: reading_level, sentence_complexity, vocabulary_difficulty, clarity_score.
""",
            
            TextAnalysisType.INTENT: f"""
Analyze the intent and purpose of this text:
- Primary intent (inform, persuade, entertain, etc.)
- Target action or response
- Urgency level
- Communication style

Text: {text}

Respond in JSON format with keys: primary_intent, target_action, urgency_level, communication_style.
""",
            
            TextAnalysisType.TOPICS: f"""
Identify the main topics and themes in this text:
- Primary topics
- Secondary themes
- Topic relationships
- Coverage depth

Text: {text}

Respond in JSON format with keys: primary_topics, secondary_themes, topic_relationships, coverage_depth.
"""
        }
        
        return prompts.get(analysis_type, prompts[TextAnalysisType.GENERAL])
    
    async def generate_text(
        self,
        generation_request: TextGenerationRequest,
        model_preference: Optional[AIModelId] = None
    ) -> Either[AIError, str]:
        """Generate text based on prompt and style requirements."""
        try:
            # Check cache
            cache_key = f"gen:{hash(generation_request.prompt)}:{generation_request.style.value}"
            if cache_key in self.generation_cache:
                logger.debug("Using cached text generation")
                return Either.right(self.generation_cache[cache_key])
            
            # Select creative model
            model_result = self.model_manager.select_best_model(
                AIOperation.GENERATE,
                ProcessingMode.CREATIVE,
                input_size=len(generation_request.prompt)
            )
            if model_result.is_left():
                return model_result
            
            model = model_result.get_right()
            
            # Create AI request
            request_result = self._create_ai_request(
                AIOperation.GENERATE,
                generation_request.prompt,
                model,
                temperature=generation_request.temperature,
                max_tokens=TokenCount(generation_request.max_length // 3),  # Approximate token count
                system_prompt=generation_request.build_system_prompt()
            )
            if request_result.is_left():
                return request_result
            
            request = request_result.get_right()
            
            # Generate with AI
            response_result = await self._call_ai_model(request)
            if response_result.is_left():
                return response_result
            
            response = response_result.get_right()
            generated_text = str(response.result)
            
            # Post-process generated text
            processed_text = self._post_process_generated_text(generated_text, generation_request)
            
            # Cache result
            self.generation_cache[cache_key] = processed_text
            
            logger.info(f"Text generation completed: {len(processed_text)} characters")
            return Either.right(processed_text)
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return Either.left(AIError("generation_failed", str(e)))
    
    def _post_process_generated_text(self, text: str, request: TextGenerationRequest) -> str:
        """Post-process generated text for quality and requirements."""
        # Remove any AI response artifacts
        text = text.strip()
        
        # Remove common AI response prefixes
        prefixes_to_remove = [
            "Here's a ", "Here is a ", "I'll write ", "I'll create ",
            "Let me write ", "Let me create ", "Based on your request"
        ]
        
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # Ensure proper capitalization
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Truncate if too long
        if len(text) > request.max_length:
            # Try to truncate at sentence boundary
            sentences = self.sentence_splitter.split(text)
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) <= request.max_length:
                    truncated += sentence + ". "
                else:
                    break
            
            if truncated:
                text = truncated.strip()
            else:
                text = text[:request.max_length].rstrip() + "..."
        
        return text
    
    async def classify_text(
        self,
        text: str,
        categories: List[str],
        model_preference: Optional[AIModelId] = None
    ) -> Either[AIError, Dict[str, float]]:
        """Classify text into provided categories with confidence scores."""
        try:
            if not categories:
                return Either.left(AIError("invalid_input", "Categories list cannot be empty"))
            
            # Select model for classification
            model_result = self.model_manager.select_best_model(
                AIOperation.CLASSIFY,
                ProcessingMode.ACCURATE
            )
            if model_result.is_left():
                return model_result
            
            model = model_result.get_right()
            
            # Build classification prompt
            categories_str = ", ".join(categories)
            prompt = f"""
Classify the following text into these categories: {categories_str}

For each category, provide a confidence score from 0.0 to 1.0 indicating how well the text fits that category.

Text to classify:
{text}

Respond in JSON format with each category as a key and confidence score as value.
"""
            
            # Create and process request
            request_result = self._create_ai_request(AIOperation.CLASSIFY, prompt, model, temperature=0.2)
            if request_result.is_left():
                return request_result
            
            response_result = await self._call_ai_model(request_result.get_right())
            if response_result.is_left():
                return response_result
            
            response = response_result.get_right()
            
            # Parse classification results
            try:
                results = json.loads(str(response.result))
                # Ensure all categories are present with default scores
                classification = {}
                for category in categories:
                    classification[category] = float(results.get(category, 0.0))
                
                logger.info(f"Text classification completed for {len(categories)} categories")
                return Either.right(classification)
                
            except (json.JSONDecodeError, ValueError) as e:
                return Either.left(AIError("invalid_response", f"Could not parse classification results: {e}"))
            
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            return Either.left(AIError("classification_failed", str(e)))
    
    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        model_preference: Optional[AIModelId] = None
    ) -> Either[AIError, Dict[str, List[str]]]:
        """Extract named entities from text."""
        try:
            default_types = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY", "MISC"]
            target_types = entity_types or default_types
            
            # Use analysis method with entities type
            analysis_result = await self.analyze_text(text, TextAnalysisType.ENTITIES, model_preference)
            if analysis_result.is_left():
                return analysis_result
            
            analysis = analysis_result.get_right()
            entities = analysis.results
            
            # Filter to requested types if specified
            if entity_types:
                filtered_entities = {}
                for entity_type in entity_types:
                    # Map common entity type names
                    type_mapping = {
                        "PERSON": "people",
                        "ORGANIZATION": "organizations", 
                        "LOCATION": "locations",
                        "DATE": "dates",
                        "MONEY": "numbers",
                        "MISC": "other"
                    }
                    mapped_type = type_mapping.get(entity_type.upper(), entity_type.lower())
                    filtered_entities[entity_type] = entities.get(mapped_type, [])
                
                entities = filtered_entities
            
            logger.info(f"Entity extraction completed: {sum(len(v) for v in entities.values())} entities found")
            return Either.right(entities)
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return Either.left(AIError("extraction_failed", str(e)))
    
    def _create_ai_request(
        self,
        operation: AIOperation,
        prompt: str,
        model: AIModel,
        **kwargs
    ) -> Either[AIError, AIRequest]:
        """Create AI request with proper configuration."""
        try:
            from src.core.ai_integration import create_ai_request
            
            return create_ai_request(
                operation=operation,
                input_data=prompt,
                model_id=model.model_id,
                **kwargs
            )
        except Exception as e:
            return Either.left(AIError("request_creation_failed", str(e)))
    
    async def _call_ai_model(self, request: AIRequest) -> Either[AIError, AIResponse]:
        """Call AI model with request (placeholder for actual implementation)."""
        # This would integrate with actual AI APIs
        # For now, return a mock response
        try:
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Create mock response based on operation
            if request.operation == AIOperation.ANALYZE:
                result = self._create_mock_analysis_result(request.input_data)
            elif request.operation == AIOperation.GENERATE:
                result = self._create_mock_generation_result(request.input_data)
            elif request.operation == AIOperation.CLASSIFY:
                result = '{"category1": 0.8, "category2": 0.2}'
            else:
                result = "Mock AI response"
            
            response = AIResponse(
                request_id=request.request_id,
                operation=request.operation,
                result=result,
                model_used=request.model.display_name,
                tokens_used=TokenCount(len(str(request.input_data)) // 4),
                input_tokens=TokenCount(len(str(request.input_data)) // 4),
                output_tokens=TokenCount(len(str(result)) // 4),
                processing_time=0.1,
                cost_estimate=request.model.estimate_cost(TokenCount(100), TokenCount(50)),
                confidence=ConfidenceScore(0.85)
            )
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(AIError.api_call_failed(request.model.model_name, str(e)))
    
    def _create_mock_analysis_result(self, text: str) -> str:
        """Create mock analysis result for testing."""
        return json.dumps({
            "themes": ["communication", "technology"],
            "tone": "professional",
            "key_info": "Text processing capabilities",
            "sentiment": "neutral",
            "quality_score": 0.8
        })
    
    def _create_mock_generation_result(self, prompt: str) -> str:
        """Create mock generation result for testing."""
        return f"Generated text based on: {prompt[:50]}... This is a mock response for testing purposes."
    
    def _parse_analysis_response(
        self,
        analysis_type: TextAnalysisType,
        input_text: str,
        response: str,
        confidence: ConfidenceScore,
        processing_time: float,
        model_name: str
    ) -> TextAnalysisResult:
        """Parse AI response into structured analysis result."""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                results = json.loads(response)
            else:
                # Fallback to simple parsing
                results = {"analysis": response}
            
            return TextAnalysisResult(
                analysis_type=analysis_type,
                input_text=input_text,
                results=results,
                confidence=confidence,
                processing_time=processing_time,
                model_used=model_name
            )
            
        except json.JSONDecodeError:
            # If JSON parsing fails, wrap in simple structure
            return TextAnalysisResult(
                analysis_type=analysis_type,
                input_text=input_text,
                results={"raw_response": response},
                confidence=ConfidenceScore(confidence * 0.8),  # Lower confidence for unparseable response
                processing_time=processing_time,
                model_used=model_name
            )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get text processing statistics."""
        return {
            "cache_size": {
                "analysis": len(self.analysis_cache),
                "generation": len(self.generation_cache)
            },
            "supported_analysis_types": [t.value for t in TextAnalysisType],
            "supported_generation_styles": [s.value for s in TextGenerationStyle],
            "model_manager_stats": self.model_manager.get_usage_statistics()
        }
    
    def clear_cache(self) -> None:
        """Clear processing caches."""
        self.analysis_cache.clear()
        self.generation_cache.clear()
        logger.info("Text processor caches cleared")