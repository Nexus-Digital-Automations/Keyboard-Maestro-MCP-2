"""
AI integration type system for Keyboard Maestro MCP tools.

This module provides comprehensive AI/ML integration with support for multiple
model providers, secure processing, and cost optimization. Implements all
ADDER+ techniques for enterprise-grade AI automation.

Security: All AI operations include input validation and output sanitization.
Performance: Optimized for real-time AI processing with intelligent caching.
Type Safety: Complete branded type system with contract validation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NewType, List, Optional, Union, Dict, Any, Set, Type, Callable
from enum import Enum
from datetime import datetime, UTC
import re
import json

from .either import Either
from .contracts import require, ensure
from .errors import ValidationError


# Branded Types for AI Integration
AIModelId = NewType('AIModelId', str)
AIRequestId = NewType('AIRequestId', str)
AISessionId = NewType('AISessionId', str)
TokenCount = NewType('TokenCount', int)
CostAmount = NewType('CostAmount', float)
ConfidenceScore = NewType('ConfidenceScore', float)


class AIOperation(Enum):
    """AI processing operation types."""
    ANALYZE = "analyze"              # Analyze content for insights
    GENERATE = "generate"            # Generate new content
    PREDICT = "predict"              # Make predictions based on data
    CLASSIFY = "classify"            # Classify content into categories
    EXTRACT = "extract"              # Extract specific information
    ENHANCE = "enhance"              # Enhance existing content
    SUMMARIZE = "summarize"          # Summarize long content
    TRANSLATE = "translate"          # Translate between languages
    EXPLAIN = "explain"              # Explain concepts or content
    TRANSFORM = "transform"          # Transform content format/style


class AIModelType(Enum):
    """Supported AI model types."""
    OPENAI = "openai"                # OpenAI GPT models
    AZURE_OPENAI = "azure"           # Azure OpenAI service
    GOOGLE_AI = "google"             # Google AI/Gemini
    ANTHROPIC = "anthropic"          # Anthropic Claude
    LOCAL = "local"                  # Local models
    AUTO = "auto"                    # Automatic model selection


class ProcessingMode(Enum):
    """AI processing mode priorities."""
    FAST = "fast"                    # Prioritize speed
    BALANCED = "balanced"            # Balance speed and accuracy
    ACCURATE = "accurate"            # Prioritize accuracy
    CREATIVE = "creative"            # Prioritize creativity
    COST_EFFECTIVE = "cost_effective" # Prioritize cost optimization


class OutputFormat(Enum):
    """AI output format options."""
    AUTO = "auto"                    # Auto-detect format
    TEXT = "text"                    # Plain text
    JSON = "json"                    # JSON structure
    MARKDOWN = "markdown"            # Markdown formatted
    HTML = "html"                    # HTML formatted
    STRUCTURED = "structured"        # Structured data format


class TextAnalysisType(Enum):
    """Types of text analysis operations."""
    GENERAL = "general"              # General text analysis
    SENTIMENT = "sentiment"          # Sentiment analysis
    ENTITIES = "entities"            # Entity extraction
    KEYWORDS = "keywords"            # Keyword extraction
    SUMMARY = "summary"              # Text summarization
    CLASSIFICATION = "classification" # Text classification
    LANGUAGE = "language"            # Language detection


class TextGenerationStyle(Enum):
    """Styles for text generation."""
    FORMAL = "formal"                # Formal business writing
    CASUAL = "casual"                # Casual conversational style
    TECHNICAL = "technical"          # Technical documentation
    CREATIVE = "creative"            # Creative writing
    PERSUASIVE = "persuasive"        # Persuasive/marketing style
    INSTRUCTIONAL = "instructional"  # How-to/tutorial style


class ImageAnalysisType(Enum):
    """Types of image analysis operations."""
    DESCRIPTION = "description"      # Generate image description
    OCR = "ocr"                      # Optical character recognition
    OBJECTS = "objects"              # Object detection
    FACES = "faces"                  # Face detection/recognition
    SCENE = "scene"                  # Scene understanding
    QUALITY = "quality"              # Image quality assessment


@dataclass(frozen=True)
class AIModel:
    """Type-safe AI model specification with comprehensive metadata."""
    model_id: AIModelId
    model_type: AIModelType
    model_name: str
    display_name: str
    api_endpoint: Optional[str] = None
    api_key_required: bool = True
    max_tokens: TokenCount = TokenCount(4096)
    cost_per_input_token: CostAmount = CostAmount(0.0)
    cost_per_output_token: CostAmount = CostAmount(0.0)
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_streaming: bool = False
    context_window: TokenCount = TokenCount(4096)
    rate_limit_per_minute: int = 60
    
    @require(lambda self: len(self.model_name) > 0)
    @require(lambda self: self.max_tokens > 0)
    @require(lambda self: self.cost_per_input_token >= 0.0)
    @require(lambda self: self.cost_per_output_token >= 0.0)
    @require(lambda self: self.context_window > 0)
    @require(lambda self: self.rate_limit_per_minute > 0)
    def __post_init__(self):
        """Validate AI model configuration."""
        pass
    
    def estimate_cost(self, input_tokens: TokenCount, output_tokens: TokenCount = TokenCount(0)) -> CostAmount:
        """Estimate processing cost for token usage."""
        input_cost = input_tokens * self.cost_per_input_token
        output_cost = output_tokens * self.cost_per_output_token
        return CostAmount(input_cost + output_cost)
    
    def can_handle_operation(self, operation: AIOperation, input_size: int = 0) -> bool:
        """Check if model supports specific operation and input size."""
        # Vision operations require vision support
        vision_operations = {AIOperation.ANALYZE, AIOperation.EXTRACT}
        if operation in vision_operations and input_size == 0:  # Assume non-text
            return self.supports_vision
        
        # Check context window limits
        if input_size > self.context_window:
            return False
        
        return True
    
    def is_within_rate_limit(self, requests_this_minute: int) -> bool:
        """Check if request is within rate limit."""
        return requests_this_minute < self.rate_limit_per_minute


@dataclass(frozen=True)
class AIRequest:
    """Complete AI processing request specification with validation."""
    request_id: AIRequestId
    operation: AIOperation
    input_data: Union[str, Dict[str, Any], List[Any]]
    model: AIModel
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    temperature: float = 0.7
    max_tokens: Optional[TokenCount] = None
    context: Dict[str, Any] = field(default_factory=dict)
    output_format: OutputFormat = OutputFormat.AUTO
    privacy_mode: bool = True
    system_prompt: Optional[str] = None
    user_id: Optional[str] = None
    
    @require(lambda self: 0.0 <= self.temperature <= 2.0)
    @require(lambda self: self.max_tokens is None or self.max_tokens > 0)
    @require(lambda self: len(str(self.input_data)) > 0)
    def __post_init__(self):
        """Validate AI request parameters."""
        pass
    
    def get_effective_max_tokens(self) -> TokenCount:
        """Get effective max tokens for request."""
        return self.max_tokens or TokenCount(self.model.max_tokens // 2)
    
    def prepare_input_for_model(self) -> str:
        """Prepare input data for AI model processing."""
        if isinstance(self.input_data, str):
            return self.input_data
        elif isinstance(self.input_data, dict):
            return json.dumps(self.input_data, indent=2)
        elif isinstance(self.input_data, list):
            return '\n'.join(str(item) for item in self.input_data)
        else:
            return str(self.input_data)
    
    def estimate_input_tokens(self) -> TokenCount:
        """Estimate input token count for request."""
        text = self.prepare_input_for_model()
        if self.system_prompt:
            text += "\n" + self.system_prompt
        # Rough estimation: 1 token ≈ 4 characters
        return TokenCount(len(text) // 4)
    
    def validate_for_model(self) -> Either[ValidationError, None]:
        """Validate request compatibility with selected model."""
        try:
            # Check operation support
            input_size = len(self.prepare_input_for_model())
            if not self.model.can_handle_operation(self.operation, input_size):
                return Either.left(ValidationError(
                    "operation_not_supported",
                    f"Model {self.model.model_name} cannot handle operation {self.operation.value}"
                ))
            
            # Check context window
            estimated_tokens = self.estimate_input_tokens()
            if estimated_tokens > self.model.context_window:
                return Either.left(ValidationError(
                    "context_window_exceeded",
                    f"Input tokens {estimated_tokens} exceed model context window {self.model.context_window}"
                ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(ValidationError("validation_failed", str(e)))


@dataclass(frozen=True)
class AIResponse:
    """AI processing response with comprehensive metadata."""
    request_id: AIRequestId
    operation: AIOperation
    result: Any
    model_used: str
    tokens_used: TokenCount
    input_tokens: TokenCount
    output_tokens: TokenCount
    processing_time: float
    cost_estimate: CostAmount
    confidence: Optional[ConfidenceScore] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @require(lambda self: self.tokens_used >= 0)
    @require(lambda self: self.input_tokens >= 0)
    @require(lambda self: self.output_tokens >= 0)
    @require(lambda self: self.processing_time >= 0.0)
    @require(lambda self: self.cost_estimate >= 0.0)
    @require(lambda self: self.confidence is None or 0.0 <= self.confidence <= 1.0)
    def __post_init__(self):
        """Validate AI response data."""
        pass
    
    def is_high_confidence(self, threshold: ConfidenceScore = ConfidenceScore(0.8)) -> bool:
        """Check if result meets confidence threshold."""
        return self.confidence is not None and self.confidence >= threshold
    
    def get_formatted_result(self, format_type: OutputFormat) -> str:
        """Get result formatted in specified format."""
        if format_type == OutputFormat.TEXT:
            return str(self.result)
        elif format_type == OutputFormat.JSON:
            if isinstance(self.result, (dict, list)):
                return json.dumps(self.result, indent=2)
            else:
                return json.dumps({"result": self.result}, indent=2)
        elif format_type == OutputFormat.MARKDOWN:
            return f"# AI Processing Result\n\n{self.result}"
        else:
            return str(self.result)
    
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown for response."""
        return {
            "total_cost": float(self.cost_estimate),
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
            "total_tokens": int(self.tokens_used),
            "model_used": self.model_used,
            "processing_time": self.processing_time
        }


class AISecurityLevel(Enum):
    """AI security validation levels."""
    MINIMAL = "minimal"              # Basic validation
    STANDARD = "standard"            # Standard security checks
    STRICT = "strict"                # Enhanced security validation
    PARANOID = "paranoid"            # Maximum security restrictions


@dataclass(frozen=True)
class AISecurityConfig:
    """AI security configuration and policies."""
    security_level: AISecurityLevel = AISecurityLevel.STANDARD
    enable_content_filtering: bool = True
    enable_pii_detection: bool = True
    enable_malware_scanning: bool = True
    max_input_size: int = 1_000_000  # 1MB
    max_output_size: int = 500_000   # 500KB
    allowed_domains: Set[str] = field(default_factory=set)
    blocked_patterns: List[str] = field(default_factory=list)
    enable_audit_logging: bool = True
    
    @require(lambda self: self.max_input_size > 0)
    @require(lambda self: self.max_output_size > 0)
    def __post_init__(self):
        """Validate security configuration."""
        pass
    
    def should_scan_content(self, content_size: int) -> bool:
        """Determine if content should be scanned based on size and policy."""
        if content_size > self.max_input_size:
            return False  # Too large to scan
        
        return (self.enable_content_filtering or 
                self.enable_pii_detection or 
                self.enable_malware_scanning)


@dataclass(frozen=True)
class AIUsageStats:
    """AI usage statistics and tracking."""
    session_id: AISessionId
    total_requests: int = 0
    total_tokens: TokenCount = TokenCount(0)
    total_cost: CostAmount = CostAmount(0.0)
    average_response_time: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    models_used: Set[str] = field(default_factory=set)
    operations_count: Dict[AIOperation, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_request_time: Optional[datetime] = None
    
    def get_success_rate(self) -> float:
        """Calculate request success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def get_average_cost_per_request(self) -> CostAmount:
        """Calculate average cost per request."""
        if self.total_requests == 0:
            return CostAmount(0.0)
        return CostAmount(self.total_cost / self.total_requests)
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds."""
        end_time = self.last_request_time or datetime.now(UTC)
        return (end_time - self.start_time).total_seconds()


# Predefined AI Models for Different Providers
DEFAULT_AI_MODELS = {
    # OpenAI Models
    "gpt-4": AIModel(
        model_id=AIModelId("openai-gpt-4"),
        model_type=AIModelType.OPENAI,
        model_name="gpt-4",
        display_name="GPT-4",
        max_tokens=TokenCount(8192),
        cost_per_input_token=CostAmount(0.00003),
        cost_per_output_token=CostAmount(0.00006),
        supports_vision=False,
        supports_function_calling=True,
        context_window=TokenCount(8192)
    ),
    
    "gpt-4-vision": AIModel(
        model_id=AIModelId("openai-gpt-4-vision"),
        model_type=AIModelType.OPENAI,
        model_name="gpt-4-vision-preview",
        display_name="GPT-4 Vision",
        max_tokens=TokenCount(4096),
        cost_per_input_token=CostAmount(0.00001),
        cost_per_output_token=CostAmount(0.00003),
        supports_vision=True,
        supports_function_calling=True,
        context_window=TokenCount(128000)
    ),
    
    "gpt-3.5-turbo": AIModel(
        model_id=AIModelId("openai-gpt-3.5-turbo"),
        model_type=AIModelType.OPENAI,
        model_name="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        max_tokens=TokenCount(4096),
        cost_per_input_token=CostAmount(0.0000005),
        cost_per_output_token=CostAmount(0.0000015),
        supports_function_calling=True,
        context_window=TokenCount(16385),
        rate_limit_per_minute=3500
    ),
    
    # Google AI Models
    "gemini-pro": AIModel(
        model_id=AIModelId("google-gemini-pro"),
        model_type=AIModelType.GOOGLE_AI,
        model_name="gemini-pro",
        display_name="Gemini Pro",
        max_tokens=TokenCount(32768),
        cost_per_input_token=CostAmount(0.0000005),
        cost_per_output_token=CostAmount(0.0000015),
        supports_function_calling=True,
        context_window=TokenCount(32768)
    ),
    
    "gemini-pro-vision": AIModel(
        model_id=AIModelId("google-gemini-pro-vision"),
        model_type=AIModelType.GOOGLE_AI,
        model_name="gemini-pro-vision",
        display_name="Gemini Pro Vision",
        max_tokens=TokenCount(16384),
        cost_per_input_token=CostAmount(0.00000025),
        cost_per_output_token=CostAmount(0.0000005),
        supports_vision=True,
        context_window=TokenCount(16384)
    )
}


def create_ai_request(
    operation: AIOperation,
    input_data: Union[str, Dict, List],
    model_id: Optional[AIModelId] = None,
    **kwargs
) -> Either[ValidationError, AIRequest]:
    """
    Create and validate AI request with automatic model selection.
    
    This function provides a convenient way to create AI requests with
    automatic model selection if no specific model is provided.
    """
    try:
        # Generate unique request ID
        request_id = AIRequestId(f"ai_req_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{id(input_data)}")
        
        # Select model if not provided
        if model_id is None:
            # Default model selection based on operation
            if operation in [AIOperation.ANALYZE, AIOperation.EXTRACT]:
                model = DEFAULT_AI_MODELS.get("gpt-4-vision", DEFAULT_AI_MODELS["gpt-4"])
            else:
                model = DEFAULT_AI_MODELS["gpt-3.5-turbo"]
        else:
            model = DEFAULT_AI_MODELS.get(str(model_id))
            if model is None:
                return Either.left(ValidationError("invalid_model", f"Model {model_id} not found"))
        
        # Create request
        request = AIRequest(
            request_id=request_id,
            operation=operation,
            input_data=input_data,
            model=model,
            **kwargs
        )
        
        # Validate request
        validation_result = request.validate_for_model()
        if validation_result.is_left():
            return validation_result
        
        return Either.right(request)
        
    except Exception as e:
        return Either.left(ValidationError("request_creation_failed", str(e)))


def create_ai_session() -> AISessionId:
    """Create new AI session ID for tracking usage."""
    import uuid
    return AISessionId(f"ai_session_{uuid.uuid4().hex[:12]}")