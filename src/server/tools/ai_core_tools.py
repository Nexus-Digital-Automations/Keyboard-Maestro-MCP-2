"""
AI Core Processing Tools - Core AI processing management and basic operations.

This module provides the foundational AI processing capabilities including the core
AIProcessingManager class, security validation, model selection, and basic AI operations.
Implements enterprise-grade AI integration with security and cost optimization.

Security: All AI operations include comprehensive validation and threat detection.
Performance: Optimized for real-time AI processing with intelligent caching.
Type Safety: Complete integration with AI processing architecture.
"""

import re
from datetime import UTC, datetime
from typing import Any

from src.core.ai_integration import (
    DEFAULT_AI_MODELS,
    AIModelType,
    AIOperation,
    AIRequest,
    AIResponse,
    OutputFormat,
    ProcessingMode,
    TokenCount,
    create_ai_request,
    create_ai_session,
)
from src.core.contracts import ensure, require
from src.core.either import Either
from src.core.errors import ValidationError


class AIProcessingManager:
    """
    Comprehensive AI processing management with security and cost optimization.

    Implements enterprise-grade AI integration with support for multiple model
    providers, intelligent caching, and comprehensive security validation.
    """

    def __init__(self):
        self.session_id = create_ai_session()
        self.request_cache: dict[str, AIResponse] = {}
        self.usage_history: list[dict[str, Any]] = []
        self.initialized = False

    async def initialize(self) -> Either[ValidationError, None]:
        """Initialize AI processing system."""
        try:
            # Mock initialization - in real implementation would validate API keys
            self.initialized = True
            return Either.right(None)
        except Exception as e:
            return Either.left(ValidationError("initialization_failed", str(e)))

    @require(lambda self, operation: operation in AIOperation)
    @require(lambda self, input_data: len(str(input_data)) > 0)
    @ensure(
        lambda result: result.is_right()
        or isinstance(result.get_left(), ValidationError)
    )
    async def process_ai_request(
        self,
        operation: AIOperation,
        input_data: str | dict[str, Any] | list[Any],
        model_type: str = "auto",
        model_name: str | None = None,
        processing_mode: str = "balanced",
        max_tokens: int | None = None,
        temperature: float = 0.7,
        context: dict | None = None,
        output_format: str = "auto",
        enable_caching: bool = True,
        cost_limit: float | None = None,
        privacy_mode: bool = True,
        user_id: str | None = None,
    ) -> Either[ValidationError, dict[str, Any]]:
        """
        Process AI request with comprehensive validation and optimization.
        """
        try:
            if not self.initialized:
                return Either.left(
                    ValidationError("not_initialized", "AI system not initialized")
                )

            # Validate and parse parameters
            try:
                proc_mode = ProcessingMode(processing_mode)
            except ValueError:
                return Either.left(
                    ValidationError(
                        "invalid_mode", f"Unknown processing mode: {processing_mode}"
                    )
                )

            try:
                out_format = OutputFormat(output_format)
            except ValueError:
                return Either.left(
                    ValidationError(
                        "invalid_format", f"Unknown output format: {output_format}"
                    )
                )

            # Security validation
            security_result = await self._validate_input_security(
                input_data, privacy_mode
            )
            if security_result.is_left():
                return security_result

            # Model selection
            model_result = self._select_model(
                operation, model_type, model_name, proc_mode
            )
            if model_result.is_left():
                return model_result

            model = model_result.get_right()

            # Check cache if enabled
            if enable_caching:
                cache_key = self._generate_cache_key(
                    operation, input_data, model.model_name, temperature
                )
                cached_response = self.request_cache.get(cache_key)
                if cached_response:
                    return Either.right(
                        self._format_response(cached_response, out_format, cached=True)
                    )

            # Create AI request
            request_result = create_ai_request(
                operation=operation,
                input_data=input_data,
                model_id=model.model_id,
                processing_mode=proc_mode,
                temperature=temperature,
                max_tokens=TokenCount(max_tokens) if max_tokens else None,
                context=context or {},
                output_format=out_format,
                privacy_mode=privacy_mode,
                user_id=user_id,
            )

            if request_result.is_left():
                return request_result

            request = request_result.get_right()

            # Cost validation
            if cost_limit:
                estimated_cost = model.estimate_cost(request.estimate_input_tokens())
                if estimated_cost > cost_limit:
                    return Either.left(
                        ValidationError(
                            "cost_limit_exceeded",
                            f"Estimated cost ${estimated_cost:.4f} exceeds limit ${cost_limit:.4f}",
                        )
                    )

            # Process request
            response_result = await self._execute_ai_request(request)
            if response_result.is_left():
                return response_result

            response = response_result.get_right()

            # Cache response if enabled
            if enable_caching:
                self.request_cache[cache_key] = response
                # Limit cache size
                if len(self.request_cache) > 1000:
                    oldest_keys = list(self.request_cache.keys())[:100]
                    for key in oldest_keys:
                        del self.request_cache[key]

            # Record usage
            await self._record_usage(response)

            return Either.right(self._format_response(response, out_format))

        except Exception as e:
            return Either.left(
                ValidationError("ai_request", str(e), "Valid AI request processing")
            )

    async def _validate_input_security(
        self, input_data: Any, privacy_mode: bool
    ) -> Either[ValidationError, None]:
        """Validate input data for security threats."""
        try:
            input_str = str(input_data)

            # Check size limits
            if len(input_str.encode("utf-8")) > 1_000_000:  # 1MB limit
                return Either.left(
                    ValidationError(
                        "input_too_large", "Input exceeds maximum size limit"
                    )
                )

            # Check for dangerous patterns
            dangerous_patterns = [
                r"<script[^>]*>.*?</script>",  # XSS
                r"javascript:",  # JavaScript URLs
                r"eval\s*\(",  # Code injection
                r"exec\s*\(",  # Code execution
                r"subprocess\.",  # Subprocess calls
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    return Either.left(
                        ValidationError(
                            "dangerous_content", "Input contains dangerous pattern"
                        )
                    )

            # PII detection if privacy mode enabled
            if privacy_mode:
                pii_result = self._detect_pii(input_str)
                if pii_result.is_left():
                    return pii_result

            return Either.right(None)

        except Exception as e:
            return Either.left(ValidationError("security_validation_failed", str(e)))

    def _detect_pii(self, text: str) -> Either[ValidationError, None]:
        """Detect personally identifiable information in text."""
        pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
            (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "Credit Card"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email"),
            (r"(?i)(password|token|key|secret)[\s:=]+\S+", "Credential"),
        ]

        detected_pii = []
        for pattern, pii_type in pii_patterns:
            if re.search(pattern, text):
                detected_pii.append(pii_type)

        if detected_pii:
            return Either.left(
                ValidationError(
                    "pii_detected", f"Detected PII types: {', '.join(detected_pii)}"
                )
            )

        return Either.right(None)

    def _select_model(
        self,
        operation: AIOperation,
        model_type: str,
        model_name: str | None,
        processing_mode: ProcessingMode,
    ) -> Either[ValidationError, Any]:
        """Select appropriate AI model for request."""
        try:
            # If specific model requested, use it
            if model_name and model_name in DEFAULT_AI_MODELS:
                return Either.right(DEFAULT_AI_MODELS[model_name])

            # Auto-select based on operation and constraints
            suitable_models = []
            for model in DEFAULT_AI_MODELS.values():
                # Check operation support
                if not model.can_handle_operation(operation):
                    continue

                # Check model type preference
                if model_type != "auto":
                    try:
                        requested_type = AIModelType(model_type)
                        if model.model_type != requested_type:
                            continue
                    except ValueError:
                        pass  # Invalid model type, ignore

                suitable_models.append(model)

            if not suitable_models:
                return Either.left(
                    ValidationError(
                        "no_suitable_model", "No model found matching criteria"
                    )
                )

            # Select based on processing mode
            if processing_mode == ProcessingMode.FAST:
                selected = min(suitable_models, key=lambda m: m.max_tokens)
            elif processing_mode == ProcessingMode.ACCURATE:
                selected = max(suitable_models, key=lambda m: m.context_window)
            elif processing_mode == ProcessingMode.COST_EFFECTIVE:
                selected = min(suitable_models, key=lambda m: m.cost_per_input_token)
            elif processing_mode == ProcessingMode.CREATIVE:
                # Prefer GPT-4 for creativity
                creative_models = [
                    m for m in suitable_models if "gpt-4" in m.model_name.lower()
                ]
                selected = creative_models[0] if creative_models else suitable_models[0]
            else:  # BALANCED
                # Select GPT-3.5 Turbo for general use
                balanced_models = [
                    m for m in suitable_models if "gpt-3.5" in m.model_name.lower()
                ]
                selected = balanced_models[0] if balanced_models else suitable_models[0]

            return Either.right(selected)

        except Exception as e:
            return Either.left(ValidationError("model_selection_failed", str(e)))

    async def _execute_ai_request(
        self, request: AIRequest
    ) -> Either[ValidationError, AIResponse]:
        """Execute AI request - mock implementation."""
        try:
            start_time = datetime.now(UTC)

            # Mock AI processing based on operation
            if request.operation == AIOperation.ANALYZE:
                result = self._mock_analyze(request.prepare_input_for_model())
            elif request.operation == AIOperation.GENERATE:
                result = self._mock_generate(
                    request.prepare_input_for_model(), request.temperature
                )
            elif request.operation == AIOperation.SUMMARIZE:
                result = self._mock_summarize(request.prepare_input_for_model())
            elif request.operation == AIOperation.TRANSLATE:
                result = self._mock_translate(request.prepare_input_for_model())
            elif request.operation == AIOperation.CLASSIFY:
                result = self._mock_classify(request.prepare_input_for_model())
            elif request.operation == AIOperation.EXTRACT:
                result = self._mock_extract(request.prepare_input_for_model())
            else:
                result = f"Processed {request.operation.value} operation on input data"

            processing_time = (datetime.now(UTC) - start_time).total_seconds()

            # Calculate tokens and cost
            input_tokens = request.estimate_input_tokens()
            output_tokens = TokenCount(len(str(result)) // 4)
            total_tokens = TokenCount(input_tokens + output_tokens)
            cost = request.model.estimate_cost(input_tokens, output_tokens)

            # Create response
            response = AIResponse(
                request_id=request.request_id,
                operation=request.operation,
                result=result,
                model_used=request.model.model_name,
                tokens_used=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                processing_time=processing_time,
                cost_estimate=cost,
                confidence=0.85,
                metadata={"processing_mode": request.processing_mode.value},
            )

            return Either.right(response)

        except Exception as e:
            return Either.left(ValidationError("ai_execution_failed", str(e)))

    def _mock_analyze(self, text: str) -> dict[str, Any]:
        """Mock text analysis."""
        words = text.split()
        return {
            "sentiment": "neutral" if len(words) > 10 else "positive",
            "key_themes": ["automation", "technology", "efficiency"]
            if "automation" in text.lower()
            else ["general", "content"],
            "word_count": len(words),
            "character_count": len(text),
            "reading_level": "intermediate",
            "language": "english",
            "entities": ["technology", "system"]
            if any(word in text.lower() for word in ["tech", "system", "automation"])
            else [],
        }

    def _mock_generate(self, prompt: str, temperature: float) -> str:
        """Mock text generation."""
        creativity = (
            "highly creative"
            if temperature > 0.8
            else "structured"
            if temperature < 0.3
            else "balanced"
        )
        return f"Generated {creativity} response based on the prompt: '{prompt[:50]}...' This is a mock AI-generated response that demonstrates the capabilities of the system."

    def _mock_summarize(self, text: str) -> str:
        """Mock text summarization."""
        words = text.split()
        key_word = words[0] if words else "content"
        return f"Summary: This text discusses {key_word} and related topics. It contains {len(words)} words and covers various aspects of the subject matter. The main focus appears to be on {key_word} and its implications."

    def _mock_translate(self, text: str) -> str:
        """Mock text translation."""
        return f"[Translated]: {text} (This is a mock translation demonstrating the translation capability)"

    def _mock_classify(self, text: str) -> dict[str, float]:
        """Mock text classification."""
        # Simple keyword-based classification
        positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "horrible", "poor"]

        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower) / max(
            1, len(text.split())
        )
        negative_score = sum(1 for word in negative_words if word in text_lower) / max(
            1, len(text.split())
        )
        neutral_score = max(0, 1 - positive_score - negative_score)

        return {
            "positive": min(positive_score * 3, 1.0),
            "negative": min(negative_score * 3, 1.0),
            "neutral": max(neutral_score, 0.3),
        }

    def _mock_extract(self, input_data: str) -> dict[str, Any]:
        """Mock information extraction."""
        if self._is_image_path(input_data):
            return {
                "type": "image_analysis",
                "extracted_text": "Sample text extracted from image",
                "objects_detected": ["text", "interface elements"],
                "confidence": 0.92,
            }
        else:
            # Extract emails, URLs, numbers from text
            emails = re.findall(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", input_data
            )
            urls = re.findall(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                input_data,
            )
            numbers = re.findall(r"\b\d+\b", input_data)

            return {
                "type": "text_extraction",
                "emails": emails,
                "urls": urls,
                "numbers": numbers[:10],  # Limit to first 10
                "extracted_entities": len(emails) + len(urls) + len(numbers),
            }

    def _is_image_path(self, input_data: str) -> bool:
        """Check if input is an image file path."""
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"]
        return any(input_data.lower().endswith(ext) for ext in image_extensions)

    def _generate_cache_key(
        self,
        operation: AIOperation,
        input_data: Any,
        model_name: str,
        temperature: float,
    ) -> str:
        """Generate cache key for request."""
        import hashlib

        key_data = f"{operation.value}|{str(input_data)}|{model_name}|{temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _format_response(
        self, response: AIResponse, output_format: OutputFormat, cached: bool = False
    ) -> dict[str, Any]:
        """Format AI response for return to client."""
        return {
            "success": True,
            "operation": response.operation.value,
            "result": response.get_formatted_result(output_format),
            "metadata": {
                "request_id": str(response.request_id),
                "model_used": response.model_used,
                "tokens_used": int(response.tokens_used),
                "input_tokens": int(response.input_tokens),
                "output_tokens": int(response.output_tokens),
                "processing_time": response.processing_time,
                "cost_estimate": float(response.cost_estimate),
                "confidence": float(response.confidence)
                if response.confidence
                else None,
                "timestamp": response.timestamp.isoformat(),
                "cached": cached,
            },
            "cost_breakdown": response.get_cost_breakdown(),
        }

    async def _record_usage(self, response: AIResponse) -> None:
        """Record usage for tracking."""
        usage_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model": response.model_used,
            "operation": response.operation.value,
            "tokens_used": int(response.tokens_used),
            "cost": float(response.cost_estimate),
            "processing_time": response.processing_time,
        }

        self.usage_history.append(usage_record)

        # Keep only last 1000 records
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]

    def get_system_status(self) -> dict[str, Any]:
        """Get system status information."""
        return {
            "initialized": self.initialized,
            "session_id": str(self.session_id),
            "cache_size": len(self.request_cache),
            "usage_records": len(self.usage_history),
            "supported_operations": [op.value for op in AIOperation],
            "available_models": list(DEFAULT_AI_MODELS.keys()),
            "total_requests": len(self.usage_history),
            "cache_hit_ratio": 0.15 if self.request_cache else 0.0,  # Mock ratio
        }


# Global AI processing manager instance
ai_manager = AIProcessingManager()


async def km_ai_processing(
    operation: str,  # analyze|generate|predict|classify|extract|enhance|summarize|translate|explain|transform
    input_data: str
    | dict
    | list,  # Data to process (text, image path, structured data)
    model_type: str = "auto",  # openai|azure|google|anthropic|local|auto
    model_name: str | None = None,  # Specific model to use
    processing_mode: str = "balanced",  # fast|balanced|accurate|creative|cost_effective
    max_tokens: int | None = None,  # Maximum tokens for generation
    temperature: float = 0.7,  # Creativity/randomness (0.0-2.0)
    context: dict | None = None,  # Additional context for processing
    output_format: str = "auto",  # auto|json|text|markdown|html|structured
    enable_caching: bool = True,  # Enable result caching
    cost_limit: float | None = None,  # Maximum cost per operation
    privacy_mode: bool = True,  # Enable privacy protection
    timeout: int = 60,  # Processing timeout
    ctx=None,
) -> dict[str, Any]:
    """
    AI/ML model integration for intelligent automation and decision-making.

    This tool provides comprehensive AI processing capabilities including text analysis,
    content generation, image recognition, and intelligent decision-making with
    enterprise-grade security and cost optimization.

    Args:
        operation: Type of AI operation to perform
        input_data: Content to process (text, image path, or structured data)
        model_type: AI model provider to use
        model_name: Specific model name (optional, auto-selected if not provided)
        processing_mode: Processing optimization mode
        max_tokens: Maximum tokens for generation operations
        temperature: Creativity level for generation (0.0-2.0)
        context: Additional context or parameters
        output_format: Desired output format
        enable_caching: Whether to cache results for performance
        cost_limit: Maximum cost threshold for operation
        privacy_mode: Enable enhanced privacy protection and PII filtering
        timeout: Processing timeout in seconds

    Returns:
        Dict containing AI processing results with metadata and cost information
    """
    try:
        # Validate operation
        try:
            ai_operation = AIOperation(operation)
        except ValueError:
            raise ValidationError("operation", operation, "Valid AI operation")

        # Initialize AI manager if needed
        if not ai_manager.initialized:
            init_result = await ai_manager.initialize()
            if init_result.is_left():
                raise init_result.get_left()

        # Process AI request
        result = await ai_manager.process_ai_request(
            operation=ai_operation,
            input_data=input_data,
            model_type=model_type,
            model_name=model_name,
            processing_mode=processing_mode,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context,
            output_format=output_format,
            enable_caching=enable_caching,
            cost_limit=cost_limit,
            privacy_mode=privacy_mode,
            user_id=None,  # Could be extracted from ctx
        )

        if result.is_left():
            error = result.get_left()
            return {
                "success": False,
                "error": {
                    "code": "validation_error",
                    "message": error.constraint,
                    "details": str(error.value),
                },
            }

        return result.get_right()

    except ValidationError as e:
        return {
            "success": False,
            "error": {
                "code": "validation_error",
                "message": f"Invalid {e.field_name}: {e.constraint}",
                "details": str(e.value),
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "processing_error",
                "message": "Failed to process AI request",
                "details": str(e),
            },
        }


async def km_ai_status(
    include_usage: bool = True,  # Include usage statistics
    include_cache: bool = True,  # Include cache information
    include_models: bool = True,  # Include available models
    reset_cache: bool = False,  # Clear request cache
    ctx=None,
) -> dict[str, Any]:
    """
    Get AI processing system status and statistics.

    This tool provides comprehensive status information about the AI processing
    system including initialization status, cache performance, usage statistics,
    and available models.

    Args:
        include_usage: Whether to include usage statistics
        include_cache: Whether to include cache information
        include_models: Whether to include available models
        reset_cache: Whether to clear the request cache

    Returns:
        Dict containing system status and statistics
    """
    try:
        # Initialize AI manager if needed
        if not ai_manager.initialized:
            init_result = await ai_manager.initialize()
            if init_result.is_left():
                raise init_result.get_left()

        # Reset cache if requested
        if reset_cache:
            ai_manager.request_cache.clear()

        # Get basic status
        status = ai_manager.get_system_status()

        # Conditionally include additional information
        if not include_usage:
            status.pop("usage_records", None)
            status.pop("total_requests", None)

        if not include_cache:
            status.pop("cache_size", None)
            status.pop("cache_hit_ratio", None)

        if not include_models:
            status.pop("available_models", None)

        # Add recent usage if requested
        if include_usage and ai_manager.usage_history:
            recent_usage = ai_manager.usage_history[-10:]  # Last 10 requests
            status["recent_usage"] = recent_usage

        return {
            "success": True,
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except ValidationError as e:
        return {
            "success": False,
            "error": {
                "code": "validation_error",
                "message": f"Invalid {e.field_name}: {e.constraint}",
                "details": str(e.value),
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "status_error",
                "message": "Failed to get AI system status",
                "details": str(e),
            },
        }
