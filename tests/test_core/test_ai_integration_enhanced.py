"""Comprehensive tests for core AI integration module.

import logging

logging.basicConfig(level=logging.DEBUG)
Tests cover all AI processing functionality including model management,
request/response handling, security validation, and cost optimization.
Implements property-based testing for robust validation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from hypothesis import assume, given
from hypothesis import strategies as st
from src.core.ai_integration import (
    DEFAULT_AI_MODELS,
    AIModel,
    AIModelId,
    AIModelType,
    AIOperation,
    AIRequest,
    AIRequestId,
    AIResponse,
    AISecurityConfig,
    AISecurityLevel,
    AIUsageStats,
    ConfidenceScore,
    CostAmount,
    OutputFormat,
    ProcessingMode,
    TokenCount,
    create_ai_request,
    create_ai_session,
)
from src.core.errors import ValidationError


class TestAIModel:
    """Test AI model functionality and validation."""

    def test_ai_model_creation_valid(self) -> None:
        """Test creating valid AI model."""
        model = AIModel(
            model_id=AIModelId("test-model"),
            model_type=AIModelType.OPENAI,
            model_name="gpt-test",
            display_name="Test GPT",
            max_tokens=TokenCount(1000),
            cost_per_input_token=CostAmount(0.001),
            cost_per_output_token=CostAmount(0.002),
            context_window=TokenCount(2000),
        )

        assert model.model_id == AIModelId("test-model")
        assert model.model_type == AIModelType.OPENAI
        assert model.model_name == "gpt-test"
        assert model.max_tokens == TokenCount(1000)
        assert model.context_window == TokenCount(2000)

    def test_ai_model_cost_estimation(self) -> None:
        """Test cost estimation functionality."""
        model = AIModel(
            model_id=AIModelId("cost-test"),
            model_type=AIModelType.OPENAI,
            model_name="cost-model",
            display_name="Cost Model",
            cost_per_input_token=CostAmount(0.001),
            cost_per_output_token=CostAmount(0.002),
        )

        cost = model.estimate_cost(
            input_tokens=TokenCount(100),
            output_tokens=TokenCount(50),
        )

        expected_cost = (100 * 0.001) + (50 * 0.002)
        assert cost == CostAmount(expected_cost)

    def test_ai_model_operation_support(self) -> None:
        """Test operation support checking."""
        vision_model = AIModel(
            model_id=AIModelId("vision-test"),
            model_type=AIModelType.OPENAI,
            model_name="vision-model",
            display_name="Vision Model",
            supports_vision=True,
            context_window=TokenCount(1000),
        )

        text_model = AIModel(
            model_id=AIModelId("text-test"),
            model_type=AIModelType.OPENAI,
            model_name="text-model",
            display_name="Text Model",
            supports_vision=False,
            context_window=TokenCount(1000),
        )

        # Vision operations with vision model
        assert vision_model.can_handle_operation(AIOperation.ANALYZE, 0)
        assert vision_model.can_handle_operation(AIOperation.EXTRACT, 0)

        # Text operations with text model
        assert text_model.can_handle_operation(AIOperation.GENERATE, 500)
        assert text_model.can_handle_operation(AIOperation.SUMMARIZE, 800)

        # Context window limits
        assert not text_model.can_handle_operation(AIOperation.GENERATE, 2000)

    def test_ai_model_rate_limiting(self) -> None:
        """Test rate limiting functionality."""
        model = AIModel(
            model_id=AIModelId("rate-test"),
            model_type=AIModelType.OPENAI,
            model_name="rate-model",
            display_name="Rate Model",
            rate_limit_per_minute=60,
        )

        assert model.is_within_rate_limit(30)
        assert model.is_within_rate_limit(59)
        assert not model.is_within_rate_limit(60)
        assert not model.is_within_rate_limit(100)

    @given(st.text(min_size=1), st.integers(min_value=1, max_value=100000))
    def test_ai_model_property_validation(
        self,
        model_name: str,
        max_tokens: int,
    ) -> None:
        """Property test for AI model validation."""
        assume(len(model_name.strip()) > 0)

        model = AIModel(
            model_id=AIModelId("prop-test"),
            model_type=AIModelType.OPENAI,
            model_name=model_name.strip(),
            display_name=f"Test {model_name.strip()}",
            max_tokens=TokenCount(max_tokens),
            context_window=TokenCount(max_tokens * 2),
        )

        # Properties that should always hold
        assert len(model.model_name) > 0
        assert model.max_tokens > 0
        assert model.context_window > 0
        assert model.cost_per_input_token >= 0.0
        assert model.cost_per_output_token >= 0.0
        assert model.rate_limit_per_minute > 0


class TestAIRequest:
    """Test AI request functionality and validation."""

    def test_ai_request_creation_valid(self) -> None:
        """Test creating valid AI request."""
        model = DEFAULT_AI_MODELS["gpt-3.5-turbo"]

        request = AIRequest(
            request_id=AIRequestId("test-request"),
            operation=AIOperation.GENERATE,
            input_data="Test input text",
            model=model,
            temperature=0.7,
            max_tokens=TokenCount(500),
        )

        assert request.request_id == AIRequestId("test-request")
        assert request.operation == AIOperation.GENERATE
        assert request.input_data == "Test input text"
        assert request.temperature == 0.7
        assert request.max_tokens == TokenCount(500)

    def test_ai_request_input_preparation(self) -> None:
        """Test input data preparation for different types."""
        model = DEFAULT_AI_MODELS["gpt-3.5-turbo"]

        # String input
        str_request = AIRequest(
            request_id=AIRequestId("str-test"),
            operation=AIOperation.GENERATE,
            input_data="Simple string",
            model=model,
        )
        assert str_request.prepare_input_for_model() == "Simple string"

        # Dict input
        dict_request = AIRequest(
            request_id=AIRequestId("dict-test"),
            operation=AIOperation.ANALYZE,
            input_data={"key": "value", "number": 42},
            model=model,
        )
        prepared_dict = dict_request.prepare_input_for_model()
        assert isinstance(prepared_dict, str)
        assert "key" in prepared_dict

        # List input
        list_request = AIRequest(
            request_id=AIRequestId("list-test"),
            operation=AIOperation.CLASSIFY,
            input_data=["item1", "item2", "item3"],
            model=model,
        )
        prepared_list = list_request.prepare_input_for_model()
        assert "item1" in prepared_list
        assert "item2" in prepared_list

    def test_ai_request_token_estimation(self) -> None:
        """Test token count estimation."""
        model = DEFAULT_AI_MODELS["gpt-3.5-turbo"]

        request = AIRequest(
            request_id=AIRequestId("token-test"),
            operation=AIOperation.GENERATE,
            input_data="This is a test input with some text",
            model=model,
            system_prompt="You are a helpful assistant",
        )

        tokens = request.estimate_input_tokens()
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_ai_request_model_validation(self) -> None:
        """Test request validation against model capabilities."""
        # Model with small context window
        small_model = AIModel(
            model_id=AIModelId("small-test"),
            model_type=AIModelType.OPENAI,
            model_name="small-model",
            display_name="Small Model",
            context_window=TokenCount(100),
            supports_vision=False,
        )

        # Request that exceeds context window
        large_request = AIRequest(
            request_id=AIRequestId("large-test"),
            operation=AIOperation.GENERATE,
            input_data="x" * 1000,  # Large input
            model=small_model,
        )

        validation_result = large_request.validate_for_model()
        assert validation_result.is_left()

        # Valid request
        small_request = AIRequest(
            request_id=AIRequestId("small-test"),
            operation=AIOperation.GENERATE,
            input_data="Small input",
            model=small_model,
        )

        validation_result = small_request.validate_for_model()
        assert validation_result.is_right()

    @given(
        st.text(min_size=1, max_size=1000),
        st.floats(min_value=0.0, max_value=2.0),
        st.integers(min_value=1, max_value=4096),
    )
    def test_ai_request_property_validation(
        self,
        input_text: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        """Property test for AI request validation."""
        assume(len(input_text.strip()) > 0)

        model = DEFAULT_AI_MODELS["gpt-3.5-turbo"]

        request = AIRequest(
            request_id=AIRequestId("prop-test"),
            operation=AIOperation.GENERATE,
            input_data=input_text.strip(),
            model=model,
            temperature=temperature,
            max_tokens=TokenCount(max_tokens),
        )

        # Properties that should always hold
        assert 0.0 <= request.temperature <= 2.0
        assert request.max_tokens > 0
        assert len(str(request.input_data)) > 0
        # Token count should be >= 0 (some minimal inputs might have 0 tokens)
        assert request.estimate_input_tokens() >= 0


class TestAIResponse:
    """Test AI response functionality and formatting."""

    def test_ai_response_creation_valid(self) -> None:
        """Test creating valid AI response."""
        response = AIResponse(
            request_id=AIRequestId("test-request"),
            operation=AIOperation.GENERATE,
            result="Generated text result",
            model_used="gpt-3.5-turbo",
            tokens_used=TokenCount(150),
            input_tokens=TokenCount(100),
            output_tokens=TokenCount(50),
            processing_time=1.5,
            cost_estimate=CostAmount(0.025),
            confidence=ConfidenceScore(0.85),
        )

        assert response.request_id == AIRequestId("test-request")
        assert response.operation == AIOperation.GENERATE
        assert response.result == "Generated text result"
        assert response.tokens_used == TokenCount(150)
        assert response.confidence == ConfidenceScore(0.85)

    def test_ai_response_confidence_checking(self) -> None:
        """Test confidence threshold checking."""
        high_confidence_response = AIResponse(
            request_id=AIRequestId("high-conf"),
            operation=AIOperation.CLASSIFY,
            result="Classification result",
            model_used="gpt-4",
            tokens_used=TokenCount(100),
            input_tokens=TokenCount(80),
            output_tokens=TokenCount(20),
            processing_time=0.5,
            cost_estimate=CostAmount(0.01),
            confidence=ConfidenceScore(0.95),
        )

        low_confidence_response = AIResponse(
            request_id=AIRequestId("low-conf"),
            operation=AIOperation.CLASSIFY,
            result="Classification result",
            model_used="gpt-4",
            tokens_used=TokenCount(100),
            input_tokens=TokenCount(80),
            output_tokens=TokenCount(20),
            processing_time=0.5,
            cost_estimate=CostAmount(0.01),
            confidence=ConfidenceScore(0.65),
        )

        assert high_confidence_response.is_high_confidence()
        assert not low_confidence_response.is_high_confidence()
        assert low_confidence_response.is_high_confidence(ConfidenceScore(0.6))

    def test_ai_response_formatting(self) -> None:
        """Test response formatting in different formats."""
        response_data = {
            "analysis": "Text analysis result",
            "score": 0.85,
            "categories": ["positive", "confident"],
        }

        response = AIResponse(
            request_id=AIRequestId("format-test"),
            operation=AIOperation.ANALYZE,
            result=response_data,
            model_used="gpt-4",
            tokens_used=TokenCount(200),
            input_tokens=TokenCount(150),
            output_tokens=TokenCount(50),
            processing_time=2.0,
            cost_estimate=CostAmount(0.05),
        )

        # Test JSON formatting
        json_result = response.get_formatted_result(OutputFormat.JSON)
        assert json.loads(json_result)["analysis"] == "Text analysis result"

        # Test text formatting
        text_result = response.get_formatted_result(OutputFormat.TEXT)
        assert isinstance(text_result, str)

        # Test markdown formatting
        md_result = response.get_formatted_result(OutputFormat.MARKDOWN)
        assert "# AI Processing Result" in md_result

    def test_ai_response_cost_breakdown(self) -> None:
        """Test cost breakdown functionality."""
        response = AIResponse(
            request_id=AIRequestId("cost-test"),
            operation=AIOperation.GENERATE,
            result="Generated content",
            model_used="gpt-4",
            tokens_used=TokenCount(300),
            input_tokens=TokenCount(200),
            output_tokens=TokenCount(100),
            processing_time=3.0,
            cost_estimate=CostAmount(0.075),
        )

        breakdown = response.get_cost_breakdown()

        assert breakdown["total_cost"] == 0.075
        assert breakdown["input_tokens"] == 200
        assert breakdown["output_tokens"] == 100
        assert breakdown["total_tokens"] == 300
        assert breakdown["model_used"] == "gpt-4"
        assert breakdown["processing_time"] == 3.0


class TestAISecurityConfig:
    """Test AI security configuration and validation."""

    def test_security_config_creation(self) -> None:
        """Test creating security configuration."""
        config = AISecurityConfig(
            security_level=AISecurityLevel.STRICT,
            enable_content_filtering=True,
            enable_pii_detection=True,
            max_input_size=500_000,
            blocked_patterns=["password", "ssn", "credit_card"],
        )

        assert config.security_level == AISecurityLevel.STRICT
        assert config.enable_content_filtering
        assert config.max_input_size == 500_000
        assert "password" in config.blocked_patterns

    def test_security_config_scanning_logic(self) -> None:
        """Test content scanning decision logic."""
        strict_config = AISecurityConfig(
            security_level=AISecurityLevel.STRICT,
            enable_content_filtering=True,
            max_input_size=1000,
        )

        minimal_config = AISecurityConfig(
            security_level=AISecurityLevel.MINIMAL,
            enable_content_filtering=False,
            enable_pii_detection=False,
            enable_malware_scanning=False,
        )

        # Should scan with strict config and small content
        assert strict_config.should_scan_content(500)

        # Should not scan if content too large
        assert not strict_config.should_scan_content(2000)

        # Should not scan with minimal config
        assert not minimal_config.should_scan_content(500)


class TestAIUsageStats:
    """Test AI usage statistics tracking."""

    def test_usage_stats_creation(self) -> None:
        """Test creating usage statistics."""
        session_id = create_ai_session()
        stats = AIUsageStats(session_id=session_id)

        assert stats.session_id == session_id
        assert stats.total_requests == 0
        assert stats.total_tokens == TokenCount(0)
        assert stats.total_cost == CostAmount(0.0)

    def test_usage_stats_calculations(self) -> None:
        """Test usage statistics calculations."""
        session_id = create_ai_session()

        stats = AIUsageStats(
            session_id=session_id,
            total_requests=10,
            successful_requests=8,
            failed_requests=2,
            total_cost=CostAmount(1.50),
        )

        assert stats.get_success_rate() == 0.8
        assert stats.get_average_cost_per_request() == CostAmount(0.15)

        # Test zero division handling
        empty_stats = AIUsageStats(session_id=session_id)
        assert empty_stats.get_success_rate() == 0.0
        assert empty_stats.get_average_cost_per_request() == CostAmount(0.0)

    def test_usage_stats_session_duration(self) -> None:
        """Test session duration calculation."""
        session_id = create_ai_session()

        start_time = datetime.now(UTC)
        stats = AIUsageStats(
            session_id=session_id,
            start_time=start_time,
            last_request_time=start_time,
        )

        duration = stats.get_session_duration()
        assert duration >= 0.0


class TestAIRequestFactory:
    """Test AI request factory functions."""

    def test_create_ai_request_success(self) -> None:
        """Test successful AI request creation."""
        result = create_ai_request(
            operation=AIOperation.GENERATE,
            input_data="Test input for generation",
            temperature=0.8,
            max_tokens=TokenCount(200),
        )

        assert result.is_right()

        request = result.get_right()
        assert request.operation == AIOperation.GENERATE
        assert request.input_data == "Test input for generation"
        assert request.temperature == 0.8
        assert request.max_tokens == TokenCount(200)

    def test_create_ai_request_with_model_selection(self) -> None:
        """Test request creation with specific model."""
        result = create_ai_request(
            operation=AIOperation.ANALYZE,
            input_data="Content to analyze",
            model_id=AIModelId("gpt-4"),
        )

        assert result.is_right()

        request = result.get_right()
        assert request.model.model_name == "gpt-4"

    def test_create_ai_request_invalid_model(self) -> None:
        """Test request creation with invalid model."""
        result = create_ai_request(
            operation=AIOperation.GENERATE,
            input_data="Test input",
            model_id=AIModelId("invalid-model"),
        )

        assert result.is_left()

        error = result.get_left()
        assert "Model invalid-model not found" in error.constraint

    def test_create_ai_session(self) -> None:
        """Test AI session creation."""
        session_id = create_ai_session()

        assert isinstance(session_id, str)
        assert session_id.startswith("ai_session_")
        assert len(session_id) > len("ai_session_")

    @given(
        st.sampled_from(list(AIOperation)),
        st.text(min_size=1, max_size=500),
        st.floats(min_value=0.0, max_value=2.0),
    )
    def test_create_ai_request_property_validation(
        self,
        operation: AIOperation,
        input_text: str,
        temperature: float,
    ) -> None:
        """Property test for AI request creation."""
        assume(len(input_text.strip()) > 0)

        result = create_ai_request(
            operation=operation,
            input_data=input_text.strip(),
            temperature=temperature,
        )

        # Should either succeed or fail gracefully
        if result.is_right():
            request = result.get_right()
            assert request.operation == operation
            assert 0.0 <= request.temperature <= 2.0
            assert len(str(request.input_data)) > 0
        else:
            error = result.get_left()
            assert isinstance(error, ValidationError)


class TestDefaultModels:
    """Test predefined AI models."""

    def test_default_models_exist(self) -> None:
        """Test that default models are properly defined."""
        assert "gpt-4" in DEFAULT_AI_MODELS
        assert "gpt-3.5-turbo" in DEFAULT_AI_MODELS
        assert "gemini-pro" in DEFAULT_AI_MODELS

        # Check model properties
        gpt4 = DEFAULT_AI_MODELS["gpt-4"]
        assert gpt4.model_type == AIModelType.OPENAI
        assert gpt4.supports_function_calling

        gemini = DEFAULT_AI_MODELS["gemini-pro"]
        assert gemini.model_type == AIModelType.GOOGLE_AI

    def test_default_models_validation(self) -> None:
        """Test that all default models are valid."""
        for _model_name, model in DEFAULT_AI_MODELS.items():
            # Each model should be properly configured
            assert len(model.model_name) > 0
            assert model.max_tokens > 0
            assert model.context_window > 0
            assert model.cost_per_input_token >= 0.0
            assert model.cost_per_output_token >= 0.0
            assert model.rate_limit_per_minute > 0

    def test_vision_model_capabilities(self) -> None:
        """Test vision-enabled models."""
        vision_models = [
            model for model in DEFAULT_AI_MODELS.values() if model.supports_vision
        ]

        assert len(vision_models) > 0

        for model in vision_models:
            assert model.can_handle_operation(AIOperation.ANALYZE, 0)
            assert model.can_handle_operation(AIOperation.EXTRACT, 0)


class TestAIEnums:
    """Test AI enumeration types."""

    def test_ai_operation_values(self) -> None:
        """Test AI operation enum values."""
        operations = [op.value for op in AIOperation]

        expected_operations = [
            "analyze",
            "generate",
            "predict",
            "classify",
            "extract",
            "enhance",
            "summarize",
            "translate",
            "explain",
            "transform",
        ]

        for expected in expected_operations:
            assert expected in operations

    def test_model_type_values(self) -> None:
        """Test AI model type enum values."""
        types = [t.value for t in AIModelType]

        expected_types = ["openai", "azure", "google", "anthropic", "local", "auto"]

        for expected in expected_types:
            assert expected in types

    def test_processing_mode_values(self) -> None:
        """Test processing mode enum values."""
        modes = [m.value for m in ProcessingMode]

        expected_modes = ["fast", "balanced", "accurate", "creative", "cost_effective"]

        for expected in expected_modes:
            assert expected in modes


# Integration tests for AI workflow
class TestAIIntegrationWorkflow:
    """Test complete AI processing workflows."""

    def test_complete_ai_workflow(self) -> None:
        """Test complete AI processing workflow."""
        # 1. Create AI session
        session_id = create_ai_session()
        assert session_id is not None

        # 2. Create AI request
        request_result = create_ai_request(
            operation=AIOperation.GENERATE,
            input_data="Write a haiku about programming",
            temperature=0.7,
            max_tokens=TokenCount(100),
        )

        assert request_result.is_right()
        request = request_result.get_right()

        # 3. Validate request
        validation_result = request.validate_for_model()
        assert validation_result.is_right()

        # 4. Create response (simulated)
        response = AIResponse(
            request_id=request.request_id,
            operation=request.operation,
            result="Code flows like water\nBugs hide in silent branches\nDebug brings spring rain",
            model_used=request.model.model_name,
            tokens_used=TokenCount(45),
            input_tokens=TokenCount(25),
            output_tokens=TokenCount(20),
            processing_time=1.2,
            cost_estimate=request.model.estimate_cost(TokenCount(25), TokenCount(20)),
            confidence=ConfidenceScore(0.92),
        )

        # 5. Verify response
        assert response.is_high_confidence()
        assert response.tokens_used == TokenCount(45)

        # 6. Format response
        formatted = response.get_formatted_result(OutputFormat.MARKDOWN)
        assert "# AI Processing Result" in formatted

        # 7. Get cost breakdown
        breakdown = response.get_cost_breakdown()
        assert breakdown["total_cost"] > 0
        assert breakdown["model_used"] == request.model.model_name

    def test_security_validation_workflow(self) -> None:
        """Test AI processing with security validation."""
        # Create security config
        security_config = AISecurityConfig(
            security_level=AISecurityLevel.STRICT,
            enable_content_filtering=True,
            enable_pii_detection=True,
            max_input_size=1000,
            blocked_patterns=["secret", "password", "token"],
        )

        # Test content that should be scanned
        safe_content = "Analyze this business document for key insights"
        assert security_config.should_scan_content(len(safe_content))

        # Test content that's too large
        large_content = "x" * 2000
        assert not security_config.should_scan_content(len(large_content))

        # Create request with security context
        request_result = create_ai_request(
            operation=AIOperation.ANALYZE,
            input_data=safe_content,
            privacy_mode=True,
        )

        assert request_result.is_right()
        request = request_result.get_right()
        assert request.privacy_mode

    def test_cost_optimization_workflow(self) -> None:
        """Test cost optimization strategies."""
        # Compare costs between models
        gpt4_model = DEFAULT_AI_MODELS["gpt-4"]
        gpt35_model = DEFAULT_AI_MODELS["gpt-3.5-turbo"]

        input_tokens = TokenCount(1000)
        output_tokens = TokenCount(500)

        gpt4_cost = gpt4_model.estimate_cost(input_tokens, output_tokens)
        gpt35_cost = gpt35_model.estimate_cost(input_tokens, output_tokens)

        # GPT-3.5 should be more cost-effective
        assert gpt35_cost < gpt4_cost

        # Test model selection for cost-effective processing
        cost_effective_request = create_ai_request(
            operation=AIOperation.SUMMARIZE,
            input_data="Long document to summarize...",
            processing_mode=ProcessingMode.COST_EFFECTIVE,
        )

        assert cost_effective_request.is_right()
        request = cost_effective_request.get_right()
        assert request.processing_mode == ProcessingMode.COST_EFFECTIVE
