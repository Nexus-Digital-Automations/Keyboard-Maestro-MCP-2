"""Comprehensive test coverage for AI integration core module.

Tests the complete AI integration type system including branded types,
data classes, enums, and business logic validation following ADDER+ methodology.
"""

from hypothesis import given
from hypothesis import strategies as st
from src.core.ai_integration import (
    AIModel,
    AIModelId,
    AIModelType,
    AIOperation,
    AIRequest,
    AIRequestId,
    AISessionId,
    ConfidenceScore,
    CostAmount,
    ProcessingMode,
    TokenCount,
)


class TestBrandedTypes:
    """Test branded types for AI integration."""

    def test_ai_model_id_creation(self):
        """Test AIModelId branded type creation."""
        model_id = AIModelId("gpt-4")
        assert isinstance(model_id, str)
        assert model_id == "gpt-4"

    def test_ai_request_id_creation(self):
        """Test AIRequestId branded type creation."""
        request_id = AIRequestId("req_123")
        assert isinstance(request_id, str)
        assert request_id == "req_123"

    def test_ai_session_id_creation(self):
        """Test AISessionId branded type creation."""
        session_id = AISessionId("session_abc")
        assert isinstance(session_id, str)
        assert session_id == "session_abc"

    def test_token_count_creation(self):
        """Test TokenCount branded type creation."""
        token_count = TokenCount(150)
        assert isinstance(token_count, int)
        assert token_count == 150

    def test_cost_amount_creation(self):
        """Test CostAmount branded type creation."""
        cost = CostAmount(0.05)
        assert isinstance(cost, float)
        assert cost == 0.05

    def test_confidence_score_creation(self):
        """Test ConfidenceScore branded type creation."""
        confidence = ConfidenceScore(0.95)
        assert isinstance(confidence, float)
        assert confidence == 0.95


class TestAIOperationEnum:
    """Test AIOperation enum values and behavior."""

    def test_ai_operation_values(self):
        """Test all AIOperation enum values."""
        assert AIOperation.ANALYZE.value == "analyze"
        assert AIOperation.GENERATE.value == "generate"
        assert AIOperation.PREDICT.value == "predict"
        assert AIOperation.CLASSIFY.value == "classify"
        assert AIOperation.EXTRACT.value == "extract"

    def test_ai_operation_enum_complete(self):
        """Test AIOperation enum completeness."""
        expected_operations = {
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
        }
        actual_operations = {op.value for op in AIOperation}
        assert actual_operations == expected_operations


class TestAIModelTypeEnum:
    """Test AIModelType enum values and behavior."""

    def test_ai_model_type_values(self):
        """Test all AIModelType enum values."""
        assert AIModelType.OPENAI.value == "openai"
        assert AIModelType.AZURE_OPENAI.value == "azure"
        assert AIModelType.GOOGLE_AI.value == "google"
        assert AIModelType.ANTHROPIC.value == "anthropic"
        assert AIModelType.LOCAL.value == "local"
        assert AIModelType.AUTO.value == "auto"

    def test_ai_model_type_enum_complete(self):
        """Test AIModelType enum completeness."""
        expected_types = {"openai", "azure", "google", "anthropic", "local", "auto"}
        actual_types = {mt.value for mt in AIModelType}
        assert actual_types == expected_types


class TestProcessingModeEnum:
    """Test ProcessingMode enum values and behavior."""

    def test_processing_mode_values(self):
        """Test all ProcessingMode enum values."""
        assert ProcessingMode.FAST.value == "fast"
        assert ProcessingMode.BALANCED.value == "balanced"
        assert ProcessingMode.ACCURATE.value == "accurate"
        assert ProcessingMode.CREATIVE.value == "creative"
        assert ProcessingMode.COST_EFFECTIVE.value == "cost_effective"


class TestAIModel:
    """Test AIModel dataclass functionality."""

    def test_ai_model_creation(self):
        """Test AIModel creation with valid parameters."""
        model = AIModel(
            model_id=AIModelId("gpt-4"),
            model_type=AIModelType.OPENAI,
            model_name="gpt-4",
            display_name="GPT-4",
            max_tokens=TokenCount(4096),
            cost_per_input_token=CostAmount(0.00003),
            cost_per_output_token=CostAmount(0.00006),
            rate_limit_per_minute=100,
        )

        assert model.model_id == AIModelId("gpt-4")
        assert model.model_name == "gpt-4"
        assert model.model_type == AIModelType.OPENAI
        assert model.max_tokens == TokenCount(4096)
        assert model.cost_per_input_token == CostAmount(0.00003)
        assert model.cost_per_output_token == CostAmount(0.00006)
        assert model.rate_limit_per_minute == 100

    def test_ai_model_estimate_cost(self):
        """Test AIModel cost estimation."""
        model = AIModel(
            model_id=AIModelId("gpt-4"),
            model_type=AIModelType.OPENAI,
            model_name="gpt-4",
            display_name="GPT-4",
            max_tokens=TokenCount(4096),
            cost_per_input_token=CostAmount(0.00003),
            cost_per_output_token=CostAmount(0.00006),
            rate_limit_per_minute=100,
        )

        cost = model.estimate_cost(TokenCount(1000), TokenCount(500))
        expected_cost = CostAmount(1000 * 0.00003 + 500 * 0.00006)  # input + output
        assert cost == expected_cost

    def test_ai_model_can_handle_operation(self):
        """Test AIModel operation handling capability."""
        model = AIModel(
            model_id=AIModelId("gpt-4"),
            model_type=AIModelType.OPENAI,
            model_name="gpt-4",
            display_name="GPT-4",
            context_window=TokenCount(4096),
            rate_limit_per_minute=100,
        )

        assert model.can_handle_operation(AIOperation.GENERATE, 2000) is True
        assert model.can_handle_operation(AIOperation.GENERATE, 5000) is False

    def test_ai_model_rate_limit_check(self):
        """Test AIModel rate limit validation."""
        model = AIModel(
            model_id=AIModelId("gpt-4"),
            model_type=AIModelType.OPENAI,
            model_name="gpt-4",
            display_name="GPT-4",
            rate_limit_per_minute=100,
        )

        assert model.is_within_rate_limit(50) is True
        assert model.is_within_rate_limit(100) is False
        assert model.is_within_rate_limit(150) is False


class TestAIRequest:
    """Test AIRequest dataclass functionality."""

    def test_ai_request_creation(self):
        """Test AIRequest creation with valid parameters."""
        model = AIModel(
            model_id=AIModelId("gpt-4"),
            model_type=AIModelType.OPENAI,
            model_name="gpt-4",
            display_name="GPT-4",
        )

        request = AIRequest(
            request_id=AIRequestId("req_123"),
            operation=AIOperation.ANALYZE,
            input_data="Sample text to analyze",
            model=model,
            processing_mode=ProcessingMode.BALANCED,
        )

        assert request.request_id == AIRequestId("req_123")
        assert request.operation == AIOperation.ANALYZE
        assert request.input_data == "Sample text to analyze"
        assert request.model == model
        assert request.processing_mode == ProcessingMode.BALANCED

    def test_ai_request_with_metadata(self):
        """Test AIRequest with additional metadata."""
        model = AIModel(
            model_id=AIModelId("claude-3"),
            model_type=AIModelType.ANTHROPIC,
            model_name="claude-3",
            display_name="Claude-3",
        )

        request = AIRequest(
            request_id=AIRequestId("req_456"),
            operation=AIOperation.GENERATE,
            input_data="Generate creative content",
            model=model,
            processing_mode=ProcessingMode.CREATIVE,
            context={"source": "user_input", "priority": "high"},
        )

        assert request.context == {"source": "user_input", "priority": "high"}


class TestDataclassMethods:
    """Test dataclass methods for AI integration."""

    def test_ai_request_get_effective_max_tokens(self):
        """Test AIRequest effective max tokens calculation."""
        model = AIModel(
            model_id=AIModelId("gpt-4"),
            model_type=AIModelType.OPENAI,
            model_name="gpt-4",
            display_name="GPT-4",
            max_tokens=TokenCount(4096),
        )

        request = AIRequest(
            request_id=AIRequestId("req_123"),
            operation=AIOperation.GENERATE,
            input_data="Test input",
            model=model,
        )

        effective_tokens = request.get_effective_max_tokens()
        assert effective_tokens == TokenCount(2048)  # model.max_tokens // 2

    def test_ai_request_prepare_input_for_model(self):
        """Test AIRequest input preparation."""
        model = AIModel(
            model_id=AIModelId("gpt-4"),
            model_type=AIModelType.OPENAI,
            model_name="gpt-4",
            display_name="GPT-4",
        )

        # Test string input
        request = AIRequest(
            request_id=AIRequestId("req_123"),
            operation=AIOperation.GENERATE,
            input_data="Test string",
            model=model,
        )

        prepared = request.prepare_input_for_model()
        assert prepared == "Test string"


class TestPropertyBasedValidation:
    """Property-based tests for AI integration."""

    @given(st.text(min_size=1, max_size=100))
    def test_ai_model_id_properties(self, model_name):
        """Property test for AI model ID creation."""
        model_id = AIModelId(model_name)
        assert isinstance(model_id, str)
        assert model_id == model_name

    @given(st.integers(min_value=1, max_value=10000))
    def test_token_count_properties(self, count):
        """Property test for token count handling."""
        token_count = TokenCount(count)
        assert isinstance(token_count, int)
        assert token_count == count
        assert token_count > 0

    @given(st.floats(min_value=0.0, max_value=2.0))
    def test_temperature_properties(self, temperature):
        """Property test for temperature validation in AIRequest."""
        model = AIModel(
            model_id=AIModelId("test-model"),
            model_type=AIModelType.OPENAI,
            model_name="test",
            display_name="Test Model",
        )

        if 0.0 <= temperature <= 2.0:
            request = AIRequest(
                request_id=AIRequestId("test"),
                operation=AIOperation.GENERATE,
                input_data="test",
                model=model,
                temperature=temperature,
            )
            assert request.temperature == temperature
        # Invalid temperatures would raise validation errors in __post_init__


class TestIntegrationScenarios:
    """Integration test scenarios for AI system."""

    def test_complete_ai_workflow(self):
        """Test complete AI processing workflow."""
        # Create model
        model = AIModel(
            model_id=AIModelId("gpt-4"),
            model_type=AIModelType.OPENAI,
            model_name="gpt-4",
            display_name="GPT-4",
            context_window=TokenCount(4096),
            cost_per_input_token=CostAmount(0.00003),
            cost_per_output_token=CostAmount(0.00006),
            rate_limit_per_minute=100,
        )

        # Create request
        request = AIRequest(
            request_id=AIRequestId("req_test"),
            operation=AIOperation.ANALYZE,
            input_data="Analyze this text for sentiment",
            model=model,
            processing_mode=ProcessingMode.ACCURATE,
        )

        # Validate workflow
        input_size = len(request.input_data)
        assert model.can_handle_operation(request.operation, input_size)
        cost = model.estimate_cost(TokenCount(100), TokenCount(50))  # estimated tokens
        assert isinstance(cost, float)
        assert cost > 0

    def test_rate_limiting_scenario(self):
        """Test rate limiting enforcement."""
        model = AIModel(
            model_id=AIModelId("claude-3"),
            model_type=AIModelType.ANTHROPIC,
            model_name="claude-3",
            display_name="Claude-3",
            context_window=TokenCount(8192),
            cost_per_input_token=CostAmount(0.00002),
            rate_limit_per_minute=50,
        )

        # Test within limit
        assert model.is_within_rate_limit(25) is True
        assert model.is_within_rate_limit(49) is True

        # Test at/over limit
        assert model.is_within_rate_limit(50) is False
        assert model.is_within_rate_limit(75) is False

    def test_cost_optimization_scenario(self):
        """Test cost optimization calculations."""
        models = [
            AIModel(
                model_id=AIModelId("gpt-3.5"),
                model_type=AIModelType.OPENAI,
                model_name="gpt-3.5",
                display_name="GPT-3.5",
                context_window=TokenCount(4096),
                cost_per_input_token=CostAmount(0.000002),
                rate_limit_per_minute=200,
            ),
            AIModel(
                model_id=AIModelId("gpt-4"),
                model_type=AIModelType.OPENAI,
                model_name="gpt-4",
                display_name="GPT-4",
                context_window=TokenCount(8192),
                cost_per_input_token=CostAmount(0.00003),
                rate_limit_per_minute=100,
            ),
        ]

        input_text = "Generate a creative story"
        token_estimate = TokenCount(len(input_text.split()) * 2)  # rough estimate

        costs = [model.estimate_cost(token_estimate) for model in models]

        # GPT-3.5 should be cheaper for input tokens
        assert costs[0] < costs[1]

        # Both should handle the operation
        for model in models:
            assert model.can_handle_operation(AIOperation.GENERATE, len(input_text))
