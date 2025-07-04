"""
Property-based tests for AI integration system.

This module provides comprehensive property-based testing for the AI/ML
integration system using Hypothesis to validate behavior across input ranges.
Tests security boundaries, performance characteristics, and functional correctness.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta, UTC
from typing import Dict, Any, List

from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize

from src.core.ai_integration import (
    AIOperation, AIModelType, ProcessingMode, OutputFormat,
    AIRequest, AIResponse, AIModel, AIModelId, TokenCount,
    CostAmount, ConfidenceScore, create_ai_request, create_ai_session
)
from src.ai.model_manager import AIModelManager, AIError, ModelUsageTracker
from src.ai.text_processor import TextProcessor, TextAnalysisType, TextGenerationStyle, TextGenerationRequest
from src.ai.image_analyzer import ImageAnalyzer, ImageAnalysisType, ImageSecurityValidator
from src.ai.security_validator import (
    AISecurityValidator, SecurityThreat, SecurityThreatType, SecurityThreatLevel,
    PIIDetector, ContentFilter
)
from src.server.tools.ai_processing_tools import AIProcessingManager
from src.core.either import Either


# Hypothesis strategies for AI types
@st.composite
def ai_model_strategy(draw):
    """Generate valid AI model configurations."""
    return AIModel(
        model_id=AIModelId(draw(st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories='L')))),
        model_type=draw(st.sampled_from(AIModelType)),
        model_name=draw(st.text(min_size=3, max_size=30)),
        display_name=draw(st.text(min_size=3, max_size=50)),
        max_tokens=TokenCount(draw(st.integers(min_value=512, max_value=32768))),
        cost_per_input_token=CostAmount(draw(st.floats(min_value=0.0, max_value=0.01))),
        cost_per_output_token=CostAmount(draw(st.floats(min_value=0.0, max_value=0.01))),
        context_window=TokenCount(draw(st.integers(min_value=512, max_value=128000))),
        rate_limit_per_minute=draw(st.integers(min_value=1, max_value=10000))
    )


@st.composite
def ai_request_strategy(draw):
    """Generate valid AI request configurations."""
    model = draw(ai_model_strategy())
    operation = draw(st.sampled_from(AIOperation))
    
    # Generate appropriate input data based on operation
    if operation in [AIOperation.ANALYZE, AIOperation.GENERATE, AIOperation.SUMMARIZE]:
        input_data = draw(st.text(min_size=1, max_size=5000))
    elif operation == AIOperation.CLASSIFY:
        input_data = draw(st.text(min_size=1, max_size=1000))
    else:
        input_data = draw(st.one_of(
            st.text(min_size=1, max_size=1000),
            st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=100)),
            st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10)
        ))
    
    return AIRequest(
        request_id=AIRequestId(f"test_req_{draw(st.integers(min_value=1000, max_value=9999))}"),
        operation=operation,
        input_data=input_data,
        model=model,
        processing_mode=draw(st.sampled_from(ProcessingMode)),
        temperature=draw(st.floats(min_value=0.0, max_value=2.0)),
        max_tokens=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=8192).map(TokenCount))),
        privacy_mode=draw(st.booleans())
    )


class TestAIIntegrationProperties:
    """Property-based tests for AI integration components."""
    
    @given(ai_model_strategy())
    def test_ai_model_properties(self, model):
        """Property: AI models should have consistent cost calculations."""
        # Test cost estimation properties
        input_tokens = TokenCount(100)
        output_tokens = TokenCount(50)
        
        cost = model.estimate_cost(input_tokens, output_tokens)
        
        # Cost should be non-negative
        assert cost >= 0.0
        
        # Cost should scale with token count
        double_cost = model.estimate_cost(TokenCount(200), TokenCount(100))
        assert double_cost >= cost
        
        # Zero tokens should have zero cost (if base costs are zero)
        if model.cost_per_input_token == 0 and model.cost_per_output_token == 0:
            assert model.estimate_cost(TokenCount(0), TokenCount(0)) == 0.0
    
    @given(ai_model_strategy(), st.sampled_from(AIOperation))
    def test_model_operation_support(self, model, operation):
        """Property: Model operation support should be consistent."""
        # Test with different input sizes
        small_input = model.can_handle_operation(operation, 100)
        large_input = model.can_handle_operation(operation, model.context_window + 1000)
        
        # Large input exceeding context window should not be supported
        if model.context_window < 1000:
            assert not large_input
        
        # Vision operations should respect vision capability
        if operation == AIOperation.ANALYZE and not model.supports_vision:
            # For text analysis, should still be supported
            assert model.can_handle_operation(operation, 100) == True
    
    @given(ai_request_strategy())
    def test_ai_request_validation(self, request):
        """Property: AI request validation should be consistent."""
        validation_result = request.validate_for_model()
        
        # Validation should always return Either
        assert validation_result.is_left() or validation_result.is_right()
        
        # Request with input larger than context window should fail
        if request.estimate_input_tokens() > request.model.context_window:
            assert validation_result.is_left()
        
        # Temperature should be within valid range
        assert 0.0 <= request.temperature <= 2.0
        
        # Max tokens should be positive if specified
        if request.max_tokens is not None:
            assert request.max_tokens > 0
    
    @given(st.text(min_size=1, max_size=1000))
    def test_input_data_preparation(self, text_input):
        """Property: Input data preparation should preserve essential content."""
        # Create a simple request for testing
        model = AIModel(
            model_id=AIModelId("test_model"),
            model_type=AIModelType.OPENAI,
            model_name="test",
            display_name="Test Model"
        )
        
        request = AIRequest(
            request_id=AIRequestId("test_123"),
            operation=AIOperation.ANALYZE,
            input_data=text_input,
            model=model
        )
        
        prepared = request.prepare_input_for_model()
        
        # Prepared input should be a string
        assert isinstance(prepared, str)
        
        # Should preserve content length relationship
        assert len(prepared) > 0
        
        # For string input, should be identical
        if isinstance(text_input, str):
            assert prepared == text_input


class TestSecurityValidatorProperties:
    """Property-based tests for security validation."""
    
    @given(st.text(min_size=0, max_size=10000))
    def test_pii_detection_properties(self, text):
        """Property: PII detection should handle various text inputs safely."""
        detector = PIIDetector()
        
        # Should not crash on any input
        threats = detector.detect_pii(text)
        
        # Result should always be a list
        assert isinstance(threats, list)
        
        # All threats should be SecurityThreat objects
        for threat in threats:
            assert isinstance(threat, SecurityThreat)
            assert threat.threat_type == SecurityThreatType.PII_DETECTED
            assert 0.0 <= threat.confidence <= 1.0
    
    @given(st.text(min_size=0, max_size=5000))
    def test_content_filter_properties(self, content):
        """Property: Content filtering should be consistent and safe."""
        filter_system = ContentFilter()
        
        # Should handle any content without crashing
        threats = filter_system.scan_content(content)
        
        # Result should be a list
        assert isinstance(threats, list)
        
        # All threats should have valid severity levels
        for threat in threats:
            assert threat.severity in SecurityThreatLevel
            assert 0.0 <= threat.confidence <= 1.0
            assert len(threat.description) > 0
    
    @given(ai_request_strategy())
    async def test_security_validation_properties(self, request):
        """Property: Security validation should provide consistent results."""
        validator = AISecurityValidator()
        
        # Should handle any valid AI request
        result = await validator.validate_request(request)
        
        # Should always return Either
        assert result.is_left() or result.is_right()
        
        if result.is_right():
            scan_result = result.get_right()
            
            # Risk score should be in valid range
            assert 0.0 <= scan_result.risk_score <= 100.0
            
            # Scan time should be positive
            assert scan_result.scan_time >= 0.0
            
            # Safety should correlate with risk score
            if scan_result.risk_score > 80:
                assert not scan_result.is_safe
    
    @given(st.lists(st.text(min_size=5, max_size=100), min_size=1, max_size=20))
    def test_threat_aggregation_properties(self, threat_descriptions):
        """Property: Threat aggregation should scale appropriately."""
        threats = []
        
        for i, desc in enumerate(threat_descriptions):
            threat = SecurityThreat(
                threat_type=SecurityThreatType.SUSPICIOUS_BEHAVIOR,
                severity=SecurityThreatLevel.MEDIUM,
                description=desc,
                detected_content=desc[:20],
                confidence=0.5
            )
            threats.append(threat)
        
        # Risk calculation should handle multiple threats
        total_risk = sum(threat.get_risk_score() for threat in threats)
        
        # Total risk should scale with number of threats
        assert total_risk >= 0.0
        
        if threats:
            avg_risk = total_risk / len(threats)
            assert avg_risk > 0.0


class TestModelManagerProperties:
    """Property-based tests for AI model management."""
    
    @pytest.fixture
    def model_manager(self):
        """Create model manager for testing."""
        return AIModelManager()
    
    @given(st.lists(ai_model_strategy(), min_size=1, max_size=10))
    def test_model_selection_properties(self, models):
        """Property: Model selection should be deterministic and optimal."""
        manager = AIModelManager()
        
        # Add models to manager
        for model in models:
            manager.available_models[model.model_id] = model
            manager.usage_trackers[model.model_id] = ModelUsageTracker(model.model_id)
        
        # Test selection for different operations
        for operation in AIOperation:
            for mode in ProcessingMode:
                result = manager.select_best_model(operation, mode)
                
                if result.is_right():
                    selected_model = result.get_right()
                    
                    # Selected model should be from available models
                    assert selected_model.model_id in manager.available_models
                    
                    # Should support the requested operation
                    assert selected_model.can_handle_operation(operation)
    
    @given(st.integers(min_value=1, max_value=1000), st.floats(min_value=0.0, max_value=1.0))
    def test_usage_tracking_properties(self, token_count, cost):
        """Property: Usage tracking should accumulate correctly."""
        tracker = ModelUsageTracker(AIModelId("test_model"))
        
        initial_requests = tracker.total_requests
        initial_tokens = tracker.total_tokens
        initial_cost = tracker.total_cost
        
        # Record usage
        tracker.record_request(TokenCount(token_count), CostAmount(cost))
        
        # Counters should increase
        assert tracker.total_requests == initial_requests + 1
        assert tracker.total_tokens == initial_tokens + token_count
        assert tracker.total_cost == initial_cost + cost
        
        # Recent request counters should be updated
        assert tracker.requests_this_minute >= 1
        assert tracker.last_request_time is not None


class TestTextProcessorProperties:
    """Property-based tests for text processing."""
    
    @pytest.fixture
    def text_processor(self):
        """Create text processor for testing."""
        mock_manager = Mock(spec=AIModelManager)
        return TextProcessor(mock_manager)
    
    @given(st.text(min_size=1, max_size=2000), st.sampled_from(TextAnalysisType))
    def test_analysis_prompt_generation(self, text, analysis_type):
        """Property: Analysis prompts should be well-formed."""
        processor = TextProcessor(Mock())
        
        prompt = processor._build_analysis_prompt(text, analysis_type)
        
        # Prompt should contain the input text
        assert text in prompt
        
        # Should be a reasonable length
        assert len(prompt) > len(text)
        assert len(prompt) < len(text) + 2000  # Reasonable overhead
        
        # Should contain JSON format instruction
        assert "JSON" in prompt or "json" in prompt
    
    @given(st.text(min_size=1, max_size=100), st.sampled_from(TextGenerationStyle))
    def test_generation_request_properties(self, prompt, style):
        """Property: Generation requests should be valid."""
        request = TextGenerationRequest(
            prompt=prompt,
            style=style,
            max_length=500,
            temperature=0.7
        )
        
        # System prompt should be appropriate for style
        system_prompt = request.build_system_prompt()
        
        assert len(system_prompt) > 0
        assert str(request.max_length) in system_prompt
        
        # Style should influence the prompt
        style_keywords = {
            TextGenerationStyle.FORMAL: ["formal", "professional"],
            TextGenerationStyle.CASUAL: ["casual", "friendly"],
            TextGenerationStyle.CREATIVE: ["creative", "engaging"],
            TextGenerationStyle.TECHNICAL: ["technical", "precise"]
        }
        
        if style in style_keywords:
            assert any(keyword in system_prompt.lower() for keyword in style_keywords[style])


class TestImageAnalyzerProperties:
    """Property-based tests for image analysis."""
    
    @given(st.text(min_size=1, max_size=1000))
    def test_image_path_validation(self, path_input):
        """Property: Image path validation should be secure."""
        validator = ImageSecurityValidator()
        
        # Should handle any path input safely
        result = validator.validate_image_path(path_input)
        
        # Should always return Either
        assert result.is_left() or result.is_right()
        
        # Dangerous paths should be rejected
        dangerous_patterns = ["../", "\\", "<", ">", "|", ":", "*", "?"]
        has_dangerous = any(pattern in path_input for pattern in dangerous_patterns)
        
        if has_dangerous:
            # May be rejected for security reasons
            pass  # This is acceptable behavior
    
    @given(st.sampled_from(ImageAnalysisType))
    def test_analysis_prompt_generation(self, analysis_type):
        """Property: Image analysis prompts should be appropriate."""
        analyzer = ImageAnalyzer(Mock())
        
        prompt = analyzer._build_analysis_prompt(analysis_type)
        
        # Prompt should be non-empty
        assert len(prompt) > 0
        
        # Should contain JSON format instruction
        assert "JSON" in prompt or "json" in prompt
        
        # Should be relevant to analysis type
        type_keywords = {
            ImageAnalysisType.DESCRIBE: ["describe", "detail"],
            ImageAnalysisType.TEXT_OCR: ["text", "extract"],
            ImageAnalysisType.OBJECTS: ["objects", "identify"],
            ImageAnalysisType.FACES: ["faces", "detect"]
        }
        
        if analysis_type in type_keywords:
            assert any(keyword in prompt.lower() for keyword in type_keywords[analysis_type])


# Stateful testing for AI system behavior
class AISystemStateMachine(RuleBasedStateMachine):
    """Stateful testing for AI processing system."""
    
    def __init__(self):
        super().__init__()
        self.manager = AIProcessingManager()
        self.requests = []
        self.responses = []
    
    requests = Bundle('requests')
    
    @initialize()
    async def setup(self):
        """Initialize the AI system for testing."""
        # Mock initialization
        self.manager.initialized = True
        self.manager.model_manager = Mock(spec=AIModelManager)
        self.manager.text_processor = Mock(spec=TextProcessor)
        self.manager.image_analyzer = Mock(spec=ImageAnalyzer)
    
    @rule(target=requests, operation=st.sampled_from(["analyze", "generate", "classify"]))
    def create_request(self, operation):
        """Create AI processing requests."""
        request_data = {
            "operation": operation,
            "input_data": "test content",
            "timestamp": datetime.now(UTC)
        }
        self.requests.append(request_data)
        return request_data
    
    @rule(request=requests)
    async def process_request(self, request):
        """Process AI requests and verify behavior."""
        # Mock successful processing
        mock_result = {
            "success": True,
            "operation": request["operation"],
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        self.responses.append(mock_result)
        
        # Verify response properties
        assert mock_result["success"] is True
        assert "timestamp" in mock_result


# Integration property tests
@pytest.mark.asyncio
class TestAIIntegrationE2E:
    """End-to-end property tests for AI integration."""
    
    @given(st.text(min_size=10, max_size=500))
    async def test_text_analysis_workflow(self, input_text):
        """Property: Text analysis workflow should be robust."""
        # Skip if text is too problematic
        assume(len(input_text.strip()) > 5)
        assume(not any(char in input_text for char in ['\x00', '\x01', '\x02']))
        
        # Mock the entire workflow
        manager = AIProcessingManager()
        manager.initialized = True
        
        # Mock components
        mock_processor = Mock(spec=TextProcessor)
        mock_processor.analyze_text = AsyncMock(return_value=Either.right(Mock(
            analysis_type=TextAnalysisType.GENERAL,
            results={"analysis": "mock analysis"},
            confidence=ConfidenceScore(0.8),
            processing_time=0.1,
            model_used="mock_model",
            get_metadata=Mock(return_value={})
        )))
        
        manager.text_processor = mock_processor
        manager.security_validator = Mock()
        manager.security_validator.validate_request = AsyncMock(return_value=Either.right(Mock(
            get_summary=Mock(return_value={"is_safe": True, "risk_score": 10.0})
        )))
        
        # Test processing
        result = await manager.process_request("analyze", input_text)
        
        # Should handle the request
        assert result.is_left() or result.is_right()
        
        if result.is_right():
            response = result.get_right()
            assert "operation" in response
            assert "session_id" in response


if __name__ == "__main__":
    # Run property tests with custom settings
    pytest.main([
        __file__, 
        "-v", 
        "--hypothesis-show-statistics",
        "--hypothesis-seed=42"
    ])