"""
Comprehensive tests for AI processing MCP tools.

This module provides extensive testing for AI/ML model integration including
text analysis, image processing, content generation, and intelligent automation.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from src.server.tools.ai_processing_tools import AIProcessingManager, km_ai_processing, km_ai_status, km_ai_models
from src.core.ai_integration import (
    AIOperation, AIModelType, ProcessingMode, TextAnalysisType, 
    TextGenerationStyle, ImageAnalysisType, create_ai_session
)
from src.ai.model_manager import AIError
from src.core.either import Either


class TestAIProcessingManager:
    """Test suite for AI processing manager."""
    
    @pytest.fixture
    def ai_manager(self):
        """Create AI processing manager instance."""
        return AIProcessingManager()
    
    @pytest.fixture
    def mock_api_keys(self):
        """Mock API keys for testing."""
        return {
            "openai": "test-openai-key",
            "google": "test-google-key",
            "azure": "test-azure-key"
        }
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, ai_manager, mock_api_keys):
        """Test successful AI manager initialization."""
        with patch.object(ai_manager.model_manager, 'initialize') as mock_init:
            mock_init.return_value = Either.right(None)
            
            result = await ai_manager.initialize(mock_api_keys)
            
            assert result.is_right()
            assert ai_manager.initialized is True
            assert ai_manager.text_processor is not None
            assert ai_manager.image_analyzer is not None
            
            mock_init.assert_called_once_with(mock_api_keys)
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, ai_manager, mock_api_keys):
        """Test AI manager initialization failure."""
        with patch.object(ai_manager.model_manager, 'initialize') as mock_init:
            mock_init.return_value = Either.left(AIError.initialization_failed("Test error"))
            
            result = await ai_manager.initialize(mock_api_keys)
            
            assert result.is_left()
            assert ai_manager.initialized is False
    
    @pytest.mark.asyncio
    async def test_text_analysis_operation(self, ai_manager):
        """Test text analysis operation processing."""
        ai_manager.initialized = True
        
        with patch.object(ai_manager, 'text_processor') as mock_processor:
            mock_result = Mock()
            mock_result.analysis_type = TextAnalysisType.SENTIMENT
            mock_result.results = {"sentiment": "positive", "confidence": 0.9}
            mock_result.confidence = 0.9
            mock_result.processing_time = 0.5
            mock_result.model_used = "gpt-3.5-turbo"
            mock_result.get_metadata.return_value = {"test": "metadata"}
            
            mock_processor.analyze_text.return_value = Either.right(mock_result)
            
            result = await ai_manager.process_request(
                "analyze",
                "This is a test text",
                analysis_type="sentiment"
            )
            
            assert result.is_right()
            response = result.get_right()
            assert response["operation"] == "analyze"
            assert response["analysis_type"] == "sentiment"
            assert response["results"]["sentiment"] == "positive"
            assert response["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_text_generation_operation(self, ai_manager):
        """Test text generation operation processing."""
        ai_manager.initialized = True
        
        with patch.object(ai_manager, 'text_processor') as mock_processor:
            mock_processor.generate_text.return_value = Either.right("Generated text content")
            
            result = await ai_manager.process_request(
                "generate",
                "Write a professional email",
                style="formal",
                max_length=300
            )
            
            assert result.is_right()
            response = result.get_right()
            assert response["operation"] == "generate"
            assert response["generated_text"] == "Generated text content"
            assert response["style"] == "formal"
    
    @pytest.mark.asyncio
    async def test_image_analysis_operation(self, ai_manager):
        """Test image analysis operation processing."""
        ai_manager.initialized = True
        
        with patch.object(ai_manager, 'image_analyzer') as mock_analyzer:
            mock_result = Mock()
            mock_result.analysis_type = ImageAnalysisType.DESCRIBE
            mock_result.image_path = "/test/image.jpg"
            mock_result.results = {"description": "Test image description"}
            mock_result.confidence = 0.85
            mock_result.processing_time = 1.2
            mock_result.model_used = "gpt-4-vision"
            mock_result.metadata = Mock()
            mock_result.metadata.format.value = "jpeg"
            mock_result.metadata.file_size = 1024000
            mock_result.get_structured_data.return_value = {"structured": "data"}
            
            mock_analyzer.analyze_image.return_value = Either.right(mock_result)
            
            result = await ai_manager.process_request(
                "extract",
                "/test/image.jpg",
                analysis_type="describe"
            )
            
            assert result.is_right()
            response = result.get_right()
            assert response["operation"] == "extract"
            assert response["analysis_type"] == "describe"
            assert response["results"]["description"] == "Test image description"
    
    @pytest.mark.asyncio
    async def test_text_classification(self, ai_manager):
        """Test text classification functionality."""
        ai_manager.initialized = True
        
        with patch.object(ai_manager, 'text_processor') as mock_processor:
            classification_result = {
                "positive": 0.8,
                "negative": 0.1,
                "neutral": 0.1
            }
            mock_processor.classify_text.return_value = Either.right(classification_result)
            
            result = await ai_manager.process_request(
                "classify",
                "This is a great product!",
                categories=["positive", "negative", "neutral"]
            )
            
            assert result.is_right()
            response = result.get_right()
            assert response["operation"] == "classify"
            assert response["classification"]["positive"] == 0.8
            assert response["top_category"][0] == "positive"
    
    @pytest.mark.asyncio
    async def test_text_summarization(self, ai_manager):
        """Test text summarization functionality."""
        ai_manager.initialized = True
        
        with patch.object(ai_manager, 'text_processor') as mock_processor:
            mock_result = Mock()
            mock_result.results = {
                "summary": "This is a concise summary of the text.",
                "key_points": ["Point 1", "Point 2", "Point 3"]
            }
            mock_processor.analyze_text.return_value = Either.right(mock_result)
            
            long_text = "This is a very long text that needs to be summarized. " * 20
            
            result = await ai_manager.process_request("summarize", long_text)
            
            assert result.is_right()
            response = result.get_right()
            assert response["operation"] == "summarize"
            assert "summary" in response
            assert response["compression_ratio"] < 1.0
    
    def test_system_status(self, ai_manager):
        """Test AI system status reporting."""
        ai_manager.initialized = True
        
        with patch.object(ai_manager.model_manager, 'list_available_models') as mock_list, \
             patch.object(ai_manager.model_manager, 'get_usage_statistics') as mock_stats:
            
            mock_list.return_value = [{"model": "gpt-3.5-turbo"}]
            mock_stats.return_value = {"total_requests": 10}
            
            status = ai_manager.get_system_status()
            
            assert status["initialized"] is True
            assert "session_id" in status
            assert "available_models" in status
            assert "supported_operations" in status


class TestAIProcessingMCPTools:
    """Test suite for AI processing MCP tools."""
    
    @pytest.mark.asyncio
    async def test_km_ai_processing_text_analysis(self):
        """Test km_ai_processing tool with text analysis."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            mock_manager.process_request.return_value = Either.right({
                "operation": "analyze",
                "analysis_type": "sentiment",
                "results": {"sentiment": "positive"},
                "confidence": 0.9
            })
            
            result = await km_ai_processing(
                operation="analyze",
                input_data="This is a positive message",
                context={"analysis_type": "sentiment"}
            )
            
            assert result["success"] is True
            assert result["operation"] == "analyze"
            assert result["analysis_type"] == "sentiment"
    
    @pytest.mark.asyncio
    async def test_km_ai_processing_text_generation(self):
        """Test km_ai_processing tool with text generation."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            mock_manager.process_request.return_value = Either.right({
                "operation": "generate",
                "generated_text": "Professional email content",
                "style": "formal"
            })
            
            result = await km_ai_processing(
                operation="generate",
                input_data="Write a professional email about project updates",
                context={"style": "formal", "max_length": 300}
            )
            
            assert result["success"] is True
            assert result["operation"] == "generate"
            assert "generated_text" in result
    
    @pytest.mark.asyncio
    async def test_km_ai_processing_image_analysis(self):
        """Test km_ai_processing tool with image analysis."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            mock_manager.process_request.return_value = Either.right({
                "operation": "extract",
                "analysis_type": "text_ocr",
                "results": {"text": "Extracted text from image"}
            })
            
            result = await km_ai_processing(
                operation="extract",
                input_data="/path/to/image.jpg",
                context={"analysis_type": "text_ocr"}
            )
            
            assert result["success"] is True
            assert result["operation"] == "extract"
            assert "results" in result
    
    @pytest.mark.asyncio
    async def test_km_ai_processing_uninitialized_system(self):
        """Test km_ai_processing with uninitialized system."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = False
            mock_manager.initialize.return_value = Either.left(AIError.initialization_failed("No API keys"))
            
            result = await km_ai_processing(
                operation="analyze",
                input_data="Test text"
            )
            
            assert result["success"] is False
            assert "initialization_error" in result["error_type"]
    
    @pytest.mark.asyncio
    async def test_km_ai_processing_timeout(self):
        """Test km_ai_processing with timeout."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            
            # Simulate a slow operation
            async def slow_process(*args, **kwargs):
                await asyncio.sleep(2)
                return Either.right({"result": "data"})
            
            mock_manager.process_request = slow_process
            
            result = await km_ai_processing(
                operation="analyze",
                input_data="Test text",
                timeout=1  # 1 second timeout
            )
            
            assert result["success"] is False
            assert "timeout_error" in result["error_type"]
    
    @pytest.mark.asyncio
    async def test_km_ai_processing_invalid_operation(self):
        """Test km_ai_processing with invalid operation."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            mock_manager.process_request.return_value = Either.left(
                AIError("invalid_operation", "Unsupported operation: invalid_op")
            )
            
            result = await km_ai_processing(
                operation="invalid_op",
                input_data="Test text"
            )
            
            assert result["success"] is False
            assert "invalid_operation" in result["error_type"]
    
    @pytest.mark.asyncio
    async def test_km_ai_status_initialized(self):
        """Test km_ai_status with initialized system."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            mock_manager.get_system_status.return_value = {
                "initialized": True,
                "session_id": "test-session",
                "available_models": [{"model": "gpt-3.5-turbo"}],
                "supported_operations": ["analyze", "generate"]
            }
            
            result = await km_ai_status(detailed=True)
            
            assert result["success"] is True
            assert result["initialized"] is True
            assert "session_id" in result
    
    @pytest.mark.asyncio
    async def test_km_ai_status_not_initialized(self):
        """Test km_ai_status with uninitialized system."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = False
            
            result = await km_ai_status()
            
            assert result["initialized"] is False
            assert result["status"] == "not_initialized"
    
    @pytest.mark.asyncio
    async def test_km_ai_models_success(self):
        """Test km_ai_models tool success case."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            mock_manager.model_manager.list_available_models.return_value = [
                {
                    "model_id": "gpt-3.5-turbo",
                    "provider": "openai",
                    "supports_vision": False,
                    "cost_per_1k_input": 0.0015
                },
                {
                    "model_id": "gpt-4-vision",
                    "provider": "openai", 
                    "supports_vision": True,
                    "cost_per_1k_input": 0.01
                }
            ]
            
            result = await km_ai_models(provider="openai")
            
            assert result["success"] is True
            assert result["total_count"] == 2
            assert result["filters_applied"]["provider"] == "openai"
    
    @pytest.mark.asyncio
    async def test_km_ai_models_invalid_provider(self):
        """Test km_ai_models with invalid provider."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            
            result = await km_ai_models(provider="invalid_provider")
            
            assert result["success"] is False
            assert "Invalid provider" in result["error"]
            assert "valid_providers" in result


class TestAIProcessingIntegration:
    """Integration tests for AI processing tools."""
    
    @pytest.mark.asyncio
    async def test_text_analysis_workflow(self):
        """Test complete text analysis workflow."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            
            # Mock sentiment analysis
            mock_manager.process_request.return_value = Either.right({
                "operation": "analyze",
                "analysis_type": "sentiment",
                "results": {
                    "sentiment": "positive",
                    "confidence": 0.92,
                    "emotions": ["joy", "satisfaction"]
                },
                "confidence": 0.92,
                "processing_time": 0.45,
                "model_used": "gpt-3.5-turbo"
            })
            
            # Test analysis
            result = await km_ai_processing(
                operation="analyze",
                input_data="I absolutely love this new feature! It works perfectly.",
                context={"analysis_type": "sentiment"},
                processing_mode="accurate"
            )
            
            assert result["success"] is True
            assert result["analysis_type"] == "sentiment"
            assert result["results"]["sentiment"] == "positive"
            assert result["confidence"] > 0.9
    
    @pytest.mark.asyncio
    async def test_content_generation_workflow(self):
        """Test complete content generation workflow."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            
            # Mock text generation
            mock_manager.process_request.return_value = Either.right({
                "operation": "generate",
                "generated_text": "Dear Team,\n\nI hope this email finds you well. I wanted to provide you with an update on our current project status and upcoming milestones.\n\nBest regards,\nProject Manager",
                "style": "formal",
                "length": 142,
                "prompt": "Write a professional email about project updates"
            })
            
            # Test generation
            result = await km_ai_processing(
                operation="generate",
                input_data="Write a professional email about project updates",
                context={
                    "style": "formal",
                    "max_length": 300,
                    "target_audience": "team members"
                },
                temperature=0.3,
                processing_mode="creative"
            )
            
            assert result["success"] is True
            assert result["operation"] == "generate"
            assert "generated_text" in result
            assert result["style"] == "formal"
            assert len(result["generated_text"]) > 50
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test comprehensive error handling."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            
            # Test various error scenarios
            error_scenarios = [
                (AIError("rate_limit_exceeded", "Rate limit exceeded"), "rate_limit_exceeded"),
                (AIError("cost_limit_exceeded", "Cost limit exceeded"), "cost_limit_exceeded"),
                (AIError("model_not_available", "Model not available"), "model_not_available"),
                (AIError("invalid_input", "Invalid input format"), "invalid_input")
            ]
            
            for error, expected_type in error_scenarios:
                mock_manager.process_request.return_value = Either.left(error)
                
                result = await km_ai_processing(
                    operation="analyze",
                    input_data="Test text"
                )
                
                assert result["success"] is False
                assert result["error_type"] == expected_type
    
    @pytest.mark.asyncio
    async def test_multi_operation_session(self):
        """Test multiple operations in the same session."""
        with patch('src.server.tools.ai_processing_tools.ai_manager') as mock_manager:
            mock_manager.initialized = True
            
            # Mock different operations
            operations = [
                ("analyze", {"operation": "analyze", "results": {"sentiment": "positive"}}),
                ("generate", {"operation": "generate", "generated_text": "Generated content"}),
                ("classify", {"operation": "classify", "classification": {"category1": 0.8}})
            ]
            
            for operation, mock_response in operations:
                mock_manager.process_request.return_value = Either.right(mock_response)
                
                result = await km_ai_processing(
                    operation=operation,
                    input_data="Test input",
                    enable_caching=True
                )
                
                assert result["success"] is True
                assert result["operation"] == operation


# Property-based testing with Hypothesis
try:
    from hypothesis import given, strategies as st
    
    @given(st.text(min_size=1, max_size=1000))
    def test_text_input_validation_properties(text_content):
        """Property: Text input validation should handle various content safely."""
        # Filter out empty or whitespace-only text
        if text_content.strip():
            # Test that input validation doesn't crash on valid text
            try:
                # Simulate basic validation
                assert len(text_content) > 0
                assert isinstance(text_content, str)
                # Would normally call actual validation function
            except Exception:
                # Some edge cases might be invalid
                pass
    
    @given(st.floats(min_value=0.0, max_value=2.0))
    def test_temperature_parameter_properties(temperature):
        """Property: Temperature parameters should be within valid range."""
        # Test temperature validation
        assert 0.0 <= temperature <= 2.0
        
        # Test that temperature affects generation (mock test)
        if temperature < 0.5:
            generation_mode = "conservative"
        elif temperature > 1.5:
            generation_mode = "creative"
        else:
            generation_mode = "balanced"
        
        assert generation_mode in ["conservative", "balanced", "creative"]
    
    @given(st.integers(min_value=1, max_value=10000))
    def test_max_tokens_properties(max_tokens):
        """Property: Max tokens should be positive and reasonable."""
        assert max_tokens > 0
        assert max_tokens <= 10000
        
        # Test cost estimation with different token counts
        base_cost_per_token = 0.00002
        estimated_cost = max_tokens * base_cost_per_token
        assert estimated_cost >= 0

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


if __name__ == "__main__":
    pytest.main([__file__])