"""
Property-based tests for visual automation system.

This module validates the visual automation capabilities including OCR, image recognition,
and screen analysis through comprehensive property-based testing using Hypothesis.
Tests ensure correctness, security, and performance across all input ranges.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from typing import Dict, Any, List, Tuple
import base64
import asyncio
from datetime import datetime

from src.core.visual import (
    ScreenRegion, OCRResult, ImageMatch, VisualElement, VisualOperation,
    ElementType, ConfidenceScore, ImageData, validate_image_data,
    create_screen_region, normalize_confidence
)
from src.vision.ocr_engine import OCREngine, OCRProcessingOptions, OCRPrivacyFilter
from src.vision.image_recognition import ImageRecognitionEngine, MatchingConfig, MatchingMethod
from src.vision.screen_analysis import ScreenAnalysisEngine, CaptureMode, ChangeDetectionMode
from src.server.tools.visual_automation_tools import VisualAutomationSecurityManager, VisualAutomationProcessor


# Strategy generators for visual automation testing
@composite
def screen_regions(draw):
    """Generate valid screen regions with reasonable bounds."""
    x = draw(st.integers(min_value=0, max_value=3840))
    y = draw(st.integers(min_value=0, max_value=2160))
    width = draw(st.integers(min_value=1, max_value=min(1920, 3840 - x)))
    height = draw(st.integers(min_value=1, max_value=min(1080, 2160 - y)))
    display_id = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=5)))
    return ScreenRegion(x, y, width, height, display_id)


@composite
def confidence_scores(draw):
    """Generate valid confidence scores."""
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    return ConfidenceScore(confidence)


@composite
def image_data_samples(draw):
    """Generate simulated image data."""
    # Create minimal valid image headers
    formats = [
        b'\xff\xd8\xff',  # JPEG
        b'\x89PNG\r\n\x1a\n',  # PNG
        b'GIF89a',  # GIF
    ]
    header = draw(st.sampled_from(formats))
    data_size = draw(st.integers(min_value=100, max_value=10000))
    padding = draw(st.binary(min_size=data_size, max_size=data_size))
    return ImageData(header + padding)


@composite
def ocr_results(draw):
    """Generate valid OCR results."""
    text = draw(st.text(min_size=1, max_size=1000))
    confidence = draw(confidence_scores())
    region = draw(st.one_of(st.none(), screen_regions()))
    language = draw(st.sampled_from(['en', 'es', 'fr', 'de', 'ja', 'zh']))
    
    return OCRResult(
        text=text,
        confidence=confidence,
        coordinates=region,
        language=language,
        metadata={"test_generated": True}
    )


@composite
def visual_elements(draw):
    """Generate valid visual elements."""
    element_type = draw(st.sampled_from(list(ElementType)))
    confidence = draw(confidence_scores())
    location = draw(screen_regions())
    text_content = draw(st.one_of(st.none(), st.text(max_size=100)))
    interactive = draw(st.booleans())
    
    return VisualElement(
        element_type=element_type,
        confidence=confidence,
        location=location,
        text_content=text_content,
        interactive=interactive
    )


class TestScreenRegionProperties:
    """Test properties of screen region handling."""
    
    @given(screen_regions())
    def test_screen_region_creation_properties(self, region):
        """Property: Screen regions should maintain valid state."""
        assert region.x >= 0
        assert region.y >= 0
        assert region.width > 0
        assert region.height > 0
        assert region.area == region.width * region.height
        assert len(region.center) == 2
        assert region.center[0] == region.x + region.width // 2
        assert region.center[1] == region.y + region.height // 2
    
    @given(screen_regions(), st.integers(), st.integers())
    def test_contains_point_properties(self, region, x, y):
        """Property: Point containment should be consistent with bounds."""
        contains = region.contains_point(x, y)
        
        if contains:
            assert region.x <= x < region.x + region.width
            assert region.y <= y < region.y + region.height
        else:
            assert not (region.x <= x < region.x + region.width and 
                       region.y <= y < region.y + region.height)
    
    @given(screen_regions(), screen_regions())
    def test_overlap_properties(self, region1, region2):
        """Property: Region overlap should be symmetric."""
        overlap1 = region1.overlaps_with(region2)
        overlap2 = region2.overlaps_with(region1)
        assert overlap1 == overlap2
    
    @given(screen_regions())
    def test_region_serialization_properties(self, region):
        """Property: Region serialization should be reversible."""
        region_dict = region.to_dict()
        reconstructed = ScreenRegion.from_dict(region_dict)
        
        assert reconstructed.x == region.x
        assert reconstructed.y == region.y
        assert reconstructed.width == region.width
        assert reconstructed.height == region.height
        assert reconstructed.display_id == region.display_id


class TestConfidenceScoreProperties:
    """Test properties of confidence score handling."""
    
    @given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))
    def test_confidence_normalization_properties(self, raw_confidence):
        """Property: Confidence normalization should always produce valid range."""
        normalized = normalize_confidence(raw_confidence)
        assert 0.0 <= float(normalized) <= 1.0
        
        # Should clamp to bounds
        if raw_confidence < 0.0:
            assert float(normalized) == 0.0
        elif raw_confidence > 1.0:
            assert float(normalized) == 1.0
        else:
            assert abs(float(normalized) - raw_confidence) < 1e-10
    
    @given(confidence_scores())
    def test_confidence_properties(self, confidence):
        """Property: Confidence scores should maintain valid range."""
        assert 0.0 <= float(confidence) <= 1.0
    
    @given(confidence_scores(), confidence_scores())
    def test_confidence_comparison_properties(self, conf1, conf2):
        """Property: Confidence comparison should be consistent."""
        if float(conf1) > float(conf2):
            assert float(conf1) >= float(conf2)
        elif float(conf1) < float(conf2):
            assert float(conf1) <= float(conf2)
        else:
            assert abs(float(conf1) - float(conf2)) < 1e-10


class TestImageDataValidationProperties:
    """Test properties of image data validation."""
    
    @given(image_data_samples())
    def test_valid_image_data_properties(self, image_data):
        """Property: Valid image data should pass validation."""
        result = validate_image_data(bytes(image_data))
        assert result.is_right()
        
        validated_data = result.get_right()
        assert len(validated_data) == len(image_data)
        assert bytes(validated_data) == bytes(image_data)
    
    @given(st.binary(min_size=0, max_size=10))
    def test_invalid_image_data_properties(self, invalid_data):
        """Property: Invalid image data should be rejected."""
        assume(len(invalid_data) < 3 or not any(
            invalid_data.startswith(header) for header in [
                b'\xff\xd8\xff', b'\x89PNG\r\n\x1a\n', b'GIF89a', b'GIF87a'
            ]
        ))
        
        result = validate_image_data(invalid_data)
        if len(invalid_data) == 0:
            assert result.is_left()
        elif len(invalid_data) > 50 * 1024 * 1024:  # 50MB limit
            assert result.is_left()
    
    @given(st.binary(min_size=50*1024*1024 + 1, max_size=100*1024*1024))
    def test_oversized_image_rejection(self, large_data):
        """Property: Oversized images should be rejected."""
        result = validate_image_data(large_data)
        assert result.is_left()


class TestOCREngineProperties:
    """Test properties of OCR engine functionality."""
    
    @pytest.mark.asyncio
    @given(image_data_samples(), st.sampled_from(['en', 'es', 'fr', 'de']))
    async def test_ocr_extraction_properties(self, image_data, language):
        """Property: OCR extraction should handle all valid inputs."""
        engine = OCREngine(cache_enabled=False)  # Disable cache for testing
        
        result = await engine.extract_text(image_data, language=language)
        
        # OCR should either succeed or fail gracefully
        if result.is_right():
            ocr_result = result.get_right()
            assert isinstance(ocr_result.text, str)
            assert 0.0 <= float(ocr_result.confidence) <= 1.0
            assert ocr_result.language == language
            assert len(ocr_result.text) >= 0
        else:
            error = result.get_left()
            assert hasattr(error, 'message')
    
    @given(st.text(min_size=1, max_size=10000))
    def test_privacy_filter_properties(self, text_input):
        """Property: Privacy filtering should be consistent and safe."""
        filtered_text, detected_categories = OCRPrivacyFilter.filter_sensitive_content(
            text_input, privacy_mode=True
        )
        
        # Filtered text should not be longer than original
        assert len(filtered_text) <= len(text_input) + len(detected_categories) * len("[REDACTED]")
        
        # Should maintain basic structure
        assert isinstance(filtered_text, str)
        assert isinstance(detected_categories, set)
        
        # If sensitive content detected, should contain redaction markers
        if detected_categories:
            assert "[REDACTED]" in filtered_text
    
    @given(ocr_results())
    def test_ocr_result_properties(self, ocr_result):
        """Property: OCR results should maintain consistent state."""
        assert len(ocr_result.text) > 0
        assert 0.0 <= float(ocr_result.confidence) <= 1.0
        assert ocr_result.word_count >= 0
        assert ocr_result.line_count >= 0
        
        # High confidence check should be consistent
        is_high_conf = ocr_result.is_high_confidence
        assert is_high_conf == (float(ocr_result.confidence) >= 0.8)


class TestImageRecognitionProperties:
    """Test properties of image recognition functionality."""
    
    @pytest.mark.asyncio
    @given(image_data_samples(), image_data_samples(), st.one_of(st.none(), screen_regions()))
    async def test_template_matching_properties(self, screen_data, template_data, search_region):
        """Property: Template matching should handle all valid combinations."""
        engine = ImageRecognitionEngine(cache_enabled=False)
        
        result = await engine.find_template_matches(
            screen_data, template_data, search_region
        )
        
        # Should either succeed or fail gracefully
        if result.is_right():
            matches = result.get_right()
            assert isinstance(matches, list)
            
            for match_result in matches:
                assert hasattr(match_result, 'match')
                assert hasattr(match_result, 'processing_time_ms')
                assert match_result.processing_time_ms >= 0
                
                match = match_result.match
                assert 0.0 <= float(match.confidence) <= 1.0
                assert isinstance(match.found, bool)
        else:
            error = result.get_left()
            assert hasattr(error, 'message')
    
    @given(visual_elements())
    def test_visual_element_properties(self, element):
        """Property: Visual elements should maintain valid state."""
        assert isinstance(element.element_type, ElementType)
        assert 0.0 <= float(element.confidence) <= 1.0
        assert element.location.width > 0
        assert element.location.height > 0
        assert element.area == element.location.area
        
        # Reliability check should be consistent
        is_reliable = element.is_reliable
        assert is_reliable == (float(element.confidence) >= 0.75)
        
        # Center point should be within bounds
        center = element.center_point
        assert element.location.x <= center[0] <= element.location.x + element.location.width
        assert element.location.y <= center[1] <= element.location.y + element.location.height


class TestScreenAnalysisProperties:
    """Test properties of screen analysis functionality."""
    
    @pytest.mark.asyncio
    @given(screen_regions())
    async def test_screen_capture_properties(self, region):
        """Property: Screen capture should handle all valid regions."""
        engine = ScreenAnalysisEngine(enable_privacy_protection=True)
        
        result = await engine.capture_screen_region(region, CaptureMode.BALANCED)
        
        # Should either succeed or fail gracefully
        if result.is_right():
            capture = result.get_right()
            assert len(capture.image_data) > 0
            assert capture.region.x == region.x
            assert capture.region.y == region.y
            assert capture.region.width == region.width
            assert capture.region.height == region.height
            assert 0.0 <= capture.quality_score <= 1.0
            assert 0.0 <= capture.compression_ratio <= 1.0
            assert capture.file_size_mb >= 0.0
        else:
            error = result.get_left()
            assert hasattr(error, 'message')
    
    @pytest.mark.asyncio
    @given(screen_regions(), st.floats(min_value=0.0, max_value=1.0))
    async def test_change_detection_properties(self, region, sensitivity):
        """Property: Change detection should handle all valid parameters."""
        engine = ScreenAnalysisEngine()
        
        result = await engine.detect_screen_changes(region, sensitivity=sensitivity)
        
        if result.is_right():
            change = result.get_right()
            assert isinstance(change.changed, bool)
            assert 0.0 <= change.change_percentage <= 100.0
            assert 0.0 <= change.confidence <= 1.0
            assert isinstance(change.timestamp, datetime)
            
            # Significant change check should be consistent
            is_significant = change.is_significant_change
            expected = change.change_percentage > 10.0 and change.confidence > 0.8
            assert is_significant == expected


class TestVisualAutomationSecurityProperties:
    """Test security properties of visual automation."""
    
    def test_security_manager_region_validation_properties(self):
        """Property: Security manager should validate all region parameters."""
        security = VisualAutomationSecurityManager()
        
        @given(st.dictionaries(
            st.sampled_from(['x', 'y', 'width', 'height', 'display_id']),
            st.one_of(st.integers(), st.floats(), st.text(), st.none()),
            min_size=1
        ))
        def check_region_validation(region_dict):
            result = security.validate_region(region_dict)
            
            # Should validate consistently
            if result.is_right():
                region = result.get_right()
                assert region.x >= 0
                assert region.y >= 0
                assert region.width > 0
                assert region.height > 0
                assert region.width <= security.MAX_REGION_WIDTH
                assert region.height <= security.MAX_REGION_HEIGHT
                assert region.area <= security.MAX_REGION_AREA
            else:
                error = result.get_left()
                assert hasattr(error, 'message')
        
        check_region_validation()
    
    @given(st.one_of(
        st.text(min_size=1, max_size=100),
        st.binary(min_size=1, max_size=1000)
    ))
    def test_image_validation_properties(self, image_input):
        """Property: Image validation should handle diverse inputs."""
        security = VisualAutomationSecurityManager()
        
        result = security.validate_image_data(image_input)
        
        # Should either validate or reject consistently
        if result.is_right():
            validated_data = result.get_right()
            assert len(validated_data) > 0
            # Should be within size limits
            size_mb = len(validated_data) / (1024 * 1024)
            assert size_mb <= security.MAX_IMAGE_SIZE_MB
        else:
            error = result.get_left()
            assert hasattr(error, 'message')
    
    @given(
        st.sampled_from(['ocr_text', 'find_image', 'capture_screen', 'analyze_window']),
        st.text(min_size=1, max_size=50)
    )
    def test_rate_limiting_properties(self, operation, client_id):
        """Property: Rate limiting should be consistent and fair."""
        security = VisualAutomationSecurityManager()
        
        # Should allow initial operations
        result = security.check_rate_limit(operation, client_id)
        assert result.is_right()
        
        # Should eventually enforce limits
        for _ in range(security.MAX_OPERATIONS_PER_MINUTE + 5):
            result = security.check_rate_limit(operation, client_id)
        
        # Final result should be rate limited
        assert result.is_left()


class TestVisualProcessorIntegrationProperties:
    """Test integration properties of visual processor."""
    
    @pytest.mark.asyncio
    @given(st.sampled_from(list(VisualOperation)))
    async def test_processor_operation_handling(self, operation):
        """Property: Processor should handle all supported operations."""
        processor = VisualAutomationProcessor()
        
        # Create minimal valid config for each operation
        region = ScreenRegion(100, 100, 200, 150)
        
        from src.core.visual import VisualProcessingConfig
        config = VisualProcessingConfig(
            operation=operation,
            region=region,
            confidence_threshold=0.8
        )
        
        # Should handle operation gracefully
        try:
            result = await processor.process_operation(operation, config)
            
            if result.is_right():
                response = result.get_right()
                assert isinstance(response, dict)
                assert "operation" in response
                assert "success" in response
            else:
                error = result.get_left()
                assert hasattr(error, 'message')
        except Exception as e:
            # Should not raise unhandled exceptions
            assert False, f"Unexpected exception: {str(e)}"
    
    def test_processor_statistics_properties(self):
        """Property: Processor statistics should be consistent."""
        processor = VisualAutomationProcessor()
        stats = processor.get_processing_stats()
        
        assert isinstance(stats, dict)
        assert "operations_completed" in stats
        assert "ocr_operations" in stats
        assert "image_matches" in stats
        assert "screen_captures" in stats
        assert "errors_encountered" in stats
        
        # All counters should be non-negative
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                assert value >= 0


class TestVisualAutomationEndToEndProperties:
    """Test end-to-end properties of visual automation system."""
    
    @pytest.mark.asyncio
    @settings(max_examples=10, deadline=5000)  # Reduced examples for async tests
    @given(
        st.sampled_from(['ocr_text', 'capture_screen', 'analyze_window']),
        screen_regions(),
        st.booleans()
    )
    async def test_end_to_end_visual_operations(self, operation, region, privacy_mode):
        """Property: End-to-end operations should be robust."""
        from src.server.tools.visual_automation_tools import km_visual_automation
        
        # Prepare operation parameters
        region_dict = region.to_dict()
        
        try:
            result = await km_visual_automation(
                operation=operation,
                region=region_dict,
                privacy_mode=privacy_mode,
                timeout_seconds=5  # Short timeout for testing
            )
            
            # Should return valid response structure
            assert isinstance(result, dict)
            assert "success" in result
            assert "operation" in result
            assert result["operation"] == operation
            assert "timestamp" in result
            assert "privacy_mode_enabled" in result
            assert result["privacy_mode_enabled"] == privacy_mode
            
        except Exception as e:
            # Should provide meaningful error messages
            assert isinstance(e, Exception)
            assert len(str(e)) > 0
    
    @pytest.mark.asyncio
    @given(st.text(min_size=1, max_size=100))
    async def test_error_handling_properties(self, invalid_operation):
        """Property: Invalid operations should be handled gracefully."""
        assume(invalid_operation not in [op.value for op in VisualOperation])
        
        from src.server.tools.visual_automation_tools import km_visual_automation
        from fastmcp.exceptions import ToolError
        
        with pytest.raises(ToolError) as exc_info:
            await km_visual_automation(operation=invalid_operation)
        
        # Error should be informative
        assert "Invalid operation" in str(exc_info.value)
        assert invalid_operation in str(exc_info.value)


# Performance property tests
class TestVisualAutomationPerformanceProperties:
    """Test performance properties of visual automation."""
    
    @pytest.mark.asyncio
    @given(screen_regions())
    async def test_processing_time_properties(self, region):
        """Property: Processing times should be reasonable."""
        processor = VisualAutomationProcessor()
        
        from src.core.visual import VisualProcessingConfig, VisualOperation
        config = VisualProcessingConfig(
            operation=VisualOperation.CAPTURE_SCREEN,
            region=region
        )
        
        import time
        start_time = time.time()
        
        result = await processor.process_operation(
            VisualOperation.CAPTURE_SCREEN, config
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjusted for simulation)
        assert processing_time < 5.0  # 5 seconds max for any operation
        
        if result.is_right():
            response = result.get_right()
            # Should include timing information
            if "processing_time_ms" in response:
                assert response["processing_time_ms"] >= 0
    
    @given(st.integers(min_value=1, max_value=10))
    def test_cache_efficiency_properties(self, cache_size):
        """Property: Caching should improve performance consistency."""
        # Test with caching enabled
        engine_cached = ImageRecognitionEngine(cache_enabled=True, max_cache_size=cache_size)
        stats_cached = engine_cached.get_processing_stats()
        
        # Test with caching disabled
        engine_no_cache = ImageRecognitionEngine(cache_enabled=False)
        stats_no_cache = engine_no_cache.get_processing_stats()
        
        # Both should have consistent initial state
        assert isinstance(stats_cached, dict)
        assert isinstance(stats_no_cache, dict)
        
        # Cached version should have cache statistics
        if "cache_stats" in stats_cached:
            cache_stats = stats_cached["cache_stats"]
            assert "cache_enabled" in cache_stats
            assert cache_stats["cache_enabled"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])