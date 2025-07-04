"""
Advanced visual automation MCP tools for OCR, image recognition, and screen analysis.

This module implements the km_visual_automation MCP tool, providing AI agents with
sophisticated visual automation capabilities including OCR text extraction, image
template matching, screen capture, and UI element detection.

Security: Comprehensive input validation, privacy protection, and permission management.
Performance: Optimized processing with intelligent caching and batch operations.
Integration: Seamless integration with Keyboard Maestro and existing tool ecosystem.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import logging
import asyncio
import base64
import json
from datetime import datetime

from fastmcp import Context
from fastmcp.exceptions import ToolError

from src.core.visual import (
    VisualOperation, ScreenRegion, VisualProcessingConfig, VisualError,
    ProcessingError, PermissionError, PrivacyError, ImageData, validate_image_data
)
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger
from src.vision.ocr_engine import OCREngine, OCRProcessingOptions
from src.vision.image_recognition import ImageRecognitionEngine, MatchingConfig, MatchingMethod
from src.vision.screen_analysis import ScreenAnalysisEngine, CaptureMode, ChangeDetectionMode
from src.security.input_sanitizer import InputSanitizer

logger = get_logger(__name__)


class VisualAutomationSecurityManager:
    """Security manager for visual automation operations."""
    
    # Maximum allowed region sizes (to prevent abuse)
    MAX_REGION_WIDTH = 3840   # 4K width
    MAX_REGION_HEIGHT = 2160  # 4K height
    MAX_REGION_AREA = 2073600  # 1920x1080
    
    # Rate limiting settings
    MAX_OPERATIONS_PER_MINUTE = 30
    MAX_BATCH_SIZE = 10
    
    # File size limits
    MAX_IMAGE_SIZE_MB = 50
    MAX_TEMPLATE_SIZE_MB = 10
    
    def __init__(self):
        self.operation_history: Dict[str, List[datetime]] = {}
        self.sanitizer = InputSanitizer()
    
    def validate_region(self, region_data: Dict[str, Any]) -> Either[VisualError, ScreenRegion]:
        """Validate and sanitize screen region parameters."""
        try:
            # Extract coordinates with validation
            x = int(region_data.get('x', 0))
            y = int(region_data.get('y', 0))
            width = int(region_data.get('width', 0))
            height = int(region_data.get('height', 0))
            
            # Validate coordinate bounds
            if x < 0 or y < 0:
                return Either.left(ProcessingError("Region coordinates must be non-negative"))
            
            if width <= 0 or height <= 0:
                return Either.left(ProcessingError("Region dimensions must be positive"))
            
            if width > self.MAX_REGION_WIDTH or height > self.MAX_REGION_HEIGHT:
                return Either.left(ProcessingError(
                    f"Region too large. Max dimensions: {self.MAX_REGION_WIDTH}x{self.MAX_REGION_HEIGHT}"
                ))
            
            if width * height > self.MAX_REGION_AREA:
                return Either.left(ProcessingError(
                    f"Region area too large. Max area: {self.MAX_REGION_AREA} pixels"
                ))
            
            display_id = region_data.get('display_id')
            if display_id is not None:
                display_id = int(display_id)
                if display_id < 1 or display_id > 10:  # Reasonable display limit
                    return Either.left(ProcessingError("Invalid display ID"))
            
            return Either.right(ScreenRegion(x, y, width, height, display_id))
            
        except (ValueError, TypeError) as e:
            return Either.left(ProcessingError(f"Invalid region parameters: {str(e)}"))
    
    def validate_image_data(self, image_input: Union[str, bytes]) -> Either[VisualError, ImageData]:
        """Validate and process image data input."""
        try:
            if isinstance(image_input, str):
                # Assume base64 encoded image
                if image_input.startswith('data:'):
                    # Remove data URL prefix
                    _, encoded_data = image_input.split(',', 1)
                    image_bytes = base64.b64decode(encoded_data)
                else:
                    # Direct base64
                    image_bytes = base64.b64decode(image_input)
            else:
                image_bytes = bytes(image_input)
            
            # Validate size
            size_mb = len(image_bytes) / (1024 * 1024)
            if size_mb > self.MAX_IMAGE_SIZE_MB:
                return Either.left(ProcessingError(
                    f"Image too large: {size_mb:.1f}MB. Max allowed: {self.MAX_IMAGE_SIZE_MB}MB"
                ))
            
            # Validate image format
            return validate_image_data(image_bytes)
            
        except Exception as e:
            return Either.left(ProcessingError(f"Invalid image data: {str(e)}"))
    
    def check_rate_limit(self, operation: str, client_id: str = "default") -> Either[VisualError, None]:
        """Check if operation is within rate limits."""
        now = datetime.now()
        key = f"{client_id}_{operation}"
        
        # Initialize or clean old entries
        if key not in self.operation_history:
            self.operation_history[key] = []
        
        # Remove operations older than 1 minute
        cutoff = datetime.now().timestamp() - 60
        self.operation_history[key] = [
            op_time for op_time in self.operation_history[key] 
            if op_time.timestamp() > cutoff
        ]
        
        # Check current rate
        if len(self.operation_history[key]) >= self.MAX_OPERATIONS_PER_MINUTE:
            return Either.left(ProcessingError(
                f"Rate limit exceeded. Max {self.MAX_OPERATIONS_PER_MINUTE} operations per minute"
            ))
        
        # Record this operation
        self.operation_history[key].append(now)
        return Either.right(None)
    
    def validate_operation_config(self, operation: str, config: Dict[str, Any]) -> Either[VisualError, None]:
        """Validate operation-specific configuration parameters."""
        try:
            if operation == "ocr_text":
                language = config.get('ocr_language', 'en')
                if not isinstance(language, str) or len(language) < 2:
                    return Either.left(ProcessingError("Invalid language code"))
                
                confidence = float(config.get('confidence_threshold', 0.8))
                if not (0.0 <= confidence <= 1.0):
                    return Either.left(ProcessingError("Confidence threshold must be between 0.0 and 1.0"))
            
            elif operation == "find_image":
                confidence = float(config.get('confidence_threshold', 0.8))
                if not (0.0 <= confidence <= 1.0):
                    return Either.left(ProcessingError("Confidence threshold must be between 0.0 and 1.0"))
                
                max_matches = int(config.get('max_matches', 10))
                if not (1 <= max_matches <= 100):
                    return Either.left(ProcessingError("Max matches must be between 1 and 100"))
            
            elif operation == "capture_screen":
                mode = config.get('capture_mode', 'balanced')
                valid_modes = ['full_quality', 'balanced', 'performance', 'privacy_safe', 'thumbnail']
                if mode not in valid_modes:
                    return Either.left(ProcessingError(f"Invalid capture mode. Valid modes: {valid_modes}"))
            
            return Either.right(None)
            
        except (ValueError, TypeError) as e:
            return Either.left(ProcessingError(f"Invalid configuration: {str(e)}"))


class VisualAutomationProcessor:
    """Main processor for visual automation operations."""
    
    def __init__(self):
        self.security_manager = VisualAutomationSecurityManager()
        self.ocr_engine = OCREngine(cache_enabled=True)
        self.image_recognition = ImageRecognitionEngine(cache_enabled=True)
        self.screen_analysis = ScreenAnalysisEngine(enable_privacy_protection=True)
        self.processing_stats = {
            "operations_completed": 0,
            "ocr_operations": 0,
            "image_matches": 0,
            "screen_captures": 0,
            "errors_encountered": 0
        }
        logger.info("Visual Automation Processor initialized with full engine support")
    
    async def process_operation(
        self,
        operation: VisualOperation,
        config: VisualProcessingConfig,
        image_data: Optional[ImageData] = None,
        template_data: Optional[ImageData] = None
    ) -> Either[VisualError, Dict[str, Any]]:
        """Process visual automation operation based on type."""
        try:
            logger.info(f"Processing visual operation: {operation.value}")
            
            # Dispatch to appropriate handler
            if operation == VisualOperation.OCR_TEXT:
                return await self._handle_ocr_operation(config, image_data)
            elif operation == VisualOperation.FIND_IMAGE:
                return await self._handle_image_matching(config, image_data, template_data)
            elif operation == VisualOperation.CAPTURE_SCREEN:
                return await self._handle_screen_capture(config)
            elif operation == VisualOperation.ANALYZE_WINDOW:
                return await self._handle_window_analysis(config)
            elif operation == VisualOperation.UI_ELEMENT_DETECTION:
                return await self._handle_ui_detection(config, image_data)
            elif operation == VisualOperation.COLOR_ANALYSIS:
                return await self._handle_color_analysis(config)
            elif operation == VisualOperation.MOTION_DETECTION:
                return await self._handle_motion_detection(config)
            else:
                return Either.left(ProcessingError(f"Unsupported operation: {operation.value}"))
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"Visual operation processing failed: {str(e)}")
            return Either.left(ProcessingError(f"Operation processing failed: {str(e)}"))
    
    async def _handle_ocr_operation(
        self,
        config: VisualProcessingConfig,
        image_data: Optional[ImageData]
    ) -> Either[VisualError, Dict[str, Any]]:
        """Handle OCR text extraction operation."""
        try:
            if not image_data:
                # Capture screen region for OCR
                if not config.region:
                    return Either.left(ProcessingError("Region required for screen OCR"))
                
                capture_result = await self.screen_analysis.capture_screen_region(
                    config.region, CaptureMode.BALANCED, config.privacy_mode
                )
                if capture_result.is_left():
                    return Either.left(capture_result.get_left())
                
                image_data = capture_result.get_right().image_data
            
            # Configure OCR options
            ocr_options = OCRProcessingOptions(
                dpi=300,
                contrast_enhancement=True,
                noise_reduction=True,
                extract_word_boxes=config.include_coordinates,
                extract_line_boxes=config.include_coordinates,
                confidence_threshold=float(config.confidence_threshold)
            )
            
            # Perform OCR
            ocr_result = await self.ocr_engine.extract_text(
                image_data, config.region, config.language, ocr_options, config.privacy_mode
            )
            
            if ocr_result.is_left():
                return Either.left(ocr_result.get_left())
            
            result = ocr_result.get_right()
            self.processing_stats["ocr_operations"] += 1
            
            # Format response
            response = {
                "operation": "ocr_text",
                "success": True,
                "text": str(result.text),
                "confidence": float(result.confidence),
                "language": result.language,
                "character_count": len(str(result.text)),
                "word_count": result.word_count,
                "line_count": result.line_count,
                "privacy_filtered": result.metadata.get("privacy_filtered", False),
                "processing_time_ms": result.metadata.get("processing_time_ms", 0)
            }
            
            if config.include_coordinates and result.coordinates:
                response["text_region"] = result.coordinates.to_dict()
                response["word_boxes"] = [
                    {"word": word, "region": region.to_dict()}
                    for word, region in result.word_boxes
                ]
                response["line_boxes"] = [
                    region.to_dict() for region in result.line_boxes
                ]
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(ProcessingError(f"OCR operation failed: {str(e)}"))
    
    async def _handle_image_matching(
        self,
        config: VisualProcessingConfig,
        screen_data: Optional[ImageData],
        template_data: Optional[ImageData]
    ) -> Either[VisualError, Dict[str, Any]]:
        """Handle image template matching operation."""
        try:
            if not template_data:
                return Either.left(ProcessingError("Template image required for matching"))
            
            if not screen_data:
                # Capture screen for matching
                search_region = config.region or ScreenRegion(0, 0, 1920, 1080)
                capture_result = await self.screen_analysis.capture_screen_region(
                    search_region, CaptureMode.BALANCED, config.privacy_mode
                )
                if capture_result.is_left():
                    return Either.left(capture_result.get_left())
                
                screen_data = capture_result.get_right().image_data
            
            # Configure matching
            matching_config = MatchingConfig(
                method=MatchingMethod.HYBRID_MATCHING,
                confidence_threshold=float(config.confidence_threshold),
                max_matches=config.max_results,
                enable_multi_scale=True,
                scale_tolerance=0.2,
                blur_tolerance=True,
                partial_matching=True
            )
            
            # Perform template matching
            matches_result = await self.image_recognition.find_template_matches(
                screen_data, template_data, config.region, matching_config
            )
            
            if matches_result.is_left():
                return Either.left(matches_result.get_left())
            
            match_results = matches_result.get_right()
            self.processing_stats["image_matches"] += len(match_results)
            
            # Format response
            matches = []
            for match_result in match_results:
                match = match_result.match
                match_data = {
                    "found": match.found,
                    "confidence": float(match.confidence),
                    "method": match_result.method_used.value,
                    "processing_time_ms": match_result.processing_time_ms,
                    "quality_score": match_result.quality_metrics.get("edge_alignment", 0.0)
                }
                
                if match.location and config.include_coordinates:
                    match_data["location"] = match.location.to_dict()
                    match_data["center_point"] = match.center_point
                
                if match_result.feature_points:
                    match_data["feature_points_count"] = len(match_result.feature_points)
                
                matches.append(match_data)
            
            response = {
                "operation": "find_image",
                "success": True,
                "matches_found": len(matches),
                "matches": matches,
                "total_processing_time_ms": sum(m.processing_time_ms for m in match_results),
                "best_confidence": max((float(m.match.confidence) for m in match_results), default=0.0)
            }
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(ProcessingError(f"Image matching failed: {str(e)}"))
    
    async def _handle_screen_capture(
        self,
        config: VisualProcessingConfig
    ) -> Either[VisualError, Dict[str, Any]]:
        """Handle screen capture operation."""
        try:
            if not config.region:
                return Either.left(ProcessingError("Region required for screen capture"))
            
            # Determine capture mode from config
            mode_mapping = {
                "full_quality": CaptureMode.FULL_QUALITY,
                "balanced": CaptureMode.BALANCED,
                "performance": CaptureMode.PERFORMANCE,
                "privacy_safe": CaptureMode.PRIVACY_SAFE,
                "thumbnail": CaptureMode.THUMBNAIL
            }
            
            capture_mode = mode_mapping.get(
                config.processing_options.get("capture_mode", "balanced"),
                CaptureMode.BALANCED
            )
            
            # Perform screen capture
            capture_result = await self.screen_analysis.capture_screen_region(
                config.region, capture_mode, config.privacy_mode
            )
            
            if capture_result.is_left():
                return Either.left(capture_result.get_left())
            
            capture = capture_result.get_right()
            self.processing_stats["screen_captures"] += 1
            
            # Encode image data as base64
            image_b64 = base64.b64encode(capture.image_data).decode('utf-8')
            
            response = {
                "operation": "capture_screen",
                "success": True,
                "image_data": f"data:image/png;base64,{image_b64}",
                "region": capture.region.to_dict(),
                "capture_mode": capture.capture_mode.value,
                "timestamp": capture.timestamp.isoformat(),
                "file_size_mb": capture.file_size_mb,
                "quality_score": capture.quality_score,
                "privacy_filtered": capture.privacy_filtered,
                "compression_ratio": capture.compression_ratio
            }
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(ProcessingError(f"Screen capture failed: {str(e)}"))
    
    async def _handle_window_analysis(
        self,
        config: VisualProcessingConfig
    ) -> Either[VisualError, Dict[str, Any]]:
        """Handle window analysis operation."""
        try:
            # Get window list
            windows_result = await self.screen_analysis.get_window_list(include_hidden=False)
            if windows_result.is_left():
                return Either.left(windows_result.get_left())
            
            windows = windows_result.get_right()
            
            # Filter windows by region if specified
            if config.region:
                relevant_windows = [
                    w for w in windows if config.region.overlaps_with(w.bounds)
                ]
            else:
                relevant_windows = windows
            
            # Format window information
            window_data = []
            for window in relevant_windows:
                window_info = {
                    "window_id": window.window_id,
                    "title": window.title,
                    "application": window.application_name,
                    "bundle_id": window.bundle_id,
                    "bounds": window.bounds.to_dict(),
                    "state": window.state.value,
                    "is_visible": window.is_visible,
                    "layer": window.layer,
                    "alpha": window.alpha,
                    "area": window.area
                }
                window_data.append(window_info)
            
            response = {
                "operation": "analyze_window",
                "success": True,
                "windows_found": len(window_data),
                "windows": window_data,
                "active_window": next(
                    (w for w in window_data if w["state"] == "active"), None
                )
            }
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(ProcessingError(f"Window analysis failed: {str(e)}"))
    
    async def _handle_ui_detection(
        self,
        config: VisualProcessingConfig,
        image_data: Optional[ImageData]
    ) -> Either[VisualError, Dict[str, Any]]:
        """Handle UI element detection operation."""
        try:
            if not config.region:
                return Either.left(ProcessingError("Region required for UI detection"))
            
            if not image_data:
                # Capture screen region
                capture_result = await self.screen_analysis.capture_screen_region(
                    config.region, CaptureMode.BALANCED, config.privacy_mode
                )
                if capture_result.is_left():
                    return Either.left(capture_result.get_left())
                
                image_data = capture_result.get_right().image_data
            
            # Perform UI element detection
            elements_result = await self.image_recognition.detect_ui_elements(
                image_data, config.region
            )
            
            if elements_result.is_left():
                return Either.left(elements_result.get_left())
            
            elements = elements_result.get_right()
            
            # Format element information
            element_data = []
            for element in elements:
                elem_info = {
                    "element_type": element.element_type.value,
                    "confidence": float(element.confidence),
                    "location": element.location.to_dict(),
                    "center_point": element.center_point,
                    "interactive": element.interactive,
                    "visual_state": element.visual_state,
                    "area": element.area
                }
                
                if element.text_content:
                    elem_info["text_content"] = element.text_content
                
                if element.properties:
                    elem_info["properties"] = element.properties
                
                element_data.append(elem_info)
            
            response = {
                "operation": "ui_element_detection",
                "success": True,
                "elements_found": len(element_data),
                "elements": element_data,
                "interactive_elements": [
                    e for e in element_data if e["interactive"]
                ]
            }
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(ProcessingError(f"UI detection failed: {str(e)}"))
    
    async def _handle_color_analysis(
        self,
        config: VisualProcessingConfig
    ) -> Either[VisualError, Dict[str, Any]]:
        """Handle color analysis operation."""
        try:
            if not config.region:
                return Either.left(ProcessingError("Region required for color analysis"))
            
            # Perform color analysis
            color_result = await self.screen_analysis.analyze_color_distribution(config.region)
            if color_result.is_left():
                return Either.left(color_result.get_left())
            
            color_info = color_result.get_right()
            
            response = {
                "operation": "color_analysis",
                "success": True,
                "dominant_colors": [
                    {"rgb": list(color), "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"}
                    for color in color_info.dominant_colors
                ],
                "color_palette": [
                    {
                        "rgb": list(color[:3]),
                        "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                        "percentage": color[3] * 100
                    }
                    for color in color_info.color_palette
                ],
                "average_color": {
                    "rgb": list(color_info.average_color),
                    "hex": f"#{color_info.average_color[0]:02x}{color_info.average_color[1]:02x}{color_info.average_color[2]:02x}"
                },
                "brightness": color_info.brightness,
                "contrast_ratio": color_info.contrast_ratio,
                "color_distribution": color_info.color_distribution
            }
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(ProcessingError(f"Color analysis failed: {str(e)}"))
    
    async def _handle_motion_detection(
        self,
        config: VisualProcessingConfig
    ) -> Either[VisualError, Dict[str, Any]]:
        """Handle motion/change detection operation."""
        try:
            if not config.region:
                return Either.left(ProcessingError("Region required for motion detection"))
            
            sensitivity = config.processing_options.get("sensitivity", 0.2)
            mode = config.processing_options.get("detection_mode", "content_aware")
            
            # Map mode string to enum
            mode_mapping = {
                "pixel_perfect": ChangeDetectionMode.PIXEL_PERFECT,
                "content_aware": ChangeDetectionMode.CONTENT_AWARE,
                "structural": ChangeDetectionMode.STRUCTURAL,
                "motion_only": ChangeDetectionMode.MOTION_ONLY
            }
            
            detection_mode = mode_mapping.get(mode, ChangeDetectionMode.CONTENT_AWARE)
            
            # Perform change detection
            change_result = await self.screen_analysis.detect_screen_changes(
                config.region, detection_mode, sensitivity
            )
            
            if change_result.is_left():
                return Either.left(change_result.get_left())
            
            change = change_result.get_right()
            
            response = {
                "operation": "motion_detection",
                "success": True,
                "changed": change.changed,
                "change_percentage": change.change_percentage,
                "change_type": change.change_type,
                "confidence": change.confidence,
                "is_significant": change.is_significant_change,
                "timestamp": change.timestamp.isoformat(),
                "detection_mode": detection_mode.value,
                "sensitivity": sensitivity
            }
            
            if change.changed_regions and config.include_coordinates:
                response["changed_regions"] = [
                    region.to_dict() for region in change.changed_regions
                ]
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(ProcessingError(f"Motion detection failed: {str(e)}"))
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.processing_stats.copy()
        stats.update({
            "ocr_stats": self.ocr_engine.get_cache_stats(),
            "recognition_stats": self.image_recognition.get_processing_stats(),
            "screen_analysis_stats": self.screen_analysis.get_analysis_stats()
        })
        return stats


# Main MCP Tool Implementation  
async def km_visual_automation(
    operation: str,
    region: Optional[Dict[str, int]] = None,
    image_template: Optional[str] = None,
    image_data: Optional[str] = None,
    ocr_language: str = "en",
    confidence_threshold: float = 0.8,
    include_coordinates: bool = True,
    privacy_mode: bool = True,
    timeout_seconds: int = 30,
    cache_results: bool = True,
    max_results: int = 10,
    processing_options: Optional[Dict[str, Any]] = None,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Advanced visual automation with OCR, image recognition, and screen analysis.
    
    Provides sophisticated visual automation capabilities for AI agents including
    text extraction, image template matching, screen capture, UI element detection,
    and color analysis with comprehensive privacy protection.
    
    Args:
        operation: Visual operation type (ocr_text, find_image, capture_screen, etc.)
        region: Screen region {x, y, width, height, display_id} to operate on
        image_template: Base64 image template for matching operations
        image_data: Base64 image data for analysis (optional, will capture if not provided)
        ocr_language: Language code for OCR (en, es, fr, de, etc.)
        confidence_threshold: Minimum confidence for results (0.0 to 1.0)
        include_coordinates: Include coordinate information in results
        privacy_mode: Enable privacy content filtering and protection
        timeout_seconds: Processing timeout (1 to 300 seconds)
        cache_results: Enable result caching for performance
        max_results: Maximum number of results to return
        processing_options: Additional processing configuration
        ctx: MCP context for request tracking
        
    Returns:
        Dict containing operation results, coordinates, confidence scores, and metadata
        
    Raises:
        ToolError: If validation fails or processing encounters errors
    """
    try:
        logger.info(f"km_visual_automation called: operation={operation}, privacy_mode={privacy_mode}")
        
        # Initialize processor and security manager
        processor = VisualAutomationProcessor()
        security = processor.security_manager
        
        # Rate limiting check
        client_id = ctx.meta.get("client_id", "default") if ctx else "default"
        rate_check = security.check_rate_limit(operation, client_id)
        if rate_check.is_left():
            raise ToolError(f"Rate limit exceeded: {rate_check.get_left().message}")
        
        # Validate operation type
        try:
            visual_operation = VisualOperation(operation.lower())
        except ValueError:
            valid_ops = [op.value for op in VisualOperation]
            raise ToolError(f"Invalid operation '{operation}'. Valid operations: {valid_ops}")
        
        # Validate and parse region
        screen_region = None
        if region:
            region_result = security.validate_region(region)
            if region_result.is_left():
                raise ToolError(f"Invalid region: {region_result.get_left().message}")
            screen_region = region_result.get_right()
        
        # Validate processing options
        proc_options = processing_options or {}
        config_check = security.validate_operation_config(operation, proc_options)
        if config_check.is_left():
            raise ToolError(f"Invalid configuration: {config_check.get_left().message}")
        
        # Validate image inputs
        processed_image_data = None
        if image_data:
            image_result = security.validate_image_data(image_data)
            if image_result.is_left():
                raise ToolError(f"Invalid image data: {image_result.get_left().message}")
            processed_image_data = image_result.get_right()
        
        processed_template_data = None
        if image_template:
            template_result = security.validate_image_data(image_template)
            if template_result.is_left():
                raise ToolError(f"Invalid template data: {template_result.get_left().message}")
            processed_template_data = template_result.get_right()
        
        # Create processing configuration
        config = VisualProcessingConfig(
            operation=visual_operation,
            region=screen_region,
            language=ocr_language,
            confidence_threshold=confidence_threshold,
            include_coordinates=include_coordinates,
            privacy_mode=privacy_mode,
            timeout_seconds=timeout_seconds,
            cache_results=cache_results,
            max_results=max_results,
            processing_options=proc_options
        )
        
        # Process the operation
        result = await processor.process_operation(
            visual_operation, config, processed_image_data, processed_template_data
        )
        
        if result.is_left():
            error = result.get_left()
            if isinstance(error, PermissionError):
                raise ToolError(f"Permission denied: {error.message}")
            elif isinstance(error, PrivacyError):
                raise ToolError(f"Privacy violation: {error.message}")
            else:
                raise ToolError(f"Processing failed: {error.message}")
        
        # Update processing statistics
        processor.processing_stats["operations_completed"] += 1
        
        # Add metadata to response
        response = result.get_right()
        response.update({
            "timestamp": datetime.now().isoformat(),
            "privacy_mode_enabled": privacy_mode,
            "processing_stats": processor.get_processing_stats(),
            "cache_enabled": cache_results
        })
        
        logger.info(f"Visual automation completed successfully: {operation}")
        return response
        
    except ToolError:
        raise
    except Exception as e:
        logger.error(f"km_visual_automation failed: {str(e)}")
        raise ToolError(f"Visual automation failed: {str(e)}")


# Export the tool for MCP server registration
__all__ = ["km_visual_automation"]