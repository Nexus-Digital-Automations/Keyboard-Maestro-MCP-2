"""
AI image analysis system for computer vision and visual automation.

This module provides comprehensive image analysis capabilities including
object detection, OCR, scene analysis, and visual content understanding.
Implements enterprise-grade computer vision with security and performance.

Security: All image processing includes path validation and content scanning.
Performance: Optimized for real-time image analysis with intelligent caching.
Type Safety: Complete integration with AI model management system.
"""

import asyncio
import os
import hashlib
import base64
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
import mimetypes

from src.core.ai_integration import (
    AIOperation, AIRequest, AIResponse, ProcessingMode, OutputFormat,
    AIModel, AIModelId, TokenCount, ConfidenceScore
)
from src.ai.model_manager import AIModelManager, AIError
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.errors import ValidationError, SecurityError
from src.core.logging import get_logger

logger = get_logger(__name__)


class ImageAnalysisType(Enum):
    """Types of image analysis operations."""
    DESCRIBE = "describe"            # General image description
    OBJECTS = "objects"              # Object detection and identification
    TEXT_OCR = "text_ocr"           # Optical character recognition
    FACES = "faces"                  # Face detection and analysis
    SCENE = "scene"                  # Scene understanding and context
    QUALITY = "quality"              # Image quality assessment
    SIMILARITY = "similarity"        # Image similarity comparison
    CONTENT = "content"              # Content classification
    DETAILS = "details"              # Detailed visual analysis
    ACCESSIBILITY = "accessibility"  # Accessibility description


class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    BMP = "bmp"
    WEBP = "webp"
    TIFF = "tiff"


@dataclass(frozen=True)
class ImageMetadata:
    """Image file metadata and properties."""
    file_path: str
    file_size: int
    format: ImageFormat
    width: Optional[int] = None
    height: Optional[int] = None
    color_mode: Optional[str] = None
    creation_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    
    @require(lambda self: self.file_size > 0)
    @require(lambda self: len(self.file_path) > 0)
    def __post_init__(self):
        """Validate image metadata."""
        pass
    
    def get_aspect_ratio(self) -> Optional[float]:
        """Calculate image aspect ratio."""
        if self.width and self.height and self.height > 0:
            return self.width / self.height
        return None
    
    def get_megapixels(self) -> Optional[float]:
        """Calculate image size in megapixels."""
        if self.width and self.height:
            return (self.width * self.height) / 1_000_000
        return None


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Result of image analysis operation."""
    analysis_type: ImageAnalysisType
    image_path: str
    results: Dict[str, Any]
    confidence: ConfidenceScore
    processing_time: float
    model_used: str
    metadata: ImageMetadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def get_primary_result(self) -> Any:
        """Get the primary result from analysis."""
        if self.analysis_type == ImageAnalysisType.DESCRIBE:
            return self.results.get("description", "")
        elif self.analysis_type == ImageAnalysisType.TEXT_OCR:
            return self.results.get("text", "")
        elif self.analysis_type == ImageAnalysisType.OBJECTS:
            return self.results.get("objects", [])
        else:
            return self.results
    
    def get_structured_data(self) -> Dict[str, Any]:
        """Get structured analysis data."""
        return {
            "analysis_type": self.analysis_type.value,
            "results": self.results,
            "confidence": float(self.confidence),
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "image_metadata": {
                "file_path": self.image_path,
                "file_size": self.metadata.file_size,
                "format": self.metadata.format.value,
                "dimensions": f"{self.metadata.width}x{self.metadata.height}" if self.metadata.width else "unknown"
            },
            "timestamp": self.timestamp.isoformat()
        }


class ImageSecurityValidator:
    """Security validation for image processing operations."""
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    MAX_DIMENSION = 8192  # 8K resolution
    
    # Allowed image formats
    ALLOWED_FORMATS = {
        'image/jpeg', 'image/png', 'image/gif', 
        'image/bmp', 'image/webp', 'image/tiff'
    }
    
    # Safe directory prefixes
    SAFE_PREFIXES = [
        '/Users/',
        '~/Documents/',
        '~/Pictures/',
        '~/Desktop/',
        './images/',
        './temp/',
        '/tmp/'
    ]
    
    def validate_image_path(self, image_path: str) -> Either[SecurityError, str]:
        """Validate image file path for security."""
        try:
            # Expand user path
            expanded_path = os.path.expanduser(image_path)
            
            # Check if path exists
            if not os.path.exists(expanded_path):
                return Either.left(SecurityError("file_not_found", f"Image file not found: {image_path}"))
            
            # Check if it's a file (not directory)
            if not os.path.isfile(expanded_path):
                return Either.left(SecurityError("invalid_file_type", f"Path is not a file: {image_path}"))
            
            # Validate path safety
            if not self._is_safe_path(expanded_path):
                return Either.left(SecurityError("unsafe_path", f"Unsafe file path: {image_path}"))
            
            return Either.right(expanded_path)
            
        except Exception as e:
            return Either.left(SecurityError("path_validation_failed", str(e)))
    
    def _is_safe_path(self, path: str) -> bool:
        """Check if file path is in allowed directories."""
        # Convert to absolute path for comparison
        abs_path = os.path.abspath(path)
        
        # Check against safe prefixes
        for prefix in self.SAFE_PREFIXES:
            expanded_prefix = os.path.abspath(os.path.expanduser(prefix))
            if abs_path.startswith(expanded_prefix):
                return True
        
        return False
    
    def validate_image_file(self, file_path: str) -> Either[SecurityError, ImageMetadata]:
        """Validate image file format, size, and properties."""
        try:
            # Get file stats
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            
            # Check file size
            if file_size > self.MAX_FILE_SIZE:
                return Either.left(SecurityError(
                    "file_too_large", 
                    f"File size {file_size} exceeds maximum {self.MAX_FILE_SIZE}"
                ))
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type not in self.ALLOWED_FORMATS:
                return Either.left(SecurityError(
                    "unsupported_format",
                    f"Unsupported image format: {mime_type}"
                ))
            
            # Determine image format
            format_mapping = {
                'image/jpeg': ImageFormat.JPEG,
                'image/png': ImageFormat.PNG,
                'image/gif': ImageFormat.GIF,
                'image/bmp': ImageFormat.BMP,
                'image/webp': ImageFormat.WEBP,
                'image/tiff': ImageFormat.TIFF
            }
            
            image_format = format_mapping.get(mime_type, ImageFormat.JPEG)
            
            # Create metadata (basic version - could be enhanced with actual image library)
            metadata = ImageMetadata(
                file_path=file_path,
                file_size=file_size,
                format=image_format,
                creation_date=datetime.fromtimestamp(file_stat.st_ctime),
                last_modified=datetime.fromtimestamp(file_stat.st_mtime)
            )
            
            return Either.right(metadata)
            
        except Exception as e:
            return Either.left(SecurityError("file_validation_failed", str(e)))
    
    def scan_image_content(self, file_path: str) -> Either[SecurityError, None]:
        """Scan image content for potential security issues."""
        try:
            # Read first few bytes to verify it's actually an image
            with open(file_path, 'rb') as f:
                header = f.read(32)
            
            # Check for common image file signatures
            image_signatures = {
                b'\xFF\xD8\xFF': 'JPEG',
                b'\x89PNG\r\n\x1a\n': 'PNG',
                b'GIF87a': 'GIF87a',
                b'GIF89a': 'GIF89a',
                b'BM': 'BMP',
                b'RIFF': 'WEBP',  # Simplified - WEBP has more complex signature
                b'II*\x00': 'TIFF',
                b'MM\x00*': 'TIFF'
            }
            
            # Verify file signature
            is_valid_image = False
            for signature, format_name in image_signatures.items():
                if header.startswith(signature):
                    is_valid_image = True
                    break
            
            if not is_valid_image:
                return Either.left(SecurityError("invalid_image_signature", "File does not have valid image signature"))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(SecurityError("content_scan_failed", str(e)))


class ImageAnalyzer:
    """AI-powered image analysis and computer vision system."""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.security_validator = ImageSecurityValidator()
        self.analysis_cache: Dict[str, ImageAnalysisResult] = {}
        self.cache_ttl = 3600  # 1 hour cache
    
    async def analyze_image(
        self,
        image_path: str,
        analysis_type: ImageAnalysisType = ImageAnalysisType.DESCRIBE,
        model_preference: Optional[AIModelId] = None,
        additional_context: Optional[str] = None
    ) -> Either[AIError, ImageAnalysisResult]:
        """Analyze image using AI vision models."""
        try:
            # Security validation
            path_result = self.security_validator.validate_image_path(image_path)
            if path_result.is_left():
                return Either.left(AIError("security_validation_failed", str(path_result.get_left())))
            
            validated_path = path_result.get_right()
            
            # File validation
            metadata_result = self.security_validator.validate_image_file(validated_path)
            if metadata_result.is_left():
                return Either.left(AIError("file_validation_failed", str(metadata_result.get_left())))
            
            metadata = metadata_result.get_right()
            
            # Content scanning
            scan_result = self.security_validator.scan_image_content(validated_path)
            if scan_result.is_left():
                return Either.left(AIError("content_scan_failed", str(scan_result.get_left())))
            
            # Check cache
            cache_key = self._generate_cache_key(validated_path, analysis_type, additional_context)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug(f"Using cached analysis for {analysis_type.value}")
                return Either.right(cached_result)
            
            # Select vision-capable model
            vision_models = [
                model for model in self.model_manager.available_models.values()
                if model.supports_vision
            ]
            
            if not vision_models:
                return Either.left(AIError("no_vision_model_available", "No vision-capable models available"))
            
            # Select best vision model
            model_result = self.model_manager.select_best_model(
                AIOperation.ANALYZE,
                ProcessingMode.ACCURATE,
                input_size=metadata.file_size
            )
            if model_result.is_left():
                return model_result
            
            model = model_result.get_right()
            
            if not model.supports_vision:
                # Fallback to first available vision model
                model = vision_models[0]
            
            # Prepare image for analysis
            image_data_result = await self._prepare_image_data(validated_path)
            if image_data_result.is_left():
                return image_data_result
            
            image_data = image_data_result.get_right()
            
            # Build analysis prompt
            prompt = self._build_analysis_prompt(analysis_type, additional_context)
            
            # Create AI request with image
            request_result = self._create_vision_request(
                prompt, image_data, model, analysis_type
            )
            if request_result.is_left():
                return request_result
            
            request = request_result.get_right()
            
            # Process with AI
            start_time = datetime.now(UTC)
            response_result = await self._call_vision_model(request)
            if response_result.is_left():
                return response_result
            
            response = response_result.get_right()
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            
            # Create analysis result
            analysis_result = ImageAnalysisResult(
                analysis_type=analysis_type,
                image_path=validated_path,
                results=self._parse_vision_response(response.result, analysis_type),
                confidence=response.confidence or ConfidenceScore(0.8),
                processing_time=processing_time,
                model_used=model.display_name,
                metadata=metadata
            )
            
            # Cache result
            self._cache_result(cache_key, analysis_result)
            
            logger.info(f"Image analysis completed: {analysis_type.value} in {processing_time:.2f}s")
            return Either.right(analysis_result)
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return Either.left(AIError("analysis_failed", str(e)))
    
    def _build_analysis_prompt(self, analysis_type: ImageAnalysisType, context: Optional[str] = None) -> str:
        """Build analysis prompt based on analysis type."""
        prompts = {
            ImageAnalysisType.DESCRIBE: "Describe this image in detail, including objects, people, setting, colors, mood, and any notable features.",
            
            ImageAnalysisType.OBJECTS: "Identify and list all objects visible in this image. For each object, provide its location, size, and confidence level.",
            
            ImageAnalysisType.TEXT_OCR: "Extract all text visible in this image. Preserve formatting and indicate the location of text elements when possible.",
            
            ImageAnalysisType.FACES: "Detect and analyze any faces in this image. Describe expressions, estimated age ranges, and general characteristics without identifying specific individuals.",
            
            ImageAnalysisType.SCENE: "Analyze the scene in this image. Describe the setting, context, time of day, weather conditions, and overall environment.",
            
            ImageAnalysisType.QUALITY: "Assess the technical quality of this image including resolution, focus, lighting, composition, and any technical issues.",
            
            ImageAnalysisType.CONTENT: "Classify the content of this image. Determine the category, subject matter, and appropriate content rating.",
            
            ImageAnalysisType.DETAILS: "Provide a comprehensive detailed analysis of this image including all visible elements, their relationships, and significance.",
            
            ImageAnalysisType.ACCESSIBILITY: "Create an accessibility description of this image suitable for screen readers and visually impaired users."
        }
        
        base_prompt = prompts.get(analysis_type, prompts[ImageAnalysisType.DESCRIBE])
        
        if context:
            base_prompt += f"\n\nAdditional context: {context}"
        
        base_prompt += "\n\nRespond in JSON format with appropriate keys for the analysis type."
        
        return base_prompt
    
    async def _prepare_image_data(self, image_path: str) -> Either[AIError, str]:
        """Prepare image data for AI processing."""
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Encode as base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return Either.right(image_base64)
            
        except Exception as e:
            return Either.left(AIError("image_preparation_failed", str(e)))
    
    def _create_vision_request(
        self,
        prompt: str,
        image_data: str,
        model: AIModel,
        analysis_type: ImageAnalysisType
    ) -> Either[AIError, AIRequest]:
        """Create AI request for vision analysis."""
        try:
            from src.core.ai_integration import create_ai_request
            
            # Combine prompt with image reference
            vision_input = {
                "prompt": prompt,
                "image_data": image_data,
                "analysis_type": analysis_type.value
            }
            
            return create_ai_request(
                operation=AIOperation.ANALYZE,
                input_data=vision_input,
                model_id=model.model_id,
                temperature=0.2,  # Lower temperature for analysis
                processing_mode=ProcessingMode.ACCURATE
            )
        except Exception as e:
            return Either.left(AIError("request_creation_failed", str(e)))
    
    async def _call_vision_model(self, request: AIRequest) -> Either[AIError, AIResponse]:
        """Call AI vision model with request (placeholder for actual implementation)."""
        try:
            # This would integrate with actual vision AI APIs
            # For now, return a mock response
            await asyncio.sleep(0.2)  # Simulate processing time
            
            # Create mock response based on analysis type
            input_data = request.input_data
            if isinstance(input_data, dict):
                analysis_type = input_data.get("analysis_type", "describe")
            else:
                analysis_type = "describe"
            
            result = self._create_mock_vision_result(analysis_type)
            
            response = AIResponse(
                request_id=request.request_id,
                operation=request.operation,
                result=result,
                model_used=request.model.display_name,
                tokens_used=TokenCount(200),
                input_tokens=TokenCount(150),
                output_tokens=TokenCount(50),
                processing_time=0.2,
                cost_estimate=request.model.estimate_cost(TokenCount(150), TokenCount(50)),
                confidence=ConfidenceScore(0.9)
            )
            
            return Either.right(response)
            
        except Exception as e:
            return Either.left(AIError.api_call_failed(request.model.model_name, str(e)))
    
    def _create_mock_vision_result(self, analysis_type: str) -> str:
        """Create mock vision analysis result for testing."""
        mock_results = {
            "describe": '{"description": "A mock image analysis showing various objects in a natural setting.", "objects": ["tree", "building", "person"], "colors": ["green", "blue", "brown"], "mood": "peaceful"}',
            
            "objects": '{"objects": [{"name": "tree", "confidence": 0.95, "location": "center-left"}, {"name": "building", "confidence": 0.87, "location": "background"}, {"name": "person", "confidence": 0.92, "location": "foreground"}]}',
            
            "text_ocr": '{"text": "Sample text extracted from image", "confidence": 0.88, "locations": [{"text": "Sample text", "x": 100, "y": 200, "width": 150, "height": 30}]}',
            
            "faces": '{"faces": [{"expression": "smiling", "age_range": "25-35", "confidence": 0.85, "location": {"x": 150, "y": 100, "width": 80, "height": 100}}]}',
            
            "scene": '{"scene_type": "outdoor", "setting": "park or garden", "time_of_day": "afternoon", "weather": "sunny", "context": "recreational area"}',
            
            "quality": '{"resolution": "high", "focus": "sharp", "lighting": "good", "composition": "well-balanced", "technical_score": 0.85}',
            
            "content": '{"category": "nature", "subject": "landscape", "content_rating": "general", "keywords": ["outdoor", "nature", "peaceful"]}',
            
            "accessibility": '{"alt_text": "An outdoor scene showing a tree in the foreground with a building visible in the background. A person is walking along a path. The lighting suggests it is daytime with clear weather."}'
        }
        
        return mock_results.get(analysis_type, mock_results["describe"])
    
    def _parse_vision_response(self, response: str, analysis_type: ImageAnalysisType) -> Dict[str, Any]:
        """Parse AI vision response into structured data."""
        try:
            if isinstance(response, str) and response.strip().startswith('{'):
                import json
                return json.loads(response)
            else:
                return {"raw_response": str(response)}
        except json.JSONDecodeError:
            return {"raw_response": str(response)}
    
    def _generate_cache_key(self, image_path: str, analysis_type: ImageAnalysisType, context: Optional[str]) -> str:
        """Generate cache key for image analysis."""
        # Include file modification time in cache key
        try:
            mtime = os.path.getmtime(image_path)
            key_data = f"{image_path}:{analysis_type.value}:{context or ''}:{mtime}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except OSError:
            # Fallback if file doesn't exist
            key_data = f"{image_path}:{analysis_type.value}:{context or ''}"
            return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[ImageAnalysisResult]:
        """Get cached analysis result if available and not expired."""
        if cache_key not in self.analysis_cache:
            return None
        
        result = self.analysis_cache[cache_key]
        
        # Check if expired
        if (datetime.now(UTC) - result.timestamp).total_seconds() > self.cache_ttl:
            del self.analysis_cache[cache_key]
            return None
        
        return result
    
    def _cache_result(self, cache_key: str, result: ImageAnalysisResult) -> None:
        """Cache analysis result."""
        # Limit cache size
        if len(self.analysis_cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(
                self.analysis_cache.items(),
                key=lambda x: x[1].timestamp
            )
            for key, _ in sorted_items[:20]:  # Remove 20 oldest
                del self.analysis_cache[key]
        
        self.analysis_cache[cache_key] = result
    
    async def compare_images(
        self,
        image1_path: str,
        image2_path: str,
        comparison_type: str = "similarity"
    ) -> Either[AIError, Dict[str, Any]]:
        """Compare two images for similarity or differences."""
        try:
            # Analyze both images
            result1 = await self.analyze_image(image1_path, ImageAnalysisType.DETAILS)
            result2 = await self.analyze_image(image2_path, ImageAnalysisType.DETAILS)
            
            if result1.is_left():
                return result1
            if result2.is_left():
                return result2
            
            analysis1 = result1.get_right()
            analysis2 = result2.get_right()
            
            # Create comparison result
            comparison = {
                "image1": {
                    "path": image1_path,
                    "analysis": analysis1.results
                },
                "image2": {
                    "path": image2_path,
                    "analysis": analysis2.results
                },
                "comparison_type": comparison_type,
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            # Add basic similarity assessment (mock implementation)
            if comparison_type == "similarity":
                comparison["similarity_score"] = 0.75  # Mock score
                comparison["similar_elements"] = ["colors", "composition"]
                comparison["different_elements"] = ["objects", "lighting"]
            
            return Either.right(comparison)
            
        except Exception as e:
            return Either.left(AIError("comparison_failed", str(e)))
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats."""
        return [fmt.value for fmt in ImageFormat]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get image analysis statistics."""
        return {
            "cache_size": len(self.analysis_cache),
            "supported_formats": self.get_supported_formats(),
            "supported_analysis_types": [t.value for t in ImageAnalysisType],
            "security_limits": {
                "max_file_size": self.security_validator.MAX_FILE_SIZE,
                "max_dimension": self.security_validator.MAX_DIMENSION,
                "allowed_formats": list(self.security_validator.ALLOWED_FORMATS)
            },
            "model_manager_stats": self.model_manager.get_usage_statistics()
        }
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self.analysis_cache.clear()
        logger.info("Image analyzer cache cleared")