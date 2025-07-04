"""
Computer Vision Architecture - TASK_61 Phase 1 Core Implementation

Advanced computer vision type definitions and architectural framework for AI-powered image understanding.
Provides comprehensive types, enums, and utilities for object detection, scene analysis, and intelligent image processing.

Architecture: Branded Types + Design by Contract + AI Model Integration + Deep Learning + Real-time Processing
Performance: <200ms object detection, <500ms scene analysis, <100ms classification
Security: Safe image processing, validated model inputs, comprehensive sanitization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, NewType
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import re
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure


# Branded Types for Computer Vision Type Safety
ImageContent = NewType('ImageContent', bytes)
VideoContent = NewType('VideoContent', bytes)
ModelId = NewType('ModelId', str)
ObjectId = NewType('ObjectId', str)
SceneId = NewType('SceneId', str)
AnalysisId = NewType('AnalysisId', str)
ConfidenceThreshold = NewType('ConfidenceThreshold', float)
BoundingBoxId = NewType('BoundingBoxId', str)


def create_image_content(image_data: bytes) -> ImageContent:
    """Create validated image content with security checks."""
    if not image_data or len(image_data) == 0:
        raise ValueError("Image content cannot be empty")
    if len(image_data) > 50 * 1024 * 1024:  # 50MB limit
        raise ValueError("Image content exceeds maximum size")
    
    # Basic image format validation
    image_signatures = {
        b'\xff\xd8\xff': 'JPEG',
        b'\x89\x50\x4e\x47': 'PNG',
        b'\x47\x49\x46\x38': 'GIF',
        b'\x42\x4d': 'BMP',
        b'\x52\x49\x46\x46': 'WEBP'
    }
    
    is_valid_image = any(image_data.startswith(sig) for sig in image_signatures.keys())
    if not is_valid_image:
        raise ValueError("Invalid image format")
    
    return ImageContent(image_data)


def create_model_id(model_name: str) -> ModelId:
    """Create validated model identifier."""
    if not model_name or not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', model_name):
        raise ValueError("Model ID must be a valid identifier")
    return ModelId(model_name.lower())


def create_object_id() -> ObjectId:
    """Create unique object identifier."""
    import uuid
    return ObjectId(f"obj_{uuid.uuid4().hex[:12]}")


def create_scene_id() -> SceneId:
    """Create unique scene identifier."""
    import uuid
    return SceneId(f"scene_{uuid.uuid4().hex[:12]}")


def create_analysis_id() -> AnalysisId:
    """Create unique analysis identifier."""
    import uuid
    return AnalysisId(f"analysis_{uuid.uuid4().hex[:8]}")


def create_bbox_id() -> BoundingBoxId:
    """Create unique bounding box identifier."""
    import uuid
    return BoundingBoxId(f"bbox_{uuid.uuid4().hex[:8]}")


class VisionOperation(Enum):
    """Types of computer vision operations."""
    OBJECT_DETECTION = "object_detection"
    SCENE_CLASSIFICATION = "scene_classification"
    IMAGE_SEGMENTATION = "image_segmentation"
    FACIAL_RECOGNITION = "facial_recognition"
    TEXT_DETECTION = "text_detection"
    OPTICAL_CHARACTER_RECOGNITION = "ocr"
    LANDMARK_DETECTION = "landmark_detection"
    ACTIVITY_RECOGNITION = "activity_recognition"
    DEPTH_ESTIMATION = "depth_estimation"
    MOTION_TRACKING = "motion_tracking"
    STYLE_TRANSFER = "style_transfer"
    IMAGE_ENHANCEMENT = "image_enhancement"
    ANOMALY_DETECTION = "anomaly_detection"
    SIMILARITY_MATCHING = "similarity_matching"
    CONTENT_MODERATION = "content_moderation"


class ObjectCategory(Enum):
    """Categories of detected objects."""
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    FURNITURE = "furniture"
    ELECTRONICS = "electronics"
    FOOD = "food"
    CLOTHING = "clothing"
    BUILDING = "building"
    NATURE = "nature"
    TOOL = "tool"
    SPORTS = "sports"
    MEDICAL = "medical"
    UI_ELEMENT = "ui_element"
    TEXT = "text"
    ICON = "icon"
    BUTTON = "button"
    MENU = "menu"
    WINDOW = "window"
    DIALOG = "dialog"
    UNKNOWN = "unknown"


class SceneType(Enum):
    """Types of detected scenes."""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    OFFICE = "office"
    HOME = "home"
    RESTAURANT = "restaurant"
    STREET = "street"
    NATURE = "nature"
    BEACH = "beach"
    CITY = "city"
    DESKTOP = "desktop"
    APPLICATION = "application"
    WEBSITE = "website"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    VIDEO_CALL = "video_call"
    GAME = "game"
    UNKNOWN = "unknown"


class AnalysisLevel(Enum):
    """Levels of computer vision analysis."""
    FAST = "fast"              # Quick processing, basic detection
    STANDARD = "standard"      # Standard processing, good accuracy
    DETAILED = "detailed"      # Detailed analysis, high accuracy
    COMPREHENSIVE = "comprehensive"  # Full analysis with all features


class ModelType(Enum):
    """Types of computer vision models."""
    CONVOLUTIONAL_NEURAL_NETWORK = "cnn"
    VISION_TRANSFORMER = "vit"
    YOLO = "yolo"
    RCNN = "rcnn"
    MOBILENET = "mobilenet"
    EFFICIENTNET = "efficientnet"
    RESNET = "resnet"
    DETECTRON = "detectron"
    CLIP = "clip"
    CUSTOM = "custom"


class ProcessingMode(Enum):
    """Modes for computer vision processing."""
    REAL_TIME = "real_time"    # Real-time processing for live video
    BATCH = "batch"            # Batch processing for multiple images
    STREAMING = "streaming"     # Streaming processing for continuous input
    ON_DEMAND = "on_demand"    # On-demand processing for single requests


@dataclass(frozen=True)
class VisionError(Exception):
    """Base class for computer vision processing errors."""
    message: str
    error_code: str
    operation: Optional[VisionOperation] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectDetectionError(VisionError):
    """Error in object detection processing."""
    pass


@dataclass(frozen=True)
class SceneAnalysisError(VisionError):
    """Error in scene analysis processing."""
    pass


@dataclass(frozen=True)
class ModelLoadingError(VisionError):
    """Error in computer vision model loading."""
    pass


@dataclass(frozen=True)
class ImageProcessingError(VisionError):
    """Error in image processing operations."""
    pass


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box for detected objects."""
    bbox_id: BoundingBoxId
    x: float  # Left coordinate (0-1 normalized)
    y: float  # Top coordinate (0-1 normalized)
    width: float  # Width (0-1 normalized)
    height: float  # Height (0-1 normalized)
    confidence: float
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.x <= 1.0) or not (0.0 <= self.y <= 1.0):
            raise ValueError("Coordinates must be normalized between 0.0 and 1.0")
        if not (0.0 < self.width <= 1.0) or not (0.0 < self.height <= 1.0):
            raise ValueError("Dimensions must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class DetectedObject:
    """Object detected in an image."""
    object_id: ObjectId
    category: ObjectCategory
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    attributes: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class SceneAnalysis:
    """Scene analysis result."""
    scene_id: SceneId
    scene_type: SceneType
    confidence: float
    description: str
    environment_attributes: Dict[str, Any] = field(default_factory=dict)
    lighting_conditions: Dict[str, float] = field(default_factory=dict)
    color_palette: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.complexity_score <= 1.0):
            raise ValueError("Complexity score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class TextDetection:
    """Text detected in an image."""
    text_id: str
    text_content: str
    confidence: float
    bounding_box: BoundingBox
    language: str = "en"
    font_properties: Dict[str, Any] = field(default_factory=dict)
    reading_order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class VisionModel:
    """Computer vision model configuration and metadata."""
    model_id: ModelId
    model_name: str
    model_type: ModelType
    supported_operations: List[VisionOperation]
    input_resolution: Tuple[int, int]
    supported_formats: List[str]
    processing_speed: str  # fast, medium, slow
    accuracy_level: str  # basic, standard, high, premium
    model_path: Optional[str] = None
    requires_gpu: bool = False
    memory_requirements_mb: int = 512
    batch_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.input_resolution[0] <= 0 or self.input_resolution[1] <= 0:
            raise ValueError("Input resolution must be positive")
        if self.memory_requirements_mb <= 0:
            raise ValueError("Memory requirements must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


@dataclass(frozen=True)
class VisionProcessingRequest:
    """Request for computer vision processing."""
    request_id: AnalysisId
    operation: VisionOperation
    image_content: ImageContent
    model_id: Optional[ModelId] = None
    analysis_level: AnalysisLevel = AnalysisLevel.STANDARD
    confidence_threshold: float = 0.5
    max_objects: int = 100
    processing_mode: ProcessingMode = ProcessingMode.ON_DEMAND
    roi_coordinates: Optional[Tuple[float, float, float, float]] = None  # x, y, width, height
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    
    def __post_init__(self):
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if self.max_objects <= 0:
            raise ValueError("Max objects must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")


@dataclass(frozen=True)
class VisionProcessingResult:
    """Result from computer vision processing."""
    result_id: str
    request_id: AnalysisId
    operation: VisionOperation
    success: bool
    processing_time_ms: float
    detected_objects: List[DetectedObject] = field(default_factory=list)
    scene_analysis: Optional[SceneAnalysis] = None
    text_detections: List[TextDetection] = field(default_factory=list)
    image_metadata: Dict[str, Any] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[VisionError] = field(default_factory=list)
    
    def __post_init__(self):
        if self.processing_time_ms < 0:
            raise ValueError("Processing time cannot be negative")


@dataclass(frozen=True)
class VideoAnalysis:
    """Video analysis result for motion and temporal understanding."""
    video_id: str
    duration_seconds: float
    frame_count: int
    fps: float
    motion_detected: bool
    activity_classification: List[str] = field(default_factory=list)
    object_tracking: Dict[str, List[BoundingBox]] = field(default_factory=dict)
    scene_changes: List[float] = field(default_factory=list)  # Timestamps
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        if self.frame_count <= 0:
            raise ValueError("Frame count must be positive")
        if self.fps <= 0:
            raise ValueError("FPS must be positive")


@dataclass(frozen=True)
class ImageEnhancement:
    """Image enhancement and processing result."""
    enhancement_id: str
    original_quality_score: float
    enhanced_quality_score: float
    enhancement_operations: List[str]
    processing_parameters: Dict[str, Any] = field(default_factory=dict)
    before_after_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.original_quality_score <= 1.0):
            raise ValueError("Quality scores must be between 0.0 and 1.0")
        if not (0.0 <= self.enhanced_quality_score <= 1.0):
            raise ValueError("Quality scores must be between 0.0 and 1.0")


# Utility Functions
def validate_confidence_threshold(threshold: float) -> ConfidenceThreshold:
    """Validate and create confidence threshold."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    return ConfidenceThreshold(threshold)


def validate_image_content(image_data: bytes) -> Either[VisionError, ImageContent]:
    """Validate and sanitize image content for computer vision processing."""
    try:
        if not image_data:
            return Either.left(VisionError("Empty image data", "EMPTY_IMAGE"))
        
        if len(image_data) > 100 * 1024 * 1024:  # 100MB limit
            return Either.left(VisionError(f"Image exceeds maximum size", "IMAGE_TOO_LARGE"))
        
        # Check for malicious content patterns
        malicious_patterns = [
            b'<script',  # Embedded scripts
            b'javascript:',  # JavaScript URLs
            b'<?php',  # PHP code
            b'eval(',  # Code evaluation
        ]
        
        for pattern in malicious_patterns:
            if pattern in image_data[:1024]:  # Check first 1KB
                return Either.left(VisionError("Potentially malicious content detected", "MALICIOUS_CONTENT"))
        
        return Either.right(create_image_content(image_data))
        
    except Exception as e:
        return Either.left(VisionError(f"Image validation failed: {str(e)}", "VALIDATION_ERROR"))


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Calculate intersection
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.width, box2.x + box2.width)
    y2 = min(box1.y + box1.height, box2.y + box2.height)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def filter_objects_by_confidence(
    objects: List[DetectedObject],
    threshold: float
) -> List[DetectedObject]:
    """Filter detected objects by confidence threshold."""
    return [obj for obj in objects if obj.confidence >= threshold]


def non_maximum_suppression(
    objects: List[DetectedObject],
    iou_threshold: float = 0.5
) -> List[DetectedObject]:
    """Apply Non-Maximum Suppression to remove overlapping detections."""
    if not objects:
        return objects
    
    # Sort by confidence (descending)
    sorted_objects = sorted(objects, key=lambda x: x.confidence, reverse=True)
    
    keep = []
    for obj in sorted_objects:
        # Check overlap with already kept objects
        should_keep = True
        for kept_obj in keep:
            if (obj.category == kept_obj.category and 
                calculate_iou(obj.bounding_box, kept_obj.bounding_box) > iou_threshold):
                should_keep = False
                break
        
        if should_keep:
            keep.append(obj)
    
    return keep


def merge_overlapping_boxes(
    boxes: List[BoundingBox],
    iou_threshold: float = 0.3
) -> List[BoundingBox]:
    """Merge overlapping bounding boxes."""
    if not boxes:
        return boxes
    
    merged = []
    used = set()
    
    for i, box1 in enumerate(boxes):
        if i in used:
            continue
            
        # Find all boxes that overlap with this one
        group = [box1]
        used.add(i)
        
        for j, box2 in enumerate(boxes[i+1:], i+1):
            if j not in used and calculate_iou(box1, box2) > iou_threshold:
                group.append(box2)
                used.add(j)
        
        # Merge the group into a single box
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Calculate merged bounding box
            min_x = min(box.x for box in group)
            min_y = min(box.y for box in group)
            max_x = max(box.x + box.width for box in group)
            max_y = max(box.y + box.height for box in group)
            
            # Average confidence
            avg_confidence = sum(box.confidence for box in group) / len(group)
            
            merged_box = BoundingBox(
                bbox_id=create_bbox_id(),
                x=min_x,
                y=min_y,
                width=max_x - min_x,
                height=max_y - min_y,
                confidence=avg_confidence,
                label=f"merged_{len(group)}_boxes"
            )
            merged.append(merged_box)
    
    return merged


def calculate_scene_complexity(scene: SceneAnalysis, objects: List[DetectedObject]) -> float:
    """Calculate scene complexity score based on objects and scene analysis."""
    # Base complexity from number of objects
    object_complexity = min(1.0, len(objects) / 20.0)  # Normalize to 20 objects = max complexity
    
    # Complexity from object diversity
    unique_categories = len(set(obj.category for obj in objects))
    diversity_complexity = min(1.0, unique_categories / 10.0)  # 10+ categories = max diversity
    
    # Complexity from confidence distribution
    if objects:
        confidences = [obj.confidence for obj in objects]
        confidence_std = (sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences))**0.5
        confidence_complexity = confidence_std  # Higher std = more complex
    else:
        confidence_complexity = 0.0
    
    # Combined complexity score
    complexity = (object_complexity * 0.4 + 
                 diversity_complexity * 0.4 + 
                 confidence_complexity * 0.2)
    
    return min(1.0, complexity)


@require(lambda processing_result: isinstance(processing_result, VisionProcessingResult))
def validate_processing_result(processing_result: VisionProcessingResult) -> bool:
    """Validate computer vision processing result integrity."""
    # Check confidence scores
    for obj in processing_result.detected_objects:
        if not (0.0 <= obj.confidence <= 1.0):
            return False
        if not (0.0 <= obj.bounding_box.confidence <= 1.0):
            return False
    
    for text in processing_result.text_detections:
        if not (0.0 <= text.confidence <= 1.0):
            return False
    
    # Check scene analysis if present
    if processing_result.scene_analysis:
        scene = processing_result.scene_analysis
        if not (0.0 <= scene.confidence <= 1.0):
            return False
        if not (0.0 <= scene.complexity_score <= 1.0):
            return False
    
    # Check processing time
    if processing_result.processing_time_ms < 0:
        return False
    
    return True


def is_vision_related(description: str) -> bool:
    """Check if description is related to computer vision operations."""
    vision_keywords = [
        "detect", "recognize", "identify", "find", "locate", "see", "look",
        "object", "person", "face", "text", "scene", "image", "picture",
        "visual", "vision", "camera", "screen", "capture", "analyze",
        "classification", "detection", "segmentation", "tracking", "ocr",
        "enhancement", "filter", "processing", "understanding", "reading"
    ]
    
    description_lower = description.lower()
    return any(keyword in description_lower for keyword in vision_keywords)