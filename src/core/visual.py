"""
Advanced visual automation types and processing framework.

This module implements sophisticated visual automation capabilities including OCR,
image recognition, screen analysis, and UI element detection for Keyboard Maestro.
Enables AI to leverage visual automation for applications without programmatic APIs.

Security: Comprehensive privacy protection with sensitive content filtering.
Performance: Optimized image processing with efficient caching and memory management.
Type Safety: Complete branded type system with contract-driven development.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Tuple, NewType
from enum import Enum
from datetime import datetime
import re
import base64
import uuid

from src.core.either import Either
from src.core.errors import ValidationError, SecurityError
from src.core.contracts import require, ensure

# Branded types for visual automation
ImageData = NewType('ImageData', bytes)
TemplateId = NewType('TemplateId', str)
OCRText = NewType('OCRText', str)
ConfidenceScore = NewType('ConfidenceScore', float)

class VisualOperation(Enum):
    """Supported visual automation operations with comprehensive coverage."""
    # OCR operations
    OCR_TEXT = "ocr_text"
    OCR_DOCUMENT = "ocr_document"
    OCR_HANDWRITING = "ocr_handwriting"
    
    # Image recognition
    FIND_IMAGE = "find_image"
    TEMPLATE_MATCH = "template_match"
    FEATURE_DETECTION = "feature_detection"
    
    # Screen analysis
    CAPTURE_SCREEN = "capture_screen"
    ANALYZE_WINDOW = "analyze_window"
    MONITOR_CHANGES = "monitor_changes"
    
    # UI element detection
    UI_ELEMENT_DETECTION = "ui_element_detection"
    BUTTON_DETECTION = "button_detection"
    TEXT_FIELD_DETECTION = "text_field_detection"
    MENU_DETECTION = "menu_detection"
    
    # Color and visual analysis
    COLOR_ANALYSIS = "color_analysis"
    MOTION_DETECTION = "motion_detection"
    LAYOUT_ANALYSIS = "layout_analysis"
    
    # Advanced analysis
    ACCESSIBILITY_ANALYSIS = "accessibility_analysis"
    VISUAL_DIFF = "visual_diff"


class ElementType(Enum):
    """UI element types for detection and interaction."""
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    SLIDER = "slider"
    SCROLL_BAR = "scroll_bar"
    IMAGE = "image"
    ICON = "icon"
    WINDOW = "window"
    DIALOG = "dialog"
    TOOLBAR = "toolbar"
    STATUS_BAR = "status_bar"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ScreenRegion:
    """Type-safe screen region specification with comprehensive validation."""
    x: int
    y: int
    width: int
    height: int
    display_id: Optional[int] = None  # For multi-monitor support
    
    def __post_init__(self):
        """Contract validation for screen region."""
        if not (self.x >= 0 and self.y >= 0):
            raise ValueError("Screen coordinates must be non-negative")
        if not (self.width > 0 and self.height > 0):
            raise ValueError("Region dimensions must be positive")
        if not (self.width <= 8192 and self.height <= 8192):
            raise ValueError("Region dimensions exceed maximum limits")
    
    @property
    def area(self) -> int:
        """Calculate region area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Calculate center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bottom_right(self) -> Tuple[int, int]:
        """Calculate bottom-right coordinates."""
        return (self.x + self.width, self.y + self.height)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within region."""
        return (self.x <= x < self.x + self.width and 
                self.y <= y < self.y + self.height)
    
    def overlaps_with(self, other: 'ScreenRegion') -> bool:
        """Check if this region overlaps with another."""
        return not (self.x + self.width <= other.x or
                   other.x + other.width <= self.x or
                   self.y + self.height <= other.y or
                   other.y + other.height <= self.y)
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary representation."""
        result = {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
        if self.display_id is not None:
            result["display_id"] = self.display_id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'ScreenRegion':
        """Create from dictionary with validation."""
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            display_id=data.get("display_id")
        )


@dataclass(frozen=True)
class OCRResult:
    """OCR text extraction result with comprehensive metadata."""
    text: OCRText
    confidence: ConfidenceScore
    coordinates: Optional[ScreenRegion] = None
    language: str = "en"
    line_boxes: List[ScreenRegion] = field(default_factory=list)
    word_boxes: List[Tuple[str, ScreenRegion]] = field(default_factory=list)
    character_boxes: List[Tuple[str, ScreenRegion]] = field(default_factory=list)
    text_orientation: float = 0.0  # Text rotation angle
    reading_order: List[int] = field(default_factory=list)  # Reading order indices
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Contract validation for OCR result."""
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if len(str(self.text).strip()) == 0:
            raise ValueError("OCR text cannot be empty")
        if not (-360.0 <= self.text_orientation <= 360.0):
            raise ValueError("Text orientation must be between -360 and 360 degrees")
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if result has high confidence (>= 0.8)."""
        return float(self.confidence) >= 0.8
    
    @property
    def word_count(self) -> int:
        """Count words in extracted text."""
        return len(str(self.text).split())
    
    @property
    def line_count(self) -> int:
        """Count lines in extracted text."""
        return len(str(self.text).splitlines())


@dataclass(frozen=True)
class ImageMatch:
    """Image template matching result with location and confidence."""
    found: bool
    confidence: ConfidenceScore
    location: Optional[ScreenRegion] = None
    template_id: TemplateId = TemplateId("")
    template_name: str = ""
    scale_factor: float = 1.0  # Scale at which match was found
    rotation_angle: float = 0.0  # Rotation angle of match
    method: str = "template_matching"  # Matching method used
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Contract validation for image match."""
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not (0.1 <= self.scale_factor <= 5.0):
            raise ValueError("Scale factor must be between 0.1 and 5.0")
        if not (-360.0 <= self.rotation_angle <= 360.0):
            raise ValueError("Rotation angle must be between -360 and 360 degrees")
    
    @property
    def is_good_match(self) -> bool:
        """Check if match has good confidence (>= 0.7)."""
        return float(self.confidence) >= 0.7
    
    @property
    def center_point(self) -> Optional[Tuple[int, int]]:
        """Get center point of match location."""
        if self.location:
            return self.location.center
        return None


@dataclass(frozen=True)
class VisualElement:
    """Detected visual UI element with comprehensive properties."""
    element_type: ElementType
    confidence: ConfidenceScore
    location: ScreenRegion
    text_content: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    accessibility_info: Dict[str, Any] = field(default_factory=dict)
    visual_state: str = "normal"  # normal, hover, pressed, disabled, etc.
    interactive: bool = True
    z_order: int = 0  # Layer order for overlapping elements
    parent_element: Optional[str] = None  # Parent element ID
    children: List[str] = field(default_factory=list)  # Child element IDs
    
    def __post_init__(self):
        """Contract validation for visual element."""
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.visual_state not in ["normal", "hover", "pressed", "disabled", "selected", "focused"]:
            raise ValueError("Invalid visual state")
    
    @property
    def is_reliable(self) -> bool:
        """Check if element detection is reliable (>= 0.75)."""
        return float(self.confidence) >= 0.75
    
    @property
    def center_point(self) -> Tuple[int, int]:
        """Get center point of element."""
        return self.location.center
    
    @property
    def area(self) -> int:
        """Get element area in pixels."""
        return self.location.area


@dataclass(frozen=True)
class ColorInfo:
    """Color analysis information."""
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)  # RGB tuples
    color_palette: List[Tuple[int, int, int, float]] = field(default_factory=list)  # RGB + percentage
    average_color: Tuple[int, int, int] = (0, 0, 0)
    brightness: float = 0.0  # 0.0 to 1.0
    contrast_ratio: float = 0.0
    color_distribution: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate color information."""
        if not (0.0 <= self.brightness <= 1.0):
            raise ValueError("Brightness must be between 0.0 and 1.0")
        if not (0.0 <= self.contrast_ratio <= 21.0):
            raise ValueError("Contrast ratio must be between 0.0 and 21.0")


@dataclass(frozen=True)
class VisualProcessingConfig:
    """Configuration for visual processing operations."""
    operation: VisualOperation
    region: Optional[ScreenRegion] = None
    language: str = "en"
    confidence_threshold: ConfidenceScore = ConfidenceScore(0.8)
    include_coordinates: bool = True
    privacy_mode: bool = True
    timeout_seconds: int = 30
    cache_results: bool = True
    max_results: int = 100
    processing_options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate processing configuration."""
        if not (0.0 <= float(self.confidence_threshold) <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if not (1 <= self.timeout_seconds <= 300):
            raise ValueError("Timeout must be between 1 and 300 seconds")
        if not (1 <= self.max_results <= 1000):
            raise ValueError("Max results must be between 1 and 1000")


# Error types for visual processing
class VisualError(Exception):
    """Base class for visual processing errors."""
    def __init__(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(f"{error_type}: {message}")
    
    def is_processing_error(self) -> bool:
        """Check if this is a processing error."""
        return self.error_type in ["PROCESSING_ERROR", "OCR_ERROR", "IMAGE_ERROR", "SCREEN_ERROR"]
    
    def is_security_error(self) -> bool:
        """Check if this is a security error."""
        return self.error_type in ["PERMISSION_ERROR", "PRIVACY_ERROR", "SECURITY_VIOLATION"]


class ProcessingError(VisualError):
    """Visual processing operation error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("PROCESSING_ERROR", message, details)


class PermissionError(VisualError):
    """Screen recording or access permission error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("PERMISSION_ERROR", message, details)


class PrivacyError(VisualError):
    """Privacy protection violation error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("PRIVACY_ERROR", message, details)


# Utility functions for visual processing
def create_screen_region(x: int, y: int, width: int, height: int, display_id: Optional[int] = None) -> ScreenRegion:
    """Create a validated screen region."""
    return ScreenRegion(x=x, y=y, width=width, height=height, display_id=display_id)


def merge_overlapping_regions(regions: List[ScreenRegion]) -> List[ScreenRegion]:
    """Merge overlapping screen regions into larger regions."""
    if not regions:
        return []
    
    # Sort regions by x coordinate
    sorted_regions = sorted(regions, key=lambda r: (r.x, r.y))
    merged = [sorted_regions[0]]
    
    for current in sorted_regions[1:]:
        last_merged = merged[-1]
        
        if last_merged.overlaps_with(current):
            # Merge the regions
            min_x = min(last_merged.x, current.x)
            min_y = min(last_merged.y, current.y)
            max_x = max(last_merged.x + last_merged.width, current.x + current.width)
            max_y = max(last_merged.y + last_merged.height, current.y + current.height)
            
            merged[-1] = ScreenRegion(
                x=min_x,
                y=min_y,
                width=max_x - min_x,
                height=max_y - min_y,
                display_id=last_merged.display_id or current.display_id
            )
        else:
            merged.append(current)
    
    return merged


def calculate_region_similarity(region1: ScreenRegion, region2: ScreenRegion) -> float:
    """Calculate similarity between two regions (0.0 to 1.0)."""
    if not region1.overlaps_with(region2):
        return 0.0
    
    # Calculate intersection area
    x1 = max(region1.x, region2.x)
    y1 = max(region1.y, region2.y)
    x2 = min(region1.x + region1.width, region2.x + region2.width)
    y2 = min(region1.y + region1.height, region2.y + region2.height)
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = region1.area + region2.area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def validate_image_data(image_data: bytes) -> Either[VisualError, ImageData]:
    """Validate image data format and size."""
    if len(image_data) == 0:
        return Either.left(ProcessingError("Image data is empty"))
    
    if len(image_data) > 50 * 1024 * 1024:  # 50MB limit
        return Either.left(ProcessingError("Image data too large (max 50MB)"))
    
    # Check for common image format headers
    valid_headers = [
        b'\xff\xd8\xff',          # JPEG
        b'\x89PNG\r\n\x1a\n',     # PNG
        b'GIF87a',                # GIF87a
        b'GIF89a',                # GIF89a
        b'RIFF',                  # WebP (starts with RIFF)
        b'BM',                    # BMP
        b'II*\x00',               # TIFF (little endian)
        b'MM\x00*',               # TIFF (big endian)
    ]
    
    is_valid_format = any(image_data.startswith(header) for header in valid_headers)
    if not is_valid_format:
        return Either.left(ProcessingError("Invalid or unsupported image format"))
    
    return Either.right(ImageData(image_data))


def generate_template_id() -> TemplateId:
    """Generate a unique template identifier."""
    return TemplateId(f"template_{uuid.uuid4().hex[:8]}")


def normalize_confidence(confidence: float) -> ConfidenceScore:
    """Normalize confidence score to valid range."""
    normalized = max(0.0, min(1.0, confidence))
    return ConfidenceScore(normalized)