# TASK_35: km_visual_automation - OCR, Image Recognition & Screen Analysis

**Created By**: Agent_1 (Platform Expansion) | **Priority**: HIGH | **Duration**: 6 hours
**Technique Focus**: Design by Contract + Type Safety + Computer Vision + Performance Optimization + Security Validation
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: Foundation tasks (TASK_1-20), Action builder integration (TASK_14)
**Blocking**: Visual automation workflows requiring OCR and image recognition

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/KM_MCP.md - Visual automation capabilities (lines 791-812)
- [ ] **Action Integration**: development/tasks/TASK_14.md - Action builder integration patterns
- [ ] **Interface Automation**: development/protocols/KM_MCP.md - Mouse/keyboard automation (lines 768-789)
- [ ] **Security Framework**: src/core/contracts.py - Image processing security validation
- [ ] **Testing Requirements**: tests/TESTING.md - Visual automation testing patterns

## 🎯 Problem Analysis
**Classification**: Visual Intelligence Infrastructure Gap
**Gap Identified**: No OCR, image recognition, or visual automation capabilities for AI-driven visual interaction
**Impact**: AI cannot read screen content, recognize UI elements, or perform visual-based automation

<thinking>
Root Cause Analysis:
1. Current platform focuses on programmatic automation but lacks visual intelligence
2. No OCR capabilities for reading text from images or screen regions
3. Missing image recognition for finding and clicking UI elements
4. Cannot perform screen analysis for dynamic UI automation
5. Essential for complete automation platform that can interact with any application
6. Should integrate with existing interface automation and action builder
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **Visual types**: Define branded types for OCR, image recognition, and screen analysis
- [ ] **Performance framework**: Optimized image processing and caching strategies
- [ ] **Security validation**: Safe image handling and path validation

### Phase 2: OCR Implementation
- [ ] **Text extraction**: Multi-language OCR with 100+ language support
- [ ] **Region selection**: Screen area, window, and coordinate-based OCR
- [ ] **Text validation**: Confidence scoring and result validation
- [ ] **Format handling**: Support for various image formats and color modes

### Phase 3: Image Recognition
- [ ] **Template matching**: Find UI elements using reference images
- [ ] **Fuzzy matching**: Tolerance-based matching for slight variations
- [ ] **Multi-scale detection**: Find images at different sizes and orientations
- [ ] **Performance optimization**: Efficient image processing and caching

### Phase 4: Screen Analysis
- [ ] **Pixel analysis**: Color detection and screen sampling
- [ ] **UI element detection**: Automatic button, field, and control recognition
- [ ] **Change detection**: Monitor screen regions for changes
- [ ] **Multi-monitor support**: Handle complex display configurations

### Phase 5: Integration & Testing
- [ ] **Action builder integration**: Visual actions for macro sequences
- [ ] **Interface automation**: Integration with mouse/keyboard simulation
- [ ] **TESTING.md update**: Visual automation testing coverage and validation
- [ ] **Performance optimization**: Efficient image processing and response times

## 🔧 Implementation Files & Specifications
```
src/server/tools/visual_automation_tools.py        # Main visual automation tool implementation
src/core/visual_processing.py                      # Visual processing type definitions
src/visual/ocr_engine.py                          # OCR processing and text extraction
src/visual/image_recognition.py                   # Image matching and recognition
src/visual/screen_analysis.py                     # Screen analysis and pixel detection
src/visual/performance_cache.py                   # Image processing cache and optimization
tests/tools/test_visual_automation_tools.py       # Unit and integration tests
tests/property_tests/test_visual_automation.py    # Property-based visual validation
```

### km_visual_automation Tool Specification
```python
@mcp.tool()
async def km_visual_automation(
    operation: str,                             # ocr|find_image|pixel_color|screen_analysis
    area: str = "screen",                       # screen|window|coordinates
    coordinates: Optional[Dict] = None,         # {x, y, width, height} for area selection
    language: str = "en",                       # OCR language code (100+ supported)
    image_path: Optional[str] = None,           # Template image for matching
    fuzziness: int = 85,                        # Matching tolerance (0-100)
    confidence_threshold: float = 0.8,          # Minimum confidence for results
    return_coordinates: bool = True,            # Return coordinate information
    cache_results: bool = True,                 # Enable result caching
    timeout: int = 10,                          # Processing timeout in seconds
    output_format: str = "json",                # json|text|xml output format
    preprocessing: Optional[Dict] = None,        # Image preprocessing options
    ctx = None
) -> Dict[str, Any]:
```

### Visual Processing Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Set, Tuple
from enum import Enum
import re
from pathlib import Path

class VisualOperation(Enum):
    """Visual automation operation types."""
    OCR = "ocr"
    FIND_IMAGE = "find_image"
    PIXEL_COLOR = "pixel_color"
    SCREEN_ANALYSIS = "screen_analysis"
    UI_DETECTION = "ui_detection"
    CHANGE_DETECTION = "change_detection"

class AreaType(Enum):
    """Screen area selection types."""
    SCREEN = "screen"
    WINDOW = "window"
    COORDINATES = "coordinates"
    SELECTION = "selection"

class OCRLanguage(Enum):
    """Supported OCR languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    # Add more languages as needed

@dataclass(frozen=True)
class ScreenRegion:
    """Type-safe screen region with validation."""
    x: int
    y: int
    width: int
    height: int
    
    @require(lambda self: self.x >= 0 and self.y >= 0)
    @require(lambda self: self.width > 0 and self.height > 0)
    @require(lambda self: self.width <= 10000 and self.height <= 10000)  # Reasonable limits
    def __post_init__(self):
        pass
    
    def get_center(self) -> Tuple[int, int]:
        """Get center coordinates of region."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within region."""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def intersects(self, other: 'ScreenRegion') -> bool:
        """Check if this region intersects with another."""
        return not (self.x + self.width < other.x or 
                   other.x + other.width < self.x or
                   self.y + self.height < other.y or 
                   other.y + other.height < self.y)

@dataclass(frozen=True)
class OCRResult:
    """OCR text extraction result with metadata."""
    text: str
    confidence: float
    region: ScreenRegion
    language: OCRLanguage
    words: List[Dict[str, Any]] = field(default_factory=list)
    lines: List[Dict[str, Any]] = field(default_factory=list)
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    def __post_init__(self):
        pass
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if OCR result meets confidence threshold."""
        return self.confidence >= threshold
    
    def get_word_count(self) -> int:
        """Get number of detected words."""
        return len(self.words)
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> List[str]:
        """Get words that meet minimum confidence threshold."""
        return [
            word['text'] for word in self.words 
            if word.get('confidence', 0) >= min_confidence
        ]

@dataclass(frozen=True)
class ImageMatch:
    """Image recognition match result."""
    template_path: str
    match_region: ScreenRegion
    confidence: float
    center_coordinates: Tuple[int, int]
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: len(self.template_path) > 0)
    def __post_init__(self):
        pass
    
    def is_good_match(self, threshold: float = 0.8) -> bool:
        """Check if match meets confidence threshold."""
        return self.confidence >= threshold
    
    def get_click_coordinates(self) -> Tuple[int, int]:
        """Get coordinates for clicking the matched element."""
        return self.center_coordinates

@dataclass(frozen=True)
class PixelInfo:
    """Pixel color and position information."""
    x: int
    y: int
    red: int
    green: int
    blue: int
    alpha: int = 255
    
    @require(lambda self: 0 <= self.red <= 255)
    @require(lambda self: 0 <= self.green <= 255)
    @require(lambda self: 0 <= self.blue <= 255)
    @require(lambda self: 0 <= self.alpha <= 255)
    def __post_init__(self):
        pass
    
    def get_hex_color(self) -> str:
        """Get color as hex string."""
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"
    
    def get_rgb_tuple(self) -> Tuple[int, int, int]:
        """Get RGB values as tuple."""
        return (self.red, self.green, self.blue)
    
    def is_similar_color(self, other: 'PixelInfo', tolerance: int = 10) -> bool:
        """Check if colors are similar within tolerance."""
        return (abs(self.red - other.red) <= tolerance and
                abs(self.green - other.green) <= tolerance and
                abs(self.blue - other.blue) <= tolerance)

@dataclass(frozen=True)
class VisualRequest:
    """Complete visual automation request specification."""
    operation: VisualOperation
    area_type: AreaType
    region: Optional[ScreenRegion] = None
    window_name: Optional[str] = None
    language: OCRLanguage = OCRLanguage.ENGLISH
    template_image: Optional[str] = None
    confidence_threshold: float = 0.8
    preprocessing_options: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: 0.0 <= self.confidence_threshold <= 1.0)
    def __post_init__(self):
        # Validate region is provided for coordinate-based operations
        if self.area_type == AreaType.COORDINATES and not self.region:
            raise ValueError("Region required for coordinate-based operations")
        
        # Validate template image for image matching operations
        if self.operation == VisualOperation.FIND_IMAGE and not self.template_image:
            raise ValueError("Template image required for image matching")

class OCREngine:
    """OCR text extraction engine with multi-language support."""
    
    def __init__(self):
        self.supported_languages = set(lang.value for lang in OCRLanguage)
        self.ocr_cache = {}
    
    async def extract_text(self, request: VisualRequest) -> Either[VisualError, OCRResult]:
        """Extract text from screen region using OCR."""
        try:
            # Validate request
            if request.operation != VisualOperation.OCR:
                return Either.left(VisualError.invalid_operation(request.operation))
            
            # Get screen region
            region = await self._get_screen_region(request)
            if region.is_left():
                return region
            
            screen_region = region.get_right()
            
            # Capture screen region
            image_data = await self._capture_screen_region(screen_region)
            if image_data.is_left():
                return image_data
            
            image = image_data.get_right()
            
            # Preprocess image if options provided
            processed_image = await self._preprocess_image(image, request.preprocessing_options)
            
            # Perform OCR
            ocr_result = await self._perform_ocr(
                processed_image, 
                request.language, 
                screen_region
            )
            
            return ocr_result
            
        except Exception as e:
            return Either.left(VisualError.ocr_failed(str(e)))
    
    async def _capture_screen_region(self, region: ScreenRegion) -> Either[VisualError, Any]:
        """Capture screen region as image."""
        try:
            # Use macOS screencapture or similar
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Capture specific region
            cmd = [
                'screencapture',
                '-R', f'{region.x},{region.y},{region.width},{region.height}',
                '-t', 'png',
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return Either.left(VisualError.screen_capture_failed(result.stderr))
            
            # Load image
            from PIL import Image
            image = Image.open(temp_path)
            
            # Clean up temp file
            import os
            os.unlink(temp_path)
            
            return Either.right(image)
            
        except Exception as e:
            return Either.left(VisualError.screen_capture_failed(str(e)))
    
    async def _perform_ocr(self, image: Any, language: OCRLanguage, region: ScreenRegion) -> Either[VisualError, OCRResult]:
        """Perform OCR on image using Tesseract or similar."""
        try:
            import pytesseract
            
            # Configure language
            lang_code = language.value
            if lang_code not in self.supported_languages:
                lang_code = 'en'  # Default to English
            
            # Perform OCR with detailed output
            data = pytesseract.image_to_data(
                image, 
                lang=lang_code,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text_parts = []
            words = []
            total_confidence = 0
            valid_words = 0
            
            for i in range(len(data['text'])):
                word_text = data['text'][i].strip()
                if word_text:
                    confidence = float(data['conf'][i]) / 100.0 if data['conf'][i] != -1 else 0.0
                    
                    # Create word info
                    word_info = {
                        'text': word_text,
                        'confidence': confidence,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                    words.append(word_info)
                    
                    if confidence > 0.3:  # Only include reasonably confident words
                        text_parts.append(word_text)
                        total_confidence += confidence
                        valid_words += 1
            
            # Calculate overall confidence
            overall_confidence = total_confidence / valid_words if valid_words > 0 else 0.0
            
            # Join text
            extracted_text = ' '.join(text_parts)
            
            return Either.right(OCRResult(
                text=extracted_text,
                confidence=overall_confidence,
                region=region,
                language=language,
                words=words
            ))
            
        except Exception as e:
            return Either.left(VisualError.ocr_processing_failed(str(e)))

class ImageRecognition:
    """Image recognition and template matching engine."""
    
    def __init__(self):
        self.template_cache = {}
        self.match_cache = {}
    
    async def find_image(self, request: VisualRequest) -> Either[VisualError, List[ImageMatch]]:
        """Find template image on screen."""
        try:
            if request.operation != VisualOperation.FIND_IMAGE:
                return Either.left(VisualError.invalid_operation(request.operation))
            
            if not request.template_image:
                return Either.left(VisualError.missing_template())
            
            # Validate template image exists
            template_path = Path(request.template_image)
            if not template_path.exists():
                return Either.left(VisualError.template_not_found(request.template_image))
            
            # Get screen region
            region = await self._get_screen_region(request)
            if region.is_left():
                return region
            
            screen_region = region.get_right()
            
            # Capture screen
            screen_image = await self._capture_screen_region(screen_region)
            if screen_image.is_left():
                return screen_image
            
            screen = screen_image.get_right()
            
            # Load template
            template = await self._load_template_image(request.template_image)
            if template.is_left():
                return template
            
            template_img = template.get_right()
            
            # Perform template matching
            matches = await self._template_match(
                screen, 
                template_img, 
                screen_region,
                request.confidence_threshold
            )
            
            return matches
            
        except Exception as e:
            return Either.left(VisualError.image_recognition_failed(str(e)))
    
    async def _template_match(self, screen: Any, template: Any, region: ScreenRegion, threshold: float) -> Either[VisualError, List[ImageMatch]]:
        """Perform template matching using OpenCV."""
        try:
            import cv2
            import numpy as np
            
            # Convert PIL images to OpenCV format
            screen_cv = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
            template_cv = cv2.cvtColor(np.array(template), cv2.COLOR_RGB2BGR)
            
            # Perform template matching
            result = cv2.matchTemplate(screen_cv, template_cv, cv2.TM_CCOEFF_NORMED)
            
            # Find all matches above threshold
            locations = np.where(result >= threshold)
            matches = []
            
            template_height, template_width = template_cv.shape[:2]
            
            for pt in zip(*locations[::-1]):  # Switch x and y coordinates
                x, y = pt
                confidence = result[y, x]
                
                # Calculate absolute coordinates
                abs_x = region.x + x
                abs_y = region.y + y
                
                match_region = ScreenRegion(
                    x=abs_x,
                    y=abs_y,
                    width=template_width,
                    height=template_height
                )
                
                center_x = abs_x + template_width // 2
                center_y = abs_y + template_height // 2
                
                match = ImageMatch(
                    template_path=str(template),
                    match_region=match_region,
                    confidence=float(confidence),
                    center_coordinates=(center_x, center_y)
                )
                
                matches.append(match)
            
            # Remove overlapping matches (non-maximum suppression)
            filtered_matches = self._remove_overlapping_matches(matches)
            
            return Either.right(filtered_matches)
            
        except Exception as e:
            return Either.left(VisualError.template_matching_failed(str(e)))
    
    def _remove_overlapping_matches(self, matches: List[ImageMatch]) -> List[ImageMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return matches
        
        # Sort by confidence (highest first)
        sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
        
        filtered = []
        for match in sorted_matches:
            # Check if this match overlaps with any already selected
            overlaps = False
            for selected in filtered:
                if match.match_region.intersects(selected.match_region):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(match)
        
        return filtered

class ScreenAnalyzer:
    """Screen analysis and pixel detection."""
    
    async def get_pixel_color(self, x: int, y: int) -> Either[VisualError, PixelInfo]:
        """Get pixel color at specific coordinates."""
        try:
            # Capture single pixel
            import subprocess
            import tempfile
            from PIL import Image
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Capture 1x1 pixel region
            cmd = [
                'screencapture',
                '-R', f'{x},{y},1,1',
                '-t', 'png',
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return Either.left(VisualError.pixel_capture_failed(result.stderr))
            
            # Load and analyze pixel
            image = Image.open(temp_path)
            pixel = image.getpixel((0, 0))
            
            # Clean up
            import os
            os.unlink(temp_path)
            
            # Handle different pixel formats
            if len(pixel) == 3:  # RGB
                r, g, b = pixel
                a = 255
            elif len(pixel) == 4:  # RGBA
                r, g, b, a = pixel
            else:
                return Either.left(VisualError.unsupported_pixel_format())
            
            pixel_info = PixelInfo(
                x=x,
                y=y,
                red=r,
                green=g,
                blue=b,
                alpha=a
            )
            
            return Either.right(pixel_info)
            
        except Exception as e:
            return Either.left(VisualError.pixel_analysis_failed(str(e)))

class VisualAutomationManager:
    """Comprehensive visual automation management."""
    
    def __init__(self):
        self.ocr_engine = OCREngine()
        self.image_recognition = ImageRecognition()
        self.screen_analyzer = ScreenAnalyzer()
        self.performance_cache = PerformanceCache()
    
    async def execute_visual_operation(self, request: VisualRequest) -> Either[VisualError, Dict[str, Any]]:
        """Execute visual automation operation."""
        try:
            # Security validation
            security_result = self._validate_request_security(request)
            if security_result.is_left():
                return security_result
            
            # Route to appropriate handler
            if request.operation == VisualOperation.OCR:
                result = await self.ocr_engine.extract_text(request)
                if result.is_right():
                    ocr_result = result.get_right()
                    return Either.right({
                        "operation": "ocr",
                        "text": ocr_result.text,
                        "confidence": ocr_result.confidence,
                        "word_count": ocr_result.get_word_count(),
                        "region": {
                            "x": ocr_result.region.x,
                            "y": ocr_result.region.y,
                            "width": ocr_result.region.width,
                            "height": ocr_result.region.height
                        },
                        "words": ocr_result.words
                    })
                else:
                    return result
            
            elif request.operation == VisualOperation.FIND_IMAGE:
                result = await self.image_recognition.find_image(request)
                if result.is_right():
                    matches = result.get_right()
                    return Either.right({
                        "operation": "find_image",
                        "matches_found": len(matches),
                        "matches": [
                            {
                                "confidence": match.confidence,
                                "center_x": match.center_coordinates[0],
                                "center_y": match.center_coordinates[1],
                                "region": {
                                    "x": match.match_region.x,
                                    "y": match.match_region.y,
                                    "width": match.match_region.width,
                                    "height": match.match_region.height
                                }
                            }
                            for match in matches
                        ]
                    })
                else:
                    return result
            
            elif request.operation == VisualOperation.PIXEL_COLOR:
                if not request.region:
                    return Either.left(VisualError.missing_coordinates())
                
                result = await self.screen_analyzer.get_pixel_color(
                    request.region.x, 
                    request.region.y
                )
                if result.is_right():
                    pixel = result.get_right()
                    return Either.right({
                        "operation": "pixel_color",
                        "x": pixel.x,
                        "y": pixel.y,
                        "color": {
                            "red": pixel.red,
                            "green": pixel.green,
                            "blue": pixel.blue,
                            "alpha": pixel.alpha,
                            "hex": pixel.get_hex_color()
                        }
                    })
                else:
                    return result
            
            else:
                return Either.left(VisualError.unsupported_operation(request.operation))
                
        except Exception as e:
            return Either.left(VisualError.execution_error(str(e)))
    
    def _validate_request_security(self, request: VisualRequest) -> Either[VisualError, None]:
        """Validate request for security compliance."""
        # Validate template image path
        if request.template_image:
            if not self._is_safe_image_path(request.template_image):
                return Either.left(VisualError.unsafe_image_path(request.template_image))
        
        # Validate screen region bounds
        if request.region:
            if not self._is_valid_screen_region(request.region):
                return Either.left(VisualError.invalid_screen_region())
        
        return Either.right(None)
    
    def _is_safe_image_path(self, path: str) -> bool:
        """Validate image path for security."""
        # Only allow specific directories
        safe_prefixes = [
            '/Users/',
            '~/Documents/',
            '~/Pictures/',
            './images/',
            './templates/'
        ]
        
        expanded_path = os.path.expanduser(path)
        return any(expanded_path.startswith(prefix) for prefix in safe_prefixes)
    
    def _is_valid_screen_region(self, region: ScreenRegion) -> bool:
        """Validate screen region bounds."""
        # Basic sanity checks
        if region.x < 0 or region.y < 0:
            return False
        if region.width <= 0 or region.height <= 0:
            return False
        if region.width > 10000 or region.height > 10000:  # Reasonable limits
            return False
        return True
```

## 🔒 Security Implementation
```python
class VisualSecurityValidator:
    """Security-first visual automation validation."""
    
    @staticmethod
    def validate_image_path_safety(path: str) -> Either[SecurityError, None]:
        """Validate image path for security."""
        try:
            import os
            from pathlib import Path
            
            # Resolve path
            resolved_path = Path(path).resolve()
            
            # Check for path traversal
            if '..' in str(resolved_path):
                return Either.left(SecurityError("Path traversal detected"))
            
            # Only allow specific file types
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
            if resolved_path.suffix.lower() not in allowed_extensions:
                return Either.left(SecurityError("Unsupported image format"))
            
            # Check file size limits
            if resolved_path.exists():
                file_size = resolved_path.stat().st_size
                if file_size > 50 * 1024 * 1024:  # 50MB limit
                    return Either.left(SecurityError("Image file too large"))
            
            return Either.right(None)
            
        except Exception:
            return Either.left(SecurityError("Invalid image path"))
    
    @staticmethod
    def validate_screen_region_safety(region: ScreenRegion) -> Either[SecurityError, None]:
        """Validate screen region for security."""
        # Check bounds
        if region.x < 0 or region.y < 0:
            return Either.left(SecurityError("Invalid screen coordinates"))
        
        # Check size limits
        if region.width > 5000 or region.height > 5000:
            return Either.left(SecurityError("Screen region too large"))
        
        # Check for reasonable coordinates (within typical screen bounds)
        if region.x > 10000 or region.y > 10000:
            return Either.left(SecurityError("Screen coordinates out of range"))
        
        return Either.right(None)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=2000), st.integers(min_value=0, max_value=2000),
       st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000))
def test_screen_region_properties(x, y, width, height):
    """Property: Valid screen regions should be accepted."""
    region = ScreenRegion(x=x, y=y, width=width, height=height)
    assert region.x == x
    assert region.y == y
    assert region.width == width
    assert region.height == height
    
    center_x, center_y = region.get_center()
    assert region.contains_point(center_x, center_y)

@given(st.text(min_size=1, max_size=1000))
def test_ocr_result_properties(text_content):
    """Property: OCR results should handle various text content."""
    # Filter out empty or whitespace-only text
    if text_content.strip():
        confidence = 0.85
        region = ScreenRegion(x=0, y=0, width=100, height=50)
        
        ocr_result = OCRResult(
            text=text_content,
            confidence=confidence,
            region=region,
            language=OCRLanguage.ENGLISH
        )
        
        assert ocr_result.text == text_content
        assert ocr_result.is_high_confidence(0.8)

@given(st.integers(min_value=0, max_value=255), st.integers(min_value=0, max_value=255),
       st.integers(min_value=0, max_value=255))
def test_pixel_info_properties(red, green, blue):
    """Property: Pixel info should handle all valid RGB values."""
    pixel = PixelInfo(x=100, y=100, red=red, green=green, blue=blue)
    
    assert pixel.red == red
    assert pixel.green == green
    assert pixel.blue == blue
    
    hex_color = pixel.get_hex_color()
    assert hex_color.startswith('#')
    assert len(hex_color) == 7
    
    rgb_tuple = pixel.get_rgb_tuple()
    assert rgb_tuple == (red, green, blue)
```

## 🏗️ Modularity Strategy
- **visual_automation_tools.py**: Main MCP tool interface (<250 lines)
- **visual_processing.py**: Type definitions and core logic (<400 lines)
- **ocr_engine.py**: OCR processing and text extraction (<300 lines)
- **image_recognition.py**: Image matching and recognition (<250 lines)
- **screen_analysis.py**: Screen analysis and pixel detection (<200 lines)
- **performance_cache.py**: Caching and optimization (<150 lines)

## ✅ Success Criteria
- Complete OCR support with 100+ languages and high accuracy text extraction
- Image recognition and template matching with fuzzy tolerance
- Pixel color detection and screen region analysis
- Multi-monitor support and complex display configuration handling
- Comprehensive security validation prevents malicious image processing
- Performance optimization with caching and efficient processing
- Property-based tests validate all visual processing scenarios
- Performance: <2s OCR processing, <1s image matching, <200ms pixel analysis
- Integration with action builder for visual automation sequences
- Documentation: Complete visual automation API with examples and guides
- TESTING.md shows 95%+ test coverage with all visual processing tests passing
- Tool enables AI to read, recognize, and interact with any visual content

## 🔄 Integration Points
- **TASK_14 (km_action_builder)**: Visual actions for macro sequences
- **Interface Automation Tools**: Mouse/keyboard automation based on visual detection
- **TASK_10 (km_macro_manager)**: Visual-triggered macro execution
- **All Existing Tools**: Enable visual validation and interaction for any automation
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- Essential for complete automation platform that can interact with any application
- Security is critical - must validate all image paths and processing operations
- OCR enables reading text from any application or image
- Image recognition enables finding and clicking UI elements automatically
- Performance optimization ensures responsive visual processing
- Success here enables AI to interact with applications through visual intelligence