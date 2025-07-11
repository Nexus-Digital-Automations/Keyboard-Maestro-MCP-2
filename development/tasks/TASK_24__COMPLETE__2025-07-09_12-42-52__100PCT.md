# TASK_24: km_visual_automation - OCR, Image Recognition, Screen Analysis

**Created By**: Agent_ADDER+ (Protocol Gap Analysis) | **Priority**: HIGH | **Duration**: 6 hours
**Technique Focus**: Computer Vision + Security Validation + Performance Optimization + Privacy Protection
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED
**Assigned**: Agent_3
**Dependencies**: TASK_21 (conditions), TASK_22 (control flow) for intelligent visual workflows
**Blocking**: Advanced UI automation and screen-based automation workflows

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Visual automation specification
- [ ] **KM Documentation**: development/protocols/KM_MCP.md - OCR and image recognition actions
- [ ] **Foundation**: src/core/types.py - Image and coordinate type definitions
- [ ] **Privacy Standards**: macOS screen recording permissions and privacy requirements
- [ ] **Testing Framework**: tests/TESTING.md - Visual automation testing requirements

## 🎯 Problem Analysis
**Classification**: Missing High-Value Functionality
**Gap Identified**: No visual automation capabilities (OCR, image recognition, screen analysis)
**Impact**: AI cannot leverage Keyboard Maestro's most powerful visual automation features

<thinking>
Root Cause Analysis:
1. Current implementation is text-based only - missing visual capabilities
2. Keyboard Maestro's OCR and image recognition are unique strengths not exposed
3. Visual automation enables UI interaction without accessibility API dependencies
4. Screen analysis allows AI to understand and respond to visual application states
5. This is essential for automating applications that don't provide programmatic APIs
6. OCR enables text extraction from images, PDFs, and screen regions
7. Image recognition enables button/element detection for UI automation
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **Visual type system**: Define branded types for images, coordinates, OCR results
- [ ] **Privacy framework**: Screen recording permission management and validation
- [ ] **Performance optimization**: Image processing, caching, and memory management

### Phase 2: Core OCR Capabilities
- [ ] **Text extraction**: OCR from screen regions, images, and documents
- [ ] **Language support**: Multi-language OCR with confidence scoring
- [ ] **Text analysis**: Pattern recognition, formatting preservation, structure extraction
- [ ] **Result filtering**: Text validation, noise removal, confidence thresholds

### Phase 3: Image Recognition & Analysis
- [ ] **Template matching**: Find UI elements using image templates
- [ ] **Feature detection**: Identify buttons, text fields, menus, and controls
- [ ] **Color analysis**: Color sampling, palette extraction, visual state detection
- [ ] **Motion detection**: Screen change detection and animation tracking

### Phase 4: Screen & Window Analysis
- [ ] **Screen capture**: Secure screenshot capture with privacy validation
- [ ] **Window detection**: Application window boundaries and content analysis
- [ ] **UI hierarchy**: Element structure analysis and relationship mapping
- [ ] **Accessibility integration**: Combine visual and accessibility API data

### Phase 5: Integration & Security
- [ ] **AppleScript integration**: Safe visual action generation and execution
- [ ] **Privacy protection**: Sensitive content detection and filtering
- [ ] **Property-based tests**: Hypothesis validation for visual processing
- [ ] **TESTING.md update**: Visual automation test coverage and performance metrics

## 🔧 Implementation Files & Specifications
```
src/server/tools/visual_automation_tools.py   # Main visual automation tool implementation
src/core/visual.py                           # Visual type definitions and processing
src/integration/km_visual.py                 # KM-specific visual integration
src/vision/ocr_engine.py                     # OCR processing and text extraction
src/vision/image_recognition.py              # Image matching and recognition
src/vision/screen_analysis.py                # Screen capture and analysis
tests/tools/test_visual_automation_tools.py  # Unit and integration tests
tests/property_tests/test_visual.py          # Property-based visual validation
```

### km_visual_automation Tool Specification
```python
@mcp.tool()
async def km_visual_automation(
    operation: str,                          # ocr_text|find_image|capture_screen|analyze_window
    region: Optional[Dict[str, int]] = None, # Screen region {x, y, width, height}
    image_template: Optional[str] = None,    # Base64 image or file path for matching
    ocr_language: str = "en",               # OCR language code
    confidence_threshold: float = 0.8,      # Minimum confidence for results
    include_coordinates: bool = True,        # Include coordinate information
    privacy_mode: bool = True,              # Enable privacy content filtering
    timeout_seconds: int = 30,              # Processing timeout
    cache_results: bool = True,             # Cache processed results
    ctx = None
) -> Dict[str, Any]:
```

### Visual Processing Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple
from enum import Enum
import base64

class VisualOperation(Enum):
    """Supported visual automation operations."""
    OCR_TEXT = "ocr_text"
    FIND_IMAGE = "find_image"
    CAPTURE_SCREEN = "capture_screen"
    ANALYZE_WINDOW = "analyze_window"
    COLOR_ANALYSIS = "color_analysis"
    MOTION_DETECTION = "motion_detection"
    TEXT_RECOGNITION = "text_recognition"
    UI_ELEMENT_DETECTION = "ui_element_detection"

@dataclass(frozen=True)
class ScreenRegion:
    """Type-safe screen region specification."""
    x: int
    y: int
    width: int
    height: int
    
    @require(lambda self: self.x >= 0 and self.y >= 0)
    @require(lambda self: self.width > 0 and self.height > 0)
    @require(lambda self: self.width <= 8192 and self.height <= 8192)  # Reasonable limits
    def __post_init__(self):
        pass
    
    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

@dataclass(frozen=True)
class OCRResult:
    """OCR text extraction result."""
    text: str
    confidence: float
    coordinates: Optional[ScreenRegion] = None
    language: str = "en"
    line_boxes: List[ScreenRegion] = field(default_factory=list)
    word_boxes: List[Tuple[str, ScreenRegion]] = field(default_factory=list)
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: len(self.text.strip()) > 0)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class ImageMatch:
    """Image template matching result."""
    found: bool
    confidence: float
    location: Optional[ScreenRegion] = None
    template_name: str = ""
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class VisualElement:
    """Detected visual UI element."""
    element_type: str  # button, text_field, menu, etc.
    confidence: float
    location: ScreenRegion
    text_content: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: len(self.element_type) > 0)
    def __post_init__(self):
        pass

class VisualProcessor:
    """Core visual processing engine."""
    
    @require(lambda region: region.width > 0 and region.height > 0)
    @ensure(lambda result: result.is_right() or result.get_left().is_processing_error())
    async def extract_text_from_region(
        self,
        region: ScreenRegion,
        language: str = "en",
        confidence_threshold: float = 0.8
    ) -> Either[VisualError, OCRResult]:
        """Extract text from screen region using OCR."""
        pass
    
    @require(lambda template: len(template) > 0)
    @ensure(lambda result: result.is_right() or result.get_left().is_processing_error())
    async def find_image_template(
        self,
        template: Union[str, bytes],
        search_region: Optional[ScreenRegion] = None,
        confidence_threshold: float = 0.8
    ) -> Either[VisualError, List[ImageMatch]]:
        """Find image template matches on screen."""
        pass
    
    async def capture_screen_region(
        self,
        region: Optional[ScreenRegion] = None,
        privacy_filter: bool = True
    ) -> Either[VisualError, bytes]:
        """Capture screen region with privacy filtering."""
        pass
    
    async def analyze_ui_elements(
        self,
        region: ScreenRegion,
        element_types: List[str] = None
    ) -> Either[VisualError, List[VisualElement]]:
        """Analyze and detect UI elements in region."""
        pass
```

## 🔒 Security & Privacy Implementation
```python
class VisualSecurityManager:
    """Security-first visual automation with privacy protection."""
    
    SENSITIVE_PATTERNS = [
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card numbers
        r'\b\d{3}-\d{2}-\d{4}\b',                        # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\bpassword\b',                                 # Password fields
        r'\b(?:pin|ssn|social|security)\b'               # Sensitive terms
    ]
    
    @staticmethod
    def validate_screen_region(region: ScreenRegion) -> Either[SecurityError, None]:
        """Validate screen region for security constraints."""
        # Get screen dimensions
        screen_width, screen_height = get_screen_dimensions()
        
        # Validate bounds
        if region.x < 0 or region.y < 0:
            return Either.left(SecurityError("Negative coordinates not allowed"))
        
        if region.x + region.width > screen_width or region.y + region.height > screen_height:
            return Either.left(SecurityError("Region extends beyond screen bounds"))
        
        # Check for suspicious large regions (potential screen scraping)
        if region.width * region.height > screen_width * screen_height * 0.8:
            return Either.left(SecurityError("Region too large - potential screen scraping"))
        
        return Either.right(None)
    
    @staticmethod
    def filter_sensitive_content(text: str, privacy_mode: bool = True) -> str:
        """Filter sensitive content from OCR results."""
        if not privacy_mode:
            return text
        
        filtered_text = text
        for pattern in VisualSecurityManager.SENSITIVE_PATTERNS:
            filtered_text = re.sub(pattern, "[REDACTED]", filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    @staticmethod
    def validate_image_template(template_data: bytes) -> Either[SecurityError, None]:
        """Validate image template for security."""
        # Check file size limits
        if len(template_data) > 10 * 1024 * 1024:  # 10MB limit
            return Either.left(SecurityError("Image template too large"))
        
        # Validate image format
        try:
            # Basic image format validation
            if not template_data.startswith((b'\xff\xd8', b'\x89PNG', b'GIF87a', b'GIF89a')):
                return Either.left(SecurityError("Invalid image format"))
        except Exception:
            return Either.left(SecurityError("Image validation failed"))
        
        return Either.right(None)
    
    @staticmethod
    async def check_screen_recording_permission() -> Either[SecurityError, None]:
        """Verify screen recording permissions are granted."""
        # Check macOS screen recording permission
        # This would use macOS APIs to verify permission status
        permission_granted = await verify_screen_recording_permission()
        
        if not permission_granted:
            return Either.left(SecurityError("Screen recording permission required"))
        
        return Either.right(None)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=1920), st.integers(min_value=0, max_value=1080))
def test_screen_region_properties(width, height):
    """Property: Screen regions should handle all valid dimensions."""
    if width > 0 and height > 0:
        region = ScreenRegion(0, 0, width, height)
        assert region.width == width
        assert region.height == height
        assert region.to_dict()["width"] == width

@given(st.text(min_size=1, max_size=1000))
def test_ocr_text_security_properties(text_input):
    """Property: OCR results should filter sensitive content in privacy mode."""
    filtered = VisualSecurityManager.filter_sensitive_content(text_input, privacy_mode=True)
    
    # Check that sensitive patterns are redacted
    for pattern in VisualSecurityManager.SENSITIVE_PATTERNS:
        matches = re.findall(pattern, text_input, re.IGNORECASE)
        if matches:
            assert "[REDACTED]" in filtered

@given(st.floats(min_value=0.0, max_value=1.0))
def test_confidence_threshold_properties(confidence):
    """Property: Confidence thresholds should handle all valid ranges."""
    ocr_result = OCRResult("test", confidence, language="en")
    assert 0.0 <= ocr_result.confidence <= 1.0
    assert ocr_result.text == "test"
```

## 🏗️ Modularity Strategy
- **visual_automation_tools.py**: Main MCP tool interface (<250 lines)
- **visual.py**: Type definitions and validation (<200 lines)
- **ocr_engine.py**: OCR processing and text extraction (<300 lines)
- **image_recognition.py**: Image matching and template detection (<250 lines)
- **screen_analysis.py**: Screen capture and UI analysis (<300 lines)
- **km_visual.py**: KM integration and AppleScript generation (<200 lines)

## 📋 Advanced Visual Capabilities

### OCR Text Extraction
```python
# Example: Extract text from specific screen region
ocr_result = await visual_processor.extract_text_from_region(
    region=ScreenRegion(100, 100, 400, 200),
    language="en",
    confidence_threshold=0.85
)

if ocr_result.is_right():
    text = ocr_result.get_right()
    print(f"Extracted: {text.text} (confidence: {text.confidence})")
```

### Image Template Matching
```python
# Example: Find button in application window
button_matches = await visual_processor.find_image_template(
    template="submit_button.png",
    search_region=ScreenRegion(0, 0, 1920, 1080),
    confidence_threshold=0.9
)

for match in button_matches.get_right():
    if match.found:
        print(f"Button found at: {match.location.to_dict()}")
```

### UI Element Detection
```python
# Example: Analyze window for interactive elements
elements = await visual_processor.analyze_ui_elements(
    region=ScreenRegion(200, 100, 800, 600),
    element_types=["button", "text_field", "menu"]
)

for element in elements.get_right():
    print(f"Found {element.element_type} at {element.location.to_dict()}")
```

### Privacy-Protected Screen Capture
```python
# Example: Capture screen with privacy filtering
screenshot = await visual_processor.capture_screen_region(
    region=ScreenRegion(0, 0, 1920, 1080),
    privacy_filter=True
)

if screenshot.is_right():
    image_data = screenshot.get_right()
    # Process captured image with sensitive content filtered
```

## ✅ Success Criteria
- Complete visual automation implementation with OCR, image recognition, and screen analysis
- Comprehensive privacy protection with sensitive content detection and filtering
- Property-based tests validate behavior across all visual processing scenarios
- Integration with condition system (TASK_21) for intelligent visual workflows
- Performance: <2s OCR processing, <1s image matching, <500ms screen capture
- Documentation: Complete API documentation with privacy considerations and examples
- TESTING.md shows 95%+ test coverage with all security and performance tests passing
- Tool enables AI to leverage Keyboard Maestro's unique visual automation capabilities

## 🔄 Integration Points
- **TASK_21 (km_add_condition)**: Visual conditions for UI state detection
- **TASK_22 (km_control_flow)**: Control flow based on visual analysis results
- **TASK_23 (km_create_trigger_advanced)**: Visual triggers for screen changes
- **All UI Automation Tasks**: Visual capabilities enable advanced UI interaction
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- This unlocks Keyboard Maestro's most powerful and unique automation capabilities
- Essential for automating applications without programmatic APIs
- Privacy and security are critical due to screen recording capabilities
- Must maintain functional programming patterns for testability and composability
- Success here enables sophisticated visual AI workflows that can understand and interact with any macOS application
- Combined with other tasks, creates comprehensive automation platform that works with any software