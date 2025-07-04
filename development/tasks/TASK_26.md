# TASK_26: km_interface_automation - Mouse, Keyboard, Drag/Drop Operations

**Created By**: Agent_ADDER+ (Protocol Gap Analysis) | **Priority**: MEDIUM | **Duration**: 5 hours
**Technique Focus**: Hardware Interaction + Accessibility APIs + Event Simulation + Security Validation
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED ✅
**Assigned**: Agent_4
**Dependencies**: TASK_24 (visual automation), TASK_16 (window manager)
**Blocking**: Advanced UI automation and application-independent interaction workflows

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Interface automation specification
- [ ] **KM Documentation**: development/protocols/KM_MCP.md - Mouse and keyboard action types
- [ ] **Visual Integration**: development/tasks/TASK_24.md - Visual automation for coordinate detection
- [ ] **macOS APIs**: Core Graphics, Accessibility APIs for event generation
- [ ] **Testing Framework**: tests/TESTING.md - Hardware interaction testing requirements

## 🎯 Problem Analysis
**Classification**: Missing Critical Functionality
**Gap Identified**: No direct hardware interaction capabilities (mouse, keyboard, drag/drop)
**Impact**: AI cannot perform basic UI interaction that doesn't rely on application-specific APIs

<thinking>
Root Cause Analysis:
1. Current implementation relies on application-specific automation (AppleScript, app control)
2. Missing fundamental hardware event simulation for universal UI interaction
3. No mouse interaction capabilities for clicking, dragging, scrolling
4. No keyboard simulation for typing, shortcuts, key combinations
5. Limited to applications that provide programmatic APIs
6. Cannot interact with legacy applications, games, or UI elements without accessibility support
7. Essential for universal automation that works with any application
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design ✅
- [x] **Input type system**: Define branded types for mouse, keyboard, and gesture events ✅
- [x] **Coordinate validation**: Screen boundary checking and multi-monitor support ✅
- [x] **Security framework**: Event rate limiting and malicious input prevention ✅

### Phase 2: Mouse Interaction ✅
- [x] **Basic mouse operations**: Click, double-click, right-click, middle-click ✅
- [x] **Mouse movement**: Smooth movement, instant positioning, relative movement ✅
- [x] **Drag and drop**: Complex drag operations with source/target validation ✅
- [x] **Scroll operations**: Wheel scrolling, trackpad gestures, smooth scrolling ✅

### Phase 3: Keyboard Simulation ✅
- [x] **Text input**: Secure text typing with proper character encoding ✅
- [x] **Key combinations**: Modifier key handling and complex shortcuts ✅
- [x] **Special keys**: Function keys, arrow keys, control sequences ✅
- [x] **Input methods**: Support for different keyboard layouts and input sources ✅

### Phase 4: Advanced Interactions ✅
- [x] **Gesture simulation**: Multi-touch gestures, trackpad operations ✅
- [x] **Timing control**: Precise timing, delays, and event sequencing ✅
- [x] **Accessibility integration**: Work with screen readers and accessibility tools ✅
- [x] **Visual coordinate integration**: Combine with TASK_24 for intelligent targeting ✅

### Phase 5: Integration & Security ✅
- [x] **AppleScript generation**: Safe hardware event XML generation ✅
- [x] **Rate limiting**: Prevent automation abuse and system overload ✅
- [x] **Property-based tests**: Hypothesis validation for all interaction types ✅
- [x] **TESTING.md update**: Hardware interaction test coverage and security validation ✅

## 🔧 Implementation Files & Specifications
```
src/server/tools/interface_automation_tools.py  # Main interface automation tool
src/core/hardware_events.py                    # Hardware event type definitions
src/interaction/mouse_controller.py            # Mouse operation implementation
src/interaction/keyboard_controller.py         # Keyboard operation implementation
src/interaction/gesture_controller.py          # Gesture and drag operation implementation
src/security/input_validator.py                # Input security and rate limiting
tests/tools/test_interface_automation_tools.py # Unit and integration tests
tests/property_tests/test_hardware_events.py   # Property-based interaction validation
```

### km_interface_automation Tool Specification
```python
@mcp.tool()
async def km_interface_automation(
    operation: str,                          # mouse_click|mouse_move|key_press|type_text|drag_drop|scroll
    coordinates: Optional[Dict[str, int]] = None,  # Target coordinates {x, y}
    text_content: Optional[str] = None,      # Text to type
    key_combination: Optional[List[str]] = None,  # Key combination to press
    button: str = "left",                    # Mouse button (left, right, middle)
    click_count: int = 1,                    # Number of clicks (1, 2, 3)
    drag_destination: Optional[Dict[str, int]] = None,  # Drag target coordinates
    modifier_keys: Optional[List[str]] = None,  # Modifier keys (cmd, opt, shift, ctrl)
    duration_ms: int = 100,                  # Action duration in milliseconds
    smooth_movement: bool = True,            # Use smooth mouse movement
    delay_between_events: int = 50,          # Delay between multiple events
    validate_coordinates: bool = True,       # Validate coordinates are on screen
    ctx = None
) -> Dict[str, Any]:
```

### Hardware Event Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple
from enum import Enum

class MouseButton(Enum):
    """Mouse button types."""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"
    BUTTON_4 = "button4"
    BUTTON_5 = "button5"

class KeyCode(Enum):
    """Special key codes."""
    ENTER = "enter"
    RETURN = "return"
    TAB = "tab"
    SPACE = "space"
    ESCAPE = "escape"
    DELETE = "delete"
    BACKSPACE = "backspace"
    ARROW_UP = "up"
    ARROW_DOWN = "down"
    ARROW_LEFT = "left"
    ARROW_RIGHT = "right"
    F1 = "f1"
    F2 = "f2"
    # ... F3-F12
    HOME = "home"
    END = "end"
    PAGE_UP = "pageup"
    PAGE_DOWN = "pagedown"

class ModifierKey(Enum):
    """Modifier keys."""
    COMMAND = "cmd"
    OPTION = "opt"
    SHIFT = "shift"
    CONTROL = "ctrl"
    FUNCTION = "fn"

@dataclass(frozen=True)
class Coordinate:
    """Type-safe screen coordinate."""
    x: int
    y: int
    
    @require(lambda self: self.x >= 0 and self.y >= 0)
    @require(lambda self: self.x <= 8192 and self.y <= 8192)  # Reasonable screen limits
    def __post_init__(self):
        pass
    
    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y}

@dataclass(frozen=True)
class MouseEvent:
    """Mouse interaction event specification."""
    operation: str  # click, move, drag, scroll
    position: Coordinate
    button: MouseButton = MouseButton.LEFT
    click_count: int = 1
    duration_ms: int = 100
    
    @require(lambda self: 1 <= self.click_count <= 10)
    @require(lambda self: 10 <= self.duration_ms <= 5000)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class KeyboardEvent:
    """Keyboard interaction event specification."""
    operation: str  # press, type, combination
    key_code: Optional[KeyCode] = None
    text_content: Optional[str] = None
    modifiers: List[ModifierKey] = field(default_factory=list)
    duration_ms: int = 50
    
    @require(lambda self: self.key_code is not None or self.text_content is not None)
    @require(lambda self: self.text_content is None or len(self.text_content) <= 10000)
    @require(lambda self: 10 <= self.duration_ms <= 1000)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class DragOperation:
    """Drag and drop operation specification."""
    source: Coordinate
    destination: Coordinate
    duration_ms: int = 500
    smooth_movement: bool = True
    
    @require(lambda self: 100 <= self.duration_ms <= 10000)
    def __post_init__(self):
        pass

class MouseController:
    """Hardware mouse control with security validation."""
    
    @require(lambda coord: coord.x >= 0 and coord.y >= 0)
    @ensure(lambda result: result.is_right() or result.get_left().is_hardware_error())
    async def click_at_position(
        self,
        position: Coordinate,
        button: MouseButton = MouseButton.LEFT,
        click_count: int = 1,
        duration_ms: int = 100
    ) -> Either[HardwareError, Dict[str, Any]]:
        """Perform mouse click at specified position."""
        # Validate position is on screen
        validation_result = self._validate_screen_position(position)
        if validation_result.is_left():
            return Either.left(validation_result.get_left())
        
        # Rate limit check
        rate_limit_result = await self._check_rate_limit()
        if rate_limit_result.is_left():
            return Either.left(rate_limit_result.get_left())
        
        # Execute mouse click via Core Graphics
        return await self._execute_mouse_click(position, button, click_count, duration_ms)
    
    @require(lambda source, dest: source != dest)
    @ensure(lambda result: result.is_right() or result.get_left().is_hardware_error())
    async def drag_and_drop(
        self,
        source: Coordinate,
        destination: Coordinate,
        duration_ms: int = 500,
        smooth_movement: bool = True
    ) -> Either[HardwareError, Dict[str, Any]]:
        """Perform drag and drop operation."""
        # Validate both positions
        for position in [source, destination]:
            validation_result = self._validate_screen_position(position)
            if validation_result.is_left():
                return Either.left(validation_result.get_left())
        
        # Execute drag operation
        return await self._execute_drag_operation(source, destination, duration_ms, smooth_movement)
    
    async def move_to_position(
        self,
        position: Coordinate,
        duration_ms: int = 200,
        smooth_movement: bool = True
    ) -> Either[HardwareError, Dict[str, Any]]:
        """Move mouse cursor to specified position."""
        validation_result = self._validate_screen_position(position)
        if validation_result.is_left():
            return Either.left(validation_result.get_left())
        
        return await self._execute_mouse_movement(position, duration_ms, smooth_movement)
    
    def _validate_screen_position(self, position: Coordinate) -> Either[HardwareError, None]:
        """Validate position is within screen bounds."""
        # Get screen dimensions (would use actual macOS APIs)
        screen_width, screen_height = get_main_screen_dimensions()
        
        if position.x < 0 or position.y < 0:
            return Either.left(HardwareError("Negative coordinates not allowed"))
        
        if position.x >= screen_width or position.y >= screen_height:
            return Either.left(HardwareError("Position outside screen bounds"))
        
        return Either.right(None)

class KeyboardController:
    """Hardware keyboard control with security validation."""
    
    @require(lambda text: len(text) <= 10000)
    @ensure(lambda result: result.is_right() or result.get_left().is_hardware_error())
    async def type_text(
        self,
        text: str,
        delay_between_chars: int = 50
    ) -> Either[HardwareError, Dict[str, Any]]:
        """Type text with character-by-character timing."""
        # Validate text content
        validation_result = self._validate_text_content(text)
        if validation_result.is_left():
            return Either.left(validation_result.get_left())
        
        # Rate limit check
        rate_limit_result = await self._check_rate_limit()
        if rate_limit_result.is_left():
            return Either.left(rate_limit_result.get_left())
        
        # Execute text typing
        return await self._execute_text_typing(text, delay_between_chars)
    
    @require(lambda keys: len(keys) <= 10)
    @ensure(lambda result: result.is_right() or result.get_left().is_hardware_error())
    async def press_key_combination(
        self,
        keys: List[str],
        duration_ms: int = 100
    ) -> Either[HardwareError, Dict[str, Any]]:
        """Press key combination with proper modifier handling."""
        # Validate key combination
        validation_result = self._validate_key_combination(keys)
        if validation_result.is_left():
            return Either.left(validation_result.get_left())
        
        # Execute key combination
        return await self._execute_key_combination(keys, duration_ms)
    
    def _validate_text_content(self, text: str) -> Either[HardwareError, None]:
        """Validate text content for security."""
        # Check for dangerous control sequences
        dangerous_sequences = ['\x1b', '\x00', '\x7f']  # ESC, NULL, DEL
        
        for sequence in dangerous_sequences:
            if sequence in text:
                return Either.left(HardwareError("Dangerous control sequence in text"))
        
        # Check for excessive length
        if len(text) > 10000:
            return Either.left(HardwareError("Text too long"))
        
        return Either.right(None)
    
    def _validate_key_combination(self, keys: List[str]) -> Either[HardwareError, None]:
        """Validate key combination is safe and valid."""
        if len(keys) > 10:
            return Either.left(HardwareError("Too many keys in combination"))
        
        # Validate each key
        valid_keys = {
            # Modifier keys
            "cmd", "command", "opt", "option", "shift", "ctrl", "control", "fn",
            # Letter keys
            *[chr(i) for i in range(ord('a'), ord('z') + 1)],
            # Number keys
            *[str(i) for i in range(10)],
            # Function keys
            *[f"f{i}" for i in range(1, 13)],
            # Special keys
            "space", "enter", "return", "tab", "escape", "delete", "backspace",
            "up", "down", "left", "right", "home", "end", "pageup", "pagedown"
        }
        
        for key in keys:
            if key.lower() not in valid_keys:
                return Either.left(HardwareError(f"Invalid key: {key}"))
        
        return Either.right(None)
```

## 🔒 Security Implementation
```python
class HardwareSecurityManager:
    """Security-first hardware interaction with rate limiting."""
    
    def __init__(self):
        self._rate_limiter = RateLimiter()
        self._event_history: List[Tuple[datetime, str]] = []
    
    async def check_rate_limit(self, operation: str) -> Either[SecurityError, None]:
        """Check if operation is within rate limits."""
        current_time = datetime.now()
        
        # Clean old events (older than 1 minute)
        self._event_history = [
            (time, op) for time, op in self._event_history
            if current_time - time < timedelta(minutes=1)
        ]
        
        # Count recent events
        recent_events = len([
            op for time, op in self._event_history
            if current_time - time < timedelta(seconds=10)
        ])
        
        # Rate limit: max 100 events per 10 seconds
        if recent_events >= 100:
            return Either.left(SecurityError("Rate limit exceeded"))
        
        # Add current event
        self._event_history.append((current_time, operation))
        
        return Either.right(None)
    
    @staticmethod
    def validate_coordinate_safety(coord: Coordinate) -> Either[SecurityError, None]:
        """Validate coordinate is safe for interaction."""
        # Prevent clicks on dangerous system areas
        dangerous_areas = [
            (0, 0, 100, 50),        # Menu bar corners
            (0, 0, 200, 100),       # Apple menu area
        ]
        
        for dx, dy, dw, dh in dangerous_areas:
            if dx <= coord.x <= dx + dw and dy <= coord.y <= dy + dh:
                return Either.left(SecurityError("Coordinate in dangerous system area"))
        
        return Either.right(None)
    
    @staticmethod
    def validate_text_safety(text: str) -> Either[SecurityError, None]:
        """Validate text content for security."""
        # Check for password-like patterns
        password_patterns = [
            r'password\s*[:=]\s*\S+',
            r'pass\s*[:=]\s*\S+',
            r'secret\s*[:=]\s*\S+',
            r'token\s*[:=]\s*\S+'
        ]
        
        text_lower = text.lower()
        for pattern in password_patterns:
            if re.search(pattern, text_lower):
                return Either.left(SecurityError("Text contains password-like pattern"))
        
        # Check for script injection
        script_patterns = [
            r'<script',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\('
        ]
        
        for pattern in script_patterns:
            if re.search(pattern, text_lower):
                return Either.left(SecurityError("Text contains script injection pattern"))
        
        return Either.right(None)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=1920), st.integers(min_value=0, max_value=1080))
def test_coordinate_validation_properties(x, y):
    """Property: Coordinate validation should handle all screen positions."""
    coord = Coordinate(x, y)
    controller = MouseController()
    
    result = controller._validate_screen_position(coord)
    if x < 1920 and y < 1080:
        assert result.is_right()
    else:
        assert result.is_left()

@given(st.text(min_size=1, max_size=100))
def test_text_typing_security_properties(text):
    """Property: Text typing should validate all input safely."""
    controller = KeyboardController()
    validation = controller._validate_text_content(text)
    
    # Should either pass validation or fail with security error
    assert validation.is_right() or validation.get_left().code == "SECURITY_ERROR"

@given(st.lists(st.sampled_from(["cmd", "shift", "a", "1", "f1"]), min_size=1, max_size=5))
def test_key_combination_properties(keys):
    """Property: Key combinations should be validated properly."""
    controller = KeyboardController()
    validation = controller._validate_key_combination(keys)
    
    # All valid keys should pass
    assert validation.is_right()
```

## 🏗️ Modularity Strategy
- **interface_automation_tools.py**: Main MCP tool interface (<250 lines)
- **hardware_events.py**: Type definitions and validation (<200 lines)
- **mouse_controller.py**: Mouse interaction implementation (<300 lines)
- **keyboard_controller.py**: Keyboard interaction implementation (<300 lines)
- **gesture_controller.py**: Advanced gesture and drag operations (<250 lines)
- **input_validator.py**: Security validation and rate limiting (<200 lines)

## 📋 Advanced Interface Automation Examples

### Precise Mouse Operations
```python
# Example: Click specific UI element with validation
result = await interface_automation.click_at_position(
    position=Coordinate(500, 300),
    button=MouseButton.LEFT,
    click_count=2,  # Double-click
    duration_ms=150
)
```

### Complex Drag Operations
```python
# Example: Drag file from source to destination
result = await interface_automation.drag_and_drop(
    source=Coordinate(100, 200),
    destination=Coordinate(800, 400),
    duration_ms=1000,
    smooth_movement=True
)
```

### Secure Text Input
```python
# Example: Type text with security validation
result = await interface_automation.type_text(
    text="Hello, World!",
    delay_between_chars=50  # Natural typing speed
)
```

### Keyboard Combinations
```python
# Example: Complex keyboard shortcut
result = await interface_automation.press_key_combination(
    keys=["cmd", "shift", "4"],  # Screenshot shortcut
    duration_ms=100
)
```

## ✅ Success Criteria
- Complete hardware interaction implementation with mouse, keyboard, and gesture support
- Comprehensive security validation with rate limiting and dangerous input prevention
- Property-based tests validate behavior across all interaction scenarios and edge cases
- Integration with visual automation (TASK_24) for intelligent coordinate targeting
- Performance: <50ms simple operations, <500ms complex drag operations
- Documentation: Complete API documentation with security considerations and examples
- TESTING.md shows 95%+ test coverage with all security and hardware tests passing
- Tool enables universal UI automation that works with any application regardless of API support

## 🔄 Integration Points
- **TASK_24 (km_visual_automation)**: Visual coordinate detection for intelligent targeting
- **TASK_16 (km_window_manager)**: Window positioning combined with interface interaction
- **TASK_21/22 (conditions/control_flow)**: Conditional interface automation workflows
- **All Future UI Tasks**: Universal interaction capabilities for any application
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- This completes the universal automation foundation - no application-specific APIs required
- Essential for automating legacy applications, games, or any UI without accessibility support
- Security is critical - hardware events can be used maliciously if not properly controlled
- Must maintain functional programming patterns for testability and composability
- Success here enables AI to interact with any macOS application through visual + hardware automation
- Combined with visual automation (TASK_24), creates complete computer vision + interaction platform