"""
Hardware event type system for universal interface automation.

This module defines comprehensive types for mouse, keyboard, and gesture events
with security validation and coordinate management for safe hardware interaction.

Security: All events include validation and rate limiting protection.
Performance: Efficient event processing with minimal overhead.
Type Safety: Complete branded type system with contract-driven development.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
from datetime import datetime, timedelta
import re

from src.core.either import Either
from src.core.errors import ValidationError, SecurityError
from src.core.contracts import require, ensure


class MouseButton(Enum):
    """Mouse button types for click operations."""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"
    BUTTON_4 = "button4"
    BUTTON_5 = "button5"


class KeyCode(Enum):
    """Special key codes for keyboard operations."""
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
    F3 = "f3"
    F4 = "f4"
    F5 = "f5"
    F6 = "f6"
    F7 = "f7"
    F8 = "f8"
    F9 = "f9"
    F10 = "f10"
    F11 = "f11"
    F12 = "f12"
    HOME = "home"
    END = "end"
    PAGE_UP = "pageup"
    PAGE_DOWN = "pagedown"


class ModifierKey(Enum):
    """Modifier keys for keyboard combinations."""
    COMMAND = "cmd"
    OPTION = "opt"
    SHIFT = "shift"
    CONTROL = "ctrl"
    FUNCTION = "fn"


class ScrollDirection(Enum):
    """Scroll direction for mouse wheel operations."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class Coordinate:
    """Type-safe screen coordinate with validation."""
    x: int
    y: int
    
    def __post_init__(self):
        """Contract validation for coordinate bounds."""
        if not (0 <= self.x <= 8192):
            raise ValueError(f"X coordinate {self.x} outside valid range (0-8192)")
        if not (0 <= self.y <= 8192):
            raise ValueError(f"Y coordinate {self.y} outside valid range (0-8192)")
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format."""
        return {"x": self.x, "y": self.y}
    
    def distance_to(self, other: 'Coordinate') -> float:
        """Calculate distance to another coordinate."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass(frozen=True)
class MouseEvent:
    """Mouse interaction event specification with validation."""
    operation: str  # click, move, drag, scroll
    position: Coordinate
    button: MouseButton = MouseButton.LEFT
    click_count: int = 1
    duration_ms: int = 100
    event_id: str = field(default_factory=lambda: f"mouse_{datetime.now().timestamp()}")
    
    def __post_init__(self):
        """Contract validation for mouse event parameters."""
        if not (1 <= self.click_count <= 10):
            raise ValueError("Click count must be between 1 and 10")
        if not (10 <= self.duration_ms <= 5000):
            raise ValueError("Duration must be between 10 and 5000 milliseconds")
        
        valid_operations = {"click", "move", "drag", "scroll"}
        if self.operation not in valid_operations:
            raise ValueError(f"Operation must be one of: {valid_operations}")


@dataclass(frozen=True)
class KeyboardEvent:
    """Keyboard interaction event specification with validation."""
    operation: str  # press, type, combination
    key_code: Optional[KeyCode] = None
    text_content: Optional[str] = None
    modifiers: List[ModifierKey] = field(default_factory=list)
    duration_ms: int = 50
    event_id: str = field(default_factory=lambda: f"key_{datetime.now().timestamp()}")
    
    def __post_init__(self):
        """Contract validation for keyboard event parameters."""
        if self.key_code is None and self.text_content is None:
            raise ValueError("Either key_code or text_content must be provided")
        
        if self.text_content is not None and len(self.text_content) > 10000:
            raise ValueError("Text content too long (max 10000 characters)")
        
        if not (10 <= self.duration_ms <= 1000):
            raise ValueError("Duration must be between 10 and 1000 milliseconds")
        
        valid_operations = {"press", "type", "combination"}
        if self.operation not in valid_operations:
            raise ValueError(f"Operation must be one of: {valid_operations}")


@dataclass(frozen=True)
class DragOperation:
    """Drag and drop operation specification with validation."""
    source: Coordinate
    destination: Coordinate
    duration_ms: int = 500
    smooth_movement: bool = True
    button: MouseButton = MouseButton.LEFT
    event_id: str = field(default_factory=lambda: f"drag_{datetime.now().timestamp()}")
    
    def __post_init__(self):
        """Contract validation for drag operation parameters."""
        if not (100 <= self.duration_ms <= 10000):
            raise ValueError("Duration must be between 100 and 10000 milliseconds")
        
        if self.source == self.destination:
            raise ValueError("Source and destination coordinates cannot be the same")
    
    def distance(self) -> float:
        """Calculate drag distance."""
        return self.source.distance_to(self.destination)


@dataclass(frozen=True)
class ScrollEvent:
    """Scroll operation specification with validation."""
    position: Coordinate
    direction: ScrollDirection
    amount: int = 3  # Number of scroll units
    duration_ms: int = 200
    smooth_scroll: bool = True
    event_id: str = field(default_factory=lambda: f"scroll_{datetime.now().timestamp()}")
    
    def __post_init__(self):
        """Contract validation for scroll event parameters."""
        if not (1 <= self.amount <= 20):
            raise ValueError("Scroll amount must be between 1 and 20")
        if not (50 <= self.duration_ms <= 2000):
            raise ValueError("Duration must be between 50 and 2000 milliseconds")


class GestureType(Enum):
    """Types of multi-touch gestures."""
    SWIPE = "swipe"
    PINCH = "pinch"
    ROTATE = "rotate"
    TWO_FINGER_TAP = "two_finger_tap"
    THREE_FINGER_TAP = "three_finger_tap"
    FOUR_FINGER_TAP = "four_finger_tap"


class SwipeDirection(Enum):
    """Direction for swipe gestures."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class GestureEvent:
    """Multi-touch gesture event specification with validation."""
    gesture_type: GestureType
    position: Coordinate
    direction: Optional[SwipeDirection] = None  # For swipes
    scale: Optional[float] = None  # For pinch: 0.5 = zoom out, 2.0 = zoom in
    rotation_degrees: Optional[float] = None  # For rotate: -180 to 180
    finger_count: int = 2
    duration_ms: int = 300
    event_id: str = field(default_factory=lambda: f"gesture_{datetime.now().timestamp()}")
    
    def __post_init__(self):
        """Contract validation for gesture event parameters."""
        if not (2 <= self.finger_count <= 5):
            raise ValueError("Finger count must be between 2 and 5")
        if not (50 <= self.duration_ms <= 2000):
            raise ValueError("Duration must be between 50 and 2000 milliseconds")
        
        # Validate gesture-specific parameters
        if self.gesture_type == GestureType.SWIPE and self.direction is None:
            raise ValueError("Swipe gesture requires valid direction")
        
        if self.gesture_type == GestureType.PINCH and (self.scale is None or not (0.1 <= self.scale <= 10.0)):
            raise ValueError("Pinch gesture requires scale between 0.1 and 10.0")
        
        if self.gesture_type == GestureType.ROTATE and (self.rotation_degrees is None or not (-180 <= self.rotation_degrees <= 180)):
            raise ValueError("Rotate gesture requires degrees between -180 and 180")


class HardwareEventValidator:
    """Security-first validation for hardware events."""
    
    # Dangerous system areas to avoid clicking
    DANGEROUS_AREAS = [
        (0, 0, 100, 50),        # Menu bar corners
        (0, 0, 200, 100),       # Apple menu area
    ]
    
    # Dangerous text patterns to prevent
    DANGEROUS_TEXT_PATTERNS = [
        r'password\s*[:=]\s*\S+',
        r'pass\s*[:=]\s*\S+',
        r'secret\s*[:=]\s*\S+',
        r'token\s*[:=]\s*\S+',
        r'<script',
        r'javascript:',
        r'eval\s*\(',
        r'exec\s*\(',
        r'rm\s+-rf',
        r'sudo\s+',
    ]
    
    @staticmethod
    def validate_coordinate_safety(coord: Coordinate) -> Either[SecurityError, None]:
        """Validate coordinate is safe for interaction."""
        # Check dangerous system areas
        for dx, dy, dw, dh in HardwareEventValidator.DANGEROUS_AREAS:
            if dx <= coord.x <= dx + dw and dy <= coord.y <= dy + dh:
                return Either.left(SecurityError(
                    "DANGEROUS_COORDINATE",
                    f"Coordinate ({coord.x}, {coord.y}) is in dangerous system area"
                ))
        
        return Either.right(None)
    
    @staticmethod
    def validate_text_safety(text: str) -> Either[SecurityError, None]:
        """Validate text content for security threats."""
        if not text:
            return Either.right(None)
        
        text_lower = text.lower()
        
        # Check for dangerous patterns
        for pattern in HardwareEventValidator.DANGEROUS_TEXT_PATTERNS:
            if re.search(pattern, text_lower):
                return Either.left(SecurityError(
                    "DANGEROUS_TEXT_PATTERN",
                    f"Text contains dangerous pattern: {pattern}"
                ))
        
        # Check for control characters
        dangerous_chars = ['\x1b', '\x00', '\x7f']  # ESC, NULL, DEL
        for char in dangerous_chars:
            if char in text:
                return Either.left(SecurityError(
                    "DANGEROUS_CONTROL_CHAR",
                    f"Text contains dangerous control character: {repr(char)}"
                ))
        
        return Either.right(None)
    
    @staticmethod
    def validate_key_combination(keys: List[str]) -> Either[SecurityError, None]:
        """Validate key combination is safe and valid."""
        if len(keys) > 10:
            return Either.left(SecurityError(
                "TOO_MANY_KEYS",
                "Key combination has too many keys (max 10)"
            ))
        
        # Valid key set
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
                return Either.left(SecurityError(
                    "INVALID_KEY",
                    f"Invalid key in combination: {key}"
                ))
        
        return Either.right(None)
    
    @staticmethod
    def validate_drag_distance(drag_op: DragOperation) -> Either[SecurityError, None]:
        """Validate drag operation distance and destination."""
        distance = drag_op.distance()
        
        # Prevent extremely long drags that might be malicious
        if distance > 3000:  # pixels
            return Either.left(SecurityError(
                "DRAG_TOO_LONG",
                f"Drag distance {distance:.1f} pixels exceeds safe limit (3000)"
            ))
        
        # Validate both coordinates
        for coord in [drag_op.source, drag_op.destination]:
            coord_result = HardwareEventValidator.validate_coordinate_safety(coord)
            if coord_result.is_left():
                return coord_result
        
        return Either.right(None)


class RateLimiter:
    """Rate limiting for hardware events to prevent abuse."""
    
    def __init__(self):
        self._event_history: List[Tuple[datetime, str]] = []
    
    def check_rate_limit(self, operation: str) -> Either[SecurityError, None]:
        """Check if operation is within rate limits."""
        current_time = datetime.now()
        
        # Clean old events (older than 1 minute)
        self._event_history = [
            (time, op) for time, op in self._event_history
            if current_time - time < timedelta(minutes=1)
        ]
        
        # Count recent events by type
        recent_mouse_events = len([
            op for time, op in self._event_history
            if current_time - time < timedelta(seconds=10) and op.startswith("mouse")
        ])
        
        recent_keyboard_events = len([
            op for time, op in self._event_history
            if current_time - time < timedelta(seconds=10) and op.startswith("key")
        ])
        
        # Rate limits by operation type
        if operation.startswith("mouse") and recent_mouse_events >= 50:
            return Either.left(SecurityError(
                "MOUSE_RATE_LIMIT",
                "Mouse event rate limit exceeded (50 per 10 seconds)"
            ))
        
        if operation.startswith("key") and recent_keyboard_events >= 100:
            return Either.left(SecurityError(
                "KEYBOARD_RATE_LIMIT",
                "Keyboard event rate limit exceeded (100 per 10 seconds)"
            ))
        
        # Add current event
        self._event_history.append((current_time, operation))
        
        return Either.right(None)


# Utility functions for screen dimensions and hardware info
def get_screen_dimensions() -> Tuple[int, int]:
    """Get main screen dimensions (width, height)."""
    # This would use actual macOS APIs in production
    # For now, return common screen dimensions
    return (1920, 1080)


def get_all_screen_info() -> List[Dict[str, Any]]:
    """Get information about all connected screens."""
    # This would use actual macOS APIs in production
    return [
        {
            "id": 0,
            "name": "Main Display",
            "width": 1920,
            "height": 1080,
            "origin_x": 0,
            "origin_y": 0,
            "is_main": True
        }
    ]


# Helper functions for common operations
def create_mouse_click(x: int, y: int, button: MouseButton = MouseButton.LEFT, count: int = 1) -> MouseEvent:
    """Create a validated mouse click event."""
    return MouseEvent(
        operation="click",
        position=Coordinate(x, y),
        button=button,
        click_count=count
    )


def create_text_input(text: str) -> KeyboardEvent:
    """Create a validated text input event."""
    return KeyboardEvent(
        operation="type",
        text_content=text
    )


def create_key_combination(keys: List[str]) -> KeyboardEvent:
    """Create a validated key combination event."""
    # Convert string keys to modifiers where appropriate
    modifiers = []
    regular_keys = []
    
    for key in keys:
        try:
            modifier = ModifierKey(key.lower())
            modifiers.append(modifier)
        except ValueError:
            regular_keys.append(key)
    
    # For combinations, we use the first non-modifier key as the main key
    main_key = regular_keys[0] if regular_keys else None
    
    return KeyboardEvent(
        operation="combination",
        key_code=KeyCode(main_key) if main_key and main_key in [k.value for k in KeyCode] else None,
        text_content=main_key if main_key and main_key not in [k.value for k in KeyCode] else None,
        modifiers=modifiers
    )


def create_drag_drop(source_x: int, source_y: int, dest_x: int, dest_y: int) -> DragOperation:
    """Create a validated drag and drop operation."""
    return DragOperation(
        source=Coordinate(source_x, source_y),
        destination=Coordinate(dest_x, dest_y)
    )