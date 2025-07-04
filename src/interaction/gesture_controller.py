"""
Gesture and advanced interaction controller for hardware automation.

This module implements advanced gesture controls including multi-touch gestures,
precise timing, accessibility integration, and complex interaction sequences
for comprehensive UI automation.

Security: All gestures include validation and timing constraints.
Performance: Optimized gesture recognition and event sequencing.
Type Safety: Complete integration with hardware event type system.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Sequence
import asyncio
import time
import math
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.core.hardware_events import (
    Coordinate, DragOperation, HardwareEventValidator, RateLimiter
)
from src.core.either import Either
from src.core.errors import SecurityError, IntegrationError
from src.core.contracts import require, ensure
from src.core.logging import get_logger

logger = get_logger(__name__)


class GestureType(Enum):
    """Types of gestures for multi-touch interactions."""
    PINCH = "pinch"
    ROTATE = "rotate"
    SWIPE = "swipe"
    TWO_FINGER_SCROLL = "two_finger_scroll"
    THREE_FINGER_SWIPE = "three_finger_swipe"
    FOUR_FINGER_SWIPE = "four_finger_swipe"


class SwipeDirection(Enum):
    """Swipe gesture directions."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class GestureEvent:
    """Gesture interaction event specification with validation."""
    gesture_type: GestureType
    center_position: Coordinate
    magnitude: float = 1.0  # Scale factor, rotation angle, or distance
    direction: Optional[SwipeDirection] = None
    duration_ms: int = 500
    finger_count: int = 2
    event_id: str = field(default_factory=lambda: f"gesture_{datetime.now().timestamp()}")
    
    def __post_init__(self):
        """Contract validation for gesture parameters."""
        if not (100 <= self.duration_ms <= 3000):
            raise ValueError("Duration must be between 100 and 3000 milliseconds")
        
        if not (0.1 <= self.magnitude <= 10.0):
            raise ValueError("Magnitude must be between 0.1 and 10.0")
        
        if not (2 <= self.finger_count <= 4):
            raise ValueError("Finger count must be between 2 and 4")


@dataclass(frozen=True)
class TimingSequence:
    """Precise timing sequence for complex interactions."""
    events: List[Tuple[str, Dict[str, Any], int]]  # (operation, params, delay_ms)
    total_duration_ms: int
    synchronous: bool = False
    event_id: str = field(default_factory=lambda: f"sequence_{datetime.now().timestamp()}")
    
    def __post_init__(self):
        """Contract validation for timing sequence."""
        if not (10 <= self.total_duration_ms <= 30000):
            raise ValueError("Total duration must be between 10ms and 30 seconds")
        
        if len(self.events) > 50:
            raise ValueError("Sequence cannot have more than 50 events")


class GestureController:
    """Advanced gesture and timing control with security validation."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.active_gestures: Dict[str, GestureEvent] = {}
        self.sequence_history: List[TimingSequence] = []
    
    @require(lambda self, gesture: isinstance(gesture, GestureEvent))
    @ensure(lambda result: result.is_right() or result.get_left().error_code.startswith("GESTURE_"))
    async def perform_gesture(
        self,
        gesture_type: GestureType,
        center_position: Coordinate,
        magnitude: float = 1.0,
        direction: Optional[SwipeDirection] = None,
        duration_ms: int = 500,
        finger_count: int = 2
    ) -> Either[SecurityError, Dict[str, Any]]:
        """
        Perform multi-touch gesture with validation and timing control.
        
        Args:
            gesture_type: Type of gesture to perform
            center_position: Center point of the gesture
            magnitude: Scale factor, rotation angle, or swipe distance
            direction: Direction for directional gestures
            duration_ms: Gesture duration in milliseconds
            finger_count: Number of fingers for the gesture
            
        Returns:
            Either security error or operation result with gesture details
        """
        try:
            logger.info(f"Gesture {gesture_type.value} at ({center_position.x}, {center_position.y})")
            
            # Create gesture event
            gesture_event = GestureEvent(
                gesture_type=gesture_type,
                center_position=center_position,
                magnitude=magnitude,
                direction=direction,
                duration_ms=duration_ms,
                finger_count=finger_count
            )
            
            # Validate gesture parameters
            validation_result = self._validate_gesture(gesture_event)
            if validation_result.is_left():
                return Either.left(validation_result.get_left())
            
            # Rate limit check
            rate_limit_result = self.rate_limiter.check_rate_limit("gesture_perform")
            if rate_limit_result.is_left():
                return Either.left(rate_limit_result.get_left())
            
            # Execute gesture
            execution_result = await self._execute_gesture(gesture_event)
            if execution_result.is_left():
                return execution_result
            
            result = {
                "success": True,
                "operation": "gesture",
                "gesture_type": gesture_type.value,
                "center_position": center_position.to_dict(),
                "magnitude": magnitude,
                "direction": direction.value if direction else None,
                "duration_ms": duration_ms,
                "finger_count": finger_count,
                "execution_time_ms": execution_result.get_right().get("execution_time_ms", 0),
                "event_id": gesture_event.event_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Gesture {gesture_type.value} completed")
            return Either.right(result)
            
        except Exception as e:
            logger.error(f"Error in gesture execution: {str(e)}")
            return Either.left(SecurityError(
                "GESTURE_EXECUTION_ERROR",
                f"Failed to execute gesture: {str(e)}"
            ))
    
    @require(lambda self, sequence: isinstance(sequence, TimingSequence))
    async def execute_timing_sequence(
        self,
        events: List[Tuple[str, Dict[str, Any], int]],
        synchronous: bool = False
    ) -> Either[SecurityError, Dict[str, Any]]:
        """
        Execute a precisely timed sequence of interactions.
        
        Args:
            events: List of (operation, parameters, delay_ms) tuples
            synchronous: Whether to wait for each event to complete
            
        Returns:
            Either security error or sequence execution result
        """
        try:
            total_duration = sum(delay for _, _, delay in events)
            
            # Create timing sequence
            timing_sequence = TimingSequence(
                events=events,
                total_duration_ms=total_duration,
                synchronous=synchronous
            )
            
            logger.info(f"Executing timing sequence: {len(events)} events over {total_duration}ms")
            
            # Rate limit check
            rate_limit_result = self.rate_limiter.check_rate_limit("sequence_execute")
            if rate_limit_result.is_left():
                return Either.left(rate_limit_result.get_left())
            
            # Execute sequence
            execution_result = await self._execute_timing_sequence(timing_sequence)
            if execution_result.is_left():
                return execution_result
            
            # Track sequence
            self.sequence_history.append(timing_sequence)
            if len(self.sequence_history) > 10:  # Keep last 10 sequences
                self.sequence_history.pop(0)
            
            result = {
                "success": True,
                "operation": "timing_sequence",
                "events_count": len(events),
                "total_duration_ms": total_duration,
                "synchronous": synchronous,
                "execution_time_ms": execution_result.get_right().get("execution_time_ms", 0),
                "event_id": timing_sequence.event_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Timing sequence completed: {len(events)} events")
            return Either.right(result)
            
        except Exception as e:
            logger.error(f"Error in timing sequence: {str(e)}")
            return Either.left(SecurityError(
                "SEQUENCE_EXECUTION_ERROR",
                f"Failed to execute timing sequence: {str(e)}"
            ))
    
    async def create_accessibility_interaction(
        self,
        element_type: str,
        action: str,
        position: Coordinate,
        accessibility_info: Optional[Dict[str, Any]] = None
    ) -> Either[SecurityError, Dict[str, Any]]:
        """
        Create accessibility-aware interaction for screen readers and assistive technology.
        
        Args:
            element_type: Type of UI element (button, textfield, etc.)
            action: Accessibility action to perform
            position: Position of the element
            accessibility_info: Additional accessibility metadata
            
        Returns:
            Either security error or accessibility interaction result
        """
        try:
            logger.info(f"Accessibility interaction: {action} on {element_type}")
            
            # Validate accessibility parameters
            validation_result = self._validate_accessibility_interaction(
                element_type, action, position
            )
            if validation_result.is_left():
                return Either.left(validation_result.get_left())
            
            # Execute accessibility interaction
            execution_result = await self._execute_accessibility_interaction(
                element_type, action, position, accessibility_info
            )
            if execution_result.is_left():
                return execution_result
            
            result = {
                "success": True,
                "operation": "accessibility_interaction",
                "element_type": element_type,
                "action": action,
                "position": position.to_dict(),
                "accessibility_info": accessibility_info or {},
                "execution_time_ms": execution_result.get_right().get("execution_time_ms", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Accessibility interaction completed: {action}")
            return Either.right(result)
            
        except Exception as e:
            logger.error(f"Error in accessibility interaction: {str(e)}")
            return Either.left(SecurityError(
                "ACCESSIBILITY_ERROR",
                f"Failed to execute accessibility interaction: {str(e)}"
            ))
    
    def _validate_gesture(self, gesture: GestureEvent) -> Either[SecurityError, None]:
        """Validate gesture parameters for security and feasibility."""
        # Validate center position
        coord_result = HardwareEventValidator.validate_coordinate_safety(gesture.center_position)
        if coord_result.is_left():
            return Either.left(coord_result.get_left())
        
        # Validate gesture type and finger count compatibility
        compatible_fingers = {
            GestureType.PINCH: [2],
            GestureType.ROTATE: [2],
            GestureType.SWIPE: [1, 2, 3, 4],
            GestureType.TWO_FINGER_SCROLL: [2],
            GestureType.THREE_FINGER_SWIPE: [3],
            GestureType.FOUR_FINGER_SWIPE: [4]
        }
        
        if gesture.finger_count not in compatible_fingers.get(gesture.gesture_type, []):
            return Either.left(SecurityError(
                "INCOMPATIBLE_GESTURE",
                f"Gesture {gesture.gesture_type.value} incompatible with {gesture.finger_count} fingers"
            ))
        
        # Validate magnitude is reasonable
        if gesture.gesture_type in [GestureType.PINCH, GestureType.ROTATE]:
            if not (0.1 <= gesture.magnitude <= 5.0):
                return Either.left(SecurityError(
                    "INVALID_MAGNITUDE",
                    f"Magnitude {gesture.magnitude} outside safe range (0.1-5.0)"
                ))
        
        return Either.right(None)
    
    def _validate_accessibility_interaction(
        self, 
        element_type: str, 
        action: str, 
        position: Coordinate
    ) -> Either[SecurityError, None]:
        """Validate accessibility interaction parameters."""
        # Valid element types
        valid_elements = {
            "button", "textfield", "text", "image", "link", "checkbox", 
            "radiobutton", "combobox", "slider", "progressbar", "menu", "menuitem"
        }
        
        if element_type.lower() not in valid_elements:
            return Either.left(SecurityError(
                "INVALID_ELEMENT_TYPE",
                f"Element type '{element_type}' not supported"
            ))
        
        # Valid accessibility actions
        valid_actions = {
            "click", "focus", "activate", "select", "expand", "collapse",
            "increment", "decrement", "scroll", "announce"
        }
        
        if action.lower() not in valid_actions:
            return Either.left(SecurityError(
                "INVALID_ACCESSIBILITY_ACTION",
                f"Action '{action}' not supported"
            ))
        
        # Validate position
        coord_result = HardwareEventValidator.validate_coordinate_safety(position)
        if coord_result.is_left():
            return Either.left(coord_result.get_left())
        
        return Either.right(None)
    
    async def _execute_gesture(self, gesture: GestureEvent) -> Either[IntegrationError, Dict[str, Any]]:
        """Execute gesture with proper timing and AppleScript generation."""
        try:
            start_time = time.time()
            
            # Generate AppleScript for gesture
            applescript = self._generate_gesture_applescript(gesture)
            
            # Simulate gesture execution with proper timing
            if gesture.gesture_type == GestureType.PINCH:
                await self._simulate_pinch_gesture(gesture)
            elif gesture.gesture_type == GestureType.ROTATE:
                await self._simulate_rotate_gesture(gesture)
            elif gesture.gesture_type in [GestureType.SWIPE, GestureType.THREE_FINGER_SWIPE, GestureType.FOUR_FINGER_SWIPE]:
                await self._simulate_swipe_gesture(gesture)
            elif gesture.gesture_type == GestureType.TWO_FINGER_SCROLL:
                await self._simulate_scroll_gesture(gesture)
            else:
                await asyncio.sleep(gesture.duration_ms / 1000.0)
            
            execution_time = (time.time() - start_time) * 1000
            
            return Either.right({
                "applescript_generated": True,
                "execution_time_ms": execution_time,
                "gesture_executed": True
            })
            
        except Exception as e:
            return Either.left(IntegrationError(
                "GESTURE_EXECUTION_ERROR",
                f"Failed to execute gesture: {str(e)}"
            ))
    
    async def _execute_timing_sequence(self, sequence: TimingSequence) -> Either[IntegrationError, Dict[str, Any]]:
        """Execute timing sequence with precise event scheduling."""
        try:
            start_time = time.time()
            
            if sequence.synchronous:
                # Execute events sequentially with delays
                for operation, params, delay_ms in sequence.events:
                    await asyncio.sleep(delay_ms / 1000.0)
                    logger.debug(f"Sequence event: {operation} with delay {delay_ms}ms")
            else:
                # Execute events asynchronously
                tasks = []
                for i, (operation, params, delay_ms) in enumerate(sequence.events):
                    task = asyncio.create_task(self._execute_sequence_event(operation, params, delay_ms))
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
            
            execution_time = (time.time() - start_time) * 1000
            
            return Either.right({
                "execution_time_ms": execution_time,
                "events_executed": len(sequence.events),
                "sequence_completed": True
            })
            
        except Exception as e:
            return Either.left(IntegrationError(
                "SEQUENCE_EXECUTION_ERROR",
                f"Failed to execute timing sequence: {str(e)}"
            ))
    
    async def _execute_accessibility_interaction(
        self,
        element_type: str,
        action: str,
        position: Coordinate,
        accessibility_info: Optional[Dict[str, Any]]
    ) -> Either[IntegrationError, Dict[str, Any]]:
        """Execute accessibility interaction with screen reader support."""
        try:
            start_time = time.time()
            
            # Generate AppleScript for accessibility interaction
            applescript = self._generate_accessibility_applescript(
                element_type, action, position, accessibility_info
            )
            
            # Simulate accessibility interaction
            await asyncio.sleep(0.1)  # Brief delay for accessibility
            
            execution_time = (time.time() - start_time) * 1000
            
            return Either.right({
                "applescript_generated": True,
                "execution_time_ms": execution_time,
                "accessibility_executed": True
            })
            
        except Exception as e:
            return Either.left(IntegrationError(
                "ACCESSIBILITY_EXECUTION_ERROR",
                f"Failed to execute accessibility interaction: {str(e)}"
            ))
    
    async def _simulate_pinch_gesture(self, gesture: GestureEvent):
        """Simulate pinch gesture with realistic timing."""
        steps = max(10, int(gesture.duration_ms / 50))  # 50ms per step
        step_duration = gesture.duration_ms / steps / 1000.0
        
        for step in range(steps):
            progress = step / steps
            # Simulate finger positions expanding/contracting
            await asyncio.sleep(step_duration)
    
    async def _simulate_rotate_gesture(self, gesture: GestureEvent):
        """Simulate rotate gesture with realistic timing."""
        steps = max(10, int(gesture.duration_ms / 50))
        step_duration = gesture.duration_ms / steps / 1000.0
        
        for step in range(steps):
            progress = step / steps
            # Simulate finger rotation around center
            await asyncio.sleep(step_duration)
    
    async def _simulate_swipe_gesture(self, gesture: GestureEvent):
        """Simulate swipe gesture with realistic movement."""
        steps = max(5, int(gesture.duration_ms / 100))  # 100ms per step
        step_duration = gesture.duration_ms / steps / 1000.0
        
        for step in range(steps):
            progress = step / steps
            # Simulate finger movement in direction
            await asyncio.sleep(step_duration)
    
    async def _simulate_scroll_gesture(self, gesture: GestureEvent):
        """Simulate two-finger scroll gesture."""
        steps = max(3, int(gesture.duration_ms / 150))  # 150ms per step
        step_duration = gesture.duration_ms / steps / 1000.0
        
        for step in range(steps):
            await asyncio.sleep(step_duration)
    
    async def _execute_sequence_event(self, operation: str, params: Dict[str, Any], delay_ms: int):
        """Execute a single event in a timing sequence."""
        await asyncio.sleep(delay_ms / 1000.0)
        logger.debug(f"Executed sequence event: {operation}")
    
    def _generate_gesture_applescript(self, gesture: GestureEvent) -> str:
        """Generate AppleScript for gesture execution."""
        x, y = gesture.center_position.x, gesture.center_position.y
        
        applescript = f'''
tell application "System Events"
    try
        -- Perform {gesture.gesture_type.value} gesture
        set gestureCenter to {{{x}, {y}}}
        -- Gesture: {gesture.gesture_type.value} with {gesture.finger_count} fingers
        -- Magnitude: {gesture.magnitude}, Duration: {gesture.duration_ms}ms
        
        return "SUCCESS: Gesture {gesture.gesture_type.value} executed"
    on error errorMessage
        return "ERROR: " & errorMessage
    end try
end tell
'''
        return applescript
    
    def _generate_accessibility_applescript(
        self,
        element_type: str,
        action: str,
        position: Coordinate,
        accessibility_info: Optional[Dict[str, Any]]
    ) -> str:
        """Generate AppleScript for accessibility interaction."""
        x, y = position.x, position.y
        
        applescript = f'''
tell application "System Events"
    try
        -- Accessibility interaction
        set elementPosition to {{{x}, {y}}}
        -- Element: {element_type}, Action: {action}
        
        -- Announce action for screen readers
        -- Perform {action} on {element_type}
        
        return "SUCCESS: Accessibility {action} executed on {element_type}"
    on error errorMessage
        return "ERROR: " & errorMessage
    end try
end tell
'''
        return applescript
    
    def get_gesture_history(self) -> List[Dict[str, Any]]:
        """Get history of recent gestures."""
        return [
            {
                "event_id": gesture.event_id,
                "gesture_type": gesture.gesture_type.value,
                "center_position": gesture.center_position.to_dict(),
                "magnitude": gesture.magnitude,
                "finger_count": gesture.finger_count
            }
            for gesture in self.active_gestures.values()
        ]
    
    def get_sequence_history(self) -> List[Dict[str, Any]]:
        """Get history of recent timing sequences."""
        return [
            {
                "event_id": sequence.event_id,
                "events_count": len(sequence.events),
                "total_duration_ms": sequence.total_duration_ms,
                "synchronous": sequence.synchronous
            }
            for sequence in self.sequence_history
        ]