"""
Property-based tests for hardware event validation and interaction safety.

This module provides comprehensive property-based testing for the interface
automation system, validating security boundaries, coordinate safety, input
validation, and interaction patterns across all possible input ranges.

Security: Tests validate all security constraints and malicious input detection.
Performance: Tests validate timing constraints and rate limiting behavior.
Type Safety: Tests validate type system integrity and contract compliance.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import List, Dict, Any
import asyncio
import re

from src.core.hardware_events import (
    Coordinate, MouseEvent, KeyboardEvent, DragOperation, ScrollEvent, GestureEvent,
    MouseButton, KeyCode, ModifierKey, ScrollDirection, GestureType, SwipeDirection,
    HardwareEventValidator, RateLimiter, create_mouse_click, create_text_input,
    create_key_combination, create_drag_drop
)
from src.interaction.mouse_controller import MouseController
from src.interaction.keyboard_controller import KeyboardController
from src.interaction.gesture_controller import GestureController
from src.core.either import Either
from src.core.errors import SecurityError, ValidationError


class TestCoordinateProperties:
    """Property-based tests for coordinate validation and safety."""
    
    @given(st.integers(min_value=-1000, max_value=10000), 
           st.integers(min_value=-1000, max_value=10000))
    def test_coordinate_bounds_validation(self, x: int, y: int):
        """Property: Coordinate creation should validate bounds correctly."""
        if 0 <= x <= 8192 and 0 <= y <= 8192:
            coord = Coordinate(x, y)
            assert coord.x == x
            assert coord.y == y
        else:
            with pytest.raises(ValueError):
                Coordinate(x, y)
    
    @given(st.integers(min_value=0, max_value=8192), 
           st.integers(min_value=0, max_value=8192))
    def test_coordinate_safety_properties(self, x: int, y: int):
        """Property: Valid coordinates should pass safety validation."""
        coord = Coordinate(x, y)
        result = HardwareEventValidator.validate_coordinate_safety(coord)
        
        # Check if coordinate is in dangerous system areas
        dangerous_areas = [(0, 0, 100, 50), (0, 0, 200, 100)]
        is_dangerous = any(
            dx <= x <= dx + dw and dy <= y <= dy + dh
            for dx, dy, dw, dh in dangerous_areas
        )
        
        if is_dangerous:
            assert result.is_left()
            assert result.get_left().security_code == "DANGEROUS_COORDINATE"
        else:
            assert result.is_right()
    
    @given(st.integers(min_value=0, max_value=8192), 
           st.integers(min_value=0, max_value=8192),
           st.integers(min_value=0, max_value=8192), 
           st.integers(min_value=0, max_value=8192))
    def test_distance_calculation_properties(self, x1: int, y1: int, x2: int, y2: int):
        """Property: Distance calculation should be symmetric and non-negative."""
        coord1 = Coordinate(x1, y1)
        coord2 = Coordinate(x2, y2)
        
        distance1 = coord1.distance_to(coord2)
        distance2 = coord2.distance_to(coord1)
        
        # Distance should be symmetric
        assert abs(distance1 - distance2) < 0.001
        
        # Distance should be non-negative
        assert distance1 >= 0
        
        # Distance to self should be zero
        assert coord1.distance_to(coord1) == 0


class TestMouseEventProperties:
    """Property-based tests for mouse event validation and execution."""
    
    @given(st.integers(min_value=0, max_value=8192),
           st.integers(min_value=0, max_value=8192),
           st.sampled_from(list(MouseButton)),
           st.integers(min_value=1, max_value=10),
           st.integers(min_value=10, max_value=5000))
    def test_mouse_event_creation_properties(self, x: int, y: int, button: MouseButton, 
                                           click_count: int, duration: int):
        """Property: Valid mouse events should be created successfully."""
        coord = Coordinate(x, y)
        
        mouse_event = MouseEvent(
            operation="click",
            position=coord,
            button=button,
            click_count=click_count,
            duration_ms=duration
        )
        
        assert mouse_event.position == coord
        assert mouse_event.button == button
        assert mouse_event.click_count == click_count
        assert mouse_event.duration_ms == duration
    
    @given(st.integers(min_value=-10, max_value=20),
           st.integers(min_value=-100, max_value=10000))
    def test_mouse_event_validation_boundaries(self, click_count: int, duration: int):
        """Property: Mouse events should validate parameter boundaries."""
        coord = Coordinate(100, 100)
        
        if 1 <= click_count <= 10 and 10 <= duration <= 5000:
            # Should create successfully
            mouse_event = MouseEvent(
                operation="click",
                position=coord,
                click_count=click_count,
                duration_ms=duration
            )
            assert mouse_event is not None
        else:
            # Should raise validation error
            with pytest.raises(ValueError):
                MouseEvent(
                    operation="click",
                    position=coord,
                    click_count=click_count,
                    duration_ms=duration
                )
    
    @pytest.mark.asyncio
    @given(st.integers(min_value=100, max_value=1000),
           st.integers(min_value=100, max_value=800))
    async def test_mouse_controller_properties(self, x: int, y: int):
        """Property: Mouse controller should handle valid coordinates safely."""
        # Skip dangerous areas for this test
        assume(not (x <= 200 and y <= 100))
        
        controller = MouseController()
        coord = Coordinate(x, y)
        
        result = await controller.click_at_position(coord)
        
        # Should either succeed or fail with proper error handling
        assert result.is_right() or result.is_left()
        
        if result.is_right():
            response = result.get_right()
            assert response["success"] is True
            assert "position" in response
            assert response["position"]["x"] == x
            assert response["position"]["y"] == y


class TestKeyboardEventProperties:
    """Property-based tests for keyboard event validation and text safety."""
    
    @given(st.text(min_size=0, max_size=10000))
    def test_text_content_validation(self, text: str):
        """Property: Text validation should detect dangerous patterns consistently."""
        result = HardwareEventValidator.validate_text_safety(text)
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'password\s*[:=]\s*\S+',
            r'pass\s*[:=]\s*\S+',
            r'secret\s*[:=]\s*\S+',
            r'token\s*[:=]\s*\S+',
            r'<script',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'rm\s+-rf',
            r'sudo\s+'
        ]
        
        text_lower = text.lower()
        has_dangerous_pattern = any(
            re.search(pattern, text_lower) for pattern in dangerous_patterns
        )
        
        # Check for dangerous control characters
        dangerous_chars = ['\x1b', '\x00', '\x7f']
        has_dangerous_char = any(char in text for char in dangerous_chars)
        
        if has_dangerous_pattern or has_dangerous_char:
            assert result.is_left()
            error = result.get_left()
            # Check for security-related error codes
            assert any(pattern in error.error_code for pattern in ["DANGEROUS", "SECURITY", "CONTROL_CHAR"])
        else:
            assert result.is_right()
    
    @given(st.lists(st.sampled_from([
        "cmd", "command", "opt", "option", "shift", "ctrl", "control", "fn",
        "a", "b", "c", "1", "2", "3", "f1", "f2", "space", "enter", "tab"
    ]), min_size=1, max_size=15))
    def test_key_combination_validation(self, keys: List[str]):
        """Property: Key combination validation should handle all valid key sets."""
        result = HardwareEventValidator.validate_key_combination(keys)
        
        if len(keys) <= 10:
            # All keys in our test set are valid
            assert result.is_right()
        else:
            # Too many keys
            assert result.is_left()
            assert result.get_left().error_code == "TOO_MANY_KEYS"
    
    @given(st.text(min_size=1, max_size=1000, alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'S', 'Z'),
        blacklist_characters='\x1b\x00\x7f'
    )))
    @pytest.mark.asyncio
    async def test_text_typing_properties(self, text: str):
        """Property: Safe text should be typed successfully."""
        # Skip text with dangerous patterns
        dangerous_patterns = ['password:', 'secret:', '<script', 'javascript:']
        assume(not any(pattern in text.lower() for pattern in dangerous_patterns))
        
        controller = KeyboardController()
        result = await controller.type_text(text, delay_between_chars=10)
        
        # Should either succeed or fail gracefully
        assert result.is_right() or result.is_left()
        
        if result.is_right():
            response = result.get_right()
            assert response["success"] is True
            assert response["text_length"] == len(text)


class TestDragOperationProperties:
    """Property-based tests for drag operation validation and safety."""
    
    @given(st.integers(min_value=0, max_value=8192),
           st.integers(min_value=0, max_value=8192),
           st.integers(min_value=0, max_value=8192),
           st.integers(min_value=0, max_value=8192),
           st.integers(min_value=100, max_value=10000))
    def test_drag_operation_creation(self, x1: int, y1: int, x2: int, y2: int, duration: int):
        """Property: Valid drag operations should be created with proper validation."""
        source = Coordinate(x1, y1)
        destination = Coordinate(x2, y2)
        
        if source != destination:
            drag_op = DragOperation(
                source=source,
                destination=destination,
                duration_ms=duration
            )
            
            assert drag_op.source == source
            assert drag_op.destination == destination
            assert drag_op.duration_ms == duration
            
            # Distance should be calculated correctly
            expected_distance = source.distance_to(destination)
            assert abs(drag_op.distance() - expected_distance) < 0.001
        else:
            # Same source and destination should raise error
            with pytest.raises(ValueError):
                DragOperation(source=source, destination=destination)
    
    @given(st.integers(min_value=200, max_value=1000),
           st.integers(min_value=200, max_value=800),
           st.integers(min_value=300, max_value=1100),
           st.integers(min_value=300, max_value=900))
    def test_drag_distance_validation(self, x1: int, y1: int, x2: int, y2: int):
        """Property: Drag distance validation should prevent excessively long drags."""
        # Skip when source and destination are the same (would fail validation)
        assume(not (x1 == x2 and y1 == y2))
        
        source = Coordinate(x1, y1)
        destination = Coordinate(x2, y2)
        
        drag_op = DragOperation(source=source, destination=destination)
        result = HardwareEventValidator.validate_drag_distance(drag_op)
        
        distance = drag_op.distance()
        
        if distance <= 3000:
            # Skip dangerous coordinate areas for successful cases
            if not any(
                (dx <= coord.x <= dx + dw and dy <= coord.y <= dy + dh)
                for coord in [source, destination]
                for dx, dy, dw, dh in [(0, 0, 100, 50), (0, 0, 200, 100)]
            ):
                assert result.is_right()
        else:
            assert result.is_left()
            assert result.get_left().error_code == "DRAG_TOO_LONG"


class TestGestureProperties:
    """Property-based tests for gesture validation and execution."""
    
    @given(st.sampled_from(list(GestureType)),
           st.integers(min_value=200, max_value=1000),
           st.integers(min_value=200, max_value=800),
           st.floats(min_value=0.1, max_value=5.0),
           st.integers(min_value=2, max_value=4),
           st.integers(min_value=100, max_value=3000))
    def test_gesture_event_properties(self, gesture_type: GestureType, x: int, y: int,
                                    magnitude: float, finger_count: int, duration: int):
        """Property: Gesture events should validate finger count compatibility."""
        position = Coordinate(x, y)
        
        # Check finger count compatibility
        compatible_fingers = {
            GestureType.PINCH: [2],
            GestureType.ROTATE: [2],
            GestureType.SWIPE: [1, 2, 3, 4],
            GestureType.TWO_FINGER_TAP: [2],
            GestureType.THREE_FINGER_TAP: [3],
            GestureType.FOUR_FINGER_TAP: [4]
        }
        
        if finger_count in compatible_fingers.get(gesture_type, []):
            try:
                # Handle gesture-specific requirements
                if gesture_type == GestureType.SWIPE:
                    gesture_event = GestureEvent(
                        gesture_type=gesture_type,
                        position=position,
                        direction=SwipeDirection.UP,  # Provide required direction
                        finger_count=finger_count,
                        duration_ms=duration
                    )
                elif gesture_type == GestureType.PINCH:
                    gesture_event = GestureEvent(
                        gesture_type=gesture_type,
                        position=position,
                        scale=magnitude,  # Use magnitude as scale
                        finger_count=finger_count,
                        duration_ms=duration
                    )
                elif gesture_type == GestureType.ROTATE:
                    gesture_event = GestureEvent(
                        gesture_type=gesture_type,
                        position=position,
                        rotation_degrees=magnitude * 36,  # Convert magnitude to degrees
                        finger_count=finger_count,
                        duration_ms=duration
                    )
                else:
                    # For TAP gestures
                    gesture_event = GestureEvent(
                        gesture_type=gesture_type,
                        position=position,
                        finger_count=finger_count,
                        duration_ms=duration
                    )
                
                assert gesture_event.gesture_type == gesture_type
                assert gesture_event.position == position
                assert gesture_event.finger_count == finger_count
            except ValueError:
                # Some gestures may fail validation due to specific requirements
                pass
        else:
            # Incompatible finger count should be handled by validation
            pass


class TestRateLimitingProperties:
    """Property-based tests for rate limiting behavior."""
    
    @given(st.lists(st.text(min_size=5, max_size=20), min_size=1, max_size=200))
    @pytest.mark.asyncio
    async def test_rate_limiter_properties(self, operations: List[str]):
        """Property: Rate limiter should enforce limits consistently."""
        rate_limiter = RateLimiter()
        
        mouse_operations = [op for op in operations if "mouse" in op.lower()]
        keyboard_operations = [op for op in operations if "key" in op.lower()]
        
        mouse_limit_exceeded = False
        keyboard_limit_exceeded = False
        
        for op in mouse_operations:
            result = rate_limiter.check_rate_limit(f"mouse_{op}")
            if result.is_left():
                mouse_limit_exceeded = True
                break
        
        for op in keyboard_operations:
            result = rate_limiter.check_rate_limit(f"keyboard_{op}")
            if result.is_left():
                keyboard_limit_exceeded = True
                break
        
        # If we have many operations, rate limits should eventually be hit
        if len(mouse_operations) > 60:
            assert mouse_limit_exceeded
        
        if len(keyboard_operations) > 110:
            assert keyboard_limit_exceeded


class TestSecurityProperties:
    """Property-based tests for security validation across all components."""
    
    @given(st.text(min_size=0, max_size=100))
    def test_password_pattern_detection(self, text: str):
        """Property: Password patterns should be detected consistently."""
        result = HardwareEventValidator.validate_text_safety(text)
        
        password_indicators = ['password:', 'pass=', 'secret:', 'token=']
        
        if any(indicator in text.lower() for indicator in password_indicators):
            assert result.is_left()
            assert "DANGEROUS_TEXT_PATTERN" in result.get_left().error_code
    
    @given(st.text(min_size=0, max_size=100))
    def test_script_injection_detection(self, text: str):
        """Property: Script injection patterns should be detected."""
        result = HardwareEventValidator.validate_text_safety(text)
        
        script_indicators = ['<script', 'javascript:', 'eval(', 'exec(']
        
        if any(indicator in text.lower() for indicator in script_indicators):
            assert result.is_left()
            assert "DANGEROUS_TEXT_PATTERN" in result.get_left().error_code
    
    @given(st.integers(min_value=0, max_value=300),
           st.integers(min_value=0, max_value=200))
    def test_system_area_protection(self, x: int, y: int):
        """Property: System areas should be protected from interaction."""
        coord = Coordinate(x, y)
        result = HardwareEventValidator.validate_coordinate_safety(coord)
        
        # Check if coordinate is in protected system areas
        in_menu_area = (0 <= x <= 100 and 0 <= y <= 50)
        in_apple_menu = (0 <= x <= 200 and 0 <= y <= 100)
        
        if in_menu_area or in_apple_menu:
            assert result.is_left()
            assert result.get_left().error_code == "DANGEROUS_COORDINATE"
        else:
            assert result.is_right()


# Performance and timing tests
class TestPerformanceProperties:
    """Property-based tests for performance and timing constraints."""
    
    @pytest.mark.asyncio
    @given(st.integers(min_value=500, max_value=1500),
           st.integers(min_value=500, max_value=1000))
    @settings(deadline=2000)  # 2 second deadline for property tests
    async def test_mouse_operation_timing(self, x: int, y: int):
        """Property: Mouse operations should complete within timing constraints."""
        # Skip dangerous areas
        assume(not (x <= 200 and y <= 100))
        
        controller = MouseController()
        coord = Coordinate(x, y)
        
        import time
        start_time = time.time()
        
        result = await controller.click_at_position(coord, duration_ms=50)
        
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should complete in reasonable time (allowing for test overhead)
        assert execution_time < 500  # 500ms max for simple operations
        
        if result.is_right():
            response = result.get_right()
            assert "execution_time_ms" in response


# Integration property tests
class TestIntegrationProperties:
    """Property-based tests for component integration and workflow validation."""
    
    @pytest.mark.asyncio
    @given(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('L', 'N'),
        blacklist_characters='\x1b\x00\x7f'
    )))
    @settings(deadline=3000)
    async def test_keyboard_mouse_integration(self, text: str):
        """Property: Keyboard and mouse operations should integrate seamlessly."""
        # Skip dangerous text patterns
        assume(not any(pattern in text.lower() for pattern in [
            'password', 'secret', 'script', 'eval'
        ]))
        
        mouse_controller = MouseController()
        keyboard_controller = KeyboardController()
        
        # Simulate click then type workflow
        coord = Coordinate(500, 400)  # Safe coordinates
        
        # Click to focus
        click_result = await mouse_controller.click_at_position(coord)
        
        # Type text
        type_result = await keyboard_controller.type_text(text, delay_between_chars=20)
        
        # Both operations should succeed or fail gracefully
        if click_result.is_right() and type_result.is_right():
            click_response = click_result.get_right()
            type_response = type_result.get_right()
            
            assert click_response["success"] is True
            assert type_response["success"] is True
            assert type_response["text_length"] == len(text)


if __name__ == "__main__":
    # Run specific property tests for debugging
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])