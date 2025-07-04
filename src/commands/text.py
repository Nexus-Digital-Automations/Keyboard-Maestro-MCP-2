"""
Text Manipulation Commands

Provides secure text input, search, and manipulation commands
with comprehensive validation and security boundaries.
"""

from __future__ import annotations
from typing import Optional, FrozenSet, Union, Pattern
from dataclasses import dataclass, field
from enum import Enum
import re
import time

from ..core.types import ExecutionContext, CommandResult, Permission, Duration
from ..core.contracts import require, ensure
from .base import BaseCommand, create_command_result, is_safe_text_content
from .validation import SecurityValidator


class TypingSpeed(Enum):
    """Typing speed options for text input."""
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"
    INSTANT = "instant"


class TextSearchMode(Enum):
    """Text search modes."""
    EXACT = "exact"
    CONTAINS = "contains"
    REGEX = "regex"
    CASE_INSENSITIVE = "case_insensitive"


@dataclass(frozen=True)
class TypeTextCommand(BaseCommand):
    """
    Safely type text with configurable speed and validation.
    
    Performs comprehensive security validation to prevent
    script injection and malicious content.
    """
    
    def get_text(self) -> str:
        """Get the text to type."""
        return self.parameters.get("text", "")
    
    def get_typing_speed(self) -> TypingSpeed:
        """Get the typing speed setting."""
        speed_str = self.parameters.get("typing_speed", "normal")
        try:
            return TypingSpeed(speed_str)
        except ValueError:
            return TypingSpeed.NORMAL
    
    def get_delay_between_keys(self) -> float:
        """Get delay between keystrokes in seconds."""
        speed = self.get_typing_speed()
        speed_delays = {
            TypingSpeed.SLOW: 0.1,
            TypingSpeed.NORMAL: 0.05,
            TypingSpeed.FAST: 0.02,
            TypingSpeed.INSTANT: 0.0
        }
        return speed_delays[speed]
    
    def _validate_impl(self) -> bool:
        """Validate text input parameters."""
        text = self.get_text()
        
        # Check if text is provided
        if not text:
            return False
        
        # Validate text content for security
        if not is_safe_text_content(text):
            return False
        
        # Validate typing speed
        try:
            self.get_typing_speed()
        except ValueError:
            return False
        
        return True
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute text typing with security validation."""
        text = self.get_text()
        delay = self.get_delay_between_keys()
        
        start_time = time.time()
        
        try:
            # Simulate typing with delays
            characters_typed = 0
            
            for char in text:
                # In a real implementation, this would send keystrokes
                # For now, we'll simulate the delay
                if delay > 0:
                    time.sleep(delay)
                characters_typed += 1
                
                # Check for execution timeout
                if time.time() - start_time > context.timeout.seconds:
                    return create_command_result(
                        success=False,
                        error_message=f"Typing timed out after {characters_typed} characters",
                        characters_typed=characters_typed,
                        partial_text=text[:characters_typed]
                    )
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            return create_command_result(
                success=True,
                output=f"Typed {characters_typed} characters",
                execution_time=execution_time,
                characters_typed=characters_typed,
                text_length=len(text),
                typing_speed=self.get_typing_speed().value
            )
            
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Text typing failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Text input requires text input permission."""
        return frozenset([Permission.TEXT_INPUT])
    
    def get_security_risk_level(self) -> str:
        """Text input has medium risk due to potential for injection."""
        return "medium"


@dataclass(frozen=True)
class FindTextCommand(BaseCommand):
    """
    Find text in the current context with pattern matching.
    
    Supports exact match, contains, and safe regex patterns
    with comprehensive validation.
    """
    
    def get_search_pattern(self) -> str:
        """Get the search pattern."""
        return self.parameters.get("pattern", "")
    
    def get_search_mode(self) -> TextSearchMode:
        """Get the search mode."""
        mode_str = self.parameters.get("mode", "exact")
        try:
            return TextSearchMode(mode_str)
        except ValueError:
            return TextSearchMode.EXACT
    
    def get_case_sensitive(self) -> bool:
        """Get case sensitivity setting."""
        return self.parameters.get("case_sensitive", True)
    
    def get_target_text(self) -> str:
        """Get the text to search in (for testing)."""
        return self.parameters.get("target_text", "")
    
    def _validate_impl(self) -> bool:
        """Validate search parameters."""
        pattern = self.get_search_pattern()
        
        # Check if pattern is provided
        if not pattern:
            return False
        
        # Validate pattern for security
        if not is_safe_text_content(pattern):
            return False
        
        # For regex mode, validate the regex pattern
        if self.get_search_mode() == TextSearchMode.REGEX:
            try:
                re.compile(pattern)
            except re.error:
                return False
        
        return True
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute text search with pattern matching."""
        pattern = self.get_search_pattern()
        mode = self.get_search_mode()
        case_sensitive = self.get_case_sensitive()
        target_text = self.get_target_text()
        
        start_time = time.time()
        
        try:
            # In a real implementation, this would search in the active application
            # For now, we'll search in the provided target_text for testing
            if not target_text:
                return create_command_result(
                    success=False,
                    error_message="No target text available for search",
                    execution_time=Duration.from_seconds(time.time() - start_time)
                )
            
            matches = []
            match_count = 0
            
            if mode == TextSearchMode.EXACT:
                search_text = target_text if case_sensitive else target_text.lower()
                search_pattern = pattern if case_sensitive else pattern.lower()
                
                if search_pattern in search_text:
                    match_count = search_text.count(search_pattern)
                    # Find all positions
                    start = 0
                    while True:
                        pos = search_text.find(search_pattern, start)
                        if pos == -1:
                            break
                        matches.append(pos)
                        start = pos + 1
            
            elif mode == TextSearchMode.CONTAINS:
                search_text = target_text if case_sensitive else target_text.lower()
                search_pattern = pattern if case_sensitive else pattern.lower()
                
                if search_pattern in search_text:
                    match_count = 1
                    matches.append(search_text.find(search_pattern))
            
            elif mode == TextSearchMode.REGEX:
                flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    regex_matches = list(re.finditer(pattern, target_text, flags))
                    match_count = len(regex_matches)
                    matches = [match.start() for match in regex_matches]
                except re.error as e:
                    return create_command_result(
                        success=False,
                        error_message=f"Regex pattern error: {str(e)}",
                        execution_time=Duration.from_seconds(time.time() - start_time)
                    )
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            return create_command_result(
                success=True,
                output=f"Found {match_count} matches for '{pattern}'",
                execution_time=execution_time,
                match_count=match_count,
                match_positions=matches,
                search_pattern=pattern,
                search_mode=mode.value,
                case_sensitive=case_sensitive
            )
            
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Text search failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Text search may require screen capture to read text."""
        return frozenset([Permission.SCREEN_CAPTURE])
    
    def get_security_risk_level(self) -> str:
        """Text search has low risk as it's read-only."""
        return "low"


@dataclass(frozen=True)
class ReplaceTextCommand(BaseCommand):
    """
    Replace text with validation and safety checks.
    
    Provides secure text replacement with pattern matching
    and comprehensive security validation.
    """
    
    def get_search_pattern(self) -> str:
        """Get the pattern to search for."""
        return self.parameters.get("search_pattern", "")
    
    def get_replacement_text(self) -> str:
        """Get the replacement text."""
        return self.parameters.get("replacement_text", "")
    
    def get_max_replacements(self) -> int:
        """Get maximum number of replacements."""
        return max(1, min(100, self.parameters.get("max_replacements", 1)))
    
    def get_case_sensitive(self) -> bool:
        """Get case sensitivity setting."""
        return self.parameters.get("case_sensitive", True)
    
    def get_target_text(self) -> str:
        """Get the text to perform replacements in (for testing)."""
        return self.parameters.get("target_text", "")
    
    def _validate_impl(self) -> bool:
        """Validate replacement parameters."""
        search_pattern = self.get_search_pattern()
        replacement_text = self.get_replacement_text()
        
        # Check if both pattern and replacement are provided
        if not search_pattern or not replacement_text:
            return False
        
        # Validate both texts for security
        if not is_safe_text_content(search_pattern):
            return False
        
        if not is_safe_text_content(replacement_text):
            return False
        
        # Validate max replacements
        max_replacements = self.get_max_replacements()
        if max_replacements < 1 or max_replacements > 100:
            return False
        
        return True
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute text replacement with safety limits."""
        search_pattern = self.get_search_pattern()
        replacement_text = self.get_replacement_text()
        max_replacements = self.get_max_replacements()
        case_sensitive = self.get_case_sensitive()
        target_text = self.get_target_text()
        
        start_time = time.time()
        
        try:
            # In a real implementation, this would replace text in the active application
            # For now, we'll work with the provided target_text for testing
            if not target_text:
                return create_command_result(
                    success=False,
                    error_message="No target text available for replacement",
                    execution_time=Duration.from_seconds(time.time() - start_time)
                )
            
            # Perform replacement with safety limits
            if case_sensitive:
                # Count existing matches first
                match_count = target_text.count(search_pattern)
                actual_replacements = min(match_count, max_replacements)
                
                # Perform limited replacements
                result_text = target_text
                for _ in range(actual_replacements):
                    result_text = result_text.replace(search_pattern, replacement_text, 1)
            else:
                # Case-insensitive replacement is more complex
                result_text = target_text
                replacements_made = 0
                
                while replacements_made < max_replacements:
                    lower_text = result_text.lower()
                    lower_pattern = search_pattern.lower()
                    
                    pos = lower_text.find(lower_pattern)
                    if pos == -1:
                        break
                    
                    # Replace while preserving case of the rest of the text
                    result_text = (
                        result_text[:pos] + 
                        replacement_text + 
                        result_text[pos + len(search_pattern):]
                    )
                    replacements_made += 1
                
                actual_replacements = replacements_made
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            return create_command_result(
                success=True,
                output=f"Replaced {actual_replacements} occurrences of '{search_pattern}' with '{replacement_text}'",
                execution_time=execution_time,
                replacements_made=actual_replacements,
                search_pattern=search_pattern,
                replacement_text=replacement_text,
                result_text=result_text,
                case_sensitive=case_sensitive
            )
            
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Text replacement failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Text replacement requires both text input and screen capture."""
        return frozenset([Permission.TEXT_INPUT, Permission.SCREEN_CAPTURE])
    
    def get_security_risk_level(self) -> str:
        """Text replacement has medium risk due to modification capabilities."""
        return "medium"