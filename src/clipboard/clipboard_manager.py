"""
Core Clipboard Operations with Security and Privacy Protection

This module implements comprehensive clipboard management for Keyboard Maestro MCP,
providing secure access to clipboard content, history, and format detection with
defensive programming and property-based validation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Pattern
from enum import Enum
import re
import time
import subprocess
import asyncio
from pathlib import Path

from ..core.types import Duration
from ..core.contracts import require, ensure
from ..integration.km_client import Either, KMError


class ClipboardFormat(Enum):
    """Supported clipboard content formats with validation."""
    TEXT = "text"
    IMAGE = "image" 
    FILE = "file"
    URL = "url"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClipboardContent:
    """Type-safe clipboard content representation with security validation."""
    content: Union[str, bytes]
    format: ClipboardFormat
    size_bytes: int
    timestamp: float
    is_sensitive: bool = False
    preview_safe: bool = True
    
    @require(lambda self: self.size_bytes >= 0)
    @require(lambda self: self.size_bytes <= 100_000_000)  # 100MB limit
    def __post_init__(self):
        """Validate clipboard content constraints."""
        pass
    
    def preview(self, max_length: int = 50) -> str:
        """Generate safe preview of clipboard content with privacy protection."""
        if self.is_sensitive:
            return "[SENSITIVE CONTENT HIDDEN]"
        
        if not self.preview_safe:
            return f"[{self.format.value.upper()} - CONTENT FILTERED]"
        
        if self.format == ClipboardFormat.TEXT:
            content_str = str(self.content)
            if len(content_str) > max_length:
                return content_str[:max_length] + "..."
            return content_str
        elif self.format == ClipboardFormat.IMAGE:
            return f"[IMAGE - {self.size_bytes} bytes]"
        elif self.format == ClipboardFormat.FILE:
            return f"[FILE - {self.size_bytes} bytes]"
        elif self.format == ClipboardFormat.URL:
            return f"[URL - {str(self.content)[:max_length]}...]" if len(str(self.content)) > max_length else f"[URL - {str(self.content)}]"
        else:
            return f"[{self.format.value.upper()} - {self.size_bytes} bytes]"
    
    def is_empty(self) -> bool:
        """Check if clipboard content is empty."""
        if self.format == ClipboardFormat.TEXT:
            return not str(self.content).strip()
        return self.size_bytes == 0


class ClipboardManager:
    """
    Secure clipboard operations with history and format detection.
    
    Implements defensive programming patterns with comprehensive security
    validation, sensitive content detection, and memory management.
    """
    
    # Sensitive content patterns for detection
    _SENSITIVE_PATTERNS: List[Pattern[str]] = [
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
        re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),  # Credit card
        re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN
        re.compile(r'\b[A-F0-9]{32}\b'),  # MD5 hash
        re.compile(r'\b[A-F0-9]{64}\b'),  # SHA256 hash
        re.compile(r'password\s*[:=]\s*\S+', re.IGNORECASE),  # Password patterns
        re.compile(r'token\s*[:=]\s*\S+', re.IGNORECASE),  # Token patterns
        re.compile(r'api[_-]?key\s*[:=]\s*\S+', re.IGNORECASE),  # API keys
        re.compile(r'secret\s*[:=]\s*\S+', re.IGNORECASE),  # Secrets
    ]
    
    def __init__(self):
        """Initialize clipboard manager with security settings."""
        self._max_content_size = 100_000_000  # 100MB
        self._max_history_size = 200
        self._detection_enabled = True
    
    @require(lambda content: isinstance(content, str))
    @require(lambda content: len(content.encode('utf-8')) <= 100_000_000)
    @ensure(lambda result: result.is_right() or result.get_left().code in ["SECURITY_ERROR", "SIZE_ERROR", "EXECUTION_ERROR"])
    async def set_clipboard(self, content: str) -> Either[KMError, bool]:
        """
        Set clipboard content with comprehensive security validation.
        
        Args:
            content: Text content to set on clipboard
            
        Returns:
            Either success status or error details
        """
        try:
            # Validate content size
            content_bytes = content.encode('utf-8')
            if len(content_bytes) > self._max_content_size:
                return Either.left(KMError.validation_error(
                    f"Content size {len(content_bytes)} exceeds maximum {self._max_content_size}"
                ))
            
            # Detect sensitive content
            if self._detection_enabled and self._detect_sensitive_content(content):
                return Either.left(KMError.validation_error(
                    "Potentially sensitive content detected. Use include_sensitive=True to override."
                ))
            
            # Set clipboard via AppleScript
            result = await self._set_clipboard_applescript(content)
            return result
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to set clipboard: {str(e)}"))
    
    @ensure(lambda result: result.is_right() or result.get_left().code in ["ACCESS_ERROR", "EXECUTION_ERROR"])
    async def get_clipboard(self, include_sensitive: bool = False) -> Either[KMError, ClipboardContent]:
        """
        Get current clipboard content with format detection and security filtering.
        
        Args:
            include_sensitive: Whether to include potentially sensitive content
            
        Returns:
            Either clipboard content or error details
        """
        try:
            # Get clipboard content via AppleScript
            result = await self._get_clipboard_applescript()
            if result.is_left():
                return result
            
            content = result.get_right()
            timestamp = time.time()
            
            # Detect format and validate content
            format_type = self._detect_format(content)
            size_bytes = len(content.encode('utf-8')) if isinstance(content, str) else len(content)
            
            # Check for sensitive content
            is_sensitive = False
            preview_safe = True
            if isinstance(content, str) and self._detection_enabled:
                is_sensitive = self._detect_sensitive_content(content)
                if is_sensitive and not include_sensitive:
                    preview_safe = False
                    content = "[SENSITIVE CONTENT FILTERED]"
            
            clipboard_content = ClipboardContent(
                content=content,
                format=format_type,
                size_bytes=size_bytes,
                timestamp=timestamp,
                is_sensitive=is_sensitive,
                preview_safe=preview_safe
            )
            
            return Either.right(clipboard_content)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to get clipboard: {str(e)}"))
    
    @require(lambda index: index >= 0 and index < 200)
    async def get_history_item(self, index: int, include_sensitive: bool = False) -> Either[KMError, ClipboardContent]:
        """
        Get item from clipboard history with bounds checking and security filtering.
        
        Args:
            index: 0-based history position (0 = most recent)
            include_sensitive: Whether to include sensitive content
            
        Returns:
            Either clipboard content or error details
        """
        try:
            # Get clipboard history via AppleScript
            result = await self._get_clipboard_history_applescript(index)
            if result.is_left():
                return result
            
            content = result.get_right()
            if not content:
                return Either.left(KMError.not_found_error(f"No clipboard history item at index {index}"))
            
            timestamp = time.time() - (index * 60)  # Estimate timestamp based on index
            
            # Detect format and validate content
            format_type = self._detect_format(content)
            size_bytes = len(content.encode('utf-8')) if isinstance(content, str) else len(content)
            
            # Check for sensitive content
            is_sensitive = False
            preview_safe = True
            if isinstance(content, str) and self._detection_enabled:
                is_sensitive = self._detect_sensitive_content(content)
                if is_sensitive and not include_sensitive:
                    preview_safe = False
                    content = "[SENSITIVE CONTENT FILTERED]"
            
            clipboard_content = ClipboardContent(
                content=content,
                format=format_type,
                size_bytes=size_bytes,
                timestamp=timestamp,
                is_sensitive=is_sensitive,
                preview_safe=preview_safe
            )
            
            return Either.right(clipboard_content)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to get clipboard history: {str(e)}"))
    
    async def get_history_list(self, count: int = 10, include_sensitive: bool = False) -> Either[KMError, List[ClipboardContent]]:
        """
        Get list of clipboard history items.
        
        Args:
            count: Number of items to retrieve (max 200)
            include_sensitive: Whether to include sensitive content
            
        Returns:
            Either list of clipboard content or error details
        """
        count = min(count, self._max_history_size)
        history_items = []
        
        for i in range(count):
            result = await self.get_history_item(i, include_sensitive)
            if result.is_right():
                history_items.append(result.get_right())
            elif result.get_left().code == "NOT_FOUND_ERROR":
                # End of history reached
                break
            else:
                # Other error - return it
                return result
        
        return Either.right(history_items)
    
    def _detect_sensitive_content(self, content: str) -> bool:
        """
        Detect potentially sensitive content using pattern matching.
        
        Args:
            content: Text content to analyze
            
        Returns:
            True if sensitive patterns detected
        """
        if not isinstance(content, str) or not content.strip():
            return False
        
        # Check content length - very long strings might contain sensitive data
        if len(content) > 10000:  # 10KB threshold
            return True
        
        # Check against sensitive patterns
        for pattern in self._SENSITIVE_PATTERNS:
            if pattern.search(content):
                return True
        
        # Check for Base64 encoded content (potential tokens/keys)
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 20 and base64_pattern.match(line):
                return True
        
        return False
    
    def _detect_format(self, content: Any) -> ClipboardFormat:
        """
        Detect clipboard content format.
        
        Args:
            content: Clipboard content to analyze
            
        Returns:
            Detected format type
        """
        if isinstance(content, bytes):
            # Check for image headers
            if content.startswith(b'\x89PNG') or content.startswith(b'\xFF\xD8\xFF'):
                return ClipboardFormat.IMAGE
            return ClipboardFormat.UNKNOWN
        
        if isinstance(content, str):
            content = content.strip()
            
            # Check for URL patterns
            url_pattern = re.compile(r'^https?://\S+$', re.IGNORECASE)
            if url_pattern.match(content):
                return ClipboardFormat.URL
            
            # Check for file path patterns
            if content.startswith('/') or content.startswith('file://'):
                if Path(content.replace('file://', '')).exists():
                    return ClipboardFormat.FILE
            
            # Default to text
            return ClipboardFormat.TEXT
        
        return ClipboardFormat.UNKNOWN
    
    async def _set_clipboard_applescript(self, content: str) -> Either[KMError, bool]:
        """Set clipboard content via AppleScript with proper escaping."""
        try:
            # Escape content for AppleScript
            escaped_content = self._escape_applescript_string(content)
            
            script = f'''
            tell application "System Events"
                set the clipboard to "{escaped_content}"
                return "success"
            end tell
            '''
            
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                return Either.left(KMError.execution_error(f"AppleScript failed: {error_msg}"))
            
            return Either.right(True)
            
        except asyncio.TimeoutError:
            return Either.left(KMError.timeout_error(Duration.from_seconds(10)))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to set clipboard: {str(e)}"))
    
    async def _get_clipboard_applescript(self) -> Either[KMError, str]:
        """Get clipboard content via AppleScript."""
        try:
            script = '''
            tell application "System Events"
                try
                    set clipboardText to (the clipboard as text)
                    return clipboardText
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                return Either.left(KMError.execution_error(f"AppleScript failed: {error_msg}"))
            
            output = stdout.decode().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:].strip()))
            
            return Either.right(output)
            
        except asyncio.TimeoutError:
            return Either.left(KMError.timeout_error(Duration.from_seconds(10)))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to get clipboard: {str(e)}"))
    
    async def _get_clipboard_history_applescript(self, index: int) -> Either[KMError, str]:
        """Get clipboard history item via AppleScript."""
        try:
            # Use Keyboard Maestro Engine to access clipboard history
            script = f'''
            tell application "Keyboard Maestro Engine"
                try
                    set historyItem to clipboard history {index + 1}
                    return historyItem as text
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                return Either.left(KMError.execution_error(f"AppleScript failed: {error_msg}"))
            
            output = stdout.decode().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.not_found_error(f"History item {index} not found"))
            
            return Either.right(output)
            
        except asyncio.TimeoutError:
            return Either.left(KMError.timeout_error(Duration.from_seconds(10)))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to get clipboard history: {str(e)}"))
    
    def _escape_applescript_string(self, text: str) -> str:
        """Escape string for safe use in AppleScript."""
        if not isinstance(text, str):
            text = str(text)
        
        # Replace dangerous characters
        text = text.replace('\\', '\\\\')  # Escape backslashes
        text = text.replace('"', '\\"')    # Escape quotes
        text = text.replace('\n', '\\n')   # Escape newlines
        text = text.replace('\r', '\\r')   # Escape carriage returns
        text = text.replace('\t', '\\t')   # Escape tabs
        
        return text