"""
Clipboard Management Module for Keyboard Maestro MCP

Provides secure clipboard operations with history management, named clipboards,
and comprehensive security validation for AI-driven text processing workflows.
"""

from .clipboard_manager import ClipboardContent, ClipboardFormat, ClipboardManager
from .named_clipboards import NamedClipboard, NamedClipboardManager

__all__ = [
    "ClipboardManager",
    "ClipboardContent",
    "ClipboardFormat",
    "NamedClipboardManager",
    "NamedClipboard",
]
