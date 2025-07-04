"""
Security utilities for the Keyboard Maestro MCP server.

This package provides input sanitization, validation, and security checks
for all user-provided data to prevent injection attacks and ensure system safety.
"""

from .input_sanitizer import InputSanitizer

__all__ = ["InputSanitizer"]