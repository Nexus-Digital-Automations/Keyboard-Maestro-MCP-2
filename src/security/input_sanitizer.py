"""
Input sanitization and validation for security.

This module provides comprehensive input sanitization to prevent injection attacks,
validate user inputs, and ensure all data meets security requirements before processing.
"""

import re
import html
from typing import List, Set
from src.core.either import Either
from src.core.errors import SecurityError, ValidationError

class InputSanitizer:
    """
    Security-focused input sanitization with comprehensive validation.
    
    Provides defense against:
    - Script injection attacks
    - Command injection
    - Path traversal
    - XSS attacks
    - ReDoS attacks
    """
    
    # Dangerous patterns that indicate potential injection attacks
    SCRIPT_INJECTION_PATTERNS = [
        r'<script[^>]*>',
        r'javascript:',
        r'eval\s*\(',
        r'exec\s*\(',
        r'system\s*\(',
        r'shell_exec\s*\(',
        r'passthru\s*\(',
        r'file_get_contents\s*\(',
        r'include\s*\(',
        r'require\s*\(',
        r'__import__\s*\(',
        r'getattr\s*\(',
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r'[;&|`$]',  # Command separators and execution
        r'\$\([^)]*\)',  # Command substitution
        r'`[^`]*`',  # Backtick execution
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./',
        r'\.\.[/\\]',
        r'[/\\]\.\./',
        r'[/\\]\.\.[/\\]',
    ]
    
    def __init__(self):
        self.strict_mode = False
    
    def sanitize_macro_identifier(self, identifier: str) -> Either[SecurityError, str]:
        """
        Sanitize macro identifier (name or UUID).
        
        Args:
            identifier: Macro name or UUID string
            
        Returns:
            Either containing sanitized identifier or security error
        """
        if not identifier or len(identifier.strip()) == 0:
            return Either.left(SecurityError("EMPTY_IDENTIFIER", "Macro identifier cannot be empty"))
        
        # Remove leading/trailing whitespace
        clean_id = identifier.strip()
        
        # Check length constraints
        if len(clean_id) > 255:
            return Either.left(SecurityError("IDENTIFIER_TOO_LONG", "Macro identifier exceeds 255 characters"))
        
        # Check for dangerous patterns
        security_check = self._check_security_patterns(clean_id, "macro_identifier")
        if security_check.is_left():
            return security_check
        
        # UUID pattern check
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        if re.match(uuid_pattern, clean_id):
            return Either.right(clean_id)
        
        # Macro name pattern check (allow alphanumeric, spaces, hyphens, dots, underscores)
        name_pattern = r'^[a-zA-Z0-9_\s\-\.]+$'
        if re.match(name_pattern, clean_id):
            return Either.right(clean_id)
        
        return Either.left(SecurityError("INVALID_IDENTIFIER_FORMAT", 
                                       "Macro identifier contains invalid characters"))
    
    def sanitize_text_content(self, text: str, strict_mode: bool = False) -> Either[SecurityError, str]:
        """
        Sanitize text content for safe processing.
        
        Args:
            text: Input text to sanitize
            strict_mode: Whether to apply strict validation
            
        Returns:
            Either containing sanitized text or security error
        """
        if text is None:
            return Either.right("")
        
        # Length validation
        max_length = 1000 if strict_mode else 10000
        if len(text) > max_length:
            return Either.left(SecurityError("TEXT_TOO_LONG", f"Text exceeds {max_length} characters"))
        
        # Security pattern checks
        security_check = self._check_security_patterns(text, "text_content")
        if security_check.is_left():
            return security_check
        
        # HTML escape for safety
        sanitized = html.escape(text)
        
        # Remove null bytes and control characters (except common whitespace)
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return Either.right(sanitized)
    
    def sanitize_file_path(self, path: str) -> Either[SecurityError, str]:
        """
        Sanitize file path to prevent traversal attacks.
        
        Args:
            path: File path to sanitize
            
        Returns:
            Either containing sanitized path or security error
        """
        if not path or len(path.strip()) == 0:
            return Either.left(SecurityError("EMPTY_PATH", "File path cannot be empty"))
        
        clean_path = path.strip()
        
        # Check for path traversal patterns
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, clean_path):
                return Either.left(SecurityError("PATH_TRAVERSAL", 
                                               f"Path traversal detected: {pattern}"))
        
        # Check for absolute paths that might be dangerous
        dangerous_prefixes = [
            '/System', '/usr/bin', '/usr/sbin', '/bin', '/sbin',
            '/private/etc', '/Library/Keychains', '/var/root',
            'C:\\Windows', 'C:\\System32'
        ]
        
        for prefix in dangerous_prefixes:
            if clean_path.startswith(prefix):
                return Either.left(SecurityError("FORBIDDEN_PATH", 
                                               f"Access denied to protected path: {prefix}"))
        
        # Additional security checks
        security_check = self._check_security_patterns(clean_path, "file_path")
        if security_check.is_left():
            return security_check
        
        return Either.right(clean_path)
    
    def sanitize_regex_pattern(self, pattern: str) -> Either[SecurityError, str]:
        """
        Sanitize regex pattern to prevent ReDoS attacks.
        
        Args:
            pattern: Regex pattern to sanitize
            
        Returns:
            Either containing validated pattern or security error
        """
        if not pattern:
            return Either.left(SecurityError("EMPTY_PATTERN", "Regex pattern cannot be empty"))
        
        # Length check
        if len(pattern) > 500:
            return Either.left(SecurityError("PATTERN_TOO_LONG", "Regex pattern too long (max 500 chars)"))
        
        # Check for dangerous regex patterns that could cause ReDoS
        dangerous_regex_patterns = [
            r'\(\?\#',      # Comment groups
            r'\(\?\>',      # Atomic groups
            r'\(\?\<',      # Lookbehind
            r'\*\+',        # Nested quantifiers
            r'\+\*',        # Nested quantifiers
            r'\{\d{4,}',    # Large repetition counts
            r'\([^)]*\)\*\([^)]*\)\*',  # Nested repetitions
        ]
        
        for danger_pattern in dangerous_regex_patterns:
            if re.search(danger_pattern, pattern):
                return Either.left(SecurityError("DANGEROUS_REGEX", 
                                               f"Potentially dangerous regex pattern: {danger_pattern}"))
        
        # Test compilation
        try:
            re.compile(pattern)
        except re.error as e:
            return Either.left(SecurityError("INVALID_REGEX", f"Invalid regex pattern: {str(e)}"))
        
        return Either.right(pattern)
    
    def sanitize_variable_name(self, name: str) -> Either[SecurityError, str]:
        """
        Sanitize variable name for KM variables.
        
        Args:
            name: Variable name to sanitize
            
        Returns:
            Either containing sanitized name or security error
        """
        if not name or len(name.strip()) == 0:
            return Either.left(SecurityError("EMPTY_VARIABLE_NAME", "Variable name cannot be empty"))
        
        clean_name = name.strip()
        
        # Length check
        if len(clean_name) > 255:
            return Either.left(SecurityError("VARIABLE_NAME_TOO_LONG", "Variable name too long (max 255 chars)"))
        
        # Format validation (alphanumeric and underscores only)
        if not re.match(r'^[a-zA-Z0-9_]+$', clean_name):
            return Either.left(SecurityError("INVALID_VARIABLE_NAME", 
                                           "Variable name must contain only alphanumeric characters and underscores"))
        
        # Security checks
        security_check = self._check_security_patterns(clean_name, "variable_name")
        if security_check.is_left():
            return security_check
        
        return Either.right(clean_name)
    
    def _check_security_patterns(self, text: str, context: str) -> Either[SecurityError, None]:
        """
        Check text for security-related patterns.
        
        Args:
            text: Text to check
            context: Context of the check for error reporting
            
        Returns:
            Either indicating success or security violation
        """
        text_lower = text.lower()
        
        # Check for script injection patterns
        for pattern in self.SCRIPT_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return Either.left(SecurityError("SCRIPT_INJECTION", 
                                               f"Potential script injection in {context}: {pattern}"))
        
        # Check for command injection patterns
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, text):
                return Either.left(SecurityError("COMMAND_INJECTION", 
                                               f"Potential command injection in {context}: {pattern}"))
        
        # Check for suspicious keywords
        suspicious_keywords = [
            'password', 'passwd', 'secret', 'token', 'key', 'auth',
            'admin', 'root', 'sudo', 'su', 'chmod', 'chown'
        ]
        
        # Only flag if appears to be in suspicious context
        for keyword in suspicious_keywords:
            if keyword in text_lower and any(char in text for char in ['=', ':', ';']):
                return Either.left(SecurityError("SUSPICIOUS_CONTENT", 
                                               f"Suspicious content detected in {context}: {keyword}"))
        
        return Either.right(None)
    
    def validate_url(self, url: str) -> Either[SecurityError, str]:
        """
        Validate and sanitize URL.
        
        Args:
            url: URL to validate
            
        Returns:
            Either containing validated URL or security error
        """
        if not url or len(url.strip()) == 0:
            return Either.left(SecurityError("EMPTY_URL", "URL cannot be empty"))
        
        clean_url = url.strip()
        
        # Length check
        if len(clean_url) > 2000:
            return Either.left(SecurityError("URL_TOO_LONG", "URL too long (max 2000 chars)"))
        
        # Allowed schemes
        allowed_schemes = ['http', 'https', 'file']
        scheme_pattern = r'^([a-zA-Z][a-zA-Z0-9+.-]*):\/\/'
        
        scheme_match = re.match(scheme_pattern, clean_url)
        if not scheme_match:
            return Either.left(SecurityError("INVALID_URL_SCHEME", "URL must have valid scheme"))
        
        scheme = scheme_match.group(1).lower()
        if scheme not in allowed_schemes:
            return Either.left(SecurityError("FORBIDDEN_URL_SCHEME", 
                                           f"URL scheme '{scheme}' not allowed"))
        
        # Security checks
        security_check = self._check_security_patterns(clean_url, "url")
        if security_check.is_left():
            return security_check
        
        return Either.right(clean_url)