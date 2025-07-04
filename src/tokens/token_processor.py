"""
Core Token Processing Engine for Keyboard Maestro MCP

Provides secure token processing with comprehensive validation, context evaluation,
and injection prevention while maintaining proper token syntax handling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable
from enum import Enum
import re
import time
import os
import platform
import subprocess
from datetime import datetime

from ..core.contracts import require, ensure
from ..core.types import Duration
from ..integration.km_client import Either, KMError


class TokenType(Enum):
    """Types of Keyboard Maestro tokens."""
    VARIABLE = "variable"           # %Variable%name%
    SYSTEM = "system"              # %CurrentUser%, %FrontWindowName%
    CALCULATION = "calculation"     # %Calculate%expression%
    DATE_TIME = "datetime"         # %ICUDateTime%format%
    CLIPBOARD = "clipboard"        # %CurrentClipboard%
    APPLICATION = "application"    # %Application%bundle_id%
    UNKNOWN = "unknown"            # Unrecognized token


class ProcessingContext(Enum):
    """Context for token processing with security implications."""
    TEXT = "text"              # Plain text context
    CALCULATION = "calculation" # Mathematical expression context
    REGEX = "regex"            # Regular expression context
    FILENAME = "filename"      # File name context
    URL = "url"               # URL context


@dataclass(frozen=True)
class TokenExpression:
    """Type-safe token expression with comprehensive validation."""
    text: str
    context: ProcessingContext = ProcessingContext.TEXT
    variables: Dict[str, str] = field(default_factory=dict)
    
    @require(lambda self: len(self.text) > 0 and len(self.text) <= 10000, "Text must be 1-10000 characters")
    @require(lambda self: self._is_safe_token_expression(self.text), "Token expression contains unsafe patterns")
    def __post_init__(self):
        pass
    
    def _is_safe_token_expression(self, text: str) -> bool:
        """Validate token expression is safe for processing."""
        # Check for dangerous patterns that could lead to code injection
        dangerous_patterns = [
            r'%Execute\s*Shell\s*Script%',  # Shell execution
            r'%Execute\s*AppleScript%.*(?:do\s+shell\s+script|system\s+events)',  # Dangerous AppleScript
            r'%.*(?:password|secret|key|token|auth|credential).*%',  # Sensitive data access
            r'%.*(?:sudo|rm\s+-rf|format|delete|drop|truncate).*%',  # Dangerous commands
            r'%.*(?:eval|exec|import|__import__).*%',  # Code execution
            r'%.*(?:file://|http://|https://|ftp://).*%',  # URL schemes in unexpected contexts
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # Context-specific validation
        if self.context == ProcessingContext.FILENAME:
            # Additional filename-specific checks
            if re.search(r'[<>:"|?*]', text):
                return False
        
        return True


@dataclass(frozen=True)
class TokenProcessingResult:
    """Result of token processing with comprehensive metadata."""
    original_text: str
    processed_text: str
    tokens_found: List[str]
    substitutions_made: int
    processing_time: float
    context: ProcessingContext
    security_warnings: List[str] = field(default_factory=list)
    
    def has_changes(self) -> bool:
        """Check if any tokens were processed."""
        return self.original_text != self.processed_text
    
    def has_security_issues(self) -> bool:
        """Check if security warnings were generated."""
        return len(self.security_warnings) > 0


class TokenProcessor:
    """Secure token processing with KM integration and comprehensive validation."""
    
    def __init__(self):
        self.system_tokens = self._initialize_system_tokens()
        self._processing_stats = {"total_processed": 0, "errors": 0, "security_violations": 0}
    
    @require(lambda self, expression: expression.text != "", "Expression text cannot be empty")
    @ensure(lambda self, expression, result: result.is_right() or result.get_left().code in ["TOKEN_ERROR", "SECURITY_ERROR", "VALIDATION_ERROR"], "Must return valid Either with expected error codes")
    async def process_tokens(self, expression: TokenExpression) -> Either[KMError, TokenProcessingResult]:
        """Process tokens with comprehensive security validation and context awareness."""
        start_time = time.time()
        security_warnings = []
        
        try:
            # Parse tokens from the text
            tokens = self._parse_tokens(expression.text)
            
            if not tokens:
                # No tokens found, return original text
                processing_time = time.time() - start_time
                result = TokenProcessingResult(
                    original_text=expression.text,
                    processed_text=expression.text,
                    tokens_found=[],
                    substitutions_made=0,
                    processing_time=processing_time,
                    context=expression.context,
                    security_warnings=[]
                )
                return Either.right(result)
            
            # Process each token
            processed_text = expression.text
            substitutions_made = 0
            
            for token_info in tokens:
                token_result = await self._process_single_token(
                    token_info, 
                    expression.variables, 
                    expression.context
                )
                
                if token_result.is_right():
                    token_value, warnings = token_result.get_right()
                    security_warnings.extend(warnings)
                    
                    # Replace token in text
                    processed_text = processed_text.replace(
                        token_info['full_match'], 
                        token_value, 
                        1  # Replace only first occurrence to maintain order
                    )
                    substitutions_made += 1
                else:
                    # Log error but continue processing other tokens
                    error = token_result.get_left()
                    security_warnings.append(f"Token '{token_info['full_match']}' failed: {error.message}")
            
            processing_time = time.time() - start_time
            self._processing_stats["total_processed"] += 1
            
            if security_warnings:
                self._processing_stats["security_violations"] += 1
            
            result = TokenProcessingResult(
                original_text=expression.text,
                processed_text=processed_text,
                tokens_found=[token['full_match'] for token in tokens],
                substitutions_made=substitutions_made,
                processing_time=processing_time,
                context=expression.context,
                security_warnings=security_warnings
            )
            
            return Either.right(result)
            
        except Exception as e:
            self._processing_stats["errors"] += 1
            return Either.left(KMError.execution_error(
                f"Token processing failed: {str(e)}",
                details={"expression": expression.text[:100]}
            ))
    
    def _initialize_system_tokens(self) -> Dict[str, Callable[[], str]]:
        """Initialize system token resolvers with security boundaries."""
        return {
            'CurrentUser': self._get_current_user,
            'CurrentDate': self._get_current_date,
            'CurrentTime': self._get_current_time,
            'LongDate': self._get_long_date,
            'ShortDate': self._get_short_date,
            'Time': self._get_time_12h,
            'SystemVersion': self._get_system_version,
            'ComputerName': self._get_computer_name,
            'FrontWindowName': self._get_front_window_name,
            'CurrentApplication': self._get_current_application,
            'CurrentClipboard': self._get_current_clipboard_preview,
        }
    
    def _parse_tokens(self, text: str) -> List[Dict[str, Any]]:
        """Parse tokens from text with type identification and security validation."""
        token_pattern = r'%([^%]+)%'
        tokens = []
        
        for match in re.finditer(token_pattern, text):
            token_content = match.group(1)
            
            # Security check for token content
            if not self._is_safe_token_content(token_content):
                continue  # Skip unsafe tokens
            
            token_info = {
                'full_match': match.group(0),
                'content': token_content,
                'start': match.start(),
                'end': match.end(),
                'type': self._identify_token_type(token_content)
            }
            tokens.append(token_info)
        
        return tokens
    
    def _is_safe_token_content(self, content: str) -> bool:
        """Validate individual token content for security."""
        # Check length
        if len(content) > 500:
            return False
        
        # Check for injection patterns
        dangerous_patterns = [
            r'[;&|`$(){}[\]]',  # Shell metacharacters
            r'\\[nt]',          # Escape sequences
            r'(?:sudo|rm|format|delete)',  # Dangerous commands
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        
        return True
    
    def _identify_token_type(self, content: str) -> TokenType:
        """Identify the type of token based on content with security awareness."""
        content_lower = content.lower()
        
        if content.startswith('Calculate'):
            return TokenType.CALCULATION
        elif content.startswith('ICUDateTime'):
            return TokenType.DATE_TIME
        elif content in self.system_tokens:
            return TokenType.SYSTEM
        elif content.startswith('Variable'):
            return TokenType.VARIABLE
        elif 'clipboard' in content_lower:
            return TokenType.CLIPBOARD
        elif 'application' in content_lower:
            return TokenType.APPLICATION
        else:
            return TokenType.UNKNOWN
    
    async def _process_single_token(
        self, 
        token_info: Dict[str, Any], 
        variables: Dict[str, str], 
        context: ProcessingContext
    ) -> Either[KMError, tuple[str, List[str]]]:
        """Process a single token with context-specific handling."""
        warnings = []
        token_type = token_info['type']
        content = token_info['content']
        
        try:
            if token_type == TokenType.SYSTEM:
                result = self._resolve_system_token(content)
                if result is None:
                    return Either.left(KMError.not_found_error(f"System token '{content}' not found"))
                return Either.right((result, warnings))
            
            elif token_type == TokenType.VARIABLE:
                return self._process_variable_token(content, variables, warnings)
            
            elif token_type == TokenType.CALCULATION:
                return self._process_calculation_token(content, context, warnings)
            
            elif token_type == TokenType.DATE_TIME:
                return self._process_datetime_token(content, warnings)
            
            elif token_type == TokenType.CLIPBOARD:
                return self._process_clipboard_token(content, warnings)
            
            else:
                warnings.append(f"Unknown token type: {content}")
                return Either.right((token_info['full_match'], warnings))
            
        except Exception as e:
            return Either.left(KMError.execution_error(
                f"Failed to process token '{content}': {str(e)}"
            ))
    
    def _resolve_system_token(self, token_name: str) -> Optional[str]:
        """Resolve system token to current value with error handling."""
        resolver = self.system_tokens.get(token_name)
        if resolver:
            try:
                return resolver()
            except Exception:
                return None
        return None
    
    def _process_variable_token(
        self, 
        content: str, 
        variables: Dict[str, str], 
        warnings: List[str]
    ) -> Either[KMError, tuple[str, List[str]]]:
        """Process variable token with scope resolution."""
        # Extract variable name from %Variable%name% format
        if content.startswith('Variable'):
            var_name = content[8:] if len(content) > 8 else ""
            
            if var_name in variables:
                value = variables[var_name]
                # Sanitize variable value for security
                if len(value) > 1000:
                    warnings.append(f"Variable '{var_name}' value truncated (too long)")
                    value = value[:1000] + "..."
                return Either.right((value, warnings))
            else:
                return Either.left(KMError.not_found_error(f"Variable '{var_name}' not found"))
        
        return Either.left(KMError.validation_error(f"Invalid variable token format: {content}"))
    
    def _process_calculation_token(
        self, 
        content: str, 
        context: ProcessingContext, 
        warnings: List[str]
    ) -> Either[KMError, tuple[str, List[str]]]:
        """Process calculation token with security validation."""
        if not content.startswith('Calculate'):
            return Either.left(KMError.validation_error("Invalid calculation token format"))
        
        # For now, return placeholder - actual calculation would need the calculator module
        warnings.append("Calculation tokens require KM engine integration")
        return Either.right(("0", warnings))
    
    def _process_datetime_token(
        self, 
        content: str, 
        warnings: List[str]
    ) -> Either[KMError, tuple[str, List[str]]]:
        """Process datetime token with format validation."""
        if content.startswith('ICUDateTime'):
            # Extract format string
            format_str = content[11:] if len(content) > 11 else ""
            try:
                # Simple datetime formatting - could be enhanced with ICU formats
                current_time = datetime.now()
                if not format_str:
                    result = current_time.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Basic format mapping - could be enhanced
                    result = current_time.strftime('%Y-%m-%d %H:%M:%S')
                return Either.right((result, warnings))
            except Exception as e:
                return Either.left(KMError.execution_error(f"DateTime formatting failed: {str(e)}"))
        
        return Either.left(KMError.validation_error("Invalid datetime token format"))
    
    def _process_clipboard_token(
        self, 
        content: str, 
        warnings: List[str]
    ) -> Either[KMError, tuple[str, List[str]]]:
        """Process clipboard token with privacy protection."""
        # For security, provide preview only
        warnings.append("Clipboard access restricted - showing preview only")
        return Either.right(("[Clipboard Content]", warnings))
    
    # System token resolver methods
    def _get_current_user(self) -> str:
        """Get current system user safely."""
        return os.getenv('USER', 'unknown')
    
    def _get_current_date(self) -> str:
        """Get current date in ISO format."""
        return datetime.now().strftime('%Y-%m-%d')
    
    def _get_current_time(self) -> str:
        """Get current time in 24-hour format."""
        return datetime.now().strftime('%H:%M:%S')
    
    def _get_long_date(self) -> str:
        """Get long date format."""
        return datetime.now().strftime('%A, %B %d, %Y')
    
    def _get_short_date(self) -> str:
        """Get short date format."""
        return datetime.now().strftime('%m/%d/%Y')
    
    def _get_time_12h(self) -> str:
        """Get time in 12-hour format."""
        return datetime.now().strftime('%I:%M:%S %p')
    
    def _get_system_version(self) -> str:
        """Get macOS system version safely."""
        try:
            return platform.mac_ver()[0]
        except Exception:
            return "unknown"
    
    def _get_computer_name(self) -> str:
        """Get computer name safely."""
        try:
            return platform.node()
        except Exception:
            return "unknown"
    
    def _get_front_window_name(self) -> Optional[str]:
        """Get front window name via safe AppleScript."""
        script = '''
        tell application "System Events"
            try
                set frontApp to first application process whose frontmost is true
                set windowName to name of front window of frontApp
                return windowName
            on error
                return ""
            end try
        end tell
        '''
        
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "Unknown Window"
        except Exception:
            return "Unknown Window"
    
    def _get_current_application(self) -> Optional[str]:
        """Get current application name safely."""
        script = '''
        tell application "System Events"
            try
                set frontApp to first application process whose frontmost is true
                return name of frontApp
            on error
                return ""
            end try
        end tell
        '''
        
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "Unknown Application"
        except Exception:
            return "Unknown Application"
    
    def _get_current_clipboard_preview(self) -> str:
        """Get safe clipboard preview."""
        return "[Clipboard Preview - Access Restricted]"
    
    async def process_tokens_in_text(self, text: str, variables: Optional[Dict[str, str]] = None) -> str:
        """Simple text-based token processing interface for web requests."""
        if not text or not text.strip():
            return text
        
        try:
            expression = TokenExpression(
                text=text,
                context=ProcessingContext.TEXT,
                variables=variables or {}
            )
            
            result = await self.process_tokens(expression)
            
            if result.is_right():
                processing_result = result.get_right()
                return processing_result.processed_text
            else:
                # Return original text if processing fails
                return text
                
        except Exception:
            # Return original text if expression creation fails
            return text
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics for monitoring."""
        return self._processing_stats.copy()