# TASK_19: km_token_processor - Token System Integration

**Created By**: Agent_ADDER+ (High-Impact Tool Implementation) | **Priority**: MEDIUM | **Duration**: 3 hours
**Technique Focus**: Token Security + Context Validation + String Processing + Variable Substitution
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED ✅ (MCP Tool Registered Successfully)
**Assigned**: Agent_11
**Dependencies**: TASK_10 (macro creation for token-based workflows)
**Blocking**: None (standalone token processing functionality)

## 📖 Required Reading (Complete before starting)
- [x] **development/protocols/KM_MCP.md**: km_token_processor specification (lines 930-941)
- [x] **src/creation/**: Macro creation patterns from TASK_10
- [x] **Keyboard Maestro Token System**: Understanding KM tokens like %CurrentUser%, %FrontWindowName%
- [x] **src/core/types.py**: String processing and validation types
- [x] **tests/TESTING.md**: String processing and token security testing

## 🎯 Implementation Overview
Create a secure token processing system that enables AI assistants to process Keyboard Maestro tokens with variable substitution, context evaluation, and security validation while preventing injection attacks and maintaining proper token syntax.

<thinking>
Token processing is essential for dynamic content:
1. Token Security: Prevent injection through malicious token content
2. Context Evaluation: Resolve tokens based on current system state
3. Variable Substitution: Replace variables with actual values safely
4. Format Validation: Ensure token syntax is correct and safe
5. System Integration: Work with KM's token processing engine
6. Performance: Efficient processing of complex token expressions
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Core Token Processing Infrastructure - COMPLETED ✅
- [x] **Token types**: Define TokenExpression, TokenType, ProcessingContext, SubstitutionResult types ✅
- [x] **Security validation**: Token content sanitization, injection prevention, syntax validation ✅
- [x] **Parser implementation**: Token syntax parser with security boundaries ✅
- [x] **Context management**: System state capture for token resolution ✅

### Phase 2: Token Resolution & KM Integration - COMPLETED ✅
- [x] **System tokens**: Process built-in KM tokens like %CurrentUser%, %FrontWindowName% ✅
- [x] **Variable tokens**: Handle variable tokens with proper scope resolution ✅
- [x] **Calculation tokens**: Process calculation tokens with security validation ✅
- [x] **KM engine integration**: Interface with Keyboard Maestro's token processing ✅

### Phase 3: Security & Validation - COMPLETED ✅
- [x] **Token sanitization**: Clean token expressions for safe processing ✅
- [x] **Context validation**: Verify token context is safe and appropriate ✅
- [x] **Result validation**: Ensure processed results are within safety bounds ✅
- [x] **Injection prevention**: Prevent code injection through token manipulation ✅

### Phase 4: MCP Tool Integration - COMPLETED ✅
- [x] **Tool implementation**: km_token_processor MCP tool with comprehensive token support ✅
- [x] **Processing modes**: text, calculation, regex contexts with appropriate validation ✅
- [x] **Response formatting**: Token processing results with security metadata ✅
- [x] **Testing integration**: Token security tests and processing validation ✅

## 🔧 Implementation Files & Specifications

### New Files to Create:

#### src/tokens/token_processor.py - Core Token Processing Engine
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import re

class TokenType(Enum):
    """Types of Keyboard Maestro tokens."""
    VARIABLE = "variable"           # %Variable%name%
    SYSTEM = "system"              # %CurrentUser%, %FrontWindowName%
    CALCULATION = "calculation"     # %Calculate%expression%
    DATE_TIME = "datetime"         # %ICUDateTime%format%
    CLIPBOARD = "clipboard"        # %CurrentClipboard%
    APPLICATION = "application"    # %Application%bundle_id%

class ProcessingContext(Enum):
    """Context for token processing."""
    TEXT = "text"              # Plain text context
    CALCULATION = "calculation" # Mathematical expression context
    REGEX = "regex"            # Regular expression context
    FILENAME = "filename"      # File name context
    URL = "url"               # URL context

@dataclass(frozen=True)
class TokenExpression:
    """Type-safe token expression."""
    text: str
    context: ProcessingContext = ProcessingContext.TEXT
    variables: Dict[str, str] = field(default_factory=dict)
    
    @require(lambda self: len(self.text) > 0 and len(self.text) <= 10000)
    @require(lambda self: self._is_safe_token_expression(self.text))
    def __post_init__(self):
        pass
    
    def _is_safe_token_expression(self, text: str) -> bool:
        """Validate token expression is safe for processing."""
        # Check for dangerous patterns
        dangerous_patterns = [
            r'%Execute\s*Shell\s*Script%',  # Shell execution
            r'%Execute\s*AppleScript%.*(?:do\s+shell\s+script|system\s+events)',  # Dangerous AppleScript
            r'%.*(?:password|secret|key|token).*%',  # Sensitive data access
            r'%.*(?:sudo|rm\s+-rf|format|delete).*%',  # Dangerous commands
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return True

@dataclass(frozen=True)
class TokenProcessingResult:
    """Result of token processing with metadata."""
    original_text: str
    processed_text: str
    tokens_found: List[str]
    substitutions_made: int
    processing_time: float
    context: ProcessingContext
    
    def has_changes(self) -> bool:
        """Check if any tokens were processed."""
        return self.original_text != self.processed_text

class TokenProcessor:
    """Secure token processing with KM integration."""
    
    def __init__(self):
        self.system_tokens = self._initialize_system_tokens()
    
    @require(lambda expression: expression.text != "")
    @ensure(lambda result: result.is_right() or result.get_left().code in ["TOKEN_ERROR", "SECURITY_ERROR"])
    async def process_tokens(self, expression: TokenExpression) -> Either[KMError, TokenProcessingResult]:
        """Process tokens with comprehensive security validation."""
        pass
    
    def _initialize_system_tokens(self) -> Dict[str, callable]:
        """Initialize system token resolvers."""
        return {
            'CurrentUser': self._get_current_user,
            'CurrentDate': self._get_current_date,
            'CurrentTime': self._get_current_time,
            'FrontWindowName': self._get_front_window_name,
            'CurrentApplication': self._get_current_application,
            'SystemVersion': self._get_system_version
        }
    
    def _parse_tokens(self, text: str) -> List[Dict[str, Any]]:
        """Parse tokens from text with type identification."""
        token_pattern = r'%([^%]+)%'
        tokens = []
        
        for match in re.finditer(token_pattern, text):
            token_content = match.group(1)
            token_info = {
                'full_match': match.group(0),
                'content': token_content,
                'start': match.start(),
                'end': match.end(),
                'type': self._identify_token_type(token_content)
            }
            tokens.append(token_info)
        
        return tokens
    
    def _identify_token_type(self, content: str) -> TokenType:
        """Identify the type of token based on content."""
        if content.startswith('Calculate'):
            return TokenType.CALCULATION
        elif content.startswith('ICUDateTime'):
            return TokenType.DATE_TIME
        elif content in self.system_tokens:
            return TokenType.SYSTEM
        elif content.startswith('Variable'):
            return TokenType.VARIABLE
        else:
            return TokenType.SYSTEM  # Default assumption
    
    def _resolve_system_token(self, token_name: str) -> Optional[str]:
        """Resolve system token to current value."""
        resolver = self.system_tokens.get(token_name)
        if resolver:
            try:
                return resolver()
            except Exception:
                return None
        return None
    
    def _get_current_user(self) -> str:
        """Get current system user."""
        import os
        return os.getenv('USER', 'unknown')
    
    def _get_current_date(self) -> str:
        """Get current date."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d')
    
    def _get_current_time(self) -> str:
        """Get current time."""
        from datetime import datetime
        return datetime.now().strftime('%H:%M:%S')
    
    def _get_front_window_name(self) -> Optional[str]:
        """Get front window name via AppleScript."""
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
            import subprocess
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def _get_current_application(self) -> Optional[str]:
        """Get current application name."""
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
            import subprocess
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def _get_system_version(self) -> str:
        """Get macOS system version."""
        try:
            import platform
            return platform.mac_ver()[0]
        except Exception:
            return "unknown"
```

#### src/tokens/km_token_integration.py - Keyboard Maestro Integration
```python
class KMTokenEngine:
    """Integration with Keyboard Maestro's token processing system."""
    
    async def process_with_km(self, text: str, context: ProcessingContext = ProcessingContext.TEXT) -> Either[KMError, str]:
        """Process tokens using KM's token processing engine."""
        # Build AppleScript to use KM's token processing
        context_param = self._context_to_km_parameter(context)
        
        script = f'''
        tell application "Keyboard Maestro Engine"
            try
                set result to process tokens "{self._escape_for_applescript(text)}" {context_param}
                return result as string
            on error errorMessage
                return "ERROR: " & errorMessage
            end try
        end tell
        '''
        
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return Either.left(KMError.execution_error(f"KM token processing failed: {result.stderr}"))
            
            output = result.stdout.strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:].strip()))
            
            return Either.right(output)
            
        except subprocess.TimeoutExpired:
            return Either.left(KMError.timeout_error("KM token processing timeout"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"KM token processing error: {str(e)}"))
    
    def _context_to_km_parameter(self, context: ProcessingContext) -> str:
        """Convert processing context to KM parameter."""
        context_map = {
            ProcessingContext.TEXT: "",
            ProcessingContext.CALCULATION: "for calculation",
            ProcessingContext.REGEX: "for regex",
            ProcessingContext.FILENAME: "for filename",
            ProcessingContext.URL: "for url"
        }
        return context_map.get(context, "")
    
    def _escape_for_applescript(self, text: str) -> str:
        """Escape text for safe AppleScript usage."""
        return text.replace('"', '\\"').replace('\\', '\\\\')
```

#### src/server/tools/token_tools.py - MCP Tool Implementation
```python
async def km_token_processor(
    text: Annotated[str, Field(
        description="Text containing Keyboard Maestro tokens",
        min_length=1,
        max_length=10000
    )],
    context: Annotated[str, Field(
        default="text",
        description="Processing context",
        pattern=r"^(text|calculation|regex|filename|url)$"
    )] = "text",
    variables: Annotated[Dict[str, str], Field(
        default_factory=dict,
        description="Variable values for token substitution"
    )] = {},
    use_km_engine: Annotated[bool, Field(
        default=True,
        description="Use Keyboard Maestro's token processing engine"
    )] = True,
    preview_only: Annotated[bool, Field(
        default=False,
        description="Preview tokens without processing"
    )] = False,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Process Keyboard Maestro tokens with comprehensive security and context support.
    
    Features:
    - Secure token parsing with injection prevention
    - Support for all major KM token types (system, variable, calculation, datetime)
    - Multiple processing contexts (text, calculation, regex, filename, URL)
    - Variable substitution with scope resolution
    - Integration with Keyboard Maestro's token processing engine
    - Preview mode for token analysis without execution
    
    Security:
    - Token content validation and sanitization
    - Prevention of dangerous token execution
    - Safe processing with bounded execution
    - Context-appropriate validation
    
    Returns processed text with token metadata and security validation results.
    """
    if ctx:
        await ctx.info(f"Processing tokens in text: {text[:50]}...")
    
    try:
        import time
        start_time = time.time()
        
        # Create token expression with validation
        token_expr = TokenExpression(
            text=text,
            context=ProcessingContext(context),
            variables=variables
        )
        
        if preview_only:
            # Just parse and return token information
            processor = TokenProcessor()
            tokens = processor._parse_tokens(text)
            
            return {
                "success": True,
                "preview": {
                    "original_text": text,
                    "tokens_found": [token['full_match'] for token in tokens],
                    "token_details": tokens,
                    "token_count": len(tokens),
                    "context": context
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "preview_mode": True
                }
            }
        
        # Choose processing method
        if use_km_engine:
            # Use Keyboard Maestro's token processing engine
            km_token = KMTokenEngine()
            km_result = await km_token.process_with_km(text, ProcessingContext(context))
            
            if km_result.is_left():
                # Fallback to local processor
                processor = TokenProcessor()
                process_result = await processor.process_tokens(token_expr)
            else:
                # Create result from KM processing
                processed_text = km_result.get_right()
                execution_time = time.time() - start_time
                
                # Parse tokens for metadata
                processor = TokenProcessor()
                tokens = processor._parse_tokens(text)
                
                process_result = Either.right(TokenProcessingResult(
                    original_text=text,
                    processed_text=processed_text,
                    tokens_found=[token['full_match'] for token in tokens],
                    substitutions_made=len(tokens),
                    processing_time=execution_time,
                    context=ProcessingContext(context)
                ))
        else:
            # Use local processor
            processor = TokenProcessor()
            process_result = await processor.process_tokens(token_expr)
        
        if process_result.is_left():
            error = process_result.get_left()
            return {
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": {"text": text[:100] + "..." if len(text) > 100 else text}
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        result = process_result.get_right()
        
        if ctx:
            await ctx.info(f"Token processing complete: {result.substitutions_made} substitutions made")
        
        return {
            "success": True,
            "processing": {
                "original_text": result.original_text,
                "processed_text": result.processed_text,
                "tokens_found": result.tokens_found,
                "substitutions_made": result.substitutions_made,
                "processing_time": result.processing_time,
                "context": result.context.value,
                "has_changes": result.has_changes(),
                "variables_used": variables
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "processing_id": str(uuid.uuid4()),
                "engine": "keyboard_maestro" if use_km_engine else "local"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "TOKEN_PROCESSING_ERROR",
                "message": str(e),
                "details": {"text": text[:100] + "..." if len(text) > 100 else text}
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat()
            }
        }
```

## ✅ Success Criteria - ALL COMPLETED ✅
- [x] Complete token processing system with security validation ✅
- [x] Support for all major KM token types (system, variable, calculation, datetime) ✅
- [x] Multiple processing contexts with appropriate validation ✅
- [x] Real Keyboard Maestro token engine integration ✅
- [x] Comprehensive security validation against token injection ✅
- [x] Property-based testing for token processing scenarios and edge cases ✅
- [x] Performance meets sub-500ms processing targets for most token expressions ✅
- [x] Integration with macro creation for token-based workflows ✅
- [x] TESTING.md updated with token processing and security tests ✅
- [x] Documentation with token reference and security guidelines ✅

## 🎨 Usage Examples

### Basic Token Processing
```python
# Process system tokens
result = await client.call_tool("km_token_processor", {
    "text": "Hello %CurrentUser%, today is %CurrentDate%",
    "context": "text"
})

# Process variable tokens
result = await client.call_tool("km_token_processor", {
    "text": "Processing file: %Variable%FileName%",
    "variables": {"FileName": "document.pdf"},
    "context": "filename"
})
```

### Advanced Token Operations
```python
# Process calculation tokens
result = await client.call_tool("km_token_processor", {
    "text": "Result: %Calculate%price * (1 + tax_rate)%",
    "variables": {"price": "100", "tax_rate": "0.08"},
    "context": "calculation"
})

# Preview tokens without processing
result = await client.call_tool("km_token_processor", {
    "text": "Window: %FrontWindowName% in %CurrentApplication%",
    "preview_only": True
})
```