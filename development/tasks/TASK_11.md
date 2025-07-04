# TASK_11: km_clipboard_manager - Clipboard Operations

**Created By**: Agent_ADDER+ (High-Impact Tool Implementation) | **Priority**: HIGH | **Duration**: 3 hours
**Technique Focus**: Defensive Programming + Data Security + Property-Based Testing + Memory Management
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED ✅ (MCP Tool Registered Successfully)
**Assigned**: Agent_2 
**Dependencies**: TASK_10 (km_create_macro foundation)
**Blocking**: None (standalone clipboard functionality)
**Completion**: All clipboard operations implemented and registered in main.py:255-300

## 📖 Required Reading (Complete before starting)
- [x] **development/protocols/KM_MCP.md**: km_clipboard_manager specification (lines 536-549)
- [x] **src/integration/km_client.py**: AppleScript integration patterns for system operations
- [x] **src/core/types.py**: Security types and validation patterns
- [x] **macOS Clipboard API**: Understanding pasteboard operations and security
- [x] **tests/TESTING.md**: Current test framework and security testing requirements

## 🎯 Implementation Overview
Create a comprehensive clipboard management system that provides AI assistants with full clipboard control including current content access, history management, named clipboards for persistent storage, and multiple format support with robust security validation.

<thinking>
Clipboard operations are essential for text processing workflows:
1. Security Critical: Clipboard may contain sensitive data (passwords, tokens, personal info)
2. Format Support: Text, images, files, URLs with proper type validation
3. History Management: Access to clipboard history with size limits and cleanup
4. Named Clipboards: Persistent storage for workflow data between operations
5. Memory Management: Efficient handling of large clipboard content (images, files)
6. Privacy Protection: Option to exclude sensitive content from history
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Core Clipboard Infrastructure
- [ ] **Clipboard types**: Define ClipboardContent, ClipboardHistory, NamedClipboard types
- [ ] **Security validation**: Content scanning, sensitive data detection, size limits
- [ ] **Format handling**: Text, image, file, URL format support with validation
- [ ] **Memory management**: Efficient storage and cleanup for large content

### Phase 2: AppleScript & System Integration
- [ ] **Clipboard access**: AppleScript commands for reading/writing clipboard content
- [ ] **History management**: Access to Keyboard Maestro's clipboard history (200 items)
- [ ] **Named clipboards**: Persistent clipboard storage with naming and organization
- [ ] **Format detection**: Automatic content type detection and validation

### Phase 3: Security & Privacy Features
- [ ] **Sensitive data detection**: Identify and handle passwords, tokens, personal data
- [ ] **Content filtering**: Option to exclude sensitive content from operations
- [ ] **Size validation**: Prevent memory issues with oversized content
- [ ] **Access logging**: Audit trail for clipboard operations

### Phase 4: MCP Tool Integration
- [ ] **Tool implementation**: km_clipboard_manager MCP tool with comprehensive operations
- [ ] **Operation modes**: get, set, get_history, manage_named clipboards
- [ ] **Response formatting**: Secure responses with content preview and metadata
- [ ] **Testing integration**: Property-based tests for all clipboard scenarios

## 🔧 Implementation Files & Specifications

### New Files to Create:

#### src/clipboard/clipboard_manager.py - Core Clipboard Operations
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import re

from ..core.types import Duration
from ..core.contracts import require, ensure

class ClipboardFormat(Enum):
    """Supported clipboard content formats."""
    TEXT = "text"
    IMAGE = "image" 
    FILE = "file"
    URL = "url"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class ClipboardContent:
    """Type-safe clipboard content representation."""
    content: Union[str, bytes]
    format: ClipboardFormat
    size_bytes: int
    timestamp: float
    is_sensitive: bool = False
    
    @require(lambda self: self.size_bytes >= 0)
    @require(lambda self: self.size_bytes <= 100_000_000)  # 100MB limit
    def __post_init__(self):
        pass
    
    def preview(self, max_length: int = 50) -> str:
        """Generate safe preview of clipboard content."""
        if self.is_sensitive:
            return "[SENSITIVE CONTENT HIDDEN]"
        if self.format == ClipboardFormat.TEXT:
            return str(self.content)[:max_length] + ("..." if len(str(self.content)) > max_length else "")
        return f"[{self.format.value.upper()} - {self.size_bytes} bytes]"

class ClipboardManager:
    """Secure clipboard operations with history and named clipboard support."""
    
    @require(lambda content: isinstance(content, str) and len(content) <= 1_000_000)
    @ensure(lambda result: result.is_right() or result.get_left().code in ["SECURITY_ERROR", "SIZE_ERROR"])
    async def set_clipboard(self, content: str) -> Either[KMError, bool]:
        """Set clipboard content with security validation."""
        pass
    
    @ensure(lambda result: result.is_right() or result.get_left().code == "ACCESS_ERROR")
    async def get_clipboard(self) -> Either[KMError, ClipboardContent]:
        """Get current clipboard content with format detection."""
        pass
    
    @require(lambda index: index >= 0 and index < 200)
    async def get_history_item(self, index: int) -> Either[KMError, ClipboardContent]:
        """Get item from clipboard history with bounds checking."""
        pass
    
    def _detect_sensitive_content(self, content: str) -> bool:
        """Detect potentially sensitive content (passwords, tokens, etc.)."""
        pass
    
    def _detect_format(self, content: Any) -> ClipboardFormat:
        """Detect clipboard content format."""
        pass
```

#### src/clipboard/named_clipboards.py - Named Clipboard System
```python
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class NamedClipboard:
    """Named clipboard for persistent storage."""
    name: str
    content: ClipboardContent
    created_at: float
    accessed_at: float
    access_count: int = 0
    
    @require(lambda self: len(self.name) > 0 and len(self.name) <= 100)
    @require(lambda self: re.match(r'^[a-zA-Z0-9_\-\s]+$', self.name))
    def __post_init__(self):
        pass

class NamedClipboardManager:
    """Manage named clipboards with persistence and organization."""
    
    @require(lambda name: len(name) > 0 and len(name) <= 100)
    @ensure(lambda result: result.is_right() or result.get_left().code == "NAME_CONFLICT")
    async def create_named_clipboard(self, name: str, content: ClipboardContent) -> Either[KMError, bool]:
        """Create named clipboard with conflict detection."""
        pass
    
    async def list_named_clipboards(self) -> Either[KMError, List[NamedClipboard]]:
        """List all named clipboards with metadata."""
        pass
    
    async def delete_named_clipboard(self, name: str) -> Either[KMError, bool]:
        """Delete named clipboard with validation."""
        pass
```

#### src/server/tools/clipboard_tools.py - MCP Tool Implementation
```python
async def km_clipboard_manager(
    operation: Annotated[str, Field(
        description="Clipboard operation type",
        pattern=r"^(get|set|get_history|manage_named)$"
    )],
    clipboard_name: Annotated[Optional[str], Field(
        default=None,
        description="For named clipboards - clipboard name",
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-\s]*$"
    )] = None,
    history_index: Annotated[Optional[int], Field(
        default=None,
        description="0-based history position for get_history",
        ge=0,
        le=199
    )] = None,
    content: Annotated[Optional[str], Field(
        default=None,
        description="Content to set (for set operation)",
        max_length=1_000_000
    )] = None,
    format: Annotated[str, Field(
        default="text",
        description="Content format filter",
        pattern=r"^(text|image|file|url)$"
    )] = "text",
    include_sensitive: Annotated[bool, Field(
        default=False,
        description="Include potentially sensitive content"
    )] = False,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Comprehensive clipboard management with security and privacy protection.
    
    Operations:
    - get: Retrieve current clipboard content with format detection
    - set: Set clipboard content with security validation
    - get_history: Access clipboard history by index
    - manage_named: Create, list, or delete named clipboards
    
    Security Features:
    - Automatic sensitive content detection and filtering
    - Size limits to prevent memory issues
    - Content preview for large items
    - Access logging for audit trails
    
    Returns clipboard operations results with metadata and security status.
    """
    if ctx:
        await ctx.info(f"Performing clipboard operation: {operation}")
    
    try:
        clipboard_manager = ClipboardManager()
        
        if operation == "get":
            # Get current clipboard content
            pass
        elif operation == "set":
            # Set clipboard content with validation
            pass
        elif operation == "get_history":
            # Get historical clipboard item
            pass
        elif operation == "manage_named":
            # Manage named clipboards
            pass
        
    except Exception as e:
        # Comprehensive error handling
        pass
```

### Files to Enhance:

#### src/core/types.py - Add Clipboard Types
```python
# Add to existing types
ClipboardId = NewType('ClipboardId', str)
ClipboardHistory = NewType('ClipboardHistory', List[ClipboardContent])

class ClipboardPermission(Enum):
    """Clipboard access permission levels."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    HISTORY_ACCESS = "history_access"
    NAMED_CLIPBOARD = "named_clipboard"
```

#### src/integration/km_client.py - Add Clipboard Methods
```python
def get_clipboard_applescript(self) -> Either[KMError, str]:
    """Get clipboard content via AppleScript."""
    script = '''
    tell application "Keyboard Maestro Engine"
        try
            set clipboardText to (the clipboard as text)
            return clipboardText
        on error errorMessage
            return "ERROR: " & errorMessage
        end try
    end tell
    '''
    pass

def set_clipboard_applescript(self, content: str) -> Either[KMError, bool]:
    """Set clipboard content via AppleScript with escaping."""
    pass

def get_clipboard_history_applescript(self, index: int) -> Either[KMError, str]:
    """Get clipboard history item via AppleScript."""
    pass
```

## 🏗️ Modularity Strategy
- **src/clipboard/**: New directory for clipboard functionality (<250 lines each)
- **clipboard_manager.py**: Core operations and security (240 lines)
- **named_clipboards.py**: Named clipboard management (180 lines)  
- **src/server/tools/clipboard_tools.py**: MCP tool implementation (200 lines)
- **Enhance existing files**: Minimal additions to types.py and km_client.py

## 🔒 Security Implementation
1. **Sensitive Data Detection**: Pattern matching for passwords, tokens, credit cards, SSNs
2. **Content Filtering**: Option to exclude sensitive content from operations and history
3. **Size Validation**: Prevent memory exhaustion with large clipboard content
4. **Format Validation**: Verify content format matches expected type
5. **Access Logging**: Log all clipboard operations for security auditing
6. **Memory Safety**: Proper cleanup of sensitive content from memory

## 📊 Performance Targets
- **Clipboard Access**: <50ms for current content retrieval
- **History Access**: <100ms for history item retrieval
- **Content Setting**: <100ms for text content up to 1MB
- **Sensitive Detection**: <200ms for content scanning
- **Memory Usage**: <50MB for clipboard operations

## ✅ Success Criteria
- [ ] All advanced techniques implemented (defensive programming, property testing, security)
- [ ] Complete security validation with sensitive content detection
- [ ] Support for text, image, file, and URL clipboard formats
- [ ] Real Keyboard Maestro clipboard integration (no mock data)
- [ ] Named clipboard system with persistence and organization
- [ ] Comprehensive error handling with security event logging
- [ ] Property-based testing covers all clipboard scenarios including edge cases
- [ ] Performance meets sub-200ms response targets for most operations
- [ ] Integration with existing MCP framework and security model
- [ ] TESTING.md updated with clipboard security tests
- [ ] Full documentation with privacy and security guidelines

## 🎨 Usage Examples

### Basic Clipboard Operations
```python
# Get current clipboard content
result = await client.call_tool("km_clipboard_manager", {
    "operation": "get",
    "format": "text"
})

# Set clipboard content with security validation
result = await client.call_tool("km_clipboard_manager", {
    "operation": "set",
    "content": "Hello, clipboard!"
})
```

### Advanced History and Named Clipboards
```python
# Access clipboard history
result = await client.call_tool("km_clipboard_manager", {
    "operation": "get_history",
    "history_index": 5,
    "include_sensitive": False
})

# Create named clipboard for workflow
result = await client.call_tool("km_clipboard_manager", {
    "operation": "manage_named",
    "clipboard_name": "workflow_data",
    "content": "important workflow data"
})
```

## 🧪 Testing Strategy
- **Property-Based Testing**: Random content with various formats and sizes
- **Security Testing**: Inject sensitive patterns, test filtering effectiveness
- **Performance Testing**: Large content handling, memory usage validation
- **Privacy Testing**: Verify sensitive content is properly filtered
- **Integration Testing**: Real clipboard operations with Keyboard Maestro
- **Edge Case Testing**: Empty clipboard, corrupted content, format mismatches