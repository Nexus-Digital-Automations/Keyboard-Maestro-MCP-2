# TASK_13: km_file_operations - File System Automation

**Created By**: Agent_ADDER+ (High-Impact Tool Implementation) | **Priority**: MEDIUM | **Duration**: 3 hours
**Technique Focus**: Path Security + File Validation + Transaction Safety + Permission Management
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED ✅ (MCP Tool Registered Successfully)
**Assigned**: Agent_2
**Dependencies**: TASK_12 (app control for file app integration)
**Blocking**: None (standalone file system functionality)
**Completion**: All file operations implemented and registered in main.py:303-338

## 📖 Required Reading (Complete before starting)
- [x] **development/protocols/KM_MCP.md**: km_file_operations specification (lines 708-721)
- [x] **src/integration/km_client.py**: AppleScript patterns for system operations
- [x] **src/core/types.py**: Security validation and permission models
- [x] **macOS File System Security**: Understanding sandbox restrictions and permissions
- [x] **tests/TESTING.md**: File system testing requirements and security

## 🎯 Implementation Overview
Create a secure file system automation engine that enables AI assistants to perform file operations (copy, move, delete, rename) while maintaining strict security boundaries, path validation, and transaction safety for reliable workflow automation.

<thinking>
File operations are critical but security-sensitive:
1. Security Critical: Must validate all paths to prevent directory traversal and unauthorized access
2. Transaction Safety: Atomic operations with rollback capability for failed operations  
3. Permission Model: Respect system permissions and sandbox restrictions
4. Path Validation: Sanitize and validate all file paths against security policies
5. Error Recovery: Handle permission errors, disk space, locked files gracefully
6. Integration: Work with app control for opening files in specific applications
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Core File System Infrastructure
- [x] **File operation types**: Define FilePath, FileOperation, OperationResult, TransactionId types
- [x] **Security validation**: Path sanitization, directory traversal prevention, permission checking
- [x] **Transaction system**: Atomic operations with rollback capability for multi-step operations
- [x] **Permission management**: Verify system permissions for file system access

### Phase 2: AppleScript & System Integration
- [x] **File operations**: Copy, move, delete, rename operations via AppleScript/shell
- [x] **Directory management**: Create, remove directories with intermediate path creation
- [x] **Attribute management**: Get/set file attributes, permissions, metadata
- [x] **Error handling**: Comprehensive error recovery for file system failures

### Phase 3: Security & Safety Features
- [x] **Path validation**: Prevent directory traversal, validate against allowed directories
- [x] **Overwrite protection**: Configurable overwrite behavior with backup options
- [x] **Disk space checking**: Pre-operation validation of available space
- [x] **Atomic operations**: Transaction-safe multi-file operations with rollback

### Phase 4: MCP Tool Integration
- [x] **Tool implementation**: km_file_operations MCP tool with operation types
- [x] **Operation modes**: copy, move, delete, rename, create_folder, get_info
- [x] **Response formatting**: Operation results with metadata and security status
- [x] **Testing integration**: File system operation tests with security validation

## 🔧 Implementation Files & Specifications

### New Files to Create:

#### src/filesystem/file_operations.py - Core File System Operations
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pathlib import Path
import shutil
import os

from ..core.types import Duration
from ..core.contracts import require, ensure

class FileOperationType(Enum):
    """Supported file system operations."""
    COPY = "copy"
    MOVE = "move" 
    DELETE = "delete"
    RENAME = "rename"
    CREATE_FOLDER = "create_folder"
    GET_INFO = "get_info"

@dataclass(frozen=True)
class FilePath:
    """Type-safe file path with security validation."""
    path: str
    
    @require(lambda self: len(self.path) > 0 and len(self.path) <= 1000)
    @require(lambda self: not any(dangerous in self.path for dangerous in ["../", "..\\", "~/"]))
    def __post_init__(self):
        pass
    
    def is_safe_path(self) -> bool:
        """Validate path is safe for operations."""
        try:
            resolved = Path(self.path).resolve()
            # Check if path is within allowed directories
            return True  # Implement specific security policy
        except Exception:
            return False
    
    def exists(self) -> bool:
        """Check if path exists."""
        return Path(self.path).exists()

@dataclass(frozen=True)
class FileOperationRequest:
    """Type-safe file operation request."""
    operation: FileOperationType
    source_path: FilePath
    destination_path: Optional[FilePath] = None
    overwrite: bool = False
    create_intermediate: bool = False
    backup_existing: bool = False
    
    @require(lambda self: self.source_path.is_safe_path())
    @require(lambda self: not self.destination_path or self.destination_path.is_safe_path())
    def __post_init__(self):
        pass

class FileOperationManager:
    """Secure file operations with transaction safety and rollback."""
    
    @require(lambda request: request.source_path.exists() or request.operation == FileOperationType.CREATE_FOLDER)
    @ensure(lambda result: result.is_right() or result.get_left().code in ["PERMISSION_ERROR", "PATH_ERROR", "DISK_SPACE_ERROR"])
    async def execute_operation(self, request: FileOperationRequest) -> Either[KMError, Dict[str, Any]]:
        """Execute file operation with comprehensive validation and error handling."""
        pass
    
    @require(lambda source: source.is_safe_path())
    @require(lambda dest: dest.is_safe_path())
    async def copy_file(self, source: FilePath, destination: FilePath, overwrite: bool = False) -> Either[KMError, bool]:
        """Copy file with security validation and overwrite protection."""
        pass
    
    @require(lambda source: source.is_safe_path())
    @require(lambda dest: dest.is_safe_path())
    async def move_file(self, source: FilePath, destination: FilePath, overwrite: bool = False) -> Either[KMError, bool]:
        """Move file with atomic operation and rollback capability."""
        pass
    
    @require(lambda path: path.is_safe_path())
    async def delete_file(self, path: FilePath, secure_delete: bool = False) -> Either[KMError, bool]:
        """Delete file with optional secure deletion."""
        pass
    
    def _validate_permissions(self, path: FilePath, operation: FileOperationType) -> bool:
        """Validate user has permissions for operation on path."""
        pass
    
    def _check_disk_space(self, path: FilePath, required_bytes: int) -> bool:
        """Check available disk space for operation."""
        pass
    
    def _create_backup(self, path: FilePath) -> Optional[FilePath]:
        """Create backup of file before operation."""
        pass
```

#### src/filesystem/path_security.py - Path Security Validation
```python
from typing import Set, List
from pathlib import Path
import re

class PathSecurity:
    """Comprehensive path security validation and sanitization."""
    
    # Allowed directories for file operations (configurable)
    ALLOWED_DIRECTORIES = {
        Path.home() / "Documents",
        Path.home() / "Downloads", 
        Path.home() / "Desktop",
        Path("/tmp")
    }
    
    # Dangerous path patterns
    DANGEROUS_PATTERNS = [
        r"\.\.[\\/]",  # Directory traversal
        r"^[\\/]",     # Absolute root paths
        r"~[\\/]",     # Home directory shortcuts
        r"\$\{.*\}",   # Environment variable expansion
        r"`.*`",       # Command substitution
    ]
    
    @classmethod
    def validate_path(cls, path: str) -> bool:
        """Comprehensive path security validation."""
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, path):
                return False
        
        try:
            resolved_path = Path(path).resolve()
            
            # Check if path is within allowed directories
            for allowed_dir in cls.ALLOWED_DIRECTORIES:
                try:
                    resolved_path.relative_to(allowed_dir)
                    return True
                except ValueError:
                    continue
            
            return False
        except Exception:
            return False
    
    @classmethod
    def sanitize_path(cls, path: str) -> Optional[str]:
        """Sanitize path for safe operations."""
        # Remove dangerous characters and patterns
        sanitized = re.sub(r'[<>:"|?*]', '', path)
        sanitized = re.sub(r'\.\.[\\/]', '', sanitized)
        
        if cls.validate_path(sanitized):
            return sanitized
        return None
```

#### src/server/tools/file_operation_tools.py - MCP Tool Implementation
```python
async def km_file_operations(
    operation: Annotated[str, Field(
        description="File operation type",
        pattern=r"^(copy|move|delete|rename|create_folder|get_info)$"
    )],
    source_path: Annotated[str, Field(
        description="Source file or folder path",
        min_length=1,
        max_length=1000
    )],
    destination_path: Annotated[Optional[str], Field(
        default=None,
        description="Destination path for copy/move/rename operations",
        max_length=1000
    )] = None,
    overwrite: Annotated[bool, Field(
        default=False,
        description="Overwrite existing files"
    )] = False,
    create_intermediate: Annotated[bool, Field(
        default=False,
        description="Create missing intermediate directories"
    )] = False,
    backup_existing: Annotated[bool, Field(
        default=False,
        description="Create backup of existing files before overwrite"
    )] = False,
    secure_delete: Annotated[bool, Field(
        default=False,
        description="Secure deletion (multiple overwrite passes)"
    )] = False,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Secure file system operations with comprehensive validation and safety features.
    
    Operations:
    - copy: Copy files or directories with overwrite protection
    - move: Move files or directories with atomic operations
    - delete: Delete files or directories with optional secure deletion
    - rename: Rename files or directories in place
    - create_folder: Create directories with intermediate path creation
    - get_info: Get file/directory metadata and attributes
    
    Security Features:
    - Path validation and sanitization against directory traversal
    - Permission checking for all operations
    - Disk space validation before operations
    - Transaction safety with rollback capability
    - Backup creation for destructive operations
    
    Returns operation results with file metadata and security status.
    """
    if ctx:
        await ctx.info(f"Performing file operation: {operation} on {source_path}")
    
    try:
        # Validate and sanitize paths
        from ...filesystem.path_security import PathSecurity
        
        if not PathSecurity.validate_path(source_path):
            raise ValidationError(f"Source path validation failed: {source_path}")
        
        if destination_path and not PathSecurity.validate_path(destination_path):
            raise ValidationError(f"Destination path validation failed: {destination_path}")
        
        # Create file operation request
        source_file_path = FilePath(source_path)
        destination_file_path = FilePath(destination_path) if destination_path else None
        
        request = FileOperationRequest(
            operation=FileOperationType(operation),
            source_path=source_file_path,
            destination_path=destination_file_path,
            overwrite=overwrite,
            create_intermediate=create_intermediate,
            backup_existing=backup_existing
        )
        
        # Execute operation
        file_manager = FileOperationManager()
        
        if ctx:
            await ctx.report_progress(50, 100, f"Executing {operation}")
        
        result = await file_manager.execute_operation(request)
        
        if result.is_right():
            operation_result = result.get_right()
            return {
                "success": True,
                "operation": operation,
                "source_path": source_path,
                "destination_path": destination_path,
                "result": operation_result,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "operation_id": str(uuid.uuid4())
                }
            }
        else:
            error = result.get_left()
            return {
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "operation_id": str(uuid.uuid4())
                }
            }
            
    except Exception as e:
        # Comprehensive error handling
        return {
            "success": False,
            "error": {
                "code": "OPERATION_ERROR",
                "message": str(e),
                "details": {"operation": operation, "source": source_path}
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat()
            }
        }
```

## 🏗️ Modularity Strategy
- **src/filesystem/**: New directory for file system operations (<250 lines each)
- **file_operations.py**: Core operations and transaction safety (240 lines)
- **path_security.py**: Path validation and security (150 lines)
- **src/server/tools/file_operation_tools.py**: MCP tool implementation (200 lines)
- **Enhance existing files**: Minimal additions to types.py

## 🔒 Security Implementation
1. **Path Validation**: Comprehensive path sanitization and directory traversal prevention
2. **Permission Checking**: Verify user permissions for all file operations
3. **Allowed Directory Policy**: Restrict operations to approved directory trees
4. **Overwrite Protection**: Configurable overwrite behavior with backup options
5. **Atomic Operations**: Transaction-safe operations with rollback capability
6. **Secure Deletion**: Optional secure deletion with multiple overwrite passes

## 📊 Performance Targets
- **File Copy**: <5 seconds for files up to 100MB
- **File Move**: <2 seconds for local moves, <10 seconds for cross-volume
- **File Delete**: <1 second for individual files
- **Directory Operations**: <3 seconds for directory creation/deletion
- **Path Validation**: <50ms for security checks

## ✅ Success Criteria
- [ ] All advanced techniques implemented (path security, transaction safety, permission management)
- [ ] Complete security validation with directory traversal prevention
- [ ] Support for copy, move, delete, rename, create_folder, and get_info operations
- [ ] Real file system integration with proper error handling
- [ ] Transaction safety with rollback capability for failed operations
- [ ] Comprehensive error handling with permission and disk space validation
- [ ] Property-based testing covers all file operation scenarios and security edge cases
- [ ] Performance meets sub-10-second targets for most operations
- [ ] Integration with existing MCP framework and security model
- [ ] TESTING.md updated with file system security tests
- [ ] Full documentation with security policies and allowed directory configuration

## 🎨 Usage Examples

### Basic File Operations
```python
# Copy file with overwrite protection
result = await client.call_tool("km_file_operations", {
    "operation": "copy",
    "source_path": "/Users/user/Documents/source.txt",
    "destination_path": "/Users/user/Documents/backup.txt",
    "overwrite": False,
    "backup_existing": True
})

# Create directory structure
result = await client.call_tool("km_file_operations", {
    "operation": "create_folder",
    "source_path": "/Users/user/Documents/projects/new_project",
    "create_intermediate": True
})
```

### Advanced File Management
```python
# Secure file deletion
result = await client.call_tool("km_file_operations", {
    "operation": "delete",
    "source_path": "/Users/user/Documents/sensitive.txt",
    "secure_delete": True
})

# Get file information
result = await client.call_tool("km_file_operations", {
    "operation": "get_info",
    "source_path": "/Users/user/Documents/report.pdf"
})
```

## 🧪 Testing Strategy
- **Property-Based Testing**: Random file operations with various path combinations
- **Security Testing**: Directory traversal attempts, unauthorized path access
- **Performance Testing**: Large file operations, disk space scenarios
- **Transaction Testing**: Rollback scenarios and atomic operation validation
- **Permission Testing**: Operations with insufficient permissions
- **Integration Testing**: Real file system operations with security validation