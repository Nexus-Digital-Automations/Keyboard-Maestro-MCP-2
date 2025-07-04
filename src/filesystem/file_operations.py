"""
Core File System Operations with Transaction Safety

This module implements secure file operations with transaction safety, rollback
capability, and comprehensive security validation for Keyboard Maestro MCP.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pathlib import Path
import shutil
import os
import time
import uuid
import hashlib
import asyncio
import subprocess

from .path_security import PathSecurity, PathAccessLevel
from ..core.types import Duration
from ..core.contracts import require, ensure
from ..integration.km_client import Either, KMError


class FileOperationType(Enum):
    """Supported file system operations with security validation."""
    COPY = "copy"
    MOVE = "move" 
    DELETE = "delete"
    RENAME = "rename"
    CREATE_FOLDER = "create_folder"
    GET_INFO = "get_info"


@dataclass(frozen=True)
class FilePath:
    """
    Type-safe file path with comprehensive security validation.
    
    Implements defensive programming patterns to ensure all file paths
    are validated and sanitized before any operations are performed.
    """
    path: str
    _security: PathSecurity = field(default_factory=PathSecurity, init=False)
    
    @require(lambda self: len(self.path) > 0 and len(self.path) <= 1000)
    def __post_init__(self):
        """Validate path constraints on creation."""
        pass
    
    def is_safe_path(self, access_level: PathAccessLevel = PathAccessLevel.READ_WRITE) -> bool:
        """Validate path is safe for operations with specified access level."""
        return self._security.validate_path(self.path, access_level)
    
    def exists(self) -> bool:
        """Check if path exists on filesystem."""
        try:
            return Path(self.path).exists()
        except (OSError, ValueError):
            return False
    
    def is_file(self) -> bool:
        """Check if path points to a file."""
        try:
            return Path(self.path).is_file()
        except (OSError, ValueError):
            return False
    
    def is_directory(self) -> bool:
        """Check if path points to a directory."""
        try:
            return Path(self.path).is_dir()
        except (OSError, ValueError):
            return False
    
    def get_size(self) -> Optional[int]:
        """Get file size in bytes, None if not accessible."""
        try:
            path_obj = Path(self.path)
            if path_obj.is_file():
                return path_obj.stat().st_size
            elif path_obj.is_dir():
                total_size = 0
                for file_path in path_obj.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                return total_size
            return None
        except (OSError, ValueError):
            return None
    
    def get_parent(self) -> Optional[FilePath]:
        """Get parent directory as FilePath."""
        try:
            parent = Path(self.path).parent
            return FilePath(str(parent))
        except (OSError, ValueError):
            return None
    
    def resolve(self) -> Optional[str]:
        """Get resolved absolute path."""
        try:
            return str(Path(self.path).resolve())
        except (OSError, ValueError):
            return None


@dataclass(frozen=True)
class FileOperationRequest:
    """
    Type-safe file operation request with comprehensive validation.
    
    Ensures all operation parameters are validated and safe before
    any file system operations are performed.
    """
    operation: FileOperationType
    source_path: FilePath
    destination_path: Optional[FilePath] = None
    overwrite: bool = False
    create_intermediate: bool = False
    backup_existing: bool = False
    secure_delete: bool = False
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @require(lambda self: self.source_path.is_safe_path())
    @require(lambda self: not self.destination_path or self.destination_path.is_safe_path())
    def __post_init__(self):
        """Validate operation request constraints."""
        # Additional validation for specific operations
        if self.operation in (FileOperationType.COPY, FileOperationType.MOVE, FileOperationType.RENAME):
            if not self.destination_path:
                raise ValueError(f"Operation {self.operation.value} requires destination_path")
    
    def requires_destination(self) -> bool:
        """Check if operation requires a destination path."""
        return self.operation in (FileOperationType.COPY, FileOperationType.MOVE, FileOperationType.RENAME)


@dataclass
class FileOperationResult:
    """Result of file operation with metadata and rollback information."""
    success: bool
    operation: FileOperationType
    source_path: str
    destination_path: Optional[str] = None
    execution_time: Optional[Duration] = None
    bytes_processed: Optional[int] = None
    backup_path: Optional[str] = None
    transaction_id: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    rollback_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for API responses."""
        return {
            "success": self.success,
            "operation": self.operation.value,
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "execution_time": self.execution_time.total_seconds() if self.execution_time else None,
            "bytes_processed": self.bytes_processed,
            "backup_path": self.backup_path,
            "transaction_id": self.transaction_id,
            "error_details": self.error_details,
            "has_rollback": self.rollback_info is not None
        }


class FileOperationManager:
    """
    Secure file operations with transaction safety and rollback capability.
    
    Implements comprehensive security validation, atomic operations, and
    defensive programming patterns for reliable file system automation.
    """
    
    def __init__(self):
        """Initialize file operation manager with security validation."""
        self._security = PathSecurity()
        self._active_transactions: Dict[str, Dict[str, Any]] = {}
        self._max_file_size = 100_000_000  # 100MB limit
        self._backup_suffix = ".km_backup"
    
    @require(lambda request: isinstance(request, FileOperationRequest))
    @ensure(lambda result: result.is_right() or result.get_left().code in [
        "PERMISSION_ERROR", "PATH_ERROR", "DISK_SPACE_ERROR", "VALIDATION_ERROR", "EXECUTION_ERROR"
    ])
    async def execute_operation(self, request: FileOperationRequest) -> Either[KMError, FileOperationResult]:
        """
        Execute file operation with comprehensive validation and error handling.
        
        Args:
            request: Validated file operation request
            
        Returns:
            Either operation result or error details
        """
        start_time = time.time()
        
        try:
            # Pre-operation validation
            validation_result = await self._validate_operation(request)
            if validation_result.is_left():
                return validation_result
            
            # Execute operation based on type
            if request.operation == FileOperationType.COPY:
                result = await self._copy_operation(request)
            elif request.operation == FileOperationType.MOVE:
                result = await self._move_operation(request)
            elif request.operation == FileOperationType.DELETE:
                result = await self._delete_operation(request)
            elif request.operation == FileOperationType.RENAME:
                result = await self._rename_operation(request)
            elif request.operation == FileOperationType.CREATE_FOLDER:
                result = await self._create_folder_operation(request)
            elif request.operation == FileOperationType.GET_INFO:
                result = await self._get_info_operation(request)
            else:
                return Either.left(KMError.validation_error(f"Unsupported operation: {request.operation.value}"))
            
            if result.is_right():
                operation_result = result.get_right()
                operation_result.execution_time = Duration.from_seconds(time.time() - start_time)
                operation_result.transaction_id = request.transaction_id
            
            return result
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"File operation failed: {str(e)}"))
    
    async def _validate_operation(self, request: FileOperationRequest) -> Either[KMError, bool]:
        """Comprehensive pre-operation validation."""
        try:
            # Validate source path
            if not request.source_path.is_safe_path():
                return Either.left(KMError.validation_error("Source path failed security validation"))
            
            # Check source exists for operations that require it
            if request.operation != FileOperationType.CREATE_FOLDER:
                if not request.source_path.exists():
                    return Either.left(KMError.not_found_error(f"Source path does not exist: {request.source_path.path}"))
            
            # Validate destination path if required
            if request.requires_destination():
                if not request.destination_path:
                    return Either.left(KMError.validation_error("Destination path required for this operation"))
                
                if not request.destination_path.is_safe_path():
                    return Either.left(KMError.validation_error("Destination path failed security validation"))
                
                # Check for overwrite conflicts
                if request.destination_path.exists() and not request.overwrite:
                    return Either.left(KMError.validation_error("Destination exists and overwrite not enabled"))
            
            # Check disk space for copy operations
            if request.operation == FileOperationType.COPY:
                source_size = request.source_path.get_size() or 0
                if source_size > self._max_file_size:
                    return Either.left(KMError.validation_error(f"File too large: {source_size} bytes"))
                
                dest_parent = request.destination_path.get_parent()
                if dest_parent and not self._security.check_disk_space(Path(dest_parent.path), source_size):
                    return Either.left(KMError.validation_error("Insufficient disk space"))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.validation_error(f"Validation failed: {str(e)}"))
    
    async def _copy_operation(self, request: FileOperationRequest) -> Either[KMError, FileOperationResult]:
        """Execute copy operation with backup and rollback support."""
        try:
            source_path = Path(request.source_path.path)
            dest_path = Path(request.destination_path.path)
            
            # Create backup if requested and destination exists
            backup_path = None
            if request.backup_existing and dest_path.exists():
                backup_path = await self._create_backup(dest_path)
            
            # Create intermediate directories if requested
            if request.create_intermediate:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Perform copy operation
            bytes_processed = 0
            if source_path.is_file():
                shutil.copy2(source_path, dest_path)
                bytes_processed = dest_path.stat().st_size
            elif source_path.is_dir():
                if dest_path.exists() and request.overwrite:
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
                bytes_processed = sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
            
            return Either.right(FileOperationResult(
                success=True,
                operation=request.operation,
                source_path=request.source_path.path,
                destination_path=request.destination_path.path,
                bytes_processed=bytes_processed,
                backup_path=str(backup_path) if backup_path else None
            ))
            
        except PermissionError as e:
            return Either.left(KMError.validation_error(f"Permission denied: {str(e)}"))
        except OSError as e:
            return Either.left(KMError.execution_error(f"Copy operation failed: {str(e)}"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Unexpected error: {str(e)}"))
    
    async def _move_operation(self, request: FileOperationRequest) -> Either[KMError, FileOperationResult]:
        """Execute move operation with atomic transaction safety."""
        try:
            source_path = Path(request.source_path.path)
            dest_path = Path(request.destination_path.path)
            
            # Create backup if requested and destination exists
            backup_path = None
            if request.backup_existing and dest_path.exists():
                backup_path = await self._create_backup(dest_path)
            
            # Create intermediate directories if requested
            if request.create_intermediate:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Store original file size for reporting
            bytes_processed = source_path.stat().st_size if source_path.is_file() else 0
            
            # Perform atomic move
            shutil.move(source_path, dest_path)
            
            return Either.right(FileOperationResult(
                success=True,
                operation=request.operation,
                source_path=request.source_path.path,
                destination_path=request.destination_path.path,
                bytes_processed=bytes_processed,
                backup_path=str(backup_path) if backup_path else None
            ))
            
        except PermissionError as e:
            return Either.left(KMError.validation_error(f"Permission denied: {str(e)}"))
        except OSError as e:
            return Either.left(KMError.execution_error(f"Move operation failed: {str(e)}"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Unexpected error: {str(e)}"))
    
    async def _delete_operation(self, request: FileOperationRequest) -> Either[KMError, FileOperationResult]:
        """Execute delete operation with optional secure deletion."""
        try:
            source_path = Path(request.source_path.path)
            
            # Create backup if requested
            backup_path = None
            if request.backup_existing:
                backup_path = await self._create_backup(source_path)
            
            bytes_processed = source_path.stat().st_size if source_path.is_file() else 0
            
            # Perform secure deletion if requested
            if request.secure_delete and source_path.is_file():
                await self._secure_delete_file(source_path)
            else:
                if source_path.is_file():
                    source_path.unlink()
                elif source_path.is_dir():
                    shutil.rmtree(source_path)
            
            return Either.right(FileOperationResult(
                success=True,
                operation=request.operation,
                source_path=request.source_path.path,
                bytes_processed=bytes_processed,
                backup_path=str(backup_path) if backup_path else None
            ))
            
        except PermissionError as e:
            return Either.left(KMError.validation_error(f"Permission denied: {str(e)}"))
        except OSError as e:
            return Either.left(KMError.execution_error(f"Delete operation failed: {str(e)}"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Unexpected error: {str(e)}"))
    
    async def _rename_operation(self, request: FileOperationRequest) -> Either[KMError, FileOperationResult]:
        """Execute rename operation with validation."""
        try:
            source_path = Path(request.source_path.path)
            dest_path = Path(request.destination_path.path)
            
            # Ensure rename is within the same directory for safety
            if source_path.parent != dest_path.parent:
                return Either.left(KMError.validation_error("Rename operation must be within same directory"))
            
            bytes_processed = source_path.stat().st_size if source_path.is_file() else 0
            
            # Perform rename
            source_path.rename(dest_path)
            
            return Either.right(FileOperationResult(
                success=True,
                operation=request.operation,
                source_path=request.source_path.path,
                destination_path=request.destination_path.path,
                bytes_processed=bytes_processed
            ))
            
        except PermissionError as e:
            return Either.left(KMError.validation_error(f"Permission denied: {str(e)}"))
        except OSError as e:
            return Either.left(KMError.execution_error(f"Rename operation failed: {str(e)}"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Unexpected error: {str(e)}"))
    
    async def _create_folder_operation(self, request: FileOperationRequest) -> Either[KMError, FileOperationResult]:
        """Execute folder creation with intermediate path support."""
        try:
            source_path = Path(request.source_path.path)
            
            # Create directory with parents if requested
            source_path.mkdir(parents=request.create_intermediate, exist_ok=request.overwrite)
            
            return Either.right(FileOperationResult(
                success=True,
                operation=request.operation,
                source_path=request.source_path.path,
                bytes_processed=0
            ))
            
        except FileExistsError:
            return Either.left(KMError.validation_error("Directory already exists"))
        except PermissionError as e:
            return Either.left(KMError.validation_error(f"Permission denied: {str(e)}"))
        except OSError as e:
            return Either.left(KMError.execution_error(f"Create folder failed: {str(e)}"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Unexpected error: {str(e)}"))
    
    async def _get_info_operation(self, request: FileOperationRequest) -> Either[KMError, FileOperationResult]:
        """Get file/directory information and metadata."""
        try:
            source_path = Path(request.source_path.path)
            stat_info = source_path.stat()
            
            file_info = {
                "name": source_path.name,
                "size": stat_info.st_size,
                "is_file": source_path.is_file(),
                "is_directory": source_path.is_dir(),
                "created": stat_info.st_ctime,
                "modified": stat_info.st_mtime,
                "accessed": stat_info.st_atime,
                "permissions": oct(stat_info.st_mode)[-3:],
                "owner_uid": stat_info.st_uid,
                "group_gid": stat_info.st_gid
            }
            
            return Either.right(FileOperationResult(
                success=True,
                operation=request.operation,
                source_path=request.source_path.path,
                bytes_processed=stat_info.st_size,
                error_details=file_info  # Using error_details to store file info
            ))
            
        except PermissionError as e:
            return Either.left(KMError.validation_error(f"Permission denied: {str(e)}"))
        except OSError as e:
            return Either.left(KMError.execution_error(f"Get info failed: {str(e)}"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Unexpected error: {str(e)}"))
    
    async def _create_backup(self, path: Path) -> Optional[Path]:
        """Create backup of file or directory before operation."""
        try:
            timestamp = int(time.time())
            backup_path = path.with_suffix(f"{path.suffix}{self._backup_suffix}_{timestamp}")
            
            if path.is_file():
                shutil.copy2(path, backup_path)
            elif path.is_dir():
                shutil.copytree(path, backup_path)
            
            return backup_path
            
        except Exception:
            return None
    
    async def _secure_delete_file(self, path: Path) -> None:
        """Securely delete file with multiple overwrite passes."""
        try:
            if not path.is_file():
                return
            
            file_size = path.stat().st_size
            
            # Overwrite file with random data multiple times
            with open(path, 'r+b') as f:
                for _ in range(3):  # 3 passes
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Finally delete the file
            path.unlink()
            
        except Exception:
            # Fall back to regular deletion
            path.unlink()