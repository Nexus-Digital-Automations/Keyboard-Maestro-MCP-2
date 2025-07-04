"""
MCP File Operation Tools for Keyboard Maestro

Provides secure file system operations with comprehensive validation,
transaction safety, and path security for AI assistant workflows.
"""

from typing import Dict, Any, Optional, Annotated
from datetime import datetime, UTC
import uuid

from pydantic import Field
from fastmcp import Context

from ...filesystem.file_operations import (
    FileOperationManager,
    FileOperationRequest,
    FileOperationType,
    FilePath
)
from ...filesystem.path_security import PathSecurity
from ...core.contracts import require, ensure


class ValidationError(Exception):
    """File operation validation error."""
    pass


@require(lambda operation: operation in ["copy", "move", "delete", "rename", "create_folder", "get_info"])
@require(lambda source_path: len(source_path) > 0 and len(source_path) <= 1000)
@ensure(lambda result: isinstance(result, dict) and "success" in result)
async def km_file_operations(
    operation: Annotated[str, Field(
        description="File operation type: copy, move, delete, rename, create_folder, get_info",
        pattern=r"^(copy|move|delete|rename|create_folder|get_info)$"
    )],
    source_path: Annotated[str, Field(
        description="Source file or folder path (absolute path required)",
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
        description="Allow overwriting existing files"
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
        description="Use secure deletion (multiple overwrite passes)"
    )] = False,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Secure file system operations with comprehensive validation and safety features.
    
    Supported Operations:
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
    - Restricted to allowed directory trees
    
    Performance Targets:
    - File operations: <10 seconds for most operations
    - Path validation: <50ms for security checks
    - Large files: Up to 100MB supported
    
    Returns:
        Dict containing operation results, metadata, and security status
    
    Raises:
        ValidationError: If path validation fails
        PermissionError: If insufficient permissions
        OSError: If file system operation fails
    """
    if ctx:
        await ctx.info(f"Starting file operation: {operation} on {source_path}")
    
    try:
        # Validate and sanitize paths using comprehensive security validation
        if not PathSecurity.validate_path(source_path):
            raise ValidationError(f"Source path validation failed: {source_path}")
        
        if destination_path and not PathSecurity.validate_path(destination_path):
            raise ValidationError(f"Destination path validation failed: {destination_path}")
        
        # Create type-safe file path objects
        source_file_path = FilePath(source_path)
        destination_file_path = FilePath(destination_path) if destination_path else None
        
        # Validate paths are safe for operations
        if not source_file_path.is_safe_path():
            raise ValidationError(f"Source path security check failed: {source_path}")
        
        if destination_file_path and not destination_file_path.is_safe_path():
            raise ValidationError(f"Destination path security check failed: {destination_path}")
        
        # Create file operation request with validation
        request = FileOperationRequest(
            operation=FileOperationType(operation),
            source_path=source_file_path,
            destination_path=destination_file_path,
            overwrite=overwrite,
            create_intermediate=create_intermediate,
            backup_existing=backup_existing,
            secure_delete=secure_delete
        )
        
        # Execute operation with comprehensive error handling
        file_manager = FileOperationManager()
        
        if ctx:
            await ctx.report_progress(25, 100, f"Validating {operation} operation")
        
        # Execute the file operation
        result = await file_manager.execute_operation(request)
        
        if ctx:
            await ctx.report_progress(75, 100, f"Completing {operation} operation")
        
        # Process results
        if result.is_right():
            operation_result = result.get_right()
            
            # Prepare successful response
            response = {
                "success": True,
                "operation": operation,
                "source_path": source_path,
                "destination_path": destination_path,
                "result": operation_result.to_dict(),
                "security_status": {
                    "path_validated": True,
                    "permissions_checked": True,
                    "transaction_safe": True
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "operation_id": str(uuid.uuid4()),
                    "execution_time": operation_result.execution_time.total_seconds() if operation_result.execution_time else None,
                    "bytes_processed": operation_result.bytes_processed
                }
            }
            
            # Add backup information if available
            if operation_result.backup_path:
                response["backup_created"] = operation_result.backup_path
            
            if ctx:
                await ctx.info(f"File operation completed successfully: {operation}")
            
            return response
            
        else:
            # Handle operation errors
            error = result.get_left()
            
            error_response = {
                "success": False,
                "operation": operation,
                "source_path": source_path,
                "destination_path": destination_path,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details or {}
                },
                "security_status": {
                    "path_validated": True,
                    "permissions_checked": True,
                    "operation_failed": True
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "operation_id": str(uuid.uuid4())
                }
            }
            
            if ctx:
                await ctx.error(f"File operation failed: {error.message}")
            
            return error_response
    
    except ValidationError as e:
        # Path validation or parameter validation errors
        error_response = {
            "success": False,
            "operation": operation,
            "source_path": source_path,
            "destination_path": destination_path,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": str(e),
                "details": {
                    "validation_type": "path_security",
                    "failed_path": source_path if "Source path" in str(e) else destination_path
                }
            },
            "security_status": {
                "path_validated": False,
                "security_violation": True
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "operation_id": str(uuid.uuid4())
            }
        }
        
        if ctx:
            await ctx.error(f"Path validation failed: {str(e)}")
        
        return error_response
    
    except PermissionError as e:
        # Permission-related errors
        error_response = {
            "success": False,
            "operation": operation,
            "source_path": source_path,
            "destination_path": destination_path,
            "error": {
                "code": "PERMISSION_ERROR",
                "message": f"Permission denied: {str(e)}",
                "details": {
                    "error_type": "permission_denied",
                    "operation_attempted": operation
                }
            },
            "security_status": {
                "path_validated": True,
                "permission_denied": True
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "operation_id": str(uuid.uuid4())
            }
        }
        
        if ctx:
            await ctx.error(f"Permission denied for {operation}: {str(e)}")
        
        return error_response
    
    except Exception as e:
        # Comprehensive error handling for unexpected errors
        error_response = {
            "success": False,
            "operation": operation,
            "source_path": source_path,
            "destination_path": destination_path,
            "error": {
                "code": "OPERATION_ERROR",
                "message": f"Unexpected error: {str(e)}",
                "details": {
                    "error_type": type(e).__name__,
                    "operation": operation,
                    "source": source_path
                }
            },
            "security_status": {
                "path_validated": True,
                "unexpected_error": True
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "operation_id": str(uuid.uuid4())
            }
        }
        
        if ctx:
            await ctx.error(f"Unexpected error in {operation}: {str(e)}")
        
        return error_response