"""
Path Security Validation and Sanitization

This module implements comprehensive path security validation to prevent
directory traversal attacks, unauthorized file access, and injection vulnerabilities
while maintaining usability for legitimate file operations.
"""

from __future__ import annotations
from typing import Set, List, Optional, Pattern
from pathlib import Path
import re
import os
import stat
from enum import Enum

from ..core.contracts import require, ensure


class PathAccessLevel(Enum):
    """File system access level permissions."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    FULL_ACCESS = "full_access"


class PathSecurity:
    """
    Comprehensive path security validation and sanitization.
    
    Implements defensive programming patterns to prevent directory traversal,
    unauthorized access, and injection attacks while allowing legitimate
    file operations within approved directory trees.
    """
    
    # Default allowed directories for file operations (configurable via environment)
    DEFAULT_ALLOWED_DIRECTORIES = {
        "~/Documents",
        "~/Downloads", 
        "~/Desktop",
        "/tmp",
        "~/Library/Application Support/Keyboard Maestro MCP"
    }
    
    # System directories that should never be accessible
    PROTECTED_DIRECTORIES = {
        "/System",
        "/usr/bin",
        "/usr/sbin",
        "/sbin",
        "/bin",
        "/etc",
        "/var/log",
        "/private/etc",
        "/private/var",
        "/Library/System",
        "/Applications/Utilities"
    }
    
    # Dangerous path patterns for validation
    DANGEROUS_PATTERNS: List[Pattern[str]] = [
        re.compile(r"\.\.[\\/]", re.IGNORECASE),  # Directory traversal
        re.compile(r"^[\\/]", re.IGNORECASE),     # Absolute root paths (unless allowed)
        re.compile(r"~[\\/]", re.IGNORECASE),     # Uncontrolled home shortcuts
        re.compile(r"\$\{[^}]*\}", re.IGNORECASE), # Environment variable expansion
        re.compile(r"`[^`]*`", re.IGNORECASE),    # Command substitution
        re.compile(r"\|", re.IGNORECASE),         # Pipe operators
        re.compile(r"[;&]", re.IGNORECASE),       # Command separators
        re.compile(r"<[^>]*>", re.IGNORECASE),    # Redirection operators
    ]
    
    # Characters that should be sanitized from paths
    UNSAFE_CHARACTERS = r'[<>:"|?*\x00-\x1f\x7f-\xff]'
    
    def __init__(self, custom_allowed_dirs: Optional[Set[str]] = None):
        """
        Initialize path security with optional custom allowed directories.
        
        Args:
            custom_allowed_dirs: Custom set of allowed directory patterns
        """
        self._allowed_directories = custom_allowed_dirs or self.DEFAULT_ALLOWED_DIRECTORIES
        self._resolved_allowed_dirs = self._resolve_allowed_directories()
    
    @staticmethod
    def _resolve_allowed_directories() -> Set[Path]:
        """Resolve allowed directory patterns to absolute paths."""
        resolved = set()
        
        for dir_pattern in PathSecurity.DEFAULT_ALLOWED_DIRECTORIES:
            try:
                # Expand user home directory
                expanded = os.path.expanduser(dir_pattern)
                path_obj = Path(expanded).resolve()
                
                # Create directory if it doesn't exist (for app-specific dirs)
                if "Keyboard Maestro MCP" in str(path_obj):
                    path_obj.mkdir(parents=True, exist_ok=True)
                
                if path_obj.exists():
                    resolved.add(path_obj)
            except (OSError, ValueError):
                # Skip invalid paths
                continue
        
        return resolved
    
    @require(lambda path: isinstance(path, str) and len(path) > 0)
    @staticmethod
    def validate_path(path: str, access_level: PathAccessLevel = PathAccessLevel.READ_WRITE) -> bool:
        """
        Comprehensive path security validation.
        
        Args:
            path: File path to validate
            access_level: Required access level
            
        Returns:
            True if path is safe for operations
        """
        try:
            # Basic sanitization
            if not PathSecurity._check_basic_safety(path):
                return False
            
            # Check for dangerous patterns
            if PathSecurity._contains_dangerous_patterns(path):
                return False
            
            # Resolve path safely
            resolved_path = PathSecurity._safe_resolve_path(path)
            if not resolved_path:
                return False
            
            # Check against protected directories
            if PathSecurity._is_protected_directory(resolved_path):
                return False
            
            # Check if path is within allowed directories
            if not PathSecurity._is_within_allowed_directories(resolved_path):
                return False
            
            # Check file system permissions
            if not PathSecurity._check_permissions(resolved_path, access_level):
                return False
            
            return True
            
        except Exception:
            # Any exception during validation means unsafe
            return False
    
    @require(lambda path: isinstance(path, str))
    @ensure(lambda result: result is None or self.validate_path(result))
    def sanitize_path(self, path: str) -> Optional[str]:
        """
        Sanitize path for safe operations.
        
        Args:
            path: Path to sanitize
            
        Returns:
            Sanitized path or None if cannot be made safe
        """
        try:
            # Remove unsafe characters
            sanitized = re.sub(self.UNSAFE_CHARACTERS, '', path.strip())
            
            # Remove dangerous patterns
            for pattern in self.DANGEROUS_PATTERNS:
                sanitized = pattern.sub('', sanitized)
            
            # Normalize path separators
            sanitized = sanitized.replace('\\', '/')
            
            # Remove duplicate slashes
            sanitized = re.sub(r'/+', '/', sanitized)
            
            # Remove leading/trailing whitespace and slashes
            sanitized = sanitized.strip('/ ')
            
            if len(sanitized) == 0:
                return None
            
            # Validate the sanitized path
            if self.validate_path(sanitized, PathAccessLevel.READ_ONLY):
                return sanitized
            
            return None
            
        except Exception:
            return None
    
    @staticmethod
    def _check_basic_safety(path: str) -> bool:
        """Check basic path safety requirements."""
        if not path or len(path) > 1000:
            return False
        
        # Check for null bytes
        if '\x00' in path:
            return False
        
        # Check for excessive directory traversal attempts
        if path.count('../') > 3 or path.count('..\\') > 3:
            return False
        
        return True
    
    @staticmethod
    def _contains_dangerous_patterns(path: str) -> bool:
        """Check if path contains dangerous patterns."""
        for pattern in PathSecurity.DANGEROUS_PATTERNS:
            if pattern.search(path):
                return True
        return False
    
    @staticmethod
    def _safe_resolve_path(path: str) -> Optional[Path]:
        """Safely resolve path to absolute form."""
        try:
            # Handle relative paths by making them relative to user home
            if not os.path.isabs(path):
                # If it doesn't start with ~, make it relative to home
                if not path.startswith('~'):
                    path = os.path.join(os.path.expanduser('~/Documents'), path)
                else:
                    path = os.path.expanduser(path)
            
            resolved = Path(path).resolve()
            return resolved
            
        except (OSError, ValueError, RuntimeError):
            return None
    
    @staticmethod
    def _is_protected_directory(path: Path) -> bool:
        """Check if path is within protected system directories."""
        path_str = str(path)
        
        for protected_dir in PathSecurity.PROTECTED_DIRECTORIES:
            try:
                protected_path = Path(protected_dir).resolve()
                if path == protected_path or path_str.startswith(str(protected_path) + '/'):
                    return True
            except (OSError, ValueError):
                continue
        
        return False
    
    @staticmethod
    def _is_within_allowed_directories(path: Path) -> bool:
        """Check if path is within allowed directories."""
        for allowed_dir in PathSecurity._resolve_allowed_directories():
            try:
                # Check if path is within or equal to allowed directory
                path.relative_to(allowed_dir)
                return True
            except ValueError:
                # Not within this allowed directory, try next
                continue
        
        return False
    
    @staticmethod
    def _check_permissions(path: Path, access_level: PathAccessLevel) -> bool:
        """Check if we have required permissions for the path."""
        try:
            # For non-existent paths, check parent directory
            check_path = path if path.exists() else path.parent
            
            # Check if parent exists
            if not check_path.exists():
                return False
            
            # Get file stats
            stat_info = check_path.stat()
            
            # Check read permission
            if not os.access(check_path, os.R_OK):
                return False
            
            # Check write permission if required
            if access_level in (PathAccessLevel.READ_WRITE, PathAccessLevel.FULL_ACCESS):
                if not os.access(check_path, os.W_OK):
                    return False
            
            # Check execute permission for directories
            if check_path.is_dir() and not os.access(check_path, os.X_OK):
                return False
            
            return True
            
        except (OSError, ValueError):
            return False
    
    @staticmethod
    def get_safe_temp_path(prefix: str = "km_mcp_") -> Optional[Path]:
        """
        Get a safe temporary file path within allowed directories.
        
        Args:
            prefix: Filename prefix
            
        Returns:
            Safe temporary file path or None
        """
        import tempfile
        import uuid
        
        try:
            # Try to use /tmp if it's in allowed directories
            temp_dir = Path("/tmp")
            allowed_dirs = PathSecurity._resolve_allowed_directories()
            if temp_dir in allowed_dirs:
                temp_file = temp_dir / f"{prefix}{uuid.uuid4().hex}"
                return temp_file
            
            # Fallback to user's Documents directory
            docs_dir = Path.home() / "Documents"
            if docs_dir in allowed_dirs:
                temp_file = docs_dir / f"{prefix}{uuid.uuid4().hex}"
                return temp_file
            
            return None
            
        except Exception:
            return None
    
    def check_disk_space(self, path: Path, required_bytes: int) -> bool:
        """
        Check if there's enough disk space for an operation.
        
        Args:
            path: Path to check space for
            required_bytes: Required space in bytes
            
        Returns:
            True if enough space available
        """
        try:
            # Get parent directory if file doesn't exist
            check_path = path if path.exists() else path.parent
            
            # Get disk usage statistics
            stat_result = os.statvfs(check_path)
            available_bytes = stat_result.f_bavail * stat_result.f_frsize
            
            # Add 10% safety margin
            safety_margin = required_bytes * 0.1
            required_with_margin = required_bytes + safety_margin
            
            return available_bytes >= required_with_margin
            
        except (OSError, ValueError):
            return False