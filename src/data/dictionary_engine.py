"""
Core dictionary management engine with type-safe operations.

This module provides the main dictionary management functionality with 
comprehensive validation, security checking, and error handling.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path

from ..core.data_structures import (
    DictionaryId, DictionaryPath, DictionaryMetadata, DataSchema, SchemaId,
    MergeStrategy, MergeResult, MergeConflict, SecurityLimits, DEFAULT_SECURITY_LIMITS
)
from ..core.either import Either
from ..core.errors import ValidationError, SecurityError, DataError, create_error_context


class DataSecurityManager:
    """Security-first data management with validation and protection."""
    
    # Dangerous patterns for security validation
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',           # XSS prevention
        r'javascript:',                         # JavaScript URL prevention
        r'eval\s*\(',                          # Code injection prevention
        r'__[a-zA-Z_]+__',                     # Python internal attributes
        r'\.\./',                              # Path traversal prevention
        r'import\s+',                          # Import statement prevention
        r'exec\s*\(',                          # Exec function prevention
        r'subprocess\.',                       # Subprocess prevention
    ]
    
    @staticmethod
    def validate_dictionary_name(name: str) -> Either[SecurityError, None]:
        """Validate dictionary name for security constraints."""
        if not name or len(name.strip()) == 0:
            return Either.left(SecurityError("INVALID_NAME", "Dictionary name cannot be empty"))
        
        if len(name) > 100:
            return Either.left(SecurityError("NAME_TOO_LONG", "Dictionary name too long"))
        
        # Allow alphanumeric, underscores, hyphens, dots
        if not all(c.isalnum() or c in '_-.' for c in name):
            return Either.left(SecurityError("INVALID_CHARS", "Invalid characters in dictionary name"))
        
        # Prevent reserved names
        reserved_names = {"__system__", "__internal__", "null", "undefined", "system"}
        if name.lower() in reserved_names:
            return Either.left(SecurityError("RESERVED_NAME", f"Name '{name}' is reserved"))
        
        return Either.right(None)
    
    @staticmethod
    def validate_key_path(path: DictionaryPath, limits: SecurityLimits = DEFAULT_SECURITY_LIMITS) -> Either[SecurityError, None]:
        """Validate key path for security and depth limits."""
        if len(path.path) > limits.max_key_length:
            return Either.left(SecurityError("PATH_TOO_LONG", f"Key path exceeds {limits.max_key_length} characters"))
        
        segments = path.segments()
        if len(segments) > limits.max_nesting_depth:
            return Either.left(SecurityError("PATH_TOO_DEEP", f"Key path exceeds {limits.max_nesting_depth} levels"))
        
        # Check for dangerous patterns in path segments
        for segment in segments:
            if not segment or len(segment.strip()) == 0:
                return Either.left(SecurityError("EMPTY_SEGMENT", "Key path contains empty segment"))
            
            for pattern in DataSecurityManager.DANGEROUS_PATTERNS:
                if re.search(pattern, segment, re.IGNORECASE):
                    return Either.left(SecurityError("DANGEROUS_PATTERN", f"Dangerous pattern in key: {segment}"))
        
        return Either.right(None)
    
    @staticmethod
    def validate_value_content(value: Any, limits: SecurityLimits = DEFAULT_SECURITY_LIMITS) -> Either[SecurityError, None]:
        """Validate value content for security and size limits."""
        # Calculate size and validate serializability
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            size_bytes = len(serialized.encode('utf-8'))
            
            size_result = limits.validate_size(size_bytes, "value")
            if size_result.is_left():
                return size_result
                
        except (TypeError, ValueError, RecursionError) as e:
            return Either.left(SecurityError("NOT_SERIALIZABLE", f"Value not JSON serializable: {str(e)}"))
        
        # Check depth
        depth = DataSecurityManager.calculate_data_depth(value)
        depth_result = limits.validate_depth(depth)
        if depth_result.is_left():
            return depth_result
        
        # Check for dangerous content
        return DataSecurityManager._validate_content_recursive(value)
    
    @staticmethod
    def _validate_content_recursive(value: Any) -> Either[SecurityError, None]:
        """Recursively validate content for dangerous patterns."""
        if isinstance(value, str):
            for pattern in DataSecurityManager.DANGEROUS_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    return Either.left(SecurityError("DANGEROUS_CONTENT", f"Dangerous content detected: {pattern}"))
        
        elif isinstance(value, dict):
            for k, v in value.items():
                # Validate key
                if isinstance(k, str):
                    key_result = DataSecurityManager._validate_content_recursive(k)
                    if key_result.is_left():
                        return key_result
                
                # Validate value
                value_result = DataSecurityManager._validate_content_recursive(v)
                if value_result.is_left():
                    return value_result
        
        elif isinstance(value, list):
            for item in value:
                item_result = DataSecurityManager._validate_content_recursive(item)
                if item_result.is_left():
                    return item_result
        
        return Either.right(None)
    
    @staticmethod
    def calculate_data_depth(data: Any, current_depth: int = 0) -> int:
        """Calculate nesting depth of data structure."""
        if current_depth > DEFAULT_SECURITY_LIMITS.max_nesting_depth:
            return current_depth
        
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(
                DataSecurityManager.calculate_data_depth(v, current_depth + 1)
                for v in data.values()
            )
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(
                DataSecurityManager.calculate_data_depth(item, current_depth + 1)
                for item in data
            )
        else:
            return current_depth


class DictionaryEngine:
    """Core dictionary management engine with comprehensive validation."""
    
    def __init__(self, storage_path: Optional[Path] = None, security_limits: SecurityLimits = DEFAULT_SECURITY_LIMITS):
        self.storage_path = storage_path or Path.home() / ".km-mcp" / "dictionaries"
        self.security_limits = security_limits
        self._dictionaries: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, DictionaryMetadata] = {}
        self._schemas: Dict[str, DataSchema] = {}
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def create_dictionary(
        self,
        name: str,
        initial_data: Optional[Dict[str, Any]] = None,
        schema: Optional[DataSchema] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> Either[DataError, DictionaryMetadata]:
        """Create new dictionary with optional schema validation."""
        try:
            # Validate name
            name_result = DataSecurityManager.validate_dictionary_name(name)
            if name_result.is_left():
                return Either.left(DataError("create_dictionary", f"Invalid name: {name_result.get_left().message}"))
            
            # Check if dictionary already exists
            if name in self._dictionaries:
                return Either.left(DataError("create_dictionary", f"Dictionary '{name}' already exists"))
            
            # Validate initial data if provided
            data = initial_data or {}
            if data:
                content_result = DataSecurityManager.validate_value_content(data, self.security_limits)
                if content_result.is_left():
                    return Either.left(DataError("create_dictionary", f"Invalid initial data: {content_result.get_left().message}"))
                
                # Validate against schema if provided
                if schema:
                    schema_result = self._validate_against_schema(data, schema)
                    if schema_result.is_left():
                        return Either.left(DataError("create_dictionary", f"Schema validation failed: {schema_result.get_left().message}"))
            
            # Calculate metadata
            size_bytes = len(json.dumps(data, ensure_ascii=False).encode('utf-8'))
            key_count = self._count_keys(data)
            nested_levels = DataSecurityManager.calculate_data_depth(data)
            
            # Create metadata
            metadata = DictionaryMetadata.create_new(
                name=name,
                initial_size=size_bytes,
                schema_id=schema.schema_id if schema else None,
                description=description,
                tags=tags
            ).update_size(size_bytes, key_count, nested_levels)
            
            # Store dictionary and metadata
            self._dictionaries[name] = data.copy()
            self._metadata[name] = metadata
            if schema:
                self._schemas[schema.schema_id] = schema
            
            # Persist to storage
            await self._persist_dictionary(name)
            
            return Either.right(metadata)
            
        except Exception as e:
            context = create_error_context("create_dictionary", "dictionary_engine", name=name, error=str(e))
            return Either.left(DataError("create_dictionary", f"Unexpected error: {str(e)}", context))
    
    async def get_value(
        self,
        name: str,
        path: Optional[DictionaryPath] = None
    ) -> Either[DataError, Any]:
        """Retrieve value from dictionary by path."""
        try:
            # Check if dictionary exists
            if name not in self._dictionaries:
                return Either.left(DataError("get_value", f"Dictionary '{name}' not found"))
            
            data = self._dictionaries[name]
            
            # Return entire dictionary if no path specified
            if path is None:
                return Either.right(data.copy())
            
            # Validate path
            path_result = DataSecurityManager.validate_key_path(path, self.security_limits)
            if path_result.is_left():
                return Either.left(DataError("get_value", f"Invalid path: {path_result.get_left().message}"))
            
            # Navigate to value
            current = data
            for segment in path.segments():
                if not isinstance(current, dict) or segment not in current:
                    return Either.left(DataError("get_value", f"Path '{path.path}' not found"))
                current = current[segment]
            
            return Either.right(current)
            
        except Exception as e:
            context = create_error_context("get_value", "dictionary_engine", name=name, path=path.path if path else None)
            return Either.left(DataError("get_value", f"Unexpected error: {str(e)}", context))
    
    async def set_value(
        self,
        name: str,
        path: DictionaryPath,
        value: Any,
        create_path: bool = True
    ) -> Either[DataError, None]:
        """Set value in dictionary at specified path."""
        try:
            # Check if dictionary exists
            if name not in self._dictionaries:
                return Either.left(DataError("set_value", f"Dictionary '{name}' not found"))
            
            # Validate path and value
            path_result = DataSecurityManager.validate_key_path(path, self.security_limits)
            if path_result.is_left():
                return Either.left(DataError("set_value", f"Invalid path: {path_result.get_left().message}"))
            
            value_result = DataSecurityManager.validate_value_content(value, self.security_limits)
            if value_result.is_left():
                return Either.left(DataError("set_value", f"Invalid value: {value_result.get_left().message}"))
            
            # Check schema validation if schema exists
            metadata = self._metadata[name]
            if metadata.schema_id and metadata.schema_id in self._schemas:
                schema = self._schemas[metadata.schema_id]
                # Create temporary data structure for validation
                temp_data = self._dictionaries[name].copy()
                self._set_value_at_path(temp_data, path, value, create_path)
                
                schema_result = self._validate_against_schema(temp_data, schema)
                if schema_result.is_left():
                    return Either.left(DataError("set_value", f"Schema validation failed: {schema_result.get_left().message}"))
            
            # Set the value
            data = self._dictionaries[name]
            self._set_value_at_path(data, path, value, create_path)
            
            # Update metadata
            size_bytes = len(json.dumps(data, ensure_ascii=False).encode('utf-8'))
            key_count = self._count_keys(data)
            nested_levels = DataSecurityManager.calculate_data_depth(data)
            
            self._metadata[name] = metadata.update_size(size_bytes, key_count, nested_levels)
            
            # Persist changes
            await self._persist_dictionary(name)
            
            return Either.right(None)
            
        except Exception as e:
            context = create_error_context("set_value", "dictionary_engine", name=name, path=path.path)
            return Either.left(DataError("set_value", f"Unexpected error: {str(e)}", context))
    
    async def delete_key(
        self,
        name: str,
        path: DictionaryPath
    ) -> Either[DataError, None]:
        """Delete key from dictionary."""
        try:
            # Check if dictionary exists
            if name not in self._dictionaries:
                return Either.left(DataError("delete_key", f"Dictionary '{name}' not found"))
            
            # Validate path
            path_result = DataSecurityManager.validate_key_path(path, self.security_limits)
            if path_result.is_left():
                return Either.left(DataError("delete_key", f"Invalid path: {path_result.get_left().message}"))
            
            # Navigate to parent and delete key
            data = self._dictionaries[name]
            segments = path.segments()
            
            if len(segments) == 1:
                # Top-level key
                if segments[0] not in data:
                    return Either.left(DataError("delete_key", f"Key '{segments[0]}' not found"))
                del data[segments[0]]
            else:
                # Nested key
                current = data
                for segment in segments[:-1]:
                    if not isinstance(current, dict) or segment not in current:
                        return Either.left(DataError("delete_key", f"Path '{path.path}' not found"))
                    current = current[segment]
                
                final_key = segments[-1]
                if not isinstance(current, dict) or final_key not in current:
                    return Either.left(DataError("delete_key", f"Key '{final_key}' not found"))
                del current[final_key]
            
            # Update metadata
            metadata = self._metadata[name]
            size_bytes = len(json.dumps(data, ensure_ascii=False).encode('utf-8'))
            key_count = self._count_keys(data)
            nested_levels = DataSecurityManager.calculate_data_depth(data)
            
            self._metadata[name] = metadata.update_size(size_bytes, key_count, nested_levels)
            
            # Persist changes
            await self._persist_dictionary(name)
            
            return Either.right(None)
            
        except Exception as e:
            context = create_error_context("delete_key", "dictionary_engine", name=name, path=path.path)
            return Either.left(DataError("delete_key", f"Unexpected error: {str(e)}", context))
    
    def list_dictionaries(self) -> List[DictionaryMetadata]:
        """List all dictionaries with their metadata."""
        return list(self._metadata.values())
    
    def get_dictionary_metadata(self, name: str) -> Optional[DictionaryMetadata]:
        """Get metadata for specific dictionary."""
        return self._metadata.get(name)
    
    async def _persist_dictionary(self, name: str) -> None:
        """Persist dictionary to storage."""
        if name not in self._dictionaries:
            return
        
        # Create dictionary file
        dict_file = self.storage_path / f"{name}.json"
        with open(dict_file, 'w', encoding='utf-8') as f:
            json.dump(self._dictionaries[name], f, ensure_ascii=False, indent=2)
        
        # Create metadata file
        metadata_file = self.storage_path / f"{name}.metadata.json"
        metadata = self._metadata[name]
        metadata_dict = {
            "dictionary_id": metadata.dictionary_id,
            "name": metadata.name,
            "created_at": metadata.created_at.isoformat(),
            "modified_at": metadata.modified_at.isoformat(),
            "size_bytes": metadata.size_bytes,
            "key_count": metadata.key_count,
            "nested_levels": metadata.nested_levels,
            "schema_id": metadata.schema_id,
            "checksum": metadata.checksum,
            "description": metadata.description,
            "tags": metadata.tags
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    
    def _set_value_at_path(self, data: Dict[str, Any], path: DictionaryPath, value: Any, create_path: bool) -> None:
        """Internal method to set value at path."""
        segments = path.segments()
        current = data
        
        # Navigate to parent, creating path if needed
        for segment in segments[:-1]:
            if segment not in current:
                if not create_path:
                    raise KeyError(f"Path segment '{segment}' not found")
                current[segment] = {}
            elif not isinstance(current[segment], dict):
                if not create_path:
                    raise TypeError(f"Path segment '{segment}' is not a dictionary")
                current[segment] = {}
            current = current[segment]
        
        # Set final value
        final_key = segments[-1]
        current[final_key] = value
    
    def _count_keys(self, data: Any) -> int:
        """Count total number of keys in nested structure."""
        if isinstance(data, dict):
            return len(data) + sum(self._count_keys(v) for v in data.values())
        elif isinstance(data, list):
            return sum(self._count_keys(item) for item in data)
        else:
            return 0
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: DataSchema) -> Either[ValidationError, None]:
        """Validate data against schema (basic implementation)."""
        # This is a simplified schema validation
        # In a full implementation, you would use a proper JSON schema library
        schema_dict = schema.schema
        
        if schema_dict.get("type") == "object":
            if not isinstance(data, dict):
                return Either.left(ValidationError("Schema validation failed: expected object"))
            
            required = schema_dict.get("required", [])
            for req_field in required:
                if req_field not in data:
                    return Either.left(ValidationError(f"Required field '{req_field}' missing"))
            
            properties = schema_dict.get("properties", {})
            for key, value in data.items():
                if key in properties:
                    prop_schema = properties[key]
                    prop_type = prop_schema.get("type")
                    
                    if prop_type == "string" and not isinstance(value, str):
                        return Either.left(ValidationError(f"Field '{key}' must be string"))
                    elif prop_type == "number" and not isinstance(value, (int, float)):
                        return Either.left(ValidationError(f"Field '{key}' must be number"))
                    elif prop_type == "integer" and not isinstance(value, int):
                        return Either.left(ValidationError(f"Field '{key}' must be integer"))
                    elif prop_type == "boolean" and not isinstance(value, bool):
                        return Either.left(ValidationError(f"Field '{key}' must be boolean"))
                    elif prop_type == "array" and not isinstance(value, list):
                        return Either.left(ValidationError(f"Field '{key}' must be array"))
                    elif prop_type == "object" and not isinstance(value, dict):
                        return Either.left(ValidationError(f"Field '{key}' must be object"))
        
        return Either.right(None)