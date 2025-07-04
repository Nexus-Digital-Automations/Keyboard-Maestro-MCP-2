"""
Core data structure types for dictionary and JSON management.

This module defines branded types, protocols, and data structures for type-safe
data management with comprehensive validation and security boundaries.
"""

from __future__ import annotations
from typing import NewType, Any, Dict, List, Optional, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import re
import json
import hashlib

from .types import Duration
from .either import Either
from .errors import ValidationError, SecurityError, DataError


# Branded Types for Data Management
DictionaryId = NewType('DictionaryId', str)
SchemaId = NewType('SchemaId', str)
QueryId = NewType('QueryId', str)
DataChecksum = NewType('DataChecksum', str)


class DataOperation(Enum):
    """Supported dictionary and JSON operations."""
    CREATE = "create"
    READ = "read" 
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"
    MERGE = "merge"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    EXPORT = "export"
    IMPORT = "import"


class MergeStrategy(Enum):
    """Data merging strategies."""
    DEEP = "deep"           # Recursive merge of nested structures
    SHALLOW = "shallow"     # Top-level merge only
    REPLACE = "replace"     # Replace existing values
    APPEND = "append"       # Append to arrays
    UNION = "union"         # Union for sets/unique values


class DataFormat(Enum):
    """Supported data export/import formats."""
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    XML = "xml"
    PLIST = "plist"


@dataclass(frozen=True)
class DictionaryPath:
    """Type-safe dictionary key path specification."""
    path: str
    separator: str = "."
    
    def __post_init__(self):
        if not self.path or len(self.path.strip()) == 0:
            raise ValueError("Path cannot be empty")
        if self.path.startswith(self.separator) or self.path.endswith(self.separator):
            raise ValueError("Path cannot start or end with separator")
        if self.separator * 2 in self.path:
            raise ValueError("Path cannot contain consecutive separators")
    
    def segments(self) -> List[str]:
        """Split path into individual key segments."""
        return self.path.split(self.separator)
    
    def parent(self) -> Optional[DictionaryPath]:
        """Get parent path if not root."""
        segments = self.segments()
        if len(segments) <= 1:
            return None
        return DictionaryPath(self.separator.join(segments[:-1]), self.separator)
    
    def child(self, key: str) -> DictionaryPath:
        """Create child path by appending key."""
        if not key or self.separator in key:
            raise ValueError("Invalid child key")
        return DictionaryPath(f"{self.path}{self.separator}{key}", self.separator)
    
    def key_name(self) -> str:
        """Get the final key name from the path."""
        return self.segments()[-1]
    
    def depth(self) -> int:
        """Get the depth (number of segments) of the path."""
        return len(self.segments())


@dataclass(frozen=True) 
class DataSchema:
    """JSON Schema wrapper with validation capabilities."""
    schema: Dict[str, Any]
    schema_id: SchemaId
    strict_mode: bool = True
    allow_additional: bool = False
    
    def __post_init__(self):
        if not isinstance(self.schema, dict):
            raise ValueError("Schema must be a dictionary")
        if not self.schema.get("type") and not self.schema.get("$schema"):
            raise ValueError("Schema must specify type or $schema")
    
    @classmethod
    def create_object_schema(
        cls,
        properties: Dict[str, Dict[str, Any]],
        required: Optional[List[str]] = None,
        schema_id: Optional[SchemaId] = None
    ) -> DataSchema:
        """Create an object schema with specified properties."""
        schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False
        }
        if required:
            schema["required"] = required
        
        if schema_id is None:
            schema_id = SchemaId(f"schema_{hashlib.md5(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:8]}")
        
        return cls(schema=schema, schema_id=schema_id)
    
    def get_property_schema(self, path: DictionaryPath) -> Optional[Dict[str, Any]]:
        """Get schema for specific property path."""
        current_schema = self.schema
        
        for segment in path.segments():
            if current_schema.get("type") != "object":
                return None
            
            properties = current_schema.get("properties", {})
            if segment not in properties:
                return None
            
            current_schema = properties[segment]
        
        return current_schema


@dataclass(frozen=True)
class DictionaryMetadata:
    """Dictionary metadata and statistics."""
    dictionary_id: DictionaryId
    name: str
    created_at: datetime
    modified_at: datetime
    size_bytes: int
    key_count: int
    nested_levels: int
    schema_id: Optional[SchemaId] = None
    checksum: Optional[DataChecksum] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.name or len(self.name.strip()) == 0:
            raise ValueError("Dictionary name cannot be empty")
        if self.size_bytes < 0 or self.key_count < 0 or self.nested_levels < 0:
            raise ValueError("Metadata values cannot be negative")
    
    @classmethod
    def create_new(
        cls,
        name: str,
        initial_size: int = 0,
        schema_id: Optional[SchemaId] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> DictionaryMetadata:
        """Create metadata for a new dictionary."""
        now = datetime.now()
        dictionary_id = DictionaryId(f"dict_{hashlib.sha256(f'{name}_{now.isoformat()}'.encode()).hexdigest()[:16]}")
        
        return cls(
            dictionary_id=dictionary_id,
            name=name,
            created_at=now,
            modified_at=now,
            size_bytes=initial_size,
            key_count=0,
            nested_levels=0,
            schema_id=schema_id,
            description=description,
            tags=tags or []
        )
    
    def update_size(self, new_size: int, new_key_count: int, new_nested_levels: int) -> DictionaryMetadata:
        """Create updated metadata with new size information."""
        from dataclasses import replace
        return replace(
            self,
            modified_at=datetime.now(),
            size_bytes=new_size,
            key_count=new_key_count,
            nested_levels=new_nested_levels
        )


@dataclass(frozen=True)
class QueryResult:
    """Result of a data query operation."""
    query: str
    results: List[Any]
    total_matches: int
    execution_time: Duration
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_empty(self) -> bool:
        """Check if query returned no results."""
        return self.total_matches == 0
    
    def first_result(self) -> Optional[Any]:
        """Get the first result if available."""
        return self.results[0] if self.results else None


@dataclass(frozen=True)
class MergeConflict:
    """Information about a merge conflict."""
    path: DictionaryPath
    current_value: Any
    incoming_value: Any
    conflict_type: str
    resolution: Optional[str] = None
    
    def is_resolved(self) -> bool:
        """Check if conflict has been resolved."""
        return self.resolution is not None


@dataclass(frozen=True)
class MergeResult:
    """Result of a merge operation."""
    success: bool
    conflicts: List[MergeConflict]
    keys_added: int
    keys_modified: int
    keys_removed: int
    execution_time: Duration
    
    def has_conflicts(self) -> bool:
        """Check if merge had any conflicts."""
        return len(self.conflicts) > 0
    
    def unresolved_conflicts(self) -> List[MergeConflict]:
        """Get list of unresolved conflicts."""
        return [c for c in self.conflicts if not c.is_resolved()]


class DataValidator(Protocol):
    """Protocol for data validation operations."""
    
    def validate_value(self, value: Any, path: Optional[DictionaryPath] = None) -> Either[ValidationError, None]:
        """Validate a value according to schema or rules."""
        ...
    
    def validate_structure(self, data: Dict[str, Any]) -> Either[ValidationError, List[str]]:
        """Validate entire data structure."""
        ...


class DataTransformer(Protocol):
    """Protocol for data transformation operations."""
    
    def transform_keys(self, data: Dict[str, Any], strategy: str) -> Either[ValidationError, Dict[str, Any]]:
        """Transform keys according to naming strategy."""
        ...
    
    def transform_values(self, data: Dict[str, Any], transformer: str) -> Either[ValidationError, Dict[str, Any]]:
        """Transform values according to transformation rules."""
        ...


@dataclass(frozen=True)
class SecurityLimits:
    """Security limits for data operations."""
    max_dictionary_size: int = 50 * 1024 * 1024  # 50MB
    max_key_length: int = 1000
    max_value_size: int = 10 * 1024 * 1024       # 10MB per value
    max_nesting_depth: int = 20
    max_query_complexity: int = 100
    max_export_size: int = 100 * 1024 * 1024     # 100MB
    
    def validate_size(self, size: int, size_type: str) -> Either[SecurityError, None]:
        """Validate size against appropriate limit."""
        limits = {
            "dictionary": self.max_dictionary_size,
            "key": self.max_key_length,
            "value": self.max_value_size,
            "export": self.max_export_size
        }
        
        limit = limits.get(size_type)
        if limit is None:
            return Either.left(SecurityError(f"Unknown size type: {size_type}"))
        
        if size > limit:
            return Either.left(SecurityError(f"{size_type} size {size} exceeds limit {limit}"))
        
        return Either.right(None)
    
    def validate_depth(self, depth: int) -> Either[SecurityError, None]:
        """Validate nesting depth."""
        if depth > self.max_nesting_depth:
            return Either.left(SecurityError(f"Nesting depth {depth} exceeds limit {self.max_nesting_depth}"))
        return Either.right(None)


# Default security limits instance
DEFAULT_SECURITY_LIMITS = SecurityLimits()