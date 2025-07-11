# TASK_38: km_dictionary_manager - Advanced Data Structures & JSON Handling

**Created By**: Agent_3 (ADDER+ Protocol) | **Priority**: HIGH | **Duration**: 6 hours
**Technique Focus**: Data Validation + JSON Processing + Schema Management + Performance Optimization
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Agent_3
**Dependencies**: Foundation platform (TASK_1-20), condition system (TASK_21), control flow (TASK_22)
**Blocking**: Plugin ecosystem (TASK_39) and advanced automation workflows requiring structured data

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Data structure handling specification
- [ ] **KM Documentation**: development/protocols/KM_MCP.md - Dictionary and JSON actions in Keyboard Maestro
- [ ] **Foundation**: src/core/types.py - Existing type system and validation patterns
- [ ] **Token Integration**: src/server/tools/token_processor_tools.py - Token substitution in JSON data
- [ ] **Testing Framework**: tests/TESTING.md - Current test coverage and patterns

## 🎯 Problem Analysis
**Classification**: Essential Data Management Infrastructure
**Gap Identified**: No structured data management capabilities for complex automation workflows
**Impact**: AI cannot handle complex data structures, JSON processing, or schema-driven automation

<thinking>
Root Cause Analysis:
1. Current implementation focuses on simple operations - lacks structured data capabilities
2. Keyboard Maestro's dictionary and JSON features are powerful but unexposed through MCP
3. Complex automation workflows require data transformation, validation, and persistence
4. JSON processing enables API integration and data exchange between macros
5. Schema validation ensures data integrity in automated workflows
6. Dictionary operations enable key-value storage and retrieval patterns
7. This is essential for building sophisticated data-driven automation systems
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **Data type system**: Define branded types for dictionaries, JSON objects, schemas
- [ ] **Validation framework**: Schema validation, data integrity, type checking
- [ ] **Performance optimization**: Large data handling, memory management, caching

### Phase 2: Core Dictionary Operations
- [ ] **Dictionary management**: Create, read, update, delete operations with validation
- [ ] **Key-value operations**: Get/set operations with nested path support
- [ ] **Data transformation**: Type conversion, format transformation, filtering
- [ ] **Persistence**: Save/load dictionaries with integrity validation

### Phase 3: JSON Processing Engine
- [ ] **JSON parsing**: Safe parsing with schema validation and error handling
- [ ] **JSON generation**: Structure-aware generation with formatting options
- [ ] **Schema validation**: JSON Schema support with comprehensive error reporting
- [ ] **Data extraction**: JSONPath queries and data mining capabilities

### Phase 4: Advanced Data Operations
- [ ] **Data merging**: Deep merge operations with conflict resolution
- [ ] **Template processing**: Data-driven template expansion and substitution
- [ ] **Query engine**: Advanced querying and filtering capabilities
- [ ] **Export/import**: Multiple format support (JSON, YAML, CSV, XML)

### Phase 5: Integration & Security
- [ ] **Token integration**: TASK_19 token processor integration for dynamic data
- [ ] **Security validation**: Injection prevention, size limits, content filtering
- [ ] **Property-based tests**: Hypothesis validation for data processing
- [ ] **TESTING.md update**: Dictionary management test coverage and performance metrics

## 🔧 Implementation Files & Specifications
```
src/server/tools/dictionary_manager_tools.py   # Main dictionary management tool implementation
src/core/data_structures.py                    # Data type definitions and validation
src/data/dictionary_engine.py                  # Core dictionary operations
src/data/json_processor.py                     # JSON processing and schema validation
src/data/schema_manager.py                     # Schema management and validation
src/data/query_engine.py                       # Advanced querying capabilities
tests/tools/test_dictionary_manager_tools.py   # Unit and integration tests
tests/property_tests/test_data_structures.py   # Property-based data validation
```

### km_dictionary_manager Tool Specification
```python
@mcp.tool()
async def km_dictionary_manager(
    operation: str,                              # create|read|update|delete|query|merge|transform
    dictionary_name: str,                        # Dictionary identifier
    key_path: Optional[str] = None,             # Nested key path (e.g., "user.profile.name")
    value: Optional[Any] = None,                # Value for set operations
    schema: Optional[Dict[str, Any]] = None,    # JSON schema for validation
    query: Optional[str] = None,                # JSONPath query for data extraction
    merge_strategy: str = "deep",               # Merge strategy: deep|shallow|replace
    format_output: str = "json",               # Output format: json|yaml|csv|xml
    validate_schema: bool = True,               # Enable schema validation
    timeout_seconds: int = 30,                  # Operation timeout
    ctx = None
) -> Dict[str, Any]:
```

### Data Structure Type System
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json
from datetime import datetime

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

@dataclass(frozen=True)
class DictionaryPath:
    """Type-safe dictionary key path specification."""
    path: str
    separator: str = "."
    
    @require(lambda self: len(self.path.strip()) > 0)
    @require(lambda self: not self.path.startswith(self.separator))
    @require(lambda self: not self.path.endswith(self.separator))
    def __post_init__(self):
        pass
    
    def segments(self) -> List[str]:
        """Split path into individual key segments."""
        return self.path.split(self.separator)
    
    def parent(self) -> Optional['DictionaryPath']:
        """Get parent path if not root."""
        segments = self.segments()
        if len(segments) <= 1:
            return None
        return DictionaryPath(self.separator.join(segments[:-1]), self.separator)
    
    def child(self, key: str) -> 'DictionaryPath':
        """Create child path by appending key."""
        return DictionaryPath(f"{self.path}{self.separator}{key}", self.separator)

@dataclass(frozen=True) 
class DataSchema:
    """JSON Schema wrapper with validation capabilities."""
    schema: Dict[str, Any]
    strict_mode: bool = True
    allow_additional: bool = False
    
    @require(lambda self: isinstance(self.schema, dict))
    @require(lambda self: "type" in self.schema or "$schema" in self.schema)
    def __post_init__(self):
        pass
    
    def validate_data(self, data: Any) -> Either[ValidationError, None]:
        """Validate data against schema."""
        pass
    
    def get_property_schema(self, path: DictionaryPath) -> Optional[Dict[str, Any]]:
        """Get schema for specific property path."""
        pass

@dataclass(frozen=True)
class DictionaryMetadata:
    """Dictionary metadata and statistics."""
    name: str
    created_at: datetime
    modified_at: datetime
    size_bytes: int
    key_count: int
    nested_levels: int
    schema_hash: Optional[str] = None
    checksum: Optional[str] = None
    
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: self.size_bytes >= 0)
    @require(lambda self: self.key_count >= 0)
    def __post_init__(self):
        pass

class DictionaryManager:
    """Core dictionary management engine with type safety."""
    
    @require(lambda name: len(name.strip()) > 0)
    @ensure(lambda result: result.is_right() or result.get_left().is_data_error())
    async def create_dictionary(
        self,
        name: str,
        initial_data: Optional[Dict[str, Any]] = None,
        schema: Optional[DataSchema] = None
    ) -> Either[DataError, DictionaryMetadata]:
        """Create new dictionary with optional schema validation."""
        pass
    
    @require(lambda name: len(name.strip()) > 0)
    @ensure(lambda result: result.is_right() or result.get_left().is_data_error())
    async def get_value(
        self,
        name: str,
        path: Optional[DictionaryPath] = None
    ) -> Either[DataError, Any]:
        """Retrieve value from dictionary by path."""
        pass
    
    @require(lambda name: len(name.strip()) > 0)
    @require(lambda value: value is not None)
    @ensure(lambda result: result.is_right() or result.get_left().is_data_error())
    async def set_value(
        self,
        name: str,
        path: DictionaryPath,
        value: Any,
        create_path: bool = True
    ) -> Either[DataError, None]:
        """Set value in dictionary at specified path."""
        pass
    
    @require(lambda name: len(name.strip()) > 0)
    @ensure(lambda result: result.is_right() or result.get_left().is_data_error())
    async def delete_key(
        self,
        name: str,
        path: DictionaryPath
    ) -> Either[DataError, None]:
        """Delete key from dictionary."""
        pass
    
    async def merge_dictionaries(
        self,
        target_name: str,
        source_data: Dict[str, Any],
        strategy: MergeStrategy = MergeStrategy.DEEP
    ) -> Either[DataError, None]:
        """Merge data into existing dictionary."""
        pass
```

## 🔒 Security & Data Integrity Implementation
```python
class DataSecurityManager:
    """Security-first data management with validation and protection."""
    
    # Size limits for security
    MAX_DICTIONARY_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_KEY_LENGTH = 1000
    MAX_VALUE_SIZE = 10 * 1024 * 1024       # 10MB per value
    MAX_NESTING_DEPTH = 20
    
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',           # XSS prevention
        r'javascript:',                         # JavaScript URL prevention
        r'eval\s*\(',                          # Code injection prevention
        r'__[a-zA-Z_]+__',                     # Python internal attributes
        r'\.\./',                              # Path traversal prevention
    ]
    
    @staticmethod
    def validate_dictionary_name(name: str) -> Either[SecurityError, None]:
        """Validate dictionary name for security constraints."""
        if not name or len(name) == 0:
            return Either.left(SecurityError("Dictionary name cannot be empty"))
        
        if len(name) > 100:
            return Either.left(SecurityError("Dictionary name too long"))
        
        # Allow alphanumeric, underscores, hyphens, dots
        if not all(c.isalnum() or c in '_-.' for c in name):
            return Either.left(SecurityError("Invalid characters in dictionary name"))
        
        return Either.right(None)
    
    @staticmethod
    def validate_key_path(path: DictionaryPath) -> Either[SecurityError, None]:
        """Validate key path for security and depth limits."""
        if len(path.path) > DataSecurityManager.MAX_KEY_LENGTH:
            return Either.left(SecurityError("Key path too long"))
        
        segments = path.segments()
        if len(segments) > DataSecurityManager.MAX_NESTING_DEPTH:
            return Either.left(SecurityError("Key path too deeply nested"))
        
        # Check for dangerous patterns in path segments
        for segment in segments:
            for pattern in DataSecurityManager.DANGEROUS_PATTERNS:
                if re.search(pattern, segment, re.IGNORECASE):
                    return Either.left(SecurityError(f"Dangerous pattern in key: {segment}"))
        
        return Either.right(None)
    
    @staticmethod
    def validate_value_content(value: Any) -> Either[SecurityError, None]:
        """Validate value content for security and size limits."""
        # Calculate size
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            if len(serialized.encode('utf-8')) > DataSecurityManager.MAX_VALUE_SIZE:
                return Either.left(SecurityError("Value too large"))
        except (TypeError, ValueError):
            return Either.left(SecurityError("Value not JSON serializable"))
        
        # Check for dangerous content in string values
        if isinstance(value, str):
            for pattern in DataSecurityManager.DANGEROUS_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    return Either.left(SecurityError("Dangerous content in value"))
        
        # Recursively check nested structures
        if isinstance(value, dict):
            for k, v in value.items():
                key_result = DataSecurityManager.validate_value_content(k)
                if key_result.is_left():
                    return key_result
                
                value_result = DataSecurityManager.validate_value_content(v)
                if value_result.is_left():
                    return value_result
        
        elif isinstance(value, list):
            for item in value:
                item_result = DataSecurityManager.validate_value_content(item)
                if item_result.is_left():
                    return item_result
        
        return Either.right(None)
    
    @staticmethod
    def calculate_data_depth(data: Any, current_depth: int = 0) -> int:
        """Calculate nesting depth of data structure."""
        if current_depth > DataSecurityManager.MAX_NESTING_DEPTH:
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
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-."))
def test_dictionary_name_properties(name):
    """Property: Valid dictionary names should pass validation."""
    result = DataSecurityManager.validate_dictionary_name(name)
    
    # Names with only valid characters should pass
    if all(c.isalnum() or c in '_-.' for c in name):
        assert result.is_right()
    else:
        assert result.is_left()

@given(st.recursive(
    st.one_of(
        st.text(max_size=100),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.none()
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=10),
        st.dictionaries(st.text(max_size=20), children, max_size=10)
    ),
    max_leaves=50
))
def test_data_validation_properties(test_data):
    """Property: Data validation should handle all valid JSON structures."""
    validation_result = DataSecurityManager.validate_value_content(test_data)
    
    # Calculate expected depth
    depth = DataSecurityManager.calculate_data_depth(test_data)
    
    # Should pass if within depth and size limits
    try:
        json.dumps(test_data)
        serializable = True
    except (TypeError, ValueError):
        serializable = False
    
    if serializable and depth <= DataSecurityManager.MAX_NESTING_DEPTH:
        # Additional checks based on content size
        try:
            size = len(json.dumps(test_data, ensure_ascii=False).encode('utf-8'))
            if size <= DataSecurityManager.MAX_VALUE_SIZE:
                # Should pass if no dangerous patterns
                should_pass = True
                if isinstance(test_data, str):
                    for pattern in DataSecurityManager.DANGEROUS_PATTERNS:
                        if re.search(pattern, test_data, re.IGNORECASE):
                            should_pass = False
                            break
                
                if should_pass:
                    assert validation_result.is_right()
                else:
                    assert validation_result.is_left()
        except Exception:
            pass  # Size check failed, validation should fail

@given(st.lists(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"), min_size=1, max_size=10))
def test_dictionary_path_properties(path_segments):
    """Property: Dictionary paths should handle all valid segment combinations."""
    path_str = ".".join(path_segments)
    
    try:
        path = DictionaryPath(path_str)
        assert path.segments() == path_segments
        
        # Parent path should have one less segment
        parent = path.parent()
        if len(path_segments) > 1:
            assert parent is not None
            assert len(parent.segments()) == len(path_segments) - 1
        else:
            assert parent is None
        
        # Child path should have one more segment
        child = path.child("test")
        assert len(child.segments()) == len(path_segments) + 1
        assert child.segments()[-1] == "test"
        
    except Exception:
        # Path creation should only fail for invalid inputs
        assert any(len(segment) == 0 for segment in path_segments)
```

## 🏗️ Modularity Strategy
- **dictionary_manager_tools.py**: Main MCP tool interface (<250 lines)
- **data_structures.py**: Type definitions and validation (<200 lines)
- **dictionary_engine.py**: Core dictionary operations (<300 lines)
- **json_processor.py**: JSON processing and schema validation (<250 lines)
- **schema_manager.py**: Schema management and validation (<200 lines)
- **query_engine.py**: Advanced querying capabilities (<250 lines)

## 📋 Advanced Data Capabilities

### Dictionary Operations
```python
# Example: Create and manage structured data
dictionary_result = await dictionary_manager.create_dictionary(
    name="user_profiles",
    initial_data={"users": {}, "metadata": {"created": "2024-01-01"}},
    schema=user_profile_schema
)

# Set nested values with path notation
await dictionary_manager.set_value(
    name="user_profiles",
    path=DictionaryPath("users.john.profile.name"),
    value="John Doe"
)

# Query data with JSONPath
query_result = await dictionary_manager.query_data(
    name="user_profiles", 
    query="$.users[*].profile.name"
)
```

### JSON Processing and Schema Validation
```python
# Example: Process JSON with schema validation
json_result = await json_processor.process_json(
    data=raw_json_data,
    schema=api_response_schema,
    validate=True,
    transform_keys="camelCase"
)

# Merge data structures with conflict resolution
merge_result = await dictionary_manager.merge_dictionaries(
    target_name="config",
    source_data=new_settings,
    strategy=MergeStrategy.DEEP
)
```

### Advanced Querying and Export
```python
# Example: Complex data querying and export
export_result = await dictionary_manager.export_data(
    name="analytics_data",
    query="$.events[?(@.type=='click')].timestamp",
    format="csv",
    include_headers=True
)

# Transform data with templates
transform_result = await dictionary_manager.transform_data(
    name="user_data",
    template=email_template,
    output_format="json"
)
```

## ✅ Success Criteria
- Complete dictionary and JSON management implementation with schema validation
- Comprehensive security validation with injection prevention and size limits
- Property-based tests validate behavior across all data structure scenarios
- Integration with token system (TASK_19) for dynamic data processing
- Performance: <100ms simple operations, <1s complex queries, <5s large exports
- Documentation: Complete API documentation with security considerations and examples
- TESTING.md shows 95%+ test coverage with all security and performance tests passing
- Tool enables sophisticated data-driven automation workflows

## 🔄 Integration Points
- **TASK_19 (km_token_processor)**: Token substitution in JSON data and templates
- **TASK_21 (km_add_condition)**: Conditions based on data values and structure
- **TASK_22 (km_control_flow)**: Control flow based on data queries and validation
- **TASK_39 (km_plugin_ecosystem)**: Data exchange format for plugin systems
- **All Automation Tasks**: Structured data storage and retrieval for workflows

## 📋 Notes
- This provides the essential data infrastructure for sophisticated automation workflows
- Enables data-driven automation with schema validation and integrity checking
- Essential for API integration and complex data transformation scenarios
- Must maintain functional programming patterns for testability and composability
- Success here enables advanced workflows that can process, validate, and transform structured data
- Combined with other tasks, creates comprehensive automation platform with enterprise-grade data management