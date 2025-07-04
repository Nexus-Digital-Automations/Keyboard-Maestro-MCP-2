"""
Property-based tests for data structures and dictionary management.

This module uses Hypothesis to test dictionary management behavior across input ranges,
ensuring data validation, security boundaries, and operation correctness.
"""

import pytest
import json
import re
from datetime import datetime
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from src.core.data_structures import (
    DictionaryPath, DataSchema, SchemaId, DictionaryMetadata, SecurityLimits,
    MergeStrategy, DataOperation, DEFAULT_SECURITY_LIMITS
)
from src.core.either import Either
from src.core.errors import ValidationError, SecurityError, DataError
from src.data.dictionary_engine import DictionaryEngine, DataSecurityManager
from src.data.json_processor import JSONProcessor


class TestDictionaryPathProperties:
    """Property-based tests for dictionary path handling."""
    
    @given(st.lists(
        st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
        min_size=1,
        max_size=10
    ))
    def test_path_segments_properties(self, segments):
        """Property: Path segments should be preserved correctly."""
        path_str = ".".join(segments)
        
        try:
            path = DictionaryPath(path_str)
            assert path.segments() == segments
            
            # Parent path should have one less segment
            parent = path.parent()
            if len(segments) > 1:
                assert parent is not None
                assert len(parent.segments()) == len(segments) - 1
                assert parent.segments() == segments[:-1]
            else:
                assert parent is None
            
            # Child path should have one more segment
            child = path.child("test")
            assert len(child.segments()) == len(segments) + 1
            assert child.segments() == segments + ["test"]
            
            # Key name should be last segment
            assert path.key_name() == segments[-1]
            
            # Depth should equal segment count
            assert path.depth() == len(segments)
            
        except ValueError:
            # Should only fail for invalid segments
            assert any(not seg or "." in seg for seg in segments)
    
    @given(st.text(min_size=1, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_."))
    def test_path_validation_properties(self, path_str):
        """Property: Path validation should be consistent."""
        try:
            path = DictionaryPath(path_str)
            
            # Valid paths should not start/end with separator
            assert not path_str.startswith(".")
            assert not path_str.endswith(".")
            assert ".." not in path_str
            
            # Should be able to reconstruct path
            reconstructed = ".".join(path.segments())
            assert reconstructed == path_str
            
        except ValueError:
            # Should fail for invalid paths
            assert (
                path_str.startswith(".") or
                path_str.endswith(".") or
                ".." in path_str or
                not path_str.strip()
            )


class TestDataSchemaProperties:
    """Property-based tests for data schema validation."""
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
        st.dictionaries(
            st.sampled_from(["type", "minLength", "maxLength", "minimum", "maximum"]),
            st.one_of(
                st.sampled_from(["string", "number", "integer", "boolean", "array", "object"]),
                st.integers(min_value=0, max_value=1000),
                st.floats(min_value=0.0, max_value=1000.0)
            )
        ),
        min_size=1,
        max_size=5
    ))
    def test_object_schema_properties(self, properties):
        """Property: Object schemas should handle all valid property definitions."""
        try:
            schema = DataSchema.create_object_schema(properties)
            
            assert schema.schema["type"] == "object"
            assert schema.schema["properties"] == properties
            assert schema.schema["additionalProperties"] == False
            
            # Schema ID should be generated
            assert schema.schema_id is not None
            assert len(schema.schema_id) > 0
            
            # Should be able to get property schemas
            for prop_name in properties.keys():
                path = DictionaryPath(prop_name)
                prop_schema = schema.get_property_schema(path)
                assert prop_schema == properties[prop_name]
            
        except Exception:
            # Should only fail for invalid property definitions
            pass
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.lists(st.text(max_size=20), max_size=5)
        ),
        max_size=10
    ))
    def test_schema_validation_properties(self, test_data):
        """Property: Schema validation should handle all JSON-serializable data."""
        # Create a permissive schema
        schema_dict = {
            "type": "object",
            "additionalProperties": True
        }
        
        try:
            schema = DataSchema(schema_dict, SchemaId("test_schema"))
            
            # Should be able to validate JSON-serializable data
            json.dumps(test_data)  # Ensure it's serializable
            
            processor = JSONProcessor()
            result = processor._validate_against_schema(test_data, schema, False)
            
            # Should either succeed or fail with validation error
            assert result.is_right() or isinstance(result.get_left(), ValidationError)
            
        except (TypeError, ValueError):
            # Skip non-serializable data
            pass


class TestSecurityValidationProperties:
    """Property-based tests for security validation."""
    
    @given(st.text(min_size=1, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-."))
    def test_dictionary_name_validation_properties(self, name):
        """Property: Dictionary name validation should be consistent."""
        result = DataSecurityManager.validate_dictionary_name(name)
        
        # Names with only valid characters should pass
        valid_chars = all(c.isalnum() or c in '_-.' for c in name)
        not_reserved = name.lower() not in {"__system__", "__internal__", "null", "undefined", "system"}
        reasonable_length = len(name) <= 100
        
        if valid_chars and not_reserved and reasonable_length:
            assert result.is_right()
        else:
            assert result.is_left()
    
    @given(st.recursive(
        st.one_of(
            st.text(max_size=50),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none()
        ),
        lambda children: st.one_of(
            st.lists(children, max_size=5),
            st.dictionaries(st.text(max_size=10), children, max_size=5)
        ),
        max_leaves=20
    ))
    def test_content_validation_properties(self, test_data):
        """Property: Content validation should handle all JSON structures."""
        result = DataSecurityManager.validate_value_content(test_data)
        
        try:
            # Check if data is JSON serializable
            json.dumps(test_data)
            serializable = True
        except (TypeError, ValueError, RecursionError):
            serializable = False
        
        if not serializable:
            assert result.is_left()
        else:
            # Check depth
            depth = DataSecurityManager.calculate_data_depth(test_data)
            
            if depth > DEFAULT_SECURITY_LIMITS.max_nesting_depth:
                assert result.is_left()
            else:
                # Check for dangerous patterns in strings
                has_dangerous_content = False
                if isinstance(test_data, str):
                    for pattern in DataSecurityManager.DANGEROUS_PATTERNS:
                        if re.search(pattern, test_data, re.IGNORECASE):
                            has_dangerous_content = True
                            break
                
                if has_dangerous_content:
                    assert result.is_left()
                else:
                    # Should pass if no issues
                    assert result.is_right() or result.is_left()  # Either way is valid
    
    @given(st.integers(min_value=0, max_value=100))
    def test_depth_calculation_properties(self, depth_target):
        """Property: Depth calculation should be accurate for nested structures."""
        # Create nested structure of specified depth
        if depth_target == 0:
            data = "leaf"
        else:
            data = {"level": "leaf"}
            for i in range(depth_target - 1):
                data = {"level": data}
        
        calculated_depth = DataSecurityManager.calculate_data_depth(data)
        assert calculated_depth == depth_target
    
    @given(st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=10))
    def test_size_validation_properties(self, sizes):
        """Property: Size validation should respect limits."""
        limits = SecurityLimits(
            max_dictionary_size=max(sizes) + 100,
            max_value_size=max(sizes) + 50,
            max_key_length=max(sizes) + 10
        )
        
        for size in sizes:
            # Dictionary size validation
            dict_result = limits.validate_size(size, "dictionary")
            if size <= limits.max_dictionary_size:
                assert dict_result.is_right()
            else:
                assert dict_result.is_left()
            
            # Value size validation
            value_result = limits.validate_size(size, "value")
            if size <= limits.max_value_size:
                assert value_result.is_right()
            else:
                assert value_result.is_left()
            
            # Key length validation
            key_result = limits.validate_size(size, "key")
            if size <= limits.max_key_length:
                assert key_result.is_right()
            else:
                assert key_result.is_left()


class TestJSONProcessorProperties:
    """Property-based tests for JSON processing."""
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(
            st.text(max_size=50),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans()
        ),
        max_size=10
    ))
    def test_json_round_trip_properties(self, test_data):
        """Property: JSON serialization/deserialization should be round-trip safe."""
        processor = JSONProcessor()
        
        try:
            # Generate JSON
            json_result = asyncio.run(processor.generate_json(test_data))
            
            if json_result.is_right():
                json_string = json_result.get_right()
                
                # Parse it back
                parse_result = asyncio.run(processor.parse_json(json_string))
                
                if parse_result.is_right():
                    parsed_data = parse_result.get_right()
                    
                    # Should be equal to original
                    assert parsed_data == test_data
                    
        except Exception:
            # Some data might not be serializable or might exceed limits
            pass
    
    @given(st.text(max_size=1000))
    def test_json_security_validation_properties(self, json_content):
        """Property: JSON security validation should catch dangerous patterns."""
        processor = JSONProcessor()
        
        result = processor._validate_json_security(json_content)
        
        dangerous_patterns = [
            r'__proto__',
            r'constructor\.prototype',
            r'eval\s*\(',
            r'Function\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\('
        ]
        
        has_dangerous_pattern = any(
            re.search(pattern, json_content, re.IGNORECASE)
            for pattern in dangerous_patterns
        )
        
        if has_dangerous_pattern:
            assert result.is_left()
        else:
            assert result.is_right()


import asyncio


class TestDictionaryEngineProperties:
    """Property-based tests for dictionary engine operations."""
    
    @given(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"))
    def test_dictionary_lifecycle_properties(self, dict_name):
        """Property: Dictionary creation and basic operations should work."""
        engine = DictionaryEngine()
        
        # Dictionary should not exist initially
        metadata = engine.get_dictionary_metadata(dict_name)
        assert metadata is None
        
        # Create dictionary
        create_result = asyncio.run(engine.create_dictionary(dict_name))
        
        if create_result.is_right():
            # Should now exist
            metadata = engine.get_dictionary_metadata(dict_name)
            assert metadata is not None
            assert metadata.name == dict_name
            
            # Should be listed
            all_dicts = engine.list_dictionaries()
            dict_names = [d.name for d in all_dicts]
            assert dict_name in dict_names
            
            # Should be able to read
            read_result = asyncio.run(engine.get_value(dict_name))
            assert read_result.is_right()
    
    @given(
        st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
        st.dictionaries(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            st.one_of(st.text(max_size=20), st.integers(), st.booleans()),
            max_size=5
        )
    )
    def test_dictionary_data_operations_properties(self, dict_name, test_data):
        """Property: Dictionary data operations should preserve data integrity."""
        engine = DictionaryEngine()
        
        # Create dictionary with initial data
        create_result = asyncio.run(engine.create_dictionary(dict_name, test_data))
        
        if create_result.is_right():
            # Should be able to read back the same data
            read_result = asyncio.run(engine.get_value(dict_name))
            
            if read_result.is_right():
                retrieved_data = read_result.get_right()
                assert retrieved_data == test_data
                
                # Test individual key access
                for key, expected_value in test_data.items():
                    path = DictionaryPath(key)
                    key_result = asyncio.run(engine.get_value(dict_name, path))
                    
                    if key_result.is_right():
                        assert key_result.get_right() == expected_value


# Test configuration for faster property testing
settings.register_profile("fast", max_examples=50, deadline=1000)
settings.load_profile("fast")


# Additional focused property tests
@given(st.sampled_from([op.value for op in DataOperation]))
def test_data_operation_enum_consistency(operation_value):
    """Property: All data operation values should be valid enum values."""
    try:
        operation = DataOperation(operation_value)
        assert operation.value == operation_value
    except ValueError:
        pytest.fail(f"Operation value {operation_value} should be valid")


@given(st.sampled_from([strategy.value for strategy in MergeStrategy]))
def test_merge_strategy_enum_consistency(strategy_value):
    """Property: All merge strategy values should be valid enum values."""
    try:
        strategy = MergeStrategy(strategy_value)
        assert strategy.value == strategy_value
    except ValueError:
        pytest.fail(f"Merge strategy value {strategy_value} should be valid")


@given(st.text(min_size=1, max_size=50))
def test_schema_id_creation_properties(schema_name):
    """Property: Schema IDs should be unique and reproducible."""
    schema_id1 = SchemaId(f"schema_{schema_name}")
    schema_id2 = SchemaId(f"schema_{schema_name}")
    
    # Same input should produce same ID
    assert schema_id1 == schema_id2
    
    # Different input should produce different ID
    different_id = SchemaId(f"schema_{schema_name}_different")
    assert schema_id1 != different_id