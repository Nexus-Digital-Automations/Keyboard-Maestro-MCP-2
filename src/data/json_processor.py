"""
JSON processing and schema validation engine.

This module provides comprehensive JSON processing capabilities including
parsing, validation, transformation, and query operations.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..core.data_structures import (
    DataSchema, SchemaId, QueryResult, DataFormat, SecurityLimits, DEFAULT_SECURITY_LIMITS
)
from ..core.either import Either
from ..core.errors import ValidationError, DataError, create_error_context
from ..core.types import Duration


class JSONProcessor:
    """Comprehensive JSON processing with security validation."""
    
    def __init__(self, security_limits: SecurityLimits = DEFAULT_SECURITY_LIMITS):
        self.security_limits = security_limits
    
    async def parse_json(
        self,
        json_string: str,
        schema: Optional[DataSchema] = None,
        strict_mode: bool = True
    ) -> Either[DataError, Dict[str, Any]]:
        """Parse JSON string with optional schema validation."""
        try:
            start_time = datetime.now()
            
            # Validate input size
            size_result = self.security_limits.validate_size(len(json_string.encode('utf-8')), "value")
            if size_result.is_left():
                return Either.left(DataError("parse_json", f"Input too large: {size_result.get_left().message}"))
            
            # Basic security checks
            security_result = self._validate_json_security(json_string)
            if security_result.is_left():
                return security_result
            
            # Parse JSON
            try:
                data = json.loads(json_string)
            except json.JSONDecodeError as e:
                return Either.left(DataError("parse_json", f"Invalid JSON: {str(e)}"))
            
            # Validate structure depth
            depth = self._calculate_depth(data)
            depth_result = self.security_limits.validate_depth(depth)
            if depth_result.is_left():
                return Either.left(DataError("parse_json", f"Structure too deep: {depth_result.get_left().message}"))
            
            # Schema validation if provided
            if schema:
                schema_result = self._validate_against_schema(data, schema, strict_mode)
                if schema_result.is_left():
                    return Either.left(DataError("parse_json", f"Schema validation failed: {schema_result.get_left().message}"))
            
            return Either.right(data)
            
        except Exception as e:
            context = create_error_context("parse_json", "json_processor", error=str(e))
            return Either.left(DataError("parse_json", f"Unexpected error: {str(e)}", context))
    
    async def generate_json(
        self,
        data: Dict[str, Any],
        format_options: Optional[Dict[str, Any]] = None,
        schema: Optional[DataSchema] = None
    ) -> Either[DataError, str]:
        """Generate JSON string from data with formatting options."""
        try:
            # Validate data structure
            depth = self._calculate_depth(data)
            depth_result = self.security_limits.validate_depth(depth)
            if depth_result.is_left():
                return Either.left(DataError("generate_json", f"Data too deep: {depth_result.get_left().message}"))
            
            # Schema validation if provided
            if schema:
                schema_result = self._validate_against_schema(data, schema, True)
                if schema_result.is_left():
                    return Either.left(DataError("generate_json", f"Schema validation failed: {schema_result.get_left().message}"))
            
            # Apply formatting options
            options = format_options or {}
            indent = options.get("indent", 2)
            sort_keys = options.get("sort_keys", True)
            ensure_ascii = options.get("ensure_ascii", False)
            separators = options.get("separators")
            
            # Generate JSON
            json_string = json.dumps(
                data,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=ensure_ascii,
                separators=separators
            )
            
            # Validate output size
            size_result = self.security_limits.validate_size(len(json_string.encode('utf-8')), "value")
            if size_result.is_left():
                return Either.left(DataError("generate_json", f"Output too large: {size_result.get_left().message}"))
            
            return Either.right(json_string)
            
        except (TypeError, ValueError) as e:
            return Either.left(DataError("generate_json", f"Serialization failed: {str(e)}"))
        except Exception as e:
            context = create_error_context("generate_json", "json_processor", error=str(e))
            return Either.left(DataError("generate_json", f"Unexpected error: {str(e)}", context))
    
    async def query_json(
        self,
        data: Dict[str, Any],
        query: str,
        query_type: str = "jsonpath"
    ) -> Either[DataError, QueryResult]:
        """Query JSON data using JSONPath or other query languages."""
        try:
            start_time = datetime.now()
            
            if query_type == "jsonpath":
                # Basic JSONPath implementation (simplified)
                results = self._execute_jsonpath_query(data, query)
            elif query_type == "jq":
                # Would integrate with jq if available
                return Either.left(DataError("query_json", "JQ queries not yet implemented"))
            else:
                return Either.left(DataError("query_json", f"Unsupported query type: {query_type}"))
            
            end_time = datetime.now()
            execution_time = Duration.from_seconds((end_time - start_time).total_seconds())
            
            query_result = QueryResult(
                query=query,
                results=results,
                total_matches=len(results),
                execution_time=execution_time,
                metadata={"query_type": query_type}
            )
            
            return Either.right(query_result)
            
        except Exception as e:
            context = create_error_context("query_json", "json_processor", query=query, query_type=query_type)
            return Either.left(DataError("query_json", f"Query execution failed: {str(e)}", context))
    
    async def transform_json(
        self,
        data: Dict[str, Any],
        transformations: Dict[str, Any]
    ) -> Either[DataError, Dict[str, Any]]:
        """Transform JSON data according to transformation rules."""
        try:
            result = data.copy()
            
            # Apply key transformations
            if "key_case" in transformations:
                case_style = transformations["key_case"]
                result = self._transform_keys(result, case_style)
            
            # Apply value transformations
            if "value_transforms" in transformations:
                value_transforms = transformations["value_transforms"]
                result = self._transform_values(result, value_transforms)
            
            # Apply structure transformations
            if "structure" in transformations:
                structure_transforms = transformations["structure"]
                result = self._transform_structure(result, structure_transforms)
            
            return Either.right(result)
            
        except Exception as e:
            context = create_error_context("transform_json", "json_processor", error=str(e))
            return Either.left(DataError("transform_json", f"Transformation failed: {str(e)}", context))
    
    def _validate_json_security(self, json_string: str) -> Either[DataError, None]:
        """Validate JSON string for security issues."""
        # Check for potential security patterns
        dangerous_patterns = [
            r'__proto__',                          # Prototype pollution
            r'constructor\.prototype',             # Prototype manipulation
            r'eval\s*\(',                         # Code injection
            r'Function\s*\(',                     # Function constructor
            r'setTimeout\s*\(',                   # Timer functions
            r'setInterval\s*\(',                  # Timer functions
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, json_string, re.IGNORECASE):
                return Either.left(DataError("validate_json_security", f"Dangerous pattern detected: {pattern}"))
        
        return Either.right(None)
    
    def _calculate_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate nesting depth of data structure."""
        if current_depth > self.security_limits.max_nesting_depth:
            return current_depth
        
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(
                self._calculate_depth(v, current_depth + 1)
                for v in data.values()
            )
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(
                self._calculate_depth(item, current_depth + 1)
                for item in data
            )
        else:
            return current_depth
    
    def _validate_against_schema(self, data: Any, schema: DataSchema, strict_mode: bool) -> Either[ValidationError, None]:
        """Validate data against JSON schema (simplified implementation)."""
        schema_dict = schema.schema
        
        # Basic type validation
        expected_type = schema_dict.get("type")
        if expected_type:
            if expected_type == "object" and not isinstance(data, dict):
                return Either.left(ValidationError("Expected object type"))
            elif expected_type == "array" and not isinstance(data, list):
                return Either.left(ValidationError("Expected array type"))
            elif expected_type == "string" and not isinstance(data, str):
                return Either.left(ValidationError("Expected string type"))
            elif expected_type == "number" and not isinstance(data, (int, float)):
                return Either.left(ValidationError("Expected number type"))
            elif expected_type == "integer" and not isinstance(data, int):
                return Either.left(ValidationError("Expected integer type"))
            elif expected_type == "boolean" and not isinstance(data, bool):
                return Either.left(ValidationError("Expected boolean type"))
        
        # Object validation
        if isinstance(data, dict) and expected_type == "object":
            # Required properties
            required = schema_dict.get("required", [])
            for req_prop in required:
                if req_prop not in data:
                    return Either.left(ValidationError(f"Required property '{req_prop}' missing"))
            
            # Property validation
            properties = schema_dict.get("properties", {})
            for prop, value in data.items():
                if prop in properties:
                    # Recursive validation would go here
                    prop_schema = DataSchema(properties[prop], SchemaId(f"prop_{prop}"))
                    prop_result = self._validate_against_schema(value, prop_schema, strict_mode)
                    if prop_result.is_left():
                        return prop_result
                elif strict_mode and not schema.allow_additional:
                    return Either.left(ValidationError(f"Additional property '{prop}' not allowed"))
        
        # Array validation
        if isinstance(data, list) and expected_type == "array":
            items_schema = schema_dict.get("items")
            if items_schema:
                item_schema = DataSchema(items_schema, SchemaId("array_items"))
                for i, item in enumerate(data):
                    item_result = self._validate_against_schema(item, item_schema, strict_mode)
                    if item_result.is_left():
                        return Either.left(ValidationError(f"Array item {i} validation failed: {item_result.get_left().message}"))
        
        return Either.right(None)
    
    def _execute_jsonpath_query(self, data: Dict[str, Any], query: str) -> List[Any]:
        """Execute simplified JSONPath query."""
        # This is a very basic JSONPath implementation
        # In production, you would use a proper JSONPath library like jsonpath-ng
        
        if query == "$":
            return [data]
        elif query.startswith("$."):
            # Simple property access
            path = query[2:]  # Remove "$."
            parts = path.split(".")
            
            current = [data]
            for part in parts:
                new_current = []
                for item in current:
                    if isinstance(item, dict) and part in item:
                        new_current.append(item[part])
                    elif part == "*" and isinstance(item, dict):
                        new_current.extend(item.values())
                    elif part.endswith("[*]") and isinstance(item, dict):
                        key = part[:-3]  # Remove "[*]"
                        if key in item and isinstance(item[key], list):
                            new_current.extend(item[key])
                current = new_current
            
            return current
        else:
            # Unsupported query format
            return []
    
    def _transform_keys(self, data: Dict[str, Any], case_style: str) -> Dict[str, Any]:
        """Transform keys to specified case style."""
        if case_style == "camelCase":
            return self._to_camel_case(data)
        elif case_style == "snake_case":
            return self._to_snake_case(data)
        elif case_style == "kebab-case":
            return self._to_kebab_case(data)
        else:
            return data
    
    def _to_camel_case(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert keys to camelCase."""
        if isinstance(data, dict):
            return {
                self._snake_to_camel(k): self._to_camel_case(v) if isinstance(v, dict) else v
                for k, v in data.items()
            }
        return data
    
    def _to_snake_case(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert keys to snake_case."""
        if isinstance(data, dict):
            return {
                self._camel_to_snake(k): self._to_snake_case(v) if isinstance(v, dict) else v
                for k, v in data.items()
            }
        return data
    
    def _to_kebab_case(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert keys to kebab-case."""
        if isinstance(data, dict):
            return {
                k.replace("_", "-").replace(" ", "-").lower(): self._to_kebab_case(v) if isinstance(v, dict) else v
                for k, v in data.items()
            }
        return data
    
    def _snake_to_camel(self, snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split('_')
        return components[0] + ''.join(word.capitalize() for word in components[1:])
    
    def _camel_to_snake(self, camel_str: str) -> str:
        """Convert camelCase to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _transform_values(self, data: Dict[str, Any], transforms: Dict[str, str]) -> Dict[str, Any]:
        """Transform values according to transformation rules."""
        result = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = self._transform_values(value, transforms)
            elif key in transforms:
                transform_type = transforms[key]
                if transform_type == "string":
                    result[key] = str(value)
                elif transform_type == "int":
                    try:
                        result[key] = int(value)
                    except (ValueError, TypeError):
                        result[key] = value
                elif transform_type == "float":
                    try:
                        result[key] = float(value)
                    except (ValueError, TypeError):
                        result[key] = value
                elif transform_type == "bool":
                    result[key] = bool(value)
                else:
                    result[key] = value
            else:
                result[key] = value
        
        return result
    
    def _transform_structure(self, data: Dict[str, Any], structure_transforms: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data structure according to rules."""
        # Basic structure transformation (can be extended)
        result = data.copy()
        
        # Rename keys
        if "rename_keys" in structure_transforms:
            rename_map = structure_transforms["rename_keys"]
            for old_key, new_key in rename_map.items():
                if old_key in result:
                    result[new_key] = result.pop(old_key)
        
        # Remove keys
        if "remove_keys" in structure_transforms:
            remove_keys = structure_transforms["remove_keys"]
            for key in remove_keys:
                result.pop(key, None)
        
        return result