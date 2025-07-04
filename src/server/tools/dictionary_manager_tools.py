"""
Dictionary manager MCP tools for advanced data structures and JSON handling.

This module provides comprehensive dictionary management capabilities through
MCP tools, enabling sophisticated data-driven automation workflows.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ...core.data_structures import (
    DictionaryPath, DataSchema, SchemaId, DataOperation, MergeStrategy,
    DataFormat, SecurityLimits, DEFAULT_SECURITY_LIMITS
)
from ...core.either import Either
from ...core.errors import DataError, create_error_context
from ...data.dictionary_engine import DictionaryEngine
from ...data.json_processor import JSONProcessor

logger = logging.getLogger(__name__)


class DictionaryManagerTools:
    """Comprehensive dictionary and JSON management tools."""
    
    def __init__(self):
        self.engine = DictionaryEngine()
        self.json_processor = JSONProcessor()
        self.security_limits = DEFAULT_SECURITY_LIMITS
    
    async def km_dictionary_manager(
        self,
        operation: str,
        dictionary_name: str,
        key_path: Optional[str] = None,
        value: Optional[Any] = None,
        schema: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        merge_strategy: str = "deep",
        format_output: str = "json",
        validate_schema: bool = True,
        timeout_seconds: int = 30,
        ctx = None
    ) -> Dict[str, Any]:
        """
        Comprehensive dictionary and JSON management tool.
        
        Supports create, read, update, delete, query, merge, and transform operations
        on structured data with schema validation and security protection.
        """
        try:
            start_time = datetime.now()
            
            # Validate operation
            try:
                data_operation = DataOperation(operation)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid operation: {operation}",
                    "valid_operations": [op.value for op in DataOperation]
                }
            
            # Validate dictionary name
            if not dictionary_name or len(dictionary_name.strip()) == 0:
                return {
                    "success": False,
                    "error": "Dictionary name is required"
                }
            
            # Parse schema if provided
            data_schema = None
            if schema and validate_schema:
                try:
                    schema_id = SchemaId(f"schema_{dictionary_name}_{hash(json.dumps(schema, sort_keys=True)) % 10000}")
                    data_schema = DataSchema(schema=schema, schema_id=schema_id)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Invalid schema: {str(e)}"
                    }
            
            # Parse key path if provided
            dict_path = None
            if key_path:
                try:
                    dict_path = DictionaryPath(key_path)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Invalid key path: {str(e)}"
                    }
            
            # Execute operation
            if data_operation == DataOperation.CREATE:
                result = await self._handle_create(dictionary_name, value, data_schema)
            elif data_operation == DataOperation.READ:
                result = await self._handle_read(dictionary_name, dict_path, format_output)
            elif data_operation == DataOperation.UPDATE:
                result = await self._handle_update(dictionary_name, dict_path, value)
            elif data_operation == DataOperation.DELETE:
                result = await self._handle_delete(dictionary_name, dict_path)
            elif data_operation == DataOperation.QUERY:
                result = await self._handle_query(dictionary_name, query)
            elif data_operation == DataOperation.MERGE:
                result = await self._handle_merge(dictionary_name, value, merge_strategy)
            elif data_operation == DataOperation.TRANSFORM:
                result = await self._handle_transform(dictionary_name, value)
            elif data_operation == DataOperation.VALIDATE:
                result = await self._handle_validate(dictionary_name, data_schema)
            elif data_operation == DataOperation.EXPORT:
                result = await self._handle_export(dictionary_name, format_output, query)
            elif data_operation == DataOperation.IMPORT:
                result = await self._handle_import(dictionary_name, value, format_output, data_schema)
            else:
                return {
                    "success": False,
                    "error": f"Operation {operation} not implemented"
                }
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Add metadata to result
            if isinstance(result, dict) and result.get("success"):
                result["metadata"] = {
                    "operation": operation,
                    "dictionary_name": dictionary_name,
                    "execution_time_seconds": execution_time,
                    "timestamp": end_time.isoformat()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Dictionary manager error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "operation": operation,
                "dictionary_name": dictionary_name
            }
    
    async def _handle_create(
        self,
        dictionary_name: str,
        initial_data: Optional[Any],
        schema: Optional[DataSchema]
    ) -> Dict[str, Any]:
        """Handle dictionary creation."""
        data = initial_data if isinstance(initial_data, dict) else {}
        
        result = await self.engine.create_dictionary(
            name=dictionary_name,
            initial_data=data,
            schema=schema
        )
        
        if result.is_right():
            metadata = result.get_right()
            return {
                "success": True,
                "message": f"Dictionary '{dictionary_name}' created successfully",
                "dictionary_id": metadata.dictionary_id,
                "metadata": {
                    "name": metadata.name,
                    "created_at": metadata.created_at.isoformat(),
                    "size_bytes": metadata.size_bytes,
                    "key_count": metadata.key_count,
                    "nested_levels": metadata.nested_levels,
                    "schema_id": metadata.schema_id
                }
            }
        else:
            error = result.get_left()
            return {
                "success": False,
                "error": error.message,
                "error_category": error.category.value if hasattr(error, 'category') else "unknown"
            }
    
    async def _handle_read(
        self,
        dictionary_name: str,
        path: Optional[DictionaryPath],
        format_output: str
    ) -> Dict[str, Any]:
        """Handle reading dictionary values."""
        result = await self.engine.get_value(dictionary_name, path)
        
        if result.is_right():
            data = result.get_right()
            
            # Format output
            if format_output == "json":
                try:
                    formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
                except TypeError:
                    formatted_data = str(data)
            elif format_output == "yaml":
                # Would use PyYAML if available
                formatted_data = str(data)
            else:
                formatted_data = data
            
            return {
                "success": True,
                "data": data,
                "formatted_data": formatted_data,
                "path": path.path if path else None,
                "format": format_output
            }
        else:
            error = result.get_left()
            return {
                "success": False,
                "error": error.message,
                "path": path.path if path else None
            }
    
    async def _handle_update(
        self,
        dictionary_name: str,
        path: Optional[DictionaryPath],
        value: Any
    ) -> Dict[str, Any]:
        """Handle updating dictionary values."""
        if path is None:
            return {
                "success": False,
                "error": "Key path is required for update operation"
            }
        
        result = await self.engine.set_value(dictionary_name, path, value)
        
        if result.is_right():
            # Get updated metadata
            metadata = self.engine.get_dictionary_metadata(dictionary_name)
            return {
                "success": True,
                "message": f"Value updated at path '{path.path}'",
                "path": path.path,
                "value": value,
                "metadata": {
                    "size_bytes": metadata.size_bytes if metadata else 0,
                    "key_count": metadata.key_count if metadata else 0,
                    "modified_at": metadata.modified_at.isoformat() if metadata else None
                }
            }
        else:
            error = result.get_left()
            return {
                "success": False,
                "error": error.message,
                "path": path.path
            }
    
    async def _handle_delete(
        self,
        dictionary_name: str,
        path: Optional[DictionaryPath]
    ) -> Dict[str, Any]:
        """Handle deleting dictionary keys."""
        if path is None:
            return {
                "success": False,
                "error": "Key path is required for delete operation"
            }
        
        result = await self.engine.delete_key(dictionary_name, path)
        
        if result.is_right():
            metadata = self.engine.get_dictionary_metadata(dictionary_name)
            return {
                "success": True,
                "message": f"Key deleted at path '{path.path}'",
                "path": path.path,
                "metadata": {
                    "size_bytes": metadata.size_bytes if metadata else 0,
                    "key_count": metadata.key_count if metadata else 0,
                    "modified_at": metadata.modified_at.isoformat() if metadata else None
                }
            }
        else:
            error = result.get_left()
            return {
                "success": False,
                "error": error.message,
                "path": path.path
            }
    
    async def _handle_query(
        self,
        dictionary_name: str,
        query: Optional[str]
    ) -> Dict[str, Any]:
        """Handle querying dictionary data."""
        if not query:
            return {
                "success": False,
                "error": "Query is required for query operation"
            }
        
        # Get dictionary data
        get_result = await self.engine.get_value(dictionary_name)
        if get_result.is_left():
            error = get_result.get_left()
            return {
                "success": False,
                "error": f"Failed to get dictionary: {error.message}"
            }
        
        data = get_result.get_right()
        
        # Execute query
        query_result = await self.json_processor.query_json(data, query)
        
        if query_result.is_right():
            result = query_result.get_right()
            return {
                "success": True,
                "query": query,
                "results": result.results,
                "total_matches": result.total_matches,
                "execution_time_seconds": result.execution_time.total_seconds(),
                "metadata": result.metadata
            }
        else:
            error = query_result.get_left()
            return {
                "success": False,
                "error": error.message,
                "query": query
            }
    
    async def _handle_merge(
        self,
        dictionary_name: str,
        source_data: Any,
        merge_strategy: str
    ) -> Dict[str, Any]:
        """Handle merging data into dictionary."""
        if not isinstance(source_data, dict):
            return {
                "success": False,
                "error": "Source data must be a dictionary for merge operation"
            }
        
        try:
            strategy = MergeStrategy(merge_strategy)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid merge strategy: {merge_strategy}",
                "valid_strategies": [s.value for s in MergeStrategy]
            }
        
        # Get current dictionary data
        get_result = await self.engine.get_value(dictionary_name)
        if get_result.is_left():
            error = get_result.get_left()
            return {
                "success": False,
                "error": f"Failed to get dictionary: {error.message}"
            }
        
        current_data = get_result.get_right()
        
        # Perform merge based on strategy
        try:
            if strategy == MergeStrategy.DEEP:
                merged_data = self._deep_merge(current_data, source_data)
            elif strategy == MergeStrategy.SHALLOW:
                merged_data = {**current_data, **source_data}
            elif strategy == MergeStrategy.REPLACE:
                merged_data = source_data.copy()
            else:
                return {
                    "success": False,
                    "error": f"Merge strategy {merge_strategy} not implemented"
                }
            
            # Update dictionary with merged data
            for key, value in merged_data.items():
                path = DictionaryPath(key)
                set_result = await self.engine.set_value(dictionary_name, path, value)
                if set_result.is_left():
                    error = set_result.get_left()
                    return {
                        "success": False,
                        "error": f"Failed to set merged value: {error.message}",
                        "key": key
                    }
            
            metadata = self.engine.get_dictionary_metadata(dictionary_name)
            return {
                "success": True,
                "message": f"Data merged using {merge_strategy} strategy",
                "merge_strategy": merge_strategy,
                "keys_merged": len(source_data),
                "metadata": {
                    "size_bytes": metadata.size_bytes if metadata else 0,
                    "key_count": metadata.key_count if metadata else 0,
                    "modified_at": metadata.modified_at.isoformat() if metadata else None
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Merge failed: {str(e)}",
                "merge_strategy": merge_strategy
            }
    
    async def _handle_transform(
        self,
        dictionary_name: str,
        transformations: Any
    ) -> Dict[str, Any]:
        """Handle transforming dictionary data."""
        if not isinstance(transformations, dict):
            return {
                "success": False,
                "error": "Transformations must be a dictionary"
            }
        
        # Get dictionary data
        get_result = await self.engine.get_value(dictionary_name)
        if get_result.is_left():
            error = get_result.get_left()
            return {
                "success": False,
                "error": f"Failed to get dictionary: {error.message}"
            }
        
        data = get_result.get_right()
        
        # Apply transformations
        transform_result = await self.json_processor.transform_json(data, transformations)
        
        if transform_result.is_right():
            transformed_data = transform_result.get_right()
            
            # Update dictionary with transformed data
            # This would replace the entire dictionary - could be refined
            for key, value in transformed_data.items():
                path = DictionaryPath(key)
                set_result = await self.engine.set_value(dictionary_name, path, value)
                if set_result.is_left():
                    error = set_result.get_left()
                    return {
                        "success": False,
                        "error": f"Failed to set transformed value: {error.message}",
                        "key": key
                    }
            
            return {
                "success": True,
                "message": "Data transformed successfully",
                "transformations_applied": list(transformations.keys()),
                "data": transformed_data
            }
        else:
            error = transform_result.get_left()
            return {
                "success": False,
                "error": error.message,
                "transformations": transformations
            }
    
    async def _handle_validate(
        self,
        dictionary_name: str,
        schema: Optional[DataSchema]
    ) -> Dict[str, Any]:
        """Handle validating dictionary against schema."""
        if not schema:
            return {
                "success": False,
                "error": "Schema is required for validation operation"
            }
        
        # Get dictionary data
        get_result = await self.engine.get_value(dictionary_name)
        if get_result.is_left():
            error = get_result.get_left()
            return {
                "success": False,
                "error": f"Failed to get dictionary: {error.message}"
            }
        
        data = get_result.get_right()
        
        # Validate against schema
        validation_result = self.json_processor._validate_against_schema(data, schema, True)
        
        if validation_result.is_right():
            return {
                "success": True,
                "message": "Dictionary validates against schema",
                "schema_id": schema.schema_id,
                "valid": True
            }
        else:
            error = validation_result.get_left()
            return {
                "success": True,  # Operation succeeded, but validation failed
                "message": "Dictionary does not validate against schema",
                "schema_id": schema.schema_id,
                "valid": False,
                "validation_error": error.message
            }
    
    async def _handle_export(
        self,
        dictionary_name: str,
        format_output: str,
        query: Optional[str]
    ) -> Dict[str, Any]:
        """Handle exporting dictionary data."""
        # Get dictionary data
        get_result = await self.engine.get_value(dictionary_name)
        if get_result.is_left():
            error = get_result.get_left()
            return {
                "success": False,
                "error": f"Failed to get dictionary: {error.message}"
            }
        
        data = get_result.get_right()
        
        # Apply query filter if provided
        if query:
            query_result = await self.json_processor.query_json(data, query)
            if query_result.is_right():
                result_data = query_result.get_right()
                data = {"query_results": result_data.results}
            else:
                error = query_result.get_left()
                return {
                    "success": False,
                    "error": f"Query failed: {error.message}",
                    "query": query
                }
        
        # Format data for export
        try:
            if format_output == "json":
                export_data = json.dumps(data, indent=2, ensure_ascii=False)
            elif format_output == "yaml":
                # Would use PyYAML if available
                export_data = str(data)
            elif format_output == "csv":
                # Basic CSV export for flat data
                if isinstance(data, dict) and all(isinstance(v, (str, int, float, bool)) for v in data.values()):
                    import csv
                    import io
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(data.keys())
                    writer.writerow(data.values())
                    export_data = output.getvalue()
                else:
                    return {
                        "success": False,
                        "error": "CSV export requires flat dictionary structure"
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported export format: {format_output}",
                    "supported_formats": ["json", "yaml", "csv"]
                }
            
            return {
                "success": True,
                "message": f"Dictionary exported as {format_output}",
                "format": format_output,
                "export_data": export_data,
                "size_bytes": len(export_data.encode('utf-8'))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Export failed: {str(e)}",
                "format": format_output
            }
    
    async def _handle_import(
        self,
        dictionary_name: str,
        import_data: Any,
        format_input: str,
        schema: Optional[DataSchema]
    ) -> Dict[str, Any]:
        """Handle importing data into dictionary."""
        try:
            # Parse import data based on format
            if format_input == "json":
                if isinstance(import_data, str):
                    parse_result = await self.json_processor.parse_json(import_data, schema)
                    if parse_result.is_left():
                        error = parse_result.get_left()
                        return {
                            "success": False,
                            "error": f"JSON parsing failed: {error.message}"
                        }
                    data = parse_result.get_right()
                elif isinstance(import_data, dict):
                    data = import_data
                else:
                    return {
                        "success": False,
                        "error": "JSON import data must be string or dictionary"
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported import format: {format_input}",
                    "supported_formats": ["json"]
                }
            
            # Create or update dictionary with imported data
            create_result = await self.engine.create_dictionary(
                name=dictionary_name,
                initial_data=data,
                schema=schema
            )
            
            if create_result.is_right():
                metadata = create_result.get_right()
                return {
                    "success": True,
                    "message": f"Data imported from {format_input}",
                    "format": format_input,
                    "metadata": {
                        "dictionary_id": metadata.dictionary_id,
                        "size_bytes": metadata.size_bytes,
                        "key_count": metadata.key_count
                    }
                }
            else:
                error = create_result.get_left()
                return {
                    "success": False,
                    "error": f"Import failed: {error.message}",
                    "format": format_input
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Import failed: {str(e)}",
                "format": format_input
            }
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep merge of two dictionaries."""
        result = target.copy()
        
        for key, value in source.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


# Create global instance
dictionary_manager_tools = DictionaryManagerTools()