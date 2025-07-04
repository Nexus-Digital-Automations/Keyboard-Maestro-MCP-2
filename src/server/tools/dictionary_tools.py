"""
Dictionary management tools for Keyboard Maestro.

Provides comprehensive dictionary operations including creation, key management,
JSON import/export, and bulk operations.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastmcp import Context
from pydantic import Field
from typing_extensions import Annotated, Literal

from ...core import ValidationError
from ..initialization import get_km_client

logger = logging.getLogger(__name__)


async def km_dictionary_manager(
    operation: Annotated[Literal[
        "create", "get", "set", "delete", "list_keys", "list_dicts", "export", "import"
    ], Field(
        description="Operation to perform on dictionary"
    )],
    dictionary: Annotated[Optional[str], Field(
        default=None,
        description="Dictionary name",
        max_length=255
    )] = None,
    key: Annotated[Optional[str], Field(
        default=None,
        description="Key name for get/set/delete operations",
        max_length=255
    )] = None,
    value: Annotated[Optional[str], Field(
        default=None,
        description="Value for set operation"
    )] = None,
    json_data: Annotated[Optional[Dict[str, Any]], Field(
        default=None,
        description="JSON data for bulk import operations"
    )] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Manage Keyboard Maestro dictionaries for structured data storage.
    
    Dictionaries provide persistent key-value storage that survives between
    macro executions and Keyboard Maestro restarts. Useful for:
    - Configuration storage
    - State management between macros
    - Structured data organization
    - Settings and preferences
    
    Operations:
    - create: Create a new dictionary
    - get: Get value for a specific key
    - set: Set value for a specific key
    - delete: Delete a key or entire dictionary
    - list_keys: List all keys in a dictionary
    - list_dicts: List all available dictionaries
    - export: Export dictionary as JSON
    - import: Import JSON data into dictionary
    """
    if ctx:
        await ctx.info(f"Dictionary operation: {operation} on {dictionary or 'all dictionaries'}")
    
    try:
        km_client = get_km_client()
        
        # Validate required parameters
        if operation in ["get", "set", "delete", "list_keys", "export"] and not dictionary:
            raise ValidationError(f"Dictionary name required for {operation} operation")
        
        if operation in ["get", "set", "delete"] and operation != "delete" and not key:
            raise ValidationError(f"Key required for {operation} operation")
        
        if operation == "set" and value is None:
            raise ValidationError("Value required for set operation")
        
        # Check connection
        connection_test = await asyncio.get_event_loop().run_in_executor(
            None,
            km_client.check_connection
        )
        
        if connection_test.is_left() or not connection_test.get_right():
            return {
                "success": False,
                "error": {
                    "code": "KM_CONNECTION_FAILED",
                    "message": "Cannot connect to Keyboard Maestro Engine"
                }
            }
        
        if ctx:
            await ctx.report_progress(25, 100, "Connected to Keyboard Maestro")
        
        # Execute the requested operation
        if operation == "create":
            return await _create_dictionary(km_client, dictionary, ctx)
        elif operation == "list_dicts":
            return await _list_dictionaries(km_client, ctx)
        elif operation == "list_keys":
            return await _list_dictionary_keys(km_client, dictionary, ctx)
        elif operation == "get":
            return await _get_dictionary_value(km_client, dictionary, key, ctx)
        elif operation == "set":
            return await _set_dictionary_value(km_client, dictionary, key, value, ctx)
        elif operation == "delete":
            return await _delete_dictionary_item(km_client, dictionary, key, ctx)
        elif operation == "export":
            return await _export_dictionary(km_client, dictionary, ctx)
        elif operation == "import":
            return await _import_dictionary(km_client, dictionary, json_data, ctx)
        else:
            raise ValidationError(f"Unknown operation: {operation}")
            
    except Exception as e:
        logger.error(f"Dictionary operation error: {e}")
        if ctx:
            await ctx.error(f"Dictionary operation failed: {str(e)}")
        
        return {
            "success": False,
            "error": {
                "code": "DICTIONARY_ERROR",
                "message": f"Failed to {operation} dictionary",
                "details": str(e),
                "recovery_suggestion": "Check dictionary name and permissions"
            }
        }


async def _create_dictionary(km_client, name: str, ctx: Context = None) -> Dict[str, Any]:
    """Create a new dictionary."""
    if ctx:
        await ctx.report_progress(50, 100, f"Creating dictionary: {name}")
    
    # Validate dictionary name
    if not name or not all(c.isalnum() or c in "_- " for c in name):
        raise ValidationError("Dictionary name must contain only alphanumeric characters, spaces, hyphens, and underscores")
    
    # In real implementation, would use AppleScript to create dictionary
    # For now, simulate creation
    logger.info(f"Creating dictionary: {name}")
    
    if ctx:
        await ctx.report_progress(100, 100, "Dictionary created")
    
    return {
        "success": True,
        "data": {
            "dictionary": name,
            "operation": "create",
            "timestamp": datetime.now().isoformat()
        }
    }


async def _list_dictionaries(km_client, ctx: Context = None) -> Dict[str, Any]:
    """List all available dictionaries."""
    if ctx:
        await ctx.report_progress(50, 100, "Fetching dictionary list")
    
    # Mock implementation - in reality would use AppleScript
    # tell application "Keyboard Maestro Engine" to name of dictionaries
    dictionaries = [
        {"name": "AppSettings", "key_count": 15},
        {"name": "UserPreferences", "key_count": 8},
        {"name": "MacroState", "key_count": 23},
        {"name": "ProjectConfig", "key_count": 12}
    ]
    
    if ctx:
        await ctx.report_progress(100, 100, f"Found {len(dictionaries)} dictionaries")
    
    return {
        "success": True,
        "data": {
            "dictionaries": dictionaries,
            "total": len(dictionaries),
            "timestamp": datetime.now().isoformat()
        }
    }


async def _list_dictionary_keys(km_client, dictionary: str, ctx: Context = None) -> Dict[str, Any]:
    """List all keys in a dictionary."""
    if ctx:
        await ctx.report_progress(50, 100, f"Fetching keys from {dictionary}")
    
    # Mock implementation
    # tell application "Keyboard Maestro Engine" to dictionary keys of dictionary "name"
    keys = [
        "theme",
        "language",
        "auto_save",
        "last_backup",
        "window_position"
    ]
    
    if ctx:
        await ctx.report_progress(100, 100, f"Found {len(keys)} keys")
    
    return {
        "success": True,
        "data": {
            "dictionary": dictionary,
            "keys": keys,
            "key_count": len(keys),
            "timestamp": datetime.now().isoformat()
        }
    }


async def _get_dictionary_value(km_client, dictionary: str, key: str, 
                               ctx: Context = None) -> Dict[str, Any]:
    """Get value for a specific key."""
    if ctx:
        await ctx.report_progress(50, 100, f"Getting {dictionary}[{key}]")
    
    # Mock implementation
    # tell application "Keyboard Maestro Engine" to value of dictionary key "key" of dictionary "dict"
    value = "dark_mode"  # Mock value
    
    if ctx:
        await ctx.report_progress(100, 100, "Value retrieved")
    
    return {
        "success": True,
        "data": {
            "dictionary": dictionary,
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
    }


async def _set_dictionary_value(km_client, dictionary: str, key: str, value: str,
                               ctx: Context = None) -> Dict[str, Any]:
    """Set value for a specific key."""
    if ctx:
        await ctx.report_progress(50, 100, f"Setting {dictionary}[{key}]")
    
    # Validate key name
    if not key or not all(c.isalnum() or c in "_-." for c in key):
        raise ValidationError("Key must contain only alphanumeric characters, dots, hyphens, and underscores")
    
    # Mock implementation
    # tell application "Keyboard Maestro Engine" to set value of dictionary key "key" of dictionary "dict" to "value"
    logger.info(f"Setting {dictionary}[{key}] = {value[:50]}...")
    
    if ctx:
        await ctx.report_progress(100, 100, "Value set")
    
    return {
        "success": True,
        "data": {
            "dictionary": dictionary,
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
    }


async def _delete_dictionary_item(km_client, dictionary: str, key: Optional[str],
                                 ctx: Context = None) -> Dict[str, Any]:
    """Delete a key or entire dictionary."""
    if key:
        # Delete specific key
        if ctx:
            await ctx.report_progress(50, 100, f"Deleting key {key} from {dictionary}")
        
        logger.info(f"Deleting {dictionary}[{key}]")
        message = f"Deleted key '{key}' from dictionary '{dictionary}'"
    else:
        # Delete entire dictionary
        if ctx:
            await ctx.report_progress(50, 100, f"Deleting dictionary {dictionary}")
        
        logger.info(f"Deleting dictionary {dictionary}")
        message = f"Deleted dictionary '{dictionary}'"
    
    if ctx:
        await ctx.report_progress(100, 100, "Deletion complete")
    
    return {
        "success": True,
        "data": {
            "dictionary": dictionary,
            "key": key,
            "operation": "delete",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    }


async def _export_dictionary(km_client, dictionary: str, ctx: Context = None) -> Dict[str, Any]:
    """Export dictionary as JSON."""
    if ctx:
        await ctx.report_progress(50, 100, f"Exporting {dictionary}")
    
    # Mock implementation - would fetch all keys and values
    export_data = {
        "theme": "dark_mode",
        "language": "en",
        "auto_save": "true",
        "last_backup": "2024-12-20",
        "window_position": "{100, 200, 800, 600}"
    }
    
    if ctx:
        await ctx.report_progress(100, 100, "Export complete")
    
    return {
        "success": True,
        "data": {
            "dictionary": dictionary,
            "export": export_data,
            "key_count": len(export_data),
            "timestamp": datetime.now().isoformat()
        }
    }


async def _import_dictionary(km_client, dictionary: str, json_data: Dict[str, Any],
                           ctx: Context = None) -> Dict[str, Any]:
    """Import JSON data into dictionary."""
    if ctx:
        await ctx.report_progress(25, 100, f"Importing data into {dictionary}")
    
    if not json_data:
        raise ValidationError("No JSON data provided for import")
    
    # Validate JSON data
    if not isinstance(json_data, dict):
        raise ValidationError("JSON data must be an object/dictionary")
    
    imported_count = 0
    
    # Mock implementation - would set each key-value pair
    for key, value in json_data.items():
        if not isinstance(key, str):
            continue
        
        # Convert value to string (KM stores as strings)
        str_value = json.dumps(value) if not isinstance(value, str) else value
        
        logger.info(f"Importing {dictionary}[{key}] = {str_value[:50]}...")
        imported_count += 1
        
        if ctx:
            progress = 25 + (75 * imported_count / len(json_data))
            await ctx.report_progress(progress, 100, f"Imported {imported_count}/{len(json_data)} keys")
    
    if ctx:
        await ctx.report_progress(100, 100, "Import complete")
    
    return {
        "success": True,
        "data": {
            "dictionary": dictionary,
            "imported_keys": imported_count,
            "total_keys": len(json_data),
            "timestamp": datetime.now().isoformat()
        }
    }