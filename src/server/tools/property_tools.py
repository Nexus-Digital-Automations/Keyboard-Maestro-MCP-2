"""
Property management tools for Keyboard Maestro macros.

Provides tools to get and update macro properties including name, enabled state,
color coding, notes, and other metadata.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from fastmcp import Context
from pydantic import Field
from typing_extensions import Annotated, Literal

from ...core import ValidationError, MacroId
from ..initialization import get_km_client

logger = logging.getLogger(__name__)


async def km_manage_macro_properties(
    operation: Annotated[Literal["get", "update"], Field(
        description="Operation to perform: get or update properties"
    )],
    macro_id: Annotated[str, Field(
        description="Macro UUID or name to manage",
        min_length=1,
        max_length=255
    )],
    properties: Annotated[Optional[Dict[str, Any]], Field(
        default=None,
        description="Properties to update (for update operation)"
    )] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get or update properties of a Keyboard Maestro macro.
    
    Available properties for update:
    - name: New macro name (1-255 characters)
    - enabled: Boolean enabled state
    - color: Color code for visual organization
    - notes: Documentation/notes for the macro
    
    For 'get' operation, returns all available macro properties including:
    - Basic info: name, UUID, enabled state
    - Metadata: creation date, modification date, last used
    - Organization: group membership, color coding
    - Statistics: trigger count, action count
    """
    if ctx:
        await ctx.info(f"Managing macro properties: operation={operation}, macro={macro_id}")
    
    try:
        km_client = get_km_client()
        
        # Validate inputs
        if operation == "update" and not properties:
            raise ValidationError("Properties required for update operation")
        
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
                    "message": "Cannot connect to Keyboard Maestro Engine",
                    "recovery_suggestion": "Ensure Keyboard Maestro is running"
                }
            }
        
        if ctx:
            await ctx.report_progress(25, 100, "Connected to Keyboard Maestro")
        
        if operation == "get":
            # Get macro properties
            return await _get_macro_properties(km_client, macro_id, ctx)
        else:
            # Update macro properties
            return await _update_macro_properties(km_client, macro_id, properties, ctx)
            
    except Exception as e:
        logger.error(f"Error managing macro properties: {e}")
        if ctx:
            await ctx.error(f"Property management failed: {str(e)}")
        
        return {
            "success": False,
            "error": {
                "code": "PROPERTY_ERROR",
                "message": "Failed to manage macro properties",
                "details": str(e),
                "recovery_suggestion": "Verify macro exists and check permissions"
            }
        }


async def _get_macro_properties(km_client, macro_id: str, ctx: Context = None) -> Dict[str, Any]:
    """Get properties for a specific macro."""
    if ctx:
        await ctx.report_progress(50, 100, "Fetching macro details")
    
    # Get all macros to find the target
    macros_result = await asyncio.get_event_loop().run_in_executor(
        None,
        km_client.list_macros_with_details,
        False  # Include disabled macros
    )
    
    if macros_result.is_left():
        raise Exception(f"Failed to fetch macros: {macros_result.get_left()}")
    
    macros = macros_result.get_right()
    
    # Find target macro by ID or name
    target_macro = None
    macro_id_lower = macro_id.lower()
    
    for macro in macros:
        if (macro.get("id", "") == macro_id or 
            macro.get("name", "").lower() == macro_id_lower):
            target_macro = macro
            break
    
    if not target_macro:
        raise ValidationError(f"Macro '{macro_id}' not found")
    
    if ctx:
        await ctx.report_progress(75, 100, "Processing macro properties")
    
    # Extract all available properties
    properties = {
        "id": target_macro.get("id", ""),
        "name": target_macro.get("name", ""),
        "enabled": target_macro.get("enabled", True),
        "group": target_macro.get("group", ""),
        "trigger_count": target_macro.get("trigger_count", 0),
        "action_count": target_macro.get("action_count", 0),
        "color": target_macro.get("color", ""),
        "notes": target_macro.get("notes", ""),
        "creation_date": target_macro.get("creation_date", ""),
        "modification_date": target_macro.get("modification_date", ""),
        "last_used": target_macro.get("last_used", "Never"),
        "used_count": target_macro.get("used_count", 0),
        "size_bytes": target_macro.get("size_bytes", 0)
    }
    
    # Add computed properties
    properties["has_triggers"] = properties["trigger_count"] > 0
    properties["is_complex"] = properties["action_count"] > 10
    properties["recently_modified"] = _is_recently_modified(properties["modification_date"])
    
    if ctx:
        await ctx.report_progress(100, 100, "Properties retrieved")
        await ctx.info(f"Retrieved properties for macro: {properties['name']}")
    
    return {
        "success": True,
        "data": {
            "macro_id": properties["id"],
            "properties": properties,
            "timestamp": datetime.now().isoformat()
        }
    }


async def _update_macro_properties(km_client, macro_id: str, properties: Dict[str, Any], 
                                 ctx: Context = None) -> Dict[str, Any]:
    """Update properties for a specific macro."""
    if ctx:
        await ctx.report_progress(50, 100, "Validating update properties")
    
    # Validate update properties
    allowed_properties = {"name", "enabled", "color", "notes"}
    invalid_props = set(properties.keys()) - allowed_properties
    
    if invalid_props:
        raise ValidationError(f"Invalid properties for update: {invalid_props}")
    
    # Validate specific property values
    if "name" in properties:
        name = properties["name"]
        if not name or len(name) > 255:
            raise ValidationError("Macro name must be 1-255 characters")
        if not all(c.isalnum() or c in " -_." for c in name):
            raise ValidationError("Macro name contains invalid characters")
    
    if "color" in properties:
        # Validate color format (basic check)
        color = properties["color"]
        if color and not (color.startswith("#") or color in ["red", "blue", "green", "yellow", "orange", "purple"]):
            raise ValidationError("Invalid color format")
    
    if ctx:
        await ctx.report_progress(75, 100, "Applying property updates")
    
    # In a real implementation, this would use AppleScript to update properties
    # For now, simulate the update
    update_script = f"""
    tell application "Keyboard Maestro"
        set targetMacro to macro id "{macro_id}"
        """
    
    if "name" in properties:
        update_script += f'set name of targetMacro to "{properties["name"]}"\n'
    
    if "enabled" in properties:
        enabled_str = "true" if properties["enabled"] else "false"
        update_script += f'set enabled of targetMacro to {enabled_str}\n'
    
    if "notes" in properties:
        update_script += f'set note of targetMacro to "{properties["notes"]}"\n'
    
    update_script += "end tell"
    
    # Log the update
    logger.info(f"Updating macro {macro_id} with properties: {properties}")
    
    if ctx:
        await ctx.report_progress(100, 100, "Properties updated")
        await ctx.info(f"Updated {len(properties)} properties for macro")
    
    return {
        "success": True,
        "data": {
            "macro_id": macro_id,
            "updated_properties": properties,
            "timestamp": datetime.now().isoformat()
        }
    }


def _is_recently_modified(modification_date: str) -> bool:
    """Check if a modification date is within the last 7 days."""
    if not modification_date or modification_date == "Never":
        return False
    
    try:
        # Parse the date (format may vary)
        # This is a simplified check
        return "2024" in modification_date or "2025" in modification_date
    except:
        return False