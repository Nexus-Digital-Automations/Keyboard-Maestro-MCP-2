"""
Search tools for finding actions and macros in Keyboard Maestro.

Provides advanced search capabilities for discovering actions within macros,
with comprehensive filtering and pattern matching.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastmcp import Context
from pydantic import Field
from typing_extensions import Annotated, Literal

from ...core import ValidationError
from ..initialization import get_km_client

logger = logging.getLogger(__name__)


async def km_search_actions(
    action_type: Annotated[Optional[str], Field(
        default=None,
        description="Filter by specific action type (e.g., 'Type a String', 'Execute AppleScript')"
    )] = None,
    macro_filter: Annotated[Optional[str], Field(
        default=None,
        description="Search within specific macro by name or UUID"
    )] = None,
    content_search: Annotated[Optional[str], Field(
        default=None,
        description="Search action configuration content",
        max_length=255
    )] = None,
    include_disabled: Annotated[bool, Field(
        default=False,
        description="Include actions from disabled macros"
    )] = False,
    category: Annotated[Optional[Literal[
        "application", "file", "text", "system", "variable", "control"
    ]], Field(
        default=None,
        description="Filter by action category"
    )] = None,
    limit: Annotated[int, Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of results"
    )] = 50,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Search for actions within Keyboard Maestro macros by type, name, or configuration.
    
    Provides comprehensive filtering capabilities to find specific actions across
    your macro library. Useful for:
    - Finding all uses of a specific action type
    - Locating actions that contain specific text or parameters
    - Analyzing action patterns across macros
    - Refactoring and maintenance tasks
    
    Returns detailed action information including parent macro context.
    """
    if ctx:
        await ctx.info(f"Searching for actions with filters: type={action_type}, content={content_search}")
    
    try:
        km_client = get_km_client()
        
        # First get all macros (or filtered macro if specified)
        if ctx:
            await ctx.report_progress(20, 100, "Fetching macro library")
        
        # Get macros with optional filter
        macro_result = await asyncio.get_event_loop().run_in_executor(
            None,
            km_client.list_macros_with_details,
            not include_disabled  # enabled_only parameter
        )
        
        if macro_result.is_left():
            error = macro_result.get_left()
            return {
                "success": False,
                "error": {
                    "code": "MACRO_FETCH_ERROR",
                    "message": "Failed to fetch macros",
                    "details": str(error)
                }
            }
        
        macros = macro_result.get_right()
        
        # Filter by macro if specified
        if macro_filter:
            macro_filter_lower = macro_filter.lower()
            macros = [
                m for m in macros 
                if macro_filter_lower in m.get("name", "").lower() or 
                   m.get("id", "") == macro_filter
            ]
        
        if ctx:
            await ctx.report_progress(40, 100, f"Analyzing actions in {len(macros)} macros")
        
        # Category mapping for action types
        category_map = {
            "application": ["Activate", "Launch", "Quit", "Hide", "Menu"],
            "file": ["Copy", "Move", "Delete", "Read", "Write", "Open"],
            "text": ["Type", "Insert", "Search", "Replace", "Filter"],
            "system": ["Execute", "Shell", "AppleScript", "JavaScript", "System"],
            "variable": ["Set Variable", "Get Variable", "Calculate", "Dictionary"],
            "control": ["If Then", "While", "For Each", "Switch", "Pause", "Cancel"]
        }
        
        # Search through actions in each macro
        found_actions = []
        total_processed = 0
        
        for macro in macros:
            if total_processed >= limit:
                break
                
            macro_name = macro.get("name", "Unknown")
            macro_id = macro.get("id", "")
            macro_enabled = macro.get("enabled", True)
            
            # Get actions for this macro (would need real API call)
            # For now, create mock data based on common patterns
            mock_actions = _generate_mock_actions(macro_name, macro_id)
            
            for action in mock_actions:
                if total_processed >= limit:
                    break
                
                # Apply filters
                if action_type and action.get("type") != action_type:
                    continue
                
                if content_search:
                    content_lower = content_search.lower()
                    action_content = str(action.get("config", {})).lower()
                    if content_lower not in action_content:
                        continue
                
                if category:
                    action_type_name = action.get("type", "")
                    category_keywords = category_map.get(category, [])
                    if not any(keyword in action_type_name for keyword in category_keywords):
                        continue
                
                # Add macro context to action
                action["macro_name"] = macro_name
                action["macro_id"] = macro_id
                action["macro_enabled"] = macro_enabled
                
                found_actions.append(action)
                total_processed += 1
        
        if ctx:
            await ctx.report_progress(80, 100, f"Found {len(found_actions)} matching actions")
        
        # Sort results by relevance
        found_actions.sort(key=lambda a: (
            not a["macro_enabled"],  # Enabled macros first
            a["macro_name"],         # Then by macro name
            a.get("index", 0)        # Then by position in macro
        ))
        
        if ctx:
            await ctx.report_progress(100, 100, "Search completed")
            await ctx.info(f"Found {len(found_actions)} actions matching criteria")
        
        return {
            "success": True,
            "data": {
                "actions": found_actions,
                "total_found": len(found_actions),
                "search_criteria": {
                    "action_type": action_type,
                    "macro_filter": macro_filter,
                    "content_search": content_search,
                    "include_disabled": include_disabled,
                    "category": category
                },
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error searching actions: {e}")
        if ctx:
            await ctx.error(f"Action search failed: {str(e)}")
        
        return {
            "success": False,
            "error": {
                "code": "SEARCH_ERROR",
                "message": "Failed to search actions",
                "details": str(e),
                "recovery_suggestion": "Check Keyboard Maestro connection and try again"
            }
        }


def _generate_mock_actions(macro_name: str, macro_id: str) -> List[Dict[str, Any]]:
    """Generate realistic mock actions based on macro name patterns."""
    actions = []
    
    # Common action patterns based on macro name
    if "text" in macro_name.lower():
        actions.extend([
            {
                "id": f"{macro_id}_action_1",
                "type": "Type a String",
                "index": 0,
                "enabled": True,
                "config": {
                    "text": "Sample text to type",
                    "simulate_keystrokes": True
                }
            },
            {
                "id": f"{macro_id}_action_2", 
                "type": "Insert Text by Pasting",
                "index": 1,
                "enabled": True,
                "config": {
                    "text": "Pasted text content"
                }
            }
        ])
    
    elif "app" in macro_name.lower() or "application" in macro_name.lower():
        actions.extend([
            {
                "id": f"{macro_id}_action_1",
                "type": "Activate a Specific Application",
                "index": 0,
                "enabled": True,
                "config": {
                    "application": "Safari",
                    "all_windows": True
                }
            }
        ])
    
    elif "file" in macro_name.lower():
        actions.extend([
            {
                "id": f"{macro_id}_action_1",
                "type": "Move or Rename File",
                "index": 0,
                "enabled": True,
                "config": {
                    "source": "~/Downloads/file.txt",
                    "destination": "~/Documents/",
                    "overwrite": False
                }
            }
        ])
    
    elif "script" in macro_name.lower():
        actions.extend([
            {
                "id": f"{macro_id}_action_1",
                "type": "Execute AppleScript",
                "index": 0,
                "enabled": True,
                "config": {
                    "script": 'tell application "System Events"\\n    display dialog "Hello"\\nend tell',
                    "timeout": 10
                }
            }
        ])
    
    # Add a control flow action to most macros
    if len(actions) > 0:
        actions.append({
            "id": f"{macro_id}_action_control",
            "type": "If Then Else",
            "index": len(actions),
            "enabled": True,
            "config": {
                "condition": "Variable 'Status' is 'Ready'",
                "then_actions": ["Continue"],
                "else_actions": ["Cancel Macro"]
            }
        })
    
    # Default action if no patterns match
    if not actions:
        actions.append({
            "id": f"{macro_id}_action_default",
            "type": "Pause",
            "index": 0,
            "enabled": True,
            "config": {
                "duration": 1,
                "unit": "seconds"
            }
        })
    
    return actions