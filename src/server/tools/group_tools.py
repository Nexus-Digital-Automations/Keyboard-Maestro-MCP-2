"""
Macro group management tools.

Contains tools for listing and managing Keyboard Maestro macro groups
with comprehensive statistics and organization capabilities.
"""

import logging
import subprocess
from datetime import datetime, UTC
from typing import Any, Dict, List

from fastmcp import Context
from pydantic import Field
from typing_extensions import Annotated

from ..initialization import get_km_client
from ..utils import parse_group_applescript_records

logger = logging.getLogger(__name__)


async def km_list_macro_groups(
    include_macro_count: Annotated[bool, Field(
        default=True,
        description="Include count of macros in each group"
    )] = True,
    include_enabled_count: Annotated[bool, Field(
        default=True,
        description="Include count of enabled macros in each group"
    )] = True,
    sort_by: Annotated[str, Field(
        default="name",
        description="Sort groups by: name, macro_count, enabled_count"
    )] = "name",
    ctx: Context = None
) -> Dict[str, Any]:
    """
    List all macro groups from Keyboard Maestro with optional statistics.
    
    Provides comprehensive group information including macro counts,
    group status, and organizational structure.
    """
    if ctx:
        await ctx.info("Retrieving macro groups from Keyboard Maestro")
    
    try:
        # Get KM client
        km_client = get_km_client()
        
        if ctx:
            await ctx.report_progress(20, 100, "Connecting to Keyboard Maestro")
        
        # Use AppleScript to get detailed group information
        script = '''
        tell application "Keyboard Maestro"
            set groupList to every macro group
            set groupData to {}
            
            repeat with currentGroup in groupList
                set groupName to name of currentGroup
                set groupMacros to every macro of currentGroup
                set enabledMacros to 0
                set totalMacros to count of groupMacros
                
                repeat with currentMacro in groupMacros
                    if enabled of currentMacro then
                        set enabledMacros to enabledMacros + 1
                    end if
                end repeat
                
                set groupRecord to {¬
                    groupName:groupName, ¬
                    totalMacros:totalMacros, ¬
                    enabledMacros:enabledMacros, ¬
                    enabled:(enabled of currentGroup)¬
                }
                set groupData to groupData & {groupRecord}
            end repeat
            
            return groupData
        end tell
        '''
        
        if ctx:
            await ctx.report_progress(40, 100, "Executing AppleScript query")
        
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            if ctx:
                await ctx.error(f"AppleScript failed: {result.stderr}")
            return {
                "success": False,
                "error": {
                    "code": "KM_CONNECTION_FAILED",
                    "message": "Cannot retrieve macro groups from Keyboard Maestro",
                    "details": result.stderr,
                    "recovery_suggestion": "Ensure Keyboard Maestro is running and accessible"
                }
            }
        
        if ctx:
            await ctx.report_progress(60, 100, "Parsing group data")
        
        # Parse AppleScript records for groups (different from macro parsing)
        groups_data = parse_group_applescript_records(result.stdout)
        
        # Transform to standard format
        groups = []
        for group_data in groups_data:
            group_info = {
                "name": group_data.get("groupName", ""),
                "enabled": group_data.get("enabled", True)
            }
            
            if include_macro_count:
                group_info["macro_count"] = group_data.get("totalMacros", 0)
            
            if include_enabled_count:
                group_info["enabled_macro_count"] = group_data.get("enabledMacros", 0)
            
            groups.append(group_info)
        
        if ctx:
            await ctx.report_progress(80, 100, "Applying sorting and formatting")
        
        # Sort groups
        if sort_by == "name":
            groups.sort(key=lambda g: g["name"].lower())
        elif sort_by == "macro_count" and include_macro_count:
            groups.sort(key=lambda g: g.get("macro_count", 0), reverse=True)
        elif sort_by == "enabled_count" and include_enabled_count:
            groups.sort(key=lambda g: g.get("enabled_macro_count", 0), reverse=True)
        
        if ctx:
            await ctx.report_progress(100, 100, f"Retrieved {len(groups)} macro groups")
            await ctx.info(f"Found {len(groups)} macro groups")
        
        # Calculate summary statistics
        total_macros = sum(g.get("macro_count", 0) for g in groups) if include_macro_count else None
        total_enabled = sum(g.get("enabled_macro_count", 0) for g in groups) if include_enabled_count else None
        enabled_groups = sum(1 for g in groups if g.get("enabled", True))
        
        return {
            "success": True,
            "data": {
                "groups": groups,
                "summary": {
                    "total_groups": len(groups),
                    "enabled_groups": enabled_groups,
                    "disabled_groups": len(groups) - enabled_groups,
                    "total_macros": total_macros,
                    "total_enabled_macros": total_enabled
                }
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "sort_by": sort_by,
                "include_counts": {
                    "macro_count": include_macro_count,
                    "enabled_count": include_enabled_count
                }
            }
        }
        
    except subprocess.TimeoutExpired:
        if ctx:
            await ctx.error("Timeout retrieving macro groups")
        return {
            "success": False,
            "error": {
                "code": "TIMEOUT_ERROR",
                "message": "Timeout retrieving macro groups from Keyboard Maestro",
                "details": "AppleScript execution exceeded 30 seconds",
                "recovery_suggestion": "Check Keyboard Maestro responsiveness"
            }
        }
    except Exception as e:
        logger.exception("Error in km_list_macro_groups")
        if ctx:
            await ctx.error(f"Group listing error: {e}")
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": "Failed to retrieve macro groups",
                "details": str(e),
                "recovery_suggestion": "Check Keyboard Maestro connection and permissions"
            }
        }