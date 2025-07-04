"""
MCP Resources and prompts for the Keyboard Maestro server.

Contains server status resource, help documentation, and prompt definitions.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastmcp.prompts import Message
from pydantic import Field
from typing_extensions import Annotated

from .initialization import get_km_client

logger = logging.getLogger(__name__)


def get_server_status() -> Dict[str, Any]:
    """Get current server status and configuration."""
    # Test KM connection status
    km_client = get_km_client()
    try:
        # Quick sync test - we'll make this async in the future
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connection_test = loop.run_until_complete(
            km_client.list_macros_async(enabled_only=True)
        )
        loop.close()
        
        if connection_test.is_right():
            km_status = "connected"
            macro_count = len(connection_test.get_right())
        else:
            km_status = "disconnected"
            macro_count = 0
    except Exception:
        km_status = "error"
        macro_count = 0
    
    return {
        "server_name": "KeyboardMaestroMCP",
        "version": "1.0.0",
        "status": "running",
        "engine_status": "initialized",
        "km_connection": km_status,
        "km_macro_count": macro_count,
        "tools_available": 10,  # Current tool count (added km_list_macro_groups)
        "tools_planned": 51,   # Total planned tools
        "integration_methods": ["applescript", "web_api", "url_scheme"],
        "features": {
            "macro_execution": True,
            "macro_listing": True,      # Now implemented with real data
            "variable_management": True,
            "real_time_sync": False,    # TASK_7 implementation
            "enhanced_metadata": True,  # TASK_6 completed
            "trigger_management": False,
            "plugin_system": False,
            "ocr_integration": False
        },
        "task_progress": {
            "task_1_core_engine": "completed",
            "task_2_km_integration": "completed", 
            "task_3_command_library": "completed",
            "task_4_testing_framework": "completed",
            "task_5_real_api_integration": "in_progress",
            "task_6_enhanced_metadata": "completed",
            "task_7_realtime_sync": "planned"
        }
    }


def get_tool_help() -> str:
    """Get comprehensive help for all available tools."""
    return """
# Keyboard Maestro MCP Tools Help

## Available Tools

### km_execute_macro
Execute a Keyboard Maestro macro with comprehensive error handling.
- **identifier**: Macro name or UUID
- **trigger_value**: Optional parameter to pass to macro
- **method**: Execution method (applescript, url, web, remote)
- **timeout**: Maximum execution time (1-300 seconds)

### km_list_macros ✨ REAL DATA
List and filter your actual Keyboard Maestro macros (no longer mock data).
- **group_filter**: Filter by macro group name
- **enabled_only**: Only show enabled macros (default: true)
- **sort_by**: Sort field (name, last_used, created_date, group)
- **limit**: Maximum results (1-100, default: 20)

**Data Source**: Live data from Keyboard Maestro via AppleScript and Web API
**Features**: Real macro names, groups, trigger counts, and status

### km_search_macros_advanced ✨ NEW - TASK_6
Advanced macro search with comprehensive filtering and metadata analysis.
- **query**: Search text for macro names, groups, or content
- **scope**: Search scope (name_only, name_and_group, full_content, metadata_only)
- **action_categories**: Filter by action types (text_manipulation, application_control, etc.)
- **complexity_levels**: Filter by complexity (simple, moderate, complex, advanced)
- **min_usage_count**: Minimum execution count filter
- **sort_by**: Advanced sorting (name, last_used, usage_frequency, complexity, success_rate)

**Features**: Enhanced metadata, usage analytics, optimization suggestions, similarity detection

### km_analyze_macro_metadata ✨ NEW - TASK_6
Deep analysis of individual macro metadata and patterns.
- **macro_id**: Macro ID or name to analyze
- **include_relationships**: Include similarity and relationship analysis

**Features**: Complexity analysis, usage patterns, optimization suggestions, similar macro detection

### km_list_macro_groups ✨ NEW
List all macro groups from Keyboard Maestro with comprehensive statistics.
- **include_macro_count**: Include count of macros in each group (default: true)
- **include_enabled_count**: Include count of enabled macros in each group (default: true)
- **sort_by**: Sort groups by name, macro_count, or enabled_count (default: name)

**Features**: Group organization overview, macro distribution analysis, real-time group statistics

### km_variable_manager
Manage Keyboard Maestro variables across all scopes.
- **operation**: get, set, delete, or list
- **name**: Variable name (required for get/set/delete)
- **value**: Variable value (required for set)
- **scope**: global, local, instance, or password
- **instance_id**: For local/instance variables

## Resources

### km://server/status
Get current server status, KM connection health, and feature availability.

### km://help/tools  
This help documentation.

## Integration Methods
- **AppleScript**: Primary method for reliable macro access
- **Web API**: HTTP fallback when AppleScript unavailable
- **URL Scheme**: Direct macro triggering (planned)

## Error Codes
- KM_CONNECTION_FAILED: Cannot connect to Keyboard Maestro
- INVALID_PARAMETER: Parameter validation failed
- PERMISSION_DENIED: Insufficient permissions
- TIMEOUT_ERROR: Operation timed out
- EXECUTION_ERROR: Macro execution failed
- SYSTEM_ERROR: Unexpected system error

## Setup Requirements
- Keyboard Maestro running and accessible
- Accessibility permissions for AppleScript (if needed)
- Web server enabled on port 4490 (for fallback)

## Recent Updates
**TASK_5**: Real API Integration
✅ Real macro listing replaces mock data
✅ Multiple API integration methods
✅ Enhanced error handling and recovery
✅ Live connection status monitoring

**TASK_6**: Enhanced Macro Discovery & Metadata ✨ NEW
✅ Advanced search with comprehensive filtering
✅ Rich metadata extraction and analysis
✅ Usage analytics and optimization suggestions
✅ Smart filtering and pattern recognition
✅ Hierarchical organization and complexity scoring
✅ Similar macro detection and relationship mapping

For more tools and features, see the development roadmap.
    """.strip()


def create_macro_prompt(
    task_description: Annotated[str, Field(
        description="Description of the automation task to create a macro for"
    )],
    app_context: Annotated[Optional[str], Field(
        default=None,
        description="Specific application or context for the automation"
    )] = None
) -> List[Message]:
    """Generate a structured prompt for creating Keyboard Maestro macros."""
    
    system_prompt = """You are an expert Keyboard Maestro macro developer. Help create efficient, reliable macros for macOS automation tasks."""
    
    user_prompt = f"""
Task: {task_description}

{f"Context: {app_context}" if app_context else ""}

Please provide a detailed macro design including:

1. **Macro Name and Purpose**
2. **Triggers** (hotkeys, application events, etc.)
3. **Actions Sequence** (step-by-step automation)
4. **Variables** (if needed for data storage)
5. **Error Handling** (conditions and fallbacks)
6. **Testing Strategy** (how to verify it works)

Focus on reliability, user experience, and maintainability.
    """
    
    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt.strip())
    ]