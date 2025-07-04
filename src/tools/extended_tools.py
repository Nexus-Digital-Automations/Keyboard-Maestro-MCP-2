"""
Extended tools registration for additional Keyboard Maestro functionality.

Registers search, property management, dictionary, engine control,
interface automation, and notification tools.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_extended_tools(mcp: 'FastMCP') -> None:
    """
    Register extended MCP tools for comprehensive Keyboard Maestro automation.
    
    Includes:
    - km_search_actions: Search for actions within macros
    - km_manage_macro_properties: Get/update macro properties
    - km_dictionary_manager: Manage KM dictionaries
    - km_engine_control: Control KM engine operations
    - km_interface_automation: Automate mouse and keyboard
    - km_notifications: Display system notifications
    
    Args:
        mcp: FastMCP server instance
    """
    try:
        # Import tools from server.tools package
        from ..server.tools import (
            km_search_actions,
            km_manage_macro_properties,
            km_dictionary_manager,
            km_engine_control,
            km_interface_automation,
            km_notifications
        )
        
        # Register search tools
        mcp.tool()(km_search_actions)
        logger.info("Registered km_search_actions tool")
        
        # Register property tools
        mcp.tool()(km_manage_macro_properties)
        logger.info("Registered km_manage_macro_properties tool")
        
        # Register dictionary tools
        mcp.tool()(km_dictionary_manager)
        logger.info("Registered km_dictionary_manager tool")
        
        # Register engine tools
        mcp.tool()(km_engine_control)
        logger.info("Registered km_engine_control tool")
        
        # Register interface tools
        mcp.tool()(km_interface_automation)
        logger.info("Registered km_interface_automation tool")
        
        # Register notification tools
        mcp.tool()(km_notifications)
        logger.info("Registered km_notifications tool")
        
        logger.info("Successfully registered 6 extended tools")
        
    except Exception as e:
        logger.error(f"Error registering extended tools: {e}")
        raise