"""
Dynamic Tool Registration Engine for FastMCP.

This module handles the automatic registration of discovered tools with the FastMCP
server, eliminating the need for manual tool registration boilerplate.
"""

import logging
from typing import Any, Dict, List, Optional, get_origin, get_args
from functools import wraps

from fastmcp import FastMCP
from pydantic import Field
from typing_extensions import Annotated

from .tool_registry import ToolMetadata, ToolParameter, get_tool_registry

logger = logging.getLogger(__name__)


class DynamicToolRegistrar:
    """Handles dynamic registration of tools with FastMCP."""
    
    def __init__(self, mcp_server: FastMCP):
        self.mcp_server = mcp_server
        self.registry = get_tool_registry()
        self.registered_tools: Dict[str, ToolMetadata] = {}
    
    def register_all_tools(self) -> None:
        """Register all discovered tools with the FastMCP server."""
        tools = self.registry.get_all_tools()
        
        for tool_name, metadata in tools.items():
            try:
                self._register_tool(metadata)
                self.registered_tools[tool_name] = metadata
                logger.debug(f"Registered tool: {tool_name}")
            except Exception as e:
                logger.error(f"Failed to register tool {tool_name}: {e}")
        
        logger.info(f"Successfully registered {len(self.registered_tools)} tools")
        self._log_registration_summary()
    
    def _register_tool(self, metadata: ToolMetadata) -> None:
        """Register a single tool with FastMCP."""
        # Create the dynamic wrapper function
        wrapper_func = self._create_tool_wrapper(metadata)
        
        # Apply the @mcp.tool() decorator
        decorated_func = self.mcp_server.tool()(wrapper_func)
        
        # Store reference to prevent garbage collection
        setattr(self, f"_tool_{metadata.name}", decorated_func)
    
    def _create_tool_wrapper(self, metadata: ToolMetadata) -> callable:
        """Create a wrapper function for the tool that maintains type safety."""
        
        # Import the actual tool function at registration time
        try:
            module = __import__(metadata.module_name, fromlist=[metadata.name])
            actual_tool_func = getattr(module, metadata.name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import tool {metadata.name} from {metadata.module_name}: {e}")
            # Create a placeholder function that returns an error
            async def error_wrapper(ctx=None):
                return {
                    "success": False,
                    "error": f"Tool {metadata.name} not available: {e}",
                    "tool": metadata.name
                }
            error_wrapper.__name__ = metadata.name
            error_wrapper.__doc__ = f"Error: {metadata.name} not available"
            return error_wrapper
        
        # Build the explicit parameter list from metadata
        param_names = [param.name for param in metadata.parameters] + ['ctx']
        
        # Create a wrapper with explicit parameters (FastMCP doesn't support **kwargs)
        if metadata.is_async:
            # Create async wrapper with explicit parameters
            exec_code = f"""
async def tool_wrapper({', '.join(f'{name}=None' for name in param_names)}):
    '''Dynamically generated async tool wrapper.'''
    try:
        kwargs = {{{', '.join(f"'{name}': {name}" for name in param_names)}}}
        return await actual_tool_func(**kwargs)
    except Exception as e:
        logger.error(f"Error executing tool {metadata.name}: {{e}}")
        return {{
            "success": False,
            "error": str(e),
            "tool": "{metadata.name}"
        }}
"""
        else:
            # Create sync wrapper with explicit parameters
            exec_code = f"""
def tool_wrapper({', '.join(f'{name}=None' for name in param_names)}):
    '''Dynamically generated sync tool wrapper.'''
    try:
        kwargs = {{{', '.join(f"'{name}': {name}" for name in param_names)}}}
        return actual_tool_func(**kwargs)
    except Exception as e:
        logger.error(f"Error executing tool {metadata.name}: {{e}}")
        return {{
            "success": False,
            "error": str(e),
            "tool": "{metadata.name}"
        }}
"""
        
        # Execute the dynamically created function
        local_vars = {
            'actual_tool_func': actual_tool_func,
            'logger': logger,
            'metadata': metadata
        }
        exec(exec_code, globals(), local_vars)
        tool_wrapper = local_vars['tool_wrapper']
        
        # Preserve the original function's signature and metadata
        tool_wrapper.__name__ = metadata.name
        tool_wrapper.__doc__ = metadata.docstring
        
        # Copy annotations from the original function
        try:
            if hasattr(actual_tool_func, '__annotations__'):
                tool_wrapper.__annotations__ = actual_tool_func.__annotations__.copy()
        except Exception:
            pass
        
        return tool_wrapper
    
    def _log_registration_summary(self) -> None:
        """Log a summary of tool registration by category."""
        category_summary = self.registry.get_category_summary()
        
        logger.info("Tool Registration Summary:")
        for category, count in category_summary.items():
            logger.info(f"  {category}: {count} tools")
        
        logger.info(f"Total: {sum(category_summary.values())} tools registered")
    
    def get_registered_tools(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self.registered_tools.keys())
    
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific registered tool."""
        return self.registered_tools.get(tool_name)
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools in a specific category."""
        return [
            name for name, metadata in self.registered_tools.items()
            if metadata.category == category
        ]


def register_tools_dynamically(mcp_server: FastMCP) -> DynamicToolRegistrar:
    """
    Register all discovered tools with the FastMCP server.
    
    This function replaces the need for manual tool registration in main.py.
    
    Args:
        mcp_server: The FastMCP server instance
        
    Returns:
        DynamicToolRegistrar instance for further management
    """
    registrar = DynamicToolRegistrar(mcp_server)
    registrar.register_all_tools()
    return registrar


class ToolRegistrationError(Exception):
    """Raised when tool registration fails."""
    pass


def validate_tool_signature(metadata: ToolMetadata) -> bool:
    """
    Validate that a tool has a proper signature for FastMCP registration.
    
    Args:
        metadata: Tool metadata to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if function exists and is callable
        if not callable(metadata.function):
            logger.warning(f"Tool {metadata.name} is not callable")
            return False
        
        # Check if async function for MCP compatibility
        if not metadata.is_async:
            logger.warning(f"Tool {metadata.name} is not async (recommended for MCP)")
        
        # Validate parameters have proper type annotations
        for param in metadata.parameters:
            if param.annotation is None:
                logger.warning(f"Parameter {param.name} in {metadata.name} lacks type annotation")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating tool signature for {metadata.name}: {e}")
        return False


def get_tool_help_text(metadata: ToolMetadata) -> str:
    """Generate help text for a tool based on its metadata."""
    help_lines = [
        f"Tool: {metadata.name}",
        f"Category: {metadata.category}",
        f"Description: {metadata.docstring.split('.')[0] if metadata.docstring else 'No description available'}",
        "",
        "Parameters:"
    ]
    
    for param in metadata.parameters:
        param_desc = f"  {param.name}"
        if param.is_optional:
            param_desc += " (optional)"
        param_desc += f": {param.description or 'No description'}"
        help_lines.append(param_desc)
    
    return "\n".join(help_lines)