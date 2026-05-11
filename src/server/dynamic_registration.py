"""Dynamic Tool Registration Engine for FastMCP.

This module handles the automatic registration of discovered tools with the FastMCP
server, eliminating the need for manual tool registration boilerplate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .tool_registry import ToolMetadata, get_tool_registry

if TYPE_CHECKING:
    from fastmcp import Context, FastMCP


logger = logging.getLogger(__name__)


class DynamicToolRegistrar:
    """Handles dynamic registration of tools with FastMCP."""

    def __init__(self, mcp_server: FastMCP):
        self.mcp_server = mcp_server
        self.registry = get_tool_registry()
        self.registered_tools: dict[str, ToolMetadata] = {}

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
        """Return the underlying tool function so FastMCP can introspect its real signature.

        A previous version wrapped every tool in a closure with ``(*args, **kwargs)``
        plumbing, which made FastMCP refuse to register any of them
        ("Functions with *args are not supported as tools").
        """
        try:
            module = __import__(metadata.module_name, fromlist=[metadata.name])
            return getattr(module, metadata.name)
        except (ImportError, AttributeError) as import_error:
            error_message = str(import_error)
            logger.error(
                "Failed to import tool %s from %s: %s",
                metadata.name,
                metadata.module_name,
                error_message,
            )

            async def missing_tool(_ctx: Context | Any = None) -> dict[str, Any]:
                return {
                    "success": False,
                    "error": f"Tool {metadata.name} not available: {error_message}",
                    "tool": metadata.name,
                }

            missing_tool.__name__ = metadata.name
            missing_tool.__doc__ = f"Error: {metadata.name} not available"
            return missing_tool

    def _log_registration_summary(self) -> None:
        """Log a summary of tool registration by category."""
        category_summary = self.registry.get_category_summary()

        logger.info("Tool Registration Summary:")
        for category, count in category_summary.items():
            logger.info(f"  {category}: {count} tools")

        logger.info(f"Total: {sum(category_summary.values())} tools registered")

    def get_registered_tools(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self.registered_tools.keys())

    def get_tool_metadata(self, tool_name: str) -> ToolMetadata | None:
        """Get metadata for a specific registered tool."""
        return self.registered_tools.get(tool_name)

    def get_tools_by_category(self, category: str) -> list[str]:
        """Get tools in a specific category."""
        return [
            name
            for name, metadata in self.registered_tools.items()
            if metadata.category == category
        ]


def register_tools_dynamically(mcp_server: FastMCP) -> DynamicToolRegistrar:
    """Register all discovered tools with the FastMCP server.

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


def validate_tool_signature(metadata: ToolMetadata) -> bool:
    """Validate that a tool has a proper signature for FastMCP registration.

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
                logger.warning(
                    f"Parameter {param.name} in {metadata.name} lacks type annotation",
                )
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
        "Parameters:",
    ]

    for param in metadata.parameters:
        param_desc = f"  {param.name}"
        if param.is_optional:
            param_desc += " (optional)"
        param_desc += f": {param.description or 'No description'}"
        help_lines.append(param_desc)

    return "\n".join(help_lines)
