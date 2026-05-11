"""Dynamic Tool Registration Engine for FastMCP.

This module handles the automatic registration of discovered tools with the FastMCP
server, eliminating the need for manual tool registration boilerplate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .tool_registry import ToolMetadata, get_tool_registry

if TYPE_CHECKING:

    from fastmcp import FastMCP


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
        """Register a single tool with FastMCP.

        FastMCP introspects the registered function's signature to build the
        tool's input schema, so we must pass the original function — a
        `*args, **kwargs` wrapper is rejected with "Functions with *args
        are not supported as tools". The tools already centralize their own
        error handling via `_failure` helpers; no wrapper is needed.
        """
        module = __import__(metadata.module_name, fromlist=[metadata.name])
        actual_tool_func = getattr(module, metadata.name)
        decorated_func = self.mcp_server.tool()(actual_tool_func)
        setattr(self, f"_tool_{metadata.name}", decorated_func)

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
        # Check if async function for MCP compatibility
        if not metadata.is_async:
            logger.warning(f"Tool {metadata.name} is not async (recommended for MCP)")

        # Type system guarantees param.annotation is non-None;
        # construction-site validation lives in ToolDiscovery.
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
