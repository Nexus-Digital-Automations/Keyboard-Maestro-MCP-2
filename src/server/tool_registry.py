"""Tool Discovery and Registration System for Keyboard Maestro MCP.

This module provides automated tool discovery, metadata extraction, and dynamic
registration capabilities to eliminate boilerplate code while maintaining type safety.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_type_hints

from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Represents a tool parameter with type and validation information."""

    name: str
    annotation: type[Any]
    default: Any = None
    description: str = ""
    field_info: FieldInfo | None = None
    is_optional: bool = False


@dataclass
class ToolMetadata:
    """Comprehensive metadata for a discovered tool."""

    name: str
    function: Callable[..., Any]
    module_name: str
    docstring: str
    parameters: list[ToolParameter] = field(default_factory=list)
    return_type: type[Any] = dict[str, Any]
    is_async: bool = True
    category: str = "general"


class ToolDiscovery:
    """Discovers and extracts metadata from MCP tools in the tools directory."""

    def __init__(self, tools_package: str = "src.server.tools"):
        self.tools_package = tools_package
        self.discovered_tools: dict[str, ToolMetadata] = {}

    def discover_all_tools(self) -> dict[str, ToolMetadata]:
        """Discover all tools in the tools package."""
        try:
            # Import the tools package
            tools_module = importlib.import_module(self.tools_package)
            if tools_module.__file__ is None:
                logger.error(f"Tools package {self.tools_package} has no __file__")
                return {}
            tools_path = Path(tools_module.__file__).parent

            logger.info(f"Discovering tools in: {tools_path}")

            # Iterate through all modules in the tools directory
            for _finder, module_name, ispkg in pkgutil.iter_modules([str(tools_path)]):
                if ispkg or module_name.startswith("_"):
                    continue

                full_module_name = f"{self.tools_package}.{module_name}"
                try:
                    self._discover_tools_in_module(full_module_name)
                except Exception as e:
                    logger.warning(
                        f"Failed to discover tools in {full_module_name}: {e}",
                    )

            logger.info(f"Discovered {len(self.discovered_tools)} tools")
            return self.discovered_tools

        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")
            return {}

    def _discover_tools_in_module(self, module_name: str) -> None:
        """Discover tools in a specific module."""
        try:
            module = importlib.import_module(module_name)

            # Look for functions that match tool naming pattern
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("km_") and not name.startswith("_"):
                    try:
                        metadata = self._extract_tool_metadata(name, obj, module_name)
                        if metadata:
                            self.discovered_tools[name] = metadata
                            logger.debug(f"Discovered tool: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to extract metadata for {name}: {e}")

        except Exception as e:
            logger.error(f"Failed to import module {module_name}: {e}")

    def _extract_tool_metadata(
        self,
        name: str,
        func: Callable[..., Any],
        module_name: str,
    ) -> ToolMetadata | None:
        """Extract comprehensive metadata from a tool function."""
        try:
            # Get function signature and type hints
            signature = inspect.signature(func)
            type_hints = get_type_hints(func, include_extras=True)

            # Extract parameters
            parameters = []
            for param_name, param in signature.parameters.items():
                if param_name == "ctx":  # Skip context parameter
                    continue

                param_metadata = self._extract_parameter_metadata(
                    param_name,
                    param,
                    type_hints.get(param_name),
                )
                if param_metadata:
                    parameters.append(param_metadata)

            # Determine category from module name
            category = self._determine_category(module_name)

            # Create metadata object
            metadata = ToolMetadata(
                name=name,
                function=func,
                module_name=module_name,
                docstring=inspect.getdoc(func) or "",
                parameters=parameters,
                return_type=type_hints.get("return", dict[str, Any]),
                is_async=inspect.iscoroutinefunction(func),
                category=category,
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract metadata for {name}: {e}")
            return None

    def _extract_parameter_metadata(
        self,
        name: str,
        param: inspect.Parameter,
        type_hint: Any,
    ) -> ToolParameter | None:
        """Extract parameter metadata including Pydantic Field information."""
        try:
            # Handle Annotated types (common in FastMCP tools)
            annotation = type_hint
            field_info = None
            description = ""

            if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Annotated:
                args = get_args(type_hint)
                if args:
                    annotation = args[0]
                    # Look for Field information in annotations
                    for arg in args[1:]:
                        try:
                            if isinstance(arg, FieldInfo):
                                field_info = arg
                                description = getattr(arg, "description", "") or ""
                                break
                        except Exception as e:
                            # S112 fix: Log exception instead of silent continue
                            logger.debug(
                                f"Failed to check Field instance for {param.name}: {e}",
                            )
                            continue

            # Determine if parameter is optional
            is_optional = False
            try:
                is_optional = param.default is not inspect.Parameter.empty or (
                    hasattr(annotation, "__origin__")
                    and annotation.__origin__ is type(type(None) | int)
                    and type(None) in get_args(annotation)
                )
            except Exception:
                # If optional detection fails, assume required
                is_optional = param.default is not inspect.Parameter.empty

            return ToolParameter(
                name=name,
                annotation=type_hint,
                default=param.default
                if param.default is not inspect.Parameter.empty
                else None,
                description=description,
                field_info=field_info,
                is_optional=is_optional,
            )

        except Exception as e:
            logger.warning(f"Failed to extract parameter metadata for {name}: {e}")
            return None

    def _determine_category(self, module_name: str) -> str:
        """Determine tool category from module name."""
        if "core" in module_name:
            return "core"
        if "advanced" in module_name:
            return "advanced"
        if "sync" in module_name:
            return "synchronization"
        if "clipboard" in module_name:
            return "clipboard"
        if "file" in module_name:
            return "file_operations"
        if "window" in module_name:
            return "window_management"
        if "notification" in module_name:
            return "notifications"
        if "calculator" in module_name:
            return "calculations"
        if "token" in module_name:
            return "token_processing"
        if "condition" in module_name:
            return "conditional_logic"
        if "control_flow" in module_name:
            return "control_flow"
        if "trigger" in module_name:
            return "triggers"
        if "audit" in module_name:
            return "security_audit"
        if "analytics" in module_name:
            return "analytics"
        if "workflow" in module_name:
            return "workflow_intelligence"
        if "iot" in module_name:
            return "iot_integration"
        if "voice" in module_name:
            return "voice_control"
        if "quantum" in module_name:
            return "quantum_ready"
        if "ai" in module_name or "processing" in module_name:
            return "ai_intelligence"
        if "plugin" in module_name:
            return "plugin_ecosystem"
        return "general"


class ToolRegistry:
    """Central registry for managing discovered tools."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolMetadata] = {}
        self.categories: dict[str, list[str]] = {}
        self.discovery = ToolDiscovery()

    def discover_and_register_tools(self) -> None:
        """Discover all tools and register them in the registry."""
        self.tools = self.discovery.discover_all_tools()
        self._organize_by_category()

    def _organize_by_category(self) -> None:
        """Organize tools by category for easier management."""
        self.categories.clear()
        for tool_name, metadata in self.tools.items():
            category = metadata.category
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(tool_name)

    def get_tool(self, name: str) -> ToolMetadata | None:
        """Get tool metadata by name."""
        return self.tools.get(name)

    def get_tools_by_category(self, category: str) -> list[ToolMetadata]:
        """Get all tools in a specific category."""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]

    def get_all_tools(self) -> dict[str, ToolMetadata]:
        """Get all registered tools."""
        return self.tools.copy()

    def get_tool_count(self) -> int:
        """Get total number of registered tools."""
        return len(self.tools)

    def get_category_summary(self) -> dict[str, int]:
        """Get summary of tools by category."""
        return {category: len(tools) for category, tools in self.categories.items()}


# Global registry instance
_tool_registry = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        _tool_registry.discover_and_register_tools()
    return _tool_registry


def discover_tools() -> dict[str, ToolMetadata]:
    """Convenience function to discover all tools."""
    registry = get_tool_registry()
    return registry.get_all_tools()
