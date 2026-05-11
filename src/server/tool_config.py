"""Tool Configuration and Metadata Schema for Keyboard Maestro MCP.

This module defines the configuration schema for tools, including categorization,
security policies, and validation rules.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for organization and feature grouping."""

    CORE = "core"
    ADVANCED = "advanced"
    CLIPBOARD = "clipboard"
    WINDOW_MANAGEMENT = "window_management"
    NOTIFICATIONS = "notifications"
    TOKEN_PROCESSING = "token_processing"  # noqa: S105 # Enum value, not password
    CONDITIONAL_LOGIC = "conditional_logic"
    CONTROL_FLOW = "control_flow"
    TRIGGERS = "triggers"
    SECURITY_AUDIT = "security_audit"
    GENERAL = "general"


class SecurityLevel(Enum):
    """Security levels for tool execution."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


@dataclass
class ToolSecurityPolicy:
    """Security policy configuration for a tool."""

    level: SecurityLevel = SecurityLevel.STANDARD
    requires_authentication: bool = False
    allowed_contexts: set[str] = field(default_factory=set)
    rate_limit_per_minute: int | None = None
    audit_level: str = "standard"
    input_validation: bool = True
    output_sanitization: bool = True


@dataclass
class ToolValidationRules:
    """Validation rules for tool parameters and execution."""

    required_parameters: set[str] = field(default_factory=set)
    parameter_constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    timeout_seconds: int = 30
    max_retries: int = 3
    validate_return_type: bool = True


@dataclass
class ToolConfiguration:
    """Complete configuration for a tool."""

    name: str
    category: ToolCategory
    description: str
    module_path: str
    enabled: bool = True
    priority: int = 0
    security_policy: ToolSecurityPolicy = field(default_factory=ToolSecurityPolicy)
    validation_rules: ToolValidationRules = field(default_factory=ToolValidationRules)
    dependencies: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    version: str = "1.0.0"
    experimental: bool = False


class ToolConfigurationManager:
    """Manages tool configurations and policies."""

    def __init__(self):
        self.configurations: dict[str, ToolConfiguration] = {}
        self._load_default_configurations()

    def _load_default_configurations(self) -> None:
        """Seed defaults for tools that need non-discovery-time policies.

        Authoritative tool registration lives in ``ToolDiscovery`` (auto-glob
        of ``src.server.tools``). This map only carries security/audit
        overrides for tools that warrant stricter-than-default policy.
        """
        self.configurations["km_execute_macro"] = ToolConfiguration(
            name="km_execute_macro",
            category=ToolCategory.CORE,
            description="Run a Keyboard Maestro macro",
            module_path="src.server.tools.core_tools",
            priority=10,
            security_policy=ToolSecurityPolicy(
                level=SecurityLevel.STANDARD,
                audit_level="detailed",
            ),
        )
        self.configurations["km_clipboard_manager"] = ToolConfiguration(
            name="km_clipboard_manager",
            category=ToolCategory.CLIPBOARD,
            description="Clipboard inspection via Keyboard Maestro",
            module_path="src.server.tools.token_tools",
            priority=6,
            security_policy=ToolSecurityPolicy(
                level=SecurityLevel.STANDARD,
                input_validation=True,
                output_sanitization=True,
            ),
        )

    def get_configuration(self, tool_name: str) -> ToolConfiguration | None:
        """Get configuration for a specific tool."""
        return self.configurations.get(tool_name)

    def get_tools_by_category(self, category: ToolCategory) -> list[ToolConfiguration]:
        """Get all tools in a specific category."""
        return [
            config
            for config in self.configurations.values()
            if config.category == category
        ]

    def get_enabled_tools(self) -> list[str]:
        """Get list of enabled tool names."""
        return [name for name, config in self.configurations.items() if config.enabled]

    def set_tool_enabled(self, tool_name: str, enabled: bool) -> bool:
        """Enable or disable a tool."""
        if tool_name in self.configurations:
            self.configurations[tool_name].enabled = enabled
            return True
        return False

    def get_category_summary(self) -> dict[str, int]:
        """Get summary of tools by category."""
        summary = {}
        for config in self.configurations.values():
            category = config.category.value
            summary[category] = summary.get(category, 0) + 1
        return summary

    def validate_configuration(self, tool_name: str) -> bool:
        """Validate a tool configuration."""
        config = self.configurations.get(tool_name)
        if not config:
            return False

        try:
            # Basic validation
            if not config.name or not config.module_path:
                return False

            # Security policy validation
            if (
                config.security_policy.rate_limit_per_minute
                and config.security_policy.rate_limit_per_minute < 1
            ):
                return False

            # Validation rules check
            return not config.validation_rules.timeout_seconds < 1

        except Exception as e:
            logger.error(f"Configuration validation error for {tool_name}: {e}")
            return False


# Global configuration manager instance
_config_manager = None


def get_tool_config_manager() -> ToolConfigurationManager:
    """Get the global tool configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ToolConfigurationManager()
    return _config_manager
