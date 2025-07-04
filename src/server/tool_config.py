"""
Tool Configuration and Metadata Schema for Keyboard Maestro MCP.

This module defines the configuration schema for tools, including categorization,
security policies, and validation rules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for organization and feature grouping."""
    CORE = "core"
    ADVANCED = "advanced"
    SYNCHRONIZATION = "synchronization"
    CLIPBOARD = "clipboard"
    FILE_OPERATIONS = "file_operations"
    WINDOW_MANAGEMENT = "window_management"
    NOTIFICATIONS = "notifications"
    CALCULATIONS = "calculations"
    TOKEN_PROCESSING = "token_processing"
    CONDITIONAL_LOGIC = "conditional_logic"
    CONTROL_FLOW = "control_flow"
    TRIGGERS = "triggers"
    SECURITY_AUDIT = "security_audit"
    ANALYTICS = "analytics"
    WORKFLOW_INTELLIGENCE = "workflow_intelligence"
    IOT_INTEGRATION = "iot_integration"
    VOICE_CONTROL = "voice_control"
    BIOMETRIC_SECURITY = "biometric_security"
    QUANTUM_READY = "quantum_ready"
    AI_INTELLIGENCE = "ai_intelligence"
    PLUGIN_ECOSYSTEM = "plugin_ecosystem"
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
    allowed_contexts: Set[str] = field(default_factory=set)
    rate_limit_per_minute: Optional[int] = None
    audit_level: str = "standard"
    input_validation: bool = True
    output_sanitization: bool = True


@dataclass
class ToolValidationRules:
    """Validation rules for tool parameters and execution."""
    required_parameters: Set[str] = field(default_factory=set)
    parameter_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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
    dependencies: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    experimental: bool = False


class ToolConfigurationManager:
    """Manages tool configurations and policies."""
    
    def __init__(self):
        self.configurations: Dict[str, ToolConfiguration] = {}
        self._load_default_configurations()
    
    def _load_default_configurations(self) -> None:
        """Load default configurations for all tool categories."""
        
        # Core Tools
        core_tools = [
            "km_execute_macro", "km_list_macros", "km_variable_manager"
        ]
        for tool in core_tools:
            self.configurations[tool] = ToolConfiguration(
                name=tool,
                category=ToolCategory.CORE,
                description=f"Core macro operation: {tool}",
                module_path=f"src.server.tools.core_tools",
                priority=10,
                security_policy=ToolSecurityPolicy(
                    level=SecurityLevel.STANDARD,
                    audit_level="detailed"
                )
            )
        
        # Advanced Tools
        advanced_tools = [
            "km_search_macros_advanced", "km_analyze_macro_metadata"
        ]
        for tool in advanced_tools:
            self.configurations[tool] = ToolConfiguration(
                name=tool,
                category=ToolCategory.ADVANCED,
                description=f"Advanced macro operation: {tool}",
                module_path="src.server.tools.advanced_tools",
                priority=8
            )
        
        # Synchronization Tools
        sync_tools = [
            "km_start_realtime_sync", "km_stop_realtime_sync", 
            "km_sync_status", "km_force_sync"
        ]
        for tool in sync_tools:
            self.configurations[tool] = ToolConfiguration(
                name=tool,
                category=ToolCategory.SYNCHRONIZATION,
                description=f"Real-time synchronization: {tool}",
                module_path="src.server.tools.sync_tools",
                priority=7
            )
        
        # File Operations
        self.configurations["km_file_operations"] = ToolConfiguration(
            name="km_file_operations",
            category=ToolCategory.FILE_OPERATIONS,
            description="Secure file system operations",
            module_path="src.server.tools.file_operation_tools",
            priority=6,
            security_policy=ToolSecurityPolicy(
                level=SecurityLevel.STRICT,
                audit_level="comprehensive"
            )
        )
        
        # Window Management
        window_tools = ["km_window_manager", "km_window_manager_advanced"]
        for tool in window_tools:
            self.configurations[tool] = ToolConfiguration(
                name=tool,
                category=ToolCategory.WINDOW_MANAGEMENT,
                description=f"Window management: {tool}",
                module_path="src.server.tools.window_tools" if tool == "km_window_manager" else "src.server.tools.advanced_window_tools",
                priority=5
            )
        
        # Clipboard Operations
        self.configurations["km_clipboard_manager"] = ToolConfiguration(
            name="km_clipboard_manager",
            category=ToolCategory.CLIPBOARD,
            description="Comprehensive clipboard management",
            module_path="src.server.tools.clipboard_tools",
            priority=6,
            security_policy=ToolSecurityPolicy(
                level=SecurityLevel.STANDARD,
                input_validation=True,
                output_sanitization=True
            )
        )
        
        # Calculations
        calculation_tools = ["km_calculator", "km_token_processor"]
        for tool in calculation_tools:
            category = ToolCategory.CALCULATIONS if "calculator" in tool else ToolCategory.TOKEN_PROCESSING
            self.configurations[tool] = ToolConfiguration(
                name=tool,
                category=category,
                description=f"Mathematical and token operations: {tool}",
                module_path=f"src.server.tools.{tool.replace('km_', '').replace('_processor', '')}_tools",
                priority=5,
                security_policy=ToolSecurityPolicy(
                    level=SecurityLevel.STANDARD,
                    input_validation=True
                )
            )
        
        # Notifications
        notification_tools = [
            "km_notifications", "km_notification_status", "km_dismiss_notifications"
        ]
        for tool in notification_tools:
            self.configurations[tool] = ToolConfiguration(
                name=tool,
                category=ToolCategory.NOTIFICATIONS,
                description=f"User notification system: {tool}",
                module_path="src.server.tools.notification_tools",
                priority=4
            )
        
        # Control Flow and Conditions
        control_tools = ["km_add_condition", "km_control_flow", "km_create_trigger_advanced"]
        for tool in control_tools:
            if "condition" in tool:
                category = ToolCategory.CONDITIONAL_LOGIC
                module = "condition_tools"
            elif "control_flow" in tool:
                category = ToolCategory.CONTROL_FLOW
                module = "control_flow_tools"
            else:
                category = ToolCategory.TRIGGERS
                module = "advanced_trigger_tools"
                
            self.configurations[tool] = ToolConfiguration(
                name=tool,
                category=category,
                description=f"Flow control: {tool}",
                module_path=f"src.server.tools.{module}",
                priority=7
            )
        
        # Enterprise and AI Tools
        enterprise_tools = [
            ("km_audit_system", ToolCategory.SECURITY_AUDIT, "audit_system_tools"),
            ("km_analytics_engine", ToolCategory.ANALYTICS, "analytics_engine_tools"),
            ("km_analyze_workflow_intelligence", ToolCategory.WORKFLOW_INTELLIGENCE, "workflow_intelligence_tools"),
            ("km_create_workflow_from_description", ToolCategory.WORKFLOW_INTELLIGENCE, "workflow_intelligence_tools"),
        ]
        
        for tool, category, module in enterprise_tools:
            self.configurations[tool] = ToolConfiguration(
                name=tool,
                category=category,
                description=f"Enterprise feature: {tool}",
                module_path=f"src.server.tools.{module}",
                priority=9,
                security_policy=ToolSecurityPolicy(
                    level=SecurityLevel.ENTERPRISE,
                    audit_level="comprehensive"
                )
            )
        
        # IoT and Advanced Integration
        iot_tools = [
            "km_control_iot_devices", "km_monitor_sensors", 
            "km_manage_smart_home", "km_coordinate_iot_workflows"
        ]
        for tool in iot_tools:
            self.configurations[tool] = ToolConfiguration(
                name=tool,
                category=ToolCategory.IOT_INTEGRATION,
                description=f"IoT integration: {tool}",
                module_path="src.server.tools.iot_integration_tools",
                priority=6,
                experimental=True
            )
        
        # Plugin Ecosystem
        self.configurations["km_plugin_ecosystem"] = ToolConfiguration(
            name="km_plugin_ecosystem",
            category=ToolCategory.PLUGIN_ECOSYSTEM,
            description="Plugin management and custom action creation",
            module_path="src.server.tools.plugin_ecosystem_tools",
            priority=5,
            security_policy=ToolSecurityPolicy(
                level=SecurityLevel.STRICT,
                audit_level="comprehensive"
            )
        )
    
    def get_configuration(self, tool_name: str) -> Optional[ToolConfiguration]:
        """Get configuration for a specific tool."""
        return self.configurations.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolConfiguration]:
        """Get all tools in a specific category."""
        return [
            config for config in self.configurations.values()
            if config.category == category
        ]
    
    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tool names."""
        return [
            name for name, config in self.configurations.items()
            if config.enabled
        ]
    
    def set_tool_enabled(self, tool_name: str, enabled: bool) -> bool:
        """Enable or disable a tool."""
        if tool_name in self.configurations:
            self.configurations[tool_name].enabled = enabled
            return True
        return False
    
    def get_category_summary(self) -> Dict[str, int]:
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
            if config.security_policy.rate_limit_per_minute and config.security_policy.rate_limit_per_minute < 1:
                return False
            
            # Validation rules check
            if config.validation_rules.timeout_seconds < 1:
                return False
            
            return True
            
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