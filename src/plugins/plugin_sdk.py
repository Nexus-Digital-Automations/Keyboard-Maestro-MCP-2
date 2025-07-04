"""
Plugin SDK providing development framework for creating Keyboard Maestro plugins.

This module provides a comprehensive SDK for plugin developers with base classes,
utilities, and development tools to create high-quality plugins efficiently.
"""

import asyncio
import logging
import inspect
from typing import Dict, Any, List, Optional, Type, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..core.plugin_architecture import (
    PluginInterface, PluginMetadata, PluginConfiguration, CustomAction, 
    PluginHook, PluginError, CustomActionParameter, ActionId, HookId,
    PluginId, PluginPermissions, SecurityProfile, ApiVersion
)
from ..core.either import Either
from ..core.types import Duration
from ..core.errors import create_error_context
from .api_bridge import PluginAPIBridge

logger = logging.getLogger(__name__)


@dataclass
class SDKConfiguration:
    """SDK configuration for plugin development."""
    debug_mode: bool = False
    auto_reload: bool = True
    validation_strict: bool = True
    performance_monitoring: bool = True
    logging_level: str = "INFO"
    max_action_timeout: Duration = Duration.from_seconds(60.0)
    max_memory_mb: int = 256
    allowed_network_hosts: List[str] = field(default_factory=list)


class PluginLogger:
    """Enhanced logging for plugins with context and performance tracking."""
    
    def __init__(self, plugin_id: PluginId, sdk_config: SDKConfiguration):
        self.plugin_id = plugin_id
        self.sdk_config = sdk_config
        self.logger = logging.getLogger(f"plugin.{plugin_id}")
        self.performance_data: Dict[str, List[float]] = {}
    
    def debug(self, message: str, **context):
        """Debug level logging with context."""
        self._log_with_context("DEBUG", message, context)
    
    def info(self, message: str, **context):
        """Info level logging with context."""
        self._log_with_context("INFO", message, context)
    
    def warning(self, message: str, **context):
        """Warning level logging with context."""
        self._log_with_context("WARNING", message, context)
    
    def error(self, message: str, **context):
        """Error level logging with context."""
        self._log_with_context("ERROR", message, context)
    
    def _log_with_context(self, level: str, message: str, context: Dict[str, Any]):
        """Log message with plugin context."""
        formatted_message = f"[{self.plugin_id}] {message}"
        if context:
            formatted_message += f" | Context: {context}"
        
        getattr(self.logger, level.lower())(formatted_message)
    
    def log_performance(self, operation: str, duration_ms: float):
        """Log performance metrics for operations."""
        if self.sdk_config.performance_monitoring:
            if operation not in self.performance_data:
                self.performance_data[operation] = []
            
            self.performance_data[operation].append(duration_ms)
            
            # Log slow operations
            if duration_ms > 1000:  # > 1 second
                self.warning(f"Slow operation detected: {operation}", 
                           duration_ms=duration_ms)
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all tracked operations."""
        summary = {}
        for operation, durations in self.performance_data.items():
            if durations:
                summary[operation] = {
                    "count": len(durations),
                    "avg_ms": sum(durations) / len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations)
                }
        return summary


class ValidationHelper:
    """Helper class for parameter and data validation."""
    
    @staticmethod
    def validate_string(value: Any, min_length: int = 0, max_length: int = 1000, 
                       pattern: Optional[str] = None) -> Either[str, str]:
        """Validate string parameters."""
        if not isinstance(value, str):
            return Either.left(f"Expected string, got {type(value).__name__}")
        
        if len(value) < min_length:
            return Either.left(f"String too short: {len(value)} < {min_length}")
        
        if len(value) > max_length:
            return Either.left(f"String too long: {len(value)} > {max_length}")
        
        if pattern:
            import re
            if not re.match(pattern, value):
                return Either.left(f"String doesn't match pattern: {pattern}")
        
        return Either.right(value)
    
    @staticmethod
    def validate_number(value: Any, min_val: Optional[float] = None, 
                       max_val: Optional[float] = None) -> Either[str, Union[int, float]]:
        """Validate numeric parameters."""
        if not isinstance(value, (int, float)):
            return Either.left(f"Expected number, got {type(value).__name__}")
        
        if min_val is not None and value < min_val:
            return Either.left(f"Number too small: {value} < {min_val}")
        
        if max_val is not None and value > max_val:
            return Either.left(f"Number too large: {value} > {max_val}")
        
        return Either.right(value)
    
    @staticmethod
    def validate_enum(value: Any, allowed_values: List[Any]) -> Either[str, Any]:
        """Validate enum/choice parameters."""
        if value not in allowed_values:
            return Either.left(f"Invalid choice: {value}, allowed: {allowed_values}")
        return Either.right(value)
    
    @staticmethod
    def sanitize_path(path: str) -> Either[str, Path]:
        """Sanitize and validate file paths."""
        try:
            clean_path = Path(path).resolve()
            
            # Basic security checks
            if ".." in str(clean_path):
                return Either.left("Path traversal not allowed")
            
            if not clean_path.is_absolute():
                return Either.left("Only absolute paths allowed")
            
            return Either.right(clean_path)
            
        except Exception as e:
            return Either.left(f"Invalid path: {str(e)}")


class ActionBuilder:
    """Builder for creating custom actions with validation."""
    
    def __init__(self, plugin_id: PluginId):
        self.plugin_id = plugin_id
        self.actions: List[CustomAction] = []
    
    def add_action(self, action_id: ActionId, name: str, description: str,
                   handler: Callable, return_type: Type = str,
                   timeout: Duration = Duration.from_seconds(30.0)) -> 'ActionBuilder':
        """Add a custom action to the builder."""
        # Validate handler signature
        sig = inspect.signature(handler)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation
            else:
                param_type = str  # Default to string
            
            required = param.default == inspect.Parameter.empty
            default_value = param.default if not required else None
            
            action_param = CustomActionParameter(
                name=param_name,
                param_type=param_type,
                description=f"Parameter: {param_name}",
                required=required,
                default_value=default_value
            )
            parameters.append(action_param)
        
        action = CustomAction(
            action_id=action_id,
            name=name,
            description=description,
            parameters=parameters,
            return_type=return_type,
            handler=handler,
            plugin_id=self.plugin_id,
            execution_timeout=timeout
        )
        
        self.actions.append(action)
        return self
    
    def build(self) -> List[CustomAction]:
        """Build and return the list of custom actions."""
        return self.actions.copy()


class HookBuilder:
    """Builder for creating plugin hooks."""
    
    def __init__(self, plugin_id: PluginId):
        self.plugin_id = plugin_id
        self.hooks: List[PluginHook] = []
    
    def add_hook(self, hook_id: HookId, event_type: str, handler: Callable) -> 'HookBuilder':
        """Add a hook to the builder."""
        hook = PluginHook(hook_id, event_type, handler)
        self.hooks.append(hook)
        return self
    
    def build(self) -> List[PluginHook]:
        """Build and return the list of hooks."""
        return self.hooks.copy()


class BasePlugin(PluginInterface):
    """
    Base class for all plugins providing common functionality and structure.
    
    This class implements the PluginInterface and provides a foundation for
    plugin development with built-in logging, validation, and API access.
    """
    
    def __init__(self, metadata: PluginMetadata, configuration: Optional[PluginConfiguration] = None):
        self._metadata = metadata
        self._configuration = configuration or PluginConfiguration(
            plugin_id=metadata.identifier,
            settings={},
            security_profile=SecurityProfile.STANDARD
        )
        self._api_bridge: Optional[PluginAPIBridge] = None
        self._logger: Optional[PluginLogger] = None
        self._sdk_config = SDKConfiguration()
        self._initialized = False
        self._active = False
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self._metadata
    
    @property
    def configuration(self) -> PluginConfiguration:
        """Get plugin configuration."""
        return self._configuration
    
    @property
    def logger(self) -> PluginLogger:
        """Get plugin logger."""
        if not self._logger:
            self._logger = PluginLogger(self.metadata.identifier, self._sdk_config)
        return self._logger
    
    @property
    def api(self) -> Optional[PluginAPIBridge]:
        """Get API bridge for calling MCP tools."""
        return self._api_bridge
    
    async def initialize(self, api_bridge: PluginAPIBridge) -> Either[PluginError, None]:
        """Initialize plugin with API bridge."""
        try:
            self._api_bridge = api_bridge
            self.logger.info("Plugin initializing", api_version=api_bridge.api_config["version"].value)
            
            # Perform custom initialization
            init_result = await self.on_initialize()
            if init_result.is_left():
                return init_result
            
            self._initialized = True
            self.logger.info("Plugin initialized successfully")
            return Either.right(None)
            
        except Exception as e:
            error_msg = f"Plugin initialization failed: {str(e)}"
            self.logger.error(error_msg, exception=str(e))
            return Either.left(PluginError.initialization_failed(error_msg))
    
    async def activate(self) -> Either[PluginError, None]:
        """Activate plugin and register actions."""
        try:
            if not self._initialized:
                return Either.left(PluginError.activation_failed("Plugin not initialized"))
            
            self.logger.info("Plugin activating")
            
            # Perform custom activation
            activation_result = await self.on_activate()
            if activation_result.is_left():
                return activation_result
            
            self._active = True
            self.logger.info("Plugin activated successfully")
            return Either.right(None)
            
        except Exception as e:
            error_msg = f"Plugin activation failed: {str(e)}"
            self.logger.error(error_msg, exception=str(e))
            return Either.left(PluginError.activation_failed(error_msg))
    
    async def deactivate(self) -> Either[PluginError, None]:
        """Deactivate plugin and cleanup resources."""
        try:
            self.logger.info("Plugin deactivating")
            
            # Perform custom deactivation
            deactivation_result = await self.on_deactivate()
            if deactivation_result.is_left():
                return deactivation_result
            
            self._active = False
            self.logger.info("Plugin deactivated successfully")
            return Either.right(None)
            
        except Exception as e:
            error_msg = f"Plugin deactivation failed: {str(e)}"
            self.logger.error(error_msg, exception=str(e))
            return Either.left(PluginError(error_msg, "DEACTIVATION_FAILED"))
    
    async def get_custom_actions(self) -> List[CustomAction]:
        """Get list of custom actions provided by plugin."""
        try:
            return await self.register_actions()
        except Exception as e:
            self.logger.error("Failed to get custom actions", exception=str(e))
            return []
    
    async def get_hooks(self) -> List[PluginHook]:
        """Get list of event hooks provided by plugin."""
        try:
            return await self.register_hooks()
        except Exception as e:
            self.logger.error("Failed to get hooks", exception=str(e))
            return []
    
    async def call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Either[PluginError, Any]:
        """
        Call an MCP tool through the API bridge.
        
        This method provides a convenient interface for plugins to access
        the 38+ available MCP tools with automatic permission checking.
        """
        if not self._api_bridge:
            return Either.left(PluginError("API bridge not available", "API_NOT_AVAILABLE"))
        
        if not self._active:
            return Either.left(PluginError("Plugin not active", "PLUGIN_NOT_ACTIVE"))
        
        try:
            self.logger.debug(f"Calling MCP tool: {tool_name}", parameters=parameters)
            
            result = await self._api_bridge.call_tool(
                plugin_id=self.metadata.identifier,
                plugin_permissions=self.metadata.permissions,
                tool_name=tool_name,
                parameters=parameters
            )
            
            if result.is_right():
                self.logger.debug(f"MCP tool call successful: {tool_name}")
            else:
                error = result.get_left()
                self.logger.warning(f"MCP tool call failed: {tool_name}", error=error.message)
            
            return result
            
        except Exception as e:
            error_msg = f"MCP tool call exception: {str(e)}"
            self.logger.error(error_msg, tool=tool_name, exception=str(e))
            return Either.left(PluginError.execution_error(error_msg))
    
    # Abstract methods for plugin developers to implement
    
    @abstractmethod
    async def on_initialize(self) -> Either[PluginError, None]:
        """Custom initialization logic for the plugin."""
        pass
    
    @abstractmethod
    async def on_activate(self) -> Either[PluginError, None]:
        """Custom activation logic for the plugin."""
        pass
    
    @abstractmethod
    async def on_deactivate(self) -> Either[PluginError, None]:
        """Custom deactivation logic for the plugin."""
        pass
    
    @abstractmethod
    async def register_actions(self) -> List[CustomAction]:
        """Register custom actions provided by this plugin."""
        pass
    
    @abstractmethod
    async def register_hooks(self) -> List[PluginHook]:
        """Register event hooks provided by this plugin."""
        pass


class UtilityPlugin(BasePlugin):
    """Specialized base class for utility plugins with common patterns."""
    
    def __init__(self, metadata: PluginMetadata, configuration: Optional[PluginConfiguration] = None):
        super().__init__(metadata, configuration)
        self._action_builder = ActionBuilder(metadata.identifier)
        self._hook_builder = HookBuilder(metadata.identifier)
    
    def action(self, action_id: str, name: str, description: str, 
               timeout: Duration = Duration.from_seconds(30.0)):
        """Decorator for registering action methods."""
        def decorator(func):
            self._action_builder.add_action(
                action_id=ActionId(action_id),
                name=name,
                description=description,
                handler=func,
                timeout=timeout
            )
            return func
        return decorator
    
    def hook(self, hook_id: str, event_type: str):
        """Decorator for registering hook methods."""
        def decorator(func):
            self._hook_builder.add_hook(
                hook_id=HookId(hook_id),
                event_type=event_type,
                handler=func
            )
            return func
        return decorator
    
    async def register_actions(self) -> List[CustomAction]:
        """Return actions registered via decorators."""
        return self._action_builder.build()
    
    async def register_hooks(self) -> List[PluginHook]:
        """Return hooks registered via decorators."""
        return self._hook_builder.build()


class IntegrationPlugin(BasePlugin):
    """Specialized base class for integration plugins with external services."""
    
    def __init__(self, metadata: PluginMetadata, configuration: Optional[PluginConfiguration] = None):
        super().__init__(metadata, configuration)
        self._connection_pool = {}
        self._retry_config = {
            "max_retries": 3,
            "backoff_factor": 1.5,
            "timeout_seconds": 30
        }
    
    async def get_connection(self, service_name: str) -> Any:
        """Get connection to external service with pooling."""
        if service_name not in self._connection_pool:
            connection = await self.create_connection(service_name)
            self._connection_pool[service_name] = connection
        
        return self._connection_pool[service_name]
    
    async def create_connection(self, service_name: str) -> Any:
        """Create connection to external service - override in subclass."""
        raise NotImplementedError("Subclasses must implement create_connection")
    
    async def close_connections(self):
        """Close all open connections."""
        for service_name, connection in self._connection_pool.items():
            try:
                if hasattr(connection, 'close'):
                    await connection.close()
                self.logger.debug(f"Closed connection to {service_name}")
            except Exception as e:
                self.logger.warning(f"Error closing connection to {service_name}", exception=str(e))
        
        self._connection_pool.clear()
    
    async def on_deactivate(self) -> Either[PluginError, None]:
        """Clean up connections on deactivation."""
        await self.close_connections()
        return Either.right(None)


# SDK utility functions

def create_plugin_metadata(
    identifier: str,
    name: str,
    version: str,
    description: str,
    author: str,
    plugin_type: str = "utility",
    permissions: Optional[PluginPermissions] = None
) -> PluginMetadata:
    """Utility function to create plugin metadata."""
    from ..core.plugin_architecture import PluginType
    
    return PluginMetadata(
        identifier=PluginId(identifier),
        name=name,
        version=version,
        description=description,
        author=author,
        plugin_type=PluginType(plugin_type),
        api_version=ApiVersion.V1_0,
        permissions=permissions or PluginPermissions.standard()
    )


def validate_plugin_structure(plugin_path: Path) -> Either[str, None]:
    """Validate plugin directory structure."""
    required_files = ["main.py", "manifest.json"]
    
    for file_name in required_files:
        file_path = plugin_path / file_name
        if not file_path.exists():
            return Either.left(f"Required file missing: {file_name}")
    
    return Either.right(None)


async def test_plugin_action(plugin: BasePlugin, action_id: str, parameters: Dict[str, Any]) -> Either[str, Any]:
    """Test a plugin action with given parameters."""
    try:
        actions = await plugin.get_custom_actions()
        action = next((a for a in actions if a.action_id == action_id), None)
        
        if not action:
            return Either.left(f"Action not found: {action_id}")
        
        result = await action.execute(parameters)
        return result
        
    except Exception as e:
        return Either.left(f"Test execution failed: {str(e)}")


# Example plugin template

class ExamplePlugin(UtilityPlugin):
    """Example plugin demonstrating SDK usage."""
    
    async def on_initialize(self) -> Either[PluginError, None]:
        """Initialize the example plugin."""
        self.logger.info("Example plugin initializing")
        return Either.right(None)
    
    async def on_activate(self) -> Either[PluginError, None]:
        """Activate the example plugin."""
        self.logger.info("Example plugin activating")
        return Either.right(None)
    
    async def on_deactivate(self) -> Either[PluginError, None]:
        """Deactivate the example plugin."""
        self.logger.info("Example plugin deactivating")
        return Either.right(None)
    
    @UtilityPlugin.action("hello_world", "Hello World", "Say hello to the world")
    async def hello_world(self, name: str = "World") -> str:
        """Example action that says hello."""
        return f"Hello, {name}!"
    
    @UtilityPlugin.action("get_system_info", "Get System Info", "Get current system information")
    async def get_system_info(self) -> Dict[str, Any]:
        """Example action that gets system info via MCP tools."""
        # Call an MCP tool to get system information
        result = await self.call_mcp_tool("km_variable_manager", {
            "operation": "get",
            "name": "CurrentUser"
        })
        
        if result.is_right():
            user = result.get_right()
            return {"current_user": user, "status": "success"}
        else:
            return {"error": result.get_left().message, "status": "error"}