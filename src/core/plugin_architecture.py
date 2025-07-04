"""
Core plugin architecture types and protocols for the plugin ecosystem.

This module defines the fundamental plugin architecture with branded types,
security boundaries, lifecycle management, and API specifications.
"""

from __future__ import annotations
from typing import NewType, Any, Dict, List, Optional, Set, Callable, Type, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re
import inspect
import hashlib
from datetime import datetime
from pathlib import Path

from .types import Duration
from .either import Either
from .errors import ValidationError, SecurityError, create_error_context


# Branded Types for Plugin Management
PluginId = NewType('PluginId', str)
ActionId = NewType('ActionId', str)
HookId = NewType('HookId', str)
PermissionId = NewType('PermissionId', str)
PluginChecksum = NewType('PluginChecksum', str)


class PluginType(Enum):
    """Plugin types and categories."""
    ACTION = "action"              # Custom action plugins
    INTEGRATION = "integration"    # Third-party service integrations
    TRANSFORMATION = "transformation"  # Data transformation plugins
    TRIGGER = "trigger"            # Custom trigger plugins
    INTERFACE = "interface"        # UI and interaction plugins
    UTILITY = "utility"            # Helper and utility plugins
    BRIDGE = "bridge"              # API bridge plugins


class SecurityProfile(Enum):
    """Plugin security profiles with increasing restrictions."""
    NONE = "none"                  # No security restrictions (development only)
    STANDARD = "standard"          # Basic security validation
    STRICT = "strict"              # Enhanced security with limitations
    SANDBOX = "sandbox"            # Full sandboxed execution


class PluginStatus(Enum):
    """Plugin lifecycle status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"
    UPDATING = "updating"
    UNINSTALLING = "uninstalling"


class ApiVersion(Enum):
    """Supported API versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


@dataclass(frozen=True)
class PluginPermissions:
    """Type-safe plugin permission system."""
    permissions: Set[PermissionId]
    resource_limits: Dict[str, int] = field(default_factory=dict)
    network_access: bool = False
    file_system_access: bool = False
    system_integration: bool = False
    
    def has_permission(self, permission: PermissionId) -> bool:
        """Check if specific permission is granted."""
        return permission in self.permissions
    
    def requires_elevated_access(self) -> bool:
        """Check if plugin requires elevated access privileges."""
        return self.network_access or self.file_system_access or self.system_integration
    
    @classmethod
    def minimal(cls) -> PluginPermissions:
        """Create minimal permission set."""
        return cls(permissions=set(), resource_limits={"memory_mb": 64, "cpu_percent": 5})
    
    @classmethod
    def standard(cls) -> PluginPermissions:
        """Create standard permission set."""
        return cls(
            permissions={PermissionId("basic_execution")},
            resource_limits={"memory_mb": 256, "cpu_percent": 15}
        )


@dataclass(frozen=True)
class PluginAPI:
    """Plugin API specification with version compatibility."""
    version: ApiVersion
    compatible_versions: Set[ApiVersion]
    endpoints: Dict[str, Type]
    permissions: PluginPermissions
    
    def __post_init__(self):
        if self.version not in self.compatible_versions:
            # Add current version to compatible versions
            object.__setattr__(self, 'compatible_versions', 
                             self.compatible_versions | {self.version})
    
    def is_compatible(self, required_version: ApiVersion) -> bool:
        """Check if API version is compatible."""
        return required_version in self.compatible_versions
    
    def get_endpoint(self, name: str) -> Optional[Type]:
        """Get endpoint type by name."""
        return self.endpoints.get(name)


@dataclass(frozen=True)
class PluginDependency:
    """Plugin dependency specification."""
    plugin_id: PluginId
    version_requirement: str  # Semantic version requirement (e.g., ">=1.0.0")
    optional: bool = False
    
    def __post_init__(self):
        if not self._is_valid_version_requirement(self.version_requirement):
            raise ValueError(f"Invalid version requirement: {self.version_requirement}")
    
    def _is_valid_version_requirement(self, requirement: str) -> bool:
        """Validate version requirement format."""
        # Support semantic versioning with operators
        pattern = r'^(>=|<=|>|<|==|!=)?\s*\d+\.\d+(\.\d+)?(-[a-zA-Z0-9]+)?$'
        return re.match(pattern, requirement) is not None
    
    def is_satisfied_by(self, version: str) -> bool:
        """Check if dependency is satisfied by given version."""
        # Simplified version checking - would use proper semver library
        required_version = self.version_requirement.lstrip(">=<!=")
        return version >= required_version


@dataclass(frozen=True)
class PluginMetadata:
    """Comprehensive plugin metadata and manifest information."""
    identifier: PluginId
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    api_version: ApiVersion
    dependencies: List[PluginDependency] = field(default_factory=list)
    permissions: PluginPermissions = field(default_factory=PluginPermissions.minimal)
    entry_point: str = "main"
    configuration_schema: Optional[Dict[str, Any]] = None
    icon_path: Optional[str] = None
    homepage_url: Optional[str] = None
    repository_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    checksum: Optional[PluginChecksum] = None
    
    def __post_init__(self):
        if not self.identifier or not self._is_valid_identifier(self.identifier):
            raise ValueError(f"Invalid plugin identifier: {self.identifier}")
        if not self.name or len(self.name.strip()) == 0:
            raise ValueError("Plugin name cannot be empty")
        if not self._is_valid_version(self.version):
            raise ValueError(f"Invalid version format: {self.version}")
    
    def _is_valid_identifier(self, identifier: str) -> bool:
        """Validate plugin identifier format."""
        # Allow alphanumeric, hyphens, underscores, dots
        pattern = r'^[a-zA-Z][a-zA-Z0-9_.-]*$'
        return bool(re.match(pattern, identifier))
    
    def _is_valid_version(self, version: str) -> bool:
        """Validate version format (semantic versioning)."""
        pattern = r'^\d+\.\d+(\.\d+)?(-[a-zA-Z0-9]+)?$'
        return bool(re.match(pattern, version))
    
    def get_full_identifier(self) -> str:
        """Get full plugin identifier with version."""
        return f"{self.identifier}@{self.version}"
    
    def calculate_checksum(self, plugin_content: bytes) -> PluginChecksum:
        """Calculate plugin content checksum."""
        sha256_hash = hashlib.sha256(plugin_content).hexdigest()
        return PluginChecksum(sha256_hash[:16])  # First 16 chars for brevity
    
    def verify_checksum(self, plugin_content: bytes) -> bool:
        """Verify plugin content against stored checksum."""
        if not self.checksum:
            return False
        calculated = self.calculate_checksum(plugin_content)
        return calculated == self.checksum


@dataclass(frozen=True)
class CustomActionParameter:
    """Custom action parameter specification."""
    name: str
    param_type: Type
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name or not self._is_valid_parameter_name(self.name):
            raise ValueError(f"Invalid parameter name: {self.name}")
    
    def _is_valid_parameter_name(self, name: str) -> bool:
        """Validate parameter name format."""
        pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, name))
    
    def validate_value(self, value: Any) -> Either[ValidationError, None]:
        """Validate parameter value against type and constraints."""
        try:
            # Required parameter check
            if self.required and value is None:
                return Either.left(ValidationError(f"Required parameter '{self.name}' is missing"))
            
            # Type checking
            if value is not None and not isinstance(value, self.param_type):
                return Either.left(ValidationError(
                    f"Parameter '{self.name}' must be of type {self.param_type.__name__}"
                ))
            
            # Constraint validation
            if value is not None and self.constraints:
                for constraint, constraint_value in self.constraints.items():
                    if constraint == "min_length" and hasattr(value, "__len__"):
                        if len(value) < constraint_value:
                            return Either.left(ValidationError(
                                f"Parameter '{self.name}' must have at least {constraint_value} characters"
                            ))
                    elif constraint == "max_length" and hasattr(value, "__len__"):
                        if len(value) > constraint_value:
                            return Either.left(ValidationError(
                                f"Parameter '{self.name}' must have at most {constraint_value} characters"
                            ))
                    elif constraint == "min_value" and isinstance(value, (int, float)):
                        if value < constraint_value:
                            return Either.left(ValidationError(
                                f"Parameter '{self.name}' must be at least {constraint_value}"
                            ))
                    elif constraint == "max_value" and isinstance(value, (int, float)):
                        if value > constraint_value:
                            return Either.left(ValidationError(
                                f"Parameter '{self.name}' must be at most {constraint_value}"
                            ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(ValidationError(f"Parameter validation error: {str(e)}"))


@dataclass(frozen=True)
class CustomAction:
    """Custom action definition with comprehensive specification."""
    action_id: ActionId
    name: str
    description: str
    parameters: List[CustomActionParameter]
    return_type: Type
    handler: Callable
    plugin_id: PluginId
    execution_timeout: Duration = Duration.from_seconds(30.0)
    
    def __post_init__(self):
        if not self.action_id or not self.name:
            raise ValueError("Action ID and name are required")
        if not callable(self.handler):
            raise ValueError("Handler must be callable")
    
    def validate_parameters(self, params: Dict[str, Any]) -> Either[ValidationError, None]:
        """Validate action parameters against specification."""
        try:
            # Validate each parameter
            for param_spec in self.parameters:
                param_value = params.get(param_spec.name)
                validation_result = param_spec.validate_value(param_value)
                if validation_result.is_left():
                    return validation_result
            
            # Check for unexpected parameters
            expected_params = {p.name for p in self.parameters}
            provided_params = set(params.keys())
            unexpected = provided_params - expected_params
            
            if unexpected:
                return Either.left(ValidationError(
                    f"Unexpected parameters: {', '.join(unexpected)}"
                ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(ValidationError(f"Parameter validation failed: {str(e)}"))
    
    async def execute(self, params: Dict[str, Any]) -> Either[PluginError, Any]:
        """Execute custom action with comprehensive validation and error handling."""
        try:
            # Validate parameters
            validation_result = self.validate_parameters(params)
            if validation_result.is_left():
                error = validation_result.get_left()
                return Either.left(PluginError.parameter_validation_error(error.message))
            
            # Add default values for missing optional parameters
            final_params = {}
            for param_spec in self.parameters:
                if param_spec.name in params:
                    final_params[param_spec.name] = params[param_spec.name]
                elif not param_spec.required and param_spec.default_value is not None:
                    final_params[param_spec.name] = param_spec.default_value
            
            # Execute handler with timeout
            import asyncio
            if inspect.iscoroutinefunction(self.handler):
                result = await asyncio.wait_for(
                    self.handler(**final_params),
                    timeout=self.execution_timeout.total_seconds()
                )
            else:
                result = self.handler(**final_params)
            
            return Either.right(result)
            
        except asyncio.TimeoutError:
            return Either.left(PluginError.execution_timeout(self.action_id))
        except Exception as e:
            context = create_error_context("execute_action", "custom_action", 
                                         action_id=self.action_id, error=str(e))
            return Either.left(PluginError.execution_error(str(e), context))


@dataclass(frozen=True)
class PluginConfiguration:
    """Plugin configuration and runtime settings."""
    plugin_id: PluginId
    settings: Dict[str, Any]
    enabled: bool = True
    auto_update: bool = False
    security_profile: SecurityProfile = SecurityProfile.STANDARD
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.plugin_id:
            raise ValueError("Plugin ID is required")
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get configuration setting with default."""
        return self.settings.get(key, default)
    
    def update_setting(self, key: str, value: Any) -> PluginConfiguration:
        """Create new configuration with updated setting."""
        new_settings = self.settings.copy()
        new_settings[key] = value
        return PluginConfiguration(
            plugin_id=self.plugin_id,
            settings=new_settings,
            enabled=self.enabled,
            auto_update=self.auto_update,
            security_profile=self.security_profile,
            resource_limits=self.resource_limits
        )
    
    def validate_setting(self, key: str, value: Any, schema: Optional[Dict[str, Any]] = None) -> Either[ValidationError, None]:
        """Validate setting against schema if provided."""
        if not schema:
            return Either.right(None)
        
        try:
            # Basic schema validation would go here
            # In a full implementation, would use jsonschema or similar
            if key in schema:
                expected_type = schema[key].get("type")
                if expected_type and not isinstance(value, eval(expected_type)):
                    return Either.left(ValidationError(
                        f"Setting '{key}' must be of type {expected_type}"
                    ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(ValidationError(f"Setting validation failed: {str(e)}"))


class PluginHook:
    """Plugin hook for event-driven plugin activation."""
    
    def __init__(self, hook_id: HookId, event_type: str, handler: Callable):
        self.hook_id = hook_id
        self.event_type = event_type
        self.handler = handler
        self.active = True
    
    async def execute(self, event_data: Dict[str, Any]) -> Either[PluginError, Any]:
        """Execute hook handler with event data."""
        try:
            if not self.active:
                return Either.left(PluginError.hook_disabled(self.hook_id))
            
            if inspect.iscoroutinefunction(self.handler):
                result = await self.handler(event_data)
            else:
                result = self.handler(event_data)
            
            return Either.right(result)
            
        except Exception as e:
            return Either.left(PluginError.hook_execution_failed(self.hook_id, str(e)))


class PluginInterface(Protocol):
    """Protocol defining the plugin interface."""
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        ...
    
    @property
    def configuration(self) -> PluginConfiguration:
        """Get plugin configuration."""
        ...
    
    async def initialize(self, api_bridge: 'PluginAPIBridge') -> Either[PluginError, None]:
        """Initialize plugin with API bridge."""
        ...
    
    async def activate(self) -> Either[PluginError, None]:
        """Activate plugin and register actions."""
        ...
    
    async def deactivate(self) -> Either[PluginError, None]:
        """Deactivate plugin and cleanup resources."""
        ...
    
    async def get_custom_actions(self) -> List[CustomAction]:
        """Get list of custom actions provided by plugin."""
        ...
    
    async def get_hooks(self) -> List[PluginHook]:
        """Get list of event hooks provided by plugin."""
        ...


# Error types for plugin operations
class PluginError(Exception):
    """Base class for plugin-related errors."""
    
    def __init__(self, message: str, error_code: str = "PLUGIN_ERROR", context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    @classmethod
    def plugin_not_found(cls, plugin_id: PluginId) -> PluginError:
        return cls(f"Plugin not found: {plugin_id}", "PLUGIN_NOT_FOUND")
    
    @classmethod
    def invalid_plugin_format(cls, details: str) -> PluginError:
        return cls(f"Invalid plugin format: {details}", "INVALID_FORMAT")
    
    @classmethod
    def initialization_failed(cls, details: str) -> PluginError:
        return cls(f"Plugin initialization failed: {details}", "INIT_FAILED")
    
    @classmethod
    def activation_failed(cls, details: str) -> PluginError:
        return cls(f"Plugin activation failed: {details}", "ACTIVATION_FAILED")
    
    @classmethod
    def execution_error(cls, details: str, context: Optional[Dict[str, Any]] = None) -> PluginError:
        return cls(f"Plugin execution error: {details}", "EXECUTION_ERROR", context)
    
    @classmethod
    def execution_timeout(cls, action_id: ActionId) -> PluginError:
        return cls(f"Action execution timeout: {action_id}", "EXECUTION_TIMEOUT")
    
    @classmethod
    def parameter_validation_error(cls, details: str) -> PluginError:
        return cls(f"Parameter validation failed: {details}", "PARAMETER_ERROR")
    
    @classmethod
    def permission_denied(cls, permission: str) -> PluginError:
        return cls(f"Permission denied: {permission}", "PERMISSION_DENIED")
    
    @classmethod
    def security_violation(cls, details: str) -> PluginError:
        return cls(f"Security violation: {details}", "SECURITY_VIOLATION")
    
    @classmethod
    def hook_execution_failed(cls, hook_id: HookId, details: str) -> PluginError:
        return cls(f"Hook execution failed {hook_id}: {details}", "HOOK_ERROR")