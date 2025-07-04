"""
Plugin lifecycle management and orchestration system.

This module provides comprehensive plugin management including installation,
loading, activation, and execution with security validation and resource monitoring.
"""

import asyncio
import json
import logging
import importlib.util
import sys
from typing import Dict, List, Optional, Set, Any, Type
from pathlib import Path
from datetime import datetime
import tempfile
import zipfile
import shutil

from ..core.plugin_architecture import (
    PluginMetadata, PluginConfiguration, CustomAction, PluginHook,
    PluginId, ActionId, HookId, PluginStatus, SecurityProfile, ApiVersion,
    PluginError, PluginInterface, PluginPermissions
)
from ..core.either import Either
from ..core.errors import create_error_context
from .security_sandbox import PluginSecurityManager, PluginSandbox
# Note: Import PluginAPIBridge only when needed to avoid circular imports

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for tracking installed and available plugins."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path.home() / ".km-mcp" / "plugins" / "registry.json"
        self.plugins: Dict[PluginId, PluginMetadata] = {}
        self.installed_plugins: Dict[PluginId, Path] = {}
        self._ensure_registry_directory()
    
    def _ensure_registry_directory(self):
        """Ensure registry directory exists."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def load_registry(self) -> Either[PluginError, None]:
        """Load plugin registry from disk."""
        try:
            if not self.registry_path.exists():
                # Create empty registry
                await self.save_registry()
                return Either.right(None)
            
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
            
            # Reconstruct plugin metadata
            for plugin_data in registry_data.get("plugins", []):
                metadata = self._deserialize_metadata(plugin_data)
                self.plugins[metadata.identifier] = metadata
                
                plugin_path = Path(plugin_data.get("install_path", ""))
                if plugin_path.exists():
                    self.installed_plugins[metadata.identifier] = plugin_path
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError.initialization_failed(f"Failed to load registry: {str(e)}"))
    
    async def save_registry(self) -> Either[PluginError, None]:
        """Save plugin registry to disk."""
        try:
            registry_data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "plugins": []
            }
            
            for plugin_id, metadata in self.plugins.items():
                plugin_data = self._serialize_metadata(metadata)
                plugin_data["install_path"] = str(self.installed_plugins.get(plugin_id, ""))
                registry_data["plugins"].append(plugin_data)
            
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError.initialization_failed(f"Failed to save registry: {str(e)}"))
    
    def register_plugin(self, metadata: PluginMetadata, install_path: Path):
        """Register plugin in registry."""
        self.plugins[metadata.identifier] = metadata
        self.installed_plugins[metadata.identifier] = install_path
    
    def unregister_plugin(self, plugin_id: PluginId):
        """Unregister plugin from registry."""
        self.plugins.pop(plugin_id, None)
        self.installed_plugins.pop(plugin_id, None)
    
    def get_plugin_metadata(self, plugin_id: PluginId) -> Optional[PluginMetadata]:
        """Get plugin metadata by ID."""
        return self.plugins.get(plugin_id)
    
    def get_plugin_path(self, plugin_id: PluginId) -> Optional[Path]:
        """Get plugin installation path."""
        return self.installed_plugins.get(plugin_id)
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all registered plugins."""
        return list(self.plugins.values())
    
    def _serialize_metadata(self, metadata: PluginMetadata) -> Dict[str, Any]:
        """Serialize plugin metadata to dictionary."""
        return {
            "identifier": metadata.identifier,
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "author": metadata.author,
            "plugin_type": metadata.plugin_type.value,
            "api_version": metadata.api_version.value,
            "dependencies": [
                {
                    "plugin_id": dep.plugin_id,
                    "version_requirement": dep.version_requirement,
                    "optional": dep.optional
                }
                for dep in metadata.dependencies
            ],
            "permissions": {
                "permissions": list(metadata.permissions.permissions),
                "resource_limits": metadata.permissions.resource_limits,
                "network_access": metadata.permissions.network_access,
                "file_system_access": metadata.permissions.file_system_access,
                "system_integration": metadata.permissions.system_integration
            },
            "entry_point": metadata.entry_point,
            "configuration_schema": metadata.configuration_schema,
            "icon_path": metadata.icon_path,
            "homepage_url": metadata.homepage_url,
            "repository_url": metadata.repository_url,
            "created_at": metadata.created_at.isoformat(),
            "checksum": metadata.checksum
        }
    
    def _deserialize_metadata(self, data: Dict[str, Any]) -> PluginMetadata:
        """Deserialize plugin metadata from dictionary."""
        from ..core.plugin_architecture import PluginType, ApiVersion, PluginDependency, PluginPermissions, PermissionId
        
        dependencies = []
        for dep_data in data.get("dependencies", []):
            dependencies.append(PluginDependency(
                plugin_id=PluginId(dep_data["plugin_id"]),
                version_requirement=dep_data["version_requirement"],
                optional=dep_data.get("optional", False)
            ))
        
        permissions_data = data.get("permissions", {})
        permissions = PluginPermissions(
            permissions={PermissionId(p) for p in permissions_data.get("permissions", [])},
            resource_limits=permissions_data.get("resource_limits", {}),
            network_access=permissions_data.get("network_access", False),
            file_system_access=permissions_data.get("file_system_access", False),
            system_integration=permissions_data.get("system_integration", False)
        )
        
        return PluginMetadata(
            identifier=PluginId(data["identifier"]),
            name=data["name"],
            version=data["version"],
            description=data["description"],
            author=data["author"],
            plugin_type=PluginType(data["plugin_type"]),
            api_version=ApiVersion(data["api_version"]),
            dependencies=dependencies,
            permissions=permissions,
            entry_point=data.get("entry_point", "main"),
            configuration_schema=data.get("configuration_schema"),
            icon_path=data.get("icon_path"),
            homepage_url=data.get("homepage_url"),
            repository_url=data.get("repository_url"),
            created_at=datetime.fromisoformat(data["created_at"]),
            checksum=data.get("checksum")
        )


class PluginLoader:
    """Plugin loading and module management."""
    
    def __init__(self):
        self.loaded_modules: Dict[PluginId, Any] = {}
    
    async def load_plugin_from_path(self, plugin_path: Path, metadata: PluginMetadata) -> Either[PluginError, PluginInterface]:
        """Load plugin from file system path."""
        try:
            # Determine if it's a zip file or directory
            if plugin_path.suffix == '.zip':
                return await self._load_from_zip(plugin_path, metadata)
            elif plugin_path.is_dir():
                return await self._load_from_directory(plugin_path, metadata)
            else:
                return Either.left(PluginError.invalid_plugin_format(f"Unsupported plugin format: {plugin_path}"))
            
        except Exception as e:
            return Either.left(PluginError.initialization_failed(f"Failed to load plugin: {str(e)}"))
    
    async def _load_from_zip(self, zip_path: Path, metadata: PluginMetadata) -> Either[PluginError, PluginInterface]:
        """Load plugin from zip file."""
        try:
            # Extract to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_file:
                    zip_file.extractall(temp_path)
                
                # Find plugin directory (usually the first directory in zip)
                plugin_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
                if not plugin_dirs:
                    return Either.left(PluginError.invalid_plugin_format("No plugin directory found in zip"))
                
                plugin_dir = plugin_dirs[0]
                return await self._load_from_directory(plugin_dir, metadata)
            
        except Exception as e:
            return Either.left(PluginError.initialization_failed(f"Failed to load from zip: {str(e)}"))
    
    async def _load_from_directory(self, plugin_dir: Path, metadata: PluginMetadata) -> Either[PluginError, PluginInterface]:
        """Load plugin from directory."""
        try:
            # Look for entry point file
            entry_point_file = plugin_dir / f"{metadata.entry_point}.py"
            if not entry_point_file.exists():
                return Either.left(PluginError.invalid_plugin_format(
                    f"Entry point file not found: {metadata.entry_point}.py"
                ))
            
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(
                f"plugin_{metadata.identifier}",
                entry_point_file
            )
            
            if not spec or not spec.loader:
                return Either.left(PluginError.invalid_plugin_format("Invalid Python module"))
            
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules temporarily for loading
            sys.modules[spec.name] = module
            
            try:
                spec.loader.exec_module(module)
                
                # Look for plugin class
                plugin_class = getattr(module, 'Plugin', None)
                if not plugin_class:
                    return Either.left(PluginError.invalid_plugin_format("Plugin class not found"))
                
                # Create plugin instance
                plugin_instance = plugin_class(metadata)
                
                # Store loaded module
                self.loaded_modules[metadata.identifier] = module
                
                return Either.right(plugin_instance)
                
            finally:
                # Clean up sys.modules
                sys.modules.pop(spec.name, None)
            
        except Exception as e:
            return Either.left(PluginError.initialization_failed(f"Failed to load module: {str(e)}"))
    
    def unload_plugin(self, plugin_id: PluginId):
        """Unload plugin module."""
        self.loaded_modules.pop(plugin_id, None)


class PluginManager:
    """Comprehensive plugin lifecycle and management system."""
    
    def __init__(self, plugins_directory: Optional[Path] = None):
        self.plugins_directory = plugins_directory or Path.home() / ".km-mcp" / "plugins"
        self.plugins_directory.mkdir(parents=True, exist_ok=True)
        
        self.registry = PluginRegistry()
        self.loader = PluginLoader()
        self.security_manager = PluginSecurityManager()
        from .api_bridge import PluginAPIBridge
        self.api_bridge = PluginAPIBridge()
        
        # Runtime state
        self.loaded_plugins: Dict[PluginId, PluginInterface] = {}
        self.plugin_configurations: Dict[PluginId, PluginConfiguration] = {}
        self.custom_actions: Dict[ActionId, CustomAction] = {}
        self.plugin_hooks: Dict[str, List[PluginHook]] = {}  # event_type -> hooks
        
        # Default API configuration
        self.plugin_api = {
            "version": ApiVersion.V1_0,
            "compatible_versions": {ApiVersion.V1_0},
        }
    
    async def initialize(self) -> Either[PluginError, None]:
        """Initialize plugin manager and load registry."""
        try:
            # Load registry
            registry_result = await self.registry.load_registry()
            if registry_result.is_left():
                return registry_result
            
            # Initialize API bridge
            bridge_result = await self.api_bridge.initialize()
            if bridge_result.is_left():
                return Either.left(PluginError.initialization_failed(f"API bridge init failed: {bridge_result.get_left()}"))
            
            logger.info("Plugin manager initialized successfully")
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError.initialization_failed(f"Manager initialization failed: {str(e)}"))
    
    async def install_plugin(self, plugin_source: str, configuration: Optional[PluginConfiguration] = None) -> Either[PluginError, PluginMetadata]:
        """Install plugin from source with comprehensive validation."""
        try:
            # Parse plugin source (file path, URL, etc.)
            source_path = Path(plugin_source)
            if not source_path.exists():
                return Either.left(PluginError.plugin_not_found(PluginId(plugin_source)))
            
            # Load and validate plugin metadata
            metadata_result = await self._extract_plugin_metadata(source_path)
            if metadata_result.is_left():
                return metadata_result
            
            metadata = metadata_result.get_right()
            
            # Security validation
            security_result = await self.security_manager.validate_plugin(source_path, metadata)
            if security_result.is_left():
                return security_result
            
            # Check dependencies
            deps_result = await self._validate_dependencies(metadata.dependencies)
            if deps_result.is_left():
                return deps_result
            
            # Install plugin files
            install_path = self.plugins_directory / metadata.identifier
            if install_path.exists():
                # Remove existing installation
                shutil.rmtree(install_path)
            
            # Copy/extract plugin
            if source_path.suffix == '.zip':
                with zipfile.ZipFile(source_path, 'r') as zip_file:
                    zip_file.extractall(install_path)
            else:
                shutil.copytree(source_path, install_path)
            
            # Register plugin
            self.registry.register_plugin(metadata, install_path)
            await self.registry.save_registry()
            
            # Create default configuration if not provided
            if configuration is None:
                configuration = PluginConfiguration(
                    plugin_id=metadata.identifier,
                    settings={},
                    security_profile=SecurityProfile.STANDARD
                )
            
            self.plugin_configurations[metadata.identifier] = configuration
            
            logger.info(f"Plugin installed successfully: {metadata.identifier}")
            return Either.right(metadata)
            
        except Exception as e:
            context = create_error_context("install_plugin", "plugin_manager", source=plugin_source)
            return Either.left(PluginError.initialization_failed(f"Installation failed: {str(e)}", context))
    
    async def load_plugin(self, plugin_id: PluginId) -> Either[PluginError, None]:
        """Load plugin into memory."""
        try:
            if plugin_id in self.loaded_plugins:
                return Either.right(None)  # Already loaded
            
            # Get plugin metadata and path
            metadata = self.registry.get_plugin_metadata(plugin_id)
            if not metadata:
                return Either.left(PluginError.plugin_not_found(plugin_id))
            
            plugin_path = self.registry.get_plugin_path(plugin_id)
            if not plugin_path or not plugin_path.exists():
                return Either.left(PluginError.plugin_not_found(plugin_id))
            
            # Load plugin
            load_result = await self.loader.load_plugin_from_path(plugin_path, metadata)
            if load_result.is_left():
                return load_result
            
            plugin_instance = load_result.get_right()
            
            # Initialize plugin
            init_result = await plugin_instance.initialize(self.api_bridge)
            if init_result.is_left():
                return init_result
            
            # Store loaded plugin
            self.loaded_plugins[plugin_id] = plugin_instance
            
            logger.info(f"Plugin loaded successfully: {plugin_id}")
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError.initialization_failed(f"Loading failed: {str(e)}"))
    
    async def activate_plugin(self, plugin_id: PluginId) -> Either[PluginError, None]:
        """Activate plugin and register its actions and hooks."""
        try:
            # Ensure plugin is loaded
            if plugin_id not in self.loaded_plugins:
                load_result = await self.load_plugin(plugin_id)
                if load_result.is_left():
                    return load_result
            
            plugin = self.loaded_plugins[plugin_id]
            
            # Activate plugin
            activate_result = await plugin.activate()
            if activate_result.is_left():
                return activate_result
            
            # Register custom actions
            custom_actions = await plugin.get_custom_actions()
            for action in custom_actions:
                self.custom_actions[action.action_id] = action
            
            # Register hooks
            hooks = await plugin.get_hooks()
            for hook in hooks:
                if hook.event_type not in self.plugin_hooks:
                    self.plugin_hooks[hook.event_type] = []
                self.plugin_hooks[hook.event_type].append(hook)
            
            logger.info(f"Plugin activated: {plugin_id} with {len(custom_actions)} actions and {len(hooks)} hooks")
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError.activation_failed(f"Activation failed: {str(e)}"))
    
    async def execute_custom_action(self, action_id: ActionId, parameters: Dict[str, Any]) -> Either[PluginError, Any]:
        """Execute custom action with security validation."""
        try:
            if action_id not in self.custom_actions:
                return Either.left(PluginError(f"Action not found: {action_id}", "ACTION_NOT_FOUND"))
            
            action = self.custom_actions[action_id]
            
            # Get plugin configuration for security profile
            config = self.plugin_configurations.get(action.plugin_id)
            if not config:
                return Either.left(PluginError(f"Plugin configuration not found: {action.plugin_id}", "CONFIG_NOT_FOUND"))
            
            # Execute in security sandbox
            sandbox = PluginSandbox(config.security_profile)
            return await sandbox.execute_action(action, parameters)
            
        except Exception as e:
            return Either.left(PluginError.execution_error(str(e)))
    
    async def trigger_hooks(self, event_type: str, event_data: Dict[str, Any]) -> List[Either[PluginError, Any]]:
        """Trigger all hooks for specific event type."""
        results = []
        
        hooks = self.plugin_hooks.get(event_type, [])
        for hook in hooks:
            try:
                result = await hook.execute(event_data)
                results.append(result)
            except Exception as e:
                error = PluginError.hook_execution_failed(hook.hook_id, str(e))
                results.append(Either.left(error))
        
        return results
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins with their status."""
        plugins = []
        
        for metadata in self.registry.list_plugins():
            plugin_info = {
                "id": metadata.identifier,
                "name": metadata.name,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "type": metadata.plugin_type.value,
                "status": "loaded" if metadata.identifier in self.loaded_plugins else "installed",
                "permissions": {
                    "network_access": metadata.permissions.network_access,
                    "file_system_access": metadata.permissions.file_system_access,
                    "system_integration": metadata.permissions.system_integration
                }
            }
            plugins.append(plugin_info)
        
        return plugins
    
    def get_custom_actions(self) -> List[Dict[str, Any]]:
        """Get list of all available custom actions."""
        actions = []
        
        for action in self.custom_actions.values():
            action_info = {
                "id": action.action_id,
                "name": action.name,
                "description": action.description,
                "plugin_id": action.plugin_id,
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.param_type.__name__,
                        "description": param.description,
                        "required": param.required,
                        "default_value": param.default_value
                    }
                    for param in action.parameters
                ]
            }
            actions.append(action_info)
        
        return actions
    
    async def _extract_plugin_metadata(self, plugin_path: Path) -> Either[PluginError, PluginMetadata]:
        """Extract plugin metadata from plugin source."""
        try:
            # Look for manifest file
            manifest_path = None
            
            if plugin_path.is_dir():
                manifest_path = plugin_path / "manifest.json"
            elif plugin_path.suffix == '.zip':
                with zipfile.ZipFile(plugin_path, 'r') as zip_file:
                    if "manifest.json" in zip_file.namelist():
                        with tempfile.TemporaryDirectory() as temp_dir:
                            zip_file.extract("manifest.json", temp_dir)
                            manifest_path = Path(temp_dir) / "manifest.json"
            
            if not manifest_path or not manifest_path.exists():
                return Either.left(PluginError.invalid_plugin_format("manifest.json not found"))
            
            # Parse manifest
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            # Convert to PluginMetadata
            return Either.right(self.registry._deserialize_metadata(manifest_data))
            
        except Exception as e:
            return Either.left(PluginError.invalid_plugin_format(f"Failed to parse manifest: {str(e)}"))
    
    async def _validate_dependencies(self, dependencies: List) -> Either[PluginError, None]:
        """Validate plugin dependencies are satisfied."""
        try:
            for dependency in dependencies:
                if not dependency.optional:
                    # Check if required dependency is installed
                    dep_metadata = self.registry.get_plugin_metadata(dependency.plugin_id)
                    if not dep_metadata:
                        return Either.left(PluginError(
                            f"Required dependency not found: {dependency.plugin_id}",
                            "DEPENDENCY_NOT_FOUND"
                        ))
                    
                    # Check version compatibility
                    if not dependency.is_satisfied_by(dep_metadata.version):
                        return Either.left(PluginError(
                            f"Dependency version mismatch: {dependency.plugin_id} requires {dependency.version_requirement}",
                            "DEPENDENCY_VERSION_MISMATCH"
                        ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError(f"Dependency validation failed: {str(e)}", "DEPENDENCY_ERROR"))