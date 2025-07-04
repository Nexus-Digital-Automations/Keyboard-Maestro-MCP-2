"""
Plugin ecosystem MCP tools for custom action creation and plugin management.

This module provides comprehensive plugin ecosystem capabilities through
MCP tools, enabling plugin installation, management, and custom action execution.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

from ...core.plugin_architecture import (
    PluginId, ActionId, PluginType, PluginStatus, SecurityProfile,
    PluginMetadata, PluginConfiguration, PluginError
)
from ...core.either import Either
from ...core.errors import create_error_context
from ...plugins.plugin_manager import PluginManager
from ...plugins.security_sandbox import PluginSecurityManager
from ...plugins.api_bridge import PluginAPIBridge

logger = logging.getLogger(__name__)


class PluginEcosystemTools:
    """Comprehensive plugin ecosystem management tools."""
    
    def __init__(self):
        self.plugin_manager = PluginManager()
        self.security_manager = PluginSecurityManager()
        self.api_bridge = PluginAPIBridge()
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure plugin manager is initialized."""
        if not self._initialized:
            init_result = await self.plugin_manager.initialize()
            if init_result.is_left():
                raise Exception(f"Plugin manager initialization failed: {init_result.get_left()}")
            self._initialized = True
    
    async def km_plugin_ecosystem(
        self,
        operation: str,
        plugin_identifier: Optional[str] = None,
        plugin_source: Optional[str] = None,
        action_name: Optional[str] = None,
        parameters: Optional[Dict] = None,
        plugin_config: Optional[Dict] = None,
        security_profile: str = "standard",
        api_version: str = "1.0",
        auto_update: bool = False,
        dependency_resolution: bool = True,
        validation_level: str = "strict",
        timeout: int = 60,
        ctx = None
    ) -> Dict[str, Any]:
        """
        Comprehensive plugin ecosystem management tool.
        
        Operations:
        - install: Install plugin from source
        - uninstall: Remove plugin and cleanup
        - list: List all plugins with status
        - activate: Activate plugin and register actions
        - deactivate: Deactivate plugin
        - execute: Execute custom action
        - configure: Update plugin configuration
        - status: Get plugin status and health
        - actions: List available custom actions
        - marketplace: Plugin marketplace operations
        
        Args:
            operation: Operation to perform
            plugin_identifier: Plugin ID for operations
            plugin_source: Plugin source (file, URL, marketplace)
            action_name: Custom action to execute
            parameters: Action parameters or operation parameters
            plugin_config: Plugin configuration settings
            security_profile: Security profile (none|standard|strict|sandbox)
            api_version: Plugin API version
            auto_update: Enable automatic updates
            dependency_resolution: Resolve plugin dependencies
            validation_level: Validation level (none|basic|standard|strict)
            timeout: Operation timeout in seconds
            ctx: Execution context
            
        Returns:
            Operation result with status and data
        """
        try:
            await self._ensure_initialized()
            
            # Validate operation
            valid_operations = {
                'install', 'uninstall', 'list', 'activate', 'deactivate',
                'execute', 'configure', 'status', 'actions', 'marketplace'
            }
            
            if operation not in valid_operations:
                return {
                    'success': False,
                    'error': f"Invalid operation: {operation}",
                    'valid_operations': list(valid_operations)
                }
            
            # Route to appropriate handler
            if operation == 'install':
                return await self._handle_install(
                    plugin_source, plugin_config, security_profile,
                    auto_update, dependency_resolution, validation_level, timeout
                )
            elif operation == 'uninstall':
                return await self._handle_uninstall(plugin_identifier, timeout)
            elif operation == 'list':
                return await self._handle_list(parameters or {})
            elif operation == 'activate':
                return await self._handle_activate(plugin_identifier, timeout)
            elif operation == 'deactivate':
                return await self._handle_deactivate(plugin_identifier, timeout)
            elif operation == 'execute':
                return await self._handle_execute(action_name, parameters or {}, timeout)
            elif operation == 'configure':
                return await self._handle_configure(plugin_identifier, plugin_config or {})
            elif operation == 'status':
                return await self._handle_status(plugin_identifier)
            elif operation == 'actions':
                return await self._handle_actions(plugin_identifier)
            elif operation == 'marketplace':
                return await self._handle_marketplace(parameters or {})
            
        except Exception as e:
            logger.error(f"Plugin ecosystem operation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _handle_install(
        self,
        plugin_source: Optional[str],
        plugin_config: Optional[Dict],
        security_profile: str,
        auto_update: bool,
        dependency_resolution: bool,
        validation_level: str,
        timeout: int
    ) -> Dict[str, Any]:
        """Handle plugin installation."""
        try:
            if not plugin_source:
                return {
                    'success': False,
                    'error': "Plugin source is required for installation"
                }
            
            # Parse security profile
            try:
                security_prof = SecurityProfile(security_profile.lower())
            except ValueError:
                return {
                    'success': False,
                    'error': f"Invalid security profile: {security_profile}",
                    'valid_profiles': [p.value for p in SecurityProfile]
                }
            
            # Create plugin configuration
            config = None
            if plugin_config:
                # Extract plugin ID from source for configuration
                # For now, use a placeholder - would be determined from plugin metadata
                config = PluginConfiguration(
                    plugin_id=PluginId("temp_id"),
                    settings=plugin_config,
                    auto_update=auto_update,
                    security_profile=security_prof
                )
            
            # Install plugin
            install_result = await asyncio.wait_for(
                self.plugin_manager.install_plugin(plugin_source, config),
                timeout=timeout
            )
            
            if install_result.is_left():
                error = install_result.get_left()
                return {
                    'success': False,
                    'error': str(error),
                    'error_code': getattr(error, 'error_code', 'INSTALL_FAILED'),
                    'source': plugin_source
                }
            
            metadata = install_result.get_right()
            
            return {
                'success': True,
                'operation': 'install',
                'plugin': {
                    'id': metadata.identifier,
                    'name': metadata.name,
                    'version': metadata.version,
                    'description': metadata.description,
                    'author': metadata.author,
                    'type': metadata.plugin_type.value,
                    'api_version': metadata.api_version.value,
                    'permissions': {
                        'network_access': metadata.permissions.network_access,
                        'file_system_access': metadata.permissions.file_system_access,
                        'system_integration': metadata.permissions.system_integration
                    }
                },
                'security_profile': security_profile,
                'auto_update': auto_update,
                'timestamp': datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Installation timeout after {timeout} seconds",
                'source': plugin_source
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Installation failed: {e}",
                'source': plugin_source
            }
    
    async def _handle_uninstall(
        self,
        plugin_identifier: Optional[str],
        timeout: int
    ) -> Dict[str, Any]:
        """Handle plugin uninstallation."""
        try:
            if not plugin_identifier:
                return {
                    'success': False,
                    'error': "Plugin identifier is required for uninstallation"
                }
            
            plugin_id = PluginId(plugin_identifier)
            
            # Get plugin info before uninstalling
            plugin_info = None
            plugins = self.plugin_manager.list_plugins()
            for plugin in plugins:
                if plugin['id'] == plugin_identifier:
                    plugin_info = plugin
                    break
            
            # Uninstall plugin
            uninstall_result = await asyncio.wait_for(
                self.plugin_manager.uninstall_plugin(plugin_id),
                timeout=timeout
            )
            
            if uninstall_result.is_left():
                error = uninstall_result.get_left()
                return {
                    'success': False,
                    'error': str(error),
                    'error_code': getattr(error, 'error_code', 'UNINSTALL_FAILED'),
                    'plugin_id': plugin_identifier
                }
            
            return {
                'success': True,
                'operation': 'uninstall',
                'plugin_id': plugin_identifier,
                'plugin_info': plugin_info,
                'timestamp': datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Uninstallation timeout after {timeout} seconds",
                'plugin_id': plugin_identifier
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Uninstallation failed: {e}",
                'plugin_id': plugin_identifier
            }
    
    async def _handle_list(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle plugin listing."""
        try:
            # Get filter parameters
            status_filter = parameters.get('status_filter')
            type_filter = parameters.get('type_filter')
            include_actions = parameters.get('include_actions', False)
            include_permissions = parameters.get('include_permissions', True)
            
            # Get all plugins
            plugins = self.plugin_manager.list_plugins()
            
            # Apply filters
            filtered_plugins = []
            for plugin in plugins:
                # Status filter
                if status_filter and plugin.get('status') != status_filter:
                    continue
                
                # Type filter
                if type_filter and plugin.get('type') != type_filter:
                    continue
                
                # Enhance plugin info
                enhanced_plugin = plugin.copy()
                
                # Add custom actions if requested
                if include_actions:
                    custom_actions = self.plugin_manager.get_custom_actions()
                    plugin_actions = [
                        action for action in custom_actions
                        if action['plugin_id'] == plugin['id']
                    ]
                    enhanced_plugin['custom_actions'] = plugin_actions
                
                # Remove permissions if not requested
                if not include_permissions:
                    enhanced_plugin.pop('permissions', None)
                
                filtered_plugins.append(enhanced_plugin)
            
            return {
                'success': True,
                'operation': 'list',
                'plugins': filtered_plugins,
                'total_count': len(filtered_plugins),
                'filters_applied': {
                    'status_filter': status_filter,
                    'type_filter': type_filter,
                    'include_actions': include_actions,
                    'include_permissions': include_permissions
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Plugin listing failed: {e}",
                'operation': 'list'
            }
    
    async def _handle_activate(
        self,
        plugin_identifier: Optional[str],
        timeout: int
    ) -> Dict[str, Any]:
        """Handle plugin activation."""
        try:
            if not plugin_identifier:
                return {
                    'success': False,
                    'error': "Plugin identifier is required for activation"
                }
            
            plugin_id = PluginId(plugin_identifier)
            
            # Activate plugin
            activate_result = await asyncio.wait_for(
                self.plugin_manager.activate_plugin(plugin_id),
                timeout=timeout
            )
            
            if activate_result.is_left():
                error = activate_result.get_left()
                return {
                    'success': False,
                    'error': str(error),
                    'error_code': getattr(error, 'error_code', 'ACTIVATION_FAILED'),
                    'plugin_id': plugin_identifier
                }
            
            # Get updated plugin info
            plugins = self.plugin_manager.list_plugins()
            plugin_info = None
            for plugin in plugins:
                if plugin['id'] == plugin_identifier:
                    plugin_info = plugin
                    break
            
            # Get registered actions
            custom_actions = self.plugin_manager.get_custom_actions()
            plugin_actions = [
                action for action in custom_actions
                if action['plugin_id'] == plugin_identifier
            ]
            
            return {
                'success': True,
                'operation': 'activate',
                'plugin_id': plugin_identifier,
                'plugin_info': plugin_info,
                'registered_actions': plugin_actions,
                'action_count': len(plugin_actions),
                'timestamp': datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Activation timeout after {timeout} seconds",
                'plugin_id': plugin_identifier
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Activation failed: {e}",
                'plugin_id': plugin_identifier
            }
    
    async def _handle_deactivate(
        self,
        plugin_identifier: Optional[str],
        timeout: int
    ) -> Dict[str, Any]:
        """Handle plugin deactivation."""
        try:
            if not plugin_identifier:
                return {
                    'success': False,
                    'error': "Plugin identifier is required for deactivation"
                }
            
            plugin_id = PluginId(plugin_identifier)
            
            # Get actions before deactivation
            custom_actions = self.plugin_manager.get_custom_actions()
            plugin_actions = [
                action for action in custom_actions
                if action['plugin_id'] == plugin_identifier
            ]
            
            # Deactivate plugin
            deactivate_result = await asyncio.wait_for(
                self.plugin_manager.deactivate_plugin(plugin_id),
                timeout=timeout
            )
            
            if deactivate_result.is_left():
                error = deactivate_result.get_left()
                return {
                    'success': False,
                    'error': str(error),
                    'error_code': getattr(error, 'error_code', 'DEACTIVATION_FAILED'),
                    'plugin_id': plugin_identifier
                }
            
            return {
                'success': True,
                'operation': 'deactivate',
                'plugin_id': plugin_identifier,
                'unregistered_actions': plugin_actions,
                'action_count': len(plugin_actions),
                'timestamp': datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Deactivation timeout after {timeout} seconds",
                'plugin_id': plugin_identifier
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Deactivation failed: {e}",
                'plugin_id': plugin_identifier
            }
    
    async def _handle_execute(
        self,
        action_name: Optional[str],
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Handle custom action execution."""
        try:
            if not action_name:
                return {
                    'success': False,
                    'error': "Action name is required for execution"
                }
            
            action_id = ActionId(action_name)
            
            # Execute custom action
            execute_result = await asyncio.wait_for(
                self.plugin_manager.execute_custom_action(action_id, parameters),
                timeout=timeout
            )
            
            if execute_result.is_left():
                error = execute_result.get_left()
                return {
                    'success': False,
                    'error': str(error),
                    'error_code': getattr(error, 'error_code', 'EXECUTION_FAILED'),
                    'action_name': action_name,
                    'parameters': parameters
                }
            
            result = execute_result.get_right()
            
            return {
                'success': True,
                'operation': 'execute',
                'action_name': action_name,
                'parameters': parameters,
                'result': result,
                'result_type': type(result).__name__,
                'timestamp': datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Execution timeout after {timeout} seconds",
                'action_name': action_name
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Execution failed: {e}",
                'action_name': action_name,
                'parameters': parameters
            }
    
    async def _handle_configure(
        self,
        plugin_identifier: Optional[str],
        plugin_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle plugin configuration update."""
        try:
            if not plugin_identifier:
                return {
                    'success': False,
                    'error': "Plugin identifier is required for configuration"
                }
            
            plugin_id = PluginId(plugin_identifier)
            
            # Get current configuration
            current_config = self.plugin_manager.plugin_configurations.get(plugin_id)
            if not current_config:
                return {
                    'success': False,
                    'error': f"Plugin configuration not found: {plugin_identifier}"
                }
            
            # Update configuration
            updated_config = current_config
            for key, value in plugin_config.items():
                if key == 'settings':
                    # Merge settings
                    new_settings = current_config.settings.copy()
                    new_settings.update(value if isinstance(value, dict) else {})
                    updated_config = updated_config.update_setting('_all_settings', new_settings)
                elif key == 'security_profile':
                    try:
                        security_prof = SecurityProfile(value.lower())
                        updated_config = PluginConfiguration(
                            plugin_id=current_config.plugin_id,
                            settings=current_config.settings,
                            enabled=current_config.enabled,
                            auto_update=current_config.auto_update,
                            security_profile=security_prof,
                            resource_limits=current_config.resource_limits
                        )
                    except ValueError:
                        return {
                            'success': False,
                            'error': f"Invalid security profile: {value}",
                            'valid_profiles': [p.value for p in SecurityProfile]
                        }
                elif hasattr(current_config, key):
                    # Update other configuration attributes
                    setattr(updated_config, key, value)
            
            # Store updated configuration
            self.plugin_manager.plugin_configurations[plugin_id] = updated_config
            
            return {
                'success': True,
                'operation': 'configure',
                'plugin_id': plugin_identifier,
                'updated_config': {
                    'settings': updated_config.settings,
                    'enabled': updated_config.enabled,
                    'auto_update': updated_config.auto_update,
                    'security_profile': updated_config.security_profile.value
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Configuration update failed: {e}",
                'plugin_id': plugin_identifier
            }
    
    async def _handle_status(self, plugin_identifier: Optional[str]) -> Dict[str, Any]:
        """Handle plugin status inquiry."""
        try:
            if plugin_identifier:
                # Single plugin status
                plugin_id = PluginId(plugin_identifier)
                
                # Get plugin status
                status = self.plugin_manager.get_plugin_status(plugin_id)
                if status is None:
                    return {
                        'success': False,
                        'error': f"Plugin not found: {plugin_identifier}"
                    }
                
                # Get plugin info
                plugins = self.plugin_manager.list_plugins()
                plugin_info = None
                for plugin in plugins:
                    if plugin['id'] == plugin_identifier:
                        plugin_info = plugin
                        break
                
                # Get API usage stats if available
                usage_stats = {}
                if hasattr(self.api_bridge, 'get_plugin_usage_stats'):
                    usage_stats = self.api_bridge.get_plugin_usage_stats(plugin_id)
                
                # Get custom actions
                custom_actions = self.plugin_manager.get_custom_actions()
                plugin_actions = [
                    action for action in custom_actions
                    if action['plugin_id'] == plugin_identifier
                ]
                
                return {
                    'success': True,
                    'operation': 'status',
                    'plugin_id': plugin_identifier,
                    'status': status.value if status else 'unknown',
                    'plugin_info': plugin_info,
                    'custom_actions': plugin_actions,
                    'action_count': len(plugin_actions),
                    'usage_stats': usage_stats,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # System-wide status
                plugins = self.plugin_manager.list_plugins()
                custom_actions = self.plugin_manager.get_custom_actions()
                
                status_summary = {
                    'installed': 0,
                    'loaded': 0,
                    'active': 0,
                    'error': 0
                }
                
                for plugin in plugins:
                    status = plugin.get('status', 'unknown')
                    if status == 'installed':
                        status_summary['installed'] += 1
                    elif status == 'loaded':
                        status_summary['loaded'] += 1
                        status_summary['installed'] += 1
                    elif status == 'active':
                        status_summary['active'] += 1
                        status_summary['loaded'] += 1
                        status_summary['installed'] += 1
                    elif status == 'error':
                        status_summary['error'] += 1
                
                return {
                    'success': True,
                    'operation': 'status',
                    'system_status': status_summary,
                    'total_plugins': len(plugins),
                    'total_actions': len(custom_actions),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Status inquiry failed: {e}",
                'plugin_id': plugin_identifier
            }
    
    async def _handle_actions(self, plugin_identifier: Optional[str]) -> Dict[str, Any]:
        """Handle custom actions listing."""
        try:
            custom_actions = self.plugin_manager.get_custom_actions()
            
            if plugin_identifier:
                # Filter actions for specific plugin
                plugin_actions = [
                    action for action in custom_actions
                    if action['plugin_id'] == plugin_identifier
                ]
                
                return {
                    'success': True,
                    'operation': 'actions',
                    'plugin_id': plugin_identifier,
                    'actions': plugin_actions,
                    'action_count': len(plugin_actions),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # All actions grouped by plugin
                actions_by_plugin = {}
                for action in custom_actions:
                    plugin_id = action['plugin_id']
                    if plugin_id not in actions_by_plugin:
                        actions_by_plugin[plugin_id] = []
                    actions_by_plugin[plugin_id].append(action)
                
                return {
                    'success': True,
                    'operation': 'actions',
                    'actions_by_plugin': actions_by_plugin,
                    'total_actions': len(custom_actions),
                    'plugin_count': len(actions_by_plugin),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Actions listing failed: {e}",
                'plugin_id': plugin_identifier
            }
    
    async def _handle_marketplace(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle plugin marketplace operations."""
        try:
            marketplace_operation = parameters.get('marketplace_operation', 'search')
            
            if marketplace_operation == 'search':
                # Plugin marketplace search (placeholder implementation)
                query = parameters.get('query', '')
                category = parameters.get('category')
                
                # This would connect to a real plugin marketplace
                marketplace_results = [
                    {
                        'id': 'example-plugin-1',
                        'name': 'Example Plugin',
                        'version': '1.0.0',
                        'description': 'An example plugin for demonstration',
                        'author': 'Plugin Author',
                        'category': 'utility',
                        'download_url': 'https://example.com/plugin1.zip',
                        'rating': 4.5,
                        'downloads': 1000
                    }
                ]
                
                return {
                    'success': True,
                    'operation': 'marketplace',
                    'marketplace_operation': 'search',
                    'query': query,
                    'category': category,
                    'results': marketplace_results,
                    'result_count': len(marketplace_results),
                    'timestamp': datetime.now().isoformat()
                }
            
            elif marketplace_operation == 'info':
                # Get plugin info from marketplace
                plugin_id = parameters.get('plugin_id')
                if not plugin_id:
                    return {
                        'success': False,
                        'error': "Plugin ID required for marketplace info"
                    }
                
                # This would fetch from real marketplace
                plugin_info = {
                    'id': plugin_id,
                    'name': 'Example Plugin',
                    'version': '1.0.0',
                    'description': 'Detailed plugin description',
                    'author': 'Plugin Author',
                    'category': 'utility',
                    'permissions': ['basic_execution'],
                    'download_url': f'https://example.com/{plugin_id}.zip',
                    'rating': 4.5,
                    'downloads': 1000,
                    'updated_at': '2024-01-01T00:00:00Z'
                }
                
                return {
                    'success': True,
                    'operation': 'marketplace',
                    'marketplace_operation': 'info',
                    'plugin_info': plugin_info,
                    'timestamp': datetime.now().isoformat()
                }
            
            else:
                return {
                    'success': False,
                    'error': f"Unknown marketplace operation: {marketplace_operation}",
                    'valid_operations': ['search', 'info']
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Marketplace operation failed: {e}",
                'operation': 'marketplace'
            }