"""
MCP tools for plugin management and ecosystem operations.

These tools provide the interface for managing the plugin ecosystem
through the Model Context Protocol (MCP) server.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..plugins.plugin_manager import PluginManager
from ..plugins.marketplace import PluginMarketplace, SearchQuery, PluginCategory
from ..plugins.security_sandbox import PluginSecurityManager
from ..core.plugin_architecture import (
    PluginId, PluginError, SecurityProfile, PluginPermissions, PermissionId
)
from ..core.either import Either
from ..core.errors import create_error_context

logger = logging.getLogger(__name__)

# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None
_marketplace: Optional[PluginMarketplace] = None


async def get_plugin_manager() -> PluginManager:
    """Get or create global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
        await _plugin_manager.initialize()
    return _plugin_manager


async def get_marketplace() -> PluginMarketplace:
    """Get or create global marketplace instance."""
    global _marketplace
    if _marketplace is None:
        _marketplace = PluginMarketplace()
        await _marketplace.initialize()
    return _marketplace


async def km_plugin_manager(
    operation: str,
    plugin_id: Optional[str] = None,
    plugin_path: Optional[str] = None,
    configuration: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Comprehensive plugin management operations.
    
    Operations:
    - list: List all installed plugins
    - install: Install plugin from path
    - load: Load plugin into memory
    - activate: Activate plugin
    - deactivate: Deactivate plugin
    - uninstall: Remove plugin
    - status: Get plugin status
    - metrics: Get plugin metrics
    """
    try:
        manager = await get_plugin_manager()
        
        if operation == "list":
            plugins = manager.list_plugins()
            return {
                "success": True,
                "data": {
                    "plugins": plugins,
                    "total_count": len(plugins)
                },
                "metadata": {
                    "operation": "list",
                    "timestamp": "now"
                }
            }
        
        elif operation == "install":
            if not plugin_path:
                raise ValueError("plugin_path required for install operation")
            
            path = Path(plugin_path)
            if not path.exists():
                raise ValueError(f"Plugin path not found: {plugin_path}")
            
            # Create configuration if provided
            plugin_config = None
            if configuration:
                plugin_config = await _create_plugin_configuration(configuration)
            
            result = await manager.install_plugin(str(path), plugin_config)
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.error_code,
                        "message": error.message,
                        "details": "Plugin installation failed"
                    }
                }
            
            metadata = result.get_right()
            return {
                "success": True,
                "data": {
                    "plugin_id": metadata.identifier,
                    "name": metadata.name,
                    "version": metadata.version,
                    "status": "installed"
                },
                "metadata": {
                    "operation": "install",
                    "timestamp": "now"
                }
            }
        
        elif operation == "load":
            if not plugin_id:
                raise ValueError("plugin_id required for load operation")
            
            result = await manager.load_plugin(PluginId(plugin_id))
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.error_code,
                        "message": error.message
                    }
                }
            
            return {
                "success": True,
                "data": {
                    "plugin_id": plugin_id,
                    "status": "loaded"
                }
            }
        
        elif operation == "activate":
            if not plugin_id:
                raise ValueError("plugin_id required for activate operation")
            
            result = await manager.activate_plugin(PluginId(plugin_id))
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.error_code,
                        "message": error.message
                    }
                }
            
            # Get custom actions registered by the plugin
            actions = manager.get_custom_actions()
            plugin_actions = [a for a in actions if a.get("plugin_id") == plugin_id]
            
            return {
                "success": True,
                "data": {
                    "plugin_id": plugin_id,
                    "status": "active",
                    "custom_actions": len(plugin_actions),
                    "actions": plugin_actions
                }
            }
        
        elif operation == "status":
            if not plugin_id:
                raise ValueError("plugin_id required for status operation")
            
            plugins = manager.list_plugins()
            plugin_info = next((p for p in plugins if p["id"] == plugin_id), None)
            
            if not plugin_info:
                return {
                    "success": False,
                    "error": {
                        "code": "PLUGIN_NOT_FOUND",
                        "message": f"Plugin not found: {plugin_id}"
                    }
                }
            
            # Get API metrics if available
            metrics = {}
            if manager.api_bridge:
                metrics = manager.api_bridge.get_plugin_metrics(PluginId(plugin_id))
            
            return {
                "success": True,
                "data": {
                    "plugin_info": plugin_info,
                    "metrics": metrics
                }
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    except Exception as e:
        logger.error(f"Plugin manager operation failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": {
                "code": "OPERATION_FAILED",
                "message": str(e),
                "operation": operation
            }
        }


async def km_plugin_marketplace(
    operation: str,
    query: Optional[str] = None,
    category: Optional[str] = None,
    plugin_id: Optional[str] = None,
    limit: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Plugin marketplace operations for discovery and installation.
    
    Operations:
    - search: Search for plugins
    - details: Get plugin details
    - featured: Get featured plugins
    - install: Install from marketplace
    - updates: Check for updates
    """
    try:
        marketplace = await get_marketplace()
        
        if operation == "search":
            # Build search query
            search_query = SearchQuery(
                query=query,
                category=PluginCategory(category) if category else None,
                limit=limit
            )
            
            result = await marketplace.search_plugins(search_query)
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.error_code,
                        "message": error.message
                    }
                }
            
            plugins = result.get_right()
            return {
                "success": True,
                "data": {
                    "plugins": [_serialize_marketplace_entry(entry) for entry in plugins],
                    "total_results": len(plugins),
                    "query": query,
                    "category": category
                }
            }
        
        elif operation == "details":
            if not plugin_id:
                raise ValueError("plugin_id required for details operation")
            
            result = await marketplace.get_plugin_details(PluginId(plugin_id))
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.error_code,
                        "message": error.message
                    }
                }
            
            entry = result.get_right()
            return {
                "success": True,
                "data": _serialize_marketplace_entry(entry)
            }
        
        elif operation == "featured":
            result = await marketplace.get_featured_plugins()
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.error_code,
                        "message": error.message
                    }
                }
            
            featured = result.get_right()
            return {
                "success": True,
                "data": {
                    "featured_plugins": [_serialize_marketplace_entry(entry) for entry in featured],
                    "count": len(featured)
                }
            }
        
        elif operation == "install":
            if not plugin_id:
                raise ValueError("plugin_id required for install operation")
            
            # Get plugin manager for installation
            manager = await get_plugin_manager()
            
            # Install from marketplace
            result = await marketplace.install_plugin(PluginId(plugin_id), manager.plugins_directory)
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.error_code,
                        "message": error.message
                    }
                }
            
            plugin_path = result.get_right()
            return {
                "success": True,
                "data": {
                    "plugin_id": plugin_id,
                    "install_path": str(plugin_path),
                    "status": "installed"
                }
            }
        
        else:
            raise ValueError(f"Unknown marketplace operation: {operation}")
    
    except Exception as e:
        logger.error(f"Marketplace operation failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": {
                "code": "MARKETPLACE_ERROR",
                "message": str(e),
                "operation": operation
            }
        }


async def km_plugin_security(
    operation: str,
    plugin_path: Optional[str] = None,
    plugin_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Plugin security operations and validation.
    
    Operations:
    - scan: Security scan of plugin
    - approve: Approve plugin for installation
    - block: Block plugin from execution
    - report: Get security report
    """
    try:
        security_manager = PluginSecurityManager()
        
        if operation == "scan":
            if not plugin_path:
                raise ValueError("plugin_path required for scan operation")
            
            path = Path(plugin_path)
            if not path.exists():
                raise ValueError(f"Plugin path not found: {plugin_path}")
            
            result = security_manager.get_security_report(path)
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.error_code,
                        "message": error.message
                    }
                }
            
            report = result.get_right()
            return {
                "success": True,
                "data": {
                    "security_report": report,
                    "plugin_path": plugin_path
                }
            }
        
        elif operation == "approve":
            if not plugin_id:
                raise ValueError("plugin_id required for approve operation")
            
            security_manager.approve_plugin(plugin_id)
            return {
                "success": True,
                "data": {
                    "plugin_id": plugin_id,
                    "status": "approved"
                }
            }
        
        elif operation == "block":
            if not plugin_id:
                raise ValueError("plugin_id required for block operation")
            
            security_manager.block_plugin(plugin_id)
            return {
                "success": True,
                "data": {
                    "plugin_id": plugin_id,
                    "status": "blocked"
                }
            }
        
        else:
            raise ValueError(f"Unknown security operation: {operation}")
    
    except Exception as e:
        logger.error(f"Security operation failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": {
                "code": "SECURITY_ERROR",
                "message": str(e),
                "operation": operation
            }
        }


async def km_plugin_actions(
    operation: str,
    action_id: Optional[str] = None,
    plugin_id: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute and manage custom plugin actions.
    
    Operations:
    - list: List available custom actions
    - execute: Execute custom action
    - info: Get action information
    """
    try:
        manager = await get_plugin_manager()
        
        if operation == "list":
            actions = manager.get_custom_actions()
            
            # Filter by plugin if specified
            if plugin_id:
                actions = [a for a in actions if a.get("plugin_id") == plugin_id]
            
            return {
                "success": True,
                "data": {
                    "custom_actions": actions,
                    "total_count": len(actions)
                }
            }
        
        elif operation == "execute":
            if not action_id:
                raise ValueError("action_id required for execute operation")
            
            if parameters is None:
                parameters = {}
            
            result = await manager.execute_custom_action(action_id, parameters)
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.error_code,
                        "message": error.message
                    }
                }
            
            action_result = result.get_right()
            return {
                "success": True,
                "data": {
                    "action_id": action_id,
                    "result": action_result,
                    "parameters": parameters
                }
            }
        
        elif operation == "info":
            if not action_id:
                raise ValueError("action_id required for info operation")
            
            actions = manager.get_custom_actions()
            action_info = next((a for a in actions if a["id"] == action_id), None)
            
            if not action_info:
                return {
                    "success": False,
                    "error": {
                        "code": "ACTION_NOT_FOUND",
                        "message": f"Custom action not found: {action_id}"
                    }
                }
            
            return {
                "success": True,
                "data": {
                    "action_info": action_info
                }
            }
        
        else:
            raise ValueError(f"Unknown action operation: {operation}")
    
    except Exception as e:
        logger.error(f"Plugin action operation failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": {
                "code": "ACTION_ERROR",
                "message": str(e),
                "operation": operation
            }
        }


async def km_plugin_development(
    operation: str,
    plugin_name: Optional[str] = None,
    template_type: str = "utility",
    target_directory: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Plugin development tools and templates.
    
    Operations:
    - create_template: Create plugin template
    - validate: Validate plugin structure
    - package: Package plugin for distribution
    - test: Test plugin functionality
    """
    try:
        if operation == "create_template":
            if not plugin_name or not target_directory:
                raise ValueError("plugin_name and target_directory required")
            
            target_path = Path(target_directory)
            plugin_path = target_path / plugin_name
            
            # Create plugin template
            await _create_plugin_template(plugin_name, template_type, plugin_path)
            
            return {
                "success": True,
                "data": {
                    "plugin_name": plugin_name,
                    "template_type": template_type,
                    "plugin_path": str(plugin_path),
                    "files_created": [
                        "main.py",
                        "manifest.json",
                        "README.md",
                        "__init__.py"
                    ]
                }
            }
        
        elif operation == "validate":
            if not target_directory:
                raise ValueError("target_directory required for validate")
            
            plugin_path = Path(target_directory)
            
            # Validate plugin structure
            from ..plugins.plugin_sdk import validate_plugin_structure
            result = validate_plugin_structure(plugin_path)
            
            if result.is_left():
                error_msg = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": "VALIDATION_FAILED",
                        "message": error_msg
                    }
                }
            
            return {
                "success": True,
                "data": {
                    "plugin_path": str(plugin_path),
                    "validation_status": "passed",
                    "structure": "valid"
                }
            }
        
        else:
            raise ValueError(f"Unknown development operation: {operation}")
    
    except Exception as e:
        logger.error(f"Plugin development operation failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": {
                "code": "DEVELOPMENT_ERROR",
                "message": str(e),
                "operation": operation
            }
        }


# Helper functions

async def _create_plugin_configuration(config_data: Dict[str, Any]):
    """Create plugin configuration from data."""
    from ..core.plugin_architecture import PluginConfiguration
    
    plugin_id = PluginId(config_data.get("plugin_id", ""))
    settings = config_data.get("settings", {})
    security_profile = SecurityProfile(config_data.get("security_profile", "standard"))
    
    return PluginConfiguration(
        plugin_id=plugin_id,
        settings=settings,
        security_profile=security_profile,
        enabled=config_data.get("enabled", True),
        auto_update=config_data.get("auto_update", False)
    )


def _serialize_marketplace_entry(entry) -> Dict[str, Any]:
    """Serialize marketplace entry for JSON response."""
    return {
        "plugin_id": entry.metadata.identifier,
        "name": entry.metadata.name,
        "version": entry.metadata.version,
        "description": entry.metadata.description,
        "author": entry.metadata.author,
        "category": entry.category.value,
        "status": entry.status.value,
        "rating": entry.get_average_rating(),
        "price": entry.price,
        "is_free": entry.is_free(),
        "featured": entry.featured,
        "verified_developer": entry.verified_developer,
        "tags": list(entry.tags),
        "download_info": {
            "file_size": entry.download_info.file_size,
            "download_url": entry.download_info.download_url
        }
    }


async def _create_plugin_template(plugin_name: str, template_type: str, target_path: Path):
    """Create plugin template files."""
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Create main.py
    main_content = f'''"""
{plugin_name} plugin for Keyboard Maestro MCP.

This plugin was generated from the {template_type} template.
"""

from src.plugins.plugin_sdk import UtilityPlugin, create_plugin_metadata
from src.core.plugin_architecture import PluginPermissions, PermissionId
from src.core.either import Either

class Plugin(UtilityPlugin):
    """Main plugin class."""
    
    def __init__(self):
        metadata = create_plugin_metadata(
            identifier="{plugin_name.lower().replace(' ', '-')}",
            name="{plugin_name}",
            version="1.0.0",
            description="A {template_type} plugin for automation tasks",
            author="Plugin Developer",
            plugin_type="{template_type}",
            permissions=PluginPermissions(
                permissions={{PermissionId("basic_execution")}},
                network_access=False,
                file_system_access=False
            )
        )
        super().__init__(metadata)
    
    async def on_initialize(self):
        """Initialize the plugin."""
        self.logger.info("Plugin initializing")
        return Either.right(None)
    
    async def on_activate(self):
        """Activate the plugin."""
        self.logger.info("Plugin activating")
        return Either.right(None)
    
    async def on_deactivate(self):
        """Deactivate the plugin."""
        self.logger.info("Plugin deactivating")
        return Either.right(None)
    
    @UtilityPlugin.action("hello", "Say Hello", "Say hello to someone")
    async def hello(self, name: str = "World") -> str:
        """Example action that says hello."""
        return f"Hello, {{name}}!"
'''
    
    with open(target_path / "main.py", 'w') as f:
        f.write(main_content)
    
    # Create manifest.json
    manifest = {
        "identifier": plugin_name.lower().replace(' ', '-'),
        "name": plugin_name,
        "version": "1.0.0",
        "description": f"A {template_type} plugin for automation tasks",
        "author": "Plugin Developer",
        "plugin_type": template_type,
        "api_version": "1.0",
        "entry_point": "main",
        "permissions": {
            "permissions": ["basic_execution"],
            "network_access": False,
            "file_system_access": False,
            "system_integration": False
        }
    }
    
    with open(target_path / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create README.md
    readme_content = f'''# {plugin_name}

A {template_type} plugin for Keyboard Maestro MCP.

## Description

This plugin provides automation capabilities for [describe your plugin's purpose].

## Actions

- **Say Hello**: Example action that greets someone

## Installation

1. Copy this plugin directory to your plugins folder
2. Use the plugin manager to install and activate it

## Usage

After activation, the plugin's custom actions will be available through the MCP interface.

## Development

This plugin was created using the Keyboard Maestro MCP plugin template system.
'''
    
    with open(target_path / "README.md", 'w') as f:
        f.write(readme_content)
    
    # Create __init__.py
    with open(target_path / "__init__.py", 'w') as f:
        f.write(f'"""{plugin_name} plugin."""\n')


# MCP tool metadata for registration

PLUGIN_MANAGEMENT_TOOLS = {
    "km_plugin_manager": {
        "name": "km_plugin_manager",
        "description": "Comprehensive plugin management operations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "install", "load", "activate", "deactivate", "status"],
                    "description": "Operation to perform"
                },
                "plugin_id": {
                    "type": "string",
                    "description": "Plugin identifier for operations"
                },
                "plugin_path": {
                    "type": "string",
                    "description": "Path to plugin for installation"
                },
                "configuration": {
                    "type": "object",
                    "description": "Plugin configuration settings"
                }
            },
            "required": ["operation"]
        }
    },
    "km_plugin_marketplace": {
        "name": "km_plugin_marketplace",
        "description": "Plugin marketplace operations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["search", "details", "featured", "install", "updates"],
                    "description": "Marketplace operation"
                },
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "category": {
                    "type": "string",
                    "description": "Plugin category filter"
                },
                "plugin_id": {
                    "type": "string",
                    "description": "Plugin identifier"
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum results"
                }
            },
            "required": ["operation"]
        }
    },
    "km_plugin_security": {
        "name": "km_plugin_security",
        "description": "Plugin security operations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["scan", "approve", "block", "report"],
                    "description": "Security operation"
                },
                "plugin_path": {
                    "type": "string",
                    "description": "Path to plugin for scanning"
                },
                "plugin_id": {
                    "type": "string",
                    "description": "Plugin identifier"
                }
            },
            "required": ["operation"]
        }
    },
    "km_plugin_actions": {
        "name": "km_plugin_actions",
        "description": "Execute and manage custom plugin actions",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "execute", "info"],
                    "description": "Action operation"
                },
                "action_id": {
                    "type": "string",
                    "description": "Custom action identifier"
                },
                "plugin_id": {
                    "type": "string",
                    "description": "Filter by plugin"
                },
                "parameters": {
                    "type": "object",
                    "description": "Action parameters"
                }
            },
            "required": ["operation"]
        }
    },
    "km_plugin_development": {
        "name": "km_plugin_development",
        "description": "Plugin development tools and templates",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create_template", "validate", "package", "test"],
                    "description": "Development operation"
                },
                "plugin_name": {
                    "type": "string",
                    "description": "Name for new plugin"
                },
                "template_type": {
                    "type": "string",
                    "enum": ["utility", "integration", "automation"],
                    "default": "utility",
                    "description": "Plugin template type"
                },
                "target_directory": {
                    "type": "string",
                    "description": "Target directory for operations"
                }
            },
            "required": ["operation"]
        }
    }
}