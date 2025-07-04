# Plugin Ecosystem Documentation

## Overview

The Keyboard Maestro MCP Plugin Ecosystem provides unlimited extensibility to the existing 38 MCP tools through a comprehensive plugin architecture. This system enables custom action creation, secure plugin execution, marketplace integration, and advanced development frameworks.

## Architecture Components

### Core Plugin System (`plugin_architecture.py`)

**Branded Types for Type Safety:**
- `PluginId`, `ActionId`, `HookId` - Unique identifiers with compile-time validation
- `PluginMetadata` - Comprehensive plugin information with validation
- `CustomAction` - Action definitions with parameter specifications
- `PluginPermissions` - Fine-grained permission system

**Security Profiles:**
- `NONE` - Development only, no restrictions
- `STANDARD` - Basic security validation  
- `STRICT` - Enhanced security with limitations
- `SANDBOX` - Full isolated execution

### Plugin Management (`plugin_manager.py`)

**Core Functions:**
- Plugin installation with dependency resolution
- Lifecycle management (load, activate, deactivate)
- Registry persistence and synchronization
- Custom action execution with security validation

**Key Classes:**
- `PluginRegistry` - Centralized plugin tracking
- `PluginLoader` - Dynamic module loading
- `PluginManager` - Orchestrates all plugin operations

### Security System (`security_sandbox.py`)

**Multi-Layer Security:**
- Static code analysis with AST parsing
- Runtime resource monitoring (CPU, memory, disk, network)
- Dangerous pattern detection
- Security rating system (SAFE, WARNING, DANGEROUS)

**Resource Limits by Profile:**
```python
SecurityProfile.SANDBOX:  # Most restrictive
    memory_mb: 128, cpu_percent: 5, execution_time: 15s
    
SecurityProfile.STRICT:
    memory_mb: 256, cpu_percent: 10, execution_time: 30s
    
SecurityProfile.STANDARD:
    memory_mb: 512, cpu_percent: 25, execution_time: 60s
```

### API Bridge (`api_bridge.py`)

**Secure Tool Access:**
- Permission-based filtering of available tools
- Rate limiting per plugin (configurable limits)
- Resource usage tracking and enforcement
- Comprehensive audit logging

**Available Tools:**
- All 38 existing MCP tools accessible through secure bridge
- Tools categorized by permission requirements
- Rate limits and resource costs per tool

### Plugin SDK (`plugin_sdk.py`)

**Development Framework:**
- `BasePlugin` - Abstract base class with lifecycle management
- `UtilityPlugin` - Decorator-based action registration
- `IntegrationPlugin` - External service connection pooling
- `ValidationHelper` - Parameter validation utilities

**Development Tools:**
- Action and hook builders with type validation
- Performance monitoring and logging
- Template generation for rapid development

### Marketplace (`marketplace.py`)

**Plugin Discovery:**
- Search with filtering by category, rating, price
- Featured plugins and developer verification
- Automatic update checking
- Secure installation with integrity verification

**Security Features:**
- Checksum verification for all downloads
- Digital signature validation (configurable)
- Security scanning before installation
- File size and content validation

## Usage Examples

### Creating a Custom Plugin

```python
from src.plugins.plugin_sdk import UtilityPlugin, create_plugin_metadata
from src.core.plugin_architecture import PluginPermissions, PermissionId

class MyPlugin(UtilityPlugin):
    def __init__(self):
        metadata = create_plugin_metadata(
            identifier="my-awesome-plugin",
            name="My Awesome Plugin", 
            version="1.0.0",
            description="Does awesome things",
            author="Plugin Developer",
            permissions=PluginPermissions(
                permissions={PermissionId("macro_read")},
                network_access=False
            )
        )
        super().__init__(metadata)
    
    @UtilityPlugin.action("process_text", "Process Text", "Process text input")
    async def process_text(self, text: str, operation: str = "uppercase") -> str:
        """Custom action that processes text."""
        if operation == "uppercase":
            return text.upper()
        elif operation == "lowercase":
            return text.lower()
        else:
            return text
    
    async def on_initialize(self):
        self.logger.info("Plugin initializing")
        return Either.right(None)
    
    async def on_activate(self):
        self.logger.info("Plugin activated")
        return Either.right(None)
    
    async def on_deactivate(self):
        return Either.right(None)
```

### Installing and Managing Plugins

```python
# Through MCP tools
await mcp_client.call_tool("km_plugin_manager", {
    "operation": "install",
    "plugin_path": "/path/to/plugin",
    "configuration": {
        "security_profile": "standard",
        "enabled": True
    }
})

await mcp_client.call_tool("km_plugin_manager", {
    "operation": "activate", 
    "plugin_id": "my-awesome-plugin"
})

# Execute custom actions
await mcp_client.call_tool("km_plugin_actions", {
    "operation": "execute",
    "action_id": "process_text",
    "parameters": {
        "text": "Hello World",
        "operation": "uppercase"
    }
})
```

### Marketplace Integration

```python
# Search for plugins
await mcp_client.call_tool("km_plugin_marketplace", {
    "operation": "search",
    "query": "text processing",
    "category": "utility",
    "limit": 10
})

# Install from marketplace
await mcp_client.call_tool("km_plugin_marketplace", {
    "operation": "install",
    "plugin_id": "text-processor-pro"
})
```

## MCP Tool Interface

### Plugin Management Tools

1. **`km_plugin_manager`** - Core plugin lifecycle management
   - Operations: list, install, load, activate, deactivate, status
   - Parameters: operation, plugin_id, plugin_path, configuration

2. **`km_plugin_marketplace`** - Plugin discovery and marketplace
   - Operations: search, details, featured, install, updates  
   - Parameters: operation, query, category, plugin_id, limit

3. **`km_plugin_security`** - Security operations and validation
   - Operations: scan, approve, block, report
   - Parameters: operation, plugin_path, plugin_id

4. **`km_plugin_actions`** - Custom action execution
   - Operations: list, execute, info
   - Parameters: operation, action_id, plugin_id, parameters

5. **`km_plugin_development`** - Development tools and templates
   - Operations: create_template, validate, package, test
   - Parameters: operation, plugin_name, template_type, target_directory

## Security Considerations

### Permission System
Plugins must declare required permissions:
- `macro_read` - Read macro information
- `macro_write` - Create/modify macros
- `variable_access` - Variable operations
- `file_access` - File system operations
- `network_access` - Network operations
- `system_access` - System-level operations

### Sandboxing
All plugin execution occurs within security boundaries:
- Resource limits enforced at OS level where possible
- Network access restricted to allowed hosts
- File system access limited to plugin directories
- CPU and memory usage monitored continuously

### Code Analysis
Static analysis performs comprehensive security scanning:
- Dangerous function detection (eval, exec, system calls)
- Import analysis for restricted modules
- Pattern matching for suspicious code
- Security rating assignment with recommendations

## Performance Characteristics

- **Plugin Loading**: < 2 seconds for typical plugins
- **Action Execution**: < 100ms average response time
- **API Bridge Calls**: < 500ms for complex operations
- **Security Scanning**: < 5 seconds for thorough analysis
- **Marketplace Search**: < 1 second for typical queries

## Plugin Development Guidelines

### Best Practices
1. **Type Safety** - Use branded types and parameter validation
2. **Error Handling** - Return Either types for all operations
3. **Logging** - Use plugin logger for debugging and monitoring
4. **Security** - Declare minimal required permissions
5. **Performance** - Implement efficient algorithms and caching
6. **Documentation** - Provide clear action descriptions and examples

### Directory Structure
```
my-plugin/
├── main.py              # Plugin entry point
├── manifest.json        # Plugin metadata
├── README.md           # Documentation
├── requirements.txt    # Dependencies (optional)
├── tests/             # Plugin tests
│   └── test_plugin.py
└── assets/            # Icons, resources
    └── icon.png
```

### Manifest Format
```json
{
    "identifier": "my-plugin",
    "name": "My Plugin",
    "version": "1.0.0", 
    "description": "Plugin description",
    "author": "Developer Name",
    "plugin_type": "utility",
    "api_version": "1.0",
    "entry_point": "main",
    "permissions": {
        "permissions": ["macro_read"],
        "network_access": false,
        "file_system_access": false,
        "system_integration": false
    },
    "dependencies": [
        {
            "plugin_id": "required-plugin",
            "version_requirement": ">=1.0.0",
            "optional": false
        }
    ]
}
```

## Testing Framework

The plugin ecosystem includes comprehensive property-based testing using Hypothesis to validate:

- Plugin metadata creation and validation
- Permission system consistency  
- Security boundary enforcement
- API bridge rate limiting
- Marketplace search functionality
- Plugin lifecycle state transitions
- Custom action parameter validation

Tests are located in `tests/property_tests/test_plugin_properties.py` and provide extensive coverage of edge cases and security boundaries.

## Integration with Existing Tools

The plugin ecosystem seamlessly integrates with all 38 existing MCP tools:

- Plugins can call any existing tool through the API bridge
- Permission-based filtering ensures security boundaries
- Rate limiting prevents abuse
- Resource monitoring tracks usage
- Comprehensive audit logging for compliance

This creates unlimited extensibility while maintaining the security and reliability of the core system.

## Future Enhancements

Planned improvements include:
- Hot-reload capability for development
- Plugin dependency graph visualization  
- Advanced marketplace features (ratings, reviews)
- Cross-platform plugin distribution
- Plugin analytics and usage insights
- AI-powered plugin recommendations
- Collaborative plugin development tools

The plugin ecosystem represents a significant advancement in automation capabilities, enabling users to extend the system infinitely while maintaining enterprise-grade security and performance standards.