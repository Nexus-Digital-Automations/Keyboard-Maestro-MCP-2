"""
Tests for plugin ecosystem MCP tools.

This module provides comprehensive testing for the plugin ecosystem including
plugin installation, lifecycle management, custom action execution, and security validation.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from src.server.tools.plugin_ecosystem_tools import PluginEcosystemTools
from src.core.plugin_architecture import (
    PluginId, ActionId, PluginStatus, SecurityProfile, PluginError,
    PluginMetadata, PluginConfiguration, PluginType, ApiVersion
)


class TestPluginEcosystemTools:
    """Test suite for plugin ecosystem tools."""
    
    @pytest.fixture
    def plugin_tools(self):
        """Create plugin ecosystem tools instance."""
        return PluginEcosystemTools()
    
    @pytest.fixture
    def sample_plugin_metadata(self):
        """Create sample plugin metadata."""
        return {
            "identifier": "test-plugin",
            "name": "Test Plugin",
            "version": "1.0.0",
            "description": "A test plugin for unit testing",
            "author": "Test Author",
            "plugin_type": "action",
            "api_version": "1.0",
            "dependencies": [],
            "permissions": [],
            "entry_point": "main"
        }
    
    @pytest.fixture
    def sample_plugin_directory(self, sample_plugin_metadata):
        """Create a sample plugin directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_dir = Path(temp_dir) / "test-plugin"
            plugin_dir.mkdir()
            
            # Create manifest
            with open(plugin_dir / "manifest.json", 'w') as f:
                json.dump(sample_plugin_metadata, f)
            
            # Create entry point
            with open(plugin_dir / "main.py", 'w') as f:
                f.write("""
from src.core.plugin_architecture import PluginInterface, CustomAction

class Plugin(PluginInterface):
    async def initialize(self, api_bridge):
        return Either.right(None)
    
    async def activate(self):
        return Either.right(None)
    
    async def deactivate(self):
        return Either.right(None)
    
    async def get_custom_actions(self):
        return []
    
    async def get_hooks(self):
        return []
""")
            
            yield plugin_dir
    
    @pytest.mark.asyncio
    async def test_plugin_installation_success(self, plugin_tools, sample_plugin_directory):
        """Test successful plugin installation."""
        with patch.object(plugin_tools.plugin_manager, 'install_plugin') as mock_install:
            mock_metadata = Mock()
            mock_metadata.identifier = "test-plugin"
            mock_metadata.name = "Test Plugin"
            mock_metadata.version = "1.0.0"
            mock_metadata.description = "Test plugin"
            mock_metadata.author = "Test Author"
            mock_metadata.plugin_type = PluginType.ACTION
            mock_metadata.api_version = ApiVersion.V1_0
            mock_metadata.permissions = Mock()
            mock_metadata.permissions.network_access = False
            mock_metadata.permissions.file_system_access = False
            mock_metadata.permissions.system_integration = False
            
            from src.core.either import Either
            mock_install.return_value = Either.right(mock_metadata)
            
            result = await plugin_tools.km_plugin_ecosystem(
                operation="install",
                plugin_source=str(sample_plugin_directory),
                security_profile="standard"
            )
            
            assert result['success'] is True
            assert result['operation'] == 'install'
            assert result['plugin']['id'] == 'test-plugin'
            assert result['plugin']['name'] == 'Test Plugin'
            assert result['security_profile'] == 'standard'
            
            mock_install.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_plugin_installation_failure(self, plugin_tools):
        """Test plugin installation failure."""
        with patch.object(plugin_tools.plugin_manager, 'install_plugin') as mock_install:
            from src.core.either import Either
            mock_error = PluginError("Installation failed", "INSTALL_ERROR")
            mock_install.return_value = Either.left(mock_error)
            
            result = await plugin_tools.km_plugin_ecosystem(
                operation="install",
                plugin_source="/nonexistent/path"
            )
            
            assert result['success'] is False
            assert 'error' in result
            assert result['error_code'] == 'INSTALL_ERROR'
    
    @pytest.mark.asyncio
    async def test_plugin_list_operation(self, plugin_tools):
        """Test plugin listing operation."""
        mock_plugins = [
            {
                'id': 'plugin1',
                'name': 'Plugin 1',
                'status': 'active',
                'type': 'action'
            },
            {
                'id': 'plugin2',
                'name': 'Plugin 2', 
                'status': 'loaded',
                'type': 'utility'
            }
        ]
        
        with patch.object(plugin_tools.plugin_manager, 'list_plugins', return_value=mock_plugins):
            result = await plugin_tools.km_plugin_ecosystem(operation="list")
            
            assert result['success'] is True
            assert result['operation'] == 'list'
            assert len(result['plugins']) == 2
            assert result['total_count'] == 2
    
    @pytest.mark.asyncio
    async def test_plugin_activation(self, plugin_tools):
        """Test plugin activation."""
        with patch.object(plugin_tools.plugin_manager, 'activate_plugin') as mock_activate, \
             patch.object(plugin_tools.plugin_manager, 'list_plugins') as mock_list, \
             patch.object(plugin_tools.plugin_manager, 'get_custom_actions') as mock_actions:
            
            from src.core.either import Either
            mock_activate.return_value = Either.right(None)
            mock_list.return_value = [{'id': 'test-plugin', 'status': 'active'}]
            mock_actions.return_value = [
                {'id': 'action1', 'plugin_id': 'test-plugin', 'name': 'Test Action'}
            ]
            
            result = await plugin_tools.km_plugin_ecosystem(
                operation="activate",
                plugin_identifier="test-plugin"
            )
            
            assert result['success'] is True
            assert result['operation'] == 'activate'
            assert result['plugin_id'] == 'test-plugin'
            assert result['action_count'] == 1
            
            mock_activate.assert_called_once_with(PluginId('test-plugin'))
    
    @pytest.mark.asyncio
    async def test_custom_action_execution(self, plugin_tools):
        """Test custom action execution."""
        with patch.object(plugin_tools.plugin_manager, 'execute_custom_action') as mock_execute:
            from src.core.either import Either
            mock_execute.return_value = Either.right("Action executed successfully")
            
            result = await plugin_tools.km_plugin_ecosystem(
                operation="execute",
                action_name="test_action",
                parameters={"param1": "value1"}
            )
            
            assert result['success'] is True
            assert result['operation'] == 'execute'
            assert result['action_name'] == 'test_action'
            assert result['result'] == "Action executed successfully"
            
            mock_execute.assert_called_once_with(
                ActionId('test_action'),
                {"param1": "value1"}
            )
    
    @pytest.mark.asyncio
    async def test_plugin_configuration_update(self, plugin_tools):
        """Test plugin configuration update."""
        mock_config = Mock()
        mock_config.plugin_id = PluginId("test-plugin")
        mock_config.settings = {"key1": "value1"}
        mock_config.enabled = True
        mock_config.auto_update = False
        mock_config.security_profile = SecurityProfile.STANDARD
        mock_config.resource_limits = {}
        
        with patch.object(plugin_tools.plugin_manager, 'plugin_configurations', {'test-plugin': mock_config}):
            result = await plugin_tools.km_plugin_ecosystem(
                operation="configure",
                plugin_identifier="test-plugin",
                plugin_config={
                    "settings": {"key2": "value2"},
                    "enabled": True
                }
            )
            
            assert result['success'] is True
            assert result['operation'] == 'configure'
            assert result['plugin_id'] == 'test-plugin'
    
    @pytest.mark.asyncio
    async def test_plugin_status_inquiry(self, plugin_tools):
        """Test plugin status inquiry."""
        with patch.object(plugin_tools.plugin_manager, 'get_plugin_status') as mock_status, \
             patch.object(plugin_tools.plugin_manager, 'list_plugins') as mock_list, \
             patch.object(plugin_tools.plugin_manager, 'get_custom_actions') as mock_actions:
            
            mock_status.return_value = PluginStatus.ACTIVE
            mock_list.return_value = [{'id': 'test-plugin', 'status': 'active'}]
            mock_actions.return_value = [
                {'id': 'action1', 'plugin_id': 'test-plugin'}
            ]
            
            result = await plugin_tools.km_plugin_ecosystem(
                operation="status",
                plugin_identifier="test-plugin"
            )
            
            assert result['success'] is True
            assert result['operation'] == 'status'
            assert result['plugin_id'] == 'test-plugin'
            assert result['status'] == 'active'
            assert result['action_count'] == 1
    
    @pytest.mark.asyncio
    async def test_marketplace_search(self, plugin_tools):
        """Test marketplace search functionality."""
        result = await plugin_tools.km_plugin_ecosystem(
            operation="marketplace",
            parameters={
                "marketplace_operation": "search",
                "query": "test",
                "category": "utility"
            }
        )
        
        assert result['success'] is True
        assert result['operation'] == 'marketplace'
        assert result['marketplace_operation'] == 'search'
        assert result['query'] == 'test'
        assert result['category'] == 'utility'
        assert 'results' in result
    
    @pytest.mark.asyncio
    async def test_invalid_operation(self, plugin_tools):
        """Test handling of invalid operations."""
        result = await plugin_tools.km_plugin_ecosystem(operation="invalid_operation")
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Invalid operation' in result['error']
        assert 'valid_operations' in result
    
    @pytest.mark.asyncio
    async def test_missing_plugin_identifier(self, plugin_tools):
        """Test operations requiring plugin identifier without providing it."""
        result = await plugin_tools.km_plugin_ecosystem(operation="activate")
        
        assert result['success'] is False
        assert 'Plugin identifier is required' in result['error']
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, plugin_tools):
        """Test timeout handling in operations."""
        with patch.object(plugin_tools.plugin_manager, 'install_plugin') as mock_install:
            # Simulate a timeout
            async def slow_install(*args, **kwargs):
                await asyncio.sleep(2)
                from src.core.either import Either
                return Either.right(Mock())
            
            mock_install.side_effect = slow_install
            
            result = await plugin_tools.km_plugin_ecosystem(
                operation="install",
                plugin_source="/some/path",
                timeout=1  # 1 second timeout
            )
            
            assert result['success'] is False
            assert 'timeout' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_security_profile_validation(self, plugin_tools):
        """Test security profile validation."""
        with patch.object(plugin_tools.plugin_manager, 'install_plugin') as mock_install:
            from src.core.either import Either
            mock_install.return_value = Either.right(Mock())
            
            # Test invalid security profile
            result = await plugin_tools.km_plugin_ecosystem(
                operation="install",
                plugin_source="/some/path",
                security_profile="invalid_profile"
            )
            
            assert result['success'] is False
            assert 'Invalid security profile' in result['error']
            assert 'valid_profiles' in result


class TestPluginEcosystemIntegration:
    """Integration tests for plugin ecosystem."""
    
    @pytest.mark.asyncio
    async def test_plugin_lifecycle_integration(self):
        """Test complete plugin lifecycle integration."""
        tools = PluginEcosystemTools()
        
        with patch.object(tools.plugin_manager, 'install_plugin') as mock_install, \
             patch.object(tools.plugin_manager, 'activate_plugin') as mock_activate, \
             patch.object(tools.plugin_manager, 'execute_custom_action') as mock_execute, \
             patch.object(tools.plugin_manager, 'deactivate_plugin') as mock_deactivate, \
             patch.object(tools.plugin_manager, 'uninstall_plugin') as mock_uninstall:
            
            from src.core.either import Either
            
            # Mock successful operations
            mock_metadata = Mock()
            mock_metadata.identifier = "test-plugin"
            mock_metadata.name = "Test Plugin"
            mock_metadata.version = "1.0.0"
            mock_metadata.description = "Test"
            mock_metadata.author = "Author"
            mock_metadata.plugin_type = PluginType.ACTION
            mock_metadata.api_version = ApiVersion.V1_0
            mock_metadata.permissions = Mock()
            mock_metadata.permissions.network_access = False
            mock_metadata.permissions.file_system_access = False
            mock_metadata.permissions.system_integration = False
            
            mock_install.return_value = Either.right(mock_metadata)
            mock_activate.return_value = Either.right(None)
            mock_execute.return_value = Either.right("Success")
            mock_deactivate.return_value = Either.right(None)
            mock_uninstall.return_value = Either.right(None)
            
            # Test install
            install_result = await tools.km_plugin_ecosystem(
                operation="install",
                plugin_source="/test/plugin"
            )
            assert install_result['success'] is True
            
            # Test activate
            activate_result = await tools.km_plugin_ecosystem(
                operation="activate",
                plugin_identifier="test-plugin"
            )
            assert activate_result['success'] is True
            
            # Test execute
            execute_result = await tools.km_plugin_ecosystem(
                operation="execute",
                action_name="test_action",
                parameters={}
            )
            assert execute_result['success'] is True
            
            # Test deactivate
            deactivate_result = await tools.km_plugin_ecosystem(
                operation="deactivate",
                plugin_identifier="test-plugin"
            )
            assert deactivate_result['success'] is True
            
            # Test uninstall
            uninstall_result = await tools.km_plugin_ecosystem(
                operation="uninstall",
                plugin_identifier="test-plugin"
            )
            assert uninstall_result['success'] is True


if __name__ == "__main__":
    pytest.main([__file__])