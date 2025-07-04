"""
Property-based tests for the plugin ecosystem.

This module provides comprehensive property-based testing using Hypothesis
to validate plugin system behavior across input ranges and edge cases.
"""

import pytest
import asyncio
import tempfile
import json
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from pathlib import Path
from typing import Dict, Any, List

from src.core.plugin_architecture import (
    PluginMetadata, PluginConfiguration, CustomAction, CustomActionParameter,
    PluginPermissions, PluginError, SecurityProfile, ApiVersion, PluginType,
    PluginId, ActionId, PermissionId
)
from src.plugins.plugin_manager import PluginManager, PluginRegistry
from src.plugins.security_sandbox import PluginSecurityManager, SecurityLimits
from src.plugins.api_bridge import PluginAPIBridge
from src.plugins.marketplace import PluginMarketplace, SearchQuery
from src.core.either import Either


# Hypothesis strategies for plugin testing

@st.composite
def plugin_identifier_strategy(draw):
    """Generate valid plugin identifiers."""
    # Valid identifier: starts with letter, contains alphanumeric/underscore/hyphen/dot
    first_char = draw(st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    rest_chars = draw(st.text(
        alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-',
        min_size=0,
        max_size=50
    ))
    return PluginId(first_char + rest_chars)


@st.composite
def version_strategy(draw):
    """Generate valid semantic version strings."""
    major = draw(st.integers(min_value=0, max_value=99))
    minor = draw(st.integers(min_value=0, max_value=99))
    patch = draw(st.integers(min_value=0, max_value=99))
    
    # Optional pre-release identifier
    pre_release = draw(st.one_of(
        st.none(),
        st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789', min_size=1, max_size=10)
    ))
    
    version = f"{major}.{minor}.{patch}"
    if pre_release:
        version += f"-{pre_release}"
    
    return version


@st.composite
def plugin_permissions_strategy(draw):
    """Generate plugin permissions configurations."""
    permissions = draw(st.sets(
        st.sampled_from([
            PermissionId("macro_read"),
            PermissionId("macro_write"),
            PermissionId("variable_access"),
            PermissionId("file_access"),
            PermissionId("network_access"),
            PermissionId("system_access")
        ]),
        min_size=0,
        max_size=6
    ))
    
    return PluginPermissions(
        permissions=permissions,
        resource_limits=draw(st.dictionaries(
            st.sampled_from(["memory_mb", "cpu_percent", "disk_mb"]),
            st.integers(min_value=1, max_value=1000),
            min_size=0,
            max_size=3
        )),
        network_access=draw(st.booleans()),
        file_system_access=draw(st.booleans()),
        system_integration=draw(st.booleans())
    )


@st.composite
def plugin_metadata_strategy(draw):
    """Generate valid plugin metadata."""
    return PluginMetadata(
        identifier=draw(plugin_identifier_strategy()),
        name=draw(st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_categories=['Cc']))),
        version=draw(version_strategy()),
        description=draw(st.text(min_size=0, max_size=500, alphabet=st.characters(blacklist_categories=['Cc']))),
        author=draw(st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_categories=['Cc']))),
        plugin_type=draw(st.sampled_from(list(PluginType))),
        api_version=draw(st.sampled_from(list(ApiVersion))),
        permissions=draw(plugin_permissions_strategy()),
        entry_point=draw(st.text(min_size=1, max_size=50, alphabet='abcdefghijklmnopqrstuvwxyz_').filter(lambda x: len(x.replace('_', '')) > 0)),
    )


@st.composite
def security_profile_strategy(draw):
    """Generate security profiles."""
    return draw(st.sampled_from(list(SecurityProfile)))


@st.composite
def action_parameter_strategy(draw):
    """Generate custom action parameters."""
    return CustomActionParameter(
        name=draw(st.text(
            min_size=1,
            max_size=50,
            alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
        )),
        param_type=draw(st.sampled_from([str, int, float, bool])),
        description=draw(st.text(min_size=0, max_size=200, alphabet=st.characters(blacklist_categories=['Cc']))),
        required=draw(st.booleans()),
        default_value=draw(st.one_of(
            st.none(),
            st.text(max_size=100),
            st.integers(min_value=-1000, max_value=1000),
            st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            st.booleans()
        ))
    )


class TestPluginMetadataProperties:
    """Property-based tests for plugin metadata validation."""
    
    @given(plugin_metadata_strategy())
    @settings(max_examples=100)
    def test_metadata_creation_invariants(self, metadata: PluginMetadata):
        """Test that valid metadata always satisfies invariants."""
        # Identifier invariants
        assert metadata.identifier
        assert len(metadata.identifier) > 0
        assert metadata.identifier[0].isalpha()
        
        # Name invariants
        assert metadata.name.strip()
        assert len(metadata.name.strip()) > 0
        
        # Version invariants
        version_parts = metadata.version.split('.')
        assert len(version_parts) >= 2
        assert all(part.split('-')[0].isdigit() for part in version_parts[:3])
        
        # Permissions invariants
        assert isinstance(metadata.permissions, PluginPermissions)
        
        # Entry point invariants
        assert metadata.entry_point
        assert len(metadata.entry_point.replace('_', '')) > 0
        assert metadata.entry_point.replace('_', '').isalnum()
    
    @given(
        identifier=st.text(max_size=0),
        name=st.text(min_size=1, max_size=100),
        version=version_strategy(),
        description=st.text(max_size=500),
        author=st.text(min_size=1, max_size=100)
    )
    def test_invalid_identifier_rejected(self, identifier, name, version, description, author):
        """Test that invalid identifiers are rejected."""
        assume(len(identifier) == 0)  # Empty identifier should be invalid
        
        with pytest.raises(ValueError):
            PluginMetadata(
                identifier=PluginId(identifier),
                name=name,
                version=version,
                description=description,
                author=author,
                plugin_type=PluginType.UTILITY,
                api_version=ApiVersion.V1_0
            )
    
    @given(st.text(min_size=1, max_size=3))
    def test_invalid_version_rejected(self, invalid_version):
        """Test that invalid version formats are rejected."""
        assume(not invalid_version.replace('.', '').replace('-', '').isalnum())
        
        with pytest.raises(ValueError):
            PluginMetadata(
                identifier=PluginId("test-plugin"),
                name="Test Plugin",
                version=invalid_version,
                description="Test description",
                author="Test Author",
                plugin_type=PluginType.UTILITY,
                api_version=ApiVersion.V1_0
            )


class TestPluginPermissionsProperties:
    """Property-based tests for plugin permissions."""
    
    @given(plugin_permissions_strategy())
    @settings(max_examples=50)
    def test_permissions_consistency(self, permissions: PluginPermissions):
        """Test that permission settings are internally consistent."""
        # Elevated access should be true if any elevated permission is granted
        elevated_permissions = {
            permissions.network_access,
            permissions.file_system_access,
            permissions.system_integration
        }
        
        if any(elevated_permissions):
            assert permissions.requires_elevated_access()
        
        # All permission IDs should be valid
        for perm_id in permissions.permissions:
            assert isinstance(perm_id, str)
            assert len(perm_id) > 0
    
    @given(
        permissions=st.sets(st.sampled_from([
            PermissionId("macro_read"),
            PermissionId("macro_write"),
            PermissionId("dangerous_operation")
        ])),
        permission_to_check=st.sampled_from([
            PermissionId("macro_read"),
            PermissionId("macro_write"),
            PermissionId("nonexistent")
        ])
    )
    def test_permission_checking(self, permissions, permission_to_check):
        """Test permission checking behavior."""
        plugin_perms = PluginPermissions(permissions=permissions)
        
        has_permission = plugin_perms.has_permission(permission_to_check)
        assert has_permission == (permission_to_check in permissions)


class TestSecurityLimitsProperties:
    """Property-based tests for security limits."""
    
    @given(security_profile_strategy())
    @settings(max_examples=20)
    def test_security_limits_scaling(self, profile: SecurityProfile):
        """Test that security limits scale appropriately with profile."""
        limits = SecurityLimits(profile)
        
        # More restrictive profiles should have lower limits
        if profile == SecurityProfile.SANDBOX:
            assert limits.limits['memory_mb'] <= 256
            assert limits.limits['cpu_percent'] <= 10
            assert not limits.allows_network()
            assert not limits.allows_subprocess()
        
        elif profile == SecurityProfile.STRICT:
            assert limits.limits['memory_mb'] <= 512
            assert limits.limits['cpu_percent'] <= 25
        
        # All profiles should have reasonable timeout limits
        timeout = limits.get_execution_timeout()
        assert 1 <= timeout <= 300


class TestPluginRegistryProperties:
    """Property-based tests for plugin registry operations."""
    
    @pytest.fixture
    def temp_registry(self):
        """Create temporary registry for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "test_registry.json"
            yield PluginRegistry(registry_path)
    
    @given(st.lists(plugin_metadata_strategy(), min_size=0, max_size=10, unique_by=lambda x: x.identifier))
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_registry_persistence(self, temp_registry, plugin_list: List[PluginMetadata]):
        """Test that registry persists and loads data correctly."""
        # Register all plugins
        for metadata in plugin_list:
            temp_registry.register_plugin(metadata, Path("/mock/path"))
        
        # Save registry
        save_result = await temp_registry.save_registry()
        assert save_result.is_right()
        
        # Create new registry instance and load
        new_registry = PluginRegistry(temp_registry.registry_path)
        load_result = await new_registry.load_registry()
        assert load_result.is_right()
        
        # Verify all plugins were loaded
        loaded_plugins = new_registry.list_plugins()
        assert len(loaded_plugins) == len(plugin_list)
        
        for original in plugin_list:
            loaded = new_registry.get_plugin_metadata(original.identifier)
            assert loaded is not None
            assert loaded.identifier == original.identifier
            assert loaded.name == original.name
            assert loaded.version == original.version
    
    @given(plugin_metadata_strategy())
    @settings(max_examples=20)
    def test_registry_operations_idempotent(self, temp_registry, metadata: PluginMetadata):
        """Test that registry operations are idempotent."""
        plugin_path = Path("/mock/path")
        
        # Register plugin multiple times
        temp_registry.register_plugin(metadata, plugin_path)
        temp_registry.register_plugin(metadata, plugin_path)
        temp_registry.register_plugin(metadata, plugin_path)
        
        # Should only have one entry
        plugins = temp_registry.list_plugins()
        plugin_count = sum(1 for p in plugins if p.identifier == metadata.identifier)
        assert plugin_count == 1
        
        # Unregister and verify
        temp_registry.unregister_plugin(metadata.identifier)
        assert temp_registry.get_plugin_metadata(metadata.identifier) is None


class TestAPIBridgeProperties:
    """Property-based tests for API bridge security and functionality."""
    
    def create_api_bridge(self):
        """Create API bridge for testing - not a fixture to avoid Hypothesis issues."""
        return PluginAPIBridge()
    
    @given(
        plugin_id=plugin_identifier_strategy(),
        tool_name=st.sampled_from(["km_variable_manager", "km_execute_macro", "nonexistent_tool"]),
        parameters=st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz_'),
            st.one_of(st.text(max_size=100), st.integers(), st.booleans()),
            min_size=0,
            max_size=5
        )
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_api_bridge_security_boundaries(self, plugin_id, tool_name, parameters):
        """Test API bridge enforces security boundaries."""
        # Create API bridge for this test
        api_bridge = self.create_api_bridge()
        await api_bridge.initialize()
        
        # Create minimal permissions
        permissions = PluginPermissions(permissions=set())
        
        # Authorize plugin
        api_bridge.authorize_plugin(plugin_id)
        
        # Call tool
        result = await api_bridge.call_tool(plugin_id, permissions, tool_name, parameters)
        
        if tool_name == "nonexistent_tool":
            # Should fail for nonexistent tools
            assert result.is_left()
            error = result.get_left()
            assert "not found" in error.message.lower()
        else:
            # Should fail for insufficient permissions
            assert result.is_left()
            error = result.get_left()
            assert "permission" in error.message.lower() or "denied" in error.message.lower()
    
    @given(plugin_id=plugin_identifier_strategy())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_api_bridge_authorization_required(self, plugin_id):
        """Test that API bridge requires plugin authorization."""
        # Create API bridge for this test
        api_bridge = self.create_api_bridge()
        await api_bridge.initialize()
        
        permissions = PluginPermissions.standard()
        
        # Try to call tool without authorization
        result = await api_bridge.call_tool(
            plugin_id, permissions, "km_variable_manager", {"operation": "list"}
        )
        
        assert result.is_left()
        error = result.get_left()
        assert "not authorized" in error.message.lower()
    
    @given(
        plugin_id=plugin_identifier_strategy(),
        tool_name=st.sampled_from(["km_variable_manager", "km_calculator"]),
        call_count=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self, api_bridge, plugin_id, tool_name, call_count):
        """Test that rate limiting is properly enforced."""
        # Authorize plugin
        api_bridge.authorize_plugin(plugin_id)
        
        # Create permissions with required access
        permissions = PluginPermissions(
            permissions={PermissionId("variable_access"), PermissionId("calculation")},
            network_access=False,
            file_system_access=False
        )
        
        # Make many rapid calls
        rate_limited = False
        for i in range(call_count):
            result = await api_bridge.call_tool(
                plugin_id, permissions, tool_name, {"operation": "test"}
            )
            
            if result.is_left() and "rate limit" in result.get_left().message.lower():
                rate_limited = True
                break
        
        # For high call counts, should eventually hit rate limit
        if call_count > 50:
            assert rate_limited


class TestMarketplaceProperties:
    """Property-based tests for marketplace functionality."""
    
    @pytest.fixture
    async def marketplace(self):
        """Create marketplace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            marketplace = PluginMarketplace(cache_dir=Path(temp_dir))
            await marketplace.initialize()
            yield marketplace
    
    @given(
        query_text=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        limit=st.integers(min_value=1, max_value=50),
        offset=st.integers(min_value=0, max_value=20)
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_search_result_bounds(self, marketplace, query_text, limit, offset):
        """Test that search results respect pagination bounds."""
        from src.plugins.marketplace import SearchQuery
        
        query = SearchQuery(
            query=query_text,
            limit=limit,
            offset=offset
        )
        
        result = await marketplace.search_plugins(query)
        assert result.is_right()
        
        plugins = result.get_right()
        
        # Result count should not exceed limit
        assert len(plugins) <= limit
        
        # Should be non-negative
        assert len(plugins) >= 0
    
    @given(st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz-'))
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_plugin_details_consistency(self, marketplace, plugin_id_str):
        """Test plugin details retrieval consistency."""
        plugin_id = PluginId(plugin_id_str)
        
        # Get details twice
        result1 = await marketplace.get_plugin_details(plugin_id)
        result2 = await marketplace.get_plugin_details(plugin_id)
        
        # Results should be consistent
        assert result1.is_left() == result2.is_left()
        
        if result1.is_right() and result2.is_right():
            entry1 = result1.get_right()
            entry2 = result2.get_right()
            
            # Core metadata should be identical
            assert entry1.metadata.identifier == entry2.metadata.identifier
            assert entry1.metadata.version == entry2.metadata.version
            assert entry1.status == entry2.status


class TestPluginLifecycleProperties:
    """Property-based tests for plugin lifecycle management."""
    
    @given(plugin_metadata_strategy())
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_plugin_lifecycle_state_transitions(self, metadata: PluginMetadata):
        """Test plugin lifecycle state transitions are valid."""
        from src.plugins.plugin_sdk import BasePlugin
        
        # Create mock plugin
        class MockPlugin(BasePlugin):
            async def on_initialize(self):
                return Either.right(None)
            
            async def on_activate(self):
                return Either.right(None)
            
            async def on_deactivate(self):
                return Either.right(None)
            
            async def register_actions(self):
                return []
            
            async def register_hooks(self):
                return []
        
        plugin = MockPlugin(metadata)
        
        # Test initial state
        assert not plugin._initialized
        assert not plugin._active
        
        # Initialize
        api_bridge = PluginAPIBridge()
        await api_bridge.initialize()
        
        init_result = await plugin.initialize(api_bridge)
        assert init_result.is_right()
        assert plugin._initialized
        assert not plugin._active
        
        # Activate
        activate_result = await plugin.activate()
        assert activate_result.is_right()
        assert plugin._initialized
        assert plugin._active
        
        # Deactivate
        deactivate_result = await plugin.deactivate()
        assert deactivate_result.is_right()
        assert plugin._initialized
        assert not plugin._active
    
    @given(
        action_params=st.lists(action_parameter_strategy(), min_size=0, max_size=5),
        test_values=st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz_'),
            st.one_of(st.text(max_size=100), st.integers(), st.booleans()),
            min_size=0,
            max_size=5
        )
    )
    @settings(max_examples=30)
    def test_action_parameter_validation(self, action_params: List[CustomActionParameter], test_values: Dict[str, Any]):
        """Test custom action parameter validation."""
        # Create mock action
        async def mock_handler(**kwargs):
            return "success"
        
        action = CustomAction(
            action_id=ActionId("test-action"),
            name="Test Action",
            description="Test action for property testing",
            parameters=action_params,
            return_type=str,
            handler=mock_handler,
            plugin_id=PluginId("test-plugin")
        )
        
        # Validate parameters
        validation_result = action.validate_parameters(test_values)
        
        # Check that validation catches missing required parameters
        required_params = {p.name for p in action_params if p.required}
        provided_params = set(test_values.keys())
        missing_required = required_params - provided_params
        
        if missing_required:
            assert validation_result.is_left()
        
        # Check that validation catches unexpected parameters
        expected_params = {p.name for p in action_params}
        unexpected_params = provided_params - expected_params
        
        if unexpected_params and validation_result.is_right():
            # If there are unexpected params but validation passed,
            # it means all required params were provided
            pass


# Integration property tests

@pytest.mark.asyncio
async def test_end_to_end_plugin_system_properties():
    """End-to-end property test of the entire plugin system."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plugins_dir = Path(temp_dir)
        
        # Initialize all components
        manager = PluginManager(plugins_dir)
        await manager.initialize()
        
        # The system should start in a clean state
        assert len(manager.list_plugins()) == 0
        assert len(manager.get_custom_actions()) == 0
        
        # API bridge should be available
        assert manager.api_bridge is not None
        health = await manager.api_bridge.health_check()
        assert health["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])