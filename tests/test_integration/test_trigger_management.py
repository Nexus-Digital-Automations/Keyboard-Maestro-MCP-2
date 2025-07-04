"""
Tests for trigger registration and event routing system.

Tests the TASK_2 Phase 2 implementation of trigger management
with property-based testing and integration validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

# Hypothesis for property-based testing
from hypothesis import given, strategies as st, assume

# Local imports
from src.integration.triggers import (
    TriggerRegistrationManager, 
    EventRouter,
    TriggerDefinition,
    EventRoutingRule,
    TriggerInfo,
    TriggerState,
    TriggerStatus,
    TriggerMetadata,
    create_hotkey_routing_rule,
    create_application_routing_rule
)
from src.integration.events import KMEvent, TriggerType, EventPriority
from src.integration.km_client import KMClient, Either, KMError, ConnectionConfig
from src.core.types import TriggerId, MacroId


class TestTriggerRegistrationManager:
    """Test trigger registration with KM system."""
    
    @pytest.fixture
    def mock_km_client(self):
        """Mock KM client for testing."""
        client = Mock(spec=KMClient)
        client.register_trigger_async = AsyncMock()
        client.activate_trigger_async = AsyncMock()
        client.deactivate_trigger_async = AsyncMock()
        client.list_triggers_async = AsyncMock()
        return client
    
    @pytest.fixture
    def trigger_manager(self, mock_km_client):
        """Trigger registration manager with mock client."""
        return TriggerRegistrationManager(mock_km_client)
    
    @pytest.fixture
    def sample_trigger_def(self):
        """Sample trigger definition for testing."""
        return TriggerDefinition(
            trigger_id=TriggerId("test-trigger-123"),
            macro_id=MacroId("test-macro-456"),
            trigger_type=TriggerType.HOTKEY,
            configuration={"key": "F1", "modifiers": ["Command"]},
            name="Test Hotkey",
            description="Test hotkey trigger",
            enabled=True
        )
    
    @pytest.mark.asyncio
    async def test_successful_trigger_registration(self, trigger_manager, mock_km_client, sample_trigger_def):
        """Test successful trigger registration flow."""
        # Mock successful registration
        mock_km_client.register_trigger_async.return_value = Either.right(sample_trigger_def.trigger_id)
        
        # Register trigger
        result = await trigger_manager.register_trigger(sample_trigger_def)
        
        # Verify success
        assert result.is_right()
        assert result.get_right() == sample_trigger_def.trigger_id
        
        # Verify client was called
        mock_km_client.register_trigger_async.assert_called_once()
        
        # Verify state updated
        state = trigger_manager.get_current_state()
        assert state.has_trigger(sample_trigger_def.trigger_id)
        
        trigger_info = state.get_trigger(sample_trigger_def.trigger_id)
        assert trigger_info.status == TriggerStatus.REGISTERED
        assert trigger_info.macro_id == sample_trigger_def.macro_id
    
    @pytest.mark.asyncio
    async def test_trigger_registration_validation_failure(self, trigger_manager, mock_km_client):
        """Test trigger registration with invalid configuration."""
        # Create trigger with invalid config (script injection)
        bad_trigger_def = TriggerDefinition(
            trigger_id=TriggerId("bad-trigger"),
            macro_id=MacroId("test-macro"),
            trigger_type=TriggerType.HOTKEY,
            configuration={"script": "<script>alert('xss')</script>"},  # Should be caught by validation
            enabled=True
        )
        
        # Register trigger
        result = await trigger_manager.register_trigger(bad_trigger_def)
        
        # Should fail validation
        assert result.is_left()
        assert "Invalid trigger configuration" in result.get_left().message
        
        # Client should not be called
        mock_km_client.register_trigger_async.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_trigger_activation(self, trigger_manager, mock_km_client, sample_trigger_def):
        """Test trigger activation flow."""
        # First register the trigger
        mock_km_client.register_trigger_async.return_value = Either.right(sample_trigger_def.trigger_id)
        await trigger_manager.register_trigger(sample_trigger_def)
        
        # Mock successful activation
        mock_km_client.activate_trigger_async.return_value = Either.right(True)
        
        # Activate trigger
        result = await trigger_manager.activate_trigger(sample_trigger_def.trigger_id)
        
        # Verify success
        assert result.is_right()
        assert result.get_right() is True
        
        # Verify state updated
        state = trigger_manager.get_current_state()
        trigger_info = state.get_trigger(sample_trigger_def.trigger_id)
        assert trigger_info.status == TriggerStatus.ACTIVE
        assert sample_trigger_def.trigger_id in state.active_triggers
    
    @pytest.mark.asyncio
    async def test_state_synchronization(self, trigger_manager, mock_km_client):
        """Test state synchronization with KM."""
        # Mock KM response
        km_triggers = [
            {"triggerId": "trigger-1", "status": "active"},
            {"triggerId": "trigger-2", "status": "inactive"}
        ]
        mock_km_client.list_triggers_async.return_value = Either.right(km_triggers)
        
        # Sync state
        result = await trigger_manager.sync_state_with_km()
        
        # Verify success
        assert result.is_right()
        
        # Verify client was called
        mock_km_client.list_triggers_async.assert_called_once()


class TestEventRouter:
    """Test event routing to macro handlers."""
    
    @pytest.fixture
    def mock_macro_engine(self):
        """Mock macro engine for testing."""
        engine = Mock()
        engine.execute_macro_async = AsyncMock()
        return engine
    
    @pytest.fixture
    def mock_trigger_manager(self):
        """Mock trigger manager for testing."""
        manager = Mock()
        manager.get_current_state = Mock()
        manager._state_lock = asyncio.Lock()
        manager._state = Mock()
        return manager
    
    @pytest.fixture
    def event_router(self, mock_macro_engine, mock_trigger_manager):
        """Event router with mock dependencies."""
        return EventRouter(mock_macro_engine, mock_trigger_manager)
    
    @pytest.fixture
    def sample_trigger_info(self):
        """Sample trigger info for testing."""
        return TriggerInfo(
            trigger_id=TriggerId("test-trigger"),
            macro_id=MacroId("test-macro"),
            trigger_type=TriggerType.HOTKEY,
            status=TriggerStatus.ACTIVE,
            configuration={"key": "F1"},
            metadata=TriggerMetadata(name="Test Trigger")
        )
    
    @pytest.fixture
    def sample_km_event(self):
        """Sample KM event for testing."""
        return KMEvent.create(
            trigger_type=TriggerType.HOTKEY,
            trigger_id=TriggerId("test-trigger"),
            payload={"trigger_value": "test_value", "key": "F1"},
            priority=EventPriority.NORMAL
        )
    
    @pytest.mark.asyncio
    async def test_successful_event_routing(
        self, 
        event_router, 
        mock_macro_engine, 
        mock_trigger_manager, 
        sample_trigger_info, 
        sample_km_event
    ):
        """Test successful event routing to macro execution."""
        # Setup mocks
        trigger_state = Mock()
        trigger_state.get_trigger.return_value = sample_trigger_info
        mock_trigger_manager.get_current_state.return_value = trigger_state
        
        # Mock successful macro execution
        execution_result = Mock()
        execution_result.status.value = "completed"
        mock_macro_engine.execute_macro_async.return_value = execution_result
        
        # Mock state update
        mock_trigger_manager._state.with_triggered.return_value = Mock()
        
        # Route event
        result = await event_router.route_event(sample_km_event)
        
        # Verify success
        assert result.is_right()
        assert result.get_right() is True
        
        # Verify macro engine was called
        mock_macro_engine.execute_macro_async.assert_called_once_with(
            macro_id=sample_trigger_info.macro_id,
            trigger_value=sample_km_event.get_payload_value("trigger_value"),
            context_data=sample_km_event.payload
        )
    
    @pytest.mark.asyncio
    async def test_event_routing_trigger_not_found(
        self, 
        event_router, 
        mock_trigger_manager, 
        sample_km_event
    ):
        """Test event routing when trigger is not found."""
        # Setup mocks - trigger not found
        trigger_state = Mock()
        trigger_state.get_trigger.return_value = None
        mock_trigger_manager.get_current_state.return_value = trigger_state
        
        # Route event
        result = await event_router.route_event(sample_km_event)
        
        # Verify failure
        assert result.is_left()
        assert "not found" in result.get_left().message
    
    @pytest.mark.asyncio
    async def test_event_routing_with_rules(
        self, 
        event_router, 
        mock_macro_engine, 
        mock_trigger_manager, 
        sample_trigger_info, 
        sample_km_event
    ):
        """Test event routing with custom routing rules."""
        # Add routing rule
        hotkey_rule = create_hotkey_routing_rule()
        event_router.add_routing_rule(hotkey_rule)
        
        # Setup mocks
        trigger_state = Mock()
        trigger_state.get_trigger.return_value = sample_trigger_info
        mock_trigger_manager.get_current_state.return_value = trigger_state
        
        execution_result = Mock()
        execution_result.status.value = "completed"
        mock_macro_engine.execute_macro_async.return_value = execution_result
        
        mock_trigger_manager._state.with_triggered.return_value = Mock()
        
        # Route event
        result = await event_router.route_event(sample_km_event)
        
        # Verify success
        assert result.is_right()
        
        # Verify rule matched (hotkey rule should match hotkey event)
        assert len(event_router._routing_rules) == 1
    
    def test_routing_rule_creation(self):
        """Test built-in routing rule creation."""
        hotkey_rule = create_hotkey_routing_rule()
        app_rule = create_application_routing_rule()
        
        # Test hotkey rule
        assert hotkey_rule.priority == 100
        
        # Test application rule
        assert app_rule.priority == 90
        
        # Test rule ordering (higher priority first)
        rules = [app_rule, hotkey_rule]
        rules.sort(key=lambda r: r.priority, reverse=True)
        assert rules[0] == hotkey_rule  # Higher priority first


class TestTriggerDefinition:
    """Test trigger definition functionality."""
    
    def test_trigger_definition_creation(self):
        """Test basic trigger definition creation."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test-trigger"),
            macro_id=MacroId("test-macro"),
            trigger_type=TriggerType.HOTKEY,
            configuration={"key": "F1", "modifiers": ["Command"]},
            name="Test Hotkey"
        )
        
        assert trigger_def.trigger_id == "test-trigger"
        assert trigger_def.macro_id == "test-macro"
        assert trigger_def.trigger_type == TriggerType.HOTKEY
        assert trigger_def.enabled is True  # Default value
    
    def test_trigger_definition_conversion(self):
        """Test conversion to KM format."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test-trigger"),
            macro_id=MacroId("test-macro"),
            trigger_type=TriggerType.APPLICATION,
            configuration={"app": "TextEdit"},
            name="App Trigger"
        )
        
        km_format = trigger_def.to_km_format()
        
        assert km_format["triggerId"] == "test-trigger"
        assert km_format["macroId"] == "test-macro"
        assert km_format["type"] == "application"
        assert km_format["config"]["app"] == "TextEdit"
        assert km_format["name"] == "App Trigger"
        assert km_format["enabled"] is True
    
    def test_trigger_definition_km_client_conversion(self):
        """Test conversion to KMClient TriggerDefinition."""
        trigger_def = TriggerDefinition(
            trigger_id=TriggerId("test-trigger"),
            macro_id=MacroId("test-macro"),
            trigger_type=TriggerType.HOTKEY,
            configuration={"key": "F2"},
            enabled=False
        )
        
        km_trigger_def = trigger_def.to_km_trigger_definition()
        
        assert km_trigger_def.trigger_id == "test-trigger"
        assert km_trigger_def.macro_id == "test-macro"
        assert km_trigger_def.trigger_type == TriggerType.HOTKEY
        assert km_trigger_def.configuration["key"] == "F2"
        assert km_trigger_def.enabled is False


# Property-based tests for trigger management

@given(
    trigger_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd', 'Pc'])),
    macro_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd', 'Pc'])),
    key_name=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd']))
)
def test_trigger_definition_properties(trigger_id, macro_id, key_name):
    """Property-based test for trigger definition invariants."""
    assume(len(trigger_id.strip()) > 0)
    assume(len(macro_id.strip()) > 0)
    assume(len(key_name.strip()) > 0)
    
    # Create trigger definition
    trigger_def = TriggerDefinition(
        trigger_id=TriggerId(trigger_id),
        macro_id=MacroId(macro_id),
        trigger_type=TriggerType.HOTKEY,
        configuration={"key": key_name}
    )
    
    # Test invariants
    assert trigger_def.trigger_id == trigger_id
    assert trigger_def.macro_id == macro_id
    assert trigger_def.trigger_type == TriggerType.HOTKEY
    assert "key" in trigger_def.configuration
    assert trigger_def.enabled is True  # Default value
    
    # Test conversions preserve data
    km_format = trigger_def.to_km_format()
    assert km_format["triggerId"] == trigger_id
    assert km_format["macroId"] == macro_id
    assert km_format["config"]["key"] == key_name
    
    km_trigger_def = trigger_def.to_km_trigger_definition()
    assert km_trigger_def.trigger_id == trigger_id
    assert km_trigger_def.macro_id == macro_id
    assert km_trigger_def.configuration["key"] == key_name


@given(
    payload_data=st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd', 'Pc'])),
        values=st.one_of(
            st.text(min_size=0, max_size=100, alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Nd', 'Pc', 'Zs'])),
            st.integers(min_value=-1000, max_value=1000),
            st.booleans()
        ),
        min_size=0,
        max_size=10
    )
)
def test_km_event_payload_properties(payload_data):
    """Property-based test for KM event payload handling."""
    # Create event with arbitrary payload
    event = KMEvent.create(
        trigger_type=TriggerType.HOTKEY,
        trigger_id=TriggerId("test-trigger"),
        payload=payload_data
    )
    
    # Test payload preservation
    assert event.payload == payload_data
    
    # Test payload access methods
    for key, value in payload_data.items():
        assert event.get_payload_value(key) == value
    
    # Test default value handling
    assert event.get_payload_value("nonexistent_key", "default") == "default"
    
    # Test payload modification creates new event
    if payload_data:
        key = list(payload_data.keys())[0]
        modified_event = event.with_payload(key, "modified_value")
        assert modified_event.get_payload_value(key) == "modified_value"
        assert event.get_payload_value(key) != "modified_value"  # Original unchanged


if __name__ == "__main__":
    # Run basic smoke test
    import sys
    
    try:
        # Test imports
        from src.integration.triggers import TriggerRegistrationManager, EventRouter
        from src.integration.events import KMEvent, TriggerType
        from src.integration.km_client import KMClient, ConnectionConfig
        
        # Test basic functionality
        config = ConnectionConfig()
        client = KMClient(config)
        manager = TriggerRegistrationManager(client)
        
        print("✅ All trigger management components imported successfully")
        print("✅ Basic object creation works")
        print("✅ Integration test ready for execution")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        sys.exit(1)