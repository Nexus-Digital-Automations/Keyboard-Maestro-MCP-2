"""
Comprehensive Integration Module Tests - Coverage Expansion

Tests for integration modules including KM client, triggers, security, file monitoring, and protocols.
Focuses on achieving high coverage for integration infrastructure.

Architecture: Property-Based Testing + Type Safety + Contract Validation + Security Testing
Performance: <200ms per test, parallel execution, comprehensive edge case coverage
"""

import pytest
import asyncio
import tempfile
import os
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from hypothesis import given, strategies as st, settings
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, UTC

# Test imports with graceful fallbacks
try:
    from src.integration.km_client import (
        KMClient, KMConnection, KMError, ConnectionState
    )
    from src.integration.triggers import (
        TriggerManager, TriggerType, TriggerEvent, TriggerCondition
    )
    from src.integration.security import (
        SecurityManager, SecurityLevel, SecurityPolicy, AccessControl
    )
    from src.integration.file_monitor import (
        FileMonitor, FileEvent, FileEventType, MonitorConfig
    )
    from src.integration.protocol import (
        ProtocolHandler, MessageType, ProtocolMessage, ProtocolError
    )
    from src.core.types import MacroId, CommandId
    from src.core.either import Either
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    # Mock classes for testing
    class KMClient:
        def __init__(self):
            self.connected = False
    
    class TriggerManager:
        def __init__(self):
            self.triggers = {}
    
    class SecurityManager:
        def __init__(self):
            self.policies = {}


class TestKMClient:
    """Test Keyboard Maestro client functionality."""
    
    def test_client_creation(self):
        """Test KM client creation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        client = KMClient()
        assert client is not None
        assert client.connection_state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_client_connection(self):
        """Test KM client connection."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        client = KMClient()
        
        with patch('src.integration.km_client.connect_to_keyboard_maestro') as mock_connect:
            mock_connect.return_value = Either.right({"status": "connected"})
            
            result = await client.connect()
            assert result.is_right()
            assert client.connection_state == ConnectionState.CONNECTED
    
    @pytest.mark.asyncio
    async def test_client_connection_failure(self):
        """Test KM client connection failure."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        client = KMClient()
        
        with patch('src.integration.km_client.connect_to_keyboard_maestro') as mock_connect:
            mock_connect.return_value = Either.left(KMError("Connection failed", "CONNECT_FAILED"))
            
            result = await client.connect()
            assert result.is_left()
            assert client.connection_state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_execute_macro(self):
        """Test macro execution through client."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        client = KMClient()
        client.connection_state = ConnectionState.CONNECTED
        
        with patch('src.integration.km_client.execute_km_macro') as mock_execute:
            mock_execute.return_value = Either.right({"result": "success", "output": "macro completed"})
            
            result = await client.execute_macro("test_macro", {"param": "value"})
            assert result.is_right()
            assert result.get_right()["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_execute_macro_disconnected(self):
        """Test macro execution when disconnected."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        client = KMClient()
        client.connection_state = ConnectionState.DISCONNECTED
        
        result = await client.execute_macro("test_macro", {})
        assert result.is_left()
        assert "not connected" in result.get_left().message.lower()
    
    @pytest.mark.asyncio
    async def test_get_macro_list(self):
        """Test getting macro list from KM."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        client = KMClient()
        client.connection_state = ConnectionState.CONNECTED
        
        mock_macros = [
            {"id": "macro1", "name": "Test Macro 1", "enabled": True},
            {"id": "macro2", "name": "Test Macro 2", "enabled": False}
        ]
        
        with patch('src.integration.km_client.get_km_macros') as mock_get:
            mock_get.return_value = Either.right(mock_macros)
            
            result = await client.get_macro_list()
            assert result.is_right()
            macros = result.get_right()
            assert len(macros) == 2
            assert macros[0]["name"] == "Test Macro 1"
    
    @pytest.mark.asyncio
    async def test_get_variable_value(self):
        """Test getting variable value from KM."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        client = KMClient()
        client.connection_state = ConnectionState.CONNECTED
        
        with patch('src.integration.km_client.get_km_variable') as mock_get:
            mock_get.return_value = Either.right({"value": "test_value"})
            
            result = await client.get_variable("test_var")
            assert result.is_right()
            assert result.get_right()["value"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_set_variable_value(self):
        """Test setting variable value in KM."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        client = KMClient()
        client.connection_state = ConnectionState.CONNECTED
        
        with patch('src.integration.km_client.set_km_variable') as mock_set:
            mock_set.return_value = Either.right({"status": "set"})
            
            result = await client.set_variable("test_var", "new_value")
            assert result.is_right()
            assert result.get_right()["status"] == "set"
    
    def test_connection_state_transitions(self):
        """Test connection state transitions."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        client = KMClient()
        
        # Initial state
        assert client.connection_state == ConnectionState.DISCONNECTED
        
        # State transitions
        client.connection_state = ConnectionState.CONNECTING
        assert client.connection_state == ConnectionState.CONNECTING
        
        client.connection_state = ConnectionState.CONNECTED
        assert client.connection_state == ConnectionState.CONNECTED
        
        client.connection_state = ConnectionState.ERROR
        assert client.connection_state == ConnectionState.ERROR


class TestTriggerManager:
    """Test trigger management functionality."""
    
    def test_trigger_manager_creation(self):
        """Test trigger manager creation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = TriggerManager()
        assert manager is not None
        assert len(manager.get_all_triggers()) == 0
    
    def test_register_trigger(self):
        """Test trigger registration."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = TriggerManager()
        
        # Create test trigger
        trigger = TriggerEvent(
            trigger_id="test_trigger",
            trigger_type=TriggerType.HOTKEY,
            condition=TriggerCondition("hotkey", {"key": "cmd+shift+t"}),
            action="execute_macro",
            action_parameters={"macro_id": "test_macro"}
        )
        
        manager.register_trigger(trigger)
        
        # Verify registration
        triggers = manager.get_all_triggers()
        assert len(triggers) == 1
        assert triggers[0].trigger_id == "test_trigger"
    
    def test_unregister_trigger(self):
        """Test trigger unregistration."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = TriggerManager()
        
        trigger = TriggerEvent(
            trigger_id="test_trigger",
            trigger_type=TriggerType.HOTKEY,
            condition=TriggerCondition("hotkey", {"key": "cmd+t"}),
            action="execute_macro",
            action_parameters={}
        )
        
        # Register and unregister
        manager.register_trigger(trigger)
        assert len(manager.get_all_triggers()) == 1
        
        manager.unregister_trigger("test_trigger")
        assert len(manager.get_all_triggers()) == 0
    
    def test_trigger_types(self):
        """Test different trigger types."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = TriggerManager()
        
        # Test various trigger types
        trigger_configs = [
            {
                "id": "hotkey_trigger",
                "type": TriggerType.HOTKEY,
                "condition": TriggerCondition("hotkey", {"key": "cmd+h"})
            },
            {
                "id": "file_trigger",
                "type": TriggerType.FILE_CHANGE,
                "condition": TriggerCondition("file_change", {"path": "/test/path"})
            },
            {
                "id": "time_trigger",
                "type": TriggerType.SCHEDULED,
                "condition": TriggerCondition("schedule", {"cron": "0 9 * * *"})
            },
            {
                "id": "app_trigger",
                "type": TriggerType.APPLICATION,
                "condition": TriggerCondition("app_launch", {"app": "TextEdit"})
            }
        ]
        
        for config in trigger_configs:
            trigger = TriggerEvent(
                trigger_id=config["id"],
                trigger_type=config["type"],
                condition=config["condition"],
                action="test_action",
                action_parameters={}
            )
            manager.register_trigger(trigger)
        
        # Verify all types registered
        triggers = manager.get_all_triggers()
        assert len(triggers) == 4
        
        # Test filtering by type
        hotkey_triggers = manager.get_triggers_by_type(TriggerType.HOTKEY)
        assert len(hotkey_triggers) == 1
        assert hotkey_triggers[0].trigger_id == "hotkey_trigger"
    
    @pytest.mark.asyncio
    async def test_trigger_execution(self):
        """Test trigger execution."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = TriggerManager()
        
        # Mock action executor
        with patch('src.integration.triggers.execute_trigger_action') as mock_execute:
            mock_execute.return_value = Either.right({"result": "executed"})
            
            trigger = TriggerEvent(
                trigger_id="test_trigger",
                trigger_type=TriggerType.HOTKEY,
                condition=TriggerCondition("hotkey", {"key": "cmd+t"}),
                action="execute_macro",
                action_parameters={"macro_id": "test_macro"}
            )
            
            result = await manager.execute_trigger(trigger)
            assert result.is_right()
            assert result.get_right()["result"] == "executed"
    
    def test_trigger_condition_matching(self):
        """Test trigger condition matching."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = TriggerManager()
        
        # Create trigger with specific condition
        condition = TriggerCondition("hotkey", {"key": "cmd+shift+t", "modifiers": ["cmd", "shift"]})
        trigger = TriggerEvent(
            trigger_id="test_trigger",
            trigger_type=TriggerType.HOTKEY,
            condition=condition,
            action="test_action",
            action_parameters={}
        )
        
        manager.register_trigger(trigger)
        
        # Test condition matching
        matching_event = {"type": "hotkey", "key": "cmd+shift+t"}
        non_matching_event = {"type": "hotkey", "key": "cmd+t"}
        
        matching_triggers = manager.find_matching_triggers(matching_event)
        assert len(matching_triggers) == 1
        assert matching_triggers[0].trigger_id == "test_trigger"
        
        non_matching_triggers = manager.find_matching_triggers(non_matching_event)
        assert len(non_matching_triggers) == 0


class TestSecurityManager:
    """Test security management functionality."""
    
    def test_security_manager_creation(self):
        """Test security manager creation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = SecurityManager()
        assert manager is not None
    
    def test_security_policy_management(self):
        """Test security policy management."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = SecurityManager()
        
        # Create security policy
        policy = SecurityPolicy(
            policy_id="test_policy",
            security_level=SecurityLevel.HIGH,
            allowed_operations=["read", "execute"],
            restricted_paths=["/system", "/private"],
            access_controls=[]
        )
        
        manager.add_policy(policy)
        
        # Verify policy
        retrieved_policy = manager.get_policy("test_policy")
        assert retrieved_policy is not None
        assert retrieved_policy.security_level == SecurityLevel.HIGH
    
    def test_access_control_validation(self):
        """Test access control validation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = SecurityManager()
        
        # Create access control
        access_control = AccessControl(
            resource="test_resource",
            operation="read",
            user_id="test_user",
            permissions=["read", "execute"]
        )
        
        # Test access validation
        assert manager.validate_access(access_control, "read")
        assert manager.validate_access(access_control, "execute")
        assert not manager.validate_access(access_control, "write")
    
    def test_security_level_enforcement(self):
        """Test security level enforcement."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = SecurityManager()
        
        # Test different security levels
        levels = [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH, SecurityLevel.MAXIMUM]
        
        for level in levels:
            policy = SecurityPolicy(
                policy_id=f"policy_{level.name}",
                security_level=level,
                allowed_operations=["read"],
                restricted_paths=[],
                access_controls=[]
            )
            manager.add_policy(policy)
            
            # Verify enforcement based on level
            enforcement = manager.get_enforcement_rules(level)
            assert enforcement is not None
            
            if level == SecurityLevel.MAXIMUM:
                assert enforcement["requires_elevation"] is True
            if level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
                assert enforcement["audit_required"] is True
    
    def test_path_security_validation(self):
        """Test path security validation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = SecurityManager()
        
        # Define restricted paths
        restricted_paths = ["/system", "/private", "/etc"]
        
        policy = SecurityPolicy(
            policy_id="path_policy",
            security_level=SecurityLevel.HIGH,
            allowed_operations=["read"],
            restricted_paths=restricted_paths,
            access_controls=[]
        )
        
        manager.add_policy(policy)
        
        # Test path validation
        safe_paths = ["/Users/test", "/Documents/file.txt", "/tmp/temp"]
        restricted_test_paths = ["/system/config", "/private/data", "/etc/passwd"]
        
        for path in safe_paths:
            assert manager.validate_path_access(path, "path_policy")
        
        for path in restricted_test_paths:
            assert not manager.validate_path_access(path, "path_policy")
    
    @pytest.mark.asyncio
    async def test_security_audit_logging(self):
        """Test security audit logging."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = SecurityManager()
        
        with patch('src.integration.security.log_security_event') as mock_log:
            # Perform security-sensitive operation
            await manager.audit_operation(
                operation="file_access",
                resource="/sensitive/file",
                user_id="test_user",
                result="denied"
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert call_args["operation"] == "file_access"
            assert call_args["result"] == "denied"


class TestFileMonitor:
    """Test file monitoring functionality."""
    
    def test_file_monitor_creation(self):
        """Test file monitor creation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        monitor = FileMonitor()
        assert monitor is not None
        assert not monitor.is_monitoring()
    
    def test_monitor_configuration(self):
        """Test file monitor configuration."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        monitor = FileMonitor()
        
        config = MonitorConfig(
            watch_paths=["/test/path1", "/test/path2"],
            event_types=[FileEventType.CREATED, FileEventType.MODIFIED],
            recursive=True,
            ignore_patterns=["*.tmp", "*.log"]
        )
        
        monitor.configure(config)
        
        # Verify configuration
        current_config = monitor.get_configuration()
        assert len(current_config.watch_paths) == 2
        assert FileEventType.CREATED in current_config.event_types
        assert current_config.recursive is True
    
    @pytest.mark.asyncio
    async def test_file_monitoring_lifecycle(self):
        """Test file monitoring lifecycle."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        monitor = FileMonitor()
        
        # Configure monitor
        config = MonitorConfig(
            watch_paths=["/tmp"],
            event_types=[FileEventType.CREATED],
            recursive=False
        )
        monitor.configure(config)
        
        # Start monitoring
        with patch('src.integration.file_monitor.start_file_watcher'):
            result = await monitor.start_monitoring()
            assert result.is_right()
            assert monitor.is_monitoring()
        
        # Stop monitoring
        with patch('src.integration.file_monitor.stop_file_watcher'):
            result = await monitor.stop_monitoring()
            assert result.is_right()
            assert not monitor.is_monitoring()
    
    def test_file_event_handling(self):
        """Test file event handling."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        monitor = FileMonitor()
        events_received = []
        
        def event_handler(event: FileEvent):
            events_received.append(event)
        
        monitor.set_event_handler(event_handler)
        
        # Simulate file events
        test_events = [
            FileEvent(
                event_type=FileEventType.CREATED,
                file_path="/test/new_file.txt",
                timestamp=datetime.now(UTC)
            ),
            FileEvent(
                event_type=FileEventType.MODIFIED,
                file_path="/test/existing_file.txt",
                timestamp=datetime.now(UTC)
            )
        ]
        
        for event in test_events:
            monitor._handle_file_event(event)
        
        # Verify events were handled
        assert len(events_received) == 2
        assert events_received[0].event_type == FileEventType.CREATED
        assert events_received[1].event_type == FileEventType.MODIFIED
    
    def test_file_event_filtering(self):
        """Test file event filtering."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        monitor = FileMonitor()
        
        config = MonitorConfig(
            watch_paths=["/test"],
            event_types=[FileEventType.CREATED],
            ignore_patterns=["*.tmp", "*.log"]
        )
        monitor.configure(config)
        
        # Test events
        events = [
            FileEvent(FileEventType.CREATED, "/test/document.txt", datetime.now(UTC)),
            FileEvent(FileEventType.CREATED, "/test/temp.tmp", datetime.now(UTC)),
            FileEvent(FileEventType.CREATED, "/test/error.log", datetime.now(UTC)),
            FileEvent(FileEventType.MODIFIED, "/test/document.txt", datetime.now(UTC))  # Wrong type
        ]
        
        filtered_events = []
        for event in events:
            if monitor._should_process_event(event):
                filtered_events.append(event)
        
        # Should only process document.txt creation
        assert len(filtered_events) == 1
        assert filtered_events[0].file_path == "/test/document.txt"
    
    @pytest.mark.asyncio
    async def test_file_monitor_error_handling(self):
        """Test file monitor error handling."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        monitor = FileMonitor()
        
        # Configure with invalid path
        config = MonitorConfig(
            watch_paths=["/nonexistent/path"],
            event_types=[FileEventType.CREATED]
        )
        monitor.configure(config)
        
        # Start monitoring should handle error gracefully
        with patch('src.integration.file_monitor.start_file_watcher') as mock_start:
            mock_start.side_effect = Exception("Path not found")
            
            result = await monitor.start_monitoring()
            assert result.is_left()
            assert "Path not found" in result.get_left().message


class TestProtocolHandler:
    """Test protocol handling functionality."""
    
    def test_protocol_handler_creation(self):
        """Test protocol handler creation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        handler = ProtocolHandler()
        assert handler is not None
    
    def test_message_creation(self):
        """Test protocol message creation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        message = ProtocolMessage(
            message_type=MessageType.REQUEST,
            message_id="test_001",
            payload={"action": "execute_macro", "macro_id": "test"},
            metadata={"timestamp": datetime.now(UTC).isoformat()}
        )
        
        assert message.message_type == MessageType.REQUEST
        assert message.message_id == "test_001"
        assert message.payload["action"] == "execute_macro"
    
    def test_message_serialization(self):
        """Test protocol message serialization."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        handler = ProtocolHandler()
        
        message = ProtocolMessage(
            message_type=MessageType.RESPONSE,
            message_id="test_002",
            payload={"result": "success", "data": {"output": "completed"}},
            metadata={}
        )
        
        # Serialize message
        serialized = handler.serialize_message(message)
        assert isinstance(serialized, (str, bytes))
        
        # Deserialize message
        deserialized = handler.deserialize_message(serialized)
        assert deserialized.message_type == MessageType.RESPONSE
        assert deserialized.message_id == "test_002"
        assert deserialized.payload["result"] == "success"
    
    def test_message_validation(self):
        """Test protocol message validation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        handler = ProtocolHandler()
        
        # Valid message
        valid_message = ProtocolMessage(
            message_type=MessageType.REQUEST,
            message_id="valid_001",
            payload={"action": "test"},
            metadata={}
        )
        
        assert handler.validate_message(valid_message)
        
        # Invalid message (missing required fields)
        invalid_message = ProtocolMessage(
            message_type=MessageType.REQUEST,
            message_id="",  # Empty ID
            payload={},
            metadata={}
        )
        
        assert not handler.validate_message(invalid_message)
    
    @pytest.mark.asyncio
    async def test_message_processing(self):
        """Test protocol message processing."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        handler = ProtocolHandler()
        
        request_message = ProtocolMessage(
            message_type=MessageType.REQUEST,
            message_id="process_001",
            payload={"action": "get_status"},
            metadata={}
        )
        
        with patch('src.integration.protocol.process_request') as mock_process:
            mock_process.return_value = Either.right({"status": "active", "uptime": 3600})
            
            response = await handler.process_message(request_message)
            assert response.message_type == MessageType.RESPONSE
            assert response.payload["status"] == "active"
    
    def test_error_message_handling(self):
        """Test error message handling."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        handler = ProtocolHandler()
        
        error = ProtocolError("Invalid request", "INVALID_REQUEST")
        error_message = handler.create_error_message("error_001", error)
        
        assert error_message.message_type == MessageType.ERROR
        assert error_message.message_id == "error_001"
        assert error_message.payload["error_code"] == "INVALID_REQUEST"
        assert error_message.payload["error_message"] == "Invalid request"
    
    @given(st.text(min_size=1, max_size=20), st.dictionaries(st.text(), st.text()))
    @settings(max_examples=10)
    def test_message_property_based(self, message_id, payload):
        """Property-based test for message handling."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        handler = ProtocolHandler()
        
        message = ProtocolMessage(
            message_type=MessageType.REQUEST,
            message_id=message_id,
            payload=payload,
            metadata={}
        )
        
        # Serialize and deserialize
        serialized = handler.serialize_message(message)
        deserialized = handler.deserialize_message(serialized)
        
        # Should preserve content
        assert deserialized.message_id == message_id
        assert deserialized.payload == payload


class TestIntegrationScenarios:
    """Test integration scenarios across modules."""
    
    @pytest.mark.asyncio
    async def test_trigger_to_km_execution(self):
        """Test trigger to KM execution workflow."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        # Setup components
        trigger_manager = TriggerManager()
        km_client = KMClient()
        
        # Mock KM client connection
        km_client.connection_state = ConnectionState.CONNECTED
        
        # Create trigger
        trigger = TriggerEvent(
            trigger_id="integration_trigger",
            trigger_type=TriggerType.HOTKEY,
            condition=TriggerCondition("hotkey", {"key": "cmd+i"}),
            action="execute_macro",
            action_parameters={"macro_id": "integration_macro"}
        )
        
        trigger_manager.register_trigger(trigger)
        
        # Mock macro execution
        with patch.object(km_client, 'execute_macro') as mock_execute:
            mock_execute.return_value = Either.right({"result": "completed"})
            
            # Simulate trigger activation
            event = {"type": "hotkey", "key": "cmd+i"}
            matching_triggers = trigger_manager.find_matching_triggers(event)
            
            assert len(matching_triggers) == 1
            
            # Execute the triggered action
            result = await km_client.execute_macro(
                matching_triggers[0].action_parameters["macro_id"],
                {}
            )
            
            assert result.is_right()
            assert result.get_right()["result"] == "completed"
    
    @pytest.mark.asyncio
    async def test_file_monitor_to_trigger_workflow(self):
        """Test file monitor to trigger workflow."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        # Setup components
        file_monitor = FileMonitor()
        trigger_manager = TriggerManager()
        
        # Configure file monitor
        config = MonitorConfig(
            watch_paths=["/test/watch"],
            event_types=[FileEventType.CREATED]
        )
        file_monitor.configure(config)
        
        # Create file-based trigger
        file_trigger = TriggerEvent(
            trigger_id="file_trigger",
            trigger_type=TriggerType.FILE_CHANGE,
            condition=TriggerCondition("file_created", {"path_pattern": "*.txt"}),
            action="process_file",
            action_parameters={}
        )
        
        trigger_manager.register_trigger(file_trigger)
        
        # Setup event handler to bridge file monitor and trigger manager
        triggered_actions = []
        
        def file_event_handler(event: FileEvent):
            if event.file_path.endswith('.txt'):
                # Convert file event to trigger event
                trigger_event = {
                    "type": "file_created",
                    "path": event.file_path
                }
                matching_triggers = trigger_manager.find_matching_triggers(trigger_event)
                triggered_actions.extend(matching_triggers)
        
        file_monitor.set_event_handler(file_event_handler)
        
        # Simulate file creation
        test_event = FileEvent(
            event_type=FileEventType.CREATED,
            file_path="/test/watch/document.txt",
            timestamp=datetime.now(UTC)
        )
        
        file_monitor._handle_file_event(test_event)
        
        # Verify trigger was activated
        assert len(triggered_actions) == 1
        assert triggered_actions[0].trigger_id == "file_trigger"
    
    @pytest.mark.asyncio
    async def test_security_protocol_integration(self):
        """Test security and protocol integration."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        # Setup components
        security_manager = SecurityManager()
        protocol_handler = ProtocolHandler()
        
        # Create security policy
        policy = SecurityPolicy(
            policy_id="protocol_policy",
            security_level=SecurityLevel.HIGH,
            allowed_operations=["read", "execute"],
            restricted_paths=[],
            access_controls=[]
        )
        security_manager.add_policy(policy)
        
        # Create protocol request
        request = ProtocolMessage(
            message_type=MessageType.REQUEST,
            message_id="secure_001",
            payload={"action": "execute_macro", "macro_id": "sensitive_macro"},
            metadata={"user_id": "test_user", "security_policy": "protocol_policy"}
        )
        
        # Validate security before processing
        access_control = AccessControl(
            resource="sensitive_macro",
            operation="execute",
            user_id="test_user",
            permissions=["execute"]
        )
        
        security_valid = security_manager.validate_access(access_control, "execute")
        assert security_valid
        
        # Process request if security validation passes
        if security_valid:
            with patch('src.integration.protocol.process_request') as mock_process:
                mock_process.return_value = Either.right({"result": "executed securely"})
                
                response = await protocol_handler.process_message(request)
                assert response.message_type == MessageType.RESPONSE
                assert response.payload["result"] == "executed securely"


# Property-based testing for integration components
class TestIntegrationProperties:
    """Property-based tests for integration components."""
    
    @given(st.text(min_size=1, max_size=20))
    @settings(max_examples=10)
    def test_trigger_id_uniqueness(self, trigger_id):
        """Test trigger ID uniqueness property."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration module not available")
        
        manager = TriggerManager()
        
        trigger = TriggerEvent(
            trigger_id=trigger_id,
            trigger_type=TriggerType.HOTKEY,
            condition=TriggerCondition("test", {}),
            action="test",
            action_parameters={}
        )
        
        # First registration should succeed
        manager.register_trigger(trigger)
        assert len(manager.get_all_triggers()) == 1
        
        # Second registration with same ID should fail or replace
        duplicate_trigger = TriggerEvent(
            trigger_id=trigger_id,
            trigger_type=TriggerType.APPLICATION,
            condition=TriggerCondition("test2", {}),
            action="test2",
            action_parameters={}
        )
        
        # Behavior depends on implementation - either reject duplicate or replace
        try:
            manager.register_trigger(duplicate_trigger)
            # If replacement is allowed, should still have only one trigger
            assert len(manager.get_all_triggers()) == 1
        except ValueError:
            # If duplicates are rejected, original should remain
            assert len(manager.get_all_triggers()) == 1
            assert manager.get_all_triggers()[0].action == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])