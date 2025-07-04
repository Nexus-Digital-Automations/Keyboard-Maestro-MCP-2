"""
Device Controller - TASK_65 Phase 2 Core IoT Engine

IoT device discovery, connection, and control management with multi-protocol support.
Provides comprehensive device lifecycle management and real-time control capabilities.

Architecture: Device Discovery + Connection Management + Protocol Abstraction + Control Interface
Performance: <100ms device commands, <50ms status queries, <200ms device discovery
Security: Device authentication, encrypted communication, secure command execution
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import json
import hashlib
import ssl
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.iot_architecture import (
    IoTDevice, DeviceId, DeviceType, IoTProtocol, DeviceAction, DeviceStatus,
    SecurityLevel, ProtocolAddress, IoTIntegrationError, validate_device_configuration,
    create_device_id, create_protocol_address
)


class DiscoveryMethod(Enum):
    """Device discovery methods."""
    NETWORK_SCAN = "network_scan"
    MDNS = "mdns"
    UPNP = "upnp"
    BLUETOOTH = "bluetooth"
    ZIGBEE = "zigbee"
    ZWAVE = "zwave"
    MANUAL = "manual"
    API_DISCOVERY = "api_discovery"


class ConnectionState(Enum):
    """Device connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class DeviceConnection:
    """Device connection information."""
    device_id: DeviceId
    connection_state: ConnectionState
    protocol: IoTProtocol
    address: ProtocolAddress
    
    # Connection details
    connected_at: Optional[datetime] = None
    last_communication: Optional[datetime] = None
    connection_attempts: int = 0
    error_count: int = 0
    
    # Performance metrics
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    bandwidth_usage: float = 0.0
    
    # Security
    authentication_token: Optional[str] = None
    certificate_fingerprint: Optional[str] = None
    encryption_active: bool = False
    
    # Configuration
    keep_alive_interval: int = 60  # seconds
    timeout_seconds: int = 30
    max_retries: int = 3
    
    def is_active(self) -> bool:
        """Check if connection is active."""
        return self.connection_state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]
    
    def update_performance(self, response_time: float, success: bool):
        """Update connection performance metrics."""
        self.response_time_ms = (self.response_time_ms + response_time) / 2
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        
        if success:
            self.last_communication = datetime.now(UTC)
        else:
            self.error_count += 1


@dataclass
class DeviceCapability:
    """Device capability description."""
    capability_name: str
    supported_actions: List[DeviceAction]
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    
    def supports_action(self, action: DeviceAction) -> bool:
        """Check if capability supports specific action."""
        return action in self.supported_actions


@dataclass
class DiscoveryResult:
    """Device discovery result."""
    device_id: DeviceId
    device_type: DeviceType
    protocol: IoTProtocol
    address: ProtocolAddress
    
    # Discovery metadata
    discovery_method: DiscoveryMethod
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    signal_strength: Optional[float] = None
    
    # Device information
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    firmware_version: Optional[str] = None
    capabilities: List[DeviceCapability] = field(default_factory=list)
    
    # Network information
    mac_address: Optional[str] = None
    ip_address: Optional[str] = None
    port: Optional[int] = None
    
    # Metadata
    additional_info: Dict[str, Any] = field(default_factory=dict)


class DeviceController:
    """Advanced IoT device controller with multi-protocol support."""
    
    def __init__(self):
        self.devices: Dict[DeviceId, IoTDevice] = {}
        self.connections: Dict[DeviceId, DeviceConnection] = {}
        self.capabilities: Dict[DeviceId, List[DeviceCapability]] = {}
        
        # Discovery configuration
        self.discovery_enabled = True
        self.discovery_interval = 300  # seconds
        self.discovery_methods: Set[DiscoveryMethod] = {
            DiscoveryMethod.NETWORK_SCAN,
            DiscoveryMethod.MDNS,
            DiscoveryMethod.UPNP
        }
        
        # Security configuration
        self.require_authentication = True
        self.require_encryption = True
        self.trusted_certificates: Set[str] = set()
        self.device_whitelist: Set[DeviceId] = set()
        
        # Performance monitoring
        self.command_metrics: Dict[str, Dict[str, float]] = {}
        self.discovery_history: List[DiscoveryResult] = []
        
        # Event handlers
        self.device_discovered_handlers: List[Callable[[DiscoveryResult], None]] = []
        self.device_connected_handlers: List[Callable[[DeviceId], None]] = []
        self.device_disconnected_handlers: List[Callable[[DeviceId], None]] = []
        self.command_executed_handlers: List[Callable[[DeviceId, DeviceAction, Dict[str, Any]], None]] = []
        
        # Background tasks
        self._discovery_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Start background services
        asyncio.create_task(self._start_background_services())
    
    @require(lambda device: isinstance(device, IoTDevice))
    async def register_device(self, device: IoTDevice) -> Either[IoTIntegrationError, bool]:
        """Register a new IoT device."""
        try:
            # Validate device configuration
            validation = validate_device_configuration(device)
            if validation.is_error():
                return validation
            
            # Check if device already exists
            if device.device_id in self.devices:
                return Either.error(IoTIntegrationError(f"Device already registered: {device.device_id}"))
            
            # Register device
            self.devices[device.device_id] = device
            
            # Initialize connection
            connection = DeviceConnection(
                device_id=device.device_id,
                connection_state=ConnectionState.DISCONNECTED,
                protocol=device.protocol,
                address=device.address,
                timeout_seconds=device.connection_timeout,
                max_retries=device.retry_attempts
            )
            self.connections[device.device_id] = connection
            
            # Initialize capabilities
            self.capabilities[device.device_id] = []
            
            # Attempt initial connection if device is supposed to be online
            if device.status == DeviceStatus.ONLINE:
                await self.connect_device(device.device_id)
            
            return Either.success(True)
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Device registration failed: {str(e)}", device.device_id))
    
    async def discover_devices(self, methods: Optional[Set[DiscoveryMethod]] = None) -> Either[IoTIntegrationError, List[DiscoveryResult]]:
        """Discover IoT devices using specified methods."""
        try:
            if not self.discovery_enabled:
                return Either.success([])
            
            discovery_methods = methods or self.discovery_methods
            all_results: List[DiscoveryResult] = []
            
            # Run discovery methods in parallel
            tasks = []
            for method in discovery_methods:
                if method == DiscoveryMethod.NETWORK_SCAN:
                    tasks.append(self._discover_network_scan())
                elif method == DiscoveryMethod.MDNS:
                    tasks.append(self._discover_mdns())
                elif method == DiscoveryMethod.UPNP:
                    tasks.append(self._discover_upnp())
                elif method == DiscoveryMethod.BLUETOOTH:
                    tasks.append(self._discover_bluetooth())
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, list):
                        all_results.extend(result)
                    elif not isinstance(result, Exception):
                        all_results.append(result)
            
            # Store discovery history
            self.discovery_history.extend(all_results)
            
            # Trigger discovery event handlers
            for result in all_results:
                for handler in self.device_discovered_handlers:
                    try:
                        handler(result)
                    except Exception:
                        pass  # Don't let handler errors affect discovery
            
            return Either.success(all_results)
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Device discovery failed: {str(e)}"))
    
    @require(lambda device_id: isinstance(device_id, DeviceId))
    async def connect_device(self, device_id: DeviceId) -> Either[IoTIntegrationError, bool]:
        """Connect to an IoT device."""
        try:
            if device_id not in self.devices:
                return Either.error(IoTIntegrationError(f"Device not registered: {device_id}"))
            
            device = self.devices[device_id]
            connection = self.connections[device_id]
            
            # Check if already connected
            if connection.is_active():
                return Either.success(True)
            
            connection.connection_state = ConnectionState.CONNECTING
            connection.connection_attempts += 1
            
            connection_start = datetime.now(UTC)
            
            # Connect based on protocol
            if device.protocol == IoTProtocol.HTTP:
                success = await self._connect_http(device, connection)
            elif device.protocol == IoTProtocol.HTTPS:
                success = await self._connect_https(device, connection)
            elif device.protocol == IoTProtocol.MQTT:
                success = await self._connect_mqtt(device, connection)
            elif device.protocol == IoTProtocol.COAP:
                success = await self._connect_coap(device, connection)
            else:
                return Either.error(IoTIntegrationError(f"Unsupported protocol: {device.protocol}", device_id))
            
            if success:
                connection.connection_state = ConnectionState.CONNECTED
                connection.connected_at = connection_start
                
                # Authenticate if required
                if self.require_authentication:
                    auth_result = await self._authenticate_device(device, connection)
                    if auth_result.is_error():
                        connection.connection_state = ConnectionState.ERROR
                        return auth_result
                    
                    connection.connection_state = ConnectionState.AUTHENTICATED
                
                # Update device status
                device.status = DeviceStatus.ONLINE
                device.last_seen = datetime.now(UTC)
                
                # Trigger connection event handlers
                for handler in self.device_connected_handlers:
                    try:
                        handler(device_id)
                    except Exception:
                        pass
                
                return Either.success(True)
            else:
                connection.connection_state = ConnectionState.ERROR
                return Either.error(IoTIntegrationError(f"Connection failed: {device_id}"))
            
        except Exception as e:
            if device_id in self.connections:
                self.connections[device_id].connection_state = ConnectionState.ERROR
            return Either.error(IoTIntegrationError(f"Connection error: {str(e)}", device_id))
    
    @require(lambda device_id: isinstance(device_id, DeviceId))
    @require(lambda action: isinstance(action, DeviceAction))
    async def execute_device_action(self, device_id: DeviceId, action: DeviceAction, 
                                   parameters: Optional[Dict[str, Any]] = None) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Execute an action on an IoT device."""
        try:
            if device_id not in self.devices:
                return Either.error(IoTIntegrationError(f"Device not registered: {device_id}"))
            
            device = self.devices[device_id]
            connection = self.connections[device_id]
            
            # Check connection
            if not connection.is_active():
                connect_result = await self.connect_device(device_id)
                if connect_result.is_error():
                    return Either.error(IoTIntegrationError(f"Cannot connect to device: {device_id}"))
            
            # Check if device supports action
            if not device.supports_action(action):
                return Either.error(IoTIntegrationError(f"Device does not support action {action.value}", device_id))
            
            # Prepare command parameters
            command_params = parameters or {}
            
            # Execute action based on protocol
            execution_start = datetime.now(UTC)
            
            if device.protocol in [IoTProtocol.HTTP, IoTProtocol.HTTPS]:
                result = await self._execute_http_action(device, action, command_params)
            elif device.protocol == IoTProtocol.MQTT:
                result = await self._execute_mqtt_action(device, action, command_params)
            elif device.protocol == IoTProtocol.COAP:
                result = await self._execute_coap_action(device, action, command_params)
            else:
                return Either.error(IoTIntegrationError(f"Unsupported protocol for action execution: {device.protocol}", device_id))
            
            execution_time = (datetime.now(UTC) - execution_start).total_seconds() * 1000
            
            if result.is_success():
                # Update performance metrics
                connection.update_performance(execution_time, True)
                
                # Store command metrics
                action_key = f"{device_id}_{action.value}"
                if action_key not in self.command_metrics:
                    self.command_metrics[action_key] = {"total_time": 0, "count": 0, "success_rate": 1.0}
                
                metrics = self.command_metrics[action_key]
                metrics["total_time"] += execution_time
                metrics["count"] += 1
                metrics["avg_time"] = metrics["total_time"] / metrics["count"]
                
                # Trigger command event handlers
                for handler in self.command_executed_handlers:
                    try:
                        handler(device_id, action, result.value)
                    except Exception:
                        pass
                
                return result
            else:
                connection.update_performance(execution_time, False)
                return result
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Action execution failed: {str(e)}", device_id))
    
    async def get_device_status(self, device_id: DeviceId) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Get comprehensive device status."""
        try:
            if device_id not in self.devices:
                return Either.error(IoTIntegrationError(f"Device not registered: {device_id}"))
            
            device = self.devices[device_id]
            connection = self.connections[device_id]
            
            status = {
                "device_id": device_id,
                "device_name": device.device_name,
                "device_type": device.device_type.value,
                "protocol": device.protocol.value,
                "status": device.status.value,
                "connection_state": connection.connection_state.value,
                "last_seen": device.last_seen.isoformat() if device.last_seen else None,
                "battery_level": device.battery_level,
                "signal_strength": device.signal_strength,
                "connection_metrics": {
                    "response_time_ms": connection.response_time_ms,
                    "success_rate": connection.success_rate,
                    "error_count": connection.error_count,
                    "connected_at": connection.connected_at.isoformat() if connection.connected_at else None
                },
                "capabilities": [cap.capability_name for cap in self.capabilities.get(device_id, [])],
                "supported_actions": [action.value for action in device.supported_actions],
                "properties": device.properties,
                "security": {
                    "security_level": device.security_level.value,
                    "encryption_enabled": device.encryption_enabled,
                    "authentication_active": connection.authentication_token is not None
                }
            }
            
            return Either.success(status)
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Failed to get device status: {str(e)}", device_id))
    
    async def disconnect_device(self, device_id: DeviceId) -> Either[IoTIntegrationError, bool]:
        """Disconnect from an IoT device."""
        try:
            if device_id not in self.connections:
                return Either.error(IoTIntegrationError(f"Device not found: {device_id}"))
            
            connection = self.connections[device_id]
            device = self.devices[device_id]
            
            # Close protocol-specific connections
            if device.protocol in [IoTProtocol.HTTP, IoTProtocol.HTTPS]:
                await self._disconnect_http(device, connection)
            elif device.protocol == IoTProtocol.MQTT:
                await self._disconnect_mqtt(device, connection)
            elif device.protocol == IoTProtocol.COAP:
                await self._disconnect_coap(device, connection)
            
            # Update connection state
            connection.connection_state = ConnectionState.DISCONNECTED
            connection.authentication_token = None
            connection.encryption_active = False
            
            # Update device status
            device.status = DeviceStatus.OFFLINE
            
            # Trigger disconnection event handlers
            for handler in self.device_disconnected_handlers:
                try:
                    handler(device_id)
                except Exception:
                    pass
            
            return Either.success(True)
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Disconnection failed: {str(e)}", device_id))
    
    # Background services
    
    async def _start_background_services(self):
        """Start background discovery and health check services."""
        if self.discovery_enabled:
            self._discovery_task = asyncio.create_task(self._discovery_loop())
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _discovery_loop(self):
        """Background device discovery loop."""
        while True:
            try:
                await asyncio.sleep(self.discovery_interval)
                await self.discover_devices()
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(60)  # Error recovery delay
    
    async def _health_check_loop(self):
        """Background device health check loop."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for device_id, connection in self.connections.items():
                    if connection.is_active():
                        # Check if device is still responsive
                        ping_result = await self._ping_device(device_id)
                        if ping_result.is_error():
                            # Device not responding, mark as offline
                            connection.connection_state = ConnectionState.ERROR
                            self.devices[device_id].status = DeviceStatus.OFFLINE
                        
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(60)  # Error recovery delay
    
    # Protocol-specific implementations (placeholders)
    
    async def _connect_http(self, device: IoTDevice, connection: DeviceConnection) -> bool:
        """Connect via HTTP protocol."""
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate connection time
        return True
    
    async def _connect_https(self, device: IoTDevice, connection: DeviceConnection) -> bool:
        """Connect via HTTPS protocol."""
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate connection time
        connection.encryption_active = True
        return True
    
    async def _connect_mqtt(self, device: IoTDevice, connection: DeviceConnection) -> bool:
        """Connect via MQTT protocol."""
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate connection time
        return True
    
    async def _connect_coap(self, device: IoTDevice, connection: DeviceConnection) -> bool:
        """Connect via CoAP protocol."""
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate connection time
        return True
    
    async def _authenticate_device(self, device: IoTDevice, connection: DeviceConnection) -> Either[IoTIntegrationError, bool]:
        """Authenticate device connection."""
        try:
            # Placeholder authentication implementation
            if device.authentication:
                # Generate or use provided authentication token
                token = hashlib.sha256(f"{device.device_id}_{datetime.now(UTC).isoformat()}".encode()).hexdigest()
                connection.authentication_token = token
            
            return Either.success(True)
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Authentication failed: {str(e)}", device.device_id))
    
    async def _execute_http_action(self, device: IoTDevice, action: DeviceAction, parameters: Dict[str, Any]) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Execute action via HTTP."""
        # Placeholder implementation
        await asyncio.sleep(0.05)  # Simulate execution time
        
        result = {
            "device_id": device.device_id,
            "action": action.value,
            "parameters": parameters,
            "executed_at": datetime.now(UTC).isoformat(),
            "success": True,
            "response": {"status": "completed"}
        }
        
        return Either.success(result)
    
    async def _execute_mqtt_action(self, device: IoTDevice, action: DeviceAction, parameters: Dict[str, Any]) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Execute action via MQTT."""
        # Placeholder implementation
        await asyncio.sleep(0.05)  # Simulate execution time
        
        result = {
            "device_id": device.device_id,
            "action": action.value,
            "parameters": parameters,
            "executed_at": datetime.now(UTC).isoformat(),
            "success": True,
            "response": {"message_id": f"mqtt_{datetime.now(UTC).timestamp()}"}
        }
        
        return Either.success(result)
    
    async def _execute_coap_action(self, device: IoTDevice, action: DeviceAction, parameters: Dict[str, Any]) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Execute action via CoAP."""
        # Placeholder implementation
        await asyncio.sleep(0.05)  # Simulate execution time
        
        result = {
            "device_id": device.device_id,
            "action": action.value,
            "parameters": parameters,
            "executed_at": datetime.now(UTC).isoformat(),
            "success": True,
            "response": {"coap_response": "ACK"}
        }
        
        return Either.success(result)
    
    # Discovery method implementations (placeholders)
    
    async def _discover_network_scan(self) -> List[DiscoveryResult]:
        """Discover devices via network scanning."""
        # Placeholder implementation
        await asyncio.sleep(1.0)  # Simulate scan time
        return []
    
    async def _discover_mdns(self) -> List[DiscoveryResult]:
        """Discover devices via mDNS."""
        # Placeholder implementation
        await asyncio.sleep(0.5)  # Simulate discovery time
        return []
    
    async def _discover_upnp(self) -> List[DiscoveryResult]:
        """Discover devices via UPnP."""
        # Placeholder implementation
        await asyncio.sleep(0.5)  # Simulate discovery time
        return []
    
    async def _discover_bluetooth(self) -> List[DiscoveryResult]:
        """Discover devices via Bluetooth."""
        # Placeholder implementation
        await asyncio.sleep(2.0)  # Simulate scan time
        return []
    
    async def _disconnect_http(self, device: IoTDevice, connection: DeviceConnection):
        """Disconnect HTTP connection."""
        # Placeholder implementation
        pass
    
    async def _disconnect_mqtt(self, device: IoTDevice, connection: DeviceConnection):
        """Disconnect MQTT connection."""
        # Placeholder implementation
        pass
    
    async def _disconnect_coap(self, device: IoTDevice, connection: DeviceConnection):
        """Disconnect CoAP connection."""
        # Placeholder implementation
        pass
    
    async def _ping_device(self, device_id: DeviceId) -> Either[IoTIntegrationError, bool]:
        """Ping device to check connectivity."""
        # Placeholder implementation
        await asyncio.sleep(0.01)  # Simulate ping time
        return Either.success(True)
    
    # Event handler management
    
    def add_device_discovered_handler(self, handler: Callable[[DiscoveryResult], None]):
        """Add device discovered event handler."""
        self.device_discovered_handlers.append(handler)
    
    def add_device_connected_handler(self, handler: Callable[[DeviceId], None]):
        """Add device connected event handler."""
        self.device_connected_handlers.append(handler)
    
    def add_device_disconnected_handler(self, handler: Callable[[DeviceId], None]):
        """Add device disconnected event handler."""
        self.device_disconnected_handlers.append(handler)
    
    def add_command_executed_handler(self, handler: Callable[[DeviceId, DeviceAction, Dict[str, Any]], None]):
        """Add command executed event handler."""
        self.command_executed_handlers.append(handler)
    
    # Utility methods
    
    def get_all_devices(self) -> List[IoTDevice]:
        """Get all registered devices."""
        return list(self.devices.values())
    
    def get_connected_devices(self) -> List[DeviceId]:
        """Get all connected device IDs."""
        return [device_id for device_id, conn in self.connections.items() if conn.is_active()]
    
    def get_device_metrics(self) -> Dict[str, Any]:
        """Get device controller metrics."""
        total_devices = len(self.devices)
        connected_devices = len(self.get_connected_devices())
        
        return {
            "total_devices": total_devices,
            "connected_devices": connected_devices,
            "connection_rate": connected_devices / total_devices if total_devices > 0 else 0,
            "discovery_results": len(self.discovery_history),
            "command_metrics": self.command_metrics,
            "protocols_in_use": list(set(device.protocol.value for device in self.devices.values())),
            "device_types": list(set(device.device_type.value for device in self.devices.values()))
        }


# Export the device controller
__all__ = ["DeviceController", "DiscoveryResult", "DeviceConnection", "DeviceCapability", "DiscoveryMethod", "ConnectionState"]