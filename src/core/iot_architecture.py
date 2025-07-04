"""
IoT Architecture - TASK_65 Phase 1 Architecture & Design

Complete IoT device integration type system with support for multiple protocols,
sensors, automation workflows, and smart home capabilities.

Architecture: IoT Device Management + Protocol Abstraction + Automation Engine + Security Framework
Performance: <100ms device commands, <50ms sensor readings, <200ms workflow execution
Security: Device authentication, encrypted communication, secure automation workflows
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import json
import asyncio
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.errors import ValidationError, SecurityError


# Branded Types for IoT Architecture
class DeviceId(str):
    """Branded type for IoT device identifiers."""
    pass

class SensorId(str):
    """Branded type for sensor identifiers."""
    pass

class SceneId(str):
    """Branded type for smart home scene identifiers."""
    pass

class WorkflowId(str):
    """Branded type for IoT workflow identifiers."""
    pass

class ProtocolAddress(str):
    """Branded type for protocol-specific addresses."""
    pass


def create_device_id(identifier: str) -> DeviceId:
    """Create a validated device ID."""
    if not identifier or len(identifier) > 100:
        raise ValidationError("device_id", identifier, "must be 1-100 characters")
    return DeviceId(identifier)

def create_sensor_id(identifier: str) -> SensorId:
    """Create a validated sensor ID."""
    if not identifier or len(identifier) > 100:
        raise ValidationError("sensor_id", identifier, "must be 1-100 characters")
    return SensorId(identifier)

def create_scene_id(identifier: str) -> SceneId:
    """Create a validated scene ID."""
    if not identifier or len(identifier) > 100:
        raise ValidationError("scene_id", identifier, "must be 1-100 characters")
    return SceneId(identifier)

def create_workflow_id(identifier: str) -> WorkflowId:
    """Create a validated workflow ID."""
    if not identifier or len(identifier) > 100:
        raise ValidationError("workflow_id", identifier, "must be 1-100 characters")
    return WorkflowId(identifier)

def create_protocol_address(address: str) -> ProtocolAddress:
    """Create a validated protocol address."""
    if not address or len(address) > 200:
        raise ValidationError("protocol_address", address, "must be 1-200 characters")
    return ProtocolAddress(address)


class IoTProtocol(Enum):
    """IoT communication protocols."""
    MQTT = "mqtt"
    HTTP = "http"
    HTTPS = "https"
    COAP = "coap"
    ZIGBEE = "zigbee"
    ZWAVE = "zwave"
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"
    THREAD = "thread"
    MATTER = "matter"


class DeviceType(Enum):
    """IoT device types."""
    LIGHT = "light"
    SWITCH = "switch"
    SENSOR = "sensor"
    THERMOSTAT = "thermostat"
    CAMERA = "camera"
    LOCK = "lock"
    GARAGE_DOOR = "garage_door"
    SPRINKLER = "sprinkler"
    FAN = "fan"
    BLINDS = "blinds"
    SPEAKER = "speaker"
    TV = "tv"
    APPLIANCE = "appliance"
    SECURITY = "security"
    ENERGY = "energy"
    WEATHER = "weather"
    AIR_QUALITY = "air_quality"
    MOTION = "motion"
    CUSTOM = "custom"


class SensorType(Enum):
    """Sensor data types."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    MOTION = "motion"
    SOUND = "sound"
    AIR_QUALITY = "air_quality"
    ENERGY = "energy"
    WATER = "water"
    GAS = "gas"
    SMOKE = "smoke"
    CO2 = "co2"
    UV = "uv"
    WIND = "wind"
    RAIN = "rain"
    SOIL = "soil"
    PH = "ph"
    PROXIMITY = "proximity"
    VIBRATION = "vibration"
    MAGNETIC = "magnetic"
    CUSTOM = "custom"


class DeviceAction(Enum):
    """Device actions."""
    ON = "on"
    OFF = "off"
    TOGGLE = "toggle"
    SET_VALUE = "set_value"
    GET_VALUE = "get_value"
    GET_STATUS = "get_status"
    INCREASE = "increase"
    DECREASE = "decrease"
    OPEN = "open"
    CLOSE = "close"
    LOCK = "lock"
    UNLOCK = "unlock"
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    RESET = "reset"
    CALIBRATE = "calibrate"
    UPDATE = "update"
    REBOOT = "reboot"


class AutomationTrigger(Enum):
    """Automation trigger types."""
    SENSOR_THRESHOLD = "sensor_threshold"
    TIME_SCHEDULE = "time_schedule"
    DEVICE_STATE = "device_state"
    SCENE_ACTIVATION = "scene_activation"
    USER_PRESENCE = "user_presence"
    WEATHER_CONDITION = "weather_condition"
    ENERGY_USAGE = "energy_usage"
    MANUAL_TRIGGER = "manual_trigger"
    WEBHOOK = "webhook"
    API_EVENT = "api_event"
    GEOFENCE = "geofence"
    SUNRISE_SUNSET = "sunrise_sunset"
    SECURITY_EVENT = "security_event"


class WorkflowExecutionMode(Enum):
    """Workflow execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"
    EVENT_DRIVEN = "event_driven"
    ADAPTIVE = "adaptive"


class SecurityLevel(Enum):
    """IoT security levels."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


class DeviceStatus(Enum):
    """Device connection status."""
    ONLINE = "online"
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IoTDevice:
    """IoT device configuration and state."""
    device_id: DeviceId
    device_name: str
    device_type: DeviceType
    protocol: IoTProtocol
    address: ProtocolAddress
    
    # Device capabilities
    capabilities: List[str] = field(default_factory=list)
    supported_actions: List[DeviceAction] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Connection settings
    authentication: Dict[str, str] = field(default_factory=dict)
    connection_timeout: int = 30
    retry_attempts: int = 3
    
    # Device state
    status: DeviceStatus = DeviceStatus.UNKNOWN
    last_seen: Optional[datetime] = None
    battery_level: Optional[float] = None
    signal_strength: Optional[float] = None
    
    # Security settings
    security_level: SecurityLevel = SecurityLevel.STANDARD
    encryption_enabled: bool = True
    certificate_path: Optional[str] = None
    
    # Metadata
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    firmware_version: Optional[str] = None
    location: Optional[str] = None
    room: Optional[str] = None
    
    # Configuration
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def is_online(self) -> bool:
        """Check if device is online."""
        return self.status == DeviceStatus.ONLINE
    
    def supports_action(self, action: DeviceAction) -> bool:
        """Check if device supports specific action."""
        return action in self.supported_actions
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get device property with default."""
        return self.properties.get(key, default)


@dataclass(frozen=True)
class SensorReading:
    """Sensor data reading."""
    sensor_id: SensorId
    sensor_type: SensorType
    value: Union[float, int, str, bool]
    unit: Optional[str] = None
    
    # Reading metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    quality: float = 1.0  # 0.0 to 1.0
    accuracy: Optional[float] = None
    
    # Context
    location: Optional[str] = None
    device_id: Optional[DeviceId] = None
    
    # Validation
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if reading is within valid range."""
        if self.min_value is not None and isinstance(self.value, (int, float)):
            if self.value < self.min_value:
                return False
        if self.max_value is not None and isinstance(self.value, (int, float)):
            if self.value > self.max_value:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality,
            "accuracy": self.accuracy,
            "location": self.location,
            "device_id": self.device_id,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "metadata": self.metadata
        }


@dataclass
class AutomationCondition:
    """Automation trigger condition."""
    condition_id: str
    trigger_type: AutomationTrigger
    
    # Condition parameters
    sensor_id: Optional[SensorId] = None
    device_id: Optional[DeviceId] = None
    threshold_value: Optional[Union[float, str, bool]] = None
    comparison_operator: str = "="  # >, <, >=, <=, ==, !=
    
    # Time-based conditions
    schedule_time: Optional[str] = None  # HH:MM format
    schedule_days: List[str] = field(default_factory=list)  # Monday, Tuesday, etc.
    
    # Location-based conditions
    geofence_center: Optional[Tuple[float, float]] = None  # lat, lon
    geofence_radius: Optional[float] = None  # meters
    
    # Event conditions
    webhook_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    
    # Condition state
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    # Configuration
    cooldown_minutes: int = 0
    maximum_triggers: Optional[int] = None
    
    def evaluate(self, sensor_reading: Optional[SensorReading] = None, 
                device_state: Optional[Dict[str, Any]] = None) -> bool:
        """Evaluate condition against current data."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if (self.cooldown_minutes > 0 and self.last_triggered and 
            datetime.now(UTC) - self.last_triggered < timedelta(minutes=self.cooldown_minutes)):
            return False
        
        # Check maximum triggers
        if self.maximum_triggers and self.trigger_count >= self.maximum_triggers:
            return False
        
        # Evaluate based on trigger type
        if self.trigger_type == AutomationTrigger.SENSOR_THRESHOLD:
            return self._evaluate_sensor_threshold(sensor_reading)
        elif self.trigger_type == AutomationTrigger.DEVICE_STATE:
            return self._evaluate_device_state(device_state)
        elif self.trigger_type == AutomationTrigger.TIME_SCHEDULE:
            return self._evaluate_time_schedule()
        # Add other trigger type evaluations as needed
        
        return False
    
    def _evaluate_sensor_threshold(self, reading: Optional[SensorReading]) -> bool:
        """Evaluate sensor threshold condition."""
        if not reading or reading.sensor_id != self.sensor_id:
            return False
        
        if self.threshold_value is None:
            return False
        
        value = reading.value
        threshold = self.threshold_value
        
        if self.comparison_operator == ">":
            return value > threshold
        elif self.comparison_operator == "<":
            return value < threshold
        elif self.comparison_operator == ">=":
            return value >= threshold
        elif self.comparison_operator == "<=":
            return value <= threshold
        elif self.comparison_operator == "==":
            return value == threshold
        elif self.comparison_operator == "!=":
            return value != threshold
        
        return False
    
    def _evaluate_device_state(self, state: Optional[Dict[str, Any]]) -> bool:
        """Evaluate device state condition."""
        if not state or not self.device_id:
            return False
        
        device_value = state.get(self.device_id)
        return device_value == self.threshold_value
    
    def _evaluate_time_schedule(self) -> bool:
        """Evaluate time-based schedule condition."""
        if not self.schedule_time:
            return False
        
        now = datetime.now(UTC)
        current_time = now.strftime("%H:%M")
        current_day = now.strftime("%A")
        
        time_match = current_time == self.schedule_time
        day_match = not self.schedule_days or current_day in self.schedule_days
        
        return time_match and day_match


@dataclass
class AutomationAction:
    """Automation action to execute."""
    action_id: str
    action_type: str  # device_control, scene_activation, notification, etc.
    
    # Target configuration
    device_id: Optional[DeviceId] = None
    scene_id: Optional[SceneId] = None
    
    # Action parameters
    action: Optional[DeviceAction] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution settings
    delay_seconds: int = 0
    timeout_seconds: int = 30
    retry_attempts: int = 2
    
    # Validation
    required_device_status: Optional[DeviceStatus] = None
    conditions: List[str] = field(default_factory=list)
    
    async def execute(self, context: Dict[str, Any] = None) -> Either[str, Dict[str, Any]]:
        """Execute the automation action."""
        try:
            if self.delay_seconds > 0:
                await asyncio.sleep(self.delay_seconds)
            
            # Implementation would depend on action type
            # This is a placeholder for the actual execution logic
            result = {
                "action_id": self.action_id,
                "action_type": self.action_type,
                "executed_at": datetime.now(UTC).isoformat(),
                "parameters": self.parameters,
                "success": True
            }
            
            return Either.success(result)
            
        except Exception as e:
            return Either.error(f"Action execution failed: {str(e)}")


@dataclass
class SmartHomeScene:
    """Smart home scene configuration."""
    scene_id: SceneId
    scene_name: str
    description: Optional[str] = None
    
    # Scene configuration
    device_settings: Dict[DeviceId, Dict[str, Any]] = field(default_factory=dict)
    actions: List[AutomationAction] = field(default_factory=list)
    
    # Activation settings
    activation_delay: int = 0  # seconds
    fade_duration: int = 0  # seconds for gradual changes
    
    # Schedule and conditions
    auto_activate_conditions: List[AutomationCondition] = field(default_factory=list)
    schedule: Optional[Dict[str, Any]] = None
    
    # Scene metadata
    icon: Optional[str] = None
    color: Optional[str] = None
    category: Optional[str] = None
    
    # Usage tracking
    activation_count: int = 0
    last_activated: Optional[datetime] = None
    favorite: bool = False
    
    # Configuration
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    async def activate(self, context: Dict[str, Any] = None) -> Either[str, Dict[str, Any]]:
        """Activate the scene."""
        try:
            if self.activation_delay > 0:
                await asyncio.sleep(self.activation_delay)
            
            results = []
            
            # Execute scene actions
            for action in self.actions:
                result = await action.execute(context)
                if result.is_success():
                    results.append(result.value)
                else:
                    return Either.error(f"Scene activation failed: {result.error}")
            
            # Update activation tracking
            self.activation_count += 1
            self.last_activated = datetime.now(UTC)
            
            return Either.success({
                "scene_id": self.scene_id,
                "scene_name": self.scene_name,
                "activated_at": self.last_activated.isoformat(),
                "actions_executed": len(results),
                "results": results
            })
            
        except Exception as e:
            return Either.error(f"Scene activation error: {str(e)}")


@dataclass
class IoTWorkflow:
    """IoT automation workflow."""
    workflow_id: WorkflowId
    workflow_name: str
    description: Optional[str] = None
    
    # Workflow configuration
    triggers: List[AutomationCondition] = field(default_factory=list)
    actions: List[AutomationAction] = field(default_factory=list)
    execution_mode: WorkflowExecutionMode = WorkflowExecutionMode.SEQUENTIAL
    
    # Dependencies
    device_dependencies: Dict[DeviceId, List[DeviceId]] = field(default_factory=dict)
    required_devices: Set[DeviceId] = field(default_factory=set)
    
    # Execution settings
    max_execution_time: int = 300  # seconds
    retry_on_failure: bool = True
    continue_on_error: bool = False
    
    # Performance settings
    parallel_limit: int = 5
    timeout_seconds: int = 60
    
    # Monitoring
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    success_count: int = 0
    
    # Status
    enabled: bool = True
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    
    # Configuration
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def is_triggered(self, sensor_readings: List[SensorReading] = None,
                    device_states: Dict[DeviceId, Dict[str, Any]] = None) -> bool:
        """Check if workflow should be triggered."""
        if not self.enabled or not self.triggers:
            return False
        
        for trigger in self.triggers:
            if trigger.evaluate(
                sensor_reading=sensor_readings[0] if sensor_readings else None,
                device_state=device_states
            ):
                return True
        
        return False
    
    async def execute(self, context: Dict[str, Any] = None) -> Either[str, Dict[str, Any]]:
        """Execute the IoT workflow."""
        try:
            execution_start = datetime.now(UTC)
            
            if self.execution_mode == WorkflowExecutionMode.SEQUENTIAL:
                results = await self._execute_sequential()
            elif self.execution_mode == WorkflowExecutionMode.PARALLEL:
                results = await self._execute_parallel()
            elif self.execution_mode == WorkflowExecutionMode.CONDITIONAL:
                results = await self._execute_conditional(context)
            else:
                results = await self._execute_sequential()  # Default fallback
            
            execution_time = (datetime.now(UTC) - execution_start).total_seconds()
            
            # Update metrics
            self.execution_count += 1
            self.last_execution = execution_start
            self.performance_metrics["avg_execution_time"] = (
                self.performance_metrics.get("avg_execution_time", 0) * (self.execution_count - 1) + execution_time
            ) / self.execution_count
            
            if results.is_success():
                self.success_count += 1
            else:
                self.error_count += 1
            
            return results
            
        except Exception as e:
            self.error_count += 1
            return Either.error(f"Workflow execution failed: {str(e)}")
    
    async def _execute_sequential(self) -> Either[str, Dict[str, Any]]:
        """Execute actions sequentially."""
        results = []
        
        for action in self.actions:
            result = await action.execute()
            if result.is_success():
                results.append(result.value)
            elif not self.continue_on_error:
                return Either.error(f"Sequential execution failed: {result.error}")
        
        return Either.success({
            "workflow_id": self.workflow_id,
            "execution_mode": "sequential",
            "actions_completed": len(results),
            "results": results
        })
    
    async def _execute_parallel(self) -> Either[str, Dict[str, Any]]:
        """Execute actions in parallel."""
        # Limit parallel execution
        semaphore = asyncio.Semaphore(self.parallel_limit)
        
        async def execute_with_semaphore(action):
            async with semaphore:
                return await action.execute()
        
        tasks = [execute_with_semaphore(action) for action in self.actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            elif hasattr(result, 'is_success') and result.is_success():
                successful_results.append(result.value)
            else:
                errors.append("Unknown execution error")
        
        if errors and not self.continue_on_error:
            return Either.error(f"Parallel execution failed: {errors[0]}")
        
        return Either.success({
            "workflow_id": self.workflow_id,
            "execution_mode": "parallel",
            "actions_completed": len(successful_results),
            "errors": len(errors),
            "results": successful_results
        })
    
    async def _execute_conditional(self, context: Dict[str, Any] = None) -> Either[str, Dict[str, Any]]:
        """Execute actions based on conditions."""
        # Conditional execution logic would be implemented here
        # For now, fall back to sequential execution
        return await self._execute_sequential()


class IoTIntegrationError(Exception):
    """IoT integration specific errors."""
    
    def __init__(self, message: str, device_id: Optional[DeviceId] = None, 
                 sensor_id: Optional[SensorId] = None):
        super().__init__(message)
        self.device_id = device_id
        self.sensor_id = sensor_id


# Utility functions for IoT architecture

def validate_device_configuration(device: IoTDevice) -> Either[IoTIntegrationError, bool]:
    """Validate IoT device configuration."""
    try:
        # Validate device ID
        if not device.device_id or len(device.device_id) > 100:
            return Either.error(IoTIntegrationError("Invalid device ID", device.device_id))
        
        # Validate protocol address
        if not device.address:
            return Either.error(IoTIntegrationError("Protocol address required", device.device_id))
        
        # Validate security settings
        if device.security_level == SecurityLevel.MAXIMUM and not device.encryption_enabled:
            return Either.error(IoTIntegrationError("Encryption required for maximum security", device.device_id))
        
        return Either.success(True)
        
    except Exception as e:
        return Either.error(IoTIntegrationError(f"Device validation failed: {str(e)}", device.device_id))


def create_default_device_capabilities(device_type: DeviceType) -> List[str]:
    """Create default capabilities for device type."""
    capabilities_map = {
        DeviceType.LIGHT: ["on_off", "brightness", "color"],
        DeviceType.SWITCH: ["on_off"],
        DeviceType.SENSOR: ["read_value", "get_status"],
        DeviceType.THERMOSTAT: ["temperature_control", "mode_setting", "schedule"],
        DeviceType.CAMERA: ["video_stream", "recording", "motion_detection"],
        DeviceType.LOCK: ["lock_unlock", "status_check", "access_log"],
        DeviceType.GARAGE_DOOR: ["open_close", "status_check"],
        DeviceType.SPRINKLER: ["on_off", "zone_control", "schedule"],
        DeviceType.FAN: ["on_off", "speed_control", "oscillation"],
        DeviceType.BLINDS: ["open_close", "position_control"],
        DeviceType.SPEAKER: ["audio_playback", "volume_control", "source_selection"],
        DeviceType.TV: ["power_control", "channel_control", "volume_control"],
        DeviceType.APPLIANCE: ["on_off", "mode_setting"],
        DeviceType.SECURITY: ["status_monitoring", "alert_generation"],
        DeviceType.ENERGY: ["consumption_monitoring", "control"],
        DeviceType.WEATHER: ["data_collection", "forecasting"],
        DeviceType.AIR_QUALITY: ["air_monitoring", "purification"],
        DeviceType.MOTION: ["motion_detection", "occupancy_sensing"],
        DeviceType.CUSTOM: ["custom_control"]
    }
    
    return capabilities_map.get(device_type, ["basic_control"])


# Export all classes and functions
__all__ = [
    # Branded types
    "DeviceId", "SensorId", "SceneId", "WorkflowId", "ProtocolAddress",
    "create_device_id", "create_sensor_id", "create_scene_id", "create_workflow_id", "create_protocol_address",
    
    # Enums
    "IoTProtocol", "DeviceType", "SensorType", "DeviceAction", "AutomationTrigger",
    "WorkflowExecutionMode", "SecurityLevel", "DeviceStatus",
    
    # Data structures
    "IoTDevice", "SensorReading", "AutomationCondition", "AutomationAction",
    "SmartHomeScene", "IoTWorkflow",
    
    # Utilities
    "IoTIntegrationError", "validate_device_configuration", "create_default_device_capabilities"
]