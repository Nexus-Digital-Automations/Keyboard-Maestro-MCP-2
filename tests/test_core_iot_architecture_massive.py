"""Comprehensive tests for src/core/iot_architecture.py - MASSIVE 415 statements coverage.

🚨 CRITICAL COVERAGE ENFORCEMENT: Phase 8 targeting highest-impact zero-coverage modules.
This test covers src/core/iot_architecture.py (415 statements - 6th HIGHEST IMPACT) to achieve
significant progress toward mandatory 95% coverage threshold.

Coverage Focus: IoT device management, automation workflows, smart home scenes, sensor readings,
protocol abstraction, device validation, workflow execution, and all IoT integration functionality.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from src.core.constants import IDENTIFIER_LENGTH_LIMIT
from src.core.either import Either
from src.core.errors import ValidationError
from src.core.iot_architecture import (
    AutomationAction,
    AutomationCondition,
    AutomationTrigger,
    DeviceAction,
    DeviceId,
    DeviceStatus,
    DeviceType,
    IoTDevice,
    IoTIntegrationError,
    IoTProtocol,
    IoTWorkflow,
    ProtocolAddress,
    SceneId,
    SecurityLevel,
    SensorId,
    SensorReading,
    SensorType,
    SmartHomeScene,
    WorkflowExecutionMode,
    WorkflowId,
    create_default_device_capabilities,
    create_device_id,
    create_protocol_address,
    create_scene_id,
    create_sensor_id,
    create_workflow_id,
    validate_device_configuration,
)


class TestBrandedTypes:
    """Comprehensive tests for branded type creation functions."""

    def test_create_device_id_success(self):
        """Test successful device ID creation."""
        device_id = create_device_id("device_001")
        assert device_id == "device_001"
        assert isinstance(device_id, DeviceId)

    def test_create_device_id_empty_string(self):
        """Test device ID creation with empty string."""
        with pytest.raises(ValidationError) as exc_info:
            create_device_id("")
        assert "device_id" in str(exc_info.value)
        assert "must be 1-100 characters" in str(exc_info.value)

    def test_create_device_id_too_long(self):
        """Test device ID creation with too long string."""
        long_id = "x" * (IDENTIFIER_LENGTH_LIMIT + 1)
        with pytest.raises(ValidationError) as exc_info:
            create_device_id(long_id)
        assert "device_id" in str(exc_info.value)
        assert "must be 1-100 characters" in str(exc_info.value)

    def test_create_sensor_id_success(self):
        """Test successful sensor ID creation."""
        sensor_id = create_sensor_id("sensor_temperature_001")
        assert sensor_id == "sensor_temperature_001"
        assert isinstance(sensor_id, SensorId)

    def test_create_sensor_id_edge_cases(self):
        """Test sensor ID creation edge cases."""
        # Empty string
        with pytest.raises(ValidationError):
            create_sensor_id("")

        # Too long
        with pytest.raises(ValidationError):
            create_sensor_id("x" * (IDENTIFIER_LENGTH_LIMIT + 1))

    def test_create_scene_id_success(self):
        """Test successful scene ID creation."""
        scene_id = create_scene_id("evening_lights")
        assert scene_id == "evening_lights"
        assert isinstance(scene_id, SceneId)

    def test_create_scene_id_validation(self):
        """Test scene ID validation."""
        with pytest.raises(ValidationError):
            create_scene_id("")

        with pytest.raises(ValidationError):
            create_scene_id("x" * (IDENTIFIER_LENGTH_LIMIT + 1))

    def test_create_workflow_id_success(self):
        """Test successful workflow ID creation."""
        workflow_id = create_workflow_id("morning_routine")
        assert workflow_id == "morning_routine"
        assert isinstance(workflow_id, WorkflowId)

    def test_create_workflow_id_validation(self):
        """Test workflow ID validation."""
        with pytest.raises(ValidationError):
            create_workflow_id("")

        with pytest.raises(ValidationError):
            create_workflow_id("x" * (IDENTIFIER_LENGTH_LIMIT + 1))

    def test_create_protocol_address_success(self):
        """Test successful protocol address creation."""
        address = create_protocol_address("192.168.1.100:1883")
        assert address == "192.168.1.100:1883"
        assert isinstance(address, ProtocolAddress)

    def test_create_protocol_address_validation(self):
        """Test protocol address validation."""
        with pytest.raises(ValidationError):
            create_protocol_address("")

        with pytest.raises(ValidationError):
            create_protocol_address("x" * 201)  # Too long


class TestEnumerations:
    """Comprehensive tests for IoT enumeration classes."""

    def test_iot_protocol_values(self):
        """Test IoT protocol enumeration values."""
        assert IoTProtocol.MQTT.value == "mqtt"
        assert IoTProtocol.HTTP.value == "http"
        assert IoTProtocol.HTTPS.value == "https"
        assert IoTProtocol.COAP.value == "coap"
        assert IoTProtocol.ZIGBEE.value == "zigbee"
        assert IoTProtocol.ZWAVE.value == "zwave"
        assert IoTProtocol.BLUETOOTH.value == "bluetooth"
        assert IoTProtocol.WIFI.value == "wifi"
        assert IoTProtocol.THREAD.value == "thread"
        assert IoTProtocol.MATTER.value == "matter"

    def test_device_type_values(self):
        """Test device type enumeration values."""
        assert DeviceType.LIGHT.value == "light"
        assert DeviceType.SWITCH.value == "switch"
        assert DeviceType.SENSOR.value == "sensor"
        assert DeviceType.THERMOSTAT.value == "thermostat"
        assert DeviceType.CAMERA.value == "camera"
        assert DeviceType.LOCK.value == "lock"
        assert DeviceType.GARAGE_DOOR.value == "garage_door"
        assert DeviceType.SPRINKLER.value == "sprinkler"
        assert DeviceType.FAN.value == "fan"
        assert DeviceType.BLINDS.value == "blinds"
        assert DeviceType.SPEAKER.value == "speaker"
        assert DeviceType.TV.value == "tv"
        assert DeviceType.APPLIANCE.value == "appliance"
        assert DeviceType.SECURITY.value == "security"
        assert DeviceType.ENERGY.value == "energy"
        assert DeviceType.WEATHER.value == "weather"
        assert DeviceType.AIR_QUALITY.value == "air_quality"
        assert DeviceType.MOTION.value == "motion"
        assert DeviceType.CUSTOM.value == "custom"

    def test_sensor_type_values(self):
        """Test sensor type enumeration values."""
        assert SensorType.TEMPERATURE.value == "temperature"
        assert SensorType.HUMIDITY.value == "humidity"
        assert SensorType.PRESSURE.value == "pressure"
        assert SensorType.LIGHT.value == "light"
        assert SensorType.MOTION.value == "motion"
        assert SensorType.SOUND.value == "sound"
        assert SensorType.AIR_QUALITY.value == "air_quality"
        assert SensorType.ENERGY.value == "energy"
        assert SensorType.WATER.value == "water"
        assert SensorType.GAS.value == "gas"
        assert SensorType.SMOKE.value == "smoke"
        assert SensorType.CO2.value == "co2"
        assert SensorType.UV.value == "uv"
        assert SensorType.WIND.value == "wind"
        assert SensorType.RAIN.value == "rain"
        assert SensorType.SOIL.value == "soil"
        assert SensorType.PH.value == "ph"
        assert SensorType.PROXIMITY.value == "proximity"
        assert SensorType.VIBRATION.value == "vibration"
        assert SensorType.MAGNETIC.value == "magnetic"
        assert SensorType.CUSTOM.value == "custom"

    def test_device_action_values(self):
        """Test device action enumeration values."""
        assert DeviceAction.ON.value == "on"
        assert DeviceAction.OFF.value == "off"
        assert DeviceAction.TOGGLE.value == "toggle"
        assert DeviceAction.SET_VALUE.value == "set_value"
        assert DeviceAction.GET_VALUE.value == "get_value"
        assert DeviceAction.GET_STATUS.value == "get_status"
        assert DeviceAction.INCREASE.value == "increase"
        assert DeviceAction.DECREASE.value == "decrease"
        assert DeviceAction.OPEN.value == "open"
        assert DeviceAction.CLOSE.value == "close"
        assert DeviceAction.LOCK.value == "lock"
        assert DeviceAction.UNLOCK.value == "unlock"
        assert DeviceAction.PLAY.value == "play"
        assert DeviceAction.PAUSE.value == "pause"
        assert DeviceAction.STOP.value == "stop"
        assert DeviceAction.RESET.value == "reset"
        assert DeviceAction.CALIBRATE.value == "calibrate"
        assert DeviceAction.UPDATE.value == "update"
        assert DeviceAction.REBOOT.value == "reboot"

    def test_automation_trigger_values(self):
        """Test automation trigger enumeration values."""
        assert AutomationTrigger.SENSOR_THRESHOLD.value == "sensor_threshold"
        assert AutomationTrigger.TIME_SCHEDULE.value == "time_schedule"
        assert AutomationTrigger.DEVICE_STATE.value == "device_state"
        assert AutomationTrigger.SCENE_ACTIVATION.value == "scene_activation"
        assert AutomationTrigger.USER_PRESENCE.value == "user_presence"
        assert AutomationTrigger.WEATHER_CONDITION.value == "weather_condition"
        assert AutomationTrigger.ENERGY_USAGE.value == "energy_usage"
        assert AutomationTrigger.MANUAL_TRIGGER.value == "manual_trigger"
        assert AutomationTrigger.WEBHOOK.value == "webhook"
        assert AutomationTrigger.API_EVENT.value == "api_event"
        assert AutomationTrigger.GEOFENCE.value == "geofence"
        assert AutomationTrigger.SUNRISE_SUNSET.value == "sunrise_sunset"
        assert AutomationTrigger.SECURITY_EVENT.value == "security_event"

    def test_workflow_execution_mode_values(self):
        """Test workflow execution mode enumeration values."""
        assert WorkflowExecutionMode.SEQUENTIAL.value == "sequential"
        assert WorkflowExecutionMode.PARALLEL.value == "parallel"
        assert WorkflowExecutionMode.CONDITIONAL.value == "conditional"
        assert WorkflowExecutionMode.PIPELINE.value == "pipeline"
        assert WorkflowExecutionMode.EVENT_DRIVEN.value == "event_driven"
        assert WorkflowExecutionMode.ADAPTIVE.value == "adaptive"

    def test_security_level_values(self):
        """Test security level enumeration values."""
        assert SecurityLevel.BASIC.value == "basic"
        assert SecurityLevel.STANDARD.value == "standard"
        assert SecurityLevel.HIGH.value == "high"
        assert SecurityLevel.MAXIMUM.value == "maximum"

    def test_device_status_values(self):
        """Test device status enumeration values."""
        assert DeviceStatus.ONLINE.value == "online"
        assert DeviceStatus.OFFLINE.value == "offline"
        assert DeviceStatus.CONNECTING.value == "connecting"
        assert DeviceStatus.ERROR.value == "error"
        assert DeviceStatus.MAINTENANCE.value == "maintenance"
        assert DeviceStatus.UNKNOWN.value == "unknown"


class TestIoTDevice:
    """Comprehensive tests for IoTDevice class."""

    @pytest.fixture
    def sample_iot_device(self):
        """Create sample IoT device for testing."""
        return IoTDevice(
            device_id=create_device_id("smart_light_001"),
            device_name="Living Room Smart Light",
            device_type=DeviceType.LIGHT,
            protocol=IoTProtocol.WIFI,
            address=create_protocol_address("192.168.1.50"),
            capabilities=["on_off", "brightness", "color"],
            supported_actions=[DeviceAction.ON, DeviceAction.OFF, DeviceAction.SET_VALUE],
            properties={"brightness": 75, "color": "warm_white"},
            status=DeviceStatus.ONLINE,
            last_seen=datetime.now(UTC),
            battery_level=None,  # AC powered
            signal_strength=0.85,
            security_level=SecurityLevel.STANDARD,
            encryption_enabled=True,
            manufacturer="SmartHome Corp",
            model="SL-2024",
            firmware_version="1.2.3",
            location="Living Room",
            room="Living Room",
            tags={"lighting", "smart_home"},
        )

    def test_iot_device_creation_success(self, sample_iot_device):
        """Test successful IoT device creation."""
        device = sample_iot_device

        assert device.device_id == "smart_light_001"
        assert device.device_name == "Living Room Smart Light"
        assert device.device_type == DeviceType.LIGHT
        assert device.protocol == IoTProtocol.WIFI
        assert device.address == "192.168.1.50"
        assert "on_off" in device.capabilities
        assert DeviceAction.ON in device.supported_actions
        assert device.properties["brightness"] == 75
        assert device.status == DeviceStatus.ONLINE
        assert device.security_level == SecurityLevel.STANDARD
        assert device.encryption_enabled is True
        assert device.manufacturer == "SmartHome Corp"
        assert device.location == "Living Room"
        assert "lighting" in device.tags

    def test_iot_device_is_online_true(self, sample_iot_device):
        """Test device is_online method when device is online."""
        device = sample_iot_device
        device.status = DeviceStatus.ONLINE
        assert device.is_online() is True

    def test_iot_device_is_online_false(self, sample_iot_device):
        """Test device is_online method when device is offline."""
        device = sample_iot_device
        device.status = DeviceStatus.OFFLINE
        assert device.is_online() is False

    def test_iot_device_supports_action_true(self, sample_iot_device):
        """Test device supports_action method for supported action."""
        device = sample_iot_device
        assert device.supports_action(DeviceAction.ON) is True
        assert device.supports_action(DeviceAction.OFF) is True
        assert device.supports_action(DeviceAction.SET_VALUE) is True

    def test_iot_device_supports_action_false(self, sample_iot_device):
        """Test device supports_action method for unsupported action."""
        device = sample_iot_device
        assert device.supports_action(DeviceAction.LOCK) is False
        assert device.supports_action(DeviceAction.UNLOCK) is False

    def test_iot_device_get_property_exists(self, sample_iot_device):
        """Test device get_property method for existing property."""
        device = sample_iot_device
        assert device.get_property("brightness") == 75
        assert device.get_property("color") == "warm_white"

    def test_iot_device_get_property_not_exists(self, sample_iot_device):
        """Test device get_property method for non-existing property."""
        device = sample_iot_device
        assert device.get_property("nonexistent") is None
        assert device.get_property("missing", "default_value") == "default_value"

    def test_iot_device_defaults(self):
        """Test IoT device with default values."""
        device = IoTDevice(
            device_id=create_device_id("minimal_device"),
            device_name="Minimal Device",
            device_type=DeviceType.SENSOR,
            protocol=IoTProtocol.HTTP,
            address=create_protocol_address("http://sensor.local"),
        )

        assert device.capabilities == []
        assert device.supported_actions == []
        assert device.properties == {}
        assert device.status == DeviceStatus.UNKNOWN
        assert device.last_seen is None
        assert device.battery_level is None
        assert device.signal_strength is None
        assert device.security_level == SecurityLevel.STANDARD
        assert device.encryption_enabled is True
        assert device.manufacturer is None
        assert device.location is None
        assert device.tags == set()

    def test_iot_device_battery_powered(self):
        """Test IoT device with battery power."""
        device = IoTDevice(
            device_id=create_device_id("battery_sensor"),
            device_name="Battery Sensor",
            device_type=DeviceType.SENSOR,
            protocol=IoTProtocol.ZIGBEE,
            address=create_protocol_address("zigbee://sensor_network/0x001"),
            battery_level=0.75,
            signal_strength=0.9,
        )

        assert device.battery_level == 0.75
        assert device.signal_strength == 0.9


class TestSensorReading:
    """Comprehensive tests for SensorReading class."""

    @pytest.fixture
    def sample_sensor_reading(self):
        """Create sample sensor reading for testing."""
        return SensorReading(
            sensor_id=create_sensor_id("temp_sensor_001"),
            sensor_type=SensorType.TEMPERATURE,
            value=23.5,
            unit="°C",
            timestamp=datetime(2024, 7, 11, 12, 0, 0, tzinfo=UTC),
            quality=0.95,
            accuracy=0.1,
            location="Living Room",
            device_id=create_device_id("weather_station_001"),
            min_value=-10.0,
            max_value=50.0,
            metadata={"calibrated": True, "sensor_type": "DS18B20"},
        )

    def test_sensor_reading_creation_success(self, sample_sensor_reading):
        """Test successful sensor reading creation."""
        reading = sample_sensor_reading

        assert reading.sensor_id == "temp_sensor_001"
        assert reading.sensor_type == SensorType.TEMPERATURE
        assert reading.value == 23.5
        assert reading.unit == "°C"
        assert reading.quality == 0.95
        assert reading.accuracy == 0.1
        assert reading.location == "Living Room"
        assert reading.device_id == "weather_station_001"
        assert reading.min_value == -10.0
        assert reading.max_value == 50.0
        assert reading.metadata["calibrated"] is True

    def test_sensor_reading_is_valid_true(self, sample_sensor_reading):
        """Test sensor reading validation for valid reading."""
        reading = sample_sensor_reading
        assert reading.is_valid() is True

    def test_sensor_reading_is_valid_below_min(self, sample_sensor_reading):
        """Test sensor reading validation for value below minimum."""
        reading = sample_sensor_reading
        reading = SensorReading(
            sensor_id=reading.sensor_id,
            sensor_type=reading.sensor_type,
            value=-15.0,  # Below min_value of -10.0
            min_value=-10.0,
            max_value=50.0,
        )
        assert reading.is_valid() is False

    def test_sensor_reading_is_valid_above_max(self, sample_sensor_reading):
        """Test sensor reading validation for value above maximum."""
        reading = sample_sensor_reading
        reading = SensorReading(
            sensor_id=reading.sensor_id,
            sensor_type=reading.sensor_type,
            value=55.0,  # Above max_value of 50.0
            min_value=-10.0,
            max_value=50.0,
        )
        assert reading.is_valid() is False

    def test_sensor_reading_is_valid_no_limits(self):
        """Test sensor reading validation with no min/max limits."""
        reading = SensorReading(
            sensor_id=create_sensor_id("no_limit_sensor"),
            sensor_type=SensorType.HUMIDITY,
            value=999.0,  # Any value should be valid
            min_value=None,
            max_value=None,
        )
        assert reading.is_valid() is True

    def test_sensor_reading_is_valid_string_value(self):
        """Test sensor reading validation with string value."""
        reading = SensorReading(
            sensor_id=create_sensor_id("status_sensor"),
            sensor_type=SensorType.CUSTOM,
            value="active",  # String value
            min_value=10.0,
            max_value=50.0,
        )
        # String values should be valid regardless of numeric limits
        assert reading.is_valid() is True

    def test_sensor_reading_to_dict(self, sample_sensor_reading):
        """Test sensor reading to_dict method."""
        reading = sample_sensor_reading
        data = reading.to_dict()

        assert data["sensor_id"] == "temp_sensor_001"
        assert data["sensor_type"] == "temperature"
        assert data["value"] == 23.5
        assert data["unit"] == "°C"
        assert data["quality"] == 0.95
        assert data["accuracy"] == 0.1
        assert data["location"] == "Living Room"
        assert data["device_id"] == "weather_station_001"
        assert data["min_value"] == -10.0
        assert data["max_value"] == 50.0
        assert data["metadata"]["calibrated"] is True
        assert "timestamp" in data  # ISO format timestamp

    def test_sensor_reading_defaults(self):
        """Test sensor reading with default values."""
        reading = SensorReading(
            sensor_id=create_sensor_id("minimal_sensor"),
            sensor_type=SensorType.MOTION,
            value=True,
        )

        assert reading.unit is None
        assert reading.quality == 1.0
        assert reading.accuracy is None
        assert reading.location is None
        assert reading.device_id is None
        assert reading.min_value is None
        assert reading.max_value is None
        assert reading.metadata == {}


class TestAutomationCondition:
    """Comprehensive tests for AutomationCondition class."""

    @pytest.fixture
    def sensor_threshold_condition(self):
        """Create sensor threshold automation condition."""
        return AutomationCondition(
            condition_id="temp_high_condition",
            trigger_type=AutomationTrigger.SENSOR_THRESHOLD,
            sensor_id=create_sensor_id("temp_sensor_001"),
            threshold_value=25.0,
            comparison_operator=">",
            enabled=True,
        )

    @pytest.fixture
    def time_schedule_condition(self):
        """Create time schedule automation condition."""
        return AutomationCondition(
            condition_id="morning_schedule",
            trigger_type=AutomationTrigger.TIME_SCHEDULE,
            schedule_time="07:00",
            schedule_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            enabled=True,
        )

    @pytest.fixture
    def device_state_condition(self):
        """Create device state automation condition."""
        return AutomationCondition(
            condition_id="door_open_condition",
            trigger_type=AutomationTrigger.DEVICE_STATE,
            device_id=create_device_id("front_door_001"),
            threshold_value="open",
            enabled=True,
        )

    def test_automation_condition_creation(self, sensor_threshold_condition):
        """Test automation condition creation."""
        condition = sensor_threshold_condition

        assert condition.condition_id == "temp_high_condition"
        assert condition.trigger_type == AutomationTrigger.SENSOR_THRESHOLD
        assert condition.sensor_id == "temp_sensor_001"
        assert condition.threshold_value == 25.0
        assert condition.comparison_operator == ">"
        assert condition.enabled is True
        assert condition.last_triggered is None
        assert condition.trigger_count == 0

    def test_automation_condition_evaluate_disabled(self, sensor_threshold_condition):
        """Test automation condition evaluation when disabled."""
        condition = sensor_threshold_condition
        condition.enabled = False

        reading = SensorReading(
            sensor_id=create_sensor_id("temp_sensor_001"),
            sensor_type=SensorType.TEMPERATURE,
            value=30.0,  # Above threshold
        )

        assert condition.evaluate(sensor_reading=reading) is False

    def test_automation_condition_evaluate_cooldown(self, sensor_threshold_condition):
        """Test automation condition evaluation during cooldown period."""
        condition = sensor_threshold_condition
        condition.cooldown_minutes = 10
        condition.last_triggered = datetime.now(UTC) - timedelta(minutes=5)  # Within cooldown

        reading = SensorReading(
            sensor_id=create_sensor_id("temp_sensor_001"),
            sensor_type=SensorType.TEMPERATURE,
            value=30.0,  # Above threshold
        )

        assert condition.evaluate(sensor_reading=reading) is False

    def test_automation_condition_evaluate_max_triggers(self, sensor_threshold_condition):
        """Test automation condition evaluation when max triggers reached."""
        condition = sensor_threshold_condition
        condition.maximum_triggers = 5
        condition.trigger_count = 5  # Already at maximum

        reading = SensorReading(
            sensor_id=create_sensor_id("temp_sensor_001"),
            sensor_type=SensorType.TEMPERATURE,
            value=30.0,  # Above threshold
        )

        assert condition.evaluate(sensor_reading=reading) is False

    def test_sensor_threshold_evaluation_greater_than_true(self, sensor_threshold_condition):
        """Test sensor threshold evaluation - greater than (true)."""
        condition = sensor_threshold_condition
        condition.comparison_operator = ">"

        reading = SensorReading(
            sensor_id=create_sensor_id("temp_sensor_001"),
            sensor_type=SensorType.TEMPERATURE,
            value=30.0,  # 30 > 25
        )

        assert condition.evaluate(sensor_reading=reading) is True

    def test_sensor_threshold_evaluation_greater_than_false(self, sensor_threshold_condition):
        """Test sensor threshold evaluation - greater than (false)."""
        condition = sensor_threshold_condition
        condition.comparison_operator = ">"

        reading = SensorReading(
            sensor_id=create_sensor_id("temp_sensor_001"),
            sensor_type=SensorType.TEMPERATURE,
            value=20.0,  # 20 <= 25
        )

        assert condition.evaluate(sensor_reading=reading) is False

    def test_sensor_threshold_evaluation_all_operators(self, sensor_threshold_condition):
        """Test sensor threshold evaluation with all comparison operators."""
        condition = sensor_threshold_condition
        condition.threshold_value = 25.0

        reading = SensorReading(
            sensor_id=create_sensor_id("temp_sensor_001"),
            sensor_type=SensorType.TEMPERATURE,
            value=30.0,
        )

        # Test all operators
        condition.comparison_operator = ">"
        assert condition.evaluate(sensor_reading=reading) is True

        condition.comparison_operator = "<"
        assert condition.evaluate(sensor_reading=reading) is False

        condition.comparison_operator = ">="
        assert condition.evaluate(sensor_reading=reading) is True

        condition.comparison_operator = "<="
        assert condition.evaluate(sensor_reading=reading) is False

        condition.comparison_operator = "=="
        reading.value = 25.0
        assert condition.evaluate(sensor_reading=reading) is True

        condition.comparison_operator = "!="
        assert condition.evaluate(sensor_reading=reading) is False

    def test_sensor_threshold_evaluation_wrong_sensor(self, sensor_threshold_condition):
        """Test sensor threshold evaluation with wrong sensor ID."""
        condition = sensor_threshold_condition

        reading = SensorReading(
            sensor_id=create_sensor_id("wrong_sensor"),  # Different sensor
            sensor_type=SensorType.TEMPERATURE,
            value=30.0,
        )

        assert condition.evaluate(sensor_reading=reading) is False

    def test_sensor_threshold_evaluation_no_reading(self, sensor_threshold_condition):
        """Test sensor threshold evaluation with no reading."""
        condition = sensor_threshold_condition
        assert condition.evaluate(sensor_reading=None) is False

    def test_device_state_evaluation_true(self, device_state_condition):
        """Test device state evaluation (true)."""
        condition = device_state_condition

        device_state = {
            "front_door_001": "open"  # Matches threshold_value
        }

        assert condition.evaluate(device_state=device_state) is True

    def test_device_state_evaluation_false(self, device_state_condition):
        """Test device state evaluation (false)."""
        condition = device_state_condition

        device_state = {
            "front_door_001": "closed"  # Doesn't match threshold_value
        }

        assert condition.evaluate(device_state=device_state) is False

    def test_device_state_evaluation_no_state(self, device_state_condition):
        """Test device state evaluation with no state."""
        condition = device_state_condition
        assert condition.evaluate(device_state=None) is False

    @patch('src.core.iot_architecture.datetime')
    def test_time_schedule_evaluation_true(self, mock_datetime, time_schedule_condition):
        """Test time schedule evaluation (true)."""
        # Mock current time to match schedule
        mock_now = Mock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%H:%M": "07:00",
            "%A": "Monday"
        }[fmt]
        mock_datetime.now.return_value = mock_now

        condition = time_schedule_condition
        assert condition.evaluate() is True

    @patch('src.core.iot_architecture.datetime')
    def test_time_schedule_evaluation_wrong_time(self, mock_datetime, time_schedule_condition):
        """Test time schedule evaluation - wrong time."""
        mock_now = Mock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%H:%M": "08:00",  # Wrong time
            "%A": "Monday"
        }[fmt]
        mock_datetime.now.return_value = mock_now

        condition = time_schedule_condition
        assert condition.evaluate() is False

    @patch('src.core.iot_architecture.datetime')
    def test_time_schedule_evaluation_wrong_day(self, mock_datetime, time_schedule_condition):
        """Test time schedule evaluation - wrong day."""
        mock_now = Mock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%H:%M": "07:00",
            "%A": "Saturday"  # Wrong day
        }[fmt]
        mock_datetime.now.return_value = mock_now

        condition = time_schedule_condition
        assert condition.evaluate() is False

    def test_unknown_trigger_type_evaluation(self):
        """Test evaluation with unknown trigger type."""
        condition = AutomationCondition(
            condition_id="unknown_condition",
            trigger_type=AutomationTrigger.WEBHOOK,  # Not implemented
            enabled=True,
        )

        assert condition.evaluate() is False


class TestAutomationAction:
    """Comprehensive tests for AutomationAction class."""

    @pytest.fixture
    def device_control_action(self):
        """Create device control automation action."""
        return AutomationAction(
            action_id="turn_on_lights",
            action_type="device_control",
            device_id=create_device_id("living_room_lights"),
            action=DeviceAction.ON,
            parameters={"brightness": 80, "color": "warm_white"},
            delay_seconds=0,
            timeout_seconds=30,
            retry_attempts=2,
        )

    @pytest.fixture
    def scene_activation_action(self):
        """Create scene activation automation action."""
        return AutomationAction(
            action_id="activate_evening_scene",
            action_type="scene_activation",
            scene_id=create_scene_id("evening_lights"),
            parameters={"fade_duration": 5},
        )

    def test_automation_action_creation(self, device_control_action):
        """Test automation action creation."""
        action = device_control_action

        assert action.action_id == "turn_on_lights"
        assert action.action_type == "device_control"
        assert action.device_id == "living_room_lights"
        assert action.action == DeviceAction.ON
        assert action.parameters["brightness"] == 80
        assert action.delay_seconds == 0
        assert action.timeout_seconds == 30
        assert action.retry_attempts == 2

    @pytest.mark.asyncio
    async def test_automation_action_execute_success(self, device_control_action):
        """Test successful automation action execution."""
        action = device_control_action

        result = await action.execute()

        assert result.is_success()
        data = result.get_right()
        assert data["action_id"] == "turn_on_lights"
        assert data["action_type"] == "device_control"
        assert data["success"] is True
        assert "executed_at" in data

    @pytest.mark.asyncio
    async def test_automation_action_execute_with_delay(self, device_control_action):
        """Test automation action execution with delay."""
        action = device_control_action
        action.delay_seconds = 1

        start_time = datetime.now(UTC)
        result = await action.execute()
        end_time = datetime.now(UTC)

        assert result.is_success()
        # Verify delay was applied (should take at least 1 second)
        assert (end_time - start_time).total_seconds() >= 1.0

    @pytest.mark.asyncio
    async def test_automation_action_execute_scene_activation(self, scene_activation_action):
        """Test scene activation action execution."""
        action = scene_activation_action

        result = await action.execute()

        assert result.is_success()
        data = result.get_right()
        assert data["action_id"] == "activate_evening_scene"
        assert data["action_type"] == "scene_activation"

    @pytest.mark.asyncio
    async def test_automation_action_execute_with_context(self, device_control_action):
        """Test automation action execution with context."""
        action = device_control_action
        context = {"user_id": "user_001", "trigger_source": "manual"}

        result = await action.execute(context)

        assert result.is_success()

    def test_automation_action_defaults(self):
        """Test automation action with default values."""
        action = AutomationAction(
            action_id="minimal_action",
            action_type="notification",
        )

        assert action.device_id is None
        assert action.scene_id is None
        assert action.action is None
        assert action.parameters == {}
        assert action.delay_seconds == 0
        assert action.timeout_seconds == 30
        assert action.retry_attempts == 2
        assert action.required_device_status is None
        assert action.conditions == []


class TestSmartHomeScene:
    """Comprehensive tests for SmartHomeScene class."""

    @pytest.fixture
    def sample_scene(self):
        """Create sample smart home scene."""
        return SmartHomeScene(
            scene_id=create_scene_id("evening_relaxation"),
            scene_name="Evening Relaxation",
            description="Perfect lighting for evening relaxation",
            device_settings={
                create_device_id("living_room_lights"): {
                    "brightness": 30,
                    "color": "warm_white"
                },
                create_device_id("bedroom_lights"): {
                    "brightness": 20,
                    "color": "amber"
                }
            },
            actions=[
                AutomationAction(
                    action_id="dim_lights",
                    action_type="device_control",
                    device_id=create_device_id("living_room_lights"),
                    action=DeviceAction.SET_VALUE,
                    parameters={"brightness": 30}
                )
            ],
            activation_delay=2,
            fade_duration=5,
            icon="🌙",
            color="#FFE5B4",
            category="relaxation",
        )

    def test_smart_home_scene_creation(self, sample_scene):
        """Test smart home scene creation."""
        scene = sample_scene

        assert scene.scene_id == "evening_relaxation"
        assert scene.scene_name == "Evening Relaxation"
        assert scene.description == "Perfect lighting for evening relaxation"
        assert len(scene.device_settings) == 2
        assert len(scene.actions) == 1
        assert scene.activation_delay == 2
        assert scene.fade_duration == 5
        assert scene.icon == "🌙"
        assert scene.category == "relaxation"
        assert scene.activation_count == 0
        assert scene.last_activated is None
        assert scene.favorite is False

    @pytest.mark.asyncio
    async def test_smart_home_scene_activate_success(self, sample_scene):
        """Test successful scene activation."""
        scene = sample_scene

        result = await scene.activate()

        assert result.is_success()
        data = result.get_right()
        assert data["scene_id"] == "evening_relaxation"
        assert data["scene_name"] == "Evening Relaxation"
        assert data["actions_executed"] == 1
        assert "activated_at" in data

        # Verify activation tracking updated
        assert scene.activation_count == 1
        assert scene.last_activated is not None

    @pytest.mark.asyncio
    async def test_smart_home_scene_activate_with_delay(self, sample_scene):
        """Test scene activation with delay."""
        scene = sample_scene
        scene.activation_delay = 1

        start_time = datetime.now(UTC)
        result = await scene.activate()
        end_time = datetime.now(UTC)

        assert result.is_success()
        # Verify delay was applied
        assert (end_time - start_time).total_seconds() >= 1.0

    @pytest.mark.asyncio
    async def test_smart_home_scene_activate_with_context(self, sample_scene):
        """Test scene activation with context."""
        scene = sample_scene
        context = {"user_id": "user_001", "source": "voice_command"}

        result = await scene.activate(context)

        assert result.is_success()

    @pytest.mark.asyncio
    async def test_smart_home_scene_activate_no_actions(self):
        """Test scene activation with no actions."""
        scene = SmartHomeScene(
            scene_id=create_scene_id("empty_scene"),
            scene_name="Empty Scene",
            actions=[],  # No actions
        )

        result = await scene.activate()

        assert result.is_success()
        data = result.get_right()
        assert data["actions_executed"] == 0

    def test_smart_home_scene_defaults(self):
        """Test smart home scene with default values."""
        scene = SmartHomeScene(
            scene_id=create_scene_id("minimal_scene"),
            scene_name="Minimal Scene",
        )

        assert scene.description is None
        assert scene.device_settings == {}
        assert scene.actions == []
        assert scene.activation_delay == 0
        assert scene.fade_duration == 0
        assert scene.auto_activate_conditions == []
        assert scene.schedule is None
        assert scene.icon is None
        assert scene.color is None
        assert scene.category is None
        assert scene.activation_count == 0
        assert scene.last_activated is None
        assert scene.favorite is False


class TestIoTWorkflow:
    """Comprehensive tests for IoTWorkflow class."""

    @pytest.fixture
    def sample_workflow(self):
        """Create sample IoT workflow."""
        return IoTWorkflow(
            workflow_id=create_workflow_id("morning_routine"),
            workflow_name="Morning Routine",
            description="Automated morning routine workflow",
            triggers=[
                AutomationCondition(
                    condition_id="morning_time",
                    trigger_type=AutomationTrigger.TIME_SCHEDULE,
                    schedule_time="07:00",
                    schedule_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                )
            ],
            actions=[
                AutomationAction(
                    action_id="turn_on_lights",
                    action_type="device_control",
                    device_id=create_device_id("bedroom_lights"),
                    action=DeviceAction.ON,
                ),
                AutomationAction(
                    action_id="start_coffee",
                    action_type="device_control",
                    device_id=create_device_id("coffee_maker"),
                    action=DeviceAction.ON,
                ),
            ],
            execution_mode=WorkflowExecutionMode.SEQUENTIAL,
            max_execution_time=300,
            retry_on_failure=True,
            continue_on_error=False,
        )

    def test_iot_workflow_creation(self, sample_workflow):
        """Test IoT workflow creation."""
        workflow = sample_workflow

        assert workflow.workflow_id == "morning_routine"
        assert workflow.workflow_name == "Morning Routine"
        assert workflow.description == "Automated morning routine workflow"
        assert len(workflow.triggers) == 1
        assert len(workflow.actions) == 2
        assert workflow.execution_mode == WorkflowExecutionMode.SEQUENTIAL
        assert workflow.max_execution_time == 300
        assert workflow.retry_on_failure is True
        assert workflow.continue_on_error is False
        assert workflow.enabled is True
        assert workflow.execution_count == 0
        assert workflow.error_count == 0
        assert workflow.success_count == 0

    def test_iot_workflow_is_triggered_false_disabled(self, sample_workflow):
        """Test workflow trigger check when workflow is disabled."""
        workflow = sample_workflow
        workflow.enabled = False

        assert workflow.is_triggered() is False

    def test_iot_workflow_is_triggered_false_no_triggers(self, sample_workflow):
        """Test workflow trigger check with no triggers."""
        workflow = sample_workflow
        workflow.triggers = []

        assert workflow.is_triggered() is False

    def test_iot_workflow_is_triggered_true(self, sample_workflow):
        """Test workflow trigger check when conditions are met."""
        workflow = sample_workflow

        # Mock the trigger evaluation to return True
        with patch.object(workflow.triggers[0], 'evaluate', return_value=True):
            assert workflow.is_triggered() is True

    def test_iot_workflow_is_triggered_false(self, sample_workflow):
        """Test workflow trigger check when conditions are not met."""
        workflow = sample_workflow

        # Mock the trigger evaluation to return False
        with patch.object(workflow.triggers[0], 'evaluate', return_value=False):
            assert workflow.is_triggered() is False

    @pytest.mark.asyncio
    async def test_iot_workflow_execute_sequential_success(self, sample_workflow):
        """Test successful sequential workflow execution."""
        workflow = sample_workflow
        workflow.execution_mode = WorkflowExecutionMode.SEQUENTIAL

        result = await workflow.execute()

        assert result.is_success()
        data = result.get_right()
        assert data["workflow_id"] == "morning_routine"
        assert data["execution_mode"] == "sequential"
        assert data["actions_completed"] == 2

        # Verify metrics updated
        assert workflow.execution_count == 1
        assert workflow.success_count == 1
        assert workflow.error_count == 0
        assert workflow.last_execution is not None

    @pytest.mark.asyncio
    async def test_iot_workflow_execute_parallel_success(self, sample_workflow):
        """Test successful parallel workflow execution."""
        workflow = sample_workflow
        workflow.execution_mode = WorkflowExecutionMode.PARALLEL
        workflow.parallel_limit = 5

        result = await workflow.execute()

        assert result.is_success()
        data = result.get_right()
        assert data["workflow_id"] == "morning_routine"
        assert data["execution_mode"] == "parallel"

    @pytest.mark.asyncio
    async def test_iot_workflow_execute_conditional_fallback(self, sample_workflow):
        """Test conditional workflow execution (falls back to sequential)."""
        workflow = sample_workflow
        workflow.execution_mode = WorkflowExecutionMode.CONDITIONAL

        result = await workflow.execute()

        assert result.is_success()
        # Should fall back to sequential execution
        data = result.get_right()
        assert data["workflow_id"] == "morning_routine"

    @pytest.mark.asyncio
    async def test_iot_workflow_execute_with_context(self, sample_workflow):
        """Test workflow execution with context."""
        workflow = sample_workflow
        context = {"user_id": "user_001", "manual_trigger": True}

        result = await workflow.execute(context)

        assert result.is_success()

    @pytest.mark.asyncio
    async def test_iot_workflow_execute_continue_on_error(self, sample_workflow):
        """Test workflow execution with continue_on_error enabled."""
        workflow = sample_workflow
        workflow.continue_on_error = True

        # Mock one action to fail
        with patch.object(workflow.actions[0], 'execute', return_value=Either.error("Action failed")):
            result = await workflow.execute()

            # Should still succeed because continue_on_error is True
            assert result.is_success()

    @pytest.mark.asyncio
    async def test_iot_workflow_execute_stop_on_error(self, sample_workflow):
        """Test workflow execution with continue_on_error disabled."""
        workflow = sample_workflow
        workflow.continue_on_error = False

        # Mock one action to fail
        with patch.object(workflow.actions[0], 'execute', return_value=Either.error("Action failed")):
            result = await workflow.execute()

            # Should fail because continue_on_error is False
            assert result.is_error()

    def test_iot_workflow_defaults(self):
        """Test IoT workflow with default values."""
        workflow = IoTWorkflow(
            workflow_id=create_workflow_id("minimal_workflow"),
            workflow_name="Minimal Workflow",
        )

        assert workflow.description is None
        assert workflow.triggers == []
        assert workflow.actions == []
        assert workflow.execution_mode == WorkflowExecutionMode.SEQUENTIAL
        assert workflow.device_dependencies == {}
        assert workflow.required_devices == set()
        assert workflow.max_execution_time == 300
        assert workflow.retry_on_failure is True
        assert workflow.continue_on_error is False
        assert workflow.parallel_limit == 5
        assert workflow.timeout_seconds == 60
        assert workflow.performance_metrics == {}
        assert workflow.error_count == 0
        assert workflow.success_count == 0
        assert workflow.enabled is True
        assert workflow.last_execution is None
        assert workflow.execution_count == 0


class TestIoTIntegrationError:
    """Comprehensive tests for IoTIntegrationError class."""

    def test_iot_integration_error_basic(self):
        """Test basic IoT integration error creation."""
        error = IoTIntegrationError("Device connection failed")

        assert str(error) == "Device connection failed"
        assert error.device_id is None
        assert error.sensor_id is None

    def test_iot_integration_error_with_device_id(self):
        """Test IoT integration error with device ID."""
        device_id = create_device_id("failed_device")
        error = IoTIntegrationError("Device not responding", device_id=device_id)

        assert str(error) == "Device not responding"
        assert error.device_id == device_id
        assert error.sensor_id is None

    def test_iot_integration_error_with_sensor_id(self):
        """Test IoT integration error with sensor ID."""
        sensor_id = create_sensor_id("failed_sensor")
        error = IoTIntegrationError("Sensor reading failed", sensor_id=sensor_id)

        assert str(error) == "Sensor reading failed"
        assert error.device_id is None
        assert error.sensor_id == sensor_id

    def test_iot_integration_error_with_both_ids(self):
        """Test IoT integration error with both device and sensor IDs."""
        device_id = create_device_id("failed_device")
        sensor_id = create_sensor_id("failed_sensor")
        error = IoTIntegrationError(
            "Device and sensor communication failed",
            device_id=device_id,
            sensor_id=sensor_id
        )

        assert str(error) == "Device and sensor communication failed"
        assert error.device_id == device_id
        assert error.sensor_id == sensor_id


class TestDeviceValidation:
    """Comprehensive tests for device validation functions."""

    @pytest.fixture
    def valid_device(self):
        """Create valid IoT device for testing."""
        return IoTDevice(
            device_id=create_device_id("valid_device"),
            device_name="Valid Device",
            device_type=DeviceType.LIGHT,
            protocol=IoTProtocol.WIFI,
            address=create_protocol_address("192.168.1.100"),
            security_level=SecurityLevel.STANDARD,
            encryption_enabled=True,
        )

    def test_validate_device_configuration_success(self, valid_device):
        """Test successful device validation."""
        result = validate_device_configuration(valid_device)

        assert result.is_success()
        assert result.get_right() is True

    def test_validate_device_configuration_invalid_device_id(self, valid_device):
        """Test device validation with invalid device ID."""
        # Create device with invalid ID directly (bypassing create_device_id validation)
        invalid_device = IoTDevice(
            device_id="",  # Empty device ID
            device_name="Invalid Device",
            device_type=DeviceType.LIGHT,
            protocol=IoTProtocol.WIFI,
            address=create_protocol_address("192.168.1.100"),
        )

        result = validate_device_configuration(invalid_device)

        assert result.is_error()
        error = result.get_left()
        assert isinstance(error, IoTIntegrationError)
        assert "Invalid device ID" in str(error)

    def test_validate_device_configuration_long_device_id(self, valid_device):
        """Test device validation with too long device ID."""
        # Create device with long ID directly
        invalid_device = IoTDevice(
            device_id="x" * 101,  # Too long device ID
            device_name="Invalid Device",
            device_type=DeviceType.LIGHT,
            protocol=IoTProtocol.WIFI,
            address=create_protocol_address("192.168.1.100"),
        )

        result = validate_device_configuration(invalid_device)

        assert result.is_error()
        error = result.get_left()
        assert isinstance(error, IoTIntegrationError)
        assert "Invalid device ID" in str(error)

    def test_validate_device_configuration_missing_address(self, valid_device):
        """Test device validation with missing protocol address."""
        invalid_device = IoTDevice(
            device_id=create_device_id("valid_device"),
            device_name="Invalid Device",
            device_type=DeviceType.LIGHT,
            protocol=IoTProtocol.WIFI,
            address="",  # Empty address
        )

        result = validate_device_configuration(invalid_device)

        assert result.is_error()
        error = result.get_left()
        assert isinstance(error, IoTIntegrationError)
        assert "Protocol address required" in str(error)

    def test_validate_device_configuration_security_mismatch(self, valid_device):
        """Test device validation with security level/encryption mismatch."""
        invalid_device = IoTDevice(
            device_id=create_device_id("invalid_security_device"),
            device_name="Security Mismatch Device",
            device_type=DeviceType.LIGHT,
            protocol=IoTProtocol.WIFI,
            address=create_protocol_address("192.168.1.100"),
            security_level=SecurityLevel.MAXIMUM,
            encryption_enabled=False,  # Encryption required for maximum security
        )

        result = validate_device_configuration(invalid_device)

        assert result.is_error()
        error = result.get_left()
        assert isinstance(error, IoTIntegrationError)
        assert "Encryption required for maximum security" in str(error)


class TestDefaultCapabilities:
    """Comprehensive tests for default device capabilities function."""

    def test_create_default_device_capabilities_light(self):
        """Test default capabilities for light device."""
        capabilities = create_default_device_capabilities(DeviceType.LIGHT)
        expected = ["on_off", "brightness", "color"]
        assert capabilities == expected

    def test_create_default_device_capabilities_switch(self):
        """Test default capabilities for switch device."""
        capabilities = create_default_device_capabilities(DeviceType.SWITCH)
        expected = ["on_off"]
        assert capabilities == expected

    def test_create_default_device_capabilities_sensor(self):
        """Test default capabilities for sensor device."""
        capabilities = create_default_device_capabilities(DeviceType.SENSOR)
        expected = ["read_value", "get_status"]
        assert capabilities == expected

    def test_create_default_device_capabilities_thermostat(self):
        """Test default capabilities for thermostat device."""
        capabilities = create_default_device_capabilities(DeviceType.THERMOSTAT)
        expected = ["temperature_control", "mode_setting", "schedule"]
        assert capabilities == expected

    def test_create_default_device_capabilities_camera(self):
        """Test default capabilities for camera device."""
        capabilities = create_default_device_capabilities(DeviceType.CAMERA)
        expected = ["video_stream", "recording", "motion_detection"]
        assert capabilities == expected

    def test_create_default_device_capabilities_lock(self):
        """Test default capabilities for lock device."""
        capabilities = create_default_device_capabilities(DeviceType.LOCK)
        expected = ["lock_unlock", "status_check", "access_log"]
        assert capabilities == expected

    def test_create_default_device_capabilities_garage_door(self):
        """Test default capabilities for garage door device."""
        capabilities = create_default_device_capabilities(DeviceType.GARAGE_DOOR)
        expected = ["open_close", "status_check"]
        assert capabilities == expected

    def test_create_default_device_capabilities_sprinkler(self):
        """Test default capabilities for sprinkler device."""
        capabilities = create_default_device_capabilities(DeviceType.SPRINKLER)
        expected = ["on_off", "zone_control", "schedule"]
        assert capabilities == expected

    def test_create_default_device_capabilities_fan(self):
        """Test default capabilities for fan device."""
        capabilities = create_default_device_capabilities(DeviceType.FAN)
        expected = ["on_off", "speed_control", "oscillation"]
        assert capabilities == expected

    def test_create_default_device_capabilities_blinds(self):
        """Test default capabilities for blinds device."""
        capabilities = create_default_device_capabilities(DeviceType.BLINDS)
        expected = ["open_close", "position_control"]
        assert capabilities == expected

    def test_create_default_device_capabilities_speaker(self):
        """Test default capabilities for speaker device."""
        capabilities = create_default_device_capabilities(DeviceType.SPEAKER)
        expected = ["audio_playback", "volume_control", "source_selection"]
        assert capabilities == expected

    def test_create_default_device_capabilities_tv(self):
        """Test default capabilities for TV device."""
        capabilities = create_default_device_capabilities(DeviceType.TV)
        expected = ["power_control", "channel_control", "volume_control"]
        assert capabilities == expected

    def test_create_default_device_capabilities_appliance(self):
        """Test default capabilities for appliance device."""
        capabilities = create_default_device_capabilities(DeviceType.APPLIANCE)
        expected = ["on_off", "mode_setting"]
        assert capabilities == expected

    def test_create_default_device_capabilities_security(self):
        """Test default capabilities for security device."""
        capabilities = create_default_device_capabilities(DeviceType.SECURITY)
        expected = ["status_monitoring", "alert_generation"]
        assert capabilities == expected

    def test_create_default_device_capabilities_energy(self):
        """Test default capabilities for energy device."""
        capabilities = create_default_device_capabilities(DeviceType.ENERGY)
        expected = ["consumption_monitoring", "control"]
        assert capabilities == expected

    def test_create_default_device_capabilities_weather(self):
        """Test default capabilities for weather device."""
        capabilities = create_default_device_capabilities(DeviceType.WEATHER)
        expected = ["data_collection", "forecasting"]
        assert capabilities == expected

    def test_create_default_device_capabilities_air_quality(self):
        """Test default capabilities for air quality device."""
        capabilities = create_default_device_capabilities(DeviceType.AIR_QUALITY)
        expected = ["air_monitoring", "purification"]
        assert capabilities == expected

    def test_create_default_device_capabilities_motion(self):
        """Test default capabilities for motion device."""
        capabilities = create_default_device_capabilities(DeviceType.MOTION)
        expected = ["motion_detection", "occupancy_sensing"]
        assert capabilities == expected

    def test_create_default_device_capabilities_custom(self):
        """Test default capabilities for custom device."""
        capabilities = create_default_device_capabilities(DeviceType.CUSTOM)
        expected = ["custom_control"]
        assert capabilities == expected

    def test_create_default_device_capabilities_unknown_type(self):
        """Test default capabilities for unknown device type."""
        # Create a mock device type that's not in the map
        with patch.object(DeviceType, 'UNKNOWN_TYPE', 'unknown_type', create=True):
            capabilities = create_default_device_capabilities(DeviceType.UNKNOWN_TYPE)
            expected = ["basic_control"]
            assert capabilities == expected


class TestModuleExports:
    """Test module exports and all functionality."""

    def test_module_exports_available(self):
        """Test that all expected exports are available."""
        from src.core.iot_architecture import __all__

        # Verify key exports are present
        expected_exports = [
            "IoTDevice",
            "SensorReading",
            "AutomationCondition",
            "AutomationAction",
            "SmartHomeScene",
            "IoTWorkflow",
            "IoTIntegrationError",
            "DeviceId",
            "SensorId",
            "SceneId",
            "WorkflowId",
            "ProtocolAddress",
            "create_device_id",
            "create_sensor_id",
            "create_scene_id",
            "create_workflow_id",
            "create_protocol_address",
            "validate_device_configuration",
            "create_default_device_capabilities",
        ]

        for export in expected_exports:
            assert export in __all__, f"Expected export {export} not found in __all__"

    def test_import_all_classes(self):
        """Test importing all main classes."""
        # Test that all classes can be imported successfully
        classes = [
            IoTDevice,
            SensorReading,
            AutomationCondition,
            AutomationAction,
            SmartHomeScene,
            IoTWorkflow,
            IoTIntegrationError,
        ]

        for cls in classes:
            assert cls is not None
            assert hasattr(cls, '__name__')

    def test_import_all_enums(self):
        """Test importing all enumeration classes."""
        enums = [
            IoTProtocol,
            DeviceType,
            SensorType,
            DeviceAction,
            AutomationTrigger,
            WorkflowExecutionMode,
            SecurityLevel,
            DeviceStatus,
        ]

        for enum_cls in enums:
            assert enum_cls is not None
            assert hasattr(enum_cls, '__members__')

    def test_import_all_functions(self):
        """Test importing all utility functions."""
        functions = [
            create_device_id,
            create_sensor_id,
            create_scene_id,
            create_workflow_id,
            create_protocol_address,
            validate_device_configuration,
            create_default_device_capabilities,
        ]

        for func in functions:
            assert func is not None
            assert callable(func)
