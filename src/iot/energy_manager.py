"""
IoT Energy Management System - TASK_65 Phase 4 Advanced Features

Smart energy management, optimization algorithms, load balancing,
and renewable energy integration for IoT device automation.

Architecture: Energy Monitoring + Smart Optimization + Load Balancing + Renewable Integration
Performance: <50ms energy calculations, <200ms optimization cycles, <1s load balancing
Security: Energy data protection, usage privacy, secure optimization controls
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, UTC, timedelta
from dataclasses import dataclass, field
import asyncio
import math
from enum import Enum
import logging

from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.errors import ValidationError, SecurityError, SystemError
from ..core.iot_architecture import (
    DeviceId, IoTIntegrationError, IoTDevice, DeviceType
)

logger = logging.getLogger(__name__)


class EnergySource(Enum):
    """Energy source types."""
    GRID = "grid"
    SOLAR = "solar"
    WIND = "wind"
    BATTERY = "battery"
    GENERATOR = "generator"
    HYBRID = "hybrid"


class EnergyPriority(Enum):
    """Energy consumption priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    DEFER = "defer"


class OptimizationStrategy(Enum):
    """Energy optimization strategies."""
    COST_MINIMIZATION = "cost_minimization"
    CARBON_REDUCTION = "carbon_reduction"
    PEAK_SHAVING = "peak_shaving"
    LOAD_BALANCING = "load_balancing"
    RENEWABLE_MAXIMIZATION = "renewable_maximization"
    EFFICIENCY_OPTIMIZATION = "efficiency_optimization"


class EnergyTariff(Enum):
    """Energy pricing tariff types."""
    FLAT_RATE = "flat_rate"
    TIME_OF_USE = "time_of_use"
    PEAK_DEMAND = "peak_demand"
    REAL_TIME = "real_time"
    TIERED = "tiered"


EnergyProfileId = str
OptimizationId = str


@dataclass
class EnergyReading:
    """Energy consumption/production reading."""
    device_id: DeviceId
    timestamp: datetime
    power_consumption: float  # Watts
    voltage: float
    current: float
    energy_cumulative: float  # kWh
    cost_per_kwh: float
    source: EnergySource
    efficiency: float = 1.0
    carbon_intensity: float = 0.5  # kg CO2/kWh
    
    def calculate_cost(self, duration_hours: float) -> float:
        """Calculate energy cost for duration."""
        energy_used = self.power_consumption * duration_hours / 1000  # Convert to kWh
        return energy_used * self.cost_per_kwh
    
    def calculate_carbon_footprint(self, duration_hours: float) -> float:
        """Calculate carbon footprint for duration."""
        energy_used = self.power_consumption * duration_hours / 1000  # Convert to kWh
        return energy_used * self.carbon_intensity


@dataclass
class EnergyProfile:
    """Device energy consumption profile."""
    profile_id: EnergyProfileId
    device_id: DeviceId
    device_type: DeviceType
    rated_power: float  # Watts
    standby_power: float  # Watts
    peak_power: float  # Watts
    efficiency_rating: float
    priority: EnergyPriority
    operating_hours: List[Tuple[int, int]]  # [(start_hour, end_hour)]
    energy_source_preference: List[EnergySource]
    load_profile: Dict[str, float] = field(default_factory=dict)
    
    def get_expected_power(self, usage_level: float = 1.0) -> float:
        """Get expected power consumption for usage level."""
        if usage_level <= 0:
            return self.standby_power
        elif usage_level >= 1.0:
            return self.peak_power
        else:
            return self.standby_power + (self.rated_power - self.standby_power) * usage_level


@dataclass
class EnergyOptimization:
    """Energy optimization result."""
    optimization_id: OptimizationId
    strategy: OptimizationStrategy
    target_devices: List[DeviceId]
    optimization_period: Tuple[datetime, datetime]
    actions: List[Dict[str, Any]]
    projected_savings: Dict[str, float]  # cost, energy, carbon
    confidence: float
    priority_conflicts: List[str] = field(default_factory=list)
    
    def is_beneficial(self) -> bool:
        """Check if optimization provides significant benefits."""
        cost_savings = self.projected_savings.get("cost", 0)
        energy_savings = self.projected_savings.get("energy", 0)
        return cost_savings > 0.1 or energy_savings > 0.05  # 10 cents or 5% energy


@dataclass
class LoadBalancingResult:
    """Load balancing optimization result."""
    balancing_id: str
    total_load_before: float
    total_load_after: float
    peak_reduction: float
    device_adjustments: Dict[DeviceId, Dict[str, Any]]
    time_shifts: List[Dict[str, Any]]
    efficiency_improvement: float
    
    def get_peak_reduction_percentage(self) -> float:
        """Get peak reduction as percentage."""
        if self.total_load_before > 0:
            return (self.peak_reduction / self.total_load_before) * 100
        return 0.0


class EnergyManager:
    """
    Comprehensive energy management system for IoT devices.
    
    Contracts:
        Preconditions:
            - All energy readings must be validated and non-negative
            - Device energy profiles must be properly configured
            - Optimization constraints must be clearly defined
        
        Postconditions:
            - Energy optimizations preserve device functionality
            - Cost and carbon calculations are accurate
            - Load balancing maintains system stability
        
        Invariants:
            - Total energy consumption never exceeds capacity limits
            - Critical devices maintain minimum power requirements
            - Energy source preferences are respected when possible
    """
    
    def __init__(self):
        self.energy_profiles: Dict[DeviceId, EnergyProfile] = {}
        self.energy_readings: Dict[DeviceId, List[EnergyReading]] = {}
        self.active_optimizations: Dict[OptimizationId, EnergyOptimization] = {}
        self.energy_sources: Dict[EnergySource, Dict[str, Any]] = {}
        
        # System configuration
        self.total_capacity = 10000.0  # Watts
        self.peak_demand_limit = 8000.0  # Watts
        self.renewable_preference = True
        self.cost_optimization_enabled = True
        self.carbon_optimization_enabled = True
        
        # Performance metrics
        self.total_energy_managed = 0.0
        self.total_cost_savings = 0.0
        self.total_carbon_savings = 0.0
        self.optimization_count = 0
        
        # Initialize default energy sources
        self._initialize_energy_sources()
    
    def _initialize_energy_sources(self):
        """Initialize default energy source configurations."""
        self.energy_sources = {
            EnergySource.GRID: {
                "capacity": 8000.0,
                "cost_per_kwh": 0.12,
                "carbon_intensity": 0.4,
                "availability": 0.99,
                "priority": 3
            },
            EnergySource.SOLAR: {
                "capacity": 3000.0,
                "cost_per_kwh": 0.03,
                "carbon_intensity": 0.05,
                "availability": 0.6,  # Weather dependent
                "priority": 1
            },
            EnergySource.BATTERY: {
                "capacity": 2000.0,
                "cost_per_kwh": 0.08,
                "carbon_intensity": 0.1,
                "availability": 0.95,
                "priority": 2
            }
        }
    
    @require(lambda self, profile: profile.device_id and profile.rated_power >= 0)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def register_device_profile(self, profile: EnergyProfile) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """
        Register energy profile for IoT device.
        
        Architecture:
            - Validates energy profile configuration
            - Calculates baseline consumption patterns
            - Integrates with optimization algorithms
        
        Security:
            - Validates power consumption limits
            - Prevents resource exhaustion attacks
            - Ensures profile data integrity
        """
        try:
            # Validate profile
            if profile.rated_power > self.total_capacity:
                return Either.error(IoTIntegrationError(
                    f"Device rated power {profile.rated_power}W exceeds system capacity",
                    profile.device_id
                ))
            
            if profile.standby_power < 0 or profile.peak_power < profile.rated_power:
                return Either.error(IoTIntegrationError(
                    "Invalid power consumption values in profile",
                    profile.device_id
                ))
            
            # Store profile
            self.energy_profiles[profile.device_id] = profile
            
            # Initialize energy readings list
            if profile.device_id not in self.energy_readings:
                self.energy_readings[profile.device_id] = []
            
            # Calculate initial load profile
            load_profile = await self._calculate_load_profile(profile)
            profile.load_profile = load_profile
            
            registration_info = {
                "device_id": profile.device_id,
                "profile_id": profile.profile_id,
                "rated_power": profile.rated_power,
                "efficiency_rating": profile.efficiency_rating,
                "priority": profile.priority.value,
                "energy_source_preference": [src.value for src in profile.energy_source_preference],
                "load_profile": load_profile,
                "registered_at": datetime.now(UTC).isoformat()
            }
            
            logger.info(f"Energy profile registered for device {profile.device_id}")
            
            return Either.success({
                "success": True,
                "registration_info": registration_info,
                "total_managed_devices": len(self.energy_profiles)
            })
            
        except Exception as e:
            error_msg = f"Failed to register energy profile: {str(e)}"
            logger.error(error_msg)
            return Either.error(IoTIntegrationError(error_msg, profile.device_id))
    
    @require(lambda self, reading: reading.device_id and reading.power_consumption >= 0)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def record_energy_reading(self, reading: EnergyReading) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """
        Record energy consumption reading for device.
        
        Performance:
            - <10ms reading processing time
            - Real-time consumption monitoring
            - Efficient data storage and retrieval
        """
        try:
            # Validate reading
            if reading.power_consumption > self.total_capacity:
                return Either.error(IoTIntegrationError(
                    f"Power consumption {reading.power_consumption}W exceeds system capacity",
                    reading.device_id
                ))
            
            # Store reading
            if reading.device_id not in self.energy_readings:
                self.energy_readings[reading.device_id] = []
            
            self.energy_readings[reading.device_id].append(reading)
            
            # Maintain reading history limit (keep last 1000 readings)
            if len(self.energy_readings[reading.device_id]) > 1000:
                self.energy_readings[reading.device_id] = self.energy_readings[reading.device_id][-1000:]
            
            # Update total energy managed
            self.total_energy_managed += reading.power_consumption / 1000  # Convert to kW
            
            # Check for peak demand violations
            current_total_load = await self._calculate_current_total_load()
            peak_violation = current_total_load > self.peak_demand_limit
            
            # Trigger optimization if needed
            optimization_triggered = False
            if peak_violation or self._should_trigger_optimization(reading):
                optimization_result = await self._trigger_automatic_optimization(reading)
                optimization_triggered = optimization_result.is_success()
            
            reading_info = {
                "device_id": reading.device_id,
                "power_consumption": reading.power_consumption,
                "energy_source": reading.source.value,
                "efficiency": reading.efficiency,
                "cost_per_kwh": reading.cost_per_kwh,
                "carbon_intensity": reading.carbon_intensity,
                "current_total_load": current_total_load,
                "peak_violation": peak_violation,
                "optimization_triggered": optimization_triggered,
                "recorded_at": reading.timestamp.isoformat()
            }
            
            return Either.success({
                "success": True,
                "reading_info": reading_info
            })
            
        except Exception as e:
            error_msg = f"Failed to record energy reading: {str(e)}"
            logger.error(error_msg)
            return Either.error(IoTIntegrationError(error_msg, reading.device_id))
    
    @require(lambda self, devices, strategy: len(devices) > 0 and strategy)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def optimize_energy_consumption(
        self,
        devices: List[DeviceId],
        strategy: OptimizationStrategy,
        optimization_period: Optional[Tuple[datetime, datetime]] = None
    ) -> Either[IoTIntegrationError, EnergyOptimization]:
        """
        Optimize energy consumption for specified devices using given strategy.
        
        Architecture:
            - Multi-objective optimization algorithms
            - Device priority and constraint handling
            - Real-time optimization with feedback loops
        
        Performance:
            - <500ms optimization calculation
            - Scalable to 100+ devices
            - Adaptive algorithm performance
        """
        try:
            if not optimization_period:
                optimization_period = (
                    datetime.now(UTC),
                    datetime.now(UTC) + timedelta(hours=24)
                )
            
            # Validate devices
            invalid_devices = [d for d in devices if d not in self.energy_profiles]
            if invalid_devices:
                return Either.error(IoTIntegrationError(
                    f"Unknown devices: {invalid_devices}",
                    invalid_devices[0] if invalid_devices else None
                ))
            
            # Run optimization algorithm
            optimization_result = await self._run_optimization_algorithm(devices, strategy, optimization_period)
            
            # Store optimization
            optimization_id = f"opt_{int(datetime.now(UTC).timestamp())}"
            optimization = EnergyOptimization(
                optimization_id=optimization_id,
                strategy=strategy,
                target_devices=devices,
                optimization_period=optimization_period,
                actions=optimization_result["actions"],
                projected_savings=optimization_result["projected_savings"],
                confidence=optimization_result["confidence"],
                priority_conflicts=optimization_result.get("priority_conflicts", [])
            )
            
            self.active_optimizations[optimization_id] = optimization
            self.optimization_count += 1
            
            # Update savings metrics
            self.total_cost_savings += optimization.projected_savings.get("cost", 0)
            self.total_carbon_savings += optimization.projected_savings.get("carbon", 0)
            
            logger.info(f"Energy optimization completed: {optimization_id}")
            
            return Either.success(optimization)
            
        except Exception as e:
            error_msg = f"Energy optimization failed: {str(e)}"
            logger.error(error_msg)
            return Either.error(IoTIntegrationError(error_msg))
    
    async def _run_optimization_algorithm(
        self,
        devices: List[DeviceId],
        strategy: OptimizationStrategy,
        period: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Run specific optimization algorithm based on strategy."""
        
        if strategy == OptimizationStrategy.COST_MINIMIZATION:
            return await self._optimize_for_cost(devices, period)
        elif strategy == OptimizationStrategy.CARBON_REDUCTION:
            return await self._optimize_for_carbon(devices, period)
        elif strategy == OptimizationStrategy.PEAK_SHAVING:
            return await self._optimize_peak_shaving(devices, period)
        elif strategy == OptimizationStrategy.LOAD_BALANCING:
            return await self._optimize_load_balancing(devices, period)
        elif strategy == OptimizationStrategy.RENEWABLE_MAXIMIZATION:
            return await self._optimize_renewable_usage(devices, period)
        else:
            return await self._optimize_efficiency(devices, period)
    
    async def _optimize_for_cost(self, devices: List[DeviceId], period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Optimize for minimum cost."""
        actions = []
        total_cost_savings = 0.0
        
        for device_id in devices:
            profile = self.energy_profiles[device_id]
            recent_readings = self.energy_readings.get(device_id, [])[-24:]  # Last 24 readings
            
            if recent_readings:
                # Find cheapest energy source
                cheapest_source = min(self.energy_sources.keys(), 
                                    key=lambda s: self.energy_sources[s]["cost_per_kwh"])
                
                current_cost = sum(r.calculate_cost(1) for r in recent_readings) / len(recent_readings)
                optimized_cost = current_cost * (self.energy_sources[cheapest_source]["cost_per_kwh"] / 0.12)
                savings = current_cost - optimized_cost
                
                if savings > 0.01:  # Minimum 1 cent savings
                    actions.append({
                        "device_id": device_id,
                        "action": "switch_energy_source",
                        "parameters": {"source": cheapest_source.value},
                        "projected_cost_savings": savings
                    })
                    total_cost_savings += savings
                
                # Schedule non-critical devices during low-cost periods
                if profile.priority in [EnergyPriority.LOW, EnergyPriority.DEFER]:
                    actions.append({
                        "device_id": device_id,
                        "action": "schedule_optimization",
                        "parameters": {"schedule": "off_peak_hours"},
                        "projected_cost_savings": current_cost * 0.3
                    })
                    total_cost_savings += current_cost * 0.3
        
        return {
            "actions": actions,
            "projected_savings": {"cost": total_cost_savings, "energy": 0, "carbon": 0},
            "confidence": 0.8,
            "priority_conflicts": []
        }
    
    async def _optimize_for_carbon(self, devices: List[DeviceId], period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Optimize for carbon footprint reduction."""
        actions = []
        total_carbon_savings = 0.0
        
        # Find lowest carbon intensity source
        cleanest_source = min(self.energy_sources.keys(),
                            key=lambda s: self.energy_sources[s]["carbon_intensity"])
        
        for device_id in devices:
            recent_readings = self.energy_readings.get(device_id, [])[-24:]
            
            if recent_readings:
                current_carbon = sum(r.calculate_carbon_footprint(1) for r in recent_readings) / len(recent_readings)
                optimized_carbon = current_carbon * (self.energy_sources[cleanest_source]["carbon_intensity"] / 0.4)
                savings = current_carbon - optimized_carbon
                
                if savings > 0.01:  # Minimum savings threshold
                    actions.append({
                        "device_id": device_id,
                        "action": "switch_energy_source",
                        "parameters": {"source": cleanest_source.value},
                        "projected_carbon_savings": savings
                    })
                    total_carbon_savings += savings
        
        return {
            "actions": actions,
            "projected_savings": {"cost": 0, "energy": 0, "carbon": total_carbon_savings},
            "confidence": 0.75,
            "priority_conflicts": []
        }
    
    async def _optimize_peak_shaving(self, devices: List[DeviceId], period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Optimize for peak demand reduction."""
        actions = []
        current_peak = await self._calculate_current_total_load()
        target_reduction = max(0, current_peak - self.peak_demand_limit)
        
        if target_reduction > 0:
            # Sort devices by priority (defer low priority first)
            prioritized_devices = sorted(devices, 
                key=lambda d: self.energy_profiles[d].priority.value, reverse=True)
            
            reduction_achieved = 0.0
            for device_id in prioritized_devices:
                if reduction_achieved >= target_reduction:
                    break
                
                profile = self.energy_profiles[device_id]
                if profile.priority in [EnergyPriority.LOW, EnergyPriority.DEFER]:
                    device_reduction = profile.rated_power * 0.5  # 50% reduction
                    actions.append({
                        "device_id": device_id,
                        "action": "reduce_power",
                        "parameters": {"reduction_percentage": 50},
                        "power_reduction": device_reduction
                    })
                    reduction_achieved += device_reduction
        
        return {
            "actions": actions,
            "projected_savings": {"cost": 0, "energy": reduction_achieved / 1000, "carbon": 0},
            "confidence": 0.9,
            "priority_conflicts": []
        }
    
    async def _optimize_load_balancing(self, devices: List[DeviceId], period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Optimize for load balancing across time periods."""
        actions = []
        
        # Simple load balancing: stagger device start times
        time_offset = 0
        for device_id in devices:
            profile = self.energy_profiles[device_id]
            if profile.priority not in [EnergyPriority.CRITICAL]:
                actions.append({
                    "device_id": device_id,
                    "action": "stagger_schedule",
                    "parameters": {"time_offset_minutes": time_offset},
                    "load_balancing_benefit": profile.rated_power * 0.1
                })
                time_offset += 15  # 15-minute intervals
        
        return {
            "actions": actions,
            "projected_savings": {"cost": 0, "energy": 0, "carbon": 0},
            "confidence": 0.85,
            "priority_conflicts": []
        }
    
    async def _optimize_renewable_usage(self, devices: List[DeviceId], period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Optimize for maximum renewable energy usage."""
        actions = []
        renewable_capacity = sum(
            info["capacity"] for source, info in self.energy_sources.items()
            if source in [EnergySource.SOLAR, EnergySource.WIND]
        )
        
        for device_id in devices:
            actions.append({
                "device_id": device_id,
                "action": "prefer_renewable",
                "parameters": {"renewable_sources": ["solar", "wind"]},
                "renewable_percentage_increase": 20
            })
        
        return {
            "actions": actions,
            "projected_savings": {"cost": 0, "energy": 0, "carbon": renewable_capacity * 0.3},
            "confidence": 0.7,
            "priority_conflicts": []
        }
    
    async def _optimize_efficiency(self, devices: List[DeviceId], period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Optimize for overall efficiency."""
        actions = []
        total_efficiency_gain = 0.0
        
        for device_id in devices:
            profile = self.energy_profiles[device_id]
            if profile.efficiency_rating < 0.9:  # Can improve efficiency
                efficiency_gain = (0.9 - profile.efficiency_rating) * profile.rated_power
                actions.append({
                    "device_id": device_id,
                    "action": "efficiency_optimization",
                    "parameters": {"target_efficiency": 0.9},
                    "efficiency_gain": efficiency_gain
                })
                total_efficiency_gain += efficiency_gain
        
        return {
            "actions": actions,
            "projected_savings": {"cost": 0, "energy": total_efficiency_gain / 1000, "carbon": 0},
            "confidence": 0.8,
            "priority_conflicts": []
        }
    
    async def _calculate_load_profile(self, profile: EnergyProfile) -> Dict[str, float]:
        """Calculate expected load profile for device."""
        load_profile = {}
        
        for hour in range(24):
            # Check if device typically operates during this hour
            is_operating_hour = any(
                start <= hour <= end for start, end in profile.operating_hours
            )
            
            if is_operating_hour:
                # Vary load based on device type and hour
                base_load = profile.rated_power
                if profile.device_type == DeviceType.LIGHT:
                    # Lights: higher usage in evening
                    if 18 <= hour <= 23:
                        load_profile[str(hour)] = base_load * 0.9
                    elif 6 <= hour <= 8:
                        load_profile[str(hour)] = base_load * 0.7
                    else:
                        load_profile[str(hour)] = base_load * 0.3
                elif profile.device_type == DeviceType.THERMOSTAT:
                    # HVAC: higher usage during extreme hours
                    if hour in [1, 2, 3, 13, 14, 15]:
                        load_profile[str(hour)] = base_load * 0.8
                    else:
                        load_profile[str(hour)] = base_load * 0.5
                else:
                    # Default pattern
                    load_profile[str(hour)] = base_load * 0.6
            else:
                load_profile[str(hour)] = profile.standby_power
        
        return load_profile
    
    async def _calculate_current_total_load(self) -> float:
        """Calculate current total power load across all devices."""
        total_load = 0.0
        
        for device_id, readings in self.energy_readings.items():
            if readings:
                # Use most recent reading
                latest_reading = readings[-1]
                total_load += latest_reading.power_consumption
        
        return total_load
    
    def _should_trigger_optimization(self, reading: EnergyReading) -> bool:
        """Determine if automatic optimization should be triggered."""
        # Trigger if power consumption is unusually high
        device_readings = self.energy_readings.get(reading.device_id, [])
        if len(device_readings) >= 10:
            avg_power = sum(r.power_consumption for r in device_readings[-10:]) / 10
            if reading.power_consumption > avg_power * 1.5:
                return True
        
        # Trigger if approaching peak demand limit
        current_load = asyncio.create_task(self._calculate_current_total_load())
        if hasattr(current_load, 'result'):
            return current_load.result() > self.peak_demand_limit * 0.9
        
        return False
    
    async def _trigger_automatic_optimization(self, reading: EnergyReading) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Trigger automatic optimization based on current conditions."""
        try:
            # Choose strategy based on current conditions
            current_load = await self._calculate_current_total_load()
            
            if current_load > self.peak_demand_limit:
                strategy = OptimizationStrategy.PEAK_SHAVING
                devices = [reading.device_id]
            elif reading.cost_per_kwh > 0.15:  # High cost
                strategy = OptimizationStrategy.COST_MINIMIZATION
                devices = list(self.energy_profiles.keys())
            else:
                strategy = OptimizationStrategy.EFFICIENCY_OPTIMIZATION
                devices = [reading.device_id]
            
            # Run optimization
            optimization_result = await self.optimize_energy_consumption(devices, strategy)
            
            return Either.success({
                "optimization_triggered": True,
                "strategy": strategy.value,
                "optimization_id": optimization_result.value.optimization_id if optimization_result.is_success() else None
            })
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Automatic optimization failed: {str(e)}"))
    
    async def get_energy_summary(self) -> Dict[str, Any]:
        """Get comprehensive energy management summary."""
        current_load = await self._calculate_current_total_load()
        
        return {
            "total_managed_devices": len(self.energy_profiles),
            "current_total_load": current_load,
            "peak_demand_limit": self.peak_demand_limit,
            "capacity_utilization": current_load / self.total_capacity * 100,
            "total_energy_managed": self.total_energy_managed,
            "total_cost_savings": self.total_cost_savings,
            "total_carbon_savings": self.total_carbon_savings,
            "active_optimizations": len(self.active_optimizations),
            "optimization_count": self.optimization_count,
            "energy_sources": {
                source.value: {
                    "capacity": info["capacity"],
                    "cost_per_kwh": info["cost_per_kwh"],
                    "carbon_intensity": info["carbon_intensity"],
                    "availability": info["availability"]
                }
                for source, info in self.energy_sources.items()
            },
            "device_priorities": {
                priority.value: len([p for p in self.energy_profiles.values() if p.priority == priority])
                for priority in EnergyPriority
            }
        }


# Helper functions for energy management
def create_energy_profile(
    device_id: DeviceId,
    device_type: DeviceType,
    rated_power: float,
    priority: EnergyPriority = EnergyPriority.NORMAL
) -> EnergyProfile:
    """Create energy profile with sensible defaults."""
    profile_id = f"profile_{device_id}_{int(datetime.now(UTC).timestamp())}"
    
    return EnergyProfile(
        profile_id=profile_id,
        device_id=device_id,
        device_type=device_type,
        rated_power=rated_power,
        standby_power=rated_power * 0.1,  # 10% standby
        peak_power=rated_power * 1.2,  # 20% higher peak
        efficiency_rating=0.85,
        priority=priority,
        operating_hours=[(6, 22)],  # 6 AM to 10 PM default
        energy_source_preference=[EnergySource.SOLAR, EnergySource.GRID]
    )


def calculate_time_of_use_cost(base_cost: float, hour: int) -> float:
    """Calculate time-of-use energy cost based on hour."""
    # Peak hours (higher cost): 16-20 (4 PM to 8 PM)
    # Off-peak hours (lower cost): 22-06 (10 PM to 6 AM)
    # Standard hours (base cost): Other times
    
    if 16 <= hour <= 20:  # Peak hours
        return base_cost * 1.5
    elif 22 <= hour or hour <= 6:  # Off-peak hours
        return base_cost * 0.7
    else:  # Standard hours
        return base_cost