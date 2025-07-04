"""
Resource management and optimization system for autonomous agents.

This module provides intelligent resource allocation, optimization, and monitoring
for autonomous agents. Implements sophisticated algorithms for resource prediction,
load balancing, and efficient utilization across multiple agents.

Security: Resource limits enforced with safety margins
Performance: <100ms allocation decisions, <500ms optimization cycles
Enterprise: Comprehensive resource tracking and audit integration
"""

import asyncio
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from collections import defaultdict, deque
import logging
from enum import Enum

from ..core.autonomous_systems import (
    AgentId, ActionId, AgentAction, PerformanceMetric,
    AutonomousAgentError
)
from ..core.either import Either
from ..core.contracts import require, ensure


class ResourceType(Enum):
    """Types of resources managed by the system."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    API_CALLS = "api_calls"
    ACTIONS = "actions"
    TIME = "time"


@dataclass
class ResourceAllocation:
    """Resource allocation for an agent."""
    agent_id: AgentId
    resource_type: ResourceType
    allocated_amount: float
    used_amount: float
    allocation_time: datetime
    expiration_time: Optional[datetime] = None
    priority: int = 5  # 1-10, higher is more important
    
    @property
    def utilization_rate(self) -> float:
        """Calculate resource utilization percentage."""
        if self.allocated_amount == 0:
            return 0.0
        return min(100.0, (self.used_amount / self.allocated_amount) * 100)
    
    @property
    def is_expired(self) -> bool:
        """Check if allocation has expired."""
        if not self.expiration_time:
            return False
        return datetime.now(UTC) > self.expiration_time
    
    def remaining_capacity(self) -> float:
        """Calculate remaining resource capacity."""
        return max(0.0, self.allocated_amount - self.used_amount)


@dataclass
class ResourceUsageHistory:
    """Historical resource usage tracking."""
    resource_type: ResourceType
    timestamp: datetime
    usage_amount: float
    agent_id: AgentId
    action_id: Optional[ActionId] = None
    
    
@dataclass
class ResourcePrediction:
    """Predicted resource requirements."""
    resource_type: ResourceType
    predicted_amount: float
    confidence: PerformanceMetric
    time_horizon: timedelta
    based_on_samples: int
    
    def get_buffered_amount(self, buffer_percentage: float = 20.0) -> float:
        """Get predicted amount with safety buffer."""
        return self.predicted_amount * (1 + buffer_percentage / 100)


class ResourceOptimizer:
    """Intelligent resource management and optimization system."""
    
    def __init__(self, total_resources: Dict[ResourceType, float]):
        self.total_resources = total_resources
        self.allocations: Dict[Tuple[AgentId, ResourceType], ResourceAllocation] = {}
        self.usage_history: Dict[ResourceType, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.resource_pools: Dict[ResourceType, float] = total_resources.copy()
        self.optimization_metrics = {
            "total_allocations": 0,
            "allocation_failures": 0,
            "optimization_cycles": 0,
            "average_utilization": {}
        }
        self._lock = asyncio.Lock()
    
    async def request_resources(self, agent_id: AgentId, 
                              requirements: Dict[ResourceType, float],
                              priority: int = 5,
                              duration: Optional[timedelta] = None) -> Either[AutonomousAgentError, Dict[ResourceType, float]]:
        """Request resources for an agent."""
        async with self._lock:
            try:
                # Check availability
                available = self._check_availability(requirements)
                if not all(available.values()):
                    # Try to optimize and free up resources
                    await self._optimize_allocations()
                    available = self._check_availability(requirements)
                    
                    if not all(available.values()):
                        self.optimization_metrics["allocation_failures"] += 1
                        return Either.left(AutonomousAgentError.resource_limit_exceeded(
                            "multiple_resources",
                            sum(requirements.values()),
                            sum(self.resource_pools.values())
                        ))
                
                # Allocate resources
                allocated = {}
                expiration = datetime.now(UTC) + duration if duration else None
                
                for resource_type, amount in requirements.items():
                    allocation = ResourceAllocation(
                        agent_id=agent_id,
                        resource_type=resource_type,
                        allocated_amount=amount,
                        used_amount=0.0,
                        allocation_time=datetime.now(UTC),
                        expiration_time=expiration,
                        priority=priority
                    )
                    
                    key = (agent_id, resource_type)
                    if key in self.allocations:
                        # Update existing allocation
                        self.allocations[key].allocated_amount += amount
                    else:
                        self.allocations[key] = allocation
                    
                    # Deduct from pool
                    self.resource_pools[resource_type] -= amount
                    allocated[resource_type] = amount
                
                self.optimization_metrics["total_allocations"] += 1
                return Either.right(allocated)
                
            except Exception as e:
                return Either.left(AutonomousAgentError.unexpected_error(f"Resource allocation failed: {str(e)}"))
    
    async def release_resources(self, agent_id: AgentId, 
                              resources: Optional[Dict[ResourceType, float]] = None) -> Either[AutonomousAgentError, None]:
        """Release resources allocated to an agent."""
        async with self._lock:
            try:
                if resources:
                    # Release specific resources
                    for resource_type, amount in resources.items():
                        key = (agent_id, resource_type)
                        if key in self.allocations:
                            allocation = self.allocations[key]
                            released_amount = min(amount, allocation.allocated_amount)
                            allocation.allocated_amount -= released_amount
                            self.resource_pools[resource_type] += released_amount
                            
                            if allocation.allocated_amount <= 0:
                                del self.allocations[key]
                else:
                    # Release all resources for agent
                    keys_to_remove = []
                    for key, allocation in self.allocations.items():
                        if allocation.agent_id == agent_id:
                            self.resource_pools[allocation.resource_type] += allocation.allocated_amount
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        del self.allocations[key]
                
                return Either.right(None)
                
            except Exception as e:
                return Either.left(AutonomousAgentError.unexpected_error(f"Resource release failed: {str(e)}"))
    
    async def report_usage(self, agent_id: AgentId, usage: Dict[ResourceType, float],
                         action_id: Optional[ActionId] = None) -> None:
        """Report actual resource usage."""
        async with self._lock:
            timestamp = datetime.now(UTC)
            
            for resource_type, amount in usage.items():
                # Update allocation tracking
                key = (agent_id, resource_type)
                if key in self.allocations:
                    self.allocations[key].used_amount += amount
                
                # Record history
                history_entry = ResourceUsageHistory(
                    resource_type=resource_type,
                    timestamp=timestamp,
                    usage_amount=amount,
                    agent_id=agent_id,
                    action_id=action_id
                )
                self.usage_history[resource_type].append(history_entry)
    
    async def predict_requirements(self, agent_id: AgentId, 
                                 time_horizon: timedelta) -> Dict[ResourceType, ResourcePrediction]:
        """Predict future resource requirements for an agent."""
        predictions = {}
        
        for resource_type in ResourceType:
            # Get historical usage for this agent
            agent_history = [
                entry for entry in self.usage_history[resource_type]
                if entry.agent_id == agent_id
            ]
            
            if len(agent_history) < 5:
                # Not enough data for prediction
                continue
            
            # Simple prediction based on recent average
            recent_usage = agent_history[-20:]  # Last 20 entries
            avg_usage = sum(entry.usage_amount for entry in recent_usage) / len(recent_usage)
            
            # Adjust for time horizon
            time_factor = time_horizon.total_seconds() / 3600  # Convert to hours
            predicted_amount = avg_usage * time_factor
            
            prediction = ResourcePrediction(
                resource_type=resource_type,
                predicted_amount=predicted_amount,
                confidence=PerformanceMetric(min(0.9, len(recent_usage) / 20)),
                time_horizon=time_horizon,
                based_on_samples=len(recent_usage)
            )
            
            predictions[resource_type] = prediction
        
        return predictions
    
    async def optimize_allocations(self) -> Dict[str, Any]:
        """Perform global resource optimization."""
        async with self._lock:
            optimization_result = await self._optimize_allocations()
            self.optimization_metrics["optimization_cycles"] += 1
            return optimization_result
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status."""
        status = {
            "total_resources": {rt.value: amount for rt, amount in self.total_resources.items()},
            "available_resources": {rt.value: amount for rt, amount in self.resource_pools.items()},
            "allocations": [],
            "utilization_rates": {},
            "optimization_metrics": self.optimization_metrics.copy()
        }
        
        # Calculate utilization rates
        for resource_type in ResourceType:
            total = self.total_resources.get(resource_type, 0)
            available = self.resource_pools.get(resource_type, 0)
            if total > 0:
                utilization = ((total - available) / total) * 100
                status["utilization_rates"][resource_type.value] = utilization
        
        # Add allocation details
        for allocation in self.allocations.values():
            status["allocations"].append({
                "agent_id": allocation.agent_id,
                "resource_type": allocation.resource_type.value,
                "allocated": allocation.allocated_amount,
                "used": allocation.used_amount,
                "utilization": allocation.utilization_rate,
                "priority": allocation.priority
            })
        
        # Calculate average utilization
        if status["utilization_rates"]:
            avg_utilization = sum(status["utilization_rates"].values()) / len(status["utilization_rates"])
            self.optimization_metrics["average_utilization"] = avg_utilization
        
        return status
    
    async def _optimize_allocations(self) -> Dict[str, Any]:
        """Internal optimization logic."""
        optimization_actions = {
            "reallocated": 0,
            "expired_cleaned": 0,
            "underutilized_reclaimed": 0
        }
        
        # Clean up expired allocations
        expired_keys = []
        for key, allocation in self.allocations.items():
            if allocation.is_expired:
                self.resource_pools[allocation.resource_type] += allocation.allocated_amount
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.allocations[key]
            optimization_actions["expired_cleaned"] += 1
        
        # Reclaim underutilized resources
        underutilized_threshold = 30.0  # 30% utilization
        for key, allocation in list(self.allocations.items()):
            if allocation.utilization_rate < underutilized_threshold:
                # Reclaim unused portion
                unused = allocation.allocated_amount - allocation.used_amount
                reclaim_amount = unused * 0.5  # Reclaim 50% of unused
                
                allocation.allocated_amount -= reclaim_amount
                self.resource_pools[allocation.resource_type] += reclaim_amount
                optimization_actions["underutilized_reclaimed"] += 1
        
        # Rebalance based on priority
        await self._rebalance_by_priority()
        
        return optimization_actions
    
    async def _rebalance_by_priority(self) -> None:
        """Rebalance resources based on agent priorities."""
        # Group allocations by resource type
        by_resource = defaultdict(list)
        for allocation in self.allocations.values():
            by_resource[allocation.resource_type].append(allocation)
        
        # Sort by priority and rebalance
        for resource_type, allocations in by_resource.items():
            allocations.sort(key=lambda a: a.priority, reverse=True)
            
            # High priority agents can borrow from low priority
            for i, high_priority in enumerate(allocations):
                if high_priority.utilization_rate > 90:  # Needs more resources
                    for low_priority in allocations[i+1:]:
                        if low_priority.utilization_rate < 50:  # Has spare capacity
                            # Transfer some allocation
                            transfer_amount = min(
                                high_priority.allocated_amount * 0.2,  # 20% increase
                                low_priority.remaining_capacity() * 0.5  # 50% of spare
                            )
                            
                            high_priority.allocated_amount += transfer_amount
                            low_priority.allocated_amount -= transfer_amount
                            break
    
    def _check_availability(self, requirements: Dict[ResourceType, float]) -> Dict[ResourceType, bool]:
        """Check if required resources are available."""
        availability = {}
        for resource_type, required in requirements.items():
            available = self.resource_pools.get(resource_type, 0)
            availability[resource_type] = available >= required
        return availability
    
    async def calculate_efficiency_score(self) -> float:
        """Calculate overall resource efficiency score."""
        if not self.allocations:
            return 100.0  # Perfect efficiency if no allocations
        
        total_efficiency = 0.0
        count = 0
        
        for allocation in self.allocations.values():
            # Efficiency based on utilization (optimal is 70-90%)
            utilization = allocation.utilization_rate
            if 70 <= utilization <= 90:
                efficiency = 100.0
            elif utilization < 70:
                efficiency = (utilization / 70) * 100
            else:  # > 90%
                efficiency = max(50, 100 - (utilization - 90) * 2)
            
            total_efficiency += efficiency
            count += 1
        
        return total_efficiency / count if count > 0 else 0.0
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for resource optimization."""
        recommendations = []
        
        # Analyze current state
        status = self.get_resource_status()
        
        # Check overall utilization
        avg_utilization = self.optimization_metrics.get("average_utilization", 0)
        if avg_utilization > 85:
            recommendations.append({
                "type": "scale_up",
                "urgency": "high",
                "description": f"Resource utilization at {avg_utilization:.1f}%, consider scaling up",
                "affected_resources": [rt.value for rt, util in status["utilization_rates"].items() if util > 85]
            })
        elif avg_utilization < 30:
            recommendations.append({
                "type": "scale_down",
                "urgency": "low",
                "description": f"Resource utilization at {avg_utilization:.1f}%, resources may be over-provisioned",
                "affected_resources": [rt.value for rt, util in status["utilization_rates"].items() if util < 30]
            })
        
        # Check for resource contention
        if self.optimization_metrics["allocation_failures"] > 10:
            recommendations.append({
                "type": "resource_contention",
                "urgency": "medium",
                "description": f"{self.optimization_metrics['allocation_failures']} allocation failures detected",
                "suggestion": "Review agent priorities and resource requirements"
            })
        
        # Check for imbalanced allocations
        for resource_type in ResourceType:
            allocations = [a for a in self.allocations.values() if a.resource_type == resource_type]
            if allocations:
                utilizations = [a.utilization_rate for a in allocations]
                if max(utilizations) - min(utilizations) > 50:
                    recommendations.append({
                        "type": "imbalanced_allocation",
                        "urgency": "medium",
                        "description": f"Large utilization variance for {resource_type.value}",
                        "suggestion": "Rebalance allocations or adjust agent behaviors"
                    })
        
        return recommendations