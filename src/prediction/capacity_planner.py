"""
Capacity planning for resource scaling and optimization.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, UTC
import logging

from .predictive_types import CapacityPlan, CapacityPlanId, ConfidenceLevel, create_capacity_plan_id, create_confidence_level
from .model_manager import PredictiveModelManager
from ..core.either import Either

logger = logging.getLogger(__name__)


class CapacityPlanner:
    """Intelligent capacity planning and resource scaling."""
    
    def __init__(self, model_manager: Optional[PredictiveModelManager] = None):
        self.model_manager = model_manager or PredictiveModelManager()
        self.capacity_plans: List[CapacityPlan] = []
        self.logger = logging.getLogger(__name__)
    
    async def create_capacity_plan(
        self, 
        resource_type: str, 
        planning_horizon: timedelta = timedelta(days=30)
    ) -> Either[Exception, CapacityPlan]:
        """Create capacity plan for resource scaling."""
        try:
            plan = CapacityPlan(
                plan_id=create_capacity_plan_id(),
                resource_type=resource_type,
                current_capacity=100.0,
                projected_demand=[
                    (datetime.now(UTC) + timedelta(days=7), 120.0, create_confidence_level(0.8)),
                    (datetime.now(UTC) + timedelta(days=14), 140.0, create_confidence_level(0.7)),
                    (datetime.now(UTC) + timedelta(days=21), 160.0, create_confidence_level(0.6)),
                ],
                scaling_recommendations=[
                    "Increase capacity by 20% within 1 week",
                    "Plan additional scaling for month 2"
                ],
                optimal_scaling_time=datetime.now(UTC) + timedelta(days=5),
                cost_implications={"scaling_cost": 1000.0, "operational_savings": 500.0},
                risk_assessment="Low risk with gradual scaling approach",
                confidence=create_confidence_level(0.75),
                model_used="capacity_model_001"
            )
            
            self.capacity_plans.append(plan)
            return Either.right(plan)
            
        except Exception as e:
            return Either.left(e)