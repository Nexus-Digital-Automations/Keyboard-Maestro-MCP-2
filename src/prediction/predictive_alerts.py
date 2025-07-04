"""
Predictive alert system for proactive notification and action triggering.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, UTC
import logging

from .predictive_types import PredictiveAlert, AlertId, AlertSeverity, ConfidenceLevel, create_alert_id, create_confidence_level
from .model_manager import PredictiveModelManager
from ..core.either import Either

logger = logging.getLogger(__name__)


class PredictiveAlertSystem:
    """Proactive alert system with predictive capabilities."""
    
    def __init__(self, model_manager: Optional[PredictiveModelManager] = None):
        self.model_manager = model_manager or PredictiveModelManager()
        self.active_alerts: List[PredictiveAlert] = []
        self.alert_history: List[PredictiveAlert] = []
        self.logger = logging.getLogger(__name__)
    
    async def generate_predictive_alert(
        self, 
        alert_data: Dict[str, Any]
    ) -> Either[Exception, PredictiveAlert]:
        """Generate predictive alert based on analysis."""
        try:
            alert = PredictiveAlert(
                alert_id=create_alert_id(),
                alert_type=alert_data.get("type", "performance_warning"),
                severity=AlertSeverity(alert_data.get("severity", "warning")),
                title=alert_data.get("title", "Predictive Alert"),
                description=alert_data.get("description", "System issue predicted"),
                predicted_occurrence=datetime.now(UTC) + timedelta(hours=2),
                confidence=create_confidence_level(alert_data.get("confidence", 0.8)),
                affected_systems=alert_data.get("affected_systems", ["automation_system"]),
                recommended_actions=alert_data.get("actions", ["Monitor system", "Check resources"]),
                escalation_threshold=timedelta(hours=1),
                auto_resolution=alert_data.get("auto_resolution", False),
                model_used="alert_predictor_001"
            )
            
            self.active_alerts.append(alert)
            return Either.right(alert)
            
        except Exception as e:
            return Either.left(e)
    
    def get_active_alerts(self) -> List[PredictiveAlert]:
        """Get all active predictive alerts."""
        return self.active_alerts.copy()
    
    def acknowledge_alert(self, alert_id: AlertId) -> bool:
        """Acknowledge and resolve an alert."""
        for i, alert in enumerate(self.active_alerts):
            if alert.alert_id == alert_id:
                resolved_alert = self.active_alerts.pop(i)
                self.alert_history.append(resolved_alert)
                return True
        return False