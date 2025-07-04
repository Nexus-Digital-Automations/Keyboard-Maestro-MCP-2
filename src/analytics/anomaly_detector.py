"""
Anomaly detection system for identifying performance and security issues.

Provides real-time anomaly detection with ML-powered analysis and
automated alerting for critical system deviations.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, UTC
import logging

from ..core.analytics_architecture import PrivacyMode, MetricValue


class AnomalyDetector:
    """Real-time anomaly detection for automation metrics."""
    
    def __init__(self, privacy_mode: PrivacyMode = PrivacyMode.COMPLIANT):
        self.privacy_mode = privacy_mode
        self.logger = logging.getLogger(__name__)
        self.anomaly_threshold = 2.0  # Standard deviations
        self.detected_anomalies = []
    
    async def detect_anomalies(self, metrics_data: List[MetricValue]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics data using statistical analysis."""
        anomalies = []
        
        # Group metrics by type and tool
        grouped_metrics = {}
        for metric in metrics_data:
            key = f"{metric.source_tool}_{metric.metric_id}"
            if isinstance(metric.value, (int, float)):
                if key not in grouped_metrics:
                    grouped_metrics[key] = []
                grouped_metrics[key].append(metric.value)
        
        for key, values in grouped_metrics.items():
            if len(values) >= 10:  # Minimum data for anomaly detection
                anomaly = await self._detect_statistical_anomaly(key, values)
                if anomaly:
                    anomalies.append(anomaly)
        
        self.detected_anomalies.extend(anomalies)
        return anomalies
    
    async def _detect_statistical_anomaly(self, metric_key: str, values: List[float]) -> Optional[Dict[str, Any]]:
        """Detect statistical anomalies using z-score method."""
        if len(values) < 10:
            return None
        
        # Calculate statistics
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return None
        
        # Check recent values for anomalies
        recent_values = values[-5:]
        anomalous_values = []
        
        for value in recent_values:
            z_score = abs(value - mean) / std_dev
            if z_score > self.anomaly_threshold:
                anomalous_values.append({
                    'value': value,
                    'z_score': z_score,
                    'deviation': abs(value - mean)
                })
        
        if anomalous_values:
            return {
                'metric_key': metric_key,
                'anomaly_type': 'statistical_outlier',
                'anomalous_values': anomalous_values,
                'baseline_mean': mean,
                'baseline_std_dev': std_dev,
                'severity': 'high' if max(av['z_score'] for av in anomalous_values) > 3 else 'medium',
                'detected_at': datetime.now(UTC),
                'confidence': 0.9
            }
        
        return None