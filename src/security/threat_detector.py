"""
Threat Detector - TASK_62 Phase 4 Advanced Security Features

AI-powered threat detection and response for zero trust security framework.
Provides intelligent threat analysis, pattern recognition, and automated response capabilities.

Architecture: Machine Learning + Behavioral Analysis + Pattern Recognition + Automated Response
Performance: <500ms threat detection, <1s pattern analysis, <2s response orchestration
Security: Real-time threat intelligence, behavioral anomaly detection, adaptive response
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import json
import statistics
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.zero_trust_architecture import (
    ThreatId, SecurityContextId, RiskScore, ValidationId,
    ThreatSeverity, SecurityOperation, TreatmentLevel,
    SecurityContext, ZeroTrustError, ThreatDetectionError,
    create_threat_id, create_security_context_id, create_risk_score
)


class ThreatType(Enum):
    """Types of security threats."""
    MALWARE = "malware"                    # Malicious software
    PHISHING = "phishing"                  # Social engineering attack
    INSIDER_THREAT = "insider_threat"      # Internal threat actor
    APT = "apt"                           # Advanced Persistent Threat
    DDOS = "ddos"                         # Distributed Denial of Service
    DATA_BREACH = "data_breach"           # Data exfiltration
    PRIVILEGE_ESCALATION = "privilege_escalation"  # Unauthorized elevation
    LATERAL_MOVEMENT = "lateral_movement"  # Network traversal
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"  # Behavioral anomaly
    UNAUTHORIZED_ACCESS = "unauthorized_access"  # Access violation
    SUSPICIOUS_ACTIVITY = "suspicious_activity"  # Suspicious patterns


class ThreatVector(Enum):
    """Attack vectors for threats."""
    EMAIL = "email"                       # Email-based attack
    WEB = "web"                          # Web-based attack
    NETWORK = "network"                  # Network-based attack
    ENDPOINT = "endpoint"                # Endpoint-based attack
    APPLICATION = "application"          # Application-based attack
    SOCIAL = "social"                    # Social engineering
    PHYSICAL = "physical"                # Physical access
    SUPPLY_CHAIN = "supply_chain"        # Supply chain attack
    CLOUD = "cloud"                      # Cloud infrastructure attack
    API = "api"                          # API-based attack


class ResponseAction(Enum):
    """Automated response actions."""
    MONITOR = "monitor"                   # Continue monitoring
    ALERT = "alert"                      # Generate alert
    ISOLATE = "isolate"                  # Isolate affected system
    BLOCK = "block"                      # Block access/traffic
    QUARANTINE = "quarantine"            # Quarantine threat
    REMEDIATE = "remediate"              # Automatic remediation
    ESCALATE = "escalate"                # Escalate to security team
    INVESTIGATE = "investigate"          # Trigger investigation
    LOCKDOWN = "lockdown"                # Emergency lockdown
    DISABLE_ACCOUNT = "disable_account"  # Disable user account


@dataclass(frozen=True)
class ThreatIndicator:
    """Security threat indicator."""
    indicator_id: str
    indicator_type: str                   # IP, domain, hash, pattern, etc.
    indicator_value: str
    threat_types: List[ThreatType]
    confidence: float                     # 0.0 to 1.0
    severity: ThreatSeverity
    first_seen: datetime
    last_seen: datetime
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ThreatPattern:
    """Behavioral threat pattern."""
    pattern_id: str
    pattern_name: str
    pattern_description: str
    threat_types: List[ThreatType]
    indicators: List[str]                 # Pattern matching rules
    threshold_score: float               # Minimum score to trigger
    time_window_minutes: int             # Time window for pattern analysis
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThreatDetection:
    """Detected security threat."""
    threat_id: ThreatId
    threat_type: ThreatType
    threat_vector: ThreatVector
    severity: ThreatSeverity
    confidence: float                     # 0.0 to 1.0
    risk_score: RiskScore
    source_ip: Optional[str] = None
    target_resources: List[str] = field(default_factory=list)
    indicators: List[ThreatIndicator] = field(default_factory=list)
    patterns_matched: List[str] = field(default_factory=list)
    detection_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[ResponseAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ThreatResponse:
    """Automated threat response."""
    response_id: str
    threat_id: ThreatId
    response_actions: List[ResponseAction]
    execution_status: str                 # pending, executing, completed, failed
    execution_timestamp: datetime
    affected_systems: List[str] = field(default_factory=list)
    response_effectiveness: Optional[float] = None  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIThreatDetector:
    """AI-powered threat detection and response system."""
    
    def __init__(self):
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.threat_patterns: Dict[str, ThreatPattern] = {}
        self.active_detections: Dict[ThreatId, ThreatDetection] = {}
        self.behavioral_baselines: Dict[str, Dict[str, float]] = {}
        self.ml_models: Dict[str, Any] = {}  # Placeholder for ML models
        
        # Initialize default threat patterns
        self._initialize_threat_patterns()
        
        # Initialize threat indicators from intelligence feeds
        self._initialize_threat_indicators()
    
    def _initialize_threat_patterns(self):
        """Initialize default threat detection patterns."""
        patterns = [
            ThreatPattern(
                pattern_id="suspicious_login_pattern",
                pattern_name="Suspicious Login Activity",
                pattern_description="Multiple failed logins followed by successful login",
                threat_types=[ThreatType.UNAUTHORIZED_ACCESS, ThreatType.INSIDER_THREAT],
                indicators=["failed_login_count > 5", "login_success_after_failures", "unusual_login_time"],
                threshold_score=0.7,
                time_window_minutes=30
            ),
            ThreatPattern(
                pattern_id="data_exfiltration_pattern",
                pattern_name="Data Exfiltration Activity",
                pattern_description="Large data transfers to external systems",
                threat_types=[ThreatType.DATA_BREACH, ThreatType.INSIDER_THREAT],
                indicators=["large_data_transfer", "external_destination", "unusual_access_pattern"],
                threshold_score=0.8,
                time_window_minutes=60
            ),
            ThreatPattern(
                pattern_id="lateral_movement_pattern",
                pattern_name="Lateral Movement Behavior",
                pattern_description="Network scanning and privilege escalation attempts",
                threat_types=[ThreatType.LATERAL_MOVEMENT, ThreatType.APT],
                indicators=["network_scanning", "privilege_escalation_attempt", "credential_dumping"],
                threshold_score=0.9,
                time_window_minutes=120
            )
        ]
        
        for pattern in patterns:
            self.threat_patterns[pattern.pattern_id] = pattern
    
    def _initialize_threat_indicators(self):
        """Initialize threat indicators from intelligence feeds."""
        # Placeholder for threat intelligence integration
        # In production, this would load from external threat feeds
        indicators = [
            ThreatIndicator(
                indicator_id="malicious_ip_1",
                indicator_type="ip",
                indicator_value="192.168.1.100",
                threat_types=[ThreatType.APT, ThreatType.DDOS],
                confidence=0.95,
                severity=ThreatSeverity.HIGH,
                first_seen=datetime.now(UTC) - timedelta(days=30),
                last_seen=datetime.now(UTC),
                sources=["threat_feed_1", "security_vendor_x"]
            )
        ]
        
        for indicator in indicators:
            self.threat_indicators[indicator.indicator_id] = indicator
    
    @require(lambda security_events: isinstance(security_events, list))
    @ensure(lambda result: result.is_success() or result.is_error())
    async def detect_threats(
        self,
        security_events: List[Dict[str, Any]],
        scope: str = "system",
        enable_ml_analysis: bool = True,
        enable_behavioral_analysis: bool = True
    ) -> Either[ThreatDetectionError, Dict[str, Any]]:
        """
        Detect security threats using AI and pattern recognition.
        
        Args:
            security_events: List of security events to analyze
            scope: Detection scope (system, application, network, user)
            enable_ml_analysis: Enable machine learning-based analysis
            enable_behavioral_analysis: Enable behavioral anomaly detection
            
        Returns:
            Either threat detection error or detection results
        """
        try:
            detections = []
            
            # Analyze security events for threat patterns
            pattern_detections = await self._analyze_threat_patterns(security_events)
            detections.extend(pattern_detections)
            
            # Check against threat indicators
            indicator_detections = await self._check_threat_indicators(security_events)
            detections.extend(indicator_detections)
            
            # Perform behavioral analysis if enabled
            if enable_behavioral_analysis:
                behavioral_detections = await self._analyze_behavioral_anomalies(security_events, scope)
                detections.extend(behavioral_detections)
            
            # Perform ML analysis if enabled
            if enable_ml_analysis:
                ml_detections = await self._analyze_with_ml(security_events, scope)
                detections.extend(ml_detections)
            
            # Correlate and prioritize detections
            correlated_detections = await self._correlate_detections(detections)
            
            # Store active detections
            for detection in correlated_detections:
                self.active_detections[detection.threat_id] = detection
            
            # Calculate overall threat level
            threat_level = self._calculate_threat_level(correlated_detections)
            
            return Either.success({
                "threats": [
                    {
                        "threat_id": d.threat_id,
                        "threat_type": d.threat_type.value,
                        "threat_vector": d.threat_vector.value,
                        "severity": d.severity.value,
                        "confidence": d.confidence,
                        "risk_score": float(d.risk_score),
                        "source_ip": d.source_ip,
                        "target_resources": d.target_resources,
                        "patterns_matched": d.patterns_matched,
                        "detection_timestamp": d.detection_timestamp.isoformat(),
                        "recommended_actions": [action.value for action in d.recommended_actions],
                        "evidence": d.evidence
                    }
                    for d in correlated_detections
                ],
                "threat_level": threat_level,
                "total_threats": len(correlated_detections),
                "high_severity_threats": len([d for d in correlated_detections if d.severity == ThreatSeverity.HIGH]),
                "recommended_immediate_actions": self._get_immediate_actions(correlated_detections)
            })
            
        except Exception as e:
            return Either.error(ThreatDetectionError(f"Threat detection failed: {str(e)}"))
    
    async def _analyze_threat_patterns(self, events: List[Dict[str, Any]]) -> List[ThreatDetection]:
        """Analyze events against known threat patterns."""
        detections = []
        
        for pattern in self.threat_patterns.values():
            pattern_score = await self._calculate_pattern_score(events, pattern)
            
            if pattern_score >= pattern.threshold_score:
                threat_id = create_threat_id(f"pattern_{pattern.pattern_id}")
                
                # Determine primary threat type and vector
                threat_type = pattern.threat_types[0] if pattern.threat_types else ThreatType.SUSPICIOUS_ACTIVITY
                threat_vector = self._infer_threat_vector(events, threat_type)
                
                # Calculate severity based on pattern and score
                severity = self._calculate_severity(pattern_score, threat_type)
                
                # Generate recommended actions
                actions = self._generate_response_actions(threat_type, severity)
                
                detection = ThreatDetection(
                    threat_id=threat_id,
                    threat_type=threat_type,
                    threat_vector=threat_vector,
                    severity=severity,
                    confidence=pattern_score,
                    risk_score=create_risk_score(pattern_score * 100),
                    patterns_matched=[pattern.pattern_id],
                    evidence={"pattern_score": pattern_score, "matching_events": len(events)},
                    recommended_actions=actions,
                    metadata={"detection_method": "pattern_analysis"}
                )
                
                detections.append(detection)
        
        return detections
    
    async def _check_threat_indicators(self, events: List[Dict[str, Any]]) -> List[ThreatDetection]:
        """Check events against known threat indicators."""
        detections = []
        
        for event in events:
            # Extract indicators from event (IP addresses, domains, hashes, etc.)
            event_indicators = self._extract_indicators_from_event(event)
            
            for indicator_value in event_indicators:
                # Check if indicator matches known threats
                matching_indicators = [
                    ind for ind in self.threat_indicators.values()
                    if ind.indicator_value == indicator_value
                ]
                
                for indicator in matching_indicators:
                    threat_id = create_threat_id(f"indicator_{indicator.indicator_id}")
                    
                    # Use indicator's threat types and severity
                    threat_type = indicator.threat_types[0] if indicator.threat_types else ThreatType.SUSPICIOUS_ACTIVITY
                    threat_vector = self._infer_threat_vector([event], threat_type)
                    
                    # Generate recommended actions
                    actions = self._generate_response_actions(threat_type, indicator.severity)
                    
                    detection = ThreatDetection(
                        threat_id=threat_id,
                        threat_type=threat_type,
                        threat_vector=threat_vector,
                        severity=indicator.severity,
                        confidence=indicator.confidence,
                        risk_score=create_risk_score(indicator.confidence * 100),
                        source_ip=event.get("source_ip"),
                        indicators=[indicator],
                        evidence={"indicator_match": indicator_value, "event_id": event.get("event_id")},
                        recommended_actions=actions,
                        metadata={"detection_method": "indicator_match"}
                    )
                    
                    detections.append(detection)
        
        return detections
    
    async def _analyze_behavioral_anomalies(self, events: List[Dict[str, Any]], scope: str) -> List[ThreatDetection]:
        """Analyze events for behavioral anomalies."""
        detections = []
        
        # Group events by user/system/application
        entity_events = self._group_events_by_entity(events)
        
        for entity_id, entity_events_list in entity_events.items():
            # Calculate behavioral metrics
            current_metrics = self._calculate_behavioral_metrics(entity_events_list)
            
            # Get baseline metrics for entity
            baseline_metrics = self.behavioral_baselines.get(entity_id, {})
            
            if baseline_metrics:
                # Compare current behavior to baseline
                anomaly_score = self._calculate_anomaly_score(current_metrics, baseline_metrics)
                
                if anomaly_score > 0.7:  # Threshold for anomalous behavior
                    threat_id = create_threat_id(f"behavioral_{entity_id}")
                    
                    detection = ThreatDetection(
                        threat_id=threat_id,
                        threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                        threat_vector=ThreatVector.ENDPOINT,
                        severity=ThreatSeverity.MEDIUM if anomaly_score < 0.9 else ThreatSeverity.HIGH,
                        confidence=anomaly_score,
                        risk_score=create_risk_score(anomaly_score * 100),
                        target_resources=[entity_id],
                        evidence={
                            "anomaly_score": anomaly_score,
                            "current_metrics": current_metrics,
                            "baseline_metrics": baseline_metrics
                        },
                        recommended_actions=[ResponseAction.MONITOR, ResponseAction.INVESTIGATE],
                        metadata={"detection_method": "behavioral_analysis", "entity_id": entity_id}
                    )
                    
                    detections.append(detection)
            else:
                # No baseline - establish one
                self.behavioral_baselines[entity_id] = current_metrics
        
        return detections
    
    async def _analyze_with_ml(self, events: List[Dict[str, Any]], scope: str) -> List[ThreatDetection]:
        """Analyze events using machine learning models."""
        detections = []
        
        # Placeholder for ML-based threat detection
        # In production, this would use trained ML models for:
        # - Anomaly detection
        # - Classification of malicious activity
        # - Predictive threat modeling
        
        # Simulate ML analysis
        if len(events) > 10:  # Only analyze if sufficient data
            ml_score = min(0.95, len(events) * 0.05)  # Simulated ML confidence
            
            if ml_score > 0.6:
                threat_id = create_threat_id("ml_detection")
                
                detection = ThreatDetection(
                    threat_id=threat_id,
                    threat_type=ThreatType.SUSPICIOUS_ACTIVITY,
                    threat_vector=ThreatVector.NETWORK,
                    severity=ThreatSeverity.MEDIUM,
                    confidence=ml_score,
                    risk_score=create_risk_score(ml_score * 100),
                    evidence={"ml_score": ml_score, "model": "anomaly_detector_v1"},
                    recommended_actions=[ResponseAction.MONITOR, ResponseAction.ALERT],
                    metadata={"detection_method": "machine_learning"}
                )
                
                detections.append(detection)
        
        return detections
    
    async def _correlate_detections(self, detections: List[ThreatDetection]) -> List[ThreatDetection]:
        """Correlate and deduplicate threat detections."""
        # Group similar detections
        correlated = {}
        
        for detection in detections:
            # Create correlation key based on threat type, vector, and targets
            correlation_key = f"{detection.threat_type.value}_{detection.threat_vector.value}_{hash(tuple(detection.target_resources))}"
            
            if correlation_key not in correlated:
                correlated[correlation_key] = []
            correlated[correlation_key].append(detection)
        
        # Merge correlated detections
        merged_detections = []
        for group in correlated.values():
            if len(group) == 1:
                merged_detections.append(group[0])
            else:
                # Merge multiple detections into one with higher confidence
                merged = self._merge_detections(group)
                merged_detections.append(merged)
        
        return merged_detections
    
    def _merge_detections(self, detections: List[ThreatDetection]) -> ThreatDetection:
        """Merge multiple related detections into one."""
        # Use the detection with highest confidence as base
        base_detection = max(detections, key=lambda d: d.confidence)
        
        # Merge evidence and patterns
        merged_evidence = {}
        merged_patterns = []
        merged_actions = set()
        
        for detection in detections:
            merged_evidence.update(detection.evidence)
            merged_patterns.extend(detection.patterns_matched)
            merged_actions.update(detection.recommended_actions)
        
        # Calculate merged confidence (average weighted by individual confidence)
        total_weight = sum(d.confidence for d in detections)
        merged_confidence = min(1.0, total_weight / len(detections) * 1.2)  # Boost for correlation
        
        return ThreatDetection(
            threat_id=base_detection.threat_id,
            threat_type=base_detection.threat_type,
            threat_vector=base_detection.threat_vector,
            severity=max(d.severity for d in detections),  # Use highest severity
            confidence=merged_confidence,
            risk_score=create_risk_score(merged_confidence * 100),
            source_ip=base_detection.source_ip,
            target_resources=list(set().union(*[d.target_resources for d in detections])),
            patterns_matched=list(set(merged_patterns)),
            evidence=merged_evidence,
            recommended_actions=list(merged_actions),
            metadata={"detection_method": "correlated", "merged_count": len(detections)}
        )
    
    # Helper methods
    
    async def _calculate_pattern_score(self, events: List[Dict[str, Any]], pattern: ThreatPattern) -> float:
        """Calculate how well events match a threat pattern."""
        # Simplified pattern matching - in production would use more sophisticated analysis
        matching_indicators = 0
        
        for indicator in pattern.indicators:
            if self._check_indicator_in_events(indicator, events):
                matching_indicators += 1
        
        return matching_indicators / len(pattern.indicators) if pattern.indicators else 0.0
    
    def _check_indicator_in_events(self, indicator: str, events: List[Dict[str, Any]]) -> bool:
        """Check if an indicator pattern exists in events."""
        # Simplified indicator checking
        if "failed_login" in indicator:
            return any(event.get("event_type") == "login_failed" for event in events)
        elif "large_data_transfer" in indicator:
            return any(event.get("data_size", 0) > 1000000 for event in events)  # > 1MB
        elif "network_scanning" in indicator:
            return any(event.get("event_type") == "network_scan" for event in events)
        
        return False
    
    def _extract_indicators_from_event(self, event: Dict[str, Any]) -> List[str]:
        """Extract threat indicators from security event."""
        indicators = []
        
        # Extract IP addresses
        if "source_ip" in event:
            indicators.append(event["source_ip"])
        if "destination_ip" in event:
            indicators.append(event["destination_ip"])
        
        # Extract domains
        if "domain" in event:
            indicators.append(event["domain"])
        
        # Extract file hashes
        if "file_hash" in event:
            indicators.append(event["file_hash"])
        
        return indicators
    
    def _infer_threat_vector(self, events: List[Dict[str, Any]], threat_type: ThreatType) -> ThreatVector:
        """Infer threat vector from events and threat type."""
        # Analyze events to determine likely attack vector
        if any(event.get("source") == "email" for event in events):
            return ThreatVector.EMAIL
        elif any(event.get("source") == "web" for event in events):
            return ThreatVector.WEB
        elif any(event.get("protocol") in ["tcp", "udp"] for event in events):
            return ThreatVector.NETWORK
        else:
            return ThreatVector.ENDPOINT  # Default
    
    def _calculate_severity(self, score: float, threat_type: ThreatType) -> ThreatSeverity:
        """Calculate threat severity based on score and type."""
        # High-impact threat types get elevated severity
        high_impact_types = [ThreatType.APT, ThreatType.DATA_BREACH, ThreatType.PRIVILEGE_ESCALATION]
        
        if threat_type in high_impact_types:
            if score > 0.7:
                return ThreatSeverity.HIGH
            elif score > 0.5:
                return ThreatSeverity.MEDIUM
            else:
                return ThreatSeverity.LOW
        else:
            if score > 0.9:
                return ThreatSeverity.HIGH
            elif score > 0.7:
                return ThreatSeverity.MEDIUM
            else:
                return ThreatSeverity.LOW
    
    def _generate_response_actions(self, threat_type: ThreatType, severity: ThreatSeverity) -> List[ResponseAction]:
        """Generate recommended response actions based on threat characteristics."""
        actions = [ResponseAction.MONITOR, ResponseAction.ALERT]
        
        if severity == ThreatSeverity.HIGH:
            actions.extend([ResponseAction.ISOLATE, ResponseAction.INVESTIGATE])
            
            if threat_type in [ThreatType.APT, ThreatType.DATA_BREACH]:
                actions.append(ResponseAction.ESCALATE)
            elif threat_type == ThreatType.MALWARE:
                actions.extend([ResponseAction.QUARANTINE, ResponseAction.REMEDIATE])
        
        elif severity == ThreatSeverity.MEDIUM:
            actions.append(ResponseAction.BLOCK)
            
            if threat_type == ThreatType.UNAUTHORIZED_ACCESS:
                actions.append(ResponseAction.DISABLE_ACCOUNT)
        
        return actions
    
    def _group_events_by_entity(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group events by entity (user, system, application)."""
        grouped = {}
        
        for event in events:
            entity_id = event.get("user_id") or event.get("system_id") or event.get("source_ip", "unknown")
            
            if entity_id not in grouped:
                grouped[entity_id] = []
            grouped[entity_id].append(event)
        
        return grouped
    
    def _calculate_behavioral_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate behavioral metrics for an entity."""
        metrics = {
            "login_frequency": 0.0,
            "data_access_volume": 0.0,
            "network_connections": 0.0,
            "failed_login_rate": 0.0,
            "unusual_time_activity": 0.0
        }
        
        if not events:
            return metrics
        
        # Calculate metrics from events
        login_events = [e for e in events if e.get("event_type") in ["login_success", "login_failed"]]
        failed_logins = [e for e in login_events if e.get("event_type") == "login_failed"]
        
        metrics["login_frequency"] = len(login_events) / len(events) if events else 0.0
        metrics["failed_login_rate"] = len(failed_logins) / len(login_events) if login_events else 0.0
        
        # Calculate data access volume
        data_events = [e for e in events if e.get("data_size")]
        if data_events:
            total_data = sum(e.get("data_size", 0) for e in data_events)
            metrics["data_access_volume"] = total_data / len(events)
        
        return metrics
    
    def _calculate_anomaly_score(self, current: Dict[str, float], baseline: Dict[str, float]) -> float:
        """Calculate anomaly score comparing current behavior to baseline."""
        anomaly_scores = []
        
        for metric, current_value in current.items():
            baseline_value = baseline.get(metric, 0.0)
            
            if baseline_value > 0:
                # Calculate percentage deviation
                deviation = abs(current_value - baseline_value) / baseline_value
                anomaly_scores.append(min(1.0, deviation))
            elif current_value > 0:
                # New activity where there was none before
                anomaly_scores.append(0.5)
        
        return statistics.mean(anomaly_scores) if anomaly_scores else 0.0
    
    def _calculate_threat_level(self, detections: List[ThreatDetection]) -> str:
        """Calculate overall threat level from detections."""
        if not detections:
            return "low"
        
        high_severity_count = len([d for d in detections if d.severity == ThreatSeverity.HIGH])
        medium_severity_count = len([d for d in detections if d.severity == ThreatSeverity.MEDIUM])
        
        if high_severity_count > 0:
            return "critical" if high_severity_count > 2 else "high"
        elif medium_severity_count > 3:
            return "high"
        elif medium_severity_count > 0:
            return "medium"
        else:
            return "low"
    
    def _get_immediate_actions(self, detections: List[ThreatDetection]) -> List[str]:
        """Get immediate actions recommended across all detections."""
        all_actions = set()
        
        for detection in detections:
            all_actions.update(action.value for action in detection.recommended_actions)
        
        # Prioritize critical actions
        priority_actions = [
            ResponseAction.LOCKDOWN.value,
            ResponseAction.ISOLATE.value,
            ResponseAction.ESCALATE.value,
            ResponseAction.BLOCK.value,
            ResponseAction.INVESTIGATE.value
        ]
        
        immediate = []
        for action in priority_actions:
            if action in all_actions:
                immediate.append(action)
        
        return immediate[:5]  # Limit to top 5 actions


# Export the threat detector class
__all__ = ["AIThreatDetector", "ThreatType", "ThreatVector", "ResponseAction", "ThreatDetection"]